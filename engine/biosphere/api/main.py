"""FastAPI application factory.

Two things here are product requirements rather than plumbing, and both are easy to get
subtly wrong:

**1. Split CORS.** The dashboard is a first-party app and its write routes stay locked to
``settings.cors_allow_origins``. The player, however, is embedded in an ``<iframe>`` on
third-party listing portals, so the public read routes and the media mount must answer *any*
origin. One app-wide CORS middleware cannot express both, so :class:`ScopedCORSMiddleware`
dispatches to one of two ``CORSMiddleware`` instances by path prefix.

**2. Cross-origin textures.** A WebGL panorama is uploaded to the GPU as a texture. When the
image comes from another origin the browser requires ``Access-Control-Allow-Origin`` *and*
``Cross-Origin-Resource-Policy: cross-origin`` — with COEP-enabled embedders, a missing CORP
header blocks the fetch outright and the player shows a black sphere with no error the user
can act on. The media mount therefore sets both unconditionally, plus long immutable caching
(URLs carry a ``?v=`` token so a reprocessed scene still busts the cache).

Framing is allowed via ``Content-Security-Policy: frame-ancestors *``; ``X-Frame-Options`` is
deliberately never sent, because it has no "allow any origin" value and its mere presence
would break every embed.
"""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from typing import Any

import orjson
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from biosphere.api.deps import current_user
from biosphere.api.routes import PRIVATE_ROUTERS, PUBLIC_ROUTERS
from biosphere.api.routes.properties import sweep_staging
from biosphere.core.config import Settings, get_settings
from biosphere.core.db import init_db
from biosphere.core.jobs import get_job_manager
from biosphere.core.logging import bind_context, clear_context, configure_logging, get_logger
from biosphere.core.models import ApiException
from biosphere.core.storage import get_storage
from biosphere.pano.types import StitchError

__all__ = ["create_app", "app", "API_PREFIX", "ORJSONResponse"]

log = get_logger("biosphere.api")

API_PREFIX = "/api/v1"


class ORJSONResponse(JSONResponse):
    """orjson-backed JSON response.

    Defined here rather than imported from ``fastapi.responses``: that symbol is deprecated
    in FastAPI 0.141 (it now serialises response models to bytes via pydantic directly) and
    instantiating it emits a warning on every single response. Numpy scalars still leak out
    of the CV code into error payloads occasionally, so ``OPT_SERIALIZE_NUMPY`` stays on.
    """

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY, default=str)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

#: Bilingual copy for the HTTP statuses a client can actually provoke.
_STATUS_COPY: dict[int, tuple[str, str, str]] = {
    400: ("bad_request", "The request was malformed.", "İstek hatalı."),
    401: ("unauthorized", "Authentication is required.", "Kimlik doğrulaması gerekli."),
    403: ("forbidden", "You do not have access to this resource.", "Bu kaynağa erişiminiz yok."),
    404: ("not_found", "The requested resource was not found.", "İstenen kayıt bulunamadı."),
    405: ("method_not_allowed", "That method is not allowed here.", "Bu adres için geçersiz yöntem."),
    409: ("conflict", "The request conflicts with the current state.", "İstek mevcut durumla çakışıyor."),
    413: ("payload_too_large", "The upload is too large.", "Yükleme çok büyük."),
    415: ("unsupported_media", "That file type is not supported.", "Bu dosya türü desteklenmiyor."),
    422: ("validation_error", "The request could not be validated.", "İstek doğrulanamadı."),
    429: ("rate_limited", "Too many requests. Slow down.", "Çok fazla istek gönderildi."),
    500: ("internal_error", "Something went wrong on our side.", "Sunucu tarafında bir hata oluştu."),
    503: ("unavailable", "The service is temporarily unavailable.", "Servis geçici olarak kullanılamıyor."),
    507: ("insufficient_storage", "The server is out of storage.", "Sunucuda depolama alanı kalmadı."),
}


def _error_body(code: str, message: str, message_tr: str, hint: str | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "message_tr": message_tr, "hint": hint}


# --------------------------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------------------------


class RequestContextMiddleware:
    """Assign a request id, bind it to the log context, and emit one access line per request.

    Written as raw ASGI rather than ``BaseHTTPMiddleware`` on purpose: ``BaseHTTPMiddleware``
    wraps the response in an anyio task pair, which interferes with long-lived
    ``text/event-stream`` responses — exactly the endpoint this service depends on most.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        request_id = headers.get("x-request-id") or uuid.uuid4().hex[:16]
        method = scope.get("method", "GET")
        path = scope.get("path", "")
        started = time.perf_counter()
        status_code = 500

        bind_context(request_id=request_id, method=method, path=path)

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message)["x-request-id"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            log.info(
                "http.request",
                status=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            clear_context()


class ScopedCORSMiddleware:
    """Two CORS policies in one app, selected by path prefix."""

    def __init__(
        self,
        app: Any,
        *,
        public_prefixes: tuple[str, ...],
        allow_origins: list[str],
    ) -> None:
        self.app = app
        self.public_prefixes = public_prefixes
        self._restricted = CORSMiddleware(
            app,
            allow_origins=allow_origins,
            allow_credentials=True,
            # PUT is the chunked upload (``/properties/uploads/{id}/parts/{n}``): a dashboard on
            # another origin than the engine sends it too, not only the phone behind the proxy.
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
            max_age=600,
        )
        self._permissive = CORSMiddleware(
            app,
            # ``allow_origins=["*"]`` makes the header a literal ``*`` (rather than an echo of
            # the caller's origin), which is what a shared CDN cache and a no-credentials
            # embed both want. The regex is kept so the intent survives a config edit.
            allow_origins=["*"],
            allow_origin_regex=".*",
            # Credentials are off, and must stay off: ``*`` plus credentials is rejected by
            # every browser, and the public routes need no cookies.
            allow_credentials=False,
            allow_methods=["GET", "HEAD", "OPTIONS"],
            allow_headers=["*"],
            max_age=86400,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", ""))
        chosen = (
            self._permissive
            if any(path.startswith(prefix) for prefix in self.public_prefixes)
            else self._restricted
        )
        await chosen(scope, receive, send)


class MediaHeadersMiddleware:
    """Wrap the static mount so every panorama byte is usable as a cross-origin texture."""

    IMMUTABLE_SUFFIXES = (".jpg", ".jpeg", ".webp", ".png", ".ktx2", ".basis", ".sog", ".ply", ".spz")

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", "")).lower()
        immutable = path.endswith(self.IMMUTABLE_SUFFIXES)

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["access-control-allow-origin"] = "*"
                headers["cross-origin-resource-policy"] = "cross-origin"
                headers["timing-allow-origin"] = "*"
                headers["cache-control"] = (
                    "public, max-age=31536000, immutable" if immutable else "public, max-age=300"
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)


# --------------------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------------------


def _register_exception_handlers(app: FastAPI) -> None:
    """Everything a client sees is the ``ApiError`` shape from types.ts — no exceptions."""

    @app.exception_handler(ApiException)
    async def _handle_api_exception(_: Request, exc: ApiException) -> ORJSONResponse:
        return ORJSONResponse(exc.payload(), status_code=exc.status_code)

    @app.exception_handler(StitchError)
    async def _handle_stitch_error(_: Request, exc: StitchError) -> ORJSONResponse:
        return ORJSONResponse(
            _error_body(
                "stitch_failed",
                getattr(exc, "user_message_en", "The panorama could not be created."),
                getattr(exc, "user_message_tr", "Panorama oluşturulamadı."),
                str(exc) or None,
            ),
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(_: Request, exc: StarletteHTTPException) -> ORJSONResponse:
        code, message, message_tr = _STATUS_COPY.get(
            exc.status_code, ("http_error", "Request failed.", "İstek başarısız oldu.")
        )
        detail = exc.detail if isinstance(exc.detail, str) else None
        return ORJSONResponse(
            _error_body(code, detail or message, message_tr, None),
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> ORJSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(p) for p in first.get("loc", ())[1:]) or "body"
        reason = str(first.get("msg", "invalid value"))
        return ORJSONResponse(
            _error_body(
                "validation_error",
                f"'{location}' is invalid: {reason}.",
                f"'{location}' alanı geçersiz: {reason}.",
                "Check the request body against the API contract.",
            ),
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> ORJSONResponse:
        # The traceback goes to the log, never to the client: internal paths and SQL are not
        # the user's business, and leaking them is how a stack trace becomes a CVE.
        log.error("http.unhandled", error=str(exc), traceback=traceback.format_exc())
        return ORJSONResponse(
            _error_body(
                "internal_error",
                "Something went wrong on our side.",
                "Sunucu tarafında bir hata oluştu.",
                "Try again in a moment.",
            ),
            status_code=500,
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application."""
    settings = settings or get_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # noqa: ANN202 - FastAPI's own signature
        init_db()
        manager = get_job_manager()
        # The worker threads publish SSE events onto *this* loop; without capturing it here
        # every cross-thread publish would be a silent no-op.
        manager.bind_loop(asyncio.get_running_loop())
        reaped = manager.reap_orphans()
        # Parts nobody came back for. A failed finalise keeps them on purpose so a retry is
        # free, which means nothing else ever removes them — and an abandoned capture is half
        # a gigabyte of it.
        swept = sweep_staging(settings)
        # The health probes that import torch and pycolmap start warming now, so the
        # dashboard's first poll after a restart is not the request that pays for them.
        from biosphere.api.routes.health import start_warm_up

        start_warm_up(settings)
        storage = get_storage()
        log.info(
            "startup",
            app=settings.app_name,
            env=settings.env,
            media_root=str(settings.media_root),
            free_gb=round(storage.free_bytes() / (1024**3), 2),
            database=settings.database_url,
            cors_allow_origins=settings.cors_allow_origins,
            workers=settings.worker_concurrency,
            reaped_jobs=reaped,
            swept_uploads=swept,
        )
        try:
            yield
        finally:
            manager.shutdown(wait=False)
            log.info("shutdown")

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary="Phone video to embeddable 360 tour.",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    api = APIRouter(prefix=API_PREFIX)
    for router in PUBLIC_ROUTERS:
        api.include_router(router)
    # One dependency for every private router, declared here rather than on each handler.
    # A per-handler check is a check a new endpoint can be written without; this one is a
    # check a new endpoint has to be moved out of `PRIVATE_ROUTERS` to escape.
    for router in PRIVATE_ROUTERS:
        api.include_router(router, dependencies=[Depends(current_user)])
    app.include_router(api)

    # Only the DELIVERED media is exposed, and it is mounted a subtree at a time. Mounting the
    # whole media root would also publish the SQLite database, every raw upload and every
    # scratch directory sitting next to it, so each new deliverable earns its own mount rather
    # than widening this one.
    for name, directory in (
        ("media", settings.properties_dir),
        ("restyle_media", settings.restyle_dir),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        app.mount(
            f"{settings.media_url_prefix}/{directory.name}",
            MediaHeadersMiddleware(StaticFiles(directory=str(directory), check_dir=False)),
            name=name,
        )

    _register_exception_handlers(app)

    # Added last => outermost, so the access log covers CORS rejections too.
    app.add_middleware(
        ScopedCORSMiddleware,
        public_prefixes=(f"{API_PREFIX}/public", settings.media_url_prefix),
        allow_origins=list(settings.cors_allow_origins),
    )
    app.add_middleware(RequestContextMiddleware)
    return app


app = create_app()
