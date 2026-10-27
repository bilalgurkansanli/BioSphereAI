"""Virtual staging over an ordinary photograph.

Three decisions here are product rather than plumbing.

**A photograph is addressed by its own bytes.** There is no ``RestylePhoto`` table. The id is
a hash of the uploaded file, so the same listing photo dragged in tomorrow is the same id and
lands on the same rendered styles — which is how people actually use a tool like this, and it
costs nothing to support. A row would add an ownership and expiry lifecycle to a feature that
does not have one yet, and rows outlive the files they point at.

**One style per request, not five.** A pass is seconds of a whole GPU and five of them is most
of a minute; a single endpoint returning all five would hold a request open for that long and
give the user nothing to look at until the last one landed. Asking per style lets the rail fill
in as the answers arrive, makes each result independently cacheable, and keeps every response
inside an ordinary timeout.

**The renders are serialised behind one lock.** Not per photograph, as ``empty_room`` does —
per PROCESS. Two diffusion passes on one card is not twice the throughput, it is an
out-of-memory error on an eight-gigabyte laptop GPU and a failed request for both callers. The
card does one thing at a time, so the queue is honest about that.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from biosphere.api import schemas as sc
from biosphere.api.deps import CurrentUser, SessionDep, SettingsDep, StorageDep, gpu_lock, require_scene
from biosphere.core.config import Settings
from biosphere.core.logging import get_logger
from biosphere.core.models import ApiException
from biosphere.core.storage import LocalStorage, write_jpeg

router = APIRouter(prefix="/ai/restyle", tags=["ai"])

log = get_logger("biosphere.api.restyle")

#: Longest side of the stored photograph and of every render made from it.
#:
#: NOT A BANDWIDTH DECISION. The diffusion pass runs at roughly a 512-pixel square of latent
#: (``restyle.WORK_PIXELS``) whatever it is handed, so the styled room carries about that much
#: real detail and everything above it is interpolation. Delivering a 4000-pixel file would be
#: delivering upscaled blur at twenty times the bytes, and it would invite the viewer to zoom
#: into detail that was never there. The original is stored at the same cap so the two halves
#: of the before/after slider are the same size — a slider between images of different
#: resolutions shows the seam as a sharpness step.
_MAX_SIDE = 1536

#: Refused before anything is decoded. A room photograph is under ten megabytes; anything much
#: larger is a mistake or a probe.
_MAX_BYTES = 32 * 1024 * 1024

#: Length of the content-addressed id. 64 bits of a blake2b over the file: collisions are not a
#: security boundary here (the id grants no authority) but a collision would show somebody
#: else's room, so it is sized well past birthday range for any plausible corpus.
_ID_CHARS = 16

_META = "meta.json"
_ORIGINAL = "original.jpg"

#: What produced a cached style, versioned; an older render is stale. 2: curtains held by
#: default, and the loft palette's glowing orange removed (see ``ai.restyle``).
_STYLE_ALGORITHM = 2

#: One model pass at a time, shared with every other route that reaches for the card. See
#: :data:`biosphere.api.deps.gpu_lock`.
_render_lock = gpu_lock


# --------------------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------------------


def _photo_id(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=_ID_CHARS // 2).hexdigest()


def _valid_id(photo_id: str) -> str:
    """Reject anything that could leave the restyle tree before it is used as a path."""
    cleaned = str(photo_id).strip().lower()
    if len(cleaned) != _ID_CHARS or any(c not in "0123456789abcdef" for c in cleaned):
        raise ApiException(
            "invalid_photo_id",
            "That photo id is not valid.",
            "Bu fotoğraf kimliği geçersiz.",
            status_code=422,
        )
    return cleaned


def _fit(image: np.ndarray, longest: int = _MAX_SIDE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = longest / float(max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def _meta_path(directory: Path) -> Path:
    return directory / _META


def _read_meta(directory: Path) -> dict[str, Any]:
    try:
        return json.loads(_meta_path(directory).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing or corrupt sidecar is not a failure
        return {}


def _write_meta(directory: Path, meta: dict[str, Any]) -> None:
    _meta_path(directory).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _ready_styles(directory: Path) -> list[str]:
    from biosphere.ai import restyle as engine

    return [s for s in engine.STYLE_IDS if (directory / f"{s}.jpg").is_file()]


def _photo_schema(storage: LocalStorage, photo_id: str, directory: Path) -> sc.RestylePhoto:
    meta = _read_meta(directory)
    return sc.RestylePhoto(
        photo_id=photo_id,
        original_url=storage.url_for(directory / _ORIGINAL),
        width=int(meta.get("width", 0)),
        height=int(meta.get("height", 0)),
        ready_styles=_ready_styles(directory),
    )


def _require_photo(storage: LocalStorage, settings: Settings, photo_id: str) -> Path:
    directory = settings.restyle_dir / photo_id
    if not (directory / _ORIGINAL).is_file():
        raise ApiException(
            "photo_not_found",
            "That photograph is no longer on the server.",
            "Bu fotoğraf sunucuda bulunamadı.",
            hint="Upload it again.",
            status_code=404,
        )
    return directory


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------


@router.get("/styles", response_model=sc.RestyleStylesResponse, summary="The decorating styles")
def list_styles() -> sc.RestyleStylesResponse:
    """The catalogue, and whether this deployment can render it.

    ``available`` is answered WITHOUT loading a model — the page that draws the style rail asks
    this on mount, and a health-shaped call that pulls four gigabytes onto a GPU would make
    opening the page the most expensive thing the service does. It reports that the weights and
    the card are present, which is the honest limit of what can be known for free.
    """
    from biosphere.ai import restyle as engine
    from biosphere.ai import sd_inpaint

    return sc.RestyleStylesResponse(
        styles=[
            sc.RestyleStyle(id=s.id, label_tr=s.label_tr, label_en=s.label_en) for s in engine.STYLES
        ],
        available=engine.is_loaded() or sd_inpaint.cuda_device() is not None,
    )


@router.post("/photos", response_model=sc.RestylePhoto, status_code=201, summary="Upload a room photo")
async def upload_photo(
    storage: StorageDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File()],
) -> sc.RestylePhoto:
    """Take one photograph and give back the id every later call uses.

    The upload is decoded here rather than sniffed: if OpenCV can turn the bytes into an image
    then it is an image, which is a stronger statement than any extension or magic-number check
    and is exactly the question the rest of this feature needs answered.
    """
    payload = await file.read(_MAX_BYTES + 1)
    if not payload:
        raise ApiException(
            "no_file", "No file was uploaded.", "Hiçbir dosya yüklenmedi.", status_code=422
        )
    if len(payload) > _MAX_BYTES:
        raise ApiException(
            "payload_too_large",
            f"That photograph is larger than {_MAX_BYTES // (1024 * 1024)} MB.",
            f"Bu fotoğraf {_MAX_BYTES // (1024 * 1024)} MB sınırından büyük.",
            status_code=413,
        )

    image = await run_in_threadpool(cv2.imdecode, np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ApiException(
            "unsupported_media",
            "That file is not an image this server can read.",
            "Bu dosya okunabilir bir görsel değil.",
            hint="Use a JPEG, PNG or WebP photograph.",
            status_code=415,
        )

    photo_id = _photo_id(payload)
    directory = storage.restyle_dir(photo_id)
    fitted = _fit(image)

    # Written every time, even on a repeat upload: it is cheap, and it repairs a directory
    # whose original was lost while its renders survived.
    await run_in_threadpool(write_jpeg, directory / _ORIGINAL, fitted, 92)
    await run_in_threadpool(
        _write_meta,
        directory,
        {
            "photo_id": photo_id,
            "width": int(fitted.shape[1]),
            "height": int(fitted.shape[0]),
            "source_width": int(image.shape[1]),
            "source_height": int(image.shape[0]),
            "uploaded_at": time.time(),
        },
    )
    log.info(
        "restyle.photo_stored",
        photo_id=photo_id,
        size=f"{fitted.shape[1]}x{fitted.shape[0]}",
        ready=len(_ready_styles(directory)),
    )
    return _photo_schema(storage, photo_id, directory)


@router.get("/photos/{photo_id}", response_model=sc.RestylePhoto, summary="An uploaded photo's state")
def get_photo(photo_id: str, storage: StorageDep, settings: SettingsDep) -> sc.RestylePhoto:
    """Which styles are already rendered. What a reloaded page asks before re-requesting any."""
    cleaned = _valid_id(photo_id)
    return _photo_schema(storage, cleaned, _require_photo(storage, settings, cleaned))


@router.post("", response_model=sc.RestyleResponse, summary="Furnish a photo in one style")
async def restyle(
    payload: sc.RestyleRequest, storage: StorageDep, settings: SettingsDep
) -> sc.RestyleResponse:
    """Render one style of one photograph, off the cache when it exists.

    The cache is the rendered file itself, which is right for a deterministic pipeline: the
    seed is a function of the photograph, the mask and the style id, so re-running would
    reproduce the same image at the cost of a whole GPU pass.
    """
    from biosphere.ai import restyle as engine

    photo_id = _valid_id(payload.photo_id)
    style = engine.style_by_id(payload.style)
    if style is None:
        raise ApiException(
            "unknown_style",
            f"'{payload.style}' is not a style. Use one of: {', '.join(engine.STYLE_IDS)}.",
            f"'{payload.style}' geçerli bir stil değil. Şunlardan biri olmalı: {', '.join(engine.STYLE_IDS)}.",
            status_code=422,
        )

    directory = _require_photo(storage, settings, photo_id)
    original_url = storage.url_for(directory / _ORIGINAL)
    target = directory / f"{style.id}.jpg"
    meta = _read_meta(directory)
    cached_meta = dict(meta.get("styles", {}).get(style.id, {}))

    def usable(entry: dict[str, Any]) -> bool:
        """The file on disk is this take of this style, made by the current algorithm."""
        return (
            int(entry.get("algorithm", 1)) == _STYLE_ALGORITHM
            and int(entry.get("variation", 0)) == payload.variation
        )

    if target.is_file() and not payload.refresh and usable(cached_meta):
        return sc.RestyleResponse(
            photo_id=photo_id,
            style=style.id,
            image_url=storage.url_for(target),
            original_url=original_url,
            protected_fraction=float(cached_meta.get("protected_fraction", 0.0)),
            elapsed_ms=int(cached_meta.get("elapsed_ms", 0)),
            cached=True,
            model_backed=bool(cached_meta.get("model_backed", True)),
            variation=payload.variation,
        )

    image = await run_in_threadpool(cv2.imread, str(directory / _ORIGINAL))
    if image is None:  # pragma: no cover - the file existed a moment ago
        raise ApiException(
            "photo_unreadable",
            "The stored photograph could not be read.",
            "Kayıtlı fotoğraf okunamadı.",
            status_code=500,
        )

    async with _render_lock:
        # Re-checked inside the lock: while this request waited, the caller ahead of it may
        # have been asking for exactly this style, and a second pass would spend a GPU minute
        # to produce a byte-identical file.
        meta = _read_meta(directory)
        cached_meta = dict(meta.get("styles", {}).get(style.id, {}))
        if target.is_file() and not payload.refresh and usable(cached_meta):
            return sc.RestyleResponse(
                photo_id=photo_id,
                style=style.id,
                image_url=storage.url_for(target),
                original_url=original_url,
                protected_fraction=float(cached_meta.get("protected_fraction", 0.0)),
                elapsed_ms=int(cached_meta.get("elapsed_ms", 0)),
                cached=True,
                model_backed=bool(cached_meta.get("model_backed", True)),
                variation=payload.variation,
            )
        result = await run_in_threadpool(
            lambda: engine.restyle_photo(
                image, style, window_dressing=payload.window_dressing, variation=payload.variation
            )
        )

    if result is None:
        raise ApiException(
            "restyle_unavailable",
            "Virtual staging is not available on this server.",
            "Sanal dekorasyon bu sunucuda kullanılamıyor.",
            hint="It needs a CUDA GPU with the diffusion and ControlNet weights present.",
            status_code=503,
        )

    await run_in_threadpool(write_jpeg, target, result.image, 92)
    meta = _read_meta(directory)
    meta.setdefault("styles", {})[style.id] = {
        "seed": result.seed,
        "elapsed_ms": result.elapsed_ms,
        "protected_fraction": round(result.protected_fraction, 5),
        "model_backed": result.model_backed,
        "algorithm": _STYLE_ALGORITHM,
        "variation": payload.variation,
        "rendered_at": time.time(),
    }
    await run_in_threadpool(_write_meta, directory, meta)

    return sc.RestyleResponse(
        photo_id=photo_id,
        style=style.id,
        image_url=storage.url_for(target),
        original_url=original_url,
        protected_fraction=result.protected_fraction,
        elapsed_ms=result.elapsed_ms,
        cached=False,
        model_backed=result.model_backed,
        variation=payload.variation,
    )


# --------------------------------------------------------------------------------------
# The sphere
# --------------------------------------------------------------------------------------

#: Where a scene's restyled panoramas live, one directory per style, beside ``staged/``.
_SCENE_DIRNAME = "restyled"

#: The restyled master is capped here, exactly as the emptied one is.
#:
#: Every pixel of it was regenerated by a model working through 512-pixel windows, so a
#: 8192-wide delivery would be an eightfold upscale of invented detail — more bytes for the
#: phone, no more information, and an invitation to zoom into something that was never there.
_SCENE_MAX_WIDTH = 4096


@router.post(
    "/scene",
    response_model=sc.RestyleSceneResponse,
    summary="Furnish a whole 360 scene in one style",
)
async def restyle_scene(
    payload: sc.RestyleSceneRequest, session: SessionDep, storage: StorageDep
,
    user: CurrentUser,
) -> sc.RestyleSceneResponse:
    """Restyle a scene's panorama and deliver it as its own resolution ladder.

    MINUTES, NOT SECONDS, AND IT SHOULD BE A JOB. Sixteen windows at a diffusion pass each is
    about two minutes on the reference card, so this sits under the same long deadline
    ``/ai/empty-room`` uses and carries the same debt: the engine already owns a queue with
    progress, reconnection and survival across a closed tab, and this belongs in it. Widening
    that queue past a stitch is a schema change rather than a refactor, so until it happens the
    honest thing is to say so here rather than to pretend a two-minute request is normal.

    The cache makes it bearable in the meantime: keyed on the source panorama's own mtime, so
    a reprocessed scene invalidates every style and an unchanged one answers instantly.
    """
    from biosphere.ai import empty_room as delivery
    from biosphere.ai import restyle as engine
    from biosphere.ai import restyle_pano

    scene = require_scene(session, payload.scene_id, user)
    if scene.status != "ready" or not scene.panorama:
        raise ApiException(
            "scene_not_ready",
            "This scene has no finished panorama yet.",
            "Bu sahnenin panoraması henüz hazır değil.",
            hint="Wait for processing to finish.",
            status_code=409,
        )
    style = engine.style_by_id(payload.style)
    if style is None:
        raise ApiException(
            "unknown_style",
            f"'{payload.style}' is not a style. Use one of: {', '.join(engine.STYLE_IDS)}.",
            f"'{payload.style}' geçerli bir stil değil. Şunlardan biri olmalı: {', '.join(engine.STYLE_IDS)}.",
            status_code=422,
        )
    property_id, scene_id = scene.property_id, scene.id
    session.commit()

    source_path, _level = delivery.source_level(scene, storage)
    target_dir = storage.scene_dir(property_id, scene_id) / _SCENE_DIRNAME / style.id
    key = {
        # 2: curtains held by default (the player sends no value), and the loft palette.
        "version": 2,
        "style": style.id,
        "source": source_path.name,
        "source_mtime": int(source_path.stat().st_mtime),
        "window_dressing": bool(payload.window_dressing),
    }

    cached = _read_meta(target_dir)
    if not payload.refresh and all(cached.get(k) == v for k, v in key.items()):
        levels = delivery.staged_levels(
            target_dir, storage, int(cached.get("width", 0)), int(cached.get("height", 0))
        )
        if levels is not None:
            return sc.RestyleSceneResponse(
                scene_id=scene_id,
                style=style.id,
                levels=levels,
                protected_fraction=float(cached.get("protected_fraction", 0.0)),
                elapsed_ms=int(cached.get("elapsed_ms", 0)),
                cached=True,
                model_backed=bool(cached.get("model_backed", True)),
            )

    def _run() -> Any:
        pano = cv2.imread(str(source_path))
        if pano is None:  # pragma: no cover - the file was stat'ed a moment ago
            # Raised rather than returned as None: "the panorama on disk is unreadable" and
            # "this box has no GPU" are different failures with different fixes, and giving
            # them one error code would send an operator to look at the wrong machine.
            raise ApiException(
                "panorama_unreadable",
                "The scene's panorama could not be read from disk.",
                "Sahnenin panoraması diskten okunamadı.",
                hint="Reprocess the scene.",
                status_code=500,
            )
        height, width = pano.shape[:2]
        if width > _SCENE_MAX_WIDTH:
            target_h = max(2, round(_SCENE_MAX_WIDTH * height / float(width)))
            pano = delivery.resize_cyclic(pano, _SCENE_MAX_WIDTH, target_h, cv2.INTER_AREA)
        return restyle_pano.restyle_equirect(
            pano, style, window_dressing=payload.window_dressing
        )

    async with _render_lock:
        cached = _read_meta(target_dir)
        if not payload.refresh and all(cached.get(k) == v for k, v in key.items()):
            levels = delivery.staged_levels(
                target_dir, storage, int(cached.get("width", 0)), int(cached.get("height", 0))
            )
            if levels is not None:
                return sc.RestyleSceneResponse(
                    scene_id=scene_id,
                    style=style.id,
                    levels=levels,
                    protected_fraction=float(cached.get("protected_fraction", 0.0)),
                    elapsed_ms=int(cached.get("elapsed_ms", 0)),
                    cached=True,
                    model_backed=bool(cached.get("model_backed", True)),
                )
        result = await run_in_threadpool(_run)

    if result is None:
        raise ApiException(
            "restyle_unavailable",
            "Virtual staging is not available on this server.",
            "Sanal dekorasyon bu sunucuda kullanılamıyor.",
            hint="It needs a CUDA GPU with the diffusion and ControlNet weights present.",
            status_code=503,
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    levels = await run_in_threadpool(delivery.write_levels, target_dir, result.panorama, storage)
    await run_in_threadpool(
        _write_meta,
        target_dir,
        {
            **key,
            "width": int(levels.width),
            "height": int(levels.height),
            "windows": result.windows,
            "elapsed_ms": result.elapsed_ms,
            "restyled_fraction": round(result.restyled_fraction, 5),
            "protected_fraction": round(result.protected_fraction, 5),
            "model_backed": result.model_backed,
            "rendered_at": time.time(),
        },
    )
    log.info(
        "restyle.scene_done",
        scene_id=scene_id,
        style=style.id,
        windows=result.windows,
        seconds=round(result.elapsed_ms / 1000.0, 1),
    )
    return sc.RestyleSceneResponse(
        scene_id=scene_id,
        style=style.id,
        levels=levels,
        protected_fraction=result.protected_fraction,
        elapsed_ms=result.elapsed_ms,
        cached=False,
        model_backed=result.model_backed,
    )
