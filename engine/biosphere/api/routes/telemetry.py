"""Development-only sink for the phone's capture performance trace.

The guided capture runs on a mid-range Android handset that nobody on this project can
attach a profiler to, and two rounds of lag fixes have already shipped on reasoning rather
than measurement. ``web/src/lib/capture/perfTrace.ts`` closes that hole by having the phone
measure itself and post one small summary with ``navigator.sendBeacon``; this is the other
end of that wire.

**This endpoint is unauthenticated by construction.** ``sendBeacon`` cannot carry a
credential the page does not already hold in a cookie, and the whole point is that the trace
survives the tab being swiped away, when no code of ours is running to sign anything. An
unauthenticated write endpoint is acceptable *only* because it is switched off outside
development. Two guards enforce that:

  1. ``settings.env`` must name a development environment, otherwise the route answers 404 —
     not 403, because a 403 confirms the endpoint exists and invites someone to look for the
     flag that turns it on.
  2. The body is capped at 64 KiB and the cap is applied while streaming, not after reading.
     ``Content-Length`` is checked first as a cheap rejection, but it is a client-supplied
     number and a chunked request omits it entirely, so it can never be the only check.

The JSONL file is written under ``settings.media_root / "logs"``. That directory is
deliberately *not* under ``properties/``: the static mount in ``api.main`` publishes ``properties/``
to any origin, so a trace written there would be a public document containing the user's
full user-agent string.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import orjson
from fastapi import APIRouter, Request, Response
from fastapi.concurrency import run_in_threadpool

from biosphere.api.deps import SettingsDep
from biosphere.core.config import Settings
from biosphere.core.logging import get_logger
from biosphere.core.models import ApiException, utcnow

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

log = get_logger("biosphere.telemetry")

#: A summary is well under 4 KB. The cap is generous enough that a future field cannot
#: silently start failing, and small enough that the endpoint is not a disk-filling primitive.
MAX_BODY_BYTES = 64 * 1024

#: Rotated rather than truncated: the trace that explains the lag may be the one from
#: yesterday, and losing it to a rotation is worse than keeping 16 MB on disk. Two files is
#: the whole retention policy — an unbounded dev log lives on the same volume as the
#: panoramas, and filling that volume breaks the actual product.
MAX_LOG_BYTES = 8 * 1024 * 1024

TRACE_FILENAME = "capture-traces.jsonl"

#: Guards the append. ``O_APPEND`` is atomic for small writes on POSIX but carries no such
#: guarantee on Windows, which is this project's primary development platform.
_write_lock = threading.Lock()

#: Environments in which the endpoint is live. ``settings.env`` defaults to "development".
_DEV_ENVS = frozenset({"development", "dev", "local"})


def _is_development(settings: Settings) -> bool:
    return str(settings.env).strip().lower() in _DEV_ENVS


def _not_found() -> ApiException:
    return ApiException(
        "not_found",
        "The requested resource was not found.",
        "İstenen kayıt bulunamadı.",
        status_code=404,
    )


def _too_large() -> ApiException:
    return ApiException(
        "payload_too_large",
        f"A capture trace may not exceed {MAX_BODY_BYTES // 1024} KB.",
        f"Yakalama izi en fazla {MAX_BODY_BYTES // 1024} KB olabilir.",
        status_code=413,
    )


async def _read_capped(request: Request) -> bytes:
    """Read the body, refusing to buffer more than :data:`MAX_BODY_BYTES`."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise _too_large()
    buffer = bytearray()
    async for chunk in request.stream():
        buffer.extend(chunk)
        if len(buffer) > MAX_BODY_BYTES:
            # Raised mid-stream on purpose: a lying Content-Length must not be able to make
            # this handler allocate an arbitrary amount before the check runs.
            raise _too_large()
    return bytes(buffer)


def _trace_path(settings: Settings) -> Path:
    directory = settings.media_root / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / TRACE_FILENAME


def _append(settings: Settings, record: dict[str, Any]) -> None:
    """Append one line. Blocking disk IO, so callers run it off the event loop."""
    path = _trace_path(settings)
    line = orjson.dumps(record, default=str) + b"\n"
    with _write_lock:
        try:
            if path.exists() and path.stat().st_size + len(line) > MAX_LOG_BYTES:
                path.replace(path.with_name(path.name + ".1"))
        except OSError:
            # A rotation that fails is not a reason to lose the trace we are holding.
            pass
        with path.open("ab") as handle:
            handle.write(line)


def _log_fields(trace: dict[str, Any]) -> dict[str, Any]:
    """The handful of numbers worth having in the log line itself.

    The full document goes to the JSONL file, not into structlog: a 3 KB blob on one log
    line is unreadable in the console renderer and drowns the request log it sits in. These
    are the fields you grep for when deciding whether to go and read the file.
    """
    frames = trace.get("frames") if isinstance(trace.get("frames"), dict) else {}
    long_tasks = trace.get("longTasks") if isinstance(trace.get("longTasks"), dict) else {}
    device = trace.get("device") if isinstance(trace.get("device"), dict) else {}
    return {
        "label": trace.get("label"),
        "duration_ms": trace.get("durationMs"),
        "fps": frames.get("fps"),
        "frame_p50_ms": frames.get("p50Ms"),
        "frame_p95_ms": frames.get("p95Ms"),
        "frame_p99_ms": frames.get("p99Ms"),
        "hitches": frames.get("hitches"),
        "long_tasks": long_tasks.get("count"),
        "long_task_ms": long_tasks.get("totalMs"),
        "blocked_fraction": long_tasks.get("blockedFraction"),
        "stages": sorted(trace["stages"]) if isinstance(trace.get("stages"), dict) else None,
        "cores": device.get("hardwareConcurrency"),
        "user_agent": device.get("userAgent"),
    }


@router.post(
    "/capture",
    status_code=204,
    summary="Record a capture performance trace (development only)",
)
async def capture_trace(request: Request, settings: SettingsDep) -> Response:
    """Accept one ``PerfSummary`` from the phone.

    Returns 204 with no body because the sender is ``sendBeacon``, which discards the
    response entirely — anything rendered here would be written and never read.
    """
    if not _is_development(settings):
        raise _not_found()

    body = await _read_capped(request)
    try:
        trace = orjson.loads(body) if body else None
    except orjson.JSONDecodeError as exc:
        raise ApiException(
            "bad_request",
            "The trace body is not valid JSON.",
            "İz gövdesi geçerli JSON değil.",
            status_code=400,
        ) from exc
    if not isinstance(trace, dict):
        raise ApiException(
            "bad_request",
            "The trace body must be a JSON object.",
            "İz gövdesi bir JSON nesnesi olmalı.",
            status_code=400,
        )

    log.info("capture.trace", **_log_fields(trace))
    # The client's own clock is in ``trace["at"]`` and a phone's clock is routinely minutes
    # off; ``received_at`` is what orders the file.
    await run_in_threadpool(
        _append, settings, {"received_at": utcnow().isoformat(), "bytes": len(body), "trace": trace}
    )
    return Response(status_code=204)
