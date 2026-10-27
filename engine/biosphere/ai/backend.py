"""Where the model work actually happens: a remote worker, this process, or nowhere.

The models do not have to live where the service does
------------------------------------------------------
Emptying a room is seconds of a whole GPU. The machine running this engine is also somebody's
desktop and their other projects, and its card may be the wrong one for the job — on the
reference workstation a single inpainting window took **185 seconds** on a 4 GB GTX 1650,
saturating both the card and its memory, while a laptop on the same desk would do it in a
few. Moving the *service* to that laptop would drag the media library, the database and the
job queue with it. Moving only the two model calls moves nothing else.

So this module is the seam, and it is deliberately narrow. The worker knows nothing about
panoramas, rooms or furniture: it answers "what is in this photograph" and "redraw this
photograph without the masked part". Everything BioSphere-specific — the window geometry, the
definition of a room's shell, the order the windows are filled in — stays here, where it can
be read and tested without a GPU.

Resolution order, and why local is last
----------------------------------------
1. ``settings.ai_worker_url`` — a worker on the network. Preferred whenever configured.
2. This process, if ``settings.ai_local_models`` allows it AND no worker is configured.
   Setting a worker URL turns the local path off rather than leaving it as a fallback: two
   copies of the same weights on two machines is exactly what the setting exists to prevent,
   and a silent fallback onto the desktop's own card is the failure it was added to stop.
3. Nothing, in which case the caller uses the classical path and the response says
   ``model_backed: false``. That is a degraded answer, never an error.

Which inpainter, and why ``auto`` will not pick the diffusion one
------------------------------------------------------------------
There are two, and ``auto`` chooses LaMa or nothing. That is not a view about model families,
it is what a real capture measured: :mod:`biosphere.ai.sd_inpaint` empties a synthetic room
beautifully and, on a photographed bedroom, draws a NEW BED into the bed-shaped hole — at an
18 % mask, unmoved by guidance from 6 to 15 or by a negative prompt naming the furniture. A
generative inpainter answers "what belongs in this hole"; it cannot be asked for an absence.
The full numbers are in that module's header, and they are the reason it is opt-in.

So: ``auto`` -> LaMa when its weights are here, otherwise nothing — and nothing is not a
failure, it is ``empty_room`` falling back to the classical per-surface fill, which has no
vocabulary for objects and therefore cannot invent one. On the reference capture that path
empties the room in 53 s, soft but honest.

``cyclic`` still decides what a request IS, and the diffusion path refuses one of them
outright. ``cyclic=False`` is a perspective window: square, 1024 px, an ordinary photograph.
``cyclic=True`` is the equirect canvas: 2:1, four thousand pixels wide, and nothing like the
512-square box every diffusion checkpoint was trained in — asked for it, a diffusion model
draws the same wall twice. LaMa has no training size and takes the canvas natively, so that
pass is LaMa's alone; it covers only the window rims, the poles and whatever no window reached.

``settings.ai_inpainter`` = ``sd`` overrides all of this for an operator who has looked at the
output and wants it anyway.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from biosphere.core.logging import get_logger

__all__ = [
    "segment_photo",
    "inpaint_photo",
    "local_inpaint",
    "local_inpainter",
    "ready",
    "available",
    "describe",
    "reset",
    "SEGMENT_TIMEOUT_S",
    "INPAINT_TIMEOUT_S",
    "READY_TIMEOUT_S",
]

log = get_logger("biosphere.ai.backend")

#: A segmentation is one forward pass; an inpaint may be dozens of diffusion steps. Both are
#: generous because the alternative to waiting is a run that silently falls back to the
#: classical path and delivers a worse room with no explanation.
SEGMENT_TIMEOUT_S = 60.0
INPAINT_TIMEOUT_S = 300.0

#: How long the worker's readiness probe may take.
#:
#: NOT A LIVENESS TIMEOUT, AND THE OLD FIVE SECONDS WAS THE WRONG NUMBER FOR IT. The worker's
#: ``/health`` reports what it can do by *checking*, which on the first call means pulling a
#: segmenter and an inpainter onto the card — tens of seconds for a diffusion checkpoint,
#: longer on a cold filesystem. Timing out at five turned "the worker is warming up" into
#: "there is no worker", and the room came back off the classical filler with
#: ``model_backed: false`` for no reason a user could see. This is called once per emptying,
#: never on a hot path, so a generous number costs nothing and a mean one costs the feature.
READY_TIMEOUT_S = 120.0

_lock = threading.Lock()
_state: dict[str, Any] = {"mode": None}


def _settings() -> Any:
    from biosphere.core.config import get_settings

    return get_settings()


def _worker_url() -> str:
    try:
        return str(_settings().ai_worker_url or "").strip().rstrip("/")
    except Exception:  # noqa: BLE001 - the engine must run without the API package
        return ""


def _local_allowed() -> bool:
    try:
        return bool(_settings().ai_use_local_models)
    except Exception:  # noqa: BLE001
        return True


def _preference() -> str:
    """``auto`` | ``sd`` | ``lama``. Unknown values fall back to ``auto`` rather than failing."""
    try:
        value = str(_settings().ai_inpainter or "auto").strip().lower()
    except Exception:  # noqa: BLE001
        return "auto"
    return value if value in ("auto", "sd", "lama") else "auto"


# ======================================================================================
# Choosing an inpainter
# ======================================================================================


def local_inpainter(*, cyclic: bool = False) -> str | None:
    """Name of the inpainter this machine would use for that shape of request. LOADS IT.

    ``None`` means neither is usable here, which is a fallback and not an error — see the
    module docstring for what each is asked and why ``cyclic`` decides it.
    """
    from biosphere.ai import lama, sd_inpaint

    preference = _preference()
    # The canvas belongs to LaMa whatever the preference says: a diffusion pass over a 2:1
    # canvas does not produce a worse room, it produces a duplicated one, and an operator who
    # set `sd` was choosing a better window fill rather than asking for that.
    if cyclic:
        return "lama" if preference in ("auto", "lama") and lama.available() else None

    if preference == "sd":
        return "sd" if sd_inpaint.available() else None

    # `auto` DOES NOT REACH FOR THE DIFFUSION MODEL, and that is a measurement rather than a
    # ranking of model families. Asked to empty a real bedroom it drew a new bed into the
    # bed-shaped hole at an 18 % mask, and somebody else's flat at 74 %; guidance from 6 to 15
    # and a negative prompt naming the furniture changed neither. It answers "what belongs
    # here", and a text prompt cannot ask a generative model for an absence. See
    # `sd_inpaint`'s header for the full numbers.
    #
    # So `auto` is LaMa or nothing, and nothing means `empty_room` uses the classical
    # per-surface fill -- which has no vocabulary for objects and therefore cannot invent one.
    # An operator who wants the diffusion path asks for it by name and looks at the result.
    return "lama" if lama.available() else None


def local_inpaint(image: np.ndarray, hole: np.ndarray, *, cyclic: bool = True) -> np.ndarray | None:
    """Run whichever local inpainter fits this request. ``None`` when none does.

    Shared by :func:`inpaint_photo`'s local branch and by the worker's ``/inpaint``, so the
    engine and the GPU box cannot drift apart on which model answers which question.
    """
    from biosphere.ai import lama, sd_inpaint

    chosen = local_inpainter(cyclic=cyclic)
    if chosen is None:
        return None
    module = sd_inpaint if chosen == "sd" else lama
    result = module.inpaint(image, hole, cyclic=cyclic)
    if result is not None:
        return result[0]

    # A diffusion pass that fell over -- almost always out of memory on a small card -- is
    # worth one retry through LaMa rather than a hole left in the room. The other direction
    # is not symmetric: LaMa failing means its weights are gone, and SD would be loading a
    # second model into a process that just ran out of room.
    if chosen == "sd" and lama.available():
        log.warning("backend.sd_failed_falling_back", to="lama")
        fallback = lama.inpaint(image, hole, cyclic=cyclic)
        if fallback is not None:
            return fallback[0]
    return None


# ======================================================================================
# Remote
# ======================================================================================


def _post(path: str, files: dict[str, tuple[str, bytes, str]], timeout: float) -> bytes | None:
    """One call to the worker. Returns the raw body, or ``None`` if it could not be made."""
    import httpx

    url = f"{_worker_url()}{path}"
    try:
        response = httpx.post(url, files=files, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - a worker that is off is not an error here
        log.warning("backend.worker_unreachable", url=url, error=str(exc)[:160])
        return None
    if response.status_code != 200:
        log.warning("backend.worker_error", url=url, status=response.status_code, body=response.text[:200])
        return None
    return response.content


def _encode(image: np.ndarray, suffix: str = ".png") -> bytes:
    import cv2

    ok, buffer = cv2.imencode(suffix, image)
    if not ok:  # pragma: no cover - only on a corrupt array
        raise ValueError(f"could not encode {suffix}")
    return buffer.tobytes()


def _decode(data: bytes, flags: int) -> np.ndarray | None:
    import cv2

    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


# ======================================================================================
# The two calls
# ======================================================================================


def segment_photo(image: np.ndarray) -> np.ndarray | None:
    """ADE20K class ids for one ordinary photograph, or ``None`` with no backend.

    Class ids rather than a furniture mask on purpose: what counts as "the room" is a
    product decision (see ``furniture_seg._ROOM_SHELL``) and belongs in the product, not in
    a GPU service that should be replaceable without renegotiating it.
    """
    if _worker_url():
        body = _post(
            "/segment",
            {"image": ("view.jpg", _encode(image, ".jpg"), "image/jpeg")},
            SEGMENT_TIMEOUT_S,
        )
        if body is None:
            return None
        labels = _decode(body, 0)  # single channel
        return None if labels is None else labels.astype(np.int32)

    if not _local_allowed():
        return None
    from biosphere.ai import furniture_seg

    return furniture_seg.segment_photo_local(image)


def inpaint_photo(image: np.ndarray, hole: np.ndarray, *, cyclic: bool = True) -> np.ndarray | None:
    """Redraw ``image`` without the masked part, or ``None`` with no backend."""
    if _worker_url():
        body = _post(
            "/inpaint",
            {
                "image": ("image.png", _encode(image, ".png"), "image/png"),
                "mask": ("mask.png", _encode(hole, ".png"), "image/png"),
                "cyclic": ("cyclic", b"1" if cyclic else b"0", "text/plain"),
            },
            INPAINT_TIMEOUT_S,
        )
        if body is None:
            return None
        return _decode(body, 1)  # colour

    if not _local_allowed():
        return None
    return local_inpaint(image, hole, cyclic=cyclic)


# ======================================================================================
# Reporting
# ======================================================================================


def describe() -> dict[str, Any]:
    """What this process will do, WITHOUT loading anything to find out.

    `/health` is polled every few seconds by the dashboard. An answer that pulled a
    900 MB segmenter onto the GPU to produce it would make the health check the single
    most expensive request in the service.
    """
    worker = _worker_url()
    if worker:
        return {"mode": "worker", "url": worker}
    if not _local_allowed():
        return {"mode": "off"}

    from biosphere.ai import furniture_seg, lama, sd_inpaint

    return {
        "mode": "local",
        # Already-loaded state only; no call here constructs a model.
        "segmenter": furniture_seg.loaded_backend(),
        "inpainter": "sd" if sd_inpaint.is_loaded() else ("lama" if lama.is_loaded() else None),
        "sd_checkpoint": sd_inpaint.model_ref(),
        "preference": _preference(),
        "weights_found": lama.model_path() is not None,
    }


def ready() -> bool:
    """True when both halves can run, checked as cheaply as the mode allows.

    For a worker this is one HTTP call. For the local path it does load the models, which is
    right: the only caller is about to use them, and finding out now rather than half way
    through a room is the difference between a clean fallback and a half-emptied panorama.
    """
    if _worker_url():
        import httpx

        try:
            response = httpx.get(f"{_worker_url()}/health", timeout=READY_TIMEOUT_S)
            body = response.json() if response.status_code == 200 else {}
        except Exception:  # noqa: BLE001
            return False
        return bool(body.get("segment")) and bool(body.get("inpaint"))

    if not _local_allowed():
        return False
    from biosphere.ai import furniture_seg

    # An inpainter for the WINDOWS is what makes a run model-backed. The canvas pass is a few
    # percent of the sphere and `empty_room` already copes with it not happening, so requiring
    # LaMa here as well would turn a fully diffusion-filled room into a classical one over a
    # cleanup pass nobody would have noticed.
    return furniture_seg.available() and local_inpainter(cyclic=False) is not None


#: Kept as the older name; `ready` says better what it costs.
available = ready


def reset() -> None:
    with _lock:
        _state["mode"] = None
