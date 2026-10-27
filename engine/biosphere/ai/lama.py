"""LaMa — the learned inpainter, wired in as ``empty_room``'s fill backend.

Why a model here and classical CV everywhere else in this package
-----------------------------------------------------------------
The rest of the engine is deterministic geometry, and deliberately so: a panorama's poses,
seams and exposure are *measurements*, and a network that guesses them would be guessing at
something the data already answers. Emptying a room is the opposite kind of problem. The
pixels behind a sofa were never photographed, so there is nothing to measure — every method
is inventing, and the only question is which invention looks like a room.

Classical filling answers that badly, and the failure has a name a user gave it: it looks
like the furniture was *painted over*. That is precisely what a normalised-convolution
pyramid does — it produces the smooth field the surrounding pixels imply, so a sofa becomes
a sofa-shaped wash of wall colour. Copying texture along a row fixes the grain and not the
structure: a skirting board still stops at one edge of the hole and restarts at the other.

LaMa is built for exactly this case. Its fast Fourier convolutions have a receptive field
covering the whole image from the first layer, which is why it continues long structures —
a skirting board, a floorboard, the line where a wall meets a floor — across holes far
larger than any patch-based method can bridge.

Offline by construction
-----------------------
Weights are only ever *found*, never fetched: :func:`_locate` looks in the configured models
directory and then in the torch hub cache, and returns ``None`` if neither has them.
``settings.ai_allow_downloads`` is not consulted because nothing here downloads. A missing
model is not an error — :data:`biosphere.ai.empty_room.FILL_BACKEND` falls back to the
classical filler and the response says ``model_backed: false``, which is the whole reason
that field is in the contract.

The equirect wrap
-----------------
The canvas is cyclic in yaw and the network is not. Every call is therefore made on a
wrap-padded image and cropped back, so a hole straddling column 0 is inpainted with its real
neighbours on both sides rather than against a border. The FFC layers' own FFT assumes
periodicity, which makes this the natural framing for them as well as the correct one.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from biosphere.core.logging import get_logger

__all__ = ["available", "is_loaded", "inpaint", "model_path", "reset"]

log = get_logger("biosphere.ai.lama")

#: Filenames we recognise, in preference order.
_FILENAMES = ("big-lama.pt", "lama.pt", "big-lama-regular.pt")

#: Long edge the network actually runs at, per device.
#:
#: MEASURED ON THE REFERENCE MACHINE (GTX 1650, 4 GB): 1024x2048 costs 2.0 s and 1.27 GB,
#: 1536x3072 costs 8.2 s and 2.6 GB. The larger one is chosen for CUDA because the output is
#: then upscaled by only 1.33x to reach a 4096 master, and a synthesised region that has been
#: enlarged four times looks exactly like what the user objected to in the first place.
#:
#: On CPU the same call is roughly thirty times slower, so the size drops to something that
#: finishes inside a request rather than inside a coffee break.
_LONG_EDGE = {"cuda": 3072, "cpu": 1024}

#: The network needs both sides divisible by this.
_STRIDE = 8

#: Wrap padding, as a fraction of the width. Enough that the seam has real content on both
#: sides; small enough not to buy a third of a second of extra convolution.
_WRAP_FRACTION = 0.06

_lock = threading.Lock()
_state: dict[str, Any] = {"module": None, "device": None, "checked": False, "path": None}


def _locate() -> Path | None:
    """Find the weights on disk. Never fetches anything."""
    candidates: list[Path] = []
    try:
        from biosphere.core.config import settings

        if settings.ai_models_dir:
            candidates.append(Path(settings.ai_models_dir))
    except Exception:  # noqa: BLE001 - the engine must import without the API package
        pass

    hub = os.environ.get("TORCH_HOME")
    candidates.append(Path(hub) / "hub" / "checkpoints" if hub else Path.home() / ".cache" / "torch" / "hub" / "checkpoints")

    for directory in candidates:
        for name in _FILENAMES:
            path = directory / name
            if path.is_file():
                return path
    return None


def model_path() -> Path | None:
    return _locate()


def _load() -> tuple[Any, str] | None:
    """Load once per process. Returns ``(module, device)`` or ``None``."""
    if _state["checked"]:
        module = _state["module"]
        return (module, _state["device"]) if module is not None else None

    with _lock:
        if _state["checked"]:
            module = _state["module"]
            return (module, _state["device"]) if module is not None else None
        _state["checked"] = True

        path = _locate()
        if path is None:
            log.info("lama.unavailable", reason="weights not found on disk")
            return None
        try:
            import torch
        except Exception as exc:  # noqa: BLE001 - torch is an optional extra
            log.info("lama.unavailable", reason=f"torch missing: {exc}")
            return None

        device = "cpu"
        try:
            from biosphere.core.config import settings

            gpu_allowed = bool(settings.enable_gpu)
        except Exception:  # noqa: BLE001
            gpu_allowed = True
        try:
            if gpu_allowed and torch.cuda.is_available():
                device = "cuda"
        except Exception:  # noqa: BLE001 - a broken driver must not break the endpoint
            device = "cpu"

        try:
            module = torch.jit.load(str(path), map_location=device).eval()
        except Exception as exc:  # noqa: BLE001
            log.warning("lama.load_failed", path=str(path), error=str(exc))
            return None

        _state.update({"module": module, "device": device, "path": path})
        log.info("lama.ready", path=str(path), device=device)
        return module, device


def available() -> bool:
    """True when a call to :func:`inpaint` will actually use the network. LOADS IT."""
    return _load() is not None


def is_loaded() -> bool:
    """Whether the weights are already resident. Never loads them."""
    return _state["module"] is not None


def reset() -> None:
    """Drop the loaded model. For tests, and for freeing VRAM in a long-lived process."""
    with _lock:
        _state.update({"module": None, "device": None, "checked": False, "path": None})


def _wrap_pad(image: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return image
    return np.concatenate([image[:, -pad:], image, image[:, :pad]], axis=1)


def inpaint(
    image: np.ndarray, hole: np.ndarray, *, cyclic: bool = True
) -> tuple[np.ndarray, bool] | None:
    """Fill ``hole`` (255 = replace). ``None`` when the model is absent.

    ``cyclic`` says whether the image wraps horizontally. TRUE FOR AN EQUIRECT AND FALSE FOR
    A PERSPECTIVE WINDOW, and the difference is not cosmetic: wrap-padding a window glues the
    right-hand side of a room to its left, so the network is asked to continue a wall into a
    completely different one. The equirect genuinely does wrap, and there padding it is what
    stops a hole on the seam being inpainted against a border.

    The returned image is the same size as the input. Only the hole is synthesised; the
    caller still composites, because the network's output differs everywhere by a fraction of
    a grey level and the panorama outside the mask must stay byte-identical.
    """
    loaded = _load()
    if loaded is None:
        return None
    module, device = loaded

    import torch

    height, width = image.shape[:2]
    if not np.any(hole):
        return image.copy(), True

    long_edge = _LONG_EDGE.get(device, 1024)
    scale = min(1.0, long_edge / float(max(width, height)))
    work_w = max(_STRIDE, int(round(width * scale)))
    work_h = max(_STRIDE, int(round(height * scale)))

    small = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else image
    small_hole = (
        cv2.resize(hole, (work_w, work_h), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else hole
    )

    left = int(round(work_w * _WRAP_FRACTION)) if cyclic else 0
    if left:
        small = _wrap_pad(small, left)
        small_hole = _wrap_pad(small_hole, left)

    # Both sides have to land on the network's stride. The make-up goes on the right and the
    # bottom with replicated edges, so the crop back is a fixed offset rather than arithmetic
    # that has to be re-derived — and getting that wrong is a tensor-size error deep inside
    # a traced graph, which is exactly how this was found.
    padded_h, padded_w = small.shape[:2]
    extra_w = (-padded_w) % _STRIDE
    extra_h = (-padded_h) % _STRIDE
    if extra_w or extra_h:
        small = cv2.copyMakeBorder(small, 0, extra_h, 0, extra_w, cv2.BORDER_REPLICATE)
        small_hole = cv2.copyMakeBorder(small_hole, 0, extra_h, 0, extra_w, cv2.BORDER_CONSTANT, value=0)

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    mask = torch.from_numpy((small_hole > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            out = module(tensor, mask)
    except Exception as exc:  # noqa: BLE001 - OOM on a small card is a real possibility
        log.warning("lama.inference_failed", error=str(exc)[:200], size=f"{small.shape[0]}x{small.shape[1]}")
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return None

    result = out.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    del out, tensor, mask
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    filled = cv2.cvtColor((result * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    filled = filled[:work_h, left : left + work_w]
    if (filled.shape[1], filled.shape[0]) != (width, height):
        filled = cv2.resize(filled, (width, height), interpolation=cv2.INTER_CUBIC)
    return filled, True
