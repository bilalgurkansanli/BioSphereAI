"""Virtual staging: erase furniture from an equirectangular panorama.

**This is classical computer vision, not a generative model.** ``model_backed`` is therefore
reported as ``False`` on every response. The pipeline is:

    polygon (or an automatic proposal)  ->  raster mask  ->  dilate  ->  inpaint  ->  composite

Inpainting uses ``cv2.xphoto.inpaint`` with FSR — a frequency-selective reconstruction that
extends the surrounding texture's spectrum into the hole and looks markedly better than
diffusion-based fills on carpet, tiles and wood grain — and falls back to
``cv2.inpaint(..., INPAINT_TELEA)`` on failure or timeout.

FSR is *expensive*: measured on this machine, ``INPAINT_FSR_BEST`` costs roughly 5.5 ms per
hole pixel and ``INPAINT_FSR_FAST`` roughly 0.16 ms, both super-linear in the crop. A
whole-sofa mask on a 4096 px panorama would run for minutes, so the algorithm is chosen
against a wall-clock budget and enforced with a hard deadline rather than hoped about.

Where a model would slot in
---------------------------
:data:`INPAINT_BACKEND` is the seam. A LaMa (or SD-inpaint) backend and a SAM-based proposal
step implement the same two callables — ``(image, hole_mask, budget) -> (image, model_backed)``
and ``(image, strength) -> [Proposal]`` — and nothing else in the service changes; the
response already carries ``model_backed`` so the UI can label the result honestly.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
import numpy as np

from biosphere.api.schemas import HideFurnitureResponse, RemovedObject
from biosphere.core.logging import get_logger
from biosphere.core.models import ApiException, Scene
from biosphere.core.storage import LocalStorage, write_jpeg, write_png

__all__ = [
    "hide_furniture",
    "propose_regions",
    "INPAINT_BACKEND",
    "Proposal",
    "TIME_BUDGET_S",
]

log = get_logger("biosphere.ai.hide_furniture")

#: Wall-clock budget for one inpaint call. Also the hard deadline for the FSR thread.
TIME_BUDGET_S = 6.0

#: Measured cost per hole pixel (seconds). Used to pick an algorithm that can finish.
_FSR_BEST_S_PER_PX = 5.5e-3
_FSR_FAST_S_PER_PX = 1.6e-4

_LEVEL_PREFERENCE = ("hd", "uhd", "sd", "preview")


@dataclass(slots=True)
class Proposal:
    """One automatically detected removal candidate."""

    label: str
    confidence: float
    area_px: int
    mask: np.ndarray


# --------------------------------------------------------------------------------------
# Inpainting
# --------------------------------------------------------------------------------------


def _run_with_deadline(fn: Callable[[], Any], budget: float) -> Any | None:
    """Run ``fn`` on a helper thread and give up after ``budget`` seconds.

    OpenCV releases the GIL inside ``inpaint``, so the deadline is real. The thread is left
    to finish on its own (C++ code cannot be interrupted safely); it is a daemon, so it can
    never hold up shutdown, and its result is simply discarded.
    """
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - reported to the caller as a fallback
            box["error"] = exc

    thread = Thread(target=_target, name="inpaint", daemon=True)
    thread.start()
    thread.join(budget)
    if thread.is_alive() or "error" in box:
        return None
    return box.get("value")


def _classical_inpaint(
    image: np.ndarray, hole: np.ndarray, budget: float = TIME_BUDGET_S
) -> tuple[np.ndarray, bool]:
    """FSR with a TELEA fallback. ``hole`` is 255 where pixels must be replaced."""
    hole_px = int(np.count_nonzero(hole))
    if hole_px == 0:
        return image.copy(), False

    # xphoto's convention is the opposite of cv2.inpaint's: non-zero means *valid*.
    valid = cv2.bitwise_not(hole)

    algorithm: int | None = None
    if hole_px * _FSR_BEST_S_PER_PX <= budget:
        algorithm = cv2.xphoto.INPAINT_FSR_BEST
    elif hole_px * _FSR_FAST_S_PER_PX <= budget:
        algorithm = cv2.xphoto.INPAINT_FSR_FAST

    if algorithm is not None:
        def _fsr() -> np.ndarray:
            dst = np.zeros_like(image)
            cv2.xphoto.inpaint(image, valid, dst, algorithm)
            return dst

        result = _run_with_deadline(_fsr, budget)
        if result is not None and np.any(result):
            return result, False
        log.info("inpaint.fsr_fallback", hole_px=hole_px, budget=budget)

    radius = max(3, int(round(math.sqrt(hole_px) * 0.05)))
    return cv2.inpaint(image, hole, radius, cv2.INPAINT_TELEA), False


#: Swap-in point for a learned inpainter (LaMa/SD). Must return ``(image, model_backed)``.
INPAINT_BACKEND: Callable[..., tuple[np.ndarray, bool]] = _classical_inpaint


# --------------------------------------------------------------------------------------
# Automatic proposals
# --------------------------------------------------------------------------------------


def _label_for(w: int, h: int, bottom_frac: float, area_frac: float) -> str:
    """A deliberately small taxonomy — anything finer would be pretending."""
    aspect = w / max(h, 1)
    if aspect >= 2.2 and bottom_frac > 0.78:
        return "rug"
    if area_frac >= 0.012 and 0.35 <= aspect <= 2.8:
        return "furniture"
    return "clutter"


def propose_regions(image: np.ndarray, strength: float = 0.5, limit: int = 3) -> list[Proposal]:
    """Deterministically segment the lower half and rank removal candidates.

    Mean-shift filtering flattens texture into piecewise-constant regions, flood fill turns
    those into connected components, and the ranking is a weighted sum of three signals that
    actually correlate with "a thing standing on the floor":

    * **size** — a plausible object covers between 0.4% and 12% of the visible floor half;
    * **colour contrast against the floor** — measured in HSV against the dominant colour of
      the bottom rows, which in an indoor panorama *is* the floor;
    * **position** — objects meet the floor below the horizon, not at the nadir where the
      tripod/photographer sits.

    Fully deterministic: same panorama in, same proposals out.
    """
    full_h, full_w = image.shape[:2]
    horizon = full_h // 2
    lower = image[horizon:, :]
    if lower.size == 0:
        return []

    scale = min(1.0, 512.0 / float(full_w))
    small = cv2.resize(
        lower,
        (max(16, int(full_w * scale)), max(8, int(lower.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    h, w = small.shape[:2]

    filtered = cv2.pyrMeanShiftFiltering(small, sp=12, sr=26, maxLevel=1)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    floor_band = hsv[int(h * 0.86) :, :].reshape(-1, 3)
    floor_hsv = np.median(floor_band, axis=0) if floor_band.size else np.array([0.0, 0.0, 128.0])

    # Flood fill with a shared mask: already-labelled pixels block later fills, so the whole
    # sweep stays linear in the pixel count.
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    labels = np.zeros((h, w), np.int32)
    next_id = 0
    for y in range(0, h, 3):
        for x in range(0, w, 3):
            if ff_mask[y + 1, x + 1] != 0 or next_id >= 240:
                continue
            next_id += 1
            cv2.floodFill(
                filtered,
                ff_mask,
                (x, y),
                0,
                (9, 9, 9),
                (9, 9, 9),
                4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (next_id << 8),
            )
            labels[ff_mask[1:-1, 1:-1] == next_id] = next_id

    total = float(h * w)
    scored: list[tuple[float, Proposal]] = []
    for region_id in range(1, next_id + 1):
        region = labels == region_id
        area = int(region.sum())
        if area < 40:
            continue
        area_frac = area / total
        if area_frac < 0.002 or area_frac > 0.35:
            continue

        ys, xs = np.nonzero(region)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        bottom_frac = (y1 + 1) / h
        centre_frac = (ys.mean() + 0.5) / h

        size_score = float(np.clip(1.0 - abs(math.log(area_frac / 0.03)) / 2.6, 0.0, 1.0))

        mean_hsv = hsv[region].mean(axis=0)
        hue_delta = abs(float(mean_hsv[0]) - float(floor_hsv[0]))
        hue_delta = min(hue_delta, 180.0 - hue_delta)  # hue is circular
        colour_score = float(
            np.clip(
                hue_delta / 45.0
                + abs(float(mean_hsv[1]) - float(floor_hsv[1])) / 90.0
                + abs(float(mean_hsv[2]) - float(floor_hsv[2])) / 110.0,
                0.0,
                1.0,
            )
        )
        # Sitting on the floor: below the horizon, above the nadir smear.
        position_score = float(np.clip(1.0 - abs(centre_frac - 0.5) * 1.7, 0.0, 1.0))

        score = 0.40 * size_score + 0.40 * colour_score + 0.20 * position_score
        if score < 0.36 - 0.16 * float(strength):
            continue

        mask_small = (region.astype(np.uint8)) * 255
        mask_full = np.zeros((full_h, full_w), np.uint8)
        mask_full[horizon:, :] = cv2.resize(
            mask_small, (full_w, full_h - horizon), interpolation=cv2.INTER_NEAREST
        )
        scored.append(
            (
                score,
                Proposal(
                    label=_label_for(bw, bh, bottom_frac, area_frac),
                    confidence=round(float(np.clip(score, 0.05, 0.95)), 3),
                    area_px=int(np.count_nonzero(mask_full)),
                    mask=mask_full,
                ),
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [proposal for _, proposal in scored[:limit]]


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def _level_path(scene: Scene, storage: LocalStorage) -> tuple[Path, str]:
    """Resolve the working image: the ``hd`` rung, or the best one that exists."""
    scene_dir = storage.scene_dir(scene.property_id, scene.id)
    for level in _LEVEL_PREFERENCE:
        candidate = scene_dir / f"pano_{level}.jpg"
        if candidate.exists():
            return candidate, level
    raise ApiException(
        "scene_not_ready",
        "This scene has no panorama yet.",
        "Bu sahnenin panoraması henüz hazır değil.",
        hint="Wait for processing to finish, then try again.",
        status_code=409,
    )


def _imread(path: Path) -> np.ndarray:
    """Read an image without tripping over non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ApiException(
            "panorama_unreadable",
            "The stored panorama could not be read.",
            "Kayıtlı panorama okunamadı.",
            status_code=500,
        )
    return image


def _next_index(scene_dir: Path) -> int:
    return 1 + len(list(scene_dir.glob("hidden_*.jpg")))


def hide_furniture(
    scene: Scene,
    polygon: Sequence[tuple[float, float]],
    *,
    storage: LocalStorage,
    strength: float | None = None,
    budget: float = TIME_BUDGET_S,
) -> HideFurnitureResponse:
    """Remove the masked region (or the strongest automatic proposals) from a scene."""
    started = time.perf_counter()
    strength_value = 0.5 if strength is None else float(min(max(strength, 0.0), 1.0))

    source_path, level = _level_path(scene, storage)
    image = _imread(source_path)
    height, width = image.shape[:2]

    hole = np.zeros((height, width), np.uint8)
    removed: list[RemovedObject] = []

    if polygon and len(polygon) >= 3:
        # Polygon coordinates are defined against the hd rung (types.ts); if we are working
        # on a different rung because hd does not exist, scale them across.
        hd_path = storage.scene_dir(scene.property_id, scene.id) / "pano_hd.jpg"
        if level != "hd" and hd_path.exists():  # pragma: no cover - hd is always written
            reference_w = _imread(hd_path).shape[1]
            factor = width / float(reference_w)
        else:
            factor = 1.0
        pts = np.array(
            [[float(x) * factor, float(y) * factor] for x, y in polygon], dtype=np.float32
        )
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.fillPoly(hole, [np.round(pts).astype(np.int32)], 255)
        x, y, bw, bh = cv2.boundingRect(np.round(pts).astype(np.int32))
        area = int(np.count_nonzero(hole))
        removed.append(
            RemovedObject(
                label=_label_for(bw, bh, (y + bh) / height, area / float(height * width)),
                # The user drew this outline; there is no detector uncertainty to report.
                confidence=1.0,
                area_px=area,
            )
        )
    else:
        for proposal in propose_regions(image, strength_value):
            hole = cv2.bitwise_or(hole, proposal.mask)
            removed.append(
                RemovedObject(
                    label=proposal.label, confidence=proposal.confidence, area_px=proposal.area_px
                )
            )

    if not np.any(hole):
        raise ApiException(
            "nothing_to_remove",
            "No removable object was found; draw an outline around what should disappear.",
            "Kaldırılacak bir nesne bulunamadı; silinmesini istediğiniz alanı çizin.",
            hint="Send a polygon with at least three points.",
            status_code=422,
        )

    # Dilation matters: object boundaries carry contact shadows and colour bleed, and leaving
    # a one-pixel halo of sofa behind is more obvious than the fill itself.
    radius = max(2, int(round((3.0 + 14.0 * strength_value) * max(width / 4096.0, 0.25))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    hole = cv2.dilate(hole, kernel)

    # Inpaint only a padded crop: cost scales with the region, not the panorama.
    x, y, bw, bh = cv2.boundingRect(hole)
    pad = max(24, int(round(max(bw, bh) * 0.35)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + bw + pad), min(height, y + bh + pad)
    crop = np.ascontiguousarray(image[y0:y1, x0:x1])
    crop_hole = np.ascontiguousarray(hole[y0:y1, x0:x1])

    filled, model_backed = INPAINT_BACKEND(crop, crop_hole, budget)

    # Feathered composite so the patch does not show a hard rim.
    alpha = cv2.GaussianBlur(crop_hole.astype(np.float32) / 255.0, (0, 0), 1.5)[..., None]
    blended = (filled.astype(np.float32) * alpha + crop.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8
    )
    result = image.copy()
    result[y0:y1, x0:x1] = blended

    scene_dir = storage.scene_dir(scene.property_id, scene.id)
    index = _next_index(scene_dir)
    image_path = write_jpeg(scene_dir / f"hidden_{index}.jpg", result, 92)
    mask_path = write_png(scene_dir / f"hidden_{index}_mask.png", hole)

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    log.info(
        "hide_furniture.done",
        scene_id=scene.id,
        level=level,
        objects=len(removed),
        elapsed_ms=elapsed_ms,
    )
    return HideFurnitureResponse(
        scene_id=scene.id,
        image_url=storage.url_for(image_path),
        mask_url=storage.url_for(mask_path),
        removed_objects=removed,
        elapsed_ms=elapsed_ms,
        model_backed=model_backed,
    )
