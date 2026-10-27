"""Semantic furniture segmentation, through perspective windows onto the sphere.

Why this exists: colour is not enough, and that is measured
-----------------------------------------------------------
``empty_room`` finds furnishings by how far a pixel sits from the colour its surface shows
in most directions. On a procedurally generated room that reaches an F1 of about 0.6. On a
real, lived-in bedroom it inverts: the wall seed band is itself covered in shelves, boxes and
hanging clothes, so the "wall palette" learns the clutter, the actual wall then reads as
foreign, and the mask comes back as the room with object-shaped holes in it. Measured on the
reference capture: 52 % of the sphere selected, with the desk, the chair and the bed among
the few things NOT selected.

No amount of tuning fixes that, because the premise fails: in a cluttered room there is no
clean band of any surface to learn from. A white desk against a white wall is not separable
by colour under any threshold. What separates them is knowing what a desk is.

Why perspective windows rather than the panorama
------------------------------------------------
Every detector worth using was trained on ordinary photographs. An equirectangular canvas is
not one: straight edges bow, verticals converge toward the poles, and a sofa spanning 90
degrees of yaw is smeared across a quarter of the image. Feeding it in directly wastes the
model.

So the sphere is sampled the way a camera would see it — a ring of rectilinear views with a
normal field of view — the detector runs on those, and the instance masks are projected back
onto the sphere through the same geometry. Views overlap deliberately: an object cut in half
by one window is whole in its neighbour, and the union is what counts.

Two adapters, and the licence matters
--------------------------------------
:func:`available` reports which, if any, backend this machine can actually run. Neither is a
dependency of this project and nothing here downloads:

``torchvision`` — Mask R-CNN on COCO. BSD-licensed, weights included in that licence, which
    is the one to prefer for a product that ships closed. Needs the weights present locally.
``ultralytics`` — YOLO segmentation. Better and faster, and **AGPL-3.0**: linking it into a
    proprietary service obliges you to publish that service's source, unless you hold a
    commercial licence from them. That is a business decision, not a technical one, so this
    module will use it if it is installed and will never install it.

COCO has no class for a wall, a floor or a ceiling. Every instance an indoor detector finds
is therefore something standing in the room rather than the room itself, which is why the
class filter below is an exclusion list of nothing rather than an inclusion list of guesses.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from biosphere.ai import views
from biosphere.core.logging import get_logger

__all__ = [
    "available",
    "backend_name",
    "loaded_backend",
    "segment_photo_local",
    "segment_equirect",
    "segment_equirect_labels",
    "CURTAIN_CLASSES",
    "reset",
    "VIEW_PLAN",
    "SURFACE_OF_CLASS",
]

log = get_logger("biosphere.ai.furniture_seg")

#: Where the sphere is sampled from, as ``(pitch_deg, n_yaw)`` rows.
#:
#: Two rows, not three. Furniture stands on the floor and is seen below the horizon; the
#: upper row exists to catch what hangs on a wall and what sits on top of a wardrobe. Eight
#: yaws at a 90-degree field of view gives 45 degrees of overlap between neighbours, so
#: nothing is only ever seen at the edge of a window where a detector is least reliable.
VIEW_PLAN: tuple[tuple[float, int], ...] = ((-30.0, 8), (5.0, 8))

#: Rectilinear window size and field of view. 640 is what these detectors were trained near,
#: and smaller than the inpainter's window on purpose: a segmenter is deciding what a thing
#: IS, which does not improve with pixels the way redrawing it does, and Mask2Former costs
#: 0.83 s per window at this size against three times that at 1024.
VIEW_PIXELS = 640
VIEW_FOV_DEG = views.VIEW_FOV_DEG

# The projection lives in `views` so the segmenter and the inpainter cannot drift apart on
# it; a half-degree of disagreement would put masks slightly off the objects they are for.
_rotation = views.rotation
_intrinsics = views.intrinsics
_render_view = views.render_view
_accumulate = views.paint_mask

#: Detections below this score are noise. Deliberately low: a missed sofa is worse than a
#: spurious one, because the fill handles an over-large mask gracefully and cannot invent
#: the half of a sofa that was left behind.
MIN_SCORE = 0.35

#: Share of the windows that can see a direction which must agree before it is called a
#: furnishing. A half is the plain majority; the windows overlap threefold at most, so this
#: is "two of three" where coverage is densest and "one of one" where a direction is seen
#: only once, which is the right behaviour at both ends.
_VOTE_SHARE = 0.5

#: Stand-in class for "an object detector found something here". Never in `_ROOM_SHELL`,
#: and outside ADE20K's own 0-149 so it can never collide with a real label.
_ID_FURNITURE = 254

#: Masks are resolved on this canvas before being handed back, matching `empty_room`'s own
#: decision width so the two masks can simply be OR-ed.
_MASK_WIDTH = 1024

#: The scene-parsing checkpoint. ADE20K rather than COCO, and that is the whole design.
#:
#: COCO knows eighty *objects* and has no class for a wall, a floor or a ceiling, so with it
#: "what is the room" can only be inferred from what was not detected — and everything COCO
#: has never heard of (a stack of boxes, clothes on a hook, a shelf of clutter) silently
#: becomes room. ADE20K labels the shell explicitly, so the question this feature actually
#: asks — is this pixel the room, or something standing in it — is answered directly by the
#: model rather than reconstructed from its silence.
MASK2FORMER_MODEL = "facebook/mask2former-swin-large-ade-semantic"

#: ADE20K ids that ARE the room and must survive. Everything else in the 150 goes.
#:
#: Three groups, and each boundary is a product decision rather than a technical one:
#:
#: *Building fabric* — wall, floor, ceiling, window, door, column, stairs. Uncontroversial.
#:
#: *Window dressing* — curtains and blinds. They belong to the window rather than to the
#: furniture, and the reference this feature was measured against (Matterport's own emptied
#: room) keeps them. Removing them is also the expensive mistake: a curtain covers a whole
#: wall of a room, and behind it is a window nobody photographed, so taking it out replaces
#: real glass with invented glass and costs the inpainter the context it needs for the rest.
#:
#: *Fixtures* — a bath, a sink, a toilet, a radiator, a fitted counter, a chandelier. These
#: are plumbed or wired into the building and an estate agent's "empty room" keeps them; a
#: bathroom photographed without its toilet looks like a mistake, not like a bare room. Note
#: that `lamp` is NOT here while `light`, `chandelier` and `sconce` are: the first is a thing
#: standing on the floor, the others are fittings.
#:
#: *Outdoors* — sky, trees, a road, a neighbouring building. Seen through a window, and
#: erasing them would fill the view with invented wall.
_ROOM_SHELL: frozenset[int] = frozenset({
    # building fabric
    0, 3, 5, 8, 14, 38, 42, 53, 58, 59, 95, 96, 121,
    # window dressing
    18, 63,
    # fixtures that are plumbed, wired or fitted
    37, 45, 47, 49, 65, 70, 71, 73, 82, 85, 118, 124, 129, 134, 145, 146,
    # the world outside a window
    1, 2, 4, 6, 9, 11, 13, 16, 21, 25, 26, 29, 34, 46, 48, 52, 60, 61, 68,
    72, 79, 84, 87, 91, 94, 109, 113, 128, 140,
})

#: ADE20K class -> which of the engine's three surfaces it is, for the classes that are one.
#:
#: The fill needs this and cannot get it from the geometry alone: `empty_room`'s own shell is
#: a monotone three-band model of a column, which is right about a room's *shape* and wrong
#: wherever a wardrobe pushed the estimated junction up. The model says which pixels are
#: floor with no such ambiguity, so the texture the fill borrows always comes from the right
#: surface. A window and a door are grouped with the wall because they are in its plane.
SURFACE_OF_CLASS: dict[int, int] = {
    5: 0,                      # ceiling
    0: 1, 8: 1, 14: 1, 58: 1,  # wall, windowpane, door, screen door
    3: 2, 6: 2, 52: 2,         # floor, road and path (a floor seen through a doorway)
}

_YOLO_FILENAMES = ("yolo11x-seg.pt", "yolo11l-seg.pt", "yolo11m-seg.pt", "yolo11s-seg.pt", "yolo11n-seg.pt", "yolov8x-seg.pt", "yolov8n-seg.pt")

_lock = threading.Lock()
_state: dict[str, Any] = {"checked": False, "kind": None, "model": None, "device": "cpu"}


# ======================================================================================
# Backend discovery
# ======================================================================================


def _model_dirs() -> list[Path]:
    out: list[Path] = []
    try:
        from biosphere.core.config import settings

        if settings.ai_models_dir:
            out.append(Path(settings.ai_models_dir))
    except Exception:  # noqa: BLE001
        pass
    hub = os.environ.get("TORCH_HOME")
    out.append(Path(hub) / "hub" / "checkpoints" if hub else Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
    return out


def _find(names: tuple[str, ...]) -> Path | None:
    for directory in _model_dirs():
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _device() -> str:
    try:
        import torch

        from biosphere.core.config import settings

        if bool(settings.enable_gpu) and torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _load() -> tuple[str, Any, str] | None:
    """``(kind, model, device)`` for the best backend on this machine, or ``None``."""
    if _state["checked"]:
        return (_state["kind"], _state["model"], _state["device"]) if _state["model"] is not None else None

    with _lock:
        if _state["checked"]:
            return (_state["kind"], _state["model"], _state["device"]) if _state["model"] is not None else None
        _state["checked"] = True
        device = _device()

        # 1. Mask2Former on ADE20K: the only one of the three that knows what a wall is.
        try:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

            # Found, not fetched, unless downloads are allowed — the same rule every other
            # model in this package follows. Without it every engine start made a round of HEAD
            # requests to the Hub for weights that were already on disk.
            from biosphere.ai import sd_inpaint

            local_only = not sd_inpaint.downloads_allowed()
            processor = AutoImageProcessor.from_pretrained(MASK2FORMER_MODEL, local_files_only=local_only)
            network = Mask2FormerForUniversalSegmentation.from_pretrained(
                MASK2FORMER_MODEL, local_files_only=local_only
            )
            network.eval().to(device)
            _state.update({"kind": "mask2former", "model": (network, processor), "device": device})
            log.info("furniture_seg.ready", backend="mask2former", model=MASK2FORMER_MODEL, device=device)
            return "mask2former", (network, processor), device
        except Exception as exc:  # noqa: BLE001 - transformers is an optional extra
            log.info("furniture_seg.mask2former_unavailable", error=str(exc)[:160])

        # 2. Ultralytics, when the operator has chosen to install it. See the licence note.
        weights = _find(_YOLO_FILENAMES)
        if weights is not None:
            try:
                from ultralytics import YOLO

                model = YOLO(str(weights))
                _state.update({"kind": "ultralytics", "model": model, "device": device})
                log.info("furniture_seg.ready", backend="ultralytics", weights=str(weights), device=device)
                return "ultralytics", model, device
            except Exception as exc:  # noqa: BLE001
                log.info("furniture_seg.ultralytics_unavailable", error=str(exc)[:160])

        # 3. torchvision Mask R-CNN, weights strictly from the local cache.
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn

            if _find(("maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",)) is None:
                raise FileNotFoundError("mask r-cnn weights are not cached")
            model = maskrcnn_resnet50_fpn(weights="DEFAULT")
            model.eval().to(device)
            _state.update({"kind": "torchvision", "model": model, "device": device})
            log.info("furniture_seg.ready", backend="torchvision", device=device)
            return "torchvision", model, device
        except Exception as exc:  # noqa: BLE001
            log.info("furniture_seg.unavailable", reason=str(exc)[:160])

        return None


def available() -> bool:
    return _load() is not None


def backend_name() -> str | None:
    """Which backend would be used. LOADS IT to find out."""
    loaded = _load()
    return loaded[0] if loaded else None


def loaded_backend() -> str | None:
    """Which backend is already resident. Never loads one."""
    return _state["kind"] if _state["model"] is not None else None


def reset() -> None:
    with _lock:
        _state.update({"checked": False, "kind": None, "model": None, "device": "cpu"})


# ======================================================================================
# Sphere <-> perspective
# ======================================================================================


# ======================================================================================
# Inference
# ======================================================================================


def _labels_mask2former(model: Any, image: np.ndarray, device: str) -> np.ndarray:
    """ADE20K class id per pixel."""
    import torch

    network, processor = model
    size = image.shape[:2]
    inputs = processor(images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = network(**inputs)
    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[size])[0]
    return seg.cpu().numpy().astype(np.int32)


def _instances_ultralytics(model: Any, image: np.ndarray, device: str) -> np.ndarray:
    """Union of every instance mask the detector found, as uint8."""
    height, width = image.shape[:2]
    out = np.zeros((height, width), np.uint8)
    results = model.predict(image, verbose=False, conf=MIN_SCORE, device=device)
    for result in results:
        masks = getattr(result, "masks", None)
        if masks is None:
            continue
        for plane in masks.data.cpu().numpy():
            resized = cv2.resize(plane.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            out[resized > 0.5] = 255
    return out


def _instances_torchvision(model: Any, image: np.ndarray, device: str) -> np.ndarray:
    import torch

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
    with torch.no_grad():
        prediction = model([tensor])[0]
    out = np.zeros((height, width), np.uint8)
    for score, plane in zip(
        prediction["scores"].cpu().numpy(), prediction["masks"].cpu().numpy(), strict=False
    ):
        if score >= MIN_SCORE:
            out[plane[0] > 0.5] = 255
    return out


def segment_photo_local(image: np.ndarray) -> np.ndarray | None:
    """Label one ORDINARY PHOTOGRAPH with ADE20K class ids, on this machine.

    This is the whole of what a GPU has to be able to do for this feature, which is why it
    is a function over a plain photo rather than over a panorama: it is what
    :mod:`biosphere.ai.worker` exposes, so the heavy half can run on another machine without
    that machine knowing anything about rooms, sweeps or equirect canvases.

    The two object detectors cannot answer in this vocabulary — COCO has no class for a wall
    — so they answer in the only part of it they can: :data:`_ID_FURNITURE` where they found
    an instance and ``wall`` everywhere else. That is a coarse approximation and it is why
    Mask2Former is preferred; it is still better than nothing, because in an indoor room
    every COCO instance is something standing in it.
    """
    loaded = _load()
    if loaded is None:
        return None
    kind, model, device = loaded

    if kind == "mask2former":
        return _labels_mask2former(model, image, device)

    found = (
        _instances_ultralytics(model, image, device)
        if kind == "ultralytics"
        else _instances_torchvision(model, image, device)
    )
    labels = np.zeros(found.shape, np.int32)  # 0 is `wall`: "part of the room"
    labels[found > 0] = _ID_FURNITURE
    return labels


def segment_equirect(pano: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """``(furniture, surface)`` on a ``_MASK_WIDTH`` canvas, or ``None`` with no backend.

    ``surface`` is 0/1/2 for ceiling/wall/floor and 255 elsewhere. A missing backend is not
    an error: the caller falls back to the colour segmentation and reports
    ``model_backed: false``.
    """
    answered = segment_equirect_labels(pano)
    return None if answered is None else (answered[0], answered[1])


#: ADE20K ids the plane fill treats as a layer IN FRONT of a wall rather than as the wall:
#: curtain and blind. Held like every other part of the shell, and continued downward rather
#: than painted over when a removed chair stood in front of one — see ``room_planes``.
CURTAIN_CLASSES: frozenset[int] = frozenset({18, 63})


def segment_equirect_labels(pano: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(furniture, surface, classes)``: :func:`segment_equirect` plus the class of every pixel.

    ``classes`` is the ADE20K id each direction was given by the window that saw it MOST
    HEAD-ON (255 where no window looked). The emptying fill needs it and the vote cannot give
    it: a vote says "furnishing or not", while the fill has to tell wall paint from a door, a
    window and a curtain — all of them "not a furnishing", none of them a sample of the paint.
    """
    from biosphere.ai import backend  # noqa: PLC0415 - avoids a circular import at module scope

    width = _MASK_WIDTH
    height = width // 2
    votes = np.zeros((height, width), np.uint16)
    seen = np.zeros((height, width), np.uint16)
    surface_out = np.full((height, width), 255, np.uint8)
    classes_out = np.full((height, width), 255, np.uint8)
    centrality = np.full((height, width), -1.0, np.float32)
    K = _intrinsics(VIEW_PIXELS, VIEW_FOV_DEG)

    keep = np.zeros(256, bool)
    for class_id in _ROOM_SHELL:
        keep[class_id] = True

    answered = 0
    for pitch_deg, n_yaw in VIEW_PLAN:
        for index in range(n_yaw):
            yaw_deg = 360.0 * index / float(n_yaw)
            rotation = _rotation(yaw_deg, pitch_deg)
            view = _render_view(pano, rotation, K, VIEW_PIXELS)

            labels = backend.segment_photo(view)
            if labels is None:
                continue
            answered += 1
            clipped = np.clip(labels, 0, 255)

            views.paint_votes(votes, seen, (~keep[clipped]).astype(np.uint8) * 255, rotation, K)
            for class_id, which in SURFACE_OF_CLASS.items():
                plane = (clipped == class_id).astype(np.uint8) * 255
                if plane.any():
                    views.paint_mask(surface_out, plane, rotation, K, value=which)
            views.paint_labels(classes_out, centrality, np.minimum(clipped, 254).astype(np.uint8), rotation, K)

    if answered == 0:
        return None

    # A direction is a furnishing when most of the windows that could see it said so.
    out = np.zeros((height, width), np.uint8)
    looked = seen > 0
    out[looked & (votes.astype(np.float32) >= _VOTE_SHARE * seen.astype(np.float32))] = 255

    log.info(
        "furniture_seg.done",
        windows=answered,
        covered=round(float((out > 0).mean()), 4),
        unseen=round(float((~looked).mean()), 4),
    )
    return out, surface_out, classes_out
