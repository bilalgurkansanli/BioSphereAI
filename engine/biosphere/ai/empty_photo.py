"""Taking the furniture out of an ordinary photograph.

The sibling of :mod:`biosphere.ai.empty_room`, and much smaller, because almost everything hard
about that module is hard *because the canvas is a sphere*. On an equirect a skirting board is a
sine wave, the seam column wraps, the poles cover no solid angle, and no inpainter has ever seen
a room drawn that way — so that module has to model the room as a monotone three-band
segmentation of every column, fill per surface, and treat the caps separately.

A photograph is none of those things. It is the thing every model here was trained on. So this
module is two steps:

1. **Which pixels are furniture** — :func:`biosphere.ai.restyle.read_room`, the same function
   the restyle path uses. That shared definition is the point rather than a convenience: a
   curtain is protected in both features or in neither, and two modules that each decided for
   themselves would eventually disagree in front of a user who had just used the other one.

2. **Fill the hole** — :func:`biosphere.ai.backend.inpaint_photo` with ``cyclic=False``, which
   is LaMa on a machine that has it. Reconstruction rather than generation, deliberately, and
   for the reason ``sd_inpaint`` measured: asked for a bed-shaped hole in a bedroom, a diffusion
   inpainter draws a bed. That is the mechanism the restyle feature is built on and exactly the
   failure this one must not have.

What the fill actually produces, measured
------------------------------------------
MEASURED on two 90-degree crops of a real stitched bedroom panorama:

  hole 21.6 %   bed and chair gone, curtains and walls intact, and the floor where the bed
                stood comes back as a grey wash rather than as floorboards.
  hole 33.4 %   a dark smear across the corner the chest of drawers occupied.

Swept and RULED OUT as causes: LaMa's working resolution (384, 512, 768 and 1024 all give the
same smear), whether the masked region is zeroed before the call (the traced model does it
itself), and filling the hole in connected components smallest-first so each piece has more
context — the pattern ``room_fill`` uses on the sphere, which here found only three components
and changed nothing.

So this is the fill's ceiling on this material, not a bug in the mask — which is accurate to
the pixel; see ``scripts/bench_empty_photo.py``. It is also the SAME ceiling ``/ai/empty-room``
already ships on the sphere, since both end in the same inpainter. That module's header states
the trade in the words a user gave it, and they hold here: soft is a defect a viewer forgives,
where an invented wardrobe is a misrepresentation.

The honest instrument is therefore not a threshold, it is the comparison: the UI puts this
result behind a before/after slider and prints :attr:`EmptyPhotoOutput.removed_fraction`, so
the person deciding whether to publish it can see exactly what was replaced.
:data:`MAX_HOLE_FRACTION` only catches the case that is hopeless before it starts.

An ALREADY EMPTY room is not a failure either. ``read_room`` says so itself by coming back in
``stage`` mode, and this module returns ``removed_fraction = 0`` with the photograph unchanged
instead of inventing work.

The mask is grown, the fill is not replaced
-------------------------------------------
:func:`complete_mask` grows ``read_room``'s furnishings over what the segmenter called wall or
floor but plainly is not — the things standing on a removed nightstand, the bed's shadow, a dark
bag — because a fill can only be as good as what it is asked to remove, and measured on the
reference photograph those were the things left hanging in the air.

The panorama's plane fill (:mod:`biosphere.ai.room_planes`) is NOT used here, and that was
measured rather than assumed. A photograph has to be calibrated before its floor is a plane, and
from its own lines (vertical and horizontal vanishing points) five perspective crops of the
reference bedroom, whose true cameras were known, came back 3-8 degrees out in pitch and up to
60 % out in focal length; in three of the five not one stretch of floor line was in view to
place a wall by. A wrong floor plane draws the floor up the wall. Until a photograph can be
calibrated reliably — a phone's own gravity and lens in the upload would do it — LaMa stays.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from biosphere.core.logging import get_logger

__all__ = ["empty_photo", "EmptyPhotoOutput", "MAX_HOLE_FRACTION"]

log = get_logger("biosphere.ai.empty_photo")

#: Above this share of the frame the fill is refused rather than attempted.
#:
#: NOT A PERFORMANCE LIMIT, and deliberately not tight. LaMa reconstructs a hole from what
#: surrounds it, and past roughly half the frame there is not enough photograph left to
#: reconstruct from — what comes back is a smooth field with a room's colours, which reads as a
#: damaged photograph rather than as an empty room.
#:
#: It is not set where the quality starts to fall off, because quality falls off gradually and
#: a threshold there would refuse pictures a user might well accept. It is set where the answer
#: stops being about this room at all. Everything between is reported honestly and shown beside
#: the original — see the fill's measured behaviour in the module docstring.
MAX_HOLE_FRACTION = 0.55

#: Whether the hole is grown over what the segmenter called room but is not (:func:`complete_mask`).
#: A switch only so ``scripts/bench_empty_photo.py`` can show the two masks side by side.
COMPLETE_MASK = True

#: How far the composite feathers at the hole's edge, as a fraction of the long side.
_FEATHER = 0.003


@dataclass(slots=True)
class EmptyPhotoOutput:
    """The bare room, and an honest account of how much of it was reconstructed."""

    image: np.ndarray
    mask: np.ndarray
    #: Share of the frame that was replaced. The headline number: it is how much of the picture
    #: is now a reconstruction rather than a photograph.
    removed_fraction: float
    #: What the segmenter named on the way past. Useful to show, and the same list the restyle
    #: path builds its prompt from.
    objects: tuple[str, ...]
    elapsed_ms: int
    #: True when a learned inpainter ran. False never ships today — there is no classical
    #: fallback here — but the field exists so the API answer carries the same honest flag every
    #: other AI route does.
    model_backed: bool = True


#: Chroma (OpenCV Lab units) from the room's own wall or floor colour, and L below it, at which
#: a pixel the segmenter called wall or floor is something standing on it. The same two tests,
#: and the same reasons, as ``empty_room._OUTLIER_DE``; measured on the reference bedroom's
#: photographs the floor's own variation is under 6 in chroma and 15 in L.
_PHOTO_CHROMA = 14.0
_PHOTO_DARK = 25.0
#: Floor beside what is removed and this much darker than the room's floor is its shadow.
_PHOTO_SHADOW = 10.0
#: What stands ON a removed piece — a lamp on a nightstand, a bottle on a desk — sits in a band
#: just above it. A share of the frame height: 8 % is a bedside lamp seen across a bedroom.
_ON_TOP_BAND = 0.08
#: Largest share of the frame ONE piece of "wall" the tests add may cover. What stands on a
#: removed nightstand or hangs above it is small; a large stretch of "wall" that fails the paint
#: test beside a removed piece is an opening or a fitting the segmenter did not name. Measured on
#: the reference dresser photograph, whose doorway the segmenter called wall: the doorway (2.5 %
#: of the frame) and the whiteboard went with the dresser and came back a grey smear; at 1.2 %
#: both stay, and the reference bedroom photograph's completion is unchanged.
_WALL_PIECE_MAX = 0.012


def complete_mask(image: np.ndarray, hole: np.ndarray, masks: object) -> np.ndarray:
    """Grow the hole over what the segmenter called room but plainly is not.

    MEASURED ON THE REFERENCE PHOTOGRAPH: the things a nightstand carried were called "wall",
    so they stayed when the nightstand went and hung in the air above where it had stood; and the
    floor in the bed's shadow stayed dark when the bed went, a grey halo in its shape. Three
    tests catch those, each only where it touches what is already being removed:

    * colour — wall or floor far from the room's own wall or floor colour (a dark bag, a rug);
    * shadow — floor beside the hole and darker than the room's floor;
    * on top — anything in a band just above the hole that is not the wall's own paint.

    Nothing the product holds (a window, a door, a curtain, a radiator) is ever added, and no
    single piece of "wall" larger than :data:`_WALL_PIECE_MAX` — a doorway the segmenter called
    wall fails the paint test too, and it is the room's.
    """
    labels = getattr(masks, "labels", None)
    held = getattr(masks, "held", None)
    surfaces = getattr(masks, "surfaces", None)
    if labels is None or held is None or not surfaces:
        return hole
    height, width = hole.shape[:2]
    removed = hole > 127
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float32)
    wall = np.isin(labels, list(surfaces.get("wall", set())))
    floor = np.isin(labels, list(surfaces.get("floor", set())))
    candidate = np.zeros((height, width), bool)
    on_wall = np.zeros((height, width), bool)
    for region, shadow in ((wall, False), (floor, True)):
        clean = region & ~removed
        if int(clean.sum()) < 200:
            continue
        ref = np.median(lab[clean], axis=0)
        diff = lab - ref
        chroma = np.sqrt(diff[..., 1] ** 2 + diff[..., 2] ** 2)
        lightness = diff[..., 0] * (100.0 / 255.0)
        outlier = region & ((chroma > _PHOTO_CHROMA) | (lightness < -_PHOTO_DARK))
        if shadow:
            candidate |= outlier
            near = cv2.dilate(removed.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            candidate |= region & near & (lightness < -_PHOTO_SHADOW)
        else:
            on_wall |= outlier
            band = max(3, int(round(_ON_TOP_BAND * height)))
            kernel = np.zeros((2 * band + 1, 1), np.uint8)
            kernel[band:] = 1  # grows a region UPWARD only
            above = cv2.dilate(removed.astype(np.uint8), kernel) > 0
            on_wall |= region & above & ((chroma > _PHOTO_CHROMA / 2) | (np.abs(lightness) > 12.0))
    if on_wall.any():
        _, pieces, stats, _ = cv2.connectedComponentsWithStats(on_wall.astype(np.uint8), connectivity=8)
        large = np.nonzero(stats[1:, cv2.CC_STAT_AREA] > _WALL_PIECE_MAX * height * width)[0] + 1
        if large.size:
            on_wall &= ~np.isin(pieces, large)
        candidate |= on_wall
    candidate &= held == 0
    if not candidate.any():
        return hole
    touching = cv2.dilate(removed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    count, components = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    taken = np.unique(components[touching & candidate])
    taken = taken[taken > 0]
    if taken.size == 0:
        return hole
    added = np.isin(components, taken)
    grow = max(3, int(round(max(height, width) * 0.004)) | 1)
    added = cv2.dilate(added.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))) > 0
    added &= held == 0
    log.info("empty_photo.mask_completed", added=round(float(added.mean()), 4), components=int(taken.size))
    return np.where(added | removed, np.uint8(255), np.uint8(0))


def empty_photo(
    image: np.ndarray,
    *,
    window_dressing: bool = True,
    max_hole: float = MAX_HOLE_FRACTION,
) -> EmptyPhotoOutput | None:
    """Erase the furnishings from one ordinary photograph. ``None`` when nothing can answer.

    ``None`` means there is no inpainting backend on this machine — a degraded answer, not an
    error, and the caller reports it. An empty room, or one whose furniture covers too much of
    the frame to reconstruct around, comes back as a RESULT with ``removed_fraction`` saying so.
    """
    from biosphere.ai import backend, restyle

    started = time.perf_counter()
    height, width = image.shape[:2]

    masks = restyle.read_room(image, window_dressing=window_dressing)
    if not masks.segmented:
        log.info("empty_photo.unavailable", reason="no scene-parsing backend on this machine")
        return None

    hole = masks.editable if masks.mode == "replace" else np.zeros((height, width), np.uint8)
    if masks.mode == "replace" and COMPLETE_MASK:
        hole = complete_mask(image, hole, masks)
    share = float((hole > 127).mean())

    if share <= 0.0:
        # Nothing the segmenter recognises is standing in this room. Handing the photograph
        # back unchanged is the truthful answer; inventing a hole would not be.
        log.info("empty_photo.nothing_to_remove", mode=masks.mode)
        return EmptyPhotoOutput(
            image=image.copy(),
            mask=hole,
            removed_fraction=0.0,
            objects=masks.objects,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    if share > max_hole:
        log.warning("empty_photo.hole_too_large", share=round(share, 3), limit=max_hole)
        return EmptyPhotoOutput(
            image=image.copy(),
            mask=hole,
            removed_fraction=0.0,
            objects=masks.objects,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    filled = backend.inpaint_photo(image, hole, cyclic=False)
    if filled is None:
        log.info("empty_photo.unavailable", reason="no inpainting backend on this machine")
        return None
    if (filled.shape[1], filled.shape[0]) != (width, height):
        filled = cv2.resize(filled, (width, height), interpolation=cv2.INTER_CUBIC)

    # An inpainter rewrites every pixel it is given, so the photograph outside the hole comes
    # back from the original. Feathered, or the join shows as an outline of the mask — the
    # silhouette of the very furniture that was supposed to have gone.
    alpha = restyle.feathered_alpha(hole, max(1.0, max(height, width) * _FEATHER))[..., None]
    out = np.rint(
        np.clip(filled.astype(np.float32) * alpha + image.astype(np.float32) * (1.0 - alpha), 0, 255)
    ).astype(np.uint8)

    elapsed = int((time.perf_counter() - started) * 1000)
    log.info(
        "empty_photo.done",
        removed=round(share, 4),
        objects=list(masks.objects),
        ms=elapsed,
    )
    return EmptyPhotoOutput(
        image=out,
        mask=hole,
        removed_fraction=share,
        objects=masks.objects,
        elapsed_ms=elapsed,
    )
