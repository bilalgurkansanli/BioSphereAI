"""Emptying the room: take every furnishing out of a panorama and leave the shell.

**Two halves, and only the first is learned.** WHAT to remove comes from a scene-parsing
model when one is installed (``model_backed`` says so) and from the colour model below when
not. WHAT IS BEHIND IT is never asked of a model that could invent: with a segmented room the
hole is rebuilt on the room's own planes (:mod:`biosphere.ai.room_planes` — the walls, floor and
ceiling fitted to the junctions the panorama shows), and the colour path below is the fallback.

Measured on the reference bedroom's three captures (``scripts/bench_empty_room.py``), the plane
fill replaced the LaMa fill's grey and black smears — mean colour distance (CIE76) of the
rebuilt surfaces from their own visible colour 10.9-13.2 against LaMa's 18.8-26.6, and the share
of the hole more than 15 L* darker than its surface 0.09-0.11 against 0.43-0.71 — and the mask
grew over what the segmenter called room but was not (:func:`_complete_mask`: a black bag
"floor", a wardrobe's glass reflecting the ceiling light).
The honest limits of the colour path are stated below rather than glossed over.

Why this is not :mod:`biosphere.ai.hide_furniture` with a bigger limit
---------------------------------------------------------------------
That module answers "erase *this* thing": the caller draws an outline, the hole is small,
and ``cv2.xphoto.inpaint`` fills it by extending the surrounding texture's spectrum inward.
Both halves of it stop working when the request becomes "erase *everything*":

* **Detection.** Its proposal step segments the lower half into blobs and ranks the three
  that look most object-like. Ranking is the wrong shape of answer for "all of them" —
  there is no threshold on that ranking that means "and nothing else", because the score is
  relative to the other blobs in the same room.
* **Filling.** A whole-room mask covers 10-40 % of the canvas in patches hundreds of pixels
  across. Generic inpainting is a *diffusion* process: over a hole that size Telea produces
  a smear and FSR would not finish this decade (measured at 5.5 ms per hole pixel, a 3 %
  mask on a 4096 canvas is nine hours).

So this module inverts the question. It does not look for furniture. It works out where the
**room** is — ceiling, walls, floor — and calls everything that is not the room a furnishing.
That is a decision with an absolute meaning, and it hands the filler something a generic
inpainter never has: it knows the hole is a piece of *wall* or a piece of *floor*, and it
knows where else in the same panorama that surface is visible.

The room model
--------------
An indoor equirectangular panorama, read down a single column, is a three-part story:
ceiling, then wall, then floor, in that order and never any other. That is not a heuristic
about furniture, it is the shape of a room seen from inside it. Two consequences drive the
whole file:

1. **The layout is a monotone 3-state segmentation of every column**, solved exactly by
   Viterbi (:func:`_segment_columns`) against per-sector colour models. Not thresholding,
   not clustering — the ordering constraint is free information and it is what stops a dark
   rug being labelled "ceiling" because it happens to match the ceiling's colour.
2. **A furnishing is a pixel that does not look like the surface it stands in front of.**
   The residual is measured against that surface's *local* colour model, so a red sofa on a
   grey floor and a grey sofa on a red floor are both caught, and a window in a wall is not.

Filling, and why it is not inpainting
-------------------------------------
The fill is split by frequency, and each half exploits something specific to this geometry:

* **Low frequency** — a normalised-convolution pyramid (push-pull) run *per surface*, so
  floor colour can never bleed up a wall. It is O(N), has no radius limit, and over a hole
  of any size it produces the smooth shading field the surface would have had. This is the
  part that makes a whole sofa disappear instead of smearing.
* **High frequency** — grain is transplanted from elsewhere on the *same row*. On an
  equirect canvas a row is a circle of constant elevation, so a horizontal shift samples the
  same surface at the same distance and therefore at the **same texture scale**. That is a
  geometric fact of the projection, not a lucky heuristic, and it is why the recovered floor
  reads as flooring rather than as plastic. Wrapping is free: the row is cyclic.

What this cannot do, stated plainly
-----------------------------------
It separates surfaces by colour. A white sofa against a white wall in flat light is not
separable by colour, and no amount of tuning here will change that — that case needs a
semantic segmenter. :data:`SEGMENT_BACKEND` is the seam where one drops in: it implements
``(image, shell, strength) -> mask`` and nothing else in the service changes, exactly as
``hide_furniture.INPAINT_BACKEND`` does for the filler. ``removed_fraction`` in the response
is what the UI should show instead of a claim of completeness.

Nothing inside the polar caps is ever touched. Those directions were never photographed —
the stitcher synthesised them and may have composited branding there — so replacing them
with a second synthesis would be inventing on top of invention.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import orjson

from biosphere.api.schemas import EmptyRoomResponse, PanoramaLevels, RemovedObject
from biosphere.core.logging import get_logger
from biosphere.core.models import ApiException, Scene
from biosphere.core.storage import LocalStorage, write_jpeg, write_png, write_webp
from biosphere.pano import geometry as geo

__all__ = [
    "empty_room",
    "source_level",
    "write_levels",
    "resize_cyclic",
    "staged_levels",
    "STAGED_LEVEL_WIDTHS",
    "estimate_shell",
    "furniture_mask",
    "fill_room",
    "RoomShell",
    "RoomPlan",
    "SEGMENT_BACKEND",
    "FILL_BACKEND",
    "CEILING",
    "WALL",
    "FLOOR",
    "STAGED_DIRNAME",
    "DEFAULT_STRENGTH",
    "TIME_BUDGET_S",
]

log = get_logger("biosphere.ai.empty_room")

_EPS = 1e-6

#: Surface ids. The order is the order they appear down a column and the DP depends on it.
CEILING, WALL, FLOOR = 0, 1, 2
_N_SURFACES = 3

#: How hard the default press pushes.
#:
#: CHOSEN AGAINST REAL CAPTURES, NOT THE SYNTHETIC ROOMS. On the procedural fixtures more is
#: simply better, because their surfaces are clean and a higher threshold only finds more
#: furniture. Real rooms are not like that: at 0.6 a cluttered bedroom came back with its
#: walls visibly bleached, because a busy room offers many pixels that differ from the paint
#: behind them and the fill obligingly replaced all of them with paint. At 0.35 the same
#: capture kept its room and lost its rug and its clutter, but kept the bed.
#:
#: 0.5 sits between them, and the tie is broken by which failure the user can live with. A
#: sofa left standing is visible, self-explanatory and half a job done. A bleached wall is
#: damage, and it makes the whole feature look broken rather than limited. The endpoint takes
#: `strength`, so wanting more is one request away.
DEFAULT_STRENGTH = 0.5

#: Wall clock for one full run, guarded at the stage boundaries.
#:
#: Measured on the reference machine (GTX 1650): the classical path is 1.8 s for a 4096-wide
#: panorama. With the models it is about 30 s -- sixteen perspective windows through
#: Mask2Former at 0.8 s each, then one LaMa pass at 8 s -- so the budget has to cover that
#: and it is still a *ceiling*: a run that needs all of it has gone wrong. The result is
#: cached per scene, so this is paid once and never by a user pressing the toggle twice.
TIME_BUDGET_S = 120.0

#: Everything is decided at this width. Furniture is a large-blob phenomenon and the colour
#: models are per-5-degree-sector, so nothing here is improved by more pixels — while the
#: DP, the morphology and the connected components all cost linearly in them.
_DECISION_WIDTH = 1024

#: The staged panorama is delivered no wider than this, whatever the source.
#:
#: A REAL CEILING, NOT A ROUND NUMBER. The fill's low-frequency half is a float32 pyramid
#: over the whole canvas; at 8192x4096 that is 536 MB for level 0 alone before the pyramid,
#: which is the same wall that forced `blend.blend_tiled` to exist. It is also the honest
#: resolution: a third of these pixels are synthesised, and delivering invented grain at 8K
#: costs four times the memory and the bytes to say nothing more than 4K already says.
_STAGED_MAX_WIDTH = 4096

#: Yaw sectors for the local colour models. 72 is 5 degrees, the same lattice
#: `pano/coverage.py` and the phone's capture grid use — so "the wall in that direction"
#: means the same thing in all three places.
_SECTORS = 72

#: Elevation bands (degrees) that seed the FIRST pass only, before a layout exists.
#:
#: Deliberately crude, and only ever used to bootstrap: they are fixed elevations, so they
#: cannot know that this room's ceiling is low or that its floor starts high. Pass two
#: replaces them with bands measured off the layout itself — see :func:`_seed_regions`.
_BOOTSTRAP_BANDS: dict[int, tuple[float, float]] = {
    CEILING: (40.0, 70.0),
    WALL: (6.0, 28.0),
    FLOOR: (-65.0, -42.0),
}

#: Where each surface is sampled once a layout is known, as fractions of that surface's own
#: extent in the column. THIS IS WHERE THE FURNITURE IS AVOIDED, and each one is a claim
#: about rooms that can be checked:
#:
#:   CEILING  the middle of it, clear of the cornice shadow at the edge.
#:   WALL     the top of the wall. Furniture stands on the floor; only a full-height
#:            wardrobe reaches the top of a wall, and then only in a few directions.
#:   FLOOR    the PERIMETER — the strip just inside the skirting, not the ring around the
#:            photographer's feet. That ring was the original choice and it was wrong by
#:            measurement: on the reference rooms it came out 82 % furniture, because a rug
#:            is laid in the middle of a room and that is exactly where you stand. The
#:            perimeter is the part of a floor a rug does not reach.
_SEED_SPANS: dict[int, tuple[float, float]] = {
    CEILING: (0.10, 0.98),
    WALL: (0.06, 0.42),
    FLOOR: (0.04, 0.26),
}

#: A colour must hold this share of a sector's seed band to count as present in it...
_SECTOR_PRESENCE_SHARE = 0.05
#: ...and be present in this share of all sectors to be one of the surface's own colours.
#:
#: THE ARCHITECTURE IS WHAT GOES ALL THE WAY ROUND. A wall's paint appears in nearly every
#: direction; a wardrobe appears in two. Nothing else available here separates a large flat
#: brown rectangle that is furniture from a large flat brown rectangle that is a door, and
#: this does, without recognising either.
#:
#: The floor asks for more than the other two because its seed band is the dirtiest: measured
#: on the reference rooms, the perimeter strip comes back 56 % furniture, because furniture is
#: pushed against walls. At a low bar the six pieces in one room between them cover enough
#: directions to be voted architecture — and the palette then contained the wardrobe's brown
#: and the sofa's blue to within 1.0 and 4.1 Lab units, which is why neither was ever removed.
_GLOBAL_SECTOR_SHARE: dict[int, float] = {CEILING: 0.35, WALL: 0.30, FLOOR: 0.45}

#: Below this many directions a colour does not "recur" at all, and there is no global
#: palette to relax down to — the surface genuinely differs everywhere. Only then does the
#: per-sector fallback in :func:`_surface_models` take over.
_MIN_GLOBAL_SECTORS = 3

#: How many global colours a surface may have.
#:
#: The ceiling and the walls get many because THEIR PALETTE HAS TO SPAN THEIR SHADING: a
#: ceiling is 35 Lab units brighter under the light than in the corner, and at 8 units to a
#: bin that is five entries before any second material is described. Starved of them, the
#: shadowed half of every ceiling reads as an object and gets erased. The floor gets few for
#: the opposite reason — its seed band is the contaminated one, and every extra entry there
#: is another chance to admit a sofa.
_MAX_GLOBAL: dict[int, int] = {CEILING: 8, WALL: 8, FLOOR: 4}

#: Colours kept for one sector alone, so a window, a doorway or an accent wall is still the
#: room rather than a hole in it. Held to a far higher bar than a global colour — it has to
#: dominate its own sector — because this is the door through which a wardrobe would walk in
#: and call itself a wall.
#:
#: A FLOOR HAS NO LOCAL COLOURS, and that is not a tuning choice. A floor is one material laid
#: all the way to the walls; that is what makes it a floor. So anything that dominates the
#: floor in one direction and not the others is, by construction, something standing on it.
#: Turning this off for FLOOR alone is what finally let whole sofas be seen.
_LOCAL_DOMINANT_SHARE: dict[int, float] = {CEILING: 0.30, WALL: 0.22, FLOOR: 1.0}
_MAX_LOCAL: dict[int, int] = {CEILING: 2, WALL: 3, FLOOR: 0}

#: Never remove anything outside this elevation band.
#:
#: `poles.fill_uncovered` synthesised every pixel above 60 degrees, and `PipelineConfig`
#: composites branding over a 30-degree zenith disc and a 42-degree nadir one. Those
#: directions hold no furniture, hold no photograph, and may hold a logo the operator put
#: there on purpose. Inpainting them would be a second synthesis on top of the first.
#: Sized to the branding `PipelineConfig` actually composites: a 30-degree zenith disc is a
#: 15-degree radius from the pole, a 42-degree nadir one is 21 degrees, so the protected caps
#: begin at +75 and -69. A degree of margin either way, and not one more — every extra degree
#: of nadir protection leaves a visible ring of un-emptied floor around the viewer's own feet.
_PROTECT_ABOVE_DEG = 76.0
_PROTECT_BELOW_DEG = -70.0

#: A sector band needs at least this many pixels before its colour model is believed.
_MIN_SEED_PIXELS = 24

#: A SURFACE HAS A PALETTE, NOT A COLOUR, and that is the difference between this working
#: and not. A ceiling is lit in the middle and shadowed at the corners; a wall has a window
#: in it; a floor has boards and grout. Described by one colour, every one of those reads as
#: "not the room" and the mask fills with the room itself. Described by the handful of
#: colours the surface demonstrably has, only what is genuinely foreign is left over.
#:
#: The palette is taken from a coarse Lab histogram — deterministic, unlike k-means, which
#: this has to be: the same panorama must empty the same way every time it is asked.
_PALETTE_BIN = 8.0
#: Palette entries closer together than this are the same colour counted twice.
_PALETTE_MERGE_DE = 9.0

#: Lightness carries less weight than chroma in every distance here.
#:
#: The same wall is 40 Lab units darker in the corner than under the light, and no palette
#: covers every shading of every surface. Chroma barely moves under that — a grey wall stays
#: grey in shadow — so down-weighting L is what lets a fixed threshold survive real lighting.
#: Not zero: a charcoal sofa against a white wall differs in nothing else, and at 0.3 a 40
#: unit lightness gap still contributes 22, comfortably over the threshold.
#:
#: 0.3 rather than 0.5 by measurement. Swept against ground truth on three reference rooms,
#: dropping it trades a little recall (0.525 -> 0.503) for a lot of precision (0.658 ->
#: 0.722), because what it stops flagging is the shaded half of the room's own ceiling.
#: Erasing a real ceiling is the worse error of the two: a sofa left standing is a job half
#: done, while a smear where the cornice used to be is damage.
_L_WEIGHT = 0.3

#: Lab units of disagreement that count as "one unit of surprise" when the layout is solved.
#: Only sets the exchange rate against `_elevation_prior`; it is not a decision threshold.
_COST_SCALE = 12.0

#: The two junction curves are robustified over a window this wide, as a fraction of the
#: canvas. Wide enough to span a wardrobe (a 0.6 m unit two metres away subtends 17 degrees,
#: which is 5 % of the canvas) and narrow enough to keep a room's actual corners.
_JUNCTION_WINDOW_FRAC = 0.09
_JUNCTION_SMOOTH_FRAC = 0.012

#: Measured; see :func:`_cyclic_percentile` for the hypothesis this rejected.
_JUNCTION_QUANTILE: dict[int, float] = {FLOOR: 50.0, CEILING: 50.0}

#: How many times the layout and the colour models are solved against each other.
#: Two: pass one exists only to give pass two somewhere to measure its bands from, and a
#: third changed the mask by under a percent on every reference room while costing a third
#: of the run.
_SHELL_PASSES = 2

#: Cost of a surface change inside a column. Small: exactly two changes are permitted by
#: construction, so this only discourages a hairline segment at the very top or bottom.
_SWITCH_PENALTY = 1.5

#: Weighted Lab distance at which a pixel stops being its surface, as a function of
#: `strength`. Absolute, not relative to the surface's own spread: a scale learned from the
#: band grows with every object standing in it, so a busy room would forgive more and more
#: of itself.
#:
#: The span is measured, not guessed. Sweeping the threshold against ground truth on the
#: reference rooms puts the best F1 between 7 and 14 weighted-Lab units depending on how
#: much the room's own shading varies, so the range brackets that with a margin either side
#: and `strength` moves across it.
_TAU_CONSERVATIVE, _TAU_AGGRESSIVE = 18.0, 7.0

#: The lower of the two hysteresis levels, as a fraction of the upper. A region has to reach
#: this AND touch a core to be taken. Deliberately generous: an object's mid-tones sit much
#: closer to the room than its extremes do, and taking a whole object is what the fill is
#: good at -- leaving half of one is what it cannot hide.
_HYSTERESIS_RATIO = 0.55

#: How far growth may reach from a core, as a fraction of the canvas width. At the 1024 px
#: decision width this is about twenty pixels, or seven degrees of sphere -- the scale of an
#: object's own shading, not of a room.
_GROWTH_REACH_FRAC = 0.02

#: A removable region must cover at least this fraction of the decision canvas. Below it we
#: are looking at compression noise and picture frames' own mouldings, and removing those
#: costs more in fill artefacts than the ghost of a doorknob costs in realism.
_MIN_COMPONENT_FRAC = 2.0e-4

#: ...and it must be THICK, not merely large. This is the rule that keeps the room itself
#: out of the mask, and it needed its own criterion because area does not express it: the
#: ceiling-wall shadow line and the skirting board run the entire way round the panorama, so
#: they are among the LARGEST connected regions on the canvas while being four pixels thick.
#: A furnishing is a blob. Measured as the largest inscribed radius, in pixels at the
#: decision width, so ~2 degrees of sphere across at the narrowest.
_MIN_RADIUS_AT_1024 = 5.0

#: How far the mask is grown before filling, in pixels at 4096 wide, at strength 0.5.
#: Contact shadows and colour bleed live just outside an object's silhouette, and a one-pixel
#: rim of sofa left behind reads worse than the fill it surrounds.
_DILATE_PX_AT_4096 = 9.0

#: More of the sphere than this and the run is refused. THE CAP IS THE SAFETY ARGUMENT, NOT
#: THE THRESHOLD -- the same shape of guard as `blend.max_ghost_repair_fraction`.
#:
#: It differs by who drew the mask, because the two are not making the same kind of claim.
#: A colour test says "this pixel is far from the wall's palette"; at half the sphere that is
#: not a heavily furnished room, it is a failed palette, and the measured example is exactly
#: that -- 52 % selected with holes where the furniture actually was. A scene-parsing model
#: says "this pixel is a bed", and half a sphere of beds and desks is an ordinary bedroom:
#: the three reference captures come in at 0.47, 0.48 and 0.52 with the furniture correctly
#: identified. Same number, opposite meanings, so they get different limits.
_MAX_REMOVED_FRACTION_CLASSICAL = 0.45
_MAX_REMOVED_FRACTION_MODEL = 0.80

#: Grain gain when the transplanted high band is added back. Slightly under 1: the source
#: row is a different part of the same surface, so its grain is right in character and
#: wrong in detail, and a shade less of it is more convincing than a shade more.
_GRAIN_GAIN = 0.85

#: How much of the room's own grain goes back on top of a MODEL fill, and how large a
#: deviation counts as grain rather than as an edge. Separate from `_GRAIN_GAIN`, which
#: belongs to the classical filler: that one has no texture of its own and needs all of it,
#: while the network's output already carries some and needs only the fine tooth it lacks.
_GRAIN_REINJECT_GAIN = 0.55
_GRAIN_CLAMP = 7.0

#: Candidate yaw shifts for the grain transplant, as fractions of the canvas width. Ordered
#: outward from a half turn: the far side of the room is the least likely to be occluded by
#: the same piece of furniture we are removing.
_TRANSPLANT_SHIFTS = (0.50, 0.33, 0.67, 0.25, 0.75, 0.17, 0.83, 0.42, 0.58, 0.08, 0.92)

#: Where the staged pyramid lives inside the scene directory.
STAGED_DIRNAME = "staged"

#: Bumped whenever the algorithm changes in a way that invalidates a cached result.
#: 2: the mask comes from a scene-parsing model and the fill from LaMa. Everything written
#: by version 1 is a different picture and has to be recomputed rather than served.
#: 3: `auto` no longer selects the diffusion fill. Anything cached while it did contains
#: furniture the model invented -- a made bed where a bed was removed -- and serving that
#: from disk would be worse than the run that produced it, because nobody would be watching
#: when it appeared. See `biosphere.ai.sd_inpaint` for what was measured.
#: 4: a segmented room is rebuilt on its own planes (`room_planes`) and its mask grown over
#: what the segmenter called room but is not (`_complete_mask`). Every version-3 file is the
#: LaMa fill's grey and black smears, which is the thing this version exists to replace.
_CACHE_VERSION = 4

_LEVEL_PREFERENCE = ("uhd", "hd", "sd", "preview")
STAGED_LEVEL_WIDTHS = (("hd", 4096), ("sd", 2048), ("preview", 512))


# ======================================================================================
# Data
# ======================================================================================


@dataclass(slots=True)
class RoomShell:
    """Where the room's own surfaces are, on the decision canvas.

    ``ceiling_row`` and ``floor_row`` are per-column boundaries in rows: ceiling occupies
    ``[0, ceiling_row)``, wall ``[ceiling_row, floor_row)``, floor ``[floor_row, H)``. They
    are floats because they are smoothed across columns, and cyclic because column 0 and
    column W-1 are neighbours.
    """

    ceiling_row: np.ndarray
    floor_row: np.ndarray
    labels: np.ndarray
    #: Per-surface distance field in units of that surface's own spread, shape (3, H, W).
    distance: np.ndarray

    @property
    def size(self) -> tuple[int, int]:
        return int(self.labels.shape[0]), int(self.labels.shape[1])


@dataclass(slots=True)
class RoomPlan:
    """What a run decided, before anything was written to disk."""

    image: np.ndarray
    mask: np.ndarray
    shell: RoomShell
    regions: list[RemovedObject]
    removed_fraction: float
    #: True when a learned inpainter produced the fill rather than the classical one.
    model_backed: bool = False
    #: The room the plane fill drew on, and what each plane did — for the bench and the log.
    room: Any = None
    fill: str = ""
    planes: list[Any] = field(default_factory=list)


# ======================================================================================
# Small numeric helpers
# ======================================================================================


def _gate(value: np.ndarray, off_at: float, free_at: float, weight: float) -> np.ndarray:
    """``weight`` where ``value <= off_at``, 0 where ``value >= free_at``, linear between."""
    span = max(free_at - off_at, _EPS)
    return (weight * np.clip((free_at - value) / span, 0.0, 1.0)).astype(np.float32)


def _wrap_pad(image: np.ndarray, pad: int) -> np.ndarray:
    """Extend horizontally with the canvas's own opposite edge. The seam is not an edge."""
    if pad <= 0:
        return image
    return np.concatenate([image[:, -pad:], image, image[:, :pad]], axis=1)


def _pyr_down_cyclic(image: np.ndarray) -> np.ndarray:
    """``cv2.pyrDown`` that respects the horizontal wrap.

    Two padded columns survive one halving as one column each side, so the crop is exact
    and no border mode is ever consulted for a pixel that has a real neighbour.
    """
    down = cv2.pyrDown(_wrap_pad(image, 2))
    return np.ascontiguousarray(down[:, 1:-1])


def _pyr_up_cyclic(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """``cv2.pyrUp`` to an exact size, respecting the horizontal wrap."""
    up = cv2.pyrUp(_wrap_pad(image, 1), dstsize=(width + 4, height))
    return np.ascontiguousarray(up[:, 2:-2])


def resize_cyclic(image: np.ndarray, width: int, height: int, interpolation: int) -> np.ndarray:
    """Resize without letting the seam column interpolate against a clamped border.

    Public alongside :func:`source_level` and :func:`write_levels`: anything that derives a
    panorama from a scene's own and delivers it has to cap the master the same way, and a
    caller that reached for a plain ``cv2.resize`` would put a visible line down the seam of
    every rung it wrote."""
    src_h, src_w = image.shape[:2]
    if (src_w, src_h) == (width, height):
        return image
    pad = 8
    scaled_pad = max(1, int(round(pad * width / float(src_w))))
    out = cv2.resize(
        _wrap_pad(image, pad), (width + 2 * scaled_pad, height), interpolation=interpolation
    )
    return np.ascontiguousarray(out[:, scaled_pad : scaled_pad + width])


def _cyclic_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur along a cyclic 1-D signal."""
    if sigma <= 0.0:
        return values.astype(np.float32)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    radius = min(radius, values.size)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= float(kernel.sum())
    padded = np.concatenate([values[-radius:], values, values[:radius]]).astype(np.float32)
    return np.convolve(padded, kernel, mode="same")[radius:-radius].astype(np.float32)


def _cyclic_percentile(values: np.ndarray, window: int, q: float) -> np.ndarray:
    """Rolling percentile along a cyclic 1-D signal, used to robustify a junction curve.

    ``q`` IS 50, AND THAT IS A RESULT RATHER THAN A DEFAULT. The reasoning that argued for a
    higher one is worth recording so it is not re-derived: a furnishing whose colour the
    floor's palette already contains — a wooden chest on a wooden floor — is read as floor,
    so the boundary should be placed at the top of the chest rather than behind it, and the
    curve should spike one way only. Swept against ground truth on three reference rooms, it
    does not: q of 50, 65, 75 and 85 score F1 0.591, 0.583, 0.581 and 0.579, so the errors
    are two-sided after all and a plain median is the best of them. The parameter stays
    because the window is what does the work, and a median is the right estimator over it.
    """
    width = int(values.size)
    window = max(3, min(int(window) | 1, width if width % 2 else width - 1))
    pad = window // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]]).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.percentile(windows, q, axis=-1).astype(np.float32)


def _cyclic_fill(values: np.ndarray, trusted: np.ndarray) -> np.ndarray:
    """Replace untrusted entries by interpolating cyclically between the trusted ones."""
    width = int(values.size)
    index = np.nonzero(trusted)[0]
    if index.size == 0:
        return values.astype(np.float32)
    if index.size == width:
        return values.astype(np.float32)
    # Three copies of the sample set give np.interp a monotone axis that covers the wrap in
    # both directions, so a gap straddling column 0 is interpolated rather than clamped.
    xs = np.concatenate([index - width, index, index + width]).astype(np.float64)
    ys = np.tile(values[index].astype(np.float64), 3)
    return np.interp(np.arange(width, dtype=np.float64), xs, ys).astype(np.float32)


def _imread(path: Path) -> np.ndarray:
    """Read an image without tripping over non-ASCII paths on Windows."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ApiException(
            "panorama_unreadable",
            "The stored panorama could not be read.",
            "Kayıtlı panorama okunamadı.",
            status_code=500,
        )
    return image


# ======================================================================================
# Colour models
# ======================================================================================


def _to_lab(image: np.ndarray) -> np.ndarray:
    """CIE Lab, float32, with the L channel back on its natural 0..100 scale.

    Lab because the distance that decides "is this the same surface" has to behave like the
    difference a person sees. In BGR a shadowed patch of the same carpet is as far from lit
    carpet as carpet is from a sofa.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def _delta_e(diff: np.ndarray) -> np.ndarray:
    """Lab distance with lightness down-weighted. See :data:`_L_WEIGHT`."""
    return np.sqrt(
        _L_WEIGHT * diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2, dtype=np.float32
    )


def _bin_keys(pixels: np.ndarray) -> np.ndarray:
    """Pack each pixel's coarse Lab bin into one integer, so bins can be counted by sorting."""
    binned = np.clip(np.floor(pixels / _PALETTE_BIN).astype(np.int32), -32, 31) + 32
    return (binned[:, 0] << 12) | (binned[:, 1] << 6) | binned[:, 2]


def _modes_of(pixels: np.ndarray, keys: np.ndarray, wanted: list[int]) -> list[np.ndarray]:
    """Mean colour of each requested bin. The mean, not the bin centre: the bin is 8 Lab
    units across and its occupants are rarely spread evenly inside it."""
    out: list[np.ndarray] = []
    for key in wanted:
        selected = keys == key
        if selected.any():
            out.append(pixels[selected].mean(axis=0).astype(np.float32))
    return out


def _sector_of_column(width: int) -> np.ndarray:
    columns = np.arange(width, dtype=np.int64)
    return np.minimum(columns * _SECTORS // max(width, 1), _SECTORS - 1)


def _seed_regions(
    phi_deg: np.ndarray, width: int, shell: RoomShell | None
) -> dict[int, np.ndarray]:
    """Where each surface's colour is sampled from. Bootstrap bands, then layout-relative.

    Returns one boolean mask per surface. Nothing inside the polar caps is ever sampled:
    those pixels were synthesised by the stitcher and describe its inpainter, not the room.
    """
    height = phi_deg.size
    rows = np.arange(height, dtype=np.float32)[:, None]
    protect = ((phi_deg > _PROTECT_ABOVE_DEG) | (phi_deg < _PROTECT_BELOW_DEG))[:, None]

    regions: dict[int, np.ndarray] = {}
    if shell is None:
        for surface, (low_deg, high_deg) in _BOOTSTRAP_BANDS.items():
            band = ((phi_deg >= low_deg) & (phi_deg <= high_deg))[:, None]
            regions[surface] = np.broadcast_to(band & ~protect, (height, width)).copy()
        return regions

    ceiling = shell.ceiling_row[None, :]
    floor = shell.floor_row[None, :]
    extents = {
        CEILING: (np.zeros_like(ceiling), np.maximum(ceiling, 1.0)),
        WALL: (ceiling, np.maximum(floor - ceiling, 1.0)),
        FLOOR: (floor, np.maximum(float(height) - floor, 1.0)),
    }
    for surface, (origin, span) in extents.items():
        low, high = _SEED_SPANS[surface]
        regions[surface] = (
            (rows >= origin + low * span) & (rows < origin + high * span) & ~protect
        )
    return regions


def _surface_models(
    lab: np.ndarray, regions: dict[int, np.ndarray], sector_of_col: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-surface, per-sector colour palettes. Returns ``(modes, counts)``.

    Two kinds of entry, and the distinction is the heart of the method:

    **Global** — a colour the surface shows in at least :data:`_GLOBAL_SECTOR_SHARE` of all
    directions. That is what architecture looks like from the middle of a room: paint,
    plaster, floorboards. A sofa cannot fake it, because a sofa is somewhere rather than
    everywhere.

    **Local** — a colour that dominates one sector's own seed band, kept for that sector and
    its two neighbours. Without this a window, a doorway or an accent wall would be foreign
    to its own wall and get erased. With it, a wardrobe that happens to fill its sector gets
    in too; that is the trade, it is bounded by the much higher share this asks for, and the
    blob tests downstream do not care either way.
    """
    slots = max(_MAX_GLOBAL.values()) + 3 * max(_MAX_LOCAL.values())
    modes = np.zeros((_N_SURFACES, _SECTORS, slots, 3), np.float32)
    counts = np.zeros((_N_SURFACES, _SECTORS), np.int32)
    columns_of = [np.nonzero(sector_of_col == sector)[0] for sector in range(_SECTORS)]

    for surface in range(_N_SURFACES):
        region = regions[surface]
        pixels_by_sector: list[np.ndarray] = []
        keys_by_sector: list[np.ndarray] = []
        for sector in range(_SECTORS):
            cols = columns_of[sector]
            if cols.size == 0:
                pixels_by_sector.append(np.zeros((0, 3), np.float32))
                keys_by_sector.append(np.zeros(0, np.int64))
                continue
            pixels = lab[:, cols][region[:, cols]]
            pixels_by_sector.append(pixels)
            keys_by_sector.append(_bin_keys(pixels) if pixels.size else np.zeros(0, np.int64))

        presence: dict[int, int] = {}
        for keys in keys_by_sector:
            if keys.size < _MIN_SEED_PIXELS:
                continue
            unique, occurrences = np.unique(keys, return_counts=True)
            floor_count = _SECTOR_PRESENCE_SHARE * keys.size
            for key, count in zip(unique.tolist(), occurrences.tolist(), strict=True):
                if count >= floor_count:
                    presence[key] = presence.get(key, 0) + 1

        pooled = lab[region]
        pooled_keys = _bin_keys(pooled) if pooled.size else np.zeros(0, np.int64)
        ranked = sorted(presence.items(), key=lambda item: (-item[1], item[0]))
        wanted = [key for key, seen in ranked if seen >= _GLOBAL_SECTOR_SHARE[surface] * _SECTORS]
        if not wanted and ranked and ranked[0][1] >= _MIN_GLOBAL_SECTORS:
            # Nothing cleared the bar, but something does recur. RELAX THE BAR RATHER THAN
            # ABANDON THE IDEA: a room lit hard from one side, or one with a wall of glass,
            # can leave no colour in half of all directions while still having colours that
            # come back. Dropping straight through to the per-sector fallback below cost 3.5
            # points of recall on the reference rooms, because a sector's own median in a
            # furnished direction is the furniture.
            wanted = [key for key, seen in ranked if seen >= ranked[0][1] * 0.6]
        global_modes = _modes_of(pooled, pooled_keys, wanted[: _MAX_GLOBAL[surface]])
        # A SURFACE WITH NO COLOUR IN COMMON ACROSS DIRECTIONS IS NOT A SURFACE THIS CAN
        # DESCRIBE GLOBALLY, and pretending otherwise is catastrophic rather than merely
        # inaccurate. Falling back to the pooled median alone declares everything far from
        # the middle of the range foreign: measured on a smooth left-to-right gradient, that
        # selected 54 % of the sphere for removal. So the fallback is LOCAL — every sector
        # is at least told its own colour — and the surface is described sector by sector
        # instead of by an average of things that were never the same.
        degenerate = not global_modes
        if degenerate and pooled.size:
            global_modes = [np.median(pooled, axis=0).astype(np.float32)]
        del pooled, pooled_keys

        max_local = _MAX_LOCAL[surface]
        local_modes: list[list[np.ndarray]] = []
        for sector in range(_SECTORS):
            keys, pixels = keys_by_sector[sector], pixels_by_sector[sector]
            if keys.size < _MIN_SEED_PIXELS:
                local_modes.append([])
                continue
            if degenerate:
                # See above: describe it where it is, since it is nothing anywhere else.
                local_modes.append([np.median(pixels, axis=0).astype(np.float32)])
                continue
            if max_local == 0:
                local_modes.append([])
                continue
            unique, occurrences = np.unique(keys, return_counts=True)
            order = np.argsort(-occurrences)
            dominant = [
                int(unique[i])
                for i in order
                if occurrences[i] >= _LOCAL_DOMINANT_SHARE[surface] * keys.size
            ][:max_local]
            local_modes.append(_modes_of(pixels, keys, dominant))

        for sector in range(_SECTORS):
            slot = 0
            candidates = list(global_modes)
            for offset in (0, -1, 1):
                candidates.extend(local_modes[(sector + offset) % _SECTORS])
            for colour in candidates:
                if slot >= slots:
                    break
                if any(
                    float(_delta_e(colour - modes[surface, sector, j])) < _PALETTE_MERGE_DE
                    for j in range(slot)
                ):
                    continue
                modes[surface, sector, slot] = colour
                slot += 1
            counts[surface, sector] = slot
        del pixels_by_sector, keys_by_sector

    return modes, counts


def _distance_field(
    lab: np.ndarray, modes: np.ndarray, counts: np.ndarray, sector_of_col: np.ndarray
) -> np.ndarray:
    """Weighted Lab distance from every pixel to every surface's palette. Shape ``(3,H,W)``."""
    height, width = lab.shape[:2]
    out = np.empty((_N_SURFACES, height, width), np.float32)
    slots = modes.shape[2]
    slot_index = np.arange(slots)[None, :]

    for surface in range(_N_SURFACES):
        per_column = modes[surface][sector_of_col]              # (W, slots, 3)
        counts_col = counts[surface][sector_of_col]             # (W,)
        diff = lab[:, :, None, :] - per_column[None, :, :, :]   # (H, W, slots, 3)
        distance = _delta_e(diff)
        del diff
        unused = slot_index >= counts_col[:, None]              # (W, slots)
        if unused.any():
            distance[:, unused] = np.inf
        nearest = distance.min(axis=2)
        del distance
        # A sector with no palette at all cannot vouch for anything: make it maximally
        # surprising rather than silently the best match.
        out[surface] = np.where(np.isfinite(nearest), nearest, 1.0e3)
        del nearest
    return out


# ======================================================================================
# Room layout
# ======================================================================================


def _elevation_prior(phi_deg: np.ndarray) -> np.ndarray:
    """Per-row cost of claiming each surface, purely from elevation. Shape ``(3, H)``.

    Not a hard cut anywhere. A small room's ceiling genuinely reaches down toward the
    horizon and its floor reaches up; what the prior encodes is that neither crosses it.
    """
    prior = np.zeros((_N_SURFACES, phi_deg.size), np.float32)
    prior[CEILING] = _gate(phi_deg, off_at=0.0, free_at=25.0, weight=4.0)
    prior[FLOOR] = _gate(-phi_deg, off_at=0.0, free_at=20.0, weight=4.0)
    # Walls are free across the middle of the sphere and increasingly implausible past the
    # elevation where a room stops having walls.
    prior[WALL] = _gate(50.0 - np.abs(phi_deg), off_at=-15.0, free_at=5.0, weight=3.0)
    return prior


def _segment_columns(cost: np.ndarray, switch_penalty: float = _SWITCH_PENALTY) -> np.ndarray:
    """Exact monotone 3-state segmentation of every column, by Viterbi.

    States may only advance — ceiling to wall to floor, never back — which is the whole
    point: it is the ordering of a room seen from inside, and it is free information that
    no per-pixel classifier can use. Vectorised across columns, so the loop is over the
    canvas's rows and not over its pixels.
    """
    n_states, height, width = cost.shape
    back = np.zeros((height, n_states, width), np.uint8)

    accumulated = cost[:, 0, :].astype(np.float32).copy()
    for row in range(1, height):
        nxt = np.empty_like(accumulated)
        nxt[0] = accumulated[0] + cost[0, row]
        for state in (1, 2):
            stay = accumulated[state]
            advance = accumulated[state - 1] + switch_penalty
            takes_advance = advance < stay
            nxt[state] = np.where(takes_advance, advance, stay) + cost[state, row]
            back[row, state] = takes_advance
        accumulated = nxt

    labels = np.empty((height, width), np.uint8)
    columns = np.arange(width)
    state = np.argmin(accumulated, axis=0).astype(np.int32)
    labels[height - 1] = state
    for row in range(height - 1, 0, -1):
        state = state - back[row, state, columns].astype(np.int32)
        labels[row - 1] = state
    return labels


def _boundaries(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column ``(ceiling_row, floor_row)``. Exact, because the labelling is monotone."""
    height = labels.shape[0]
    ceiling_row = (labels == CEILING).sum(axis=0).astype(np.float32)
    floor_row = (height - (labels == FLOOR).sum(axis=0)).astype(np.float32)
    return ceiling_row, np.maximum(floor_row, ceiling_row)


def _labels_from_boundaries(
    ceiling_row: np.ndarray, floor_row: np.ndarray, height: int
) -> np.ndarray:
    rows = np.arange(height, dtype=np.float32)[:, None]
    return np.where(
        rows < ceiling_row[None, :], CEILING, np.where(rows < floor_row[None, :], WALL, FLOOR)
    ).astype(np.uint8)


def _solve_layout(distance: np.ndarray, phi_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One monotone segmentation of the whole canvas, as two smoothed boundary curves."""
    prior = _elevation_prior(phi_deg)
    cost = distance / _COST_SCALE + prior[:, :, None]
    labels = _segment_columns(cost)
    del cost
    ceiling_row, floor_row = _boundaries(labels)
    del labels
    return _cyclic_smooth(ceiling_row, 2.0), _cyclic_smooth(floor_row, 2.0)


def _complete_junctions(
    distance: np.ndarray, ceiling_row: np.ndarray, floor_row: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the two junctions across the columns where a furnishing hides them.

    THIS IS WHY THE WALL CONTINUES BEHIND THE WARDROBE. A sofa straddles the wall-floor
    junction, so in those columns the segmentation puts the boundary somewhere inside the
    sofa rather than behind it — the boundary is not wrong so much as *displaced*, and a
    displaced boundary sends the filler to the wrong surface for a band the height of the
    sofa. Columns whose junction sits in something foreign are therefore dropped, and the
    curve is interpolated across them from the columns either side, where the junction is
    plainly visible. The room's own shape does the completing.
    """
    _, height, width = distance.shape
    provisional = _labels_from_boundaries(ceiling_row, floor_row, height)
    residual = np.take_along_axis(distance, provisional[None].astype(np.intp), axis=0)[0]
    occluded = residual > _TAU_CONSERVATIVE
    del provisional, residual

    band = max(4, height // 32)
    rows = np.arange(height)[:, None]
    minimum_trusted = max(8, width // 64)

    for surface, curve in ((FLOOR, floor_row), (CEILING, ceiling_row)):
        near = np.abs(rows - curve[None, :]) <= band
        noise = (occluded & near).sum(axis=0) / float(max(2 * band, 1))
        trusted = noise < 0.35
        # A room where no column has a visible junction cannot be completed from itself, and
        # the provisional curve is then strictly better than interpolating out of nothing.
        if int(trusted.sum()) >= minimum_trusted:
            curve[:] = _cyclic_fill(curve, trusted)
        curve[:] = _cyclic_percentile(
            curve, window=max(3, int(round(width * _JUNCTION_WINDOW_FRAC))), q=_JUNCTION_QUANTILE[surface]
        )
        curve[:] = _cyclic_smooth(curve, sigma=max(2.0, width * _JUNCTION_SMOOTH_FRAC))

    ceiling_row = np.clip(ceiling_row, 0.0, float(height))
    floor_row = np.clip(floor_row, ceiling_row, float(height))
    return ceiling_row, floor_row


def estimate_shell(lab: np.ndarray, phi_deg: np.ndarray) -> RoomShell:
    """Find the room's ceiling, walls and floor, and how far every pixel is from each.

    The layout and the colour models define each other, so they are solved together, twice.

    Pass one seeds the palettes from fixed elevation bands. It has to: there is no layout
    yet. Those bands are crude — a fixed band cannot know that this room's ceiling is low —
    and worse, one of them is not even reliably the surface it claims. Measured on the
    reference rooms, the original ring-around-your-feet floor band came out **82 % furniture**,
    which taught the floor that a sofa was a floor colour, and nothing downstream could
    recover from that.

    Pass two throws those bands away and samples each surface where the *layout* says it is:
    the middle of the ceiling, the top of the wall, the perimeter of the floor. Those are the
    parts of a room furniture does not reach, and they can only be named once there is a
    layout to name them against.
    """
    height, width = lab.shape[:2]
    sector_of_col = _sector_of_column(width)

    shell: RoomShell | None = None
    for _ in range(_SHELL_PASSES):
        regions = _seed_regions(phi_deg, width, shell)
        modes, counts = _surface_models(lab, regions, sector_of_col)
        del regions
        distance = _distance_field(lab, modes, counts, sector_of_col)
        del modes, counts
        ceiling_row, floor_row = _solve_layout(distance, phi_deg)
        ceiling_row, floor_row = _complete_junctions(distance, ceiling_row, floor_row)
        shell = RoomShell(
            ceiling_row=ceiling_row,
            floor_row=floor_row,
            labels=_labels_from_boundaries(ceiling_row, floor_row, height),
            distance=distance,
        )

    assert shell is not None  # noqa: S101 - _SHELL_PASSES is a positive constant
    return shell


# ======================================================================================
# What to remove
# ======================================================================================


def _surface_residual(shell: RoomShell) -> np.ndarray:
    """How far each pixel is from the surface the shell says is behind it."""
    return np.take_along_axis(shell.distance, shell.labels[None].astype(np.intp), axis=0)[0]


def _classical_segment(lab: np.ndarray, shell: RoomShell, strength: float) -> np.ndarray:
    """Everything that does not look like the surface it is standing in front of.

    HYSTERESIS, NOT A THRESHOLD, and it is the difference between removing a sofa and
    removing the *parts of a sofa that happen to be far from the wall colour*. A single cut
    takes the dark blue of the cushions and leaves the mid-grey of the arm, and a half-erased
    object reads as damage rather than as an absence -- which is exactly the complaint that
    it looks like the furniture was painted over.

    So: a high threshold finds cores that are unmistakably not the room, and every connected
    region that reaches a lower threshold *and touches a core* comes with it. An object is
    connected to itself, so this takes objects whole; a stretch of shaded wall is connected
    to more shaded wall and to no core at all, so it is left alone. The same two-level latch
    the capture detectors use, for the same reason.

    Despeckling, the blob test and hole filling belong to :func:`furniture_mask`, so a
    learned backend inherits all of them by returning the one thing only a segmenter can.
    """
    _ = lab
    residual = _surface_residual(shell)
    high = _TAU_CONSERVATIVE + (_TAU_AGGRESSIVE - _TAU_CONSERVATIVE) * float(strength)
    low = high * _HYSTERESIS_RATIO

    core = residual > high
    if not core.any():
        return np.zeros(residual.shape, np.uint8)

    # GROWTH IS BOUNDED, and it has to be. Taking whole connected candidate regions is the
    # textbook hysteresis and it inverted the mask on a real room: at a low threshold the
    # candidate set is one region spanning the entire canvas -- a room's own shading clears
    # any threshold low enough to catch an object's mid-tones -- so every core dragged the
    # whole panorama in with it, and what survived were the objects, whose colours the
    # contaminated palette had learned. Measured: 52 % of the sphere selected, with holes
    # exactly where the furniture was.
    #
    # Morphological reconstruction with a step limit instead: an object's mid-tones lie
    # within a few dozen pixels of a core, so growth reaches them and stops well short of
    # the room.
    candidate = (residual > low).astype(np.uint8)
    grown = core.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(max(1, int(round(residual.shape[1] * _GROWTH_REACH_FRAC)))):
        expanded = cv2.dilate(grown, kernel) & candidate
        if np.array_equal(expanded, grown):
            break
        grown = expanded
    return grown * 255


#: Swap-in point for a semantic segmenter (SAM, or an ADE20K-trained model). Must return a
#: uint8 mask, 255 where a pixel is a furnishing rather than the room itself.
SEGMENT_BACKEND: Callable[[np.ndarray, RoomShell, float], np.ndarray] = _classical_segment


def _protect_poles(mask: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    protected = (phi_deg > _PROTECT_ABOVE_DEG) | (phi_deg < _PROTECT_BELOW_DEG)
    if protected.any():
        mask[protected, :] = 0
    return mask


def _describe(
    label_image: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    index: int,
    shell: RoomShell,
    residual: np.ndarray,
) -> RemovedObject:
    """A deliberately small taxonomy. Anything finer would be pretending to recognise."""
    height, width = shell.size
    x, y, box_w, box_h, area = (int(stats[index, k]) for k in range(5))
    centre_x, centre_y = float(centroids[index, 0]), float(centroids[index, 1])
    area_frac = area / float(height * width)
    aspect = box_w / float(max(box_h, 1))
    surface = int(shell.labels[int(np.clip(centre_y, 0, height - 1)), int(np.clip(centre_x, 0, width - 1))])

    if surface == CEILING:
        name = "ceiling_item"
    elif surface == WALL and area_frac < 0.004:
        name = "wall_item"
    elif aspect >= 2.4 and (y + box_h) > height * 0.86:
        name = "rug"
    elif area_frac >= 0.0035:
        name = "furniture"
    else:
        name = "clutter"

    # Confidence is how far the region actually was from the room, not a guess about what it
    # is: the only thing measured here is "this is not the surface behind it".
    region = label_image == index
    strength = float(np.mean(residual[region])) if region.any() else 0.0
    confidence = float(np.clip((strength - _TAU_AGGRESSIVE) / 6.0, 0.05, 0.95))
    return RemovedObject(label=name, confidence=round(confidence, 3), area_px=int(area))


def _semantic_mask(
    image: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(furniture, surface, classes)`` from a scene-parsing model, on the decision canvas.

    ``None`` when no model is installed, which is not an error: the caller falls back to the
    colour segmentation and says so through the response's ``model_backed``. ``classes`` is the
    ADE20K id per direction, which the plane fill needs to tell wall paint from a door, a
    window and a curtain — see :func:`_room_model`.
    """
    from biosphere.ai import furniture_seg  # noqa: PLC0415 - torch stays out of module scope

    result = furniture_seg.segment_equirect_labels(image)
    if result is None:
        return None
    mask, surface, classes = result
    return (
        resize_cyclic(mask, width, height, cv2.INTER_NEAREST),
        resize_cyclic(surface, width, height, cv2.INTER_NEAREST),
        resize_cyclic(classes, width, height, cv2.INTER_NEAREST),
    )


def furniture_mask(
    lab: np.ndarray, shell: RoomShell, phi_deg: np.ndarray, strength: float
) -> tuple[np.ndarray, list[RemovedObject]]:
    """The removal mask on the decision canvas, plus one entry per surviving region.

    A FURNISHING IS A BLOB, and the two tests below are what enforce it. Area alone does
    not: the shadow line where the ceiling meets the wall, and the skirting board where the
    wall meets the floor, both run the whole way round the panorama and are therefore among
    the largest connected regions on it while being a few pixels thick. Both are the room.
    Both were being erased — and the smear left behind where a straight architectural line
    used to be is far more noticeable than the sofa that stayed.
    """
    raw = SEGMENT_BACKEND(lab, shell, strength)
    return describe_mask(raw, shell, phi_deg)


def describe_mask(
    raw: np.ndarray, shell: RoomShell, phi_deg: np.ndarray
) -> tuple[np.ndarray, list[RemovedObject]]:
    """Clean a raw removal mask and report what survived. Shared by every segmenter.

    A FURNISHING IS A BLOB, and the two tests below are what enforce it. Area alone does
    not: the shadow line where the ceiling meets the wall, and the skirting board where the
    wall meets the floor, both run the whole way round the panorama and are therefore among
    the largest connected regions on it while being a few pixels thick. Both are the room.

    Kept out of the backends so that a model inherits the pole protection, the despeckle,
    the blob test and the silhouette close by returning nothing but a per-pixel decision.
    """
    height, width = shell.size
    raw = _protect_poles(raw.copy(), phi_deg)

    # Despeckle first. A close before this would weld the specks into the very lines the
    # blob test is about to look for.
    speck = max(3, int(round(width / 220.0)) | 1)
    opened = cv2.morphologyEx(
        raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (speck, speck))
    )

    count, label_image, stats, centroids = cv2.connectedComponentsWithStats(
        (opened > 0).astype(np.uint8), connectivity=8
    )
    # One distance transform for the whole mask; a region's largest inscribed radius is then
    # just its maximum, which is exactly "how thick is the thickest part of this thing".
    thickness = cv2.distanceTransform((opened > 0).astype(np.uint8), cv2.DIST_L2, 5)
    residual = _surface_residual(shell)

    minimum_area = max(16, int(round(_MIN_COMPONENT_FRAC * height * width)))
    minimum_radius = max(2.0, _MIN_RADIUS_AT_1024 * width / 1024.0)

    kept = np.zeros((height, width), np.uint8)
    regions: list[RemovedObject] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < minimum_area:
            continue
        selected = label_image == index
        if float(thickness[selected].max()) < minimum_radius:
            continue
        kept[selected] = 255
        regions.append(_describe(label_image, stats, centroids, index, shell, residual))

    # Only now: a hole inside a sofa is still sofa, and the fill needs the whole silhouette.
    if regions:
        fill = max(5, int(round(width / 96.0)) | 1)
        kept = cv2.morphologyEx(
            kept, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill, fill))
        )
        kept = _protect_poles(kept, phi_deg)

    regions.sort(key=lambda item: item.area_px, reverse=True)
    return kept, regions


# ======================================================================================
# Filling
# ======================================================================================


def _push_pull(colour: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Normalised-convolution pyramid: the smooth field the valid pixels imply, everywhere.

    Down the pyramid each level carries a colour sum and a coverage; up it, a pixel keeps
    its own value where it has one and takes the level above's where it does not. Two
    properties are the reason this and not an inpainter: the cost is linear in the pixel
    count regardless of how big the hole is, and there is no distance at which the fill runs
    out of information — a hole the size of a wardrobe is filled from the whole wall, not
    from its own rim.
    """
    height, width = weight.shape[:2]
    total = float(weight.sum())
    if total < 1.0:
        return np.zeros_like(colour)

    colours = [colour * weight[..., None]]
    weights = [weight]
    levels = max(1, int(math.log2(max(min(height, width), 4))) - 1)
    for _ in range(levels):
        if min(weights[-1].shape[:2]) <= 4:
            break
        colours.append(_pyr_down_cyclic(colours[-1]))
        weights.append(_pyr_down_cyclic(weights[-1]))

    coarse_w = np.maximum(weights[-1], _EPS)[..., None]
    out = colours[-1] / coarse_w
    for level in range(len(colours) - 2, -1, -1):
        level_h, level_w = weights[level].shape[:2]
        upper = _pyr_up_cyclic(out, level_h, level_w)
        alpha = np.clip(weights[level], 0.0, 1.0)[..., None]
        own = colours[level] / np.maximum(weights[level], _EPS)[..., None]
        out = own * alpha + upper * (1.0 - alpha)
        del upper, alpha, own
    del colours, weights
    return out


def _low_frequency_base(
    image: np.ndarray, labels: np.ndarray, valid: np.ndarray, work_width: int
) -> np.ndarray:
    """Per-surface push-pull, computed small and lifted back up.

    PER SURFACE, and that is the whole reason the result reads as a room. A single push-pull
    over the whole canvas fills the hole under a sofa from the wall *and* the floor, and the
    junction comes back as a soft vertical wash across both. Constraining each surface to its
    own valid pixels means the wall behind the wardrobe is built out of wall.

    Computed at `work_width` because this is by construction the low band: resolving it at
    4096 costs four times the memory to carry frequencies it does not contain.
    """
    height, width = image.shape[:2]
    work_w = min(work_width, width)
    work_h = max(2, int(round(work_w * height / float(width))))

    small = resize_cyclic(image, work_w, work_h, cv2.INTER_AREA)
    labels_s = resize_cyclic(labels, work_w, work_h, cv2.INTER_NEAREST)
    valid_s = resize_cyclic(valid, work_w, work_h, cv2.INTER_NEAREST)

    base_s = np.zeros_like(small)
    filled_any = False
    for surface in range(_N_SURFACES):
        selected = labels_s == surface
        if not selected.any():
            continue
        weight = (selected & (valid_s > 0)).astype(np.float32)
        if float(weight.sum()) < 32.0:
            # Every pixel of this surface is being removed. Nothing local to learn from, so
            # fall through to the global pass below rather than invent a colour.
            continue
        field = _push_pull(small, weight)
        base_s[selected] = field[selected]
        filled_any = True
        del field, weight

    # Whatever no surface could fill — an entirely occluded surface — is filled from the
    # whole panorama. Worse, and stated: it is a colour from the room rather than a smear.
    unresolved = ~np.any(
        np.stack([(labels_s == k) & (base_s[..., 0] != 0.0) for k in range(_N_SURFACES)]), axis=0
    )
    if unresolved.any() or not filled_any:
        weight = (valid_s > 0).astype(np.float32)
        field = _push_pull(small, weight)
        base_s[unresolved] = field[unresolved]
        del field, weight

    del small, labels_s, valid_s
    return resize_cyclic(base_s, width, height, cv2.INTER_CUBIC)


def _transplant_grain(
    detail: np.ndarray, labels: np.ndarray, valid: np.ndarray, hole: np.ndarray
) -> np.ndarray:
    """Take the high band from elsewhere on the same row of the same surface.

    A row of an equirect canvas is a circle of constant elevation. Sampling along it keeps
    the surface's distance from the camera, and therefore the projected size of its texture,
    exactly right — which a patch taken from anywhere else on the canvas would not. Wrapping
    costs nothing because the row is cyclic, and the source is required to be a *valid* pixel
    so a removed sofa can never be transplanted as grain.

    Indexed rather than rolled: a shift of a 4096x2048x3 float32 array is 100 MB of copy per
    candidate, while the pixels that still need one are a few percent of that.
    """
    height, width = hole.shape[:2]
    out = np.zeros_like(detail)
    rows, cols = np.nonzero(hole)
    if rows.size == 0:
        return out

    valid_bool = valid > 0
    for fraction in _TRANSPLANT_SHIFTS:
        if rows.size == 0:
            break
        shift = int(round(fraction * width)) % width
        if shift == 0:
            continue
        source = (cols + shift) % width
        usable = valid_bool[rows, source] & (labels[rows, source] == labels[rows, cols])
        if usable.any():
            take_rows, take_cols, take_src = rows[usable], cols[usable], source[usable]
            out[take_rows, take_cols] = detail[take_rows, take_src]
            keep = ~usable
            rows, cols = rows[keep], cols[keep]
    # Anything still unmatched keeps a zero residual: smooth, which is the honest look for a
    # direction in which this surface is nowhere else visible.
    return out


def _composite(original: np.ndarray, filled: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Take the fill inside the hole and nothing outside it.

    Full strength in the middle, ramping to nothing at the rim, so the repair has no border
    of its own AND no pixel outside the mask is touched at all. The mask was already grown
    past the object's contact shadow, so the ramp lives on background either way.
    """
    feather = max(2.0, original.shape[1] / 512.0)
    alpha = np.clip(cv2.distanceTransform(hole, cv2.DIST_L2, 5) / feather, 0.0, 1.0)[..., None]
    out = original.astype(np.float32) * (1.0 - alpha) + filled.astype(np.float32) * alpha
    return out.astype(np.uint8)


def _classical_fill(image: np.ndarray, hole: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, bool]:
    """Rebuild the room behind ``hole``. ``hole`` is 255 where pixels must be replaced."""
    height, width = image.shape[:2]
    if not np.any(hole):
        return image.copy(), False

    image_f = image.astype(np.float32)
    valid = np.where(hole > 0, np.uint8(0), np.uint8(255))

    base = _low_frequency_base(image_f, labels, valid, work_width=2048)

    # The split frequency follows the canvas so the grain is the same physical scale at any
    # delivered width.
    sigma = max(1.2, width / 1024.0)
    detail = image_f - cv2.GaussianBlur(_wrap_pad(image_f, 16), (0, 0), sigma)[:, 16:-16]
    # CLAMPED, for the reason `_reinject_grain` already records and this path was missing:
    # the split band carries edges as well as texture, and transplanting an edge pastes it
    # into the middle of a surface that has no such line. On a real bedroom that showed as
    # scratches dragged across the reconstructed floor -- the bed frame and the skirting,
    # copied along rows of constant elevation. The clamp keeps only what a surface has
    # everywhere, which is what "grain" was supposed to mean here.
    #
    # The GAIN stays where it was and deliberately differs from `_GRAIN_REINJECT_GAIN`: this
    # filler's base has no texture at all and needs the whole of what survives the clamp.
    np.clip(detail, -_GRAIN_CLAMP, _GRAIN_CLAMP, out=detail)
    grain = _transplant_grain(detail, labels, valid, hole)
    del detail

    filled = base + grain * _GRAIN_GAIN
    del base, grain
    np.clip(filled, 0.0, 255.0, out=filled)
    out = _composite(image, filled.astype(np.uint8), hole)
    del filled
    return out, False


def _learned_or_classical(
    image: np.ndarray, hole: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, bool]:
    """LaMa when its weights are on this machine, the classical filler when they are not.

    THE MODEL IS THE DEFAULT AND THE CLASSICAL PATH IS THE FALLBACK, which is the opposite
    of how the rest of this engine works and is right for exactly one reason: the pixels
    behind a sofa were never photographed. Every other stage measures something that is in
    the data. This one invents, and a network trained on rooms invents rooms, while a
    normalised-convolution pyramid invents a smooth wash — which is what "it looks like you
    painted over the furniture" means, precisely and fairly.

    Structure is the difference that shows. LaMa's Fourier convolutions see the whole canvas
    from the first layer, so a skirting board, a floorboard and the line where a wall meets a
    floor carry across the hole instead of stopping at its rim.
    """
    from biosphere.ai import backend, room_fill  # noqa: PLC0415 - torch stays out of module scope

    if not backend.ready():
        return _classical_fill(image, hole, labels)

    # THROUGH WINDOWS, NOT ON THE CANVAS. Asked to rebuild a room on the equirect directly,
    # the network produces a smooth field with no skirting and no floorboards -- it has never
    # seen a room drawn in that projection and there is no straight line in its input to
    # continue. Seen through a 90-degree window the same room is an ordinary photograph, and
    # it draws a wall meeting a floor. See `room_fill`.
    pano, leftover = room_fill.fill_by_views(image, hole, backend.inpaint_photo)

    # The fade at each window's rim, the poles, and anything no window reached. A few percent
    # of the sphere, where the canvas-wide pass's softness costs nothing.
    if np.any(leftover):
        result = backend.inpaint_photo(pano, leftover, cyclic=True)
        if result is not None:
            pano = _composite(pano, result, leftover)

    return _composite(image, _reinject_grain(image, pano, hole, labels), hole), True


def _reinject_grain(
    original: np.ndarray, filled: np.ndarray, hole: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Put the room's own grain back on top of the model's structure.

    LaMa is a structure machine and a texture compromise. Over a mask covering half the
    sphere its output is *correct and soft*: the skirting continues, the wall meets the floor
    where it should, and the floor itself arrives as a smooth field with none of the grain a
    floor has. That reads as a photograph of an empty room taken slightly out of focus.

    THE SAME GAIN SERVES THE DIFFUSION FILL, for a different reason arriving at the same
    place. That model invents plenty of texture of its own — but it runs at its 512-pixel
    training size and its window is enlarged back to 1024 on the way out, and a 2x enlargement
    is exactly what strips the fine tooth off whatever it drew. So both fills arrive carrying
    their structure and missing their grain, and both want the room's own.

    So the two halves come from where each is actually available. Structure from the network,
    which is the only thing that can invent it; grain from the panorama, which already
    contains the real thing. The transplant runs along rows of constant elevation and takes
    only from the same surface, so a floor borrows floor at its own distance and therefore at
    its own texture scale -- which is the whole reason the equirect projection makes this
    correct rather than merely convenient.
    """
    if _GRAIN_REINJECT_GAIN <= 0.0:
        return filled

    # GRAIN, NOT STRUCTURE, and the distinction is the whole of it. A wide split band carries
    # the skirting board and the cornice as well as the plaster, and transplanting those
    # pastes an edge into the middle of a wall: the first attempt at this put what looked
    # like scratches across every reconstructed surface. A fine band and a hard clamp keep
    # only what a surface has everywhere -- the tooth of the paint, the flecks in the floor.
    sigma = max(1.0, original.shape[1] / 2048.0)
    original_f = original.astype(np.float32)
    detail = original_f - cv2.GaussianBlur(_wrap_pad(original_f, 16), (0, 0), sigma)[:, 16:-16]
    np.clip(detail, -_GRAIN_CLAMP, _GRAIN_CLAMP, out=detail)

    valid = np.where(hole > 0, np.uint8(0), np.uint8(255))
    grain = _transplant_grain(detail, labels, valid, hole > 0)
    del detail

    out = filled.astype(np.float32) + grain * _GRAIN_REINJECT_GAIN
    del grain
    np.clip(out, 0.0, 255.0, out=out)
    return out.astype(np.uint8)


#: Swap-in point for the inpainter. Must return ``(image, model_backed)``. The surface labels
#: are passed because a method that wants them can use them and one that does not can ignore
#: them — the classical filler needs them to keep floor colour off a wall, LaMa does not.
FILL_BACKEND: Callable[..., tuple[np.ndarray, bool]] = _learned_or_classical

#: Whether a segmented room is rebuilt on its own planes (``room_planes``) before anything else
#: is asked. A switch rather than a constant only so ``scripts/bench_empty_room.py`` can put the
#: two fills side by side on the same mask.
PLANE_FILL = True


def fill_room(
    image: np.ndarray, hole: np.ndarray, labels: np.ndarray, *, allow_model: bool = True
) -> tuple[np.ndarray, bool]:
    if not allow_model:
        return _classical_fill(image, hole, labels)
    return FILL_BACKEND(image, hole, labels)


# ======================================================================================
# The room's own planes
# ======================================================================================


@dataclass(slots=True)
class RoomModel:
    """What the plane fill needs to know about a room, on the delivered canvas."""

    box: Any
    #: 0 ceiling, 1 wall PAINT, 2 floor, 255 anything else — only these are samples of a plane.
    surface: np.ndarray
    #: 255 where something the product keeps (a door, a window, a curtain, a radiator) stands.
    held: np.ndarray
    #: 1 where that something is a curtain, which the fill continues rather than paints over.
    layer: np.ndarray
    #: Rows the photographed floor line sits below the box's, per column (``floor_line_shift``).
    floor_shift: np.ndarray | None = None
    #: How the box was fitted, and what each plane's fill did.
    fit: dict[str, Any] = field(default_factory=dict)
    report: list[Any] = field(default_factory=list)


#: ADE20K ids of the room's own three surfaces, as `furniture_seg` labels them.
_ADE_WALL, _ADE_FLOOR, _ADE_CEILING = 0, 3, 5


def _room_model(image: np.ndarray, model_surface: np.ndarray, classes: np.ndarray, hole: np.ndarray) -> RoomModel | None:
    """The room's box and per-pixel surface roles, or ``None`` when no box can be fitted.

    The box is fitted to the junctions the panorama actually shows (:func:`room_planes.fit_box`),
    with :func:`biosphere.plan.outline.outline_from_surface` as the starting point and the
    fallback. Not the outline as it stands: it is built to draw a plan, and in a cluttered room it
    reads furniture feet as the floor's edge — see ``fit_box`` for what that measured.
    """
    from biosphere.ai import furniture_seg, room_planes  # noqa: PLC0415 - keeps import cost lazy
    from biosphere.plan import outline as plan_outline  # noqa: PLC0415

    try:
        outline = plan_outline.outline_from_surface(model_surface)
    except Exception as exc:  # noqa: BLE001 - a room the fit cannot read falls back, it does not fail
        log.info("empty_room.no_box", reason=str(exc)[:160])
        return None
    if len(outline.polygon) < 3:
        return None
    box = room_planes.RoomBox.from_outline(outline)
    if not box.convex_around_eye():
        # An L-shaped fit is traced exactly, but a fit that puts the camera outside its own room
        # is not a room; the learned or classical fill does better than a wrong box.
        polygon = box.polygon
        inside = cv2.pointPolygonTest(polygon.astype(np.float32).reshape(-1, 1, 2), (0.0, 0.0), False) > 0
        if not inside:
            log.info("empty_room.no_box", reason="camera outside the fitted outline")
            return None

    height, width = image.shape[:2]
    full = resize_cyclic(classes, width, height, cv2.INTER_NEAREST)
    fitted, shift, fit_report = room_planes.fit_room(image, classes, hole, box)
    if fitted is not None:
        box = fitted
    surface = np.full((height, width), 255, np.uint8)
    surface[full == _ADE_CEILING] = CEILING
    surface[full == _ADE_WALL] = WALL
    surface[full == _ADE_FLOOR] = FLOOR
    keep = np.zeros(256, bool)
    for class_id in furniture_seg._ROOM_SHELL:  # noqa: SLF001 - the one definition of "the room"
        keep[class_id] = True
    keep[[_ADE_WALL, _ADE_FLOOR, _ADE_CEILING]] = False
    held = np.where(keep[full], np.uint8(255), np.uint8(0))
    curtain = np.zeros(256, bool)
    for class_id in furniture_seg.CURTAIN_CLASSES:
        curtain[class_id] = True
    layer = curtain[full].astype(np.uint8)
    log.info("empty_room.box", walls=box.walls, ceiling=round(float(box.ceiling), 3), fit=fit_report)
    return RoomModel(box=box, surface=surface, held=held, layer=layer, floor_shift=shift, fit=fit_report)


#: Chroma distance (a*b*) from the room's own wall or floor colour at which a pixel the
#: segmenter called wall or floor is treated as something STANDING on it. Chroma, because the
#: same paint is 30 L* darker in a corner than under the lamp but keeps its hue; the rug the
#: model called floor in the reference bedroom sits 20-40 chroma from the laminate.
_OUTLIER_DE: dict[int, float] = {_ADE_WALL: 14.0, _ADE_FLOOR: 16.0}

#: L* under the surface's own at which a pixel is something dark standing on it.
_DARK_DROP_L = 25.0


#: L* below the room's floor at which floor beside a removed object counts as its shadow.
#: The reference bedroom's white laminate varies by 3-5 under its light; a bed's shadow is 10-25.
_SHADOW_DROP_L = 8.0


def _complete_mask(image: np.ndarray, mask: np.ndarray, classes: np.ndarray, room: RoomModel) -> np.ndarray:
    """Grow the removal mask over what the segmenter called room but the room says is not.

    THE SEGMENTER'S MISSES ARE THE MOST VISIBLE THING ABOUT AN EMPTIED ROOM, because the fill
    removes everything around them: on the reference bedroom a black bag and a chair base the
    model called "floor" were left standing on an empty floor, the white top panel of a wardrobe
    it called "wall" floated where the wardrobe had been, and its glass doors' reflections of
    the ceiling light — called "ceiling" — hung in mid-air. Each is caught by a fact about rooms
    the per-pixel model cannot use:

    * GEOMETRY. Wall paint does not lie below the floor line, and ceiling does not hang below the
      ceiling line. The box says where both lines are.
    * COLOUR. A surface is one paint or one tiling: a pixel far from its surface's own colour in
      the same direction (:data:`_OUTLIER_DE`) is not that surface.

    Only candidates TOUCHING what is already being removed are taken — a dark picture frame
    alone on a wall stays, the bag beside the removed bed goes — and nothing the product holds
    (a door, a window, a curtain, a radiator) is ever added. Doorways are left alone entirely: the
    next room's floor and walls are, rightly, neither this room's colour nor in its box.
    """
    from biosphere.ai import room_planes  # noqa: PLC0415

    dh, dw = classes.shape
    height, width = mask.shape
    small = resize_cyclic(image, dw, dh, cv2.INTER_AREA)
    lab = _to_lab(small)
    removed = resize_cyclic(mask, dw, dh, cv2.INTER_AREA) > 64

    rows = np.arange(dh, dtype=np.float64)[:, None]
    shift = room.floor_shift if room.floor_shift is not None else np.zeros(width)
    columns = (np.arange(dw) * width) // dw
    floor_line = room_planes.floor_rows(room.box, dw, dh) + shift[columns] * (dh / float(height))
    theta = 2.0 * np.pi * (np.arange(dw) + 0.5) / dw - np.pi
    reach = room_planes.trace(room.box, np.stack([np.sin(theta), np.zeros(dw), np.cos(theta)], axis=1))
    distance = np.hypot(reach.wall_point[:, 0], reach.wall_point[:, 2])
    elevation = np.arctan2(room.box.eye_to_ceiling, np.maximum(distance, 1e-6))
    ceiling_line = (0.5 * np.pi - elevation) / np.pi * dh - 0.5
    degree = dh / 180.0
    expect_floor = rows > floor_line[None, :] + 1.0 * degree
    expect_wall = (rows < floor_line[None, :] - 2.0 * degree) & (rows > ceiling_line[None, :] + 2.0 * degree)

    candidate = ((classes == _ADE_WALL) & expect_floor) | ((classes == _ADE_CEILING) & expect_wall)

    # CONTACT SHADOWS. The floor right beside where something stood is darker than the floor —
    # it was in that thing's shadow — and once the thing is gone the shadow is a dark halo in the
    # shape of what was removed. Floor within a couple of degrees of the mask and clearly darker
    # than the room's floor goes with it.
    floor_clean = (classes == _ADE_FLOOR) & expect_floor & ~removed
    if int(floor_clean.sum()) >= 200:
        floor_light = float(np.median(lab[..., 0][floor_clean]))
        near = cv2.dilate(removed.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        candidate |= (classes == _ADE_FLOOR) & expect_floor & near & (lab[..., 0] < floor_light - _SHADOW_DROP_L)

    for class_id, threshold in _OUTLIER_DE.items():
        where = (classes == class_id) & (expect_wall if class_id == _ADE_WALL else expect_floor)
        clean = where & ~removed
        if int(clean.sum()) < 200:
            continue
        # ONE REFERENCE FOR THE ROOM, not one per direction: a black bag the model called floor
        # can be most of the "floor" in its own few degrees, and a local median then calls the
        # bag the floor. A room's floor is one material and its walls one paint.
        ref = np.median(lab[clean], axis=0)
        diff = lab[where] - ref
        chroma = np.sqrt(diff[:, 1] ** 2 + diff[:, 2] ** 2)
        # Colour that is not the surface's, or a surface far DARKER than itself: the black chair
        # base the model called floor sat 50 L* under the floor at no chroma at all. Brighter is
        # not counted — a glossy floor's reflections and a lit wall are brighter, and nothing
        # standing in a room is a hole of light.
        spot = np.zeros((dh, dw), bool)
        spot[where] = (chroma > threshold) | (diff[:, 0] < -_DARK_DROP_L)
        candidate |= spot

    from biosphere.ai import furniture_seg  # noqa: PLC0415

    keep = np.zeros(256, bool)
    for class_id in furniture_seg._ROOM_SHELL:  # noqa: SLF001 - the one definition of "the room"
        keep[class_id] = True
    keep[[_ADE_WALL, _ADE_FLOOR, _ADE_CEILING]] = False
    candidate &= ~keep[classes]
    # Around a door — the leaf and the opening beside it — only the geometric rule applies.
    # Local, not per column: measured on the reference bedroom, a sliver of the wardrobe's glass
    # called "door" put a whole column of reflections out of reach.
    reach_px = 2 * int(round(4 * dw / 360)) + 1
    door = cv2.dilate((classes == 14).astype(np.uint8), np.ones((reach_px, reach_px), np.uint8)) > 0
    candidate &= ~door | (expect_floor & (classes == _ADE_WALL))
    _, phi = geo.equirect_angles(dw, dh, dtype=np.float32)
    candidate &= ((np.degrees(phi) <= _PROTECT_ABOVE_DEG) & (np.degrees(phi) >= _PROTECT_BELOW_DEG))[:, None]
    if not candidate.any():
        return mask

    touching = cv2.dilate(removed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    taken = np.unique(labels[touching & candidate])
    taken = taken[taken > 0]
    if taken.size == 0:
        return mask
    added = np.isin(labels, taken)
    log.info("empty_room.mask_completed", added=round(float(added.mean()), 4), components=int(taken.size))
    grown = resize_cyclic(added.astype(np.uint8) * 255, width, height, cv2.INTER_NEAREST)
    radius = max(2, int(round(_DILATE_PX_AT_4096 * width / 4096.0)))
    grown = cv2.dilate(grown, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    _, phi_full = geo.equirect_angles(width, height, dtype=np.float32)
    grown = _protect_poles(grown, np.degrees(phi_full).astype(np.float32))
    full_classes = resize_cyclic(classes, width, height, cv2.INTER_NEAREST)
    grown[keep[full_classes]] = 0
    return np.maximum(mask, grown)


def _pole_caps(width: int, height: int) -> np.ndarray:
    """255 over the protected caps: the stitcher's synthesis and branding, never photographed."""
    _, phi = geo.equirect_angles(width, height, dtype=np.float32)
    phi_deg = np.degrees(phi)
    caps = ((phi_deg > _PROTECT_ABOVE_DEG) | (phi_deg < _PROTECT_BELOW_DEG)).astype(np.uint8) * 255
    return np.repeat(caps[:, None], width, axis=1)


def _plane_fill(
    image: np.ndarray, hole: np.ndarray, labels: np.ndarray, room: RoomModel, *, allow_model: bool
) -> np.ndarray:
    """Rebuild ``hole`` on the room's planes; whatever no plane answered falls to ``fill_room``.

    See :mod:`biosphere.ai.room_planes` for why the planes and not a network: on the reference
    bedroom every learned fill turned the wardrobe and the bed into grey and black smears,
    because the pixels around a hole that size are the remains of more furniture.
    """
    from biosphere.ai import room_planes  # noqa: PLC0415

    height, width = image.shape[:2]
    camera = room_planes.EquirectCamera(width, height)
    painted, filled, report = room_planes.fill_planes(
        image,
        camera,
        room.box,
        hole,
        room.surface,
        held=room.held,
        layer=room.layer,
        floor_shift=room.floor_shift,
        exclude=_pole_caps(width, height),
    )
    leftover = (hole > 127) & ~filled
    if leftover.any():
        # A sliver no plane could answer for — a wall the camera barely saw. Small by
        # construction, which is the size the older fills are good at.
        rest, _ = fill_room(painted, leftover.astype(np.uint8) * 255, labels, allow_model=allow_model)
        painted[leftover] = rest[leftover]
    room.report = report
    return _composite(image, painted, hole)


# ======================================================================================
# Planning one run
# ======================================================================================


def plan_empty_room(image: np.ndarray, strength: float, *, allow_model: bool = True) -> RoomPlan:
    """Decide what to remove and rebuild it, all in memory. No storage, no scene, no I/O.

    Separated from :func:`empty_room` so the algorithm can be tested against a procedurally
    generated room without a database row, a media root or a scene on disk.
    """
    height, width = image.shape[:2]
    decision_w = min(_DECISION_WIDTH, width)
    decision_h = max(2, int(round(decision_w * height / float(width))))

    decision = resize_cyclic(image, decision_w, decision_h, cv2.INTER_AREA)
    _, phi = geo.equirect_angles(decision_w, decision_h, dtype=np.float32)
    phi_deg = np.degrees(phi).astype(np.float32)

    lab = _to_lab(decision)
    shell = estimate_shell(lab, phi_deg)

    # A SCENE-PARSING MODEL WHEN THERE IS ONE, COLOUR ONLY WHEN THERE IS NOT.
    #
    # The colour segmentation reaches an F1 near 0.6 on a procedurally generated room and
    # inverts on a real cluttered one: with shelves, boxes and hanging clothes covering the
    # walls, the "wall palette" learns the clutter and the wall itself becomes the foreign
    # thing. A model that has been shown a hundred thousand rooms does not have that failure
    # mode, and it answers the question this feature actually asks -- is this the room, or is
    # it something standing in it -- rather than a proxy for it.
    semantic = _semantic_mask(image, decision_h, decision_w) if allow_model else None
    model_backed = semantic is not None
    small_surface = shell.labels
    model_surface = classes = None
    if semantic is not None:
        semantic_mask, model_surface, classes = semantic
        small_mask, regions = describe_mask(semantic_mask, shell, phi_deg)
        # The model is authoritative about which pixels are wall, floor and ceiling; the
        # geometric shell covers what it left unlabelled, which is exactly the furniture
        # about to be removed and therefore exactly where the fill needs to know what
        # surface belongs behind it.
        small_surface = np.where(model_surface == 255, shell.labels, model_surface).astype(np.uint8)
    else:
        small_mask, regions = furniture_mask(lab, shell, phi_deg, strength)
    del lab, decision

    # Lift the decision to the delivered canvas. NEAREST on the labels (they are ids, not
    # numbers); LINEAR then a threshold on the mask, which puts the boundary at the same
    # place a human would have drawn it rather than on the decision canvas's pixel grid.
    labels = resize_cyclic(small_surface, width, height, cv2.INTER_NEAREST)
    mask = resize_cyclic(small_mask, width, height, cv2.INTER_LINEAR)
    mask = np.where(mask >= 128, np.uint8(255), np.uint8(0))
    # Kept undilated for the room fit: a junction visible BETWEEN a desk's legs is a junction,
    # and the dilation below exists for the fill's sake, not to hide the room from the fit.
    furniture_only = mask.copy()

    if np.any(mask):
        radius = max(2, int(round(_DILATE_PX_AT_4096 * (0.5 + strength) * width / 4096.0)))
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        )
        # The dilation can walk into a pole; the protection is absolute, so re-apply it.
        _, phi_full = geo.equirect_angles(width, height, dtype=np.float32)
        mask = _protect_poles(mask, np.degrees(phi_full).astype(np.float32))

    # The fill reports whether a model drew it. Not folded into `model_backed` below — see the
    # note there — but worth having in the log, because "the room came back soft" and "the room
    # came back invented" are different complaints with different fixes.
    #
    # THE ROOM'S OWN PLANES FIRST whenever the segmenter gave a room to fit a box to. See
    # `room_planes` for the measurement: LaMa through windows turned the reference bedroom's
    # wardrobe and beds into grey and black smears, and no learned fill was asked for an
    # absence without inventing something.
    room = (
        _room_model(image, model_surface, classes, furniture_only)
        if (np.any(mask) and model_surface is not None and classes is not None and PLANE_FILL)
        else None
    )
    if room is not None:
        mask = _complete_mask(image, mask, classes, room)

    weights = geo.sphere_solid_angle_weights(height, width)
    removed_fraction = float(weights[mask > 0].sum()) if np.any(mask) else 0.0
    del weights

    if room is not None:
        result, fill_model, fill_name = _plane_fill(image, mask, labels, room, allow_model=allow_model), False, "planes"
    else:
        result, fill_model = fill_room(image, mask, labels, allow_model=allow_model)
        fill_name = "learned" if fill_model else "classical"
    log.info("empty_room.filled", fill=fill_name, fill_model=bool(fill_model), mask_model=bool(model_backed))
    del labels
    return RoomPlan(
        image=result,
        mask=mask,
        shell=shell,
        regions=regions,
        removed_fraction=removed_fraction,
        # THE SEGMENTATION, AND ONLY THE SEGMENTATION. A learned fill behind a colour-guessed
        # mask is still guessing where the furniture is, and that is the half a user can see
        # going wrong — which is what this flag has always meant and said.
        #
        # It also used to require a learned FILL, and that quietly became a bug the moment the
        # default fill stopped being one: `_MAX_REMOVED_FRACTION_MODEL` is selected by this
        # flag, and its own note is explicit that the cap "differs by who drew the mask". A
        # scene-parsing model saying half a sphere is beds and desks is an ordinary bedroom;
        # a colour test saying the same thing is a failed palette. Tying the flag to the fill
        # judged the first by the second's limit and refused rooms it had read correctly.
        model_backed=bool(model_backed),
        room=room,
        fill=fill_name,
        planes=list(room.report) if room is not None else [],
    )


# ======================================================================================
# Scene entry point
# ======================================================================================


def source_level(scene: Scene, storage: LocalStorage) -> tuple[Path, str]:
    """The widest rung on disk; a derived master is built from it and then capped.

    Public because :mod:`biosphere.ai.restyle_pano` delivers its own derived panorama and has
    to start from the same rung and end in the same shape of ladder. "Take a panorama that was
    made from a scene's own and hand it to the player" is one job with one right answer, and a
    second implementation of it would be a second place for the preview rung to go missing.
    """
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


def _cache_key(source: Path, strength: float) -> dict[str, Any]:
    try:
        mtime = int(source.stat().st_mtime)
    except OSError:  # pragma: no cover - the file was just read
        mtime = 0
    return {
        "version": _CACHE_VERSION,
        "strength": round(float(strength), 3),
        "source": source.name,
        "source_mtime": mtime,
    }


def staged_levels(
    staged_dir: Path, storage: LocalStorage, width: int, height: int
) -> PanoramaLevels | None:
    """Read the pyramid back off disk, or ``None`` if a rung the player needs is missing.

    The master's size comes from the cache record rather than from the file, because reading
    it back would mean decoding a 4K JPEG to learn two integers that were known when it was
    written — and the cheap way of doing that (a reduced decode, multiplied back up) is only
    exact while every width stays a multiple of eight.
    """
    urls: dict[str, str | None] = {"preview_url": None, "sd_url": None, "hd_url": None}
    for name, _ in STAGED_LEVEL_WIDTHS:
        path = staged_dir / f"pano_{name}.jpg"
        if path.exists():
            urls[f"{name}_url"] = storage.url_for(path)
        elif name != "sd":
            # `sd` is legitimately absent when the master is small; the other two are not.
            return None
    if not urls["preview_url"] or not urls["hd_url"] or width <= 0 or height <= 0:
        return None
    return PanoramaLevels(
        preview_url=urls["preview_url"],
        sd_url=urls["sd_url"],
        hd_url=urls["hd_url"],
        # Never a `uhd`: see `_STAGED_MAX_WIDTH`. The player's ladder simply tops out here.
        uhd_url=None,
        width=width,
        height=height,
        cube_urls=None,
    )


def write_levels(staged_dir: Path, panorama: np.ndarray, storage: LocalStorage) -> PanoramaLevels:
    """Write the delivered pyramid, mirroring what the stitch job writes for a scene.

    Public for the same reason as :func:`source_level`."""
    height, width = panorama.shape[:2]
    urls: dict[str, str | None] = {"preview_url": None, "sd_url": None, "hd_url": None}

    for name, target in STAGED_LEVEL_WIDTHS:
        level_w = min(target, width)
        # THE PREVIEW IS NEVER SKIPPED. It is the rung the player paints first and the one
        # `Player` gates the whole canvas on, so a master small enough that 512 px is not a
        # downscale still gets one — blurred at its own size, which is what makes it a
        # preview rather than a duplicate. Only `sd` may be dropped, and only when it would
        # be the same width as the master and therefore the same bytes twice.
        if name == "sd" and level_w >= width:
            continue
        level_h = max(1, int(round(level_w * height / float(width))))
        image = (
            panorama
            if level_w == width and level_h == height
            else resize_cyclic(panorama, level_w, level_h, cv2.INTER_AREA)
        )
        if name == "preview":
            image = cv2.GaussianBlur(image, (0, 0), max(level_w / 256.0, 0.8))
            urls["preview_url"] = storage.url_for(write_jpeg(staged_dir / "pano_preview.jpg", image, 70))
            write_webp(staged_dir / "pano_preview.webp", image, 65)
        else:
            urls[f"{name}_url"] = storage.url_for(write_jpeg(staged_dir / f"pano_{name}.jpg", image, 92))
        if image is not panorama:
            del image

    return PanoramaLevels(
        preview_url=urls["preview_url"] or "",
        sd_url=urls["sd_url"],
        hd_url=urls["hd_url"],
        uhd_url=None,
        width=width,
        height=height,
        cube_urls=None,
    )


def empty_room(
    scene: Scene,
    *,
    storage: LocalStorage,
    strength: float | None = None,
    refresh: bool = False,
    allow_model: bool = True,
    budget: float = TIME_BUDGET_S,
) -> EmptyRoomResponse:
    """Remove every furnishing from a scene's panorama and deliver the result as a pyramid.

    Idempotent and cached: the same scene at the same strength is computed once and then
    answered off disk, because the button that calls this is a *toggle* and a user flicking
    it back and forth must not queue a minute of CPU per flick.
    """
    started = time.perf_counter()
    strength_value = DEFAULT_STRENGTH if strength is None else float(min(max(strength, 0.0), 1.0))

    source_path, level_name = source_level(scene, storage)
    staged_dir = storage.scene_dir(scene.property_id, scene.id) / STAGED_DIRNAME
    meta_path = staged_dir / "meta.json"
    key = _cache_key(source_path, strength_value)

    if not refresh and meta_path.exists():
        try:
            cached = orjson.loads(meta_path.read_bytes())
        except (OSError, orjson.JSONDecodeError):
            cached = None
        if isinstance(cached, dict) and all(cached.get(k) == v for k, v in key.items()):
            levels = staged_levels(
                staged_dir, storage, int(cached.get("width", 0)), int(cached.get("height", 0))
            )
            if levels is not None:
                return EmptyRoomResponse(
                    scene_id=scene.id,
                    levels=levels,
                    mask_url=storage.url_for(staged_dir / "mask.png")
                    if (staged_dir / "mask.png").exists()
                    else None,
                    regions=[RemovedObject(**item) for item in cached.get("regions", [])],
                    removed_fraction=float(cached.get("removed_fraction", 0.0)),
                    strength=strength_value,
                    elapsed_ms=int(round((time.perf_counter() - started) * 1000)),
                    cached=True,
                    model_backed=bool(cached.get("model_backed", False)),
                )

    image = _imread(source_path)
    height, width = image.shape[:2]
    if width > _STAGED_MAX_WIDTH:
        target_h = max(2, int(round(_STAGED_MAX_WIDTH * height / float(width))))
        image = resize_cyclic(image, _STAGED_MAX_WIDTH, target_h, cv2.INTER_AREA)

    plan = plan_empty_room(image, strength_value, allow_model=allow_model)
    del image

    cap = _MAX_REMOVED_FRACTION_MODEL if plan.model_backed else _MAX_REMOVED_FRACTION_CLASSICAL
    if plan.removed_fraction > cap:
        raise ApiException(
            "room_not_recognised",
            (
                f"{plan.removed_fraction:.0%} of this panorama does not look like a wall, a "
                "floor or a ceiling, so there is no room here to keep."
            ),
            (
                f"Bu panoramanın %{plan.removed_fraction * 100:.0f}'i duvar, zemin ya da tavana "
                "benzemiyor; geriye bırakılacak bir oda kalmıyor."
            ),
            hint=(
                "Furnishings are found by how far they sit from the surface behind them, "
                "which needs an indoor panorama with recognisable walls and a floor."
            ),
            status_code=422,
        )

    if not plan.regions:
        raise ApiException(
            "nothing_to_remove",
            "Nothing in this panorama separates from the room itself.",
            "Bu panoramada odadan ayrışan bir eşya bulunamadı.",
            hint=(
                "Furnishings are found by how far they sit from the colour of the wall or "
                "floor behind them; raise 'strength', or outline the object with "
                "/ai/hide-furniture instead."
            ),
            status_code=422,
        )

    elapsed = time.perf_counter() - started
    if elapsed > budget:  # pragma: no cover - a guard, not a path
        log.warning("empty_room.over_budget", scene_id=scene.id, seconds=round(elapsed, 2))

    staged_dir.mkdir(parents=True, exist_ok=True)
    levels = write_levels(staged_dir, plan.image, storage)
    mask_path = write_png(staged_dir / "mask.png", plan.mask)

    payload = dict(key)
    payload["width"] = int(levels.width)
    payload["height"] = int(levels.height)
    payload["removed_fraction"] = round(plan.removed_fraction, 5)
    payload["model_backed"] = bool(plan.model_backed)
    payload["regions"] = [item.model_dump() for item in plan.regions]
    meta_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    log.info(
        "empty_room.done",
        scene_id=scene.id,
        source_level=level_name,
        regions=len(plan.regions),
        removed_fraction=round(plan.removed_fraction, 4),
        elapsed_ms=elapsed_ms,
    )
    return EmptyRoomResponse(
        scene_id=scene.id,
        levels=levels,
        mask_url=storage.url_for(mask_path),
        regions=plan.regions,
        removed_fraction=plan.removed_fraction,
        strength=strength_value,
        elapsed_ms=elapsed_ms,
        cached=False,
        model_backed=plan.model_backed,
    )
