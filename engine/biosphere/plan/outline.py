"""The shape of one room, read off its own panorama.

The whole feature rests on one line of geometry the engine already trusts twice — in
:mod:`biosphere.ai.measure` for single-view metrology and in the player's ``portalGeometry``
for floor rings:

    a direction below the horizon meets the floor at  h / tan(depression)

So if you can find, for every azimuth, the elevation at which the *wall meets the floor*, you
have the distance to the wall in every direction: a room's plan, at the price of one number
you have to assume (the camera's height) and one you have to detect (that junction).

Finding the junction: two boundaries, and why the higher one wins
----------------------------------------------------------------
The junction is found from the semantic surface map that :func:`biosphere.ai.furniture_seg
.segment_equirect` already produces for the restyling and emptying tools — deliberately the
same one, because two different opinions about where the floor is would put the plan's walls
somewhere other than the walls those tools protect.

The floor boundary alone is not enough, and the reason is the ordinary contents of a bedroom:
a wardrobe standing against the far wall hides the last two metres of floor in front of it,
so the topmost floor pixel in those columns is the wardrobe's foot and the wall comes out
*nearer* than it is. Furniture can only ever do that — it stands on the floor and occludes
away from the camera — so the error has a known sign, and a rule follows from it:

    the ceiling line, above the same wall, is rarely occluded by anything;
    and since occlusion can only shorten a radius, the LONGER of the two answers is the
    better one.

The ceiling line needs the ceiling's height to become a distance, and that is not assumed —
it is measured. In every column where both boundaries are visible the two see the same wall,
so ``tan(elevation_ceiling) / tan(depression_floor)`` is the ceiling height above the eye in
camera heights, and it is the *same number* in every such column. The estimator below takes
the mode of that ratio rather than its mean, because the columns that disagree disagree for a
reason (a wardrobe below, a high cupboard above) rather than by noise.

Measured on three procedurally furnished rooms whose dimensions are known exactly
(``scripts/bench_floorplan.py``), against a perfect surface map: floor boundary alone, 88 cm
of mean radial error and 0.50 IoU with the true floor; with the ceiling rule, **1.4 cm and
0.99**. The rule is not a refinement, it is the difference between a plan and a guess.

A measurement, and then a room
------------------------------
The profile above is what was measured. It is NOT what gets drawn, and the difference is the
second half of this module: the outline is a rectilinear room fitted to the profile, because
a profile is the distance to the nearest thing in every direction and a room is not that
shape. See the "Profile -> a room" section for what that fit may and may not claim.

Measured end to end on the same three rooms, with the real segmenter rather than a perfect
mask: the fitted room's floor agrees with the true floor to **0.984-0.991 IoU**, and its
dimensions to within 9 cm, against 0.88-0.95 for the drawn-as-measured outline.

What this is not
----------------
It is not a reconstruction. There is no depth here, no multi-view geometry and no learned
layout model; there is a semantic mask, an assumed height and trigonometry. Its failure modes
are therefore legible rather than mysterious, and each one is a named number below.

The known limits, in the order they matter:

* **A mirror is a room.** Nothing here can tell one from a doorway, and unlike a doorway a
  mirror can be big enough for the fit to believe.
* **Rooms that are neither a rectangle nor an L come out as the nearest of the two.** That
  ceiling is deliberate; a model free enough to draw any shape draws the measurement's noise.
* **The scale is an assumption**, and the whole plan scales linearly with it.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from biosphere.core.logging import get_logger
from biosphere.plan.types import (
    BINS,
    CAMERA_HEIGHT_M,
    MAX_RADIUS_H,
    MIN_RADIUS_H,
    BinSource,
    RadialProfile,
    RoomOutline,
)

__all__ = [
    "SURFACE_CEILING",
    "SURFACE_WALL",
    "SURFACE_FLOOR",
    "MIN_RUN_DEG",
    "CEILING_RANGE_H",
    "profile_from_surface",
    "polygon_from_profile",
    "outline_from_surface",
    "outline_from_pano",
]

log = get_logger("biosphere.plan.outline")

#: The three values :func:`biosphere.ai.furniture_seg.segment_equirect` writes, plus the
#: 255 it leaves where no window could see. Named here so this module never spells them as
#: bare integers, and imported nowhere else so the two files cannot drift apart silently.
SURFACE_CEILING = 0
SURFACE_WALL = 1
SURFACE_FLOOR = 2

#: How much uninterrupted surface has to be found before a boundary is believed, in degrees
#: of elevation.
#:
#: A semantic mask frays at every edge and speckles in the middle of large flat regions; one
#: stray floor pixel four degrees above the true junction moves that wall a metre and a half
#: further away. Requiring a run means the boundary is the start of the actual region rather
#: than the first pixel that agreed with it. One and a half degrees is four rows on the 512-
#: row canvas the segmenter answers on, and is under a tenth of the angle any real junction
#: subtends.
MIN_RUN_DEG = 1.5

#: Plausible ceiling height above the FLOOR, in camera heights, as an inclusive range.
#:
#: 1.45 to 2.6 camera heights is 2.3 m to 4.2 m at the assumed height: a low modern flat at
#: one end and a high old apartment at the other. Outside it the ratio estimate is rejected
#: rather than clamped, because a value out here does not mean "an unusual room", it means the
#: ceiling boundary was not the ceiling — and a clamped wrong number is a wrong number that
#: has stopped announcing itself.
CEILING_RANGE_H = (1.45, 2.6)

#: Share of columns that must offer both boundaries before the ceiling height is fixed at all.
#: Under it the profile is floor-only and says so, which is the honest answer for a capture
#: that never looked up.
_MIN_CEILING_SUPPORT = 0.12

#: Width of the circular median filter run over the binned profile, in bins (degrees).
#:
#: Five degrees removes the single-column spikes a mask leaves at a picture frame or a lamp
#: without rounding off a corner: a corner two metres away subtends about 50 degrees of the
#: profile, ten times this.
_SMOOTH_BINS = 5

#: How far a simplified polygon may sit from the profile it came from, in camera heights.
#:
#: 0.22 camera heights is about 35 cm, which is roughly the shallowest wall feature a plan
#: has any business drawing — an alcove, a chimney breast, the return beside a doorway.
#: Below it the outline keeps the wobble of the mask and reads as a cave rather than a room:
#: on the reference bedroom, 14 cm produced twenty-nine vertices against seventeen here, and
#: the extra twelve are all mask noise. Swept against synthetic ground truth in
#: ``scripts/bench_floorplan.py``, where the floor IoU is flat from 14 cm to 48 cm (0.907 to
#: 0.927) — so this number costs accuracy nothing and is chosen for legibility.
_SIMPLIFY_EPSILON_H = 0.22


# ======================================================================================
# Boundaries
# ======================================================================================


def _first_run(mask: np.ndarray, run: int) -> np.ndarray:
    """Row index of the first run of ``run`` true values down each column, or -1.

    ``mask`` is ``(rows, cols)`` and is always scanned from row 0 downwards, so the caller
    orients it by choosing which slice of the canvas to hand over. See
    :func:`_column_boundaries` for why the two surfaces are scanned from opposite ends.
    """
    rows = mask.shape[0]
    if rows < run:
        return np.full(mask.shape[1], -1, np.int32)
    integral = np.concatenate([np.zeros((1, mask.shape[1]), np.int32), np.cumsum(mask, axis=0, dtype=np.int32)])
    # windows[i] counts the true values in rows [i, i+run).
    windows = integral[run:] - integral[:-run]
    full = windows >= run
    found = full.any(axis=0)
    index = full.argmax(axis=0).astype(np.int32)
    index[~found] = -1
    return index


def _bottom_of_first_run(mask: np.ndarray, run: int) -> np.ndarray:
    """Row index of the last true value of the first run down each column, or -1."""
    start = _first_run(mask, run)
    rows = mask.shape[0]
    ladder = np.arange(rows, dtype=np.int32)[:, None]
    ends = (ladder >= start[None, :]) & ~mask
    found = ends.any(axis=0)
    bottom = ends.argmax(axis=0).astype(np.int32) - 1
    bottom[~found] = rows - 1  # the run reaches the bottom edge of the window
    bottom[start < 0] = -1
    return bottom


def _column_boundaries(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(floor_elevation, ceiling_elevation)`` per column, in radians, NaN where absent.

    ``floor_elevation`` is negative (the junction is below the horizon) and
    ``ceiling_elevation`` positive. Both are measured at the pixel centre, which is a
    quarter of a degree on this canvas and an order below the mask's own accuracy.
    """
    height, width = surface.shape[:2]
    horizon = height // 2
    run = max(2, int(round(MIN_RUN_DEG * height / 180.0)))

    def elevation(rows: np.ndarray) -> np.ndarray:
        out = np.full(width, np.nan, np.float64)
        ok = rows >= 0
        out[ok] = 0.5 * math.pi - math.pi * (rows[ok] + 0.5) / float(height)
        return out

    # THE TWO SCANS GO OPPOSITE WAYS, and the asymmetry is the whole robustness of this
    # function rather than an oversight.
    #
    # The floor is scanned DOWNWARD FROM THE HORIZON and the first run wins: furniture
    # standing on the floor cuts the visible floor into a far strip at the wall's foot and a
    # near one at the viewer's feet, and it is the far strip — the topmost — that is the wall.
    #
    # The ceiling is scanned DOWNWARD FROM THE ZENITH and the first run's *bottom* wins,
    # because a ceiling is one region and it contains the top of the sphere. Taking instead
    # the lowest thing labelled ceiling is what a room punishes: measured on a real bedroom, a
    # dark window at night came back as ceiling, its island sat five degrees above the horizon,
    # and the wall it implied was twelve metres away in a four-metre room.
    below = surface[horizon:, :] == SURFACE_FLOOR
    floor_rows = _first_run(below, run)
    floor_rows = np.where(floor_rows >= 0, floor_rows + horizon, -1)

    above = (surface[:horizon, :] == SURFACE_CEILING).astype(np.uint8)
    # Bridge the gaps a light fitting or a beam leaves in the middle of a ceiling, so the
    # region below one stays part of the same run rather than starting a new, lower one.
    above = cv2.morphologyEx(above, cv2.MORPH_CLOSE, np.ones((2 * run + 1, 1), np.uint8)) > 0
    ceiling_rows = _bottom_of_first_run(above, run)

    return elevation(floor_rows), elevation(ceiling_rows)


def _ceiling_height(floor_elev: np.ndarray, ceiling_elev: np.ndarray) -> float | None:
    """Ceiling height above the floor in camera heights, or ``None``.

    In a column where both boundaries are on the same wall,
    ``tan(ceiling_elev) / tan(-floor_elev)`` is the ceiling's height above the eye, and it is
    the same in every such column. Where they are not on the same wall — a wardrobe hiding the
    floor, a tall cupboard hiding the ceiling line — the ratio is wrong in a *direction*, so
    the estimator has to be a mode rather than a mean: the true value is the one many columns
    repeat, not the average of the true ones and the wrong ones.
    """
    both = np.isfinite(floor_elev) & np.isfinite(ceiling_elev)
    if float(both.mean()) < _MIN_CEILING_SUPPORT:
        return None
    ratio = np.tan(ceiling_elev[both]) / np.maximum(np.tan(-floor_elev[both]), 1e-6)
    height = 1.0 + ratio
    lo, hi = CEILING_RANGE_H
    height = height[(height >= lo) & (height <= hi)]
    if height.size < max(8, int(_MIN_CEILING_SUPPORT * floor_elev.size * 0.25)):
        return None

    # Mode by the densest window: the peak of a coarse histogram, refined to the median of
    # everything inside it. A plain median would be dragged by whichever occlusion is more
    # common in this particular room.
    counts, edges = np.histogram(height, bins=24, range=CEILING_RANGE_H)
    peak = int(counts.argmax())
    inside = height[(height >= edges[peak]) & (height <= edges[peak + 1])]
    return float(np.median(inside)) if inside.size else None


def _radius_from_elevations(
    floor_elev: np.ndarray, ceiling_elev: np.ndarray, ceiling_height: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column radius in camera heights and which boundary produced it.

    ``source`` is 0 for none, 1 for floor, 2 for ceiling — an integer because this is the
    vectorised half; :func:`profile_from_surface` names them.
    """
    width = floor_elev.shape[0]
    radius = np.full(width, np.nan, np.float64)
    source = np.zeros(width, np.int8)

    has_floor = np.isfinite(floor_elev)
    r_floor = np.full(width, np.nan, np.float64)
    r_floor[has_floor] = 1.0 / np.maximum(np.tan(-floor_elev[has_floor]), 1e-6)
    radius[has_floor] = r_floor[has_floor]
    source[has_floor] = 1

    if ceiling_height is not None:
        has_ceiling = np.isfinite(ceiling_elev)
        r_ceiling = np.full(width, np.nan, np.float64)
        r_ceiling[has_ceiling] = (ceiling_height - 1.0) / np.maximum(
            np.tan(ceiling_elev[has_ceiling]), 1e-6
        )
        # Occlusion only ever shortens a radius, so the longer answer is the better one.
        #
        # There is deliberately NO CAP on how much further the ceiling may put a wall. The
        # first version had one — a ratio, to stop a mislabelled dark window asking for a wall
        # twelve metres away in a four-metre room — and it cost more than it bought: on three
        # synthetic rooms with known dimensions it held the mean radial error at 15 cm, and
        # removing it dropped that to 1.4 cm, because real occlusion in a furnished room is
        # routinely a factor of three or four. The mislabelled window is now excluded by
        # something structural instead: a ceiling is the region containing the zenith, and
        # that window was an island. See :func:`_column_boundaries`.
        better = has_ceiling & (~has_floor | (r_ceiling > radius))
        radius[better] = r_ceiling[better]
        source[better] = 2

    out_of_range = np.isfinite(radius) & ((radius < MIN_RADIUS_H) | (radius > MAX_RADIUS_H))
    radius[out_of_range] = np.nan
    source[out_of_range] = 0
    return radius, source


# ======================================================================================
# Binning, filling and smoothing
# ======================================================================================


def _bin_columns(values: np.ndarray, source: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-column answers to ``bins`` azimuth bins by median.

    Column ``u`` of a ``W``-wide equirect covers azimuth ``2*pi*(u+0.5)/W - pi``, so the bin
    is ``floor((u+0.5)/W * bins)`` and no wrap-around arithmetic is needed.
    """
    width = values.shape[0]
    index = np.floor((np.arange(width) + 0.5) / width * bins).astype(np.int32)
    index = np.clip(index, 0, bins - 1)

    radius = np.full(bins, np.nan, np.float64)
    kind = np.zeros(bins, np.int8)
    for b in range(bins):
        take = index == b
        vals = values[take]
        good = np.isfinite(vals)
        if not good.any():
            continue
        radius[b] = float(np.median(vals[good]))
        kinds = source[take][good]
        # Ties go to the floor: it is the boundary that needed no assumed ceiling height.
        kind[b] = 1 if int((kinds == 1).sum()) >= int((kinds == 2).sum()) else 2
    return radius, kind


def _fill_circular(values: np.ndarray) -> np.ndarray:
    """Interpolate the NaN bins from their neighbours, around the circle.

    A gap is a direction where neither boundary was visible — under a wardrobe that reaches
    the ceiling, or in the wedge a capture never covered. Drawing straight across it is the
    only thing a closed outline can do; :class:`RadialProfile` records that those bins were
    never measured so nothing downstream counts them as evidence.
    """
    known = np.isfinite(values)
    if not known.any():
        return values
    if known.all():
        return values
    n = values.shape[0]
    idx = np.flatnonzero(known)
    # Unwrap once around the circle so interpolation across bin 0 works without a special case.
    extended_idx = np.concatenate([idx - n, idx, idx + n])
    extended_val = np.concatenate([values[idx], values[idx], values[idx]])
    return np.interp(np.arange(n), extended_idx, extended_val)


def _median_circular(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values
    half = width // 2
    padded = np.concatenate([values[-half:], values, values[:half]])
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(windows, axis=1)


def profile_from_surface(
    surface: np.ndarray, *, bins: int = BINS, use_ceiling: bool = True
) -> RadialProfile:
    """Turn a ceiling/wall/floor map into a distance in every direction.

    Pure array work with no model in it, which is the point: the perception half of this
    feature is one function call away in :func:`outline_from_pano`, and everything after it
    can be checked against a mask drawn by hand or ray-traced from a known room.
    """
    surface = np.asarray(surface)
    if surface.ndim != 2:
        raise ValueError("surface must be a single-channel label map")

    floor_elev, ceiling_elev = _column_boundaries(surface)
    ceiling_height = _ceiling_height(floor_elev, ceiling_elev) if use_ceiling else None
    radius, source = _radius_from_elevations(floor_elev, ceiling_elev, ceiling_height)

    binned, kind = _bin_columns(radius, source, bins)
    filled = _fill_circular(binned)
    if not np.isfinite(filled).any():
        # Nothing at all was seen. A profile still has to exist for the caller to inspect, so
        # it is the smallest legal room with every bin marked unmeasured.
        filled = np.full(bins, MIN_RADIUS_H, np.float64)
        kind = np.zeros(bins, np.int8)

    smoothed = _median_circular(np.nan_to_num(filled, nan=MIN_RADIUS_H), _SMOOTH_BINS)
    smoothed = np.clip(smoothed, MIN_RADIUS_H, MAX_RADIUS_H)

    names: dict[int, BinSource] = {0: "none", 1: "floor", 2: "ceiling"}
    return RadialProfile(
        radius=tuple(float(v) for v in smoothed),
        source=tuple(names[int(k)] for k in kind),
        ceiling_height=ceiling_height,
    )


# ======================================================================================
# Profile -> a room
# ======================================================================================
#
# THE OUTLINE IS A ROOM FITTED TO THE PROFILE, NOT THE PROFILE ITSELF, and that is the most
# consequential decision in this module.
#
# The first version drew the measurement: the radial profile, simplified. It is faithful and
# it is unusable. A profile is the distance to the nearest thing in every direction, so a
# real bedroom comes out as a blob with a dozen small bites taken out of it (a wardrobe the
# ceiling line could not see past, a curtain, a stack of boxes) and two long wedges sticking
# out of it (the floor of the next room, seen through an open door). Nobody looks at that and
# sees a room, because a room is not that shape: flats are built out of rectangles.
#
# So the drawing answers a different question from the measurement. The measurement is "how
# far is the nearest surface in this direction", kept in :class:`RadialProfile` and never
# thrown away. The drawing is "which rectilinear room best explains those distances", and the
# distinction matters because the two fail differently: an occlusion pulls one direction of
# the measurement inward and leaves the fitted room where it is, which is exactly the
# behaviour a plan needs.
#
# The fit is deliberately shallow — a rectangle, with at most one corner cut out of it — and
# that ceiling is a claim about rooms rather than a limit of the arithmetic. A model free
# enough to draw any shape would draw the blob again.
#
# ANGLES IN THIS SECTION ARE MEASURED FROM +X, not as bearings, because that is the frame a
# wall direction lives in. :func:`_turn` converts: turning by ``a`` DECREASES an angle from
# +x by ``a``, so a room whose walls run at ``a`` becomes axis-aligned under ``_turn(p, a)``.


#: How much any one direction may pull the fit, in metres.
#:
#: The loss is ``min(|error|, tau)``, so a direction that disagrees by 40 cm and one that
#: disagrees by four metres cost the same. That is the whole robustness of the fit, and it is
#: what makes an open doorway harmless: the wedge of far-room floor behind it is a handful of
#: directions wrong by metres, and capping their cost stops them dragging the wall out to
#: meet them. 40 cm is above the segmenter's own scatter along a wall (measured at 3.5-14 cm)
#: and well below anything a wall could be mistaken for.
_LOSS_CAP_M = 0.4

#: How close to a wall the camera is allowed to have stood. Below this the fit is describing
#: a cupboard rather than a room, and the search should not go there.
_MIN_HALF_EXTENT_M = 0.35

#: Search radius and step for each wall, in metres. Coordinate descent over this grid rather
#: than a continuous optimiser: the loss is capped and therefore not smooth, there are only
#: four parameters, and 2 cm is finer than anything downstream can use.
_SEARCH_SPAN_M = 2.0
_SEARCH_STEP_M = 0.02
_SEARCH_SWEEPS = 3

#: A corner cut has to earn its two extra vertices twice over: it must remove at least this
#: much floor, and improve the fit by at least this share of the rectangle's own loss.
#:
#: An L-shaped living room is a real thing and this is how it gets drawn. Around 1.5 square
#: metres is the smallest bite that reads as architecture rather than as a wardrobe; below it
#: the honest drawing is a rectangle with a wardrobe in the corner, which is what the plan of
#: an emptied room would look like anyway.
_MIN_CUT_AREA_M2 = 1.5
_MIN_CUT_GAIN = 0.12
_CUT_STEP_M = 0.05

#: Tangent half-window for the orientation estimate, in profile bins.
_TANGENT_WINDOW = 4


def _orientation(points: np.ndarray) -> float:
    """The direction the room's walls run in, from +x, modulo 90 degrees.

    Every panorama's yaw origin is wherever the photographer started turning, so a room's
    walls arrive at an arbitrary angle and the fit has to find it before it can talk about
    rectangles. Each boundary segment votes with its length at FOUR TIMES its angle, which
    maps directions a quarter turn apart onto the same point — so parallel and perpendicular
    walls reinforce one another instead of cancelling, which is the whole point when the
    thing being looked for is a grid rather than a direction.

    The tangent is taken across a window rather than between neighbours: adjacent samples of
    a profile are centimetres apart and the direction between two of them is mostly noise.
    """
    n = len(points)
    if n < 4 * _TANGENT_WINDOW:
        return 0.0
    tangent = np.roll(points, -_TANGENT_WINDOW, axis=0) - np.roll(points, _TANGENT_WINDOW, axis=0)
    length = np.hypot(tangent[:, 0], tangent[:, 1])
    good = length > 1e-6
    if not good.any():
        return 0.0
    angle = 4.0 * np.arctan2(tangent[good, 1], tangent[good, 0])
    total = np.sum(length[good] * np.exp(1j * angle))
    if abs(total) < 1e-9:
        return 0.0
    return float(np.angle(total) / 4.0)


def _turn(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``(x, z)`` points about the camera, DECREASING their angle from +x by ``angle``."""
    ca, sa = math.cos(angle), math.sin(angle)
    return np.stack(
        [points[:, 0] * ca + points[:, 1] * sa, points[:, 1] * ca - points[:, 0] * sa], axis=1
    )


def _ray_hits(polygon: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Distance from the camera to ``polygon`` along each direction, or ``inf``.

    The model's own profile, so a candidate room can be compared with the measurement in the
    units the measurement is in. Generic over the polygon rather than solved per shape: the
    rectangle and the cut-cornered version then share one scorer, and a third shape later
    would not need a third formula.
    """
    best = np.full(directions.shape[0], np.inf)
    count = len(polygon)
    for index in range(count):
        p = polygon[index]
        edge = polygon[(index + 1) % count] - p
        denominator = directions[:, 0] * edge[1] - directions[:, 1] * edge[0]
        usable = np.abs(denominator) > 1e-12
        safe = np.where(usable, denominator, 1.0)
        t = (p[0] * edge[1] - p[1] * edge[0]) / safe
        s = (p[0] * directions[:, 1] - p[1] * directions[:, 0]) / safe
        hit = usable & (t > 1e-9) & (s >= -1e-9) & (s <= 1.0 + 1e-9)
        best = np.where(hit & (t < best), t, best)
    return best


def _loss(polygon: np.ndarray, directions: np.ndarray, measured: np.ndarray) -> float:
    """Capped mean disagreement between a candidate room and the measured profile."""
    model = _ray_hits(polygon, directions)
    error = np.where(np.isfinite(model), np.abs(model - measured), _LOSS_CAP_M)
    return float(np.minimum(error, _LOSS_CAP_M).mean())


def _rectangle(bounds: np.ndarray) -> np.ndarray:
    x_lo, x_hi, z_lo, z_hi = (float(v) for v in bounds)
    return np.array([[x_lo, z_lo], [x_hi, z_lo], [x_hi, z_hi], [x_lo, z_hi]], np.float64)


def _fit_rectangle(directions: np.ndarray, measured: np.ndarray, start: np.ndarray) -> np.ndarray:
    """The four walls that best explain the measured distances.

    Coordinate descent, one wall at a time, over a grid. Each wall is independent of the
    other three for every direction that reaches it, which is why sweeping them one at a time
    settles in two or three passes rather than needing a joint optimiser.
    """
    bounds = np.array(start, np.float64)
    offsets = np.arange(-_SEARCH_SPAN_M, _SEARCH_SPAN_M + 1e-9, _SEARCH_STEP_M)
    limits = ((-math.inf, -_MIN_HALF_EXTENT_M), (_MIN_HALF_EXTENT_M, math.inf))

    for _ in range(_SEARCH_SWEEPS):
        for slot in range(4):
            low, high = limits[slot % 2]
            candidates = np.unique(np.clip(bounds[slot] + offsets, low, high))
            best_value, best_loss = float(bounds[slot]), math.inf
            for candidate in candidates:
                trial = bounds.copy()
                trial[slot] = candidate
                value = _loss(_rectangle(trial), directions, measured)
                if value < best_loss:
                    best_loss, best_value = value, float(candidate)
            bounds[slot] = best_value
    return bounds


def _with_cut(bounds: np.ndarray, corner: int, cut_x: float, cut_z: float) -> np.ndarray:
    """The rectangle with one corner squared off — an L-shaped room.

    ``corner`` indexes the rectangle's own vertex order, so the cut replaces one vertex with
    three and the ring stays wound the way it was.
    """
    x_lo, x_hi, z_lo, z_hi = (float(v) for v in bounds)
    ring = [[x_lo, z_lo], [x_hi, z_lo], [x_hi, z_hi], [x_lo, z_hi]]
    replacement = (
        [[x_lo, cut_z], [cut_x, cut_z], [cut_x, z_lo]],
        [[cut_x, z_lo], [cut_x, cut_z], [x_hi, cut_z]],
        [[x_hi, cut_z], [cut_x, cut_z], [cut_x, z_hi]],
        [[cut_x, z_hi], [cut_x, cut_z], [x_lo, cut_z]],
    )[corner]
    return np.array(ring[:corner] + replacement + ring[corner + 1 :], np.float64)


def _fit_corner_cut(
    bounds: np.ndarray, directions: np.ndarray, measured: np.ndarray, base_loss: float
) -> np.ndarray | None:
    """Take a bite out of one corner, if the room really has one taken out of it.

    Tried at every corner and kept only when it pays twice: a big enough piece of floor, and
    a big enough improvement in the fit. Without both tests the cut becomes a way of drawing
    a wardrobe, which is the failure the whole fit exists to avoid.
    """
    corners = _rectangle(bounds)
    area = float(abs(bounds[1] - bounds[0]) * abs(bounds[3] - bounds[2]))
    best: tuple[float, np.ndarray] | None = None

    for corner in range(4):
        towards_x, towards_z = float(corners[corner][0]), float(corners[corner][1])
        span_x = np.arange(0.25, abs(towards_x) - 0.2, _CUT_STEP_M) * math.copysign(1.0, towards_x)
        span_z = np.arange(0.25, abs(towards_z) - 0.2, _CUT_STEP_M) * math.copysign(1.0, towards_z)
        if span_x.size == 0 or span_z.size == 0:
            continue

        cut_x, cut_z = float(span_x[-1]), float(span_z[-1])
        score = _loss(_with_cut(bounds, corner, cut_x, cut_z), directions, measured)
        for _ in range(2):
            for value in span_x:
                trial = _loss(_with_cut(bounds, corner, float(value), cut_z), directions, measured)
                if trial < score:
                    score, cut_x = trial, float(value)
            for value in span_z:
                trial = _loss(_with_cut(bounds, corner, cut_x, float(value)), directions, measured)
                if trial < score:
                    score, cut_z = trial, float(value)

        removed = abs(towards_x - cut_x) * abs(towards_z - cut_z)
        if removed < _MIN_CUT_AREA_M2 or removed > 0.6 * area:
            continue
        if score > base_loss * (1.0 - _MIN_CUT_GAIN):
            continue
        if best is None or score < best[0]:
            best = (score, _with_cut(bounds, corner, cut_x, cut_z))

    return None if best is None else best[1]


def _simplified(points: np.ndarray, epsilon: float) -> tuple[tuple[float, float], ...]:
    """The measurement itself, simplified. Diagnostic only — see the section header."""
    approx = cv2.approxPolyDP(points.astype(np.float32).reshape(-1, 1, 2), epsilon, True)
    ring = approx.reshape(-1, 2).astype(np.float64)
    if len(ring) < 3:
        ring = points
    return tuple((float(x), float(z)) for x, z in ring)


def _area(points: np.ndarray) -> float:
    x, z = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))) * 0.5)


def polygon_from_profile(
    profile: RadialProfile,
    camera_height_m: float = CAMERA_HEIGHT_M,
    *,
    rectify: bool = True,
) -> tuple[tuple[float, float], ...]:
    """Fit a room to the profile and return it in metres, camera at the origin.

    With ``rectify=False`` the measurement is returned instead, simplified but not fitted.
    That is for looking at — ``scripts/bench_floorplan.py`` draws both — and never for the
    plan itself.
    """
    points = np.array(profile.points(camera_height_m), np.float64)
    if not rectify:
        return _simplified(points, _SIMPLIFY_EPSILON_H * camera_height_m)

    angle = _orientation(points)
    turned = _turn(points, angle)
    measured = np.hypot(points[:, 0], points[:, 1])
    # Bearings grow by `angle` under the same turn, so the model is asked about the same
    # physical directions the profile measured.
    bearings = np.array(profile.angles(), np.float64) + angle
    directions = np.stack([np.sin(bearings), np.cos(bearings)], axis=1)

    start = np.array(
        [
            min(float(np.percentile(turned[:, 0], 8.0)), -_MIN_HALF_EXTENT_M),
            max(float(np.percentile(turned[:, 0], 92.0)), _MIN_HALF_EXTENT_M),
            min(float(np.percentile(turned[:, 1], 8.0)), -_MIN_HALF_EXTENT_M),
            max(float(np.percentile(turned[:, 1], 92.0)), _MIN_HALF_EXTENT_M),
        ],
        np.float64,
    )
    bounds = _fit_rectangle(directions, measured, start)
    polygon = _rectangle(bounds)
    base = _loss(polygon, directions, measured)

    cut = _fit_corner_cut(bounds, directions, measured, base)
    if cut is not None:
        polygon = cut

    log.info(
        "plan.room_fitted",
        orientation_deg=round(math.degrees(angle), 1),
        width_m=round(float(bounds[1] - bounds[0]), 2),
        depth_m=round(float(bounds[3] - bounds[2]), 2),
        residual_cm=round(base * 100.0, 1),
        corner_cut=cut is not None,
    )
    return tuple((float(x), float(z)) for x, z in _turn(polygon, -angle))

def outline_from_surface(
    surface: np.ndarray,
    *,
    camera_height_m: float = CAMERA_HEIGHT_M,
    bins: int = BINS,
    rectify: bool = True,
    use_ceiling: bool = True,
) -> RoomOutline:
    """A surface map in, a room outline out. The half of this module with no model in it."""
    profile = profile_from_surface(surface, bins=bins, use_ceiling=use_ceiling)
    return RoomOutline(
        polygon=polygon_from_profile(profile, camera_height_m, rectify=rectify),
        profile=profile,
        camera_height_m=camera_height_m,
        source="model",
    )


def outline_from_pano(
    pano: np.ndarray,
    *,
    camera_height_m: float = CAMERA_HEIGHT_M,
    rectify: bool = True,
    use_ceiling: bool = True,
) -> RoomOutline | None:
    """Measure a room from its equirectangular panorama, or ``None`` with no segmenter.

    The segmentation is :func:`biosphere.ai.furniture_seg.segment_equirect`'s, unchanged and
    deliberately shared: it is the same pass the restyling and emptying tools run, so the
    plan's walls are the walls those tools protect. It is also the whole cost of this
    function — sixteen perspective windows through a scene-parsing model — which is why the
    result is cached beside the panorama rather than computed per request.

    ``None`` is not a failure to report as one. It means this deployment has no segmenter, in
    which case the room has no measured shape and the plan draws it as a placeholder.
    """
    from biosphere.ai import furniture_seg  # noqa: PLC0415 - optional heavy import

    answered = furniture_seg.segment_equirect(pano)
    if answered is None:
        return None
    _, surface = answered
    outline = outline_from_surface(
        surface, camera_height_m=camera_height_m, rectify=rectify, use_ceiling=use_ceiling
    )
    log.info(
        "plan.outline",
        measured=round(outline.profile.measured_fraction, 3),
        from_ceiling=round(outline.profile.ceiling_fraction, 3),
        ceiling_height=(
            None if outline.profile.ceiling_height is None else round(outline.profile.ceiling_height, 3)
        ),
        area_m2=round(outline.area_m2, 2),
        vertices=len(outline.polygon),
    )
    return outline


def outline_payload(outline: RoomOutline, *, source_mtime: int) -> dict[str, Any]:
    """The cached form: the outline plus what it was computed from."""
    payload = outline.as_dict()
    payload["source_mtime"] = int(source_mtime)
    return payload
