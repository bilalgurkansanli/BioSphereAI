"""The room behind the furniture, rebuilt on its own planes.

Why the fill needs geometry and not a better network
-----------------------------------------------------
Every learned fill this package has tried fails the same way on a furnished room, and the
failure has one cause. Measured on the reference bedroom (``scripts/bench_empty_room.py``):
LaMa through perspective windows turns a removed wardrobe, a bed and a heap of bags into grey
and black smears; Stable Diffusion draws a new bed into the bed-shaped hole; RORem draws a
window into a solid wall. A hole the size of a wardrobe is mostly a question about *what is
behind it*, and an image model can only answer that from the pixels around the hole — which,
in a furnished room, are the remains of more furniture, a contact shadow, and three different
surfaces meeting at angles the window happens to show.

The room itself answers it far better. Behind a wardrobe there is a wall; under a bed there is
a floor; and both are PLANES whose position is already measured — :mod:`biosphere.plan.outline`
fits the room's walls from the same segmentation this feature runs. On a plane, seen square-on,
the empty room is simple in exactly the ways an image model is bad at:

* a wall, rectified fronto-parallel, is paint whose colour changes smoothly, with the skirting
  and the cornice as perfectly HORIZONTAL bands — so a hole in it is filled row by row from the
  same row either side, and a moulding cannot bend, stop or double;
* a floor, rectified top-down, is a tiling — square tiles become a perfectly regular lattice,
  boards become parallel strips — so a hole in it is filled by copying the floor's own pattern
  one or more periods away, in phase, with its grout lines continuing through;
* and the line where the two meet is not drawn by anything: it is where the ray stops hitting
  the floor plane and starts hitting the wall plane, so it is straight by construction.

Nothing here has a vocabulary for objects, so nothing here can invent one. That is the
property ``sd_inpaint`` measured the diffusion path does not have, and it is the one a picture
put in front of a buyer needs most.

What it cannot know, stated plainly
-----------------------------------
What was never photographed — a socket behind the wardrobe, a scuff under the bed, the hem of
a curtain hidden by a chair — comes back as the surface's own continuation. The fill is
deliberately plain. It also inherits the layout's limits: a room that is neither a rectangle
nor an L is fitted as the nearest of the two, and a wall the fit put a few centimetres off
shows as a small step where the filled junction meets the photographed one.

Units are CAMERA HEIGHTS, exactly as in :mod:`biosphere.plan.types`: the eye is at the origin,
the floor is the plane ``y = +1`` (+Y is down, as everywhere in the engine), and the ceiling is
``y = -(ceiling - 1)``. A wall is a polygon edge in ``(x, z)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from biosphere.core.logging import get_logger

__all__ = [
    "FLOOR",
    "CEILING",
    "RoomBox",
    "PlaneRaster",
    "Camera",
    "EquirectCamera",
    "PinholeCamera",
    "trace",
    "plane_rasters",
    "rectify",
    "fill_planes",
    "PlaneFill",
    "DEFAULT_CEILING_H",
]

log = get_logger("biosphere.ai.room_planes")

#: Plane ids. Walls are numbered from 0 in polygon order; the two horizontal planes are negative
#: so a wall index can never collide with them.
FLOOR, CEILING = -1, -2

#: Ceiling height above the floor, in camera heights, when the layout could not measure one.
#: 2.6 m over the player's 1.6 m eye height: the commonest ceiling in a Turkish flat, and the
#: plan module's own default range brackets it.
DEFAULT_CEILING_H = 2.6 / 1.6

#: Rectified raster resolution, in camera heights per pixel.
#:
#: 0.0025 h is 4 mm at the assumed eye height. A 4096-wide equirect resolves 3 mm on a wall two
#: metres away and 5 mm radially on the floor at the same distance, so this is the density the
#: photograph actually carries in the middle of a room; finer would only interpolate, coarser
#: would blur the fill below the picture around it.
_PIXEL_H = 0.0025

#: How far a raster reaches past the box it was built from, in camera heights. The box is
#: a fit and the floor line is moved onto the photograph's own (:func:`floor_line_shift`); 0.25 h
#: is 40 cm, more than any correction that survives that function's own limits.
_SLACK_H = 0.25

#: No raster side longer than this. A seven-metre wall at 4 mm is 1750 px, so this only bites on
#: a hall-sized room, where the resolution is lowered rather than the memory raised.
_MAX_RASTER_SIDE = 2400

_EPS = 1e-9


# ======================================================================================
# The room
# ======================================================================================


@dataclass(frozen=True, slots=True)
class RoomBox:
    """A room as a floor polygon and a ceiling height, both in camera heights."""

    #: ``(n, 2)`` float64, ``(x, z)`` per corner, in order around the room.
    polygon: np.ndarray
    #: Ceiling height ABOVE THE FLOOR, in camera heights (so always > 1).
    ceiling: float

    @property
    def eye_to_ceiling(self) -> float:
        return float(self.ceiling - 1.0)

    @property
    def walls(self) -> int:
        return int(self.polygon.shape[0])

    def wall(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """``(start, end)`` corners of one wall, in ``(x, z)``."""
        start = self.polygon[index]
        end = self.polygon[(index + 1) % self.walls]
        return start, end

    @classmethod
    def from_outline(cls, outline: object, *, default_ceiling: float = DEFAULT_CEILING_H) -> RoomBox:
        """From a :class:`biosphere.plan.types.RoomOutline` (metres) to camera heights."""
        metres = float(getattr(outline, "camera_height_m"))
        polygon = np.asarray(getattr(outline, "polygon"), np.float64) / max(metres, _EPS)
        profile = getattr(outline, "profile", None)
        measured = getattr(profile, "ceiling_height", None) if profile is not None else None
        ceiling = float(measured) if measured else float(default_ceiling)
        return cls(polygon=_counter_clockwise(polygon), ceiling=max(ceiling, 1.05))

    def convex_around_eye(self) -> bool:
        """True when the outline is convex and the camera stands inside it."""
        poly = self.polygon
        edges = np.roll(poly, -1, axis=0) - poly
        to_eye = -poly
        turns = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        sides = edges[:, 0] * to_eye[:, 1] - edges[:, 1] * to_eye[:, 0]
        return bool(np.all(turns >= -1e-9) and np.all(sides > 0.0))

    def orientation(self) -> float:
        """Direction of the longest wall, radians from +x in the ``(x, z)`` plane.

        The floor raster is laid along it, so a tiled floor whose grout runs parallel to the
        walls comes out as an axis-aligned lattice rather than a rotated one.
        """
        best, angle = -1.0, 0.0
        for index in range(self.walls):
            start, end = self.wall(index)
            delta = end - start
            length = float(np.hypot(delta[0], delta[1]))
            if length > best:
                best, angle = length, math.atan2(float(delta[1]), float(delta[0]))
        return angle


def _counter_clockwise(polygon: np.ndarray) -> np.ndarray:
    x, z = polygon[:, 0], polygon[:, 1]
    signed = float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))
    return polygon if signed >= 0.0 else polygon[::-1].copy()


@dataclass(slots=True)
class Hits:
    """Where a set of rays meets the box."""

    #: ``FLOOR``, ``CEILING`` or a wall index, per ray.
    plane: np.ndarray
    #: The 3-D hit point, ``(..., 3)``, camera heights.
    point: np.ndarray
    #: Distance along each wall from its start corner (camera heights); NaN off the walls.
    along: np.ndarray
    #: The wall each ray's DIRECTION reaches, whether or not the floor or ceiling came first,
    #: and where — so a pixel the floor line is moved across can be put on its wall.
    wall: np.ndarray
    wall_point: np.ndarray


def trace(box: RoomBox, rays: np.ndarray) -> Hits:
    """First surface of the EMPTY room each ray meets. ``rays`` is ``(..., 3)``, any length.

    Exact for any simple polygon, convex or not: every wall is intersected and the nearest
    positive hit wins, so an L-shaped room's re-entrant corner hides what it should.
    """
    shape = rays.shape[:-1]
    d = rays.reshape(-1, 3).astype(np.float64)
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dy > _EPS, 1.0 / dy, np.inf)
        t_ceiling = np.where(dy < -_EPS, box.eye_to_ceiling / -dy, np.inf)

    t_wall = np.full(d.shape[0], np.inf)
    wall_id = np.full(d.shape[0], -1, np.int32)
    along = np.full(d.shape[0], np.nan)
    for index in range(box.walls):
        start, end = box.wall(index)
        ex, ez = float(end[0] - start[0]), float(end[1] - start[1])
        px, pz = float(start[0]), float(start[1])
        # t * (dx, dz) - s * (ex, ez) = (px, pz)
        det = -dx * ez + ex * dz
        safe = np.where(np.abs(det) > _EPS, det, 1.0)
        t = (-px * ez + ex * pz) / safe
        s = (dx * pz - dz * px) / safe
        ok = (np.abs(det) > _EPS) & (t > _EPS) & (s >= -1e-6) & (s <= 1.0 + 1e-6)
        nearer = ok & (t < t_wall)
        t_wall = np.where(nearer, t, t_wall)
        wall_id = np.where(nearer, index, wall_id)
        along = np.where(nearer, np.clip(s, 0.0, 1.0) * math.hypot(ex, ez), along)

    t_hit = np.minimum(np.minimum(t_floor, t_ceiling), t_wall)
    plane = np.where(
        t_hit == t_floor, FLOOR, np.where(t_hit == t_ceiling, CEILING, wall_id)
    ).astype(np.int32)
    point = d * np.where(np.isfinite(t_hit), t_hit, 0.0)[:, None]
    wall_point = d * np.where(np.isfinite(t_wall), t_wall, 0.0)[:, None]
    return Hits(
        plane=plane.reshape(shape),
        point=point.reshape(*shape, 3),
        along=np.where(plane >= 0, along, np.nan).reshape(shape),
        wall=wall_id.reshape(shape),
        wall_point=wall_point.reshape(*shape, 3),
    )


# ======================================================================================
# Fitting the box to the photograph
# ======================================================================================

#: ADE20K ids, by name, for the box fit. Wall paint against floor and against ceiling are the
#: only junctions trusted: a curtain hangs a hand's breadth in front of its wall and a door leaf
#: may stand open into the room, so neither foot is where the wall is.
_ADE_WALL, _ADE_FLOOR, _ADE_CEILING, _ADE_DOOR, _ADE_CURTAIN = 0, 3, 5, 14, 18

#: Fewest observations that may move a wall, or fix the ceiling height.
_MIN_JUNCTION_POINTS = 12

#: Orientation search around the outline's own estimate, degrees, and its step.
_ORIENTATION_SPAN_DEG, _ORIENTATION_STEP_DEG = 12.0, 0.25


def _junction_rows(classes: np.ndarray, upper: int, lower: int, *, run: int, from_row: int) -> np.ndarray:
    """Per column, the first row (scanning down from ``from_row``) where ``run`` pixels of
    ``upper`` sit directly on ``run`` pixels of ``lower``. -1 where there is none."""
    height, width = classes.shape
    out = np.full(width, -1, np.int64)
    found = np.zeros(width, bool)
    for row in range(max(from_row, run), height - run):
        if found.all():
            break
        hit = (classes[row - run : row] == upper).all(axis=0) & (classes[row : row + run] == lower).all(axis=0)
        hit &= ~found
        out[hit] = row
        found |= hit
    return out


def _refine_rows(grey: np.ndarray, columns: np.ndarray, rows: np.ndarray, reach: int) -> np.ndarray:
    """Move each coarse junction row to the strongest vertical edge within ``reach`` pixels."""
    gy = np.abs(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
    out = rows.astype(np.float64).copy()
    height = grey.shape[0]
    for index, (col, row) in enumerate(zip(columns.tolist(), rows.tolist(), strict=True)):
        lo, hi = max(1, row - reach), min(height - 2, row + reach)
        if hi <= lo:
            continue
        profile = gy[lo : hi + 1, col]
        best = int(np.argmax(profile))
        frac = 0.0
        if 0 < best < profile.size - 1:
            a, b, c = float(profile[best - 1]), float(profile[best]), float(profile[best + 1])
            denom = a - 2.0 * b + c
            frac = 0.0 if abs(denom) < 1e-9 else float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
        out[index] = lo + best + frac
    return out


@dataclass(slots=True)
class Junctions:
    """Where the room's own junctions are visible, as directions on the sphere."""

    #: Azimuth (radians, the engine's yaw) and depression below the horizon of wall-meets-floor.
    floor_theta: np.ndarray
    floor_depression: np.ndarray
    #: Azimuth and elevation above the horizon of wall-meets-ceiling.
    ceiling_theta: np.ndarray
    ceiling_elevation: np.ndarray
    #: Curtain-meets-floor: a wall seen through what hangs in front of it, a hand's breadth near.
    #: Used only for a wall the ceiling line never reached — curtains hung to the ceiling hide it.
    hem_theta: np.ndarray | None = None
    hem_depression: np.ndarray | None = None


def observe_junctions(image: np.ndarray, classes: np.ndarray, hole: np.ndarray) -> Junctions:
    """Read the visible floor and ceiling junctions off the segmentation, sharpened on the image.

    The segmentation says WHERE a junction is to within its own pixel — a third of a degree on
    the engine's 1024-wide canvas, which is seven centimetres of wall at two metres — and the
    strongest edge within a degree of it says exactly where. Columns with a door in them, or
    with anything being removed beside the junction, are not read: a doorway shows the next
    room's junction and a removed object's foot is not the wall's.
    """
    height, width = image.shape[:2]
    c_height, c_width = classes.shape
    grey = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2)
    # Any hole pixel inside a class cell taints it: AREA then "> 0" is a max-pool.
    taint = cv2.resize((hole > 127).astype(np.float32), (c_width, c_height), interpolation=cv2.INTER_AREA) > 0
    taint = cv2.dilate(taint.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    run = max(2, int(round(c_height / 256)))
    door = (classes == _ADE_DOOR).any(axis=0)
    door = cv2.dilate(door.astype(np.uint8)[None, :], np.ones((1, 2 * run + 1), np.uint8))[0] > 0
    sx, sy = width / float(c_width), height / float(c_height)

    def read(upper: int, lower: int, from_row: int, sharpen: bool) -> tuple[np.ndarray, np.ndarray]:
        rows_c = _junction_rows(classes, upper, lower, run=run, from_row=from_row)
        cols_c = np.nonzero((rows_c >= 0) & ~door)[0]
        rows_c = rows_c[cols_c]
        keep = ~taint[np.clip(rows_c, 0, c_height - 1), cols_c]
        cols_c, rows_c = cols_c[keep], rows_c[keep]
        cols_f = np.clip(np.round((cols_c + 0.5) * sx - 0.5).astype(np.int64), 0, width - 1)
        rows_f = rows_c.astype(np.float64) * sy
        if sharpen and cols_f.size:
            rows_f = _refine_rows(grey, cols_f, np.round(rows_f).astype(np.int64), reach=max(2, int(round(1.2 * sy))))
        theta = 2.0 * np.pi * (cols_f + 0.5) / width - np.pi
        angle = np.pi * (rows_f + 0.5) / height - 0.5 * np.pi  # positive below the horizon
        return theta, angle

    floor_theta, floor_dep = read(_ADE_WALL, _ADE_FLOOR, c_height // 2 + 1, sharpen=True)
    ok = floor_dep > np.radians(3.0)
    # The ceiling line is NOT sharpened: a cornice is three parallel edges, and "the strongest"
    # hops between them from one column to the next. The median over a wall does better.
    ceiling_theta, ceiling_angle = read(_ADE_CEILING, _ADE_WALL, run + 1, sharpen=False)
    up = -ceiling_angle > np.radians(3.0)
    hem_theta, hem_dep = read(_ADE_CURTAIN, _ADE_FLOOR, c_height // 2 + 1, sharpen=True)
    hem_ok = hem_dep > np.radians(3.0)
    return Junctions(
        floor_theta[ok],
        floor_dep[ok],
        ceiling_theta[up],
        -ceiling_angle[up],
        hem_theta[hem_ok],
        hem_dep[hem_ok],
    )


def _rectangle_hits(bounds: np.ndarray, theta: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Distance to an axis-aligned rectangle (in a frame turned by ``angle``) along each azimuth,
    and which side was hit (0 x-low, 1 x-high, 2 z-low, 3 z-high)."""
    x0, x1, z0, z1 = bounds
    dx = np.sin(theta - angle)
    dz = np.cos(theta - angle)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 1e-9, x1 / dx, np.where(dx < -1e-9, x0 / dx, np.inf))
        tz = np.where(dz > 1e-9, z1 / dz, np.where(dz < -1e-9, z0 / dz, np.inf))
    side = np.where(tx < tz, np.where(dx > 0, 1, 0), np.where(dz > 0, 3, 2))
    return np.minimum(tx, tz), side


def _side_mode(coord: np.ndarray, scale: float, *, prefer_far: bool = False) -> float | None:
    """Where most of one wall's observations agree, rather than their average.

    THE MODE, BECAUSE THE ERRORS ARE ONE-SIDED AND COMMON. A ceiling line read off the top of a
    wardrobe, or a floor line read off the foot of a curtain, is nearer than the wall; a floor
    line seen through a doorway is further. In a furnished room either kind can be a third of a
    wall's observations, which drags a median and does not move a mode: the wall is the one
    distance many columns repeat.
    """
    if coord.size < _MIN_JUNCTION_POINTS:
        return None
    width = max(0.01, 0.015 * scale)
    lo, hi = float(coord.min()), float(coord.max())
    bins = max(1, int(math.ceil((hi - lo) / width)))
    counts, edges = np.histogram(coord, bins=bins, range=(lo, lo + bins * width))
    # A three-bin window, so a wall whose line falls on a bin edge is not split in two.
    smoothed = np.convolve(np.pad(counts, 1), np.ones(3), mode="valid")
    peak = int(np.argmax(smoothed))
    if prefer_far:
        # Ceiling-line observations can only err NEAR (the top of a wardrobe is in front of the
        # wall, never behind it), so of the clusters with real support the farthest is the wall.
        strong = np.nonzero(smoothed >= _FAR_CLUSTER_SHARE * smoothed[peak])[0]
        centres = 0.5 * (edges[strong] + edges[strong + 1])
        peak = int(strong[int(np.argmax(np.abs(centres)))])
    centre = 0.5 * (edges[peak] + edges[peak + 1])
    near = coord[np.abs(coord - centre) <= 1.5 * width]
    if near.size < _MIN_JUNCTION_POINTS // 2:
        return None
    return float(np.median(near))


def _fit_sides(u: np.ndarray, v: np.ndarray, start: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Each wall of an axis-aligned rectangle at the mode of the points it owns."""
    bounds = start.copy()
    local = np.arctan2(u, v)
    found = np.zeros(4, bool)
    for _ in range(5):
        _, side = _rectangle_hits(bounds, local, 0.0)
        for index, coord in ((0, u), (1, u), (2, v), (3, v)):
            mode = _side_mode(coord[side == index], scale, prefer_far=True)
            if mode is not None and (mode < 0) == (index % 2 == 0):
                bounds[index] = mode
                found[index] = True
    return bounds, found


def fit_box(junctions: Junctions, prior: RoomBox | None = None) -> tuple[RoomBox | None, dict[str, float]]:
    """A rectangular room fitted to the visible junctions, in camera heights.

    WHY THE PLAN'S OUTLINE IS NOT USED AS IT STANDS. :mod:`biosphere.plan.outline` fits a room to
    the distance profile its own segmentation reads, which is the right tool for drawing a plan
    and the wrong one for this: in a cluttered room most of the floor junction is the FEET of
    furniture, the ceiling height it derives from them is short, and every distance read off the
    ceiling line with that height is short too. Measured on the reference bedroom, the outline
    put the wall behind the wardrobe at one camera height where the junction visible under the
    desk says two — and a fill drawn on that box painted wall halfway across the floor.

    So the fit is done in two steps that each trust what furniture cannot fake:

    1. The SHAPE, from the ceiling line alone, which furniture almost never reaches. With the
       ceiling's height left as the unit, every ceiling observation is a point on the plan up to
       one common scale; the rectangle's orientation and its four walls are fitted to those
       points (each wall at the mode of what it owns — see :func:`_side_mode`).
    2. The SCALE — the ceiling's height above the eye, ``ceiling - 1`` — from the floor line where
       it is visible: each floor observation divided by the unit rectangle's distance in its
       direction is one estimate of it. Curtain feet read low, doorways read high, so the
       estimate is the cluster nearest :data:`_SCALE_PRIOR` that enough columns agree on.
    """
    report: dict[str, float] = {}
    ct, ce = junctions.ceiling_theta, junctions.ceiling_elevation
    ft, fd = junctions.floor_theta, junctions.floor_depression
    if ct.size < 4 * _MIN_JUNCTION_POINTS:
        return None, {"too_few_ceiling": float(ct.size)}

    r_unit = 1.0 / np.tan(ce)
    sane = (r_unit > 0.2) & (r_unit < 12.0)
    ct, r_unit = ct[sane], r_unit[sane]
    px, pz = r_unit * np.sin(ct), r_unit * np.cos(ct)

    def turned(angle: float) -> tuple[np.ndarray, np.ndarray]:
        return px * math.cos(angle) + pz * math.sin(angle), -px * math.sin(angle) + pz * math.cos(angle)

    def start_for(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.array(
            [
                min(float(np.percentile(u, 3)), -0.2),
                max(float(np.percentile(u, 97)), 0.2),
                min(float(np.percentile(v, 3)), -0.2),
                max(float(np.percentile(v, 97)), 0.2),
            ]
        )

    def loss_of(angle: float) -> tuple[float, np.ndarray, np.ndarray]:
        u, v = turned(angle)
        bounds, found = _fit_sides(u, v, start_for(u, v), float(np.median(r_unit)))
        hits, _ = _rectangle_hits(bounds, np.arctan2(u, v), 0.0)
        relative = np.abs(hits - r_unit) / np.maximum(r_unit, 1e-6)
        # Share of observations the rectangle explains to 2 %: an inlier count rather than a
        # mean, so the wardrobe tops that sit in front of a wall cost nothing either way.
        return -float((relative < 0.02).mean()), bounds, found

    base = prior.orientation() if prior is not None else 0.0
    coarse = [base + math.radians(d) for d in np.arange(-45.0, 45.0, 1.0)]
    scored = [(loss_of(a)[0], a) for a in coarse]
    best_angle = min(scored)[1]
    fine = [best_angle + math.radians(d) for d in np.arange(-1.0, 1.0001, 0.1)]
    loss, angle = min((loss_of(a)[0], a) for a in fine)
    _, bounds_unit, found = loss_of(angle)
    report["orientation_deg"] = round(math.degrees(angle), 2)
    report["ceiling_inliers"] = round(-loss, 3)

    # Walls the ceiling line never reached (curtains to the ceiling, a wardrobe across a whole
    # wall) keep the outline's estimate, converted to the unit scale below once it is known.
    scale = None
    if ft.size:
        r_floor = 1.0 / np.tan(fd)
        local = ft + angle
        hits_unit, side_f = _rectangle_hits(bounds_unit, local, 0.0)
        estimates = r_floor / np.maximum(hits_unit, 1e-6)
        usable = (estimates > _SCALE_RANGE[0]) & (estimates < _SCALE_RANGE[1]) & found[side_f]
        estimates = estimates[usable]
        if estimates.size >= _MIN_JUNCTION_POINTS:
            # Clusters 3 % wide; the one nearest the prior among those with enough support.
            grid = np.arange(_SCALE_RANGE[0], _SCALE_RANGE[1], 0.01)
            support = np.array([(np.abs(estimates - g) <= 0.03 * g).sum() for g in grid])
            candidates = grid[support >= _MIN_JUNCTION_POINTS]
            if candidates.size:
                pick = float(candidates[np.argmin(np.abs(np.log(candidates / _SCALE_PRIOR)))])
                near = estimates[np.abs(estimates - pick) <= 0.03 * pick]
                scale = float(np.median(near))
                report["scale_columns"] = float(near.size)
    if scale is None:
        scale = (prior.ceiling - 1.0) if prior is not None else _SCALE_PRIOR
        report["scale_from_prior"] = 1.0
    report["ceiling"] = round(1.0 + scale, 4)

    bounds = bounds_unit * scale
    if not found.all():
        # A wall the ceiling line never reached is placed by its own floor line where that is
        # visible — wall paint first, then a curtain's hem — before the outline is asked.
        u_all, v_all = [], []
        for theta_obs, dep_obs in ((ft, fd), (junctions.hem_theta, junctions.hem_depression)):
            if theta_obs is None or theta_obs.size == 0:
                continue
            r_obs = 1.0 / np.tan(dep_obs)
            ox, oz = r_obs * np.sin(theta_obs), r_obs * np.cos(theta_obs)
            u_all.append(ox * math.cos(angle) + oz * math.sin(angle))
            v_all.append(-ox * math.sin(angle) + oz * math.cos(angle))
        if u_all:
            u_obs, v_obs = np.concatenate(u_all), np.concatenate(v_all)
            _, side_obs = _rectangle_hits(bounds, np.arctan2(u_obs, v_obs), 0.0)
            for index, coord in ((0, u_obs), (1, u_obs), (2, v_obs), (3, v_obs)):
                if found[index]:
                    continue
                mode = _side_mode(coord[side_obs == index], float(np.median(np.abs(coord))) + 1e-6)
                if mode is not None and (mode < 0) == (index % 2 == 0):
                    bounds[index] = mode
                    found[index] = True
                    report[f"side{index}_from_floor"] = 1.0
    if prior is not None and not found.all():
        # The outline's own walls, in the same turned frame, for the sides nothing observed.
        corners = prior.polygon
        u = corners[:, 0] * math.cos(angle) + corners[:, 1] * math.sin(angle)
        v = -corners[:, 0] * math.sin(angle) + corners[:, 1] * math.cos(angle)
        fallback = np.array([u.min(), u.max(), v.min(), v.max()])
        bounds = np.where(found, bounds, fallback)
        report["sides_from_outline"] = float((~found).sum())
    if bounds[0] >= -0.05 or bounds[1] <= 0.05 or bounds[2] >= -0.05 or bounds[3] <= 0.05:
        return None, {**report, "camera_outside": 1.0}

    corners_local = np.array(
        [[bounds[0], bounds[2]], [bounds[1], bounds[2]], [bounds[1], bounds[3]], [bounds[0], bounds[3]]]
    )
    c, s = math.cos(angle), math.sin(angle)
    polygon = np.column_stack(
        [corners_local[:, 0] * c - corners_local[:, 1] * s, corners_local[:, 0] * s + corners_local[:, 1] * c]
    )
    return RoomBox(polygon=_counter_clockwise(polygon), ceiling=1.0 + scale), report


def floor_rows(box: RoomBox, width: int, height: int) -> np.ndarray:
    """Row of the box's floor line in every column of an equirect canvas (float)."""
    theta = 2.0 * np.pi * (np.arange(width) + 0.5) / width - np.pi
    rays = np.stack([np.sin(theta), np.zeros(width), np.cos(theta)], axis=1)
    hits = trace(box, rays)
    distance = np.hypot(hits.wall_point[:, 0], hits.wall_point[:, 2])
    distance = np.where(hits.wall >= 0, distance, np.nan)
    depression = np.arctan2(1.0, np.maximum(distance, 1e-6))
    return (depression + 0.5 * np.pi) / np.pi * height - 0.5


#: How far from the box's floor line the photographed one is searched for, in degrees.
_SNAP_SEARCH_DEG = 4.0
#: An edge must be at least this strong (grey levels per pixel, Sobel on a blurred image) to be
#: read as the floor line. Paint against tile at the reference bedroom's exposure is 6-20.
_SNAP_MIN_EDGE = 5.0
#: Across a span this wide with no reading, the correction fades back to the box's own line.
_SNAP_REACH_DEG = 45.0


def _read_floor_line(
    gy: np.ndarray,
    base: np.ndarray,
    upper: np.ndarray,
    floor: np.ndarray,
    blocked: np.ndarray,
    search_deg: float,
) -> np.ndarray:
    """Per column, the row of the photographed floor line near ``base`` — or NaN.

    The strongest vertical edge within ``search_deg`` of the box's line that has wall (or a held
    fitting) above it and floor below it, both clear of anything being removed. Majority over a
    short probe rather than every pixel, so a speck of mislabel does not veto a clean line.
    Vectorised over columns with running sums down each column: the per-column loop it replaced
    was 14 s of a run.
    """
    height, width = gy.shape
    search = int(round(search_deg / 180.0 * height))
    probe = max(3, int(round(0.6 / 180.0 * height)))
    gap = max(2, probe // 3)
    readings = np.full(width, np.nan)
    columns = np.arange(0, width, 2)
    centre = base[columns]
    ok_cols = np.isfinite(centre)
    columns, centre = columns[ok_cols], centre[ok_cols]
    if columns.size == 0:
        return readings
    # Running sums down each (sampled) column, so a probe's mean is two lookups.
    up = np.zeros((height + 1, columns.size), np.float32)
    up[1:] = np.cumsum((upper[:, columns] & ~blocked[:, columns]).astype(np.float32), axis=0)
    fl = np.zeros((height + 1, columns.size), np.float32)
    fl[1:] = np.cumsum((floor[:, columns] & ~blocked[:, columns]).astype(np.float32), axis=0)
    offsets = np.arange(-search, search + 1)
    rows = np.rint(centre)[None, :].astype(np.int64) + offsets[:, None]
    rows = np.clip(rows, probe + gap, height - probe - gap - 1)
    col_index = np.broadcast_to(np.arange(columns.size)[None, :], rows.shape)
    strength = gy[rows, columns[None, :]]
    above = (up[rows - gap, col_index] - up[rows - gap - probe, col_index]) / probe
    below = (fl[rows + gap + probe, col_index] - fl[rows + gap, col_index]) / probe
    ok = (above > 0.7) & (below > 0.7) & (strength >= _SNAP_MIN_EDGE)
    best = np.argmax(np.where(ok, strength, -1.0), axis=0)
    found = ok[best, np.arange(columns.size)]
    readings[columns[found]] = rows[best[found], np.arange(columns.size)[found]].astype(np.float64)
    return readings


def _edge_strength(image: np.ndarray) -> np.ndarray:
    grey = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    return np.abs(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)) / 4.0


def refit_walls(box: RoomBox, rows: np.ndarray, height: int) -> tuple[RoomBox, dict[str, float]]:
    """Move each wall to the floor line photographed in its own columns.

    ``rows`` is a per-column floor-line row (NaN where unread) on a canvas ``height`` rows high.
    A wall with enough readings takes the MODE of their distances along its normal — the
    readings are in camera heights already, so no ceiling height enters — and keeps its
    orientation; the corners are re-solved from the moved lines. A wall with too few readings
    stays where it was.
    """
    width = rows.size
    found = np.nonzero(np.isfinite(rows))[0]
    report: dict[str, float] = {}
    if found.size < _MIN_JUNCTION_POINTS:
        return box, report
    theta = 2.0 * np.pi * (found + 0.5) / width - np.pi
    depression = np.pi * (rows[found] + 0.5) / height - 0.5 * np.pi
    ok = depression > np.radians(3.0)
    theta, depression = theta[ok], depression[ok]
    radius = 1.0 / np.tan(depression)
    points = np.stack([radius * np.sin(theta), radius * np.cos(theta)], axis=1)
    owner = trace(box, np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)).wall
    lines = []
    for index in range(box.walls):
        start, end = box.wall(index)
        direction = (end - start) / max(float(np.hypot(*(end - start))), _EPS)
        normal = np.array([-direction[1], direction[0]])
        offset = float(normal @ start)
        mine = points[owner == index] @ normal
        mode = _side_mode(mine, abs(offset))
        if mode is not None and mode < 0.0 and abs(mode - offset) <= 0.45 * abs(offset):
            report[f"wall{index}_moved"] = round(mode - offset, 4)
            offset = mode
        lines.append((normal, offset))
    corners = []
    for index in range(len(lines)):
        n1, o1 = lines[index - 1]
        n2, o2 = lines[index]
        matrix = np.array([n1, n2])
        if abs(float(np.linalg.det(matrix))) < 1e-6:
            corners.append(box.polygon[index])
        else:
            corners.append(np.linalg.solve(matrix, np.array([o1, o2])))
    moved = RoomBox(polygon=_counter_clockwise(np.array(corners, np.float64)), ceiling=box.ceiling)
    if not moved.convex_around_eye():
        return box, {**report, "refit_rejected": 1.0}
    return moved, report


#: The first, wide reading of the floor line, which places walls the box got badly wrong.
_REFIT_SEARCH_DEG = 12.0


def fit_room(
    image: np.ndarray,
    classes: np.ndarray,
    hole: np.ndarray,
    prior: RoomBox | None,
) -> tuple[RoomBox | None, np.ndarray | None, dict[str, object]]:
    """The room's box and its floor-line correction, from everything the panorama shows.

    Three readings, each trusting what the one before could not:

    1. :func:`fit_box` — the ceiling line's SHAPE, scaled by the visible floor line where wall
       paint meets it. Right in orientation and corners; in a crowded room a wall can still be
       a third out, because the scale rests on a handful of columns.
    2. :func:`refit_walls` on a WIDE reading of the floor line (:data:`_REFIT_SEARCH_DEG`): every
       wall with enough of its own floor line in view moves onto it. Measured on the reference
       bedroom, the wall behind the bed moved from one camera height to where its skirting is.
    3. :func:`floor_line_shift` on a NARROW reading around the moved box: the per-column
       correction that makes the filled line start exactly where the photographed one stops.
    """
    from biosphere.ai import furniture_seg  # noqa: PLC0415

    height, width = image.shape[:2]
    report: dict[str, object] = {}
    junctions = observe_junctions(image, classes, hole)
    box, fitted = fit_box(junctions, prior=prior)
    report["fit"] = fitted
    if box is None:
        box = prior
    if box is None:
        return None, None, report

    full = cv2.resize(classes, (width, height), interpolation=cv2.INTER_NEAREST)
    keep = np.zeros(256, bool)
    for class_id in furniture_seg._ROOM_SHELL:  # noqa: SLF001 - the one definition of "the room"
        keep[class_id] = True
    keep[[_ADE_FLOOR, _ADE_CEILING]] = False
    upper = keep[full]
    floor = full == _ADE_FLOOR
    blocked = hole > 127
    gy = _edge_strength(image)

    wide = _read_floor_line(gy, floor_rows(box, width, height), upper, floor, blocked, _REFIT_SEARCH_DEG)
    box, moved = refit_walls(box, wide, height)
    report["refit"] = moved
    shift, snapped = floor_line_shift(image, box, upper, floor, hole, gy=gy)
    report["floor_line"] = snapped
    return box, shift, report


def floor_line_shift(
    image: np.ndarray,
    box: RoomBox,
    upper: np.ndarray,
    floor: np.ndarray,
    hole: np.ndarray,
    *,
    gy: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Per column, how many rows the PHOTOGRAPHED floor line sits below the box's.

    WHY THE BOX'S LINE IS NOT DRAWN AS IT IS. A box fitted to a cluttered room is right in shape
    and a few per cent out in distance, and a few per cent at two metres is a degree — a visible
    step where the filled floor line meets the real one at the edge of the hole. So the line is
    read off the photograph in every column where it is in plain view (wall or a held fitting
    above, floor below, nothing being removed in between: the strongest vertical edge within
    :data:`_SNAP_SEARCH_DEG` of the box's line), cleaned with a running median, and carried
    across each hole by interpolating between the readings either side of it. The filled line
    therefore STARTS where the photographed one stops, whatever the box says in between.
    """
    height, width = hole.shape
    base = floor_rows(box, width, height)
    if gy is None:
        gy = _edge_strength(image)
    search = int(round(_SNAP_SEARCH_DEG / 180.0 * height))
    readings = _read_floor_line(gy, base, upper, floor, hole > 127, _SNAP_SEARCH_DEG) - base
    report = {"columns_read": float(np.isfinite(readings).sum())}
    found = np.nonzero(np.isfinite(readings))[0]
    if found.size < 8:
        return np.zeros(width), report

    # Running median over a few degrees of neighbours: a single column's edge can be a tile
    # joint or a cable; the floor line is what a run of columns agrees on.
    values = readings[found]
    window = max(5, int(round(2.0 / 360.0 * width)) // 2)
    cleaned = np.empty_like(values)
    for i in range(values.size):
        near = np.abs(((found - found[i]) + width // 2) % width - width // 2) <= window
        cleaned[i] = np.median(values[near])
    keep = np.abs(values - cleaned) <= max(3.0, 0.5 * search / 4)
    found, values = found[keep], cleaned[keep]
    report["columns_kept"] = float(found.size)
    if found.size < 8:
        return np.zeros(width), report

    # Cyclic interpolation between readings, faded toward zero across long unread spans.
    cols = np.arange(width)
    xs = np.concatenate([found - width, found, found + width]).astype(np.float64)
    ys = np.tile(values, 3)
    shift = np.interp(cols, xs, ys)
    order = np.sort(found)
    nxt = np.searchsorted(order, cols) % order.size
    prv = (nxt - 1) % order.size
    to_next = (order[nxt] - cols) % width
    to_prev = (cols - order[prv]) % width
    reach = _SNAP_REACH_DEG / 360.0 * width
    weight = np.clip(1.0 - (np.minimum(to_next, to_prev) - reach / 3.0) / reach, 0.0, 1.0)
    shift = shift * weight
    report["median_shift_rows"] = round(float(np.median(values)), 2)
    report["p90_abs_shift_rows"] = round(float(np.percentile(np.abs(values), 90)), 2)
    return shift, report


#: A cluster of ceiling-line observations is a candidate wall when it holds this share of the
#: strongest cluster's support; the farthest candidate wins (see :func:`_side_mode`).
_FAR_CLUSTER_SHARE = 0.4

#: The ceiling's height above the eye, in eye heights, that a scale estimate is pulled toward.
#: 0.65 is a 2.64 m ceiling over a 1.6 m eye; measured on the two reference captures of the
#: bedroom where the floor line is visible on several walls, 0.656 and 0.666.
_SCALE_PRIOR = 0.65
#: Scales outside this are not a room: 0.35 is a 2.2 m ceiling over the eye at 1.6 m, 1.3 a
#: 3.7 m one — and a doorway's far junction routinely reads past it.
_SCALE_RANGE = (0.35, 1.3)


# ======================================================================================
# Cameras: how an image's pixels become rays
# ======================================================================================


class Camera:
    """The two things the plane fill needs from an image: rays out, and pixels back."""

    width: int
    height: int

    def rays(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def project(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
        """``(x, y, in_view)`` pixel coordinates for unit rays."""
        raise NotImplementedError

    def sample(
        self, image: np.ndarray, x: np.ndarray, y: np.ndarray, interpolation: int
    ) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass(slots=True)
class EquirectCamera(Camera):
    """A levelled equirectangular panorama: the engine's own convention."""

    width: int
    height: int

    def rays(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        theta = 2.0 * np.pi * (xs + 0.5) / float(self.width) - np.pi
        phi = 0.5 * np.pi - np.pi * (ys + 0.5) / float(self.height)
        cp = np.cos(phi)
        return np.stack([cp * np.sin(theta), -np.sin(phi), cp * np.cos(theta)], axis=-1)

    def project(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        n = np.sqrt(x * x + y * y + z * z) + _EPS
        theta = np.arctan2(x, z)
        phi = np.arcsin(np.clip(-y / n, -1.0, 1.0))
        u = (theta + np.pi) / (2.0 * np.pi) * self.width - 0.5
        v = (0.5 * np.pi - phi) / np.pi * self.height - 0.5
        return u, v, np.ones(u.shape, bool)

    def sample(self, image: np.ndarray, x: np.ndarray, y: np.ndarray, interpolation: int) -> np.ndarray:
        # BORDER_WRAP rather than a padded copy: the canvas wraps in x, which is exactly what
        # wrap does, and rows are clamped first so the vertical wrap is never consulted. The
        # padded copy cost 20 s of a run — a 25 MB copy per raster per channel set.
        h, w = image.shape[:2]
        mx = np.mod(x + 0.0, w).astype(np.float32)
        my = np.clip(y, 0.0, h - 1.0).astype(np.float32)
        return cv2.remap(image, mx, my, interpolation, borderMode=cv2.BORDER_WRAP)


@dataclass(slots=True)
class PinholeCamera(Camera):
    """An ordinary photograph: intrinsics ``K`` and ``R`` mapping WORLD to CAMERA.

    The world is the engine's (+Y down) with gravity along +Y, which is what lets a photograph
    share :func:`trace` with a panorama: the floor is ``y = +1`` in both.
    """

    width: int
    height: int
    K: np.ndarray
    R: np.ndarray

    def rays(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        cam = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs, dtype=np.float64)], axis=-1)
        cam /= np.linalg.norm(cam, axis=-1, keepdims=True)
        return cam @ self.R  # R.T @ d, as a row product

    def project(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cam = rays @ self.R.T
        z = cam[..., 2]
        front = z > 1e-6
        safe = np.where(front, z, 1.0)
        x = self.K[0, 0] * cam[..., 0] / safe + self.K[0, 2]
        y = self.K[1, 1] * cam[..., 1] / safe + self.K[1, 2]
        inside = front & (x >= -0.5) & (x <= self.width - 0.5) & (y >= -0.5) & (y <= self.height - 0.5)
        return x, y, inside

    def sample(self, image: np.ndarray, x: np.ndarray, y: np.ndarray, interpolation: int) -> np.ndarray:
        return cv2.remap(
            image, x.astype(np.float32), y.astype(np.float32), interpolation, borderMode=cv2.BORDER_REPLICATE
        )


# ======================================================================================
# Rectified rasters
# ======================================================================================


@dataclass(slots=True)
class PlaneRaster:
    """A square-on view of one plane of the room, as a pixel grid.

    ``origin + col * pixel * axis_u + row * pixel * axis_v`` is the 3-D point at a pixel centre.
    For a wall, ``axis_u`` runs along it from its start corner and ``axis_v`` points down, so
    row 0 is the ceiling line and the last row the floor line — a skirting board is a run of
    rows and a cornice is another.
    """

    plane: int
    origin: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    pixel: float
    width: int
    height: int

    def points(self) -> np.ndarray:
        cols = (np.arange(self.width, dtype=np.float64) + 0.5) * self.pixel
        rows = (np.arange(self.height, dtype=np.float64) + 0.5) * self.pixel
        return (
            self.origin[None, None, :]
            + cols[None, :, None] * self.axis_u[None, None, :]
            + rows[:, None, None] * self.axis_v[None, None, :]
        )

    def coords(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fractional ``(col, row)`` of 3-D points on this plane."""
        rel = points - self.origin
        col = (rel @ self.axis_u) / self.pixel - 0.5
        row = (rel @ self.axis_v) / self.pixel - 0.5
        return col, row


def plane_rasters(box: RoomBox, pixel: float = _PIXEL_H) -> dict[int, PlaneRaster]:
    """One raster per wall, one for the floor and one for the ceiling."""
    out: dict[int, PlaneRaster] = {}
    top = -box.eye_to_ceiling

    tall = box.ceiling + _SLACK_H
    longest = max(float(np.hypot(*(box.wall(i)[1] - box.wall(i)[0]))) for i in range(box.walls))
    # ONE pixel size for every wall, so row k of every wall raster is the same height above the
    # floor — which is what lets the walls be read together (`wall_profile`).
    step = max(pixel, longest / _MAX_RASTER_SIDE, tall / _MAX_RASTER_SIDE)
    for index in range(box.walls):
        start, end = box.wall(index)
        delta = end - start
        length = float(np.hypot(delta[0], delta[1]))
        if length < 1e-3:
            continue
        out[index] = PlaneRaster(
            plane=index,
            origin=np.array([start[0], top, start[1]], np.float64),
            axis_u=np.array([delta[0] / length, 0.0, delta[1] / length], np.float64),
            axis_v=np.array([0.0, 1.0, 0.0], np.float64),
            pixel=step,
            width=max(2, int(math.ceil(length / step))),
            # Below the floor line by `_SLACK_H`: where the floor line is moved down onto the
            # photographed one, the pixels it hands to the wall land here.
            height=max(2, int(math.ceil(tall / step))),
        )

    angle = box.orientation()
    axis_u = np.array([math.cos(angle), 0.0, math.sin(angle)], np.float64)
    axis_v = np.array([-math.sin(angle), 0.0, math.cos(angle)], np.float64)
    flat = box.polygon
    u = flat[:, 0] * axis_u[0] + flat[:, 1] * axis_u[2]
    v = flat[:, 0] * axis_v[0] + flat[:, 1] * axis_v[2]
    # Past the walls by `_SLACK_H` on every side, for the same reason the walls reach below
    # the floor: a floor line moved outward onto the photographed one hands pixels to the floor.
    span_u = float(u.max() - u.min()) + 2.0 * _SLACK_H
    span_v = float(v.max() - v.min()) + 2.0 * _SLACK_H
    step = max(pixel, span_u / _MAX_RASTER_SIDE, span_v / _MAX_RASTER_SIDE)
    for plane, height in ((FLOOR, 1.0), (CEILING, top)):
        corner = (float(u.min()) - _SLACK_H) * axis_u + (float(v.min()) - _SLACK_H) * axis_v
        out[plane] = PlaneRaster(
            plane=plane,
            origin=np.array([corner[0], height, corner[2]], np.float64),
            axis_u=axis_u,
            axis_v=axis_v,
            pixel=step,
            width=max(2, int(math.ceil(span_u / step))),
            height=max(2, int(math.ceil(span_v / step))),
        )
    return out


@dataclass(slots=True)
class Rectified:
    """One plane, as the image saw it."""

    raster: PlaneRaster
    colour: np.ndarray        # (h, w, 3) float32
    #: The plane's own surface was visible here and nothing is being removed in front of it.
    valid: np.ndarray         # (h, w) bool
    #: This raster pixel is behind the hole: the fill must produce it.
    hole: np.ndarray          # (h, w) bool
    #: Something the image protects (a curtain, a door) stands here. Not a sample, not a hole.
    held: np.ndarray          # (h, w) bool
    #: Class of what stands in front, for the layer-aware wall fill. 0 none, 1 curtain.
    layer: np.ndarray         # (h, w) uint8


def _outline_mask(box: RoomBox, raster: PlaneRaster) -> np.ndarray:
    """The room's outline, rasterised onto a floor or ceiling raster, grown by the slack."""
    corners = np.column_stack(
        [box.polygon[:, 0], np.full(box.walls, raster.origin[1]), box.polygon[:, 1]]
    )
    col, row = raster.coords(corners)
    # fillPoly takes integer vertices; 4 bits of sub-pixel precision keep the edge honest.
    shift = 4
    vertices = np.round(np.column_stack([col + 0.5, row + 0.5]) * (1 << shift)).astype(np.int32)
    mask = np.zeros((raster.height, raster.width), np.uint8)
    cv2.fillPoly(mask, [vertices], 1, lineType=cv2.LINE_8, shift=shift)
    grow = max(1, int(round(_SLACK_H / raster.pixel)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return mask > 0


def rectify(
    image: np.ndarray,
    camera: Camera,
    box: RoomBox,
    raster: PlaneRaster,
    *,
    hole: np.ndarray,
    surface: np.ndarray,
    held: np.ndarray | None = None,
    layer: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
) -> Rectified | None:
    """Sample ``image`` square-on to one plane. ``None`` when the image never sees the plane.

    ``surface`` is the per-pixel surface the image shows (0 ceiling, 1 wall paint, 2 floor,
    anything else not one of them). A raster pixel is a SAMPLE of this plane only when the image
    pixel it comes from is unmasked, shows this kind of surface, and the empty room's own
    geometry says the ray reaches this plane first — the last condition is what keeps the wall
    behind a re-entrant corner from borrowing the corner.
    """
    points = raster.points()
    norms = np.linalg.norm(points, axis=-1, keepdims=True)
    rays = points / np.maximum(norms, _EPS)
    x, y, inside = camera.project(rays)
    if not inside.any():
        return None

    if box.convex_around_eye():
        # THE COMMON CASE NEEDS NO RAY TRACE. From a point inside a convex room every point of
        # every wall and every point of the floor inside the outline is in plain view, so
        # visibility is just "inside the outline" — measured, the trace was 13 of 175 s of a
        # synthetic run, for an answer this rasterises in milliseconds.
        own = _outline_mask(box, raster) if raster.plane < 0 else np.ones(inside.shape, bool)
    else:
        hits = trace(box, rays)
        own = hits.plane == raster.plane
        if raster.plane >= 0:
            # The along-wall coordinate has to match too: a ray may hit the same wall index at
            # a different point only through numerical noise at a corner, and that pixel is the
            # corner of two walls rather than a sample of either.
            expected = (points - raster.origin) @ raster.axis_u
            own &= np.abs(np.nan_to_num(hits.along, nan=-1e9) - expected) < 4.0 * raster.pixel
    visible = own & inside

    colour = camera.sample(image, x, y, cv2.INTER_LINEAR).astype(np.float32)
    hole_s = camera.sample(hole, x, y, cv2.INTER_NEAREST) > 127
    surface_s = camera.sample(surface, x, y, cv2.INTER_NEAREST)
    wanted = {FLOOR: 2, CEILING: 0}.get(raster.plane, 1)
    held_s = (
        camera.sample(held, x, y, cv2.INTER_NEAREST) > 127
        if held is not None
        else np.zeros(visible.shape, bool)
    )
    layer_s = (
        camera.sample(layer, x, y, cv2.INTER_NEAREST)
        if layer is not None
        else np.zeros(visible.shape, np.uint8)
    )

    if exclude is not None:
        # Never photographed (the stitcher's own pole fill and branding): not a sample of
        # anything, however floor-coloured it is.
        visible_samples = visible & ~(camera.sample(exclude, x, y, cv2.INTER_NEAREST) > 127)
    else:
        visible_samples = visible
    return Rectified(
        raster=raster,
        colour=colour,
        valid=visible_samples & ~hole_s & (surface_s == wanted) & ~held_s,
        hole=visible & hole_s,
        held=visible & held_s & ~hole_s,
        layer=np.where(visible, layer_s, 0).astype(np.uint8),
    )


# ======================================================================================
# Filling one plane
# ======================================================================================


def _push_pull(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Normalised-convolution pyramid: the smooth field ``weight`` implies, everywhere.

    The same construction ``empty_room`` uses on the sphere, without the wrap: a plane raster
    has edges. Linear in the pixel count and with no reach limit, so a hole the size of a
    wardrobe is filled from the whole plane rather than from its rim.
    """
    if float(weight.sum()) < 1.0:
        return np.zeros_like(values, dtype=np.float32)
    vals = values.astype(np.float32)
    if vals.ndim == 2:
        vals = vals[..., None]
    channels = vals.shape[2]

    def down(array: np.ndarray) -> np.ndarray:
        result = cv2.pyrDown(array)
        return result.reshape(result.shape[0], result.shape[1], -1) if array.ndim == 3 else result

    w = weight.astype(np.float32)
    colours = [vals * w[..., None]]
    weights = [w]
    while min(weights[-1].shape[:2]) > 4:
        colours.append(down(colours[-1]).reshape(*down(weights[-1]).shape[:2], channels))
        weights.append(down(weights[-1]))
    out = colours[-1] / np.maximum(weights[-1], 1e-6)[..., None]
    for level in range(len(colours) - 2, -1, -1):
        h, w_ = weights[level].shape[:2]
        up = cv2.pyrUp(out, dstsize=(w_, h)).reshape(h, w_, channels)
        alpha = np.clip(weights[level], 0.0, 1.0)[..., None]
        own = colours[level] / np.maximum(weights[level], 1e-6)[..., None]
        out = own * alpha + up * (1.0 - alpha)
    return out.reshape(values.shape).astype(np.float32)


#: How far from a hole the plane's statistics are distrusted, in camera heights.
#:
#: The rim of a removed object is not the room: it is a contact shadow on the floor and a
#: darker band on the wall where the object blocked the light, and a fill that continues the
#: rim continues the shadow — which is exactly the grey smear the learned fills leave. 4 cm is
#: the width of the shadow a bed frame casts on the floor beside it in the reference bedroom;
#: past it the floor reads as floor.
_RIM_H = 0.025


def _sample_points(
    image: np.ndarray, col: np.ndarray, row: np.ndarray, interpolation: int, border: int, value: float = 0.0
) -> np.ndarray:
    """``cv2.remap`` at a flat list of points.

    remap refuses a map with a side over 32767, and a hole is millions of pixels, so the list is
    folded into a block 4096 wide and unfolded afterwards.
    """
    count = int(col.size)
    side = 4096
    rows = max(1, (count + side - 1) // side)
    pad = rows * side - count
    mx = np.concatenate([col.astype(np.float32).ravel(), np.zeros(pad, np.float32)]).reshape(rows, side)
    my = np.concatenate([row.astype(np.float32).ravel(), np.zeros(pad, np.float32)]).reshape(rows, side)
    out = cv2.remap(image, mx, my, interpolation, borderMode=border, borderValue=value)
    out = out.reshape(rows * side, -1)[:count]
    return out if image.ndim == 3 else out[:, 0]


def _rim_free(valid: np.ndarray, hole: np.ndarray, pixel: float) -> np.ndarray:
    radius = max(1, int(round(_RIM_H / pixel)))
    grown = cv2.dilate(
        hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    )
    return valid & (grown == 0)


def _consistent(colour: np.ndarray, sample: np.ndarray, *, rows: bool, floor_tolerance: float) -> np.ndarray:
    """The samples that look like the surface they are supposed to be a sample of.

    The segmentation is per pixel and so are its mistakes: the top of a desk that the mask's
    dilation did not reach, a dark bag the model called floor, the shadow under a bed frame.
    Every one of those is a "valid" sample by label and a disaster by value — a row fill
    carries a desk top's white edge the whole width of the hole, which is how the first real run
    drew a bench along the wall where a wardrobe had stood.

    A sample is kept when it is close to either of two references: its ROW's median along the
    whole wall (so a skirting, white against a coloured wall, is consistent in its own rows), or
    the plane's smooth 2-D field (so a pool of light is consistent where it falls). The
    tolerance is the larger of ``floor_tolerance`` and three robust spreads, so a plane whose
    paint genuinely varies is not stripped of half its samples.
    """
    if int(sample.sum()) < 64:
        return sample
    references = [_push_pull(colour, sample.astype(np.float32))]
    if rows:
        height = sample.shape[0]
        medians = np.full((height, 3), np.nan, np.float32)
        counts = sample.sum(axis=1)
        for row in np.nonzero(counts >= 24)[0]:
            medians[row] = np.median(colour[row][sample[row]], axis=0)
        known = np.nonzero(np.isfinite(medians[:, 0]))[0]
        if known.size >= 2:
            for channel in range(3):
                medians[:, channel] = np.interp(np.arange(height), known, medians[known, channel])
            medians = cv2.medianBlur(np.ascontiguousarray(medians[:, None, :]).astype(np.float32), 5)[:, 0, :] if height >= 5 else medians
            references.append(np.broadcast_to(medians[:, None, :], colour.shape))
    distance = np.min(
        np.stack([np.abs(colour - ref).max(axis=2) for ref in references]), axis=0
    )
    spread = float(np.median(distance[sample])) * 1.4826
    tolerance = max(floor_tolerance, 3.0 * spread)
    kept = sample & (distance <= tolerance)
    # And the surface's own HUE: a rug the model called floor is a few levels from the laminate in
    # brightness and twenty in colour, and it tinted the reference bedroom's rebuilt floor beige.
    # On a wall the hue is the ROW's: a white skirting under lilac paint is the wall's own, and a
    # single hue for the whole wall threw every skirting sample away.
    if int(kept.sum()) >= 64:
        lab = cv2.cvtColor(np.clip(colour, 0, 255).astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)
        reference = np.broadcast_to(np.median(lab[kept][:, 1:], axis=0), lab[..., 1:].shape).copy()
        if rows:
            for row in np.nonzero(kept.sum(axis=1) >= 24)[0]:
                reference[row] = np.median(lab[row][kept[row]][:, 1:], axis=0)
        off_hue = np.sqrt(((lab[..., 1:] - reference) ** 2).sum(axis=2))
        kept &= off_hue <= max(_HUE_TOLERANCE, 3.0 * 1.4826 * float(np.median(off_hue[kept])))
    return kept


def _row_fill(colour: np.ndarray, valid: np.ndarray, todo: np.ndarray, *, reach: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill ``todo`` pixels along their ROW from the nearest valid run on each side.

    The endpoint of each side is the median of the ``reach`` valid pixels nearest the gap, so a
    lone dark pixel at the edge of a hole does not decide the colour of a metre of wall. Between
    the two endpoints the fill is a straight ramp — the shading a flat wall under a distant
    light actually has — and with one side only, it holds that side's colour.

    Returns ``(filled, done)``; rows with no valid pixel at all are left for the caller.
    """
    height, width = valid.shape
    out = colour.copy()
    done = np.zeros_like(todo)
    for row in range(height):
        gaps = todo[row]
        if not gaps.any():
            continue
        good = np.nonzero(valid[row])[0]
        if good.size == 0:
            continue
        cols = np.nonzero(gaps)[0]
        # Split the row's work into contiguous gaps.
        breaks = np.nonzero(np.diff(cols) > 1)[0]
        starts = np.concatenate([[cols[0]], cols[breaks + 1]])
        ends = np.concatenate([cols[breaks], [cols[-1]]])
        for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
            left = good[good < a][-reach:]
            right = good[good > b][:reach]
            if left.size == 0 and right.size == 0:
                continue
            lv = np.median(colour[row, left], axis=0) if left.size else None
            rv = np.median(colour[row, right], axis=0) if right.size else None
            span = np.arange(a, b + 1)
            if lv is not None and rv is not None:
                la, rb = float(left[-1]), float(right[0])
                t = ((span - la) / max(rb - la, 1.0))[:, None]
                out[row, a : b + 1] = lv[None, :] * (1.0 - t) + rv[None, :] * t
            else:
                out[row, a : b + 1] = (lv if lv is not None else rv)[None, :]
            done[row, a : b + 1] = True
    return out, done


def _curtain_down(colour: np.ndarray, curtain: np.ndarray, todo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continue a curtain down its own columns by repeating the curtain's own lowest stretch.

    A curtain is the same cloth all the way down: its folds are vertical and its weave does not
    change with height. So a hole below a curtain is filled from the stretch of the same curtain
    just above it, repeated downward in the same column. Holding each column's last colour
    instead was the first version, and measured on the second reference capture it read as
    paint dragged down in streaks: one flat colour per column has a fold's edge but none of its
    shading or weave.
    """
    height, width = curtain.shape
    out = colour.copy()
    done = np.zeros_like(todo)
    period = max(8, height // 10)
    for col in np.nonzero(todo.any(axis=0) & curtain.any(axis=0))[0]:
        rows = np.nonzero(curtain[:, col])[0]
        holes = np.nonzero(todo[:, col])[0]
        for row in holes:
            above = rows[rows < row]
            if above.size == 0:
                continue
            last = int(above[-1])
            # The run of curtain ending at `last`, at most `period` long.
            start = last
            while start - 1 >= 0 and curtain[start - 1, col] and last - start + 1 < period:
                start -= 1
            span = last - start + 1
            if span < 4:
                continue
            offset = (row - last - 1) % span
            out[row, col] = colour[last - offset, col]
            done[row, col] = True
    return out, done


def _column_fill(colour: np.ndarray, valid: np.ndarray, todo: np.ndarray, *, reach: int) -> tuple[np.ndarray, np.ndarray]:
    filled, done = _row_fill(
        np.ascontiguousarray(colour.transpose(1, 0, 2)),
        np.ascontiguousarray(valid.T),
        np.ascontiguousarray(todo.T),
        reach=reach,
    )
    return np.ascontiguousarray(filled.transpose(1, 0, 2)), np.ascontiguousarray(done.T)


def _grain(colour: np.ndarray, sample: np.ndarray, todo: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Fine texture for ``todo``: noise with the surface's own amplitude and colour balance.

    A plane fill without grain reads as plastic: paint has tooth and a floor has fleck, and both
    are what tells the eye the surface is a photograph. The grain is SYNTHESISED rather than
    copied. Copying blocks of the real surface was the first version and it measured worse: in
    a cluttered room the clean samples are islands between objects, every block carried a piece
    of some edge, and on the reference bedroom's floor the blocks tiled into a visible
    checkerboard. Paint tooth and laminate fleck are, to the eye, noise of a given strength and
    tint — which is exactly what can be measured from the clean surface and reproduced without
    a seam: the per-channel spread of the surface's fine detail (median absolute deviation,
    so an edge in the sample cannot inflate it) applied to one luminance noise field, blurred to
    the detail's own scale.
    """
    out = np.zeros_like(colour)
    if not todo.any() or int(sample.sum()) < 64:
        return out
    blurred = cv2.GaussianBlur(colour, (0, 0), sigma)
    detail = colour - blurred
    clean = cv2.erode(sample.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if int(clean.sum()) < 64:
        clean = sample
    picked = detail[clean]
    spread = 1.4826 * np.median(np.abs(picked - np.median(picked, axis=0)), axis=0)
    spread = np.clip(spread, 0.0, _GRAIN_MAX).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(todo.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), _GRAIN_BLUR)
    noise /= max(float(noise.std()), 1e-6)
    out = noise[..., None] * spread[None, None, :]
    out[~todo] = 0.0
    return out


#: Largest grain amplitude, grey levels (one sigma). Paint measures 1-2, laminate 2-4; anything
#: higher in the clean samples is texture the grain should not imitate.
_GRAIN_MAX = 4.0
#: Spatial scale of the synthesised grain, raster pixels: the size of a fleck at 4 mm a pixel.
_GRAIN_BLUR = 0.8


@dataclass(slots=True)
class PlaneFill:
    """What one plane's fill decided, for the bench and the log."""

    plane: int
    method: str
    hole_px: int
    sample_px: int
    lattice: tuple[tuple[float, float], tuple[float, float]] | None = None


def _period(
    texture: np.ndarray, sample: np.ndarray, *, vertical_only: bool = False
) -> tuple[str, np.ndarray, np.ndarray] | None:
    """The repeat of a surface's pattern, from its masked autocorrelation.

    ``("2d", a, b)`` for a tiling — square or brick tiles give two independent periods;
    ``("1d", a, along)`` for a pattern that repeats one way and runs freely the other — boards
    across their width, wallpaper stripes; ``None`` for paint, carpet and anything else with no
    period worth trusting, which the caller fills with shading and grain instead.

    MASKED, because most of a furnished floor is not floor: the autocorrelation of a raster with
    holes in it is dominated by the holes' own shape. Normalising each shift by the energy of the
    pixels that actually overlap at that shift (the standard masked-NCC construction) removes
    that and leaves the grout.
    """
    if int(sample.sum()) < 4000:
        return None
    # Measured at most _PERIOD_SIDE across: a repeat is found from a few dozen periods, not from
    # every pixel, and the correlation's cost is the FFT of the whole raster. Measured on a
    # synthetic run: 40 of 121 s went to full-size transforms before this.
    scale = max(1, int(math.ceil(max(sample.shape) / _PERIOD_SIDE)))
    grey = texture.mean(axis=2) if texture.ndim == 3 else texture
    if scale > 1:
        size = (max(8, sample.shape[1] // scale), max(8, sample.shape[0] // scale))
        grey = cv2.resize(grey.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        sample = cv2.resize(sample.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.99
    height, width = sample.shape
    hp = grey - cv2.GaussianBlur(grey, (0, 0), max(1.5, 6.0 / scale))
    hp = np.where(sample, hp, 0.0).astype(np.float32)
    m = sample.astype(np.float32)
    # Padded by half, not doubled: only shifts up to a third of the raster are searched, and a
    # pad that wide already keeps the circular correlation from wrapping into them.
    fh = cv2.getOptimalDFTSize(height + height // 2 + 2)
    fw = cv2.getOptimalDFTSize(width + width // 2 + 2)

    def spectrum(a: np.ndarray) -> np.ndarray:
        padded = np.zeros((fh, fw), np.float32)
        padded[:height, :width] = a
        return cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)

    def xcorr(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
        product = cv2.mulSpectrums(fa, fb, 0, conjB=True)
        return cv2.idft(product, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

    f_hp, f_m, f_e = spectrum(hp), spectrum(m), spectrum(hp * hp)
    num = xcorr(f_hp, f_hp)
    e1 = xcorr(f_e, f_m)
    e2 = xcorr(f_m, f_e)
    overlap = xcorr(f_m, f_m)
    ncc = num / np.sqrt(np.maximum(e1 * e2, 1e-6))
    # A shift at which only a sliver of the sample overlaps itself is noise, however high.
    ncc = np.where(overlap > 0.3 * float(m.sum()), ncc, 0.0)
    ncc = np.fft.fftshift(ncc).astype(np.float32)
    cy, cx = fh // 2, fw // 2
    span_y, span_x = min(height // 3, cy - 2), min(width // 3, cx - 2)
    window = ncc[cy - span_y : cy + span_y + 1, cx - span_x : cx + span_x + 1].copy()
    yy, xx = np.mgrid[-span_y : span_y + 1, -span_x : span_x + 1]

    # 1. A RIDGE — the correlation staying high along one direction — is a pattern that runs
    #    freely that way: boards along their length, wallpaper stripes along theirs. Found first,
    #    because along a ridge every shift is a local maximum and the 2-D peak search below would
    #    mistake the ridge's first pixel for a period. Measured on the reference bedroom's
    #    laminate: the strongest "peak" was one pixel across and six along the boards (0.47).
    ridge = _ridge(window, span_x, span_y)
    if ridge is not None:
        direction, ridge_score = ridge
        if vertical_only and abs(float(direction[1])) < 0.94:
            # On a wall a horizontal ridge is a moulding, which the row fill already carries;
            # copying it by offset would only repeat whatever edge sits in the samples.
            return None
        across = np.array([-direction[1], direction[0]])
        repeat = _repeat_across(window, span_x, span_y, direction, across)
        if repeat is None:
            # A ridge with no clean repeat across it is grain, not a pattern: copying it by
            # offset repeated the samples' own blemishes in rows (measured: a black bag on the
            # reference bedroom's floor came back as a train of black blocks). Grain does it.
            return None
        return "1d", across * repeat * scale, direction

    window[(np.abs(yy) < 6) & (np.abs(xx) < 6)] = -1.0
    # A wall is paint far more often than it is paper, and a false lattice on paint draws tiles
    # on it: measured on the reference bedroom, plain lilac walls passed the floor's threshold
    # on the repeat of their own shading. A wall's pattern has to be unmistakable.
    threshold = _WALL_PERIOD_NCC if vertical_only else _PERIOD_NCC
    peaks = (window >= cv2.dilate(window, np.ones((7, 7), np.uint8))) & (window > threshold)
    py, px = np.nonzero(peaks)
    if py.size == 0:
        return None

    def parabola(lo: float, mid: float, hi: float) -> float:
        denom = lo - 2.0 * mid + hi
        return 0.0 if abs(denom) < 1e-9 else float(np.clip(0.5 * (lo - hi) / denom, -0.5, 0.5))

    def refined(y: int, x: int) -> np.ndarray:
        """Sub-pixel peak, one parabola per axis: ten periods of drift is what this saves."""
        last_y, last_x = window.shape[0] - 1, window.shape[1] - 1
        oy = parabola(window[y - 1, x], window[y, x], window[y + 1, x]) if 0 < y < last_y else 0.0
        ox = parabola(window[y, x - 1], window[y, x], window[y, x + 1]) if 0 < x < last_x else 0.0
        return np.array([xx[y, x] + ox, yy[y, x] + oy], np.float64)

    vecs = [refined(int(y), int(x)) for y, x in zip(py, px, strict=True)]
    lengths = np.array([float(np.hypot(*v)) for v in vecs])
    order = np.argsort(lengths)
    first = int(order[0])
    a = vecs[first]
    a_len = float(lengths[first])
    if a_len * scale < 6.0:
        return None
    for index in order[1:]:
        b = vecs[int(index)]
        cross = abs(a[0] * b[1] - a[1] * b[0])
        if cross > 0.5 * a_len * float(lengths[int(index)]):
            return "2d", a * scale, b * scale
    return None


def _profile_along(window: np.ndarray, cx: int, cy: int, direction: np.ndarray, radii: np.ndarray, band: int = 0) -> np.ndarray:
    """Mean correlation along a ray from the centre, optionally averaged over a band across it."""
    normal = np.array([-direction[1], direction[0]])
    offsets = np.arange(-band, band + 1, dtype=np.float64)
    xs = np.rint(cx + direction[0] * radii[:, None] + normal[0] * offsets[None, :]).astype(np.int64)
    ys = np.rint(cy + direction[1] * radii[:, None] + normal[1] * offsets[None, :]).astype(np.int64)
    inside = (xs >= 0) & (xs < window.shape[1]) & (ys >= 0) & (ys < window.shape[0])
    values = np.where(inside, window[np.clip(ys, 0, window.shape[0] - 1), np.clip(xs, 0, window.shape[1] - 1)], 0.0)
    counts = inside.sum(axis=1)
    return np.where(counts > 0, values.sum(axis=1) / np.maximum(counts, 1), -1.0)


def _ridge(window: np.ndarray, cx: int, cy: int) -> tuple[np.ndarray, float] | None:
    """The direction along which the correlation stays high, and how high, or ``None``.

    Scored over a range of distances well past the zero-shift peak's own width, in half-degree
    steps; a ridge must hold :data:`_RIDGE_NCC` on average along all of it.
    """
    radii = np.arange(4, max(5, min(cx, cy, 60)), 2)
    if radii.size < 4:
        return None
    scores = []
    for angle in np.radians(np.arange(0.0, 180.0, 1.0)):
        direction = np.array([math.cos(angle), math.sin(angle)])
        scores.append((float(np.mean(_profile_along(window, cx, cy, direction, radii))), float(angle)))
    best, angle = max(scores)
    # Absolute AND relative: a laminate's seams correlate weakly under glare (0.15-0.22 measured
    # on the reference bedroom) but sharply in one direction only, while paint is flat in all.
    typical = float(np.median([score for score, _ in scores]))
    if best < _RIDGE_NCC or best < _RIDGE_CONTRAST * max(typical, 0.01):
        return None
    # Half-degree refinement around the winner: the ridge is narrow (0.22 at 76 deg, 0.17 at 78).
    fine = []
    for delta in np.radians(np.arange(-1.0, 1.01, 0.25)):
        direction = np.array([math.cos(angle + delta), math.sin(angle + delta)])
        fine.append((float(np.mean(_profile_along(window, cx, cy, direction, radii))), angle + float(delta)))
    best, angle = max(fine)
    return np.array([math.cos(angle), math.sin(angle)]), best


def _repeat_across(
    window: np.ndarray, cx: int, cy: int, direction: np.ndarray, across: np.ndarray
) -> float | None:
    """Spacing of the repeat perpendicular to a ridge (board width, stripe pitch), or ``None``.

    The correlation across the ridge, averaged along it over a band so a single row of noise
    cannot fake a seam, must come back up to a clear local maximum after first falling away.
    """
    radii = np.arange(1, max(2, min(cx, cy) - 2))
    profile = _profile_along(window, cx, cy, across, radii, band=6)
    if profile.size < 12:
        return None
    falling = np.nonzero(profile < 0.5 * profile[0])[0]
    if falling.size == 0:
        return None
    start = int(falling[0])
    tail = profile[start:]
    if tail.size < 3:
        return None
    peak = int(np.argmax(tail)) + start
    if profile[peak] < _PERIOD_NCC * 0.6 or peak >= profile.size - 1 or radii[peak] < 6:
        return None
    return float(radii[peak])


#: Step along a ridge between candidate copy offsets, in raster pixels (4 cm at 0.0025 h).
_ALONG_STEP_PX = 16.0

#: A ridge must keep this much correlation on average to count, and this many times the
#: typical direction's: the reference bedroom's laminate measured 0.22 along its boards and a
#: median of 0.03 over all directions; flat paint has no direction standing out at all.
_RIDGE_NCC = 0.12
_RIDGE_CONTRAST = 3.0

#: How far a copy source's pattern may sit from its surface's shading (a fraction): a laminate
#: seam is 0.1-0.2 below, a black object 0.8.
_SOURCE_DEVIATION = 0.3

#: Normalised correlation a shift must reach to count as a period of the pattern. Paint's best
#: non-zero shift is noise well under this; a board or tile repeat sits far over it.
_PERIOD_NCC = 0.35

#: The same for a wall, where a false positive is a tiled pattern on plain paint.
_WALL_PERIOD_NCC = 0.6

#: The period is searched on a raster no larger than this on its long side.
_PERIOD_SIDE = 512

#: How many of the nearest repeats a hole pixel may copy its pattern from directly. Beyond them
#: it takes the pattern's AVERAGE at its own phase (see :func:`_cell_average`): measured, an
#: unbounded search over a fine tiling ran to ten thousand offsets and 38 s per plane.
_COPY_OFFSETS = 48

#: How far a pattern may be copied from, as a fraction of the raster's diagonal. Past this the
#: copy is from a part of the floor under different light and in a different state of wear, and
#: grain does better.
_PERIOD_REACH = 0.6


def _offsets(period: tuple[str, np.ndarray, np.ndarray], reach: float) -> list[tuple[int, int]]:
    """Integer copy offsets the pattern repeats at, nearest first."""
    kind, a, b = period
    a_len = float(np.hypot(*a))
    if kind == "2d":
        b_len = float(np.hypot(*b))
        n_a, n_b = int(reach / a_len) + 1, int(reach / max(b_len, 1.0)) + 1
        steps = [(i * a + j * b) for i in range(-n_a, n_a + 1) for j in range(-n_b, n_b + 1)]
    else:
        # Free along `b` (a unit vector): any step that way is the same pattern. Stepped finely
        # (:data:`_ALONG_STEP_PX`) so the nearest offsets are the ones that carry a board along
        # its own length, which is what a missing piece of floor most needs.
        n_a = int(reach / a_len) + 1
        n_t = int(reach / _ALONG_STEP_PX) + 1
        steps = [(i * a + t * _ALONG_STEP_PX * b) for i in range(-n_a, n_a + 1) for t in range(-n_t, n_t + 1)]
    out: dict[tuple[int, int], float] = {}
    for v in steps:
        length = float(np.hypot(*v))
        if length < 1.0 or length > reach:
            continue
        key = (int(round(float(v[0]))), int(round(float(v[1]))))
        out[key] = min(out.get(key, length), length)
    return [key for key, _ in sorted(out.items(), key=lambda item: item[1])]


def _copy_pattern(
    texture: np.ndarray, sample: np.ndarray, hole: np.ndarray, offsets: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """For every hole pixel, the pattern at the NEAREST offset that lands on clean surface.

    Done by whole-image shifts rather than per pixel: each offset is one masked slice copy over
    the hole's bounding box, and it stops as soon as nothing is left. Measured on a synthetic
    floor: the per-pixel version spent 111 s, this one well under one.
    """
    height, width = hole.shape
    out = np.zeros_like(texture)
    remaining = hole.copy()
    ys, xs = np.nonzero(remaining)
    if ys.size == 0:
        return out, np.zeros_like(hole)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    for count, (dx, dy) in enumerate(offsets):
        ty0, ty1 = max(y0, -dy), min(y1, height - dy)
        tx0, tx1 = max(x0, -dx), min(x1, width - dx)
        if ty0 < ty1 and tx0 < tx1:
            target = remaining[ty0:ty1, tx0:tx1]
            take = target & sample[ty0 + dy : ty1 + dy, tx0 + dx : tx1 + dx]
            if take.any():
                out[ty0:ty1, tx0:tx1][take] = texture[ty0 + dy : ty1 + dy, tx0 + dx : tx1 + dx][take]
                target[take] = False
        if count % 64 == 63:
            ys, xs = np.nonzero(remaining)
            if ys.size == 0:
                break
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return out, hole & ~remaining


def _cell_average(
    texture: np.ndarray, sample: np.ndarray, todo: np.ndarray, period: tuple[str, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The pattern's mean at each pixel's PHASE within the repeat, for ``todo`` pixels.

    Every pixel of a periodic surface has a phase — where it sits inside its tile, or across its
    board — and every pixel at the same phase shows the same thing up to wear and light. So the
    clean surface is binned by phase and averaged, and a hole pixel too far from any clean
    repeat to copy one directly takes the average at its own phase: the grout line in exactly
    the right place, the tile face at its mean colour, and the grain added on top by the caller.
    """
    kind, a, b = period
    height, width = sample.shape
    ys, xs = np.mgrid[0:height, 0:width]
    pts = np.stack([xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)], axis=1)
    if kind == "2d":
        basis = np.column_stack([a, b])
        phase = (pts @ np.linalg.inv(basis).T) % 1.0
        bins = (max(4, int(round(np.hypot(*a)))), max(4, int(round(np.hypot(*b)))))
        index = (
            np.minimum((phase[:, 0] * bins[0]).astype(np.int64), bins[0] - 1) * bins[1]
            + np.minimum((phase[:, 1] * bins[1]).astype(np.int64), bins[1] - 1)
        )
        n_bins = bins[0] * bins[1]
    else:
        a_len = float(np.hypot(*a))
        phase = ((pts @ (a / a_len)) / a_len) % 1.0
        n_bins = max(4, int(round(a_len)))
        index = np.minimum((phase * n_bins).astype(np.int64), n_bins - 1)
    index = index.reshape(height, width)
    channels = texture.shape[2] if texture.ndim == 3 else 1
    flat = texture.reshape(-1, channels)
    chosen = sample.ravel()
    sums = np.zeros((n_bins, channels), np.float64)
    counts = np.zeros(n_bins, np.float64)
    np.add.at(sums, index.ravel()[chosen], flat[chosen])
    np.add.at(counts, index.ravel()[chosen], 1.0)
    mean = sums / np.maximum(counts, 1.0)[:, None]
    out = np.zeros_like(texture)
    known = np.zeros_like(todo)
    rows, cols = np.nonzero(todo)
    picked = index[rows, cols]
    has = counts[picked] >= 3
    values = mean[picked[has]].astype(texture.dtype)
    out[rows[has], cols[has]] = values if texture.ndim == 3 else values[:, 0]
    known[rows[has], cols[has]] = True
    return out, known


def _pattern_fill(
    colour: np.ndarray,
    shade: np.ndarray,
    sample: np.ndarray,
    hole: np.ndarray,
    seed: int,
    grain_gain: float,
    *,
    wall: bool = False,
) -> tuple[np.ndarray, str, tuple[str, np.ndarray, np.ndarray] | None]:
    """Shading times pattern over ``hole``: the pattern copied in phase when it has a period,
    fine grain from the same surface when it does not."""
    out = colour.copy()
    texture = colour / np.maximum(shade, 1.0)
    # A copy SOURCE must look like the surface: a seam or a grout line darkens the surface by a
    # tenth or two, a black bag the segmenter called floor by four fifths. Copying the second
    # repeats it across the hole.
    sample = sample & (np.abs(texture.mean(axis=2) - 1.0) <= _SOURCE_DEVIATION)
    period = _period(texture, sample, vertical_only=wall)
    todo = hole.copy()
    method = "grain"
    if period is not None:
        # The pattern's MEAN at each pixel's phase, not a copy of a neighbour. Copying the
        # nearest clean repeat was the first version: it reproduces a real floor's variation
        # where the samples are clean, and where they are not — the islands of floor between
        # furniture, each with a shadow on one side — it tiled those blemishes into rows of
        # overlapping scales across the reference bedroom's floor. The phase mean draws every
        # seam and grout line exactly where the pattern puts it, and grain does the rest.
        method = f"period-{period[0]}"
        cell, known = _cell_average(texture, sample, todo, period)
        grain = _grain(colour, sample, todo, sigma=1.3, seed=seed)
        use = todo & known
        out[use] = cell[use] * shade[use] + grain[use] * grain_gain
        todo &= ~use
    if todo.any():
        grain = _grain(colour, sample, todo, sigma=1.3, seed=seed)
        out[todo] = shade[todo] + grain[todo] * grain_gain
    return out, method, period


def _meet_the_edge(fill: np.ndarray, colour: np.ndarray, sample: np.ndarray, hole: np.ndarray, pixel: float) -> np.ndarray:
    """Pull a model fill onto the photograph at the hole's edge, fading inward.

    A model of a surface is right on average and wrong by a few levels at any given edge, and a
    few levels along a long straight edge is a visible seam. So the disagreement is measured on a
    thin band of clean surface just outside the hole, spread smoothly over the plane, and added
    back inside the hole with a weight that falls off over :data:`_EDGE_DECAY_H` — the fill
    meets the real wall exactly and becomes the model a hand's breadth in.
    """
    if not hole.any():
        return fill
    band_px = max(2, int(round(0.015 / pixel)))
    near = cv2.dilate(hole.astype(np.uint8), np.ones((2 * band_px + 1, 2 * band_px + 1), np.uint8)) > 0
    band = near & ~hole & sample
    if int(band.sum()) < 16:
        return fill
    difference = np.clip(colour - fill, -_EDGE_CLAMP, _EDGE_CLAMP) * band[..., None]
    spread = _push_pull(difference, band.astype(np.float32))
    distance = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.exp(-distance * pixel / _EDGE_DECAY_H)[..., None]
    out = fill.copy()
    out[hole] = fill[hole] + (spread * weight)[hole]
    return out


#: How far the floor's local shading is carried into a hole before the room's own floor colour
#: takes over (camera heights; 0.15 h is 24 cm, about a bed's contact shadow and a half).
_FLOOR_DECAY_H = 0.15

#: How far into a hole the edge correction reaches (camera heights; 0.05 h is 8 cm).
_EDGE_DECAY_H = 0.05
#: Largest disagreement the edge correction will pull in: a contact shadow left outside the
#: mask is a disagreement too, and it should not be carried into the room.
_EDGE_CLAMP = 18.0


def _period_report(
    period: tuple[str, np.ndarray, np.ndarray] | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if period is None:
        return None
    _, a, b = period
    return (round(float(a[0]), 2), round(float(a[1]), 2)), (round(float(b[0]), 2), round(float(b[1]), 2))


def _wall_samples(rect: Rectified) -> np.ndarray:
    """A wall raster's trustworthy paint samples: visible, clear of every hole's rim, and
    consistent with their row or their wall's smooth field (:func:`_consistent`)."""
    statistics = _rim_free(rect.valid, rect.hole, rect.raster.pixel)
    # Where the rim exclusion leaves a row with nothing at all, fall back to the raw valid set
    # for that row rather than leaving it to the vertical pass.
    rows_empty = ~statistics.any(axis=1) & rect.valid.any(axis=1)
    statistics[rows_empty] = rect.valid[rows_empty]
    return _consistent(rect.colour, statistics, rows=True, floor_tolerance=_WALL_TOLERANCE)


def wall_profile(rects: list[Rectified], samples: list[np.ndarray], floor_row: int) -> np.ndarray | None:
    """The room's walls as one column: paint colour per height, over every wall.

    WHY ONE PROFILE FOR THE ROOM. A wardrobe that fills a wall leaves that wall with almost
    nothing of its own to be rebuilt from — measured on the reference bedroom, 4 % of one wall
    raster was visible paint, all of it in the cornice — and a fill that only listens to its own
    wall turns a door frame's white edge into a bench along the floor. But a room's walls share
    their paint, their cornice and their skirting, and they share them AT THE SAME HEIGHT, which
    is exactly what a rectified wall's row is. So the room's own walls, together, say what a row
    of this room's wall looks like; each wall then only has to say how its light differs.

    THREE BANDS, TRUSTED DIFFERENTLY. The top of a wall (the cornice) is seen all round the room
    and its row medians are believed as they are, white moulding and all. The body of the wall
    is believed only in the PAINT's own hue — the hue the body agrees on — so white furniture the
    segmenter called wall cannot whiten a row (it did, on the second reference capture, from a
    metre down). And the bottom, where the furniture stands and almost nothing is seen, is the
    body's own trend carried down to the floor: paint continues to the floor unless the room
    shows otherwise.

    Rows line up across walls because every wall raster starts at the ceiling line and shares one
    pixel size (:func:`plane_rasters`); ``floor_row`` is the floor line's row.
    """
    if not rects or floor_row < 20:
        return None
    height = max(r.colour.shape[0] for r in rects)
    top = int(_PROFILE_CORNICE * floor_row)
    body_end = int(_PROFILE_BODY_END * floor_row)

    def lab_of(values: np.ndarray) -> np.ndarray:
        pixels = np.clip(values, 0, 255).astype(np.uint8)[None, :, :]
        return cv2.cvtColor(pixels, cv2.COLOR_BGR2Lab)[0].astype(np.float32)

    rows_values: list[list[np.ndarray]] = []
    for row in range(height):
        rows_values.append(
            [r.colour[row][m[row]] for r, m in zip(rects, samples, strict=True) if row < m.shape[0] and m[row].any()]
        )
    body = [np.concatenate(v) for v in rows_values[top:body_end] if v]
    if not body:
        return None
    body_all = np.concatenate(body)
    hue = np.median(lab_of(body_all)[:, 1:], axis=0)

    profile = np.full((height, 3), np.nan, np.float32)
    counts = np.zeros(height, np.int64)
    votes = np.full((height, 3), np.nan, np.float32)
    for row, per_wall in enumerate(rows_values):
        if not per_wall:
            continue
        values = np.concatenate(per_wall)
        if row >= top:
            off = np.sqrt(((lab_of(values)[:, 1:] - hue) ** 2).sum(axis=1))
            values = values[off <= _HUE_TOLERANCE]
        counts[row] = values.shape[0]
        if values.shape[0] >= 24:
            profile[row] = np.median(values, axis=0)
        # ONE VOTE PER WALL, for the rows the paint test cannot speak for (a white skirting is
        # not the paint's hue): a moulding the room has is seen on several walls, white
        # furniture the segmenter called wall on one.
        walls_here = [np.median(v, axis=0) for v in per_wall if v.shape[0] >= 12]
        if len(walls_here) >= 2:
            stack = np.stack(walls_here)
            if float(np.abs(stack - np.median(stack, axis=0)).max()) <= 2.0 * _WALL_TOLERANCE:
                votes[row] = np.median(stack, axis=0)
    # A row only speaks for the room when a fair share of the room's walls is visible in it.
    # The bottom of every wall is where the furniture stands, so its rows are the thinnest:
    # measured on the reference bedroom, 80 samples a row across four walls against 2000 under
    # the cornice, and those 80 were the white of a mislabelled floor and the grey of contact
    # shadows, which drew a white bench along every wall.
    reference = float(np.percentile(counts[counts > 0], 75)) if (counts > 0).any() else 0.0
    reliable = counts >= max(24.0, _PROFILE_MIN_SHARE * reference)
    profile[~reliable] = np.nan

    # The body's trend, robustly, carried to the floor.
    body_rows = np.nonzero(reliable[top:body_end])[0] + top
    if body_rows.size >= 8:
        design = np.stack([np.ones(body_rows.size), body_rows / float(floor_row)], axis=1)
        trend = np.zeros((2, 3))
        for channel in range(3):
            y = profile[body_rows, channel].astype(np.float64)
            weights = np.ones_like(y)
            for _ in range(3):
                w = np.sqrt(weights)
                coef, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
                residual = np.abs(y - design @ coef)
                weights = 1.0 / np.maximum(1.0, residual / max(2.0, 1.4826 * float(np.median(residual))))
            trend[:, channel] = coef
        below = np.arange(body_end, height)
        anchor = (np.array([1.0, body_end / float(floor_row)]) @ trend).astype(np.float32)
        carried = (np.stack([np.ones(below.size), below / float(floor_row)], axis=1) @ trend).astype(np.float32)
        # The trend's slope, not its licence: a few per cent either way over the furniture zone.
        profile[below] = np.clip(carried, anchor * (1.0 - _TREND_LIMIT), anchor * (1.0 + _TREND_LIMIT))
        # Where the walls agree on something else down there (a skirting seen on two walls or
        # more), that is the room's, and it is drawn — but only a skirting's height above the
        # floor line. Higher up, "the walls agree on white" was the white furniture the segmenter
        # called wall standing against two of them, and it drew a white bench round the room.
        skirting = below[below >= floor_row - int(round(_SKIRTING_ROWS_SHARE * floor_row))]
        agreed = skirting[np.isfinite(votes[skirting, 0])]
        profile[agreed] = votes[agreed]

    known = np.nonzero(np.isfinite(profile[:, 0]))[0]
    if known.size < 8:
        return None
    for channel in range(3):
        profile[:, channel] = np.interp(np.arange(height), known, profile[known, channel])
    return cv2.medianBlur(np.ascontiguousarray(profile[:, None, :]), 5)[:, 0, :] if height >= 5 else profile


#: How far above the floor line a band of the walls' own colour may be drawn as a skirting, as
#: a share of the wall's height: 5 % is 13 cm on a 2.6 m wall, a tall skirting board.
_SKIRTING_ROWS_SHARE = 0.05

#: Top share of a wall's height read as cornice (believed in any colour), and where the body
#: ends and the furniture zone begins (carried down from the body's trend), both as shares of
#: the floor line's row. A 2.6 m wall: the top 40 cm, and everything below 1.3 m — measured on
#: the reference captures, white furniture the segmenter called wall reached 1.2 m on three
#: walls of four, and a body that reached lower turned the bottom of every wall white.
_PROFILE_CORNICE = 0.15
_PROFILE_BODY_END = 0.50

#: How far the carried trend may move the paint's colour over the furniture zone (a share).
_TREND_LIMIT = 0.06


def _fill_wall(
    rect: Rectified, seed: int, statistics: np.ndarray, profile: np.ndarray | None
) -> tuple[np.ndarray, PlaneFill]:
    """The room's wall profile, lit the way this wall is lit, with this wall's own mouldings and
    pattern on top; curtains carried down their own columns.

    WHY THE PROFILE. See :func:`wall_profile`: a row of wall looks like the same row of every
    other wall in the room, and that is what lets a wall the camera barely saw be rebuilt at all.

    WHY THE LIGHT IS THE WALL'S OWN. Each wall faces the window at its own angle, so this wall's
    samples divided by the profile — its SHADING — is smoothed into a field and carried across
    the hole: the paint darkens toward a corner here exactly as it does around the hole.

    WHY ROWS FOR WHAT IS LEFT. Square-on, everything horizontal about a wall is a row: a dado
    rail on this wall only, the cornice where it differs. What the profile does not already say
    is carried across the hole row by row from the same row either side — never in 2-D, which
    rounds a moulding off at the corners of the hole.

    WHY CURTAINS ARE DIFFERENT. A curtain is held, not removed, and it hangs in front of the
    wall. Where a chair stood in front of one, the honest continuation is MORE CURTAIN, down to
    the floor, not wall paint — a row fill would carry the wall's colour across the curtain's
    hem and leave a bite out of it. So a hole column with curtain directly above it is filled
    downward from that curtain first; the wall fill takes over everywhere else.
    """
    colour, hole = rect.colour, rect.hole
    pixel = rect.raster.pixel
    height = colour.shape[0]
    reach = max(6, int(round(0.03 / pixel)))

    out = colour.copy()
    todo = hole.copy()

    curtain = (rect.layer == 1) & rect.held
    if curtain.any() and todo.any():
        # Columns of the hole that have curtain somewhere above them, carried downward.
        curtain_out, curtain_done = _curtain_down(colour, curtain, todo)
        out[curtain_done] = curtain_out[curtain_done]
        todo &= ~curtain_done

    if profile is not None:
        rows = np.broadcast_to(profile[:height][:, None, :], colour.shape).astype(np.float32)
    else:
        rows = _push_pull(colour, statistics.astype(np.float32))

    # This wall's light, as a function of position ALONG the wall only: its samples over the
    # profile, the median down each column, smoothed sideways and carried across columns with
    # nothing in them. NOT a 2-D field. Furniture covers the bottom metre of every wall, so the
    # samples a 2-D field would extrapolate downward from are the dim band just above the
    # furniture — measured on the reference bedroom, a grey band under every emptied wall with
    # a hard line where the furniture's top had been. How light falls with height is the
    # profile's business, and the profile is read from every wall at once.
    ratio = colour / np.maximum(rows, 1.0)
    width = colour.shape[1]
    column_light = np.full((width, 3), np.nan, np.float32)
    # The BODY of the wall only — between the cornice and the skirting. A wall seen only in
    # its cornice and a door frame's edge says nothing about how its paint is lit, and measured
    # on the reference bedroom's hidden wall, listening to it whitened the whole wall.
    floor_row = int(round(min(height, (rect.raster.height - _SLACK_H / pixel))))
    body = np.zeros(height, bool)
    body[int(_PROFILE_CORNICE * floor_row) : int(_PROFILE_BODY_END * floor_row)] = True
    in_body = statistics & body[:, None]
    # And only what is lit like this wall's paint. The night-black glass of a window the
    # segmenter called wall passes every row test — the whole row is glass — and set its columns'
    # light to a fifth: measured on the third reference capture, a grey slab from the window's
    # sill to the floor. Judged against the wall's OWN median, not the profile, so a wall painted
    # darker than the room is still lit as itself.
    lum_ratio = ratio.mean(axis=2)
    if in_body.any():
        typical = float(np.median(lum_ratio[in_body]))
        in_body &= (lum_ratio > typical * _LIGHT_BAND[0]) & (lum_ratio < typical * _LIGHT_BAND[1])
    counts = in_body.sum(axis=0)
    for col in np.nonzero(counts >= 8)[0]:
        column_light[col] = np.median(ratio[in_body[:, col], col], axis=0)
    known = np.nonzero(np.isfinite(column_light[:, 0]))[0]
    coverage = known.size / float(max(width, 1))
    if known.size >= 2 and coverage >= _MIN_LIGHT_COVERAGE:
        for channel in range(3):
            column_light[:, channel] = np.interp(np.arange(width), known, column_light[known, channel])
        column_light = cv2.GaussianBlur(
            np.ascontiguousarray(column_light[None, :, :]), (0, 1), sigmaX=max(1.0, 0.10 / pixel), sigmaY=0.0
        )[0]
    else:
        column_light = np.ones((width, 3), np.float32)
    base = rows * np.clip(column_light, 0.7, 1.4)[None, :, :]

    # What the profile and the light do not explain, row by row from either side of the hole.
    across = max(1.0, 0.06 / pixel)
    sample_f = statistics.astype(np.float32)
    residual = (colour - base) * sample_f[..., None]
    weight = cv2.GaussianBlur(sample_f, (0, 1), sigmaX=across, sigmaY=0.0)
    residual = cv2.GaussianBlur(residual, (0, 1), sigmaX=across, sigmaY=0.0) / np.maximum(weight, 1e-4)[..., None]
    residual_ok = statistics & (weight > 0.3)
    carried, done = _row_fill(residual, residual_ok, todo, reach=reach)
    if coverage < _MIN_LIGHT_COVERAGE:
        # Too little of this wall to say anything of its own: the room's profile stands.
        done[:] = False
    shade = base.copy()
    shade[done] += np.clip(carried[done], -_WALL_TOLERANCE, _WALL_TOLERANCE)
    shade = np.where(residual_ok[..., None], base + residual, shade)

    shade = _meet_the_edge(shade, colour, statistics, todo, pixel)
    patterned, method, period = _pattern_fill(
        colour, shade, statistics, todo, seed, grain_gain=0.8 if coverage >= _MIN_LIGHT_COVERAGE else 0.3, wall=True
    )
    out[todo] = patterned[todo]
    return out, PlaneFill(
        rect.raster.plane,
        f"{'profile' if profile is not None else 'own'}+{method}",
        int(hole.sum()),
        int(statistics.sum()),
        _period_report(period),
    )


def _smooth_surface(colour: np.ndarray, sample: np.ndarray, seed: int) -> np.ndarray:
    """The surface's light as a quadratic over the plane, fitted robustly to its clean samples.

    What a floor deep inside a large hole should look like is the room's floor under the room's
    light — brighter under the lamp, darker toward the walls — and a quadratic is the smallest
    model that says that and cannot say anything else: it extrapolates smoothly into a bed-sized
    hole, where the rim-carried shading turned to the shadow at the hole's edge and the room's
    flat median read a third too dark beside the bright floor around the viewer. Fitted with
    iteratively reweighted least squares, so a missed bag on the floor does not tilt it.
    """
    height, width = sample.shape
    ys, xs = np.nonzero(sample)
    if ys.size < 50:
        return np.broadcast_to(np.median(colour[sample], axis=0), colour.shape).astype(np.float32) if ys.size else colour.copy()
    rng = np.random.default_rng(seed)
    pick = rng.choice(ys.size, size=min(ys.size, 20000), replace=False)
    u = xs[pick] / float(width) - 0.5
    v = ys[pick] / float(height) - 0.5
    design = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)
    values = colour[ys[pick], xs[pick]].astype(np.float64)
    weights = np.ones(u.size)
    for _ in range(4):
        w = np.sqrt(weights)[:, None]
        coef, *_ = np.linalg.lstsq(design * w, values * w, rcond=None)
        residual = np.abs(values - design @ coef).max(axis=1)
        scale = max(1.4826 * float(np.median(residual)), 2.0)
        weights = 1.0 / np.maximum(1.0, residual / (2.0 * scale))
    gu = (np.arange(width) / float(width) - 0.5)[None, :]
    gv = (np.arange(height) / float(height) - 0.5)[:, None]
    field = np.zeros(colour.shape, np.float32)
    terms = (np.ones((height, width)), gu + 0 * gv, gv + 0 * gu, gu * gu + 0 * gv, gu * gv, gv * gv + 0 * gu)
    for index, term in enumerate(terms):
        field += term[..., None].astype(np.float32) * coef[index][None, None, :].astype(np.float32)
    low, high = np.percentile(values, 5, axis=0), np.percentile(values, 95, axis=0)
    return np.clip(field, low, high).astype(np.float32)


def _fill_floor(rect: Rectified, seed: int) -> tuple[np.ndarray, PlaneFill]:
    """Illumination times the floor's own pattern, the pattern copied in phase.

    Two fields, because a floor is two things. Its SHADING — the pool of light under a window,
    the fall-off toward a corner — is low-frequency and is extrapolated smoothly from the floor
    that stands clear of every hole (see :data:`_RIM_H`: the rim is shadow, not floor). Its
    PATTERN — grout, boards, fleck — is high-frequency and is copied from the floor itself, a
    whole number of periods away when the floor is tiled, so the grout lines run straight
    through the hole and meet the real ones on the far side in phase.
    """
    colour, valid, hole = rect.colour, rect.valid, rect.hole
    pixel = rect.raster.pixel
    statistics = _rim_free(valid, hole, pixel)
    if statistics.sum() < 64:
        statistics = valid
    statistics = _consistent(colour, statistics, rows=False, floor_tolerance=_FLOOR_TOLERANCE)

    # Shading: a heavily smoothed field through the clean floor. 8 cm is more than a board's
    # width and short of a pool of light, so the pattern goes and the light stays.
    sigma = max(2.0, 0.08 / pixel)
    sample_f = statistics.astype(np.float32)
    weight = cv2.GaussianBlur(sample_f, (0, 0), sigma)
    lit = cv2.GaussianBlur(colour * sample_f[..., None], (0, 0), sigma)
    shade_local = lit / np.maximum(weight, 1e-4)[..., None]
    trusted = weight > 0.2
    shade = _push_pull(np.where(trusted[..., None], shade_local, 0.0), trusted.astype(np.float32))
    # DEEP IN A LARGE HOLE THE FLOOR IS THE ROOM'S FLOOR, NOT ITS RIM'S. The pyramid carries the
    # nearest clean floor inward with no limit, and the nearest clean floor to a removed bed is
    # the strip in its shadow: measured on the reference bedroom, the floor under the wardrobe and
    # the heap beside it came back a flat grey a third darker than the room's. So the extrapolated
    # shading decays, over :data:`_FLOOR_DECAY_H`, toward the floor's own robust median.
    if hole.any() and statistics.any():
        room_floor = _smooth_surface(colour, statistics, seed)
        reach = cv2.distanceTransform((~trusted).astype(np.uint8), cv2.DIST_L2, 5) * pixel
        keep = np.exp(-reach / _FLOOR_DECAY_H)[..., None]
        shade = shade * keep + room_floor * (1.0 - keep)
    shade = np.where(trusted[..., None], shade_local, shade)
    shade = _meet_the_edge(shade, colour, statistics, hole, pixel)

    out, method, period = _pattern_fill(colour, shade, statistics, hole, seed, grain_gain=1.0)
    return out, PlaneFill(rect.raster.plane, method, int(hole.sum()), int(statistics.sum()), _period_report(period))


def _fill_ceiling(rect: Rectified, seed: int) -> tuple[np.ndarray, PlaneFill]:
    """A ceiling is paint under light: a smooth field plus its own grain."""
    statistics = _rim_free(rect.valid, rect.hole, rect.raster.pixel)
    if statistics.sum() < 64:
        statistics = rect.valid
    out = rect.colour.copy()
    smooth = _push_pull(rect.colour, statistics.astype(np.float32))
    grain = _grain(rect.colour, statistics, rect.hole, sigma=1.2, seed=seed)
    out[rect.hole] = smooth[rect.hole] + grain[rect.hole] * 0.8
    return out, PlaneFill(rect.raster.plane, "smooth", int(rect.hole.sum()), int(statistics.sum()))


#: Grey levels a wall sample may sit from its row's median, or from the wall's smooth field,
#: and still count as paint (see :func:`_consistent`). 16 keeps a painted wall's shading and a
#: skirting's own rows, and drops the white edge of a desk top on a lilac wall (40+).
_WALL_TOLERANCE = 16.0
#: The same for a floor, looser because a floor's own pattern is strong: grout under glare and
#: board-to-board variation reach 20; a black bag on white tile is 150.
_FLOOR_TOLERANCE = 30.0

#: Chroma (OpenCV Lab units, a* b* about 128) a sample may sit from its surface's median hue.
#: Lilac paint in shadow and under the lamp stays within 3-4; a beige rug is 15-25 off white.
_HUE_TOLERANCE = 8.0

#: Share of a wall's columns that must show paint in the wall's body before the wall's own
#: light and its own mouldings are listened to; below it the room's profile is drawn as it is.
_MIN_LIGHT_COVERAGE = 0.25

#: The band, as multiples of a wall's median brightness over the profile, its body samples must
#: sit in to set its column light. Wide on purpose: a wall beside the window is lit twice as
#: brightly at one end as at the other, and that gradient is what the light is for. Measured on
#: the reference captures against no band: the third capture's dark-smear share 0.178 -> 0.088,
#: mean colour distance 14.0 -> 13.2; the first unchanged, the second 11.0 -> 10.9.
_LIGHT_BAND = (0.6, 1.6)

#: Share of a well-seen row's sample count a wall row needs before its own median is believed
#: (:func:`wall_profile`). 0.2: the reference bedroom's junk rows sat at 4 %, its cornice and
#: paint rows at 60-100 %.
_PROFILE_MIN_SHARE = 0.2

#: A plane needs this much of itself visible before it is filled from itself. Below it the
#: plane is a sliver the camera barely saw, and the caller's fallback does better.
_MIN_SAMPLE_PX = 400


def fill_planes(
    image: np.ndarray,
    camera: Camera,
    box: RoomBox,
    hole: np.ndarray,
    surface: np.ndarray,
    *,
    held: np.ndarray | None = None,
    layer: np.ndarray | None = None,
    floor_shift: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
    pixel: float = _PIXEL_H,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[PlaneFill]]:
    """Rebuild ``hole`` from the room's planes. Returns ``(image, filled, report)``.

    ``filled`` marks the hole pixels a plane answered for; the rest — a sliver of a plane the
    camera barely saw, a ray that leaves the box through a gap in the fit — are the caller's to
    finish. Only hole pixels are written: everything else comes back byte-identical.
    """
    height, width = image.shape[:2]
    out = image.copy()
    hole_bool = hole > 127
    filled = np.zeros((height, width), bool)
    report: list[PlaneFill] = []
    if not hole_bool.any():
        return out, filled, report

    ys, xs = np.nonzero(hole_bool)
    rays = camera.rays(xs.astype(np.float64), ys.astype(np.float64))
    hits = trace(box, rays)
    rasters = plane_rasters(box, pixel)
    if floor_shift is not None and isinstance(camera, EquirectCamera):
        # The floor line where the photograph puts it (see `floor_line_shift`): pixels between
        # the box's line and the photographed one change plane, and land in the rasters' slack.
        line = floor_rows(box, width, height) + floor_shift
        below = ys.astype(np.float64) >= line[xs]
        to_floor = below & (hits.plane >= 0) & (rays[:, 1] > 1e-6)
        to_wall = ~below & (hits.plane == FLOOR) & (hits.wall >= 0)
        hits.plane[to_floor] = FLOOR
        hits.point[to_floor] = rays[to_floor] / rays[to_floor, 1:2]
        hits.plane[to_wall] = hits.wall[to_wall]
        hits.point[to_wall] = hits.wall_point[to_wall]

    # Every wall is rectified first, whether or not it has a hole: the room's wall profile is
    # read from all of them (see `wall_profile`).
    rectified: dict[int, Rectified] = {}
    for plane_id, raster in rasters.items():
        rect = rectify(
            image, camera, box, raster, hole=hole, surface=surface, held=held, layer=layer, exclude=exclude
        )
        if rect is not None:
            rectified[plane_id] = rect
    walls = [pid for pid in rectified if pid >= 0]
    wall_samples = {pid: _wall_samples(rectified[pid]) for pid in walls}
    floor_row = int(round(box.ceiling / rasters[walls[0]].pixel)) if walls else 0
    profile = wall_profile([rectified[pid] for pid in walls], [wall_samples[pid] for pid in walls], floor_row)

    for plane_id, raster in rasters.items():
        on_plane = hits.plane == plane_id
        if not on_plane.any() or plane_id not in rectified:
            continue
        rect = rectified[plane_id]
        seed = 7919 * (plane_id + 3)
        if plane_id == FLOOR:
            if int(rect.valid.sum()) < _MIN_SAMPLE_PX:
                continue
            painted, info = _fill_floor(rect, seed)
        elif plane_id == CEILING:
            if int(rect.valid.sum()) < _MIN_SAMPLE_PX:
                continue
            painted, info = _fill_ceiling(rect, seed)
        else:
            if int(rect.valid.sum()) < _MIN_SAMPLE_PX and profile is None:
                continue
            painted, info = _fill_wall(rect, seed, wall_samples[plane_id], profile)
        report.append(info)
        if progress is not None:
            progress(f"plane {plane_id}: {info.method}")

        col, row = raster.coords(hits.point[on_plane])
        sampled = _sample_points(
            np.clip(painted, 0, 255).astype(np.float32), col, row, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE
        )
        # Only where the raster genuinely produced something: a hole pixel whose raster
        # neighbourhood was never filled (outside the plane's visible part) is left alone.
        known = _sample_points(
            (rect.hole | rect.valid | rect.held).astype(np.float32),
            col,
            row,
            cv2.INTER_LINEAR,
            cv2.BORDER_CONSTANT,
        )
        good = known > 0.5
        idx_y, idx_x = ys[on_plane][good], xs[on_plane][good]
        out[idx_y, idx_x] = np.clip(np.rint(sampled[good]), 0, 255).astype(np.uint8)
        filled[idx_y, idx_x] = True

    log.info(
        "room_planes.filled",
        planes=[(r.plane, r.method) for r in report],
        filled=round(float(filled.sum()) / max(1, int(hole_bool.sum())), 4),
    )
    return out, filled, report
