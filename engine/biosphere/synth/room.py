"""A closed-form virtual room: the only "scene" the synthetic harness ever needs.

Why an analytic tracer instead of a mesh renderer? Because the harness has to answer
one question honestly — *how much quality did the stitcher lose?* — and that question
only has a meaning if the reference panorama and the simulated video frames are two
samplings of the **same continuous function**. A mesh pipeline would introduce its own
rasterisation, its own filtering and its own colour management between the two, and the
degradation we measured would be partly ours. Here the scene is a handful of
axis-aligned boxes and a set of pure ``f(u, v)`` surface functions, so a ray fired from
anywhere in the room returns the exact same radiance no matter which renderer asked.

Two design choices carry most of the weight:

*Every surface function is band-limited by the caller's pixel footprint.* Each ray
carries the angular size of the pixel it belongs to; the tracer converts that into a
surface-space filter width and every stripe, seam, plank and mullion is evaluated as a
**box-filtered** feature rather than point-sampled. Without this a 3 mm floor seam
aliases violently, and the equirect reference and a perspective frame would alias
*differently* — which would show up as tens of dB of fake "stitching loss".

*The tracer is re-entrant from any origin.* Parallax is then free and exact: the capture
simulator simply traces from the phone's true optical centre instead of the observer's
spine, and the resulting disagreement with the reference panorama is real physics, not a
warp applied after the fact.

Conventions follow :mod:`biosphere.pano.types`: world **+Y points down**, so the floor
plane is ``y = 0`` and the ceiling is ``y = -height``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

__all__ = [
    "Box",
    "WallRect",
    "Light",
    "StyleParams",
    "Room",
    "STYLES",
    "FACE_XLO",
    "FACE_XHI",
    "FACE_CEILING",
    "FACE_FLOOR",
    "FACE_ZLO",
    "FACE_ZHI",
    "INNER_FACE_BASE",
    "WALL_FACES",
]

# --------------------------------------------------------------------------------------
# Face identifiers
# --------------------------------------------------------------------------------------

FACE_XLO: Final[int] = 0  # wall at x = lo.x, inward normal +X
FACE_XHI: Final[int] = 1  # wall at x = hi.x, inward normal -X
FACE_CEILING: Final[int] = 2  # y = lo.y (remember: +Y is *down*), inward normal +Y
FACE_FLOOR: Final[int] = 3  # y = hi.y, inward normal -Y
FACE_ZLO: Final[int] = 4  # wall at z = lo.z, inward normal +Z
FACE_ZHI: Final[int] = 5  # wall at z = hi.z, inward normal -Z
INNER_FACE_BASE: Final[int] = 8  # furniture face ids start here: 8 + 6*box + 2*axis + side
WALL_FACES: Final[tuple[int, ...]] = (FACE_XLO, FACE_XHI, FACE_ZLO, FACE_ZHI)

_MISS: Final[int] = -1

# Inward normals of the enclosing box, indexed by face id.
_OUTER_NORMALS: Final[np.ndarray] = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float32,
)

# Outward normals of a furniture box, indexed by 2*axis + side (side 0 = the `lo` face).
_INNER_NORMALS: Final[np.ndarray] = np.array(
    [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

_EPS: Final[float] = 1e-9
_MIN_FOOTPRINT: Final[float] = 2.0e-5  # metres; keeps the box filters unconditional
_GRAZE_CLAMP: Final[float] = 0.18  # stops the footprint exploding at grazing incidence

# Width of the reconstruction filter, in pixels. A one-pixel box leaves a lot of energy
# above Nyquist — enough that an equirect render and a perspective render of the same wall
# alias *differently* and disagree by several dB after resampling. Offline renderers reach
# for a 2-4 pixel Mitchell or Gaussian for the same reason; 1.35 is the point where the
# harness's own reprojection error stops falling and the image is still crisp.
_FILTER_WIDTH: Final[float] = 1.35


# ======================================================================================
# Deterministic hash noise
# ======================================================================================

_H1: Final[np.uint64] = np.uint64(0x9E3779B97F4A7C15)
_H2: Final[np.uint64] = np.uint64(0xBF58476D1CE4E5B9)
_H3: Final[np.uint64] = np.uint64(0x94D049BB133111EB)
_SH30: Final[np.uint64] = np.uint64(30)
_SH27: Final[np.uint64] = np.uint64(27)
_SH31: Final[np.uint64] = np.uint64(31)
_SH40: Final[np.uint64] = np.uint64(40)
_INT_BIAS: Final[int] = 1 << 24


def _hash01(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """SplitMix-style integer hash of a 2D lattice cell -> float32 in [0, 1).

    Integer arithmetic rather than the usual ``fract(sin(...))`` trick, because the
    latter loses all its entropy in float32 once the arguments grow past a few thousand,
    and the harness must stay bit-reproducible across resolutions.
    """
    # `view` rather than `astype`: the bias makes every value positive, so the int64 bits
    # are already the uint64 we want and the copy is pure overhead in the hottest loop.
    a = (ix + _INT_BIAS).view(np.uint64)
    b = (iy + _INT_BIAS).view(np.uint64)
    # Fold the seed in Python integers first: numpy would warn about the scalar overflow
    # even though wrap-around is exactly what a hash wants.
    s = np.uint64((int(seed) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF)
    h = a * _H1
    h += b * _H2
    h += s
    h ^= h >> _SH30
    h *= _H2
    h ^= h >> _SH27
    h *= _H3
    h ^= h >> _SH31
    return (h >> _SH40).astype(np.float32) * np.float32(1.0 / 16777216.0)


def _value_noise(u: np.ndarray, v: np.ndarray, freq: float, seed: int) -> np.ndarray:
    """Smooth value noise on a lattice of period ``1 / freq`` metres, centred on zero."""
    x = u * np.float32(freq)
    y = v * np.float32(freq)
    xi = np.floor(x)
    yi = np.floor(y)
    xf = x - xi
    yf = y - yi
    sx = xf * xf * (np.float32(3.0) - np.float32(2.0) * xf)
    sy = yf * yf * (np.float32(3.0) - np.float32(2.0) * yf)
    ix = xi.astype(np.int64)
    iy = yi.astype(np.int64)
    n00 = _hash01(ix, iy, seed)
    n10 = _hash01(ix + 1, iy, seed)
    n01 = _hash01(ix, iy + 1, seed)
    n11 = _hash01(ix + 1, iy + 1, seed)
    a = n00 + (n10 - n00) * sx
    b = n01 + (n11 - n01) * sx
    return (a + (b - a) * sy) - np.float32(0.5)


def _fbm(
    u: np.ndarray,
    v: np.ndarray,
    freq: float,
    octaves: int,
    fp: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fractal value noise, with each octave rolled off once the pixel outruns it.

    The roll-off is what makes the same wall look identical at 1024 px and at 8192 px
    apart from detail the coarser canvas physically cannot represent.
    """
    total = np.zeros_like(u)
    # The normalisation counts *every* octave, whether or not the loop below skips it, so
    # that the early-out is a pure speed-up and never changes the surface it produces.
    norm = np.float32(sum(0.5**o for o in range(octaves)))
    amp = 1.0
    f = float(freq)
    fp_min = float(fp.min()) if fp.size else 0.0
    for o in range(octaves):
        # Every later octave is finer than this one, so once the *sharpest* pixel in the
        # batch cannot resolve this frequency, none of them can resolve any of the rest.
        if amp / (1.0 + (2.5 * fp_min * f) ** 2) < 0.002 * float(norm):
            break
        fade = np.float32(1.0) / (np.float32(1.0) + (np.float32(2.5) * fp * np.float32(f)) ** 2)
        total += np.float32(amp) * fade * _value_noise(u, v, f, seed + 977 * o)
        amp *= 0.5
        f *= 2.03
    return total / norm


# ======================================================================================
# Box-filtered feature primitives
# ======================================================================================


def _band(dist: np.ndarray, half_width: float, fp: np.ndarray) -> np.ndarray:
    """Coverage of a stripe of half-width ``half_width`` by a pixel of half-width ``fp``.

    Exact box filter: the fraction of ``[d - fp, d + fp]`` that lies inside
    ``[-half_width, half_width]``. Reduces to a hard edge as ``fp -> 0`` and to the
    stripe's duty cycle as ``fp`` grows, which is precisely what a mip-map would do.
    """
    lo = np.maximum(dist - fp, np.float32(-half_width))
    hi = np.minimum(dist + fp, np.float32(half_width))
    return np.clip((hi - lo) / (np.float32(2.0) * fp), 0.0, 1.0)


def _periodic_band(
    x: np.ndarray, period: float, half_width: float, fp: np.ndarray
) -> np.ndarray:
    """Box-filtered coverage of an infinite stripe grid of the given period.

    :func:`_band` only integrates the *nearest* stripe, so once the pixel is wider than
    the period the answer has to fall back to the duty cycle or the surface would fade to
    the base colour instead of to its true mean.
    """
    p = np.float32(period)
    t = x / p
    d = np.abs((t - np.floor(t)) - np.float32(0.5)) * p
    fp_eff = np.minimum(fp, np.float32(0.5 * period))
    cov = _band(d, half_width, np.maximum(fp_eff, np.float32(_MIN_FOOTPRINT)))
    duty = np.float32(min(1.0, 2.0 * half_width / period))
    mix = np.clip((fp - np.float32(0.5 * period)) / np.float32(0.5 * period), 0.0, 1.0)
    return cov * (np.float32(1.0) - mix) + duty * mix


def _rect_cov(
    du: np.ndarray, dv: np.ndarray, hw: float, hh: float, fp: np.ndarray
) -> np.ndarray:
    """Box-filtered coverage of an axis-aligned rectangle (separable in u and v)."""
    return _band(np.abs(du), hw, fp) * _band(np.abs(dv), hh, fp)


def _outline_cov(
    du: np.ndarray, dv: np.ndarray, hw: float, hh: float, t: float, fp: np.ndarray
) -> np.ndarray:
    """Coverage of a rectangular *outline* of thickness ``t`` (outer minus inner)."""
    outer = _rect_cov(du, dv, hw, hh, fp)
    inner = _rect_cov(du, dv, max(hw - t, 1e-4), max(hh - t, 1e-4), fp)
    return np.clip(outer - inner, 0.0, 1.0)


def _disc_cov(r: np.ndarray, radius: float, fp: np.ndarray) -> np.ndarray:
    """Box-filtered coverage of a disc of the given radius (1D approximation)."""
    return _band(r, radius, fp)


def _cell_lerp(
    x: np.ndarray, cell: float, fp: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Box-filter a piecewise-constant cell field: ``(index, neighbour, neighbour_weight)``.

    A hashed-per-cell colour (a plank's tone, a block in a painting, a rooftop height) is
    a step function, and point-sampling one aliases as violently as any hard edge — worse,
    because the steps are high contrast and land at arbitrary sub-pixel positions. Here the
    pixel box is intersected with the nearest cell boundary and the two cells are blended
    by the resulting area, which is the exact box filter whenever the pixel does not span
    more than two cells (and degrades to a 50/50 average when it does).
    """
    t = x / np.float32(cell)
    i = np.floor(t)
    f = t - i
    left = f < np.float32(0.5)
    nb = np.where(left, i - np.float32(1.0), i + np.float32(1.0))
    d = np.where(left, f, np.float32(1.0) - f) * np.float32(cell)
    w = np.clip((fp - d) / (np.float32(2.0) * fp), 0.0, 0.5)
    return i.astype(np.int64), nb.astype(np.int64), w.astype(np.float32)


def _cell_value(
    u: np.ndarray, v: np.ndarray, cell_u: float, cell_v: float, fp: np.ndarray, seed: int
) -> np.ndarray:
    """Box-filtered hashed scalar in [0, 1) on a 2D cell grid."""
    iu, nu, wu = _cell_lerp(u, cell_u, fp)
    iv, nv, wv = _cell_lerp(v, cell_v, fp)
    a = _hash01(iu, iv, seed)
    b = _hash01(nu, iv, seed)
    c = _hash01(iu, nv, seed)
    e = _hash01(nu, nv, seed)
    top = a + (b - a) * wu
    bot = c + (e - c) * wu
    return top + (bot - top) * wv


def _mix(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear blend ``a -> b`` with a per-element weight broadcast over the colour axis."""
    return a + (b - a) * t[:, None]


def _rgb(r: float, g: float, b: float, n: int) -> np.ndarray:
    out = np.empty((n, 3), dtype=np.float32)
    out[:, 0] = r
    out[:, 1] = g
    out[:, 2] = b
    return out


# ======================================================================================
# Scene description
# ======================================================================================


@dataclass(slots=True)
class Box:
    """One piece of axis-aligned furniture, traced from the outside."""

    lo: np.ndarray
    hi: np.ndarray
    material: str = "matte"  # matte | wood | fabric
    tint: tuple[float, float, float] = (0.55, 0.52, 0.50)
    seed: int = 0

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, dtype=np.float32).reshape(3)
        self.hi = np.asarray(self.hi, dtype=np.float32).reshape(3)
        if np.any(self.hi <= self.lo):
            raise ValueError(f"degenerate furniture box: lo={self.lo} hi={self.hi}")

    def contains(self, p: np.ndarray, margin: float = 0.0) -> bool:
        p = np.asarray(p, dtype=np.float32).reshape(3)
        return bool(np.all(p > self.lo - margin) and np.all(p < self.hi + margin))


@dataclass(slots=True)
class WallRect:
    """A rectangle painted onto one of the four vertical walls.

    ``u`` runs left-to-right as seen from inside the room and ``v`` is height above the
    floor, both in metres, matching :meth:`Room.surface_uv`.
    """

    face: int
    u0: float
    v0: float
    u1: float
    v1: float
    kind: str = "picture"  # window | picture | door
    seed: int = 0

    @property
    def centre(self) -> tuple[float, float]:
        return (0.5 * (self.u0 + self.u1), 0.5 * (self.v0 + self.v1))

    @property
    def half_size(self) -> tuple[float, float]:
        return (0.5 * abs(self.u1 - self.u0), 0.5 * abs(self.v1 - self.v0))


@dataclass(slots=True)
class Light:
    """A point light standing in for a small area source."""

    pos: np.ndarray
    colour: tuple[float, float, float] = (1.0, 0.97, 0.92)
    power: float = 6.0
    is_fixture: bool = True  # draw a visible disc on the ceiling for this one

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float32).reshape(3)


@dataclass(frozen=True, slots=True)
class StyleParams:
    """How aggressively a style textures the room.

    ``low_texture`` is the interesting one: it is deliberately close to the failure mode
    of every naive stitcher — large matte surfaces with almost no gradient — while still
    leaving enough landmarks that a *good* stitcher can solve it. If it had no landmarks
    at all the regression suite would only be proving that the problem is impossible.
    """

    name: str
    wall_stripe: float
    wall_grain: float
    floor_grain: float
    floor_seam: float
    ceiling_grain: float
    specular_ks: float
    shininess: float
    n_pictures: int
    n_furniture: int
    ambient: float = 0.26
    # Large-scale paint unevenness (cycles per metre are low, so it survives heavy
    # filtering). This is what keeps a matte wall *just* trackable at coarse scales
    # without giving it the fine detail that would make it an easy case.
    wall_mottle: float = 0.0


STYLES: Final[dict[str, StyleParams]] = {
    "rich": StyleParams(
        name="rich",
        wall_stripe=0.085,
        wall_grain=0.030,
        floor_grain=0.10,
        floor_seam=0.45,
        ceiling_grain=0.012,
        specular_ks=0.02,
        shininess=24.0,
        n_pictures=3,
        n_furniture=6,
        wall_mottle=0.012,
    ),
    "low_texture": StyleParams(
        name="low_texture",
        wall_stripe=0.0,
        wall_grain=0.005,
        floor_grain=0.020,
        floor_seam=0.30,
        ceiling_grain=0.003,
        specular_ks=0.0,
        shininess=24.0,
        n_pictures=3,
        n_furniture=4,
        ambient=0.30,
        wall_mottle=0.030,
    ),
    "specular": StyleParams(
        name="specular",
        wall_stripe=0.070,
        wall_grain=0.026,
        floor_grain=0.09,
        floor_seam=0.40,
        ceiling_grain=0.010,
        specular_ks=0.38,
        shininess=64.0,
        n_pictures=3,
        n_furniture=6,
        wall_mottle=0.012,
    ),
}

_PICTURE_PALETTE: Final[np.ndarray] = np.array(
    [
        [0.86, 0.20, 0.16],
        [0.12, 0.28, 0.62],
        [0.94, 0.82, 0.18],
        [0.10, 0.10, 0.12],
        [0.93, 0.92, 0.88],
        [0.18, 0.52, 0.33],
        [0.75, 0.42, 0.12],
        [0.55, 0.20, 0.60],
    ],
    dtype=np.float32,
)


# ======================================================================================
# The room
# ======================================================================================


@dataclass(eq=False)
class Room:
    """A rectangular room with furniture, wall features, lights and an observer.

    The enclosing box spans ``x in [0, width_m]``, ``z in [0, depth_m]`` and
    ``y in [-height_m, 0]`` — floor at ``y = 0``, ceiling above it at negative ``y``,
    because world +Y is down.
    """

    width_m: float = 5.2
    depth_m: float = 3.9
    height_m: float = 2.7
    eye_height_m: float = 1.55
    eye: np.ndarray | None = None
    boxes: list[Box] = field(default_factory=list)
    rects: list[WallRect] = field(default_factory=list)
    rug: tuple[float, float, float, float] | None = None  # (x0, z0, x1, z1) in metres
    lights: list[Light] = field(default_factory=list)
    style: str = "rich"
    seed: int = 0
    # Fixed baseline exposure of the virtual sensor. Chosen so that painted surfaces land
    # around 0.45-0.6 and only the window and the light fixtures clip, which is what gives
    # the auto-exposure simulation something real to hunt for.
    exposure: float = 0.72

    # Derived; filled in by __post_init__.
    lo: np.ndarray = field(init=False, repr=False)
    hi: np.ndarray = field(init=False, repr=False)
    params: StyleParams = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ValueError(f"unknown style {self.style!r}; expected one of {sorted(STYLES)}")
        self.params = STYLES[self.style]
        self.lo = np.array([0.0, -self.height_m, 0.0], dtype=np.float32)
        self.hi = np.array([self.width_m, 0.0, self.depth_m], dtype=np.float32)
        if self.eye is None:
            self.eye = np.array(
                [0.5 * self.width_m + 0.18, -self.eye_height_m, 0.5 * self.depth_m - 0.24],
                dtype=np.float32,
            )
        else:
            self.eye = np.asarray(self.eye, dtype=np.float32).reshape(3)
        if not self.lights:
            self.lights = self._default_lights()
        self._fixtures = np.array(
            [[float(li.pos[0]), float(li.pos[2])] for li in self.lights if li.is_fixture],
            dtype=np.float32,
        ).reshape(-1, 2)
        self._light_pos = np.stack([li.pos for li in self.lights]).astype(np.float32)
        self._light_col = np.array([li.colour for li in self.lights], dtype=np.float32)
        self._light_pow = np.array([li.power for li in self.lights], dtype=np.float32)
        self._rects_by_face: dict[int, list[WallRect]] = {f: [] for f in WALL_FACES}
        for r in self.rects:
            if r.face not in self._rects_by_face:
                raise ValueError(f"wall rect on non-wall face {r.face}")
            self._rects_by_face[r.face].append(r)

    # -- geometry helpers -------------------------------------------------------------

    def _default_lights(self) -> list[Light]:
        w, d, h = self.width_m, self.depth_m, self.height_m
        return [
            Light(np.array([0.38 * w, -h + 0.22, 0.55 * d]), (1.0, 0.96, 0.89), 2.6, True),
            Light(np.array([0.80 * w, -h + 0.22, 0.28 * d]), (1.0, 0.95, 0.86), 1.05, True),
        ]

    def wall_extent(self, face: int) -> tuple[float, float]:
        """``(u_span, v_span)`` of a wall face, in metres."""
        if face in (FACE_XLO, FACE_XHI):
            return (self.depth_m, self.height_m)
        if face in (FACE_ZLO, FACE_ZHI):
            return (self.width_m, self.height_m)
        raise ValueError(f"face {face} is not a wall")

    def contains_eye(self, p: np.ndarray, margin: float = 0.35) -> bool:
        """True if ``p`` is inside the room and clear of every piece of furniture."""
        p = np.asarray(p, dtype=np.float32).reshape(3)
        if np.any(p <= self.lo + 0.25) or np.any(p >= self.hi - 0.25):
            return False
        return not any(b.contains(p, margin) for b in self.boxes)

    # -- surface parameterisation -----------------------------------------------------

    def surface_uv(self, face: int, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Surface coordinates in metres for a hit on outer face ``face``.

        Walls use ``u`` = left-to-right as seen from inside, ``v`` = height above the
        floor. The floor and the ceiling use plan coordinates ``(x, z)``.
        """
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        if face == FACE_XLO:
            return self.hi[2] - z, -y
        if face == FACE_XHI:
            return z - self.lo[2], -y
        if face == FACE_ZLO:
            return self.hi[0] - x, -y
        if face == FACE_ZHI:
            return x - self.lo[0], -y
        if face in (FACE_CEILING, FACE_FLOOR):
            return x - self.lo[0], z - self.lo[2]
        raise ValueError(f"unknown outer face {face}")

    # ==================================================================================
    # Ray tracing
    # ==================================================================================

    def trace(
        self,
        origin: np.ndarray,
        dirs: np.ndarray,
        *,
        pixel_angle: float | np.ndarray = 0.0,
        chunk: int = 1 << 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trace ``dirs`` from ``origin``.

        Returns ``(rgb, depth, face_id)`` with ``rgb`` in linear display units — walls
        land around 0.4-0.7 and windows well above 1.0, so the caller still has the
        headroom that an auto-exposure simulation needs.

        ``pixel_angle`` is the angular size of one pixel in radians (scalar or per-ray).
        It is what turns the procedural textures from point samples into properly
        filtered ones; pass 0 only if you deliberately want aliasing.
        """
        o = np.asarray(origin, dtype=np.float32).reshape(3)
        d = np.asarray(dirs, dtype=np.float32).reshape(-1, 3)
        n = int(d.shape[0])
        pa = np.asarray(pixel_angle, dtype=np.float32)
        if pa.ndim not in (0, 1):
            raise ValueError("pixel_angle must be a scalar or a (N,) array")
        if pa.ndim == 1 and pa.shape[0] != n:
            raise ValueError("per-ray pixel_angle must match the number of rays")

        rgb = np.empty((n, 3), dtype=np.float32)
        depth = np.empty(n, dtype=np.float32)
        fid = np.empty(n, dtype=np.int32)
        step = max(int(chunk), 1)
        for start in range(0, n, step):
            end = min(start + step, n)
            pa_c = pa if pa.ndim == 0 else pa[start:end]
            rgb[start:end], depth[start:end], fid[start:end] = self._trace_chunk(
                o, d[start:end], pa_c
            )
        return rgb, depth, fid

    # -- internals --------------------------------------------------------------------

    def _trace_chunk(
        self, o: np.ndarray, d: np.ndarray, pa: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = d.shape[0]
        norm = np.sqrt(np.einsum("ij,ij->i", d, d))
        np.maximum(norm, _EPS, out=norm)
        dn = d / norm[:, None]

        safe = np.where(np.abs(dn) < 1e-8, np.float32(1e-8), dn)
        inv = np.float32(1.0) / safe

        # --- enclosing box, from the inside: the nearest *exit* is the hit ------------
        best_t = np.full(m, np.float32(np.inf))
        axis = np.zeros(m, dtype=np.int64)
        for a in range(3):
            ta = (self.lo[a] - o[a]) * inv[:, a]
            tb = (self.hi[a] - o[a]) * inv[:, a]
            far = np.maximum(ta, tb)
            upd = far < best_t
            best_t = np.where(upd, far, best_t)
            axis = np.where(upd, a, axis)
        dsel = np.take_along_axis(dn, axis[:, None], axis=1)[:, 0]
        face = (axis * 2 + (dsel > 0)).astype(np.int32)
        inner_idx = np.full(m, -1, dtype=np.int32)

        # --- furniture, from the outside: the nearest *entry* wins --------------------
        for bi, box in enumerate(self.boxes):
            tmin = np.full(m, np.float32(-np.inf))
            tmax = np.full(m, np.float32(np.inf))
            iaxis = np.zeros(m, dtype=np.int64)
            for a in range(3):
                ta = (box.lo[a] - o[a]) * inv[:, a]
                tb = (box.hi[a] - o[a]) * inv[:, a]
                near = np.minimum(ta, tb)
                far = np.maximum(ta, tb)
                upd = near > tmin
                tmin = np.where(upd, near, tmin)
                iaxis = np.where(upd, a, iaxis)
                np.minimum(tmax, far, out=tmax)
            hit = (tmin <= tmax) & (tmin > np.float32(1e-4)) & (tmin < best_t)
            if not bool(hit.any()):
                continue
            dsel_i = np.take_along_axis(dn, iaxis[:, None], axis=1)[:, 0]
            # travelling in +axis means we enter through the `lo` face (side 0)
            side = (dsel_i <= 0).astype(np.int64)
            f_inner = (INNER_FACE_BASE + 6 * bi + 2 * iaxis + side).astype(np.int32)
            best_t = np.where(hit, tmin, best_t)
            face = np.where(hit, f_inner, face)
            inner_idx = np.where(hit, np.int32(bi), inner_idx)

        miss = ~np.isfinite(best_t)
        if bool(miss.any()):
            best_t = np.where(miss, np.float32(1e6), best_t)
            face = np.where(miss, np.int32(_MISS), face)

        p = o[None, :] + best_t[:, None] * dn
        rgb = self._shade(p, dn, best_t, face, inner_idx, pa)
        if bool(miss.any()):
            rgb[miss] = np.float32(0.05)
            best_t = np.where(miss, np.float32(np.inf), best_t)
        return rgb, best_t, face

    def _footprint(self, t: np.ndarray, dn: np.ndarray, normal: np.ndarray, pa: np.ndarray) -> np.ndarray:
        """Half-width, in surface metres, of the pixel footprint at each hit."""
        cos_i = np.abs(dn @ normal.reshape(3))
        np.maximum(cos_i, np.float32(_GRAZE_CLAMP), out=cos_i)
        fp = np.float32(0.5 * _FILTER_WIDTH) * t * pa / cos_i
        return np.maximum(fp, np.float32(_MIN_FOOTPRINT))

    def _shade(
        self,
        p: np.ndarray,
        dn: np.ndarray,
        t: np.ndarray,
        face: np.ndarray,
        inner_idx: np.ndarray,
        pa: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros((p.shape[0], 3), dtype=np.float32)
        pa_arr = pa if pa.ndim == 1 else np.full(p.shape[0], float(pa), dtype=np.float32)

        for f in (FACE_XLO, FACE_XHI, FACE_CEILING, FACE_FLOOR, FACE_ZLO, FACE_ZHI):
            sel = face == f
            if not bool(sel.any()):
                continue
            idx = np.flatnonzero(sel)
            normal = _OUTER_NORMALS[f]
            fp = self._footprint(t[idx], dn[idx], normal, pa_arr[idx])
            u, v = self.surface_uv(f, p[idx])
            albedo, emissive, ks = self._outer_surface(f, u, v, fp)
            out[idx] = self._light(p[idx], dn[idx], normal, albedo, emissive, ks)

        for bi, box in enumerate(self.boxes):
            sel = inner_idx == bi
            if not bool(sel.any()):
                continue
            idx = np.flatnonzero(sel)
            local = face[idx] - (INNER_FACE_BASE + 6 * bi)
            pb = p[idx]
            for lf in range(6):
                sub = np.flatnonzero(local == lf)
                if sub.size == 0:
                    continue
                gi = idx[sub]
                normal = _INNER_NORMALS[lf]
                fp = self._footprint(t[gi], dn[gi], normal, pa_arr[gi])
                ax = lf // 2
                ua, va = (1, 2) if ax == 0 else ((0, 2) if ax == 1 else (0, 1))
                u = pb[sub, ua] - box.lo[ua]
                v = pb[sub, va] - box.lo[va]
                albedo, emissive, ks = self._furniture_surface(box, u, v, fp)
                out[gi] = self._light(p[gi], dn[gi], normal, albedo, emissive, ks)
        return out

    # -- lighting ---------------------------------------------------------------------

    def _light(
        self,
        p: np.ndarray,
        dn: np.ndarray,
        normal: np.ndarray,
        albedo: np.ndarray,
        emissive: np.ndarray,
        ks: float,
    ) -> np.ndarray:
        n_vec = normal.reshape(1, 3)
        # Hemispheric ambient: surfaces facing up (world -Y) catch more bounce light, so
        # the ceiling and the floor never come out identical.
        up = float(max(0.0, -normal[1]))
        amb = np.float32(self.params.ambient * (0.85 + 0.30 * up))
        out = albedo * amb
        view = -dn
        for li in range(self._light_pos.shape[0]):
            dl = self._light_pos[li][None, :] - p
            dist2 = np.einsum("ij,ij->i", dl, dl) + np.float32(1e-4)
            ld = dl / np.sqrt(dist2)[:, None]
            ndl = np.einsum("ij,ij->i", np.broadcast_to(n_vec, ld.shape), ld)
            np.maximum(ndl, 0.0, out=ndl)
            # Softened inverse-square. A bare 1/r^2 point light blows out whatever it is
            # nearest to; real fixtures are area sources in a room full of bounce, and the
            # softened falloff keeps the dynamic range where a phone sensor would put it.
            atten = self._light_pow[li] / (np.float32(1.0) + np.float32(0.5) * dist2)
            w = ndl * atten
            out += albedo * (self._light_col[li][None, :] * w[:, None])
            if ks > 0.0:
                hv = ld + view
                hv /= np.sqrt(np.einsum("ij,ij->i", hv, hv))[:, None] + np.float32(_EPS)
                ndh = np.einsum("ij,ij->i", np.broadcast_to(n_vec, hv.shape), hv)
                np.maximum(ndh, 0.0, out=ndh)
                spec = np.float32(ks) * ndh ** np.float32(self.params.shininess) * atten
                out += self._light_col[li][None, :] * (spec * (ndl > 0))[:, None]
        out += emissive
        out *= np.float32(self.exposure)
        return np.maximum(out, 0.0)

    # ==================================================================================
    # Procedural surfaces
    # ==================================================================================

    def _outer_surface(
        self, face: int, u: np.ndarray, v: np.ndarray, fp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        ks = self.params.specular_ks
        if face == FACE_CEILING:
            return (*self.ceiling(u, v, fp), ks * 0.3)
        if face == FACE_FLOOR:
            return (*self.wood_floor(u, v, fp), ks)
        albedo, emissive = self.wallpaper(u, v, fp, face)
        albedo, emissive = self.skirting(u, v, fp, albedo, emissive)
        for rect in self._rects_by_face.get(face, ()):
            albedo, emissive = self._apply_rect(rect, u, v, fp, albedo, emissive)
        return albedo, emissive, ks * 0.5

    def _apply_rect(
        self,
        rect: WallRect,
        u: np.ndarray,
        v: np.ndarray,
        fp: np.ndarray,
        albedo: np.ndarray,
        emissive: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cu, cv = rect.centre
        hw, hh = rect.half_size
        du = u - np.float32(cu)
        dv = v - np.float32(cv)
        pad = float(fp.max()) + 1e-3
        near = (np.abs(du) <= hw + pad) & (np.abs(dv) <= hh + pad)
        if not bool(near.any()):
            return albedo, emissive
        idx = np.flatnonzero(near)
        cov = _rect_cov(du[idx], dv[idx], hw, hh, fp[idx])
        lu = du[idx] + np.float32(hw)
        lv = dv[idx] + np.float32(hh)
        if rect.kind == "window":
            ra, re = self.window(lu, lv, fp[idx], 2.0 * hw, 2.0 * hh, rect.seed)
        elif rect.kind == "door":
            ra, re = self.door(lu, lv, fp[idx], 2.0 * hw, 2.0 * hh, rect.seed)
        else:
            ra, re = self.picture(lu, lv, fp[idx], 2.0 * hw, 2.0 * hh, rect.seed)
        albedo[idx] = _mix(albedo[idx], ra, cov)
        emissive[idx] = _mix(emissive[idx], re, cov)
        return albedo, emissive

    # -- textures ---------------------------------------------------------------------

    def wallpaper(
        self, u: np.ndarray, v: np.ndarray, fp: np.ndarray, face: int = FACE_ZHI
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subtle vertical stripes over a fine grain, tinted slightly per wall."""
        n = u.shape[0]
        p = self.params
        tint = _WALL_TINTS[face % len(_WALL_TINTS)]
        base = _rgb(tint[0], tint[1], tint[2], n)
        if p.wall_stripe > 0.0:
            stripe = _periodic_band(u, 0.163, 0.021, fp)
            base *= (np.float32(1.0) - np.float32(p.wall_stripe) * stripe)[:, None]
        grain = _fbm(u, v, 7.0, 4, fp, self.seed * 31 + face)
        base *= (np.float32(1.0) + np.float32(2.0 * p.wall_grain) * grain)[:, None]
        if p.wall_mottle > 0.0:
            mottle = _fbm(u, v, 1.15, 3, fp, self.seed * 31 + face + 401)
            base *= (np.float32(1.0) + np.float32(2.0 * p.wall_mottle) * mottle)[:, None]
        return np.clip(base, 0.0, 4.0), np.zeros((n, 3), dtype=np.float32)

    def skirting(
        self,
        u: np.ndarray,
        v: np.ndarray,
        fp: np.ndarray,
        albedo: np.ndarray,
        emissive: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The white board along the bottom of every wall — a strong horizontal landmark."""
        top = 0.108
        cov = _band(np.abs(v - np.float32(0.5 * top)), 0.5 * top, fp)
        board = _rgb(0.90, 0.895, 0.88, u.shape[0])
        groove = _band(np.abs(v - np.float32(top - 0.014)), 0.005, fp)
        board *= (np.float32(1.0) - np.float32(0.35) * groove)[:, None]
        albedo = _mix(albedo, board, cov)
        return albedo, emissive

    def wood_floor(
        self, u: np.ndarray, v: np.ndarray, fp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Planks running along +x: per-plank tone, butt joints, seams and grain."""
        n = u.shape[0]
        p = self.params
        plank_w = 0.186
        plank_l = 1.34
        row = np.floor(v / np.float32(plank_w))
        # Stagger the butt joints row by row so the floor does not read as a grid.
        offset = _hash01(row.astype(np.int64), np.zeros(n, dtype=np.int64), self.seed + 5) * np.float32(
            plank_l
        )
        tone = _cell_value(u + offset, v, plank_l, plank_w, fp, self.seed + 11) - np.float32(0.5)
        base = _rgb(0.50, 0.355, 0.215, n)
        base *= (np.float32(1.0) + np.float32(0.30 * p.floor_seam + 0.10) * tone)[:, None]
        grain = _fbm(u * np.float32(0.35), v * np.float32(9.0), 3.0, 4, fp, self.seed + 17)
        base *= (np.float32(1.0) + np.float32(2.0 * p.floor_grain) * grain)[:, None]
        seam = _periodic_band(v, plank_w, 0.0026, fp)
        butt = _periodic_band(u + offset, plank_l, 0.0022, fp)
        dark = np.clip(seam + butt, 0.0, 1.0)
        base *= (np.float32(1.0) - np.float32(0.55 * p.floor_seam + 0.06) * dark)[:, None]
        if self.rug is not None:
            base = self._rug(u, v, fp, base)
        return np.clip(base, 0.0, 4.0), np.zeros((n, 3), dtype=np.float32)

    def _rug(
        self, u: np.ndarray, v: np.ndarray, fp: np.ndarray, floor: np.ndarray
    ) -> np.ndarray:
        """A bordered, patterned rug.

        Not decoration: a rug is the one thing that gives a *floor* trackable structure at
        a scale a phone actually resolves, and without it the nadir of a low-texture room
        is a featureless plane that no solver can constrain.
        """
        x0, z0, x1, z1 = self.rug
        cu, cv = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
        hw, hh = 0.5 * abs(x1 - x0), 0.5 * abs(z1 - z0)
        du, dv = u - np.float32(cu), v - np.float32(cv)
        cov = _rect_cov(du, dv, hw, hh, fp)
        if not bool((cov > 0.0).any()):
            return floor
        n = u.shape[0]
        body = _rgb(0.42, 0.40, 0.44, n)
        cell = 0.30
        check = np.abs(
            _periodic_band(u, 2.0 * cell, 0.5 * cell, fp)
            - _periodic_band(v, 2.0 * cell, 0.5 * cell, fp)
        )
        body = _mix(body, _rgb(0.60, 0.34, 0.30, n), check * np.float32(0.75))
        motif = _periodic_band(u + v, 0.42, 0.055, fp) * _periodic_band(u - v, 0.42, 0.055, fp)
        body = _mix(body, _rgb(0.86, 0.82, 0.70, n), motif)
        border = _outline_cov(du, dv, hw - 0.02, hh - 0.02, 0.11, fp)
        body = _mix(body, _rgb(0.20, 0.18, 0.22, n), border)
        fringe = _outline_cov(du, dv, hw, hh, 0.02, fp)
        body = _mix(body, _rgb(0.80, 0.77, 0.68, n), fringe)
        return _mix(floor, body, cov)

    def ceiling(
        self, u: np.ndarray, v: np.ndarray, fp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Near-flat off-white plus the glowing discs of the fixtures."""
        n = u.shape[0]
        base = _rgb(0.88, 0.875, 0.855, n)
        grain = _fbm(u, v, 4.0, 3, fp, self.seed + 23)
        base *= (np.float32(1.0) + np.float32(2.0 * self.params.ceiling_grain) * grain)[:, None]
        emissive = np.zeros((n, 3), dtype=np.float32)
        for k in range(self._fixtures.shape[0]):
            fx = self._fixtures[k, 0] - self.lo[0]
            fz = self._fixtures[k, 1] - self.lo[2]
            r = np.sqrt((u - fx) ** 2 + (v - fz) ** 2)
            disc = _disc_cov(r, 0.215, fp)
            ring = _band(np.abs(r - np.float32(0.235)), 0.020, fp)
            emissive += (np.float32(1.45) * disc)[:, None] * np.array(
                [1.0, 0.97, 0.90], dtype=np.float32
            )[None, :]
            base *= (np.float32(1.0) - np.float32(0.25) * ring)[:, None]
        return np.clip(base, 0.0, 4.0), emissive

    def window(
        self,
        u: np.ndarray,
        v: np.ndarray,
        fp: np.ndarray,
        width: float,
        height: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """A blown-out exterior behind a frame and mullions.

        Emissive values run far above 1.0 on purpose: this is the auto-exposure stressor,
        and a phone panning past it must actually clip and then hunt.
        """
        n = u.shape[0]
        cu, cv = 0.5 * width, 0.5 * height
        du, dv = u - np.float32(cu), v - np.float32(cv)
        albedo = _rgb(0.86, 0.85, 0.83, n)
        emissive = np.zeros((n, 3), dtype=np.float32)

        # Sky gradient plus a hazy horizon: brightest at the top.
        sky_t = np.clip(v / np.float32(max(height, 1e-3)), 0.0, 1.0)
        sky = np.empty((n, 3), dtype=np.float32)
        sky[:, 0] = 1.55 + 0.55 * sky_t
        sky[:, 1] = 1.62 + 0.50 * sky_t
        sky[:, 2] = 1.78 + 0.42 * sky_t
        horizon = _band(np.abs(v - np.float32(0.34 * height)), 0.05 * height, fp)
        sky *= (np.float32(1.0) - np.float32(0.18) * horizon)[:, None]

        # A skyline of blocks so the window is not a featureless white slab. Both the
        # rooftop step *and* the vertical join between neighbouring blocks are box
        # filtered; an unfiltered skyline is a wall of high-contrast vertical edges and
        # aliases worse than anything else in the room.
        block_w = max(width / 7.0, 0.06)
        seven = np.full(n, 7, dtype=np.int64)
        bi, bn, bw = _cell_lerp(u, block_w, fp)
        scale = np.float32(0.26 * height)
        inv = np.float32(1.0) / np.maximum(np.float32(2.0) * fp, np.float32(1e-4))

        def roof(idx: np.ndarray) -> np.ndarray:
            bh = _hash01(idx, seven, seed + 3) * scale
            return np.clip((np.float32(0.30 * height) + bh - v) * inv + np.float32(0.5), 0.0, 1.0)

        below = roof(bi) * (np.float32(1.0) - bw) + roof(bn) * bw
        city = _rgb(0.78, 0.79, 0.86, n)
        sky = _mix(sky, city, below * np.float32(0.85))
        emissive += sky

        glass = _rect_cov(du, dv, cu - 0.055, cv - 0.055, fp)
        emissive *= glass[:, None]
        frame = _rgb(0.42, 0.34, 0.26, n)
        albedo = _mix(albedo, frame, np.float32(1.0) - glass)

        mull_v = _band(np.abs(du), 0.017, fp)
        mull_h = _band(np.abs(dv - np.float32(0.08 * height)), 0.016, fp)
        mull = np.clip(mull_v + mull_h, 0.0, 1.0) * glass
        emissive *= (np.float32(1.0) - mull)[:, None]
        albedo = _mix(albedo, frame, mull)
        return np.clip(albedo, 0.0, 4.0), emissive

    def picture(
        self,
        u: np.ndarray,
        v: np.ndarray,
        fp: np.ndarray,
        width: float,
        height: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """High-contrast blocks inside a dark frame: the feature-rich landmark.

        ``low_texture`` rooms lean on these entirely, so they are deliberately loud.
        """
        n = u.shape[0]
        cu, cv = 0.5 * width, 0.5 * height
        du, dv = u - np.float32(cu), v - np.float32(cv)
        cell_u = max(width / 5.0, 0.05)
        cell_v = max(height / 4.0, 0.05)
        n_pal = _PICTURE_PALETTE.shape[0]
        iu, nu, wu = _cell_lerp(u, cell_u, fp)
        iv, nv, wv = _cell_lerp(v, cell_v, fp)

        def block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            pick = (_hash01(a, b, seed + 41) * np.float32(n_pal)).astype(np.int64)
            np.clip(pick, 0, n_pal - 1, out=pick)
            return _PICTURE_PALETTE[pick]

        top = _mix(block(iu, iv), block(nu, iv), wu)
        bot = _mix(block(iu, nv), block(nu, nv), wu)
        art = _mix(top, bot, wv)
        # Beyond two cells per pixel the two-tap blend runs out; fade to the palette mean
        # so the painting dissolves into its average colour instead of into noise.
        cell = min(cell_u, cell_v)
        blend = np.clip((np.float32(2.0) * fp - np.float32(cell)) / np.float32(cell), 0.0, 1.0)
        art = _mix(art, np.broadcast_to(_PICTURE_PALETTE.mean(axis=0), art.shape).copy(), blend)
        grid = np.clip(
            _periodic_band(u, cell_u, 0.006, fp) + _periodic_band(v, cell_v, 0.006, fp), 0.0, 1.0
        )
        art = _mix(art, _rgb(0.06, 0.06, 0.07, n), grid * np.float32(0.9))

        mat = _rect_cov(du, dv, cu - 0.030, cv - 0.030, fp)
        art = _mix(_rgb(0.95, 0.94, 0.91, n), art, mat)
        frame = _rect_cov(du, dv, cu - 0.012, cv - 0.012, fp)
        art = _mix(_rgb(0.14, 0.10, 0.08, n), art, frame)
        return np.clip(art, 0.0, 4.0), np.zeros((n, 3), dtype=np.float32)

    def door(
        self,
        u: np.ndarray,
        v: np.ndarray,
        fp: np.ndarray,
        width: float,
        height: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """A panelled door with architrave and a handle."""
        n = u.shape[0]
        cu, cv = 0.5 * width, 0.5 * height
        du, dv = u - np.float32(cu), v - np.float32(cv)
        leaf = _rgb(0.80, 0.78, 0.74, n)
        grain = _fbm(u, v, 5.0, 3, fp, seed + 61)
        leaf *= (np.float32(1.0) + np.float32(0.05) * grain)[:, None]

        for cy, ch in ((0.30 * height, 0.19 * height), (0.70 * height, 0.19 * height)):
            outline = _outline_cov(du, v - np.float32(cy), 0.32 * width, ch, 0.020, fp)
            inner = _rect_cov(du, v - np.float32(cy), 0.32 * width - 0.020, ch - 0.020, fp)
            leaf = _mix(leaf, _rgb(0.58, 0.56, 0.53, n), outline)
            leaf = _mix(leaf, _rgb(0.86, 0.84, 0.80, n), inner * np.float32(0.55))

        arch = _outline_cov(du, dv, cu - 0.006, cv - 0.006, 0.045, fp)
        leaf = _mix(leaf, _rgb(0.93, 0.92, 0.89, n), arch)
        rh = np.sqrt((u - np.float32(width - 0.13)) ** 2 + (v - np.float32(0.40 * height)) ** 2)
        leaf = _mix(leaf, _rgb(0.72, 0.62, 0.28, n), _disc_cov(rh, 0.030, fp))
        return np.clip(leaf, 0.0, 4.0), np.zeros((n, 3), dtype=np.float32)

    def _furniture_surface(
        self, box: Box, u: np.ndarray, v: np.ndarray, fp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        n = u.shape[0]
        tint = box.tint
        base = _rgb(tint[0], tint[1], tint[2], n)
        p = self.params
        if box.material == "wood":
            grain = _fbm(u * np.float32(0.3), v * np.float32(7.0), 4.0, 4, fp, box.seed)
            base *= (np.float32(1.0) + np.float32(0.9 * p.floor_grain + 0.03) * grain)[:, None]
            edge = _periodic_band(v, 0.42, 0.004, fp)
            base *= (np.float32(1.0) - np.float32(0.25 * p.floor_seam) * edge)[:, None]
            ks = p.specular_ks * 0.8
        elif box.material == "fabric":
            weave = _fbm(u, v, 34.0, 3, fp, box.seed + 7)
            base *= (np.float32(1.0) + np.float32(1.4 * p.wall_grain + 0.02) * weave)[:, None]
            ks = 0.0
        else:
            grain = _fbm(u, v, 6.0, 3, fp, box.seed + 13)
            base *= (np.float32(1.0) + np.float32(1.2 * p.wall_grain + 0.012) * grain)[:, None]
            ks = p.specular_ks * 0.4
        return np.clip(base, 0.0, 4.0), np.zeros((n, 3), dtype=np.float32), float(ks)

    # ==================================================================================
    # Deterministic layout generation
    # ==================================================================================

    @classmethod
    def random(cls, seed: int = 7, style: str = "rich") -> Room:
        """Build a plausible room from ``seed``. Identical seeds give identical rooms."""
        if style not in STYLES:
            raise ValueError(f"unknown style {style!r}; expected one of {sorted(STYLES)}")
        params = STYLES[style]
        rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        w = 5.2 + float(rng.uniform(-0.55, 0.75))
        d = 3.9 + float(rng.uniform(-0.45, 0.65))
        h = 2.7 + float(rng.uniform(-0.18, 0.28))
        eye_h = 1.55 + float(rng.uniform(-0.07, 0.09))
        eye = np.array(
            [
                0.5 * w + float(rng.uniform(-0.55, 0.55)),
                -eye_h,
                0.5 * d + float(rng.uniform(-0.45, 0.45)),
            ],
            dtype=np.float32,
        )

        lights = [
            Light(
                np.array([w * float(rng.uniform(0.30, 0.45)), -h + 0.22, d * float(rng.uniform(0.45, 0.62))]),
                (1.0, 0.96, 0.89),
                2.55 + float(rng.uniform(-0.25, 0.4)),
                True,
            ),
            Light(
                np.array([w * float(rng.uniform(0.70, 0.86)), -h + 0.22, d * float(rng.uniform(0.22, 0.38))]),
                (1.0, 0.94, 0.85),
                1.02 + float(rng.uniform(-0.15, 0.28)),
                True,
            ),
        ]

        # --- wall features -----------------------------------------------------------
        faces = list(WALL_FACES)
        rng.shuffle(faces)
        window_face = int(faces[0])
        door_face = int(faces[1])
        rects: list[WallRect] = []

        wu = d if window_face in (FACE_XLO, FACE_XHI) else w
        win_w = min(1.62, max(0.9, wu * 0.34))
        win_cu = float(rng.uniform(0.30, 0.70)) * wu
        win_cu = float(np.clip(win_cu, 0.55 * win_w + 0.15, wu - 0.55 * win_w - 0.15))
        rects.append(
            WallRect(
                face=window_face,
                u0=win_cu - 0.5 * win_w,
                v0=0.92,
                u1=win_cu + 0.5 * win_w,
                v1=0.92 + 1.28,
                kind="window",
                seed=int(rng.integers(1, 1 << 20)),
            )
        )
        # The window is also a light source; put a broad cool source just inside it.
        lights.append(
            Light(
                cls._wall_point(window_face, w, d, h, win_cu, 1.55, inset=1.05),
                (0.86, 0.92, 1.0),
                1.50 + float(rng.uniform(-0.18, 0.38)),
                False,
            )
        )

        du_span = d if door_face in (FACE_XLO, FACE_XHI) else w
        door_w = 0.92
        door_cu = float(np.clip(float(rng.uniform(0.22, 0.78)) * du_span, 0.7, du_span - 0.7))
        rects.append(
            WallRect(
                face=door_face,
                u0=door_cu - 0.5 * door_w,
                v0=0.0,
                u1=door_cu + 0.5 * door_w,
                v1=2.06,
                kind="door",
                seed=int(rng.integers(1, 1 << 20)),
            )
        )

        for k in range(params.n_pictures):
            span_of = {f: (d if f in (FACE_XLO, FACE_XHI) else w) for f in WALL_FACES}
            placed = False
            for attempt in range(24):
                face = int(faces[(k + 2 + attempt) % len(faces)])
                span = span_of[face]
                pw = float(rng.uniform(0.95, 1.48))
                ph = pw * float(rng.uniform(0.60, 0.88))
                half = 0.5 * pw + 0.14
                if 2.0 * half >= span:
                    continue
                cu = float(np.clip(float(rng.uniform(0.12, 0.88)) * span, half, span - half))
                cv = float(rng.uniform(1.32, 1.82))
                if _overlaps(rects, face, cu, cv, pw, ph):
                    continue
                rects.append(
                    WallRect(
                        face=face,
                        u0=cu - 0.5 * pw,
                        v0=cv - 0.5 * ph,
                        u1=cu + 0.5 * pw,
                        v1=cv + 0.5 * ph,
                        kind="picture",
                        seed=int(rng.integers(1, 1 << 20)),
                    )
                )
                placed = True
                break
            if not placed:  # every wall is busy; a room with fewer pictures is still valid
                continue

        # --- furniture ---------------------------------------------------------------
        catalogue = [
            ("wood", (0.60, 0.44, 0.28), (1.55, 0.74, 0.62)),  # table
            ("fabric", (0.34, 0.38, 0.46), (1.85, 0.78, 0.86)),  # sofa
            ("wood", (0.52, 0.38, 0.25), (0.98, 1.82, 0.38)),  # bookcase
            ("matte", (0.72, 0.70, 0.66), (0.62, 0.58, 0.55)),  # side cabinet
            ("fabric", (0.46, 0.30, 0.28), (0.82, 0.86, 0.82)),  # armchair
            ("matte", (0.24, 0.24, 0.26), (1.22, 0.52, 0.44)),  # sideboard
            ("wood", (0.66, 0.50, 0.32), (0.52, 1.10, 0.52)),  # plant stand
        ]
        boxes: list[Box] = []
        room_lo = np.array([0.0, -h, 0.0], dtype=np.float32)
        room_hi = np.array([w, 0.0, d], dtype=np.float32)
        attempts = 0
        while len(boxes) < params.n_furniture and attempts < 400:
            attempts += 1
            mat, tint, size = catalogue[len(boxes) % len(catalogue)]
            sx, sy, sz = (float(s) * float(rng.uniform(0.85, 1.15)) for s in size)
            x0 = float(rng.uniform(0.10, max(0.11, w - sx - 0.10)))
            z0 = float(rng.uniform(0.10, max(0.11, d - sz - 0.10)))
            lo = np.array([x0, -sy, z0], dtype=np.float32)
            hi = np.array([x0 + sx, 0.0, z0 + sz], dtype=np.float32)
            if np.any(hi > room_hi + 1e-3) or np.any(lo < room_lo - 1e-3):
                continue
            cand = Box(lo=lo, hi=hi, material=mat, tint=tint, seed=int(rng.integers(1, 1 << 20)))
            # Keep the observer's own footprint clear. A box that merely misses the eye
            # point in 3D can still be the thing the eye is standing *on*, which no real
            # capture ever is, and which would hide most of the floor from the reference.
            if (
                lo[0] - 0.55 < eye[0] < hi[0] + 0.55
                and lo[2] - 0.55 < eye[2] < hi[2] + 0.55
            ):
                continue
            if any(_boxes_overlap(cand, b, 0.06) for b in boxes):
                continue
            boxes.append(cand)

        rug_w = min(3.2, 0.56 * w)
        rug_d = min(2.4, 0.58 * d)
        rug_cx = float(np.clip(0.5 * w + float(rng.uniform(-0.5, 0.5)), 0.5 * rug_w + 0.15, w - 0.5 * rug_w - 0.15))
        rug_cz = float(np.clip(0.5 * d + float(rng.uniform(-0.4, 0.4)), 0.5 * rug_d + 0.15, d - 0.5 * rug_d - 0.15))
        rug = (rug_cx - 0.5 * rug_w, rug_cz - 0.5 * rug_d, rug_cx + 0.5 * rug_w, rug_cz + 0.5 * rug_d)

        room = cls(
            width_m=w,
            depth_m=d,
            height_m=h,
            eye_height_m=eye_h,
            eye=eye,
            boxes=boxes,
            rects=rects,
            rug=rug,
            lights=lights,
            style=style,
            seed=int(seed) & 0x7FFFFFFF,
        )
        if not room.contains_eye(eye, margin=0.30):
            room.eye = np.array([0.5 * w, -eye_h, 0.5 * d], dtype=np.float32)
        return room

    @staticmethod
    def _wall_point(
        face: int, w: float, d: float, h: float, u: float, v: float, inset: float
    ) -> np.ndarray:
        """Inverse of :meth:`surface_uv` for a wall, pushed ``inset`` metres into the room."""
        y = -v
        if face == FACE_XLO:
            return np.array([inset, y, d - u], dtype=np.float32)
        if face == FACE_XHI:
            return np.array([w - inset, y, u], dtype=np.float32)
        if face == FACE_ZLO:
            return np.array([w - u, y, inset], dtype=np.float32)
        if face == FACE_ZHI:
            return np.array([u, y, d - inset], dtype=np.float32)
        raise ValueError(f"face {face} is not a wall")


_WALL_TINTS: Final[tuple[tuple[float, float, float], ...]] = (
    (0.735, 0.722, 0.690),
    (0.700, 0.706, 0.700),
    (0.745, 0.730, 0.700),
    (0.712, 0.712, 0.704),
    (0.728, 0.718, 0.696),
    (0.706, 0.716, 0.706),
)


def _overlaps(rects: list[WallRect], face: int, cu: float, cv: float, pw: float, ph: float) -> bool:
    for r in rects:
        if r.face != face:
            continue
        ru, rv = r.centre
        rhw, rhh = r.half_size
        if abs(ru - cu) < rhw + 0.5 * pw + 0.10 and abs(rv - cv) < rhh + 0.5 * ph + 0.10:
            return True
    return False


def _boxes_overlap(a: Box, b: Box, gap: float) -> bool:
    return bool(np.all(a.lo - gap < b.hi) and np.all(b.lo - gap < a.hi))


def pixel_angle_equirect(height: int, supersample: int = 1) -> float:
    """Angular size of one equirect pixel in radians (meridional, hence constant)."""
    return float(math.pi / (float(height) * max(int(supersample), 1)))


def pixel_angle_pinhole(focal_px: float, supersample: int = 1) -> float:
    """Angular size of one pinhole pixel at the optical axis, in radians."""
    return float(1.0 / (max(float(focal_px), 1e-6) * max(int(supersample), 1)))
