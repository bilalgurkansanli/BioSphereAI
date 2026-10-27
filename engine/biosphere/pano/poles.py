"""Closing every hole a capture leaves, at the poles and anywhere else.

:func:`fill_uncovered` is the entry point and it takes the whole problem: no pixel of a
delivered panorama may be black.  In a flat photograph a missing corner is a crop; in a 360
viewer it is a void the user turns their head and falls into, and any plausible fill beats
that.  Nor is a hole in the middle of the sphere unusual -- an agent sweeping a room turns
too fast, blocks the lens with an arm, or skips the bit behind the wardrobe -- so the poles
are treated as the special case that needs different geometry, not as the only case.

Why the poles are special
-------------------------
Nobody points a phone straight up, and the tripod (or the operator's shoes) occupies the
nadir, so almost every real capture arrives with two circular holes.  The tempting fix --
run an inpainter on the equirect canvas -- produces the worst artefact in the whole
pipeline: near the poles one canvas row spans a few arc-seconds of real angle while one
canvas column still spans a full degree, so any isotropic inpainting kernel is violently
anisotropic on the sphere and smears the fill into radial streaks that converge on the
pole like a bad zoom blur.

So we change coordinates first.  A stereographic projection of the cap is conformal: it
preserves angles, small circles stay circles, and an isotropic kernel in the plane *is*
isotropic on the sphere.  Inpainting there and projecting back gives a fill that looks
like ceiling rather than like a smear.  Away from the poles none of that applies -- a row
and a column span comparable angles around the equator -- so those holes are inpainted in
the equirect domain directly, at a downscale chosen against a measured time budget, and
cropped with wraparound so a hole straddling ``u = 0`` is filled as the one region it is.

Filling never touches coverage
------------------------------
``covered_fraction``, the reported gaps and the zenith/nadir flags are measured from the
camera masks and describe what the capture actually reached.  They must read the same
whether or not a pixel was later filled in: the fill is a presentation decision, coverage
is a factual one, and the number that sends an agent back to sweep the corner they missed
cannot be allowed to congratulate them for the renderer's guesswork.  The coverage mask is
handed to the filler read-only for exactly that reason.

The composite reaches about six degrees past the hole, but its alpha ramp is deliberately
much tighter than that -- see ``_JOIN_DEG``.  Alpha must be exactly 1 on every uncovered
pixel or the black canvas shows through, so the ramp can only fall off across *captured*
pixels, and every one it touches is replaced by a twice-resampled copy of itself.  A wide,
comfortable-looking feather therefore buys an invisible join at the price of a visibly soft
ring, which is the wrong trade.

Level, not just texture
-----------------------
An inpainter answers "what colour continues this boundary", and for a large hole every
diffusion-based method converges on the *mean* of the ring it is attached to.  For a room
that is systematically the wrong answer at both ends: a ceiling is normally the brightest
surface in an interior and a floor the darkest, so a cap filled with the average of the
upper wall reads as a dull grey lid over a bright room -- and looking up is the first thing
a viewer does in a 360 tour.

So the fill's radial *level* is set separately from its texture.  The mean colour of the
covered ring is fitted against elevation over the last ten degrees of real data, per
channel so warmth carries as well as brightness, and that trend is extrapolated into the
cap as a purely radial correction anchored at exactly zero where the real pixels begin.
The result still contains no invented structure -- the correction is a smooth function of
colatitude alone -- it simply arrives at the pole at the level the room was heading for
instead of at the wall's average.

That extrapolation is gated on the evidence actually supporting it, because a ten-degree
measurement carried fifty degrees to the pole is a fivefold lever and rooms are full of
things that break it.  A wall that darkens into a cornice before the ceiling brightens
gives a *downward* trend at the top -- follow it and the ceiling goes black, which is far
worse than grey.  So the fit must be a good straight line (weighted R^2), its slope is
shrunk by its own standard error before anything is carried anywhere, and it is tapered to
zero slope at the pole, which the azimuthal mean of any continuous field on a sphere has to
be anyway.  Where the ring does not support a trend the cap simply holds the ring's level:
no better than before, but never worse.

``pole_method='gradient'`` skips all of that and extrapolates the colour of the ring of
real pixels bounding the hole, heavily blurred.  It is cheap, it can never hallucinate,
and for a flat ceiling it is very nearly indistinguishable from the inpainted result.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g
from .types import PipelineConfig
from .warp import to_uint8

__all__ = [
    "fill_uncovered",
    "fill_poles",
    "cap_frame",
    "stereographic_rays",
    "extract_cap",
    "composite_nadir_logo",
    "composite_pole_logo",
    "RingTrend",
    "ring_trend",
    "trend_colour",
]

_LOG = logging.getLogger(__name__)

#: Elevation above which a pixel counts as "polar" and may be replaced.
_CAP_DEG = 60.0
#: How far past the hole's boundary ring the stereographic patch reaches, so the inpainter
#: always has real pixels to extrapolate from.  A single-row sweep can leave the hole
#: reaching down to 25 deg elevation, so this is derived from the coverage, not fixed.
_CONTEXT_DEG = 8.0
#: The cap never reaches past this colatitude; beyond it the conformal patch would need
#: unbounded extent (the projection point itself is at 180 deg).
_MAX_REACH_DEG = 85.0
#: Fraction of a canvas row that must be covered for the polar hole to count as finished.
_CAP_ROW_SEEN = 0.25
#: How far past the hole the composite region extends, so the join has room.
_FEATHER_DEG = 6.0
#: How far the alpha ramp reaches *into genuinely captured pixels*.
#:
#: Deliberately much narrower than the 6 degree composite band.  Alpha has to be exactly 1
#: on every uncovered pixel (anything less lets the black canvas show through), so the ramp
#: can only fall off over *covered* territory -- and every covered pixel it touches gets
#: replaced by a version of itself that has been round-tripped through the stereographic
#: patch and back, i.e. resampled twice and measurably softer.  A six-degree ramp softens a
#: six-degree ring of real content; the seam-gradient metric picks that up immediately.
#: Two degrees is still 45 px at 8K, far more than enough to hide a join between two
#: renderings of the same scene.
_JOIN_DEG = 2.0
#: Rows nearest the hole are skipped when measuring the trend: they sit under the blend's
#: mask edge and under the join ramp, so their means are not clean scene radiance.
_TREND_SKIP_DEG = 1.5
#: How much real data the radial trend is fitted over.
_TREND_WINDOW_DEG = 10.0
_TREND_MIN_ROWS = 3
#: Sanity bounds.  The slope clamp rejects a fit dominated by a bright window at the top of
#: one wall; the delta clamp bounds how far the extrapolation may travel over the tens of
#: degrees between the ring and the pole, so a plausible gradient can never run to white.
_TREND_MAX_SLOPE = 4.0  # grey levels per degree of elevation
_TREND_MAX_DELTA = 90.0  # grey levels
#: Standard errors of the fitted slope subtracted from it before it is extrapolated.
#:
#: This is the guard against amplifying a local feature -- a picture rail, a cornice shadow
#: -- into a global one, and it replaces an earlier rule that simply refused to travel
#: further than the luminance range observed *inside the ten-degree window*.  That rule was
#: not a statement about confidence at all: a wall with a clean, gentle gradient varies by
#: only a few grey levels over ten degrees, so the better the evidence for a trend the
#: harder it was clamped, and the zenith fill came out at the ring's own level no matter how
#: convincing the ring was.  Subtracting the slope's own standard error instead lets a clean
#: fit travel its full distance and shrinks a noisy one toward zero, which is what "do not
#: over-extrapolate" actually means.  Two sigma is deliberately conservative given the lever:
#: ten degrees of measurement carried fifty degrees to the pole.
_TREND_SLOPE_SIGMA = 2.0
#: Goodness of fit below which the ring is not described by a line at all and nothing is
#: extrapolated; and the value above which the full trend is used.  In between the
#: extrapolation is scaled linearly.  A real upper wall is often *curved* -- it darkens
#: into a cornice before the ceiling brightens -- and projecting a straight line through
#: that lands the cap nowhere near the truth, so the fit has to earn its confidence.
_TREND_MIN_R2 = 0.80
_TREND_FULL_R2 = 0.94
#: Radial bins used to measure and correct the fill's own profile.
_TREND_BINS = 96
#: How fast the inpainter's own azimuthal structure is faded out toward the pole.  Exactly
#: zero at the ring whatever this is -- see the flattening in :func:`_apply_radial_trend`.
_TREND_FLATTEN_EXP = 1.0
#: What counts as an annulus of purely captured pixels, fit to read the fill's level from.
_TREND_CLEAN_FRAC = 0.02
_TREND_CLEAN_MIN_PX = 256

#: Fallback cap resolution.  The real one is derived from the canvas by :func:`cap_size_for`.
_CAP_SIZE = 1024
#: Bounds on that derivation.  The upper one is a memory bound, not an optical one: the
#: boundary work below runs on downscaled copies, but the cap itself is three float32
#: channels while it is being corrected, which is 200 MB at 4096 and 50 at 2048.
_CAP_SIZE_MIN = 256
_CAP_SIZE_MAX = 2048
#: The inpainters are run on this downscale: a hole has no high-frequency content to
#: preserve, and FSR on a megapixel image with a large hole takes tens of seconds.
_INPAINT_SIZE = 512
_ROW_CHUNK = 262_144

#: Samples the cap fill may carry across the hole, and the floor on the working size.
#:
#: The scale is chosen from how far the fill has to travel rather than fixed, which is the
#: same rule the equirect filler follows (``_EQUIRECT_HOLE_SAMPLES``).  It is set much
#: higher here, and the reason is worth writing down because the equirect number looks like
#: it should apply: away from a pole a hole is a strip between two pieces of real wall and
#: the fill is an interpolation, so resolving it coarsely costs nothing real.  A pole hole is
#: a disc with real pixels only on the outside, so the fill is an *extrapolation*, and every
#: bit of resolution it is denied at the rim is a mismatch against the ring it has to join.
#:
#: Measured on the reference bedroom, sweeping this constant and reading the per-azimuth
#: brightness step across the boundary (RMS over sixteen sectors, zenith / nadir):
#:
#:     44 samples ->  1.13 / 5.31      176 samples -> 0.71 / 1.53
#:     88 samples ->  0.88 / 2.69      352 samples -> 0.73 / 1.87
#:
#: -- against 3.99 / 4.17 for the fixed 512 the module used before.  The nadir is what
#: separates them: its ring is a floor with a rug and furniture on it rather than a plain
#: ceiling, so there is real structure at the boundary and a fill resolved at 44 samples
#: cannot meet any of it.  352 is no better than 176 and costs an inpaint four times the
#: area, so 176 it is.
_CAP_HOLE_SAMPLES = 176
_CAP_INPAINT_MIN = 64
#: How far the fill is cross-faded into the real pixels *at the cap's own resolution*, in
#: cap pixels.
#:
#: The cap grid is conformal, so near its rim one cap pixel spans about 0.17 deg where one
#: equirect row spans 0.088: the cap is half the canvas's resolution exactly where the two
#: meet.  A hard ``where(known, real, fill)`` there is a hard edge on the coarse grid, and
#: resampling it back onto the finer canvas turns it into the 2 px staircase the audit saw
#: cutting through the ceiling light.  Feathering over one and a half cap pixels costs about
#: 0.15 deg of real content and removes the staircase entirely; the bias below keeps the
#: half-way point on the *hole* side of the boundary so the fill does not creep outward.
_CAP_JOIN_PX = 1.5
_CAP_JOIN_BIAS = 1.5


# ======================================================================================
# Stereographic cap geometry
# ======================================================================================


def cap_frame(which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal ``(axis, e1, e2)`` for a pole cap.  World +Y is *down*."""
    if which == "zenith":
        return (
            np.array([0.0, -1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
    return (
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )


def plane_extent(colatitude_deg: float) -> float:
    """Half-width of the stereographic plane that reaches ``colatitude_deg`` from the pole.

    With the projection point at the far pole and the plane tangent at the near one, a
    colatitude ``c`` maps to plane radius ``2 tan(c/2)``; the equator therefore lands at 2.
    """
    return 2.0 * math.tan(math.radians(colatitude_deg) * 0.5)


def stereographic_rays(
    axis: np.ndarray, e1: np.ndarray, e2: np.ndarray, extent: float, size: int
) -> np.ndarray:
    """``(size, size, 3)`` unit directions for the stereographic cap grid."""
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * (2.0 * extent) - extent
    X = np.broadcast_to(t[None, :], (size, size))
    Y = np.broadcast_to(t[:, None], (size, size))
    r2 = X * X + Y * Y
    denom = 4.0 + r2
    rays = (
        (4.0 * X)[..., None] * e1[None, None, :]
        + (4.0 * Y)[..., None] * e2[None, None, :]
        + (4.0 - r2)[..., None] * axis[None, None, :]
    ) / denom[..., None]
    return rays.astype(np.float32)


def cap_size_for(extent: float, canvas_h: int) -> int:
    """Cap resolution that matches the canvas everywhere inside it.

    The conformal grid is uniform in the plane and the sphere is not, so one cap pixel is
    widest at the pole -- ``4 * extent / size`` radians there against
    ``4 * extent / (size * (1 + extent^2 / 4))`` at the rim.  Matching the coarsest of those
    to one equirect row gives ``size >= 4 * extent * h / pi``.

    A fixed 1024 was not that.  For the reference bedroom's 43 degree cap on a 4096-wide
    canvas it wanted 2052, so the cap ran at half the canvas resolution -- and the two grids
    meet at the hole boundary, which is where every artefact the audit found lives.  Half
    resolution there cost two things that were both measured: the hard known/fill switch
    inside the cap turned into a 2 px staircase across the ceiling light once resampled
    back, and the real pixels the join ramp round-trips came back 10-20 % down on Laplacian
    variance.  It is also why grain could not be matched at the ring's own scale: the cap
    had already low-passed the ring before anything measured it.
    """
    want = 4.0 * float(extent) * float(canvas_h) / math.pi
    size = int(np.clip(math.ceil(want / 64.0) * 64, _CAP_SIZE_MIN, _CAP_SIZE_MAX))
    return size


def _fill_weight(known: np.ndarray) -> np.ndarray:
    """How much of each cap pixel is invented: 1 across the hole, 0 on captured pixels.

    The complement of the cross-fade :func:`_inpaint_cap` uses, and every later correction
    is applied through it, so "the fill" means the same set of pixels to all of them.  A
    correction faded out with a plain blurred hole mask instead would be at half strength
    exactly *on* the boundary, which is the one place it has to be exact: measured on the
    reference bedroom that left 4.9 of the 8.4 grey level step still there.
    """
    w = cv2.GaussianBlur(
        (known > 0).astype(np.float32), (0, 0), _CAP_JOIN_PX, borderType=cv2.BORDER_REPLICATE
    )
    np.clip(w * _CAP_JOIN_BIAS, 0.0, 1.0, out=w)
    return 1.0 - w


def _plane_coords(rays: np.ndarray, axis: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse stereographic: unit directions -> plane ``(X, Y)``."""
    da = rays @ axis
    d1 = rays @ e1
    d2 = rays @ e2
    denom = np.maximum(1.0 + da, 1e-4)
    return (2.0 * d1 / denom).astype(np.float32), (2.0 * d2 / denom).astype(np.float32)


def _polar_edge_row(coverage: np.ndarray, which: str) -> int:
    """Canvas row where the polar hole ends: the first one, walking from the pole, that the
    capture actually reached.  ``-1`` when the whole canvas is empty.

    "Reached" is a quarter of the row's azimuth, not most of it.  A quarter is already
    enough real content for an inpainter to extrapolate a cap from, while a majority test
    cannot tell a genuine pole hole from a wedge that happens to hold half the sphere.
    """
    seen = np.nonzero(coverage.mean(axis=1) / 255.0 >= _CAP_ROW_SEEN)[0]
    if seen.size == 0:
        return -1
    return int(seen.min()) if which == "zenith" else int(seen.max())


def cap_reach_deg(coverage: np.ndarray, which: str) -> float:
    """Colatitude the stereographic patch must reach to include real pixels.

    Found by walking away from the pole for as long as the canvas rows are empty, and
    stopping at the first row where the capture starts to appear.  A fixed reach would be
    wrong in both directions: too small and the patch is *entirely* hole, so the inpainter
    has nothing to work from and returns flat grey; too large and the 1024 px patch spends
    its resolution on the equator instead of on the hole.

    The walk is contiguous, and it stops at the first sign of coverage rather than at the
    first *mostly covered* row, because those are different questions once a capture has
    holes that are not at the poles.  A three-row sweep that missed a 185 degree wedge above
    the horizon has no mostly-covered row anywhere between the zenith and the equator -- the
    wedge holds every one of them just under half -- so the old test walked the "cap" down
    to its 85 degree limit and handed the entire upper hemisphere, wedge included, to a
    conformal patch centred on a pole it has nothing to do with.  A pole cap should end
    where the polar hole ends; everything below that is a job for :func:`fill_uncovered`.
    """
    h = coverage.shape[0]
    edge = _polar_edge_row(coverage, which)
    if edge < 0:
        colat = _MAX_REACH_DEG
    elif which == "zenith":
        colat = (float(edge) + 0.5) / h * 180.0
    else:
        colat = 180.0 - (float(edge) + 0.5) / h * 180.0
    return float(np.clip(colat + _CONTEXT_DEG, 90.0 - _CAP_DEG + 5.0, _MAX_REACH_DEG))


def extract_cap(
    pano: np.ndarray,
    coverage: np.ndarray,
    which: str,
    *,
    size: int | None = None,
    reach_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample a pole cap into a conformal square.  Returns ``(image, known_mask, extent)``."""
    axis, e1, e2 = cap_frame(which)
    if reach_deg is None:
        reach_deg = cap_reach_deg(coverage, which)
    extent = plane_extent(reach_deg)
    if size is None:
        size = cap_size_for(extent, int(pano.shape[0]))
    rays = stereographic_rays(axis, e1, e2, extent, size)
    h, w = pano.shape[:2]
    u, v = g.equirect_from_rays(rays, w, h)
    del rays
    # Coverage-normalised sampling. The canvas is exact black outside coverage, and a plain
    # bilinear sample straddling the coverage edge mixes that black into the ring the
    # inpainter extrapolates from -- then the composite kept those darkened pixels as
    # "known". Measured on the reference capture: a dark dotted line along the whole fill
    # boundary, median 4.6 / p90 30.7 grey levels, 19 % of columns dipping by more than 20.
    # Weighting the sample by coverage and dividing by the sampled coverage removes the
    # black from the average; a sample that is not fully covered is declared unknown so the
    # inpainter, not the join, decides what goes there.
    cov = (coverage > 0).astype(np.float32)
    pre = pano.astype(np.float32) * cov[..., None]
    num = g.sample_equirect(pre, u, v, interpolation=cv2.INTER_LINEAR)
    den = g.sample_equirect(cov, u, v, interpolation=cv2.INTER_LINEAR)
    del u, v, pre, cov
    if den.ndim == 3:
        den = den[..., 0]
    img = to_uint8(num / np.maximum(den, 1e-3)[..., None])
    img[den < 0.5] = 0
    known = np.where(den >= 0.999, np.uint8(255), np.uint8(0))
    del num, den
    known = np.ascontiguousarray(known)
    # The square's corners reach sqrt(2) further from the pole than its edges -- for a
    # 75 deg cap that is content from *below* the horizon.  Letting the inpainter use it
    # drags the fill toward distant colours instead of the ring bounding the hole, so
    # everything past the nominal reach is declared unknown.
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * (2.0 * extent) - extent
    r2 = t[None, :] ** 2 + t[:, None] ** 2
    known[r2 > extent * extent] = 0
    del t, r2
    return img, known, extent


# ======================================================================================
# Radial brightness / colour trend of the ring bounding the hole
# ======================================================================================


@dataclass(slots=True)
class RingTrend:
    """How the covered ring's colour was heading when the capture ran out.

    ``confidence`` is what keeps this honest.  Extrapolating a ten-degree measurement fifty
    degrees to the pole is only defensible when the ring really is following a straight
    line; when it is not -- a wall that darkens into a cornice before the ceiling brightens
    is the common case -- the fit is discarded and the cap simply holds the ring's level.
    """

    colour: np.ndarray  # BGR at the hole boundary
    slope: np.ndarray  # BGR grey levels per degree of elevation, shrunk by its own error
    anchor_deg: float  # elevation of the hole boundary
    confidence: float  # 0..1, from the goodness of fit
    max_delta: float  # absolute cap on the grey levels the extrapolation may travel

    @property
    def pole_colatitude(self) -> float:
        return max(1.0, 90.0 - abs(self.anchor_deg))


def ring_trend(
    pano: np.ndarray,
    coverage: np.ndarray,
    which: str,
    *,
    window_deg: float = _TREND_WINDOW_DEG,
    skip_deg: float = _TREND_SKIP_DEG,
) -> RingTrend | None:
    """Fit mean colour against elevation over the real pixels bounding a pole hole.

    Rows are weighted by how many covered pixels they actually contain, so a half-covered
    row near the edge cannot swing the fit.  Returns ``None`` when there is not enough
    covered data to say anything at all.
    """
    h, w = pano.shape[:2]
    cov = coverage > 0
    per_row = cov.sum(axis=1).astype(np.float64)
    enough = per_row > 0.5 * w
    idx = np.nonzero(enough)[0]
    if idx.size == 0:
        return None

    rows_per_deg = h / 180.0
    skip = int(round(skip_deg * rows_per_deg))
    win = max(_TREND_MIN_ROWS, int(round(window_deg * rows_per_deg)))
    # The ring has to bound the same hole the cap is about to fill, so the boundary comes
    # from the shared polar-edge rule rather than from "the first row that is mostly
    # covered" -- on a capture whose upper hemisphere is half eaten by a wedge those are
    # tens of degrees apart, and the fit would describe a ring the cap never touches.
    polar_edge = _polar_edge_row(coverage, which)
    if which == "zenith":
        edge = polar_edge if polar_edge >= 0 else int(idx.min())
        lo, hi = edge + skip, edge + skip + win
    else:
        edge = polar_edge if polar_edge >= 0 else int(idx.max())
        lo, hi = edge - skip - win + 1, edge - skip + 1
    lo, hi = max(0, lo), min(h, hi)
    sel = np.arange(lo, hi)
    sel = sel[enough[sel]] if sel.size else sel
    if sel.size == 0:
        sel = idx[: _TREND_MIN_ROWS] if which == "zenith" else idx[-_TREND_MIN_ROWS :]
    if sel.size == 0:
        return None

    counts = per_row[sel]
    sums = (pano[sel].astype(np.float64) * cov[sel][..., None]).sum(axis=1)
    means = sums / counts[:, None]
    anchor = 90.0 - 180.0 * (edge + 0.5) / h
    x = (90.0 - 180.0 * (sel + 0.5) / h) - anchor
    del sums

    s0 = float(counts.sum())
    s1 = float((counts * x).sum())
    s2 = float((counts * x * x).sum())
    t0 = (counts[:, None] * means).sum(axis=0)
    t1 = (counts[:, None] * x[:, None] * means).sum(axis=0)
    det = s0 * s2 - s1 * s1
    if sel.size < _TREND_MIN_ROWS or abs(det) < 1e-9:
        return RingTrend(t0 / max(s0, 1e-9), np.zeros(3), float(anchor), 0.0, 0.0)
    c0 = (s2 * t0 - s1 * t1) / det
    slope = np.clip((s0 * t1 - s1 * t0) / det, -_TREND_MAX_SLOPE, _TREND_MAX_SLOPE)

    # Weighted R^2 on luminance: does a straight line actually describe this ring?
    lum = means.mean(axis=1)
    fit = float(c0.mean()) + float(slope.mean()) * x
    mean_lum = float((counts * lum).sum() / s0)
    ss_res = float((counts * (lum - fit) ** 2).sum())
    ss_tot = float((counts * (lum - mean_lum) ** 2).sum())
    r2 = 0.0 if ss_tot <= 1e-9 else max(0.0, 1.0 - ss_res / ss_tot)
    confidence = float(
        np.clip((r2 - _TREND_MIN_R2) / max(_TREND_FULL_R2 - _TREND_MIN_R2, 1e-6), 0.0, 1.0)
    )

    # Shrink each channel's slope by its own standard error before anyone extrapolates it.
    #
    # ``counts`` are reliability weights, not observation counts -- a row is one measurement
    # of the ring however many pixels went into its mean -- so they are normalised to mean 1
    # before the residual variance is formed.  Weighting them as frequencies would divide by
    # the pixel count and report a standard error some thirty times too small.
    wn = counts / max(float(counts.mean()), 1e-9)
    n0 = float(wn.sum())
    n1 = float((wn * x).sum())
    n2 = float((wn * x * x).sum())
    dn = n0 * n2 - n1 * n1
    resid = means - (c0[None, :] + x[:, None] * slope[None, :])
    sigma2 = (wn[:, None] * resid**2).sum(axis=0) / max(sel.size - 2, 1)
    se = np.sqrt(np.maximum(sigma2, 0.0) * n0 / max(dn, 1e-9))
    shrunk = np.sign(slope) * np.maximum(np.abs(slope) - _TREND_SLOPE_SIGMA * se, 0.0)
    # A slope pinned at the clamp is not a measurement of the ceiling, it is a lamp, a
    # window or a cornice shadow inside the ten-degree window; and however good the fit,
    # the extrapolation may not run the fill into saturation. Measured on the reference
    # capture: a +4/deg clamped slope carried 36 degrees pushed the ring's 240 to a flat
    # 255 over the whole cap -- the "white blob" -- while the photographed ceiling beside
    # it averaged 220 with 4.5 % of pixels at 254 or above.
    hit_clamp = bool(np.any(np.abs(slope) >= _TREND_MAX_SLOPE - 1e-9))
    if hit_clamp:
        confidence *= 0.3
    headroom = float(np.clip(250.0 - float(c0.mean()), 0.0, 255.0))
    max_delta = float(min(_TREND_MAX_DELTA, 0.5 * headroom))
    _LOG.info(
        "poles: %s ring trend slope %.3f/deg (+/-%.3f -> %.3f), R2 %.3f -> confidence %.2f, "
        "clamped=%s, max travel %.0f",
        which, float(slope.mean()), float(se.mean()), float(shrunk.mean()), r2, confidence,
        hit_clamp, max_delta,
    )
    del means, counts, x, lum, resid, wn
    return RingTrend(c0, shrunk, float(anchor), confidence, max_delta)


def trend_colour(trend: RingTrend, elev_deg) -> np.ndarray:
    """Evaluate a fitted ring trend at one or many elevations.  Returns ``(..., 3)`` BGR.

    The extrapolation is tapered so its derivative reaches zero exactly at the pole.  That
    is not a fudge: the azimuthal mean of any continuous field on a sphere *must* have zero
    radial derivative at the pole, or the surface has a cone singularity there.  A straight
    line carried to the pole violates that and overshoots; the sine taper matches the
    fitted slope at the ring, flattens as it approaches the pole, and travels 2/pi of the
    linear distance on the way.
    """
    d = np.asarray(elev_deg, dtype=np.float64) - trend.anchor_deg
    if trend.confidence <= 0.0:
        return np.broadcast_to(trend.colour, d.shape + (3,)).copy()
    span = trend.pole_colatitude
    u = np.clip(np.abs(d), 0.0, span)
    eff = np.sign(d) * (2.0 * span / np.pi) * np.sin(np.pi * u / (2.0 * span))
    delta = eff[..., None] * trend.slope * trend.confidence
    np.clip(delta, -trend.max_delta, trend.max_delta, out=delta)
    return np.clip(trend.colour + delta, 0.0, 255.0)


def _radial_lookup(prof: np.ndarray, r: np.ndarray, rmax: float) -> np.ndarray:
    """Evaluate a binned radial profile at every pixel, interpolating between bins.

    Nearest-bin indexing quantises the profile into concentric annuli of equal *radius*,
    and near the pole a few pixels of radius is a whole bin of value.  On the reference
    bedroom's 2048 cap that showed as three or four concentric rings drawn around the
    zenith, inside a fill whose entire claim is that it is smooth.
    """
    nb = int(prof.shape[0])
    centres = ((np.arange(nb, dtype=np.float32) + 0.5) / nb * rmax).astype(np.float32)
    flat_r = r.ravel().astype(np.float32)
    out = np.empty((flat_r.size, prof.shape[1]), dtype=np.float32)
    for c in range(prof.shape[1]):
        out[:, c] = np.interp(flat_r, centres, prof[:, c].astype(np.float32))
    return out.reshape(r.shape + (prof.shape[1],))


def _apply_radial_trend(
    cap: np.ndarray,
    known: np.ndarray,
    extent: float,
    which: str,
    trend: RingTrend,
) -> np.ndarray:
    """Re-level an inpainted cap so its radial profile follows the ring's trend.

    Works entirely on azimuthal averages: the cap's own profile against radius is measured,
    the trend's profile is evaluated at the same radii, and the difference is added back.
    Being a function of colatitude alone, it can brighten or warm the fill but cannot
    introduce structure that was never observed.

    The correction is anchored -- forced to zero -- at one radius, and *which* radius sets
    the level of the entire fill: the azimuthal means cancel everywhere else, so the pole
    ends up at ``profile(anchor) + trend travel``.  The anchor is therefore the innermost
    annulus that is made of nothing but captured pixels, measured over those pixels alone.
    The obvious choice -- the outermost annulus that is still mostly hole -- is a trap: it
    straddles the coverage boundary, so its mean mixes the inpainter's guess with the mask
    edge, where the blender has been feathering real content into the black outside.  On the
    room fixture that edge reads 17 grey levels darker than the clean ring one annulus
    further out, and anchoring there dragged the whole ceiling down with it.  This is the
    same contamination ``_TREND_SKIP_DEG`` keeps out of the fit itself.
    """
    size = cap.shape[0]
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * (2.0 * extent) - extent
    r = np.sqrt(t[None, :] ** 2 + t[:, None] ** 2)
    hole = (known == 0) & (r <= extent)
    if not hole.any():
        return cap

    nb = _TREND_BINS
    rmax = float(r.max()) + 1e-6
    bins = np.clip((r / rmax * nb).astype(np.int32), 0, nb - 1)
    flat = bins.ravel()
    counts = np.bincount(flat, minlength=nb).astype(np.float64)
    hole_frac = np.bincount(flat, weights=hole.ravel().astype(np.float64), minlength=nb)
    hole_frac /= np.maximum(counts, 1.0)
    mostly_hole = np.nonzero(hole_frac >= 0.5)[0]
    if mostly_hole.size == 0:
        return cap
    ring_bin = int(mostly_hole.max())

    real = ((known > 0) & (r <= extent)).ravel().astype(np.float64)
    real_n = np.bincount(flat, weights=real, minlength=nb)
    prof = np.empty((nb, 3), dtype=np.float64)
    prof_real = np.empty((nb, 3), dtype=np.float64)
    for c in range(3):
        band = cap[..., c].astype(np.float64).ravel()
        prof[:, c] = np.bincount(flat, weights=band, minlength=nb) / np.maximum(counts, 1.0)
        prof_real[:, c] = np.bincount(flat, weights=band * real, minlength=nb) / np.maximum(
            real_n, 1.0
        )
        del band
    del flat, counts, real

    colat = np.degrees(2.0 * np.arctan((np.arange(nb) + 0.5) / nb * rmax / 2.0))
    elev = (90.0 - colat) if which == "zenith" else (colat - 90.0)
    target = trend_colour(trend, elev)

    # Innermost annulus made purely of captured pixels, searched outward from the hole.
    clean = np.nonzero(
        (hole_frac <= _TREND_CLEAN_FRAC)
        & (real_n >= _TREND_CLEAN_MIN_PX)
        & (np.arange(nb) > ring_bin)
    )[0]
    delta = target - prof
    if clean.size:
        anchor_bin = int(clean.min())
        delta -= target[anchor_bin] - prof_real[anchor_bin]
    else:
        # No annulus is clean enough to read a level off, so fall back to continuity with
        # whatever the cap already has at the hole's edge.
        anchor_bin = ring_bin
        delta -= delta[ring_bin]
    np.clip(delta, -trend.max_delta, trend.max_delta, out=delta)
    # Nothing outside the hole may move, and the correction must stay low-frequency.
    delta[ring_bin + 1 :] = 0.0
    delta = cv2.GaussianBlur(delta.astype(np.float32).reshape(nb, 1, 3), (1, 9), 0).reshape(nb, 3)
    del target, prof, prof_real, colat, elev, real_n

    soft = cv2.GaussianBlur(
        hole.astype(np.float32), (0, 0), max(1.0, size / 192.0), borderType=cv2.BORDER_REPLICATE
    )
    out = cap.astype(np.float32) + _radial_lookup(delta, r, rmax) * soft[..., None]
    # The inpainter carries the ring's azimuthal light/dark arcs straight to the pole as
    # cloud-like lobes (measured +/-11 grey levels of low-passed azimuthal variation inside
    # the fill). Nothing above the ring was photographed, so inside the hole the fill is
    # pulled to its own azimuthal mean profile.
    #
    # The fade is a smoothstep in radius now, not a linear ramp between two radii.  Both
    # ends of that ramp were kinks in an otherwise smooth field, and the outer one --
    # 60 % of the way out to the ring -- was visible: with the inpainter resolved coarsely
    # enough to have no creases of its own (see ``_CAP_HOLE_SAMPLES``) what remains inside
    # the hole is Telea's front, whose iso-levels are inward offsets of the coverage
    # polygon, and the ramp drew a hard circle across those facets.  A smoothstep keeps
    # every bit of the inpainter's azimuthal structure at the ring, where it is what makes
    # the fill continuous with the real pixels, and none of it at the pole, where nothing
    # was photographed -- with no radius in between that a viewer can pick out.
    prof_now = np.zeros((nb, 3), dtype=np.float32)
    cnt_now = np.bincount(bins.ravel(), minlength=nb).astype(np.float32)
    for c in range(3):
        prof_now[:, c] = np.bincount(bins.ravel(), weights=out[..., c].ravel(), minlength=nb) / np.maximum(cnt_now, 1.0)
    r_ring = max((ring_bin + 0.5) / nb * rmax, 1e-6)
    flat_w = (1.0 - _pole_fade(r, r_ring)) ** _TREND_FLATTEN_EXP * soft
    out += (_radial_lookup(prof_now, r, rmax) - out) * flat_w[..., None]
    del delta, soft, bins, hole, r, prof_now, cnt_now, flat_w
    _LOG.info(
        "poles: %s fill re-levelled toward %s at the pole",
        which,
        np.round(trend_colour(trend, 90.0 if which == "zenith" else -90.0), 1),
    )
    return to_uint8(out)


# ======================================================================================
# Inpainting
# ======================================================================================


#: How the cap's hole is filled where FSR cannot reach: "harmonic" (membrane interpolation,
#: see :func:`_harmonic_inpaint`) or "telea" (OpenCV's fast-marching inpaint).
_CAP_FILL = "harmonic"


def _harmonic_inpaint(img: np.ndarray, known: np.ndarray) -> np.ndarray | None:
    """Fill the unknown pixels with the smoothest surface through the known ones.

    Solves Laplace's equation over the hole with the captured pixels as fixed boundary
    values: every filled pixel is the average of its four neighbours. ``None`` when the
    system cannot be solved, so the caller falls back to Telea.

    WHY NOT TELEA. Telea marches inward from the boundary and transports each boundary
    pixel's value along the inward normal -- on a disc-shaped hole, radially. Whatever the
    capture's last row happened to clip at the rim is therefore drawn as a ray to the pole: a
    ceiling lamp cut by the coverage edge became bright streaks across the synthesised ceiling,
    and the rug and blanket around the nadir fanned out round the logo like a zoom blur.
    A harmonic fill cannot do either. It obeys the maximum principle -- nothing inside is
    brighter or darker than the boundary that bounds it -- and a boundary feature's influence
    decays inward as a soft glow rather than travelling as a line.

    It is also the right object on the sphere, not just in the plane: the cap is a
    stereographic (conformal) projection, and the 2-D Laplacian is conformally invariant, so a
    harmonic fill of the cap IS the harmonic fill of that piece of sphere.
    """
    import scipy.sparse as sp  # noqa: PLC0415 - only on this path
    import scipy.sparse.linalg as spla  # noqa: PLC0415

    h, w = known.shape[:2]
    unk = known == 0
    n = int(unk.sum())
    if n == 0:
        return img.copy()
    known_any = ~unk
    if not known_any.any():
        return None
    idx = np.full((h, w), -1, dtype=np.int64)
    idx[unk] = np.arange(n, dtype=np.int64)
    ys, xs = np.nonzero(unk)
    src = img.astype(np.float64)
    channels = src.shape[2] if src.ndim == 3 else 1
    src = src.reshape(h, w, channels)
    rhs = np.zeros((n, channels), dtype=np.float64)
    deg = np.zeros(n, dtype=np.float64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    me = np.arange(n, dtype=np.int64)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        deg += inside
        nyi, nxi = ny[inside], nx[inside]
        q = idx[nyi, nxi]
        p = me[inside]
        free = q >= 0
        rows.append(p[free])
        cols.append(q[free])
        fixed = ~free
        np.add.at(rhs, p[fixed], src[nyi[fixed], nxi[fixed]])
    # A whisper of pull toward the captured mean keeps an island with no captured neighbour
    # (pure Neumann, singular) solvable; at 1e-6 it is invisible everywhere else.
    eps = 1e-6
    rhs += eps * src[known_any].mean(axis=0)[None, :]
    r = np.concatenate([me] + rows)
    c = np.concatenate([me] + cols)
    v = np.concatenate([deg + eps] + [np.full(a.size, -1.0) for a in rows])
    A = sp.csc_matrix((v, (r, c)), shape=(n, n))
    try:
        lu = spla.splu(A)
        sol = lu.solve(rhs)
    except Exception as exc:  # noqa: BLE001 - Telea is always there behind this
        _LOG.warning("poles: harmonic fill failed (%s), using Telea", exc)
        return None
    out = src.copy()
    out[unk] = sol
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out.reshape(img.shape)


def _inpaint_cap(img: np.ndarray, known: np.ndarray, extent: float | None = None) -> np.ndarray:
    """Fill ``img`` wherever ``known`` is 0: FSR where it can reach across, otherwise the
    harmonic (membrane) fill, with Telea as the last resort.

    Harmonic rather than Telea, measured on the reference bedroom: with the rest of the cap
    pipeline unchanged, Telea's front left faceted wedges in the zenith and a yellow-black fan
    from the blanket at the nadir, and the "deep streak" energy halfway to the pole read
    0.71 / 1.46 grey levels (zenith / nadir) against 0.64 / 1.23 for the membrane.

    What it costs, on the holdout bench (10 deg of real pixels hidden and refilled): the
    brightness step across the new edge rises -- 4.6 -> 4.8 grey levels RMS at the zenith,
    2.3 -> 4.6 at the nadir -- because a membrane meets a textured ring as a smooth surface
    where Telea copies the ring's own pixels into the first rows. PSNR of the refilled band
    is a wash (27.5 -> 27.3 dB zenith, 17.6 -> 18.1 dB nadir). A soft step was judged the
    lesser artefact than a fan; `_CAP_FILL = "telea"` restores the other side of the trade.

    The working scale is chosen from how far the fill actually has to travel rather than
    fixed, and the join back onto the real pixels is feathered at the cap's own resolution.
    Both are measured: see ``_CAP_HOLE_SAMPLES`` and ``_CAP_JOIN_PX``.
    """
    hole_frac = float((known == 0).mean())
    if hole_frac <= 1e-6:
        return img
    if hole_frac > 0.995:
        # Nothing at all to extrapolate from; a flat mid-grey is the honest answer.
        out = img.copy()
        out[:] = 128
        return out

    full = int(img.shape[0])
    # Diameter of the disc with the hole's area.  A pole hole is a disc, so this is the
    # distance the fill has to carry, and it is what decides both the working scale and
    # whether FSR can reach across at all.  The square's corners are marked unknown too and
    # are not part of it: counting them made the reference bedroom's zenith hole read as
    # 1003 cap pixels across when the hole itself is 845, and the working scale followed.
    if extent is None:
        hole_area = float((known == 0).sum())
    else:
        hole_area = float(((known == 0) & (_cap_radius(full, extent) <= extent)).sum())
    hole_width = 2.0 * math.sqrt(hole_area / math.pi)
    size = int(
        np.clip(
            round(full * _CAP_HOLE_SAMPLES / max(hole_width, 1.0)),
            min(_CAP_INPAINT_MIN, full),
            min(_INPAINT_SIZE, full),
        )
    )
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    kn = cv2.resize(known, (size, size), interpolation=cv2.INTER_AREA)
    kn = np.where(kn > 200, np.uint8(255), np.uint8(0))
    if not (kn > 0).any():
        kn[:] = 0
        filled = np.full_like(small, 128)
    else:
        filled = None
        # FSR reconstructs *structure* from surrounding frequencies, which is what you
        # want for a hole punched in textured content.  Past about two thirds missing
        # there is no surrounding structure left to reconstruct from and its output starts
        # inventing; Telea's boundary-inward diffusion degrades far more gracefully into
        # "the colour of the ring, smoothly continued", which is the honest answer for a
        # ceiling nobody photographed.
        #
        # The reach test is the same one the equirect filler applies, and it was missing
        # here.  A pole cap is the one place the hole is guaranteed to be wider than FSR's
        # search window, so on every real capture this branch was taken and could not
        # possibly work: measured on the reference bedroom, 422 working pixels to cross
        # against a twelve pixel reach.  What came back was the radial creasing the audit
        # reported, not a reconstruction.
        reach_px = hole_width * (size / float(full))
        if hole_frac <= 0.65 and reach_px <= _FSR_REACH_PX:
            algo = cv2.xphoto.INPAINT_FSR_BEST if hole_frac < 0.45 else cv2.xphoto.INPAINT_FSR_FAST
            try:
                dst = np.zeros_like(small)
                cv2.xphoto.inpaint(small, kn, dst, algo)
                if np.isfinite(dst).all() and dst.shape == small.shape and _fsr_took(dst, kn):
                    filled = dst
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("poles: xphoto FSR inpaint failed (%s), using Telea", exc)
        if filled is None and _CAP_FILL == "harmonic":
            filled = _harmonic_inpaint(small, kn)
        if filled is None:
            hole = cv2.dilate((kn == 0).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
            filled = cv2.inpaint(small, hole, 5.0, cv2.INPAINT_TELEA)
            del hole
    _LOG.info(
        "poles: cap %dpx inpainted at %d (hole %.0f px across, %.0f%% of the patch)",
        full, size, hole_width, 100.0 * hole_frac,
    )
    del small, kn

    up = cv2.resize(filled, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC).astype(
        np.float32
    )
    del filled
    # Cross-fade rather than switch, over the shared fill weight, so that a later correction
    # applied through the same weight lands on exactly the pixels the fill occupies.
    fw = _fill_weight(known)
    src = np.where((known > 0)[..., None], img, up.astype(np.uint8)).astype(np.float32)
    out = to_uint8(up * fw[..., None] + src * (1.0 - fw)[..., None])
    del up, fw, src
    return np.ascontiguousarray(out)


# ======================================================================================
# Making the boundary disappear: level, then grain
# ======================================================================================
#
# ``_apply_radial_trend`` sets the fill's level as a function of colatitude alone, which is
# the right shape for the *pole* and the wrong one for the *boundary*.  A ceiling light just
# inside the hole throws a glow that is 6 grey levels brighter on one side of the cap than
# the other; an azimuthal mean cannot represent that, so the fill meets the real ring with a
# step whose size depends on which way you are looking.  Measured on the reference bedroom:
# 225.1 just inside the boundary against 228.5 just outside it, and the eye reads a
# 3.4 level ring on a flat ceiling as a painted-on dome edge.
#
# So the two things that make a boundary visible are corrected against the real pixels
# themselves, at every azimuth: its local mean, and its grain.  Neither invents structure --
# the level correction is a heavily low-passed difference field and the grain is zero-mean
# noise -- but between them the fill arrives at the ring at the ring's own brightness and
# with the ring's own roughness, which is what "no visible boundary" actually requires.

#: The correction is deliberately low-frequency, so it is measured and spread on a copy of
#: the cap no larger than this.  A 2048 cap in three float32 channels is 50 MB and the field
#: needs half a dozen of them; at 512 the whole thing costs 3 MB and a fifth of a second,
#: and nothing it can represent is lost -- the smallest feature in it is 32 working pixels
#: across.
_BOUNDARY_WORK = 1024
#: Sigma of the two one-sided local means, in degrees of sphere.
#:
#: This one has to be SMALL, and that is not obvious.  Each mean is one-sided -- the real
#: one looks outward, the fill's inward -- so on a field with a radial gradient each is
#: biased by (gradient x offset), in opposite directions, and the difference inherits both.
#: The ceiling around the reference bedroom's zenith hole is exactly that field: the lamp's
#: glow falls 8 grey levels over the first 2.4 degrees outside the boundary.  Measured: at
#: 1.6 deg the correction matched the fill to the ring's two-degree *average* and left the
#: boundary 4.9 levels apart -- worse than the 3.4 it started from.  At 0.5 deg the two
#: means are both close to the boundary value and the bias is about one level.  The price of
#: a narrow kernel is a noisy difference, and the wide spread below is what pays it.
_BOUNDARY_LP_DEG = 0.5
#: Azimuth bins the rim mismatch is reduced to, and how far it is smoothed around the
#: circle.  Twenty degrees is wide enough that a cornice crossing the rim at one azimuth
#: cannot become a lobe, and narrow enough to follow a lamp's glow, which on the reference
#: bedroom spans about a quadrant.
_BOUNDARY_AZ_BINS = 360
_BOUNDARY_AZ_SMOOTH_DEG = 20.0
#: How much of the azimuthal part of the mismatch is carried inward.  See the measurement
#: in :func:`_match_boundary_level`.
_BOUNDARY_AZ_GAIN = 1.0
#: Ceiling on the correction, per channel.  A mismatch larger than this is not a lamp, it is
#: the inpainter having failed, and dragging the whole cap after it would make things worse.
_BOUNDARY_MAX_STEP = 24.0
#: How azimuthal detail fades toward the pole, for the trend's flattening and the grain.
#:
#: A smoothstep in ``r / r_edge`` running the whole way, not from some inner radius
#: outward: a linear fade beginning at half the boundary radius has a kink in its derivative
#: there, and on the reference bedroom's zenith that kink was plainly visible as a second,
#: circular edge inside the polygonal one.  It is NOT how the boundary-level correction
#: decays any more -- held at nearly full strength most of the way in, that turned every rim
#: feature into a ray; see :func:`_harmonic_extension`.
def _pole_fade(r: np.ndarray, r_edge: float) -> np.ndarray:
    t = np.clip(r / max(r_edge, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)

#: Correlation length of the grain, and the width of the annulus either side of the boundary
#: the match is measured over -- both in degrees of sphere, so they mean the same thing
#: whatever resolution the cap ended up at.
#:
#: The grain is LOW-passed white noise, not high-passed.  High-passing was the first
#: attempt and it is exactly wrong: it leaves every frequency above the cutoff, Nyquist
#: included, so most of the amplitude sat where the cap-to-canvas resample, the JPEG
#: quantiser and the viewer's own downscale each take their cut.  Measured on the reference
#: bedroom, a cap carrying 1.12 of local deviation delivered 0.86 into the PNG and 0.69 into
#: the JPEG.  Low-passed noise puts the same amplitude at a scale all three keep.
_GRAIN_SCALE_DEG = 0.12
_GRAIN_BAND_DEG = 2.2
#: Window the local standard deviation is measured over, in degrees.  0.55 deg is the 7 px
#: window the audit's numbers were taken with, on a 4096-wide delivery.
_GRAIN_WINDOW_DEG = 0.55
#: How far past the hole's edge the canvas-side ring measurement starts, in degrees.  Far
#: enough out that the join ramp and the blend's own mask edge are not in the window.
_GRAIN_RING_SKIP_DEG = 0.6
#: Hard ceiling on the grain amplitude.  Outside a cap boundary the "local standard
#: deviation" of real content is mostly *structure* -- a rug's pattern, a cornice -- and
#: matching a rug with noise would be a worse lie than leaving the fill smooth.  Matching
#: the first few grey levels is what removes the step; matching all of it is not the goal.
_GRAIN_MAX = 7.0
#: Grain fades toward the pole on the same smoothstep (:func:`_pole_fade`), so the cap
#: around the pole stays as smooth as it has always been -- ``test_poles_are_filled_
#: plausibly`` measures exactly that, and a fill that fabricates detail at the pole deserves
#: to fail it.
#:
#: Fixed seed, so two renders of the same capture are the same image.
_GRAIN_SEED = 0x50_1E_5


def _cap_radius(size: int, extent: float) -> np.ndarray:
    """``(size, size)`` stereographic plane radius for a cap grid."""
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * (2.0 * extent) - extent
    return np.sqrt(t[None, :] ** 2 + t[:, None] ** 2)


def _cap_deg_per_px(size: int, extent: float, r: float) -> float:
    """Angular size of one cap pixel at plane radius ``r``, in degrees."""
    return math.degrees((2.0 * extent / size) * 2.0 / (1.0 + 0.25 * r * r))


def _cap_masks(known: np.ndarray, extent: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(radius, hole, real, boundary_radius)`` for a cap.

    ``boundary_radius`` is the mean plane radius of the hole's rim, which is the natural
    unit for "how far in are we" on a cap whose hole is a polygon rather than a circle.
    """
    size = int(known.shape[0])
    r = _cap_radius(size, extent)
    inside = r <= extent
    hole = (known == 0) & inside
    real = (known > 0) & inside
    rim = hole & (cv2.dilate((real).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    if rim.any():
        r_edge = float(r[rim].mean())
    elif hole.any():
        r_edge = float(r[hole].max())
    else:
        r_edge = float(extent)
    return r, hole, real, max(r_edge, 1e-6)


#: Radial samples of the polar grid the harmonic extension is evaluated on.
_HARMONIC_RADIAL_SAMPLES = 160
#: How far inside the rim the boundary correction holds its full strength before it starts
#: to decay harmonically, in degrees of sphere. Without a plateau the fast azimuthal modes
#: are already half gone a degree in, which is where a viewer compares the fill with the
#: ring. It buys back little, and the bench says why: on the reference capture (10 deg of
#: real pixels hidden and refilled, per-sector step 1.2 deg either side of the new edge, grey
#: levels RMS) the nadir read 2.8 with the old Telea path, 2.3 with Telea plus this harmonic
#: correction, 4.7 with the membrane fill and no plateau, 4.6 with it. The step is the price
#: of the membrane FILL (see `_inpaint_cap`), not of this extension, and it is paid for the
#: fan and the wedges that fill removes.
_BOUNDARY_PLATEAU_DEG = 1.5


def _harmonic_extension(
    values: np.ndarray,
    r: np.ndarray,
    hole: np.ndarray,
    real: np.ndarray,
    idx: np.ndarray,
    r_edge: float,
    plateau: float = 0.0,
) -> np.ndarray:
    """The harmonic function on the cap whose rim values are ``values`` (``(nb, C)``).

    ``idx`` is each pixel's azimuth bin. The rim is not a circle -- the hole is the union of
    camera frusta -- so each azimuth is measured against its own rim radius ``R(theta)``
    (the mean radius of the hole pixels touching captured ones in that bin, smoothed round
    the circle; ``r_edge`` where a bin has none), and pixels at or beyond it take the rim
    value itself. ``plateau`` (plane units) holds the full rim value that far inside the
    rim before the decay starts. Returns ``(size, size, C)`` float32.
    """
    nb, channels = values.shape
    rim = hole & (cv2.dilate(real.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    cnt = np.bincount(idx[rim].ravel(), minlength=nb).astype(np.float64)
    tot = np.bincount(idx[rim].ravel(), weights=r[rim].ravel().astype(np.float64), minlength=nb)
    radius = np.where(cnt > 0, tot / np.maximum(cnt, 1.0), r_edge)
    k = int(max(3, round(_BOUNDARY_AZ_SMOOTH_DEG / 360.0 * nb)) | 1)
    pad = np.concatenate([radius[-k:], radius, radius[:k]]).astype(np.float32).reshape(1, -1)
    radius = cv2.GaussianBlur(pad, (k, 1), 0).ravel()[k : k + nb]
    radius = np.maximum(radius, 1e-6)

    coeffs = np.fft.rfft(values.astype(np.float64), axis=0)  # (nb // 2 + 1, C)
    modes = np.arange(coeffs.shape[0], dtype=np.float64)[:, None]
    rho = np.linspace(0.0, 1.0, _HARMONIC_RADIAL_SAMPLES)
    polar = np.empty((rho.size, nb + 1, channels), dtype=np.float32)
    for i, p in enumerate(rho):
        ring = np.fft.irfft(coeffs * (p ** modes), n=nb, axis=0)
        polar[i, :nb] = ring
        polar[i, nb] = ring[0]  # closes the circle for the interpolation below
    # Continuous azimuth bin of every pixel, matching how `idx` was quantised.
    size = r.shape[0]
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * 2.0 - 1.0
    theta = np.arctan2(np.broadcast_to(t[:, None], (size, size)), np.broadcast_to(t[None, :], (size, size)))
    az = ((theta + math.pi) / (2.0 * math.pi) * nb - 0.5).astype(np.float32) % nb
    inner = np.maximum(radius - float(plateau), 0.5 * radius)
    rel = np.clip(r / inner[idx], 0.0, 1.0).astype(np.float32) * (rho.size - 1)
    out = np.stack(
        [cv2.remap(np.ascontiguousarray(polar[..., c]), az, rel, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE) for c in range(channels)],
        axis=-1,
    )
    return out.astype(np.float32)


def _match_boundary_level(
    cap: np.ndarray, known: np.ndarray, extent: float, which: str
) -> np.ndarray:
    """Add the fill's mismatch against the real ring back into the fill, in POLAR form.

    Both sides of the boundary are reduced to a local mean with the same narrow kernel --
    one over hole pixels only, one over real pixels only -- so their difference on the rim
    is exactly "how much darker the fill arrives than the ceiling it is joining".  That
    difference is then collapsed to a function of AZIMUTH ALONE, smoothed around the circle,
    and carried inward as its harmonic extension (see :func:`_harmonic_extension` and the
    note at the call below: carried in as ``m(theta) * fade(r)`` it drew every rim feature
    as a ray to the pole).

    Collapsing it is the whole design, and the first version did not.  Spreading the
    mismatch inward with a wide isotropic blur is a diffusion from the boundary, and the
    boundary is a polygon -- the union of the camera frusta -- so the correction came back
    with the polygon's facets in it and painted a faceted plateau in the middle of the cap
    that was more visible than the edge it had just fixed.  (Telea does the same thing for
    the same reason; see ``_apply_radial_trend``.)  A field that is smooth in ``theta`` and
    smooth in ``r`` has no facets available to it, whatever shape the hole is.

    The azimuthal part must vanish at the pole -- every azimuth meets there, so anything
    depending on ``theta`` is multi-valued at ``r = 0`` -- and the harmonic extension takes
    every mode k >= 1 there by construction.  The azimuthal *mean* of the mismatch does not
    vanish: that is a level error in the fill, and it is carried the whole way in.
    """
    full = int(cap.shape[0])
    size = min(_BOUNDARY_WORK, full)
    small = cv2.resize(cap, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    kn = cv2.resize(known, (size, size), interpolation=cv2.INTER_AREA)
    r, hole, real, r_edge = _cap_masks(np.where(kn > 127, np.uint8(255), np.uint8(0)), extent)
    del kn
    if not hole.any() or not real.any():
        del small, r, hole, real
        return cap

    deg_px = _cap_deg_per_px(size, extent, r_edge)
    s_lp = max(1.5, _BOUNDARY_LP_DEG / deg_px)

    def blur(a: np.ndarray, s: float) -> np.ndarray:
        return cv2.GaussianBlur(a, (0, 0), s, borderType=cv2.BORDER_REPLICATE)

    hw = hole.astype(np.float32)
    rw = real.astype(np.float32)
    dh, dr = blur(hw, s_lp), blur(rw, s_lp)
    diff = blur(small * hw[..., None], s_lp) / np.maximum(dh, 1e-4)[..., None]
    diff = blur(small * rw[..., None], s_lp) / np.maximum(dr, 1e-4)[..., None] - diff
    # Weight: only where both sides are actually represented, i.e. on the rim itself.
    w = (dh * dr).astype(np.float32)
    w[~hole] = 0.0
    del dh, dr, hw, rw, small

    t = (np.arange(size, dtype=np.float32) + 0.5) / size * (2.0 * extent) - extent
    theta = np.arctan2(
        np.broadcast_to(t[:, None], (size, size)), np.broadcast_to(t[None, :], (size, size))
    )
    nb = _BOUNDARY_AZ_BINS
    idx = np.clip(((theta + math.pi) / (2.0 * math.pi) * nb).astype(np.int32), 0, nb - 1)
    del theta
    flat_i, flat_w = idx.ravel(), w.ravel()
    den = np.bincount(flat_i, weights=flat_w, minlength=nb)
    if float(den.sum()) <= 0.0:
        del r, hole, real, diff, w, idx, flat_i, flat_w
        return cap
    m = np.stack(
        [np.bincount(flat_i, weights=flat_w * diff[..., c].ravel(), minlength=nb) for c in range(3)],
        axis=1,
    )
    del flat_i, flat_w, diff, w
    # Bins with no rim in them (a hole that does not surround the pole) take the circle's
    # average rather than zero, so an absent measurement cannot pull the fill anywhere.
    seen = den > 1e-6
    m[seen] /= den[seen][:, None]
    if not seen.all():
        m[~seen] = m[seen].mean(axis=0)
    # Cyclic smoothing: the mismatch is a lamp's glow or a wall's shading, not a texture.
    k = int(max(3, round(_BOUNDARY_AZ_SMOOTH_DEG / 360.0 * nb)) | 1)
    pad = np.concatenate([m[-k:], m, m[:k]], axis=0).astype(np.float32)[None, ...]
    m = cv2.GaussianBlur(pad, (1, k), 0)[0, k : k + nb]
    del pad, seen, den
    np.clip(m, -_BOUNDARY_MAX_STEP, _BOUNDARY_MAX_STEP, out=m)

    m0 = m.mean(axis=0)
    # THE AZIMUTHAL PART IS EXTENDED HARMONICALLY, NOT CARRIED STRAIGHT IN. The first version
    # of this correction was ``m(theta) * smoothstep(r / r_edge)``: whatever the rim measured
    # at an azimuth, at nearly full strength most of the way to the pole. That is a RAY, and
    # on the reference capture it was the sunburst the review saw -- the ceiling lamp's arcs
    # and the cornice drawn as bright spokes across the synthesised ceiling, the rug and the
    # blanket fanned out round the nadir mark. (Dumping the cap after each stage showed the
    # inpainted fill smooth and this step adding every spoke.)
    #
    # The smooth function that takes given values on a circle is the harmonic one: Fourier
    # mode k of the rim mismatch falls off as (r / R)^k (the Poisson integral). The mean
    # (k = 0) is a level error and still reaches the pole, a one-sided glow (k = 1) tilts
    # the whole cap gently, and a lamp's 20-degree bump (k ~ 18) is gone within a tenth of
    # the radius -- a soft halo at the rim instead of a spoke. It also vanishes at the pole
    # for every k >= 1, which is what ``_pole_fade`` had to enforce by hand.
    plateau = _BOUNDARY_PLATEAU_DEG / deg_px * (2.0 * extent / size)
    corr = (
        _BOUNDARY_AZ_GAIN
        * _harmonic_extension(m - m0[None, :], r, hole, real, idx, r_edge, plateau=plateau)
        + m0[None, None, :]
    )
    reported = (float(m0.mean()), float(np.sqrt((corr[hole] ** 2).mean())))
    del idx, m, r, hole, real

    corr = cv2.resize(corr, (full, full), interpolation=cv2.INTER_LINEAR)
    # The correction belongs to the fill, not to the captured pixels: applying it on one
    # side of the boundary and not on the other is precisely what removes the step, and the
    # shared fill weight is what says which side a pixel is on.
    out = to_uint8(cap.astype(np.float32) + corr * _fill_weight(known)[..., None])
    _LOG.info(
        "poles: %s boundary level matched, %+.2f mean / %.2f rms grey levels",
        which, reported[0], reported[1],
    )
    del corr
    return out


def _ring_roughness(pano: np.ndarray, coverage: np.ndarray, which: str) -> float:
    """Typical local standard deviation of the real pixels ringing a pole hole.

    Measured on the CANVAS, not on the cap, and that is the whole point of the function.
    The conformal patch resamples the ring on its way in -- at the rim one cap pixel spans
    1.5 canvas columns -- so a roughness read off the cap is a roughness the cap itself
    reduced, and grain matched to it lands about half as rough as the ceiling beside it.
    Measured on the reference bedroom: 0.64 read on the cap against 1.18 on the canvas.

    The median, not the mean.  The annulus outside a cap boundary is mostly plain ceiling
    with a cornice or a picture rail crossing it, and the mean local deviation there is ten
    times the median -- it is a measurement of the cornice.  Grain cannot reproduce a
    cornice and should not try; what makes the boundary invisible is that the *typical*
    patch of fill is as rough as the typical patch of ceiling.
    """
    h, w = pano.shape[:2]
    edge = _polar_edge_row(coverage, which)
    if edge < 0:
        return 0.0
    rows_per_deg = h / 180.0
    lo = int(round(_GRAIN_RING_SKIP_DEG * rows_per_deg))
    hi = int(round((_GRAIN_RING_SKIP_DEG + _GRAIN_BAND_DEG) * rows_per_deg))
    sel = np.arange(edge + lo, edge + hi) if which == "zenith" else np.arange(edge - hi, edge - lo)
    sel = sel[(sel >= 0) & (sel < h)]
    if sel.size < 3:
        return 0.0
    win = int(max(3, round(_GRAIN_WINDOW_DEG * rows_per_deg)) | 1)
    # The window has to be square ON THE SPHERE, not on the canvas.  One equirect column at
    # colatitude c spans sin(c) of a row, so at the nadir ring of the reference bedroom a
    # 7x7 canvas window is 0.55 x 0.24 degrees, and it read the floor as 1.8 rough where a
    # square 0.55 deg window reads 4.1.  Grain matched to the narrow reading comes out less
    # than half as rough as the floor it is joining.
    colat = (edge + 0.5) / h * 180.0
    colat = colat if which == "zenith" else 180.0 - colat
    win_c = int(max(3, min(w // 4, round(win / max(math.sin(math.radians(colat)), 0.2)))) | 1)
    band = cv2.cvtColor(pano[sel.min() : sel.max() + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(band, (win_c, win))
    sd = np.sqrt(np.maximum(cv2.blur(band * band, (win_c, win)) - mu * mu, 0.0))
    # Only windows made entirely of captured pixels: the canvas is exact black outside
    # coverage, and one black pixel in the window is worth more than any real texture.
    ok = cv2.erode(
        (coverage[sel.min() : sel.max() + 1] > 0).astype(np.uint8),
        np.ones((win, win_c), np.uint8),
    ) > 0
    if int(ok.sum()) < 256:
        return 0.0
    return float(np.median(sd[ok]))


def _add_matched_grain(
    cap: np.ndarray, known: np.ndarray, extent: float, which: str, target: float
) -> np.ndarray:
    """Give the fill the ring's roughness, so texture does not stop dead at the boundary.

    A diffusion fill is smooth by construction, and smooth beside rough is a boundary even
    when the two are at the same brightness: on the reference bedroom the typical local
    standard deviation was 0.7 inside the nadir cap against 2.2 on the floor outside it, and
    the rug's texture simply stopped.  The deficit is made up with zero-mean noise added in
    quadrature, and faded out over the inner half of the cap.

    ``target`` comes from :func:`_ring_roughness`, measured on the canvas rather than here,
    for the reason given there.  It is capped at ``_GRAIN_MAX``: matching a rug's pattern
    with noise is a worse lie than leaving the fill smooth, and the cap is what keeps this a
    boundary treatment rather than a texture synthesiser.
    """
    size = int(cap.shape[0])
    if target <= 0.0:
        return cap
    r, hole, real, r_edge = _cap_masks(known, extent)
    if not hole.any() or not real.any():
        return cap

    # Both scales are stated in degrees and converted here, at the boundary's own radius.
    deg_px = _cap_deg_per_px(size, extent, r_edge)
    sigma = max(0.8, _GRAIN_SCALE_DEG / deg_px)
    band = max(4.0, _GRAIN_BAND_DEG / deg_px)
    win = int(max(3, round(_GRAIN_WINDOW_DEG / deg_px)) | 1)
    lum = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Local standard deviation over a small window, which is the quantity the boundary is
    # visible *in* -- an eye comparing two patches of ceiling is comparing exactly this.
    # A pure high-pass residual would be the cleaner measure of grain and is the wrong one:
    # it scores the fill's biggest deficit, that it has no local shading at all, as zero.
    mu = cv2.blur(lum, (win, win))
    sd = np.sqrt(np.maximum(cv2.blur(lum * lum, (win, win)) - mu * mu, 0.0))
    del lum, mu

    d_hole = cv2.distanceTransform((hole).astype(np.uint8), cv2.DIST_L2, 3)
    inner = hole & (d_hole > win) & (d_hole <= band)
    del d_hole
    if int(inner.sum()) < 64:
        del r, hole, real, sd, inner
        return cap
    have = float(np.median(sd[inner]))
    del inner
    need = float(np.clip(math.sqrt(max(target * target - have * have, 0.0)), 0.0, _GRAIN_MAX))
    if need < 0.25:
        _LOG.info("poles: %s grain already matches the ring (%.2f vs %.2f)", which, have, target)
        del r, hole, real, sd
        return cap

    rng = np.random.default_rng(_GRAIN_SEED)
    n = rng.standard_normal((size, size)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    # Scaled so its own local standard deviation over the same window is ``need``, not so
    # its global one is: a band-limited field's local and global spreads are not the same
    # number, and the boundary is judged on the local one.
    nm = cv2.blur(n, (win, win))
    n_local = float(
        np.median(np.sqrt(np.maximum(cv2.blur(n * n, (win, win)) - nm * nm, 0.0)))
    )
    n *= need / max(n_local, 1e-6)
    del nm
    keep = _pole_fade(r, r_edge)
    keep *= _fill_weight(known)
    out = to_uint8(cap.astype(np.float32) + (n * keep)[..., None])
    _LOG.info(
        "poles: %s grain %.2f -> %.2f against a ring of %.2f (added %.2f)",
        which, have, math.sqrt(have * have + need * need), target, need,
    )
    del r, hole, real, sd, n, keep
    return out


# ======================================================================================
# Composite back onto the equirect canvas
# ======================================================================================


def _band_rows(which: str, height: int, reach_deg: float) -> tuple[int, int]:
    """Equirect rows the cap composite may touch: pole out to ``reach_deg`` colatitude.

    The composite deliberately runs past the nominal 60 deg cap and out to wherever real
    pixels begin, because a single-row sweep leaves one continuous hole from the pole down
    to ~25 deg elevation.  Filling only the top of it would leave a black ring, which is a
    far worse artefact than a slightly generous conformal fill.  The per-pixel alpha still
    guarantees that nothing covered is overwritten.
    """
    colat = float(np.clip(reach_deg + _FEATHER_DEG, 5.0, 179.0))
    span = int(math.ceil(colat / 180.0 * height)) + 2
    if which == "zenith":
        return 0, int(min(height, span))
    return int(max(0, height - span)), height


def _hole_alpha(coverage: np.ndarray, r0: int, r1: int, height: int) -> np.ndarray:
    """Feathered "this pixel needs filling" weight for the band ``[r0, r1)``.

    Exactly 1 wherever no camera reached -- anything less and the black canvas shows
    through -- then ramping to 0 over ``_JOIN_DEG`` of covered pixels.  The ramp is built
    from a distance transform rather than a blur of the hole mask: a blur straddles the
    coverage boundary and therefore sits at ~0.5 *on* it, which quietly hands half of every
    boundary pixel to the resampled cap.
    """
    hole = (coverage[r0:r1] == 0).astype(np.uint8)
    ramp = max(2.0, _JOIN_DEG / 180.0 * height)
    # Distance from each covered pixel to the nearest uncovered one.
    dist = cv2.distanceTransform((hole == 0).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(1.0 - dist / ramp, 0.0, 1.0).astype(np.float32)
    del dist
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.8, ramp / 4.0), borderType=cv2.BORDER_REPLICATE)
    np.maximum(alpha, hole.astype(np.float32), out=alpha)
    del hole
    return alpha


def _composite_cap(
    pano: np.ndarray,
    coverage: np.ndarray,
    cap: np.ndarray,
    extent: float,
    which: str,
    reach_deg: float,
    filled: np.ndarray | None = None,
) -> None:
    """Composite a resampled cap over the band it owns, recording what it actually wrote.

    ``filled`` is set to 255 on every uncovered pixel this cap replaced.  Only the cap knows
    that: the band it touches is a row range, but within it the alpha is zero wherever the
    conformal patch does not reach, so "the cap ran" and "this pixel is no longer black" are
    different facts and the general filler needs the second one.
    """
    axis, e1, e2 = cap_frame(which)
    h, w = pano.shape[:2]
    r0, r1 = _band_rows(which, h, reach_deg)
    if r1 <= r0:
        return
    alpha_band = _hole_alpha(coverage, r0, r1, h)
    size = cap.shape[0]
    rows = max(1, _ROW_CHUNK // max(1, w))
    for a0 in range(r0, r1, rows):
        a1 = min(r1, a0 + rows)
        theta, phi = g.equirect_angles(w, h, v0=a0, h=a1 - a0)
        rays = g.rays_from_equirect(theta, phi)
        X, Y = _plane_coords(rays, axis, e1, e2)
        del rays
        mx = (X + extent) / (2.0 * extent) * size - 0.5
        my = (Y + extent) / (2.0 * extent) * size - 0.5
        del X, Y
        samp = cv2.remap(
            cap, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        inside = (mx >= 0) & (mx <= size - 1) & (my >= 0) & (my <= size - 1)
        del mx, my
        al = alpha_band[a0 - r0 : a1 - r0] * inside
        del inside
        dest = pano[a0:a1]
        np.copyto(
            dest,
            to_uint8(
                dest.astype(np.float32) * (1.0 - al[..., None])
                + samp.astype(np.float32) * al[..., None]
            ),
        )
        if filled is not None:
            filled[a0:a1][al > 0.5] = 255
        del samp, al, dest
    del alpha_band


# ======================================================================================
# Gradient (ring extrapolation) method
# ======================================================================================


def _gradient_fill(
    pano: np.ndarray,
    coverage: np.ndarray,
    which: str,
    reach_deg: float,
    trend: RingTrend | None = None,
    filled: np.ndarray | None = None,
) -> None:
    h, w = pano.shape[:2]
    r0, r1 = _band_rows(which, h, reach_deg)
    if r1 <= r0:
        return
    per_row = coverage.mean(axis=1) / 255.0
    if which == "zenith":
        candidates = np.nonzero(per_row[r0:r1] > 0.97)[0]
        if candidates.size == 0:
            ring_rows = slice(max(0, r1 - 8), r1)
            edge = r1
        else:
            edge = r0 + int(candidates.min())
            ring_rows = slice(edge, min(h, edge + max(4, (r1 - r0) // 8)))
        fill_rows = range(r0, edge)
        depth = max(1, edge - r0)
    else:
        candidates = np.nonzero(per_row[r0:r1] > 0.97)[0]
        if candidates.size == 0:
            ring_rows = slice(r0, min(h, r0 + 8))
            edge = r0
        else:
            edge = r0 + int(candidates.max())
            ring_rows = slice(max(0, edge - max(4, (r1 - r0) // 8)), edge + 1)
        fill_rows = range(edge + 1, r1)
        depth = max(1, r1 - edge)
    del per_row

    ring = pano[ring_rows].astype(np.float32).mean(axis=0)  # (w, 3)
    # Heavy cyclic blur: the extrapolated colour must carry no structure of its own.
    kw = max(3, (w // 8) | 1)
    wide = np.concatenate([ring[-kw:], ring, ring[:kw]], axis=0)[None, ...]
    wide = cv2.GaussianBlur(wide, (kw, 1), kw / 4.0, borderType=cv2.BORDER_REPLICATE)
    ring = wide[0, kw : kw + w]
    del wide
    ring_mean = ring.mean(axis=0)

    anchor_elev = 90.0 - 180.0 * (edge + 0.5) / h
    for v in fill_rows:
        t = abs(v - edge) / float(depth)
        t = float(np.clip(t, 0.0, 1.0))
        # The ring's azimuthal structure fades out toward the pole (it is not evidence
        # about the cap), while the radial trend carries the level and colour with it.
        colour = ring * (1.0 - t) + ring_mean[None, :] * t
        if trend is not None:
            elev = 90.0 - 180.0 * (v + 0.5) / h
            shift = trend_colour(trend, elev) - trend_colour(trend, anchor_elev)
            colour = colour + shift[None, :].astype(np.float32)
        row_alpha = (coverage[v] == 0).astype(np.float32)[:, None]
        pano[v] = to_uint8(pano[v].astype(np.float32) * (1.0 - row_alpha) + colour * row_alpha)
        if filled is not None:
            filled[v][coverage[v] == 0] = 255
        del colour, row_alpha
    del ring


# ======================================================================================
# Nadir logo
# ======================================================================================


def composite_nadir_logo(pano: np.ndarray, logo_path: Path, *, span_deg: float = 28.0) -> None:
    """Backwards-compatible alias for the nadir case of :func:`composite_pole_logo`."""
    composite_pole_logo(pano, logo_path, which="nadir", span_deg=span_deg)


def composite_pole_logo(
    pano: np.ndarray, logo_path: Path, *, which: str = "nadir", span_deg: float = 28.0
) -> None:
    """Alpha-composite a branding PNG onto a pole, ``span_deg`` across.

    BOTH POLES EARN THIS, for the same reason. A three-row sweep photographs a band, so the
    zenith and nadir caps are the two places `fill_uncovered` had to invent — and they are
    also the two directions a viewer looks at least. The nadir additionally carries whatever
    the operator's feet and the blur of a turning phone left there. Covering them with a mark
    replaces synthesised pixels with honest ones and costs the property nothing it had.
    """
    logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
    if logo is None:
        _LOG.warning("poles: could not read nadir logo %s", logo_path)
        return
    if logo.ndim == 2:
        logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGRA)
    if logo.shape[2] == 3:
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
    which = str(which).lower()
    if which not in ("nadir", "zenith"):
        raise ValueError(f"pole must be 'nadir' or 'zenith', not {which!r}")
    axis, e1, e2 = cap_frame(which)
    half = 0.5 * span_deg
    extent = plane_extent(half)
    h, w = pano.shape[:2]
    # Only the rows the cap can possibly touch: the top band for the zenith, the bottom for
    # the nadir, with a degree of slack so the mark's soft edge is not clipped.
    if which == "nadir":
        r_start = int(math.floor((90.0 + (90.0 - half) - 1.0) / 180.0 * h))
        r_start, r_end = max(0, min(h - 1, r_start)), h
    else:
        r_end = int(math.ceil((half + 1.0) / 180.0 * h))
        r_start, r_end = 0, max(1, min(h, r_end))
    lh, lw = logo.shape[:2]

    rows = max(1, _ROW_CHUNK // max(1, w))
    for a0 in range(r_start, r_end, rows):
        a1 = min(r_end, a0 + rows)
        theta, phi = g.equirect_angles(w, h, v0=a0, h=a1 - a0)
        rays = g.rays_from_equirect(theta, phi)
        X, Y = _plane_coords(rays, axis, e1, e2)
        del rays
        mx = ((X + extent) / (2.0 * extent) * lw - 0.5).astype(np.float32)
        my = ((Y + extent) / (2.0 * extent) * lh - 0.5).astype(np.float32)
        del X, Y
        samp = cv2.remap(logo, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        inside = (mx >= 0) & (mx <= lw - 1) & (my >= 0) & (my <= lh - 1)
        del mx, my
        al = (samp[..., 3].astype(np.float32) / 255.0) * inside
        dest = pano[a0:a1]
        np.copyto(
            dest,
            to_uint8(
                dest.astype(np.float32) * (1.0 - al[..., None])
                + samp[..., :3].astype(np.float32) * al[..., None]
            ),
        )
        del samp, al, inside, dest


# ======================================================================================
# Uncovered regions that are not at a pole
# ======================================================================================

#: Measured on this machine (Ryzen 5 3600) with ``INPAINT_FSR_FAST`` over crops between
#: 256x128 and 768x384 at 25-50 % hole: 3.0e-5 to 5.4e-5 seconds per hole pixel, rising with
#: the crop as FSR's cost is super-linear in the region rather than linear in the hole.  The
#: upper end is used, because underestimating costs a stall and overestimating costs a
#: slightly blurrier fill.  Telea is 10-15x cheaper and is the fallback.
_FSR_FAST_S_PER_PX = 5.4e-5
#: Wall-clock allowed for one region, and for all of them together.  A 185 degree wedge at
#: 4096 px is ~300k hole pixels, i.e. sixteen seconds of FSR at full resolution on a render
#: that takes ninety seconds end to end -- so the region is scaled down until it fits the
#: budget instead, which is also what makes the fill an honest blur rather than invention.
_EQUIRECT_BUDGET_S = 2.0
_EQUIRECT_TOTAL_BUDGET_S = 6.0
#: Longest side of the working crop, whatever the budget says.
_EQUIRECT_WORK_MAX = 512
#: ...and the shortest, below which there is no colour structure left to carry.
_EQUIRECT_WORK_MIN = 24
#: Samples the fill may carry across the narrow dimension of a hole.
#:
#: This is what keeps the fill honest, and it is a stronger constraint than the time budget.
#: FSR reconstructs *structure* by extending the surrounding spectrum into the hole -- that
#: is the whole point of it, and on a wall with any texture it will happily invent a
#: continuation that is as detailed as the real thing and completely made up.  Nobody
#: photographed that wall.  Resolving the fill at a couple of dozen samples across and
#: resampling up turns the same call into what it should be: a smooth, correctly coloured
#: continuation.  It also removes FSR's block-boundary steps, which on a 120 px hole
#: measured twenty grey levels in the middle of an otherwise linear ramp.
_EQUIRECT_HOLE_SAMPLES = 24
#: Share of a hole FSR may leave at exactly black before its output is rejected outright.
_FSR_UNFILLED_TOLERANCE = 0.02
#: How far across a hole FSR can actually reach, in working-scale pixels.
#:
#: Measured against a known ramp plus texture: up to twelve pixels across, FSR beats Telea
#: by 30-40 % on mean error because it carries the texture; at sixteen and beyond it loses,
#: and on the wide holes this filler exists for it fails outright -- decaying to black a few
#: pixels in from one boundary and spiking back up at the other, which looks like content
#: and is not.  It reconstructs a hole from the spectrum immediately around it, so its reach
#: is a property of its search window and no amount of budget buys more of it.
_FSR_REACH_PX = 12
#: Regions given the careful crop-and-inpaint treatment, largest first.  Matches the number
#: of gaps the coverage report is willing to name.
_EQUIRECT_MAX_REGIONS = 8
#: How far the fill's alpha ramps into captured pixels.  Same trade as ``_JOIN_DEG``: alpha
#: must be exactly 1 on every uncovered pixel, so the ramp can only eat real content.
_EQUIRECT_JOIN_DEG = 1.0


def _cyclic_span(cols: np.ndarray) -> tuple[int, int]:
    """``(start, span)`` of the smallest cyclic column range containing every set column."""
    n = int(cols.size)
    idx = np.nonzero(cols)[0]
    if idx.size == 0:
        return 0, 0
    if idx.size == n:
        return 0, n
    # The span is the complement of the widest run of unset columns.
    gaps = np.diff(np.concatenate([idx, idx[:1] + n]))
    k = int(np.argmax(gaps))
    return int(idx[(k + 1) % idx.size]), int(n - (int(gaps[k]) - 1))


def _cyclic_regions(hole: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Label uncovered regions, merging the ones that meet across the canvas edge.

    Column 0 and column ``W-1`` are neighbours on the sphere, so a wedge behind the viewer
    arrives as two components.  Inpainting them separately would fill each from three sides
    and leave a visible join down the seam that the wrap repair cannot reach, so they are
    united here and cropped as one region below.
    """
    n_labels, labels = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return labels, []
    parent = list(range(n_labels))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    left, right = labels[:, 0], labels[:, -1]
    rows = labels.shape[0]
    for r in range(rows):
        if not left[r]:
            continue
        for rr in (r - 1, r, r + 1):  # 8-connectivity across the edge
            if 0 <= rr < rows and right[rr]:
                union(int(left[r]), int(right[rr]))
    roots = np.array([find(i) for i in range(n_labels)], dtype=np.int32)
    np.take(roots, labels, out=labels)
    counts = np.bincount(labels.ravel(), minlength=n_labels)
    order = [(int(counts[lab]), int(lab)) for lab in np.unique(roots) if lab != 0 and counts[lab]]
    order.sort(reverse=True)
    del roots, counts
    return labels, order


def _fsr_took(dst: np.ndarray, valid: np.ndarray) -> bool:
    """Did FSR actually fill the hole, or did it quietly hand back black?

    It does not raise and it does not return an empty image -- it fills the rim, decays to
    zero a few pixels in, and leaves the interior at exactly ``(0, 0, 0)``.  Measured here
    on a hole spanning 56 % of its crop's width, every scale from 0.2 to 0.5 came back with
    a black core, which composited straight back into the panorama as the very wedge this
    is supposed to remove.  FSR reconstructs a hole from the spectrum *around* it, so once
    the interior is further from real pixels than its search window there is nothing to
    reconstruct from; Telea marches a front in from the boundary and always arrives.

    Exact zeros are the signal: the canvas is zeroed where no camera reached, so a fill that
    returns zero has not filled anything.  When the surrounding content really is black
    Telea returns black too, and the fallback costs a few milliseconds.
    """
    hole = valid == 0
    n = int(hole.sum())
    if n == 0:
        return True
    return float(((dst.max(axis=2) == 0) & hole).sum()) / n <= _FSR_UNFILLED_TOLERANCE


def _equirect_inpaint(
    crop: np.ndarray, valid: np.ndarray, hole_px: int, hole_short: float, budget: float
) -> np.ndarray:
    """Inpaint a crop small and resample it back up.

    Downscaling is not a compromise here, it is the point.  A hole this big has no
    high-frequency content anyone can recover, and a fill carried out at full resolution
    would either cost half a minute or start inventing wallpaper; resolving it small and
    resampling up produces exactly the smooth, correctly coloured continuation that is the
    honest answer for a patch of wall nobody photographed.

    Three limits, and the tightest wins: the fill may carry only so many samples across the
    hole (:data:`_EQUIRECT_HOLE_SAMPLES`, the honesty constraint), the crop may only be so
    large (FSR's cost is super-linear in it), and the whole call has to fit its time budget.
    """
    ch, cw = crop.shape[:2]
    by_budget = math.sqrt(budget / (_FSR_FAST_S_PER_PX * max(hole_px, 1)))
    by_honesty = _EQUIRECT_HOLE_SAMPLES / max(hole_short, 1.0)
    scale = min(1.0, by_honesty, _EQUIRECT_WORK_MAX / float(max(ch, cw)), by_budget)
    ww = int(max(_EQUIRECT_WORK_MIN, min(cw, round(cw * scale))))
    wh = int(max(_EQUIRECT_WORK_MIN, min(ch, round(ch * scale))))

    small = cv2.resize(crop, (ww, wh), interpolation=cv2.INTER_AREA)
    kn = cv2.resize(valid, (ww, wh), interpolation=cv2.INTER_AREA)
    # Conservative: a working pixel is context only if it is made entirely of real pixels.
    kn = np.where(kn > 200, np.uint8(255), np.uint8(0))
    work_hole = int((kn == 0).sum())

    filled: np.ndarray | None = None
    if (kn > 0).any() and work_hole:
        reachable = hole_short * (ww / float(cw)) <= _FSR_REACH_PX
        if reachable and work_hole * _FSR_FAST_S_PER_PX <= budget:
            try:
                dst = np.zeros_like(small)
                cv2.xphoto.inpaint(small, kn, dst, cv2.xphoto.INPAINT_FSR_FAST)
                if np.isfinite(dst).all() and dst.shape == small.shape and _fsr_took(dst, kn):
                    filled = dst
                else:
                    _LOG.info("fill: FSR left the hole empty, using Telea")
            except Exception as exc:  # noqa: BLE001 - Telea below is always available
                _LOG.warning("fill: FSR inpaint failed (%s), using Telea", exc)
        if filled is None:
            mask = cv2.dilate((kn == 0).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
            filled = cv2.inpaint(small, mask, 5.0, cv2.INPAINT_TELEA)
            del mask
    elif not (kn > 0).any():
        # No context at all in the crop: hold the surrounding level rather than guess.
        filled = np.full_like(small, int(round(float(crop[valid > 0].mean()))) if (valid > 0).any() else 128)
    else:
        filled = small
    _LOG.info(
        "fill: region %dx%d -> %dx%d, %d hole px at working scale", cw, ch, ww, wh, work_hole
    )
    del small, kn
    up = cv2.resize(filled, (cw, ch), interpolation=cv2.INTER_CUBIC)
    del filled
    return up


def _fill_equirect_holes(pano: np.ndarray, pending: np.ndarray) -> int:
    """Close every still-black region in the equirect domain.  Returns the pixels filled.

    Away from the poles the equirect distortion is mild -- one row and one column span
    comparable angles -- so an isotropic inpainting kernel is close enough to isotropic on
    the sphere, and the conformal detour the caps need would only cost resolution here.

    ``pending`` is 255 on the pixels that still have to be filled; it is consumed as the
    regions are closed, so a region cropped later sees an earlier fill as usable context
    rather than as black.
    """
    h, w = pano.shape[:2]
    labels, order = _cyclic_regions(pending > 0)
    ramp = max(2.0, _EQUIRECT_JOIN_DEG / 180.0 * h)
    spent = 0.0
    total = 0
    for count, lab in order[:_EQUIRECT_MAX_REGIONS]:
        if spent >= _EQUIRECT_TOTAL_BUDGET_S:
            break
        sel = labels == lab
        rows_any = np.nonzero(sel.any(axis=1))[0]
        start, span = _cyclic_span(sel.any(axis=0))
        if span == 0 or rows_any.size == 0:
            del sel
            continue
        r_lo, r_hi = int(rows_any.min()), int(rows_any.max()) + 1

        # Context for the inpainter, and room for the alpha ramp.
        pad = int(max(8.0, ramp, 0.35 * max(span, r_hi - r_lo)))
        r0, r1 = max(0, r_lo - pad), min(h, r_hi + pad)
        cw = min(w, span + 2 * pad)
        cols = (np.arange(start - (cw - span) // 2, start - (cw - span) // 2 + cw) % w).astype(
            np.intp
        )

        band = pano[r0:r1]
        crop = band[:, cols]
        pend = pending[r0:r1][:, cols]
        mine = sel[r0:r1][:, cols]
        del sel
        # Everything not still black is context -- captured pixels and pole caps alike.
        valid = np.where(pend > 0, np.uint8(0), np.uint8(255))

        # Area over the long side approximates the hole's narrow dimension, which is what
        # decides how many samples a fill may honestly carry across it.
        hole_short = count / float(max(span, r_hi - r_lo))
        t0 = time.perf_counter()
        up = _equirect_inpaint(crop, valid, int(count), hole_short, _EQUIRECT_BUDGET_S)
        spent += time.perf_counter() - t0
        del valid

        # Alpha: exactly 1 on this region, ramping to 0 over captured pixels, and 0 on any
        # other region still waiting its turn -- blending a blur into black is worse than
        # leaving the black for the pass that will fill it properly.
        dist = cv2.distanceTransform((~mine).astype(np.uint8), cv2.DIST_L2, 3)
        alpha = np.clip(1.0 - dist / ramp, 0.0, 1.0).astype(np.float32)
        del dist
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.8, ramp / 4.0), borderType=cv2.BORDER_REPLICATE)
        alpha[(pend > 0) & ~mine] = 0.0
        np.maximum(alpha, mine.astype(np.float32), out=alpha)

        band[:, cols] = to_uint8(
            crop.astype(np.float32) * (1.0 - alpha[..., None]) + up.astype(np.float32) * alpha[..., None]
        )
        pend[mine] = 0
        pending[r0:r1][:, cols] = pend
        total += int(count)
        del crop, up, alpha, mine, pend, band, cols

    del labels
    # Whatever is left is crumbs -- a few pixels along a mask edge, or a region beyond the
    # budget.  Telea marches in from the boundary and costs a few microseconds per pixel, so
    # sweeping them all at once guarantees the invariant that matters: no black pixels.
    if pending.any():
        rest = int((pending > 0).sum())
        mask = cv2.dilate(pending, np.ones((3, 3), np.uint8))
        swept = cv2.inpaint(pano, mask, 5.0, cv2.INPAINT_TELEA)
        np.copyto(pano, swept, where=(pending > 0)[..., None])
        del mask, swept
        pending[:] = 0
        total += rest
        _LOG.info("fill: swept %d leftover pixels with Telea", rest)
    return total


# ======================================================================================
# Entry points
# ======================================================================================


def _fill_pole_caps(
    pano: np.ndarray, coverage: np.ndarray, config: PipelineConfig, filled: np.ndarray | None = None
) -> None:
    """Fill the zenith and nadir caps in the conformal domain, in place."""
    method = str(config.pole_method).lower()
    h = pano.shape[0]
    for which in ("zenith", "nadir"):
        cap_r0, cap_r1 = (0, max(1, int((90.0 - _CAP_DEG) / 180.0 * h))) if which == "zenith" else (
            min(h - 1, int((90.0 + _CAP_DEG) / 180.0 * h)),
            h,
        )
        if not (coverage[cap_r0:cap_r1] == 0).any():
            _LOG.info("poles: %s already covered, nothing to fill", which)
            continue
        reach = cap_reach_deg(coverage, which)
        trend = ring_trend(pano, coverage, which)
        if method == "gradient":
            _gradient_fill(pano, coverage, which, reach, trend, filled)
        else:
            cap, known, extent = extract_cap(pano, coverage, which, reach_deg=reach)
            capped = _inpaint_cap(cap, known, extent)
            if trend is not None:
                capped = _apply_radial_trend(capped, known, extent, which, trend)
            # Level first, then grain: the grain match measures how rough the fill is, and
            # the level correction is smooth, so doing it the other way round would leave
            # the roughness measurement describing an image that no longer exists.
            capped = _match_boundary_level(capped, known, extent, which)
            capped = _add_matched_grain(
                capped, known, extent, which, _ring_roughness(pano, coverage, which)
            )
            del cap, known
            _composite_cap(pano, coverage, capped, extent, which, reach, filled)
            del capped
            _LOG.info("poles: %s cap reached %.0f deg colatitude", which, reach)
        _LOG.info("poles: filled %s with %s", which, method)


def _as_coverage(coverage_mask: np.ndarray) -> np.ndarray:
    """A read-only 2-D view of the coverage mask.  Never written to, ever.

    Coverage is a factual claim about what the camera saw and it is reported to the agent as
    one.  Filling a pixel is a presentation decision that must not touch it -- so the mask is
    marked non-writeable here and the fill tracks what it has done in a mask of its own.
    """
    coverage = coverage_mask[..., 0] if coverage_mask.ndim == 3 else coverage_mask
    # A fresh view either way, so flipping the flag cannot make the caller's array read-only.
    view = np.ascontiguousarray(coverage).view()
    view.flags.writeable = False
    return view


def fill_poles(pano: np.ndarray, coverage_mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Fill uncovered zenith / nadir caps in place and return the panorama."""
    if str(config.pole_method).lower() == "none":
        return pano
    _fill_pole_caps(pano, _as_coverage(coverage_mask), config)
    if config.nadir_logo is not None:
        composite_nadir_logo(pano, Path(config.nadir_logo))
    return pano


def fill_uncovered(pano: np.ndarray, coverage_mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Close every uncovered region of the panorama, wherever it is.

    A black wedge is the worst thing this pipeline can ship.  In a flat image a missing
    corner is a crop; in a 360 viewer the user turns their head and falls into a void, and
    an approximate fill beats that every time.  Holes away from the poles are not an edge
    case either -- an agent sweeping a room turns too fast, blocks the lens with an arm, or
    skips the bit behind the wardrobe -- so this closes all of them, with the pole caps as
    the special case that needs different geometry rather than as the only case handled.

    Two passes, because the sphere is not the same everywhere:

    * the caps, in the conformal stereographic domain, where an isotropic kernel in the
      plane is isotropic on the sphere and a fill does not smear into radial streaks;
    * everything else, in the equirect domain directly, where the distortion is mild and
      the detour would only cost resolution.

    Coverage is untouched by all of it.  ``covered_fraction``, the reported gaps and the
    zenith/nadir flags describe what the camera reached, and they must read the same whether
    or not a pixel was later filled in -- otherwise the one number that tells an agent to go
    back and sweep the corner starts congratulating them for the renderer's guesswork.

    ``pole_method='none'`` disables the whole thing, not just the caps.  It is the existing
    switch for "invent nothing", :class:`PipelineConfig` is a frozen contract with nowhere to
    put a second one, and a caller who has asked for no invented pixels at the poles has not
    asked for invented pixels somewhere less carefully reasoned about.
    """
    if str(config.pole_method).lower() == "none":
        return pano
    coverage = _as_coverage(coverage_mask)
    filled = np.zeros(coverage.shape, dtype=np.uint8)
    t0 = time.perf_counter()

    _fill_pole_caps(pano, coverage, config, filled)

    pending = np.where((coverage == 0) & (filled == 0), np.uint8(255), np.uint8(0))
    del filled
    n_pending = int((pending > 0).sum())
    if n_pending:
        done = _fill_equirect_holes(pano, pending)
        _LOG.info(
            "fill: closed %d of %d uncovered pixels away from the poles in %.2f s",
            done, n_pending, time.perf_counter() - t0,
        )
    del pending

    if config.nadir_logo is not None:
        composite_nadir_logo(pano, Path(config.nadir_logo))
    return pano
