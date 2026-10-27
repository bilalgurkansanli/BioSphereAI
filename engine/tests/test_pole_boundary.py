"""What the edge of a synthesised pole cap has to look like.

A pole cap is invented pixels, and the honest ambition for them is not that they are right
-- nobody photographed that ceiling -- but that a viewer cannot find where they start.  Two
things give the boundary away and both are measurable:

* a step in brightness across it, and it is a step *per azimuth*.  The hole a real capture
  leaves is the union of the camera frusta, so its outline is a polygon; a fill that is
  right on average and four grey levels out along one facet reads as a dome edge painted on
  the ceiling.  Averaging the step around the circle hides exactly the fault being looked
  for, so every test here measures it in sectors.
* a step in texture.  Smooth beside rough is a boundary even at matched brightness.

The fixture is deliberately hostile in the way a real room is: the ring bounding the hole is
much brighter on one side than the other (a ceiling light off to one side), and the hole is
a polygon rather than a disc.  A fill whose level comes only from an azimuthal average
cannot pass, which is the point.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from biosphere.pano import poles as poles_mod
from biosphere.pano.types import PipelineConfig

W, H = 1024, 512


def _config(**overrides) -> PipelineConfig:
    cfg = PipelineConfig(output_width=W, nadir_logo=None, zenith_logo=None)
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


def _ceiling_with_a_lamp(seed: int = 7, edge_colat_deg: float = 34.0) -> np.ndarray:
    """An equirect whose upper hemisphere is a ceiling lit from one side, plus grain.

    Two things here are deliberate rather than decorative.  The 18 level swing with azimuth
    is a lamp off to one side, so a fill levelled from an azimuthal mean is wrong by nine on
    each side of the cap.  The glow that rises steeply over the last few degrees before the
    hole is the harder case: it is what a light fitting just inside the cap actually does to
    the ceiling around it, and a fill that reads its level from an annulus a couple of
    degrees further out -- which is the safe-looking thing to do, because that annulus is
    made of clean captured pixels -- arrives darker than the ring it has to join.
    """
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:H, 0:W]
    elev = 90.0 - 180.0 * (v + 0.5) / H
    colat = 90.0 - elev
    theta = 2.0 * math.pi * (u + 0.5) / W
    glow = 14.0 * np.exp(-np.clip(colat - edge_colat_deg, 0.0, None) / 2.5)
    base = (
        150.0 + 0.55 * elev + glow
        + 18.0 * np.cos(theta) * np.clip(elev / 60.0, 0.0, 1.0)
    )
    grain = rng.normal(0.0, 3.0, size=(H, W))
    grain = cv2.GaussianBlur(grain, (0, 0), 1.2)
    grain *= 3.0 / max(float(grain.std()), 1e-6)
    img = np.clip(base + grain, 0, 255).astype(np.uint8)
    return np.repeat(img[:, :, None], 3, axis=2)


def _polygon_cap_hole(edge_colat_deg: float = 34.0, facets: int = 9, wobble_deg: float = 4.0):
    """``(coverage, hole)`` for a zenith hole whose outline is a polygon, not a circle."""
    v, u = np.mgrid[0:H, 0:W]
    colat = 180.0 * (v + 0.5) / H
    theta = 2.0 * math.pi * (u + 0.5) / W
    edge = edge_colat_deg + wobble_deg * np.cos(facets * theta)
    hole = colat < edge
    coverage = np.where(hole, np.uint8(0), np.uint8(255))
    return coverage, hole


def _sector_steps(pano: np.ndarray, coverage: np.ndarray, n_sectors: int = 12) -> np.ndarray:
    """Brightness step across the hole boundary, one per azimuth sector.

    Measured on the canvas in a band one to five rows either side of the boundary, which at
    this size is half a degree to two and a half degrees of sphere.
    """
    grey = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY).astype(np.float64)
    covered = (coverage > 0).astype(np.uint8)
    d_out = cv2.distanceTransform(covered, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(1 - covered, cv2.DIST_L2, 3)
    cols = np.arange(W)[None, :] * np.ones((H, 1))
    sector = (cols / W * n_sectors).astype(np.int32) % n_sectors
    band = (np.arange(H)[:, None] * np.ones((1, W))) < H * 0.35  # upper hemisphere only
    steps = []
    for s in range(n_sectors):
        inn = band & (sector == s) & (covered == 0) & (d_in >= 1) & (d_in <= 5)
        out = band & (sector == s) & (covered > 0) & (d_out >= 1) & (d_out <= 5)
        if inn.sum() > 50 and out.sum() > 50:
            steps.append(float(grey[out].mean()) - float(grey[inn].mean()))
    assert len(steps) >= n_sectors - 1, "the fixture did not produce a measurable boundary"
    return np.asarray(steps)


def _local_std(pano: np.ndarray, mask: np.ndarray, win: int = 7) -> float:
    grey = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(grey, (win, win))
    sd = np.sqrt(np.maximum(cv2.blur(grey * grey, (win, win)) - mu * mu, 0.0))
    return float(np.median(sd[mask]))


# ======================================================================================


def test_cap_boundary_has_no_step_in_any_direction():
    """The fill must meet the ring at the ring's own brightness, facet by facet."""
    gt = _ceiling_with_a_lamp()
    coverage, hole = _polygon_cap_hole()
    pano = gt.copy()
    pano[hole] = 0

    filled = poles_mod.fill_uncovered(pano, coverage, _config())
    assert int((filled.max(axis=2) == 0).sum()) == 0, "the cap was left black"

    steps = _sector_steps(filled, coverage)
    rms = float(np.sqrt(np.mean(steps**2)))
    worst = float(np.max(np.abs(steps)))
    print(f"\n[cap edge] per-sector step: rms {rms:.2f}, worst {worst:.2f} grey levels")

    # The lamp puts 36 grey levels between the bright and dark sides of the ring, so a fill
    # levelled from an azimuthal mean alone lands about 18 out on each side.
    assert rms < 4.0, f"the cap boundary steps by {rms:.1f} grey levels rms"
    assert worst < 8.0, f"one facet of the cap boundary steps by {worst:.1f} grey levels"


def test_cap_fill_is_not_smooth_beside_a_grainy_ring():
    """Texture must not stop dead at the boundary either."""
    gt = _ceiling_with_a_lamp()
    coverage, hole = _polygon_cap_hole()
    pano = gt.copy()
    pano[hole] = 0

    filled = poles_mod.fill_uncovered(pano, coverage, _config())
    d_in = cv2.distanceTransform((coverage == 0).astype(np.uint8), cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform((coverage > 0).astype(np.uint8), cv2.DIST_L2, 3)
    upper = np.zeros((H, W), bool)
    upper[: int(H * 0.35)] = True
    inner = upper & (coverage == 0) & (d_in >= 8) & (d_in <= 24)
    outer = upper & (coverage > 0) & (d_out >= 8) & (d_out <= 24)

    rough_in = _local_std(filled, inner)
    rough_out = _local_std(filled, outer)
    print(f"[cap edge] local std {rough_in:.2f} inside the fill, {rough_out:.2f} outside")
    assert rough_in > 0.45 * rough_out, "the fill is glassy beside a grainy ring"
    assert rough_in < 2.2 * rough_out, "the fill is rougher than the ceiling it is joining"


def test_the_fill_does_not_reach_over_captured_pixels():
    """A pixel a camera reached keeps its own value, to within the join's own softening.

    The cap is resampled through a conformal patch and back, so the ramp that hides the
    join necessarily costs a little sharpness on the covered side.  What it may not do is
    move the *content*: a fill that overwrites real structure -- a ceiling light sitting
    half inside the cap, say -- is a worse failure than any boundary artefact.
    """
    gt = _ceiling_with_a_lamp()
    coverage, hole = _polygon_cap_hole()
    pano = gt.copy()
    pano[hole] = 0
    # A hard bright bar just outside the boundary, the way a light fitting or a cornice sits
    # across it: the thing a too-eager fill destroys.
    d_out = cv2.distanceTransform((coverage > 0).astype(np.uint8), cv2.DIST_L2, 3)
    bar = (coverage > 0) & (d_out >= 4) & (d_out <= 9)
    pano[bar] = 250
    gt = pano.copy()

    filled = poles_mod.fill_uncovered(pano, coverage, _config())
    err = np.abs(filled[bar].astype(np.float64) - gt[bar].astype(np.float64)).mean()
    print(f"[cap edge] mean change on the bar of real pixels: {err:.2f} grey levels")
    assert err < 12.0, "the fill overwrote real content just outside the hole"
    far = (coverage > 0) & (d_out > 40)
    assert np.array_equal(filled[far], gt[far]), "the fill touched pixels far from the hole"


def test_cap_resolution_follows_the_canvas():
    """The conformal patch must not be coarser than the canvas it has to join.

    ``cap_size_for`` exists because a fixed 1024 was half the canvas's resolution at the rim
    of a 43 degree cap on a 4096-wide render, and the two grids meet exactly at the hole
    boundary.
    """
    extent = poles_mod.plane_extent(43.0)
    size = poles_mod.cap_size_for(extent, 2048)
    # One cap pixel is widest at the pole: 4 * extent / size radians.
    coarsest = math.degrees(4.0 * extent / size)
    row = 180.0 / 2048
    assert coarsest <= row * 1.02, f"{coarsest:.4f} deg per cap pixel against {row:.4f} per row"
    # ...and it must not run away on a small canvas or a tiny cap either.
    assert poles_mod.cap_size_for(poles_mod.plane_extent(8.0), 512) >= poles_mod._CAP_SIZE_MIN
    assert poles_mod.cap_size_for(poles_mod.plane_extent(85.0), 16384) <= poles_mod._CAP_SIZE_MAX


def _spoke_strength(pano: np.ndarray, hole: np.ndarray, lo: float, hi: float) -> float:
    """The brightest coherent ray in the fill between ``lo`` and ``hi`` of the hole's depth.

    A ray drawn from the rim toward the pole is, on an equirect canvas, a column that stands
    out from its neighbours ALL the way down. So each row's deviation from its own 10-degree
    azimuthal average is averaged over the depth range -- grain cancels, a ray accumulates --
    and the largest such column is the answer, in grey levels.
    """
    grey = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY).astype(np.float64)
    edge_row = int(np.nonzero(hole.all(axis=1))[0].max()) if hole.all(axis=1).any() else 0
    k = max(3, int(W * 10 / 360) | 1)
    acc = np.zeros(W)
    rows = np.linspace(lo, hi, 9)
    for f in rows:
        row = grey[int(f * edge_row)]
        pad = np.concatenate([row[-k:], row, row[:k]])
        acc += row - np.convolve(pad, np.ones(k) / k, mode="same")[k:-k]
    return float(np.abs(acc / rows.size).max())


def test_a_bright_feature_on_the_rim_does_not_become_a_spoke():
    """A lamp clipped by the coverage edge must fade into the fill, not be drawn to the pole.

    On the reference capture the boundary correction carried each azimuth's rim mismatch
    straight inward, so the ceiling lamp's two arcs and the cornice came back as bright
    spokes across the synthesised ceiling, and the rug fanned out round the nadir. Here the
    ring carries three narrow bright features right at the edge -- arcs of a light fitting --
    on an otherwise plain ceiling. Halfway to the pole the fill must be as plain as the
    ceiling it continues.
    """
    gt = _ceiling_with_a_lamp()
    coverage, hole = _polygon_cap_hole()
    pano = gt.copy()
    d_out = cv2.distanceTransform((coverage > 0).astype(np.uint8), cv2.DIST_L2, 3)
    u = np.arange(W)[None, :] * np.ones((H, 1))
    for centre in (0.12, 0.17, 0.61):
        arc = (coverage > 0) & (d_out <= 6) & (np.abs(u - centre * W) < W * 0.012)
        pano[arc] = 252
    pano[hole] = 0

    filled = poles_mod.fill_uncovered(pano, coverage, _config())
    spoke = _spoke_strength(filled, hole, 0.35, 0.65)
    print(f"\n[cap spokes] brightest coherent ray mid-cap {spoke:.2f} grey levels")
    assert spoke < SPOKE_LIMIT, f"a rim feature was drawn into the fill as a spoke ({spoke:.2f})"


def test_the_rim_correction_is_the_harmonic_extension_of_the_rim():
    """Mode k of the rim mismatch must fall off as (r / R)^k -- the Poisson integral.

    A circular hole of radius R, rim values ``cos(theta) + 0.5 cos(5 theta)``: halfway in,
    the extension must read ``0.5 cos(theta) + 0.5 * 0.5^5 cos(5 theta)``. The slow mode
    keeps half its strength (a one-sided glow tilts the whole cap), the fast one is all but
    gone (a lamp's bump stays at the rim) -- which is the whole difference from a ray.
    """
    size, R = 512, 0.8
    t = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    yy, xx = np.meshgrid(t, t, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy).astype(np.float32)
    theta = np.arctan2(yy, xx)
    hole = r < R
    real = (r >= R) & (r <= 1.0)
    nb = poles_mod._BOUNDARY_AZ_BINS
    idx = np.clip(((theta + math.pi) / (2.0 * math.pi) * nb).astype(np.int32), 0, nb - 1)
    centres = (np.arange(nb) + 0.5) / nb * 2.0 * math.pi - math.pi
    values = (np.cos(centres) + 0.5 * np.cos(5 * centres))[:, None].astype(np.float64)
    ext = poles_mod._harmonic_extension(values, r, hole, real, idx, R)[..., 0]
    ring = hole & (np.abs(r - 0.5 * R) < 0.01)
    expect = 0.5 * np.cos(theta[ring]) + 0.5 * 0.5**5 * np.cos(5 * theta[ring])
    err = float(np.abs(ext[ring] - expect).max())
    assert err < 0.03, f"not the harmonic extension: worst error {err:.3f} at r = R/2"
    assert float(np.abs(ext[hole & (r < 0.02)]).max()) < 0.03, "the azimuthal part did not vanish at the pole"


#: Measured on this fixture: the rim correction carried straight inward (the code before the
#: harmonic extension) leaves a 9.0 grey-level ray halfway to the pole; the harmonic
#: extension leaves 1.1, which is the fixture's grain. The limit sits between them.
SPOKE_LIMIT = 3.0


def test_grain_is_deterministic():
    """Two renders of one capture must be the same image, grain included."""
    gt = _ceiling_with_a_lamp()
    coverage, hole = _polygon_cap_hole()
    a, b = gt.copy(), gt.copy()
    a[hole] = 0
    b[hole] = 0
    fa = poles_mod.fill_uncovered(a, coverage, _config())
    fb = poles_mod.fill_uncovered(b, coverage, _config())
    assert np.array_equal(fa, fb), "the pole fill is not reproducible"
