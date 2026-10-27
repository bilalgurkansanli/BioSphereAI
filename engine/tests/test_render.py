"""End-to-end tests for the rendering half of the engine, with self-made ground truth.

There is no video here and no dependency on the solver.  Instead a synthetic equirect
panorama is generated procedurally, virtual pinhole views are *sampled out of it* with the
shared geometry helpers at known rotations, and those views plus their exact poses are fed
to :func:`render_panorama`.  Because the ground truth is known to the pixel, "did the
renderer work" becomes a number: solid-angle weighted PSNR against the original.

Everything the synthetic panorama contains is written as a function of ``theta`` with an
integer number of cycles, so the image is genuinely cyclic and column 0 is a real
neighbour of column W-1.  Without that, the wrap-seam test would be measuring the test
fixture's own discontinuity rather than the renderer's.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from biosphere.pano import blend as blend_mod
from biosphere.pano import coverage as coverage_mod
from biosphere.pano import geometry as g
from biosphere.pano import output as output_mod
from biosphere.pano import poles as poles_mod
from biosphere.pano import refine as refine_mod
from biosphere.pano import render as render_mod
from biosphere.pano import seams as seams_mod
from biosphere.pano import warp as warp_mod
from biosphere.pano.types import CameraPose, Frame, Intrinsics, PipelineConfig, QualityMode

GT_W, GT_H = 2048, 1024


# ======================================================================================
# Synthetic world
# ======================================================================================


def _cyclic_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that respects the panorama's horizontal periodicity."""
    pad = max(4, int(sigma * 4))
    wide = np.concatenate([img[:, -pad:], img, img[:, :pad]], axis=1)
    wide = cv2.GaussianBlur(wide, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(wide[:, pad:-pad])


def synthetic_pano(w: int = GT_W, h: int = GT_H, *, lo: float = 0.0, hi: float = 255.0) -> np.ndarray:
    """Procedural equirect: gradients, a cyclic checkerboard, colour stripes and blobs.

    Every angular term uses an integer number of cycles in ``theta`` and a cosine rather
    than a sine, so no feature edge lands exactly on the canvas seam -- otherwise a
    perfect renderer would still fail the wrap-continuity test.
    """
    theta, phi = g.equirect_angles(w, h, dtype=np.float64)
    T = theta[None, :]
    P = phi[:, None]

    # Smooth base: a vertical sky-to-floor ramp plus a slow azimuthal swirl.
    base = 0.5 + 0.32 * np.sin(P) + 0.10 * np.cos(3.0 * T) * np.cos(P)
    # Cyclic checkerboard: 32 cycles in yaw, 12 in pitch.
    check = np.sign(np.cos(32.0 * T)) * np.sign(np.cos(12.0 * (P + 0.5 * np.pi)))
    base = base + 0.11 * check * np.cos(P) ** 0.5
    del check

    img = np.empty((h, w, 3), dtype=np.float64)
    img[..., 0] = base * (0.85 + 0.15 * np.cos(6.0 * T))
    img[..., 1] = base * (0.90 + 0.10 * np.cos(4.0 * T + 1.1))
    img[..., 2] = base * (0.80 + 0.20 * np.cos(5.0 * T + 2.3))
    del base

    # A few landmark blobs so local alignment errors have something to smear.
    for yaw_deg, pitch_deg, rad, tint in (
        (25.0, 5.0, 0.22, (0.25, -0.10, -0.05)),
        (100.0, -12.0, 0.18, (-0.08, 0.22, -0.06)),
        (200.0, 8.0, 0.20, (-0.05, -0.08, 0.26)),
        (300.0, -4.0, 0.16, (0.18, 0.18, -0.12)),
    ):
        dt = np.asarray(g.wrap_pi(T - math.radians(yaw_deg)))
        dp = P - math.radians(pitch_deg)
        d2 = (dt * np.cos(P)) ** 2 + dp**2
        blob = np.exp(-d2 / (2.0 * rad * rad))
        for c in range(3):
            img[..., c] += tint[c] * blob
        del dt, dp, d2, blob

    img = np.clip(img, 0.0, 1.0)
    img = lo + img * (hi - lo)
    out = np.clip(img, 0, 255).astype(np.uint8)
    del img
    # Band-limit slightly: a real lens delivers a band-limited image, and an unfiltered
    # procedural checkerboard would otherwise alias on every resample in the chain.
    return _cyclic_blur(out, 0.8)


def render_view(pano: np.ndarray, R: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Sample a virtual pinhole view out of an equirect panorama."""
    rays_cam = g.camera_rays(intr.width, intr.height, intr.K, intr.k1, intr.k2)
    rays_world = rays_cam @ np.asarray(R, dtype=np.float32)  # R.T @ v, vectorised
    u, v = g.equirect_from_rays(rays_world, pano.shape[1], pano.shape[0])
    del rays_cam, rays_world
    return g.sample_equirect(pano, u, v, interpolation=cv2.INTER_LANCZOS4)


def make_capture(
    out_dir: Path,
    pano: np.ndarray,
    orientations: list[tuple[float, float]],
    *,
    hfov: float = 60.0,
    size: tuple[int, int] = (400, 300),
    gains: np.ndarray | None = None,
    prefix: str = "v",
) -> tuple[list[Frame], list[CameraPose]]:
    """Write virtual views to disk and return them as ``(frames, poses)``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    frames: list[Frame] = []
    poses: list[CameraPose] = []
    for k, (yaw_deg, pitch_deg) in enumerate(orientations):
        intr = Intrinsics.from_hfov(w, h, hfov)
        R = g.from_yaw_pitch_roll(math.radians(yaw_deg), math.radians(pitch_deg), 0.0)
        view = render_view(pano, R, intr)
        if gains is not None:
            view = np.clip(view.astype(np.float32) * float(gains[k]), 0, 255).astype(np.uint8)
        path = out_dir / f"{prefix}{k:03d}.png"
        assert cv2.imwrite(str(path), view)
        frames.append(Frame(index=k, path=path, timestamp=k / 8.0, width=w, height=h))
        poses.append(CameraPose(frame_index=k, R=R, intrinsics=intr))
        del view
    return frames, poses


# ======================================================================================
# Measurement helpers
# ======================================================================================


def weighted_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Solid-angle weighted PSNR over the masked region of two equirect images."""
    h, w = a.shape[:2]
    wts = g.sphere_solid_angle_weights(h, w, dtype=np.float64)
    m = mask > 0
    if int(m.sum()) < 256:
        return 0.0
    diff = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    num = float((wts[m][:, None] * diff[m]).sum())
    den = float(wts[m].sum()) * 3.0
    del diff, wts
    mse = num / max(den, 1e-12)
    return 10.0 * math.log10(255.0 * 255.0 / max(mse, 1e-9))


def unshift(img: np.ndarray, yaw_delta_rad: float, *, nearest: bool = False) -> np.ndarray:
    """Undo the renderer's global wrap rotation so the result lines up with the truth."""
    h, w = img.shape[:2]
    shift = yaw_delta_rad / (2.0 * math.pi) * w
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LANCZOS4
    out = g.sample_equirect(img, gx + np.float32(shift), gy, interpolation=interp)
    del gx, gy
    return out


def applied_delta(before: list[CameraPose], after: list[CameraPose]) -> float:
    """The global yaw the renderer rotated the world by, recovered from the poses."""
    return float(g.wrap_pi(after[0].yaw - before[0].yaw))


def single_row(n: int, span_deg: float = 360.0, start_deg: float = 0.0) -> list[tuple[float, float]]:
    if span_deg >= 359.999:
        return [(start_deg + 360.0 * k / n, 0.0) for k in range(n)]
    return [(start_deg + span_deg * k / max(1, n - 1), 0.0) for k in range(n)]


def base_config(**overrides) -> PipelineConfig:
    cfg = PipelineConfig(
        output_width=GT_W,
        interpolation="lanczos4",
        seam_finder="graphcut",
        seam_scale_width=1024,
        blend_bands=5,
        emit_levels=("preview", "uhd"),
        jpeg_quality=94,
        # No branding by default. These tests measure geometry and photometry — pole fill
        # quality above all — and a mark composited over the caps would be measuring the
        # logo instead of the filler underneath it. `test_pole_branding_*` opts back in.
        nadir_logo=None,
        zenith_logo=None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture(scope="module")
def gt() -> np.ndarray:
    return synthetic_pano()


@pytest.fixture(scope="module")
def base_render(tmp_path_factory, gt):
    """A clean 30-view single-row 360 capture, rendered once and reused."""
    root = tmp_path_factory.mktemp("base")
    frames, poses = make_capture(root / "src", gt, single_row(30), hfov=60.0, size=(400, 300))
    cfg = base_config()
    result = render_mod.render_panorama(frames, poses, cfg, out_dir=root / "out")
    return result, poses, root


@pytest.fixture(scope="module")
def gain_render(tmp_path_factory):
    """The same capture with a random per-view gain baked into every source image."""
    truth = synthetic_pano(lo=15.0, hi=175.0)
    root = tmp_path_factory.mktemp("gains")
    rng = np.random.default_rng(4242)
    gains = rng.uniform(0.7, 1.4, size=30)
    frames, poses = make_capture(
        root / "src", truth, single_row(30), hfov=60.0, size=(400, 300), gains=gains
    )
    cfg = base_config()
    result = render_mod.render_panorama(frames, poses, cfg, out_dir=root / "out")
    return result, poses, truth, gains


# ======================================================================================
# 0. Conventions the warper depends on
# ======================================================================================


def test_camera_transpose_convention_matches_geometry():
    """``rays @ R.T`` really is ``R @ d`` applied row-wise -- warp.py depends on it."""
    R = g.from_yaw_pitch_roll(0.7, -0.2, 0.11)
    d = np.array([[0.3, -0.5, 0.81], [0.0, 0.0, 1.0], [-0.6, 0.2, 0.77]], dtype=np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    vector_wise = np.stack([np.asarray(R, np.float32) @ row for row in d])
    row_wise = d @ np.asarray(R, np.float32).T
    assert np.abs(vector_wise - row_wise).max() < 1e-5
    # And geometry.fov_footprint_deg's `rays_cam @ R` is the inverse direction.
    back = row_wise @ np.asarray(R, np.float32)
    assert np.abs(back - d).max() < 1e-5


def test_warp_places_a_centred_camera_at_the_canvas_centre(gt):
    intr = Intrinsics.from_hfov(400, 300, 60.0)
    pose = CameraPose(frame_index=0, R=np.eye(3), intrinsics=intr)
    view = render_view(gt, pose.R, intr)
    patch = warp_mod.warp_camera(view, pose, GT_W, GT_H, base_config())
    x0, y0, x1, y1 = patch.rect
    assert x0 < GT_W // 2 < x1, "a forward-looking camera must straddle the canvas centre"
    assert y0 < GT_H // 2 < y1
    # The patch must reproduce the ground truth where it claims to be valid.
    canvas = np.zeros_like(gt)
    canvas[y0:y1, x0:x1] = patch.image
    mask = np.zeros(gt.shape[:2], dtype=np.uint8)
    mask[y0:y1, x0:x1] = patch.mask
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    assert weighted_psnr(canvas, gt, mask) > 30.0


# ======================================================================================
# 1. Round-trip fidelity
# ======================================================================================


def test_roundtrip_fidelity_above_30db(base_render, gt):
    result, poses_in, _root = base_render
    delta = applied_delta(poses_in, result.poses)
    pano = unshift(result.panorama, delta)
    mask = unshift(result.coverage.mask, delta, nearest=True)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    psnr = weighted_psnr(pano, gt, mask)
    print(f"\n[test 1] round-trip PSNR = {psnr:.2f} dB over {100.0 * (mask > 0).mean():.1f}% of the canvas")
    assert result.panorama.shape == (GT_H, GT_W, 3)
    assert psnr > 30.0, f"round-trip PSNR only {psnr:.2f} dB"


def test_wrap_offset_is_recorded(base_render):
    result, poses_in, root = base_render
    meta = json.loads((root / "out" / "render_meta.json").read_text(encoding="utf-8"))
    assert "wrap_offset_deg" in meta and "yaw_delta_deg" in meta
    recovered = math.degrees(applied_delta(poses_in, result.poses))
    assert abs(g.wrap_pi(math.radians(recovered - meta["yaw_delta_deg"]))) < 1e-3


# ======================================================================================
# 2. No wrap seam
# ======================================================================================


def test_no_visible_wrap_seam(base_render, gt):
    result, poses_in, _root = base_render
    cov = result.coverage.mask
    rows = (cov[:, 0] > 0) & (cov[:, -1] > 0)
    assert rows.sum() > 32, "need covered rows on both sides of the seam to measure it"

    seam_diff, median_adjacent = _seam_ratio(result)
    # How discontinuous the *truth* is across the same meridian, for context: the seam can
    # never be smoother than the scene itself.
    delta = applied_delta(poses_in, result.poses)
    ref = unshift(gt, -delta).astype(np.int32)
    truth_seam = float(np.abs(ref[rows, 0] - ref[rows, -1]).mean())
    del ref
    print(
        f"\n[test 2] seam |d| = {seam_diff:.3f}, median adjacent |d| = {median_adjacent:.3f}, "
        f"ratio = {seam_diff / max(median_adjacent, 1e-6):.2f} (ground truth seam {truth_seam:.3f})"
    )
    assert seam_diff <= 1.5 * median_adjacent, (
        f"wrap seam is visible: {seam_diff:.3f} vs median {median_adjacent:.3f}"
    )
    assert seam_diff <= truth_seam + 1.5 * median_adjacent


# ======================================================================================
# 3. Exposure compensation
# ======================================================================================


def _seam_ratio(result) -> tuple[float, float]:
    """``(mean |col 0 - col W-1|, median adjacent column difference)`` over covered rows."""
    p = result.panorama.astype(np.int32)
    cov = result.coverage.mask
    rows = (cov[:, 0] > 0) & (cov[:, -1] > 0)
    seam = float(np.abs(p[rows, 0] - p[rows, -1]).mean())
    adj = float(np.median(np.abs(p[rows, 1:] - p[rows, :-1]).mean(axis=(0, 2))))
    return seam, adj


def test_wrap_repair_is_load_bearing(tmp_path, gt, monkeypatch):
    """The repair must actually be doing the work, not riding on the wrap offset alone.

    Stub :func:`repair_wrap_band` out and the same capture's seam jumps by roughly 2 grey
    levels -- small in absolute terms but several times the median column-to-column
    difference, which is exactly the kind of edge a viewer's eye finds instantly.  If a
    refactor ever silently no-ops the repair, this is what catches it.

    Note the configuration is chosen so the canvas edge does *not* land on one of the
    fixture's checkerboard edges.  When it does (it depends on the solved wrap offset) the
    ground truth itself is discontinuous across the seam by ~30 grey levels and this ratio
    stops measuring the renderer at all.
    """
    frames, poses = make_capture(
        tmp_path / "src", gt, single_row(20), hfov=60.0, size=(320, 240), prefix="w"
    )
    cfg = base_config(output_width=1024, seam_scale_width=512, blend_bands=3)

    seam_a, adj_a = _seam_ratio(render_mod.render_panorama(frames, poses, cfg, out_dir=tmp_path / "a"))
    monkeypatch.setattr(render_mod, "repair_wrap_band", lambda pano, _f, _p, _c, **_kw: pano)
    seam_b, adj_b = _seam_ratio(render_mod.render_panorama(frames, poses, cfg, out_dir=tmp_path / "b"))
    print(
        f"\n[wrap] seam {seam_a:.3f} (median adj {adj_a:.3f}) with repair, "
        f"{seam_b:.3f} (median adj {adj_b:.3f}) without"
    )
    assert seam_a <= 1.5 * adj_a
    assert seam_b > 2.0 * seam_a, "the wrap repair is not measurably doing anything"


def test_wrap_repair_draws_a_parallaxed_edge_once():
    """Where the panorama and the owner disagree by parallax, one copy survives, not two.

    The reference capture's canvas edge fell on a close wardrobe: the panorama showed a door
    frame 8 px left of the edge (graph cut had given those columns to another camera) and
    the owner showed the same frame 13 px further right. The old 24 px cosine cross-fade
    kept 0.81-0.43 of the first and 0.69-0.31 of the second -- a ghosted double frame.
    """
    band, ph, bg, bar = 46, 64, 60.0, 230.0
    cur = np.full((ph, 2 * band, 3), bg, np.float32)
    strip = np.full((ph, 2 * band, 3), bg, np.float32)
    cur[:, band - 9 : band - 5] = bar      # canvas W-9 .. W-6, in the panorama
    strip[:, band + 4 : band + 8] = bar    # canvas 4 .. 7, as the owner saw it
    out, stats = render_mod._merge_wrap_band(cur, strip, np.ones((ph, 2 * band), bool), feather=2)

    prof = out.mean(axis=(0, 2))
    solid = prof > 0.5 * (bg + bar)
    starts = np.flatnonzero(np.diff(np.concatenate([[0], solid.astype(int), [0]])) == 1)
    assert len(starts) == 1 and int(solid.sum()) == 4, f"the frame came out {len(starts)} times: {prof.round(0)}"
    ghost = (prof > bg + 0.1 * (bar - bg)) & ~solid
    assert not ghost.any(), f"a faded copy survived at {np.flatnonzero(ghost)}"
    assert stats["disagreement"] < 1.0, "the hand-over should have found the agreeing columns"


def test_wrap_repair_leaves_agreeing_pixels_alone():
    """Owner and panorama agreeing up to an exposure offset: the merge changes nothing."""
    rng = np.random.default_rng(3)
    band, ph = 30, 48
    cur = rng.uniform(40, 200, size=(ph, 2 * band, 3)).astype(np.float32)
    cur = cv2.GaussianBlur(cur, (5, 5), 1.0)
    strip = cur - 18.0  # the owner rendered darker; its gain was not matched yet
    out, _ = render_mod._merge_wrap_band(cur, strip, np.ones((ph, 2 * band), bool), feather=2)
    assert float(np.abs(out - cur).max()) < 1.5


def test_exposure_compensation_recovers_the_panorama(gain_render):
    result, poses_in, truth, gains = gain_render
    delta = applied_delta(poses_in, result.poses)
    pano = unshift(result.panorama, delta)
    mask = unshift(result.coverage.mask, delta, nearest=True)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    psnr = weighted_psnr(pano, truth, mask)
    injected = float(np.std(1.0 / gains))
    print(
        f"\n[test 3] gain-corrupted PSNR = {psnr:.2f} dB, "
        f"exposure_spread = {result.metrics.exposure_spread:.4f} (injected {injected:.4f})"
    )
    assert psnr > 28.0, f"exposure compensation left {psnr:.2f} dB"
    assert result.metrics.exposure_spread > 0.08
    assert 0.7 * injected < result.metrics.exposure_spread < 1.4 * injected


def test_exposure_spread_is_small_without_injected_gains(base_render):
    result, _poses_in, _root = base_render
    assert result.metrics.exposure_spread < 0.05


# ======================================================================================
# 4. Seam quality
# ======================================================================================


def test_seam_gradient_energy(base_render):
    result, _poses_in, _root = base_render
    energy = result.metrics.seam_gradient_energy
    print(f"\n[test 4] seam_gradient_energy = {energy:.4f}")
    assert 0.0 < energy < 1.35, f"seams are visible: gradient energy {energy:.3f}"


# ======================================================================================
# 5. Coverage
# ======================================================================================


def test_partial_capture_reports_yaw_span_and_gaps(tmp_path, gt):
    # A wide-FOV single row so the swept arc genuinely covers the shootable elevations;
    # 10 views spanning yaw centres 0..130 with a 110 deg horizontal FOV cover 240 deg.
    orientations = [(130.0 * k / 9.0, 0.0) for k in range(10)]
    frames, poses = make_capture(
        tmp_path / "src", gt, orientations, hfov=110.0, size=(400, 400), prefix="p"
    )
    cfg = base_config(output_width=1024, seam_scale_width=512, blend_bands=4)
    result = render_mod.render_panorama(frames, poses, cfg, out_dir=tmp_path / "out")
    rep = result.coverage
    gap_spans = [y1 - y0 for (y0, y1, _p0, _p1) in rep.gaps]
    delta_deg = math.degrees(applied_delta(poses, result.poses))
    print(
        f"\n[test 5] yaw_span = {rep.yaw_span_deg:.1f} deg, covered = {100 * rep.covered_fraction:.1f}%, "
        f"yaw delta = {delta_deg:.1f} deg, "
        f"gaps = {[(round(a, 1), round(b, 1)) for a, b, _c, _d in rep.gaps]}"
    )
    assert 225.0 <= rep.yaw_span_deg <= 255.0, f"yaw span {rep.yaw_span_deg:.1f} is not ~240"
    assert rep.gaps, "a 240 deg capture must report the missing sector"
    # The missing sector runs 185..305 deg in the input frame; the reported gap is in the
    # rendered frame, so it must sit at that centre plus the applied wrap rotation.  It is
    # wider than 120 deg because a pinhole footprint's top and bottom edges curve inward,
    # so the last frame stops covering the upper elevations before its yaw limit.
    assert 110.0 <= max(gap_spans) <= 180.0, f"largest gap spans {max(gap_spans):.1f} deg, expected ~120"
    biggest = rep.gaps[0]
    gap_centre = 0.5 * (biggest[0] + biggest[1])
    expected = (245.0 + delta_deg) % 360.0
    assert abs(g.wrap_pi(math.radians(gap_centre - expected))) < math.radians(20.0), (
        f"gap centre {gap_centre:.1f} deg does not match the missing sector at {expected:.1f} deg"
    )
    assert not rep.zenith_covered and not rep.nadir_covered
    assert (tmp_path / "out" / "coverage.png").exists()


def test_full_capture_reports_no_gaps(base_render):
    result, _poses_in, _root = base_render
    rep = result.coverage
    assert rep.yaw_span_deg > 355.0
    assert rep.gaps == []
    assert rep.horizon_fraction > 0.7
    assert 0.0 < rep.covered_fraction < 1.0


def test_coverage_heatmap_is_written(tmp_path, base_render):
    result, _poses_in, _root = base_render
    path = coverage_mod.coverage_heatmap_png(result.coverage, tmp_path / "hm.png")
    img = cv2.imread(str(path))
    assert img is not None and img.shape == (512, 1024, 3)


# ======================================================================================
# 6. Pole fill
# ======================================================================================


def band_psnr(a: np.ndarray, b: np.ndarray, lo_deg: float, hi_deg: float) -> float:
    """PSNR restricted to an elevation band of two equirect images."""
    h = a.shape[0]
    v0 = max(0, int(round((90.0 - hi_deg) / 180.0 * h)))
    v1 = min(h, int(round((90.0 - lo_deg) / 180.0 * h)))
    if v1 <= v0:
        return float("nan")
    d = (a[v0:v1].astype(np.float64) - b[v0:v1].astype(np.float64)) ** 2
    return 10.0 * math.log10(255.0 * 255.0 / max(float(d.mean()), 1e-9))


def flat_ring_fill(pano: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """What the caps would look like filled with the plain mean of their boundary ring.

    This is what any diffusion inpainter converges to over a large hole, and it is the
    behaviour :mod:`poles` deliberately improves on -- so it is the baseline every pole
    test scores against.
    """
    out = pano.copy()
    rows = np.nonzero(coverage.mean(axis=1) > 0.9 * 255)[0]
    if rows.size < 24:
        return out
    top, bot = int(rows.min()), int(rows.max())
    out[:top] = pano[top : top + 12].reshape(-1, 3).mean(axis=0).astype(np.uint8)
    out[bot + 1 :] = pano[bot - 11 : bot + 1].reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return out


def test_poles_are_filled_plausibly(base_render, gt):
    result, poses_in, _root = base_render
    delta = applied_delta(poses_in, result.poses)
    pano = unshift(result.panorama, delta)
    cov = unshift(result.coverage.mask, delta, nearest=True)
    h = pano.shape[0]

    top = int((90.0 - 75.0) / 180.0 * h)
    bottom = int((90.0 + 75.0) / 180.0 * h)
    assert int((pano[:top].max(axis=2) == 0).sum()) == 0, "black pixels remain above +75 deg"
    assert int((pano[bottom:].max(axis=2) == 0).sum()) == 0, "black pixels remain below -75 deg"

    # The substantive claim: the cap continues the ring's brightness and colour *trend*,
    # so it must beat the flat ring-mean fill a diffusion inpainter would have produced.
    flat = flat_ring_fill(pano, cov)
    zen_trend, zen_flat = band_psnr(pano, gt, 57.6, 90.0), band_psnr(flat, gt, 57.6, 90.0)
    nad_trend, nad_flat = band_psnr(pano, gt, -90.0, -57.6), band_psnr(flat, gt, -90.0, -57.6)
    print(
        f"\n[test 6] zenith {zen_trend:.2f} dB vs flat-ring {zen_flat:.2f} dB; "
        f"nadir {nad_trend:.2f} dB vs flat-ring {nad_flat:.2f} dB"
    )
    assert zen_trend > zen_flat + 2.0, "the zenith fill is not following the ring's trend"
    assert nad_trend > nad_flat + 2.0, "the nadir fill is not following the ring's trend"

    # ...and it must stay honestly smooth rather than inventing ceiling texture.
    grey = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    detail = cv2.magnitude(
        cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    )
    fill_detail = float(detail[:top].mean())
    captured_detail = float(detail[cov.mean(axis=1) > 0.9 * 255].mean())
    print(f"[test 6] fill detail {fill_detail:.2f} vs captured {captured_detail:.2f}")
    assert fill_detail < 0.35 * captured_detail, "the pole fill is fabricating detail"
    del flat, grey, detail


def test_pole_fill_carries_the_trend_into_a_room(tmp_path):
    """A bright ceiling and a dark floor: the case a ring-mean fill gets wrong at both ends.

    The sphere-filling fixture used elsewhere has a gentle sinusoidal falloff; a real
    interior does not.  Its ceiling is the brightest surface and its floor the darkest, so
    filling either cap with the average of the walls is biased the wrong way -- which is
    exactly what this guards.
    """
    theta, phi = g.equirect_angles(1024, 512, dtype=np.float64)
    T, P = theta[None, :], np.degrees(phi)[:, None]
    wall = 0.42 + 0.0035 * np.clip(P, -40, 40) + 0.05 * np.sign(np.cos(28.0 * T))
    ceiling = 0.86 + 0.05 * np.cos(3.0 * T) * np.ones_like(P)
    floor = 0.17 + 0.03 * np.sign(np.cos(9.0 * T)) * np.ones_like(P)
    tc = np.clip((P - 38.0) / 10.0, 0.0, 1.0)
    tf = np.clip((-P - 40.0) / 10.0, 0.0, 1.0)
    lum = wall * (1 - tc) + ceiling * tc
    lum = lum * (1 - tf) + floor * tf
    # Warm ceiling, cool floor -- chroma must carry into the caps as well as brightness.
    warmth = np.clip((P - 20.0) / 40.0, -1.0, 1.0)
    room = np.stack([lum * (1.0 - 0.10 * warmth), lum, lum * (1.0 + 0.12 * warmth)], axis=-1)
    room = np.clip(room * 255.0, 0, 255).astype(np.uint8)
    room = _cyclic_blur(room, 1.0)

    frames, poses = make_capture(
        tmp_path / "src", room, single_row(24), hfov=70.0, size=(420, 420), prefix="room"
    )
    cfg = base_config(output_width=1024, seam_scale_width=512, blend_bands=4)
    result = render_mod.render_panorama(frames, poses, cfg, out_dir=tmp_path / "out")
    delta = applied_delta(poses, result.poses)
    pano = unshift(result.panorama, delta)
    cov = unshift(result.coverage.mask, delta, nearest=True)
    flat = flat_ring_fill(pano, cov)

    bands = [(57.6, 90.0, "zenith fill"), (32.4, 57.6, "upper wall"), (-90.0, -57.6, "nadir fill")]
    print()
    for lo, hi, name in bands:
        print(
            f"[room] {name:11s} {band_psnr(pano, room, lo, hi):6.2f} dB "
            f"(flat-ring {band_psnr(flat, room, lo, hi):6.2f} dB)"
        )
    assert band_psnr(pano, room, 57.6, 90.0) > band_psnr(flat, room, 57.6, 90.0) + 4.0
    assert band_psnr(pano, room, -90.0, -57.6) > band_psnr(flat, room, -90.0, -57.6) + 4.0

    # The ceiling must arrive bright and warm, not mid-grey and neutral.
    h = pano.shape[0]
    zen = pano[: int(h * 0.06)].reshape(-1, 3).mean(axis=0)
    ring = pano[cov.mean(axis=1) > 0.9 * 255][:2048].reshape(-1, 3).mean(axis=0)
    truth = room[: int(h * 0.06)].reshape(-1, 3).mean(axis=0)
    print(f"[room] zenith fill {zen.round(1)}  ring {ring.round(1)}  truth {truth.round(1)}")
    assert zen.mean() > ring.mean() + 25.0, "the ceiling fill did not follow the wall upward"
    assert (zen[2] - zen[0]) > 0.4 * (truth[2] - truth[0]), "the warm ceiling turned neutral"


def test_gradient_pole_method_also_fills(tmp_path, gt):
    frames, poses = make_capture(
        tmp_path / "src", gt, single_row(16), hfov=60.0, size=(320, 240), prefix="grad"
    )
    cfg = base_config(
        output_width=1024,
        seam_scale_width=512,
        blend_bands=4,
        pole_method="gradient",
        emit_cubemap=True,
    )
    result = render_mod.render_panorama(frames, poses, cfg, out_dir=tmp_path / "out")
    h = result.panorama.shape[0]
    top = int((90.0 - 75.0) / 180.0 * h)
    assert int((result.panorama[:top].max(axis=2) == 0).sum()) == 0
    # The cube map is emitted from inside the pipeline when the config asks for it.
    assert set(result.cube_faces) == {"px", "nx", "py", "ny", "pz", "nz"}
    for path in result.cube_faces.values():
        assert cv2.imread(str(path)) is not None


# ======================================================================================
# 6b. Uncovered regions that are not at a pole
# ======================================================================================
#
# A capture that misses a patch of wall is the common case, not an exotic one -- the agent
# turns too fast, an arm crosses the lens, the bit behind the wardrobe never gets swept.
# The pole filler cannot help there, and a black wedge in a 360 viewer is a void the user
# turns their head and falls into.  These tests work on a synthetic coverage mask rather
# than on a capture, because that is the only way to put a hole exactly where the claim is
# about and leave the poles alone.


def punched(pano: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """``(pano, coverage)`` with ``(r0, r1, c0, c1)`` blocks knocked out.  Columns wrap."""
    h, w = pano.shape[:2]
    out = pano.copy()
    coverage = np.full((h, w), 255, np.uint8)
    for r0, r1, c0, c1 in boxes:
        cols = np.arange(c0, c1) % w
        coverage[r0:r1, cols] = 0
        out[r0:r1, cols] = 0
    return out, coverage


def detail_of(img: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.magnitude(
        cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    )


def test_mid_sphere_hole_is_filled(gt):
    """A wedge above the horizon, nowhere near a pole, must not survive as black."""
    h, w = gt.shape[:2]
    r0, r1 = int(h * 0.20), int(h * 0.36)  # elevation +25 to +55 deg
    c0, c1 = int(w * 0.30), int(w * 0.52)  # 79 deg of yaw
    pano, coverage = punched(gt, [(r0, r1, c0, c1)])
    assert int((pano.max(axis=2) == 0).sum()) > 0, "the fixture did not punch a hole"

    filled = poles_mod.fill_uncovered(pano, coverage, base_config(output_width=w))
    hole = coverage == 0

    assert int((filled.max(axis=2) == 0).sum()) == 0, "a mid-sphere hole was left black"
    # Locally correct, not just non-black: it must beat filling with the panorama's own
    # global mean, which is what an unaware fill would amount to.
    naive = gt.copy()
    naive[hole] = gt[~hole].reshape(-1, 3).mean(axis=0).astype(np.uint8)
    got = float(np.abs(filled[hole].astype(np.float64) - gt[hole].astype(np.float64)).mean())
    flat = float(np.abs(naive[hole].astype(np.float64) - gt[hole].astype(np.float64)).mean())
    print(f"\n[fill] mid-sphere hole: mean error {got:.1f} vs global-mean fill {flat:.1f}")
    assert got < 0.7 * flat, "the fill is not following the surrounding content"

    # ...and it must stay an honest blur rather than invent wallpaper.
    detail = detail_of(filled)
    inner = cv2.erode(hole.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    fill_detail = float(detail[inner].mean())
    real_detail = float(detail[~hole].mean())
    print(f"[fill] detail inside {fill_detail:.2f} vs captured {real_detail:.2f}")
    assert fill_detail < 0.5 * real_detail, "the fill is fabricating detail"


def test_hole_across_the_canvas_edge_is_filled_as_one_region(gt):
    """Column 0 and column W-1 are neighbours; a hole spanning both is *one* hole.

    Filled as two, each half is inpainted only from the content on its own side and they
    meet at ``u = 0`` with a step -- straight through the middle of the viewer's opening
    view, where the wrap machinery cannot help because the pixels are invented rather than
    warped.  The fixture makes that unmissable: the content bounding the two sides of the
    hole differs by 180 grey levels.

    The hole is deliberately lopsided about the seam, 100 columns before it and 20 after.
    A symmetric one cannot tell the two outcomes apart: any fill that marches in from both
    boundaries meets in the middle, and for a symmetric hole the middle *is* ``u = 0``, so
    the honest answer and the broken one both put their transition there.
    """
    h, w = 256, 512
    before, after = 100, 20
    col = np.arange(w)
    # Bright just left of the hole, dark just right of it, ramping the long way round.  The
    # only smooth way across the hole is through the canvas edge.
    t = (col - after) / float(w - before - after)
    lum = np.clip(40.0 + 180.0 * t, 40.0, 220.0)
    pano = np.repeat(np.repeat(lum[None, :, None], h, axis=0), 3, axis=2).astype(np.uint8)
    pano, coverage = punched(pano, [(60, h - 60, w - before, w + after)])

    filled = poles_mod.fill_uncovered(pano, coverage, base_config(output_width=w))
    assert int((filled.max(axis=2) == 0).sum()) == 0, "the wrapping hole was left black"

    rows = slice(70, h - 70)
    band = filled[rows][:, (np.arange(w - before, w + after) % w)].astype(np.float64)
    steps = np.abs(np.diff(band, axis=1)).mean(axis=(0, 2))  # per column boundary in the hole
    seam = float(
        np.abs(filled[rows, 0].astype(np.float64) - filled[rows, w - 1].astype(np.float64)).mean()
    )
    print(
        f"\n[fill] step across u=0: {seam:.2f}; worst step inside the fill "
        f"{float(steps.max()):.2f}, median {float(np.median(steps)):.2f}"
    )
    # Two independent fills would meet with most of the 180 level difference at the edge.
    assert seam < 30.0, f"the wrapping hole was filled as two regions: {seam:.1f} level step at u=0"
    # ...and the real claim: u = 0 is not a special place inside the fill.
    assert seam <= float(steps.max()) + 1e-6, "the canvas edge is the worst seam in the fill"


def test_filling_leaves_the_coverage_report_untouched(gt):
    """Coverage is what the camera saw.  Filling a pixel must not make it covered."""
    h, w = gt.shape[:2]
    boxes = [
        (int(h * 0.22), int(h * 0.34), int(w * 0.30), int(w * 0.52)),  # mid-sphere
        (0, int(h * 0.10), 0, w),  # zenith cap
        (int(h * 0.92), h, 0, w),  # nadir cap
    ]
    pano, coverage = punched(gt, boxes)
    before = coverage.copy()
    report_before = coverage_mod.build_coverage([coverage], [(0, 0)], w, h)
    snapshot = (
        report_before.covered_fraction,
        report_before.horizon_fraction,
        report_before.yaw_span_deg,
        report_before.pitch_min_deg,
        report_before.pitch_max_deg,
        report_before.zenith_covered,
        report_before.nadir_covered,
        list(report_before.gaps),
    )

    filled = poles_mod.fill_uncovered(pano, coverage, base_config(output_width=w))

    assert int((filled.max(axis=2) == 0).sum()) == 0
    assert np.array_equal(coverage, before), "the filler wrote to the coverage mask"
    report_after = coverage_mod.build_coverage([coverage], [(0, 0)], w, h)
    assert (
        report_after.covered_fraction,
        report_after.horizon_fraction,
        report_after.yaw_span_deg,
        report_after.pitch_min_deg,
        report_after.pitch_max_deg,
        report_after.zenith_covered,
        report_after.nadir_covered,
        list(report_after.gaps),
    ) == snapshot
    # The point of the whole exercise: the panorama is complete and the report still says
    # part of the sphere was never reached, at the same number it said before.
    assert snapshot[0] < 0.95 and report_after.gaps
    assert not report_before.zenith_covered and not report_before.nadir_covered


def test_partial_capture_delivers_no_black_pixels(tmp_path, gt):
    """End to end: a 240 degree sweep leaves a wedge, and the delivered panorama has none."""
    orientations = [(130.0 * k / 9.0, 0.0) for k in range(10)]
    frames, poses = make_capture(
        tmp_path / "src", gt, orientations, hfov=110.0, size=(400, 400), prefix="wedge"
    )
    cfg = base_config(output_width=1024, seam_scale_width=512, blend_bands=4)
    result = render_mod.render_panorama(frames, poses, cfg, out_dir=tmp_path / "out")
    pano, rep = result.panorama, result.coverage
    black = int((pano.max(axis=2) == 0).sum())
    print(
        f"\n[fill] 240 deg sweep: {black} black px, covered {100 * rep.covered_fraction:.1f}%, "
        f"{len(rep.gaps)} gap(s)"
    )
    assert black == 0, f"{black} black pixels survived a partial capture"
    # ...while coverage still reports the missing sector, undisturbed by the fill.
    assert rep.covered_fraction < 0.85 and rep.gaps
    assert int((rep.mask == 0).sum()) > 0


def test_nadir_logo_is_composited(tmp_path, gt):
    logo = np.zeros((256, 256, 4), dtype=np.uint8)
    cv2.circle(logo, (128, 128), 110, (40, 30, 220, 255), -1)
    logo_path = tmp_path / "logo.png"
    assert cv2.imwrite(str(logo_path), logo)

    frames, poses = make_capture(
        tmp_path / "src", gt, single_row(12), hfov=60.0, size=(320, 240), prefix="logo"
    )
    cfg = base_config(
        output_width=1024, seam_scale_width=512, blend_bands=4, nadir_logo=logo_path
    )
    result = render_mod.render_panorama(frames, poses, cfg, out_dir=tmp_path / "out")
    pano = result.panorama
    h = pano.shape[0]
    nadir = pano[int(h * (90.0 + 84.0) / 180.0) :]
    # The logo is a saturated red disc; nothing in the synthetic world looks like that.
    reddish = (nadir[..., 2].astype(np.int32) - nadir[..., 0].astype(np.int32)) > 90
    assert reddish.mean() > 0.5, "the nadir logo did not land on the nadir"
    # ...and it must stay on the nadir: the horizon band is untouched.
    band = pano[h // 2 - 20 : h // 2 + 20]
    assert (((band[..., 2].astype(np.int32) - band[..., 0].astype(np.int32)) > 90).mean()) < 0.01


def test_vignette_is_estimated_and_removed(tmp_path):
    truth = synthetic_pano(lo=25.0, hi=205.0)
    intr = Intrinsics.from_hfov(400, 300, 60.0)
    rn2 = (intr.width * 0.5) ** 2 + (intr.height * 0.5) ** 2
    gx, gy = g.pixel_grid(intr.width, intr.height)
    r2 = (((gx - intr.cx) ** 2 + (gy - intr.cy) ** 2) / rn2).astype(np.float64)
    a_true = (-0.32, 0.08, 0.0)
    falloff = 1.0 + r2 * (a_true[0] + r2 * (a_true[1] + r2 * a_true[2]))

    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    frames: list[Frame] = []
    poses: list[CameraPose] = []
    for k, (yaw_deg, pitch_deg) in enumerate(single_row(24)):
        R = g.from_yaw_pitch_roll(math.radians(yaw_deg), math.radians(pitch_deg), 0.0)
        view = render_view(truth, R, intr).astype(np.float32) * falloff[..., None]
        path = src / f"vig{k:03d}.png"
        assert cv2.imwrite(str(path), np.clip(view, 0, 255).astype(np.uint8))
        frames.append(Frame(index=k, path=path, timestamp=k / 8.0, width=400, height=300))
        poses.append(CameraPose(frame_index=k, R=R, intrinsics=intr))
        del view

    cfg = base_config(output_width=1024, seam_scale_width=512, blend_bands=4)
    result = render_mod.render_panorama(frames, poses, cfg, out_dir=tmp_path / "out")
    fitted = result.poses[0].vignette
    r = np.linspace(0.0, 1.0, 51)
    from biosphere.pano.exposure import vignette_curve

    err = float(np.abs(vignette_curve(fitted, r) - vignette_curve(a_true, r)).max())
    print(f"\n[vignette] fitted {tuple(round(v, 4) for v in fitted)} vs {a_true}, max curve error {err:.4f}")
    assert err < 0.06, f"vignette curve is off by {err:.3f} at worst"

    delta = applied_delta(poses, result.poses)
    reference = cv2.resize(truth, (cfg.output_width, cfg.output_height), interpolation=cv2.INTER_AREA)
    mask = cv2.erode(unshift(result.coverage.mask, delta, nearest=True), np.ones((7, 7), np.uint8))
    psnr = weighted_psnr(unshift(result.panorama, delta), reference, mask)
    print(f"[vignette] PSNR after correction = {psnr:.2f} dB")
    assert psnr > 30.0


# ======================================================================================
# 7. Levels
# ======================================================================================


def test_emitted_levels_are_readable_and_descend(base_render):
    result, _poses_in, _root = base_render
    assert result.levels, "at least one level must be emitted"
    for lvl in result.levels:
        assert lvl.width == 2 * lvl.height, f"{lvl.name} is not 2:1"
        img = cv2.imread(str(lvl.path))
        assert img is not None, f"{lvl.path} is not a readable JPEG"
        assert img.shape[:2] == (lvl.height, lvl.width)
        assert lvl.bytes > 0
    ordered = sorted(result.levels, key=lambda level: -level.width)
    widths = [level.width for level in ordered]
    sizes = [level.bytes for level in ordered]
    assert widths == sorted(widths, reverse=True)
    assert all(a > b for a, b in zip(sizes, sizes[1:], strict=False)), (
        f"file sizes do not descend: {sizes}"
    )
    assert result.thumbnail is not None and result.thumbnail.shape == (360, 640, 3)


def test_write_levels_full_pyramid(tmp_path, gt):
    pano = cv2.resize(gt, (4096, 2048), interpolation=cv2.INTER_CUBIC)
    cfg = base_config(output_width=4096, emit_levels=("preview", "sd", "hd", "uhd"))
    levels = output_mod.write_levels(pano, tmp_path, cfg)
    names = [lv.name for lv in levels]
    assert names == ["preview", "sd", "uhd"], f"unexpected rungs {names} (hd must not upscale)"
    widths = [lv.width for lv in levels]
    assert widths == [512, 2048, 4096]
    for lv in levels:
        assert lv.width == 2 * lv.height
        assert cv2.imread(str(lv.path)) is not None
    sizes = [lv.bytes for lv in levels]
    assert sizes == sorted(sizes), f"file sizes must grow with width: {sizes}"
    assert (tmp_path / "preview.webp").exists()
    del pano


def test_cubemap_faces_are_square_and_consistent(tmp_path, gt):
    faces = output_mod.write_cubemap(gt, tmp_path, 128)
    assert set(faces) == {"px", "nx", "py", "ny", "pz", "nz"}
    imgs = {}
    for name, path in faces.items():
        img = cv2.imread(str(path))
        assert img is not None and img.shape == (128, 128, 3)
        imgs[name] = img
    # three.js -Z is this engine's +Z, so the nz face must show the panorama's centre.
    rays = output_mod._face_rays("nz", 128)
    u, v = g.equirect_from_rays(rays, gt.shape[1], gt.shape[0])
    centre = g.sample_equirect(gt, u, v, interpolation=cv2.INTER_LANCZOS4)
    assert weighted_psnr(imgs["nz"], centre, np.full((128, 128), 255, np.uint8)) > 24.0
    # An odd face size has a true centre pixel; it must land on the panorama's centre.
    uo, vo = g.equirect_from_rays(output_mod._face_rays("nz", 129), gt.shape[1], gt.shape[0])
    assert abs(float(uo[64, 64]) - (gt.shape[1] / 2.0 - 0.5)) < 0.01
    assert abs(float(vo[64, 64]) - (gt.shape[0] / 2.0 - 0.5)) < 0.01
    # py must look at the zenith (three.js +Y up == this engine's -Y).
    _up, v_up = g.equirect_from_rays(output_mod._face_rays("py", 129), gt.shape[1], gt.shape[0])
    assert float(v_up[64, 64]) < 0.5


# ======================================================================================
# 8. Tiled blend equivalence
# ======================================================================================


def _build_patches(tmp_path, gt, cfg, n_views=16):
    frames, poses = make_capture(
        tmp_path / "src", gt, single_row(n_views), hfov=60.0, size=(400, 300), prefix="t"
    )
    offset = render_mod.choose_wrap_offset(poses, None)
    work, delta = render_mod.apply_wrap_offset(poses, offset)
    patches = warp_mod.warp_all(frames, work, cfg.output_width, cfg.output_height, cfg)
    feather = seams_mod.find_seams(patches, cfg)
    return patches, feather, delta


def test_blend_tiled_matches_multiband(tmp_path, gt):
    cfg = base_config(output_width=1024, seam_scale_width=512, blend_bands=4)
    patches, feather, delta = _build_patches(tmp_path, gt, cfg)
    W, H = cfg.output_width, cfg.output_height

    direct = blend_mod.multiband_blend(patches, feather, W, H, cfg)
    tiled = blend_mod.blend_tiled(patches, feather, W, H, cfg, strip_rows=160)
    coverage = blend_mod.build_canvas_coverage(
        [p.mask for p in patches], [p.corner for p in patches], W, H
    )
    inner = cv2.erode(coverage, np.ones((9, 9), np.uint8))

    reference = cv2.resize(gt, (W, H), interpolation=cv2.INTER_AREA)
    psnr_direct = weighted_psnr(unshift(direct, delta), reference, inner)
    psnr_tiled = weighted_psnr(unshift(tiled, delta), reference, inner)
    psnr_pair = weighted_psnr(direct, tiled, inner)
    print(
        f"\n[test 8] direct {psnr_direct:.2f} dB, tiled {psnr_tiled:.2f} dB, "
        f"pairwise {psnr_pair:.2f} dB"
    )
    assert abs(psnr_direct - psnr_tiled) < 2.0
    assert psnr_pair > 30.0
    del patches, feather, direct, tiled, coverage, inner, reference


@pytest.mark.parametrize("finder", ["voronoi", "dp", "graphcut"])
def test_every_seam_backend_partitions_the_coverage(tmp_path, gt, finder):
    cfg = base_config(output_width=512, seam_scale_width=256, blend_bands=3, seam_finder=finder)
    patches, feather, _delta = _build_patches(tmp_path, gt, cfg, n_views=10)
    hard, _feather2, backend = seams_mod.find_seams_full(patches, cfg)
    W, H = cfg.output_width, cfg.output_height
    covered = blend_mod.build_canvas_coverage([p.mask for p in patches], [p.corner for p in patches], W, H)
    claimed = blend_mod.build_canvas_coverage(hard, [p.corner for p in patches], W, H)
    orphans = int(((covered > 0) & (claimed == 0)).sum())
    weight = blend_mod.build_canvas_coverage(feather, [p.corner for p in patches], W, H)
    starved = int(((covered > 0) & (weight == 0)).sum())
    print(f"\n[seams:{finder}] backend={backend}, orphans={orphans}, starved={starved}")
    assert orphans == 0, "every covered pixel must belong to exactly one image"
    assert starved == 0, "no covered pixel may end up with zero blend weight"
    del patches, feather, hard


def test_single_view_render_is_survivable(tmp_path, gt):
    frames, poses = make_capture(
        tmp_path / "src", gt, [(0.0, 0.0)], hfov=60.0, size=(320, 240), prefix="one"
    )
    cfg = base_config(output_width=512, seam_scale_width=256, blend_bands=3, pole_method="gradient")
    result = render_mod.render_panorama(frames, poses, cfg, out_dir=tmp_path / "out")
    assert result.panorama.shape == (256, 512, 3)
    assert 0.0 < result.coverage.covered_fraction < 0.2
    assert result.coverage.yaw_span_deg < 120.0
    assert result.levels


def test_feather_blend_is_a_usable_fallback(tmp_path, gt):
    cfg = base_config(output_width=512, seam_scale_width=256, blend_bands=3)
    patches, feather, delta = _build_patches(tmp_path, gt, cfg, n_views=12)
    W, H = cfg.output_width, cfg.output_height
    out = blend_mod.feather_blend(patches, feather, W, H, cfg)
    coverage = blend_mod.build_canvas_coverage(
        [p.mask for p in patches], [p.corner for p in patches], W, H
    )
    assert out.shape == (H, W, 3)
    assert (out[coverage == 0] == 0).all(), "uncovered pixels must be exactly black"
    reference = cv2.resize(gt, (W, H), interpolation=cv2.INTER_AREA)
    inner = cv2.erode(coverage, np.ones((9, 9), np.uint8))
    assert weighted_psnr(unshift(out, delta), reference, inner) > 22.0


# ======================================================================================
# Optical-flow refinement
# ======================================================================================


def test_optical_flow_refinement_runs_and_preserves_quality(tmp_path, gt):
    cfg = base_config(
        output_width=1024,
        seam_scale_width=512,
        blend_bands=4,
        optical_flow_refine=True,
        flow_backend="dis",
    )
    frames, poses = make_capture(
        tmp_path / "src", gt, single_row(16), hfov=60.0, size=(400, 300), prefix="f"
    )
    result = render_mod.render_panorama(frames, poses, cfg, out_dir=tmp_path / "out")
    delta = applied_delta(poses, result.poses)
    reference = cv2.resize(gt, (cfg.output_width, cfg.output_height), interpolation=cv2.INTER_AREA)
    mask = cv2.erode(unshift(result.coverage.mask, delta, nearest=True), np.ones((7, 7), np.uint8))
    psnr = weighted_psnr(unshift(result.panorama, delta), reference, mask)
    print(f"\n[flow] refined PSNR = {psnr:.2f} dB, backend = {refine_mod.select_flow_backend(cfg)}")
    assert psnr > 27.0, "flow refinement must not damage an already-aligned capture"
    assert result.metrics.stage_seconds["refine"] > 0.0


def test_flow_sanity_gate_rejects_huge_displacements(tmp_path, gt):
    cfg = base_config(
        output_width=512, seam_scale_width=256, blend_bands=3,
        optical_flow_refine=True, flow_backend="dis", flow_max_shift_px=0.0,
    )
    patches, feather, _delta = _build_patches(tmp_path, gt, cfg, n_views=10)
    refined = refine_mod.refine_overlaps(patches, cfg)
    # With a zero tolerance every pair must be rejected, so nothing may move.
    for a, b in zip(patches, refined, strict=True):
        assert np.array_equal(a.image, b.image)
    del patches, feather, refined


# ======================================================================================
# 9. Memory / full-size render
# ======================================================================================


def _peak_rss_mb() -> float:
    """Peak working set in MB.  psutil if present, else the Win32 counter, else 0."""
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().peak_wset) / 1e6
    except Exception:  # noqa: BLE001 - psutil is optional on this machine
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        kernel32 = ctypes.windll.kernel32
        # Without an explicit restype the 64-bit pseudo-handle is truncated to int32 and
        # the call silently fails.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        if get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return float(counters.PeakWorkingSetSize) / 1e6
    except Exception:  # noqa: BLE001 - non-Windows or restricted process
        pass
    return 0.0


@pytest.mark.slow
def test_full_8k_render_of_24_views(tmp_path, gt):
    orientations = single_row(24)
    frames, poses = make_capture(
        tmp_path / "src", gt, orientations, hfov=65.0, size=(1600, 1200), prefix="k"
    )
    cfg = base_config(
        output_width=8192,
        seam_scale_width=1024,
        blend_bands=5,
        emit_levels=("preview", "sd", "uhd"),
    )
    import time

    t0 = time.perf_counter()
    result = render_mod.render_panorama(frames, poses, cfg, out_dir=tmp_path / "out")
    elapsed = time.perf_counter() - t0
    peak_mb = _peak_rss_mb()

    # The 8K render must not merely complete, it must be *right*: downscaled back to the
    # ground truth's resolution it has to reproduce it.
    delta = applied_delta(poses, result.poses)
    small = cv2.resize(result.panorama, (GT_W, GT_H), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(result.coverage.mask, (GT_W, GT_H), interpolation=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    psnr = weighted_psnr(unshift(small, delta), gt, unshift(mask, delta, nearest=True))
    print(
        f"\n[test 9] 8192x4096 render of 24 views in {elapsed:.1f} s, "
        f"peak RSS {peak_mb:.0f} MB, PSNR {psnr:.2f} dB, "
        f"seam energy {result.metrics.seam_gradient_energy:.3f}; "
        f"stages = { {k: round(v, 1) for k, v in result.metrics.stage_seconds.items()} }"
    )
    assert result.panorama.shape == (4096, 8192, 3)
    assert result.coverage.covered_fraction > 0.2
    assert {lv.name for lv in result.levels} == {"preview", "sd", "uhd"}
    assert result.metrics.seam_gradient_energy < 1.6
    assert psnr > 28.0, f"the 8K render only reaches {psnr:.2f} dB"
    del result, small, mask


# ======================================================================================
# Taking structure from the seam's owner instead of from the cross-fade
# ======================================================================================


def _ghost_pair(w: int = 1024, h: int = 512, shift_px: int = 6, gain: float = 1.08):
    """Two views of the same content, offset and mismatched the way a real sweep leaves them.

    6 px is the scale of misalignment the solver actually reports on a hand-held capture
    (2.75 px mean, 15 px worst), and 8% is what survives gain compensation.
    """
    from biosphere.pano.warp import WarpedPatch

    rng = np.random.default_rng(11)
    base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 1.6)  # texture, not white noise
    moved = cv2.warpAffine(
        base, np.float32([[1, 0, shift_px], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REFLECT_101
    )
    moved = np.clip(moved.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    pose = CameraPose(frame_index=0, R=np.eye(3), intrinsics=Intrinsics.from_hfov(w, h, 43.0))
    full = np.full((h, w), 255, np.uint8)
    a = WarpedPatch(0, 0, base, full.copy(), (0, 0), w, h, pose)
    b = WarpedPatch(1, 1, moved, full.copy(), (0, 0), w, h, pose)
    hard_a = np.zeros((h, w), np.uint8); hard_a[:, : w // 2] = 255
    hard_b = np.zeros((h, w), np.uint8); hard_b[:, w // 2 :] = 255
    return a, b, [hard_a, hard_b]


def _high_frequency_ghost(out, a_img, b_img, sigma):
    """How far the result sits from BOTH sources once the low band is removed.

    A pixel that came from one camera matches that camera; a pixel that was averaged
    matches neither. High-passed first, because the low band is *supposed* to be a mixture
    — that is how the exposure step is hidden.

    `sigma` must be the SPLIT the deghoster used. Measuring a wider band than the split
    counts frequencies the low pass still owns, and grades the fix for work it never
    claimed to do.
    """
    def hp(x):
        f = x.astype(np.float32)
        return f - cv2.GaussianBlur(f, (0, 0), sigma)

    o, fa, fb = hp(out), hp(a_img), hp(b_img)
    return float(np.minimum(np.abs(o - fa), np.abs(o - fb)).mean())


def test_deghosting_takes_structure_from_one_camera_at_a_time():
    """The measurement this feature exists for, at the width it is tuned for.

    4096 on purpose: `_DEGHOST_SIGMA_FRACTION` places the split as a fraction of canvas
    width, so a smaller canvas is a narrower split and a different experiment. At 1024 the
    same code removes 51% of the ghost; at the production 4096 it removes 76%.
    """
    from biosphere.pano import blend as blend_mod

    a, b, hard = _ghost_pair(w=4096, h=2048)
    w, h = a.canvas_w, a.canvas_h
    owner, covered = blend_mod.compose_owner([a, b], hard, w, h)
    assert covered.all(), "the two half-masks must tile the canvas"

    k = 2 * max(1, round(w / 400.0)) + 1
    feather = [cv2.GaussianBlur(m, (k, k), max(0.8, k / 6.0), borderType=cv2.BORDER_REPLICATE) for m in hard]
    cfg = PipelineConfig.for_mode(QualityMode.BALANCED, blend_bands=6)
    blended = blend_mod.multiband_blend([a, b], feather, w, h, cfg)
    fixed = blend_mod.deghost_structure(blended, owner, covered, w * blend_mod._DEGHOST_SIGMA_FRACTION)

    split = w * blend_mod._DEGHOST_SIGMA_FRACTION
    before = _high_frequency_ghost(blended, a.image, b.image, split)
    after = _high_frequency_ghost(fixed, a.image, b.image, split)
    assert after < before * 0.5, f"ghosting must drop sharply: {before:.3f} -> {after:.3f}"

    # And the cross-fade must still be doing its own job: the brightness step across the
    # seam has to stay far below the 6-plus grey levels a hard cut leaves.
    def step(img):
        c = w // 2
        return abs(float(img[:, c - 40 : c - 8].mean()) - float(img[:, c + 8 : c + 40].mean()))

    # And the cross-fade must still be doing its own job. Measured here on high-contrast
    # noise, which is the harshest case for a step; on the real panorama the same code runs
    # 0.96 -> 1.31 grey levels against a hard cut's 6.27.
    assert step(fixed) < step(owner) * 0.6, f"seam step regressed: {step(fixed):.2f} vs {step(owner):.2f}"


def test_deghosting_returns_the_sharpness_the_cross_fade_was_costing():
    """Averaging two misaligned views blurs them; taking one of them does not."""
    from biosphere.pano import blend as blend_mod

    a, b, hard = _ghost_pair()
    w, h = a.canvas_w, a.canvas_h
    owner, covered = blend_mod.compose_owner([a, b], hard, w, h)
    k = 2 * max(1, round(w / 400.0)) + 1
    feather = [cv2.GaussianBlur(m, (k, k), max(0.8, k / 6.0), borderType=cv2.BORDER_REPLICATE) for m in hard]
    cfg = PipelineConfig.for_mode(QualityMode.BALANCED, blend_bands=6)
    blended = blend_mod.multiband_blend([a, b], feather, w, h, cfg)
    fixed = blend_mod.deghost_structure(blended, owner, covered, w * blend_mod._DEGHOST_SIGMA_FRACTION)

    def sharpness(img):
        return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())

    assert sharpness(fixed) > sharpness(blended), (
        f"{sharpness(fixed):.1f} must beat {sharpness(blended):.1f}"
    )


def test_deghosting_leaves_uncovered_pixels_to_the_hole_filler():
    """Where the canvas simply stops, the high-pass must see no edge to sharpen."""
    from biosphere.pano import blend as blend_mod

    a, b, hard = _ghost_pair()
    w, h = a.canvas_w, a.canvas_h
    # Punch a hole out of both owners: nobody covers the middle rows.
    for m in hard:
        m[h // 3 : 2 * h // 3, :] = 0
    owner, covered = blend_mod.compose_owner([a, b], hard, w, h)
    assert not covered.all()

    blended = np.full((h, w, 3), 120, np.uint8)
    fixed = blend_mod.deghost_structure(blended, owner, covered, w * blend_mod._DEGHOST_SIGMA_FRACTION)
    hole = ~covered
    assert np.abs(fixed[hole].astype(np.int16) - blended[hole].astype(np.int16)).max() <= 1, (
        "uncovered pixels must be handed on untouched"
    )


# ======================================================================================
# Compositing from fewer views than the solver kept
# ======================================================================================


def _row_of_poses(n: int, yaw_span_deg: float = 360.0, pitch_deg: float = 0.0):
    """`n` cameras spread evenly round a row, with sharpness increasing along the sweep."""
    frames, poses = [], []
    intr = Intrinsics.from_hfov(1080, 1440, 47.0)
    for i in range(n):
        yaw = math.radians(yaw_span_deg * i / max(1, n - 1))
        R = g.from_yaw_pitch_roll(yaw, math.radians(pitch_deg), 0.0)
        poses.append(CameraPose(frame_index=i, R=R, intrinsics=intr))
        frames.append(Frame(index=i, path=Path(f"f{i}.jpg"), timestamp=float(i),
                            width=1080, height=1440, sharpness=float(i % 7)))
    return frames, poses


def test_render_thins_the_oversampled_views_the_solver_kept():
    """The solve wants redundancy; the composite wants as few seams as will cover."""
    from biosphere.pano.render import select_render_views

    frames, poses = _row_of_poses(88)
    cfg = PipelineConfig.for_mode(QualityMode.BALANCED)
    kept_f, kept_p, n_solved = select_render_views(frames, poses, cfg)

    assert n_solved == 88
    assert len(kept_p) < 40, f"88 views at 4 deg apart should thin hard, got {len(kept_p)}"
    assert len(kept_f) == len(kept_p)
    # Still dense enough to overlap: neighbours must be closer than a frame is wide.
    yaws = sorted(p.yaw for p in kept_p)
    gaps = [math.degrees(b - a) for a, b in zip(yaws, yaws[1:])]
    assert max(gaps) < 47.0, f"a gap wider than the frame would be a hole: {max(gaps):.1f} deg"


def test_render_thinning_never_narrows_what_the_capture_reached():
    """Coverage describes the CAMERA, not the composite, and must survive thinning.

    Spreading alone clips the ends of a partial sweep: a view at the edge can sit inside the
    separation limit of its neighbour and still be the only one reaching past it. Dropping it
    would quietly shrink the reported yaw span and gap list, which is the product lying about
    the capture. Measured as it matters — the union of what the kept cameras see must contain
    everything the full set saw, direction by direction.
    """
    from biosphere.pano.render import select_render_views

    frames, poses = _row_of_poses(60, yaw_span_deg=240.0)
    cfg = PipelineConfig.for_mode(QualityMode.BALANCED)
    kept_f, kept_p, _ = select_render_views(frames, poses, cfg)
    assert len(kept_p) < len(poses), "the fixture must actually be thinned to prove anything"

    # Deliberately a different, coarser grid than `select_render_views` uses internally:
    # a guarantee that only holds on its own sampling is not a guarantee.
    lat = np.deg2rad(np.arange(-80.0, 81.0, 2.0))
    lon = np.deg2rad(np.arange(-180.0, 180.0, 2.0))
    LON, LAT = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(LAT) * np.sin(LON), -np.sin(LAT), np.cos(LAT) * np.cos(LON)],
                    axis=-1).reshape(-1, 3)
    intr = poses[0].intrinsics
    tan_h = math.tan(math.radians(intr.hfov_deg) / 2.0)
    tan_v = math.tan(math.radians(intr.vfov_deg) / 2.0)

    def union(ps):
        seen = np.zeros(dirs.shape[0], dtype=bool)
        for pose in ps:
            cam = dirs @ np.asarray(pose.R, dtype=np.float64).T
            z = cam[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                seen |= (z > 1e-9) & (np.abs(cam[:, 0] / z) <= tan_h) & (np.abs(cam[:, 1] / z) <= tan_v)
        return seen

    full, thin = union(poses), union(kept_p)
    lost = full & ~thin

    # Nothing may go missing in yaw: a lost sector is a hole in the room, and the reported
    # span and gap list would both be wrong about it.
    lost_lon = np.unique(np.degrees(LON.ravel()[lost]))
    kept_lon = np.unique(np.degrees(LON.ravel()[thin]))
    assert set(lost_lon) - set(kept_lon) == set(), "thinning opened a yaw gap"

    # Vertical reach may give up the last fraction of a degree and no more. A frame is a
    # pincushion — deepest at its centre column, ~8 deg shallower at the corners — so the
    # extreme elevation is only reached directly above a frame centre and thinning leaves
    # fewer centres. This is the one thing thinning genuinely costs, so it is bounded here
    # rather than wished away.
    reach = lambda seen: float(np.abs(np.degrees(LAT.ravel()[seen])).max())
    assert reach(full) - reach(thin) <= 2.0, (
        f"vertical reach fell {reach(full) - reach(thin):.1f} deg, more than the pincushion"
    )
    # 1.5% measured on this fixture, every cell of it in the outermost ring checked below.
    assert int(lost.sum()) / int(full.sum()) < 0.025, (
        f"lost {100.0 * lost.sum() / full.sum():.2f}% of the swept band, which is not a sliver"
    )
    # And every loss must BE that sliver, not a hole somewhere convenient to ignore.
    assert np.abs(np.degrees(LAT.ravel()[lost])).min() >= reach(thin) - 2.0, (
        "coverage was lost away from the vertical extreme"
    )


def test_render_thinning_can_be_switched_off():
    from biosphere.pano.render import select_render_views

    frames, poses = _row_of_poses(40)
    cfg = PipelineConfig.for_mode(QualityMode.BALANCED, thin_render_views=False)
    kept_f, kept_p, n = select_render_views(frames, poses, cfg)
    assert len(kept_p) == n == 40


# ======================================================================================
# Finding the doubled patches and committing them to one camera
# ======================================================================================


def _overlapping_row(n=6, patch_w=450, step=200, w=2048, h=1024, shift=1.5):
    """Views that cover a slice each and overlap their neighbours, as a real sweep does.

    The earlier fixture gave every patch the WHOLE canvas, so they overlapped everywhere and
    the blender mixed everywhere — which is why a policy measured on it looked wonderful and
    then moved `seam_gradient_energy` from 0.634 to 1.518 on a real render. Partial overlap
    is not a detail of the fixture; it is the thing being modelled.

    HALF THE PRODUCTION WIDTH, ON PURPOSE. Everything here scales with canvas width — the
    low/high split, the feather, the repair cap — so the relative claims these tests make hold
    at any size, and at 4096 the seam search plus a multi-band blend per test ran the group
    past ten minutes. A test nobody can afford to run protects nothing.
    """
    from biosphere.pano.warp import WarpedPatch

    rng = np.random.default_rng(5)
    base = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 1.4)
    intr = Intrinsics.from_hfov(patch_w, h, 46.5)
    pose = CameraPose(frame_index=0, R=np.eye(3), intrinsics=intr)
    out = []
    for i, x0 in enumerate(range(0, w - patch_w, step)):
        M = np.float32([[1, 0, float(rng.normal(0, shift))], [0, 1, float(rng.normal(0, shift / 2))]])
        img = cv2.warpAffine(base[:, x0:x0 + patch_w], M, (patch_w, h), borderMode=cv2.BORDER_REFLECT_101)
        img = np.clip(img.astype(np.float32) * (1.0 + float(rng.normal(0, 0.04))), 0, 255).astype(np.uint8)
        out.append(WarpedPatch(i, i, img, np.full((h, patch_w), 255, np.uint8), (x0, 0), w, h, pose))
    return out


@pytest.fixture(scope="module")
def ghost_scene():
    """Seams and a blend, built once. Each of these tests otherwise pays for both again."""
    from biosphere.pano import blend as blend_mod, seams as seams_mod

    patches = _overlapping_row()
    w, h = patches[0].canvas_w, patches[0].canvas_h
    # Voronoi, and a small seam scale. These tests are about the repair, not the seam finder,
    # and graph-cut at the fixture's own width is the 16-Mpx graph the module warns about --
    # it ran the group past nine minutes and never reached the fourth test.
    cfg = PipelineConfig.for_mode(
        QualityMode.BALANCED, seam_finder="voronoi", seam_scale_width=256
    )
    hard, feather, _ = seams_mod.find_seams_full(patches, cfg)
    blended = blend_mod.multiband_blend(patches, feather, w, h, cfg)
    return {
        "patches": patches, "hard": hard, "blended": blended, "cfg": cfg,
        "w": w, "h": h, "sigma": w * blend_mod._DEGHOST_SIGMA_FRACTION,
    }


def test_ghost_map_finds_the_mixture_and_not_the_content(ghost_scene):
    """A pixel from one camera matches that camera; an averaged pixel matches neither."""
    from biosphere.pano import blend as blend_mod

    sc = ghost_scene
    ghost, covered = blend_mod.ghost_map(sc["blended"], sc["patches"], sc["sigma"])
    assert covered.any()
    # Taking a patch verbatim must read as no ghost at all: the metric measures mixing, not
    # texture, or it would flag every busy wall in the room.
    clean, _ = blend_mod.ghost_map(sc["patches"][0].image, [sc["patches"][0]], sc["sigma"])
    assert float(clean.mean()) < 0.05, f"un-mixed content must not read as ghosted: {clean.mean():.3f}"
    assert float(ghost[covered].mean()) > float(clean.mean())


def test_ghost_repair_stays_inside_its_cap(ghost_scene):
    """The cap is the safety argument, not the threshold.

    A grey-level threshold alone does not bound how much canvas it selects: on the reference
    capture 1.5 grey levels took 25% of the panorama and `seam_gradient_energy` went
    0.538 -> 1.238, out of the band its own docstring calls invisible. Two synthetic fixtures
    had both predicted far less. So the area is capped outright, and that is what this pins.
    """
    from biosphere.pano import blend as blend_mod

    sc = ghost_scene
    fixed, fraction = blend_mod.repair_ghosted_regions(
        sc["blended"], sc["patches"], sc["hard"], sc["cfg"]
    )
    cap = sc["cfg"].max_ghost_repair_fraction
    assert 0.0 < fraction <= cap * 1.35, f"repair took {fraction:.2%} against a {cap:.0%} cap"

    # Everything outside the repaired regions must be left alone — a local repair that
    # quietly rewrote the flat walls would be the global version wearing a disguise.
    changed = float(np.any(fixed != sc["blended"], axis=2).mean())
    assert changed <= fraction * 1.6, f"{changed:.1%} of pixels moved for a {fraction:.1%} repair"


def test_ghost_repair_removes_the_doubling_where_it_acts(ghost_scene):
    from biosphere.pano import blend as blend_mod

    sc = ghost_scene
    before, covered = blend_mod.ghost_map(sc["blended"], sc["patches"], sc["sigma"])
    fixed, fraction = blend_mod.repair_ghosted_regions(
        sc["blended"], sc["patches"], sc["hard"], sc["cfg"]
    )
    region = np.any(fixed != sc["blended"], axis=2) & covered
    assert region.any(), "nothing was repaired, so this proves nothing"

    after, _ = blend_mod.ghost_map(fixed, sc["patches"], sc["sigma"])
    assert after[region].mean() < before[region].mean() * 0.85, (
        f"ghost where the repair acted: {before[region].mean():.2f} -> {after[region].mean():.2f}"
    )


def test_ghost_repair_never_touches_a_pixel_with_no_owner(ghost_scene):
    """No owner means no commitment.

    Zeroing the high band where the partition covers nothing would leave the low band alone
    there — a smooth blob in the middle of busy content, a worse artefact than the ghost it
    replaced. Measured on the shipped seam finder this never happens (`_repair_orphans` sees
    to it), which is exactly why it needs a test: it is a guard against a change elsewhere.
    """
    from biosphere.pano import blend as blend_mod

    sc = ghost_scene
    holed = [m.copy() for m in sc["hard"]]
    for m in holed:
        m[: m.shape[0] // 4, :] = 0
    fixed, _ = blend_mod.repair_ghosted_regions(sc["blended"], sc["patches"], holed, sc["cfg"])

    _, owner_covered = blend_mod.compose_owner(sc["patches"], holed, sc["w"], sc["h"])
    orphan = ~owner_covered
    assert orphan.any(), "the fixture must actually create orphans"
    assert np.array_equal(fixed[orphan], sc["blended"][orphan]), (
        "pixels with no owner must be handed on untouched"
    )


def test_ghost_repair_can_be_switched_off(ghost_scene):
    from biosphere.pano import blend as blend_mod
    from dataclasses import replace

    sc = ghost_scene
    off = replace(sc["cfg"], repair_ghosts=False)
    fixed, fraction = blend_mod.repair_ghosted_regions(sc["blended"], sc["patches"], sc["hard"], off)
    assert fraction == 0.0 and fixed is sc["blended"]


# ======================================================================================
# Pole branding
# ======================================================================================


def test_pole_branding_lands_on_both_caps_and_nowhere_else():
    """The mark belongs at the two poles: the directions a three-row sweep never reached.

    It used to be nadir-only, and wired into an `elif` that only ran when pole filling was
    switched off — which no shipped mode does, so it could not appear at all.
    """
    from biosphere.pano import poles as poles_mod

    h, w = 512, 1024
    pano = np.full((h, w, 3), 90, np.uint8)
    logo = np.zeros((256, 256, 4), np.uint8)
    logo[..., :3] = (0, 0, 255)
    cv2.circle(logo, (128, 128), 110, (0, 0, 255, 255), -1)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mark.png"
        cv2.imwrite(str(path), logo)
        before = pano.copy()
        poles_mod.composite_pole_logo(pano, path, which="zenith", span_deg=30.0)
        poles_mod.composite_pole_logo(pano, path, which="nadir", span_deg=40.0)

    changed = np.any(pano != before, axis=2)
    assert changed[: h // 8].any(), "nothing was painted at one pole"
    assert changed[-h // 8 :].any(), "nothing was painted at the other pole"
    # The horizon is what the viewer actually looks at and must be untouched.
    band = changed[int(h * 0.30) : int(h * 0.70)]
    assert not band.any(), f"{int(band.sum())} pixels changed around the horizon"


def test_pole_branding_covers_more_of_the_nadir_than_the_zenith():
    """The nadir carries the operator's feet and the smear below a turning phone."""
    from biosphere.pano import poles as poles_mod

    h, w = 512, 1024
    logo = np.zeros((256, 256, 4), np.uint8)
    cv2.circle(logo, (128, 128), 120, (255, 255, 255, 255), -1)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mark.png"
        cv2.imwrite(str(path), logo)
        a = np.zeros((h, w, 3), np.uint8)
        poles_mod.composite_pole_logo(a, path, which="zenith", span_deg=30.0)
        b = np.zeros((h, w, 3), np.uint8)
        poles_mod.composite_pole_logo(b, path, which="nadir", span_deg=42.0)
    assert int(np.any(b != 0, axis=2).sum()) > int(np.any(a != 0, axis=2).sum())


def test_pole_branding_refuses_a_pole_that_is_not_a_pole():
    from biosphere.pano import poles as poles_mod

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "m.png"
        cv2.imwrite(str(path), np.zeros((8, 8, 4), np.uint8))
        with pytest.raises(ValueError, match="nadir"):
            poles_mod.composite_pole_logo(np.zeros((64, 128, 3), np.uint8), path, which="middle")


def test_the_shipped_mark_is_opaque_and_fits_inside_the_protected_caps():
    """Two properties the mark has to have, and neither is cosmetic.

    OPAQUE, because its job is to *replace* the pixels `fill_uncovered` invented. The first
    placeholder sat at 88 % alpha and the synthesised ceiling showed straight through it;
    on a bright ceiling the mark all but disappeared.

    INSIDE THE GUARDS, because `empty_room` refuses to inpaint the caps on the grounds that
    they may hold a logo. A mark that reached past `_PROTECT_ABOVE_DEG` / `_PROTECT_BELOW_DEG`
    would have the furniture remover synthesising over our own branding.
    """
    from biosphere.ai.empty_room import _PROTECT_ABOVE_DEG, _PROTECT_BELOW_DEG
    from biosphere.pano.types import _BRAND_MARK

    assert _BRAND_MARK is not None, "the brand mark did not travel with the package"
    mark = cv2.imread(str(_BRAND_MARK), cv2.IMREAD_UNCHANGED)
    assert mark is not None and mark.shape[2] == 4, "the mark needs an alpha channel"

    alpha = mark[..., 3]
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot(yy + 0.5 - h / 2.0, xx + 0.5 - w / 2.0) / (w / 2.0)
    assert int(alpha[radius < 0.8].min()) == 255, "the mark is translucent; the cap shows through"

    cfg = PipelineConfig()
    reach = float(radius[alpha > 0].max())
    for which, span, guard in (
        ("nadir", cfg.nadir_logo_span_deg, 90.0 + _PROTECT_BELOW_DEG),
        ("zenith", cfg.zenith_logo_span_deg, 90.0 - _PROTECT_ABOVE_DEG),
    ):
        # Stereographic: a plane radius r sits at colatitude 2 atan(r / 2).
        plane_r = reach * poles_mod.plane_extent(span / 2.0)
        colat = math.degrees(2.0 * math.atan(plane_r / 2.0))
        assert colat < guard, (
            f"the {which} mark reaches {colat:.1f} deg from the pole, past the {guard:.0f} deg "
            "empty_room protects -- shrink DISC_RADIUS_FRAC in scripts/make_brand_mark.py"
        )
