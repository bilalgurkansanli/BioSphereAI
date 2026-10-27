"""Tests for the geometry solver.

Everything here is synthetic on purpose. Real captures are not reproducible, and the
properties that matter — does the focal come back, does the loop close, does bundle
adjustment actually converge on the truth — are only checkable against a known answer.
Ground-truth rotations are generated, correspondences are projected through the *same*
geometry primitives the solver uses, and then noise and outliers are added on top.
"""

from __future__ import annotations

import copy
import math
import shutil
import subprocess
import time

import cv2
import numpy as np
import pytest

from biosphere.pano import bundle as ba
from biosphere.pano import extract as ex
from biosphere.pano import features as feat
from biosphere.pano import geometry as g
from biosphere.pano import rotation as rot
from biosphere.pano import select as sel
from biosphere.pano import solve as solve_mod
from biosphere.pano.types import (
    Frame,
    InsufficientCoverageError,
    InsufficientOverlapError,
    Intrinsics,
    MatchPair,
    PipelineConfig,
    QualityMode,
    StitchMetrics,
    SweepPattern,
)

IMG_W, IMG_H = 1280, 720
HFOV = 67.0

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# ======================================================================================
# Synthetic capture helpers
# ======================================================================================


def _intrinsics(hfov: float = HFOV) -> Intrinsics:
    return Intrinsics.from_hfov(IMG_W, IMG_H, hfov)


def _ground_truth_rotations(n: int, rng: np.random.Generator, jitter: bool = True) -> list[np.ndarray]:
    """``n`` cameras spanning a full turn, optionally with a little pitch/roll wobble."""
    out = []
    for k in range(n):
        yaw = 2.0 * math.pi * k / n
        if jitter:
            pitch = math.radians(float(rng.uniform(-4.0, 4.0)))
            roll = math.radians(float(rng.uniform(-3.0, 3.0)))
        else:
            pitch = roll = 0.0
        out.append(g.from_yaw_pitch_roll(yaw, pitch, roll))
    return out


def _correspondences(
    R_i: np.ndarray,
    R_j: np.ndarray,
    K: np.ndarray,
    rng: np.random.Generator,
    *,
    n_pts: int = 200,
    noise_px: float = 0.5,
    outlier_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project random world rays into both cameras and keep those visible in both."""
    m = 8 * n_pts
    xs = rng.uniform(15.0, IMG_W - 15.0, m)
    ys = rng.uniform(15.0, IMG_H - 15.0, m)
    rays = np.asarray(g.unproject_pixels(xs, ys, K), dtype=np.float64)
    cam_j = (rays @ R_i) @ R_j.T
    px, py, valid = g.project_rays(cam_j, K)
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    ok = valid & (px > 10.0) & (px < IMG_W - 10.0) & (py > 10.0) & (py < IMG_H - 10.0)
    idx = np.flatnonzero(ok)[:n_pts]

    pts_i = np.stack([xs[idx], ys[idx]], axis=1)
    pts_j = np.stack([px[idx], py[idx]], axis=1)
    if noise_px > 0.0:
        pts_i = pts_i + rng.normal(0.0, noise_px, pts_i.shape)
        pts_j = pts_j + rng.normal(0.0, noise_px, pts_j.shape)
    if outlier_frac > 0.0 and idx.size:
        n_bad = max(1, int(round(outlier_frac * idx.size)))
        bad = rng.choice(idx.size, size=n_bad, replace=False)
        pts_j[bad, 0] = rng.uniform(0.0, IMG_W, n_bad)
        pts_j[bad, 1] = rng.uniform(0.0, IMG_H, n_bad)
    return pts_i.astype(np.float32), pts_j.astype(np.float32)


def _make_pair(
    i: int,
    j: int,
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    *,
    confidence: float | None = None,
    loop: bool = False,
    fit_homography: bool = True,
) -> MatchPair:
    H_fit = None
    if fit_homography:
        H_fit, _ = cv2.findHomography(
            pts_i.reshape(-1, 1, 2),
            pts_j.reshape(-1, 1, 2),
            cv2.USAC_MAGSAC,
            3.0,
            maxIters=10000,
            confidence=0.9995,
        )
    n = int(pts_i.shape[0])
    return MatchPair(
        i=i,
        j=j,
        pts_i=np.ascontiguousarray(pts_i, dtype=np.float32),
        pts_j=np.ascontiguousarray(pts_j, dtype=np.float32),
        H=None if H_fit is None else np.asarray(H_fit, dtype=np.float64),
        confidence=float(confidence if confidence is not None else n / (8.0 + 0.3 * n)),
        is_loop_closure=loop,
    )


def _gauge_aligned_errors(est: list[np.ndarray], truth: list[np.ndarray]) -> list[float]:
    """Per-camera angular error in degrees after factoring out the global gauge.

    The objective is invariant to a rotation of the world, so the estimate is only ever
    correct up to one. Camera 0 is what pins it down (bundle adjustment freezes it), so
    the gauge is read off camera 0 and then applied to everyone.
    """
    G = est[0].T @ truth[0]
    return [math.degrees(g.angular_distance(E @ G, T)) for E, T in zip(est, truth, strict=True)]


def _fake_frames(n: int, quality: np.ndarray, sharpness: np.ndarray) -> list[Frame]:
    from pathlib import Path

    return [
        Frame(
            index=k,
            path=Path(f"synthetic_{k:05d}.jpg"),
            timestamp=k / 8.0,
            width=IMG_W,
            height=IMG_H,
            sharpness=float(sharpness[k]),
            quality=float(quality[k]),
        )
        for k in range(n)
    ]


# ======================================================================================
# 1. Focal estimation
# ======================================================================================


def test_focal_from_homography_exact_on_pure_rotation():
    intr = _intrinsics()
    K = intr.K
    R = g.from_yaw_pitch_roll(math.radians(18.0), math.radians(5.0), math.radians(2.0))
    Hm = g.homography_from_rotation(R, K, K)
    fi, fj = rot.focal_from_homography(Hm, (IMG_W, IMG_H), (IMG_W, IMG_H))
    assert fi is not None and fj is not None
    assert fi == pytest.approx(intr.focal, rel=1e-6)
    assert fj == pytest.approx(intr.focal, rel=1e-6)


def test_focal_from_homography_handles_principal_point_offset():
    """The estimator assumes a centred principal point; the conjugation must undo ours."""
    intr = _intrinsics()
    K = intr.K
    assert K[0, 2] == IMG_W / 2.0 and K[1, 2] == IMG_H / 2.0
    R = g.from_yaw_pitch_roll(math.radians(25.0), 0.0, 0.0)
    Hm = g.homography_from_rotation(R, K, K)
    fi, fj = rot.focal_from_homography(Hm, (IMG_W, IMG_H), (IMG_W, IMG_H))
    # Without the T-conjugation these come out wrong by the cx/cy offset.
    assert fi == pytest.approx(intr.focal, rel=1e-6)
    assert fj == pytest.approx(intr.focal, rel=1e-6)


@pytest.mark.parametrize("hfov", [55.0, 67.0, 82.0])
def test_estimate_focal_recovers_known_focal(hfov):
    rng = np.random.default_rng(7)
    intr = _intrinsics(hfov)
    K = intr.K
    pairs: list[MatchPair] = []
    for k in range(24):
        R = g.from_yaw_pitch_roll(
            math.radians(float(rng.uniform(8.0, 35.0))),
            math.radians(float(rng.uniform(-8.0, 8.0))),
            math.radians(float(rng.uniform(-4.0, 4.0))),
        )
        Hm = g.homography_from_rotation(R, K, K)
        pairs.append(
            MatchPair(
                i=k,
                j=k + 1,
                pts_i=np.zeros((0, 2), np.float32),
                pts_j=np.zeros((0, 2), np.float32),
                H=Hm,
                confidence=1.0,
            )
        )
    focal = rot.estimate_focal(pairs, IMG_W, IMG_H)
    assert focal == pytest.approx(intr.focal, rel=0.03), (
        f"focal off by {abs(focal - intr.focal) / intr.focal:.2%} (hfov {hfov})"
    )


def test_estimate_focal_falls_back_to_prior_then_default():
    empty: list[MatchPair] = []
    assert rot.estimate_focal(empty, IMG_W, IMG_H, prior_px=1100.0) == pytest.approx(1100.0)
    default = rot.estimate_focal(empty, IMG_W, IMG_H)
    assert default == pytest.approx(Intrinsics.from_hfov(IMG_W, IMG_H, 67.0).focal)


def test_estimate_focal_clamps_absurd_estimates():
    """A homography from a near-degenerate rotation must not produce a 200-degree lens."""
    tiny = Intrinsics.from_hfov(IMG_W, IMG_H, 170.0)  # implies a focal far too short
    R = g.from_yaw_pitch_roll(math.radians(20.0), math.radians(3.0), 0.0)
    Hm = g.homography_from_rotation(R, tiny.K, tiny.K)
    pair = MatchPair(0, 1, np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), H=Hm)
    focal = rot.estimate_focal([pair], IMG_W, IMG_H)
    hfov = math.degrees(2.0 * math.atan((IMG_W / 2.0) / focal))
    assert rot.MIN_HFOV_DEG - 1e-6 <= hfov <= rot.MAX_HFOV_DEG + 1e-6


# ======================================================================================
# 2. Bundle adjustment recovers the rotations
# ======================================================================================


@pytest.fixture(scope="module")
def ring_solve():
    """16 cameras round a full turn, noisy + 10 % gross outliers, solved end to end."""
    rng = np.random.default_rng(1729)
    n = 16
    intr = _intrinsics()
    K = intr.K
    truth = _ground_truth_rotations(n, rng, jitter=True)

    pairs: list[MatchPair] = []
    for i in range(n):
        for d in (1, 2):
            j = (i + d) % n
            a, b = (i, j) if i < j else (j, i)
            if any(p.i == a and p.j == b for p in pairs):
                continue
            pts_a, pts_b = _correspondences(
                truth[a], truth[b], K, rng, n_pts=200, noise_px=0.5, outlier_frac=0.10
            )
            assert pts_a.shape[0] >= 80, "not enough overlap in the synthetic pair"
            pairs.append(_make_pair(a, b, pts_a, pts_b, loop=(b - a) > 3))

    rot.relative_rotations(pairs, intr)
    poses0 = rot.initial_poses(n, pairs, intr)
    config = PipelineConfig.for_mode(QualityMode.BALANCED, refine_focal=True)
    poses, out_intr, metrics = ba.bundle_adjust(poses0, pairs, intr, config)
    return {
        "truth": truth,
        "poses0": poses0,
        "poses": poses,
        "intr": intr,
        "out_intr": out_intr,
        "metrics": metrics,
        "pairs": pairs,
        "config": config,
    }


def test_bundle_adjust_recovers_rotations_within_015_deg(ring_solve):
    errors = _gauge_aligned_errors(ring_solve["poses"], ring_solve["truth"])
    worst = max(errors)
    assert worst < 0.15, (
        f"worst camera off by {worst:.4f} deg; per-camera errors "
        + ", ".join(f"{e:.4f}" for e in errors)
    )


def test_bundle_adjust_improves_on_the_initialisation(ring_solve):
    before = max(_gauge_aligned_errors(ring_solve["poses0"], ring_solve["truth"]))
    after = max(_gauge_aligned_errors(ring_solve["poses"], ring_solve["truth"]))
    assert after <= before, f"BA made things worse: {before:.4f} -> {after:.4f} deg"


def test_bundle_adjust_rejects_the_gross_outliers(ring_solve):
    metrics = ring_solve["metrics"]
    assert metrics["n_outliers_dropped"] > 0, "the rejection round dropped nothing"
    assert metrics["mean_reprojection_error_px"] < 1.0
    assert metrics["max_reprojection_error_px"] <= 3.0 * ring_solve["config"].ba_f_scale


def test_bundle_adjust_refines_the_focal(ring_solve):
    assert ring_solve["out_intr"].focal == pytest.approx(ring_solve["intr"].focal, rel=0.02)


def test_bundle_adjust_honours_its_time_budget(ring_solve):
    """The promise is a clock, so the clock is what the test checks.

    An agent is told the properties will be ready by the time they reach the office, and the
    solve is 70% of the pipeline -- so a budget that is merely advisory is the same as no
    budget. `_MIN_ROUND_SECONDS` per round is the floor the implementation reserves, and
    the ceiling has to hold against it.
    """
    budget = 2.0
    config = PipelineConfig.for_mode(
        QualityMode.BALANCED, refine_focal=True, ba_max_seconds=budget
    )
    poses0 = [R.copy() for R in ring_solve["poses0"]]
    started = time.perf_counter()
    poses, _out_intr, metrics = ba.bundle_adjust(
        poses0, copy.deepcopy(ring_solve["pairs"]), ring_solve["intr"], config
    )
    elapsed = time.perf_counter() - started

    n_rounds = max(1, int(metrics["n_rounds"]))
    ceiling = max(budget, n_rounds * ba._MIN_ROUND_SECONDS) + 5.0
    assert elapsed <= ceiling, f"budget {budget}s overrun: took {elapsed:.1f}s over {n_rounds} rounds"
    assert len(poses) == len(poses0)
    # Truncated is not broken: whatever came back has to still be usable geometry.
    for R in poses:
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_bundle_adjust_time_budget_still_rejects_outliers(ring_solve):
    """A capped solve must not skip the step that does most of the work.

    The rejection round is worth more than any amount of extra annealing -- measured on a
    real capture, the annealing rounds converge towards 5.8 px and rejection finishes at
    2.6 -- so the budget is shared between rounds rather than spent by the first one.
    """
    config = PipelineConfig.for_mode(
        QualityMode.BALANCED, refine_focal=True, ba_max_seconds=2.0
    )
    _poses, _intr, metrics = ba.bundle_adjust(
        [R.copy() for R in ring_solve["poses0"]],
        copy.deepcopy(ring_solve["pairs"]),
        ring_solve["intr"],
        config,
    )
    assert metrics["n_rounds"] == len(ring_solve["metrics"]["round_seconds"]), (
        "a capped solve ran fewer rounds than an uncapped one; the plan was cut short"
    )
    assert metrics["n_outliers_dropped"] > 0, "the rejection round never ran"


def test_bundle_adjust_unbudgeted_by_default_reports_no_deadline(ring_solve):
    assert ring_solve["metrics"]["bundle_deadline_hit"] is False


def test_bundle_adjust_holds_camera_zero_fixed(ring_solve):
    """The gauge fix must be exact, otherwise the solution can drift as a rigid body."""
    assert math.degrees(g.angular_distance(ring_solve["poses"][0], ring_solve["poses0"][0])) < 1e-9


def test_bundle_adjust_output_are_proper_rotations(ring_solve):
    for R in ring_solve["poses"]:
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
        assert np.abs(R @ R.T - np.eye(3)).max() < 1e-9


def test_residual_path_stays_float64(ring_solve):
    """A stray float32 anywhere in the objective silently wrecks the Jacobian.

    ``geometry``'s projection helpers preserve dtype, so the bundle adjuster gets its
    float64 by feeding them float64 — the pixel arrays on ``_Problem`` and the rotations
    out of ``_unpack``. Nothing enforces that at runtime, and the failure is invisible in
    the reprojection metrics (they barely move); only the recovered rotations degrade, by
    up to 0.85 degrees on a level single-row sweep. So it is pinned here instead.
    """
    problem = ba._build_problem(ring_solve["pairs"], len(ring_solve["poses"]), 1729)
    assert problem.m > 0
    for name in ("xi", "yi", "xj", "yj"):
        assert getattr(problem, name).dtype == np.float64, f"_Problem.{name} lost float64"

    poses, K, _focal, k1, k2 = ba._unpack(
        ba._pack(ring_solve["poses"], ring_solve["intr"], ring_solve["config"]),
        ring_solve["poses"][0],
        len(ring_solve["poses"]),
        ring_solve["intr"],
        ring_solve["config"],
    )
    assert all(R.dtype == np.float64 for R in poses)
    assert K.dtype == np.float64

    # The two shared helpers, exercised exactly as the objective calls them.
    rays = g.unproject_pixels(problem.xi, problem.yi, K, k1, k2)
    assert rays.dtype == np.float64, "geometry.unproject_pixels downcast a float64 input"
    px, py, _valid = g.project_rays(rays, K, k1, k2)
    assert px.dtype == np.float64 and py.dtype == np.float64, "geometry.project_rays downcast"

    assert ba._residual_vector(problem, poses, K, k1, k2).dtype == np.float64


def test_float64_projection_resolves_a_jacobian_sized_step():
    """The finite-difference step must produce a change far above the numeric noise floor.

    ``_DIFF_STEP`` moves a rotation parameter enough to shift the reprojection by ~1e-2 px.
    In float32 that lands only two orders of magnitude above the 1e-4 px quantisation grid,
    which is what corrupted the Jacobian; in float64 the same probe is exact.
    """
    K = _intrinsics().K
    x = np.array([1234.0, 640.0, 100.0], dtype=np.float64)
    y = np.array([567.0, 360.0, 80.0], dtype=np.float64)
    for k1, k2 in ((0.0, 0.0), (-0.12, 0.03)):
        r0 = g.unproject_pixels(x, y, K, k1, k2)
        r1 = g.unproject_pixels(x + 1e-2, y, K, k1, k2)
        x0, _, _ = g.project_rays(r0, K, k1, k2)
        x1, _, _ = g.project_rays(r1, K, k1, k2)
        assert np.abs((x1 - x0) - 1e-2).max() < 1e-9

        # And the round trip itself must be exact well below a thousandth of a pixel.
        assert np.abs(x0 - x).max() < 1e-9



# ======================================================================================
# 3. Loop closure
# ======================================================================================


@pytest.fixture(scope="module")
def drifting_loop():
    """A chain whose measured steps each overshoot by 0.4 deg, so it misses by 6.

    The *correspondences* come from the true, closing poses; only the relative rotations
    handed to the spanning-tree initialiser carry the drift. That is exactly the real
    situation: pairwise homographies are individually fine and collectively inconsistent.
    """
    rng = np.random.default_rng(4242)
    n = 16
    intr = _intrinsics()
    K = intr.K
    # Pure yaw, so the per-step error accumulates exactly and the test is deterministic.
    truth = _ground_truth_rotations(n, rng, jitter=False)
    step_drift = g.from_yaw_pitch_roll(math.radians(6.0 / (n - 1)), 0.0, 0.0)

    pairs: list[MatchPair] = []
    for i in range(n - 1):  # consecutive: high confidence, drifted measurement
        pts_a, pts_b = _correspondences(truth[i], truth[i + 1], K, rng, n_pts=180, noise_px=0.4)
        pair = _make_pair(i, i + 1, pts_a, pts_b, confidence=10.0, fit_homography=False)
        pair.R = step_drift @ (truth[i + 1] @ truth[i].T)
        pairs.append(pair)
    for i in range(n - 2):  # skip-one: extra conditioning for BA, still drifted
        pts_a, pts_b = _correspondences(truth[i], truth[i + 2], K, rng, n_pts=150, noise_px=0.4)
        pair = _make_pair(i, i + 2, pts_a, pts_b, confidence=5.0, fit_homography=False)
        pair.R = step_drift @ step_drift @ (truth[i + 2] @ truth[i].T)
        pairs.append(pair)
    # The wrap-around closure: honest measurement, lowest confidence so the maximum
    # spanning tree leaves it out and the chain really does have to drift.
    pts_a, pts_b = _correspondences(truth[n - 1], truth[0], K, rng, n_pts=180, noise_px=0.4)
    wrap = _make_pair(0, n - 1, pts_b, pts_a, confidence=1.0, fit_homography=False)
    wrap.R = truth[n - 1] @ truth[0].T
    wrap.is_loop_closure = True
    pairs.append(wrap)

    poses0 = rot.initial_poses(n, pairs, intr)
    config = PipelineConfig.for_mode(QualityMode.BALANCED, refine_focal=True)
    poses, out_intr, metrics = ba.bundle_adjust(poses0, pairs, intr, config)
    return {
        "truth": truth,
        "poses0": poses0,
        "poses": poses,
        "pairs": pairs,
        "metrics": metrics,
        "out_intr": out_intr,
    }


def test_naive_chain_does_not_close(drifting_loop):
    before = ba.loop_closure_residual(drifting_loop["poses0"], drifting_loop["pairs"])
    assert before == pytest.approx(6.0, abs=0.2), f"expected ~6 deg of drift, got {before:.3f}"


def test_bundle_adjust_closes_the_loop(drifting_loop):
    after = ba.loop_closure_residual(drifting_loop["poses"], drifting_loop["pairs"])
    assert after < 0.5, f"loop still open by {after:.4f} deg"


def test_closed_loop_matches_ground_truth(drifting_loop):
    worst = max(_gauge_aligned_errors(drifting_loop["poses"], drifting_loop["truth"]))
    assert worst < 0.5, f"worst camera off by {worst:.4f} deg after closing the loop"


def test_loop_closure_residual_is_zero_without_loop_pairs():
    poses = [np.eye(3), g.from_yaw_pitch_roll(0.3, 0.0, 0.0)]
    pair = MatchPair(0, 1, np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))
    pair.R = np.eye(3)
    assert ba.loop_closure_residual(poses, [pair]) == 0.0


def test_loop_is_measured_at_the_solved_focal_not_the_starting_one():
    """A focal the bundle corrects must not come back as loop error.

    ``pair.R`` is a rotation read off pixels, so it depends on the focal it was read at. The
    statistic used to compare rotations fitted at the STARTING focal against poses solved at
    the FINAL one, and every degree of focal correction surfaced as "the loop did not close"
    -- measured on the reference capture as 3.19 deg reported against 0.83 deg actual, with
    a user-facing warning attached. Here the geometry is perfect and only the starting focal
    is wrong (12 % short), so the honest residual after the solve is essentially zero.
    """
    rng = np.random.default_rng(77)
    n = 16
    true_intr = _intrinsics()
    truth = _ground_truth_rotations(n, rng, jitter=True)
    pairs: list[MatchPair] = []
    for i in range(n):
        for d in (1, 2):
            j = (i + d) % n
            a, b = (i, j) if i < j else (j, i)
            if any(p.i == a and p.j == b for p in pairs):
                continue
            pts_a, pts_b = _correspondences(truth[a], truth[b], true_intr.K, rng, n_pts=160, noise_px=0.3)
            assert pts_a.shape[0] >= 60, "not enough overlap in the synthetic pair"
            # The wrap-around pairs (b - a > 3) are the loop closures.
            pairs.append(_make_pair(a, b, pts_a, pts_b, loop=(b - a) > 3, fit_homography=False))

    start = Intrinsics(
        focal=0.88 * true_intr.focal, cx=true_intr.cx, cy=true_intr.cy,
        width=true_intr.width, height=true_intr.height,
    )
    rot.relative_rotations(pairs, start)
    stale = [p.R.copy() for p in pairs]
    poses0 = rot.initial_poses(n, pairs, start)
    config = PipelineConfig.for_mode(QualityMode.BALANCED, refine_focal=True)
    poses, out_intr, metrics = ba.bundle_adjust(poses0, pairs, start, config)

    assert out_intr.focal == pytest.approx(true_intr.focal, rel=0.01), "the focal was not recovered"
    reported = metrics["loop_closure_residual_deg"]
    assert reported < 0.2, f"a perfect capture reported a {reported:.3f} deg loop residual"
    # And the bug this pins: the same poses against the stale, start-focal rotations.
    for p, R in zip(pairs, stale, strict=True):
        p.R = R
    assert ba.loop_closure_residual(poses, pairs) > 5.0 * max(reported, 0.05)


# ======================================================================================
# 4. Degenerate pair rejection
# ======================================================================================


def _collinear_featuresets(n_pts: int = 64) -> tuple[feat.FeatureSet, feat.FeatureSet]:
    """Two feature sets whose points all lie on one line, with matchable descriptors."""
    rng = np.random.default_rng(11)
    t = np.linspace(60.0, IMG_W - 60.0, n_pts)
    pts_i = np.stack([t, 0.4 * t + 120.0], axis=1).astype(np.float32)
    pts_j = np.stack([t + 9.0, 0.4 * t + 126.0], axis=1).astype(np.float32)
    # Well-separated descriptors: the ratio test and the mutual check both pass, so the
    # pair reaches the geometric verification stage with plenty of correspondences.
    desc = np.eye(n_pts, 128, dtype=np.float32) * 4.0 + rng.normal(0.0, 0.01, (n_pts, 128)).astype(np.float32)
    return (
        feat.FeatureSet(0, pts_i, desc, (IMG_W, IMG_H), 1.0),
        feat.FeatureSet(1, pts_j, desc.copy(), (IMG_W, IMG_H), 1.0),
    )


def test_collinear_pair_is_rejected():
    fs_i, fs_j = _collinear_featuresets()
    config = PipelineConfig.for_mode(QualityMode.BALANCED, min_pair_inliers=8)
    # Sanity: the matcher really does find the correspondences, so rejection can only
    # be coming from the geometric check.
    raw_i, raw_j = feat._raw_matches(fs_i, fs_j, config.match_ratio)
    assert raw_i.shape[0] >= 32, "the synthetic descriptors did not match at all"

    pairs = feat.match_pairs([fs_i, fs_j], [(0, 1)], _intrinsics().K, config)
    assert pairs == [], "a fully collinear point set must not produce a verified pair"


def test_is_degenerate_homography_flags_rank_deficient_and_sheared():
    assert feat.is_degenerate_homography(None)
    assert feat.is_degenerate_homography(np.zeros((3, 3)))
    assert feat.is_degenerate_homography(np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]]))
    sheared = np.array([[40.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
    assert feat.is_degenerate_homography(sheared), "80:1 anisotropy is not a rotation"
    assert not feat.is_degenerate_homography(
        g.homography_from_rotation(g.from_yaw_pitch_roll(0.3, 0.05, 0.0), _intrinsics().K, _intrinsics().K)
    )


def test_initial_poses_raises_on_a_disconnected_graph():
    intr = _intrinsics()
    pair = MatchPair(0, 1, np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), confidence=1.0)
    pair.R = np.eye(3)
    with pytest.raises(InsufficientOverlapError) as excinfo:
        rot.initial_poses(4, [pair], intr)
    message = str(excinfo.value)
    assert "2" in message and "3" in message, f"error must name the stranded frames: {message}"


# ======================================================================================
# 5. Keyframe selection
# ======================================================================================


def _linear_track(n: int, deg_per_frame: float, intr: Intrinsics) -> np.ndarray:
    """A motion track that produces exactly ``deg_per_frame`` of yaw per frame."""
    dx = -intr.focal * math.tan(math.radians(deg_per_frame))
    motion = np.zeros((n - 1, 2), dtype=np.float64)
    motion[:, 0] = dx
    return motion


def test_select_keyframes_is_evenly_spaced_and_prefers_sharp_frames():
    n = 200
    intr = _intrinsics()
    config = PipelineConfig.for_mode(QualityMode.BALANCED)
    motion = _linear_track(n, 2.0, intr)

    # Every fifth frame is the sharp one; the rest are motion-blurred.
    quality = np.full(n, 0.20)
    sharpness = np.full(n, 30.0)
    quality[::5] = 0.90
    sharpness[::5] = 400.0
    frames = _fake_frames(n, quality, sharpness)

    keyframes, pattern, yaw_span = sel.select_keyframes(frames, config, motion=motion, intr=intr)

    assert pattern is SweepPattern.SINGLE_ROW
    assert yaw_span == pytest.approx(2.0 * (n - 1), abs=1.0)
    assert config.min_keyframes <= len(keyframes) <= config.max_keyframes

    picked = [f.index for f in keyframes]
    assert picked == sorted(picked)
    assert all(i % 5 == 0 for i in picked), f"blurred frames were selected: {picked}"

    bin_deg = intr.hfov_deg * (1.0 - config.target_overlap)
    yaws = np.array(picked, dtype=np.float64) * 2.0
    gaps = np.diff(yaws)
    assert gaps.min() > 0.6 * bin_deg, f"keyframes bunched up: {gaps}"
    assert gaps.max() < 1.4 * bin_deg, f"keyframes left a hole: {gaps}"
    assert float(gaps.std()) < 0.15 * bin_deg, f"spacing is uneven: {gaps}"


def test_select_keyframes_respects_the_maximum_budget():
    n = 600
    intr = _intrinsics()
    config = PipelineConfig.for_mode(QualityMode.BALANCED, max_keyframes=20, target_overlap=0.9)
    motion = _linear_track(n, 2.0, intr)
    rng = np.random.default_rng(3)
    quality = rng.uniform(0.1, 0.9, n)
    frames = _fake_frames(n, quality, quality * 500.0)
    keyframes, _pattern, _span = sel.select_keyframes(frames, config, motion=motion, intr=intr)
    assert len(keyframes) <= 20


def test_select_keyframes_meets_the_minimum_on_a_short_sweep():
    n = 40
    intr = _intrinsics()
    config = PipelineConfig.for_mode(QualityMode.BALANCED, min_keyframes=12)
    motion = _linear_track(n, 2.0, intr)  # only ~78 deg: fewer natural bins than the floor
    quality = np.linspace(0.1, 0.9, n)
    frames = _fake_frames(n, quality, quality * 500.0)
    keyframes, _pattern, _span = sel.select_keyframes(frames, config, motion=motion, intr=intr)
    assert len(keyframes) >= 12
    assert len({f.index for f in keyframes}) == len(keyframes), "duplicate keyframes selected"


def test_select_keyframes_rejects_a_mismatched_motion_track():
    from biosphere.pano.types import StitchError

    frames = _fake_frames(10, np.ones(10), np.ones(10))
    with pytest.raises(StitchError, match="does not match"):
        sel.select_keyframes(frames, PipelineConfig(), motion=np.zeros((3, 2)))


# ======================================================================================
# 6. Sweep pattern detection
# ======================================================================================


def _row_track(pitches: list[float], yaw_per_row: float, samples: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate several constant-pitch yaw sweeps with short ramps between them."""
    yaw_parts: list[np.ndarray] = []
    pitch_parts: list[np.ndarray] = []
    yaw_cursor = 0.0
    for r, pitch in enumerate(pitches):
        yaw = yaw_cursor + np.linspace(0.0, yaw_per_row, samples)
        yaw_parts.append(yaw)
        pitch_parts.append(np.full(samples, pitch) + rng.normal(0.0, 0.4, samples))
        yaw_cursor = yaw[-1]
        if r + 1 < len(pitches):
            ramp = 6
            yaw_parts.append(yaw_cursor + np.linspace(0.0, 8.0, ramp))
            pitch_parts.append(np.linspace(pitch, pitches[r + 1], ramp))
            yaw_cursor += 8.0
    return np.concatenate(yaw_parts), np.concatenate(pitch_parts)


def test_detect_sweep_pattern_single_row():
    rng = np.random.default_rng(5)
    cum_yaw = np.linspace(0.0, 360.0, 180)
    cum_pitch = rng.normal(0.0, 1.0, 180)
    assert sel.detect_sweep_pattern(cum_yaw, cum_pitch) is SweepPattern.SINGLE_ROW


def test_detect_sweep_pattern_three_rows():
    rng = np.random.default_rng(6)
    cum_yaw, cum_pitch = _row_track([-25.0, 0.0, 25.0], 360.0, 120, rng)
    assert sel.detect_sweep_pattern(cum_yaw, cum_pitch) is SweepPattern.MULTI_ROW
    assert len(sel.pitch_clusters(cum_yaw, cum_pitch)) == 3


def test_detect_sweep_pattern_ignores_a_short_second_row():
    """Tipping up for a few seconds at the end is not a second row."""
    rng = np.random.default_rng(8)
    cum_yaw, cum_pitch = _row_track([0.0, 28.0], 360.0, 120, rng)
    cum_yaw = cum_yaw[:150]  # truncate so the second "row" only spans a fraction of a turn
    cum_pitch = cum_pitch[:150]
    assert sel.detect_sweep_pattern(cum_yaw, cum_pitch) is SweepPattern.SINGLE_ROW


def test_multi_row_selection_covers_every_row():
    rng = np.random.default_rng(9)
    cum_yaw, cum_pitch = _row_track([-24.0, 0.0, 24.0], 360.0, 90, rng)
    n = cum_yaw.size
    intr = _intrinsics()
    # Rebuild a motion track that reproduces this (yaw, pitch) walk exactly.
    motion = np.zeros((n - 1, 2), dtype=np.float64)
    motion[:, 0] = -intr.focal * np.tan(np.radians(np.diff(cum_yaw)))
    motion[:, 1] = intr.focal * np.tan(np.radians(np.diff(cum_pitch)))
    frames = _fake_frames(n, np.full(n, 0.5), np.full(n, 100.0))

    keyframes, pattern, _span = sel.select_keyframes(frames, PipelineConfig(), motion=motion, intr=intr)
    assert pattern is SweepPattern.MULTI_ROW
    picked_pitch = cum_pitch[[f.index for f in keyframes]]
    for centre in (-24.0, 0.0, 24.0):
        assert np.any(np.abs(picked_pitch - centre) < 6.0), f"row at pitch {centre} got no keyframes"


# ======================================================================================
# 7. ffprobe / ffmpeg
# ======================================================================================


def _make_test_video(path, *, size="320x240", rate=10, duration=2) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration={duration}",
        "-pix_fmt", "yuv420p", "-y", str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_probe_video_reads_stream_metadata(tmp_path):
    video = tmp_path / "testsrc.mp4"
    _make_test_video(video)
    info = ex.probe_video(video)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(10.0, abs=0.1)
    assert info.duration_s == pytest.approx(2.0, abs=0.25)
    assert info.n_frames == pytest.approx(20, abs=3)
    assert info.rotation_deg == 0
    assert info.codec
    assert (info.display_width, info.display_height) == (320, 240)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_probe_video_rejects_missing_and_non_video(tmp_path):
    from biosphere.pano.types import StitchError

    with pytest.raises(StitchError):
        ex.probe_video(tmp_path / "nope.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all")
    with pytest.raises(StitchError):
        ex.probe_video(junk)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_extract_frames_respects_the_budget_and_scores(tmp_path):
    video = tmp_path / "testsrc.mp4"
    _make_test_video(video, rate=20, duration=3)
    info = ex.probe_video(video)
    config = PipelineConfig.for_mode(
        QualityMode.BALANCED, extract_fps=10.0, max_extracted_frames=12, max_frame_width=160
    )
    frames = ex.extract_frames(info, tmp_path / "frames", config, None)

    assert 0 < len(frames) <= config.max_extracted_frames + 1
    assert all(f.path.exists() for f in frames)
    first = cv2.imread(str(frames[0].path))
    assert first is not None and first.shape[1] <= 160
    assert all(0.0 <= f.quality <= 1.0 for f in frames)
    assert max(f.sharpness for f in frames) > 0.0
    assert [f.index for f in frames] == list(range(len(frames)))
    assert frames[1].timestamp > frames[0].timestamp


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_probe_video_measures_a_browser_recording(tmp_path):
    """A live-muxed WebM declares no duration and no frame count. Both must be recovered.

    Everything the guided capture uploads is one of these. Left at zero, `_sampling_fps`
    loses the only term that clamps it to the frame budget, so a two-minute capture is
    decimated at the full `extract_fps` and the solve is handed several times the frames
    it planned for.
    """
    live = tmp_path / "live.webm"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=8",
            "-c:v", "libvpx-vp9", "-b:v", "300k", "-f", "webm", "-live", "1", str(live),
        ],
        check=True,
        capture_output=True,
    )

    info = ex.probe_video(live)
    assert info.duration_s == pytest.approx(8.0, abs=0.5)
    assert info.n_frames > 0

    config = PipelineConfig.for_mode(
        QualityMode.BALANCED, extract_fps=10.0, max_extracted_frames=12, max_frame_width=160
    )
    # The budget binds again once the length is known; at duration 0.0 it could not.
    assert ex._sampling_fps(config, info.duration_s) < config.extract_fps
    frames = ex.extract_frames(info, tmp_path / "frames", config, None)
    assert 0 < len(frames) <= config.max_extracted_frames + 1


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_extract_frames_treats_segments_as_one_capture(tmp_path):
    """A capture split by the recorder must decimate and time as if it never was.

    The phone seals and restarts its recorder whenever the encoder stalls or a call
    arrives, so the same room arrives as two files. Two things have to hold: the frame
    BUDGET is spent across the whole capture rather than per file, and the timeline stays
    monotone across the join — every downstream consumer of ``Frame.timestamp`` reads a
    backwards step as two frames being the same moment.
    """
    from biosphere.pano.solve import CaptureKind, _classify_source

    first = tmp_path / "seg0.mp4"
    second = tmp_path / "seg1.mp4"
    _make_test_video(first, rate=20, duration=3)
    _make_test_video(second, rate=20, duration=3)
    # Two videos are one capture, not a photo set — the classification is what used to send
    # segments down the stills path, where a .mp4 is simply not a readable image.
    kind, paths = _classify_source([first, second])
    assert kind is CaptureKind.VIDEO
    assert paths == [first, second]

    # A budget too large to bind, so the frame COUNT reflects the footage rather than the cap.
    roomy = PipelineConfig.for_mode(
        QualityMode.BALANCED, extract_fps=4.0, max_extracted_frames=500, max_frame_width=160
    )
    joined = ex.extract_frames([first, second], tmp_path / "joined", roomy, None)
    solo = ex.extract_frames(first, tmp_path / "solo", roomy, None)

    assert len(joined) >= 2 * len(solo) - 1, "both segments have to contribute frames"
    assert [f.index for f in joined] == list(range(len(joined)))
    assert all(f.path.exists() for f in joined)
    assert all(
        joined[i + 1].timestamp > joined[i].timestamp for i in range(len(joined) - 1)
    ), "the timeline must not step backwards at the segment join"
    # The second segment starts after the first has finished, never on top of it.
    assert joined[-1].timestamp >= ex.probe_video(first).duration_s

    # A budget tight enough to bind is spent across the CAPTURE. Sampling each segment
    # against its own duration would hand an interrupted capture twice the frames and
    # spend the whole matching budget on a room already covered.
    tight = PipelineConfig.for_mode(
        QualityMode.BALANCED, extract_fps=10.0, max_extracted_frames=20, max_frame_width=160
    )
    capped = ex.extract_frames([first, second], tmp_path / "capped", tight, None)
    assert len(capped) <= tight.max_extracted_frames + 1


def test_sampling_fps_never_exceeds_the_frame_budget():
    config = PipelineConfig.for_mode(QualityMode.BALANCED, extract_fps=8.0, max_extracted_frames=100)
    assert ex._sampling_fps(config, 10.0) == pytest.approx(8.0)
    assert ex._sampling_fps(config, 200.0) == pytest.approx(0.5)
    assert ex._sampling_fps(config, 0.0) == pytest.approx(8.0)


# ======================================================================================
# 8. Pose finalisation
# ======================================================================================


def test_finalise_poses_levels_and_zeroes_the_first_yaw():
    tilt = g.rodrigues(np.array([0.18, 0.0, 0.09]))
    poses = [
        g.from_yaw_pitch_roll(a, 0.0, 0.0) @ tilt
        for a in np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    ]
    config = PipelineConfig.for_mode(QualityMode.BALANCED, level_horizon=True)
    out = ba.finalise_poses(poses, config)
    assert math.degrees(g.yaw_pitch_roll(out[0])[0]) == pytest.approx(0.0, abs=1e-6)
    assert max(abs(math.degrees(g.yaw_pitch_roll(R)[1])) for R in out) < 1.0
    # Relative geometry must be untouched: finalisation is a change of world frame only.
    for a, b in zip(poses, out, strict=True):
        assert math.degrees(g.angular_distance(b @ out[0].T, a @ poses[0].T)) < 1e-4


def test_finalise_poses_without_levelling_still_zeroes_the_first_yaw():
    poses = [g.from_yaw_pitch_roll(a, 0.2, 0.0) for a in (0.7, 1.4, 2.1)]
    config = PipelineConfig.for_mode(QualityMode.BALANCED, level_horizon=False)
    out = ba.finalise_poses(poses, config)
    assert math.degrees(g.yaw_pitch_roll(out[0])[0]) == pytest.approx(0.0, abs=1e-6)


# ======================================================================================
# 9. End to end: solve_capture on a synthetic capture
# ======================================================================================


PANO_W, PANO_H = 4096, 2048
VIEW_W, VIEW_H = 640, 360


def _synthetic_panorama(rng: np.random.Generator) -> np.ndarray:
    """A textured equirect sphere.

    Upsampled noise rather than a photograph: it is scale-rich (so SIFT finds blobs at
    several octaves), it has no repeated structure (so there are no ambiguous matches to
    confound the test), and it is reproducible from a seed.
    """
    coarse = cv2.resize(
        rng.integers(0, 256, (256, 512, 3), dtype=np.uint8),
        (PANO_W, PANO_H),
        interpolation=cv2.INTER_LANCZOS4,
    )
    mid = cv2.resize(
        rng.integers(0, 256, (64, 128, 3), dtype=np.uint8),
        (PANO_W, PANO_H),
        interpolation=cv2.INTER_CUBIC,
    )
    pano = cv2.addWeighted(coarse, 0.6, mid, 0.4, 0.0)
    grain = rng.integers(-10, 11, pano.shape)
    return np.clip(pano.astype(np.int16) + grain, 0, 255).astype(np.uint8)


def _render_sweep(
    out_dir,
    n: int,
    step_deg: float,
    rng: np.random.Generator,
) -> tuple[list, list[np.ndarray]]:
    """Render ``n`` pinhole views of a synthetic sphere along a yaw sweep."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pano = _synthetic_panorama(rng)
    intr = Intrinsics.from_hfov(VIEW_W, VIEW_H, HFOV)
    rays = g.camera_rays(VIEW_W, VIEW_H, intr.K)
    paths, truth = [], []
    for k in range(n):
        R = g.from_yaw_pitch_roll(
            math.radians(step_deg * k),
            math.radians(float(rng.uniform(-3.0, 3.0))),
            math.radians(float(rng.uniform(-2.0, 2.0))),
        )
        truth.append(R)
        u, v = g.equirect_from_rays(rays @ R, PANO_W, PANO_H)
        view = g.sample_equirect(pano, u, v, interpolation=cv2.INTER_LANCZOS4)
        path = out_dir / f"shot_{k:03d}.jpg"
        cv2.imwrite(str(path), view, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        paths.append(path)
    return paths, truth


class _RecordingProgress:
    """Captures the stage names the solver reports, in order, without duplicates."""

    def __init__(self) -> None:
        self.stages: list[str] = []
        self.fractions: list[float] = []

    def stage(self, name: str, fraction: float, message: str = "") -> None:
        if not self.stages or self.stages[-1] != name:
            self.stages.append(name)
        self.fractions.append(fraction)

    def log(self, message: str, **fields) -> None:
        return None


@pytest.fixture(scope="module")
def solved_capture(tmp_path_factory):
    from biosphere.pano.solve import solve_capture

    root = tmp_path_factory.mktemp("e2e")
    rng = np.random.default_rng(20)
    paths, truth = _render_sweep(root / "shots", n=24, step_deg=15.0, rng=rng)
    config = PipelineConfig.for_mode(QualityMode.BALANCED, max_frame_width=VIEW_W)
    progress = _RecordingProgress()
    result = solve_capture(paths, config, work_dir=root / "work", progress=progress)
    return {"result": result, "truth": truth, "progress": progress, "root": root}


def test_solve_capture_emits_the_stages_in_order(solved_capture):
    assert solved_capture["progress"].stages == [
        "probe",
        "extract",
        "select",
        "features",
        "match",
        "bundle",
    ]
    assert all(0.0 <= f <= 1.0 for f in solved_capture["progress"].fractions)


def test_solve_capture_result_is_self_consistent(solved_capture):
    result = solved_capture["result"]
    assert len(result.frames) == len(result.poses) >= 8
    assert [f.index for f in result.frames] == list(range(len(result.frames)))
    assert all(f.path.exists() for f in result.frames)
    assert result.work_dir is not None and result.work_dir.exists()
    assert result.metrics.n_keyframes == len(result.frames)
    assert result.metrics.n_pairs == len(result.pairs) >= len(result.frames) - 1
    assert result.metrics.sweep_pattern is SweepPattern.SINGLE_ROW
    for pose in result.poses:
        assert np.linalg.det(pose.R) == pytest.approx(1.0, abs=1e-9)


def test_solve_capture_recovers_the_intrinsics(solved_capture):
    intr = Intrinsics.from_hfov(VIEW_W, VIEW_H, HFOV)
    metrics = solved_capture["result"].metrics
    assert metrics.focal_px == pytest.approx(intr.focal, rel=0.03)
    assert metrics.hfov_deg == pytest.approx(HFOV, abs=2.0)


def test_solve_capture_recovers_the_orientations(solved_capture):
    result = solved_capture["result"]
    truth = solved_capture["truth"]
    src = [f._cache["source_index"] for f in result.frames]
    R0_est, R0_true = result.poses[0].R, truth[src[0]]
    errors = [
        math.degrees(g.angular_distance(pose.R @ R0_est.T, truth[s] @ R0_true.T))
        for pose, s in zip(result.poses, src, strict=True)
    ]
    assert max(errors) < 1.0, f"worst pose off by {max(errors):.4f} deg: {errors}"


def test_solve_capture_quality_metrics_are_clean(solved_capture):
    metrics = solved_capture["result"].metrics
    assert metrics.mean_reprojection_error_px < 1.0
    assert metrics.loop_closure_residual_deg < 2.0
    assert metrics.n_loop_closures >= 1
    assert not [w for w in metrics.warnings if w.startswith(("high_reprojection", "loop_did_not"))]
    assert set(metrics.stage_seconds) == {"probe", "extract", "select", "features", "match", "bundle"}


def test_solve_capture_opens_facing_the_first_keyframe(solved_capture):
    poses = solved_capture["result"].poses
    assert math.degrees(poses[0].yaw) == pytest.approx(0.0, abs=1e-6)
    assert max(abs(math.degrees(p.pitch)) for p in poses) < 8.0


def test_solve_capture_rejects_a_partial_turn(tmp_path):
    from biosphere.pano.solve import solve_capture
    from biosphere.pano.types import InsufficientCoverageError

    rng = np.random.default_rng(31)
    paths, _truth = _render_sweep(tmp_path / "shots", n=7, step_deg=15.0, rng=rng)
    config = PipelineConfig.for_mode(QualityMode.BALANCED, max_frame_width=VIEW_W)
    with pytest.raises(InsufficientCoverageError) as excinfo:
        solve_capture(paths, config, work_dir=tmp_path / "work")
    assert "deg of yaw" in str(excinfo.value)


# ======================================================================================
# 10. Robustness: broken pose graphs and honest coverage
# ======================================================================================


def _fake_pair(i: int, j: int, n_pts: int = 40, confidence: float = 1.0) -> MatchPair:
    pts = np.random.default_rng(i * 100 + j).uniform(0, 500, (n_pts, 2)).astype(np.float32)
    return MatchPair(i=i, j=j, pts_i=pts, pts_j=pts + 3.0, confidence=confidence)


def test_connected_components_finds_the_split():
    pairs = [_fake_pair(0, 1), _fake_pair(1, 2), _fake_pair(4, 5)]
    comps = feat.connected_components(6, pairs)
    assert [len(c) for c in comps] == [3, 2, 1]
    assert comps[0] == [0, 1, 2]
    assert comps[1] == [4, 5]
    assert comps[2] == [3]
    assert feat.connected_components(3, [_fake_pair(0, 1), _fake_pair(1, 2)]) == [[0, 1, 2]]


def _frames_for(n: int) -> list[Frame]:
    from pathlib import Path

    frames = []
    for k in range(n):
        f = Frame(index=k, path=Path(f"kf_{k}.jpg"), timestamp=float(k), width=IMG_W, height=IMG_H)
        f._cache["source_index"] = k * 2
        frames.append(f)
    return frames


def test_keep_largest_component_drops_the_stragglers_and_keeps_indices_aligned():
    """One unlinkable keyframe must cost that keyframe, not the whole capture."""
    n = 8
    frames = _frames_for(n)
    # 0..5 form one chain; 6 and 7 are linked only to each other.
    pairs = [_fake_pair(k, k + 1) for k in range(5)] + [_fake_pair(6, 7)]
    prior_yaw = np.arange(n, dtype=np.float64) * 40.0
    metrics = StitchMetrics()

    kept, kept_pairs, kept_yaw, dropped = solve_mod._keep_largest_component(
        frames, pairs, prior_yaw, metrics
    )
    assert len(kept) == 6
    assert [d[0] for d in dropped] == [12, 14], "dropped entries name the source frame index"
    assert [f.index for f in kept] == list(range(6)), "Frame.index must equal list position"
    assert kept_yaw.tolist() == prior_yaw[:6].tolist()
    for p in kept_pairs:
        assert 0 <= p.i < 6 and 0 <= p.j < 6
    assert {(p.i, p.j) for p in kept_pairs} == {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}


def test_keep_largest_component_is_a_no_op_when_already_connected():
    frames = _frames_for(4)
    pairs = [_fake_pair(k, k + 1) for k in range(3)]
    prior = np.arange(4, dtype=np.float64)
    kept, kept_pairs, kept_yaw, dropped = solve_mod._keep_largest_component(
        frames, pairs, prior, StitchMetrics()
    )
    assert dropped == [] and len(kept) == 4 and len(kept_pairs) == 3


def test_check_coverage_warns_about_dropped_frames_but_still_solves():
    metrics = StitchMetrics()
    dropped = [(12, 5.0), (14, 350.0)]
    solve_mod._check_coverage(340.0, _frames_for(10), dropped, metrics)  # must not raise
    assert len(metrics.warnings) == 1
    warning = metrics.warnings[0]
    assert warning.startswith("dropped_unlinked_keyframes: 2 of 12")
    assert "5" in warning and "350" in warning, f"warning must locate the drops: {warning}"


def test_check_coverage_reports_a_split_graph_instead_of_refusing_it():
    """Half a graph is a fragment, and the user gets the fragment plus a warning.

    Refusing here used to throw away a two-minute capture the pipeline could still build
    something from. The largest piece is stitched, the rest is filled, and the warning plus
    ``covered_fraction`` are what tell the user how much of it is real.
    """
    metrics = StitchMetrics()
    solve_mod._check_coverage(350.0, _frames_for(10), [(k, k * 30.0) for k in range(11)], metrics)
    assert any(w.startswith("fragmented_capture:") for w in metrics.warnings), metrics.warnings
    fragment = next(w for w in metrics.warnings if w.startswith("fragmented_capture:"))
    assert "10 of 21" in fragment and "48%" in fragment, fragment


def test_check_coverage_still_refuses_a_sweep_that_is_not_a_turn():
    """The one floor that stays: below it there is no panorama to build at all."""
    with pytest.raises(InsufficientCoverageError, match="deg of yaw"):
        solve_mod._check_coverage(180.0, _frames_for(10), [(k, k * 30.0) for k in range(11)],
                                  StitchMetrics())


def test_check_coverage_raises_on_a_short_sweep():
    with pytest.raises(InsufficientCoverageError, match="deg of yaw"):
        solve_mod._check_coverage(180.0, _frames_for(10), [], StitchMetrics())


def test_measured_yaw_span_uses_the_estimated_focal_not_the_prior():
    """The reported span must not keep the 67-degree guess baked into it.

    A portrait phone is nearer 45 degrees across the short edge, and measuring its sweep
    against a 67-degree prior overstates the turn by ~1.6x — enough to let a 220-degree
    sweep report as a closed circle, which is exactly the claim this product must not get
    wrong.
    """
    true_focal = Intrinsics.from_hfov(IMG_W, IMG_H, 45.0).focal
    true_intr = Intrinsics(focal=true_focal, cx=IMG_W / 2, cy=IMG_H / 2, width=IMG_W, height=IMG_H)
    n_extracted, step_deg = 120, 3.0
    motion = np.zeros((n_extracted - 1, 2))
    motion[:, 0] = -true_focal * math.tan(math.radians(step_deg))
    true_span = step_deg * (n_extracted - 1)

    keyframes = []
    from pathlib import Path

    for pos, src in enumerate(range(0, n_extracted, 6)):
        f = Frame(index=pos, path=Path(f"k{pos}.jpg"), timestamp=0.0, width=IMG_W, height=IMG_H)
        f._cache["source_index"] = src
        keyframes.append(f)

    # The span is measured over the keyframes, which are the only views the renderer
    # will have; the last one sits a little short of the end of the track.
    expected = step_deg * keyframes[-1]._cache["source_index"]
    assert expected < true_span
    measured = solve_mod._measured_yaw_span(motion, true_intr, keyframes)
    assert measured == pytest.approx(expected, rel=0.02), f"{measured} vs {expected}"

    prior = Intrinsics.from_hfov(IMG_W, IMG_H, sel.DEFAULT_HFOV_DEG)
    inflated = solve_mod._measured_yaw_span(motion, prior, keyframes)
    assert inflated > 1.4 * measured, "the prior really does inflate the span"

    # A genuinely partial sweep must stay under the coverage floor after correction.
    partial = motion[: (n_extracted // 3)]
    short_kf = [f for f in keyframes if f._cache["source_index"] <= partial.shape[0]]
    assert solve_mod._measured_yaw_span(partial, true_intr, short_kf) < solve_mod.MIN_YAW_SPAN_DEG


def test_focal_clamp_envelope_admits_a_portrait_phone():
    """A 45-degree portrait capture is real hardware, not a degenerate estimate."""
    assert rot.MIN_HFOV_DEG < 45.0 and rot.MAX_HFOV_DEG > 100.0
    intr = Intrinsics.from_hfov(IMG_W, IMG_H, 45.0)
    R = g.from_yaw_pitch_roll(math.radians(12.0), math.radians(3.0), 0.0)
    H = g.homography_from_rotation(R, intr.K, intr.K)
    pair = MatchPair(0, 1, np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), H=H)
    assert rot.estimate_focal([pair], IMG_W, IMG_H) == pytest.approx(intr.focal, rel=1e-3)


# ======================================================================================
# 11. Connectivity repair, end to end
# ======================================================================================


@pytest.fixture(scope="module")
def rendered_sweep(tmp_path_factory):
    """A synthetic 360 sweep on disk, shared by the repair and drop tests."""
    root = tmp_path_factory.mktemp("repair")
    rng = np.random.default_rng(20)
    paths, truth = _render_sweep(root / "shots", n=24, step_deg=15.0, rng=rng)
    return {"root": root, "paths": paths, "truth": truth}


def test_repair_connectivity_reconnects_an_isolated_keyframe(rendered_sweep):
    """The retry must put back links that the first pass missed.

    The first pass is run normally and then every pair touching two adjacent keyframes is
    deleted, which is precisely the state a blown-out or motion-blurred frame leaves
    behind: the overlap is genuinely there, the first pass just failed to certify it. The
    repair should find those links again without any threshold being loosened for the
    capture as a whole.
    """
    from biosphere.pano.extract import load_photos

    root = rendered_sweep["root"]
    config = PipelineConfig.for_mode(QualityMode.BALANCED, max_frame_width=VIEW_W)
    frames = load_photos(list(rendered_sweep["paths"]), root / "norm", config)
    featuresets = feat.detect_features(frames, config)
    K = Intrinsics.from_hfov(VIEW_W, VIEW_H, HFOV).K
    full = feat.match_pairs(featuresets, feat.candidate_pairs(len(frames), None, config), K, config)
    assert len(feat.connected_components(len(frames), full)) == 1

    stranded = {7, 8}
    broken = [p for p in full if p.i not in stranded and p.j not in stranded]
    before = feat.connected_components(len(frames), broken)
    assert len(before) == 3, f"expected two stranded keyframes, got {[len(c) for c in before]}"

    repaired = feat.repair_connectivity(frames, broken, K, config)
    after = feat.connected_components(len(frames), repaired)
    assert len(after) == 1, f"still {[len(c) for c in after]} after repair"
    assert len(repaired) > len(broken)
    for p in repaired:
        assert p.n_inliers >= 12
        assert p.pts_i.shape == p.pts_j.shape
        assert p.H is not None


def test_every_detector_honours_the_feature_budget(rendered_sweep):
    """AKAZE has no ``nfeatures``, so ``max_features`` used to mean nothing for it.

    That is not a cosmetic inconsistency. FAST mode is the only mode that selects AKAZE,
    and with the budget ignored its yield was fixed by an OpenCV default tuned for outdoor
    photography — about 300 keypoints on an interior frame, with starved frames down at 73.
    The pose graph then shattered and a perfectly good capture was refused as
    ``InsufficientOverlapError``. Nothing downstream could ask for more, because the knob
    was not connected to anything.
    """
    from biosphere.pano.extract import load_photos

    root = rendered_sweep["root"]
    frames = load_photos(
        list(rendered_sweep["paths"])[:4],
        root / "budget",
        PipelineConfig.for_mode(QualityMode.BALANCED, max_frame_width=VIEW_W),
    )
    for detector in ("sift", "akaze"):
        counts = []
        for budget in (200, 1500):
            config = PipelineConfig.for_mode(
                QualityMode.BALANCED, detector=detector, max_features=budget,
                max_frame_width=VIEW_W,
            )
            n = [fs.n for fs in feat.detect_features(frames, config)]
            assert max(n) <= budget, f"{detector} ignored the {budget} budget: {n}"
            counts.append(max(n))
        assert counts[1] > counts[0], (
            f"{detector} yield did not respond to the budget at all: {counts}"
        )


def test_cap_to_budget_keeps_the_strongest_in_detection_order():
    """Strongest by response, but returned in the order they were detected.

    Order matters for reproducibility: the descriptor rows have to stay aligned with the
    keypoints, and a run must not depend on how equal responses happened to sort.
    """
    responses = [0.5, 0.9, 0.1, 0.7, 0.3]
    kps = [cv2.KeyPoint(float(i), 0.0, 1.0) for i in range(len(responses))]
    for kp, r in zip(kps, responses, strict=True):
        kp.response = r
    desc = np.arange(len(responses) * 3, dtype=np.float32).reshape(len(responses), 3)

    kept, kept_desc = feat._cap_to_budget(kps, desc, 3)
    assert [int(k.pt[0]) for k in kept] == [0, 1, 3], "wrong keypoints or wrong order"
    assert kept_desc.tolist() == desc[[0, 1, 3]].tolist(), "descriptors lost their alignment"

    same, same_desc = feat._cap_to_budget(kps, desc, 99)
    assert same is kps and same_desc is desc, "an unexceeded budget must not copy anything"
    assert feat._cap_to_budget(kps, None, 2)[1] is None


def test_repair_connectivity_is_a_no_op_on_a_healthy_graph(rendered_sweep):
    """It must cost nothing — and change nothing — when the graph is already connected."""
    from biosphere.pano.extract import load_photos

    config = PipelineConfig.for_mode(QualityMode.BALANCED, max_frame_width=VIEW_W)
    frames = load_photos(list(rendered_sweep["paths"]), rendered_sweep["root"] / "norm2", config)
    featuresets = feat.detect_features(frames, config)
    K = Intrinsics.from_hfov(VIEW_W, VIEW_H, HFOV).K
    pairs = feat.match_pairs(featuresets, feat.candidate_pairs(len(frames), None, config), K, config)
    assert feat.repair_connectivity(frames, pairs, K, config) is pairs


def test_solve_capture_survives_an_unlinkable_keyframe(rendered_sweep, tmp_path):
    """Two frames of pure noise must cost those frames, not the capture.

    This is the scenario that used to end the job: an agent sweeps past a window, a couple
    of frames blow out and link to nothing, and the user got an error instead of a
    panorama.
    """
    from biosphere.pano.solve import solve_capture

    rng = np.random.default_rng(5)
    paths = list(rendered_sweep["paths"])
    shots = tmp_path / "shots"
    shots.mkdir()
    corrupted = []
    for k, src in enumerate(paths):
        dst = shots / src.name
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if k in (7, 8):
            # Independent noise: sharp enough to be chosen as a keyframe, and impossible
            # to match against anything, including the other noise frame.
            img = rng.integers(0, 256, img.shape, dtype=np.uint8)
            corrupted.append(k)
        cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        paths[k] = dst

    config = PipelineConfig.for_mode(QualityMode.BALANCED, max_frame_width=VIEW_W)
    result = solve_capture(paths, config, work_dir=tmp_path / "work")

    assert len(result.frames) == len(result.poses) >= 8
    assert [f.index for f in result.frames] == list(range(len(result.frames)))
    for pose in result.poses:
        assert np.linalg.det(pose.R) == pytest.approx(1.0, abs=1e-9)
    for pair in result.pairs:
        assert 0 <= pair.i < len(result.frames) and 0 <= pair.j < len(result.frames)
    kept_sources = {f._cache["source_index"] for f in result.frames}
    assert kept_sources, "no keyframes survived"


# ======================================================================================
# 12. The loop-closure statistic must not cry wolf
# ======================================================================================


def _loop_pair(i: int, j: int, R: np.ndarray, inliers: int) -> MatchPair:
    pts = np.zeros((inliers, 2), dtype=np.float32)
    pair = MatchPair(i=i, j=j, pts_i=pts, pts_j=pts, is_loop_closure=True)
    pair.R = R
    return pair


def _ring(n: int) -> list[np.ndarray]:
    return [g.from_yaw_pitch_roll(2.0 * math.pi * k / n, 0.0, 0.0) for k in range(n)]


def test_weighted_quantile_respects_the_weights():
    v = np.array([0.0, 1.0, 10.0])
    assert ba._weighted_quantile(v, np.array([1.0, 1.0, 1.0]), 0.5) == 1.0
    # A heavy first sample drags the median onto itself.
    assert ba._weighted_quantile(v, np.array([100.0, 1.0, 1.0]), 0.5) == 0.0
    # A heavy last sample drags it the other way.
    assert ba._weighted_quantile(v, np.array([1.0, 1.0, 100.0]), 0.5) == 10.0
    assert ba._weighted_quantile(np.zeros(0), np.zeros(0), 0.5) == 0.0


def test_loop_statistic_ignores_one_flimsy_short_baseline_pair():
    """The reported bug: a good capture warned about a seam that was not there.

    The maximum was set by a five-apart pair carried by thirty-odd inliers — exactly the
    measurement the robust loss was right to discount — while the pair that actually
    closes the loop agreed to a fraction of a degree.
    """
    n = 24
    poses = _ring(n)
    pairs = [
        # Plenty of honest medium-baseline pairs, all consistent with the solve.
        _loop_pair(k, k + 5, poses[k + 5] @ poses[k].T, inliers=300)
        for k in range(0, n - 5, 2)
    ]
    # One flimsy pair that disagrees wildly.
    bad = g.from_yaw_pitch_roll(math.radians(4.0), 0.0, 0.0) @ (poses[18] @ poses[13].T)
    pairs.append(_loop_pair(13, 18, bad, inliers=30))
    # The pair that genuinely spans the loop, and which agrees.
    pairs.append(_loop_pair(0, n - 2, poses[n - 2] @ poses[0].T, inliers=120))

    stats = ba.loop_closure_stats(poses, pairs)
    assert stats.max_deg == pytest.approx(4.0, abs=0.1), "the outlier must stay visible"
    assert stats.worst_pair == (13, 18)
    assert stats.robust_deg < 0.1, f"headline was dragged by the outlier: {stats.as_dict()}"
    assert stats.n_span_pairs == 1
    assert stats.robust_deg <= solve_mod.WARN_LOOP_RESIDUAL_DEG, "would have warned wrongly"


def test_loop_statistic_tracks_a_real_drift():
    """It must still catch a loop that genuinely did not close."""
    n = 24
    truth = _ring(n)
    pairs = [_loop_pair(k, k + 5, truth[k + 5] @ truth[k].T, 300) for k in range(0, n - 5, 2)]
    pairs.append(_loop_pair(0, n - 2, truth[n - 2] @ truth[0].T, 120))

    seen = []
    for drift_deg in (0.0, 1.0, 3.0, 6.0):
        drifted = [
            truth[k] @ g.from_yaw_pitch_roll(-math.radians(drift_deg * k / (n - 1)), 0.0, 0.0).T
            for k in range(n)
        ]
        seen.append(ba.loop_closure_stats(drifted, pairs).robust_deg)

    assert seen[0] < 0.05
    assert seen == sorted(seen), f"statistic must rise with drift: {seen}"
    assert seen[-1] > 4.0, f"6 deg of drift should be unmissable: {seen}"
    # And the calibrated threshold must sit between "fine" and "clearly broken".
    assert seen[1] < solve_mod.WARN_LOOP_RESIDUAL_DEG < seen[3]


def test_loop_statistic_falls_back_when_nothing_spans_the_loop():
    """With no long-baseline pair, report the robust spread of what we do have."""
    n = 24
    poses = _ring(n)
    off = g.from_yaw_pitch_roll(math.radians(1.5), 0.0, 0.0)
    pairs = [_loop_pair(k, k + 4, off @ (poses[k + 4] @ poses[k].T), 200) for k in range(0, 12, 2)]
    stats = ba.loop_closure_stats(poses, pairs)
    assert stats.n_span_pairs == 0
    assert stats.robust_deg == pytest.approx(stats.p90_deg)
    assert stats.robust_deg == pytest.approx(1.5, abs=0.1)


def test_loop_closure_residual_is_the_headline_of_the_stats():
    n = 24
    poses = _ring(n)
    pairs = [_loop_pair(0, n - 2, poses[n - 2] @ poses[0].T, 120)]
    assert ba.loop_closure_residual(poses, pairs) == pytest.approx(
        ba.loop_closure_stats(poses, pairs).robust_deg
    )


def test_solve_capture_does_not_warn_about_a_clean_loop(solved_capture):
    """The end-to-end guard on the actual regression: no seam warning on a good capture."""
    metrics = solved_capture["result"].metrics
    assert metrics.loop_closure_residual_deg < solve_mod.WARN_LOOP_RESIDUAL_DEG
    assert not [w for w in metrics.warnings if w.startswith("loop_did_not_close")], metrics.warnings


# ======================================================================================
# 13. The pre-solve capture gate
# ======================================================================================


def _assessment(verdict, **kw):
    from biosphere.pano.validate import CaptureAssessment

    defaults = dict(
        reason="test", message_tr="tr", message_en="en", median_rot_residual_px=18.0,
        median_free_residual_px=1.5, residual_ratio=12.0, n_samples=60, n_usable_samples=55,
    )
    defaults.update(kw)
    return CaptureAssessment(verdict=verdict, **defaults)


def _gate(monkeypatch, assessment, metrics=None):
    from biosphere.pano.types import CaptureKind, NullProgress

    monkeypatch.setattr(solve_mod, "assess_capture", lambda *a, **k: assessment)
    metrics = metrics if metrics is not None else StitchMetrics()
    return solve_mod._gate_on_capture_quality(
        _frames_for(20), PipelineConfig(), CaptureKind.VIDEO, metrics, NullProgress()
    ), metrics


def test_the_gate_refuses_a_walking_capture(monkeypatch):
    from biosphere.pano.validate import CaptureVerdict, WalkingCaptureError

    monkeypatch.delenv("BIOSPHERE_FORCE_STITCH", raising=False)
    monkeypatch.delenv("BIOSPHERE_SEGMENT_ROOMS", raising=False)
    with pytest.raises(WalkingCaptureError) as excinfo:
        _gate(monkeypatch, _assessment(CaptureVerdict.WALKING))
    assert excinfo.value.payload["assessment"]["verdict"] == "walking"
    assert "segments" not in excinfo.value.payload, "segmentation must stay opt-in"


def test_the_gate_attaches_the_segmentation_when_asked(monkeypatch):
    """``BIOSPHERE_SEGMENT_ROOMS=1`` turns a bare failure into per-room re-shoot guidance."""
    from biosphere.pano import rooms
    from biosphere.pano.validate import CaptureVerdict, WalkingCaptureError

    monkeypatch.delenv("BIOSPHERE_FORCE_STITCH", raising=False)
    monkeypatch.setenv("BIOSPHERE_SEGMENT_ROOMS", "1")
    fake = [
        rooms.RoomSegment(
            index=0, start_s=0.0, end_s=12.0, start_frame=0, end_frame=96, n_samples=48,
            n_usable_samples=40, median_rot_residual_px=22.0, median_free_residual_px=1.5,
            residual_ratio=14.7, median_inlier_ratio=0.3, total_turn_deg=250.0,
            yaw_span_deg=120.0, stitchable=CaptureVerdict.WALKING,
            stitchable_reason="translation_dominates",
        )
    ]
    monkeypatch.setattr(solve_mod, "assess_capture", lambda *a, **k: _assessment(CaptureVerdict.WALKING))
    monkeypatch.setattr(rooms, "segment_rooms", lambda *a, **k: fake)

    from biosphere.pano.types import CaptureKind, NullProgress

    with pytest.raises(WalkingCaptureError) as excinfo:
        solve_mod._gate_on_capture_quality(
            _frames_for(20), PipelineConfig(), CaptureKind.VIDEO, StitchMetrics(), NullProgress()
        )
    segments = excinfo.value.payload["segments"]
    assert len(segments) == 1 and segments[0]["stitchable"] == "walking"


def test_force_stitch_turns_the_refusal_into_a_warning(monkeypatch):
    from biosphere.pano.validate import CaptureVerdict

    monkeypatch.setenv("BIOSPHERE_FORCE_STITCH", "1")
    assessment, metrics = _gate(monkeypatch, _assessment(CaptureVerdict.WALKING, forced=True))
    assert assessment.verdict is CaptureVerdict.WALKING, "the verdict itself must not be softened"
    assert any(w.startswith("forced_stitch") for w in metrics.warnings), metrics.warnings


def test_a_degraded_capture_proceeds_with_a_warning_that_names_the_problem(monkeypatch):
    from biosphere.pano.validate import CaptureVerdict

    _assess, metrics = _gate(
        monkeypatch,
        _assessment(CaptureVerdict.DEGRADED, median_rot_residual_px=3.6, focal_px=692.0,
                    analysis_width=478, residual_ratio=3.3),
    )
    warning = next(w for w in metrics.warnings if w.startswith("capture_geometry_degraded"))
    assert "3.6 px" in warning and "bend" in warning


def test_a_good_capture_passes_the_gate_silently(monkeypatch):
    from biosphere.pano.validate import CaptureVerdict

    _assess, metrics = _gate(
        monkeypatch, _assessment(CaptureVerdict.GOOD, median_rot_residual_px=0.9, residual_ratio=1.6)
    )
    assert metrics.warnings == []


def test_the_gate_does_not_add_a_stage_the_progress_bar_cannot_weight(solved_capture):
    """A frozen contract, worth pinning explicitly rather than by accident.

    ``pipeline.STAGE_WEIGHTS`` has no entry for a validation stage, so a new stage name
    would be forwarded at the bar's high-water mark and read as a hang, and a new
    ``stage_seconds`` key would break every consumer that enumerates them. The scan
    therefore reports and bills itself as part of ``extract``.
    """
    from biosphere.pano.pipeline import STAGE_WEIGHTS

    metrics = solved_capture["result"].metrics
    assert set(metrics.stage_seconds) <= set(STAGE_WEIGHTS)
    assert solved_capture["progress"].stages == [
        "probe", "extract", "select", "features", "match", "bundle"
    ]


# ======================================================================================
# Going back to the video for a direction the chosen frame could not link
# ======================================================================================


def _pool_for(n: int) -> list[Frame]:
    """A pool of extracted video frames, quality descending with index for determinism."""
    from pathlib import Path

    return [
        Frame(
            index=k,
            path=Path(f"pool_{k}.jpg"),
            timestamp=float(k),
            width=IMG_W,
            height=IMG_H,
            quality=1.0 - k / 1000.0,
        )
        for k in range(n)
    ]


def _stranded_setup():
    """Keyframes 0-1-2 linked, keyframe 3 stranded, with a video pool behind them."""
    # 1.5 deg of yaw between pool frames: a slow sweep at 30 fps puts many frames inside the
    # half-bin the rescue searches, which is the whole reason there is anything to choose
    # between. A coarse pool would make the spread untestable and unrepresentative.
    pool = _pool_for(200)
    keyframes = [pool[0], pool[60], pool[120], pool[180]]
    for position, frame in enumerate(keyframes):
        frame._cache["source_index"] = position * 60
        frame.index = position
    pairs = [
        MatchPair(0, 1, np.zeros((40, 2), np.float32), np.zeros((40, 2), np.float32), confidence=1.0),
        MatchPair(1, 2, np.zeros((40, 2), np.float32), np.zeros((40, 2), np.float32), confidence=1.0),
    ]
    cum_yaw = np.arange(200, dtype=np.float64) * 1.5
    prior_yaw = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    featuresets = [object() for _ in keyframes]
    return pool, cum_yaw, keyframes, featuresets, pairs, prior_yaw


def test_rescue_pulls_a_different_video_frame_in_for_a_stranded_direction(monkeypatch):
    """The point of shooting video: another frame of the same view gets a turn.

    Selection picks the sharpest frame in each bin BEFORE anything is matched, so it is
    guessing which frame will link. When the guess is wrong the direction used to be lost
    outright — and a lost direction is a seam in the delivered panorama, in exactly that
    place. Here the alternate links and the direction comes back.
    """
    pool, cum_yaw, keyframes, featuresets, pairs, prior_yaw = _stranded_setup()
    tried: list[int] = []

    def fake_detect(frames, config, progress=None):
        tried.append(int(frames[0].index))
        return [object()]

    def fake_match(sets, candidates, K, config, progress=None):
        # Links to the first target offered, as a real match against a neighbour would.
        i, j = candidates[0]
        return [MatchPair(i, j, np.zeros((44, 2), np.float32), np.zeros((44, 2), np.float32), confidence=0.9)]

    monkeypatch.setattr(solve_mod, "detect_features", fake_detect)
    monkeypatch.setattr(solve_mod, "match_pairs", fake_match)
    metrics = StitchMetrics()

    out_kf, out_fs, out_pairs, out_prior, rescued = solve_mod._rescue_with_alternates(
        pool, cum_yaw, keyframes, featuresets, pairs, prior_yaw,
        cum_yaw, None, PipelineConfig.for_mode(QualityMode.BALANCED), metrics,
    )

    assert rescued == 1, "the stranded direction should have been recovered"
    assert len(out_kf) == 5 and len(out_fs) == 5 and out_prior.size == 5
    assert tried, "an alternate frame must actually be looked at"

    added = [p for p in out_pairs if p.j == 4]
    assert added, f"the alternate must be linked into the graph: {out_pairs}"
    for pair in added:
        # i < j is an invariant the rest of the solver relies on; the alternate is always
        # the higher index, so no homography ever has to be reversed to restore it.
        assert pair.i < pair.j
        assert pair.i in {0, 1, 2}, "must link to the surviving component, not the stranded frame"

    # The alternate joins the MAIN component; the stranded original stays stranded and is
    # dropped by _keep_largest_component right after. Supplement, not replace — the frame
    # that would not link is not made to link, its direction is re-photographed from the
    # video instead. What matters is that the main component now covers that direction.
    components = solve_mod.connected_components(len(out_kf), out_pairs)
    assert 4 in components[0], f"the alternate must join the main component: {components}"
    assert components[1] == [3], f"the original stays stranded and will be dropped: {components}"
    assert any(w.startswith("recovered_from_video:") for w in metrics.warnings), metrics.warnings


def test_rescue_only_looks_near_the_stranded_direction(monkeypatch):
    """An alternate has to be the same view, not merely another frame of the capture."""
    pool, cum_yaw, keyframes, featuresets, pairs, prior_yaw = _stranded_setup()
    looked_at: list[int] = []

    monkeypatch.setattr(
        solve_mod, "detect_features",
        lambda frames, config, progress=None: (looked_at.append(int(frames[0].index)), [object()])[1],
    )
    monkeypatch.setattr(solve_mod, "match_pairs", lambda *a, **k: [])

    solve_mod._rescue_with_alternates(
        pool, cum_yaw, keyframes, featuresets, pairs, prior_yaw,
        cum_yaw, None, PipelineConfig.for_mode(QualityMode.BALANCED), StitchMetrics(),
    )

    assert looked_at, "alternates should have been tried"
    # The stranded keyframe is pool index 30 at 270 deg; half a bin is ~15 deg either side.
    for index in looked_at:
        assert abs(float(cum_yaw[index]) - 270.0) <= 16.0, f"pool[{index}] is a different view"
    assert len(looked_at) <= solve_mod._ALTERNATES_PER_STRANDED


def test_rescue_spreads_its_tries_instead_of_taking_the_top_of_the_quality_rank(monkeypatch):
    """Measured: quality-ranked alternates recovered 1 of 23 stranded directions.

    The frame that failed was itself the quality pick, so its nearest rivals by quality are
    its immediate neighbours — the same half-second of video, the same blur burst, the same
    failure. Spreading the tries across the window costs nothing and samples outside it.
    """
    pool, cum_yaw, keyframes, featuresets, pairs, prior_yaw = _stranded_setup()
    looked_at: list[int] = []

    monkeypatch.setattr(
        solve_mod, "detect_features",
        lambda frames, config, progress=None: (looked_at.append(int(frames[0].index)), [object()])[1],
    )
    monkeypatch.setattr(solve_mod, "match_pairs", lambda *a, **k: [])

    solve_mod._rescue_with_alternates(
        pool, cum_yaw, keyframes, featuresets, pairs, prior_yaw,
        cum_yaw, None, PipelineConfig.for_mode(QualityMode.BALANCED), StitchMetrics(),
    )

    assert len(looked_at) >= 2, looked_at
    # `_pool_for` makes quality strictly decreasing with index, so a pure quality ranking
    # would hand back a run of consecutive frames. Spreading must not.
    span = max(looked_at) - min(looked_at)
    assert span > len(looked_at), f"tries are bunched together, not spread: {sorted(looked_at)}"


def test_rescue_changes_nothing_when_no_alternate_links(monkeypatch):
    """A failed rescue must leave the solve exactly as it found it."""
    pool, cum_yaw, keyframes, featuresets, pairs, prior_yaw = _stranded_setup()
    monkeypatch.setattr(solve_mod, "detect_features", lambda f, c, progress=None: [object()])
    monkeypatch.setattr(solve_mod, "match_pairs", lambda *a, **k: [])
    metrics = StitchMetrics()

    out_kf, out_fs, out_pairs, out_prior, rescued = solve_mod._rescue_with_alternates(
        pool, cum_yaw, keyframes, featuresets, pairs, prior_yaw,
        cum_yaw, None, PipelineConfig.for_mode(QualityMode.BALANCED), metrics,
    )
    assert rescued == 0
    assert len(out_kf) == 4 and len(out_pairs) == 2 and out_prior.size == 4
    assert metrics.warnings == []


def test_rescue_is_a_no_op_on_a_healthy_graph(monkeypatch):
    """No stranded frames, no feature detection: the cost is paid only when it can help."""
    pool, cum_yaw, keyframes, featuresets, pairs, prior_yaw = _stranded_setup()
    pairs.append(
        MatchPair(2, 3, np.zeros((40, 2), np.float32), np.zeros((40, 2), np.float32), confidence=1.0)
    )
    called = False

    def boom(*a, **k):
        nonlocal called
        called = True
        return [object()]

    monkeypatch.setattr(solve_mod, "detect_features", boom)
    out_kf, _fs, out_pairs, _prior, rescued = solve_mod._rescue_with_alternates(
        pool, cum_yaw, keyframes, featuresets, pairs, prior_yaw,
        cum_yaw, None, PipelineConfig.for_mode(QualityMode.BALANCED), StitchMetrics(),
    )
    assert rescued == 0 and not called
    assert len(out_kf) == 4 and len(out_pairs) == 3
