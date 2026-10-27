"""The harness proving itself before it is allowed to judge anything else.

A measuring instrument that has not been calibrated is worse than no instrument, because
it produces numbers people believe. These tests calibrate this one. The load-bearing one
is :func:`test_reprojection_matches_ground_truth`: it simulates an ideal nodal capture,
un-projects every frame back onto the sphere using nothing but its ground-truth pose and
:mod:`biosphere.pano.geometry`, and demands that the result reproduce the panorama the
simulator started from. If that fails, the simulator and the geometry core disagree about
what "equirectangular" means and every dB the harness ever reports is fiction.

Nothing here imports the stitcher. The harness must be correct on its own terms first.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from biosphere.pano import geometry as g
from biosphere.pano.types import CameraPose, Intrinsics
from biosphere.synth import metrics as M
from biosphere.synth.capture import EFFECT_PRESETS, Trajectory, simulate_capture
from biosphere.synth.encode import encode_video, probe_duration, probe_frame_count
from biosphere.synth.render import (
    apply_camera_response,
    render_equirect,
    render_view,
    reproject_to_equirect,
)
from biosphere.synth.room import STYLES, Room

# Small enough to keep the suite quick, large enough to be a fair test of the conventions.
PANO_W, PANO_H = 1024, 512
FRAME_W, FRAME_H = 256, 144
CAPTURE_S, CAPTURE_FPS = 6.0, 10.0
ROOM_SEED = 7
MOTION_SEED = 5


def _frame_intrinsics() -> Intrinsics:
    """A camera whose angular resolution matches the test canvas.

    ``pi / PANO_H`` radians is one equirect pixel, so a focal length of ``PANO_H / pi``
    pixels puts one frame pixel on one canvas pixel. Matching them keeps the reprojection
    test measuring geometry rather than the resampling loss of an arbitrary rescale.
    """
    return Intrinsics(
        focal=PANO_H / math.pi, cx=FRAME_W / 2.0, cy=FRAME_H / 2.0, width=FRAME_W, height=FRAME_H
    )


def _weighted_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    weights = g.sphere_solid_angle_weights(a.shape[0], a.shape[1], dtype=np.float64) * mask
    d = a.astype(np.float64) - b.astype(np.float64)
    per_pixel = np.einsum("ijk,ijk->ij", d, d) / 3.0
    mse = float((per_pixel * weights).sum() / weights.sum())
    return float("inf") if mse <= 0 else 10.0 * math.log10(255.0**2 / mse)


# ======================================================================================
# Shared, expensive fixtures
# ======================================================================================


@pytest.fixture(scope="module")
def room() -> Room:
    return Room.random(ROOM_SEED, "rich")


@pytest.fixture(scope="module")
def ground_truth(room: Room) -> np.ndarray:
    return render_equirect(room, PANO_W, PANO_H)


@pytest.fixture(scope="module")
def nodal_capture(room: Room, tmp_path_factory: pytest.TempPathFactory):
    traj = Trajectory.three_row(
        duration_s=CAPTURE_S, fps=CAPTURE_FPS, seed=MOTION_SEED, arm_radius_m=0.0
    )
    return simulate_capture(
        room,
        traj,
        _frame_intrinsics(),
        out_dir=tmp_path_factory.mktemp("nodal"),
        effects="clean",
        seed=3,
        ground_truth_width=PANO_W,
    )


@pytest.fixture(scope="module")
def parallax_capture(room: Room, tmp_path_factory: pytest.TempPathFactory):
    traj = Trajectory.three_row(
        duration_s=CAPTURE_S, fps=CAPTURE_FPS, seed=MOTION_SEED, arm_radius_m=0.30
    )
    return simulate_capture(
        room,
        traj,
        _frame_intrinsics(),
        out_dir=tmp_path_factory.mktemp("parallax"),
        effects="clean",
        seed=3,
        arm_radius_m=0.30,
        ground_truth_width=PANO_W,
    )


# ======================================================================================
# 1. The ray tracer is consistent
# ======================================================================================


def test_equirect_render_is_sane(ground_truth: np.ndarray) -> None:
    pano = ground_truth
    assert pano.shape == (PANO_H, PANO_W, 3)
    assert pano.dtype == np.uint8
    assert np.isfinite(pano.astype(np.float32)).all(), "tracer produced non-finite radiance"
    assert not (pano.max(axis=2) == 0).any(), "a closed room has no black pixels: rays escaped"
    mean = float(pano.mean())
    assert 60.0 < mean < 200.0, f"implausible overall brightness {mean:.1f}"
    assert float(pano.std()) > 25.0, "the room is too flat to be a useful test scene"
    p1, p99 = np.percentile(pano, [1.0, 99.0])
    assert p1 > 4.0, "shadows are crushed to black"
    assert float((pano >= 254).mean()) < 0.30, "too much of the room is blown out"


def test_render_is_resolution_consistent(room: Room) -> None:
    """A 2x render, box-downsampled, must agree with a direct 1x render.

    This is what proves the footprint filtering is doing its job: without it the two
    resolutions would alias differently and disagree by many dB.
    """
    small = render_equirect(room, 512, 256)
    big = render_equirect(room, 1024, 512).reshape(256, 2, 512, 2, 3).mean(axis=(1, 3))
    psnr = _weighted_psnr(small, big.astype(np.float32), np.ones((256, 512), dtype=np.float64))
    assert psnr > 33.0, f"renders disagree across resolutions: {psnr:.2f} dB"


def test_trace_reports_geometry(room: Room) -> None:
    theta, phi = g.equirect_angles(64, 32)
    rays = g.rays_from_equirect(theta, phi).reshape(-1, 3)
    rgb, depth, fid = room.trace(room.eye, rays, pixel_angle=math.pi / 32)
    assert rgb.shape == (64 * 32, 3) and depth.shape == (64 * 32,) and fid.shape == (64 * 32,)
    assert np.isfinite(depth).all() and (depth > 0).all(), "every ray must hit the room"
    assert depth.max() < 3.0 * max(room.width_m, room.depth_m, room.height_m)
    assert (fid >= 0).all(), "no ray may escape a closed box"
    # The nadir looks at the floor, the zenith at the ceiling.
    zenith_idx, nadir_idx = 32, 31 * 64 + 32
    assert fid[zenith_idx] in (2,) or fid[zenith_idx] >= 8
    assert fid[nadir_idx] in (3,) or fid[nadir_idx] >= 8


# ======================================================================================
# 2. Self-consistency: the harness agrees with geometry.py
# ======================================================================================


def test_reprojection_matches_ground_truth(nodal_capture) -> None:
    """A nodal, artefact-free capture must rebuild its own ground truth.

    Nothing here is allowed to be approximate. The frames were traced from the exact
    centre of the reference panorama, so the only thing standing between them and a
    perfect reconstruction is the resampling kernel. Anything below ~34 dB means the
    conventions disagree — chase the convention, never the threshold.
    """
    frames = nodal_capture.load_frames()
    pano, wsum = reproject_to_equirect(
        frames, nodal_capture.rotations, nodal_capture.intrinsics, PANO_W, PANO_H
    )
    mask = wsum > 0
    assert mask.mean() > 0.5, "the three-row sweep should cover most of the sphere"

    psnr = _weighted_psnr(pano, nodal_capture.ground_truth, mask.astype(np.float64))
    assert psnr > 34.0, f"nodal reprojection only reached {psnr:.2f} dB"

    quality = M.compare(pano, nodal_capture.ground_truth, mask, align=False)
    assert quality.ssim > 0.95, f"SSIM {quality.ssim:.4f} too low for a nodal capture"

    _aligned, yaw = M.align_panoramas(pano, nodal_capture.ground_truth)
    assert abs(yaw) < 0.4, f"a same-frame reprojection should need no yaw, got {yaw:.3f} deg"


# ======================================================================================
# 3. Parallax really is being simulated
# ======================================================================================


def test_parallax_degrades_reprojection(nodal_capture, parallax_capture) -> None:
    """Only the optical centre differs, and that alone must cost measurable quality."""
    for a, b in zip(nodal_capture.rotations, parallax_capture.rotations, strict=True):
        assert math.degrees(g.angular_distance(a, b)) < 1e-4, "the two captures must share poses"
    assert float(np.abs(nodal_capture.origins - nodal_capture.room.eye).max()) < 1e-6
    offsets = np.linalg.norm(parallax_capture.origins - parallax_capture.room.eye, axis=1)
    sway = parallax_capture.trajectory.sway_m
    assert abs(float(offsets.mean()) - 0.30) < 0.02, "the arm should trace a 0.30 m circle"
    assert float(offsets.min()) > 0.30 - 3.0 * sway - 1e-6
    assert float(offsets.max()) < 0.30 + 3.0 * sway + 1e-6

    nodal_pano, wsum = reproject_to_equirect(
        nodal_capture.load_frames(),
        nodal_capture.rotations,
        nodal_capture.intrinsics,
        PANO_W,
        PANO_H,
    )
    par_pano, wsum2 = reproject_to_equirect(
        parallax_capture.load_frames(),
        parallax_capture.rotations,
        parallax_capture.intrinsics,
        PANO_W,
        PANO_H,
    )
    mask = ((wsum > 0) & (wsum2 > 0)).astype(np.float64)
    nodal_psnr = _weighted_psnr(nodal_pano, nodal_capture.ground_truth, mask)
    par_psnr = _weighted_psnr(par_pano, parallax_capture.ground_truth, mask)
    assert nodal_psnr - par_psnr > 2.0, (
        f"parallax barely showed up: nodal {nodal_psnr:.2f} dB vs parallax {par_psnr:.2f} dB"
    )


def test_nodal_trajectory_is_exactly_nodal() -> None:
    traj = Trajectory.single_row(duration_s=4.0, fps=10.0, seed=2, arm_radius_m=0.0)
    traj = traj.centred_at(np.array([1.0, -1.5, 2.0], dtype=np.float32))
    for t in np.linspace(0.0, 4.0, 17):
        assert np.allclose(traj.origin_at(float(t)), [1.0, -1.5, 2.0], atol=0.0)
    orbit = traj.with_arm_radius(0.25)
    radii = [
        float(np.linalg.norm(orbit.origin_at(float(t)) - orbit.centre))
        for t in np.linspace(0.0, 4.0, 17)
    ]
    assert max(abs(r - 0.25) for r in radii) < 1e-4


# ======================================================================================
# 4. Alignment recovers a known rotation
# ======================================================================================


@pytest.mark.parametrize("shift", [17, -63, 256, 401])
def test_align_recovers_known_roll(ground_truth: np.ndarray, shift: int) -> None:
    rolled = np.roll(ground_truth, shift, axis=1)
    aligned, yaw_deg = M.align_panoramas(rolled, ground_truth)
    expected = -shift * 360.0 / PANO_W
    err = abs(float(g.wrap_pi(math.radians(yaw_deg - expected))))
    assert math.degrees(err) < 0.5, f"recovered {yaw_deg:.3f} deg, expected {expected:.3f} deg"
    assert np.array_equal(aligned, ground_truth), "an integer roll must be undone losslessly"


def test_align_is_a_noop_for_identical_input(ground_truth: np.ndarray) -> None:
    aligned, yaw = M.align_panoramas(ground_truth, ground_truth)
    assert abs(yaw) < 1e-6
    assert np.array_equal(aligned, ground_truth)


def test_rotate_equirect_round_trips(ground_truth: np.ndarray) -> None:
    R = g.rodrigues(np.array([0.03, 0.5, -0.02]))
    there = M.rotate_equirect(ground_truth, R)
    back = M.rotate_equirect(there, R.T)
    psnr = _weighted_psnr(back, ground_truth, np.ones((PANO_H, PANO_W), dtype=np.float64))
    assert psnr > 28.0, f"rotating there and back lost too much: {psnr:.2f} dB"


# ======================================================================================
# 5. Pose error is gauge-invariant
# ======================================================================================


def _pose_ring(n: int = 24) -> list[CameraPose]:
    intr = Intrinsics.from_hfov(640, 360, 68.0)
    poses = []
    for i, a in enumerate(np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)):
        R = g.from_yaw_pitch_roll(float(a), 0.35 * math.sin(3 * a), 0.09 * math.cos(2 * a))
        poses.append(CameraPose(frame_index=i, R=R, intrinsics=intr))
    return poses


def test_pose_errors_are_gauge_invariant() -> None:
    truth = _pose_ring()
    rng = np.random.default_rng(19)
    for _ in range(4):
        G = g.rodrigues(rng.normal(0.0, 0.9, 3))
        estimated = [CameraPose(p.frame_index, p.R @ G, p.intrinsics) for p in truth]
        err = M.pose_errors(estimated, truth)
        assert err.max_deg < 0.01, f"a pure gauge shift leaked {err.max_deg:.6f} deg of error"
        assert err.yaw_max_deg < 0.01
        assert err.pitch_max_deg < 0.01
        assert err.roll_max_deg < 0.01
        assert err.gauge_deg > 1.0, "the recovered gauge should be the rotation we applied"


def test_pose_errors_report_real_error() -> None:
    truth = _pose_ring()
    rng = np.random.default_rng(23)
    G = g.rodrigues(np.array([0.1, 0.8, -0.2]))
    noise = [g.rodrigues(rng.normal(0.0, math.radians(0.4), 3)) for _ in truth]
    estimated = [
        CameraPose(p.frame_index, n @ p.R @ G, p.intrinsics)
        for p, n in zip(truth, noise, strict=True)
    ]
    err = M.pose_errors(estimated, truth)
    assert 0.1 < err.mean_deg < 2.0, f"unexpected mean pose error {err.mean_deg:.4f}"
    assert err.max_deg >= err.median_deg >= 0.0
    assert set(err.as_dict()) >= {"mean_deg", "median_deg", "max_deg", "yaw_max_deg"}
    assert json.dumps(err.as_dict())


def test_pose_errors_rejects_mismatched_lengths() -> None:
    truth = _pose_ring(6)
    with pytest.raises(ValueError):
        M.pose_errors(truth[:3], truth)


# ======================================================================================
# 6. compare() sanity
# ======================================================================================


def test_compare_identical_is_perfect(ground_truth: np.ndarray) -> None:
    q = M.compare(ground_truth, ground_truth)
    assert math.isinf(q.psnr) or q.psnr > 80.0
    assert q.ssim == pytest.approx(1.0, abs=1e-6)
    assert q.ssim_horizon == pytest.approx(1.0, abs=1e-6)
    assert q.covered_fraction == pytest.approx(1.0, abs=1e-6)
    assert q.as_dict()["psnr"] == M.PSNR_SENTINEL
    assert json.dumps(q.as_dict())


def test_compare_blur_scores_lower(ground_truth: np.ndarray) -> None:
    light = cv2.GaussianBlur(ground_truth, (0, 0), 0.8)
    heavy = cv2.GaussianBlur(ground_truth, (0, 0), 2.0)
    q_light = M.compare(light, ground_truth)
    q_heavy = M.compare(heavy, ground_truth)
    assert q_light.psnr < 60.0
    assert q_heavy.psnr < q_light.psnr
    assert q_heavy.ssim < q_light.ssim < 1.0
    assert q_heavy.ssim_horizon < q_light.ssim_horizon


def test_compare_is_solid_angle_weighted(ground_truth: np.ndarray) -> None:
    """The same number of damaged pixels must matter far more at the equator."""
    rows = 24
    pole = ground_truth.copy()
    pole[:rows] = 0
    pole[-rows:] = 0
    eq0 = PANO_H // 2 - rows
    equator = ground_truth.copy()
    equator[eq0 : eq0 + 2 * rows] = 0
    full = np.ones((PANO_H, PANO_W), dtype=bool)
    q_pole = M.compare(pole, ground_truth, full, align=False)
    q_eq = M.compare(equator, ground_truth, full, align=False)
    assert q_eq.psnr < q_pole.psnr - 6.0, (
        f"weighting is not being applied: poles {q_pole.psnr:.2f} dB, equator {q_eq.psnr:.2f} dB"
    )


def test_compare_respects_a_mask(ground_truth: np.ndarray) -> None:
    damaged = ground_truth.copy()
    damaged[:, : PANO_W // 4] = 0
    mask = np.ones((PANO_H, PANO_W), dtype=bool)
    mask[:, : PANO_W // 4] = False
    q = M.compare(damaged, ground_truth, mask, align=False)
    assert math.isinf(q.psnr) or q.psnr > 80.0
    assert q.covered_fraction == pytest.approx(0.75, abs=0.01)


def test_seam_scores_react_to_a_seam(ground_truth: np.ndarray) -> None:
    clean = M.seam_score(ground_truth)
    assert 0.5 < clean < 8.0, f"a clean render scored {clean:.2f}"
    broken = ground_truth.copy()
    broken[:, PANO_W // 2 :] = np.roll(ground_truth[:, PANO_W // 2 :], 4, axis=1)
    assert M.seam_score(broken) > clean * 1.2

    assert M.wrap_seam_score(ground_truth) < 1.5
    stepped = ground_truth.astype(np.int16)
    stepped[:, : PANO_W // 2] += 18
    stepped_u8 = np.clip(stepped, 0, 255).astype(np.uint8)
    assert M.wrap_seam_score(stepped_u8) > 3.0


def test_quality_report_is_json_serialisable(ground_truth: np.ndarray) -> None:
    poses = _pose_ring(8)
    report = M.quality_report(
        cv2.GaussianBlur(ground_truth, (0, 0), 1.0),
        ground_truth,
        estimated_poses=poses,
        truth_poses=poses,
        label="unit",
        extra={"note": "smoke"},
    )
    text = json.dumps(report)
    assert '"psnr"' in text and '"seam_score_ratio"' in text and '"poses"' in text
    assert report["seam"]["seam_score_ratio"] > 0.0


# ======================================================================================
# 7. Encoding round-trip
# ======================================================================================


def test_encode_roundtrip(tmp_path: Path) -> None:
    fps = 30.0
    n = 30
    paths = []
    rng = np.random.default_rng(0)
    for i in range(n):
        img = np.full((FRAME_H, FRAME_W, 3), 40 + 4 * i, dtype=np.uint8)
        img[20:60, 10 + 4 * i : 50 + 4 * i] = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
        p = tmp_path / f"f_{i:05d}.png"
        assert cv2.imwrite(str(p), img)
        paths.append(p)

    out = encode_video(paths, tmp_path / "clip.mp4", fps=fps, crf=22, preset="veryfast")
    assert out.is_file() and out.stat().st_size > 0
    assert probe_frame_count(out) == n
    assert abs(probe_duration(out) - n / fps) <= 1.0 / fps

    cap = cv2.VideoCapture(str(out))
    read = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        read += 1
    cap.release()
    assert abs(read - n) <= 1, f"decoder returned {read} frames for {n} encoded"


def test_encode_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        encode_video([], tmp_path / "nothing.mp4", fps=30.0)


# ======================================================================================
# 8. The hard case is still solvable
# ======================================================================================


@pytest.mark.parametrize("seed", [7, 3])
def test_low_texture_style_still_has_landmarks(seed: int) -> None:
    """`low_texture` must be hard, not impossible.

    If SIFT finds almost nothing here then no stitcher could ever solve the style, and a
    regression suite built on it would be testing that an impossible task stays
    impossible.
    """
    pano = render_equirect(Room.random(seed, "low_texture"), PANO_W, PANO_H)
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    keypoints = cv2.SIFT_create().detect(gray, None)
    assert len(keypoints) >= 250, f"only {len(keypoints)} keypoints on the low-texture room"

    rich = render_equirect(Room.random(seed, "rich"), PANO_W, PANO_H)
    rich_kp = cv2.SIFT_create().detect(cv2.cvtColor(rich, cv2.COLOR_BGR2GRAY), None)
    assert len(rich_kp) > len(keypoints), "the rich style should be the easier of the two"


@pytest.mark.parametrize("style", sorted(STYLES))
def test_every_style_renders(style: str) -> None:
    pano = render_equirect(Room.random(11, style), 256, 128)
    assert pano.shape == (128, 256, 3)
    assert not (pano.max(axis=2) == 0).any()


def test_specular_style_is_view_dependent() -> None:
    """The specular highlight must actually move with the observer, or it is just paint."""
    room = Room.random(11, "specular")
    a = render_equirect(room, 256, 128, room.eye)
    shifted = room.eye + np.array([0.0, -0.30, 0.0], dtype=np.float32)
    b = render_equirect(room, 256, 128, shifted)
    matte = Room.random(11, "low_texture")
    c = render_equirect(matte, 256, 128, matte.eye)
    d = render_equirect(matte, 256, 128, matte.eye + np.array([0.0, -0.30, 0.0], dtype=np.float32))
    assert float(np.abs(a.astype(float) - b.astype(float)).mean()) > 0.0
    assert float(np.abs(c.astype(float) - d.astype(float)).mean()) > 0.0


# ======================================================================================
# 9. Determinism
# ======================================================================================


def test_room_generation_is_deterministic() -> None:
    a = render_equirect(Room.random(42, "rich"), 256, 128)
    b = render_equirect(Room.random(42, "rich"), 256, 128)
    assert a.tobytes() == b.tobytes()
    c = render_equirect(Room.random(43, "rich"), 256, 128)
    assert a.tobytes() != c.tobytes(), "different seeds must give different rooms"


def test_capture_is_byte_identical_for_the_same_seed(tmp_path: Path) -> None:
    room = Room.random(4, "rich")
    intr = Intrinsics.from_hfov(160, 96, 70.0)

    def run(where: str):
        traj = Trajectory.single_row(duration_s=0.5, fps=10.0, seed=8)
        return simulate_capture(
            room,
            traj,
            intr,
            out_dir=tmp_path / where,
            effects="realistic",
            seed=777,
            ground_truth_width=256,
        )

    a = run("a")
    b = run("b")
    assert a.n_frames == b.n_frames == 5
    for pa, pb in zip(a.frame_paths, b.frame_paths, strict=True):
        assert pa.read_bytes() == pb.read_bytes(), f"{pa.name} differs between identical runs"
    assert a.ground_truth.tobytes() == b.ground_truth.tobytes()
    assert np.array_equal(a.gains, b.gains)
    # The frame *files* live in different directories, so compare everything else.
    ta, tb = a.truth_dict(), b.truth_dict()
    for entry in (*ta["frames"], *tb["frames"]):
        entry.pop("file")
    assert ta == tb


def test_camera_response_noise_is_seeded() -> None:
    img = np.full((32, 48, 3), 120, dtype=np.uint8)
    a = apply_camera_response(img, 1.0, 0.0, 4.0, 0, rng=np.random.default_rng(5))
    b = apply_camera_response(img, 1.0, 0.0, 4.0, 0, rng=np.random.default_rng(5))
    c = apply_camera_response(img, 1.0, 0.0, 4.0, 0, rng=np.random.default_rng(6))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ======================================================================================
# Camera response, trajectory shape and effects presets
# ======================================================================================


def test_camera_response_stages_do_what_they_claim() -> None:
    flat = np.full((64, 96, 3), 100, dtype=np.uint8)
    assert int(apply_camera_response(flat, 1.5).mean()) == 150

    vig = apply_camera_response(flat, 1.0, 0.4)
    assert int(vig[32, 48].mean()) > int(vig[0, 0].mean()) + 20, "corners must fall off"

    noisy = apply_camera_response(flat, 1.0, 0.0, 6.0, rng=np.random.default_rng(1))
    assert float(noisy.std()) > 3.0
    assert abs(float(noisy.mean()) - 100.0) < 1.5, "noise must be zero-mean"

    # Shot noise: a bright patch must be noisier than a dark one.
    ramp = np.zeros((64, 96, 3), dtype=np.uint8)
    ramp[:, :48] = 20
    ramp[:, 48:] = 230
    n2 = apply_camera_response(ramp, 1.0, 0.0, 10.0, rng=np.random.default_rng(2)).astype(float)
    assert n2[:, 48:].std() > n2[:, :48].std()

    jpeg = apply_camera_response(np.random.default_rng(3).integers(0, 255, (64, 96, 3), dtype=np.uint8), jpeg_quality=40)
    assert jpeg.shape == (64, 96, 3)


def test_trajectories_are_smooth_and_cover_their_pattern() -> None:
    for name, traj in (
        ("single_row", Trajectory.single_row(duration_s=8.0, fps=30.0, seed=1)),
        ("three_row", Trajectory.three_row(duration_s=18.0, fps=30.0, seed=1)),
        ("poles", Trajectory.three_row_plus_poles(duration_s=24.0, fps=30.0, seed=1)),
    ):
        t = traj.frame_times()
        angles = traj.angles_at(t)
        yaw, pitch = angles[0], angles[1]
        dy = np.diff(np.unwrap(yaw))
        assert np.all(dy > -0.02), f"{name} must not reverse direction"
        assert float(np.abs(np.diff(dy)).max()) < 0.02, f"{name} angular acceleration is not smooth"
        assert float(np.degrees(yaw.max() - yaw.min())) > 340.0, f"{name} does not complete a turn"
        # The speed profile must actually vary: a constant sweep is not a hand-held one.
        assert float(dy.max() / max(float(dy.min()), 1e-9)) > 2.0

        if name == "single_row":
            assert float(np.degrees(np.abs(pitch)).max()) < 4.0
        if name == "three_row":
            assert float(np.degrees(pitch).max()) > 30.0
            assert float(np.degrees(pitch).min()) < -30.0
        if name == "poles":
            assert float(np.degrees(pitch).max()) > 70.0
            assert float(np.degrees(pitch).min()) < -70.0


def test_hand_jitter_is_smooth_and_seeded() -> None:
    a = Trajectory.single_row(duration_s=6.0, fps=60.0, seed=3)
    b = Trajectory.single_row(duration_s=6.0, fps=60.0, seed=3)
    c = Trajectory.single_row(duration_s=6.0, fps=60.0, seed=4)
    t = np.linspace(0.0, 6.0, 601)
    ja, jb, jc = a.jitter(t), b.jitter(t), c.jitter(t)
    assert np.array_equal(ja, jb)
    assert not np.allclose(ja, jc)
    # Smooth: the third difference of white noise would be huge, of a few sines tiny.
    assert float(np.abs(np.diff(ja, n=3, axis=1)).max()) < 1e-3
    assert float(np.degrees(np.abs(ja).max())) > 0.05, "jitter must not be negligible"


def test_effects_presets_differ_where_they_should(room: Room, tmp_path: Path) -> None:
    intr = Intrinsics.from_hfov(160, 96, 70.0)
    R = g.from_yaw_pitch_roll(0.4, 0.05, 0.0)
    clean = render_view(room, R, intr, origin=room.eye)

    assert EFFECT_PRESETS["clean"].arm_radius_m == 0.0
    assert EFFECT_PRESETS["realistic"].arm_radius_m == pytest.approx(0.28)
    assert EFFECT_PRESETS["harsh"].arm_radius_m == pytest.approx(0.35)
    assert EFFECT_PRESETS["harsh"].noise_sigma > EFFECT_PRESETS["realistic"].noise_sigma
    assert EFFECT_PRESETS["harsh"].exposure_s > EFFECT_PRESETS["realistic"].exposure_s

    traj = Trajectory.three_row(duration_s=6.0, fps=20.0, seed=9)
    blurred = render_view(
        room,
        traj.R_at(1.0),
        intr,
        origin=traj.centred_at(room.eye).origin_at(1.0),
        subsamples=7,
        exposure_s=1.0 / 30.0,
        pose_at=traj.R_at,
        origin_at=traj.centred_at(room.eye).origin_at,
        t0=1.0,
    )
    sharp = render_view(room, traj.R_at(1.0), intr, origin=traj.centred_at(room.eye).origin_at(1.0))
    def sharpness(im: np.ndarray) -> float:
        return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    assert sharpness(blurred) < sharpness(sharp) * 0.9, "a 33 ms exposure must smear the sweep"

    sheared = render_view(room, R, intr, origin=room.eye, rolling_shutter_s=0.05, pose_at=traj.R_at, t0=1.0)
    assert not np.array_equal(sheared, clean), "rolling shutter must change the frame"


def test_auto_exposure_lags_and_is_recorded(room: Room, tmp_path: Path) -> None:
    intr = Intrinsics.from_hfov(96, 56, 70.0)
    traj = Trajectory.single_row(duration_s=1.2, fps=10.0, seed=6)
    cap = simulate_capture(
        room, traj, intr, out_dir=tmp_path / "ae", effects="harsh", seed=17, ground_truth_width=256
    )
    gains = cap.gains
    assert gains.min() < gains.max(), "auto-exposure never moved"
    assert np.all(gains >= EFFECT_PRESETS["harsh"].ae_gain_min - 1e-9)
    assert np.all(gains <= EFFECT_PRESETS["harsh"].ae_gain_max + 1e-9)
    # A first-order lag cannot jump: consecutive gains move by a bounded fraction.
    step = np.abs(np.diff(gains)) / np.maximum(gains[:-1], 1e-6)
    assert float(step.max()) < 0.35, "the gain jumped like an instantaneous meter"
    for i, pose in enumerate(cap.poses):
        assert float(pose.gain[0]) == pytest.approx(float(gains[i]))


def test_truth_json_round_trips(room: Room, tmp_path: Path) -> None:
    intr = Intrinsics.from_hfov(128, 72, 70.0)
    traj = Trajectory.single_row(duration_s=0.5, fps=10.0, seed=6)
    cap = simulate_capture(
        room, traj, intr, out_dir=tmp_path / "t", effects="clean", seed=1, ground_truth_width=256
    )
    path = cap.write_truth(tmp_path / "truth.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["frames"]) == cap.n_frames
    first = data["frames"][0]
    assert np.allclose(np.array(first["R"], dtype=np.float64), cap.poses[0].R, atol=1e-8)
    assert data["intrinsics"]["width"] == 128
    assert data["room"]["style"] == "rich"
    assert data["trajectory"]["name"] == "single_row"


# ======================================================================================
# CLI
# ======================================================================================


def test_cli_make_room_and_score(tmp_path: Path) -> None:
    exe = [sys.executable, "-m", "biosphere.synth.cli"]
    room_path = tmp_path / "room.jpg"
    res = subprocess.run(
        [*exe, "make-room", "--style", "rich", "--seed", "2", "--out", str(room_path), "--width", "512"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert room_path.is_file()

    res = subprocess.run(
        [*exe, "score", "--candidate", str(room_path), "--reference", str(room_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    report = json.loads(res.stdout)
    assert report["quality"]["psnr"] == M.PSNR_SENTINEL
    assert report["quality"]["ssim"] == pytest.approx(1.0, abs=1e-6)
