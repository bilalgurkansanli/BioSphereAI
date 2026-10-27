"""The walkthrough package's pure parts: PLY, rasteriser, levelling, registration, cleaning,
frame selection and pairing. No video, no GPU needed (the rasteriser runs on the CPU here)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from biosphere.splat import features, frames, level, ply, raster, register
from biosphere.splat.clean import CameraView, CleanConfig, clean_table, path_box


def _table(n: int = 5, seed: int = 0) -> ply.GaussianTable:
    rng = np.random.default_rng(seed)
    return ply.GaussianTable(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        sh_dc=rng.normal(size=(n, 3)).astype(np.float32),
        sh_rest=rng.normal(size=(n, 3, 3)).astype(np.float32),
        opacity_logit=rng.normal(size=n).astype(np.float32),
        log_scales=rng.normal(size=(n, 3)).astype(np.float32),
        quats=rng.normal(size=(n, 4)).astype(np.float32),
    )


def test_ply_round_trip_is_exact(tmp_path: Path) -> None:
    t = _table(7)
    p = ply.write_ply(tmp_path / "a.ply", t)
    back = ply.read_ply(p)
    for field in ("means", "sh_dc", "sh_rest", "opacity_logit", "log_scales", "quats"):
        assert np.array_equal(getattr(t, field), getattr(back, field)), field
    assert back.sh_degree == 1
    assert back.with_sh_degree(0).sh_rest.shape == (7, 3, 0)


def test_transformed_table_moves_and_scales() -> None:
    t = _table(4)
    R = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    moved = t.transformed(R, np.array([1.0, 2.0, 3.0]), scale=2.0)
    expect = 2.0 * (t.means.astype(np.float64) @ R.T) + np.array([1.0, 2.0, 3.0])
    assert np.allclose(moved.means, expect, atol=1e-5)
    assert np.allclose(moved.scales, t.scales * 2.0, rtol=1e-5)
    # Orientation quaternions stay unit and rotate with the frame.
    assert np.allclose(np.linalg.norm(moved.quats, axis=1), 1.0, atol=1e-5)


def test_rasteriser_draws_one_gaussian_where_it_should() -> None:
    # One opaque white blob two metres in front of a camera at the origin looking +Z.
    t = ply.GaussianTable(
        means=np.array([[0.0, 0.0, 2.0]], np.float32),
        sh_dc=np.array([[(1.0 - 0.5) / ply.SH_C0] * 3], np.float32),
        sh_rest=np.zeros((1, 3, 0), np.float32),
        opacity_logit=np.array([8.0], np.float32),
        log_scales=np.log(np.array([[0.1, 0.1, 0.1]], np.float32)),
        quats=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
    )
    view = raster.PinholeView(R=np.eye(3), t=np.zeros(3), fx=200.0, fy=200.0, cx=64.0, cy=48.0, width=128, height=96)
    rgb, alpha = raster.render(t, view, device=torch.device("cpu"))
    assert rgb.shape == (96, 128, 3)
    assert alpha[48, 64] > 0.9 and rgb[48, 64].mean() > 0.9
    assert alpha[2, 2] < 1e-3
    # Nothing is drawn when the blob is behind the camera.
    behind = ply.GaussianTable(means=-t.means, sh_dc=t.sh_dc, sh_rest=t.sh_rest, opacity_logit=t.opacity_logit, log_scales=t.log_scales, quats=t.quats)
    _, alpha2 = raster.render(behind, view, device=torch.device("cpu"))
    assert alpha2.max() == 0.0


def test_floor_plane_and_scale_from_synthetic_room() -> None:
    rng = np.random.default_rng(1)
    # A tilted world: true up is a rotated -Y; floor is a plane through the origin.
    tilt = Rotation.from_euler("xz", [12.0, -7.0], degrees=True).as_matrix()
    floor = rng.uniform([-3, 0, -3], [3, 0, 3], size=(4000, 3))
    walls = np.concatenate([rng.uniform([-3, -2.6, -3], [-2.95, 0, 3], size=(800, 3)), rng.uniform([2.95, -2.6, -3], [3, 0, 3], size=(800, 3))])
    ceiling = rng.uniform([-3, -2.6, -3], [3, -2.55, 3], size=(1500, 3))
    pts_true = np.concatenate([floor, walls, ceiling])
    cams_true = np.stack([np.linspace(-2, 2, 40), np.full(40, -1.45), np.linspace(-1, 1, 40)], axis=1)
    # An SfM frame: rotated, scaled by 3, shifted.
    s_true = 3.0
    shift = np.array([5.0, -2.0, 1.0])
    to_sfm = lambda p: s_true * (p @ tilt.T) + shift  # noqa: E731
    pts, cams = to_sfm(pts_true), to_sfm(cams_true)
    # Camera rotations: cam_from_world with the camera upright in the TRUE frame, then tilted.
    R_true = Rotation.from_euler("y", np.linspace(0, 300, 40)[:, None], degrees=True).as_matrix()
    rots = np.einsum("nij,jk->nik", R_true, tilt.T)  # cam_from_world in the SfM frame
    lev = level.estimate_leveling(rots, cams, pts, assumed_camera_height_m=1.45)
    assert np.degrees(np.arccos(abs(lev.up @ (tilt @ np.array([0, -1.0, 0]))))) < 1.0
    assert abs(lev.scale * s_true - 1.0) < 0.03  # source units -> metres
    levelled_cams = level.apply_sim3(cams, lev.R, lev.t, lev.scale)
    assert np.allclose(levelled_cams[:, 1], -1.45, atol=0.05)
    levelled_floor = level.apply_sim3(pts[:4000], lev.R, lev.t, lev.scale)
    assert abs(float(np.median(levelled_floor[:, 1]))) < 0.05


def test_sim3_from_poses_survives_outliers() -> None:
    rng = np.random.default_rng(0)
    n = 24
    src_c = np.cumsum(rng.normal(size=(n, 3)) * 0.3, axis=0)
    src_R = Rotation.random(n, random_state=1).as_matrix()
    Rt = Rotation.from_euler("xyz", [5, 40, -3], degrees=True).as_matrix()
    s, t = 2.3, np.array([1.0, -2.0, 0.5])
    tgt_c = s * (src_c @ Rt.T) + t + rng.normal(size=(n, 3)) * 0.01
    tgt_R = np.einsum("nij,jk->nik", src_R, Rt.T)
    tgt_c[3] += 5.0
    tgt_c[10] -= 4.0
    sim, mask = register.sim3_from_poses(src_R, src_c, tgt_R, tgt_c)
    assert abs(sim.scale - s) < 0.01
    assert np.degrees(np.arccos(np.clip((np.trace(sim.R.T @ Rt) - 1) / 2, -1, 1))) < 0.5
    assert np.allclose(sim.t, t, atol=0.03)
    assert mask.sum() == n - 2


def test_sim3_from_poses_never_returns_a_reflection() -> None:
    # Shared frames filmed from (almost) one spot: their centres are noise, and the least-
    # squares scale of noise is as likely negative as positive. The first real capture
    # "registered" a part at scale -0.52 this way; a similarity must keep a positive scale.
    rng = np.random.default_rng(4)
    for trial in range(20):
        n = 8
        src_c = rng.normal(size=(n, 3)) * 0.01
        src_R = Rotation.random(n, random_state=trial).as_matrix()
        tgt_c = rng.normal(size=(n, 3)) * 1.0
        sim, _ = register.sim3_from_poses(src_R, src_c, src_R, tgt_c)
        assert sim.scale > 0


def test_clean_drops_transparent_giant_and_far_splats() -> None:
    cams = np.array([[0.0, -1.5, 0.0], [2.0, -1.5, 1.0]])
    t = ply.GaussianTable(
        means=np.array([[1, -1, 0.5], [1, -1, 0.5], [1, -1, 0.5], [40, -1, 0.5]], np.float32),
        sh_dc=np.zeros((4, 3), np.float32),
        sh_rest=np.zeros((4, 3, 0), np.float32),
        opacity_logit=np.array([3.0, -6.0, 3.0, 3.0], np.float32),
        log_scales=np.log(np.array([[0.1] * 3, [0.1] * 3, [4.0, 0.1, 0.1], [0.1] * 3], np.float32)),
        quats=np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (4, 1)),
    )
    out, rep = clean_table(t, cams, config=CleanConfig())
    assert out.count == 1 and rep.n_transparent == 1 and rep.n_giant == 1 and rep.n_outside == 1
    lo, hi = path_box(cams, CleanConfig())
    assert lo[1] < -1.5 - 3.0 and hi[1] > 0.3  # reaches below the floor at y = 0


def test_frame_selection_keeps_the_sharpest_and_drops_repeats() -> None:
    n = 100
    sharp = np.arange(n, dtype=np.float64) % 10  # every 10th window has a clear best (index 9 mod 10)
    thumbs = np.zeros((n, 24, 32), np.float32)
    thumbs[50:] = 100.0  # the second half is a different view; first half is one still shot
    cfg = frames.FrameConfig(keep_per_second=1.0, min_frames=1, max_frames=20, duplicate_threshold=2.5)
    keep = frames.select_frames(sharp, thumbs, duration_s=20.0, config=cfg)
    # Each still run collapses to ONE keeper, its sharpest member (the first 9-sharp frame).
    assert keep == [9, 59]

    # A moving camera (every candidate distinct) keeps the sharpest of each window, and a
    # pause in the middle does not steal windows from the frames where the camera moved.
    thumbs2 = np.arange(n, dtype=np.float32)[:, None, None] * 10 * np.ones((1, 24, 32), np.float32)
    thumbs2[40:60] = thumbs2[40]  # a 20-frame pause
    keep2 = frames.select_frames(sharp, thumbs2, duration_s=20.0, config=cfg)
    assert len(keep2) == 20
    assert sum(1 for i in keep2 if 40 <= i < 60) == 1
    assert keep2[0] == 3  # sharpest of the first window, [0, 4) of the 80 distinct frames


def test_pairs_are_sequential_then_retrieved() -> None:
    p = features.sequential_pairs(6, 2, quadratic=True)
    assert (0, 1) in p and (0, 2) in p and (0, 4) in p and (0, 3) not in p
    g = torch.eye(6)
    g[5] = g[0]  # frame 5 looks exactly like frame 0
    seq = np.zeros(6, dtype=np.int64)
    idx = np.arange(6)
    r = features.retrieval_pairs(g, seq, idx, topk=1, min_gap=3, min_similarity=0.5)
    assert r == [(0, 5)]
    # Across sequences the gap rule does not apply.
    seq2 = np.array([0, 0, 0, 1, 1, 1])
    g2 = torch.eye(6)
    g2[3] = g2[2]
    r2 = features.retrieval_pairs(g2, seq2, np.array([0, 1, 2, 0, 1, 2]), topk=1, min_gap=3, min_similarity=0.5)
    assert (2, 3) in r2


@pytest.mark.parametrize("width", [64, 128])
def test_equirect_render_has_the_expected_shape(width: int) -> None:
    t = _table(3)
    pano = raster.render_equirect(t, np.zeros(3), width=width, face_size=32, device=torch.device("cpu"))
    assert pano.shape == (width // 2, width, 3)


def test_manifest_marks_an_unregistered_room_as_unplaced(tmp_path: Path) -> None:
    from biosphere.splat.bundle import manifest_dict
    from biosphere.splat.types import AreaResult, PropertySplatResult

    def area(key: str, placed: bool) -> AreaResult:
        sog = tmp_path / f"{key}.sog"
        sog.write_bytes(b"x")
        return AreaResult(key=key, name=key, floor_index=0, sog_path=sog, splat_count=1, sog_bytes=1, placed=placed)

    result = PropertySplatResult(
        out_dir=tmp_path,
        base=area("base", True),
        rooms=[area("kitchen", True), area("attic", False)],
        floor_plan_path=None,
        floor_plan={"hotspots": []},
        manifest_path=tmp_path / "tour.json",
        metrics={},
        warnings=[],
    )
    doc = manifest_dict(result)
    assert doc["base"]["placed"] is True
    assert [r["placed"] for r in doc["rooms"]] == [True, False]


def test_unreliable_cameras_flags_starved_and_flung_frames() -> None:
    from types import SimpleNamespace

    from biosphere.splat.sfm import unreliable_cameras

    # A walk of 12 frames, 0.3 apart; frame 5 sees almost nothing, frame 9 is flung 5 units off.
    images = []
    for i in range(12):
        centre = np.array([0.3 * i, 0.0, 0.0])
        n_points = 300
        if i == 5:
            n_points = 12
        if i == 9:
            centre = centre + np.array([0.0, 5.0, 0.0])
        images.append(
            SimpleNamespace(
                name=f"base/images/f_{i:05d}.jpg",
                has_pose=True,
                num_points3D=n_points,
                projection_center=(lambda c: (lambda: c))(centre),
            )
        )
    rec = SimpleNamespace(images={i: img for i, img in enumerate(images)})
    flagged = unreliable_cameras(rec, min_points=60, jump_factor=8.0)
    assert flagged == ["base/images/f_00005.jpg", "base/images/f_00009.jpg"]
    # Fewer than 8 frames: nothing is judged (no meaningful median step).
    small = SimpleNamespace(images={i: img for i, img in enumerate(images[:6])})
    assert unreliable_cameras(small) == []


def test_segments_from_breaks_cuts_and_drops_short_runs() -> None:
    from biosphere.splat.chain import _video_at, segments_from_breaks

    kept, dropped = segments_from_breaks(20, breaks=[2, 9, 14], min_frames=5)
    # ranges: [0,3) [3,10) [10,15) [15,20): the first is too short
    assert kept == [(3, 10), (10, 15), (15, 20)]
    assert dropped == [(0, 3)]
    assert segments_from_breaks(20, breaks=[], min_frames=5) == ([(0, 20)], [])
    assert segments_from_breaks(20, breaks=[19, 25], min_frames=5) == ([(0, 20)], [])  # a break past the last link is ignored
    # The shared timeline of two recordings maps back to the file and the second within it.
    assert _video_at(5.0, [10.0, 8.0]) == (0, 5.0)
    assert _video_at(12.5, [10.0, 8.0]) == (1, 2.5)
    assert _video_at(18.0, [10.0, 8.0]) is None


def test_split_at_jumps_paces_by_time_not_by_frame_index() -> None:
    """A bridged stretch is 30 fps and the rest is 3 fps; only a real seam may cut."""
    from types import SimpleNamespace

    from biosphere.splat.sfm import split_at_jumps

    images, times = [], {}
    x, t = 0.0, 0.0
    for i in range(50):
        # Frames 10..39 are a bridge: ten times the rate, a tenth of the step, same speed -
        # and the majority of the model, which is what makes an index-paced median wrong.
        dt = 1 / 30 if 10 <= i < 40 else 1 / 3
        x += 0.5 * dt  # a steady half a metre a second
        t += dt
        name = f"base/images/f_{i:05d}.jpg"
        times[name] = t
        images.append(
            SimpleNamespace(name=name, has_pose=True, num_points3D=300, projection_center=(lambda v: (lambda: v))(np.array([x, 0.0, 0.0])))
        )
    # ...and one real seam: from frame 42 the walk continues 8 m away.
    for j in range(42, 50):
        far = np.array([8.0 + 0.5 / 3 * (j - 42), 0.0, 0.0])
        images[j].projection_center = (lambda v: (lambda: v))(far)
    rec = SimpleNamespace(images=dict(enumerate(images)))
    own = set(times)

    with_time, _ = split_at_jumps(rec, own, jump_factor=10.0, min_frames=5, times=times)
    assert sorted(len(c) for c in with_time) == [8, 42]

    # Without the timestamps the same walk looks like a chain of jumps: every 3 fps step is
    # ten bridge-steps' worth of distance, so the model is minced and most of it falls under
    # the minimum length and is thrown away - which is what shrank a 180-frame walk to 69.
    by_index, dropped = split_at_jumps(rec, own, jump_factor=10.0, min_frames=5, times=None)
    assert sum(len(c) for c in by_index) < 40
    assert len(dropped) > 10


def test_clean_widens_needles_without_shrinking_them() -> None:
    """A needle keeps its length and its area; only its thin axes grow."""
    t = _table(3)
    t.means = np.zeros((3, 3), np.float32)
    t.opacity_logit = np.full(3, 3.0, np.float32)
    t.log_scales = np.log(
        np.array([[0.40, 0.02, 0.01], [0.05, 0.04, 0.03], [0.30, 0.30, 0.30]], np.float32)
    ).astype(np.float32)
    cams = np.zeros((4, 3), np.float64)
    out, rep = clean_table(t, cams, config=CleanConfig(max_anisotropy=6.0))
    s = out.scales
    assert out.count == 3
    assert rep.n_needles == 1  # only the first is a needle
    # The long axis is untouched; the thin ones are lifted to a sixth of it.
    assert s[0].max() == pytest.approx(0.40, rel=1e-4)
    assert s[0].min() == pytest.approx(0.40 / 6, rel=1e-4)
    # A splat already within the ratio, and a round one, are left exactly as they were.
    assert s[1] == pytest.approx(np.array([0.05, 0.04, 0.03]), rel=1e-4)
    assert s[2] == pytest.approx(np.array([0.30, 0.30, 0.30]), rel=1e-4)

    off, _ = clean_table(t, cams, config=CleanConfig(max_anisotropy=0.0))
    assert off.scales[0] == pytest.approx(np.array([0.40, 0.02, 0.01]), rel=1e-4)


def test_exposure_solver_recovers_gains_it_was_given(tmp_path: Path) -> None:
    """Frames are darkened and brightened by known amounts; the solver must undo exactly that."""
    from types import SimpleNamespace

    import cv2

    from biosphere.splat.exposure import estimate_gains

    rng = np.random.default_rng(7)
    n_images, n_points = 8, 300
    # One true colour per point, and where each point lands in each image.
    truth = rng.uniform(0.25, 0.75, n_points)
    xy = rng.uniform(4, 60, size=(n_images, n_points, 2))
    applied = np.exp(rng.uniform(-0.35, 0.35, n_images))  # what the camera's metering did
    applied /= np.exp(np.mean(np.log(applied)))  # ...up to the overall brightness, which is not recoverable

    images = {}
    for i in range(n_images):
        frame = np.full((64, 64), 0.5, dtype=np.float64)
        for p in range(n_points):
            x, y = int(round(xy[i, p, 0])), int(round(xy[i, p, 1]))
            frame[y - 1 : y + 2, x - 1 : x + 2] = truth[p]
        frame = np.clip(frame * applied[i], 0, 1)
        path = tmp_path / f"f_{i:03d}.jpg"
        cv2.imwrite(str(path), (frame * 255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])
        images[i + 1] = SimpleNamespace(
            name=path.name,
            has_pose=True,
            points2D=[SimpleNamespace(xy=xy[i, p]) for p in range(n_points)],
        )
    points3D = {
        p: SimpleNamespace(track=SimpleNamespace(length=lambda: n_images, elements=[SimpleNamespace(image_id=i + 1, point2D_idx=p) for i in range(n_images)]))
        for p in range(n_points)
    }
    rec = SimpleNamespace(images=images, points3D=points3D)

    gains, report = estimate_gains(rec, tmp_path, min_track_length=2)
    assert report.n_images == n_images
    recovered = np.array([gains[f"f_{i:03d}.jpg"] for i in range(n_images)])
    # The correction is the inverse of what the camera applied, to within JPEG rounding.
    assert recovered == pytest.approx(1.0 / applied, rel=0.03)


def test_exposure_gain_compresses_highlights_instead_of_clipping(tmp_path: Path) -> None:
    import cv2

    from biosphere.splat.exposure import apply_gains

    # A ramp from black to white, brightened by half again: the mid-tones must scale exactly
    # and the top must stay below white instead of flattening into it.
    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (8, 1))
    path = tmp_path / "ramp.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert apply_gains(tmp_path, {"ramp.jpg": 1.5}) == 1
    out = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)[4]

    assert out[64] == pytest.approx(64 * 1.5, abs=4)  # a mid-tone is exactly 1.5x
    assert out[255] < 255  # ...and the brightest pixel did not become a flat white
    assert np.all(np.diff(out.astype(int)) >= 0)  # the ramp is still a ramp: nothing is lost
    plain = np.clip(ramp[4].astype(np.float32) * 1.5, 0, 255)
    assert (plain >= 255).sum() > 80  # what a plain multiply would have flattened


def _wall(centre: tuple[float, float, float], *, n: int = 400, size: float = 3.0, seed: int = 1) -> ply.GaussianTable:
    """A dense patch of opaque splats on a plane of constant z, filling the view from nearby."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-size / 2, size / 2, size=(n, 2))
    means = np.stack([xy[:, 0] + centre[0], xy[:, 1] + centre[1], np.full(n, centre[2])], axis=1)
    return ply.GaussianTable(
        means=means.astype(np.float32),
        sh_dc=np.full((n, 3), 1.0 / ply.SH_C0, np.float32),
        sh_rest=np.zeros((n, 3, 0), np.float32),
        opacity_logit=np.full(n, 8.0, np.float32),
        log_scales=np.log(np.full((n, 3), 0.2, np.float32)),
        quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (n, 1)),
    )


def test_frame_coverage_sees_a_wall_and_not_an_empty_room() -> None:
    from biosphere.splat.coverage import frame_coverage

    poses = [(np.zeros(3), np.array([0.0, 0.0, 1.0]))]
    device = torch.device("cpu")
    near = _wall((0.0, 0.0, 2.0))
    far_behind = _wall((0.0, 0.0, -40.0))
    assert frame_coverage(near, poses, width=64, device=device) > 0.5
    # Nothing in front of the camera is exactly the case that made a swap look like a
    # teleport: the incoming cloud rendered black.
    assert frame_coverage(far_behind, poses, width=64, device=device) < 0.01


def test_redundant_areas_flags_only_what_the_base_already_covers() -> None:
    from biosphere.splat.coverage import redundant_areas

    tables = {
        "base": _wall((0.0, 0.0, 2.0), n=600),
        "same": _wall((0.0, 0.0, 2.0), n=600, seed=2),   # films the wall the base already has
        "beyond": _wall((0.0, 0.0, 42.0), n=600, seed=3),  # a room the base never reached
    }
    cameras = {
        "base": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
        "same": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
        "beyond": [[0.0, 0.0, 40.0], [0.0, 0.0, 40.2]],
    }
    rep = redundant_areas(tables, cameras, "base", samples=2, fan=0.0, device=torch.device("cpu"))
    assert rep.redundant == ["same"]
    assert rep.own["beyond"] > rep.base["beyond"]


def test_off_floor_cameras_are_left_out_of_the_route() -> None:
    from biosphere.splat.pipeline import BBOX_BELOW_FLOOR_M, BBOX_CEILING_M, _walkable

    walk = np.array([[float(i), -1.5, 0.0] for i in range(20)])
    sunk = np.array([[float(i), 8.0, 0.0] for i in range(6)])  # +y is DOWN: eight metres under
    kept, dropped = _walkable(np.concatenate([walk, sunk]))
    assert dropped == 6
    assert len(kept) == 20
    assert kept[:, 1].max() <= BBOX_BELOW_FLOOR_M and kept[:, 1].min() >= -BBOX_CEILING_M


def test_a_mis_levelled_area_keeps_its_whole_route() -> None:
    from biosphere.splat.pipeline import _walkable

    # Most of the walk outside the band is not a handful of bad poses, it is an area sitting
    # at the wrong height. Dropping it would leave the viewer nothing to choose along.
    sunk = np.array([[float(i), 5.0, 0.0] for i in range(15)])
    fine = np.array([[float(i), -1.5, 0.0] for i in range(5)])
    kept, dropped = _walkable(np.concatenate([sunk, fine]))
    assert dropped == 0
    assert len(kept) == 20


def _view_at(centre, *, width=640, height=480, f=500.0) -> object:
    """A camera at ``centre`` looking down +Z, in the property frame."""
    from biosphere.splat.clean import CameraView

    R = np.eye(3)
    return CameraView(R=R, t=-R @ np.asarray(centre, dtype=np.float64),
                      K=np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]]), width=width, height=height)


def test_visibility_keeps_what_a_camera_saw_however_far_away() -> None:
    from biosphere.splat.clean import CleanConfig, clean_table

    # Straight ahead at 2 m, straight ahead at 40 m (the street through the window), and one
    # off to the side that no camera ever framed.
    means = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 40.0], [50.0, 0.0, 2.0]], np.float32)
    t = ply.GaussianTable(
        means=means, sh_dc=np.zeros((3, 3), np.float32), sh_rest=np.zeros((3, 3, 0), np.float32),
        opacity_logit=np.full(3, 3.0, np.float32), log_scales=np.log(np.full((3, 3), 0.1, np.float32)),
        quats=np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (3, 1)),
    )
    views = [_view_at([0.0, 0.0, 0.0]), _view_at([0.0, 0.0, -0.3])]
    out, rep = clean_table(t, np.array([[0.0, 0.0, 0.0]]), config=CleanConfig(), views=views)
    assert out.count == 2 and rep.n_outside == 1
    # The distant one survives: it is in both photographs, and a box around a camera that
    # never moved would have thrown it away.
    assert float(out.means[:, 2].max()) > 30


def test_without_poses_cleaning_still_falls_back_to_the_box() -> None:
    from biosphere.splat.clean import CleanConfig, clean_table

    means = np.array([[0.0, -1.0, 0.0], [90.0, -1.0, 0.0]], np.float32)
    t = ply.GaussianTable(
        means=means, sh_dc=np.zeros((2, 3), np.float32), sh_rest=np.zeros((2, 3, 0), np.float32),
        opacity_logit=np.full(2, 3.0, np.float32), log_scales=np.log(np.full((2, 3), 0.1, np.float32)),
        quats=np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (2, 1)),
    )
    out, rep = clean_table(t, np.array([[0.0, -1.5, 0.0], [1.0, -1.5, 0.0]]), config=CleanConfig())
    assert out.count == 1 and rep.n_outside == 1


def test_dense_fusion_keeps_a_wall_that_two_views_agree_on(tmp_path: Path) -> None:
    from biosphere.splat.densify import DenseInitConfig, fuse_depths, write_point_ply

    # Two pinhole cameras 0.3 m apart, both facing a wall 2 m ahead (+Z), and a third whose
    # depth map disagrees (a floater at 1 m): only what the views agree on becomes the cloud.
    w, h = 40, 30
    K = np.array([[30.0, 0, w / 2], [0, 30.0, h / 2], [0, 0, 1]])

    def view(x: float, depth: float) -> dict:
        E = np.eye(4)
        E[0, 3] = -x  # world -> camera: the camera sits at (x, 0, 0)
        return {"depth": np.full((h, w), depth, np.float32), "conf": np.ones((h, w), np.float32), "K": K, "E": E,
                "rgb": np.full((h, w, 3), 128, np.uint8)}

    preds = {"a": view(0.0, 2.0), "b": view(0.3, 2.0), "c": view(0.15, 1.0)}
    cfg = DenseInitConfig(conf_percentile=0.0, n_check=2, min_agree=1, voxel_m=0.0, stride=1)
    xyz, rgb, counts = fuse_depths(preds, cfg)
    assert counts["candidates"] == 3 * w * h
    assert len(xyz) > 0
    assert np.allclose(xyz[:, 2], 2.0, atol=1e-4)  # nothing from the view that saw a floater
    out = write_point_ply(tmp_path / "init.ply", xyz, rgb)
    head = out.read_bytes()[:200].decode("ascii", "replace")
    assert f"element vertex {len(xyz)}" in head and "property uchar red" in head


def test_dense_init_refuses_non_commercial_checkpoints() -> None:
    from biosphere.splat.densify import load_model

    with pytest.raises(ValueError):
        load_model("DA3-GIANT", "cpu")


def test_clean_keeps_a_giant_far_from_the_walk() -> None:
    # A 4 m splat beside the walk is a smear across the room; the same splat 10 m out is the
    # street through the window, and cutting those cost 0.67 dB on a real capture.
    cams = np.array([[0.0, -1.5, 0.0], [1.0, -1.5, 0.0]])
    t = ply.GaussianTable(
        means=np.array([[0.5, -1.0, 1.0], [0.5, -1.0, 10.0]], np.float32),
        sh_dc=np.zeros((2, 3), np.float32),
        sh_rest=np.zeros((2, 3, 0), np.float32),
        opacity_logit=np.array([0.0, 0.0], np.float32),
        log_scales=np.log(np.array([[4.0, 0.1, 0.1], [4.0, 0.1, 0.1]], np.float32)),
        quats=np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (2, 1)),
    )
    view = CameraView(R=np.eye(3), t=np.array([0.0, 1.5, 0.0]), K=np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]]), width=100, height=100)
    out, rep = clean_table(t, cams, config=CleanConfig(min_views=1), views=[view])
    assert rep.n_giant == 1 and rep.n_giant_far_kept == 1
    assert out.count == 1 and float(out.means[0, 2]) == 10.0
    _, rep_all = clean_table(t, cams, config=CleanConfig(min_views=1, giant_near_m=0.0), views=[view])
    assert rep_all.n_giant == 2


def test_manifest_cameras_carry_where_they_looked_and_when(tmp_path: Path) -> None:
    # The API's manifest schema is closed, so the look direction and the filming time ride on
    # each camera entry: [x, y, z, lx, ly, lz, t]. Readers that take three numbers still work.
    from biosphere.api.schemas import WalkManifest
    from biosphere.splat.bundle import manifest_dict
    from biosphere.splat.types import AreaResult, PropertySplatResult

    sog = tmp_path / "base.sog"
    sog.write_bytes(b"x")
    base = AreaResult(key="base", name="base", floor_index=0, sog_path=sog, splat_count=1, sog_bytes=1,
                      cameras=[[0.0, -1.4, 0.0], [1.0, -1.4, 0.0]], looks=[[0.0, 0.0, 1.0, 0.5], [1.0, 0.0, 0.0, 0.83]],
                      entry_position=[0.0, -1.4, 0.0], entry_forward=[0.0, 0.0, 1.0], bbox_min=[-1, -2.7, -1], bbox_max=[2, 0.3, 1])
    result = PropertySplatResult(out_dir=tmp_path, base=base, rooms=[], floor_plan_path=None, floor_plan={"hotspots": []},
                                 manifest_path=tmp_path / "tour.json", metrics={}, warnings=[])
    doc = manifest_dict(result)
    assert doc["base"]["cameras"] == [[0.0, -1.4, 0.0, 0.0, 0.0, 1.0, 0.5], [1.0, -1.4, 0.0, 1.0, 0.0, 0.0, 0.83]]
    WalkManifest(**doc)  # the served schema accepts it


def test_turn_speeds_reads_degrees_per_second_and_skips_bridge_pairs() -> None:
    from types import SimpleNamespace

    from biosphere.splat.sfm import turn_speeds

    def img(name: str, yaw_deg: float):
        R = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
        pose = SimpleNamespace(rotation=SimpleNamespace(matrix=lambda R=R: R))
        return SimpleNamespace(name=name, has_pose=True, cam_from_world=lambda pose=pose: pose)

    # 3 fps keyframes turning 10 deg each (30 deg/s), then a 30-fps bridge pair (skipped).
    frames = {"a": (0.0, 0.0), "b": (1 / 3, 10.0), "c": (2 / 3, 20.0), "d": (2 / 3 + 1 / 30, 21.0)}
    rec = SimpleNamespace(images={k: img(k, y) for k, (_, y) in frames.items()})
    times = {k: t for k, (t, _) in frames.items()}
    speeds = turn_speeds(rec, None, times)
    assert speeds.shape == (2,)
    assert np.allclose(speeds, 30.0, atol=1e-6)
