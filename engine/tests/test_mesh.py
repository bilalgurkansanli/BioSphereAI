"""Local mesh alignment and the displaced warp it drives."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from biosphere.pano import geometry as g
from biosphere.pano import mesh as mesh_mod
from biosphere.pano import seams as seams_mod
from biosphere.pano import warp as warp_mod
from biosphere.pano.types import CameraPose, Intrinsics, MatchPair, PipelineConfig

W, H = 1024, 512
IMG_W, IMG_H = 480, 640


def _pose(yaw_deg: float, pitch_deg: float, index: int) -> CameraPose:
    intr = Intrinsics.from_hfov(IMG_W, IMG_H, 60.0)
    R = g.from_yaw_pitch_roll(math.radians(yaw_deg), math.radians(pitch_deg), 0.0)
    return CameraPose(frame_index=index, R=R, intrinsics=intr)


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((IMG_H, IMG_W, 3)) * 255).astype(np.float32), (0, 0), 1.5)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_pole_inside_is_geometric_not_a_pitch_guess():
    """A view pitched 40 degrees down with a 60-degree field does not contain the nadir."""
    down = _pose(0.0, -40.0, 0)
    u0, v0, w, h = warp_mod.footprint_window(down, W, H)
    assert w < W, "a footprint that stops 20 degrees short of the pole got the whole canvas"
    straight_down = _pose(0.0, -89.0, 1)
    u0, v0, w, h = warp_mod.footprint_window(straight_down, W, H)
    assert w == W and v0 + h == H


def test_displaced_warp_moves_content_by_the_displacement():
    pose = _pose(0.0, 0.0, 0)
    img = _texture(3)
    cfg = PipelineConfig(interpolation="linear")
    plain = warp_mod.warp_camera(img, pose, W, H, cfg)
    u0, v0, w, h = warp_mod.footprint_window(pose, W, H)
    cell = 32.0
    gw, gh = int(math.ceil(w / cell)) + 1, int(math.ceil(h / cell)) + 1
    grid = np.zeros((gh, gw, 2), np.float32)
    grid[..., 0] = 6.0  # content moves 6 px to the right everywhere
    moved = warp_mod.warp_camera(img, pose, W, H, cfg, displacement=(grid, cell, (u0, v0)))
    # Compare a central block of the plain render with the same block of the moved render
    # shifted back by 6 px, in canvas coordinates.
    cx = plain.corner[0] + plain.width // 2
    cy = plain.corner[1] + plain.height // 2
    a = plain.image[cy - plain.corner[1] - 40 : cy - plain.corner[1] + 40, cx - plain.corner[0] - 40 : cx - plain.corner[0] + 40]
    b = moved.image[cy - moved.corner[1] - 40 : cy - moved.corner[1] + 40, cx + 6 - moved.corner[0] - 40 : cx + 6 - moved.corner[0] + 40]
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()
    assert diff < 3.0, f"displaced render does not match the plain one shifted by 6 px (mean diff {diff:.2f})"


def test_mesh_solve_aligns_a_known_offset_between_two_views():
    """Two views of the same texture, one of them shifted: the meshes meet in the middle."""
    poses = [_pose(-15.0, 0.0, 0), _pose(15.0, 0.0, 1)]
    # Correspondences: project random world rays into both images, then shift view 1's
    # pixels by a constant (parallax-like) 8 px on the canvas.
    rng = np.random.default_rng(7)
    K = poses[0].intrinsics.K
    xs = rng.uniform(20, IMG_W - 20, 400)
    ys = rng.uniform(20, IMG_H - 20, 400)
    rays = g.unproject_pixels(xs, ys, K) @ poses[0].R  # world
    cam1 = rays @ poses[1].R.T
    px, py, ok = g.project_rays(cam1, K)
    ok &= (px > 10) & (px < IMG_W - 10) & (py > 10) & (py < IMG_H - 10)
    pts0 = np.stack([xs[ok], ys[ok]], 1).astype(np.float32)
    pts1 = np.stack([px[ok], py[ok]], 1).astype(np.float32)
    # A canvas-space offset of 8 px at this scale is about 2.8 deg; express it in source
    # pixels of view 1 (rough: source and canvas resolution are similar here).
    pts1_shifted = pts1 + np.array([[8.0, 0.0]], np.float32) * (IMG_W / 60.0) / (W / 360.0)
    pairs = [MatchPair(0, 1, pts0, pts1_shifted, confidence=1.0)]
    cfg = PipelineConfig()
    cell = W * cfg.mesh_cell_fraction
    meshes = mesh_mod.build_meshes(poses, W, H, cell=cell)
    cons = mesh_mod.sparse_constraints(meshes, poses, pairs, W, H)
    assert cons.n > 100
    stats = mesh_mod.solve_meshes(meshes, cons, W, cfg)
    assert stats["residual_before_median_px"] > 5.0
    assert stats["residual_after_median_px"] < 1.5
    # Both views moved, in opposite directions, by roughly half the offset each.
    d0 = meshes[0].d[..., 0]
    d1 = meshes[1].d[..., 0]
    assert d0.max() > 2.0 and d1.min() < -2.0


def test_block_shift_recovers_a_subpixel_shift():
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur((rng.random((96, 96)) * 255).astype(np.float32), (0, 0), 1.2)
    M = np.float32([[1, 0, 2.5], [0, 1, -1.5]])
    moved = cv2.warpAffine(base, M, (96, 96), flags=cv2.INTER_LINEAR)
    a = base[32:64, 32:64]
    margin = 8
    b = moved[32 - margin : 64 + margin, 32 - margin : 64 + margin]
    dx, dy, resp = mesh_mod._block_shift(a, b, margin)
    assert resp > 0.8
    assert abs(dx - 2.5) < 0.5 and abs(dy + 1.5) < 0.5


def test_seam_budget_uses_the_largest_pair_for_memory():
    """Many small overlaps must not be refused just because they sum to a large number."""
    total, largest = 50_000_000, 1_000_000

    class _P:
        def __init__(self, x, y, w, h):
            self.corner = (x, y)
            self.width = w
            self.height = h

    patches = [_P(i * 90, 0, 100, 100) for i in range(60)]
    t, big = seams_mod._pairwise_overlap_px(patches, 1.0)
    assert big == 1000
    scale, ok = seams_mod._fit_seam_scale(patches, 6000, 1.0)
    assert ok and scale == 1.0
