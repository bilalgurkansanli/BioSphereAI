"""Round-trip tests for the shared geometry core.

Every other stage of the engine trusts these conventions, so they are pinned here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biosphere.pano import geometry as g
from biosphere.pano.types import Intrinsics


def test_equirect_ray_roundtrip():
    w, h = 512, 256
    theta, phi = g.equirect_angles(w, h)
    rays = g.rays_from_equirect(theta, phi)
    u, v = g.equirect_from_rays(rays, w, h)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    assert np.abs(u - gx).max() < 1e-2
    assert np.abs(v - gy).max() < 1e-2


def test_equirect_landmarks():
    """Row 0 is the zenith, the centre column looks along +Z, and +Y is down."""
    w, h = 360, 180
    theta, phi = g.equirect_angles(w, h)
    rays = g.rays_from_equirect(theta, phi)
    zenith = rays[0, w // 2]
    nadir = rays[-1, w // 2]
    centre = rays[h // 2, w // 2]
    assert zenith[1] < -0.99, "top row must point up (world -Y)"
    assert nadir[1] > 0.99, "bottom row must point down (world +Y)"
    assert centre[2] > 0.99, "centre column must look along +Z"
    left = rays[h // 2, w // 4]
    assert left[0] < -0.99, "left of centre is world -X"


def test_rays_are_unit():
    theta, phi = g.equirect_angles(64, 32)
    rays = g.rays_from_equirect(theta, phi)
    n = np.linalg.norm(rays, axis=-1)
    assert np.abs(n - 1.0).max() < 1e-5


def test_pixel_ray_roundtrip_no_distortion():
    intr = Intrinsics.from_hfov(1920, 1080, 68.0)
    rays = g.camera_rays(intr.width, intr.height, intr.K)
    x, y, valid = g.project_rays(rays, intr.K)
    assert valid.all()
    gx, gy = g.pixel_grid(intr.width, intr.height)
    assert np.abs(x - gx).max() < 1e-2
    assert np.abs(y - gy).max() < 1e-2


def test_pixel_ray_roundtrip_with_distortion():
    intr = Intrinsics.from_hfov(1280, 720, 70.0)
    intr.k1, intr.k2 = -0.12, 0.03
    gx, gy = g.pixel_grid(intr.width, intr.height)
    rays = g.unproject_pixels(gx, gy, intr.K, intr.k1, intr.k2)
    x, y, _ = g.project_rays(rays, intr.K, intr.k1, intr.k2)
    assert np.abs(x - gx).max() < 0.05
    assert np.abs(y - gy).max() < 0.05


def test_pixel_conventions_differ_by_exactly_half_a_pixel():
    """`project_rays` speaks continuous coordinates; `equirect_from_rays` speaks remap indices.

    Both are correct for their consumer, and the gap between them is exactly 0.5 px. This
    pins that fact so nobody "fixes" one of them into agreeing with the other and shifts
    every warped patch by half a pixel in the process.
    """
    w, h = 256, 128

    # equirect_from_rays round-trips to integer indices...
    theta, phi = g.equirect_angles(w, h)
    u, _ = g.equirect_from_rays(g.rays_from_equirect(theta, phi), w, h)
    assert abs(u[0, 0].item()) < 1e-2, "column 0's ray must map back to index 0"

    # ...while project_rays round-trips to continuous coordinates.
    intr = Intrinsics.from_hfov(w, h, 60.0)
    ray_centre = g.unproject_pixels(
        np.array([intr.cx], dtype=np.float64), np.array([intr.cy], dtype=np.float64), intr.K
    )
    x, y, _ = g.project_rays(ray_centre, intr.K)
    assert float(x[0]) == pytest.approx(w / 2.0, abs=1e-9)

    rx, ry = g.to_remap_coords(x, y)
    assert float(rx[0]) == pytest.approx(w / 2.0 - 0.5, abs=1e-9)
    bx, by = g.from_remap_coords(rx, ry)
    assert float(bx[0]) == pytest.approx(float(x[0]), abs=1e-12)
    assert float(by[0]) == pytest.approx(float(y[0]), abs=1e-12)


def test_remap_conversion_lands_on_a_pixel_centre():
    """A ray through a pixel's centre must sample that pixel exactly, with no blend."""
    img = np.zeros((16, 16, 3), np.uint8)
    img[7, 5] = (10, 120, 240)
    intr = Intrinsics.from_hfov(16, 16, 60.0)

    # The continuous coordinate of pixel (5, 7)'s centre is (5.5, 7.5).
    ray = g.unproject_pixels(np.array([[5.5]]), np.array([[7.5]]), intr.K)
    x, y, _ = g.project_rays(ray, intr.K)
    mx, my = g.to_remap_coords(x.astype(np.float32), y.astype(np.float32))

    import cv2 as _cv2

    sampled = _cv2.remap(img, mx, my, _cv2.INTER_LINEAR)
    assert tuple(int(v) for v in sampled[0, 0]) == (10, 120, 240), (
        "converted coordinates must hit the pixel dead centre; a half-pixel error here "
        "blends the target with its neighbours and softens every warped patch"
    )


def test_projection_preserves_dtype():
    """Rendering wants float32 for memory; bundle adjustment needs float64 for its Jacobian.

    Silently downcasting to float32 quantises a 1000 px coordinate at ~1e-4 px, which is
    enough to corrupt a finite-difference Jacobian and collapse the trust region on a
    level single-row sweep. Both dtypes must survive a round trip.
    """
    intr = Intrinsics.from_hfov(1920, 1080, 68.0)
    for dtype in (np.float32, np.float64):
        x = np.array([[100.0, 900.0]], dtype=dtype)
        y = np.array([[50.0, 700.0]], dtype=dtype)
        rays = g.unproject_pixels(x, y, intr.K)
        assert rays.dtype == dtype
        px, py, valid = g.project_rays(rays, intr.K)
        assert px.dtype == dtype and py.dtype == dtype
        assert valid.all()
        tol = 1e-2 if dtype is np.float32 else 1e-9
        assert np.abs(px - x).max() < tol
        assert np.abs(py - y).max() < tol


def test_float64_projection_is_exact_enough_for_jacobians():
    """A one-thousandth-pixel perturbation must produce a proportional, resolvable change."""
    intr = Intrinsics.from_hfov(1920, 1080, 68.0)
    base = np.array([[1234.0]], dtype=np.float64)
    row = np.array([[567.0]], dtype=np.float64)
    r0 = g.unproject_pixels(base, row, intr.K)
    r1 = g.unproject_pixels(base + 1e-3, row, intr.K)
    x0, _, _ = g.project_rays(r0, intr.K)
    x1, _, _ = g.project_rays(r1, intr.K)
    assert abs((x1 - x0).item() - 1e-3) < 1e-9


def test_hfov_matches_intrinsics():
    intr = Intrinsics.from_hfov(1000, 500, 65.0)
    assert intr.hfov_deg == pytest.approx(65.0, abs=1e-6)


@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [(0.0, 0.0, 0.0), (0.7, 0.0, 0.0), (-2.3, 0.31, 0.0), (1.1, -0.4, 0.25), (3.0, 0.9, -0.6)],
)
def test_yaw_pitch_roll_roundtrip(yaw, pitch, roll):
    R = g.from_yaw_pitch_roll(yaw, pitch, roll)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    y2, p2, r2 = g.yaw_pitch_roll(R)
    assert g.wrap_pi(y2 - yaw) == pytest.approx(0.0, abs=1e-6)
    assert p2 == pytest.approx(pitch, abs=1e-6)
    assert g.wrap_pi(r2 - roll) == pytest.approx(0.0, abs=1e-6)


def test_yaw_direction_is_rightward():
    """A positive yaw must move the image content to the *left* on the canvas."""
    R = g.from_yaw_pitch_roll(math.radians(30.0), 0.0, 0.0)
    forward_world = R.T @ np.array([0.0, 0.0, 1.0])
    theta = math.atan2(forward_world[0], forward_world[2])
    assert math.degrees(theta) == pytest.approx(30.0, abs=1e-6)


def test_homography_rotation_roundtrip():
    intr = Intrinsics.from_hfov(1920, 1080, 68.0)
    R = g.from_yaw_pitch_roll(0.35, -0.08, 0.05)
    H = g.homography_from_rotation(R, intr.K, intr.K)
    R2 = g.rotation_from_homography(H, intr.K, intr.K)
    assert math.degrees(g.angular_distance(R, R2)) < 1e-4


def test_nearest_rotation_projects_noise():
    R = g.from_yaw_pitch_roll(0.4, 0.2, -0.1)
    noisy = R + np.random.default_rng(0).normal(0.0, 0.01, (3, 3))
    R2 = g.nearest_rotation(noisy)
    assert np.linalg.det(R2) == pytest.approx(1.0, abs=1e-9)
    assert math.degrees(g.angular_distance(R, R2)) < 2.0


def test_align_horizon_removes_tilt():
    """A sweep taken with the phone tilted must be levelled back onto the equator."""
    tilt = g.rodrigues(np.array([0.25, 0.0, 0.13]))  # arbitrary constant world tilt
    poses = [g.from_yaw_pitch_roll(a, 0.0, 0.0) @ tilt for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    C = g.align_horizon(poses)
    levelled = [R @ C.T for R in poses]
    pitches = [abs(math.degrees(g.yaw_pitch_roll(R)[1])) for R in levelled]
    assert max(pitches) < 1.0, f"horizon still tilted: max pitch {max(pitches):.3f} deg"


def test_align_horizon_is_identity_when_already_level():
    poses = [g.from_yaw_pitch_roll(a, 0.0, 0.0) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    C = g.align_horizon(poses)
    assert math.degrees(g.angular_distance(C, np.eye(3))) < 1e-4


def test_sample_equirect_wraps_horizontally():
    w, h = 64, 32
    pano = np.tile(np.arange(w, dtype=np.uint8)[None, :, None], (h, 1, 3))
    u = np.array([[-1.0, 0.0, float(w), float(w) + 1.0]], dtype=np.float32)
    v = np.full_like(u, h // 2)
    out = g.sample_equirect(pano, u, v, interpolation=1)  # INTER_LINEAR
    assert int(out[0, 0, 0]) == pytest.approx(w - 1, abs=1)
    assert int(out[0, 2, 0]) == pytest.approx(0, abs=1)


def test_solid_angle_weights_sum_to_one_and_favour_equator():
    wts = g.sphere_solid_angle_weights(180, 360, dtype=np.float64)
    assert float(wts.sum()) == pytest.approx(1.0, abs=1e-9)
    assert wts[90, 0] > wts[2, 0] * 10.0


def test_fov_footprint_spans_expected_angles():
    intr = Intrinsics.from_hfov(1000, 1000, 60.0)
    y0, y1, p0, p1 = g.fov_footprint_deg(intr, np.eye(3))
    assert y1 - y0 == pytest.approx(60.0, abs=2.0)
    assert p1 - p0 == pytest.approx(60.0, abs=2.0)


def test_fov_footprint_handles_seam_straddle():
    intr = Intrinsics.from_hfov(1000, 1000, 60.0)
    R = g.from_yaw_pitch_roll(math.pi, 0.0, 0.0)  # looking exactly at the seam
    y0, y1, _, _ = g.fov_footprint_deg(intr, R)
    assert y1 - y0 == pytest.approx(60.0, abs=2.0), "seam straddle must not collapse the span"
