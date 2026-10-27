"""Spherical / rotational geometry shared by every stage of the engine.

This module is the *single* place where the equirectangular parameterisation and the
world<->camera convention are implemented. The stitcher, the synthetic capture
simulator and the quality metrics all import from here, which is what guarantees that
a panorama rendered by the simulator and a panorama produced by the stitcher are
directly comparable pixel-for-pixel.

See :mod:`biosphere.pano.types` for the convention summary.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

__all__ = [
    "wrap_pi",
    "to_remap_coords",
    "from_remap_coords",
    "equirect_angles",
    "rays_from_equirect",
    "equirect_from_rays",
    "sample_equirect",
    "unproject_pixels",
    "project_rays",
    "pixel_grid",
    "camera_rays",
    "rodrigues",
    "rotvec",
    "nearest_rotation",
    "angular_distance",
    "rotation_from_homography",
    "homography_from_rotation",
    "yaw_pitch_roll",
    "from_yaw_pitch_roll",
    "estimate_gravity_axis",
    "align_horizon",
    "sphere_solid_angle_weights",
    "fov_footprint_deg",
]

_EPS = 1e-12


# ======================================================================================
# Angle helpers
# ======================================================================================


def wrap_pi(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles into (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


# ======================================================================================
# Pixel indexing conventions — read this before wiring anything into cv2.remap
# ======================================================================================
#
# Two conventions coexist and confusing them costs sharpness, not correctness, which is
# why it survives review so easily:
#
#   *continuous*   pixel i occupies [i, i+1) and its centre is at i + 0.5, so an image of
#                  width W spans [0, W] and its exact centre is at W/2. This is what
#                  ``Intrinsics.from_hfov`` assumes (``cx = width / 2``), what
#                  ``pixel_grid`` emits, and therefore what ``project_rays`` returns.
#
#   *index*        pixel i is addressed by the number i, so the image spans
#                  [-0.5, W-0.5] and its centre is at (W-1)/2. This is what
#                  ``cv2.remap`` expects in its map arrays, and what
#                  ``equirect_from_rays`` already returns (note its trailing ``- 0.5``).
#
# Feeding continuous coordinates straight into ``cv2.remap`` shifts every sample by half
# a pixel. The shift is uniform, so nothing misaligns and no test of *alignment* catches
# it — but every sample then lands at the worst possible interpolation phase, which
# softens the whole panorama for no reason. Convert explicitly.


def to_remap_coords(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuous pixel coordinates (from :func:`project_rays`) -> ``cv2.remap`` indices."""
    return x - 0.5, y - 0.5


def from_remap_coords(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``cv2.remap`` indices -> continuous pixel coordinates."""
    return x + 0.5, y + 0.5


# ======================================================================================
# Equirectangular <-> direction vectors
# ======================================================================================


def equirect_angles(
    width: int,
    height: int,
    *,
    u0: int = 0,
    v0: int = 0,
    w: int | None = None,
    h: int | None = None,
    dtype: type = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(theta, phi)`` for an axis-aligned window of an equirect canvas.

    The window is ``w x h`` pixels starting at ``(u0, v0)``. ``u0`` may lie outside
    ``[0, width)``; it is interpreted cyclically, which is what lets callers work on a
    seam-straddling bounding box without special-casing.
    """
    w = width if w is None else w
    h = height if h is None else h
    u = (np.arange(u0, u0 + w, dtype=np.float64) + 0.5) / float(width)
    v = (np.arange(v0, v0 + h, dtype=np.float64) + 0.5) / float(height)
    theta = (2.0 * np.pi * u - np.pi).astype(dtype)
    phi = (0.5 * np.pi - np.pi * v).astype(dtype)
    return theta, phi


def rays_from_equirect(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Outer-product ``theta`` (w,) and ``phi`` (h,) into a ``(h, w, 3)`` unit ray field."""
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    ct = np.cos(theta)[None, :]
    st = np.sin(theta)[None, :]
    cp = np.cos(phi)[:, None]
    sp = np.sin(phi)[:, None]
    h, w = phi.shape[0], theta.shape[0]
    rays = np.empty((h, w, 3), dtype=np.float32)
    rays[..., 0] = cp * st
    rays[..., 1] = -sp  # world +Y points down, +phi looks up
    rays[..., 2] = cp * ct
    return rays


def equirect_from_rays(rays: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`rays_from_equirect`: unit directions -> fractional (u, v) pixels."""
    rays = np.asarray(rays, dtype=np.float32)
    x = rays[..., 0]
    y = rays[..., 1]
    z = rays[..., 2]
    n = np.sqrt(x * x + y * y + z * z) + _EPS
    theta = np.arctan2(x, z)
    phi = np.arcsin(np.clip(-y / n, -1.0, 1.0))
    u = (theta + np.pi) / (2.0 * np.pi) * width - 0.5
    v = (0.5 * np.pi - phi) / np.pi * height - 0.5
    return u.astype(np.float32), v.astype(np.float32)


def sample_equirect(
    pano: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Sample an equirect image at fractional ``(u, v)`` with correct cyclic wrap in u.

    ``cv2.remap`` cannot mix border modes per axis, so the source is padded cyclically in
    x and reflected in y before remapping. The pad is generous enough for LANCZOS4's
    8-tap kernel.
    """
    pad = 8
    h, w = pano.shape[:2]
    padded = cv2.copyMakeBorder(pano, pad, pad, 0, 0, cv2.BORDER_REFLECT_101)
    padded = np.concatenate([padded[:, w - pad :], padded, padded[:, :pad]], axis=1)
    mx = (np.mod(u, w) + pad).astype(np.float32)
    my = (np.clip(v, -pad + 0.5, h + pad - 0.5) + pad).astype(np.float32)
    return cv2.remap(padded, mx, my, interpolation, borderMode=cv2.BORDER_REPLICATE)


# ======================================================================================
# Pinhole camera
# ======================================================================================


def pixel_grid(width: int, height: int, dtype: type = np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinate grids of shape ``(height, width)``."""
    x = np.arange(width, dtype=dtype) + 0.5
    y = np.arange(height, dtype=dtype) + 0.5
    return np.broadcast_to(x[None, :], (height, width)), np.broadcast_to(y[:, None], (height, width))


def unproject_pixels(
    x: np.ndarray,
    y: np.ndarray,
    K: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
) -> np.ndarray:
    """Pixels -> unit rays in the *camera* frame, undoing radial distortion.

    Distortion is inverted with ten fixed-point iterations, which converges to well
    under a hundredth of a pixel for the mild distortion of a phone's main lens.

    **Dtype is preserved.** Float64 input yields float64 output. Rendering feeds float32
    grids and wants float32 back for memory reasons, but bundle adjustment must stay in
    float64: quantising a 1000 px coordinate to float32 costs ~1e-4 px, which is enough
    noise to corrupt a finite-difference Jacobian and let the trust region collapse on a
    level single-row sweep, where focal and yaw spacing are nearly degenerate.
    """
    out_dtype = np.float64 if np.result_type(x, y) == np.float64 else np.float32
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    xn = (np.asarray(x, dtype=np.float64) - cx) / fx
    yn = (np.asarray(y, dtype=np.float64) - cy) / fy
    if k1 or k2:
        xu, yu = xn.copy(), yn.copy()
        for _ in range(10):
            r2 = xu * xu + yu * yu
            f = 1.0 + k1 * r2 + k2 * r2 * r2
            xu = xn / f
            yu = yn / f
        xn, yn = xu, yu
    rays = np.stack([xn, yn, np.ones_like(xn)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True) + _EPS
    return rays.astype(out_dtype, copy=False)


def project_rays(
    rays: np.ndarray,
    K: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame rays -> ``(x, y, valid)`` pixel coordinates.

    ``valid`` is False wherever the ray falls behind the image plane.

    Dtype is preserved for the same reason as :func:`unproject_pixels`: the renderer
    wants float32, the bundle adjuster needs float64.
    """
    rays = np.asarray(rays)
    if rays.dtype not in (np.float32, np.float64):
        rays = rays.astype(np.float32)
    out_dtype = rays.dtype
    z = rays[..., 2]
    valid = z > 1e-6
    zs = np.where(valid, z, 1.0)
    xn = rays[..., 0] / zs
    yn = rays[..., 1] / zs
    if k1 or k2:
        r2 = xn * xn + yn * yn
        f = 1.0 + k1 * r2 + k2 * r2 * r2
        xn = xn * f
        yn = yn * f
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return (
        (xn * fx + cx).astype(out_dtype, copy=False),
        (yn * fy + cy).astype(out_dtype, copy=False),
        valid,
    )


def camera_rays(width: int, height: int, K: np.ndarray, k1: float = 0.0, k2: float = 0.0) -> np.ndarray:
    """``(height, width, 3)`` unit rays for every pixel of an image with intrinsics ``K``."""
    x, y = pixel_grid(width, height)
    return unproject_pixels(x, y, K, k1, k2)


# ======================================================================================
# SO(3)
# ======================================================================================


def rodrigues(v: np.ndarray) -> np.ndarray:
    """Rotation vector (3,) -> rotation matrix (3, 3)."""
    R, _ = cv2.Rodrigues(np.asarray(v, dtype=np.float64).reshape(3, 1))
    return R


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> rotation vector (3,)."""
    v, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return v.reshape(3)


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    """Closest proper rotation to ``M`` in the Frobenius sense (polar decomposition)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def angular_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Geodesic angle between two rotations, in radians."""
    Rd = np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64).T
    c = (np.trace(Rd) - 1.0) * 0.5
    return float(math.acos(float(np.clip(c, -1.0, 1.0))))


def homography_from_rotation(R: np.ndarray, K_i: np.ndarray, K_j: np.ndarray) -> np.ndarray:
    """Homography mapping image *i* to image *j* under pure rotation ``R = R_j @ R_i.T``."""
    H = K_j @ R @ np.linalg.inv(K_i)
    return H / (H[2, 2] if abs(H[2, 2]) > _EPS else 1.0)


def rotation_from_homography(H: np.ndarray, K_i: np.ndarray, K_j: np.ndarray) -> np.ndarray:
    """Extract the relative rotation ``camera_i -> camera_j`` from a pure-rotation homography."""
    M = np.linalg.inv(K_j) @ np.asarray(H, dtype=np.float64) @ K_i
    return nearest_rotation(M)


def yaw_pitch_roll(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose a world->camera rotation into (yaw, pitch, roll) radians.

    Yaw is rotation about world -Y (i.e. turning right is positive), pitch is elevation
    of the optical axis, roll is rotation of the camera about its own optical axis.
    """
    d = np.asarray(R, dtype=np.float64).T @ np.array([0.0, 0.0, 1.0])
    yaw = math.atan2(d[0], d[2])
    pitch = math.asin(float(np.clip(-d[1], -1.0, 1.0)))
    R_yp = from_yaw_pitch_roll(yaw, pitch, 0.0)
    R_roll = np.asarray(R, dtype=np.float64) @ R_yp.T
    roll = math.atan2(R_roll[1, 0], R_roll[0, 0])
    return yaw, pitch, roll


def from_yaw_pitch_roll(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build a world->camera rotation from yaw / pitch / roll radians."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    # Yaw about world Y (down): turning right by +yaw rotates the camera frame by -yaw.
    Ry = np.array([[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]])
    # Pitch about camera X. The sign is chosen so that a positive pitch tips the optical
    # axis toward world -Y, i.e. the camera looks *up*.
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]])
    Rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Rx @ Ry


# ======================================================================================
# Horizon levelling
# ======================================================================================


def estimate_gravity_axis(rotations: list[np.ndarray]) -> np.ndarray:
    """Estimate the world-space sweep axis (≈ gravity) from a set of camera rotations.

    A hand-held 360 sweep is dominated by rotation about one axis. Each camera's own
    "down" vector, expressed in world coordinates, therefore clusters tightly around
    that axis. The first right singular vector of the stacked down-vectors recovers it
    robustly even when a few frames are badly oriented.
    """
    if not rotations:
        return np.array([0.0, 1.0, 0.0])
    downs = np.stack([np.asarray(R, dtype=np.float64).T @ np.array([0.0, 1.0, 0.0]) for R in rotations])
    U, S, Vt = np.linalg.svd(downs - 0.0, full_matrices=False)
    axis = Vt[0]
    if float(np.dot(axis, downs.mean(axis=0))) < 0:
        axis = -axis
    n = np.linalg.norm(axis)
    return axis / n if n > _EPS else np.array([0.0, 1.0, 0.0])


def align_horizon(rotations: list[np.ndarray]) -> np.ndarray:
    """Return a world-frame correction ``C`` such that ``R_i @ C.T`` has a level horizon.

    Applying ``C`` re-expresses every pose in a world whose +Y axis is the estimated
    gravity direction, so the panorama's horizon lands on the canvas mid-row instead of
    tilting with however the phone happened to be held.
    """
    axis = estimate_gravity_axis(rotations)
    target = np.array([0.0, 1.0, 0.0])
    v = np.cross(axis, target)
    s = float(np.linalg.norm(v))
    c = float(np.dot(axis, target))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


# ======================================================================================
# Misc
# ======================================================================================


def sphere_solid_angle_weights(height: int, width: int, dtype: type = np.float32) -> np.ndarray:
    """Per-pixel solid angle of an equirect canvas, normalised to sum to 1.

    Needed for any *fair* spherical average — near the poles an equirect pixel covers a
    vanishing patch of the sphere, so an unweighted mean over the canvas would let the
    poles dominate. Quality metrics must use this.
    """
    _, phi = equirect_angles(width, height, dtype=np.float64)
    w = np.cos(phi)[:, None]
    w = np.broadcast_to(w, (height, width)).astype(dtype)
    return w / float(w.sum())


def fov_footprint_deg(intr, R: np.ndarray) -> tuple[float, float, float, float]:
    """Angular bounding box ``(yaw_min, yaw_max, pitch_min, pitch_max)`` in degrees.

    Yaw bounds are returned unwrapped around the frame centre, so a footprint that
    straddles the +/-180 seam yields ``yaw_min < -180`` or ``yaw_max > 180`` rather than
    silently collapsing. Callers should take the result modulo 360 themselves.
    """
    w, h = intr.width, intr.height
    xs = np.linspace(0.5, w - 0.5, 24)
    ys = np.linspace(0.5, h - 0.5, 24)
    gx, gy = np.meshgrid(xs, ys)
    border = np.ones_like(gx, dtype=bool)
    border[1:-1, 1:-1] = False
    rays_cam = unproject_pixels(gx[border], gy[border], intr.K, intr.k1, intr.k2)
    rays_world = rays_cam @ np.asarray(R, dtype=np.float32)  # R.T @ v, vectorised
    theta = np.arctan2(rays_world[:, 0], rays_world[:, 2])
    phi = np.arcsin(np.clip(-rays_world[:, 1], -1.0, 1.0))
    centre = np.arctan2(
        float(np.sin(theta).mean()), float(np.cos(theta).mean())
    )
    rel = wrap_pi(theta - centre)
    return (
        math.degrees(centre + float(rel.min())),
        math.degrees(centre + float(rel.max())),
        math.degrees(float(phi.min())),
        math.degrees(float(phi.max())),
    )
