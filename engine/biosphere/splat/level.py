"""Gravity, floor and metric scale for a reconstruction that has none of the three.

Structure from motion returns a frame that is rotated arbitrarily, has its origin wherever
the first camera happened to be, and is scaled by an unknown factor. A property needs a floor
to walk on, an up to hold the camera level, and metres so that one step is one step. The
three come from three different clues:

* **Up** starts as the mean of the cameras' own up axes: the phone is held upright, and a
  few hundred frames average out the tilt of any one of them. It is then refined by the
  floor: the dominant plane among the points *below* the cameras, found by RANSAC with the
  camera prior narrowing the search, is the floor, and its normal is the true up. The prior
  keeps a wall or a table from being mistaken for the floor; the RANSAC removes the
  camera-tilt bias (a walker looks slightly down).
* **Floor height** is the plane's offset. Points below it by more than a margin are noise
  and are later discarded.
* **Scale** is the ratio between a known camera height and the median distance of the
  cameras above the floor plane. Chest-to-head height is the recording instruction, so
  1.45 m is assumed; a metric depth model can replace this assumption when one is
  available (see :func:`metric_scale`).

Everything here is expressed in the COLMAP world convention the rest of the package uses:
after :func:`leveling_transform` the floor is ``y = 0`` and up is ``-Y``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Leveling", "camera_up_prior", "fit_floor", "leveling_transform", "apply_sim3", "estimate_leveling"]

#: The recording instruction is "chest-to-head height"; this sits in the middle of that.
ASSUMED_CAMERA_HEIGHT_M = 1.45


@dataclass
class Leveling:
    up: np.ndarray  # unit vector in the source frame pointing up
    floor_point: np.ndarray  # a point on the floor plane, source frame
    floor_inlier_fraction: float
    up_prior_angle_deg: float  # how far the floor normal moved the camera prior
    camera_height_source_units: float
    scale: float  # source units -> metres
    scale_source: str  # "camera_height" | "metric_depth"
    R: np.ndarray  # noqa: N815 - the similarity that levels: p_level = scale * R p + t
    t: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "up": [float(v) for v in self.up],
            "floor_inlier_fraction": round(float(self.floor_inlier_fraction), 4),
            "up_prior_angle_deg": round(float(self.up_prior_angle_deg), 3),
            "camera_height_source_units": round(float(self.camera_height_source_units), 5),
            "scale": round(float(self.scale), 6),
            "scale_source": self.scale_source,
            "R": np.asarray(self.R).round(8).tolist(),
            "t": [float(v) for v in self.t],
        }


def camera_up_prior(rotations: np.ndarray) -> np.ndarray:
    """Mean camera up (``-y`` of each camera, in world) over ``(n, 3, 3)`` cam_from_world rotations."""
    ups = -rotations[:, 1, :]
    u = ups.mean(axis=0)
    return u / max(np.linalg.norm(u), 1e-12)


def fit_floor(
    points: np.ndarray,
    cam_centres: np.ndarray,
    up_prior: np.ndarray,
    *,
    max_normal_angle_deg: float = 20.0,
    iterations: int = 600,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, float]:
    """RANSAC the floor among points below the cameras. Returns ``(normal, point, inlier_fraction)``.

    Distances are relative to the cameras' height above the candidate plane so the routine
    works at any SfM scale: an inlier lies within 2 % of the camera height of the plane.
    """
    points = np.asarray(points, dtype=np.float64)
    cams = np.asarray(cam_centres, dtype=np.float64)
    up = np.asarray(up_prior, dtype=np.float64)
    cam_mean = np.median(cams, axis=0)  # robust to a camera the mapper threw far away
    height = (points - cam_mean) @ up  # positive = above the cameras
    below = points[height < 0.0]
    if below.shape[0] < 50:
        return up, cam_mean - up * 1.0, 0.0
    # Keep the lower third of the "below" cloud: walls and furniture legs sit above the floor.
    cut = np.percentile((below - cam_mean) @ up, 40)
    cand = below[((below - cam_mean) @ up) <= cut]
    if cand.shape[0] < 30:
        cand = below
    rng = np.random.default_rng(seed)
    cos_limit = np.cos(np.radians(max_normal_angle_deg))
    best_n, best_p, best_count = up, cand.mean(axis=0), -1
    n_cand = cand.shape[0]
    for _ in range(iterations):
        idx = rng.choice(n_cand, 3, replace=False)
        a, b, c = cand[idx]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        if n @ up < 0:
            n = -n
        if n @ up < cos_limit:
            continue
        cam_h = np.median((cams - a) @ n)
        if cam_h <= 0:
            continue
        thr = 0.02 * cam_h
        d = np.abs((cand - a) @ n)
        count = int((d < thr).sum())
        if count > best_count:
            best_count, best_n, best_p = count, n, a
    if best_count <= 0:
        return up, cand.mean(axis=0), 0.0
    # Refine on the inliers with a least-squares plane.
    cam_h = np.median((cams - best_p) @ best_n)
    d = np.abs((cand - best_p) @ best_n)
    inl = cand[d < 0.02 * cam_h]
    if inl.shape[0] >= 3:
        centre = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - centre, full_matrices=False)
        n = vt[2]
        if n @ up < 0:
            n = -n
        if n @ up >= cos_limit:
            best_n, best_p = n, centre
    return best_n, best_p, best_count / float(n_cand)


def leveling_transform(up: np.ndarray, floor_point: np.ndarray, centre: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` such that ``p' = scale * R p + t`` puts the floor at ``y'=0`` and up along ``-Y``.

    The horizontal origin lands under ``centre`` (typically the mean camera position of the
    general walkthrough); the rotation keeps the source X axis as horizontal as possible so
    the map does not spin between runs.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    y_axis = -up  # world +Y must point down
    x_hint = np.array([1.0, 0.0, 0.0])
    x_axis = x_hint - (x_hint @ y_axis) * y_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_hint = np.array([0.0, 0.0, 1.0])
        x_axis = x_hint - (x_hint @ y_axis) * y_axis
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    R = np.stack([x_axis, y_axis, z_axis])  # rows = new axes in source coordinates
    floor_y = scale * (R @ np.asarray(floor_point, dtype=np.float64))[1]
    c = scale * (R @ np.asarray(centre, dtype=np.float64))
    t = np.array([-c[0], -floor_y, -c[2]])
    return R, t


def apply_sim3(points: np.ndarray, R: np.ndarray, t: np.ndarray, scale: float) -> np.ndarray:  # noqa: N803
    return scale * (np.asarray(points, dtype=np.float64) @ np.asarray(R).T) + np.asarray(t)


def estimate_leveling(
    rotations: np.ndarray,
    cam_centres: np.ndarray,
    points: np.ndarray,
    *,
    assumed_camera_height_m: float = ASSUMED_CAMERA_HEIGHT_M,
    metric_scale: float | None = None,
) -> Leveling:
    """Level and scale from cameras and sparse points. ``metric_scale`` overrides the height prior."""
    up_prior = camera_up_prior(np.asarray(rotations, dtype=np.float64))
    normal, fpoint, frac = fit_floor(points, cam_centres, up_prior)
    angle = float(np.degrees(np.arccos(np.clip(normal @ up_prior, -1.0, 1.0))))
    cam_h = float(np.median((np.asarray(cam_centres) - fpoint) @ normal))
    if cam_h <= 1e-9:
        cam_h = float(np.median(np.linalg.norm(np.asarray(cam_centres) - fpoint, axis=1)))
    if metric_scale is not None and metric_scale > 0:
        scale, source = float(metric_scale), "metric_depth"
    else:
        scale, source = assumed_camera_height_m / cam_h, "camera_height"
    # The median, not the mean: one camera the mapper threw across the street must not move
    # the origin of the whole flat.
    centre = np.median(np.asarray(cam_centres, dtype=np.float64), axis=0)
    R, t = leveling_transform(normal, fpoint, centre, scale)
    return Leveling(
        up=normal, floor_point=fpoint, floor_inlier_fraction=frac, up_prior_angle_deg=angle,
        camera_height_source_units=cam_h, scale=scale, scale_source=source, R=R, t=t,
    )
