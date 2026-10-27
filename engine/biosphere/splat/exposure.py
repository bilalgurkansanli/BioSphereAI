"""One exposure for the whole walk, so the trainer stops explaining the phone's metering.

A phone films a flat on automatic exposure: it opens up in the hallway, stops down facing
the window, and the same wall is two thirds as bright a second later. Measured on the first
real capture, the training frames of one area spanned 1.59x in mean brightness, with jumps of
up to 32 grey levels between consecutive frames.

3DGS has no exposure parameter. Asked to reconstruct one surface that is bright in one frame
and dark in the next, the optimiser does the only thing it can: it puts a faint,
near-camera veil in front of the camera and tunes it per direction. That veil is the milky
haze over a delivered walkthrough - measured at 49,000 splats within two metres of the
camera contributing 11 % alpha across the entire frame, with no surface behind them.

The fix is to take the exposure out of the pictures before training. Every 3D point the
mapper triangulated is seen by several frames and has ONE true colour, so the frames'
disagreement about it is the gain to remove: solve for a per-frame gain that makes every
track agree, in log space, and re-centre the gains so the walk keeps its own overall
brightness. It is the photometric alignment the panorama engine already does for the same
reason, applied to the frames a splat is trained from.

What it deliberately does not do is normalise each frame to a common mean. A dark corridor
next to a sunlit room is a real difference and belongs in the model; only the disagreement
about *the same points* is metering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

__all__ = ["ExposureReport", "estimate_gains", "apply_gains", "balance_exposure"]


@dataclass
class ExposureReport:
    n_images: int = 0
    n_tracks: int = 0
    n_observations: int = 0
    #: exp() of the solved log-gains: 1.0 means the frame was already consistent.
    gain_min: float = 1.0
    gain_max: float = 1.0
    gain_spread: float = 1.0
    #: Disagreement between frames about the SAME 3D points, in grey levels. This is what the
    #: solve removes. Frame-mean brightness is deliberately not reported as a before/after:
    #: automatic exposure compresses real differences (it brightens a dark corridor), so
    #: undoing it correctly makes the frames LESS alike, not more.
    residual_before: float = 0.0
    residual_after: float = 0.0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _sample(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Mean of a 3x3 patch at each ``xy`` (float32 luminance in [0, 1]), NaN outside."""
    h, w = image.shape[:2]
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    ok = (x >= 1) & (y >= 1) & (x < w - 1) & (y < h - 1)
    out = np.full(len(xy), np.nan, dtype=np.float64)
    if not ok.any():
        return out
    xs, ys = x[ok], y[ok]
    patch = np.stack([image[ys + dy, xs + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    out[ok] = patch.mean(axis=0)
    return out


def estimate_gains(
    rec, image_dir: Path, *, min_track_length: int = 3, max_points: int = 20000, iterations: int = 12
) -> tuple[dict[str, float], ExposureReport]:
    """Per-image multiplicative gains that make the shared 3D points agree.

    Alternating least squares on ``log(observed) = log(true colour) + log(gain)``: with the
    frames fixed the true colours are a mean, with the colours fixed the gains are a mean.
    A dozen passes is far past convergence for a walk's worth of frames.
    """
    import time

    t0 = time.perf_counter()
    report = ExposureReport()
    image_dir = Path(image_dir)
    names: list[str] = []
    index: dict[int, int] = {}
    grey: list[np.ndarray] = []
    for image_id, img in rec.images.items():
        if not img.has_pose:
            continue
        path = image_dir / img.name
        if not path.exists():
            continue
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        index[image_id] = len(names)
        names.append(img.name)
        grey.append(im.astype(np.float64) / 255.0)
    report.n_images = len(names)
    if len(names) < 2:
        report.warnings.append("too_few_images_for_exposure")
        return {n: 1.0 for n in names}, report

    # Observations, gathered per image so each frame is sampled in one pass.
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    points = [(pid, pt) for pid, pt in rec.points3D.items() if pt.track.length() >= min_track_length]
    if len(points) > max_points:
        step = len(points) // max_points + 1
        points = points[::step]
    track_of: dict[int, int] = {pid: k for k, (pid, _) in enumerate(points)}
    per_image: dict[int, list[tuple[int, int]]] = {}
    for pid, pt in points:
        for el in pt.track.elements:
            if el.image_id in index:
                per_image.setdefault(el.image_id, []).append((el.point2D_idx, track_of[pid]))
    for image_id, obs in per_image.items():
        img = rec.images[image_id]
        xy = np.array([img.points2D[i].xy for i, _ in obs], dtype=np.float64)
        sampled = _sample(grey[index[image_id]], xy)
        good = np.isfinite(sampled) & (sampled > 0.01) & (sampled < 0.99)  # clipped pixels carry no gain
        if not good.any():
            continue
        rows.append(np.full(int(good.sum()), index[image_id], dtype=np.int64))
        cols.append(np.array([t for _, t in obs], dtype=np.int64)[good])
        vals.append(np.log(sampled[good]))
    if not rows:
        report.warnings.append("no_usable_observations_for_exposure")
        return {n: 1.0 for n in names}, report
    frame = np.concatenate(rows)
    track = np.concatenate(cols)
    observed = np.concatenate(vals)
    report.n_tracks = len(points)
    report.n_observations = int(len(observed))

    n_frames, n_tracks = len(names), len(points)
    gains = np.zeros(n_frames)
    frame_counts = np.bincount(frame, minlength=n_frames).astype(np.float64)
    track_counts = np.bincount(track, minlength=n_tracks).astype(np.float64)
    for _ in range(iterations):
        colour = np.bincount(track, weights=observed - gains[frame], minlength=n_tracks) / np.maximum(track_counts, 1)
        gains = np.bincount(frame, weights=observed - colour[track], minlength=n_frames) / np.maximum(frame_counts, 1)
        gains -= gains[frame_counts > 0].mean()  # the walk keeps its own overall brightness

    multipliers = np.exp(-gains)  # divide the frame by its gain
    result = {name: float(m) for name, m in zip(names, multipliers, strict=True)}
    report.gain_min = float(multipliers.min())
    report.gain_max = float(multipliers.max())
    report.gain_spread = float(multipliers.max() / max(multipliers.min(), 1e-6))
    colour = np.bincount(track, weights=observed - gains[frame], minlength=n_tracks) / np.maximum(track_counts, 1)
    report.residual_before = float(np.std(observed - colour[track]) * 255)
    report.residual_after = float(np.std(observed - colour[track] - gains[frame]) * 255)
    report.seconds = time.perf_counter() - t0
    return result, report


def apply_gains(image_dir: Path, gains: dict[str, float], *, quality: int = 95, knee: float = 0.75) -> int:
    """Apply each image's gain in place, compressing highlights instead of clipping them.

    A plain multiply destroys the top of the range: brightening a frame by 1.5 flattened up to
    26 % of its pixels to pure white on a real capture, and the trainer answered with bright
    streaks where the flat areas disagreed with their neighbours. Below ``knee`` the gain is
    exact - that is where the tracks the solve is built on live - and above it the response
    rolls off smoothly to 1, so a window keeps being a window instead of becoming a hole.

    Returns how many images were rewritten.
    """
    image_dir = Path(image_dir)
    written = 0
    for name, gain in gains.items():
        if abs(gain - 1.0) < 0.002:
            continue
        path = image_dir / name
        if not path.exists():
            continue
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is None:
            continue
        x = im.astype(np.float32) / 255.0 * gain
        head = 1.0 - knee
        out = np.where(x <= knee, x, knee + head * np.tanh(np.maximum(x - knee, 0.0) / head))
        cv2.imwrite(str(path), np.clip(out * 255.0, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
        written += 1
    return written


def balance_exposure(rec, image_dir: Path, **kwargs) -> ExposureReport:
    """Solve the gains and apply them to the images the trainer will read."""
    gains, report = estimate_gains(rec, image_dir, **kwargs)
    apply_gains(image_dir, gains)
    return report
