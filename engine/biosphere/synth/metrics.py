"""The scoring instrument: what "professional quality" means as a number.

Two things make panorama metrics different from ordinary image metrics, and getting
either of them wrong turns the whole harness into theatre.

**A panorama has no canonical orientation.** A stitcher is free to put the seam anywhere;
its output is correct up to a global rotation. Comparing raw pixels against the reference
would therefore measure the arbitrary choice of world frame rather than the quality of
the stitch. Everything here is *gauge-invariant*: :func:`align_panoramas` recovers the
rotation first, and :func:`pose_errors` solves for the unknown global rotation in closed
form before it reports a single degree of error.

**An equirect pixel is not a fixed piece of the sphere.** A pixel next to the zenith
covers a sliver; a pixel on the equator covers hundreds of times more solid angle. An
unweighted mean over the canvas is dominated by the poles, which is precisely the part of
a panorama nobody looks at and which most captures barely cover. Every average here is
weighted by :func:`~biosphere.pano.geometry.sphere_solid_angle_weights`, and the report
additionally carries horizon-band figures, because the +/-30 degree band is what a viewer
in a headset or a web viewer actually spends their time inside.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from biosphere.pano import geometry as g
from biosphere.pano.types import CameraPose

__all__ = [
    "PanoQuality",
    "PoseError",
    "rotate_equirect",
    "align_panoramas",
    "compare",
    "pose_errors",
    "seam_score",
    "wrap_seam_score",
    "quality_report",
    "PSNR_SENTINEL",
]

PSNR_SENTINEL: Final[float] = 999.0
_EPS: Final[float] = 1e-12
_HORIZON_BAND_DEG: Final[float] = 30.0
_ALIGN_BAND_DEG: Final[float] = 20.0
_SSIM_WIN: Final[int] = 7


# ======================================================================================
# Small helpers
# ======================================================================================


def _luma(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)


def _match_size(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if candidate.shape[:2] == reference.shape[:2]:
        return candidate
    h, w = reference.shape[:2]
    interp = cv2.INTER_AREA if candidate.shape[0] > h else cv2.INTER_LANCZOS4
    return cv2.resize(candidate, (w, h), interpolation=interp)


def _band_rows(height: int, half_deg: float) -> tuple[int, int]:
    """Row range covering ``|phi| <= half_deg``."""
    v0 = int(math.floor(height * (90.0 - half_deg) / 180.0))
    v1 = int(math.ceil(height * (90.0 + half_deg) / 180.0))
    return max(v0, 0), min(v1, height)


def _pitch_mask(height: int, width: int, half_deg: float) -> np.ndarray:
    v0, v1 = _band_rows(height, half_deg)
    m = np.zeros((height, width), dtype=bool)
    m[v0:v1] = True
    return m


def _weighted_mse(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    """Solid-angle-weighted mean squared error over the colour channels.

    Written out rather than delegated: an unweighted helper would silently let the poles
    (where an equirect canvas oversamples by a factor of a hundred) dominate the score.
    """
    total = float(weights.sum())
    if total <= _EPS:
        return float("nan")
    d = a.astype(np.float64) - b.astype(np.float64)
    per_pixel = np.einsum("ijk,ijk->ij", d, d) / float(d.shape[2]) if d.ndim == 3 else d * d
    return float((per_pixel * weights).sum() / total)


def _psnr_from_mse(mse: float) -> float:
    if not np.isfinite(mse):
        return float("nan")
    if mse <= _EPS:
        return float("inf")
    return float(10.0 * math.log10(255.0 * 255.0 / mse))


def _yaw_matrix(angle_rad: float) -> np.ndarray:
    """World rotation about +Y (down) that moves content toward larger equirect columns."""
    return g.rodrigues(np.array([0.0, float(angle_rad), 0.0]))


def _correction_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Compose a small world-frame correction from yaw / pitch / roll radians."""
    return (
        _yaw_matrix(yaw)
        @ g.rodrigues(np.array([float(pitch), 0.0, 0.0]))
        @ g.rodrigues(np.array([0.0, 0.0, float(roll)]))
    )


# ======================================================================================
# Rotating a panorama
# ======================================================================================


def rotate_equirect(
    pano: np.ndarray, R: np.ndarray, *, interpolation: int = cv2.INTER_LANCZOS4
) -> np.ndarray:
    """Re-render ``pano`` as if the world had been rotated by ``R``.

    ``R`` maps a direction in the panorama's own frame to the frame you want out, so a
    pure ``+yaw`` moves content toward larger columns.
    """
    h, w = pano.shape[:2]
    R = np.asarray(R, dtype=np.float32)
    theta, phi = g.equirect_angles(w, h)
    rays = g.rays_from_equirect(theta, phi).reshape(-1, 3)
    src = rays @ R  # R.T @ d, vectorised: the direction this pixel came from
    u, v = g.equirect_from_rays(src, w, h)
    return g.sample_equirect(pano, u.reshape(h, w), v.reshape(h, w), interpolation=interpolation)


# ======================================================================================
# Alignment
# ======================================================================================


def _phase_shift_columns(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Sub-pixel column shift taking ``candidate`` onto ``reference``, plus a confidence.

    Phase correlation on the ``+/-20`` degree horizon band. The band is windowed
    *vertically only*: an equirect canvas is genuinely cyclic in u, so a horizontal window
    would destroy the very periodicity that makes the FFT's circular correlation the right
    tool, while the top and bottom of the band really are hard cuts and do need tapering.
    """
    h, w = reference.shape[:2]
    v0, v1 = _band_rows(h, _ALIGN_BAND_DEG)
    a = _luma(candidate[v0:v1])
    b = _luma(reference[v0:v1])
    a = a - float(a.mean())
    b = b - float(b.mean())
    rows = v1 - v0
    win = np.hanning(max(rows, 2)).astype(np.float32)[:, None] if rows > 2 else np.ones((rows, 1), np.float32)
    (dx, _dy), response = cv2.phaseCorrelate(
        np.ascontiguousarray(a * win), np.ascontiguousarray(b * win)
    )
    # cv2.phaseCorrelate(src1, src2) reports how far src1 has to move to land on src2, so
    # dx is already the roll that takes the candidate onto the reference.
    return float(dx), float(response)


def align_panoramas(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    refine: bool = True,
    work_width: int = 512,
    refine_deg: Sequence[float] = (-0.6, -0.3, 0.0, 0.3, 0.6),
    yaw_polish_deg: Sequence[float] = (-0.5, -0.25, -0.12, 0.0, 0.12, 0.25, 0.5),
) -> tuple[np.ndarray, float]:
    """Rotate ``candidate`` onto ``reference`` and report the yaw applied, in degrees.

    A stitched panorama is only defined up to a global rotation, so a naive PSNR against
    the reference measures nothing but where the stitcher happened to put north. Yaw —
    by far the largest component, and a pure cyclic column shift — is recovered by phase
    correlation on the horizon band. A small pitch/roll search then polishes the residual
    tilt, but only replaces the yaw-only answer if it genuinely scores better, so an
    already-aligned pair is returned untouched rather than being needlessly resampled.
    """
    cand = _match_size(np.asarray(candidate), np.asarray(reference))
    ref = np.asarray(reference)
    h, w = ref.shape[:2]
    shift_px, _response = _phase_shift_columns(cand, ref)
    yaw = 2.0 * math.pi * shift_px / float(w)

    best_yaw, best_pitch, best_roll = yaw, 0.0, 0.0
    if refine and min(h, w) > 32:
        sw = min(int(work_width), w)
        sh = max(sw // 2, 8)
        small_c = cv2.resize(cand, (sw, sh), interpolation=cv2.INTER_AREA)
        small_r = cv2.resize(ref, (sw, sh), interpolation=cv2.INTER_AREA)
        weights = g.sphere_solid_angle_weights(sh, sw, dtype=np.float64)

        def score(yw: float, pt: float, rl: float) -> float:
            rot = rotate_equirect(small_c, _correction_matrix(yw, pt, rl))
            m = (rot.max(axis=2) > 0) if rot.ndim == 3 else (rot > 0)
            return -_weighted_mse(rot, small_r, weights * m)

        best = score(yaw, 0.0, 0.0)
        for pitch_deg in refine_deg:
            for roll_deg in refine_deg:
                if pitch_deg == 0.0 and roll_deg == 0.0:
                    continue
                s = score(yaw, math.radians(pitch_deg), math.radians(roll_deg))
                if s > best:
                    best, best_pitch, best_roll = s, math.radians(pitch_deg), math.radians(roll_deg)
        for dyaw_deg in yaw_polish_deg:
            if dyaw_deg == 0.0:
                continue
            cand_yaw = yaw + math.radians(dyaw_deg)
            s = score(cand_yaw, best_pitch, best_roll)
            if s > best:
                best, best_yaw = s, cand_yaw

    shift_px = best_yaw * float(w) / (2.0 * math.pi)
    k = int(round(shift_px))
    exact_roll = abs(shift_px - k) < 0.02 and best_pitch == 0.0 and best_roll == 0.0
    if exact_roll:
        # Lossless: an integer column roll is exactly a yaw on a cyclic canvas, and using
        # it keeps an identical pair identical instead of blurring it through a resample.
        aligned = np.roll(cand, k, axis=1)
    else:
        aligned = rotate_equirect(cand, _correction_matrix(best_yaw, best_pitch, best_roll))
    return aligned, float(math.degrees(g.wrap_pi(best_yaw)))


# ======================================================================================
# Panorama comparison
# ======================================================================================


@dataclass(slots=True)
class PanoQuality:
    """How close a candidate panorama is to the ground truth."""

    psnr: float
    ssim: float
    psnr_horizon: float
    ssim_horizon: float
    covered_fraction: float
    yaw_offset_deg: float
    width: int = 0
    height: int = 0

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view; infinite PSNR (an exact match) becomes a large sentinel."""

        def fin(x: float) -> float:
            if math.isinf(x):
                return PSNR_SENTINEL
            return round(float(x), 4) if math.isfinite(x) else float("nan")

        return {
            "psnr": fin(self.psnr),
            "ssim": round(float(self.ssim), 6),
            "psnr_horizon": fin(self.psnr_horizon),
            "ssim_horizon": round(float(self.ssim_horizon), 6),
            "covered_fraction": round(float(self.covered_fraction), 6),
            "yaw_offset_deg": round(float(self.yaw_offset_deg), 4),
            "width": int(self.width),
            "height": int(self.height),
        }


def _ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM on luma, computed with a cyclic horizontal pad.

    Without the wrap the two columns either side of the seam get their statistics from a
    reflected copy of themselves, which is exactly where a stitcher's worst artefact
    lives — the metric would be blindest precisely where it needs to be sharpest.
    """
    pad = _SSIM_WIN + 1
    la = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
    lb = np.concatenate([b[:, -pad:], b, b[:, :pad]], axis=1)
    _, full = structural_similarity(
        la, lb, data_range=255.0, win_size=_SSIM_WIN, full=True, gradient=False
    )
    return np.asarray(full, dtype=np.float64)[:, pad : pad + a.shape[1]]


def compare(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    align: bool = True,
) -> PanoQuality:
    """Score ``candidate`` against ``reference`` after removing the global rotation.

    ``mask`` restricts the comparison to a region — normally the coverage of the capture,
    so a panorama is not punished for a pole that was never filmed. When it is omitted the
    non-black pixels of the aligned candidate are used, which is the right default for a
    reprojection that simply leaves unseen parts of the sphere empty.
    """
    ref = np.asarray(reference)
    if ref.ndim != 3 or ref.shape[2] != 3:
        raise ValueError(f"reference must be an (h, w, 3) image, got {ref.shape}")
    if ref.shape[1] != 2 * ref.shape[0]:
        raise ValueError(f"reference must be a 2:1 equirect canvas, got {ref.shape[1]}x{ref.shape[0]}")

    if align:
        aligned, yaw_deg = align_panoramas(candidate, reference)
    else:
        aligned, yaw_deg = _match_size(np.asarray(candidate), ref), 0.0

    h, w = ref.shape[:2]
    if mask is None:
        # "Non-black" rather than "non-zero", and opened to kill speckle: a candidate that
        # has been through JPEG has ringing of a dozen counts around its coverage boundary,
        # and a strict `> 0` test would count that halo as covered and then score it as a
        # catastrophic error. Pass an explicit ``mask`` whenever the true coverage is known.
        valid = (aligned.max(axis=2) > 2).astype(np.uint8)
        valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    else:
        valid = np.asarray(mask)
        if valid.shape[:2] != (h, w):
            valid = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        valid = valid.astype(bool)

    weights = g.sphere_solid_angle_weights(h, w, dtype=np.float64)
    covered = float((weights * valid).sum())
    wmask = weights * valid
    horizon = weights * valid * _pitch_mask(h, w, _HORIZON_BAND_DEG)

    psnr = _psnr_from_mse(_weighted_mse(aligned, ref, wmask))
    psnr_h = _psnr_from_mse(_weighted_mse(aligned, ref, horizon))

    smap = _ssim_map(_luma(aligned), _luma(ref))
    ssim = float((smap * wmask).sum() / max(float(wmask.sum()), _EPS))
    ssim_h = float((smap * horizon).sum() / max(float(horizon.sum()), _EPS))

    return PanoQuality(
        psnr=psnr,
        ssim=ssim,
        psnr_horizon=psnr_h,
        ssim_horizon=ssim_h,
        covered_fraction=covered,
        yaw_offset_deg=yaw_deg,
        width=w,
        height=h,
    )


# ======================================================================================
# Pose accuracy
# ======================================================================================


@dataclass(slots=True)
class PoseError:
    """Rotation accuracy of a solve, with the arbitrary global frame divided out."""

    n: int
    mean_deg: float
    median_deg: float
    max_deg: float
    yaw_mean_deg: float
    yaw_max_deg: float
    pitch_mean_deg: float
    pitch_max_deg: float
    roll_mean_deg: float
    roll_max_deg: float
    gauge_deg: float = 0.0
    per_frame_deg: list[float] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": int(self.n),
            "mean_deg": round(float(self.mean_deg), 6),
            "median_deg": round(float(self.median_deg), 6),
            "max_deg": round(float(self.max_deg), 6),
            "yaw_mean_deg": round(float(self.yaw_mean_deg), 6),
            "yaw_max_deg": round(float(self.yaw_max_deg), 6),
            "pitch_mean_deg": round(float(self.pitch_mean_deg), 6),
            "pitch_max_deg": round(float(self.pitch_max_deg), 6),
            "roll_mean_deg": round(float(self.roll_mean_deg), 6),
            "roll_max_deg": round(float(self.roll_max_deg), 6),
            "gauge_deg": round(float(self.gauge_deg), 6),
        }


def _rotations(poses: Sequence[CameraPose | np.ndarray]) -> list[np.ndarray]:
    out = []
    for p in poses:
        R = p.R if isinstance(p, CameraPose) else p
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"expected 3x3 rotations, got {R.shape}")
        out.append(R)
    return out


def pose_errors(
    estimated: Sequence[CameraPose | np.ndarray], truth: Sequence[CameraPose | np.ndarray]
) -> PoseError:
    """Geodesic and per-axis rotation error, invariant to the solver's choice of world.

    A rotation-only solve can never recover the absolute orientation of the room; every
    estimate is off by one common rotation ``G``. Reporting error without removing it
    would mostly measure that arbitrary choice. ``G`` is the Wahba/Procrustes solution:
    it maximises ``tr(G^T sum_i R_est_i^T R_true_i)``, whose closed form is the nearest
    proper rotation to that sum.
    """
    Re = _rotations(estimated)
    Rt = _rotations(truth)
    if len(Re) != len(Rt):
        raise ValueError(f"pose count mismatch: {len(Re)} estimated vs {len(Rt)} truth")
    if not Re:
        raise ValueError("pose_errors needs at least one pose")

    M = np.zeros((3, 3), dtype=np.float64)
    for a, b in zip(Re, Rt, strict=True):
        M += a.T @ b
    G = g.nearest_rotation(M)

    geo: list[float] = []
    dyaw: list[float] = []
    dpitch: list[float] = []
    droll: list[float] = []
    for a, b in zip(Re, Rt, strict=True):
        aligned = a @ G
        geo.append(math.degrees(g.angular_distance(aligned, b)))
        ya, pa, ra = g.yaw_pitch_roll(aligned)
        yb, pb, rb = g.yaw_pitch_roll(b)
        dyaw.append(abs(math.degrees(float(g.wrap_pi(ya - yb)))))
        dpitch.append(abs(math.degrees(float(g.wrap_pi(pa - pb)))))
        droll.append(abs(math.degrees(float(g.wrap_pi(ra - rb)))))

    arr = np.array(geo, dtype=np.float64)
    return PoseError(
        n=len(geo),
        mean_deg=float(arr.mean()),
        median_deg=float(np.median(arr)),
        max_deg=float(arr.max()),
        yaw_mean_deg=float(np.mean(dyaw)),
        yaw_max_deg=float(np.max(dyaw)),
        pitch_mean_deg=float(np.mean(dpitch)),
        pitch_max_deg=float(np.max(dpitch)),
        roll_mean_deg=float(np.mean(droll)),
        roll_max_deg=float(np.max(droll)),
        gauge_deg=float(math.degrees(g.angular_distance(G, np.eye(3)))),
        per_frame_deg=geo,
    )


# ======================================================================================
# Seam detection
# ======================================================================================


def _column_gradients(pano: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solid-angle-weighted mean ``|dI/dx|`` and ``|dI/dy|`` per column, cyclic in x."""
    y = _luma(pano)
    h, w = y.shape
    weights = np.cos(g.equirect_angles(w, h, dtype=np.float64)[1])[:, None]
    wcol = float(weights.sum())
    gx = np.abs(np.roll(y, -1, axis=1) - y)  # column u vs u+1, wrapping at the seam
    gy = np.abs(np.diff(y, axis=0))
    ex = (gx * weights).sum(axis=0) / max(wcol, _EPS)
    ey = (gy * weights[:-1]).sum(axis=0) / max(float(weights[:-1].sum()), _EPS)
    return ex.astype(np.float64), ey.astype(np.float64)


def seam_score(pano: np.ndarray, *, top_k: int = 3, tail: int = 24) -> float:
    """How far the most suspicious vertical lines stand above the rest. Lower is better.

    A stitching seam is a *vertical* discontinuity: it lifts one column's horizontal
    gradient without touching that column's vertical structure, whereas most scene content
    lifts both. Each column is therefore scored by the anisotropy ``|dI/dx| / |dI/dy|``,
    normalised by the panorama's median so a genuinely stripey room does not read as a
    defect, and the ``top_k`` most suspicious columns are compared against the ``tail``
    columns immediately below them — a seam is a few columns standing above an otherwise
    smooth distribution, not a shift of the whole distribution.

    Read it honestly: an image with no vertical structure scores 1.0, but a *correct*
    panorama of a real room still scores roughly 2-4, because a wall corner is a genuine
    vertical discontinuity and no single-image statistic can tell one from a bad join.
    The actionable number is this score divided by the same score on the reference, which
    :func:`quality_report` reports as ``seam_score_ratio`` and which is near 1.0 for a
    stitch that introduced nothing of its own.
    """
    ex, ey = _column_gradients(pano)
    ratio = (ex + 1e-4) / (ey + 1e-4)
    med = float(np.median(ratio))
    if med <= _EPS:
        return 1.0
    ratio = ratio / med
    order = np.argsort(ratio)
    k = max(int(top_k), 1)
    m = max(int(tail), 1)
    if ratio.size < k + m + 1:
        raise ValueError("panorama is too narrow to compute a seam score")
    top = ratio[order[-k:]]
    ref = ratio[order[-(k + m) : -k]]
    denom = float(ref.mean())
    return float(top.mean() / denom) if denom > _EPS else float("inf")


def wrap_seam_score(pano: np.ndarray, *, neighbourhood: int = 24) -> float:
    """How much worse the cyclic seam is than its neighbouring column boundaries. 1.0 is perfect.

    Column ``W-1`` and column ``0`` are neighbours on the sphere; if the stitcher's world
    frame is even slightly inconsistent around the loop, or the last strip was blended
    against the first without wrapping, this is exactly where it shows — and it is the one
    artefact a viewer is guaranteed to pan across.

    The baseline is the median adjacent-column difference over the ``neighbourhood``
    columns either side of the seam rather than over the whole canvas. A global median is
    dominated by whatever else is in the room, so a seam sitting in a plain stretch of wall
    could score below 1.0 while being plainly visible; a local baseline asks the only
    question that matters — is this boundary worse than the ones right next to it?

    1.0 means the wrap is as ordinary as its neighbours and anything below that is simply
    a seam that landed on smooth wall (a perfect synthetic render typically scores 0.2-0.5,
    because a continuous function has *no* discontinuity there at all). Above roughly 2.0
    the join is visible to a viewer panning past it.
    """
    y = _luma(pano)
    h, w = y.shape
    weights = np.cos(g.equirect_angles(w, h, dtype=np.float64)[1])
    wsum = max(float(weights.sum()), _EPS)
    wrap = float((np.abs(y[:, 0] - y[:, w - 1]) * weights).sum() / wsum)
    adj = np.abs(np.diff(y, axis=1))
    per_col = (adj * weights[:, None]).sum(axis=0) / wsum  # boundary u <-> u+1, u = 0..w-2
    k = max(min(int(neighbourhood), (w - 1) // 2), 1)
    local = np.concatenate([per_col[:k], per_col[-k:]])
    med = float(np.median(local))
    if med <= _EPS:
        med = float(np.median(per_col))
    if med <= _EPS:
        return 1.0 if wrap <= _EPS else float("inf")
    return float(wrap / med)


# ======================================================================================
# The report
# ======================================================================================


def quality_report(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    estimated_poses: Sequence[CameraPose | np.ndarray] | None = None,
    truth_poses: Sequence[CameraPose | np.ndarray] | None = None,
    label: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything the harness knows about one stitch, as a JSON-serialisable dict.

    This is the object CI asserts on and a report page renders, so it deliberately
    includes the seam statistics and the pose error alongside the headline PSNR: a
    panorama can score well on PSNR and still have one glaring join, and a solve can have
    tiny pose error while the renderer ruins the picture.
    """
    quality = compare(candidate, reference, mask)
    cand_seam = float(seam_score(candidate))
    ref_seam = float(seam_score(reference))
    report: dict[str, Any] = {
        "label": label,
        "quality": quality.as_dict(),
        "seam": {
            "seam_score": round(cand_seam, 6),
            "wrap_seam_score": round(float(wrap_seam_score(candidate)), 6),
            "reference_seam_score": round(ref_seam, 6),
            "reference_wrap_seam_score": round(float(wrap_seam_score(reference)), 6),
            # The number to assert on: 1.0 means the stitch added no vertical structure of
            # its own, whatever the room happened to contain.
            "seam_score_ratio": round(cand_seam / ref_seam, 6) if ref_seam > _EPS else float("inf"),
        },
    }
    if estimated_poses is not None and truth_poses is not None:
        report["poses"] = pose_errors(estimated_poses, truth_poses).as_dict()
    if extra:
        report["extra"] = dict(extra)
    return report
