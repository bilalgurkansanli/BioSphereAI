"""One video's model into another's frame.

The general walkthrough is reconstructed once and becomes the property's frame: levelled,
metric, shared. Every room video is reconstructed on its own (a few hundred frames, a few
minutes) and then has to be placed in that frame: a similarity transform, seven numbers,
found through images the two reconstructions *share*.

They share images because the room's database is built with a handful of general-property
frames mixed in: the ones that look most like the room's own frames (retrieved on the same
global descriptor the matcher uses for loop closure). The room mapper poses those frames
alongside the room's, and since the general model already knows where they are, the two
sets of poses give the Sim3. The room's own frames are then the only ones kept for
training: the borrowed frames were shot at a different moment, and a splat that has seen a
room twice draws it twice.

WHY NOT PYCOLMAP'S ALIGNER. ``align_reconstructions_via_reprojections`` matches images by
name and then CHECKs that their ids agree in both models; they never do here (the room's
database numbers its images from one), and a failed CHECK aborts the *process*, not the
call. The estimate below is plain numpy: rotation from the shared frames' orientations,
scale and translation from their centres, RANSAC over both. Orientations make it robust to
the case centres alone cannot solve: a doorway is a line, and a line does not fix a roll.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .features import LocalFeatures

__all__ = ["RegistrationReport", "Sim3", "select_anchor_frames", "sim3_from_poses", "align_to_reference"]

#: Shared frames must span at least this much (metres, property frame) for their centres to
#: fix a scale; closer together they fix only a rotation.
MIN_ANCHOR_SPREAD_M = 0.3


@dataclass(frozen=True)
class Sim3:
    scale: float
    R: np.ndarray  # noqa: N815
    t: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(points, dtype=np.float64) @ self.R.T) + self.t

    def to_pycolmap(self):
        import pycolmap

        return pycolmap.Sim3d(self.scale, pycolmap.Rotation3d(self.R), self.t)


@dataclass
class RegistrationReport:
    n_anchor_frames: int = 0
    n_common_registered: int = 0
    n_inliers: int = 0
    method: str = ""
    scale: float = 1.0
    median_residual_m: float = 0.0
    max_residual_m: float = 0.0
    rotation_residual_deg: float = 0.0
    ok: bool = False
    warnings: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["scale"] = round(self.scale, 6)
        d["median_residual_m"] = round(self.median_residual_m, 4)
        d["max_residual_m"] = round(self.max_residual_m, 4)
        d["rotation_residual_deg"] = round(self.rotation_residual_deg, 3)
        return d


def select_anchor_frames(
    room: LocalFeatures,
    general: LocalFeatures,
    *,
    count: int = 24,
    door_fraction: float = 0.35,
    allowed: set[str] | None = None,
) -> list[int]:
    """Indices into ``general`` most similar to the room's frames.

    The recording instruction starts every room video under its door frame, which is the
    part of the room the general walkthrough saw too, so the first ``door_fraction`` of the
    room's frames get the vote; the rest of the room also votes, with less weight, so a room
    entered from an unusual angle still finds something. ``allowed`` restricts the pick to
    general frames that are actually posed in the general model: a frame the mapper dropped
    has no position to lend.
    """
    assert room.global_desc is not None and general.global_desc is not None
    sim = room.global_desc @ general.global_desc.T  # (n_room, n_general)
    n_room = sim.shape[0]
    n_door = max(1, int(round(door_fraction * n_room)))
    weight = torch.ones(n_room, device=sim.device)
    weight[n_door:] = 0.6
    score = (sim * weight[:, None]).amax(dim=0)  # best-matching room frame per general frame
    if allowed is not None:
        mask = torch.tensor([name in allowed for name in general.names], device=score.device)
        score = score.masked_fill(~mask, -2.0)
    k = min(count, score.numel())
    top = torch.topk(score, k).indices.tolist()
    return sorted(int(i) for i in top if score[i] > -1.5)


def _mean_rotation(rots: np.ndarray) -> np.ndarray:
    """Chordal mean of ``(n, 3, 3)`` rotations (SVD projection of the sum)."""
    U, _, Vt = np.linalg.svd(rots.sum(axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, 2] *= -1
        R = U @ Vt
    return R


def _fit_scale_translation(R: np.ndarray, src: np.ndarray, tgt: np.ndarray) -> tuple[float, np.ndarray]:  # noqa: N803
    """Least-squares ``s, t`` for ``tgt = s R src + t`` with ``R`` fixed."""
    rs = src @ R.T
    mu_s, mu_t = rs.mean(axis=0), tgt.mean(axis=0)
    a = rs - mu_s
    b = tgt - mu_t
    denom = float((a * a).sum())
    s = float((a * b).sum() / denom) if denom > 1e-12 else 1.0
    return s, mu_t - s * mu_s


def sim3_from_poses(
    src_R: np.ndarray, src_c: np.ndarray, tgt_R: np.ndarray, tgt_c: np.ndarray, *, iterations: int = 400, seed: int = 3
) -> tuple[Sim3, np.ndarray]:  # noqa: N803
    """Sim3 ``tgt = s R src + t`` from paired camera poses (``cam_from_world`` rotations, centres).

    Every pair proposes a rotation ``R_i = tgt_R_iᵀ src_R_i`` (world-to-world through the
    shared camera frame); RANSAC picks the rotation and the pair of centres that agree with
    the most others, then everything is refined on the inliers. Returns the transform and the
    inlier mask.
    """
    n = int(src_c.shape[0])
    src_c = np.asarray(src_c, dtype=np.float64)
    tgt_c = np.asarray(tgt_c, dtype=np.float64)
    rel = np.einsum("nji,njk->nik", tgt_R, src_R)  # tgt_R^T @ src_R per pair -> world_tgt <- world_src
    if n < 2:
        R = rel[0] if n == 1 else np.eye(3)
        s, t = (1.0, tgt_c[0] - R @ src_c[0]) if n == 1 else (1.0, np.zeros(3))
        return Sim3(s, R, t), np.ones(n, dtype=bool)

    rng = np.random.default_rng(seed)
    # Residual tolerance scales with the spread of the target centres: 4 % of the extent,
    # floored so a tiny anchor set still admits a few centimetres of SfM noise.
    extent = float(np.linalg.norm(tgt_c.max(axis=0) - tgt_c.min(axis=0)))
    tol = max(0.04 * extent, 0.05)
    best_mask = np.zeros(n, dtype=bool)
    best_R, best_s, best_t = np.eye(3), 1.0, np.zeros(3)
    for _ in range(iterations):
        i, j = rng.choice(n, 2, replace=False)
        R = _mean_rotation(rel[[i, j]])
        d_src = np.linalg.norm(src_c[i] - src_c[j])
        d_tgt = np.linalg.norm(tgt_c[i] - tgt_c[j])
        if d_src < 1e-9:
            continue
        s = d_tgt / d_src
        if not (0.05 < s < 20.0):
            continue
        t = tgt_c[i] - s * (R @ src_c[i])
        resid = np.linalg.norm(s * (src_c @ R.T) + t - tgt_c, axis=1)
        mask = resid < tol
        if mask.sum() > best_mask.sum():
            best_mask, best_R, best_s, best_t = mask, R, s, t
    if best_mask.sum() < 2:
        best_mask = np.ones(n, dtype=bool)
    # Refine: rotation as the chordal mean over the inliers, then s and t by least squares.
    R = _mean_rotation(rel[best_mask])
    s, t = _fit_scale_translation(R, src_c[best_mask], tgt_c[best_mask])
    resid = np.linalg.norm(s * (src_c @ R.T) + t - tgt_c, axis=1)
    mask = resid < tol
    if s <= 0:
        # A negative least-squares scale is a point reflection, not a similarity: it happens
        # when the shared frames barely move relative to each other, so their centres are
        # noise and the fit's sign is a coin toss. Measured on the first real capture: the
        # walk's opening part "registered" at scale -0.52 with a 2.6 deg rotation residual
        # and landed a metre up the wall. The RANSAC hypothesis (positive by construction)
        # is kept and the caller judges it.
        R, s, t, mask = best_R, best_s, best_t, best_mask
    elif mask.sum() >= 2 and mask.sum() > best_mask.sum():
        # The refined fit admits more inliers: refit on the larger set.
        R2 = _mean_rotation(rel[mask])
        s2, t2 = _fit_scale_translation(R2, src_c[mask], tgt_c[mask])
        if s2 > 0:
            R, s, t = R2, s2, t2
        else:
            R, s, t, mask = best_R, best_s, best_t, best_mask
    elif mask.sum() < best_mask.sum():
        # The refined fit lost inliers (a bad chordal mean): keep the RANSAC hypothesis.
        R, s, t, mask = best_R, best_s, best_t, best_mask
    else:
        mask = best_mask
    return Sim3(float(s), R, t), mask


def align_to_reference(room_rec, reference_rec, *, min_common: int = 4) -> tuple[Sim3 | None, RegistrationReport]:
    """Sim3 mapping ``room_rec`` into ``reference_rec``'s frame, via their common images.

    Returns ``(Sim3 | None, report)``; the room reconstruction is left untouched.
    """
    report = RegistrationReport(warnings=[])
    ref_by_name = {img.name: img for img in reference_rec.images.values() if img.has_pose}
    common = [img for img in room_rec.images.values() if img.has_pose and img.name in ref_by_name]
    report.n_common_registered = len(common)
    if len(common) < min_common:
        report.warnings.append(f"too_few_common_frames: {len(common)} of the borrowed frames registered in the room model")
        return None, report

    src_R = np.stack([np.asarray(img.cam_from_world().rotation.matrix(), dtype=np.float64) for img in common])
    src_c = np.stack([np.asarray(img.projection_center(), dtype=np.float64) for img in common])
    tgt_R = np.stack([np.asarray(ref_by_name[img.name].cam_from_world().rotation.matrix(), dtype=np.float64) for img in common])
    tgt_c = np.stack([np.asarray(ref_by_name[img.name].projection_center(), dtype=np.float64) for img in common])
    sim3, mask = sim3_from_poses(src_R, src_c, tgt_R, tgt_c)
    report.method = "pose_ransac"
    report.scale = sim3.scale
    report.n_inliers = int(mask.sum())
    resid = np.linalg.norm(sim3.apply(src_c) - tgt_c, axis=1)
    report.median_residual_m = float(np.median(resid[mask])) if mask.any() else float(np.median(resid))
    report.max_residual_m = float(resid[mask].max()) if mask.any() else float(resid.max())
    # Orientation agreement after the transform: how far each shared camera's rotation is
    # from where the reference says it points.
    rot_err = []
    for k in range(len(common)):
        if not mask[k]:
            continue
        R_mapped = src_R[k] @ sim3.R.T  # cam_from_world in the target frame
        d = np.trace(R_mapped.T @ tgt_R[k])
        rot_err.append(np.degrees(np.arccos(np.clip((d - 1.0) / 2.0, -1.0, 1.0))))
    report.rotation_residual_deg = float(np.median(rot_err)) if rot_err else 0.0
    # Scale is only observable when the shared frames stand apart: frames filmed from one
    # spot pin a rotation and nothing else. The reference is the property frame, in metres.
    spread = float(np.linalg.norm(tgt_c[mask].max(axis=0) - tgt_c[mask].min(axis=0))) if mask.sum() >= 2 else 0.0
    report.ok = (
        report.n_inliers >= min_common and report.rotation_residual_deg < 8.0 and sim3.scale > 0 and spread >= MIN_ANCHOR_SPREAD_M
    )
    if spread < MIN_ANCHOR_SPREAD_M:
        report.warnings.append(f"anchors_too_close: the shared frames span {spread:.2f} m, too little to fix a scale")
    if not report.ok:
        report.warnings.append(
            f"weak_alignment: {report.n_inliers} inliers, rotation residual {report.rotation_residual_deg:.1f} deg"
        )
    return sim3, report
