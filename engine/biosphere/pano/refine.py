"""Local warping that removes the ghosting a rotation model physically cannot.

A person sweeping a phone rotates about their spine, not about the lens' entrance pupil.
The optical centre therefore swings on a 20-30 cm arc, and the resulting parallax is a
*translation* between viewpoints.  No global rotation, no bundle adjustment and no choice
of focal length can undo it: near objects and far objects genuinely need different
alignments.  What is left is either to cut around the disagreement (:mod:`seams`) or to
locally deform the two images until they agree, which is what this module does.

The deformation is deliberately timid, because a bad flow field is much worse than a
little ghosting:

*Split the difference.*  Each image moves half way toward the other rather than one image
being warped onto the other.  A one-sided warp would put the entire distortion into one
frame and bend its straight lines.

*Fade the correction out.*  The displacement is multiplied by a weight that peaks in the
middle of the overlap -- where the seam runs -- and reaches exactly zero at the overlap
boundary, so no pixel outside the shared region is touched and neighbouring patches stay
consistent.

*Refuse implausible flow.*  A median displacement beyond ``flow_max_shift_px`` does not
mean strong parallax, it means the global alignment is wrong.  Warping by it would smear
the panorama into abstraction, so the pair is skipped and logged instead.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace

import cv2
import numpy as np

from .types import PipelineConfig
from .warp import WarpedPatch, patch_overlap

__all__ = ["refine_overlaps", "compute_flow", "select_flow_backend", "raft_available"]

_LOG = logging.getLogger(__name__)

#: Flow is computed on a downscale of the overlap; anything finer is wasted on a field
#: that is about to be blurred anyway, and DIS cost is quadratic in the crop size.
_FLOW_WORK_PX = 512
#: An overlap smaller than this has no room for a meaningful deformation.
_MIN_OVERLAP_SIDE = 48

_RAFT_MODEL = None


# ======================================================================================
# Back-end selection
# ======================================================================================


def raft_available() -> bool:
    """True only when RAFT can run *without touching the network*.

    Downloading 20 MB of weights in the middle of a render (or a test run) is not an
    acceptable surprise, so a missing checkpoint simply demotes us to DIS.
    """
    try:
        import torch
        from torchvision.models.optical_flow import Raft_Small_Weights
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    try:
        url = Raft_Small_Weights.DEFAULT.url
        cache = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(url))
        return os.path.isfile(cache)
    except Exception:  # noqa: BLE001
        return False


def select_flow_backend(config: PipelineConfig) -> str:
    """Resolve ``config.flow_backend`` (``auto`` / ``raft`` / ``dis``) to a concrete name."""
    want = str(config.flow_backend).lower()
    if want == "dis":
        return "dis"
    if want in ("raft", "auto"):
        if raft_available():
            return "raft"
        if want == "raft":
            _LOG.warning("refine: RAFT requested but unavailable, using DIS")
        return "dis"
    return "dis"


# ======================================================================================
# Flow back-ends
# ======================================================================================


def _dis_flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    preset = getattr(cv2, "DISOPTICAL_FLOW_PRESET_MEDIUM", 1)
    dis = cv2.DISOpticalFlow_create(preset)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    flow = dis.calc(ga, gb, None)
    del ga, gb, dis
    return np.asarray(flow, dtype=np.float32)


def _get_raft():
    global _RAFT_MODEL
    if _RAFT_MODEL is None:
        import torch
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
        model = model.eval().to("cuda")
        for p in model.parameters():
            p.requires_grad_(False)
        _RAFT_MODEL = (model, torch)
    return _RAFT_MODEL


def _raft_flow(a: np.ndarray, b: np.ndarray, *, tile: int = 512, halo: int = 64) -> np.ndarray:
    """RAFT-small on CUDA, tiled so a 4 GB card never sees more than ``tile + 2*halo``."""
    model, torch = _get_raft()
    h, w = a.shape[:2]
    out = np.zeros((h, w, 2), dtype=np.float32)
    with torch.inference_mode():
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                ty0, tx0 = max(0, y0 - halo), max(0, x0 - halo)
                ty1, tx1 = min(h, y1 + halo), min(w, x1 + halo)
                ca = a[ty0:ty1, tx0:tx1]
                cb = b[ty0:ty1, tx0:tx1]
                ph = (8 - (ty1 - ty0) % 8) % 8
                pw = (8 - (tx1 - tx0) % 8) % 8
                if ph or pw:
                    ca = cv2.copyMakeBorder(ca, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
                    cb = cv2.copyMakeBorder(cb, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
                ta = _to_tensor(ca, torch)
                tb = _to_tensor(cb, torch)
                pred = model(ta, tb)[-1][0].permute(1, 2, 0).float().cpu().numpy()
                del ta, tb
                pred = pred[: ty1 - ty0, : tx1 - tx0]
                out[y0:y1, x0:x1] = pred[y0 - ty0 : y1 - ty0, x0 - tx0 : x1 - tx0]
                del pred
        torch.cuda.empty_cache()
    return out


def _to_tensor(img: np.ndarray, torch):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None].to("cuda")
    return t * 2.0 - 1.0


def compute_flow(a: np.ndarray, b: np.ndarray, backend: str) -> np.ndarray:
    """Dense flow mapping pixels of ``a`` onto ``b``, computed at a reduced scale."""
    h, w = a.shape[:2]
    sc = min(1.0, _FLOW_WORK_PX / float(max(h, w)))
    if sc < 0.999:
        sw, sh = max(16, int(round(w * sc))), max(16, int(round(h * sc)))
        aa = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
        bb = cv2.resize(b, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        aa, bb = a, b
    if backend == "raft":
        try:
            flow = _raft_flow(aa, bb)
        except Exception as exc:  # noqa: BLE001 - OOM on a 4 GB card must not kill the render
            _LOG.warning("refine: RAFT failed (%s), falling back to DIS", exc)
            flow = _dis_flow(aa, bb)
    else:
        flow = _dis_flow(aa, bb)
    if aa is not a:
        del aa, bb
    if flow.shape[0] != h or flow.shape[1] != w:
        fx = cv2.resize(flow[..., 0], (w, h), interpolation=cv2.INTER_LINEAR) / max(sc, 1e-6)
        fy = cv2.resize(flow[..., 1], (w, h), interpolation=cv2.INTER_LINEAR) / max(sc, 1e-6)
        flow = np.stack([fx, fy], axis=-1).astype(np.float32)
        del fx, fy
    return flow


# ======================================================================================
# Refinement
# ======================================================================================


def _overlap_weight(both: np.ndarray) -> np.ndarray:
    """1 in the middle of the shared region, exactly 0 at its boundary."""
    m = (both.astype(np.uint8)) * 255
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    d = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    del padded, m
    peak = float(d.max())
    if peak <= 1e-3:
        return np.zeros(both.shape, dtype=np.float32)
    w = np.clip(d / (0.5 * peak), 0.0, 1.0).astype(np.float32)
    del d
    k = max(3, (int(round(peak * 0.4)) * 2 + 1))
    return cv2.GaussianBlur(w, (k, k), 0)


def refine_overlaps(patches: list[WarpedPatch], config: PipelineConfig) -> list[WarpedPatch]:
    """Return copies of ``patches`` with pairwise parallax corrections applied."""
    if len(patches) < 2:
        return list(patches)
    backend = select_flow_backend(config)
    out = [replace(p, image=p.image.copy()) for p in patches]
    max_shift = float(config.flow_max_shift_px)
    n_applied = 0
    n_skipped = 0

    for ia in range(len(out)):
        for ib in range(ia + 1, len(out)):
            a, b = out[ia], out[ib]
            ov = patch_overlap(a, b)
            if ov is None:
                continue
            oh, ow = ov.shape
            if oh < _MIN_OVERLAP_SIDE or ow < _MIN_OVERLAP_SIDE:
                continue
            both = (a.mask[ov.si] > 0) & (b.mask[ov.sj] > 0)
            if int(both.sum()) < _MIN_OVERLAP_SIDE * _MIN_OVERLAP_SIDE:
                del both
                continue

            crop_a = np.ascontiguousarray(a.image[ov.si])
            crop_b = np.ascontiguousarray(b.image[ov.sj])
            flow = compute_flow(crop_a, crop_b, backend)

            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            inside = mag[both]
            if inside.size == 0:
                del both, crop_a, crop_b, flow, mag, inside
                continue
            median = float(np.median(inside))
            frac_big = float((inside > max_shift).mean())
            del inside
            if median > max_shift or frac_big > 0.30:
                _LOG.info(
                    "refine: skipping pair (%d, %d): median flow %.1f px, %.0f%% above %.1f px",
                    a.index,
                    b.index,
                    median,
                    100.0 * frac_big,
                    max_shift,
                )
                n_skipped += 1
                del both, crop_a, crop_b, flow, mag
                continue
            del mag

            sigma = max(1.0, min(oh, ow) / 40.0)
            k = int(sigma * 4) | 1
            flow = cv2.GaussianBlur(flow, (k, k), sigma)
            w = _overlap_weight(both)
            del both

            gx, gy = np.meshgrid(
                np.arange(ow, dtype=np.float32), np.arange(oh, dtype=np.float32)
            )
            half_x = 0.5 * w * flow[..., 0]
            half_y = 0.5 * w * flow[..., 1]
            del flow, w

            warped_a = cv2.remap(
                crop_a, gx - half_x, gy - half_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            warped_b = cv2.remap(
                crop_b, gx + half_x, gy + half_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            a.image[ov.si] = warped_a
            b.image[ov.sj] = warped_b
            n_applied += 1
            del crop_a, crop_b, warped_a, warped_b, gx, gy, half_x, half_y

    _LOG.info(
        "refine: %s backend, %d pairs corrected, %d rejected by the sanity gate",
        backend,
        n_applied,
        n_skipped,
    )
    return out
