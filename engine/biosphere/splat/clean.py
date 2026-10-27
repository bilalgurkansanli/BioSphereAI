"""What comes off a trained cloud before it is delivered, and what must stay on it.

A trained cloud does carry debris: near-transparent splats the optimiser stopped caring
about, and huge ones that stretch across the room when the camera walks past them. Those are
property thresholds and they are cheap. The hard question is the third kind - splats
"nothing constrained" - and getting its shape wrong is expensive.

It used to be a box around the camera path. That is the wrong shape, because a walkthrough
is full of views THROUGH things: the street through a window, the room at the end of a
corridor, the ceiling two storeys up a stairwell. All of it is constrained - it appears in
photographs - and all of it stood outside the box. Measured on a real property, against frames
training never saw:

    stage                                splats     PSNR    local contrast
    as trained                          357,579    24.62 dB      35.7
    the box, 6 m around the path        323,814    20.16         34.8
    the size cap at 1.2 m               353,437    22.63         33.5
    the needle clamp at 12              357,579    24.65         19.8
    everything, as delivered            319,740    19.91         18.9
    seen by 2+ cameras, 3 m, no clamp   353,317    24.17         35.4

(the photographs themselves measure 42.8 on the same contrast scale.)

So the delivery was giving away 4.7 dB and half the picture's local contrast, and the
visitor saw it as dark smears on the ceiling and a grey wash over the floor. The criterion
that matches the purpose is visibility, not position: project a splat into every solved
camera and keep it if it lands inside one. On the same cloud that removes 69 splats out of
357,579 - which is all the unconstrained junk there ever was, the trainer having pruned the
rest itself.

The needle clamp is gone for the same reason. Widening the thin axis of an elongated
Gaussian does suppress streaks, but it suppresses the wall's texture with them, for no gain
in PSNR at all. The measurement that once argued for it was invalid: it re-clamped a file
that had already been clamped, so its "unclamped" column was nothing of the kind. A needle
seen edge-on IS a problem for the floor plan, which looks at the flat from directly above,
and that is where the clamp now lives - in the drawing, not in the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from .ply import GaussianTable

__all__ = ["CameraView", "CleanConfig", "CleanReport", "clean_table", "path_box", "seen_by"]


@dataclass(frozen=True)
class CleanConfig:
    #: Splats fainter than this are dropped. 3DGS prunes at ~0.005; 0.04 was eight times
    #: that and took 51 % of a real model with it, including the faint layers that fill a
    #: wall - measured coverage at a delivered viewpoint rose 0.845 -> 0.892 by relaxing it,
    #: for splats that compress to a few hundred kilobytes more.
    min_opacity: float = 0.01
    #: Largest axis a splat may have, in metres. A cap of 1.2 m looked safe and cost 2.0 dB
    #: of held-out PSNR for 4,000 splats: indoors the biggest legitimate Gaussians are the
    #: flat expanses - a wall, a ceiling, a floor seen edge-on - and they are what a room is
    #: mostly made of. At 3 m the same cut costs 0.44 dB and still catches the room-spanning
    #: smears the cap was written for.
    max_scale_m: float = 3.0
    #: ...and it applies only to splats within this distance of the camera path. A smear across
    #: the room is near the walk; a giant far away is the street through a window, painted by
    #: the trainer with a few big faint splats because SfM put no points out there. Measured on
    #: a 79-frame part of the first real capture (8 held-out frames): of 761 giants, the 45
    #: within 3 m of a camera are the smears (cutting them costs 0.04 dB); the other 716, median
    #: 11 m away, carried 0.67 dB - the whole cap cost 0.71 dB of the 0.77 the clean took.
    #: 0 applies the cap everywhere (the old behaviour).
    giant_near_m: float = 3.0
    #: Longest axis over shortest, after which a splat is a needle. 0 disables the clamp,
    #: which is the measured default: widening thin axes halved the delivered picture's
    #: local contrast (35.7 -> 19.8, against photographs at 42.8) and bought no PSNR at all.
    #: The earlier measurement that argued for it was invalid - it re-clamped a file that had
    #: already been clamped, so its "unclamped" column was nothing of the kind.
    max_anisotropy: float = 0.0
    #: How many solved cameras a splat must appear in to be kept, when the poses are given.
    #: A splat inside a photograph is constrained by it, wherever it stands; one no camera
    #: ever framed is a guess. This replaces the box below, which was the wrong shape: drawn
    #: around where the cameras WALKED, it cut away what they SAW - the street through a
    #: window, the room at the end of a corridor - for 4.5 dB of held-out PSNR, while the
    #: junk it was written to remove came to 69 splats out of 357,579.
    min_views: int = 2
    #: Box around the camera path, used only when no poses are supplied: horizontal padding,
    #: headroom above the cameras, and how far below the FLOOR (``y = 0``) a splat may sit.
    pad_xz_m: float = 6.0
    pad_up_m: float = 3.2
    pad_below_floor_m: float = 0.35
    drop_nan: bool = True


@dataclass
class CleanReport:
    n_in: int = 0
    n_out: int = 0
    n_transparent: int = 0
    n_giant: int = 0
    #: Giants left alone because they stand far from the walk (see ``giant_near_m``).
    n_giant_far_kept: int = 0
    n_outside: int = 0
    n_nan: int = 0
    #: Splats whose thin axes were widened because they were needles.
    n_needles: int = 0
    bbox_min: list[float] | None = None
    bbox_max: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def path_box(cam_centres: np.ndarray, config: CleanConfig) -> tuple[np.ndarray, np.ndarray]:
    """Padded axis-aligned box around the camera path, in the levelled frame (``+Y`` down)."""
    c = np.asarray(cam_centres, dtype=np.float64)
    lo = c.min(axis=0)
    hi = c.max(axis=0)
    pad_lo = np.array([config.pad_xz_m, config.pad_up_m, config.pad_xz_m])  # y is down: "up" lowers y
    pad_hi = np.array([config.pad_xz_m, 0.0, config.pad_xz_m])
    top = lo - pad_lo
    bottom = hi + pad_hi
    # The floor is y = 0 in the property frame; whatever the cameras did, the box reaches it.
    bottom[1] = max(bottom[1], 0.0) + config.pad_below_floor_m
    return top, bottom


class CameraView(NamedTuple):
    """A solved camera, enough of it to ask whether a point is inside its picture."""

    R: np.ndarray
    t: np.ndarray
    K: np.ndarray
    width: int
    height: int


def seen_by(means: np.ndarray, views: Sequence[CameraView], *, margin: float = 0.15) -> np.ndarray:
    """How many of ``views`` each point falls in front of and inside, counted per point.

    The margin lets a point sit a little outside the frame and still count: a Gaussian whose
    centre is just off the edge still paints pixels inside it.
    """
    means = np.asarray(means, dtype=np.float64)
    n = np.zeros(len(means), dtype=np.int32)
    for view in views:
        p = means @ np.asarray(view.R).T + np.asarray(view.t)
        front = p[:, 2] > 0.05
        if not front.any():
            continue
        uv = p[front] @ np.asarray(view.K).T
        u = uv[:, 0] / uv[:, 2]
        v = uv[:, 1] / uv[:, 2]
        inside = (
            (u > -margin * view.width) & (u < (1 + margin) * view.width)
            & (v > -margin * view.height) & (v < (1 + margin) * view.height)
        )
        n[np.flatnonzero(front)[inside]] += 1
    return n


def clean_table(
    table: GaussianTable,
    cam_centres: np.ndarray,
    *,
    config: CleanConfig | None = None,
    views: Sequence[CameraView] | None = None,
) -> tuple[GaussianTable, CleanReport]:
    config = config or CleanConfig()
    rep = CleanReport(n_in=table.count)
    keep = np.ones(table.count, dtype=bool)
    if config.drop_nan:
        finite = np.isfinite(table.means).all(1) & np.isfinite(table.log_scales).all(1) & np.isfinite(table.quats).all(1) & np.isfinite(table.opacity_logit)
        finite &= np.linalg.norm(table.quats, axis=1) > 1e-6
        rep.n_nan = int((~finite).sum())
        keep &= finite
    opa = table.opacity
    transparent = opa < config.min_opacity
    rep.n_transparent = int((transparent & keep).sum())
    keep &= ~transparent
    giant = table.scales.max(axis=1) > config.max_scale_m
    cams = np.asarray(cam_centres, dtype=np.float64).reshape(-1, 3)
    if config.giant_near_m > 0 and len(cams) and giant.any():
        from scipy.spatial import cKDTree

        near = np.full(table.count, np.inf)
        near[giant] = cKDTree(cams).query(table.means[giant].astype(np.float64))[0]
        rep.n_giant_far_kept = int((giant & (near >= config.giant_near_m) & keep).sum())
        giant &= near < config.giant_near_m
    rep.n_giant = int((giant & keep).sum())
    keep &= ~giant
    if views:
        inside = seen_by(table.means, views) >= config.min_views
    else:
        lo, hi = path_box(cam_centres, config)
        inside = np.all((table.means >= lo) & (table.means <= hi), axis=1)
    rep.n_outside = int((~inside & keep).sum())
    keep &= inside
    out = table.select(keep)
    if config.max_anisotropy > 0 and out.count:
        scales = out.scales
        longest = scales.max(axis=1, keepdims=True)
        widened = np.maximum(scales, longest / config.max_anisotropy)
        rep.n_needles = int((widened != scales).any(axis=1).sum())
        out.log_scales = np.log(np.maximum(widened, 1e-9)).astype(out.log_scales.dtype)
    rep.n_out = out.count
    if out.count:
        rep.bbox_min = out.means.min(axis=0).round(4).tolist()
        rep.bbox_max = out.means.max(axis=0).round(4).tolist()
    return out, rep
