"""Choosing which extracted frames actually enter the solve.

A 30-second sweep sampled at 8 fps yields ~240 frames of which maybe 40 carry new
information. Feeding all of them to the bundle adjuster costs O(n^2) matching for no
extra coverage, and — worse — the redundant near-duplicate pairs have almost no baseline
in rotation, so they contribute badly-conditioned constraints that pull the solution
around. Selection is therefore a quality gate, not just a speed optimisation.

Why phase correlation instead of feature tracking
-------------------------------------------------
Selection has to run *before* feature detection (that is the point), and it has to work
on the frames that are worst for features: motion-blurred ones. Phase correlation is a
whole-image FFT statistic — it degrades gracefully under blur, needs no descriptors, and
at 360 px wide it costs well under a millisecond per frame. We only need the motion to
about a degree, which is far inside what it delivers.

Sign conventions
----------------
``cv2.phaseCorrelate(a, b)`` returns the displacement of the *content* from ``a`` to
``b``. Content sliding right means the camera panned left, so the yaw step is the
negation of the horizontal shift. Content sliding down means the camera tilted up, and
pitch is positive when the camera looks up (see :mod:`biosphere.pano.types`), so the
pitch step takes the vertical shift unchanged.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import structlog

from .types import (
    Frame,
    Intrinsics,
    PipelineConfig,
    StitchError,
    SweepPattern,
)

__all__ = [
    "DEFAULT_HFOV_DEG",
    "estimate_frame_motion",
    "motion_to_angles",
    "detect_sweep_pattern",
    "pitch_clusters",
    "select_keyframes",
]

log = structlog.get_logger(__name__)

#: Horizontal field of view assumed for a phone's main camera before anything better is
#: known. Modern flagships sit between 65 and 70 degrees on the 1x lens.
DEFAULT_HFOV_DEG = 67.0

#: Working width for the phase-correlation pass.
_MOTION_WIDTH = 360

#: Two pitch rows are only considered distinct if their centres differ by at least this.
_ROW_SEPARATION_DEG = 8.0

#: A row has to sweep at least this much yaw to count as a row rather than a wobble.
_ROW_MIN_YAW_SPAN_DEG = 180.0


# ======================================================================================
# Frame-to-frame motion
# ======================================================================================


def _motion_gray(frame: Frame, width: int) -> np.ndarray | None:
    """Load a frame as a small 8-bit grayscale image (converted to float32 at use).

    Not cached on the frame any more, and 8-bit rather than float32. The cache held a
    float32 copy of EVERY extracted frame for the rest of the solve -- 622 MB on the
    reference 900-frame capture, and it rode along into every pickled solve -- for an image
    that is read exactly once. The float32 values are the same either way: the cast is
    exact from uint8, it just happens per pair now instead of once per frame.
    """
    img = cv2.imread(str(frame.path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w != width:
        new_h = max(1, int(round(h * width / w)))
        img = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)
    return img


def _load_motion_grays(frames: list[Frame], width: int, workers: int) -> list[np.ndarray | None]:
    """Every frame's motion image, decoded in parallel.

    The decode was a sequential loop over full-resolution JPEGs -- 900 of them on the
    reference capture, the single largest cost of the ``select`` stage -- while the phase
    correlation it feeds is under a millisecond a pair. ``cv2.imread`` and ``cv2.resize``
    release the GIL, so threads parallelise it without copying anything between processes.
    """
    if workers <= 1 or len(frames) < 8:
        return [_motion_gray(f, width) for f in frames]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda f: _motion_gray(f, width), frames))


def estimate_frame_motion(
    frames: list[Frame],
    config: PipelineConfig,
    *,
    work_width: int = _MOTION_WIDTH,
) -> np.ndarray:
    """Approximate ``(dx, dy)`` in *full-frame* pixels between consecutive frames.

    Returns an ``(n - 1, 2)`` array. Frames that fail to load contribute a zero step
    rather than aborting the run — a single unreadable jpg should not kill a capture.
    """
    n = len(frames)
    if n < 2:
        return np.zeros((0, 2), dtype=np.float64)

    workers = config.n_workers if config.n_workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    grays = _load_motion_grays(frames, work_width, workers)
    motion = np.zeros((n - 1, 2), dtype=np.float64)
    hann: np.ndarray | None = None
    prev = None if grays[0] is None else grays[0].astype(np.float32)
    for k in range(1, n):
        cur = None if grays[k] is None else grays[k].astype(np.float32)
        grays[k - 1] = None  # release as we go; only the pair in hand is needed
        if prev is None or cur is None or prev.shape != cur.shape:
            prev = cur if cur is not None else prev
            continue
        if hann is None or hann.shape != prev.shape:
            hann = cv2.createHanningWindow((prev.shape[1], prev.shape[0]), cv2.CV_32F)
        (dx, dy), _response = cv2.phaseCorrelate(prev, cur, hann)
        scale = float(frames[k].width) / float(prev.shape[1]) if prev.shape[1] else 1.0
        motion[k - 1] = (dx * scale, dy * scale)
        prev = cur

    # The FFT is cyclic, so a shift beyond half the working width aliases. Anything that
    # large is a scene cut or a decoding hiccup, not real motion; clamping it to zero is
    # safer than integrating a bogus 180-degree jump into the cumulative track.
    limit = 0.45 * float(max(f.width for f in frames))
    bad = np.abs(motion) > limit
    if bad.any():
        log.warning("estimate_frame_motion.clamped", n_clamped=int(bad.any(axis=1).sum()))
        motion[bad] = 0.0
    return motion


def motion_to_angles(
    motion: np.ndarray,
    intr: Intrinsics,
    *,
    pitch_correction: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel shifts to cumulative ``(yaw, pitch)`` tracks in degrees.

    Both returned arrays have length ``len(motion) + 1`` and start at zero, so they index
    in step with the frame list.
    """
    motion = np.asarray(motion, dtype=np.float64).reshape(-1, 2)
    focal = float(intr.focal)
    yaw_steps_flat = -np.degrees(np.arctan2(motion[:, 0], focal))
    pitch_steps = np.degrees(np.arctan2(motion[:, 1], focal))
    cum_pitch = np.concatenate([[0.0], np.cumsum(pitch_steps)])
    # ``pitch_correction`` divides each yaw step by cos(pitch), on the argument that a
    # pitched camera sees a full turn as fewer pixels. It is OFF, and measured to stay off:
    # on the reference three-row capture the flat track proposes every one of the 307 pairs
    # the matcher later verified, while the corrected track misses 89 of them -- all the
    # cross-row pairs -- and strands the entire down row (34 keyframes dropped instead of
    # 4). The phase-correlation pitch it would divide by is itself biased by the tilts
    # between rows (measured +10 deg on the up row), and the spacing of the tilted rows
    # that the flat track produces is what keeps the pose graph connected. Kept as an
    # option for a track whose pitch comes from a gyroscope.
    if pitch_correction:
        cos_pitch = np.maximum(np.cos(np.radians(cum_pitch[:-1])), 0.3)
        yaw_steps = yaw_steps_flat / cos_pitch
    else:
        yaw_steps = yaw_steps_flat
    cum_yaw = np.concatenate([[0.0], np.cumsum(yaw_steps)])
    return cum_yaw, cum_pitch


# ======================================================================================
# Sweep pattern
# ======================================================================================


def pitch_clusters(
    cum_yaw: np.ndarray,
    cum_pitch: np.ndarray,
    *,
    min_yaw_span_deg: float = _ROW_MIN_YAW_SPAN_DEG,
) -> list[np.ndarray]:
    """Group frame indices into pitch "rows".

    A histogram rather than a gap-split, because the transitions between rows are real
    samples: a gap-based clustering would be bridged by the handful of frames recorded
    while the user tilted from one row to the next. Density, on the other hand, cleanly
    separates the plateaus the user dwelt on from the ramps between them.

    Only clusters that actually sweep ``min_yaw_span_deg`` of yaw are returned; the rest
    are noise. Frames belonging to no returned cluster are attached to the nearest one so
    every frame stays selectable.
    """
    cum_pitch = np.asarray(cum_pitch, dtype=np.float64)
    cum_yaw = np.asarray(cum_yaw, dtype=np.float64)
    n = cum_pitch.size
    if n == 0:
        return []
    span = float(cum_pitch.max() - cum_pitch.min())
    if n < 4 or span < _ROW_SEPARATION_DEG:
        return [np.arange(n)]

    n_bins = max(12, min(64, n // 3))
    counts, edges = np.histogram(cum_pitch, bins=n_bins)
    threshold = max(1.0, 0.25 * float(counts.max()))
    dense = counts >= threshold

    groups: list[tuple[float, float]] = []
    start: int | None = None
    for b in range(n_bins):
        if dense[b] and start is None:
            start = b
        elif not dense[b] and start is not None:
            groups.append((edges[start], edges[b]))
            start = None
    if start is not None:
        groups.append((edges[start], edges[n_bins]))
    if not groups:
        return [np.arange(n)]

    # Merge groups whose centres are closer than a row separation.
    merged: list[tuple[float, float]] = [groups[0]]
    for lo, hi in groups[1:]:
        plo, phi = merged[-1]
        if (0.5 * (lo + hi)) - (0.5 * (plo + phi)) < _ROW_SEPARATION_DEG:
            merged[-1] = (plo, hi)
        else:
            merged.append((lo, hi))

    centres = np.array([0.5 * (lo + hi) for lo, hi in merged])
    assign = np.argmin(np.abs(cum_pitch[:, None] - centres[None, :]), axis=1)

    clusters: list[np.ndarray] = []
    for c in range(centres.size):
        idx = np.flatnonzero(assign == c)
        if idx.size == 0:
            continue
        yaw_span = float(cum_yaw[idx].max() - cum_yaw[idx].min())
        if yaw_span >= min_yaw_span_deg:
            clusters.append(idx)
    if not clusters:
        return [np.arange(n)]

    # Re-attach frames that fell in a rejected cluster to the nearest accepted row.
    accepted_centres = np.array([float(cum_pitch[idx].mean()) for idx in clusters])
    owned = np.zeros(n, dtype=bool)
    for idx in clusters:
        owned[idx] = True
    orphans = np.flatnonzero(~owned)
    if orphans.size:
        nearest = np.argmin(np.abs(cum_pitch[orphans, None] - accepted_centres[None, :]), axis=1)
        clusters = [
            np.sort(np.concatenate([idx, orphans[nearest == c]]))
            for c, idx in enumerate(clusters)
        ]
    return clusters


def detect_sweep_pattern(cum_yaw: np.ndarray, cum_pitch: np.ndarray) -> SweepPattern:
    """MULTI_ROW when at least two well-separated pitch rows each sweep >= 180 deg."""
    clusters = pitch_clusters(cum_yaw, cum_pitch)
    if len(clusters) < 2:
        return SweepPattern.SINGLE_ROW
    cum_pitch = np.asarray(cum_pitch, dtype=np.float64)
    cum_yaw = np.asarray(cum_yaw, dtype=np.float64)
    centres = sorted(float(cum_pitch[idx].mean()) for idx in clusters)
    if min(b - a for a, b in zip(centres, centres[1:], strict=False)) < _ROW_SEPARATION_DEG:
        return SweepPattern.SINGLE_ROW
    n_rows = sum(
        1
        for idx in clusters
        if float(cum_yaw[idx].max() - cum_yaw[idx].min()) >= _ROW_MIN_YAW_SPAN_DEG
    )
    pattern = SweepPattern.MULTI_ROW if n_rows >= 2 else SweepPattern.SINGLE_ROW
    log.info("detect_sweep_pattern", n_clusters=len(clusters), n_rows=n_rows, pattern=pattern.value)
    return pattern


# ======================================================================================
# Keyframe selection
# ======================================================================================


#: Angular speed above which a frame is refused as a keyframe while a slower one is
#: available in the same bin. At a phone's 30-60 ms video shutter, 12 deg/s is 8-16 px of
#: motion smear at 22 px/deg; the row transitions of a three-row sweep run at 20-60 deg/s
#: and were being chosen as keyframes because a global sharpness rank cannot tell a
#: smeared frame of a textured wall from a sharp frame of a plain one.
_SPEED_VETO_DEG_PER_S = 12.0


def frame_speeds(frames: list[Frame], motion: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Per-frame angular speed in degrees per second from the phase-correlation track."""
    n = len(frames)
    if n < 2:
        return np.zeros(n)
    motion = np.asarray(motion, dtype=np.float64).reshape(-1, 2)
    step_deg = np.degrees(np.arctan2(np.hypot(motion[:, 0], motion[:, 1]), float(intr.focal)))
    ts = np.array([float(f.timestamp) for f in frames], dtype=np.float64)
    dt = np.maximum(np.diff(ts), 1e-3)
    rate = step_deg / dt
    speed = np.zeros(n)
    speed[1:] = rate
    speed[:-1] = np.maximum(speed[:-1], rate)
    return speed


def _select_along_track(
    order: np.ndarray,
    cum_yaw: np.ndarray,
    quality: np.ndarray,
    sharp_ok: np.ndarray,
    bin_deg: float,
    speed_ok: np.ndarray | None = None,
) -> list[int]:
    """Walk one row's track, emitting the best frame of every ``bin_deg`` of yaw.

    ``order`` holds the frame indices of this row in capture order. Two details make the
    spacing come out even:

    * The anchor is reset to the *chosen* frame rather than to the bin edge, so choosing a
      frame early in a bin does not shift every later keyframe.
    * Ties on quality are broken towards the frame nearest the *ideal* spacing
      (``bin_deg`` from the anchor), not the frame nearest the anchor. Breaking towards
      the anchor collapses the spacing: each bin would keep picking its earliest good
      frame and the walk would inch forward at a fraction of a bin per step.

    The first keyframe is seeded from the opening half-bin instead of the first full bin,
    so the panorama starts near where the user actually started rather than a bin later.

    A measured caveat, recorded because it looks like a defect and acting on it makes
    things worse. ``quality`` is a *rank* over the capture, so two frames essentially never
    tie and the spacing tie-break above is unreachable on real footage: the walk picks
    whichever frame in the bin is sharpest, and steps come out at 1.5-28 deg against a
    23.7 deg bin on the reference realistic capture. Making spacing a weighted term instead
    does even the steps out (6.0/21.0/28.2) — and *halves* connectivity, because the
    resulting keyframes sit exactly at the overlap limit with no redundancy: adjacent pairs
    verifying fell from 65 to 17 and the largest pose-graph component from 10/34 to 5/20.
    The uneven oversampling is load-bearing. Any future change here has to raise
    ``target_overlap`` in step, and be measured on connectivity rather than on step spread.
    """
    m = int(order.size)
    if m == 0:
        return []

    def _best(positions: list[int], anchor: float, target: float) -> int:
        # Prefer sharp AND slow frames; fall back to sharp ones, then to the whole bucket
        # rather than leaving a gap.
        sharp = [q for q in positions if sharp_ok[order[q]]]
        if speed_ok is not None:
            pool = [q for q in sharp if speed_ok[order[q]]] or sharp or [
                q for q in positions if speed_ok[order[q]]
            ] or positions
        else:
            pool = sharp or positions
        return max(
            pool,
            key=lambda q: (
                float(quality[order[q]]),
                -abs(abs(float(cum_yaw[order[q]]) - anchor) - target),
            ),
        )

    start = float(cum_yaw[order[0]])
    head = [q for q in range(m) if abs(float(cum_yaw[order[q]]) - start) <= 0.5 * bin_deg] or [0]
    first = _best(head, start, 0.0)
    chosen: list[int] = [int(order[first])]
    anchor = float(cum_yaw[order[first]])
    p = first + 1

    while p < m:
        bucket: list[int] = []  # positions within ``order``
        span = 0.0
        while p < m:
            bucket.append(p)
            p += 1
            span = abs(float(cum_yaw[order[bucket[-1]]]) - anchor)
            if span >= bin_deg:
                break
        if span < 0.5 * bin_deg:
            # A trailing remainder shorter than half a bin is already inside the previous
            # keyframe's field of view; emitting it would only add a near-duplicate.
            break
        best = _best(bucket, anchor, bin_deg)
        chosen.append(int(order[best]))
        anchor = float(cum_yaw[order[best]])
        # Resume just after the frame we kept so the next bin starts from the new anchor.
        p = best + 1
    return chosen


def _trim_to_budget(
    chosen: list[int],
    cum_yaw: np.ndarray,
    quality: np.ndarray,
    bin_deg: float,
    max_keyframes: int,
) -> list[int]:
    """Drop the least valuable keyframes until the budget is met.

    Value combines the frame's own quality with how much angular gap its removal would
    open: dropping a frame that sits between two others 5 degrees apart is nearly free,
    dropping one that bridges a 40-degree jump is not.
    """
    chosen = list(chosen)
    while len(chosen) > max_keyframes and len(chosen) > 2:
        costs = []
        for pos in range(1, len(chosen) - 1):
            gap = abs(float(cum_yaw[chosen[pos + 1]]) - float(cum_yaw[chosen[pos - 1]]))
            costs.append((float(quality[chosen[pos]]) + gap / max(bin_deg, 1e-6), pos))
        _, worst = min(costs)
        chosen.pop(worst)
    return chosen[:max_keyframes]


def _fill_to_minimum(
    chosen: list[int],
    candidates: np.ndarray,
    quality: np.ndarray,
    min_keyframes: int,
) -> list[int]:
    """Top up a too-small selection with the best frames not already taken."""
    taken = set(chosen)
    spare = [int(i) for i in candidates if int(i) not in taken]
    spare.sort(key=lambda i: float(quality[i]), reverse=True)
    for i in spare:
        if len(chosen) >= min_keyframes:
            break
        chosen.append(i)
        taken.add(i)
    return sorted(chosen)


def select_keyframes(
    frames: list[Frame],
    config: PipelineConfig,
    *,
    motion: np.ndarray | None = None,
    intr: Intrinsics | None = None,
) -> tuple[list[Frame], SweepPattern, float]:
    """Pick the keyframes that enter the solve.

    ``motion`` and ``intr`` are injection points: the pipeline lets them default (motion
    measured from the images, intrinsics from the 67-degree phone prior), while tests and
    the synthetic harness supply a known track. Returns the keyframes in capture order,
    the detected sweep pattern, and the total yaw span in degrees.
    """
    if not frames:
        raise StitchError("no frames to select keyframes from")
    if len(frames) == 1:
        return list(frames), SweepPattern.UNKNOWN, 0.0

    if intr is None:
        intr = Intrinsics.from_hfov(frames[0].width, frames[0].height, DEFAULT_HFOV_DEG)
    if motion is None:
        motion = estimate_frame_motion(frames, config)
    motion = np.asarray(motion, dtype=np.float64).reshape(-1, 2)
    if motion.shape[0] != len(frames) - 1:
        raise StitchError(
            f"motion track length {motion.shape[0]} does not match {len(frames)} frames"
        )

    cum_yaw, cum_pitch = motion_to_angles(motion, intr, pitch_correction=False)
    pattern = detect_sweep_pattern(cum_yaw, cum_pitch)
    yaw_span = float(cum_yaw.max() - cum_yaw.min())

    quality = np.array([f.quality for f in frames], dtype=np.float64)
    sharpness = np.array([f.sharpness for f in frames], dtype=np.float64)
    sharp_cut = float(np.percentile(sharpness, float(config.min_sharpness_percentile)))
    sharp_ok = sharpness >= sharp_cut

    bin_deg = max(1.0, float(intr.hfov_deg) * (1.0 - float(config.target_overlap)))

    # Motion veto. Relative to the sweep's own pace as well as absolute, so a capture shot
    # slowly throughout still prefers its calmest moments.
    speed = frame_speeds(frames, motion, intr)
    speed_ok = speed <= max(_SPEED_VETO_DEG_PER_S, 1.5 * float(np.median(speed)))

    if pattern is SweepPattern.MULTI_ROW:
        rows = pitch_clusters(cum_yaw, cum_pitch)
    else:
        rows = [np.arange(len(frames))]

    chosen: list[int] = []
    per_row_budget = max(2, int(config.max_keyframes // max(1, len(rows))))
    for row in rows:
        row_chosen = _select_along_track(
            np.sort(row), cum_yaw, quality, sharp_ok, bin_deg, speed_ok=speed_ok
        )
        if len(row_chosen) > per_row_budget:
            row_chosen = _trim_to_budget(row_chosen, cum_yaw, quality, bin_deg, per_row_budget)
        chosen.extend(row_chosen)

    chosen = sorted(set(chosen))
    if len(chosen) > config.max_keyframes:
        chosen = _trim_to_budget(chosen, cum_yaw, quality, bin_deg, int(config.max_keyframes))
    if len(chosen) < min(int(config.min_keyframes), len(frames)):
        chosen = _fill_to_minimum(
            chosen, np.arange(len(frames)), quality, min(int(config.min_keyframes), len(frames))
        )

    keyframes = [frames[i] for i in chosen]
    log.info(
        "select_keyframes",
        n_in=len(frames),
        n_out=len(keyframes),
        pattern=pattern.value,
        bin_deg=round(bin_deg, 2),
        yaw_span_deg=round(yaw_span, 1),
        n_rows=len(rows),
    )
    return keyframes, pattern, math.fabs(yaw_span)
