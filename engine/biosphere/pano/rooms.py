"""Cutting a walkthrough into the stretches that were filmed from one place.

When :mod:`biosphere.pano.validate` returns ``walking``, the capture cannot become a
panorama — but it is not worthless. An operator walking a flat films each room in turn,
and the useful question stops being "where is the sphere" and becomes "what did you
actually film, and which parts of it were worth filming". This module answers that, so
the API can eventually say *"we found five rooms; here is how to re-shoot each"* instead
of a bare failure.

What a segment actually is
--------------------------
**A segment is a contiguous stretch of unbroken scene continuity, not a room.** The
signal is the inlier ratio between neighbouring samples: it collapses when the operator
walks through a doorway, because almost nothing visible before the doorway is still
visible after it. That is a real and useful cut, and it is what the reference analysis
(``scripts/analyse_walkthrough.py``) found five of in the motivating walkthrough.

But the same collapse happens for reasons that have nothing to do with doorways. Measured
on the *other* motivating capture — a user turning on the spot in a single bedroom for 101
seconds — scene continuity goes to **zero inliers** three separate times, for 2 to 4
seconds each, when the phone is whipped round fast enough to blur out completely. Nothing
available at this stage of the pipeline separates those two causes: the residual, the
turn rate and the flow all look the same going into a whip-pan as into a doorway, because
in both cases the last thing that happens before tracking dies is fast motion and rising
error. So this module does not claim to tell them apart, the type is documented as what it
measures rather than what one hopes it means, and callers must treat the segment count as
"stretches of continuous footage" rather than "rooms in the flat".

What is *not* claimed
---------------------
Nothing here reconstructs anything. :attr:`RoomSegment.best_pivot` finds the sub-interval
of a segment where rotation best explains the motion — which is the only part of a
walkthrough that a rotation-only stitcher could ever use — and then labels it with the
*same* :func:`~biosphere.pano.validate.classify_purity` the validator uses, so a window
that would still be refused as a capture is still labelled ``walking`` here. On the
motivating walkthrough every segment's best window comes back that way, and that is the
correct answer: a walk does not contain a hidden panorama, and a field named
``best_pivot`` must not be allowed to imply that it does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from .types import Frame, NullProgress, PipelineConfig, ProgressReporter
from .validate import (
    GOOD_ROT_RESIDUAL_PX,
    SAMPLE_FPS,
    CaptureVerdict,
    PuritySample,
    PurityScan,
    classify_purity,
    purity_stats,
    scan_rotational_purity,
)

__all__ = [
    "PivotWindow",
    "RoomSegment",
    "segment_rooms",
    "MIN_SEGMENT_S",
    "MIN_PIVOT_S",
    "CUT_ENTER_FRACTION",
    "CUT_EXIT_FRACTION",
]

log = structlog.get_logger(__name__)


# ======================================================================================
# Cut detection
# ======================================================================================

#: Width of the moving median applied to the continuity trace before anything is decided.
#: Three samples is the smallest window that removes an isolated dropout without blunting
#: a real collapse — and an isolated dropout is exactly what a single motion-blurred frame
#: produces, which the brief is explicit must not split a room on its own.
SMOOTH_SAMPLES = 3

#: A cut *begins* when smoothed continuity falls below this fraction of the capture's own
#: median, and *ends* only when it climbs back above :data:`CUT_EXIT_FRACTION` of it. Two
#: thresholds rather than one is what stops a trace hovering near the boundary from
#: chattering into a dozen spurious segments — the signal has to genuinely recover, not
#: merely stop being terrible.
#:
#: Both are fractions of the capture's own median rather than absolute numbers because the
#: baseline is not comparable between captures: the bedroom turn runs at a median inlier
#: ratio of 0.63 and the walkthrough at 0.26, so any absolute threshold that catches the
#: walkthrough's doorways cuts the bedroom to ribbons, and any that spares the bedroom
#: sees no doorways at all.
#:
#: **What this costs, measured.** Requiring a genuine recovery means a cut stays open for
#: as long as continuity keeps hovering, and on the motivating walkthrough one does — an
#: eleven-second stretch around 50-61 s never gets two consecutive samples clear of the
#: exit threshold. The five segments therefore cover 59 s of the 82-second capture (72 %)
#: where the reference analysis's simpler rule covered 78 s (95 %). The footage dropped is
#: footage whose continuity never recovered, which on a walkthrough is not footage anyone
#: could stitch — but it is a real difference in behaviour rather than a free improvement,
#: and lowering :data:`CUT_EXIT_FRACTION` trades it back for more, shorter segments. It is
#: left where it is because there is no ground truth for where that flat's rooms actually
#: are, and tuning against a single file to recover coverage would be guessing.
CUT_ENTER_FRACTION = 0.35
CUT_EXIT_FRACTION = 0.70

#: ...clamped into this band, so that a pathologically bad capture does not end up with a
#: threshold near zero (and therefore no cuts at all) and a pristine one does not end up
#: with a threshold so high that ordinary variation reads as a doorway.
CUT_THRESHOLD_FLOOR = 0.05
CUT_THRESHOLD_CEILING = 0.35

#: How long continuity must stay collapsed before it counts as a cut, and how long it must
#: stay recovered before the next segment starts. A second of sustained collapse is longer
#: than any single blurred frame and longer than a glance at the floor and back, both of
#: which the brief requires survive intact; walking through a doorway takes longer than
#: that, and every cut the reference analysis found in the motivating walkthrough lasts at
#: least this long.
MIN_CUT_S = 1.0
MIN_RECOVER_S = 0.5

#: Segments shorter than this are dropped rather than reported. Below a few seconds there
#: is not enough footage to say anything useful about a room, and the fragments that fall
#: out either side of a doorway transit are exactly this length.
MIN_SEGMENT_S = 4.0

#: Shortest interval that can be called a pivot window. Three seconds is the reference
#: analysis's own figure and about the least a person can turn through a useful arc in.
MIN_PIVOT_S = 3.0


def _moving_median(values: np.ndarray, width: int) -> np.ndarray:
    """Odd-width moving median with edge replication."""
    width = max(1, int(width) | 1)
    if width == 1 or values.size == 0:
        return np.asarray(values, dtype=np.float64)
    pad = width // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=-1)


def _continuity(samples: list[PuritySample]) -> np.ndarray:
    """Scene continuity per sample: the inlier ratio, and zero where nothing verified.

    A pair that failed outright is not missing data — it is the strongest possible
    evidence that the scene changed — so it contributes a zero rather than a gap.
    """
    return np.array(
        [s.inlier_ratio if s.usable else 0.0 for s in samples], dtype=np.float64
    )


def _find_cuts(continuity: np.ndarray, spacing_s: float) -> list[tuple[int, int]]:
    """Locate sustained collapses as ``(start, end)`` half-open sample ranges.

    A two-state machine with hysteresis: inside a segment, a run of at least ``MIN_CUT_S``
    below the enter threshold opens a cut *at the start of that run*; inside a cut, a run
    of at least ``MIN_RECOVER_S`` above the exit threshold closes it *at the start of that
    run*. Anchoring each transition at the beginning of the run that justified it — rather
    than at the sample that happened to trip it — keeps the boundary where the scene
    actually changed instead of a second late.
    """
    n = continuity.size
    if n == 0:
        return []
    smooth = _moving_median(continuity, SMOOTH_SAMPLES)
    baseline = float(np.median(smooth))
    enter = float(np.clip(CUT_ENTER_FRACTION * baseline, CUT_THRESHOLD_FLOOR, CUT_THRESHOLD_CEILING))
    exit_ = max(float(CUT_EXIT_FRACTION * baseline), enter * 1.5)

    min_cut = max(2, int(round(MIN_CUT_S / max(spacing_s, 1e-6))))
    min_recover = max(1, int(round(MIN_RECOVER_S / max(spacing_s, 1e-6))))

    cuts: list[tuple[int, int]] = []
    in_cut = False
    run_start: int | None = None
    cut_start = 0
    for k in range(n):
        if not in_cut:
            if smooth[k] < enter:
                run_start = k if run_start is None else run_start
                if k - run_start + 1 >= min_cut:
                    in_cut, cut_start, run_start = True, run_start, None
            else:
                run_start = None
        else:
            if smooth[k] >= exit_:
                run_start = k if run_start is None else run_start
                if k - run_start + 1 >= min_recover:
                    cuts.append((cut_start, run_start))
                    in_cut, run_start = False, None
            else:
                run_start = None
    if in_cut:
        cuts.append((cut_start, n))
    log.info(
        "rooms.find_cuts",
        n_samples=n,
        baseline=round(baseline, 3),
        enter=round(enter, 3),
        exit=round(exit_, 3),
        min_cut_samples=min_cut,
        n_cuts=len(cuts),
    )
    return cuts


# ======================================================================================
# Results
# ======================================================================================


@dataclass(slots=True)
class PivotWindow:
    """The stretch of a segment that rotation explains best.

    ``verdict`` comes from the same :func:`~biosphere.pano.validate.classify_purity` the
    capture validator uses, so it means exactly what it means there: only ``good`` and
    ``degraded`` describe something a rotation-only stitcher could work with, and
    ``walking`` means this window is *not* a panorama waiting to happen either.
    """

    start_s: float
    end_s: float
    median_rot_residual_px: float
    median_free_residual_px: float
    yaw_span_deg: float
    n_samples: int
    verdict: CaptureVerdict
    #: True when no window of at least :data:`MIN_PIVOT_S` met the stitchable bar and this
    #: is merely the least-bad interval available. Reporting it anyway is useful (it says
    #: where to re-shoot from); pretending it is stitchable would not be.
    is_fallback: bool = False

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "duration_s": round(self.duration_s, 2),
            "median_rot_residual_px": round(self.median_rot_residual_px, 2),
            "median_free_residual_px": round(self.median_free_residual_px, 2),
            "yaw_span_deg": round(self.yaw_span_deg, 1),
            "n_samples": self.n_samples,
            "verdict": self.verdict.value,
            "is_fallback": self.is_fallback,
        }


@dataclass(slots=True)
class RoomSegment:
    """One contiguous stretch of unbroken scene continuity.

    Usually one room of a walkthrough — but see the module docstring: a whip-pan that
    blinds the tracker for a couple of seconds cuts a single room in two, and nothing
    measurable at this stage tells that apart from a doorway.
    """

    index: int
    start_s: float
    end_s: float
    #: Positions in the *extracted frame pool* that was scanned — i.e. indices into the
    #: ``frames`` list handed to :func:`segment_rooms`, not frame numbers in the source
    #: video and not keyframe indices, neither of which exists yet at this stage.
    start_frame: int
    end_frame: int
    n_samples: int
    n_usable_samples: int
    median_rot_residual_px: float
    median_free_residual_px: float
    residual_ratio: float
    median_inlier_ratio: float
    total_turn_deg: float
    yaw_span_deg: float
    stitchable: CaptureVerdict
    stitchable_reason: str
    best_pivot: PivotWindow | None = None
    samples: list[PuritySample] = field(default_factory=list, repr=False, compare=False)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def as_dict(self) -> dict[str, Any]:
        def _num(v: float) -> float | None:
            return None if not math.isfinite(v) else round(float(v), 4)

        return {
            "index": self.index,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "duration_s": round(self.duration_s, 2),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "n_samples": self.n_samples,
            "n_usable_samples": self.n_usable_samples,
            "median_rot_residual_px": _num(self.median_rot_residual_px),
            "median_free_residual_px": _num(self.median_free_residual_px),
            "residual_ratio": _num(self.residual_ratio),
            "median_inlier_ratio": round(self.median_inlier_ratio, 3),
            "total_turn_deg": round(self.total_turn_deg, 1),
            "yaw_span_deg": round(self.yaw_span_deg, 1),
            "stitchable": self.stitchable.value,
            "stitchable_reason": self.stitchable_reason,
            "best_pivot": None if self.best_pivot is None else self.best_pivot.as_dict(),
        }


# ======================================================================================
# Pivot windows
# ======================================================================================


def _best_pivot(samples: list[PuritySample], spacing_s: float) -> PivotWindow | None:
    """Longest interval of this segment over which rotation explains the motion.

    "Explains" is judged on a moving median of the per-sample rotation-only residual, not
    on the raw samples: a single bad pair in the middle of an otherwise clean turn should
    not chop the window in half, and a single *good* pair in the middle of a walk should
    not extend one. The window is then the longest run where that smoothed residual stays
    under :data:`~biosphere.pano.validate.GOOD_ROT_RESIDUAL_PX`.

    When nothing reaches :data:`MIN_PIVOT_S` at that bar, the least-bad interval of the
    minimum length is returned instead with ``is_fallback`` set and whatever verdict its
    numbers earn — which on a real walkthrough is ``walking``, every time. That is the
    honest answer: it says where the operator came closest to standing still, and it does
    not pretend a panorama is hiding there.
    """
    if sum(1 for s in samples if s.usable) < 2:
        return None
    min_len = max(2, int(round(MIN_PIVOT_S / max(spacing_s, 1e-6))))
    # Samples that measured nothing count as infinitely bad rather than being dropped, so
    # that a window cannot quietly bridge a blackout and report the average of the footage
    # either side of it as if it were one continuous pivot.
    raw = np.array(
        [s.rot_residual_px if s.usable else math.inf for s in samples], dtype=np.float64
    )
    res = _moving_median(raw, SMOOTH_SAMPLES)

    def _window(a: int, b: int, fallback: bool) -> PivotWindow:
        chunk = samples[a:b]
        stats = purity_stats(chunk)
        verdict, _reason = classify_purity(stats)
        return PivotWindow(
            start_s=float(chunk[0].time_s),
            end_s=float(chunk[-1].time_s + chunk[-1].dt_s),
            median_rot_residual_px=stats["median_rot_residual_px"],
            median_free_residual_px=stats["median_free_residual_px"],
            yaw_span_deg=stats["yaw_span_deg"],
            n_samples=len(chunk),
            verdict=verdict,
            is_fallback=fallback,
        )

    # Longest run under the stitchable bar.
    best: tuple[int, int] | None = None
    start: int | None = None
    for k in range(res.size + 1):
        clean = k < res.size and res[k] <= GOOD_ROT_RESIDUAL_PX
        if clean and start is None:
            start = k
        elif not clean and start is not None:
            if k - start >= min_len and (best is None or (k - start) > (best[1] - best[0])):
                best = (start, k)
            start = None
    if best is not None:
        return _window(best[0], best[1], fallback=False)

    # Nothing qualified: report the least-bad interval of the minimum length. Scored on the
    # *raw* residual, not the smoothed one, so that the window reported is the window
    # measured — scoring on one series and reporting the median of another let a fallback
    # come back worse than the segment it was chosen from.
    if raw.size < min_len:
        return _window(0, raw.size, fallback=True)
    scores = [
        (float(np.median(raw[i : i + min_len])), i) for i in range(raw.size - min_len + 1)
    ]
    _score, i = min(scores)
    return _window(i, i + min_len, fallback=True)


# ======================================================================================
# Entry point
# ======================================================================================


def segment_rooms(
    frames: list[Frame],
    config: PipelineConfig,
    *,
    scan: PurityScan | None = None,
    progress: ProgressReporter | None = None,
) -> list[RoomSegment]:
    """Split a capture into the stretches that were filmed without a break in continuity.

    Costs a full-resolution purity scan — every sample at
    :data:`~biosphere.pano.validate.SAMPLE_FPS`, not the capped subsample the validator
    uses — because a segmentation cannot find a doorway it never looked at. That is
    roughly fifteen seconds on a ninety-second capture, which is why this runs only on
    the opt-in path (``BIOSPHERE_SEGMENT_ROOMS=1``) and only for a capture that has
    already been judged unstitchable. Pass ``scan`` to reuse one already taken.

    Returns the surviving segments in capture order; segments shorter than
    :data:`MIN_SEGMENT_S` are dropped, so the result may be empty and the returned
    durations need not tile the capture.
    """
    progress = progress if progress is not None else NullProgress()
    if len(frames) < 2:
        return []
    if scan is None:
        scan = scan_rotational_purity(
            frames, config, sample_fps=SAMPLE_FPS, max_samples=None, progress=progress
        )
    samples = scan.samples
    if not samples:
        return []

    spacing = scan.sample_spacing_s or (1.0 / SAMPLE_FPS)
    cuts = _find_cuts(_continuity(samples), spacing)

    # The surviving stretches are what the cuts leave behind.
    spans: list[tuple[int, int]] = []
    cursor = 0
    for a, b in cuts:
        if a > cursor:
            spans.append((cursor, a))
        cursor = b
    if cursor < len(samples):
        spans.append((cursor, len(samples)))

    segments: list[RoomSegment] = []
    for a, b in spans:
        chunk = samples[a:b]
        if not chunk:
            continue
        start_s = float(chunk[0].time_s)
        end_s = float(chunk[-1].time_s + chunk[-1].dt_s)
        if end_s - start_s < MIN_SEGMENT_S:
            continue
        stats = purity_stats(chunk)
        verdict, reason = classify_purity(stats)
        segments.append(
            RoomSegment(
                index=len(segments),
                start_s=start_s,
                end_s=end_s,
                start_frame=int(chunk[0].i),
                end_frame=int(chunk[-1].j),
                n_samples=len(chunk),
                n_usable_samples=int(stats["n_usable"]),
                median_rot_residual_px=stats["median_rot_residual_px"],
                median_free_residual_px=stats["median_free_residual_px"],
                residual_ratio=stats["residual_ratio"],
                median_inlier_ratio=stats["median_inlier_ratio"],
                total_turn_deg=stats["total_turn_deg"],
                yaw_span_deg=stats["yaw_span_deg"],
                stitchable=verdict,
                stitchable_reason=reason,
                best_pivot=_best_pivot(chunk, spacing),
                samples=chunk,
            )
        )

    log.info(
        "segment_rooms",
        n_samples=len(samples),
        n_cuts=len(cuts),
        n_segments=len(segments),
        segments=[
            f"{s.start_s:.1f}-{s.end_s:.1f}s turn={s.total_turn_deg:.0f} "
            f"rot={s.median_rot_residual_px:.1f}px {s.stitchable.value}"
            for s in segments
        ],
    )
    return segments
