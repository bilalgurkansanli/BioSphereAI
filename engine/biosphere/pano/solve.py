"""The geometry solver's entry point: capture in, solved camera orientations out.

This module owns the *order* of the pipeline and nothing else — every real decision lives
in :mod:`~biosphere.pano.extract`, :mod:`~biosphere.pano.select`,
:mod:`~biosphere.pano.features`, :mod:`~biosphere.pano.rotation` and
:mod:`~biosphere.pano.bundle`. Keeping the orchestration this thin is deliberate: it is
the layer the API and the synthetic harness both drive, so it must stay easy to read and
free of hidden state.

Failure policy
--------------
Two capture problems are unrecoverable, and both are the user's to fix, so both raise
rather than degrade:

``InsufficientCoverageError``
    The sweep never came close to a full turn (< 200 deg of yaw). Anything we produced
    would be a wide photo pretending to be a panorama.

``InsufficientOverlapError``
    The pose graph is disconnected — some keyframes share no verified features with the
    rest. Usually a fast pan across a blank wall, or a jump cut.

``WalkingCaptureError``
    The camera translated far more than it rotated — an operator walking a flat rather
    than turning on the spot. This one is decided *before* feature detection by
    :func:`~biosphere.pano.validate.assess_capture`, because it is cheap to measure and
    ruinously expensive to discover the hard way: the old behaviour was to spend the full
    two minutes and emit a mangled panorama with a mild warning.

Everything milder becomes a warning on :class:`~biosphere.pano.types.StitchMetrics`, so
the UI can say "this may look rough" while still delivering a panorama.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from .bundle import bundle_adjust, finalise_poses, loop_closure_stats
from .geometry import nearest_rotation, rodrigues, rotvec
from .devices import ResolvedDevice, resolve_device
from .extract import (
    VideoInfo,
    extract_frames,
    load_capture_sidecar,
    load_photos,
    photo_focal_prior_px,
    probe_video,
)
from .features import (
    candidate_pairs,
    connected_components,
    detect_features,
    match_pairs,
    repair_connectivity,
)
from .rotation import (
    MAX_HFOV_DEG,
    MIN_HFOV_DEG,
    estimate_focal,
    initial_poses,
    relative_rotations,
)
from .select import (
    DEFAULT_HFOV_DEG,
    estimate_frame_motion,
    motion_to_angles,
    select_keyframes,
)
from .types import (
    CameraPose,
    CaptureKind,
    Frame,
    InsufficientCoverageError,
    InsufficientOverlapError,
    StitchError,
    Intrinsics,
    MatchPair,
    NullProgress,
    PipelineConfig,
    ProgressReporter,
    SolveResult,
    StitchMetrics,
    SweepPattern,
)
from .validate import (
    CaptureAssessment,
    CaptureVerdict,
    assess_capture,
    segment_rooms_enabled,
)

__all__ = ["solve_capture", "MIN_YAW_SPAN_DEG", "WARN_YAW_SPAN_DEG"]

log = structlog.get_logger(__name__)

#: Below this the capture is not a panorama and we refuse it.
MIN_YAW_SPAN_DEG = 200.0

#: Below this it is a panorama, but one with a hole the renderer will have to fill.
WARN_YAW_SPAN_DEG = 300.0

#: Reprojection error above this means the geometry model is being strained — usually
#: parallax from a hand-held sweep that did not pivot about the lens.
WARN_REPROJECTION_PX = 2.0

#: 90th-percentile reprojection error over every correspondence above which the capture is
#: reported as parallax-limited: a rotation-only model cannot align it and the renderer's
#: local alignment is doing real work. 12 px at ~22 px/deg is half a degree.
WARN_PARALLAX_P90_PX = 12.0

#: A loop that closes worse than this genuinely degrades the delivered panorama.
#:
#: Calibrated rather than guessed. A known yaw drift was injected around the loop of a
#: solved nodal capture, the panorama re-rendered at each level, and the result scored
#: against the sphere it was rendered from:
#:
#: ===========  ==========  ============  =========
#: drift (deg)  robust_deg  ssim_horizon  psnr_horizon
#: ===========  ==========  ============  =========
#: 0.0          0.87        0.890         25.1
#: 1.0          0.46        0.881         24.4
#: 1.5          0.82        0.847         23.3
#: 2.0          1.27        0.793         22.4
#: 3.0          2.23        0.722         21.7
#: 6.0          5.18        0.672         20.2
#: ===========  ==========  ============  =========
#:
#: Two things to read off that. The statistic dips before it climbs, because this solve
#: was already about a degree "over-closed" and a small injected drift cancels it — and
#: the panorama agrees, scoring best at the same place, so the statistic is tracking
#: something real rather than wobbling. And the panorama is still at the quality bar while
#: the statistic is under ~1 deg, clearly degraded by the time it reaches 2.2.
#:
#: 2.0 therefore sits just under the first clearly-bad sample and more than doubles the
#: headroom over a healthy capture, which is what keeps it from crying wolf.
#:
#: Note the seam is *not* where this shows up: ``wrap_seam_score`` stayed between 0.75 and
#: 1.02 across the entire sweep, because the renderer chooses where to put the wrap column
#: and hides it. Loop error surfaces as misplacement across the whole panorama, not as a
#: line at the join, so it is fidelity that has to be calibrated against.
WARN_LOOP_RESIDUAL_DEG = 2.0

#: How many alternative video frames to try for each keyframe the pose graph stranded, and
#: how far from it to look for them, as a fraction of the selection bin.
#:
#: THIS IS WHAT SHOOTING VIDEO IS FOR. Keyframe selection picks the sharpest frame in every
#: bin of yaw, but it picks BEFORE anything has been matched, so "sharpest" is a guess about
#: which frame will link. When the guess is wrong the old pipeline had no recourse: the frame
#: was dropped, that direction went missing from the pose graph, and the panorama got a seam
#: or a bent line exactly there — on the reference capture, 17 keyframes at yaw 30-45, which
#: is precisely where the delivered panorama was worst.
#:
#: A video has thirty frames a second and the sweep is slow, so the same view was captured
#: many times over. Sharpness is not the only thing that decides whether a frame links —
#: rolling-shutter skew, a passing highlight, autofocus hunting mid-frame — so the next frame
#: along often matches where the chosen one could not. Four is enough to cover a hunt or a
#: blur burst without turning a failure into a long feature-detection stall; half a bin keeps
#: the alternate close enough that it still belongs to the same place in the sweep.
_ALTERNATES_PER_STRANDED = 4
_ALTERNATE_YAW_FRACTION = 0.5
#: How many surviving keyframes an alternate is offered to. The nearest few in yaw are the
#: ones it can plausibly overlap; trying the whole graph would cost time to no purpose.
_RESCUE_TARGETS = 4

#: Bridging a featureless stretch. A keyframe that shares no verified feature with
#: anything -- a blank wall, a bare ceiling corner -- is dropped by
#: :func:`_keep_largest_component`, and the render then covers that direction with whatever
#: the neighbouring row's frame edges reach. Measured on the reference capture: a
#: 62-degree hole in the up row (source frames 421-467 all see plain wall and cornice, SIFT
#: verifies not one pair among them, not even consecutive frames a quarter-second apart)
#: where the cornice came only from the top edge of the level row and was washed out at
#: the wall corner. The frames exist and are sharp; they simply carry nothing a
#: descriptor can match. So they are placed by interpolation between the solved
#: keyframes that bracket them in the sweep -- the endpoints are exact, only the fraction
#: along the gap is estimated -- and join the RENDER only, never the solve.
#:
#: The fraction comes from the motion track when the track saw the gap, and from time
#: otherwise. The track cannot be trusted blindly here: on a plain wall the phase
#: correlation locks onto the cornice, which fixes pitch but not yaw (the aperture
#: problem), and on the reference capture it measured 47 degrees across a gap the solve
#: put at 62. A horizontal cornice is also why a few degrees of yaw error inside the gap
#: is invisible: it slides the frame along the line it is aligned to.
_BRIDGE_MAX_GAP_HFOV = 3.0
#: A row runs along YAW; a row CHANGE moves in pitch. So a gap whose yaw span exceeds its
#: pitch step is a hole inside a row, and one whose pitch step dominates is the sweep moving
#: to the next row, where there is nothing to interpolate.
#:
#: TWO THINGS THAT LOOK RIGHT AND ARE NOT, both measured on the reference capture. A fixed
#: pitch step (12 degrees) rejected a genuine 49-degree hole in the up row, because a
#: hand-held row is not level and its two brackets sat at +33 and +19 -- a 13.9 degree step,
#: against the 26.2 of the real row change beside it. And clustering the poses by pitch, the
#: way `render._pitch_rows` does, finds ONE row here rather than three: the operator kept
#: recording while tilting between rows, so the pitch distribution runs unbroken from -36 to
#: +34 with no empty band anywhere to split on. The direction of the gap survives both.
#:
#: The pitch step is still bounded, because no hole inside a row spans most of a frame
#: vertically; past that the two brackets cannot be showing the same band of wall.
_BRIDGE_MAX_PITCH_STEP_FRACTION = 0.5
#: Trust the track's fraction only when its yaw change across the gap is at least this
#: share of the solved change; below that it lost the wall and time is the better clock.
_BRIDGE_TRACK_MIN_SHARE = 0.6
_BRIDGE_MAX_FRAMES = 12
_BRIDGE_MAX_PER_GAP = 6

#: Fraction of keyframes the largest connected component must retain before the capture is
#: reported as fragmented.  This GRADES, it does not gate.
#:
#: It used to raise, and refusing here was the wrong trade.  The user has already spent two
#: minutes turning slowly through three rows; telling them at the end that none of it can be
#: used, and that the fix is to do the whole thing again, throws away work the pipeline can
#: in fact still do something with.  The largest component is stitched, ``fill_uncovered``
#: closes what the camera never reached, and ``covered_fraction`` reports honestly how much
#: of the sphere is real — so a fragment arrives as a visibly incomplete panorama the user
#: can judge for themselves rather than as an error message with nothing attached.
#:
#: The floors that remain are the ones where there is genuinely nothing to deliver: fewer
#: than three keyframes, a spanning set of pairs that does not exist, and a yaw span too
#: short to be a turn at all.
MIN_COMPONENT_FRACTION = 0.60

#: A pre-scan focal is only adopted as the prior when at least this many of its samples
#: were usable. :func:`~biosphere.pano.validate._solve_focal` takes the *median* residual
#: over the samples, so a handful of them is a fit to noise; below this the 67-degree
#: default is the more honest guess. The scan measures up to
#: :data:`~biosphere.pano.validate.MAX_SAMPLES` (72) pairs, so this is a low bar that only
#: a capture the scan could barely read at all will fail.
MIN_SCAN_SAMPLES_FOR_FOCAL = 8

#: Early gate applied to the *provisional* yaw span, before the focal is known. The
#: provisional figure is measured against the 67-degree prior and can be off by well over
#: 1.5x in either direction, so this is set low enough to only reject a capture that
#: barely moved, and the real check runs once ``estimate_focal`` has spoken.
PROVISIONAL_YAW_SPAN_FLOOR_DEG = 60.0

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ======================================================================================
# Helpers
# ======================================================================================


@contextmanager
def _timed(metrics: StitchMetrics, name: str):
    """Record a stage's wall-clock time even when the stage raises."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        metrics.stage_seconds[name] = time.perf_counter() - t0


def _classify_source(source: Path | str | Iterable[Path | str]) -> tuple[CaptureKind, list[Path]]:
    """Decide whether we were handed a video or a set of stills."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            photos = sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
            if photos:
                return CaptureKind.PHOTOS, photos
            raise InsufficientCoverageError(f"no images found in {path}")
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            return CaptureKind.PHOTOS, [path]
        return CaptureKind.VIDEO, [path]
    paths = [Path(p) for p in source]
    # Several non-image files are the SEGMENTS of one interrupted capture, not a photo set.
    # A phone splits a guided capture whenever the recorder is restarted mid-sweep, and the
    # pieces have to reach `extract_frames` together or they are read as stills and refused.
    if paths and all(p.suffix.lower() not in _IMAGE_SUFFIXES for p in paths):
        return CaptureKind.VIDEO, paths
    return CaptureKind.PHOTOS, paths


def _keyframe_size(keyframes: list[Frame]) -> tuple[int, int]:
    """The frame size the solve works in, taken as the modal size of the keyframes."""
    sizes = [(f.width, f.height) for f in keyframes]
    modal = max(set(sizes), key=sizes.count)
    return int(modal[0]), int(modal[1])


def _keep_largest_component(
    keyframes: list[Frame],
    pairs: list[MatchPair],
    prior_yaw: np.ndarray,
    metrics: StitchMetrics,
) -> tuple[list[Frame], list[MatchPair], np.ndarray, list[tuple[int, float]]]:
    """Restrict the solve to the largest connected component of the pose graph.

    A keyframe that shares no verified features with the rest of the capture cannot be
    oriented, and until now that ended the job. That is the wrong trade by a wide margin:
    an agent filming a lounge sweeps past a window, two or three frames blow out, and the
    right response is to drop those frames and stitch the other forty — not to hand back
    nothing. Whether what remains is *good enough* is a separate question, answered by the
    component-fraction and yaw-span checks rather than by mere connectivity.

    Indices are rewritten so the ``Frame.index == list position == CameraPose.frame_index``
    invariant survives: pairs are renumbered onto the surviving frames and any pair
    touching a dropped frame is discarded.
    """
    n = len(keyframes)
    components = connected_components(n, pairs)
    if len(components) <= 1:
        return keyframes, pairs, prior_yaw, []

    keep = components[0]
    keep_set = set(keep)
    dropped = [
        (int(keyframes[k]._cache.get("source_index", k)), float(prior_yaw[k]))
        for k in range(n)
        if k not in keep_set
    ]

    remap = {old: new for new, old in enumerate(keep)}
    surviving = [keyframes[k] for k in keep]
    for new, frame in enumerate(surviving):
        frame.index = new
    kept_pairs = []
    for pair in pairs:
        if pair.i in keep_set and pair.j in keep_set:
            pair.i, pair.j = remap[pair.i], remap[pair.j]
            kept_pairs.append(pair)
    kept_pairs.sort(key=lambda p: (p.i, p.j))

    log.warning(
        "solve_capture.dropped_unlinked_keyframes",
        n_dropped=len(dropped),
        n_kept=len(surviving),
        component_sizes=[len(c) for c in components],
        dropped_source_indices=[d[0] for d in dropped][:20],
    )
    return surviving, kept_pairs, prior_yaw[keep], dropped


def geom_wrap_deg(delta_deg: float) -> float:
    """A yaw difference brought back to (-180, 180]; a row can straddle the wrap."""
    return (float(delta_deg) + 180.0) % 360.0 - 180.0


def _forward_yaw_pitch_deg(R: np.ndarray) -> tuple[float, float]:
    fwd = np.asarray(R, dtype=np.float64).T @ np.array([0.0, 0.0, 1.0])
    return (
        math.degrees(math.atan2(float(fwd[0]), float(fwd[2]))),
        math.degrees(math.asin(max(-1.0, min(1.0, -float(fwd[1]))))),
    )


def _bridge_track_gaps(
    pool: list[Frame],
    keyframes: list[Frame],
    poses: list[np.ndarray],
    dropped: list[tuple[int, float]],
    match_yaw: np.ndarray,
    intr: Intrinsics,
    config: PipelineConfig,
) -> tuple[list[Frame], list[np.ndarray], list[dict[str, Any]]]:
    """Frames to add to the render across the gaps the dropped keyframes left behind.

    See ``_BRIDGE_MAX_GAP_HFOV``. Returns the frames, their interpolated rotations, and one
    diagnostic dict per bridged gap. Nothing here touches the solve: the extra views carry
    no pairs, so the bundle, the loop closure and every reported error are unchanged.
    """
    if not dropped or len(keyframes) < 2 or not pool:
        return [], [], []
    src_of = [int(f._cache.get("source_index", f.index)) for f in keyframes]
    order = sorted(range(len(keyframes)), key=lambda k: src_of[k])
    sorted_src = [src_of[k] for k in order]
    hfov = float(intr.hfov_deg)
    spacing = max(1.0, hfov * (1.0 - float(config.target_overlap)))
    taken = set(src_of)

    # Group the dropped keyframes by the pair of surviving keyframes that bracket them.
    gaps: dict[tuple[int, int], list[int]] = {}
    for s, _prior in dropped:
        pos = int(np.searchsorted(np.asarray(sorted_src), s))
        if pos <= 0 or pos >= len(order):
            continue
        a, b = order[pos - 1], order[pos]
        gaps.setdefault((a, b), []).append(int(s))

    extra_frames: list[Frame] = []
    extra_R: list[np.ndarray] = []
    report: list[dict[str, Any]] = []
    for (a, b), members in sorted(gaps.items(), key=lambda kv: src_of[kv[0][0]]):
        if len(extra_frames) >= _BRIDGE_MAX_FRAMES:
            break
        sa, sb = src_of[a], src_of[b]
        Ra, Rb = np.asarray(poses[a], dtype=np.float64), np.asarray(poses[b], dtype=np.float64)
        ya, pa = _forward_yaw_pitch_deg(Ra)
        yb, pb = _forward_yaw_pitch_deg(Rb)
        yaw_step = abs(float(geom_wrap_deg(yb - ya)))
        pitch_step = abs(pb - pa)
        if pitch_step >= yaw_step or pitch_step > _BRIDGE_MAX_PITCH_STEP_FRACTION * intr.vfov_deg:
            continue  # a row change, not a hole in a row
        rel = rotvec(Rb @ Ra.T)
        gap_deg = math.degrees(float(np.linalg.norm(rel)))
        if gap_deg > _BRIDGE_MAX_GAP_HFOV * hfov or gap_deg <= 1.2 * spacing:
            continue  # not a row gap / the footprints already meet
        n_fill = min(_BRIDGE_MAX_PER_GAP, max(1, int(math.ceil(gap_deg / spacing)) - 1))

        # Fraction along the gap for every pool frame between the brackets.
        cands = [k for k in range(sa + 1, sb) if 0 <= k < len(pool) and k not in taken]
        if not cands:
            continue
        solved_yaw = abs((yb - ya + 180.0) % 360.0 - 180.0)
        track_delta = float(match_yaw[sb] - match_yaw[sa]) if sb < match_yaw.size else 0.0
        use_track = solved_yaw > 1e-6 and abs(track_delta) >= _BRIDGE_TRACK_MIN_SHARE * solved_yaw
        if use_track:
            frac = {k: float((match_yaw[k] - match_yaw[sa]) / track_delta) for k in cands}
        else:
            frac = {k: (k - sa) / float(sb - sa) for k in cands}

        chosen: list[tuple[int, float]] = []
        for m in range(1, n_fill + 1):
            t = m / (n_fill + 1.0)
            half = 0.5 / (n_fill + 1.0)
            # Spacing first, sharpness second: the three frames nearest the target, the
            # sharpest of those. On a featureless wall sharpness hardly matters, while an
            # uneven spacing costs overlap with a neighbour.
            near = sorted((k for k in cands if k not in taken), key=lambda k: abs(frac[k] - t))
            near = [k for k in near[:3] if abs(frac[k] - t) <= half] or near[:1]
            if not near:
                continue
            k = max(near, key=lambda q: float(pool[q].quality))
            taken.add(k)
            chosen.append((k, float(min(0.98, max(0.02, frac[k])))))
        for k, t in chosen:
            frame = pool[k]
            frame._cache["source_index"] = int(k)
            # Marked so the RENDER can prefer measured coverage over interpolated coverage.
            # A bridged frame's orientation is an estimate between two solved ones; where a
            # measured frame also sees that direction, compositing both puts the same cornice
            # on the canvas at two slightly different pitches. See `render.select_render_views`.
            frame._cache["bridged"] = True
            extra_frames.append(frame)
            extra_R.append(nearest_rotation(rodrigues(rel * t) @ Ra))
        report.append(
            {
                "brackets": (int(sa), int(sb)),
                "gap_deg": round(gap_deg, 1),
                "dropped": members,
                "placed": [int(k) for k, _t in chosen],
                "fraction_source": "track" if use_track else "time",
                "track_delta_deg": round(track_delta, 1),
                "solved_delta_deg": round(solved_yaw, 1),
            }
        )
    return extra_frames, extra_R, report


def _measured_yaw_span(
    motion: np.ndarray,
    intr: Intrinsics,
    keyframes: list[Frame],
) -> float:
    """Re-measure the swept yaw using the *solved* focal instead of the initial guess.

    The motion track is in pixels and was always accurate; only the pixels-to-degrees
    conversion was wrong, because it used the 67-degree prior. Redoing that conversion is
    a handful of arctangents over data we already have, and it is what makes the coverage
    figure honest: on the reference 45-degree portrait capture the prior reported 576 deg
    for a 357-degree turn, which would have let a 220-degree sweep pass as a full circle.

    The span is measured over the *keyframes that survived*, since those are the only
    views the renderer will actually have.
    """
    cum_yaw, _ = motion_to_angles(motion, intr)
    positions = [
        int(f._cache.get("source_index", f.index))
        for f in keyframes
        if 0 <= int(f._cache.get("source_index", f.index)) < cum_yaw.size
    ]
    if len(positions) < 2:
        return 0.0
    track = cum_yaw[positions]
    return float(track.max() - track.min())


def _gate_on_capture_quality(
    frames: list[Frame],
    config: PipelineConfig,
    kind: CaptureKind,
    metrics: StitchMetrics,
    progress: ProgressReporter,
) -> CaptureAssessment:
    """Judge the capture before a single feature is detected, and act on the verdict.

    This runs between extraction and selection because that is the last moment at which
    rejecting costs seconds instead of minutes, and the first at which there is anything
    to measure. It reports its progress under the ``extract`` stage rather than inventing
    one of its own: :data:`biosphere.pano.pipeline.STAGE_WEIGHTS` has no weight for a new
    stage, so a new name would be forwarded at the progress bar's high-water mark and read
    as a hang. Its cost lands in ``stage_seconds["extract"]`` for the same reason — the
    scan really is part of getting the frames ready — and is reported exactly on
    :attr:`~biosphere.pano.validate.CaptureAssessment.elapsed_s`.

    A hard verdict raises unless ``BIOSPHERE_FORCE_STITCH=1``, in which case it is
    recorded as a warning and the solve continues. When the verdict is ``walking`` and
    ``BIOSPHERE_SEGMENT_ROOMS=1``, the walkthrough is segmented and the segments travel on
    the exception so the API can offer per-room re-shoot guidance instead of a bare
    failure.
    """
    assessment = assess_capture(
        frames, config, progress=progress, kind=kind, progress_span=(0.9, 1.0)
    )
    metrics.warnings.extend(assessment.warnings)

    if assessment.verdict is CaptureVerdict.DEGRADED:
        metrics.warnings.append(
            f"capture_geometry_degraded: rotation explains the motion only to "
            f"{assessment.median_rot_residual_px:.1f} px "
            f"({assessment.median_rot_residual_deg:.2f} deg); the phone moved as well as "
            "turned, so straight lines may bend where the parallax is strongest"
        )

    if not assessment.verdict.is_fatal:
        return assessment

    if assessment.forced:
        metrics.warnings.append(
            f"forced_stitch: the capture was judged '{assessment.verdict.value}' "
            f"({assessment.reason}) but BIOSPHERE_FORCE_STITCH is set, so it was processed "
            "anyway; the result is not expected to be usable"
        )
        log.warning(
            "solve_capture.forced_past_verdict",
            verdict=assessment.verdict.value,
            detail=assessment.detail(),
        )
        return assessment

    # Segmenting costs a second full-resolution scan, so it happens only once the capture
    # is certain to be refused — never on the forced path, where the segments would be
    # computed and then thrown away.
    payload: dict[str, Any] = {}
    if assessment.verdict is CaptureVerdict.WALKING and segment_rooms_enabled():
        # Deliberately a fresh scan rather than the assessment's: that one is a capped
        # subsample, and a segmentation cannot find a doorway it never looked at.
        from .rooms import segment_rooms  # noqa: PLC0415 - only on the opt-in failure path

        payload["segments"] = [s.as_dict() for s in segment_rooms(frames, config, progress=progress)]

    log.warning("solve_capture.rejected", verdict=assessment.verdict.value, detail=assessment.detail())
    assessment.raise_if_fatal(**payload)
    return assessment  # pragma: no cover - raise_if_fatal always raises here


def _prior_focal_px(
    assessment: CaptureAssessment,
    frames: list[Frame],
    exif_prior: float | None,
) -> tuple[float | None, str]:
    """Best focal available *before* any of the solve's own features are detected.

    Until now the answer was "none", and every stage that runs before
    :func:`~biosphere.pano.rotation.estimate_focal` fell back on the 67-degree phone prior.
    That prior is not a small error on the captures this product is aimed at: a phone held
    upright measures its *horizontal* field across the short edge, about 45 degrees, so the
    guess is out by a factor of 1.5 and every angle derived from it — the whole
    phase-correlation yaw track in :mod:`biosphere.pano.select` — is inflated by the same
    factor. A 357-degree turn reads as 543.

    Most of the pipeline survives that, because it mostly compares inflated angles against
    inflated angles: keyframe spacing is ``hfov * (1 - overlap)`` measured on the same
    track, so the factor very nearly cancels.
    :func:`_measured_yaw_span` re-measures with the solved focal precisely so the coverage
    promise does not inherit it.

    "Very nearly" is load-bearing, which is why the result of this function is deliberately
    **not** used for keyframe selection. The cancellation is exact only in the small-angle
    limit: the per-frame yaw steps are a few degrees, where ``atan`` is linear, but
    ``hfov`` is 43-67 degrees, where it is not. Correcting the focal from 67 to 43 degrees
    therefore shrinks the bin by 1.56x while shrinking the track by 1.68x, and the
    keyframes come out **7.4 % further apart** in real angle. On a capture whose adjacent
    pairs are already marginal that is the difference between a connected pose graph and a
    shattered one — measured: the realistic fixture went from 33 of 52 keyframes linked to
    18 of 49, i.e. from a poor panorama to ``InsufficientOverlapError``. Re-spacing
    keyframes is a change that has to be made deliberately and paired with
    ``target_overlap``; it is not something a units fix should do behind the caller's back.

    The one place the units genuinely do not cancel, and the reason this exists, is
    :func:`~biosphere.pano.features.candidate_pairs`, which reduces the predicted separation
    modulo a full turn to find the pair that closes the loop — 360 degrees is an absolute
    quantity, so an inflated track wraps in the wrong place. Measured on the reference
    45-degree portrait sweep, 22 keyframes: against the prior it proposes 35 pairs a genuine
    200-230 degrees apart (which all fail verification) and misses the one real
    loop-closure pair; against the measured focal it proposes exactly that pair and nothing
    else.

    The fix costs nothing because the measurement already exists.
    :func:`~biosphere.pano.validate.assess_capture` has just searched for the focal that
    best explains the capture as a rotation — that search is the reason the walking check
    works at all — and then threw the answer away. On the reference sweep it recovers
    44.99 degrees against a truth of 45.0.

    EXIF still wins where it exists: it is a statement about the hardware rather than an
    inference from the pictures, and the photo path has always used it. The scan is the
    fallback for video, which has no EXIF and is where the failure lives.

    Returns ``(focal_px, source)`` with the focal in the *extracted frames'* pixels;
    ``None`` means no measurement is available and the 67-degree default stands. The caller
    uses it for the pair-search yaw prior and nothing else — see the two call sites, both of
    which say why.
    """
    if exif_prior is not None and exif_prior > 0.0:
        return float(exif_prior), "exif"
    if not frames or frames[0].width <= 0:
        return None, "default"

    width = float(frames[0].width)
    if (
        assessment.scan is None
        or assessment.focal_clamped
        or assessment.focal_px <= 0.0
        or assessment.analysis_width <= 0
        or assessment.n_usable_samples < MIN_SCAN_SAMPLES_FOR_FOCAL
    ):
        return None, "default"

    # The scan works at a downscaled analysis width; a focal in pixels scales with it.
    focal = float(assessment.focal_px) * (width / float(assessment.analysis_width))
    hfov = math.degrees(2.0 * math.atan((width / 2.0) / focal))
    if not math.isfinite(focal) or not (MIN_HFOV_DEG <= hfov <= MAX_HFOV_DEG):
        return None, "default"
    return focal, "prescan"


def _rescue_with_alternates(
    pool: list[Frame],
    cum_yaw: np.ndarray,
    keyframes: list[Frame],
    featuresets: list,
    pairs: list[MatchPair],
    prior_yaw: np.ndarray,
    match_yaw: np.ndarray,
    K: np.ndarray | None,
    config: PipelineConfig,
    metrics: StitchMetrics,
) -> tuple[list[Frame], list, list[MatchPair], np.ndarray, int]:
    """Go back to the video for the directions the chosen keyframe would not link.

    Runs after matching and after :func:`repair_connectivity`, on the keyframes that are
    still stranded in a minority component. ``repair_connectivity`` gives a frame a second
    chance at *matching*; this gives the direction a second chance at a *frame*, which is a
    different failure and the one the extra frames in a video exist to fix.

    The stranded keyframe is not replaced, it is supplemented: an alternate that links is
    appended and the stranded original falls out with the rest of the minority component in
    :func:`_keep_largest_component`. Nothing is loosened to make this work — the alternate
    faces exactly the same verification as any other pair — so a rescue means a genuine
    match was found, not that the bar was lowered.

    Returns the possibly-extended structures and how many directions were recovered.
    """
    n = len(keyframes)
    components = connected_components(n, pairs)
    if len(components) <= 1 or not pool:
        return keyframes, featuresets, pairs, prior_yaw, 0

    main = set(components[0])
    stranded = [p for p in range(n) if p not in main]
    if not stranded or not main:
        return keyframes, featuresets, pairs, prior_yaw, 0

    def _source(frame: Frame) -> int:
        return int(frame._cache.get("source_index", frame.index))

    taken = {_source(f) for f in keyframes}
    bin_deg = max(1.0, float(DEFAULT_HFOV_DEG) * (1.0 - float(config.target_overlap)))
    window = max(1.0, bin_deg * _ALTERNATE_YAW_FRACTION)

    prior = list(np.asarray(prior_yaw, dtype=np.float64).ravel())
    rescued = 0

    for position in stranded:
        src = _source(keyframes[position])
        if not (0 <= src < cum_yaw.size):
            continue
        here = float(cum_yaw[src])
        nearby = [
            k
            for k in range(len(pool))
            if k not in taken and 0 <= k < cum_yaw.size and abs(float(cum_yaw[k]) - here) <= window
        ]
        # SPREAD THE PICKS ACROSS THE WINDOW, do not just take the top of the quality rank.
        #
        # Measured: ranking alternates by quality alone recovered 1 of 23 stranded
        # directions on the reference capture. The reason is that the frame that failed was
        # ITSELF the quality pick, so the next few by quality are its immediate neighbours —
        # the same half-second, the same blur burst, the same passing highlight, and the
        # same failure. Dividing the window into slots and taking the best of each samples
        # OUTSIDE the burst for the same number of tries and the same cost.
        candidates: list[int] = []
        if nearby:
            nearby.sort()
            slots = min(_ALTERNATES_PER_STRANDED, len(nearby))
            for slot in range(slots):
                lo = (slot * len(nearby)) // slots
                hi = max(lo + 1, ((slot + 1) * len(nearby)) // slots)
                candidates.append(max(nearby[lo:hi], key=lambda k: float(pool[k].quality)))

        for k in candidates:
            alternate = pool[k]
            yaw_here = float(match_yaw[k]) if 0 <= k < match_yaw.size else here
            targets = sorted(
                main, key=lambda t: abs((prior[t] - yaw_here + 180.0) % 360.0 - 180.0)
            )[:_RESCUE_TARGETS]
            if not targets:
                break

            alternate._cache["source_index"] = k
            try:
                alt_features = detect_features([alternate], config)
            except StitchError:
                continue

            # The alternate goes LAST so every verified pair comes back as (existing, new)
            # and the i < j ordering the rest of the solver relies on holds without having
            # to reverse a homography.
            local_sets = [featuresets[t] for t in targets] + alt_features
            local = match_pairs(
                local_sets, [(m, len(targets)) for m in range(len(targets))], K, config
            )
            if not local:
                continue

            new_position = len(keyframes)
            for pair in local:
                pair.i, pair.j = targets[pair.i], new_position
            alternate.index = new_position
            keyframes.append(alternate)
            featuresets.append(alt_features[0])
            prior.append(yaw_here)
            pairs.extend(local)
            main.add(new_position)
            taken.add(k)
            rescued += 1
            break

    if rescued:
        log.info(
            "solve_capture.rescued_with_alternates",
            n_rescued=rescued,
            n_stranded=len(stranded),
        )
        metrics.warnings.append(
            f"recovered_from_video: {rescued} of {len(stranded)} directions the chosen frame "
            "could not link were recovered by matching a different frame of the video from "
            "the same point in the sweep"
        )
    return keyframes, featuresets, pairs, np.asarray(prior, dtype=np.float64), rescued


def _check_coverage(
    yaw_span: float,
    keyframes: list[Frame],
    dropped: list[tuple[int, float]],
    metrics: StitchMetrics,
) -> None:
    """Record how much of the capture survived, and refuse only what cannot be built.

    Losing frames is graded here rather than gated: a sweep that broke into pieces still
    yields a panorama from its largest piece, which is worth more to the user than an error
    at the end of a two-minute capture.  The one hard floor left is yaw span, because below
    it there is no turn to unwrap and nothing a filler could honestly close.
    """
    n_total = len(keyframes) + len(dropped)
    fraction = len(keyframes) / max(n_total, 1)
    if dropped:
        yaws = sorted(y % 360.0 for _, y in dropped)
        metrics.warnings.append(
            f"dropped_unlinked_keyframes: {len(dropped)} of {n_total} keyframes shared no "
            "verified features with the rest of the capture and were left out "
            f"(near yaw {', '.join(f'{y:.0f}' for y in yaws[:8])}"
            f"{' ...' if len(yaws) > 8 else ''} deg)"
        )
        if fraction < MIN_COMPONENT_FRACTION:
            # Reported, not refused.  See MIN_COMPONENT_FRACTION.
            metrics.warnings.append(
                f"fragmented_capture: only {len(keyframes)} of {n_total} keyframes "
                f"({fraction:.0%}) could be linked into one capture; the panorama is built "
                "from the largest connected piece and the parts the sweep broke away from "
                "it are filled in rather than photographed"
            )
    if yaw_span < MIN_YAW_SPAN_DEG:
        raise InsufficientCoverageError(
            f"the sweep covers only {yaw_span:.0f} deg of yaw "
            f"(at least {MIN_YAW_SPAN_DEG:.0f} deg is required for a panorama)"
        )


# ======================================================================================
# Entry point
# ======================================================================================


def solve_capture(
    source: Path | list[Path],
    config: PipelineConfig,
    *,
    work_dir: Path,
    progress: ProgressReporter | None = None,
) -> SolveResult:
    """Solve a capture into levelled camera orientations plus their verified pairs.

    ``source`` is either a single video file or a list of image paths (a directory of
    images also works). ``work_dir`` receives the normalised frames; it is created if
    needed and is safe to reuse between runs.
    """
    progress = progress if progress is not None else NullProgress()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "frames"
    metrics = StitchMetrics()

    kind, paths = _classify_source(source)

    # -- probe -------------------------------------------------------------------------
    with _timed(metrics, "probe"):
        progress.stage("probe", 0.0, "inspecting capture")
        infos: list[VideoInfo] = []
        sidecar_raw: dict[str, Any] | None = None
        if kind is CaptureKind.VIDEO:
            infos = [probe_video(p) for p in paths]
            sidecar = load_capture_sidecar(paths)
            if sidecar is not None:
                metrics.warnings.extend(sidecar.warnings())
                sidecar_raw = sidecar.raw
                log.info(
                    "solve_capture.sidecar",
                    path=str(sidecar.path),
                    verdict=sidecar.verdict,
                    smear_fraction=sidecar.smear_fraction,
                    exposure_s=sidecar.exposure_s,
                    mean_fps=sidecar.mean_fps,
                )
            first = infos[0]
            total_s = sum(i.duration_s for i in infos)
            piece = "" if len(infos) == 1 else f" in {len(infos)} segments"
            progress.stage(
                "probe",
                1.0,
                f"{first.display_width}x{first.display_height} @ {first.fps:.1f} fps, "
                f"{total_s:.1f}s{piece}",
            )
        else:
            progress.stage("probe", 1.0, f"{len(paths)} photos")

    # -- extract -----------------------------------------------------------------------
    with _timed(metrics, "extract"):
        progress.stage("extract", 0.0, "normalising frames")
        if kind is CaptureKind.VIDEO:
            frames = extract_frames(infos, frames_dir, config, progress)
            focal_prior = None
        else:
            frames = load_photos(list(paths), frames_dir, config, progress)
            focal_prior = photo_focal_prior_px(frames)

        metrics.n_frames_extracted = len(frames)
        if len(frames) < 2:
            raise InsufficientCoverageError(
                f"only {len(frames)} usable frame(s) in the capture; a panorama needs a sweep"
            )
        # Judge the capture here, before feature detection, so that a capture no rotation
        # model can explain costs a couple of seconds rather than the whole solve. The
        # scan also leaves behind a measured focal, which is a far better prior than the
        # 67-degree phone guess for everything between here and self-calibration.
        assessment = _gate_on_capture_quality(frames, config, kind, metrics, progress)
        measured_focal, focal_prior_source = _prior_focal_px(assessment, frames, focal_prior)

    # -- select ------------------------------------------------------------------------
    with _timed(metrics, "select"):
        progress.stage("select", 0.0, "measuring sweep motion")
        prior_intr = Intrinsics.from_hfov(frames[0].width, frames[0].height, DEFAULT_HFOV_DEG)
        if focal_prior is not None and focal_prior > 0.0:
            prior_intr = Intrinsics(
                focal=float(focal_prior),
                cx=frames[0].width / 2.0,
                cy=frames[0].height / 2.0,
                width=frames[0].width,
                height=frames[0].height,
            )
        # The focal the pre-scan measured. Deliberately *not* used for keyframe selection:
        # see :func:`_prior_focal_px` for why that would silently re-space the keyframes.
        match_intr = prior_intr
        if measured_focal is not None and measured_focal > 0.0:
            match_intr = Intrinsics(
                focal=float(measured_focal),
                cx=frames[0].width / 2.0,
                cy=frames[0].height / 2.0,
                width=frames[0].width,
                height=frames[0].height,
            )
        log.info(
            "solve_capture.focal_prior",
            source=focal_prior_source,
            measured_focal_px=None if measured_focal is None else round(float(measured_focal), 2),
            selection_hfov_deg=round(prior_intr.hfov_deg, 2),
            matching_hfov_deg=round(match_intr.hfov_deg, 2),
        )
        motion = estimate_frame_motion(frames, config)
        cum_yaw, _cum_pitch = motion_to_angles(motion, prior_intr, pitch_correction=False)
        progress.stage("select", 0.5, "choosing keyframes")
        keyframes, pattern, yaw_span = select_keyframes(
            frames, config, motion=motion, intr=prior_intr
        )
        # The pair-search prior is a different question from keyframe spacing and is the one
        # that genuinely needs the real focal, because ``candidate_pairs`` reduces it modulo
        # a full turn. Measured in the *matching* frame, not the selection frame.
        match_yaw, match_pitch = motion_to_angles(motion, match_intr, pitch_correction=False)
        prior_yaw = np.array([match_yaw[f.index] for f in keyframes], dtype=np.float64)
        prior_pitch = np.array([match_pitch[f.index] for f in keyframes], dtype=np.float64)
        # Re-index so that Frame.index, the pose list position and the pair indices all
        # agree; the original position in the extracted pool stays available on the cache.
        for pos, frame in enumerate(keyframes):
            frame._cache["source_index"] = int(frame.index)
            frame.index = pos
        progress.stage("select", 1.0, f"{len(keyframes)} keyframes, {yaw_span:.0f} deg swept")

    metrics.n_keyframes = len(keyframes)
    metrics.sweep_pattern = pattern if pattern is not SweepPattern.UNKNOWN else metrics.sweep_pattern

    # ``yaw_span`` here is measured against the 67-degree *prior*, so it is only as good
    # as that guess — on a 45-degree portrait capture it overstates the turn by 1.6x. The
    # real coverage decision therefore waits until ``estimate_focal`` has run (see below);
    # this early gate only catches a capture that barely moved at all, which is worth
    # rejecting before spending a minute on feature matching.
    if yaw_span < PROVISIONAL_YAW_SPAN_FLOOR_DEG:
        raise InsufficientCoverageError(
            f"the sweep covers only {yaw_span:.0f} deg of yaw "
            f"(at least {MIN_YAW_SPAN_DEG:.0f} deg is required for a panorama)"
        )
    if len(keyframes) < 3:
        raise InsufficientOverlapError(
            f"only {len(keyframes)} keyframes survived selection; the capture is too short"
        )

    # Left on the shipped prior on purpose. ``intr0`` feeds ``_verify_pair``'s
    # ``_MAX_PAIR_ROTATION_DEG`` sanity bound, so moving it to the measured focal changes
    # which pairs survive verification on *every* capture — including the ones that were
    # already healthy. Measured on the nodal gate capture, that alone moved the wrap-column
    # ratio from 1.21 to 4.40 and failed the acceptance gate. The measured focal earns its
    # place in the yaw prior, where the bug actually is; it has not earned a change to the
    # verification bound, which is a different claim needing its own calibration.
    width, height = _keyframe_size(keyframes)

    # -- the phone's own lens ------------------------------------------------------------
    # Identified from the sidecar, priced in this frame's pixels. Never a measurement and
    # never allowed to override one; what it gives the solve is a bound, because the
    # self-calibration's error is one-sided. See :mod:`biosphere.pano.devices`.
    device = resolve_device(sidecar_raw, width=width, height=height)
    metrics.device = device.as_dict()
    if device.focal_px is not None and device.source != "none":
        config = replace(
            config,
            focal_prior_px=float(device.focal_px),
            focal_prior_sigma=float(device.sigma_fraction),
        )
        if focal_prior is None:
            # Also the last-resort prior for self-calibration itself, which otherwise falls
            # back on a 67-degree guess when no homography yields a focal.
            focal_prior = float(device.focal_px)

    intr0 = Intrinsics.from_hfov(width, height, DEFAULT_HFOV_DEG)
    if focal_prior is not None and focal_prior > 0.0:
        intr0 = Intrinsics(
            focal=float(focal_prior), cx=width / 2.0, cy=height / 2.0, width=width, height=height
        )

    # -- features ----------------------------------------------------------------------
    with _timed(metrics, "features"):
        progress.stage("features", 0.0, "detecting features")
        featuresets = detect_features(keyframes, config, progress)
        progress.stage("features", 1.0, "features detected")

    # -- match -------------------------------------------------------------------------
    with _timed(metrics, "match"):
        progress.stage("match", 0.0, "pairing keyframes")
        candidates = candidate_pairs(
            len(keyframes), prior_yaw, config, hfov_deg=match_intr.hfov_deg,
            poses_prior_pitch=prior_pitch, vfov_deg=match_intr.vfov_deg,
        )
        pairs = match_pairs(featuresets, candidates, intr0.K, config, progress)
        pairs = repair_connectivity(keyframes, pairs, intr0.K, config, progress)
        # Second chance at a FRAME, where repair_connectivity is a second chance at a match.
        # See _ALTERNATES_PER_STRANDED: this is the step that makes shooting video worth
        # more than shooting stills.
        keyframes, featuresets, pairs, prior_yaw, n_rescued = _rescue_with_alternates(
            frames, cum_yaw, keyframes, featuresets, pairs, prior_yaw, match_yaw,
            intr0.K, config, metrics,
        )
        if n_rescued:
            progress.stage("match", 0.95, f"{n_rescued} directions recovered from the video")
        log.info(
            "solve_capture.pose_graph",
            n_keyframes=len(keyframes),
            n_candidates=len(candidates),
            n_verified=len(pairs),
            n_long_baseline=sum(1 for p in pairs if p.is_loop_closure),
            max_separation=max((p.j - p.i for p in pairs), default=0),
            component_sizes=[len(c) for c in connected_components(len(keyframes), pairs)],
        )
        progress.stage("match", 1.0, f"{len(pairs)} verified pairs")

    # -- keep the solvable part of the pose graph ---------------------------------------
    keyframes, pairs, prior_yaw, dropped = _keep_largest_component(
        keyframes, pairs, prior_yaw, metrics
    )
    metrics.n_keyframes = len(keyframes)
    metrics.n_pairs = len(pairs)
    metrics.n_loop_closures = sum(1 for p in pairs if p.is_loop_closure)
    if len(pairs) < len(keyframes) - 1:
        raise InsufficientOverlapError(
            f"only {len(pairs)} keyframe pairs verified for {len(keyframes)} keyframes; "
            "the sweep is too fast or the scene is too featureless"
        )

    # -- bundle ------------------------------------------------------------------------
    with _timed(metrics, "bundle"):
        progress.stage("bundle", 0.05, "self-calibrating focal length")
        focal = estimate_focal(pairs, width, height, focal_prior)
        intr = Intrinsics(focal=focal, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

        # Now that the focal is known, remeasure the sweep. This is the number the
        # coverage promise rests on, so it must not keep the 67-degree guess baked in.
        yaw_span = _measured_yaw_span(motion, intr, keyframes)
        _check_coverage(yaw_span, keyframes, dropped, metrics)

        relative_rotations(pairs, intr)
        progress.stage("bundle", 0.2, "building the pose graph")
        poses0 = initial_poses(len(keyframes), pairs, intr)
        loop_before = loop_closure_stats(poses0, pairs)

        progress.stage("bundle", 0.35, "refining orientations")
        poses, intr, ba_metrics = bundle_adjust(
            poses0, pairs, intr, config, progress=progress, span=(0.35, 0.9)
        )
        progress.stage("bundle", 0.9, "levelling the horizon")
        poses = finalise_poses(poses, config)
        # Fill the holes the unlinked keyframes left, for the render only (see
        # ``_BRIDGE_MAX_GAP_HFOV``). After levelling, so the interpolation is between
        # final orientations.
        bridged, bridged_R, bridge_report = _bridge_track_gaps(
            frames, keyframes, poses, dropped, match_yaw, intr, config
        )
        if bridged:
            base = len(keyframes)
            for k, (frame, R) in enumerate(zip(bridged, bridged_R)):
                frame.index = base + k
                keyframes.append(frame)
                poses.append(R)
            log.info("solve_capture.bridged_gaps", n_frames=len(bridged), gaps=bridge_report)
            metrics.warnings.append(
                f"bridged_gaps: {len(bridged)} frame(s) that shared no verified feature with "
                f"the rest of the capture were placed by interpolation across {len(bridge_report)} "
                "featureless gap(s) so the render covers them; their orientation is estimated, "
                "not measured"
            )
        progress.stage("bundle", 1.0, "solved")

    if ba_metrics.get("bundle_deadline_hit"):
        # Said out loud, because it changes what every number below it means: the solve was
        # stopped by the clock, not by convergence, so a high reprojection error here is a
        # budget symptom and not necessarily a capture problem.
        metrics.warnings.append(
            f"bundle_time_capped: the orientation solve was stopped at its "
            f"{float(ba_metrics.get('bundle_budget_seconds', 0.0)):.0f}s budget; alignment is "
            "the best reached in that time rather than the best reachable"
        )
    metrics.mean_reprojection_error_px = float(ba_metrics.get("mean_reprojection_error_px", 0.0))
    metrics.max_reprojection_error_px = float(ba_metrics.get("max_reprojection_error_px", 0.0))
    loop = loop_closure_stats(poses, pairs)
    metrics.loop_closure_residual_deg = loop.robust_deg
    # The two diagnostics beside the headline. ``loop_closure_span_pairs == 0`` is the one
    # to read first: it means *nothing in the capture measured the loop at all*, so the
    # headline fell back to a p90 over short-baseline pairs and a reassuring number there
    # says only that the neighbours agree with each other.
    metrics.loop_closure_max_deg = loop.max_deg
    metrics.loop_closure_span_pairs = loop.n_span_pairs
    metrics.parallax_median_px = float(ba_metrics.get("median_reprojection_error_all_px", 0.0))
    metrics.parallax_p90_px = float(ba_metrics.get("p90_reprojection_error_all_px", 0.0))
    metrics.parallax_fraction_over_6px = float(ba_metrics.get("frac_over_6px_all", 0.0))
    metrics.focal_px = float(intr.focal)
    metrics.hfov_deg = float(intr.hfov_deg)

    # -- what the phone says, against what the imagery says -------------------------------
    # A disagreement here is diagnostic, not decorative. The self-calibration can only err
    # LONG (see :mod:`biosphere.pano.devices`), so a solve that lands far above the lens's
    # own focal is a measurement of how far the capture is from a rotation -- and a solve
    # that lands far BELOW it cannot be explained that way at all, which means the lens
    # assumption is wrong: a different camera was selected, or the phone is not the one the
    # table thinks it is.
    if device.focal_px:
        ratio = float(intr.focal) / float(device.focal_px)
        if ratio > 1.0 + 3.0 * device.sigma_fraction:
            metrics.warnings.append(
                f"focal_above_lens: the solve wanted a {metrics.hfov_deg:.0f} deg field where "
                f"{device.profile.model or 'this phone'}'s lens is {device.hfov_deg:.0f} deg. "
                "Only a capture that moved as well as turned does that, and the field was held "
                "at the lens's own tolerance rather than followed."
            )
        elif ratio < 1.0 - 3.0 * device.sigma_fraction:
            metrics.warnings.append(
                f"focal_below_lens: the solve measured a {metrics.hfov_deg:.0f} deg field, wider "
                f"than the {device.hfov_deg:.0f} deg {device.profile.model or 'this phone'} is "
                "listed with; the capture may have used a different lens than the app selected."
            )
    if device.profile.lidar:
        # Said once, plainly, because customers ask: the scanner is in the phone and a web
        # page cannot read it. See the measure tool, which asks for a height instead.
        metrics.warnings.append(
            "lidar_present_but_unreadable: this phone has a LiDAR scanner, but a browser has "
            "no API for it, so distances come from the assumed camera height like every other "
            "capture."
        )
    if bool((sidecar_raw or {}).get("optics", {}).get("lensChangedMidCapture")):
        metrics.warnings.append(
            "lens_changed_mid_capture: the phone switched camera or zoom while recording. The "
            "solve fits one focal length for the whole file, so the frames from the other lens "
            "are placed with the wrong field of view."
        )

    # -- warnings ----------------------------------------------------------------------
    if len(keyframes) < config.min_keyframes:
        metrics.warnings.append(
            f"low_keyframe_count: {len(keyframes)} keyframes (wanted at least {config.min_keyframes})"
        )
    if yaw_span < WARN_YAW_SPAN_DEG:
        metrics.warnings.append(
            f"partial_turn: only {yaw_span:.0f} deg of yaw swept; the panorama will have a gap"
        )
    if metrics.mean_reprojection_error_px > WARN_REPROJECTION_PX:
        metrics.warnings.append(
            f"high_reprojection_error: {metrics.mean_reprojection_error_px:.2f} px mean; "
            "the camera probably translated as well as rotated"
        )
    if metrics.parallax_p90_px > WARN_PARALLAX_P90_PX:
        metrics.warnings.append(
            f"capture_parallax: {metrics.parallax_p90_px:.1f} px at the 90th percentile "
            f"({100.0 * metrics.parallax_fraction_over_6px:.0f}% of matches beyond 6 px) "
            "cannot be explained by rotation; near objects are aligned locally by the mesh "
            "and may still ghost where they overlap far ones"
        )
    if metrics.loop_closure_residual_deg > WARN_LOOP_RESIDUAL_DEG:
        metrics.warnings.append(
            f"loop_did_not_close: {metrics.loop_closure_residual_deg:.2f} deg residual; "
            "the panorama may show misaligned detail where the sweep came back round"
        )

    log.info(
        "solve_capture.done",
        kind=kind.value,
        n_frames=metrics.n_frames_extracted,
        n_keyframes=metrics.n_keyframes,
        n_pairs=metrics.n_pairs,
        n_loop_closures=metrics.n_loop_closures,
        yaw_span_deg=round(yaw_span, 1),
        focal_prior_source=focal_prior_source,
        focal_px=round(metrics.focal_px, 2),
        hfov_deg=round(metrics.hfov_deg, 2),
        mean_err_px=round(metrics.mean_reprojection_error_px, 3),
        capture_verdict=assessment.verdict.value,
        capture_rot_residual_px=round(assessment.median_rot_residual_px, 2),
        capture_scan_seconds=round(assessment.elapsed_s, 2),
        loop_before_deg=round(loop_before.robust_deg, 3),
        loop_after_deg=round(metrics.loop_closure_residual_deg, 3),
        loop_closure=loop.as_dict(),
        warnings=metrics.warnings,
        stage_seconds={k: round(v, 2) for k, v in metrics.stage_seconds.items()},
    )

    camera_poses = [
        CameraPose(frame_index=i, R=np.asarray(R, dtype=np.float64), intrinsics=intr)
        for i, R in enumerate(poses)
    ]
    # Sanity: the renderer relies on these being proper rotations.
    for pose in camera_poses:
        assert abs(float(np.linalg.det(pose.R)) - 1.0) < 1e-6, "non-rotation pose escaped the solve"

    return SolveResult(
        frames=keyframes,
        poses=camera_poses,
        pairs=pairs,
        metrics=metrics,
        work_dir=work_dir,
    )
