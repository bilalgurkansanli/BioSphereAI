"""Tests for walkthrough segmentation.

The segmenter's job is to cut a walkthrough where scene continuity genuinely breaks and
*not* to cut it anywhere else, so most of what is worth testing is the second half: a
single dropped frame, a glance at the floor, and a trace hovering around the threshold
must all leave a room intact. Those are cheap to construct exactly, so the continuity
traces here are built by hand rather than measured — which also means each test states,
in the trace itself, precisely the situation it claims to cover.

The other half is honesty about the result: a segment cut out of a walk is still a walk,
its best pivot window is still not a panorama, and the labels have to say so.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from biosphere.pano import rooms
from biosphere.pano.types import Frame, PipelineConfig, QualityMode
from biosphere.pano.validate import CaptureVerdict, PuritySample, PurityScan

SPACING = 0.25  # 4 fps, the rate the scan samples at
W, H = 480, 854


def _config() -> PipelineConfig:
    return PipelineConfig.for_mode(QualityMode.BALANCED)


def _frames(n: int = 400) -> list[Frame]:
    return [Frame(index=k, path=Path(f"f{k}.jpg"), timestamp=k / 8.0, width=W, height=H)
            for k in range(n)]


def _scan(profile: list[tuple[float, float]], *, spacing: float = SPACING) -> PurityScan:
    """Build a scan from ``(inlier_ratio, rot_residual_px)`` per sample.

    A ratio of zero means the pair did not verify at all, which is how a real scan records
    a moment it could not measure — the strongest evidence there is that the scene changed.
    """
    samples: list[PuritySample] = []
    for k, (ratio, rot) in enumerate(profile):
        usable = ratio > 0.0
        samples.append(
            PuritySample(
                i=k * 2, j=k * 2 + 2, time_s=k * spacing, dt_s=spacing, n_matches=200,
                n_inliers=int(200 * ratio), inlier_ratio=ratio,
                rot_residual_px=rot if usable else math.inf,
                free_residual_px=1.4 if usable else math.inf,
                turn_deg=2.0 if usable else 0.0, yaw_step_deg=2.0 if usable else 0.0,
                mean_flow_px=30.0, usable=usable,
            )
        )
    return PurityScan(samples, 430.0, W, H, 640.0 / W, spacing, 0.0)


def _n(seconds: float) -> int:
    return int(round(seconds / SPACING))


def _room(seconds: float, ratio: float = 0.30, rot: float = 22.0) -> list[tuple[float, float]]:
    return [(ratio, rot)] * _n(seconds)


def _blackout(seconds: float) -> list[tuple[float, float]]:
    return [(0.0, math.inf)] * _n(seconds)


# ======================================================================================
# 1. The continuity signal
# ======================================================================================


def test_moving_median_removes_an_isolated_dropout_without_moving_a_real_edge():
    trace = np.array([0.6, 0.6, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.6])
    smoothed = rooms._moving_median(trace, 3)
    assert smoothed[2] == pytest.approx(0.6), "a one-sample dropout must vanish"
    assert smoothed[6] == pytest.approx(0.0), "a sustained collapse must survive"
    assert rooms._moving_median(trace, 1).tolist() == trace.tolist()


def test_continuity_records_a_failed_pair_as_zero_not_as_missing():
    scan = _scan([(0.5, 3.0), (0.0, math.inf), (0.4, 3.0)])
    assert rooms._continuity(scan.samples).tolist() == [0.5, 0.0, 0.4]


# ======================================================================================
# 2. Hysteresis: what must NOT split a room
# ======================================================================================


def test_a_single_dropped_frame_does_not_split_a_room():
    profile = _room(12.0) + [(0.0, math.inf)] + _room(12.0)
    assert rooms._find_cuts(rooms._continuity(_scan(profile).samples), SPACING) == []


def test_a_glance_at_the_floor_does_not_split_a_room():
    """Three quarters of a second of lost tracking is a person looking down and back up.

    It is well short of the second of sustained collapse a cut requires, which is the
    whole reason that minimum exists.
    """
    profile = _room(10.0) + _blackout(0.75) + _room(10.0)
    assert rooms._find_cuts(rooms._continuity(_scan(profile).samples), SPACING) == []


def test_a_trace_hovering_at_the_threshold_does_not_chatter_into_many_segments():
    """The point of two thresholds rather than one.

    Continuity here wanders either side of the enter threshold for twenty seconds without
    ever recovering properly. With a single threshold that is a dozen cuts; with
    hysteresis the signal has to genuinely come back before a new segment starts.
    """
    rng = np.random.default_rng(4)
    marginal = [(float(v), 20.0) for v in np.clip(rng.normal(0.11, 0.05, _n(20.0)), 0.02, 0.4)]
    profile = _room(8.0) + marginal + _room(8.0)
    cuts = rooms._find_cuts(rooms._continuity(_scan(profile).samples), SPACING)
    assert len(cuts) <= 2, f"hysteresis failed to damp a marginal trace: {cuts}"


def test_the_threshold_follows_the_capture_and_not_an_absolute_number():
    """The same shape at two very different baselines must cut in the same places.

    A bedroom turn runs at a median inlier ratio of 0.63 and a walkthrough at 0.26. Any
    absolute threshold that finds the walkthrough's doorways shreds the bedroom, and any
    that spares the bedroom sees nothing in the walkthrough.
    """
    shape = lambda base: (  # noqa: E731 - a table, not a function
        [(base, 20.0)] * _n(10.0) + _blackout(2.0) + [(base, 20.0)] * _n(10.0)
    )
    high = rooms._find_cuts(rooms._continuity(_scan(shape(0.70)).samples), SPACING)
    low = rooms._find_cuts(rooms._continuity(_scan(shape(0.26)).samples), SPACING)
    assert len(high) == len(low) == 1
    assert high[0][0] == low[0][0], f"cut moved with the baseline: {high} vs {low}"


# ======================================================================================
# 3. Hysteresis: what must split
# ======================================================================================


def test_a_sustained_collapse_cuts():
    profile = _room(10.0) + _blackout(2.0) + _room(10.0)
    cuts = rooms._find_cuts(rooms._continuity(_scan(profile).samples), SPACING)
    assert len(cuts) == 1
    start, end = cuts[0]
    assert start * SPACING == pytest.approx(10.0, abs=0.5)
    assert end * SPACING == pytest.approx(12.0, abs=0.5)


def test_a_cut_that_runs_to_the_end_of_the_capture_is_closed():
    profile = _room(10.0) + _blackout(5.0)
    cuts = rooms._find_cuts(rooms._continuity(_scan(profile).samples), SPACING)
    assert len(cuts) == 1 and cuts[0][1] == len(profile)


def test_find_cuts_on_nothing_returns_nothing():
    assert rooms._find_cuts(np.zeros(0), SPACING) == []


# ======================================================================================
# 4. Segments
# ======================================================================================


def test_five_rooms_come_back_as_five_segments():
    profile = (
        _room(12.0) + _blackout(2.0)
        + _room(10.0) + _blackout(2.0)
        + _room(11.0) + _blackout(2.0)
        + _room(18.0) + _blackout(2.0)
        + _room(9.0)
    )
    segments = rooms.segment_rooms(_frames(), _config(), scan=_scan(profile))
    assert len(segments) == 5
    assert [s.index for s in segments] == [0, 1, 2, 3, 4]
    durations = [s.duration_s for s in segments]
    assert durations == pytest.approx([12.0, 10.0, 11.0, 18.0, 9.0], abs=1.0), durations
    # Segments must be ordered and disjoint, and must not span the cuts between them.
    for a, b in zip(segments, segments[1:], strict=False):
        assert a.end_s <= b.start_s


def test_a_fragment_too_short_to_say_anything_about_is_dropped():
    profile = (
        _room(12.0) + _blackout(2.0)
        + _room(2.0) + _blackout(2.0)  # a two-second fragment between two doorways
        + _room(12.0)
    )
    segments = rooms.segment_rooms(_frames(), _config(), scan=_scan(profile))
    assert len(segments) == 2
    assert all(s.duration_s >= rooms.MIN_SEGMENT_S for s in segments)


def test_an_uncut_capture_is_one_segment():
    segments = rooms.segment_rooms(_frames(), _config(), scan=_scan(_room(30.0)))
    assert len(segments) == 1
    assert segments[0].start_s == pytest.approx(0.0)
    assert segments[0].duration_s == pytest.approx(30.0, abs=0.5)


def test_segment_rooms_handles_an_empty_or_tiny_capture():
    assert rooms.segment_rooms([], _config()) == []
    assert rooms.segment_rooms(_frames(1), _config()) == []
    assert rooms.segment_rooms(_frames(), _config(), scan=_scan([])) == []


def test_a_segment_of_a_walk_is_still_labelled_a_walk():
    """The honesty requirement. Cutting a walkthrough into rooms does not make the rooms
    stitchable; it only says where the operator was when they filmed each one."""
    segments = rooms.segment_rooms(_frames(), _config(), scan=_scan(_room(20.0, rot=22.0)))
    assert segments[0].stitchable is CaptureVerdict.WALKING
    assert segments[0].stitchable_reason == "translation_dominates"


def test_a_segment_filmed_on_the_spot_is_labelled_stitchable():
    segments = rooms.segment_rooms(
        _frames(), _config(), scan=_scan(_room(20.0, ratio=0.7, rot=1.0))
    )
    assert segments[0].stitchable is CaptureVerdict.GOOD


# ======================================================================================
# 5. Pivot windows
# ======================================================================================


def test_the_best_pivot_is_the_longest_clean_stretch_not_merely_a_clean_one():
    """Two stitchable stretches in one segment: the longer one wins.

    "Longest" rather than "lowest residual" is the right objective because a panorama
    needs an arc, not a moment — a two-second window with a beautiful residual covers
    nothing worth stitching.
    """
    profile = (
        _room(6.0, ratio=0.6, rot=1.0)     # clean, 6 s
        + _room(8.0, ratio=0.5, rot=20.0)  # walking
        + _room(12.0, ratio=0.6, rot=1.0)  # clean, 12 s — this one
        + _room(6.0, ratio=0.5, rot=20.0)
    )
    pivot = rooms.segment_rooms(_frames(), _config(), scan=_scan(profile))[0].best_pivot
    assert pivot is not None and not pivot.is_fallback
    assert pivot.start_s == pytest.approx(14.0, abs=1.0), pivot
    assert pivot.duration_s == pytest.approx(12.0, abs=1.5)
    assert pivot.verdict is CaptureVerdict.GOOD


def test_a_walk_reports_its_least_bad_window_and_calls_it_a_walk():
    """The requirement the brief is most emphatic about.

    Every segment of the motivating walkthrough comes back like this: a window is still
    reported, because it says where the operator came closest to standing still and
    therefore where to re-shoot from — but it is flagged as a fallback and it keeps the
    ``walking`` verdict its numbers earn. Nothing here is allowed to imply that a sphere
    can be recovered from a walk.
    """
    profile = _room(8.0, rot=24.0) + _room(4.0, rot=13.0) + _room(8.0, rot=24.0)
    pivot = rooms.segment_rooms(_frames(), _config(), scan=_scan(profile))[0].best_pivot
    assert pivot is not None
    assert pivot.is_fallback
    assert pivot.verdict is CaptureVerdict.WALKING
    assert pivot.start_s == pytest.approx(8.0, abs=1.5), pivot
    assert pivot.median_rot_residual_px == pytest.approx(13.0, abs=0.5)


def test_the_reported_pivot_median_is_the_median_of_the_window_reported():
    """Scoring on one series and reporting another let a fallback come back *worse* than
    the segment it was chosen from — which happened on real data before this was fixed."""
    rng = np.random.default_rng(9)
    profile = [(0.3, float(v)) for v in rng.uniform(8.0, 40.0, _n(30.0))]
    segment = rooms.segment_rooms(_frames(), _config(), scan=_scan(profile))[0]
    pivot = segment.best_pivot
    assert pivot is not None
    assert pivot.median_rot_residual_px <= segment.median_rot_residual_px + 1e-9, (
        f"the best window ({pivot.median_rot_residual_px:.2f}) is worse than its whole "
        f"segment ({segment.median_rot_residual_px:.2f})"
    )


def test_a_pivot_window_does_not_bridge_a_blackout():
    """A window that spans a gap would report the average of the footage either side as if
    it were one continuous pivot. Unmeasured samples therefore count as infinitely bad."""
    profile = (
        _room(3.5, ratio=0.6, rot=1.0)
        + _blackout(0.75)  # too short to cut, but must still break a pivot window
        + _room(3.5, ratio=0.6, rot=1.0)
    )
    segment = rooms.segment_rooms(_frames(), _config(), scan=_scan(profile))[0]
    assert len(segment.samples) == _n(7.75), "this must be a single uncut segment"
    pivot = segment.best_pivot
    assert pivot is not None and not pivot.is_fallback
    assert pivot.duration_s < 4.0, f"the window bridged the blackout: {pivot}"


def test_a_segment_with_nothing_measurable_has_no_pivot():
    scan = _scan(_room(10.0, ratio=0.3, rot=20.0))
    for s in scan.samples[1:]:
        s.usable = False
    assert rooms._best_pivot(scan.samples, SPACING) is None


# ======================================================================================
# 6. Reporting
# ======================================================================================


def test_segments_serialise_for_the_error_payload():
    profile = _room(12.0) + _blackout(2.0) + _room(12.0)
    segments = rooms.segment_rooms(_frames(), _config(), scan=_scan(profile))
    import json

    payload = [s.as_dict() for s in segments]
    json.dumps(payload)  # infinities and enums must not escape into the API
    for row in payload:
        for key in ("index", "start_s", "end_s", "duration_s", "total_turn_deg",
                    "median_rot_residual_px", "stitchable", "best_pivot"):
            assert key in row, f"as_dict is missing {key}"
        assert row["stitchable"] in {v.value for v in CaptureVerdict}
        assert isinstance(row["best_pivot"], dict)
        assert row["best_pivot"]["verdict"] in {v.value for v in CaptureVerdict}


def test_a_segments_turn_is_integrated_over_its_own_span():
    """Two degrees per quarter-second over twenty seconds is 160 degrees, not 160 samples."""
    segment = rooms.segment_rooms(_frames(), _config(), scan=_scan(_room(20.0)))[0]
    assert segment.total_turn_deg == pytest.approx(8.0 * 20.0, rel=0.1)
    assert segment.yaw_span_deg == pytest.approx(8.0 * 20.0, rel=0.1)
