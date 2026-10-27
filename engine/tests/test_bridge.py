"""Bridging a featureless gap: dropped keyframes come back to the render, interpolated."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from biosphere.pano import geometry as g
from biosphere.pano import solve as solve_mod
from biosphere.pano.types import Frame, Intrinsics, PipelineConfig


def _frame(i: int, quality: float = 1.0) -> Frame:
    f = Frame(index=i, path=Path(f"frame_{i:04d}.jpg"), timestamp=i / 5.0, width=480, height=854)
    f.quality = quality
    return f


def _pose(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    return g.from_yaw_pitch_roll(math.radians(yaw_deg), math.radians(pitch_deg), 0.0)


def _setup(gap_deg: float, *, pitch_b: float = 20.0):
    """A row of keyframes at 0, 20, 40 deg, then a gap, then one at 40 + gap; source
    frames every 2 degrees of track yaw; the dropped keyframe sits in the gap."""
    intr = Intrinsics.from_hfov(480, 854, 45.0)
    cfg = PipelineConfig(target_overlap=0.55)
    n_pool = 200
    pool = [_frame(i, quality=1.0 + 0.01 * (i % 7)) for i in range(n_pool)]
    track = np.arange(n_pool) * 2.0  # 2 deg of track yaw per source frame
    src = [0, 10, 20, int(round((40.0 + gap_deg) / 2.0))]
    yaws = [0.0, 20.0, 40.0, 40.0 + gap_deg]
    keyframes = []
    for pos, s in enumerate(src):
        f = pool[s]
        f._cache["source_index"] = s
        f.index = pos
        keyframes.append(f)
    poses = [_pose(y, 20.0 if k < 3 else pitch_b) for k, y in enumerate(yaws)]
    dropped_src = (src[2] + src[3]) // 2
    return intr, cfg, pool, keyframes, poses, track, [(dropped_src, float(track[dropped_src]))]


def test_a_dropped_keyframe_is_bridged_between_its_neighbours():
    intr, cfg, pool, keyframes, poses, track, dropped = _setup(60.0)
    frames, rots, report = solve_mod._bridge_track_gaps(pool, keyframes, poses, dropped, track, intr, cfg)
    assert report and report[0]["brackets"] == (20, 50)
    assert report[0]["fraction_source"] == "track"
    # A 60 deg gap at 20 deg spacing wants two frames, at a third and two thirds.
    assert len(frames) == 2
    yaws = sorted(math.degrees(g.yaw_pitch_roll(R)[0]) for R in rots)
    assert abs(yaws[0] - 60.0) < 3.0 and abs(yaws[1] - 80.0) < 3.0
    # Pitch is interpolated, so it stays on the row.
    for R in rots:
        assert abs(math.degrees(g.yaw_pitch_roll(R)[1]) - 20.0) < 0.5
    # The chosen frames really lie inside the gap and are not the brackets.
    for f in frames:
        assert 20 < int(f._cache["source_index"]) < 50


def test_no_bridge_when_the_footprints_already_meet():
    intr, cfg, pool, keyframes, poses, track, dropped = _setup(22.0)
    frames, rots, report = solve_mod._bridge_track_gaps(pool, keyframes, poses, dropped, track, intr, cfg)
    assert frames == [] and report == []


def test_no_bridge_across_a_row_change():
    intr, cfg, pool, keyframes, poses, track, dropped = _setup(60.0, pitch_b=-20.0)
    frames, rots, report = solve_mod._bridge_track_gaps(pool, keyframes, poses, dropped, track, intr, cfg)
    assert frames == [] and report == []


def test_a_wandering_row_is_still_one_row():
    """A hand-held row is not level, and a hole inside it must still be bridged.

    Measured on the reference capture: a genuine 49-degree hole in the up row had brackets at
    +33 and +19 degrees of pitch — a 13.9 degree step — while the real row change beside it
    stepped 26.2. A fixed pitch step tight enough to reject the second rejected the first too
    and left the hole in the panorama as a stepped, torn ceiling. What separates them is the
    DIRECTION of the gap: a row runs along yaw, a row change moves in pitch.
    """
    intr, cfg, pool, keyframes, poses, track, dropped = _setup(60.0)
    poses[3] = _pose(40.0 + 60.0, 20.0 - 13.9)  # a wandering row: 60 deg of yaw, 13.9 of pitch
    frames, _rots, report = solve_mod._bridge_track_gaps(pool, keyframes, poses, dropped, track, intr, cfg)
    assert frames, "a hole inside a wandering row must still be bridged"
    assert report[0]["gap_deg"] > 1.2 * intr.hfov_deg * (1 - cfg.target_overlap)

    # A gap the sweep spent mostly climbing is a row change, whatever its total angle.
    poses[3] = _pose(40.0 + 16.0, 20.0 - 26.2)
    frames2, _r2, report2 = solve_mod._bridge_track_gaps(pool, keyframes, poses, dropped, track, intr, cfg)
    assert frames2 == [] and report2 == []


def test_time_is_the_clock_when_the_track_lost_the_wall():
    intr, cfg, pool, keyframes, poses, track, dropped = _setup(60.0)
    flat = track.copy()
    flat[20:51] = flat[20]  # the phase correlation saw nothing move across the gap
    frames, rots, report = solve_mod._bridge_track_gaps(pool, keyframes, poses, dropped, flat, intr, cfg)
    assert len(frames) == 2 and report[0]["fraction_source"] == "time"
    srcs = sorted(int(f._cache["source_index"]) for f in frames)
    assert abs(srcs[0] - 30) <= 2 and abs(srcs[1] - 40) <= 2
