"""Synthetic ground-truth harness: the instrument that makes stitch quality measurable.

The pipeline's promise — "professional quality, no visual artefacts" — is an opinion
until something can put a number on it. This package supplies the number. It builds a
virtual room analytically, renders a *perfect* equirectangular panorama of it, then
simulates a person standing at that exact point filming a 360 sweep with a phone: motion
blur, rolling shutter, lagging auto-exposure, vignetting, sensor noise, hand tremor and —
the one that actually breaks stitchers — real parallax, because the lens orbits the
user's spine rather than sitting at the optical centre.

Feed the resulting video to the pipeline, compare its output against the panorama we
started from, and every degradation is measurable in dB.

Typical use::

    from biosphere.synth import Room, Trajectory, simulate_capture, encode_video, quality_report

    room = Room.random(seed=7, style="low_texture")
    traj = Trajectory.three_row(duration_s=24.0, fps=30.0, seed=7)
    cap = simulate_capture(room, traj, intrinsics, out_dir=work, effects="realistic")
    encode_video(cap.frame_paths, work / "capture.mp4", fps=30.0)
    report = quality_report(stitched_panorama, cap.ground_truth)

Module map:

``room``     the analytic ray tracer and the procedural surfaces
``render``   ground-truth panoramas, single video frames, and the camera response
``capture``  trajectories, artefact presets, and the whole-capture simulator
``encode``   frames to H.264 and back, with an exact frame-count check
``metrics``  gauge-invariant, solid-angle-weighted scoring
``cli``      ``python -m biosphere.synth.cli`` for inspecting a regression by hand
"""

from __future__ import annotations

from biosphere.synth.capture import (
    EFFECT_PRESETS,
    CaptureEffects,
    Jitter,
    SyntheticCapture,
    Trajectory,
    simulate_capture,
)
from biosphere.synth.encode import encode_video, probe_duration, probe_frame_count
from biosphere.synth.metrics import (
    PanoQuality,
    PoseError,
    align_panoramas,
    compare,
    pose_errors,
    quality_report,
    rotate_equirect,
    seam_score,
    wrap_seam_score,
)
from biosphere.synth.render import (
    apply_camera_response,
    render_equirect,
    render_view,
    reproject_to_equirect,
)
from biosphere.synth.room import STYLES, Box, Light, Room, StyleParams, WallRect

__all__ = [
    "EFFECT_PRESETS",
    "STYLES",
    "Box",
    "CaptureEffects",
    "Jitter",
    "Light",
    "PanoQuality",
    "PoseError",
    "Room",
    "StyleParams",
    "SyntheticCapture",
    "Trajectory",
    "WallRect",
    "align_panoramas",
    "apply_camera_response",
    "compare",
    "encode_video",
    "pose_errors",
    "probe_duration",
    "probe_frame_count",
    "quality_report",
    "render_equirect",
    "render_view",
    "reproject_to_equirect",
    "rotate_equirect",
    "seam_score",
    "simulate_capture",
    "wrap_seam_score",
]
