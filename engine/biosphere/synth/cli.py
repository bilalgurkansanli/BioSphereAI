"""Command line for the synthetic harness — how a human inspects a regression.

    python -m biosphere.synth.cli make-room    --style rich --seed 7 --out room.jpg --width 4096
    python -m biosphere.synth.cli make-capture --style rich --trajectory three_row \\
                                               --effects realistic --out runs/rich_realistic
    python -m biosphere.synth.cli score        --candidate pano.jpg --reference truth.jpg

When a CI assertion fails, the number alone never explains why. ``make-capture`` therefore
writes the frames, an mp4, the ground-truth panorama and a ``truth.json`` with every pose,
gain and intrinsic — enough to reproduce the failure exactly — plus a ``reprojected.jpg``
built from the ground-truth poses, which answers the first question anybody asks: was the
capture even solvable, or did the simulator make something impossible?
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import cv2
import typer

from biosphere.pano.types import Intrinsics

from .capture import EFFECT_PRESETS, Trajectory, simulate_capture
from .encode import encode_video, probe_duration, probe_frame_count
from .metrics import quality_report
from .render import render_equirect, reproject_to_equirect
from .room import STYLES, Room

app = typer.Typer(
    name="biosphere-synth",
    help="Synthetic ground-truth harness: build rooms, fake captures, and score stitches.",
    add_completion=False,
    no_args_is_help=True,
)

_TRAJECTORIES = {
    "single_row": Trajectory.single_row,
    "three_row": Trajectory.three_row,
    "three_row_plus_poles": Trajectory.three_row_plus_poles,
}


def _echo(msg: str) -> None:
    typer.echo(msg, err=False)


def _validate(name: str, value: str, allowed: object) -> str:
    if value not in allowed:  # type: ignore[operator]
        raise typer.BadParameter(f"unknown {name} {value!r}; expected one of {sorted(allowed)}")  # type: ignore[arg-type]
    return value


@app.command("make-room")
def make_room(
    style: Annotated[str, typer.Option(help="rich | low_texture | specular")] = "rich",
    seed: Annotated[int, typer.Option(help="Layout seed; identical seeds give identical rooms.")] = 7,
    out: Annotated[Path, typer.Option(help="Where to write the panorama.")] = Path("room.jpg"),
    width: Annotated[int, typer.Option(help="Equirect width; height is width/2.")] = 4096,
    supersample: Annotated[int, typer.Option(help="Render at NxN and box-average.")] = 1,
    jpeg_quality: Annotated[int, typer.Option(help="JPEG quality when writing .jpg.")] = 95,
) -> None:
    """Render and save a ground-truth equirectangular panorama of a procedural room."""
    _validate("style", style, STYLES)
    if width % 2:
        raise typer.BadParameter("width must be even")
    room = Room.random(seed, style)
    t0 = time.perf_counter()
    pano = render_equirect(room, width, width // 2, supersample=supersample)
    dt = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)] if out.suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(out), pano, params):
        raise typer.Exit(code=1)
    _echo(
        f"{out}  {width}x{width // 2}  style={style} seed={seed}  "
        f"{room.width_m:.2f}x{room.depth_m:.2f}x{room.height_m:.2f} m  "
        f"{len(room.boxes)} boxes, {len(room.rects)} wall features  ({dt:.1f}s)"
    )


@app.command("make-capture")
def make_capture(
    out: Annotated[Path, typer.Option(help="Output directory.")],
    style: Annotated[str, typer.Option(help="rich | low_texture | specular")] = "rich",
    trajectory: Annotated[
        str, typer.Option(help="single_row | three_row | three_row_plus_poles")
    ] = "three_row",
    effects: Annotated[str, typer.Option(help="clean | realistic | harsh")] = "realistic",
    seed: Annotated[int, typer.Option(help="Room and motion seed.")] = 7,
    noise_seed: Annotated[int, typer.Option(help="Sensor-noise seed.")] = 1234,
    duration: Annotated[float, typer.Option(help="Capture length in seconds.")] = 0.0,
    fps: Annotated[float, typer.Option(help="Capture frame rate.")] = 30.0,
    frame_width: Annotated[int, typer.Option(help="Frame width in pixels.")] = 960,
    hfov: Annotated[float, typer.Option(help="Horizontal field of view, degrees.")] = 68.0,
    arm_radius: Annotated[
        float, typer.Option(help="Optical-centre orbit radius in metres; <0 uses the preset.")
    ] = -1.0,
    truth_width: Annotated[int, typer.Option(help="Ground-truth panorama width.")] = 4096,
    crf: Annotated[int, typer.Option(help="x264 CRF for the mp4.")] = 20,
    video: Annotated[bool, typer.Option(help="Encode an mp4 as well as the frames.")] = True,
    reprojection: Annotated[
        bool, typer.Option(help="Also write the ground-truth-pose reprojection.")
    ] = True,
) -> None:
    """Simulate a full capture: frames, mp4, ground truth panorama and truth.json."""
    _validate("style", style, STYLES)
    _validate("trajectory", trajectory, _TRAJECTORIES)
    _validate("effects", effects, EFFECT_PRESETS)
    if frame_width < 32:
        raise typer.BadParameter("frame_width must be at least 32")

    defaults = {"single_row": 12.0, "three_row": 24.0, "three_row_plus_poles": 32.0}
    dur = duration if duration > 0 else defaults[trajectory]
    frame_height = max(2 * int(round(frame_width * 9 / 16 / 2)), 2)
    intr = Intrinsics.from_hfov(frame_width, frame_height, hfov)

    room = Room.random(seed, style)
    traj = _TRAJECTORIES[trajectory](duration_s=dur, fps=fps, seed=seed)

    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    _echo(
        f"simulating {traj.n_frames} frames of {frame_width}x{frame_height} "
        f"({trajectory}, {effects}, style={style}, seed={seed})"
    )
    t0 = time.perf_counter()
    cap = simulate_capture(
        room,
        traj,
        intr,
        out_dir=frames_dir,
        effects=effects,
        seed=noise_seed,
        arm_radius_m=None if arm_radius < 0 else arm_radius,
        ground_truth_width=truth_width,
        progress=_TyperProgress(),
    )
    _echo(f"\nrendered in {time.perf_counter() - t0:.1f}s")

    gt_path = out / "ground_truth.jpg"
    cv2.imwrite(str(gt_path), cap.ground_truth, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    truth_path = cap.write_truth(out / "truth.json")

    summary: dict[str, object] = {
        "frames": cap.n_frames,
        "frames_dir": str(frames_dir),
        "ground_truth": str(gt_path),
        "truth_json": str(truth_path),
        "arm_radius_m": cap.trajectory.arm_radius_m,
        "gain_min": round(float(cap.gains.min()), 4),
        "gain_max": round(float(cap.gains.max()), 4),
    }

    if video:
        mp4 = encode_video(cap.frame_paths, out / "capture.mp4", fps=fps, crf=crf)
        summary["video"] = str(mp4)
        summary["video_duration_s"] = round(probe_duration(mp4), 4)
        summary["video_frames"] = probe_frame_count(mp4)

    if reprojection:
        w = min(truth_width, 2048)
        pano, wsum = reproject_to_equirect(
            cap.load_frames(),
            cap.rotations,
            intr,
            w,
            w // 2,
            gains=cap.gains.tolist(),
        )
        rp = out / "reprojected.jpg"
        cv2.imwrite(str(rp), pano, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        summary["reprojected"] = str(rp)
        summary["reprojected_coverage"] = round(float((wsum > 0).mean()), 4)

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _echo(json.dumps(summary, indent=2))


@app.command("score")
def score(
    candidate: Annotated[Path, typer.Option(help="Panorama produced by the pipeline.")],
    reference: Annotated[Path, typer.Option(help="Ground-truth panorama.")],
    mask: Annotated[Path | None, typer.Option(help="Optional coverage mask image.")] = None,
    label: Annotated[str | None, typer.Option(help="Free-form label for the report.")] = None,
    out: Annotated[Path | None, typer.Option(help="Also write the report to this path.")] = None,
) -> None:
    """Print the full quality report for a candidate panorama, as JSON."""
    cand = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
    ref = cv2.imread(str(reference), cv2.IMREAD_COLOR)
    if cand is None:
        raise typer.BadParameter(f"could not read candidate {candidate}")
    if ref is None:
        raise typer.BadParameter(f"could not read reference {reference}")
    m = None
    if mask is not None:
        raw = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise typer.BadParameter(f"could not read mask {mask}")
        m = raw > 0
    report = quality_report(cand, ref, mask=m, label=label or candidate.stem)
    text = json.dumps(report, indent=2)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    _echo(text)


@app.command("styles")
def styles() -> None:
    """List the available room styles, trajectories and effect presets."""
    _echo("styles:       " + ", ".join(sorted(STYLES)))
    _echo("trajectories: " + ", ".join(sorted(_TRAJECTORIES)))
    _echo("effects:      " + ", ".join(sorted(EFFECT_PRESETS)))


class _TyperProgress:
    """Minimal in-place progress line; the harness never needs a full progress bar."""

    def __init__(self) -> None:
        self._last = -1.0

    def stage(self, name: str, fraction: float, message: str = "") -> None:
        if fraction - self._last < 0.02 and fraction < 1.0:
            return
        self._last = fraction
        sys.stderr.write(f"\r  {name}: {fraction * 100:5.1f}%  {message}   ")
        sys.stderr.flush()

    def log(self, message: str, **fields: object) -> None:
        sys.stderr.write(message + "\n")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
