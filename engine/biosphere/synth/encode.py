"""Turning simulated frames into a video file the pipeline will actually accept.

The harness could hand the stitcher a directory of PNGs, but that would quietly skip the
first stage of the real product — decoding H.264 out of a phone recording — and with it
every artefact that stage contributes: chroma subsampling, inter-frame prediction
smearing detail out of fast pans, and the quantisation that eats exactly the low-contrast
gradients a matte wall depends on. So the harness encodes.

Frames are piped to ffmpeg rather than referenced by a glob or a concat list. A glob
constrains the caller to a numbered sequence in one directory; the concat demuxer has a
well-known habit of dropping the final frame. Piping keeps the frame count exact, which
matters because :func:`encode_video` verifies it and refuses to hand back a file that
does not contain what it was given.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "encode_video",
    "probe_duration",
    "probe_frame_count",
    "probe_stream",
    "FFmpegMissingError",
    "EncodeError",
]


class FFmpegMissingError(RuntimeError):
    """ffmpeg or ffprobe is not on PATH."""


class EncodeError(RuntimeError):
    """ffmpeg ran but produced something other than what was asked for."""


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegMissingError(
            f"{name} was not found on PATH; the synthetic harness needs ffmpeg to build "
            "the video half of a capture"
        )
    return path


def encode_video(
    frame_paths: Sequence[Path | str],
    out_path: Path | str,
    fps: float,
    crf: int = 20,
    preset: str = "medium",
    *,
    pix_fmt: str = "yuv420p",
    timeout_s: float = 900.0,
) -> Path:
    """Encode ``frame_paths`` into an H.264 MP4 at ``fps``.

    The ``scale`` filter rounds both dimensions down to even numbers because
    ``yuv420p`` subsamples chroma by two and ffmpeg refuses odd dimensions outright —
    a synthetic frame size like 255x143 would otherwise fail at the very end of a long
    render.

    Raises :class:`EncodeError` if the finished file does not contain exactly as many
    frames as were fed in.
    """
    frames = [Path(p) for p in frame_paths]
    if not frames:
        raise ValueError("encode_video needs at least one frame")
    if fps <= 0:
        raise ValueError("fps must be positive")
    missing = [p for p in frames if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frame(s) missing, first: {missing[0]}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _tool("ffmpeg"),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-framerate",
        f"{fps:.10g}",
        "-i",
        "pipe:0",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        pix_fmt,
        "-r",
        f"{fps:.10g}",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for p in frames:
            proc.stdin.write(p.read_bytes())
        proc.stdin.close()
    except BrokenPipeError as exc:  # pragma: no cover - only on an ffmpeg crash
        proc.kill()
        _, err = proc.communicate()
        raise EncodeError(f"ffmpeg closed the pipe early: {err.decode(errors='replace')}") from exc
    _, err = proc.communicate(timeout=timeout_s)
    if proc.returncode != 0:
        raise EncodeError(f"ffmpeg failed ({proc.returncode}): {err.decode(errors='replace')}")

    written = probe_frame_count(out_path)
    if written != len(frames):
        raise EncodeError(
            f"encoded frame count mismatch: fed {len(frames)}, file contains {written}"
        )
    return out_path


def probe_stream(path: Path | str) -> dict[str, object]:
    """Container + video-stream metadata as a flat dict, via ffprobe."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cmd = [
        _tool("ffprobe"),
        "-v",
        "error",
        "-count_packets",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_read_packets,r_frame_rate,avg_frame_rate,codec_name,pix_fmt:format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, check=False)
    if res.returncode != 0:
        raise EncodeError(f"ffprobe failed: {res.stderr.decode(errors='replace')}")
    data = json.loads(res.stdout.decode("utf-8"))
    streams = data.get("streams") or [{}]
    out: dict[str, object] = dict(streams[0])
    out.update({f"format_{k}": v for k, v in (data.get("format") or {}).items()})
    return out


def probe_duration(path: Path | str) -> float:
    """Container duration in seconds."""
    info = probe_stream(path)
    raw = info.get("format_duration")
    if raw is None:
        raise EncodeError(f"ffprobe reported no duration for {path}")
    return float(raw)


def probe_frame_count(path: Path | str) -> int:
    """Exact number of video packets, i.e. coded frames.

    ``nb_frames`` from the container header is often absent or a guess; counting packets
    is cheap for the short clips the harness produces and is exact.
    """
    info = probe_stream(path)
    raw = info.get("nb_read_packets")
    if raw is None:
        raise EncodeError(f"ffprobe reported no packet count for {path}")
    return int(raw)
