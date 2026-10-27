"""Video -> the frames worth posing.

A 3-minute room video is 5,400 frames; a splat wants a few hundred *sharp*, *distinct*
ones. Two passes: ffmpeg decimates to ``candidate_fps`` (with the display rotation applied,
so portrait phones come out upright, and the long side capped), then the sharpest frame of
every time window is kept and near-identical neighbours (the walker paused) are dropped.

Sharpness is the variance of the Laplacian at 480 px, the same measure the panorama engine
ranks by; it has no absolute meaning, only an order, which is all a window-best rule needs.
"""

from __future__ import annotations

import concurrent.futures as cf
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from biosphere.pano.extract import ffmpeg_binary, probe_video

__all__ = ["FrameConfig", "FrameSet", "extract_frames", "select_frames", "prepare_frames"]

_SCORE_WIDTH = 480
_DUP_SIZE = (32, 24)
_EXTRACT_TIMEOUT_S = 1800


@dataclass(frozen=True)
class FrameConfig:
    #: Decimation rate of the candidate pool.
    candidate_fps: float = 6.0
    #: Frames kept per second of video, before the cap; SplatTour quotes 1-2.
    #: 3 a second: a walk at a step a second puts frames ~25 cm apart, which is what a
    #: bare-walled flat needs for its doorway turns to stay linked; ``max_frames`` caps long walks.
    keep_per_second: float = 3.0
    min_frames: int = 80
    max_frames: int = 420
    #: Long side of the extracted frames on disk (training never uses more than this).
    long_side: int = 1920
    jpeg_quality: int = 95
    #: Mean absolute difference of 32x24 thumbnails below which two frames are the same shot.
    duplicate_threshold: float = 2.5
    n_workers: int = 8


@dataclass
class FrameSet:
    image_dir: Path
    names: list[str]
    timestamps: list[float]
    sharpness: list[float]
    width: int
    height: int
    n_candidates: int
    source_fps: float
    source_duration_s: float
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    #: Length of each source video, in order: the shared timeline maps back to a file with these.
    video_durations: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_frames": len(self.names), "n_candidates": self.n_candidates, "width": self.width,
            "height": self.height, "source_fps": self.source_fps, "source_duration_s": self.source_duration_s,
            "median_sharpness": float(np.median(self.sharpness)) if self.sharpness else 0.0,
            "seconds": round(self.seconds, 2), "warnings": list(self.warnings),
        }


def _target_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    long = max(width, height)
    if long <= long_side:
        w, h = width, height
    else:
        s = long_side / float(long)
        w, h = int(round(width * s)), int(round(height * s))
    return w - (w % 2), h - (h % 2)


def extract_frames(
    videos: list[Path], out_dir: Path, *, config: FrameConfig, progress=None
) -> tuple[list[Path], list[float], tuple[int, int], float, float, list[float]]:
    """Decimate every segment into ``out_dir`` as ``cand_%06d.jpg`` on one shared timeline."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("cand_*.jpg"):
        stale.unlink()
    infos = [probe_video(Path(v)) for v in videos]
    fps = float(config.candidate_fps)
    paths: list[Path] = []
    stamps: list[float] = []
    offset = 0.0
    size = None
    for k, (video, info) in enumerate(zip(videos, infos, strict=True)):
        w, h = _target_size(info.display_width, info.display_height, config.long_side)
        size = size or (w, h)
        start = len(paths) + 1
        cmd = [
            ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-vf", f"fps={fps:.6f},scale={w}:{h}:flags=lanczos", "-q:v", "2",
            "-start_number", str(start), "-y", str(out_dir / "cand_%06d.jpg"),
        ]
        subprocess.run(cmd, check=True, timeout=_EXTRACT_TIMEOUT_S, capture_output=True)
        produced = sorted(out_dir.glob("cand_*.jpg"))[len(paths):]
        for i, p in enumerate(produced):
            paths.append(p)
            stamps.append(offset + i / fps)
        offset += float(info.duration_s)
        if progress is not None:
            progress((k + 1) / len(videos))
    if not paths:
        raise RuntimeError("ffmpeg produced no frames")
    src_fps = float(infos[0].fps)
    return paths, stamps, size or (0, 0), src_fps, offset, [float(i.duration_s) for i in infos]


def _score(path: Path) -> tuple[float, np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0, np.zeros(_DUP_SIZE[::-1], dtype=np.float32)
    h, w = img.shape
    if w > _SCORE_WIDTH:
        img = cv2.resize(img, (_SCORE_WIDTH, int(round(h * _SCORE_WIDTH / w))), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(img, cv2.CV_64F).var())
    tiny = cv2.resize(img, _DUP_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    return sharp, tiny


def select_frames(
    sharpness: np.ndarray, thumbs: np.ndarray, duration_s: float, *, config: FrameConfig
) -> list[int]:
    """Indices to keep: the sharpest of every window of the distinct candidates.

    Repeats are removed BEFORE the windows are cut, not after: a walker who pauses (or a
    bench video that shows each photo for half a second) produces runs of identical
    candidates, and a window that straddles two runs would keep picking the sharper copy
    of the frame the previous window already kept, then drop it as a duplicate. Windowing
    over the distinct candidates keeps the frame budget where the camera actually moved.
    """
    n = int(sharpness.shape[0])
    if n == 0:
        return []
    # Runs of near-identical candidates collapse to their sharpest member.
    distinct: list[int] = []
    run_start, run_best = 0, 0
    for i in range(1, n + 1):
        if i < n and float(np.abs(thumbs[i] - thumbs[run_start]).mean()) < config.duplicate_threshold:
            if sharpness[i] > sharpness[run_best]:
                run_best = i
            continue
        distinct.append(run_best)
        run_start, run_best = i, i
    target = int(round(config.keep_per_second * max(duration_s, 1.0)))
    target = int(np.clip(target, config.min_frames, config.max_frames))
    target = min(target, len(distinct))
    if target <= 0:
        return []
    edges = np.linspace(0, len(distinct), target + 1).round().astype(int)
    keep: list[int] = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if b <= a:
            continue
        window = distinct[a:b]
        keep.append(int(window[int(np.argmax(sharpness[window]))]))
    return keep


def prepare_frames(
    videos: list[Path], work_dir: Path, *, config: FrameConfig | None = None, progress=None
) -> FrameSet:
    """Everything from the video files to ``work_dir / "images" / f_%05d.jpg``."""
    config = config or FrameConfig()
    t0 = time.perf_counter()
    work_dir = Path(work_dir)
    cand_dir = work_dir / "candidates"
    image_dir = work_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for stale in image_dir.glob("f_*.jpg"):
        stale.unlink()

    paths, stamps, (w, h), src_fps, duration, durations = extract_frames(
        videos, cand_dir, config=config, progress=(lambda f: progress("extract", 0.6 * f)) if progress else None
    )
    with cf.ThreadPoolExecutor(max_workers=max(1, config.n_workers)) as pool:
        scored = list(pool.map(_score, paths))
    sharp = np.array([s for s, _ in scored], dtype=np.float64)
    thumbs = np.stack([t for _, t in scored])
    if progress:
        progress("extract", 0.75)
    keep = select_frames(sharp, thumbs, duration, config=config)

    names: list[str] = []
    for k, i in enumerate(keep):
        name = f"f_{k:05d}.jpg"
        img = cv2.imread(str(paths[i]), cv2.IMREAD_COLOR)
        cv2.imwrite(str(image_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, int(config.jpeg_quality)])
        names.append(name)
    for p in paths:
        p.unlink(missing_ok=True)
    warnings: list[str] = []
    if len(keep) < config.min_frames:
        warnings.append(f"few_frames: only {len(keep)} usable frames from {duration:.0f}s of video")
    if sharp.size and float(np.median(sharp[keep])) < 12.0:
        warnings.append("blurry_capture: median sharpness of the kept frames is low")
    if progress:
        progress("extract", 1.0)
    result = FrameSet(
        image_dir=image_dir, names=names, timestamps=[stamps[i] for i in keep],
        sharpness=[float(sharp[i]) for i in keep], width=w, height=h, n_candidates=len(paths),
        source_fps=src_fps, source_duration_s=duration, seconds=time.perf_counter() - t0, warnings=warnings,
        video_durations=durations,
    )
    # Which moment of the video each kept frame is: the map back from a pose to the footage.
    import json

    (work_dir / "frames.json").write_text(
        json.dumps({"names": names, "timestamps": result.timestamps, "sharpness": result.sharpness, "video_durations": durations, **result.as_dict()}),
        encoding="utf-8",
    )
    return result
