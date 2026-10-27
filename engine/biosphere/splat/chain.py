"""Is the keyframe chain unbroken? Probe it, bridge what can be bridged, cut what cannot.

A walkthrough is a chain: frame *i* must share enough of the world with frame *i+1* for the
mapper to walk from one to the next. The frame selector cannot know where the chain breaks -
it picks by sharpness and time - and in an empty flat it breaks at every doorway: a white
wall fills the frame, the walker turns, and two keyframes 0.3 s apart share nothing the
detector can hold on to. What the mapper does with a broken chain is the worst outcome of
all: it joins the pieces through whatever loop closure it can find and delivers a tangle
that looks like a scene from every training view and like nothing from anywhere else.

So before matching, the chain is *probed*: every consecutive pair is matched and verified
with a fundamental matrix, and a link with fewer than ``link_min_inliers`` is *bridged* -
the source video is re-read at ``bridge_fps`` inside that gap and the extra frames are
kept if the chain holds through them. Measured on a WhatsApp-compressed walk through an
empty flat, 8 of 13 breaks bridge this way; the rest are whip pans blurred beyond any
matcher's reach (a dense matcher found nothing either). Those become *segment* boundaries:
the pipeline maps each segment on its own and registers the extra segments into the property
like rooms, or delivers them as their own areas when the walk never returns to a place the
property frame knows. A room that opens by name is a room; a tangle is not.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .features import LocalFeatures, MatchingConfig, extract_features, match_pairs
from .frames import FrameSet, ffmpeg_binary

__all__ = ["ChainConfig", "ChainReport", "probe_and_bridge", "segments_from_breaks", "link_inliers", "links_inliers"]


@dataclass(frozen=True)
class ChainConfig:
    #: Verified (fundamental-matrix) inliers a consecutive pair needs to count as linked.
    link_min_inliers: int = 30
    #: ...and inside a bridge, where consecutive frames are 33 ms apart and the geometry is
    #: nearly identical, a weaker link still carries the chain. Measured on the first real
    #: phone walk: at 30 fps the four salvageable doorway turns bottom out at 20-31 inliers,
    #: so a bridge threshold of 30 threw all four away and 18 keeps them. What it must not do
    #: is admit the hopeless ones, and it does not: those sit at 0-7 whatever the frame rate,
    #: because the frames themselves are motion-blurred to mush (Laplacian variance ~1 against
    #: ~37 for a kept frame) - no matcher recovers them, dense ones included.
    bridge_min_inliers: int = 18
    #: How densely the source is re-read inside a weak gap. The video's own rate: a turn fast
    #: enough to break the chain is exactly where every intermediate frame counts.
    bridge_fps: float = 30.0
    #: Never more than this many bridge frames in one gap (a long gap is a paused camera,
    #: and a paused camera's links are strong; a weak long gap is something else entirely).
    max_bridge_frames: int = 24
    #: Gaps longer than this are not bridged at all.
    max_gap_s: float = 4.0
    #: Segments shorter than this are dropped rather than delivered as an area.
    min_segment_frames: int = 12
    ransac_px: float = 1.5


@dataclass
class ChainReport:
    n_links: int = 0
    n_weak: int = 0
    n_bridged: int = 0
    n_added: int = 0
    n_broken: int = 0
    breaks_s: list[float] = field(default_factory=list)
    segments: list[tuple[int, int]] = field(default_factory=list)
    n_dropped_short: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["segments"] = [list(s) for s in self.segments]
        return d


def link_inliers(feats: LocalFeatures, i: int, j: int, *, config: MatchingConfig, device, ransac_px: float = 1.5) -> int:
    """Fundamental-matrix inliers between frames ``i`` and ``j`` of ``feats`` (0 when unmatched)."""
    return links_inliers(feats, [(i, j)], config=config, device=device, ransac_px=ransac_px)[0]


def links_inliers(
    feats: LocalFeatures, pairs: list[tuple[int, int]], *, config: MatchingConfig, device, ransac_px: float = 1.5
) -> list[int]:
    """:func:`link_inliers` for many pairs in one matcher pass (same numbers, one call)."""
    loose = dataclasses.replace(config, min_matches=8)
    ordered = [(i, j) if i < j else (j, i) for i, j in pairs]
    matches = match_pairs(feats, sorted(set(ordered)), config=loose, device=device)
    out: list[int] = []
    for a, b in ordered:
        m = matches.get((a, b))
        if m is None or m.shape[0] < 8:
            out.append(0)
            continue
        pa = feats.keypoints[a][m[:, 0]].astype(np.float64)
        pb = feats.keypoints[b][m[:, 1]].astype(np.float64)
        _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, ransac_px, 0.999)
        out.append(int(mask.sum()) if mask is not None else 0)
    return out


def segments_from_breaks(n: int, breaks: list[int], min_frames: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split ``range(n)`` at every ``i`` in ``breaks`` (the link i -> i+1 is broken).

    Returns ``(kept, dropped)`` ranges as ``[a, b)`` pairs; a range shorter than
    ``min_frames`` is dropped.
    """
    edges = sorted({0, n} | {b + 1 for b in breaks if 0 <= b < n - 1})
    ranges = [(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]
    kept = [r for r in ranges if r[1] - r[0] >= min_frames]
    dropped = [r for r in ranges if r[1] - r[0] < min_frames]
    return kept, dropped


def _video_at(t: float, durations: list[float]) -> tuple[int, float] | None:
    """Which video of the area's timeline holds second ``t``, and the second within it."""
    offset = 0.0
    for k, d in enumerate(durations):
        if t < offset + d:
            return k, t - offset
        offset += d
    return None


def _extract_span(video: Path, t0: float, t1: float, fps: float, size: tuple[int, int], out_dir: Path) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("b_*.jpg"):
        stale.unlink()
    w, h = size
    cmd = [
        ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-ss", f"{t0:.4f}", "-t", f"{max(t1 - t0, 0.01):.4f}", "-i", str(video),
        "-vf", f"fps={fps:.6f},scale={w}:{h}:flags=lanczos", "-q:v", "2", "-y", str(out_dir / "b_%04d.jpg"),
    ]
    subprocess.run(cmd, check=True, timeout=120, capture_output=True)
    frames = sorted(out_dir.glob("b_*.jpg"))
    return [(t0 + k / fps, p) for k, p in enumerate(frames)]


def probe_and_bridge(
    frames: FrameSet,
    videos: list[Path],
    *,
    work_dir: Path,
    matching: MatchingConfig,
    config: ChainConfig | None = None,
    device,
    progress=None,
) -> tuple[FrameSet, LocalFeatures, ChainReport]:
    """Probe the chain of ``frames``, bridge weak links from the source video, cut at breaks.

    Returns the (possibly densified and thinned) frame set, renumbered in time order inside
    ``frames.image_dir``; its features; and the report with the segment ranges. Frames of
    segments too short to deliver are removed from the set.
    """
    config = config or ChainConfig()
    t_start = time.perf_counter()
    report = ChainReport()
    work_dir = Path(work_dir)
    image_dir = Path(frames.image_dir)
    names = list(frames.names)
    stamps = list(frames.timestamps)
    n = len(names)
    if n < 2:
        report.segments = [(0, n)]
        feats = extract_features(image_dir, names, config=matching, device=device)
        return frames, feats, report

    def tell(f: float, msg: str) -> None:
        if progress is not None:
            progress(f, msg)

    feats = extract_features(image_dir, names, config=matching, device=device, progress=lambda f: tell(0.5 * f, "describing frames"))

    # 1. Probe every consecutive link.
    inliers = links_inliers(feats, [(i, i + 1) for i in range(n - 1)], config=matching, device=device, ransac_px=config.ransac_px)
    report.n_links = n - 1
    weak = [i for i, v in enumerate(inliers) if v < config.link_min_inliers]
    report.n_weak = len(weak)
    tell(0.6, "checking the chain")

    # 2. Bridge the weak links from the source video.
    added: list[tuple[float, Path, float]] = []  # (timestamp, path, sharpness)
    broken: set[int] = set()
    durations = list(frames.video_durations) if frames.video_durations else [frames.source_duration_s]
    bridge_root = work_dir / "bridge"
    for k, i in enumerate(weak):
        t0, t1 = stamps[i], stamps[i + 1]
        gap = t1 - t0
        where0, where1 = _video_at(t0, durations), _video_at(t1 - 1e-3, durations)
        if gap > config.max_gap_s or where0 is None or where1 is None or where0[0] != where1[0]:
            broken.add(i)
            continue
        vid, local0 = where0
        fps = min(config.bridge_fps, config.max_bridge_frames / max(gap, 1e-3))
        try:
            span = _extract_span(Path(videos[vid]), local0 + 0.02, local0 + gap - 0.02, fps, (frames.width, frames.height), bridge_root / f"gap_{i:05d}")
        except Exception:  # noqa: BLE001 - a bridge that cannot be cut is a break
            broken.add(i)
            continue
        span = [(t0 + (t - local0), p) for t, p in span if local0 < t < local0 + gap]
        if not span:
            broken.add(i)
            continue
        bridge_names = [p.name for _, p in span]
        bfeats = extract_features(bridge_root / f"gap_{i:05d}", bridge_names, config=matching, device=device)
        chain = LocalFeatures.concatenate([feats.subset([i, i + 1]), bfeats])  # 0 = i, 1 = i+1, 2.. = bridge in time order
        order = [0, *range(2, 2 + len(span)), 1]
        holds = True
        for a, b in zip(order[:-1], order[1:], strict=True):
            if link_inliers(chain, a, b, config=matching, device=device, ransac_px=config.ransac_px) < config.bridge_min_inliers:
                holds = False
                break
        if holds:
            report.n_bridged += 1
            for t, p in span:
                gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray is not None else 0.0
                added.append((t, p, sharp))
        else:
            broken.add(i)
        tell(0.6 + 0.3 * (k + 1) / max(len(weak), 1), "bridging weak links")
    report.n_added = len(added)

    # 3. Renumber everything in time order (bridge frames move in, originals are renamed).
    entries: list[tuple[float, Path, float, int | None]] = [
        (stamps[i], image_dir / names[i], frames.sharpness[i] if i < len(frames.sharpness) else 0.0, i) for i in range(n)
    ]
    entries += [(t, p, s, None) for t, p, s in added]
    entries.sort(key=lambda e: e[0])
    staging = image_dir.parent / "images_staging"
    staging.mkdir(parents=True, exist_ok=True)
    for k, (_t, p, _s, _) in enumerate(entries):
        shutil.move(str(p), str(staging / f"f_{k:05d}.jpg"))
    for stale in image_dir.glob("*.jpg"):
        stale.unlink()
    for p in staging.glob("f_*.jpg"):
        shutil.move(str(p), str(image_dir / p.name))
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(bridge_root, ignore_errors=True)
    new_names = [f"f_{k:05d}.jpg" for k in range(len(entries))]
    new_stamps = [e[0] for e in entries]
    new_sharp = [e[2] for e in entries]
    # Where the old link i -> i+1 was broken, the break now sits between the renumbered
    # positions of frame i's last successor... i.e. between old frame i (nothing was added
    # in a broken gap) and whatever follows it.
    old_to_new = {e[3]: k for k, e in enumerate(entries) if e[3] is not None}
    new_breaks = sorted(old_to_new[i] for i in broken)
    report.n_broken = len(new_breaks)
    report.breaks_s = [round(new_stamps[b], 2) for b in new_breaks]

    # 4. Segments; drop the ones too short to deliver.
    kept, dropped = segments_from_breaks(len(entries), new_breaks, config.min_segment_frames)
    if dropped:
        report.n_dropped_short = sum(b - a for a, b in dropped)
        report.warnings.append(
            f"short_segments_dropped: {report.n_dropped_short} frames in {len(dropped)} stretches could not be linked to the walk"
        )
    keep_idx = [k for a, b in kept for k in range(a, b)]
    # Segment ranges re-expressed over the kept list.
    remap = {k: j for j, k in enumerate(keep_idx)}
    report.segments = [(remap[a], remap[b - 1] + 1) for a, b in kept]
    final_names = [new_names[k] for k in keep_idx]
    if len(keep_idx) != len(entries):
        for k in range(len(entries)):
            if k not in remap:
                (image_dir / new_names[k]).unlink(missing_ok=True)

    tell(0.9, "describing frames")
    feats = extract_features(image_dir, final_names, config=matching, device=device)
    out = FrameSet(
        image_dir=image_dir,
        names=final_names,
        timestamps=[new_stamps[k] for k in keep_idx],
        sharpness=[new_sharp[k] for k in keep_idx],
        width=frames.width,
        height=frames.height,
        n_candidates=frames.n_candidates,
        source_fps=frames.source_fps,
        source_duration_s=frames.source_duration_s,
        seconds=frames.seconds,
        warnings=list(frames.warnings),
        video_durations=list(durations),
    )
    if report.n_broken:
        report.warnings.append(
            f"chain_breaks: tracking was lost at {report.n_broken} points ({', '.join(f'{t:.0f}s' for t in report.breaks_s)}): "
            "turns too fast or blurred; the walk is delivered in parts"
        )
    report.seconds = time.perf_counter() - t_start
    tell(1.0, "")
    return out, feats, report
