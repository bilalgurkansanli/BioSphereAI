"""Turning a capture into a pool of scored, normalised candidate frames.

The solver never touches the original media again after this stage, so everything the
later stages need to *choose* frames has to be measured here: geometry (size), motion
blur (Laplacian variance), and exposure health (brightness / clipping).

Why ffmpeg instead of ``cv2.VideoCapture``
------------------------------------------
Phone videos carry a display matrix that rotates the picture at playback time. OpenCV
ignores it, which silently yields sideways panoramas for portrait captures. ffmpeg
applies it by default, so we shell out and deliberately do *not* pass ``-noautorotate``.
ffmpeg also does the temporal decimation and the Lanczos downscale in one pass, which is
several times faster than decoding every frame into Python just to throw most away.

Why the sharpness score is rank-normalised
------------------------------------------
Laplacian variance has no absolute meaning — it scales with scene texture, exposure and
resolution. A dim living room and a bright street produce numbers two orders of magnitude
apart, so any fixed threshold is wrong for one of them. Ranking within the capture makes
the score self-calibrating: what matters is only whether *this* frame is sharper than its
neighbours, which is exactly the question keyframe selection asks.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog

from .types import (
    Frame,
    PipelineConfig,
    ProgressReporter,
    StitchError,
)

__all__ = [
    "VideoInfo",
    "CaptureSidecar",
    "load_capture_sidecar",
    "probe_video",
    "extract_frames",
    "load_photos",
    "photo_focal_prior_px",
    "score_frames",
    "ffmpeg_binary",
    "ffprobe_binary",
]

log = structlog.get_logger(__name__)

#: Width the scoring pass works at. Small enough to be free, large enough that the
#: Laplacian still sees real edges rather than resampling noise.
_SCORE_WIDTH = 480

#: 35 mm still format is 36 mm wide; the EXIF "focal length in 35 mm film" tag is defined
#: against exactly that, which is what makes the pixel conversion below exact.
_FILM_WIDTH_MM = 36.0

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ffmpeg_binary() -> str:
    """Locate ffmpeg, honouring ``BIOSPHERE_FFMPEG`` for unusual installs."""
    return os.environ.get("BIOSPHERE_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_binary() -> str:
    """Locate ffprobe, honouring ``BIOSPHERE_FFPROBE`` for unusual installs."""
    return os.environ.get("BIOSPHERE_FFPROBE") or shutil.which("ffprobe") or "ffprobe"


# ======================================================================================
# Probing
# ======================================================================================


@dataclass(slots=True)
class CaptureSidecar:
    """What the phone measured while it filmed (schema ``biosphere.capture.v1``).

    Read from the ``.json`` the capture app uploads beside the footage. A PRIOR AND A
    DIAGNOSTIC, never truth: the app's own verdict and its blur budget are surfaced as
    warnings so a soft panorama can say why it is soft, and the fields the engine may one
    day steer by are kept together here.
    """

    path: Path
    verdict: str | None
    smear_fraction: float | None
    exposure_s: float | None
    mean_fps: float | None
    pivot_radius_m: float | None
    raw: dict[str, Any]

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.verdict and self.verdict not in ("great", "gaps"):
            out.append(
                f"phone_verdict_{self.verdict}: the capture app graded this recording "
                f"'{self.verdict}' before upload"
            )
        if self.smear_fraction is not None and self.smear_fraction > 0.35:
            shutter = f" at a {1000.0 * self.exposure_s:.0f} ms shutter" if self.exposure_s else ""
            out.append(
                f"motion_blur: {100.0 * self.smear_fraction:.0f}% of the frames exceeded the "
                f"phone's motion-smear budget{shutter}; a slower sweep or more light would "
                "give sharper detail"
            )
        if self.mean_fps is not None and self.mean_fps < 20.0:
            out.append(
                f"low_frame_rate: the phone delivered {self.mean_fps:.1f} fps, so fewer "
                "candidate frames were available per direction"
            )
        # The capture app tries to cap the shutter and reverts when the sensor will not
        # cooperate; when that happened, the exposure in these frames is whatever the phone's
        # own auto-exposure chose, and the only lever left is the room and the sweep speed.
        # Measured on the reference phone: manual mode came back 3.84 stops dark because its
        # reported ISO is not the gain actually applied. See `SHUTTER_CAP_FIELD_RESULT`.
        cap = ((self.raw.get("device") or {}).get("shutterCap")) or {}
        if cap and cap.get("applied") is False and cap.get("reason") not in (None, "fast_enough", "unsupported"):
            shutter = ""
            if isinstance(cap.get("exposureBeforeS"), (int, float)) and cap["exposureBeforeS"] > 0:
                shutter = f" and stayed at {1000.0 * float(cap['exposureBeforeS']):.0f} ms"
            out.append(
                f"shutter_cap_refused: this phone would not take a shorter exposure "
                f"({cap.get('reason')}){shutter}. Sharper frames have to come from more light "
                "or a slower sweep."
            )
        return out


def load_capture_sidecar(paths: Sequence[Path | str]) -> CaptureSidecar | None:
    """Find and parse the capture sidecar uploaded beside these videos, if any."""
    seen: set[Path] = set()
    for p in paths:
        folder = Path(p).parent
        if folder in seen:
            continue
        seen.add(folder)
        for cand in sorted(folder.glob("*.json")):
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or not str(data.get("schema", "")).startswith("biosphere.capture"):
                continue
            device = data.get("device") or {}
            ts = device.get("trackSettings") or {}
            exposure = ts.get("exposureTime")
            image = data.get("image") or {}
            recording = data.get("recording") or {}
            translation = data.get("translation") or {}
            return CaptureSidecar(
                path=cand,
                verdict=str(data["verdict"]) if data.get("verdict") else None,
                smear_fraction=_as_float(image.get("fractionOverSmearBudget")),
                # Android reports exposureTime in units of 100 microseconds.
                exposure_s=(float(exposure) * 1e-4) if isinstance(exposure, (int, float)) and exposure > 0 else None,
                mean_fps=_as_float(recording.get("meanFps")),
                pivot_radius_m=_as_float(translation.get("medianPivotRadiusM")),
                raw=data,
            )
    return None


def _as_float(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


@dataclass(slots=True)
class VideoInfo:
    """What ffprobe can tell us about a capture before we decode a single frame."""

    path: Path
    width: int
    height: int
    fps: float
    duration_s: float
    rotation_deg: int
    codec: str
    n_frames: int

    @property
    def is_rotated_quarter(self) -> bool:
        """True when the display matrix swaps width and height."""
        return abs(self.rotation_deg) % 180 == 90

    @property
    def display_width(self) -> int:
        """Width *after* ffmpeg applies the display matrix — what extraction produces."""
        return self.height if self.is_rotated_quarter else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.is_rotated_quarter else self.height


def _parse_rate(value: str | None) -> float:
    """Parse an ffprobe rational such as ``30000/1001``."""
    if not value:
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rotation_from_stream(stream: dict) -> int:
    """Recover the display rotation from either of the two places ffprobe hides it.

    Modern ffmpeg reports it as ``side_data_list[].rotation`` (a Display Matrix entry);
    older muxers and many Android phones instead set the container tag ``rotate``. Both
    are consulted because a file can legitimately carry only one of them.
    """
    rotation = 0.0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                rotation = float(side["rotation"])
                break
            except (TypeError, ValueError):
                continue
        if str(side.get("side_data_type", "")).lower().startswith("display matrix"):
            # Some builds only emit the raw matrix string; the rotation line follows it.
            matrix = side.get("displaymatrix")
            if isinstance(matrix, str) and "rotation" in matrix:
                for token in matrix.replace(",", " ").split():
                    try:
                        rotation = float(token)
                        break
                    except ValueError:
                        continue
    if rotation == 0.0:
        tag = (stream.get("tags") or {}).get("rotate")
        rotation = _parse_rate(tag) if tag is not None else 0.0
    # Normalise into (-180, 180]; ffprobe's display-matrix rotation is negated relative
    # to the container tag, but only the magnitude/parity matters downstream.
    deg = int(round(rotation)) % 360
    if deg > 180:
        deg -= 360
    return deg


#: Wall-clock ceilings on the two external calls that read the USER'S file.
#:
#: Neither had one. A malformed container, a truncated recording, a stream ffmpeg cannot
#: make sense of — any of these can leave the process waiting on input that never comes, and
#: the pipeline waits with it. With `worker_concurrency = 1` that is not one lost capture: it
#: is every capture after it, queued behind a job that will never end, until somebody thinks
#: to restart the engine. An estate agent uploading something unusual would take the product
#: down for everyone.
#:
#: The numbers are generous against real work — probing reads headers, and extraction of a
#: ten-minute capture at the shipped sampling rate runs well inside a couple of minutes on
#: this hardware — so a call that exceeds them is not slow, it is stuck.
_PROBE_TIMEOUT_S = 60
_EXTRACT_TIMEOUT_S = 900


def _duration_from_packets(path: Path) -> float:
    """Demux to the last video packet to find a length the container never declared.

    Nothing is decoded — this is a sequential read of the packet index, which for a phone
    capture costs a fraction of a second. Returns 0.0 when the scan cannot answer, leaving
    the caller's existing fallbacks in charge.
    """
    cmd = [
        ffprobe_binary(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-print_format",
        "json",
        "-show_entries",
        "packet=pts_time,duration_time",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired):  # pragma: no cover - host dependent
        return 0.0
    if proc.returncode != 0 or not proc.stdout.strip():
        return 0.0
    try:
        packets = json.loads(proc.stdout).get("packets", [])
    except json.JSONDecodeError:
        return 0.0

    end = 0.0
    for packet in packets:
        pts = _parse_rate(packet.get("pts_time"))
        if pts <= 0.0 and packet.get("pts_time") not in {"0", "0.000000"}:
            continue
        end = max(end, pts + _parse_rate(packet.get("duration_time")))
    return float(end)


def probe_video(path: str | Path) -> VideoInfo:
    """Read stream metadata with ffprobe.

    Raises :class:`StitchError` when the file is unreadable or carries no video stream —
    both are user-facing problems (wrong file picked, corrupt upload), not bugs.
    """
    path = Path(path)
    if not path.exists():
        raise StitchError(f"video not found: {path}")
    cmd = [
        ffprobe_binary(),
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=_PROBE_TIMEOUT_S
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on host install
        raise StitchError("ffprobe is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise StitchError(
            f"ffprobe did not finish within {_PROBE_TIMEOUT_S}s on {path.name}; "
            "the file is probably not a readable video"
        ) from exc
    if proc.returncode != 0 or not proc.stdout.strip():
        raise StitchError(f"ffprobe failed on {path.name}: {proc.stderr.strip()[:400]}")
    try:
        meta = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise StitchError(f"ffprobe returned malformed JSON for {path.name}") from exc

    streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise StitchError(f"no video stream in {path.name}")
    stream = streams[0]
    fmt = meta.get("format", {})

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    if fps <= 0.0:
        fps = 30.0
    duration = _parse_rate(stream.get("duration")) or _parse_rate(fmt.get("duration"))
    n_frames = int(_parse_rate(stream.get("nb_frames")) or 0)
    if duration <= 0.0 and n_frames <= 0:
        # A live-muxed WebM — everything Chrome's MediaRecorder writes — declares neither.
        # See `biosphere.core.storage._duration_from_packets` for why, and note that the
        # consequences here are quieter than a rejected upload but no less wrong: the
        # sampler derives the decimation rate from the length, so a zero hands a two-minute
        # capture the full `extract_fps` and blows the frame budget, and the segments of a
        # split capture all start at t=0 on top of one another.
        duration = _duration_from_packets(path)
    if n_frames <= 0 and duration > 0:
        n_frames = int(round(duration * fps))
    if duration <= 0.0 and n_frames > 0:
        duration = n_frames / fps

    info = VideoInfo(
        path=path,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=float(fps),
        duration_s=float(duration),
        rotation_deg=_rotation_from_stream(stream),
        codec=str(stream.get("codec_name") or "unknown"),
        n_frames=int(n_frames),
    )
    log.info(
        "probe_video",
        file=path.name,
        size=f"{info.width}x{info.height}",
        display=f"{info.display_width}x{info.display_height}",
        fps=round(info.fps, 3),
        duration_s=round(info.duration_s, 3),
        rotation_deg=info.rotation_deg,
        codec=info.codec,
        n_frames=info.n_frames,
    )
    return info


# ======================================================================================
# Scoring
# ======================================================================================


def _score_one(path: Path) -> tuple[float, float, float, int, int]:
    """Return ``(sharpness, brightness, clipped_fraction, width, height)`` for one image."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return 0.0, 0.0, 1.0, 0, 0
    h, w = img.shape[:2]

    scale = _SCORE_WIDTH / float(w) if w > _SCORE_WIDTH else 1.0
    if scale < 1.0:
        small = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    brightness = float(lab[:, :, 0].mean())

    flat = small.reshape(-1)
    clipped = float(np.count_nonzero((flat == 0) | (flat == 255))) / float(flat.size)
    return sharpness, brightness, clipped, int(w), int(h)


def _clipping_penalty(clipped_fraction: float) -> float:
    """Down-weight frames that are blown out or crushed.

    A little clipping is normal (a window, a lamp). The penalty only bites past 5 % of
    pixels and never zeroes a frame out, because a clipped frame is still better than a
    hole in the panorama.
    """
    excess = max(0.0, clipped_fraction - 0.05)
    return float(np.clip(1.0 - 4.0 * excess, 0.1, 1.0))


def score_frames(frames: list[Frame], *, n_workers: int = 0) -> list[Frame]:
    """Measure sharpness / brightness / clipping and fill in the rank-based quality.

    Mutates and returns ``frames`` so callers can keep their own ordering.
    """
    if not frames:
        return frames
    workers = n_workers if n_workers > 0 else max(1, (os.cpu_count() or 4) - 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda f: _score_one(f.path), frames))

    sharp = np.array([r[0] for r in results], dtype=np.float64)
    # Rank-normalise: self-calibrating against this capture's own texture level.
    ranks = np.argsort(np.argsort(sharp)).astype(np.float64)
    denom = float(max(len(frames) - 1, 1))
    norm_sharp = ranks / denom

    for frame, (sharpness, brightness, clipped, w, h), q in zip(frames, results, norm_sharp, strict=True):
        frame.sharpness = float(sharpness)
        frame.brightness = float(brightness)
        frame.clipped_fraction = float(clipped)
        frame.quality = float(q) * _clipping_penalty(clipped)
        if w > 0 and h > 0:
            frame.width, frame.height = w, h
    return frames


# ======================================================================================
# Video -> frames
# ======================================================================================


def _sampling_fps(config: PipelineConfig, duration_s: float) -> float:
    """Pick the decimation rate that respects ``max_extracted_frames``."""
    fps = float(config.extract_fps)
    if duration_s > 0.0:
        fps = min(fps, config.max_extracted_frames / duration_s)
    # Never go below one frame every five seconds; below that the capture is unusable
    # anyway and a zero/negative rate would make ffmpeg's fps filter fail outright.
    return float(max(fps, 0.2))


def extract_frames(
    video: VideoInfo | str | Path | Sequence[VideoInfo | str | Path],
    out_dir: str | Path,
    config: PipelineConfig,
    progress: ProgressReporter | None = None,
) -> list[Frame]:
    """Decimate, downscale and score a capture into a pool of candidate frames.

    ``video`` may be SEVERAL SEGMENTS of one capture. A guided capture on a phone is
    routinely split in two: the recorder seals a segment at its own length ceiling, an
    incoming call ends the camera track outright, and the silent-encoder watchdog restarts
    the recorder onto a different codec mid-sweep. Those pieces are consecutive footage of
    the same room from the same spot, so they are one capture and not several.

    They are decimated at a SINGLE sampling rate derived from the summed duration, because
    ``max_extracted_frames`` is a budget for the capture rather than for a file: sampling
    each piece against its own duration would hand a capture twice the frames merely
    because it was interrupted, and the solve would then spend its whole matching budget on
    a room it had already covered.

    The pieces are laid out on one continuous timeline, each starting where the previous
    one's duration ended. The real-world gap while the recorder restarts is not measurable
    from the files, so the timeline is a lower bound on elapsed time — which is the correct
    direction to be wrong in, since every consumer of ``Frame.timestamp`` treats a larger
    gap as weaker evidence that two frames are related.
    """
    segments = list(video) if isinstance(video, (list, tuple)) else [video]
    infos = [v if isinstance(v, VideoInfo) else probe_video(v) for v in segments]
    if not infos:
        raise StitchError("no video segments to extract")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("frame_*.jpg"):
        stale.unlink()

    fps = _sampling_fps(config, sum(i.duration_s for i in infos))
    vf = f"fps={fps:.6f},scale='min({int(config.max_frame_width)},iw)':-2:flags=lanczos"
    if progress is not None:
        progress.stage("extract", 0.05, f"decoding at {fps:.2f} fps")
    log.info(
        "extract_frames.ffmpeg",
        fps=round(fps, 4),
        vf=vf,
        segments=len(infos),
        out_dir=str(out_dir),
    )

    frames: list[Frame] = []
    offset_s = 0.0
    for info in infos:
        # `-start_number` continues the single numbering across segments, so the pool is
        # named and sorted exactly as it was when a capture was always one file.
        cmd = [
            ffmpeg_binary(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(info.path),
            "-vf",
            vf,
            "-q:v",
            "2",
            "-start_number",
            str(len(frames) + 1),
            "-y",
            str(out_dir / "frame_%05d.jpg"),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=_EXTRACT_TIMEOUT_S
            )
        except FileNotFoundError as exc:  # pragma: no cover - depends on host install
            raise StitchError("ffmpeg is not installed or not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise StitchError(
                f"ffmpeg did not finish within {_EXTRACT_TIMEOUT_S}s; the capture is "
                "probably damaged"
            ) from exc
        if proc.returncode != 0:
            raise StitchError(f"ffmpeg failed to extract frames: {proc.stderr.strip()[:400]}")

        produced = sorted(out_dir.glob("frame_*.jpg"))[len(frames) :]
        for i, path in enumerate(produced):
            frames.append(
                Frame(
                    index=len(frames),
                    path=path,
                    timestamp=offset_s + i / fps,
                    # Per segment: the watchdog can restart the recorder onto a different
                    # codec, and iOS hands back a different display matrix after a call.
                    width=info.display_width,
                    height=info.display_height,
                )
            )
        offset_s += info.duration_s

    if not frames:
        raise StitchError("ffmpeg produced no frames; the video may be empty or corrupt")
    if progress is not None:
        progress.stage("extract", 0.6, f"scoring {len(frames)} frames")
    score_frames(frames, n_workers=config.n_workers)
    if progress is not None:
        progress.stage("extract", 1.0, f"{len(frames)} frames extracted")
    log.info(
        "extract_frames.done",
        n=len(frames),
        mean_sharpness=round(float(np.mean([f.sharpness for f in frames])), 2),
    )
    return frames


# ======================================================================================
# Photos -> frames
# ======================================================================================


def _exif_focal_px(path: Path, out_width: int) -> float | None:
    """Best-effort focal prior in pixels of the *normalised* image, or None.

    ``FocalLengthIn35mmFilm`` is the only EXIF tag that is directly convertible without
    knowing the sensor size, so it is preferred. When it is absent we fall back to the
    physical focal length combined with the focal-plane resolution, which is the only
    other self-contained way to recover the sensor width. Anything else would require
    guessing a crop factor, and a wrong prior is worse than no prior.
    """
    try:
        from PIL import Image  # imported lazily; Pillow is only needed for photo input
    except ImportError:  # pragma: no cover - Pillow is a hard dependency
        return None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            ifd = dict(exif)
            try:
                ifd.update(dict(exif.get_ifd(0x8769)))  # ExifOffset sub-IFD
            except (KeyError, AttributeError, OSError):
                # No sub-IFD: the top-level tags read above are all we get, which is
                # still enough when the camera wrote FocalLengthIn35mmFilm there.
                pass
    except (OSError, ValueError):
        return None

    def _tag(tag_id: int) -> float | None:
        value = ifd.get(tag_id)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    exif_width = _tag(0xA002) or _tag(0x0100)  # ExifImageWidth / ImageWidth

    f35 = _tag(0xA405)  # FocalLengthIn35mmFilm
    if f35 and f35 > 0.0:
        return float(f35 / _FILM_WIDTH_MM * out_width)

    focal_mm = _tag(0x920A)  # FocalLength
    fp_res = _tag(0xA20E)  # FocalPlaneXResolution
    fp_unit = _tag(0xA210) or 2.0  # 2 = inch, 3 = cm
    if focal_mm and fp_res and exif_width and fp_res > 0.0:
        unit_mm = 25.4 if int(fp_unit) == 2 else 10.0
        sensor_width_mm = exif_width / fp_res * unit_mm
        if sensor_width_mm > 0.0:
            return float(focal_mm / sensor_width_mm * out_width)
    return None


def _normalise_photo(src: Path, dst: Path, max_width: int) -> tuple[int, int]:
    """Copy ``src`` to ``dst`` upright and no wider than ``max_width``."""
    try:
        from PIL import Image, ImageOps

        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            if img.width > max_width:
                new_h = max(1, int(round(img.height * max_width / img.width)))
                # Keep the height even so downstream video-style tooling stays happy.
                new_h -= new_h % 2
                img = img.resize((max_width, max(2, new_h)), Image.LANCZOS)
            img.save(dst, format="JPEG", quality=95, subsampling=1)
            return img.width, img.height
    except (ImportError, OSError, ValueError):
        img_cv = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img_cv is None:
            raise StitchError(f"cannot read image {src.name}") from None
        h, w = img_cv.shape[:2]
        if w > max_width:
            new_h = max(2, int(round(h * max_width / w)))
            img_cv = cv2.resize(img_cv, (max_width, new_h), interpolation=cv2.INTER_AREA)
            h, w = img_cv.shape[:2]
        cv2.imwrite(str(dst), img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return int(w), int(h)


def load_photos(
    paths: list[str | Path],
    out_dir: str | Path,
    config: PipelineConfig,
    progress: ProgressReporter | None = None,
) -> list[Frame]:
    """Normalise a list of stills into the same Frame pool a video would produce.

    The EXIF focal prior of each photo is stashed on the frame's private cache so
    :func:`photo_focal_prior_px` can aggregate it later without re-reading the files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    srcs = [Path(p) for p in paths]
    srcs = [p for p in srcs if p.suffix.lower() in _IMAGE_SUFFIXES]
    if not srcs:
        raise StitchError("no readable images in the photo set")

    frames: list[Frame] = []
    for i, src in enumerate(srcs):
        dst = out_dir / f"frame_{i + 1:05d}.jpg"
        w, h = _normalise_photo(src, dst, int(config.max_frame_width))
        frame = Frame(index=i, path=dst, timestamp=float(i), width=w, height=h)
        prior = _exif_focal_px(src, w)
        if prior is not None:
            frame._cache["focal_prior_px"] = float(prior)
        frames.append(frame)
        if progress is not None and (i % 10 == 0 or i == len(srcs) - 1):
            progress.stage("extract", 0.6 * (i + 1) / len(srcs), f"normalising photo {i + 1}/{len(srcs)}")

    if progress is not None:
        progress.stage("extract", 0.6, f"scoring {len(frames)} photos")
    score_frames(frames, n_workers=config.n_workers)
    if progress is not None:
        progress.stage("extract", 1.0, f"{len(frames)} photos loaded")
    n_prior = sum(1 for f in frames if "focal_prior_px" in f._cache)
    log.info("load_photos.done", n=len(frames), n_with_exif_focal=n_prior)
    return frames


def photo_focal_prior_px(frames: list[Frame]) -> float | None:
    """Median EXIF focal prior across the photo set, or None when EXIF was useless.

    The median rather than the mean because a single mislabelled shot (a crop, a
    screenshot) would otherwise drag the prior far enough to poison the solve.
    """
    values = [
        float(f._cache["focal_prior_px"])
        for f in frames
        if "focal_prior_px" in f._cache and math.isfinite(float(f._cache["focal_prior_px"]))
    ]
    if not values:
        return None
    prior = float(np.median(values))
    log.info("photo_focal_prior_px", n=len(values), focal_px=round(prior, 2))
    return prior
