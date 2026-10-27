"""Media storage abstraction plus upload validation.

Routes and the worker never touch ``pathlib`` directly; they go through :class:`Storage`.
That indirection exists so the local-disk implementation can be swapped for an S3/R2 backed
one without editing a single route — the only thing the rest of the codebase knows is
"give me the directory for this scene" and "give me the public URL for this file".

On-disk layout (all relative to ``settings.media_root``)::

    uploads/{scene_id}/source_00.mp4         raw capture, kept so a scene can be reprocessed
    work/{job_id}/                           pipeline scratch, deleted unless keep_work_dirs
    properties/{property_id}/scenes/{scene_id}/
        pano_uhd.jpg pano_hd.jpg pano_sd.jpg
        pano_preview.jpg pano_preview.webp
        coverage.png thumb.jpg metrics.json
        cube/{px,nx,py,ny,pz,nz}.jpg
        photos/{photo_id}.jpg                the room's ordinary photographs
    restyle/{photo_id}/                      virtual staging; photo_id is the content hash
        original.jpg {style}.jpg meta.json

Uploads are *sniffed with ffprobe*, never trusted by extension: a phone that names a HEVC
capture ``.mp4`` is fine, a ``.mp4`` that is actually a zip is not, and the difference only
shows up when you decode the container.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, runtime_checkable

import cv2
import numpy as np

from biosphere.core.config import Settings, get_settings
from biosphere.core.models import ApiException

__all__ = [
    "Storage",
    "LocalStorage",
    "MediaProbe",
    "probe_media",
    "validate_media",
    "validate_capture_duration",
    "get_storage",
    "write_jpeg",
    "write_png",
    "write_webp",
    "CUBE_FACES",
    "LEVEL_TARGET_WIDTHS",
]

CHUNK_BYTES = 1024 * 1024
CUBE_FACES: tuple[str, ...] = ("px", "nx", "py", "ny", "pz", "nz")

#: Nominal width of each delivered pyramid rung. A level is always emitted at
#: ``min(target, panorama_width)`` — we never upscale, but we also never leave a rung empty,
#: because the player and the AI tools address levels by name (``hd`` in particular).
LEVEL_TARGET_WIDTHS: dict[str, int] = {"uhd": 8192, "hd": 4096, "sd": 2048, "preview": 512}

_IMAGE_CODECS = {
    "mjpeg", "png", "webp", "bmp", "gif", "tiff", "ppm", "jpeg2000", "jpegls", "heic", "avif",
}
_IMAGE_FORMATS = {
    "image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe", "tiff_pipe", "gif", "avif", "heif",
}


# --------------------------------------------------------------------------------------
# Encoding helpers (cv2.imwrite cannot handle non-ASCII paths on Windows)
# --------------------------------------------------------------------------------------


def _encode_to(path: Path, image: np.ndarray, suffix: str, params: list[int]) -> Path:
    ok, buf = cv2.imencode(suffix, image, params)
    if not ok:  # pragma: no cover - only on a corrupt array
        raise ApiException(
            "encode_failed",
            f"Could not encode image as {suffix}.",
            "Görüntü kodlanamadı.",
            status_code=500,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())
    return path


def write_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> Path:
    return _encode_to(path, image, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1])


def write_png(path: Path, image: np.ndarray, compression: int = 6) -> Path:
    return _encode_to(path, image, ".png", [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])


def write_webp(path: Path, image: np.ndarray, quality: int = 80) -> Path:
    return _encode_to(path, image, ".webp", [cv2.IMWRITE_WEBP_QUALITY, int(quality)])


# --------------------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class MediaProbe:
    """What ffprobe actually found inside the uploaded bytes."""

    kind: Literal["video", "image"]
    codec: str
    width: int
    height: int
    duration: float | None
    format_name: str


def _run_ffprobe(path: Path, settings: Settings) -> dict:
    cmd = [
        settings.ffprobe_bin,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    except FileNotFoundError as exc:
        raise ApiException(
            "ffprobe_missing",
            "ffprobe is not installed on the server, so uploads cannot be validated.",
            "Sunucuda ffprobe kurulu değil, bu yüzden yüklemeler doğrulanamıyor.",
            hint="Install ffmpeg and make sure ffprobe is on PATH.",
            status_code=500,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ApiException(
            "probe_timeout",
            "Inspecting the uploaded file timed out.",
            "Yüklenen dosya incelenirken zaman aşımı oluştu.",
            status_code=422,
        ) from exc
    if proc.returncode != 0:
        raise ApiException(
            "unsupported_media",
            "The uploaded file could not be decoded as video or image.",
            "Yüklenen dosya video veya görsel olarak çözümlenemedi.",
            hint="Export the capture as an MP4 (H.264/HEVC) and try again.",
            status_code=415,
        )
    try:
        return json.loads(proc.stdout or b"{}")
    except json.JSONDecodeError as exc:  # pragma: no cover - ffprobe always emits valid json
        raise ApiException(
            "unsupported_media",
            "The uploaded file could not be inspected.",
            "Yüklenen dosya incelenemedi.",
            status_code=415,
        ) from exc


def _looks_like_iso_bmff(path: Path) -> bool:
    """HEIC/HEIF fallback sniff: ISO base media files carry ``ftyp`` at byte 4."""
    try:
        head = path.open("rb").read(16)
    except OSError:  # pragma: no cover
        return False
    return len(head) >= 12 and head[4:8] == b"ftyp"


def _duration_from_packets(path: Path, settings: Settings) -> float | None:
    """Measure the length of a container that never wrote one down.

    Chrome's ``MediaRecorder`` muxes WebM in LIVE mode: clusters of unknown size and no
    Duration element in the Segment header, because the file is being written while it is
    still being recorded and the muxer can never seek back to fill the field in. ffprobe
    then reports no duration at all — not zero, *absent* — for the format AND the stream,
    and ``nb_frames`` with it. That is not an edge case: it is every single video the
    guided capture produces on Android.

    Read as zero, it fails the ``min_video_seconds`` gate, so a user who has just filmed a
    whole room is told their capture is too short. So the length is measured here instead,
    by demuxing to the last video packet. Demuxing only — nothing is decoded — which for a
    150 MB capture is a sequential read rather than a re-encode.

    Returns ``None`` when the scan cannot answer, so the caller keeps whatever the
    container did say rather than replacing a real number with a guess.

    (:func:`biosphere.pano.extract.probe_video` carries the same fallback for the same
    reason. The duplication is deliberate: this module must not import the pipeline, which
    pulls in OpenCV at module scope and would put seconds onto API start-up.)
    """
    cmd = [
        settings.ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-print_format", "json",
        "-show_entries", "packet=pts_time,duration_time",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        packets = json.loads(proc.stdout or b"{}").get("packets", [])
    except json.JSONDecodeError:
        return None

    end = 0.0
    for packet in packets:
        try:
            pts = float(packet["pts_time"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            pts += float(packet.get("duration_time") or 0.0)
        except (TypeError, ValueError):
            pass
        end = max(end, pts)
    return end if end > 0.0 else None


def probe_media(path: Path, settings: Settings | None = None) -> MediaProbe:
    """Decode enough of the file to know what it really is."""
    settings = settings or get_settings()
    try:
        info = _run_ffprobe(path, settings)
    except ApiException:
        # Some ffmpeg builds refuse HEIC. A valid ISO-BMFF header is still a real sniff,
        # so accept it rather than rejecting a perfectly good iPhone photo.
        if path.suffix.lower() in {".heic", ".heif"} and _looks_like_iso_bmff(path):
            return MediaProbe("image", "heic", 0, 0, None, "heif")
        raise

    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ApiException(
            "unsupported_media",
            "The uploaded file contains no image or video stream.",
            "Yüklenen dosyada görüntü akışı bulunamadı.",
            status_code=415,
        )
    stream = streams[0]
    fmt = info.get("format", {})
    format_name = str(fmt.get("format_name", ""))
    codec = str(stream.get("codec_name", ""))

    duration_raw = fmt.get("duration") or stream.get("duration")
    try:
        duration = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration = None
    if duration is None or duration <= 0.0:
        duration = _duration_from_packets(path, settings) or duration

    n_frames_raw = stream.get("nb_frames")
    try:
        n_frames = int(n_frames_raw) if n_frames_raw is not None else None
    except (TypeError, ValueError):
        n_frames = None

    is_image = (
        codec in _IMAGE_CODECS
        or any(part in _IMAGE_FORMATS for part in format_name.split(","))
        or (n_frames == 1 and (duration is None or duration < 0.5))
    )
    return MediaProbe(
        kind="image" if is_image else "video",
        codec=codec,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        duration=duration,
        format_name=format_name,
    )


def validate_capture_duration(
    probes: Sequence[MediaProbe],
    settings: Settings | None = None,
) -> float:
    """Check the length of the whole capture, across however many files it arrived in.

    Separate from :func:`validate_media` because the two questions are different. Whether a
    FILE is a decodable video of a sane length is a property of that file; whether a CAPTURE
    is long enough to contain a full turn of a room is a property of the set. Conflating
    them refused a 123-second capture on the strength of a 4.7-second opening segment.

    Returns the total, so the caller can log what it accepted.
    """
    settings = settings or get_settings()
    total = sum(p.duration or 0.0 for p in probes)
    if total < settings.min_video_seconds:
        pieces = "" if len(probes) <= 1 else f" across {len(probes)} segments"
        pieces_tr = "" if len(probes) <= 1 else f" ({len(probes)} parça toplamı)"
        raise ApiException(
            "video_too_short",
            (
                f"The capture is only {total:.1f}s long{pieces}; at least "
                f"{settings.min_video_seconds:.0f}s is needed to cover a full turn."
            ),
            (
                f"Çekim yalnızca {total:.1f} saniye{pieces_tr}; tam bir tur için en az "
                f"{settings.min_video_seconds:.0f} saniye gerekiyor."
            ),
            hint="Slowly sweep a full 360° turn while recording.",
            status_code=422,
        )
    return total


def validate_media(
    path: Path,
    *,
    expect: Literal["video", "photos"],
    settings: Settings | None = None,
) -> MediaProbe:
    """Reject anything the pipeline cannot use, with copy the user can act on."""
    settings = settings or get_settings()
    suffix = path.suffix.lower()
    allowed = (
        settings.allowed_video_suffixes if expect == "video" else settings.allowed_image_suffixes
    )
    if suffix not in allowed:
        pretty = ", ".join(allowed)
        raise ApiException(
            "unsupported_extension",
            f"'{suffix or path.name}' is not a supported file type. Allowed: {pretty}.",
            f"'{suffix or path.name}' desteklenmeyen bir dosya türü. İzin verilenler: {pretty}.",
            hint=f"Allowed extensions: {pretty}",
            status_code=415,
        )

    probe = probe_media(path, settings)

    if expect == "video":
        if probe.kind != "video":
            raise ApiException(
                "not_a_video",
                "That file is an image, but this scene expects a video capture.",
                "Bu dosya bir görsel; bu sahne için video çekimi gerekiyor.",
                hint="Switch capture_kind to 'photos' or upload a video.",
                status_code=415,
            )
        # NO MINIMUM HERE. The floor is a property of the CAPTURE, and a capture is
        # legitimately several files: the recorder seals a segment at its length ceiling, an
        # incoming call ends the track, and a stall restarts the encoder. Applying the floor
        # per file refused a two-minute capture because its first segment was 4.7 s —
        # footage the user had already shot, thrown away for arriving in two pieces. The
        # total is checked once, by the caller, in `validate_capture_duration`.
        duration = probe.duration or 0.0
        if duration > settings.max_video_seconds:
            raise ApiException(
                "video_too_long",
                (
                    f"The video is {duration:.0f}s long; the maximum is "
                    f"{settings.max_video_seconds:.0f}s."
                ),
                (
                    f"Video {duration:.0f} saniye; en fazla "
                    f"{settings.max_video_seconds:.0f} saniye olabilir."
                ),
                hint="Trim the capture to a single turn of the room.",
                status_code=422,
            )
    else:
        if probe.kind != "image":
            raise ApiException(
                "not_an_image",
                "That file is a video, but this scene expects photos.",
                "Bu dosya bir video; bu sahne için fotoğraf gerekiyor.",
                hint="Switch capture_kind to 'video' or upload photos.",
                status_code=415,
            )
    return probe


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------


@runtime_checkable
class Storage(Protocol):
    """The whole surface the API needs from a media backend."""

    def save_upload(self, src: BinaryIO, *, scene_id: str, filename: str, index: int = 0) -> Path: ...

    def scene_dir(self, property_id: str, scene_id: str) -> Path: ...

    def property_dir(self, property_id: str) -> Path: ...

    def upload_dir(self, scene_id: str) -> Path: ...

    def work_dir(self, job_id: str) -> Path: ...

    def restyle_dir(self, photo_id: str) -> Path: ...

    def scene_photos_dir(self, property_id: str, scene_id: str) -> Path: ...

    def walk_dir(self, property_id: str) -> Path: ...

    def walk_upload_dir(self, walkthrough_id: str) -> Path: ...

    def url_for(self, path: Path, *, versioned: bool = True) -> str: ...

    def delete_tree(self, path: Path) -> None: ...

    def free_bytes(self) -> int: ...


class LocalStorage:
    """:class:`Storage` against the local filesystem under ``settings.media_root``."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def root(self) -> Path:
        return self._settings.media_root

    # -- layout ----------------------------------------------------------------------
    def property_dir(self, property_id: str) -> Path:
        path = self._settings.properties_dir / property_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def scene_dir(self, property_id: str, scene_id: str) -> Path:
        path = self._settings.properties_dir / property_id / "scenes" / scene_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def upload_dir(self, scene_id: str) -> Path:
        path = self._settings.uploads_dir / scene_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def work_dir(self, job_id: str) -> Path:
        path = self._settings.work_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def restyle_dir(self, photo_id: str) -> Path:
        """Where one uploaded photograph and every style rendered from it live."""
        path = self._settings.restyle_dir / photo_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def scene_photos_dir(self, property_id: str, scene_id: str) -> Path:
        """A room's ordinary photographs. Inside the scene directory, so the property's own
        ``delete_tree`` takes them with it."""
        path = self.scene_dir(property_id, scene_id) / "photos"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def walk_dir(self, property_id: str) -> Path:
        """The property's delivered 3D walkthrough (SOG areas, thumbnails, floor plan, manifest).

        Inside the property directory so it is served by the existing media mount and removed by
        the property's ``delete_tree``.
        """
        path = self.property_dir(property_id) / "walk"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def walk_upload_dir(self, walkthrough_id: str) -> Path:
        """Where the general walkthrough video is kept (room videos live with their scene)."""
        path = self._settings.uploads_dir / f"walk_{walkthrough_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- io --------------------------------------------------------------------------
    def save_upload(
        self, src: BinaryIO, *, scene_id: str, filename: str, index: int = 0, prefix: str = "source"
    ) -> Path:
        """Stream an upload to disk in chunks, enforcing ``max_upload_bytes`` as we go.

        Reading a 500 MB phone video into memory would be an easy way to take the service
        down with three concurrent uploads, so the size limit is enforced *during* the copy
        and the partial file is removed the moment it is exceeded.

        ``prefix`` names the file family: ``source`` for a room's own capture, ``walk`` for
        the video it contributes to the property's 3D walkthrough, so the two can share the
        scene's upload directory without one reprocess touching the other's footage.
        """
        suffix = Path(filename or "").suffix.lower()
        target = self.upload_dir(scene_id) / f"{prefix}_{index:02d}{suffix}"
        limit = self._settings.max_upload_bytes
        written = 0
        try:
            with target.open("wb") as dst:
                while True:
                    chunk = src.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit:
                        raise ApiException(
                            "upload_too_large",
                            f"The file is larger than the {limit / (1024 * 1024):.0f} MB limit.",
                            f"Dosya {limit / (1024 * 1024):.0f} MB sınırından büyük.",
                            hint="Record a shorter capture or lower the recording resolution.",
                            status_code=413,
                        )
                    dst.write(chunk)
        except ApiException:
            target.unlink(missing_ok=True)
            raise
        except OSError as exc:
            target.unlink(missing_ok=True)
            raise ApiException(
                "upload_write_failed",
                "The upload could not be written to disk.",
                "Yükleme diske yazılamadı.",
                status_code=500,
            ) from exc

        if written == 0:
            target.unlink(missing_ok=True)
            raise ApiException(
                "empty_upload",
                "The uploaded file is empty.",
                "Yüklenen dosya boş.",
                status_code=422,
            )
        return target

    def url_for(self, path: Path, *, versioned: bool = True) -> str:
        """Public URL for a file under the media root.

        A ``?v=<mtime>`` cache-buster is appended by default: the media mount serves
        ``immutable`` cache headers (mandatory for the cross-origin WebGL texture path), and
        reprocessing a scene rewrites ``pano_hd.jpg`` in place — without the version token a
        browser would happily show the old panorama forever.
        """
        url = self._settings.media_url(Path(path))
        if versioned:
            try:
                url = f"{url}?v={int(Path(path).stat().st_mtime)}"
            except OSError:  # pragma: no cover - file vanished between write and URL
                pass
        return url

    def delete_tree(self, path: Path) -> None:
        """Recursively remove a directory, refusing to step outside the media root."""
        path = Path(path)
        try:
            resolved = path.resolve()
            root = self.root.resolve()
        except OSError:  # pragma: no cover
            return
        if resolved == root or root not in resolved.parents:
            raise ApiException(
                "unsafe_delete",
                "Refusing to delete a path outside the media root.",
                "Medya kökü dışındaki bir yol silinemez.",
                status_code=500,
            )
        shutil.rmtree(resolved, ignore_errors=True)

    def free_bytes(self) -> int:
        try:
            return int(shutil.disk_usage(self.root).free)
        except OSError:  # pragma: no cover
            return 0

    # -- convenience used by the worker ------------------------------------------------
    def cleanup_work(self, job_id: str) -> None:
        """Drop the scratch directory unless the operator asked to keep it for debugging."""
        if self._settings.keep_work_dirs:
            return
        path = self._settings.work_dir / job_id
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


_storage: LocalStorage | None = None


def get_storage() -> LocalStorage:
    """Process-wide storage instance (cheap object, but keeps settings resolution in one place)."""
    global _storage
    if _storage is None or _storage.settings is not get_settings():
        _storage = LocalStorage(get_settings())
    return _storage
