"""Contracts of the walkthrough pipeline: what goes in, what comes out, in which frame.

THE FRAME. Every number in a result is in the *property frame*: right-handed, ``+X`` right,
``+Y`` DOWN, ``+Z`` forward (the COLMAP/OpenCV convention the panorama engine also uses),
levelled so the floor of the general walkthrough is ``y = 0`` and up is ``-Y``, and scaled
to metres. All areas, base and rooms alike, share it. A viewer that wants ``+Y`` up rotates
the whole thing 180 degrees about ``X``; nothing here pretends otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chain import ChainConfig
from .clean import CleanConfig
from .densify import DenseInitConfig
from .features import MatchingConfig
from .frames import FrameConfig
from .sfm import SfmOptions
from .train import TrainConfig

__all__ = ["SplatQuality", "SplatConfig", "AreaInput", "AreaResult", "PropertySplatResult", "SplatError", "STAGES"]

#: Ordered stages the pipeline reports, with their cost share of one property.
STAGES: tuple[tuple[str, float], ...] = (
    ("extract", 0.06),
    ("features", 0.10),
    ("sfm", 0.14),
    ("register", 0.04),
    ("train", 0.56),
    ("clean", 0.02),
    ("export", 0.08),
)


class SplatError(RuntimeError):
    """A failure with a user-facing explanation in both languages."""

    user_message_en = "The 3D property could not be built from this capture."
    user_message_tr = "Bu çekimden 3B tur oluşturulamadı."

    def __init__(self, message: str, *, en: str | None = None, tr: str | None = None) -> None:
        super().__init__(message)
        if en:
            self.user_message_en = en
        if tr:
            self.user_message_tr = tr


@dataclass(frozen=True)
class SplatQuality:
    name: str
    frames: FrameConfig
    matching: MatchingConfig
    sfm: SfmOptions
    train: TrainConfig
    clean: CleanConfig
    #: Keyframe chain probing and bridging (see ``chain.py``); the same for every mode.
    chain: ChainConfig = field(default_factory=ChainConfig)
    #: Solve one exposure for the frames before training (see ``exposure.py``). Measured on a
    #: real phone walk: +1.29 dB at the same number of steps, for 0.6 s of arithmetic.
    balance_exposure: bool = True
    #: Start the trainer from Depth Anything 3 depth instead of the sparse SfM points (see
    #: ``densify.py``); silently the SfM points when the model is not installed.
    dense_init: DenseInitConfig = field(default_factory=DenseInitConfig)
    #: SH bands kept in the delivered SOG. 0 is view-independent colour and the smallest
    #: file; the first band costs 1.1 MB on a 353,000-splat area (4.52 -> 5.62 MB) and buys
    #: 0.31 dB, which understates it - what it actually removes is the milky veil the model
    #: puts on a wall it saw at a glancing angle, because that veil is exactly the
    #: view-dependent term. The second band changes nothing measurable (24.19 -> 24.17 dB),
    #: so one is where this stops.
    export_sh_degree: int = 1
    sog_iterations: int = 10
    #: Longest edge of the undistorted training images written to disk.
    train_image_long_side: int = 1600

    @classmethod
    def for_mode(cls, mode: str) -> SplatQuality:
        mode = (mode or "balanced").lower()
        if mode == "fast":
            return cls(
                name="fast",
                frames=FrameConfig(keep_per_second=3.0, max_frames=300, long_side=1600),
                matching=MatchingConfig(max_keypoints=3072, detection_threshold=0.02, overlap=12, retrieval_topk=8),
                sfm=SfmOptions(),
                train=TrainConfig.for_mode("fast"),
                clean=CleanConfig(),
                export_sh_degree=0,
                sog_iterations=6,
                train_image_long_side=1280,
            )
        if mode == "studio":
            return cls(
                name="studio",
                frames=FrameConfig(keep_per_second=3.0, max_frames=520, long_side=1920),
                matching=MatchingConfig(max_keypoints=4096, detection_threshold=0.02, overlap=14, retrieval_topk=12),
                sfm=SfmOptions(),
                train=TrainConfig.for_mode("studio"),
                clean=CleanConfig(),
                export_sh_degree=1,
                sog_iterations=16,
                train_image_long_side=1920,
            )
        return cls(
            name="balanced",
            frames=FrameConfig(),
            matching=MatchingConfig(),
            sfm=SfmOptions(),
            train=TrainConfig.for_mode("balanced"),
            clean=CleanConfig(),
        )


@dataclass(frozen=True)
class SplatConfig:
    quality: SplatQuality
    #: Metres assumed for the camera above the floor when no metric depth is available.
    camera_height_m: float = 1.45
    keep_work: bool = False
    #: Only the general walkthrough is trained; rooms are skipped (bench/dev switch).
    base_only: bool = False


@dataclass(frozen=True)
class AreaInput:
    """One video (possibly in segments) that becomes one streamable area."""

    key: str  # "base" for the general walkthrough, else the room's scene id
    name: str
    videos: tuple[Path, ...]
    floor_index: int = 0


@dataclass
class AreaResult:
    key: str
    name: str
    floor_index: int
    sog_path: Path | None = None
    ply_path: Path | None = None
    thumbnail_path: Path | None = None
    env_pano_path: Path | None = None
    splat_count: int = 0
    sog_bytes: int = 0
    #: Camera centres along the walk (property frame), thinned to at most 64.
    cameras: list[list[float]] = field(default_factory=list)
    #: Where the phone pointed at each of ``cameras`` (unit vector, property frame) and when
    #: (seconds into the area's video): ``[lx, ly, lz, t]``. The pointing is the view the
    #: model reproduces best, which the viewer's guided tour follows; the time orders the parts
    #: of one walk.
    looks: list[list[float]] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)
    #: Where the visitor stands on arrival and which way they face (unit, horizontal).
    entry_position: list[float] = field(default_factory=list)
    entry_forward: list[float] = field(default_factory=list)
    bbox_min: list[float] = field(default_factory=list)
    bbox_max: list[float] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failed: str | None = None
    #: False when the room could not be registered into the walkthrough and was levelled on
    #: its own: the viewer then opens it by name only, never by position.
    placed: bool = True
    #: True when the base walkthrough already covers this area's own viewpoints at least as
    #: well as it covers them itself - a piece of the same walk, filmed twice. Such an area
    #: is delivered and opens from the rail by name, but never takes the picture over as the
    #: visitor walks past: see ``splat/coverage.py`` for why that reads as teleporting.
    redundant: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "floor_index": self.floor_index,
            "splat_count": self.splat_count, "sog_bytes": self.sog_bytes, "cameras": self.cameras, "looks": self.looks,
            "centroid": self.centroid, "entry_position": self.entry_position, "entry_forward": self.entry_forward,
            "bbox_min": self.bbox_min, "bbox_max": self.bbox_max, "metrics": self.metrics,
            "warnings": self.warnings, "failed": self.failed, "redundant": self.redundant,
        }


@dataclass
class PropertySplatResult:
    out_dir: Path
    base: AreaResult
    rooms: list[AreaResult]
    floor_plan_path: Path | None
    floor_plan: dict[str, Any]
    manifest_path: Path
    metrics: dict[str, Any]
    warnings: list[str]
