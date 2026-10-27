"""Panorama engine: capture in, equirectangular panorama out.

The public surface is deliberately small — the data contracts, the shared geometry, and
the two orchestration entry points. Everything else (feature matching, seam finding,
blending, pole filling) is an implementation detail that callers should not reach into.

``geometry`` and ``types`` are safe to import anywhere; ``pipeline`` pulls in the heavy
numeric stack, so it is exposed lazily through module ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from biosphere.pano.types import (
    CameraPose,
    CaptureKind,
    CoverageReport,
    Frame,
    InsufficientCoverageError,
    InsufficientOverlapError,
    Intrinsics,
    MatchPair,
    NullProgress,
    PanoLevel,
    PipelineConfig,
    ProgressReporter,
    QualityMode,
    SolveResult,
    StitchError,
    StitchMetrics,
    StitchResult,
    SweepPattern,
)

if TYPE_CHECKING:  # pragma: no cover - import-time cost avoidance, not behaviour
    from biosphere.pano.pipeline import solve_only, stitch_capture

__all__ = [
    "CameraPose",
    "CaptureKind",
    "CoverageReport",
    "Frame",
    "InsufficientCoverageError",
    "InsufficientOverlapError",
    "Intrinsics",
    "MatchPair",
    "NullProgress",
    "PanoLevel",
    "PipelineConfig",
    "ProgressReporter",
    "QualityMode",
    "SolveResult",
    "StitchError",
    "StitchMetrics",
    "StitchResult",
    "SweepPattern",
    "solve_only",
    "stitch_capture",
]

_LAZY = {"stitch_capture", "solve_only"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from biosphere.pano import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
