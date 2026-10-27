"""Judging a capture *before* the pipeline spends two minutes failing on it.

The stitcher models the capture as a pure rotation about the optical centre. When that
assumption holds, a homography is the exact relation between two views and everything
downstream — self-calibration, bundle adjustment, seam finding — is well posed. When it
does not hold, none of it is: each direction was observed from a different point in
space, the information needed to place them on one sphere is simply not in the file, and
no amount of optimisation invents it. A walkthrough of a flat is the commonest example,
and today's pipeline would chew through one for two minutes and emit a mangled panorama
with a mild warning.

This module measures whether the assumption holds, cheaply, from the extracted frames
alone, so a hopeless capture costs seconds rather than minutes.

The discriminator: rotational purity
------------------------------------
For a pair of nearby frames, fit an unconstrained homography, then project that
homography onto the ``K R K^-1`` manifold (:func:`geometry.nearest_rotation`) and
re-measure the reprojection residual over the same inliers. The free homography is a
*floor*: it is the best any 8-dof model can do on these correspondences, so it absorbs
matching noise, JPEG mush and lens distortion. The rotation-only residual is the same
measurement with the eight degrees of freedom cut down to three. The **gap between them
is the part of the motion that is not a rotation**, and it is the only number that
separates "the user turned on the spot" from "the user walked".

Measured on the two real captures that motivated this module (see
:data:`WALKING_ROT_RESIDUAL_PX` for the full calibration table): a bedroom turn sits at
3.6 px of rotation-only residual against a 1.1 px free-homography floor; a walkthrough of
an empty flat sits at 21.2 px against a 1.6 px floor. Features match perfectly well in
both — the walkthrough's free-homography residual is no worse than several captures that
stitch fine. It is only the *model* that fails in the second, which is why every
feature-count or inlier-based health check in the pipeline waves it through.

The focal length is not assumed, it is solved for
-------------------------------------------------
This matters far more than it sounds, and getting it wrong is what makes a naive version
of this check useless. Projecting a homography onto ``K R K^-1`` requires a ``K``, and if
that ``K`` is wrong the projection cannot reproduce the homography *even for a perfect
pure rotation* — the focal error alone manufactures a residual that looks exactly like
translation. Measured on Video A (a real bedroom turn, 478 px wide) by sweeping the
assumed field of view — residuals here in that capture's own pixels, un-normalised:

=========  =========  =====================
hfov (deg) focal (px) median rot-only (px)
=========  =========  =====================
30         892        12.77
35         758         5.57
**40**     **657**     **3.13**
46         563         6.20
55         459         9.16
62         ~397       10.55  *(the fixed guess the reference script used)*
67         361        11.37
85         261        12.92
=========  =========  =====================

The same capture reads 3.1 px or 12.8 px depending only on a number nobody measured. So
the scan searches the focal that best explains the pairs as rotations, over the same
30-110 degree envelope :mod:`biosphere.pano.rotation` allows, and reports the residual at
that optimum. The search is the physically correct thing to do — self-calibration is part
of the rotation model, not an input to it — and it is also what makes the statistic
*decisive*: a pivot has a focal that explains it, and a walk does not. Swept the same way,
Video B's residual bottoms out at 21.3 px and rises in both directions; no lens makes a
walk look like a turn.

A useful by-product: the recovered field of view is a better estimate than the
homography-algebra one for exactly the captures that need judging, because
:func:`~biosphere.pano.rotation.estimate_focal` reads its answer off homographies that
parallax has already corrupted (it returns 55 deg for Video A against the 40 deg that
actually explains the motion).

Units
-----
Residuals are reported in pixels **normalised to a 640-px-wide frame**, which is the
width the reference analysis (``scripts/analyse_walkthrough.py``) used. Reprojection
error in pixels scales linearly with image scale, so without that normalisation a
1024-wide capture and a 478-wide capture of identical motion would report numbers 2.1x
apart and no single threshold could serve both. :attr:`CaptureAssessment.median_rot_residual_deg`
carries the same quantity as an angle, using the recovered focal, for anyone who would
rather not think in pixels at all.

Cost
----
Detection is the entire bill — matching and the focal search together cost under half a
second — so the scan caps itself at :data:`MAX_SAMPLES` pairs, gathers them in contiguous
runs so neighbouring pairs share a detection, and runs SIFT on frames downscaled to
:data:`ANALYSIS_WIDTH`. Measured on the two real captures: **3.5 s** on a 101-second,
810-frame capture whose full solve takes 51 s, and **3.2 s** on an 82-second one. The cost
is charged to the ``extract`` stage and reported exactly on
:attr:`CaptureAssessment.elapsed_s`.

That is the price of the check on a capture that passes. On one that does not it is the
entire cost: the walkthrough is refused about 10 s in — extraction plus the scan — against
the 51 s of solving, and the minute or so of rendering after it, that the same file would
otherwise spend producing something unusable.

A cross-check worth recording, since the focal search is the part most likely to be
doubted: on Video A the scan recovers a 38.1 degree field of view from 72 sampled pairs,
and the full bundle adjustment — a hundred times the work, on data the scan never saw —
independently reports 38.9 degrees.

Escape hatch
------------
:class:`~biosphere.pano.types.PipelineConfig` is frozen, so the override is an
environment variable: ``BIOSPHERE_FORCE_STITCH=1`` downgrades every hard verdict to a
warning. A user who insists on seeing what comes out should be able to.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np
import structlog

from . import geometry as geom
from .rotation import MAX_HFOV_DEG, MIN_HFOV_DEG
from .select import DEFAULT_HFOV_DEG
from .types import (
    CaptureKind,
    Frame,
    Intrinsics,
    NullProgress,
    PipelineConfig,
    ProgressReporter,
    StitchError,
)

__all__ = [
    "CaptureVerdict",
    "PuritySample",
    "PurityScan",
    "CaptureAssessment",
    "assess_capture",
    "scan_rotational_purity",
    "purity_stats",
    "classify_purity",
    "force_stitch_enabled",
    "segment_rooms_enabled",
    "WalkingCaptureError",
    "CaptureTooShortError",
    "CaptureTooDarkError",
    "CaptureTooBlurryError",
    "ANALYSIS_WIDTH",
    "REFERENCE_WIDTH",
    "SAMPLE_FPS",
    "MAX_SAMPLES",
    "GOOD_ROT_RESIDUAL_PX",
    "WALKING_ROT_RESIDUAL_PX",
    "WALKING_RESIDUAL_RATIO",
]

log = structlog.get_logger(__name__)


# ======================================================================================
# Scan geometry — how much of the capture we look at, and at what size
# ======================================================================================

#: Frames are downscaled to at most this width before SIFT runs. Detection is where
#: essentially all of the scan's time goes — matching and the focal search together cost
#: under half a second — so this is the knob that decides whether the check is affordable.
#: Measured on Video A: 5.4 s of detection at 640 px against 3.7 s at 400 px for 192
#: frames, with the headline statistic moving from 4.11 px to 3.95 px. 480 keeps the
#: measurement faithful on the wide captures while halving what a 1080p frame costs.
ANALYSIS_WIDTH = 480

#: Residuals are rescaled to the pixel size of a frame this wide, so that captures of
#: different resolutions are directly comparable and one threshold serves all of them.
#: 640 because that is the width the reference analysis quoted its numbers at.
REFERENCE_WIDTH = 640.0

#: Target spacing between the two frames of a measured pair, expressed as a rate. A
#: quarter of a second is short enough that the views overlap heavily (so the homography
#: is well conditioned) and long enough that a real pan produces a measurable baseline.
SAMPLE_FPS = 4.0

#: Hard cap on measured pairs. Above this the medians stop moving and we are only buying
#: latency: a 100-second capture at 4 fps would otherwise measure 400 pairs, and the
#: headline statistic differs by under 0.3 px between 48 and 96 of them.
MAX_SAMPLES = 72

#: Samples are gathered in contiguous runs of this length rather than as isolated pairs
#: spread across the capture. Both give the same temporal coverage, but a run of ``L``
#: samples chains through ``L + 1`` frames while ``L`` isolated pairs need ``2L`` — so
#: this very nearly halves the only cost that matters. The runs are still spread evenly
#: over the whole capture, so a room the operator walked through cannot hide between them.
WINDOW_SAMPLES = 8

#: SIFT budget per frame for the scan. A third of what the real solve uses: we are
#: measuring a residual over a couple of hundred inliers, not building a pose graph.
MAX_FEATURES = 1200

#: Lowe ratio for the scan's matcher, matching the reference analysis.
MATCH_RATIO = 0.8

#: A sample needs at least this many raw matches and this many RANSAC inliers before its
#: residual means anything. Below that we record the sample as unusable rather than
#: letting a wild number into the median.
MIN_SAMPLE_MATCHES = 12
MIN_SAMPLE_INLIERS = 20

#: RANSAC threshold for the scan's homography fit, expressed in *reference* pixels and
#: converted to the analysis width like every other length here, so that the inlier set a
#: sample is judged on does not silently depend on the capture's resolution.
RANSAC_THRESH_PX = 3.0
MIN_RANSAC_THRESH_PX = 1.5

#: A quarter-second pair cannot have turned this far — 60 deg in 0.25 s is 240 deg/s, four
#: times what a person filming a room does, and nothing turning that fast leaves a frame
#: sharp enough to match. When a fit claims it, the homography is degenerate rather than
#: fast: one real sample in Video B reported a 165 deg rotation and a 3259 px residual off
#: twenty inliers. Left in, a single such fit poisons the median and, worse, wrecks the
#: trapezoid integration of the turn. Samples that claim it are marked unusable, which is
#: also the honest label — nothing was measured there.
#:
#: :mod:`biosphere.pano.features` applies the same idea to the real solve with a 100 deg
#: bound; this one is tighter because it knows the baseline is a fraction of a second.
MAX_SAMPLE_TURN_DEG = 60.0


# ======================================================================================
# Verdict thresholds
# ======================================================================================

#: Rotation-only residual, normalised to a 640-px frame, at or below which the capture is
#: a clean pivot and nothing needs saying. See :data:`WALKING_ROT_RESIDUAL_PX` for the
#: calibration table this comes from.
#:
#: **This band is narrow on purpose, and it is worth being clear about what it does and
#: does not separate.** The nodal ideal measures 0.98 px. A synthetic capture with a real
#: person's arm radius — the phone orbiting the spine at 0.28-0.35 m, which is what the
#: product's own quality gate simulates — measures 2.34-2.43 px. The real bedroom turn,
#: whose ceiling beams are visibly bent, measures 3.61 px. Those last two are only 1.5x
#: apart, and the synthetic ladder is not even monotone across that region (0.28 m reads
#: 2.43 and 0.35 m reads 2.34), so there is no honest threshold that calls a careful
#: hand-held capture clean and the bedroom turn bent. Any value chosen to do that would be
#: fitted to two files rather than measured.
#:
#: So the line goes where the evidence supports one: at twice the nodal floor, i.e. the
#: point where the parallax term has grown to match the measurement noise. The consequence
#: is that ordinary hand-held captures land in ``degraded`` rather than ``good`` — which is
#: true, is what the capture guide already coaches against ("turn the phone around its own
#: axis, not your body"), and is the same thing the pipeline's existing
#: ``high_reprojection_error`` warning says about them after the fact. ``degraded`` never
#: rejects anything; the decision with real consequences is :data:`WALKING_ROT_RESIDUAL_PX`,
#: and that one has 2.4x of margin on both sides.
GOOD_ROT_RESIDUAL_PX = 2.0

#: At or above this the camera translated far more than it rotated, and a 360 sphere
#: cannot be reconstructed from the file no matter how long the solver is given.
#:
#: **Calibration.** Every row measured with :func:`scan_rotational_purity` — same
#: instrument, same 4 fps sampling, residuals normalised to a 640-px width. The synthetic
#: rows are ray-traced by :mod:`biosphere.synth` in one room (``seed=7``, style ``rich``)
#: with nothing varied but where the lens goes:
#:
#: ================================================  =======  =====  =====  ======  ========
#: capture                                           rot px   free   ratio  deg     verdict
#: ================================================  =======  =====  =====  ======  ========
#: synth pivot, arm 0.00 m (the nodal ideal)            0.98   0.59   1.67   0.072  good
#: synth pivot, arm 0.28 m (a real person's arm)        2.43   0.81   3.00   0.167  degraded
#: synth pivot, arm 0.35 m                              2.34   0.80   2.94   0.170  degraded
#: synth pivot, arm 0.60 m                              2.88   0.78   3.68   0.201  degraded
#: synth pivot, arm 1.20 m                              5.37   0.74   7.23   0.302  degraded
#: synth pivot, arm 0.28 m, "realistic" artefacts       3.45   1.19   2.89   0.239  degraded
#: synth pivot, arm 0.35 m, "harsh" artefacts           4.02   1.26   3.20   0.276  degraded
#: synth **walk** across the room, 0.8 m/s             13.70   0.85  16.19   0.985  walking
#: **Video A** bedroom turn, 101 s, 478x850             3.61   1.08   3.34   0.223  degraded
#: **Video B** walkthrough of a flat, 82 s, 576x1024   21.17   1.60  13.23   2.441  walking
#: ================================================  =======  =====  =====  ======  ========
#:
#: Four things to read off that. The statistic rises monotonically with how far the lens
#: actually moves, so it is measuring what it claims to. The free-homography floor stays
#: flat at 0.6-1.6 px down the whole table — the walkthrough matches its features *better*
#: than several of the good captures — which is exactly why no feature-count or inlier
#: health check in the pipeline can see this problem. The two real videos land 6.1x apart.
#:
#: And the two artefact rows are the ones that decide whether this check is safe to ship:
#: they are the exact scenarios ``tests/test_pipeline_quality.py`` runs the whole product
#: against, motion blur, rolling shutter, lagging auto-exposure, sensor noise, JPEG and
#: all. They measure 3.45 and 4.02 px — a factor of 2.5 clear of the reject threshold. It
#: is also worth noticing that the real bedroom turn (3.61) sits *between* them: whatever
#: else it is, it is a capture of exactly the kind the product already claims to handle,
#: which is the strongest argument there is against rejecting it.
#:
#: **A correction worth recording**: ``arm_radius_m`` is a poor proxy for walking, despite
#: being the obvious knob. Orbiting is *bounded* — over the quarter-second baseline this
#: scan measures, even a 1.20 m orbit displaces the lens by only ~0.1 m, against ~0.2 m for
#: someone strolling at 0.8 m/s. That is why the orbit ladder tops out at 5.37 px and never
#: reaches a walking verdict, and why the synthetic walking row is generated by translating
#: the camera through the room rather than by winding ``arm_radius_m`` up.
#:
#: **Where the threshold goes.** It has to sit above Video A (3.61 px — it produced a
#: usable panorama, and rejecting it would be a false alarm) and below Video B (21.17 px).
#: 10.0 is 2.8x above the first and 2.1x below the second, within a hair of the geometric
#: centre of the admissible interval, which is where a threshold belongs when its two sides
#: differ by a factor of six. The synthetic walk at 13.70 px is independent evidence that
#: the placement is not overfitted to two files: it is a capture whose motion is known by
#: construction, and it lands on the reject side.
#:
#: For scale: 10 px at a 640-px width and a typical phone focal is ~0.9 deg of angular
#: misalignment, which carried onto the 8192-px master is ~20 px of ghosting on every edge
#: in the panorama, before the worse-than-median parts of the capture are counted.
WALKING_ROT_RESIDUAL_PX = 10.0

#: Secondary confirmation for a ``walking`` verdict: the rotation-only residual must also
#: be this many times the free-homography floor. Without it, a capture that is merely hard
#: to match (dim, low texture, heavy compression) could reach the absolute threshold
#: through noise alone and be blamed for a motion problem it does not have — the floor
#: rises with the noise, and the ratio does not. Video A 3.34, Video B 13.23, the synthetic
#: walk 16.19; 6.0 sits 1.8x above the capture that must pass and 2.2x below both that must
#: fail. It is a guard, not the discriminator, which is why it is the looser of the two.
WALKING_RESIDUAL_RATIO = 6.0

#: A ``walking`` verdict is only issued when at least this many samples were usable.
#: Refusing a capture on the evidence of three measurements would be worse than letting
#: the full solve have a go and report honestly.
MIN_SAMPLES_FOR_WALKING = 8

#: Below this the capture is not long enough to judge or to stitch. Both conditions are
#: needed: a photo set is legitimately a dozen frames, and a video is legitimately dense.
MIN_FRAMES = 6
MIN_VIDEO_DURATION_S = 6.0

#: Median frame brightness (CIE L*, 0..255 as OpenCV scales it) below which the capture is
#: too dark to match. **Not calibrated against a real dark capture** — there wasn't one to
#: calibrate against — so it is deliberately set far below anything measured rather than
#: at a boundary: the two real videos come in at 163 and 144, and the synthetic captures
#: between 117 and 132. 25 is under a fifth of the darkest of those. A capture that trips
#: it is one shot with the lights off, not one shot on a dull afternoon, and the honest
#: statement is that this threshold is a floor rather than a measured discriminator.
MIN_MEAN_BRIGHTNESS = 25.0

#: "Too blurry" needs *both* symptoms, and the reason is worth stating because the
#: single-symptom version is so tempting.
#:
#: Laplacian variance has no absolute meaning — it scales with scene texture, exposure and
#: resolution, which is exactly why :mod:`biosphere.pano.extract` rank-normalises it rather
#: than thresholding it. The measured spread here proves the point in the most inconvenient
#: possible direction: the sphere-rendered nodal capture used as this module's positive
#: control measures **8.6**, and the unstitchable walkthrough measures **27**. A capture
#: that stitches perfectly is three times "blurrier" than one that cannot be stitched at
#: all. Any absolute threshold would therefore reject the wrong one.
#:
#: So the Laplacian only ever corroborates: a capture is called too blurry when the scan
#: could not use most of its samples **and** the frames are also objectively textureless.
#: Either alone is a false alarm — a photo set with wide gaps fails the first while being
#: perfectly sharp, and the control above fails the second while matching perfectly.
MIN_USABLE_SAMPLE_FRACTION = 0.35
MIN_MEAN_SHARPNESS = 12.0

#: Yaw sweep below which the panorama would have a hole. Reported only, as
#: :attr:`CaptureAssessment.completes_circle` — it never raises and never warns, because
#: :mod:`biosphere.pano.solve` measures the same thing authoritatively once the focal is
#: self-calibrated and the surviving keyframes are known, and one fact should not arrive
#: twice with two different numbers attached.
WARN_TURN_DEG = 300.0


# ======================================================================================
# Errors
# ======================================================================================


class _PayloadError(StitchError):
    """A :class:`StitchError` that can carry a structured payload for the API layer."""

    def __init__(self, message: str, **payload: Any) -> None:
        super().__init__(message)
        self.payload: dict[str, Any] = dict(payload)


class WalkingCaptureError(_PayloadError):
    """The operator walked through the scene instead of turning on the spot.

    This is unrecoverable by definition rather than by policy: each direction was
    photographed from a different point in space, so there is no single viewpoint whose
    sphere the frames describe. The copy therefore tells the user how to re-shoot rather
    than offering a lower-quality result, and mirrors the four capture-guide lines the web
    UI already shows (``capture.guide.1`` .. ``capture.guide.4`` in ``web/src/lib/i18n.ts``).

    ``payload`` carries the assessment and, when ``BIOSPHERE_SEGMENT_ROOMS=1``, the room
    segmentation, so the API can eventually offer "we found 5 rooms; here is how to
    re-shoot each" instead of a bare failure.
    """

    user_message_tr = (
        "Bu videoda kamera kendi ekseninde dönmek yerine odanın içinde ilerlemiş; "
        "yürüyerek çekilen bir görüntüden 360° panorama oluşturulamaz. "
        "Lütfen odanın ortasında durun, telefonu dikey tutun ve yürümeden kendi "
        "etrafınızda dönün: yaklaşık 15 saniyede tam bir tur atın, ardından hafif yukarı "
        "ve hafif aşağı bakarak iki tur daha çekin."
    )
    user_message_en = (
        "The camera moved through the room instead of turning on the spot, and a 360° "
        "panorama cannot be built from a walking capture. Stand in the middle of the "
        "room, hold the phone upright and turn on the spot without walking: one full "
        "circle in about fifteen seconds, then two more turns looking slightly up and "
        "slightly down."
    )


class CaptureTooShortError(_PayloadError):
    user_message_tr = (
        "Çekim çok kısa. Odanın ortasında durup yaklaşık 15 saniyede tam bir tur atın."
    )
    user_message_en = (
        "The capture is too short. Stand in the middle of the room and take about "
        "fifteen seconds to complete one full turn."
    )


class CaptureTooDarkError(_PayloadError):
    user_message_tr = (
        "Görüntü çok karanlık; kareler eşleştirilemiyor. Odanın ışıklarını açıp "
        "perdeleri aralayarak tekrar çekin."
    )
    user_message_en = (
        "The capture is too dark for the frames to be matched. Turn the lights on, open "
        "the curtains, and shoot again."
    )


class CaptureTooBlurryError(_PayloadError):
    user_message_tr = (
        "Kareler bulanık veya dokusuz; kareler eşleştirilemiyor. Telefonu daha yavaş "
        "çevirin ve odaklandığından emin olun."
    )
    user_message_en = (
        "The frames are too blurred or too featureless to be matched. Turn the phone more "
        "slowly and make sure it has focused."
    )


# ======================================================================================
# Verdict
# ======================================================================================


class CaptureVerdict(str, Enum):
    """What the pre-scan concluded about a capture.

    ``GOOD`` and ``DEGRADED`` both proceed; everything else is a hard verdict that raises
    unless ``BIOSPHERE_FORCE_STITCH=1``.
    """

    GOOD = "good"
    DEGRADED = "degraded"
    WALKING = "walking"
    TOO_SHORT = "too_short"
    TOO_DARK = "too_dark"
    TOO_BLURRY = "too_blurry"

    @property
    def is_fatal(self) -> bool:
        return self not in (CaptureVerdict.GOOD, CaptureVerdict.DEGRADED)


_ERROR_FOR_VERDICT: dict[CaptureVerdict, type[_PayloadError]] = {
    CaptureVerdict.WALKING: WalkingCaptureError,
    CaptureVerdict.TOO_SHORT: CaptureTooShortError,
    CaptureVerdict.TOO_DARK: CaptureTooDarkError,
    CaptureVerdict.TOO_BLURRY: CaptureTooBlurryError,
}


def force_stitch_enabled() -> bool:
    """True when ``BIOSPHERE_FORCE_STITCH`` asks for hard verdicts to become warnings."""
    return str(os.environ.get("BIOSPHERE_FORCE_STITCH", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def segment_rooms_enabled() -> bool:
    """True when ``BIOSPHERE_SEGMENT_ROOMS`` asks for a walkthrough to be segmented."""
    return str(os.environ.get("BIOSPHERE_SEGMENT_ROOMS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ======================================================================================
# The scan
# ======================================================================================


@dataclass(slots=True)
class PuritySample:
    """One measured frame pair.

    ``rot_residual_px`` and ``free_residual_px`` are both normalised to
    :data:`REFERENCE_WIDTH`; see the module docstring for why.
    """

    i: int  # position in the frame list of the first frame of the pair
    j: int
    time_s: float  # timestamp of the first frame
    dt_s: float  # seconds between the two frames — the baseline the residual is measured over
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    rot_residual_px: float
    free_residual_px: float
    turn_deg: float  # geodesic angle of the fitted rotation
    yaw_step_deg: float  # signed yaw component of it
    mean_flow_px: float
    usable: bool

    @property
    def residual_ratio(self) -> float:
        return self.rot_residual_px / max(self.free_residual_px, 1e-6)

    @property
    def yaw_rate_deg_s(self) -> float:
        return self.yaw_step_deg / max(self.dt_s, 1e-6)

    @property
    def turn_rate_deg_s(self) -> float:
        return self.turn_deg / max(self.dt_s, 1e-6)

    @property
    def mid_time_s(self) -> float:
        """The moment the sample describes — the midpoint of its baseline, not its start."""
        return self.time_s + 0.5 * self.dt_s


@dataclass(slots=True)
class PurityScan:
    """Everything one pass over the capture measured."""

    samples: list[PuritySample]
    focal_px: float
    analysis_width: int
    analysis_height: int
    normalisation: float  # analysis px -> reference px
    sample_spacing_s: float
    elapsed_s: float
    #: True when the focal search ran into the end of the 30-110 deg envelope, i.e. the
    #: objective never turned over. The residual is still the best available, but the
    #: focal itself should not be quoted as a calibration.
    focal_clamped: bool = False

    @property
    def usable(self) -> list[PuritySample]:
        return [s for s in self.samples if s.usable]

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.analysis_width / 2.0) / max(self.focal_px, 1e-6)))


@dataclass(slots=True)
class _Detection:
    pts: np.ndarray  # (N, 2) float32 in analysis pixels
    desc: np.ndarray | None


def _analysis_gray(frame: Frame, width: int) -> np.ndarray | None:
    """Load a frame as a CLAHE-equalised grayscale image capped at ``width`` pixels wide."""
    img = cv2.imread(str(frame.path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(
            img, (width, max(1, int(round(h * width / float(w))))), interpolation=cv2.INTER_AREA
        )
    # Interior walls and ceilings are nearly flat; the useful variation is local, which is
    # what CLAHE lifts. The real matcher does exactly the same thing, so the scan's
    # feature yield is representative of what the solve will see.
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(img)


def _detect(frame: Frame, width: int) -> tuple[_Detection | None, tuple[int, int]]:
    gray = _analysis_gray(frame, width)
    if gray is None:
        return None, (0, 0)
    sift = cv2.SIFT_create(nfeatures=MAX_FEATURES, contrastThreshold=0.015, edgeThreshold=12)
    kps, desc = sift.detectAndCompute(gray, None)
    size = (int(gray.shape[1]), int(gray.shape[0]))
    if not kps or desc is None:
        return _Detection(np.zeros((0, 2), np.float32), None), size
    return _Detection(np.array([kp.pt for kp in kps], dtype=np.float32), desc), size


def _sample_pairs(
    frames: list[Frame],
    *,
    sample_fps: float,
    max_samples: int | None,
    window: int = WINDOW_SAMPLES,
) -> tuple[list[tuple[int, int]], float]:
    """Which ``(i, j)`` index pairs to measure, and how far apart they are in seconds.

    The *spacing within a pair* is held at ``1 / sample_fps`` regardless of how many pairs
    survive the cap, because the residual depends directly on the baseline: measuring
    pairs a second apart instead of a quarter would inflate every number and quietly
    change what the thresholds mean. When the cap bites it therefore thins the samples out
    in time rather than stretching them — and it thins them in contiguous runs of
    ``window``, so consecutive pairs share a frame and the detection bill nearly halves.

    ``max_samples=None`` measures every pair at the sample rate, which is what
    :mod:`biosphere.pano.rooms` needs: a segmentation cannot find a doorway it never
    looked at.
    """
    n = len(frames)
    if n < 2:
        return [], 0.0
    span = float(frames[-1].timestamp - frames[0].timestamp)
    frame_rate = (n - 1) / span if span > 1e-6 else 1.0
    stride = max(1, int(round(frame_rate / max(sample_fps, 1e-6))))
    starts = list(range(0, n - stride, stride))
    if not starts:
        return [(0, n - 1)], span
    if max_samples is not None and len(starts) > max_samples:
        window = max(1, min(int(window), max_samples))
        n_windows = max(1, max_samples // window)
        last = max(len(starts) - window, 0)
        offsets = np.unique(np.linspace(0.0, float(last), n_windows).round().astype(int))
        keep = {starts[o + k] for o in offsets for k in range(window) if o + k < len(starts)}
        starts = sorted(keep)
    spacing = stride / frame_rate if frame_rate > 1e-6 else 0.0
    return [(i, i + stride) for i in starts], spacing


def _fit_pair(
    a: _Detection, b: _Detection, thresh_px: float = RANSAC_THRESH_PX
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, int]:
    """Ratio-test, RANSAC-verify one pair. Returns ``(H, inliers_a, inliers_b, n_matches)``."""
    empty = np.zeros((0, 2), np.float32)
    if a.desc is None or b.desc is None or len(a.desc) < 2 or len(b.desc) < 2:
        return None, empty, empty, 0
    knn = cv2.BFMatcher().knnMatch(a.desc, b.desc, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < MATCH_RATIO * n.distance]
    if len(good) < MIN_SAMPLE_MATCHES:
        return None, empty, empty, len(good)
    pa = np.array([a.pts[m.queryIdx] for m in good], dtype=np.float32)
    pb = np.array([b.pts[m.trainIdx] for m in good], dtype=np.float32)
    H, mask = cv2.findHomography(
        pa, pb, cv2.USAC_MAGSAC, float(thresh_px), maxIters=8000, confidence=0.999
    )
    if H is None or mask is None or not np.isfinite(H).all():
        return None, empty, empty, len(good)
    inl = mask.ravel().astype(bool)
    return np.asarray(H, dtype=np.float64), pa[inl], pb[inl], len(good)


def _mean_residual(pts_a: np.ndarray, pts_b: np.ndarray, H: np.ndarray) -> float:
    """Mean reprojection error of ``H`` mapping ``pts_a`` onto ``pts_b``, in analysis px."""
    p = np.asarray(pts_a, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    z = p @ H[2, :2] + H[2, 2]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x = (p @ H[0, :2] + H[0, 2]) / z
    y = (p @ H[1, :2] + H[1, 2]) / z
    d = np.stack([x, y], axis=1) - np.asarray(pts_b, dtype=np.float64)
    return float(np.linalg.norm(d, axis=1).mean())


def _K(focal: float, width: int, height: int) -> np.ndarray:
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _rotation_residuals(
    fits: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    focal: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Per-pair rotation-only residual at one assumed focal, in analysis pixels."""
    K = _K(focal, width, height)
    Kinv = np.linalg.inv(K)
    out = np.empty(len(fits), dtype=np.float64)
    for k, (H, pa, pb) in enumerate(fits):
        R = geom.nearest_rotation(Kinv @ H @ K)
        out[k] = _mean_residual(pa, pb, K @ R @ Kinv)
    return out


def _solve_focal(
    fits: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    width: int,
    height: int,
) -> tuple[float, bool]:
    """Find the focal that best explains these homographies as pure rotations.

    Robust objective: the *median* per-pair rotation-only residual, so a handful of pairs
    that genuinely contain translation cannot drag the calibration. A coarse geometric
    grid over the field-of-view envelope brackets the minimum, then a golden-section
    refinement inside the bracket resolves it to well under a degree of field of view.

    Returns ``(focal_px, clamped)``; ``clamped`` is True when the optimum ran into the end
    of the envelope, which means the objective never turned over and the estimate should
    not be trusted as a calibration (it is still the best available for the residual).

    The whole search costs a few tens of milliseconds: each evaluation is one 3x3 SVD and
    one projection of a few hundred points per pair.
    """
    if not fits:
        return float(Intrinsics.from_hfov(width, height, DEFAULT_HFOV_DEG).focal), True

    def _focal_for(hfov_deg: float) -> float:
        return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)

    def _score(focal: float) -> float:
        return float(np.median(_rotation_residuals(fits, focal, width, height)))

    lo_f, hi_f = _focal_for(MAX_HFOV_DEG), _focal_for(MIN_HFOV_DEG)
    grid = np.geomspace(lo_f, hi_f, 25)
    scores = [_score(float(f)) for f in grid]
    best = int(np.argmin(scores))
    clamped = best in (0, len(grid) - 1)

    a = float(grid[max(best - 1, 0)])
    b = float(grid[min(best + 1, len(grid) - 1)])
    if not clamped:
        # Golden-section on log(focal): the objective is smooth and single-minimum inside
        # the bracket, and working in log space keeps the steps proportional.
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        la, lb = math.log(a), math.log(b)
        c, d = lb - inv_phi * (lb - la), la + inv_phi * (lb - la)
        fc, fd = _score(math.exp(c)), _score(math.exp(d))
        for _ in range(14):
            if fc < fd:
                lb, d, fd = d, c, fc
                c = lb - inv_phi * (lb - la)
                fc = _score(math.exp(c))
            else:
                la, c, fc = c, d, fd
                d = la + inv_phi * (lb - la)
                fd = _score(math.exp(d))
        focal = math.exp(0.5 * (la + lb))
    else:
        focal = float(grid[best])
    return float(focal), clamped


def scan_rotational_purity(
    frames: list[Frame],
    config: PipelineConfig,
    *,
    sample_fps: float = SAMPLE_FPS,
    max_samples: int | None = MAX_SAMPLES,
    progress: ProgressReporter | None = None,
    progress_stage: str = "extract",
    progress_span: tuple[float, float] = (0.0, 1.0),
) -> PurityScan:
    """Measure rotational purity across the capture.

    Runs in two passes over the same correspondences, which is what makes the focal search
    affordable. The first pass fits a free homography per pair and keeps its inliers. Then
    :func:`_solve_focal` searches the focal that best explains those homographies as
    rotations — re-scoring costs nothing, because no re-matching is involved — and the
    second pass projects each homography onto the ``K R K^-1`` manifold at that focal and
    re-measures the residual over the same points.

    Solving for the focal rather than assuming the 67-degree phone prior is not a
    refinement, it is the difference between the check working and not: a portrait capture
    is nearer 40 degrees across its short edge, and forcing the wrong focal onto the
    manifold manufactures a residual indistinguishable from translation. See the module
    docstring for the measured sweep.

    ``progress_stage`` defaults to ``"extract"`` deliberately. The scan runs inside the
    extraction phase and :data:`biosphere.pano.pipeline.STAGE_WEIGHTS` is frozen, so a new
    stage name would be forwarded at the progress bar's high-water mark and read as a hang.
    """
    t0 = time.perf_counter()
    pairs, spacing = _sample_pairs(frames, sample_fps=sample_fps, max_samples=max_samples)
    if not pairs:
        return PurityScan([], 0.0, 0, 0, 1.0, 0.0, time.perf_counter() - t0)

    lo, hi = progress_span

    def _report(fraction: float, message: str) -> None:
        if progress is not None:
            progress.stage(progress_stage, lo + (hi - lo) * fraction, message)

    _report(0.0, f"checking capture geometry ({len(pairs)} samples)")

    # -- detect ------------------------------------------------------------------------
    wanted = sorted({k for pair in pairs for k in pair})
    workers = config.n_workers if config.n_workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    detections: dict[int, _Detection] = {}
    size = (0, 0)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, (det, sz) in zip(
            wanted, pool.map(lambda idx: _detect(frames[idx], ANALYSIS_WIDTH), wanted), strict=True
        ):
            if det is not None:
                detections[k] = det
            if sz[0] > 0:
                size = sz
    _report(0.5, "checking capture geometry")

    if size[0] <= 0:
        return PurityScan([], 0.0, 0, 0, 1.0, spacing, time.perf_counter() - t0)
    width, height = size
    normalisation = REFERENCE_WIDTH / float(width)
    thresh_px = max(RANSAC_THRESH_PX / normalisation, MIN_RANSAC_THRESH_PX)

    # -- pass 1: free homographies ------------------------------------------------------
    def _one(pair: tuple[int, int]):
        i, j = pair
        a, b = detections.get(i), detections.get(j)
        if a is None or b is None:
            return pair, None, np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), 0
        H, ia, ib, n_matches = _fit_pair(a, b, thresh_px)
        return pair, H, ia, ib, n_matches

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fits = list(pool.map(_one, pairs))
    _report(0.85, "checking capture geometry")

    # -- focal: the one that best explains these homographies as rotations --------------
    calib = [
        (H, ia, ib)
        for (_i, _j), H, ia, ib, _n in fits
        if H is not None and ia.shape[0] >= MIN_SAMPLE_INLIERS
    ]
    focal, focal_clamped = _solve_focal(calib, width, height)
    K = _K(focal, width, height)
    Kinv = np.linalg.inv(K)

    # -- pass 2: project onto the rotation manifold and re-measure ----------------------
    samples: list[PuritySample] = []
    for (i, j), H, ia, ib, n_matches in fits:
        t = float(frames[i].timestamp)
        dt = max(float(frames[j].timestamp - frames[i].timestamp), 1e-6)
        if H is None or ia.shape[0] < MIN_SAMPLE_INLIERS:
            samples.append(
                PuritySample(
                    i, j, t, dt, n_matches, int(ia.shape[0]), 0.0, math.inf, math.inf, 0.0, 0.0, 0.0, False
                )
            )
            continue
        free = _mean_residual(ia, ib, H) * normalisation
        R = geom.nearest_rotation(Kinv @ H @ K)
        rot = _mean_residual(ia, ib, K @ R @ Kinv) * normalisation
        yaw, _pitch, _roll = geom.yaw_pitch_roll(R)
        turn = math.degrees(geom.angular_distance(R, np.eye(3)))
        plausible = math.isfinite(rot) and math.isfinite(free) and turn <= MAX_SAMPLE_TURN_DEG
        samples.append(
            PuritySample(
                i=i,
                j=j,
                time_s=t,
                dt_s=dt,
                n_matches=n_matches,
                n_inliers=int(ia.shape[0]),
                inlier_ratio=float(ia.shape[0]) / max(n_matches, 1),
                rot_residual_px=rot if plausible else math.inf,
                free_residual_px=free if plausible else math.inf,
                turn_deg=turn if plausible else 0.0,
                yaw_step_deg=math.degrees(yaw) if plausible else 0.0,
                mean_flow_px=float(np.linalg.norm(ib - ia, axis=1).mean()) * normalisation,
                usable=plausible,
            )
        )

    elapsed = time.perf_counter() - t0
    _report(1.0, "capture geometry checked")
    scan = PurityScan(
        samples=samples,
        focal_px=float(focal),
        analysis_width=width,
        analysis_height=height,
        normalisation=normalisation,
        sample_spacing_s=spacing,
        elapsed_s=elapsed,
        focal_clamped=focal_clamped,
    )
    log.info(
        "scan_rotational_purity",
        n_samples=len(samples),
        n_usable=len(scan.usable),
        spacing_s=round(spacing, 3),
        analysis_size=f"{width}x{height}",
        focal_px=round(float(focal), 1),
        hfov_deg=round(scan.hfov_deg, 1),
        focal_clamped=focal_clamped,
        seconds=round(elapsed, 2),
    )
    return scan


# ======================================================================================
# Statistics and classification
# ======================================================================================


_EMPTY_STATS: dict[str, float] = {
    "n_samples": 0.0,
    "n_usable": 0.0,
    "usable_fraction": 0.0,
    "median_rot_residual_px": math.inf,
    "median_free_residual_px": math.inf,
    "residual_ratio": math.inf,
    "frac_below_2px": 0.0,
    "frac_below_4px": 0.0,
    "frac_clean": 0.0,
    "median_inlier_ratio": 0.0,
    "median_inliers": 0.0,
    "median_turn_rate_deg_s": 0.0,
    "total_turn_deg": 0.0,
    "yaw_span_deg": 0.0,
}


def purity_stats(samples: list[PuritySample]) -> dict[str, float]:
    """Robust summary of a set of samples. Unusable samples are excluded, not zeroed.

    The two angular figures are *integrals of a rate*, not sums over samples, because the
    scan does not measure every moment of the capture — it takes runs of samples spread
    across it. Summing per-sample steps would then report only the turning that happened
    inside the runs, which on a 100-second capture measured over sixteen seconds' worth of
    pairs understates the truth about sixfold. Integrating the yaw *rate* over the
    trapezoid between sample midpoints interpolates across the gaps instead: exact when the
    sampling is dense (which is how :mod:`biosphere.pano.rooms` calls it) and a fair
    estimate when it is not.

    ``yaw_span_deg`` is the swept extent — max minus min of the integrated track, i.e. the
    coverage figure — while ``total_turn_deg`` integrates the *unsigned* rate and is
    therefore path length, which is larger whenever the operator turned back on themselves.
    """
    usable = sorted((s for s in samples if s.usable), key=lambda s: s.mid_time_s)
    if not usable:
        return {**_EMPTY_STATS, "n_samples": float(len(samples))}

    rot = np.array([s.rot_residual_px for s in usable], dtype=np.float64)
    free = np.array([s.free_residual_px for s in usable], dtype=np.float64)
    med_rot = float(np.median(rot))
    med_free = float(np.median(free))

    t = np.array([s.mid_time_s for s in usable], dtype=np.float64)
    yaw_rate = np.array([s.yaw_rate_deg_s for s in usable], dtype=np.float64)
    turn_rate = np.array([s.turn_rate_deg_s for s in usable], dtype=np.float64)
    if t.size >= 2:
        dt = np.diff(t)
        track = np.concatenate([[0.0], np.cumsum(dt * 0.5 * (yaw_rate[1:] + yaw_rate[:-1]))])
        yaw_span = float(track.max() - track.min())
        total_turn = float(np.sum(dt * 0.5 * (turn_rate[1:] + turn_rate[:-1])))
    else:
        yaw_span = abs(float(usable[0].yaw_step_deg))
        total_turn = float(usable[0].turn_deg)

    return {
        "n_samples": float(len(samples)),
        "n_usable": float(len(usable)),
        "usable_fraction": len(usable) / float(max(len(samples), 1)),
        "median_rot_residual_px": med_rot,
        "median_free_residual_px": med_free,
        "residual_ratio": med_rot / max(med_free, 1e-6),
        "frac_below_2px": float((rot < 2.0).mean()),
        "frac_below_4px": float((rot < 4.0).mean()),
        "frac_clean": float((rot <= GOOD_ROT_RESIDUAL_PX).mean()),
        "median_inlier_ratio": float(np.median([s.inlier_ratio for s in usable])),
        "median_inliers": float(np.median([s.n_inliers for s in usable])),
        "median_turn_rate_deg_s": float(np.median(np.abs(turn_rate))),
        "total_turn_deg": total_turn,
        "yaw_span_deg": yaw_span,
    }


def classify_purity(stats: dict[str, float]) -> tuple[CaptureVerdict, str]:
    """Judge rotational purity alone. One definition, shared by the validator and rooms.

    Returns the verdict and a short machine-readable reason. Only ``GOOD``, ``DEGRADED``
    and ``WALKING`` come out of here; the exposure and duration rejects are the
    validator's business because they need the frames, not the samples.

    Deliberately just two numbers — the median residual and its ratio to the free-homography
    floor. An earlier version also required a minimum *fraction* of clean samples, which on
    the measured captures only ever restated what the median already said while adding a
    third threshold to justify. The distribution is still reported (``frac_below_2px`` and
    friends) for anyone who wants it; it just does not get a vote.
    """
    n_usable = int(stats["n_usable"])
    med = stats["median_rot_residual_px"]
    ratio = stats["residual_ratio"]

    if n_usable == 0:
        return CaptureVerdict.DEGRADED, "not_measurable"

    if med >= WALKING_ROT_RESIDUAL_PX and ratio >= WALKING_RESIDUAL_RATIO:
        if n_usable >= MIN_SAMPLES_FOR_WALKING:
            return CaptureVerdict.WALKING, "translation_dominates"
        # Too few measurements to refuse a capture on; say so and let the solve try.
        return CaptureVerdict.DEGRADED, "translation_suspected_few_samples"

    if med > GOOD_ROT_RESIDUAL_PX:
        return CaptureVerdict.DEGRADED, "parallax"
    return CaptureVerdict.GOOD, "ok"


# ======================================================================================
# The assessment
# ======================================================================================


@dataclass(slots=True)
class CaptureAssessment:
    """The verdict on a capture plus every number behind it."""

    verdict: CaptureVerdict
    reason: str
    message_tr: str
    message_en: str

    # rotational purity — the discriminator
    median_rot_residual_px: float = math.inf
    median_free_residual_px: float = math.inf
    residual_ratio: float = math.inf
    frac_below_2px: float = 0.0
    frac_below_4px: float = 0.0
    frac_clean: float = 0.0

    # angular coverage
    total_turn_deg: float = 0.0
    yaw_span_deg: float = 0.0
    completes_circle: bool = False

    # capture health
    n_frames: int = 0
    duration_s: float = 0.0
    n_samples: int = 0
    n_usable_samples: int = 0
    median_inliers: float = 0.0
    median_inlier_ratio: float = 0.0
    mean_brightness: float = 0.0
    mean_sharpness: float = 0.0
    clipped_fraction: float = 0.0

    # provenance
    focal_px: float = 0.0
    hfov_deg: float = 0.0
    focal_clamped: bool = False
    analysis_width: int = 0
    sample_spacing_s: float = 0.0
    elapsed_s: float = 0.0
    forced: bool = False
    warnings: list[str] = field(default_factory=list)
    scan: PurityScan | None = field(default=None, repr=False, compare=False)

    @property
    def median_rot_residual_deg(self) -> float:
        """The headline residual as an angle — resolution- and lens-independent."""
        if not math.isfinite(self.median_rot_residual_px) or self.focal_px <= 0.0:
            return math.inf
        # The residual is in reference-width pixels; scale the focal to match.
        focal_ref = self.focal_px * (REFERENCE_WIDTH / max(self.analysis_width, 1))
        return math.degrees(math.atan(self.median_rot_residual_px / focal_ref))

    @property
    def should_proceed(self) -> bool:
        """Whether the pipeline may continue. ``BIOSPHERE_FORCE_STITCH`` forces True."""
        return not self.verdict.is_fatal or self.forced

    def raise_if_fatal(self, **payload: Any) -> None:
        """Raise the verdict's error, unless it was downgraded by the escape hatch."""
        if self.should_proceed:
            return
        error = _ERROR_FOR_VERDICT.get(self.verdict)
        if error is None:  # pragma: no cover - every fatal verdict has an error
            error = _PayloadError
        raise error(self.detail(), assessment=self.as_dict(), **payload)

    def detail(self) -> str:
        """One-line technical explanation, for logs and for the exception message."""
        return (
            f"{self.verdict.value} ({self.reason}): rotation-only residual "
            f"{self.median_rot_residual_px:.2f} px vs free-homography floor "
            f"{self.median_free_residual_px:.2f} px (ratio {self.residual_ratio:.1f}x) over "
            f"{self.n_usable_samples}/{self.n_samples} samples; {self.yaw_span_deg:.0f} deg swept"
        )

    def as_dict(self) -> dict[str, Any]:
        def _num(v: float) -> float | None:
            return None if not math.isfinite(v) else round(float(v), 4)

        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "message_tr": self.message_tr,
            "message_en": self.message_en,
            "median_rot_residual_px": _num(self.median_rot_residual_px),
            "median_free_residual_px": _num(self.median_free_residual_px),
            "median_rot_residual_deg": _num(self.median_rot_residual_deg),
            "residual_ratio": _num(self.residual_ratio),
            "frac_below_2px": round(self.frac_below_2px, 4),
            "frac_below_4px": round(self.frac_below_4px, 4),
            "frac_clean": round(self.frac_clean, 4),
            "total_turn_deg": round(self.total_turn_deg, 1),
            "yaw_span_deg": round(self.yaw_span_deg, 1),
            "completes_circle": self.completes_circle,
            "n_frames": self.n_frames,
            "duration_s": round(self.duration_s, 2),
            "n_samples": self.n_samples,
            "n_usable_samples": self.n_usable_samples,
            "median_inliers": round(self.median_inliers, 1),
            "median_inlier_ratio": round(self.median_inlier_ratio, 4),
            "mean_brightness": round(self.mean_brightness, 2),
            "mean_sharpness": round(self.mean_sharpness, 2),
            "clipped_fraction": round(self.clipped_fraction, 4),
            "focal_px": round(self.focal_px, 2),
            "hfov_deg": round(self.hfov_deg, 2),
            "focal_clamped": self.focal_clamped,
            "analysis_width": self.analysis_width,
            "sample_spacing_s": round(self.sample_spacing_s, 3),
            "elapsed_s": round(self.elapsed_s, 3),
            "forced": self.forced,
            "warnings": list(self.warnings),
        }


_MESSAGES: dict[CaptureVerdict, tuple[str, str]] = {
    CaptureVerdict.GOOD: (
        "Çekim uygun görünüyor.",
        "The capture looks good.",
    ),
    CaptureVerdict.DEGRADED: (
        "Çekim işlenebilir, ancak telefon dönerken bir miktar yer değiştirmiş; "
        "duvar ve tavan çizgilerinde hafif bükülmeler görülebilir. Daha iyi bir sonuç "
        "için telefonu kendi ekseninde, yürümeden çevirin.",
        "The capture is usable, but the phone moved as well as turned; straight lines on "
        "walls and ceilings may bend slightly. For a better result, turn the phone on the "
        "spot without walking.",
    ),
    CaptureVerdict.WALKING: (
        WalkingCaptureError.user_message_tr,
        WalkingCaptureError.user_message_en,
    ),
    CaptureVerdict.TOO_SHORT: (
        CaptureTooShortError.user_message_tr,
        CaptureTooShortError.user_message_en,
    ),
    CaptureVerdict.TOO_DARK: (
        CaptureTooDarkError.user_message_tr,
        CaptureTooDarkError.user_message_en,
    ),
    CaptureVerdict.TOO_BLURRY: (
        CaptureTooBlurryError.user_message_tr,
        CaptureTooBlurryError.user_message_en,
    ),
}


def _frame_health(frames: list[Frame]) -> tuple[float, float, float, bool]:
    """Median brightness, median Laplacian variance, mean clipping, and whether they exist.

    :class:`~biosphere.pano.types.Frame` defaults every score to zero and only
    :func:`~biosphere.pano.extract.score_frames` fills them in. The pipeline always scores
    before it gets here, but a caller using this module directly need not have, and an
    unscored capture reads as pitch black — so it would be refused as ``too_dark`` on the
    strength of a field nobody ever wrote to. The last element says whether the scores are
    real; when they are not, the exposure and blur checks are skipped rather than guessed.
    """
    if not frames:
        return 0.0, 0.0, 0.0, False
    brightness = [f.brightness for f in frames]
    sharpness = [f.sharpness for f in frames]
    scored = max(brightness) > 0.0 or max(sharpness) > 0.0
    return (
        float(np.median(brightness)),
        float(np.median(sharpness)),
        float(np.mean([f.clipped_fraction for f in frames])),
        scored,
    )


def assess_capture(
    frames: list[Frame],
    config: PipelineConfig,
    *,
    progress: ProgressReporter | None = None,
    kind: CaptureKind | None = None,
    progress_span: tuple[float, float] = (0.0, 1.0),
) -> CaptureAssessment:
    """Decide what this capture can yield, before any of it is paid for.

    Call this straight after frame extraction and *before* feature detection: a hopeless
    capture then costs about three seconds instead of the fifty-odd the full solve would
    spend to reach the same conclusion badly.

    ``kind`` lets the duration floor apply to videos only — a photo set is legitimately a
    dozen stills with second-apart nominal timestamps, and rejecting it for being "too
    short" would be nonsense. When it is ``None`` the duration check is skipped and only
    the frame-count floor applies.
    """
    progress = progress if progress is not None else NullProgress()
    n = len(frames)
    duration = float(frames[-1].timestamp - frames[0].timestamp) if n >= 2 else 0.0
    brightness, sharpness, clipped, scored = _frame_health(frames)

    def _build(
        verdict: CaptureVerdict,
        reason: str,
        stats: dict[str, float] | None = None,
        scan: PurityScan | None = None,
        warnings: list[str] | None = None,
    ) -> CaptureAssessment:
        tr, en = _MESSAGES[verdict]
        stats = stats or purity_stats([])
        assessment = CaptureAssessment(
            verdict=verdict,
            reason=reason,
            message_tr=tr,
            message_en=en,
            median_rot_residual_px=stats["median_rot_residual_px"],
            median_free_residual_px=stats["median_free_residual_px"],
            residual_ratio=stats["residual_ratio"],
            frac_below_2px=stats["frac_below_2px"],
            frac_below_4px=stats["frac_below_4px"],
            frac_clean=stats["frac_clean"],
            total_turn_deg=stats["total_turn_deg"],
            yaw_span_deg=stats["yaw_span_deg"],
            completes_circle=stats["yaw_span_deg"] >= WARN_TURN_DEG,
            n_frames=n,
            duration_s=duration,
            n_samples=int(stats["n_samples"]),
            n_usable_samples=int(stats["n_usable"]),
            median_inliers=stats["median_inliers"],
            median_inlier_ratio=stats["median_inlier_ratio"],
            mean_brightness=brightness,
            mean_sharpness=sharpness,
            clipped_fraction=clipped,
            focal_px=scan.focal_px if scan else 0.0,
            hfov_deg=scan.hfov_deg if scan else 0.0,
            focal_clamped=scan.focal_clamped if scan else False,
            analysis_width=scan.analysis_width if scan else 0,
            sample_spacing_s=scan.sample_spacing_s if scan else 0.0,
            elapsed_s=scan.elapsed_s if scan else 0.0,
            forced=force_stitch_enabled() and verdict.is_fatal,
            warnings=list(warnings or []),
            scan=scan,
        )
        # The copy is long and bilingual; it belongs in the response, not in every log line.
        fields = {
            k: v for k, v in assessment.as_dict().items() if k not in ("message_tr", "message_en")
        }
        log.info("assess_capture", **fields)
        return assessment

    # -- the rejects that need no measurement -------------------------------------------
    if n < MIN_FRAMES or (kind is CaptureKind.VIDEO and duration < MIN_VIDEO_DURATION_S):
        return _build(CaptureVerdict.TOO_SHORT, "too_few_frames" if n < MIN_FRAMES else "too_brief")
    if scored and brightness < MIN_MEAN_BRIGHTNESS:
        return _build(CaptureVerdict.TOO_DARK, "underexposed")

    # -- measure -------------------------------------------------------------------------
    scan = scan_rotational_purity(
        frames, config, progress=progress, progress_span=progress_span
    )
    stats = purity_stats(scan.samples)

    # Deliberately no coverage warning here. The scan can see roughly how far the capture
    # turned, but :mod:`biosphere.pano.solve` measures the same thing *after*
    # self-calibrating the focal and knowing which keyframes survived, and two warnings
    # quoting different numbers for one fact is worse than one. ``yaw_span_deg`` and
    # ``completes_circle`` are still reported for anyone calling the validator directly.
    warnings: list[str] = []
    if clipped > 0.10:
        warnings.append(
            f"clipped_exposure: {clipped:.0%} of pixels are blown out or crushed; "
            "detail will be missing where the camera passed a window"
        )

    # Featureless *and* unmatchable is the only combination that means "too blurry".
    # Either alone is a false alarm: a photo set with wide gaps fails the first, a flat
    # white wall fails the second, and neither is unstitchable on its own.
    if (
        scored
        and stats["usable_fraction"] < MIN_USABLE_SAMPLE_FRACTION
        and sharpness < MIN_MEAN_SHARPNESS
    ):
        return _build(CaptureVerdict.TOO_BLURRY, "unmatchable", stats, scan, warnings)

    verdict, reason = classify_purity(stats)
    return _build(verdict, reason, stats, scan, warnings)
