"""Core data contracts for the BioSphere panorama engine.

Every stage of the pipeline consumes and produces the structures declared here.
They are intentionally plain dataclasses (not pydantic) so the numeric core stays
free of validation overhead in tight loops.

Coordinate conventions used everywhere in this package
------------------------------------------------------
World frame:      right-handed and *OpenCV-aligned* — +X right, **+Y down (gravity)**, +Z forward.
                  Choosing Y-down keeps the world and camera axes identical, so an
                  un-rotated camera has ``R == I``. Never assume Y is up here.
Camera frame:     +X right, +Y down, +Z forward (standard OpenCV camera).
Rotation ``R``:   maps *world* direction -> *camera* direction, i.e. ``p_cam = R @ d_world``.
                  The camera's viewing direction in world space is therefore ``R.T @ [0,0,1]``.
Equirectangular:  pixel (u, v) on a ``W x H`` canvas with ``W == 2 * H``.
                  ``theta = 2*pi*(u + 0.5)/W - pi``      in (-pi, pi],  yaw, +theta turns right
                  ``phi   = pi/2 - pi*(v + 0.5)/H``      in (-pi/2, pi/2], elevation, +phi looks up
                  ``ray   = (cos(phi)*sin(theta), -sin(phi), cos(phi)*cos(theta))``
                  Row 0 is the zenith, row H-1 the nadir, column u == 0 is the cyclic seam.

All of the above is implemented once in :mod:`biosphere.pano.geometry`; never re-derive it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

__all__ = [
    "QualityMode",
    "CaptureKind",
    "SweepPattern",
    "Frame",
    "Intrinsics",
    "CameraPose",
    "MatchPair",
    "PanoLevel",
    "CoverageReport",
    "StitchMetrics",
    "SolveResult",
    "StitchResult",
    "PipelineConfig",
    "ProgressReporter",
    "NullProgress",
    "StitchError",
    "InsufficientCoverageError",
    "InsufficientOverlapError",
    "EQUIRECT_ASPECT",
]

EQUIRECT_ASPECT = 2.0


class QualityMode(str, Enum):
    """Presets that trade wall-clock time against artefact suppression."""

    FAST = "fast"
    BALANCED = "balanced"
    STUDIO = "studio"


class CaptureKind(str, Enum):
    VIDEO = "video"
    PHOTOS = "photos"


class SweepPattern(str, Enum):
    """Detected shape of the user's capture motion."""

    SINGLE_ROW = "single_row"
    MULTI_ROW = "multi_row"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Frame:
    """One candidate source image pulled out of the capture."""

    index: int
    path: Path
    timestamp: float
    width: int
    height: int
    sharpness: float = 0.0
    brightness: float = 0.0
    clipped_fraction: float = 0.0
    quality: float = 0.0

    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(slots=True)
class Intrinsics:
    """Pinhole model with two radial distortion terms."""

    focal: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> Intrinsics:
        focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(focal=focal, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    @property
    def K(self) -> np.ndarray:  # noqa: N802 - matches CV literature
        return np.array(
            [[self.focal, 0.0, self.cx], [0.0, self.focal, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.focal))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.focal))

    def scaled(self, factor: float) -> Intrinsics:
        return Intrinsics(
            focal=self.focal * factor,
            cx=self.cx * factor,
            cy=self.cy * factor,
            width=int(round(self.width * factor)),
            height=int(round(self.height * factor)),
            k1=self.k1,
            k2=self.k2,
        )


@dataclass(slots=True)
class CameraPose:
    """Solved orientation and photometric correction for one keyframe."""

    frame_index: int
    R: np.ndarray  # (3, 3) world -> camera
    intrinsics: Intrinsics
    gain: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    vignette: tuple[float, float, float] = (0.0, 0.0, 0.0)  # a1..a3 of 1 + a1 r^2 + a2 r^4 + a3 r^6

    @property
    def yaw(self) -> float:
        d = self.R.T @ np.array([0.0, 0.0, 1.0])
        return math.atan2(d[0], d[2])

    @property
    def pitch(self) -> float:
        """Elevation of the optical axis in radians; positive means the camera looks up."""
        d = self.R.T @ np.array([0.0, 0.0, 1.0])
        return math.asin(float(np.clip(-d[1], -1.0, 1.0)))


@dataclass(slots=True)
class MatchPair:
    """Geometrically verified correspondence set between two keyframes."""

    i: int
    j: int
    pts_i: np.ndarray  # (N, 2) float32, pixel coords in image i
    pts_j: np.ndarray  # (N, 2) float32, pixel coords in image j
    H: np.ndarray | None = None  # (3, 3) homography mapping i -> j
    R: np.ndarray | None = None  # (3, 3) relative rotation, camera_i -> camera_j
    confidence: float = 0.0
    is_loop_closure: bool = False

    @property
    def n_inliers(self) -> int:
        return int(self.pts_i.shape[0])


# --------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class PanoLevel:
    """One resolution rung of the delivered panorama pyramid."""

    name: str  # "preview" | "sd" | "hd" | "uhd"
    width: int
    height: int
    path: Path
    bytes: int = 0


@dataclass(slots=True)
class CoverageReport:
    """Which parts of the sphere the capture actually reached."""

    mask: np.ndarray  # (h, w) uint8, 255 = covered by >=1 source image
    covered_fraction: float  # of the full sphere, 0..1
    horizon_fraction: float  # of the +/-30 deg band, 0..1
    yaw_span_deg: float
    pitch_min_deg: float
    pitch_max_deg: float
    zenith_covered: bool
    nadir_covered: bool
    gaps: list[tuple[float, float, float, float]] = field(default_factory=list)
    # gaps: (yaw_start_deg, yaw_end_deg, pitch_start_deg, pitch_end_deg)


@dataclass(slots=True)
class StitchMetrics:
    """Objective numbers the synthetic harness and the UI both consume."""

    n_frames_extracted: int = 0
    n_keyframes: int = 0
    n_pairs: int = 0
    n_loop_closures: int = 0
    mean_reprojection_error_px: float = 0.0
    max_reprojection_error_px: float = 0.0
    # Headline: the inlier-weighted median over pairs that actually span a quarter turn
    # or more. A plain median over every loop-flagged pair is robust but *not monotone*
    # in real drift — most flagged pairs sit only four to six keyframes apart and see a
    # fraction of a loop's worth of error, so they drown out the one pair that closes the
    # circle. Warn above 2.0 deg, calibrated against measured fidelity loss.
    loop_closure_residual_deg: float = 0.0
    # Worst single loop pair. Diagnostic only: one flimsy short-baseline match can put a
    # large number here on a capture that is perfectly fine.
    loop_closure_max_deg: float = 0.0
    # How many pairs genuinely span the loop. Often just one, and if it is zero the
    # headline falls back to a weighted p90 over everything.
    loop_closure_span_pairs: int = 0
    # Parallax, measured over EVERY verified correspondence after the solve rather than
    # over the survivors of the rejection round: how far the capture is from a pure
    # rotation. The survivor figures above understate it by construction -- the rejection
    # round drops exactly the near-field points that carry it.
    parallax_median_px: float = 0.0
    parallax_p90_px: float = 0.0
    parallax_fraction_over_6px: float = 0.0
    focal_px: float = 0.0
    hfov_deg: float = 0.0
    seam_gradient_energy: float = 0.0  # lower is better; measures visible seam edges
    exposure_spread: float = 0.0
    sweep_pattern: SweepPattern = SweepPattern.UNKNOWN
    stage_seconds: dict[str, float] = field(default_factory=dict)
    #: What the phone that shot this was, and what its lens implies. ``None`` for a capture
    #: with no sidecar (an upload from somewhere else). See :mod:`biosphere.pano.devices`.
    device: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_frames_extracted": self.n_frames_extracted,
            "n_keyframes": self.n_keyframes,
            "n_pairs": self.n_pairs,
            "n_loop_closures": self.n_loop_closures,
            "mean_reprojection_error_px": round(self.mean_reprojection_error_px, 4),
            "max_reprojection_error_px": round(self.max_reprojection_error_px, 4),
            "loop_closure_residual_deg": round(self.loop_closure_residual_deg, 4),
            "loop_closure_max_deg": round(self.loop_closure_max_deg, 4),
            "loop_closure_span_pairs": self.loop_closure_span_pairs,
            "parallax_median_px": round(self.parallax_median_px, 3),
            "parallax_p90_px": round(self.parallax_p90_px, 3),
            "parallax_fraction_over_6px": round(self.parallax_fraction_over_6px, 4),
            "focal_px": round(self.focal_px, 2),
            "hfov_deg": round(self.hfov_deg, 2),
            "seam_gradient_energy": round(self.seam_gradient_energy, 6),
            "exposure_spread": round(self.exposure_spread, 4),
            "sweep_pattern": self.sweep_pattern.value,
            "stage_seconds": {k: round(v, 3) for k, v in self.stage_seconds.items()},
            "warnings": list(self.warnings),
            "device": self.device,
        }


@dataclass(slots=True)
class SolveResult:
    """Hand-off between the geometry solver and the renderer.

    ``frames[i]`` and ``poses[i]`` always describe the same keyframe, and ``poses`` is
    expressed in a *levelled* world frame (horizon on the equator) unless
    ``PipelineConfig.level_horizon`` was disabled.
    """

    frames: list[Frame]
    poses: list[CameraPose]
    pairs: list[MatchPair]
    metrics: StitchMetrics
    work_dir: Path | None = None

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.poses):
            raise ValueError(
                f"frames/poses length mismatch: {len(self.frames)} vs {len(self.poses)}"
            )


@dataclass(slots=True)
class StitchResult:
    """Everything the pipeline hands back to the API layer."""

    panorama: np.ndarray  # (H, W, 3) uint8 BGR, W == 2H
    coverage: CoverageReport
    poses: list[CameraPose]
    metrics: StitchMetrics
    levels: list[PanoLevel] = field(default_factory=list)
    cube_faces: dict[str, Path] = field(default_factory=dict)
    thumbnail: np.ndarray | None = None

    # Capture time, in seconds, of the keyframe behind each entry of ``poses`` (same
    # length, same order). Populated by the pipeline rather than by the renderer, via each
    # pose's ``frame_index``. Without it a caller holding ground truth per *source frame*
    # has no way to line it up against the poses the panorama was composited from — the
    # renderer thins the solved views, so the lists differ in length and which views
    # survive is data-dependent.
    keyframe_timestamps: list[float] = field(default_factory=list)

    @property
    def width(self) -> int:
        return int(self.panorama.shape[1])

    @property
    def height(self) -> int:
        return int(self.panorama.shape[0])


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------



#: The placeholder brand mark, beside the package so it travels with an install.
_BRAND_MARK: Path | None = (
    Path(__file__).resolve().parent.parent / "assets" / "brand_placeholder.png"
)
if _BRAND_MARK is not None and not _BRAND_MARK.exists():
    _BRAND_MARK = None

@dataclass(slots=True)
class PipelineConfig:
    """Tunables for one stitch run. ``PipelineConfig.for_mode`` gives the presets."""

    mode: QualityMode = QualityMode.BALANCED

    # -- frame extraction -------------------------------------------------------------
    extract_fps: float = 8.0
    max_extracted_frames: int = 900
    max_frame_width: int = 1920
    min_sharpness_percentile: float = 35.0

    # -- keyframe selection -----------------------------------------------------------
    target_overlap: float = 0.55  # fraction of frame width shared with the previous keyframe
    min_keyframes: int = 8
    max_keyframes: int = 90

    # -- feature matching -------------------------------------------------------------
    detector: str = "sift"  # sift | akaze | orb
    max_features: int = 4000
    match_ratio: float = 0.78
    ransac_thresh_px: float = 3.0
    min_pair_inliers: int = 28
    neighbour_radius: int = 3  # how many forward neighbours to always match
    loop_search: bool = True

    # -- bundle adjustment ------------------------------------------------------------
    refine_focal: bool = True
    refine_distortion: bool = False

    #: Focal length, in the frames' own pixels, that the PHONE's lens implies -- see
    #: :mod:`biosphere.pano.devices`. ``None`` when the phone could not be identified and
    #: no sidecar was uploaded, which is the pre-2026 behaviour.
    #:
    #: This is not a competing measurement, it is the one piece of information the
    #: measurement cannot supply. Self-calibration on a hand-held sweep is biased LONG and
    #: only long: the arm's tangential motion adds flow in the same direction as the
    #: rotation, and a rotation-only model absorbs it by lengthening the focal. So the
    #: solve is allowed to move the focal freely *within* the lens's own tolerance and not
    #: past it -- ``focal_prior_tolerance`` sigmas either side. Inside the band the data
    #: still decides; outside it, the answer was going to be the bias.
    focal_prior_px: float | None = None
    #: Relative one-sigma uncertainty of ``focal_prior_px`` (0.08 = 8%).
    focal_prior_sigma: float = 0.10
    #: How many sigmas the solve may wander from the prior before it is held. Two sigmas
    #: is a 95% band: wide enough that a correct measurement is never overruled, narrow
    #: enough to cut the parallax bias, which on the reference captures ran to +20%.
    focal_prior_tolerance: float = 2.0
    # This caps ``scipy.optimize.least_squares(max_nfev=...)``, which counts *function
    # evaluations*, not iterations. With a numeric Jacobian each iteration costs about
    # ``n_params + 1`` evaluations, and a 30-keyframe solve has ~90 parameters — so the
    # old value of 120 bought barely one iteration and every solve hit the cap. Measured
    # effect of lifting it on a nodal single-row capture: mean pose error 0.335 deg ->
    # 0.191 deg, with the focal converging toward truth. The solver's own 1e-5 tolerance
    # stops it early once it has actually converged, so this is a ceiling, not a cost.
    ba_max_iterations: int = 3000
    #: Wall-clock ceiling for the whole bundle stage, in seconds. 0 disables it.
    #:
    #: `ba_max_iterations` is a ceiling on ARITHMETIC, and arithmetic is not what anybody
    #: promised. An estate agent shooting a flat was told the properties would be ready by the
    #: time they reached the office, and the honest way to keep that promise is to bound
    #: the clock rather than the iteration count -- because the same ceiling costs four
    #: minutes on a small capture and half an hour on a dense one, and it is the dense one
    #: where the wait actually hurts.
    #:
    #: Running out of budget is NOT a failure. The solver keeps the best parameters it has
    #: seen, which are the product of every completed iteration and are always at least as
    #: good as the initialisation; the panorama is built from those. What the user loses is
    #: the tail of the convergence curve, and `bundle_deadline_hit` in the metrics says so
    #: out loud rather than letting a truncated solve pass as a converged one.
    #: 360 s is not a round number, it is what is left. On the reference 900-frame capture
    #: the stages either side of the solve cost 92 s before it and 445 s of rendering after,
    #: so a fifteen-minute promise leaves the solve six minutes. Measured on that capture,
    #: the sixth minute is nearly free to give up: mean reprojection error falls 6.73 -> 5.70
    #: px between 8 s and 141 s of solving, and loop closure -- the residual that decides
    #: whether the seam lines up where the sweep comes back round -- does not improve at all,
    #: wandering between 0.98 and 1.09 deg the whole way. The tail is arithmetic, not accuracy.
    ba_max_seconds: float = 360.0
    ba_loss: str = "huber"
    ba_f_scale: float = 2.0
    level_horizon: bool = True

    # -- photometric ------------------------------------------------------------------
    exposure_compensation: bool = True
    vignette_correction: bool = True
    per_channel_gain: bool = True

    # -- rendering --------------------------------------------------------------------
    output_width: int = 8192
    interpolation: str = "lanczos4"  # linear | cubic | lanczos4
    #: Refuse a badly-softened frame the right to own a region a sharp one also covers.
    #:
    #: The seam finder chooses where to cut by how well two frames AGREE, which on a plain
    #: ceiling is "perfectly" — there is nothing there to disagree about. Ownership of large
    #: flat areas therefore fell to whichever camera won the cut, blurred or not. Measured on
    #: a real capture's delivered panorama, local sharpness across the ceiling varied 7.4x
    #: more than brightness did: the rectangular patches a viewer reads as a "shift" are a
    #: sharp frame and a motion-smeared one meeting where there is no texture to hide the
    #: change in grain. See `seams._mask_out_the_soft`.
    prefer_sharp_contributor: bool = True
    seam_finder: str = "graphcut"  # graphcut | voronoi | dp
    seam_scale_width: int = 2048
    #: Find the doubled patches after blending and commit each to one camera. See
    #: `blend.repair_ghosted_regions`. Local by construction, which is the whole point: the
    #: same substitution applied everywhere was measured at seam_gradient_energy 1.518.
    repair_ghosts: bool = True
    #: Grey levels of disagreement above which a region counts as ghosted.
    ghost_repair_threshold: float = 1.5
    #: And never more of the canvas than this, whatever the threshold selects. The cap is the
    #: safety argument, not the threshold: at 25% of the canvas the repair stops being local
    #: and `seam_gradient_energy` measured 0.538 -> 1.238 on a real render.
    max_ghost_repair_fraction: float = 0.03
    #: Composite from the views the overlap target asks for, not from every view the solver
    #: kept. See `render.select_render_views` for the measurement.
    thin_render_views: bool = True
    #: Overlap the composite keeps between neighbouring views (fraction of the field of
    #: view); see `render.select_render_views` for why it exceeds `target_overlap`.
    render_view_overlap: float = 0.75
    #: Take high frequencies from the seam's owner rather than from the cross-fade.
    #:
    #: OFF, and the reason is a measurement that contradicted the one that motivated it.
    #: In isolation -- two full-canvas patches, one straight seam -- the split removes 91%
    #: of the doubled-edge ghosting and returns 8% of sharpness for a third of a grey level
    #: of extra step. On the real pipeline it moved `seam_gradient_energy` from 0.634 to
    #: 1.518: handing detail over at a one-pixel line, across the boundaries of ninety
    #: patches rather than one, trades an area artefact for a visible cut at every seam.
    #:
    #: The isolated fixture was the flaw. Two patches that each cover the WHOLE canvas
    #: overlap everywhere, so the blender mixes everywhere and the low band spans the
    #: image; real patches overlap over a fraction of their area and the mixing is local.
    #: Any future attempt has to feather the hand-over rather than cut it, and has to be
    #: judged on `seam_gradient_energy` from a real render before it is believed.
    deghost_structure: bool = False
    blend_bands: int = 6
    blend_strength: float = 5.0

    # -- local (parallax) refinement --------------------------------------------------
    #: Local parallax correction. OFF, and an attempt to turn it on is recorded here so it
    #: is not made a third time.
    #:
    #: The reasoning that says it SHOULD be on is seductive and wrong. A person sweeping a
    #: phone rotates about their spine, not the entrance pupil, so near and far objects
    #: genuinely need different alignments and no rotation model can reconcile them;
    #: `refine.py` was written to deform the overlap until they agree. Enabling it looks
    #: like the obvious cure for the ghosting that residual parallax leaves behind.
    #:
    #: MEASURED BY ABLATION, IT MAKES THINGS WORSE, and `pipeline.py` already refuses it on
    #: exactly the captures that provoke the idea. On a 0.28 m-arm capture: with flow,
    #: straightness 2.53 px and detail 3.70; without, 0.97 px and 4.91. Flow costs a factor
    #: of 2.6 in geometry and a fifth of the sharpness, because under real translation the
    #: disagreement it warps away is a genuine depth difference, not an alignment error.
    #: Warping one view onto another does not recover the hidden surface — it bends the
    #: visible one. On the real bedroom video the same ablation ran 0.98 px -> 5.27 px.
    #:
    #: Enabling it here changes nothing for a hand-held capture (the pipeline guard disables
    #: it again and says so in a warning) and costs render time on the tripod-steady ones
    #: that slip past the guard. STUDIO opts in deliberately; nothing else should.
    optical_flow_refine: bool = False
    flow_max_shift_px: float = 24.0
    flow_backend: str = "auto"  # auto | raft | dis

    # -- local mesh alignment ---------------------------------------------------------
    #: Bend each view on a coarse mesh so the verified feature matches of overlapping views
    #: land on the same canvas pixel. This is what removes the residual parallax the
    #: rotation solve cannot: measured on the reference three-row capture, the best rotation
    #: leaves 4 px of median disagreement inside a row and 10 px between rows. Unlike the
    #: dense optical-flow refinement above, the mesh is jointly solved for all views under a
    #: stiffness prior that keeps each view locally affine, so straight lines stay straight.
    #: See :mod:`biosphere.pano.mesh`.
    mesh_align: bool = True
    #: Mesh cell size as a fraction of the canvas width (1/128 => 2.8 deg per cell). Measured
    #: on the reference capture by overlap disagreement (fraction of overlap pixels whose
    #: gradient magnitude differs by > 40): rigid 0.179, sparse-only 0.137, 1/64 cells 0.116,
    #: 1/128 cells with the stiffness below 0.089.
    mesh_cell_fraction: float = 1.0 / 128.0
    #: Pull toward zero displacement, per vertex. Keeps regions with no overlap in place.
    mesh_prior_weight: float = 0.05
    #: First-difference stiffness (neighbouring vertices move together).
    mesh_stiffness_weight: float = 0.5
    #: Second-difference stiffness (locally affine, hence line preserving).
    mesh_straightness_weight: float = 1.0
    #: Robust threshold of the data term, in canvas pixels at a 4096-wide canvas; scaled
    #: with the output width.
    mesh_huber_px: float = 3.0
    mesh_irls_iterations: int = 5
    #: Second pass: re-warp at a reduced scale with the first solution and block-match
    #: every overlap by phase correlation, so plain edges (cornice, skirting, door frames)
    #: that carry no SIFT corners still get aligned. See :func:`mesh.dense_constraints`.
    mesh_dense: bool = True
    #: Canvas width of the reduced-scale warp the block matcher runs on.
    mesh_dense_canvas: int = 2048
    #: Block size in pixels of that reduced canvas (32 px at 2048 is 5.6 degrees).
    mesh_block_px: int = 32
    #: Weight of a block constraint relative to a feature match.
    mesh_dense_weight: float = 1.0
    #: Extra weight of an edge-like block (its one constraint across the edge) over a
    #: textured one. See `mesh.dense_constraints`.
    mesh_edge_weight: float = 3.0

    # -- poles ------------------------------------------------------------------------
    fill_poles: bool = True
    pole_method: str = "stereographic_inpaint"  # stereographic_inpaint | gradient | none
    #: Branding composited onto the poles, after the caps are filled. Defaults to the mark
    #: shipped in `biosphere/assets`, on BOTH poles: they are the two directions a three-row
    #: sweep never photographed and a viewer rarely looks at, so a mark there replaces
    #: synthesised pixels rather than covering anything the camera saw. Set either to None
    #: to leave that pole bare.
    nadir_logo: Path | None = _BRAND_MARK
    zenith_logo: Path | None = _BRAND_MARK
    #: How wide each mark sits, in degrees of sphere. The nadir gets more: it carries the
    #: operator's feet and the smear a turning phone leaves directly below itself, and the
    #: traditional nadir patch is generous for exactly that reason. The zenith only has to
    #: cover the filler's own boundary, so it stays modest.
    nadir_logo_span_deg: float = 42.0
    zenith_logo_span_deg: float = 30.0

    # -- outputs ----------------------------------------------------------------------
    emit_levels: tuple[str, ...] = ("preview", "sd", "hd", "uhd")
    emit_cubemap: bool = False
    jpeg_quality: int = 92

    # -- runtime ----------------------------------------------------------------------
    n_workers: int = 0  # 0 => os.cpu_count() - 2
    seed: int = 1729

    @classmethod
    def for_mode(cls, mode: QualityMode | str, **overrides: Any) -> PipelineConfig:
        mode = QualityMode(mode)
        if mode is QualityMode.FAST:
            cfg = cls(
                mode=mode,
                extract_fps=5.0,
                max_extracted_frames=400,
                max_frame_width=1280,
                # 0.65, not the 0.45 you would guess from "fast means fewer frames".
                # Measured: at 0.45 the pose graph links only 68% of keyframes and 231 deg
                # of the sweep survives; at 0.65 it links 100% and 366 deg, and it is what
                # lets the wrap-around pair verify at all. Overlap is not a quality dial
                # here, it is what holds the graph together — and a graph that shatters
                # costs far more time in dropped work than the extra frames cost to match.
                target_overlap=0.65,
                max_keyframes=40,
                detector="akaze",
                max_features=2500,
                loop_search=True,
                ba_max_iterations=1200,
                ba_max_seconds=180.0,
                vignette_correction=False,
                output_width=4096,
                interpolation="cubic",
                seam_finder="voronoi",
                seam_scale_width=1024,
                blend_bands=4,
                optical_flow_refine=False,
                emit_levels=("preview", "sd", "hd"),
            )
        elif mode is QualityMode.STUDIO:
            cfg = cls(
                mode=mode,
                extract_fps=12.0,
                max_extracted_frames=1400,
                max_frame_width=2560,
                target_overlap=0.62,
                max_keyframes=140,
                detector="sift",
                max_features=6000,
                ba_max_iterations=9000,
                # The mode whose whole point is "take the time it needs".
                ba_max_seconds=1800.0,
                refine_distortion=True,
                output_width=8192,
                interpolation="lanczos4",
                seam_finder="graphcut",
                seam_scale_width=3072,
                blend_bands=7,
                optical_flow_refine=True,
                emit_cubemap=True,
                jpeg_quality=95,
            )
        else:
            cfg = cls(mode=QualityMode.BALANCED)
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"PipelineConfig has no field {key!r}")
            setattr(cfg, key, value)
        return cfg

    @property
    def output_height(self) -> int:
        return self.output_width // 2


# --------------------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------------------


class ProgressReporter(Protocol):
    """Sink for pipeline progress. The API layer adapts this onto the job store.

    The meaning of ``fraction`` depends on who is calling, and the two conventions meet
    inside :class:`biosphere.pano.pipeline.StageWeightedProgress`:

    * The engine's own stages (``solve``/``render``) report **stage-local** progress —
      ``0.0`` when that stage begins, ``1.0`` when it ends.
    * ``StageWeightedProgress`` consumes those, weights each stage by its measured cost,
      and forwards a **global** ``0..1`` fraction to whatever reporter the caller passed
      in. So an API-side reporter always receives overall completion, already clamped to
      be non-decreasing, and must not re-weight it.
    """

    def stage(self, name: str, fraction: float, message: str = "") -> None: ...

    def log(self, message: str, **fields: Any) -> None: ...


class NullProgress:
    """No-op reporter used by tests and CLI runs that render their own bars."""

    def stage(self, name: str, fraction: float, message: str = "") -> None:  # noqa: D102
        return None

    def log(self, message: str, **fields: Any) -> None:  # noqa: D102
        return None


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class StitchError(RuntimeError):
    """Base class for recoverable pipeline failures surfaced to the user."""

    user_message_tr = "Panorama oluşturulamadı."
    user_message_en = "The panorama could not be created."


class InsufficientOverlapError(StitchError):
    user_message_tr = (
        "Kareler arasında yeterli örtüşme bulunamadı. Lütfen daha yavaş ve düzgün bir tur çekin."
    )
    user_message_en = "Not enough overlap between frames. Please sweep more slowly and evenly."


class InsufficientCoverageError(StitchError):
    user_message_tr = "Çekim 360 dereceyi tamamlamamış. Lütfen tam tur atarak tekrar çekin."
    user_message_en = "The capture does not complete a full 360° turn. Please re-shoot a full circle."
