"""Pydantic mirror of ``web/src/lib/types.ts``.

That TypeScript file is the authoritative contract; every model here serialises to exactly
those field names and shapes. Two conventions worth calling out:

* **Timestamps** are emitted as ISO-8601 with a literal ``Z``. SQLite hands back naive
  datetimes, so :func:`_iso_z` re-attaches UTC before formatting — otherwise the browser
  would parse the string as local time and every "2 minutes ago" would be wrong by the
  server's offset.
* **Literal unions instead of enums.** ``Literal["fast", "balanced", "studio"]`` serialises
  to the bare string and rejects anything else at the edge, which is what the TS union means.

The row -> schema converters live here too, because the SSE event payloads must be
byte-identical to the REST responses; one set of converters is the only way to guarantee it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

from biosphere.core import models as m

__all__ = [
    "QualityMode", "CaptureKind", "SweepPattern", "JobStatus", "JobStage", "SceneStatus",
    "HotspotType", "NAVIGATION_TYPES", "ApiError", "PanoramaLevels", "CoverageGap", "Coverage", "StitchMetrics",
    "InitialView", "Hotspot", "Scene", "ScenePhoto", "Property", "PropertyDetail", "Job", "Paginated",
    "HealthResponse", "CreatePropertyRequest", "UpdatePropertyRequest", "UpdateSceneRequest",
    "CreateHotspotRequest", "UpdateHotspotRequest", "CreateRoomRequest",
    "UpdateScenePhotoRequest", "scene_photo_to_schema", "HideFurnitureRequest",
    "HideFurnitureResponse", "RemovedObject", "EmptyRoomRequest", "EmptyRoomResponse",
    "MeasureRequest", "MeasureResponse",
    "MeasurePoint", "MeasureReference", "MeasureSegment", "BBoxCm", "SceneJobResponse",
    "RestyleStyle", "RestyleStylesResponse", "RestylePhoto", "RestyleRequest",
    "RestyleResponse", "RestyleSceneRequest", "RestyleSceneResponse",
    "ScenePhotoVariant", "ScenePhotoRestyleRequest", "ScenePhotoEmptyRequest",
    "property_to_schema", "property_detail_to_schema", "scene_to_schema", "hotspot_to_schema",
    "job_to_schema",
    "JobKind", "WALK_STAGES", "SceneWalk", "WalkEntry", "WalkBox", "WalkArea", "WalkFloorPlanHotspot",
    "WalkFloorPlan", "WalkManifest", "Walkthrough", "WalkthroughJobResponse", "WalkthroughRoomSpec",
    "CreateWalkthroughSpec", "WalkthroughRoomParts", "CreateWalkthroughFromPartsRequest",
    "RebuildWalkthroughRequest", "walkthrough_to_schema",
    "ListingTone", "ListingRoomFact", "ListingFacts", "ListingCopy", "ListingState", "ListingUsage",
    "GenerateListingRequest",
    "InvestmentInputs", "AdvisorSourceRef", "AdvisorCard", "AdvisorPublicState", "AdvisorChatRequest",
    "AdvisorStatus", "AdvisorBudget", "AdvisorUsageSummary", "AdvisorRateRow", "InvestmentState",
]

# --------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------

QualityMode = Literal["fast", "balanced", "studio"]
#: ``none`` is a room created without a capture — no job, no panorama, ready on creation.
#: ``walk`` is a room created by the 3D walkthrough upload: it carries a walkthrough video
#: and, once built, a 3D area, but no panorama unless one is added later.
CaptureKind = Literal["video", "photos", "none", "walk"]
SweepPattern = Literal["single_row", "multi_row", "unknown"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
#: ``stitch`` builds one scene's panorama; ``walkthrough`` builds the property's 3D areas.
JobKind = Literal["stitch", "walkthrough"]
#: The union of both rails. The stitch rail is ``JOB_STAGES`` in types.ts, the walkthrough
#: rail ``WALK_STAGES``; a job's ``kind`` says which one its ``stage`` belongs to.
JobStage = Literal[
    "upload", "probe", "extract", "select", "features", "match", "bundle", "exposure",
    "warp", "seam", "blend", "refine", "poles", "levels", "done",
    "sfm", "register", "train", "clean", "export",
]
WALK_STAGES: tuple[str, ...] = (
    "upload", "extract", "features", "sfm", "register", "train", "clean", "export", "done",
)
SceneStatus = Literal["pending", "processing", "ready", "failed"]
#: ``portal`` is a ``link`` that knows it is standing on the floor.
#:
#: Both navigate to another scene and both need a ``target_scene_id``; the difference is
#: entirely in how the player draws them, and it is the difference between a property somebody
#: walks and a property somebody operates. A link is a tag on a wall — a label with an arrow, in
#: screen space, the same size wherever it is. A portal is a ring lying ON the floor at the
#: point its ray meets it, foreshortened by perspective and shrinking with distance, which is
#: what makes "you can stand there" a thing the picture says rather than a thing the label
#: claims. Nothing on the server behaves differently; the type exists so the player can tell.
HotspotType = Literal["info", "link", "media", "portal"]

#: The types that carry navigation, and therefore need a target that exists.
NAVIGATION_TYPES: tuple[str, ...] = ("link", "portal")
CubeFace = Literal["px", "nx", "py", "ny", "pz", "nz"]


def _iso_z(value: datetime) -> str:
    """ISO-8601 in UTC with a ``Z`` suffix, millisecond precision."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


Timestamp = Annotated[datetime, PlainSerializer(_iso_z, return_type=str, when_used="always")]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------------------
# Core resources
# --------------------------------------------------------------------------------------


class ApiError(_Base):
    code: str
    message: str
    message_tr: str
    hint: str | None = None


class PanoramaLevels(_Base):
    preview_url: str
    sd_url: str | None = None
    hd_url: str | None = None
    uhd_url: str | None = None
    width: int
    height: int
    cube_urls: dict[CubeFace, str] | None = None


class CoverageGap(_Base):
    yaw_start_deg: float
    yaw_end_deg: float
    pitch_start_deg: float
    pitch_end_deg: float


class Coverage(_Base):
    covered_fraction: float
    horizon_fraction: float
    yaw_span_deg: float
    pitch_min_deg: float
    pitch_max_deg: float
    zenith_covered: bool
    nadir_covered: bool
    gaps: list[CoverageGap] = Field(default_factory=list)
    map_url: str | None = None


class DeviceInfo(_Base):
    """The capture device's camera, as identified from the sidecar."""

    id: str = "generic-phone"
    brand: str = ""
    model: str = ""
    tier: str = "mid"
    #: "ua-model" | "ios-fingerprint" | "brand" | "generic" | "none" -- how sure we are.
    source: str = "none"
    #: "main" | "ultrawide" | "unknown"
    lens: str = "main"
    focal_px: float | None = None
    hfov_deg: float | None = None
    sigma_fraction: float = 0.12
    ois: bool = False
    lidar: bool = False
    candidates: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class StitchMetrics(_Base):
    n_frames_extracted: int = 0
    n_keyframes: int = 0
    n_pairs: int = 0
    n_loop_closures: int = 0
    mean_reprojection_error_px: float = 0.0
    max_reprojection_error_px: float = 0.0
    loop_closure_residual_deg: float = 0.0
    loop_closure_max_deg: float = 0.0
    loop_closure_span_pairs: int = 0
    # Parallax over EVERY verified correspondence after the solve (median, 90th percentile
    # and the fraction beyond 6 px): how far the capture is from a pure rotation.
    parallax_median_px: float = 0.0
    parallax_p90_px: float = 0.0
    parallax_fraction_over_6px: float = 0.0
    focal_px: float = 0.0
    hfov_deg: float = 0.0
    seam_gradient_energy: float = 0.0
    exposure_spread: float = 0.0
    sweep_pattern: SweepPattern = "unknown"
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    #: The phone that shot this and what its lens implies -- brand, model, tier, the focal
    #: length the hardware says it has, whether it has optical stabilisation or a LiDAR
    #: scanner, and how the identification was made. ``None`` for an upload with no capture
    #: sidecar. See :mod:`biosphere.pano.devices`.
    device: DeviceInfo | None = None


class InitialView(_Base):
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = Field(default=75.0, gt=10.0, le=140.0)


class Hotspot(_Base):
    id: str
    scene_id: str
    type: HotspotType
    yaw: float
    pitch: float
    title: str
    body: str | None = None
    icon: str | None = None
    target_scene_id: str | None = None
    media_url: str | None = None
    accent: str | None = None
    created_at: Timestamp


class ScenePhoto(_Base):
    """One ordinary photograph attached to a room.

    ``url`` is a media path like every other asset the API hands out, so the same
    ``absoluteUrl`` the player uses for panoramas resolves it. Delivered at a listing-portal
    resolution rather than a texture one — nothing here is going onto a GPU.
    """

    id: str
    scene_id: str
    url: str
    width: int
    height: int
    caption: str | None = None
    order: int
    created_at: str


class Scene(_Base):
    id: str
    property_id: str
    name: str
    status: SceneStatus
    capture_kind: CaptureKind
    quality_mode: QualityMode
    order: int
    panorama: PanoramaLevels | None = None
    coverage: Coverage | None = None
    metrics: StitchMetrics | None = None
    initial_view: InitialView
    hotspots: list[Hotspot] = Field(default_factory=list)
    #: The room's ordinary photographs, in display order. Independent of ``panorama``: a room
    #: may have both, either or neither.
    photos: list[ScenePhoto] = Field(default_factory=list)
    #: The room's 3D area from the property's walkthrough, or null when the room has none.
    walkthrough: SceneWalk | None = None
    active_job_id: str | None = None
    error: ApiError | None = None
    created_at: Timestamp


class SceneWalk(_Base):
    """A room's slice of the property's 3D walkthrough: enough to show a tile and a link."""

    status: Literal["pending", "processing", "ready", "failed"]
    url: str | None = None
    thumbnail_url: str | None = None
    splat_count: int = 0
    bytes: int = 0
    #: False when the room could not be aligned to the general walkthrough: it opens as its
    #: own space rather than as a place inside the flat.
    placed: bool = False
    error: str | None = None


class Property(_Base):
    id: str
    title: str
    description: str | None = None
    address: str | None = None
    public_token: str
    published: bool
    cover_url: str | None = None
    scene_count: int
    views: int
    created_at: Timestamp
    updated_at: Timestamp


class PropertyDetail(Property):
    scenes: list[Scene] = Field(default_factory=list)
    #: The property's 3D walkthrough, or null when none was uploaded.
    walkthrough: Walkthrough | None = None


class Job(_Base):
    id: str
    property_id: str
    #: Null for a property-level job (``kind == "walkthrough"``).
    scene_id: str | None = None
    kind: JobKind = "stitch"
    status: JobStatus
    stage: JobStage
    progress: float
    stage_progress: float
    message: str
    message_tr: str
    eta_seconds: float | None = None
    metrics: dict[str, Any] | None = None
    error: ApiError | None = None
    created_at: Timestamp
    started_at: Timestamp | None = None
    finished_at: Timestamp | None = None


class SceneJobResponse(_Base):
    """Body of ``POST /properties/{id}/scenes`` — the new scene plus the job now processing it."""

    scene: Scene
    job: Job


# --------------------------------------------------------------------------------------
# 3D walkthrough
# --------------------------------------------------------------------------------------


class WalkEntry(_Base):
    position: list[float]
    forward: list[float]


class WalkBox(_Base):
    min: list[float]
    max: list[float]


class WalkArea(_Base):
    """One streamable area of the property: the general walkthrough or a room.

    Every number is in the property frame: right-handed, +X right, +Y DOWN, +Z forward, floor at
    y = 0, metres. A Y-up renderer rotates the whole thing 180 degrees about X.
    """

    key: str
    name: str
    floor_index: int = 0
    url: str | None = None
    bytes: int = 0
    splat_count: int = 0
    thumbnail_url: str | None = None
    env_pano_url: str | None = None
    cameras: list[list[float]] = Field(default_factory=list)
    centroid: list[float] = Field(default_factory=list)
    entry: WalkEntry
    bbox: WalkBox
    placed: bool = False
    #: True when the general walkthrough already covers this area's own viewpoints. The
    #: viewer still lists it and opens it by name; it just never takes the picture over as
    #: the visitor walks past, which would look like being teleported into another room.
    redundant: bool = False
    warnings: list[str] = Field(default_factory=list)
    failed: str | None = None


class WalkFloorPlanHotspot(_Base):
    key: str
    name: str
    x_pct: float
    y_pct: float
    floor_index: int = 0


class WalkFloorPlan(_Base):
    url: str | None = None
    width: int = 0
    height: int = 0
    x0: float = 0.0
    z0: float = 0.0
    px_per_m: float = 0.0
    hotspots: list[WalkFloorPlanHotspot] = Field(default_factory=list)


class WalkManifest(_Base):
    version: int
    frame: dict[str, Any]
    base: WalkArea
    rooms: list[WalkArea] = Field(default_factory=list)
    floor_plan: WalkFloorPlan | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class Walkthrough(_Base):
    id: str
    property_id: str
    status: SceneStatus
    quality_mode: QualityMode
    room_scene_ids: list[str] = Field(default_factory=list)
    manifest: WalkManifest | None = None
    metrics: dict[str, Any] | None = None
    error: ApiError | None = None
    active_job_id: str | None = None
    created_at: Timestamp
    updated_at: Timestamp


class WalkthroughJobResponse(_Base):
    walkthrough: Walkthrough
    job: Job


class WalkthroughRoomSpec(BaseModel):
    """One room of a walkthrough upload: its name and which uploaded files are its video."""

    name: str = Field(min_length=1, max_length=200)
    #: Attach the video to an existing room (its 360 stays); omitted creates a new room.
    scene_id: str | None = None
    floor_index: int = 0
    filenames: list[str] = Field(min_length=1, max_length=16)


class CreateWalkthroughSpec(BaseModel):
    """The ``spec`` form field of ``POST /properties/{id}/walkthrough``; files travel as ``files[]``.

    Uploads ADD to the property's walkthrough: a request without a general video keeps the one
    already there, rooms join the existing rooms (a room with the same ``scene_id`` or name
    is replaced). ``defer_build`` stores the videos without starting the build, which is how
    a phone uploads one area at a time and builds once at the end (``POST .../rebuild``).
    """

    quality_mode: str = ""
    base_filenames: list[str] = Field(default_factory=list, max_length=16)
    rooms: list[WalkthroughRoomSpec] = Field(default_factory=list, max_length=40)
    defer_build: bool = False


class WalkthroughRoomParts(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    scene_id: str | None = None
    floor_index: int = 0
    files: list[ScenePartFile] = Field(min_length=1, max_length=16)


class CreateWalkthroughFromPartsRequest(BaseModel):
    upload_id: str = Field(min_length=6, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    quality_mode: str = ""
    base: list[ScenePartFile] = Field(default_factory=list, max_length=16)
    rooms: list[WalkthroughRoomParts] = Field(default_factory=list, max_length=40)
    defer_build: bool = False


class RebuildWalkthroughRequest(BaseModel):
    quality_mode: str = ""


# --------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------


class CreatePropertyRequest(_Base):
    title: str = Field(min_length=1, max_length=200)
    description: str | None = None
    address: str | None = None


class UpdatePropertyRequest(_Base):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    address: str | None = None
    published: bool | None = None
    cover_scene_id: str | None = None


class UpdateSceneRequest(_Base):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    order: int | None = None
    initial_view: InitialView | None = None


class CreateRoomRequest(_Base):
    """A room with no capture behind it. Photographs are attached afterwards."""

    name: str = Field(min_length=1, max_length=200)


class UpdateScenePhotoRequest(_Base):
    caption: str | None = None
    order: int | None = None


class CreateHotspotRequest(_Base):
    type: HotspotType = "info"
    yaw: float
    pitch: float
    title: str = Field(min_length=1, max_length=200)
    body: str | None = None
    icon: str | None = None
    target_scene_id: str | None = None
    media_url: str | None = None
    accent: str | None = None


class UpdateHotspotRequest(_Base):
    type: HotspotType | None = None
    yaw: float | None = None
    pitch: float | None = None
    title: str | None = Field(default=None, min_length=1, max_length=200)
    body: str | None = None
    icon: str | None = None
    target_scene_id: str | None = None
    media_url: str | None = None
    accent: str | None = None


# --------------------------------------------------------------------------------------
# AI
# --------------------------------------------------------------------------------------


class HideFurnitureRequest(_Base):
    scene_id: str
    polygon: list[tuple[float, float]] = Field(default_factory=list)
    strength: float | None = Field(default=None, ge=0.0, le=1.0)


class RemovedObject(_Base):
    label: str
    confidence: float
    area_px: int


class HideFurnitureResponse(_Base):
    scene_id: str
    image_url: str
    mask_url: str
    removed_objects: list[RemovedObject] = Field(default_factory=list)
    elapsed_ms: int
    model_backed: bool


class EmptyRoomRequest(_Base):
    scene_id: str
    #: How readily a pixel stops counting as part of the room. 0 removes only what is
    #: unmistakably a furnishing; 1 removes anything that differs from the surface behind it.
    #: Defaults to ``ai.empty_room.DEFAULT_STRENGTH``, which was chosen on real captures.
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Ignore the cached result and recompute. Only meaningful after a reprocess.
    refresh: bool = False


class EmptyRoomResponse(_Base):
    """The staged panorama, delivered as its own pyramid rather than as one image.

    The player streams a scene through a resolution ladder, so an emptied room has to arrive
    as the same shape of thing or the swap would either pop from blurred to sharp or hand a
    phone a texture it cannot allocate. ``levels.uhd_url`` is always null: see
    ``ai/empty_room._STAGED_MAX_WIDTH``.
    """

    scene_id: str
    levels: PanoramaLevels
    mask_url: str | None = None
    regions: list[RemovedObject] = Field(default_factory=list)
    #: Solid-angle weighted fraction of the sphere that was replaced. The honest headline
    #: number: this is classical CV, and a furnishing the colour of its wall is not found.
    removed_fraction: float
    strength: float
    elapsed_ms: int
    #: True when the answer came off disk rather than out of the algorithm.
    cached: bool
    model_backed: bool


class MeasurePoint(_Base):
    yaw: float
    pitch: float


class MeasureReference(_Base):
    yaw_a: float
    pitch_a: float
    yaw_b: float
    pitch_b: float
    length_cm: float = Field(gt=0.0)


class MeasureRequest(_Base):
    scene_id: str
    points: list[MeasurePoint] = Field(min_length=2)
    reference: MeasureReference | None = None
    camera_height_cm: float | None = Field(default=None, gt=10.0, lt=1000.0)


class MeasureSegment(_Base):
    from_: int = Field(alias="from")
    to: int
    length_cm: float | None = None
    length_relative: float
    # Additive to types.ts: explains *why* a segment has no metric length (a point at or
    # above the horizon never meets the floor plane). Optional, so TS consumers are unaffected.
    hint: str | None = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid", serialize_by_alias=True)


class BBoxCm(_Base):
    width: float
    depth: float
    height: float


class MeasureResponse(_Base):
    scene_id: str
    segments: list[MeasureSegment]
    total_cm: float | None = None
    bbox_cm: BBoxCm | None = None
    area_m2: float | None = None
    scale_source: Literal["reference", "camera_height", "relative_only"]
    confidence: float
    model_backed: bool


# --------------------------------------------------------------------------------------
# Virtual staging (restyle)
# --------------------------------------------------------------------------------------


class RestyleStyle(_Base):
    """One entry of the style catalogue, as the UI needs to render a rail of them."""

    id: str
    label_tr: str
    label_en: str


class RestyleStylesResponse(_Base):
    """The catalogue, plus whether this deployment can actually run it.

    ``available`` is reported WITHOUT loading a model, so the page that shows the rail can ask
    on mount. It answers "the weights are on this machine and there is a card", not "a pass
    will succeed" — the honest version of that only exists after a pass.
    """

    styles: list[RestyleStyle]
    available: bool


class RestylePhoto(_Base):
    """An uploaded photograph, addressed by the hash of its own bytes.

    THERE IS NO DATABASE ROW BEHIND THIS AND THAT IS DELIBERATE. The id is the content hash,
    so the same photograph uploaded twice is the same id and lands on the same cached results —
    which is exactly right for a tool people use by dragging the same listing photo in again
    after closing the tab. A row would add a lifecycle (who owns it, when does it expire) to a
    feature that does not have one yet, and rows outlive the files they point at.
    """

    photo_id: str
    original_url: str
    width: int
    height: int
    #: Which styles are already rendered for this photograph.
    ready_styles: list[str] = Field(default_factory=list)


class RestyleRequest(_Base):
    photo_id: str
    style: str
    #: Protect curtains and blinds as well as the window itself. ON by default, as on a room's
    #: own photos: a curtain is where a window is, and the first real restyle drew a wardrobe
    #: across one. See ``ai.restyle``.
    window_dressing: bool = True
    #: Ignore the cached render and recompute.
    refresh: bool = False
    #: Which take of this style: 0 is the deterministic default, and each other number is a
    #: different, equally deterministic seed — what "show me another one" asks for.
    variation: int = Field(default=0, ge=0, le=99)


class RestyleResponse(_Base):
    photo_id: str
    style: str
    image_url: str
    original_url: str
    #: Share of the frame that was composited back from the original — windows, doors,
    #: fittings and the view outside. The number a user should be told, because it is the
    #: measure of how much of the picture is still the property.
    protected_fraction: float
    elapsed_ms: int
    cached: bool
    model_backed: bool
    variation: int = 0


class ScenePhotoVariant(_Base):
    """One derived version of a room's photograph: a decorating style, or the bare room.

    ONE TYPE FOR BOTH, and the two nullable numbers are why it is worth stating. A caller
    holding a photograph wants a list of the versions it has, not two lists it has to merge —
    the UI shows them in one rail. ``kind`` says which question produced it, and exactly one of
    ``protected_fraction`` (restyle: how much of the frame is still the property) and
    ``removed_fraction`` (empty: how much of it is now a reconstruction) is meaningful.
    """

    photo_id: str
    kind: Literal["style", "empty"]
    #: The style id, or ``"empty"``.
    variant: str
    url: str
    original_url: str
    #: Restyle only. Share composited back from the original — windows, doors, fittings.
    protected_fraction: float | None = None
    #: Empty only. Share of the frame replaced by the inpainter.
    removed_fraction: float | None = None
    #: What the segmenter named in the room. The same list both features work from.
    objects: list[str] = Field(default_factory=list)
    elapsed_ms: int
    cached: bool
    model_backed: bool
    #: Restyle only: which take of the style this is (see ``RestyleRequest.variation``).
    variation: int = 0


class ScenePhotoRestyleRequest(_Base):
    style: str
    #: Let curtains and blinds be restyled. Held by default; see ``ai.restyle``.
    window_dressing: bool = True
    refresh: bool = False
    variation: int = Field(default=0, ge=0, le=99)


class ScenePhotoEmptyRequest(_Base):
    window_dressing: bool = True
    refresh: bool = False


class RestyleSceneRequest(_Base):
    scene_id: str
    style: str
    #: Held by default, as everywhere else: the player sends no value, and with this off the
    #: panorama's curtains were the one shell surface every style repainted.
    window_dressing: bool = True
    refresh: bool = False


class RestyleSceneResponse(_Base):
    """A restyled panorama, delivered as its own ladder for the same reason ``EmptyRoomResponse``
    is: the player streams a scene through resolution rungs and a styled room has to arrive as
    the same shape of thing or the swap pops."""

    scene_id: str
    style: str
    levels: PanoramaLevels
    protected_fraction: float
    elapsed_ms: int
    cached: bool
    model_backed: bool


# --------------------------------------------------------------------------------------
# Floor plan
# --------------------------------------------------------------------------------------


class PlanPlacement(_Base):
    """Where somebody put a room, in the plan's own metres and radians."""

    x: float
    z: float
    rotation: float


class PlanRoom(_Base):
    """One room on the plan, in metres, already placed in the plan's own frame.

    ``polygon`` is empty for a room with no measured outline; the client draws a disc of
    ``radius`` instead, and ``placement`` is what tells it which of the two it is looking at.
    Sending a shape and a flag saying not to trust it would be the worse contract: half the
    clients would draw the shape.
    """

    scene_id: str
    name: str
    #: ``anchor`` | ``solved`` | ``estimated`` | ``placed`` | ``unlinked``. See
    #: ``biosphere.plan.types.PlacementKind``.
    placement: str
    x: float
    z: float
    rotation: float
    polygon: list[list[float]] = Field(default_factory=list)
    radius: float
    area_m2: float
    #: True when the room has a finished panorama, i.e. somebody can stand in it.
    ready: bool
    #: Share of directions where a wall was actually found rather than interpolated. The
    #: honest headline for one room, the way ``removed_fraction`` is for an emptied one.
    measured_fraction: float


class PlanDoor(_Base):
    """A doorway, at the floor point its ring stands on."""

    from_scene_id: str
    to_scene_id: str
    x: float
    z: float
    #: True when both rooms marked this doorway, which is what fixes the far room's rotation.
    solved: bool


class FloorPlan(_Base):
    """A property seen from above.

    Empty of rooms is a legitimate answer and not an error: a property whose rooms have never
    been measured has no plan yet, and ``pending_scene_ids`` says exactly which rooms a build
    would work on.
    """

    property_id: str
    rooms: list[PlanRoom] = Field(default_factory=list)
    doors: list[PlanDoor] = Field(default_factory=list)
    #: The height the whole drawing is scaled by. A panorama cannot measure it; see
    #: ``biosphere.plan.types.CAMERA_HEIGHT_M``.
    camera_height_m: float
    #: Rooms with a panorama and no outline yet — the work ``POST .../plan/build`` would do.
    pending_scene_ids: list[str] = Field(default_factory=list)
    #: ``(min_x, min_z, max_x, max_z)`` of everything drawn, so the client can fit the view
    #: without walking the polygons.
    bounds: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    warnings: list[str] = Field(default_factory=list)
    #: False when no segmenter is available here, so a build would measure nothing.
    can_measure: bool = True


class BuildPlanRequest(_Base):
    #: Re-measure rooms that already have an outline. Only useful after a reprocess or a
    #: change to the estimator.
    refresh: bool = False


class UpdatePlanRequest(_Base):
    """Move rooms by hand.

    A scene mapped to ``null`` gives that room back to the solver, which is a different thing
    from leaving it out: leaving it out changes nothing about it. That distinction is the
    whole reason this is a mapping rather than a list — "undo my correction" has to be
    expressible, and it is the button an agent reaches for after dragging something wrong.
    """

    placements: dict[str, PlanPlacement | None] = Field(default_factory=dict)
    #: Give the whole property back to the solver. Applied before ``placements``, so a request
    #: can clear everything and then pin one room in the same call.
    reset: bool = False



# --------------------------------------------------------------------------------------
# Listing copy
# --------------------------------------------------------------------------------------

#: The four voices. See ``biosphere.ai.listing.TONES`` for what each one asks the writer for.
ListingTone = Literal["professional", "warm", "concise", "premium"]


class ListingRoomFact(_Base):
    """One room, as the copy writer was told about it."""

    scene_id: str
    name: str
    ready: bool
    has_panorama: bool
    photo_count: int
    has_walkthrough: bool
    #: Measured floor area in m², or null when the room has never been measured.
    area_m2: float | None = None
    #: Share of directions where a wall was found rather than interpolated. Below ~0.6 the
    #: area is a shape rather than a survey, and the prompt says so.
    measured_fraction: float = 0.0


class ListingFacts(_Base):
    """What the copy was written FROM.

    Returned beside the text on purpose. The paragraph is the product, but the useful line
    for an agent about to publish is which numbers went into it — and, when there are none,
    that measuring the rooms first would produce a better one.
    """

    address: str | None = None
    room_count: int
    measured_room_count: int
    #: Sum of the measured rooms only. NOT the flat's floor area: unmeasured rooms and every
    #: hallway are missing from it, which is why the copy is told never to present it as one.
    total_area_m2: float | None = None
    ceiling_height_m: float | None = None
    #: The assumption under every square metre above.
    camera_height_m: float
    has_walkthrough: bool
    rooms: list[ListingRoomFact] = Field(default_factory=list)
    #: Fingerprint of these facts. The cached copy is the answer to exactly this digest; a
    #: new room, a rename or a fresh measurement produces a different one.
    digest: str


class ListingUsage(_Base):
    """What one generation cost. Null on the offline path, which costs nothing.

    ``cost_usd`` is null for a model with no published rate in ``biosphere.ai.llm.PRICES``
    rather than guessed from a neighbouring model — a wrong price shown to somebody deciding
    whether to run this on every listing is worse than no price.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class ListingCopy(_Base):
    """A headline and a description for one property, plus everything needed to judge them."""

    title: str
    description: str
    highlights: list[str] = Field(default_factory=list)
    alt_titles: list[str] = Field(default_factory=list)
    locale: Literal["tr", "en"] = "tr"
    tone: ListingTone = "professional"
    #: False when no model wrote this — the deterministic composer did. Not an error: see
    #: ``biosphere.ai.listing``. ``warnings`` says which of the three reasons it was.
    model_backed: bool = False
    model: str | None = None
    #: How many room pictures the writer was shown. Zero on the offline path.
    images_used: int = 0
    elapsed_ms: int = 0
    cached: bool = False
    #: Tokens and money for the call that produced this. Preserved in the cache, so a copy
    #: served from disk still shows what it cost the day it was written — with ``cached``
    #: saying that today's press did not pay it again.
    usage: ListingUsage | None = None
    facts: ListingFacts
    #: Claims in the text that the facts and the pictures do not carry, plus why a model did
    #: not answer. Shown next to the copy rather than acted on: the text is never edited on
    #: the user's behalf. See ``biosphere.ai.listing.audit``.
    warnings: list[str] = Field(default_factory=list)


class ListingState(_Base):
    """What the panel shows before anybody presses anything."""

    #: Whether a model would write it here. False means a key or the client library is
    #: missing, and the button still works — it just produces the deterministic copy.
    available: bool
    model: str | None = None
    #: ``no_key`` | ``no_sdk`` | null.
    reason: str | None = None
    #: The last generation for this property, when the facts have not changed since. Null when
    #: nothing was generated yet, or when the property has moved on since it was written.
    #:
    #: Named ``latest`` rather than ``copy`` because ``copy`` is a method on every pydantic
    #: model, and a field that shadows it works until the day something calls it.
    latest: ListingCopy | None = None


class GenerateListingRequest(_Base):
    locale: Literal["tr", "en"] = "tr"
    tone: ListingTone = "professional"
    #: A free instruction from the agent ("mention the sea view", "keep it under 60 words").
    #: Additive to the tone, and outranked by the rules that keep the copy truthful.
    instructions: str | None = Field(default=None, max_length=500)
    #: Write it again even though the facts have not changed. The button an agent presses
    #: when they want a different sentence, which is a real reason and costs a call.
    refresh: bool = False


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------

T = TypeVar("T")


class Paginated(_Base, Generic[T]):  # noqa: UP046 - PEP 695 syntax needs 3.12; we support 3.11
    items: list[T]
    total: int
    limit: int
    offset: int


class HealthResponse(_Base):
    status: Literal["ok"] = "ok"
    version: str
    ffmpeg: bool
    opencv: str
    gpu: str | None = None
    media_root_free_gb: float
    #: Jobs that are running but have not reported progress recently. With one worker a
    #: stuck job blocks every capture behind it, so this is the number to watch: anything
    #: other than an empty list means the queue is not moving and nothing else will run.
    stalled_jobs: list[str] = Field(default_factory=list)
    #: Where the room-emptying models run: ``{"mode": "worker"|"local"|"off", ...}``.
    #: Reported without loading anything, so polling this stays free.
    ai: dict[str, Any] = Field(default_factory=dict)
    #: Whether the 3D walkthrough can be built here: ``{"available", "trainer", "sog_writer", "sfm"}``.
    walkthrough: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Row -> schema converters
# --------------------------------------------------------------------------------------


def hotspot_to_schema(row: m.Hotspot) -> Hotspot:
    return Hotspot(
        id=row.id,
        scene_id=row.scene_id,
        type=row.type,  # type: ignore[arg-type]
        yaw=row.yaw,
        pitch=row.pitch,
        title=row.title,
        body=row.body,
        icon=row.icon,
        target_scene_id=row.target_scene_id,
        media_url=row.media_url,
        accent=row.accent,
        created_at=m.ensure_utc(row.created_at),  # type: ignore[arg-type]
    )


def scene_photo_to_schema(row: m.ScenePhoto, url: str) -> ScenePhoto:
    return ScenePhoto(
        id=row.id,
        scene_id=row.scene_id,
        url=url,
        width=row.width,
        height=row.height,
        caption=row.caption,
        order=row.order,
        created_at=_iso_z(m.ensure_utc(row.created_at)),  # type: ignore[arg-type]
    )


def scene_to_schema(
    row: m.Scene,
    hotspots: list[m.Hotspot] | None = None,
    photos: list[ScenePhoto] | None = None,
) -> Scene:
    return Scene(
        id=row.id,
        property_id=row.property_id,
        name=row.name,
        status=row.status,  # type: ignore[arg-type]
        capture_kind=row.capture_kind,  # type: ignore[arg-type]
        quality_mode=row.quality_mode,  # type: ignore[arg-type]
        order=row.order,
        panorama=PanoramaLevels(**row.panorama) if row.panorama else None,
        coverage=Coverage(**row.coverage) if row.coverage else None,
        metrics=StitchMetrics(**row.metrics) if row.metrics else None,
        initial_view=InitialView(**(row.initial_view or m.DEFAULT_INITIAL_VIEW)),
        hotspots=[hotspot_to_schema(h) for h in (hotspots or [])],
        photos=list(photos or []),
        walkthrough=SceneWalk(**row.walkthrough) if row.walkthrough else None,
        active_job_id=row.active_job_id,
        error=ApiError(**row.error) if row.error else None,
        created_at=m.ensure_utc(row.created_at),  # type: ignore[arg-type]
    )


def walkthrough_to_schema(row: m.Walkthrough) -> Walkthrough:
    return Walkthrough(
        id=row.id,
        property_id=row.property_id,
        status=row.status,  # type: ignore[arg-type]
        quality_mode=row.quality_mode,  # type: ignore[arg-type]
        room_scene_ids=list(row.room_scene_ids or []),
        manifest=WalkManifest(**row.manifest) if row.manifest else None,
        metrics=row.metrics,
        error=ApiError(**row.error) if row.error else None,
        active_job_id=row.active_job_id,
        created_at=m.ensure_utc(row.created_at),  # type: ignore[arg-type]
        updated_at=m.ensure_utc(row.updated_at),  # type: ignore[arg-type]
    )


def _cover_url(scenes: list[m.Scene], cover_scene_id: str | None) -> str | None:
    """Prefer the explicitly chosen cover scene, else the first scene that finished."""
    ready = [s for s in scenes if s.panorama]
    chosen = next((s for s in ready if s.id == cover_scene_id), None)
    if chosen is None:
        chosen = next(iter(sorted(ready, key=lambda s: (s.order, s.created_at))), None)
    if chosen is None or not chosen.panorama:
        # A property that only has a 3D walkthrough shows a room's splat thumbnail instead.
        walk = next((s for s in scenes if (s.walkthrough or {}).get("thumbnail_url")), None)
        return walk.walkthrough.get("thumbnail_url") if walk else None
    pano = chosen.panorama
    return pano.get("sd_url") or pano.get("preview_url")


def property_to_schema(row: m.Property, scenes: list[m.Scene]) -> Property:
    return Property(
        id=row.id,
        title=row.title,
        description=row.description,
        address=row.address,
        public_token=row.public_token,
        published=row.published,
        cover_url=_cover_url(scenes, row.cover_scene_id),
        scene_count=len(scenes),
        views=row.views,
        created_at=m.ensure_utc(row.created_at),  # type: ignore[arg-type]
        updated_at=m.ensure_utc(row.updated_at),  # type: ignore[arg-type]
    )


def property_detail_to_schema(
    row: m.Property,
    scenes: list[m.Scene],
    hotspots_by_scene: dict[str, list[m.Hotspot]],
    photos_by_scene: dict[str, list[ScenePhoto]] | None = None,
    walkthrough: m.Walkthrough | None = None,
) -> PropertyDetail:
    base = property_to_schema(row, scenes)
    return PropertyDetail(
        **base.model_dump(),
        scenes=[
            scene_to_schema(
                s, hotspots_by_scene.get(s.id, []), (photos_by_scene or {}).get(s.id, [])
            )
            for s in scenes
        ],
        walkthrough=walkthrough_to_schema(walkthrough) if walkthrough is not None else None,
    )


def job_to_schema(row: m.Job) -> Job:
    return Job(
        id=row.id,
        property_id=row.property_id,
        scene_id=row.scene_id,
        kind=(row.kind or "stitch"),  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        stage=row.stage,  # type: ignore[arg-type]
        progress=row.progress,
        stage_progress=row.stage_progress,
        message=row.message,
        message_tr=row.message_tr,
        eta_seconds=row.eta_seconds,
        metrics=row.metrics,
        error=ApiError(**row.error) if row.error else None,
        created_at=m.ensure_utc(row.created_at),  # type: ignore[arg-type]
        started_at=m.ensure_utc(row.started_at),
        finished_at=m.ensure_utc(row.finished_at),
    )


class ScenePartFile(BaseModel):
    """One assembled file and the staged parts, in order, that make it up."""

    filename: str = Field(min_length=1, max_length=200)
    parts: list[int] = Field(min_length=1, max_length=512)


class CreateSceneFromPartsRequest(BaseModel):
    """Finalise a capture staged with `PUT /properties/uploads/{id}/parts/{index}`.

    The single multipart POST remains the path for ordinary captures. This exists because a
    browser has to build that whole body before it can send it, and on iOS a 270 MB capture
    never left the phone at all — no request, no error, an upload frozen at 0%. Parts are
    `Blob.slice` views, so the browser never holds more than one chunk at a time.
    """

    upload_id: str = Field(min_length=6, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=200)
    quality_mode: str = ""
    capture_kind: str = "video"
    files: list[ScenePartFile] = Field(min_length=1, max_length=32)


# --------------------------------------------------------------------------------------
# Investment assistant
# --------------------------------------------------------------------------------------
#
# The agent's figures are ``biosphere.invest.inputs.InvestmentInputs`` — defined beside the
# calculators that read them and re-exported here, so the API contract and the engine cannot
# drift apart. Everything below is what the player and the dashboard are sent.

from biosphere.invest.inputs import InvestmentInputs  # noqa: E402 - re-exported contract


class AdvisorSourceRef(_Base):
    title: str
    url: str
    #: Publication or reading date (``2026-09-17`` or ``2026-09``).
    date: str


class AdvisorCardRent(_Base):
    monthly_tl: int
    status: Literal["expected", "actual"]


class AdvisorCardAssumptions(_Base):
    """What the NET figures assume, so the card can say it."""

    vacancy_months: float
    maintenance_pct_of_rent: float
    emlak_vergisi_tl: int
    #: ``ilan`` (the agent entered it) or ``tahmin`` (an upper-bound estimate).
    emlak_vergisi_basis: str
    dask_tl: int
    #: ``ilan`` or ``varsayılan``.
    dask_basis: str
    aidat_monthly_tl: int | None = None


class AdvisorCardIncome(_Base):
    gross_pct: float
    net_pct: float
    net_annual_income_tl: int
    gross_payback_years: float
    net_payback_years: float | None = None
    assumptions: AdvisorCardAssumptions


class AdvisorCardPurchase(_Base):
    total_costs_tl: int
    total_outlay_tl: int
    costs_pct_of_price: float
    commission_pct: float
    commission_basis: str


class AdvisorCardPerM2(_Base):
    tl: int
    basis: Literal["net", "brut"]
    area_m2: float


class AdvisorCardMortgage(_Base):
    monthly_payment_tl: int
    loan_tl: int
    down_payment_pct: float
    down_payment_tl: int
    term_months: int
    monthly_rate_pct: float
    #: BDDK's loan-to-value ceiling for the price band (A/B energy class; lower classes less).
    ltv_cap_pct: float
    rate_source: AdvisorSourceRef | None = None
    rent_covers_payment_pct: float | None = None


class AdvisorCard(_Base):
    """The deterministic figures card — computed by the engine, no model involved.

    Values are raw numbers; the client formats them for its locale. ``missing`` names what the
    agent did not enter (``rent``, ``area``) so the card can say why a tile is empty.
    """

    price_tl: int
    location: str | None = None
    missing: list[str] = Field(default_factory=list)
    purchase: AdvisorCardPurchase
    rent: AdvisorCardRent | None = None
    income: AdvisorCardIncome | None = None
    per_m2: AdvisorCardPerM2 | None = None
    mortgage: AdvisorCardMortgage


class AdvisorPublicState(_Base):
    """What the player asks when it mounts: whether to show the segment, and the card.

    ``enabled: false`` carries nothing else — a property whose agent has not opted in says no
    more about itself than that, its figures included.
    """

    enabled: bool
    chat_available: bool = False
    #: Why the chat is off while the card is on: ``disabled`` | ``no_key`` | ``no_sdk`` |
    #: ``budget`` | null.
    reason: str | None = None
    card: AdvisorCard | None = None
    suggestions: list[str] = Field(default_factory=list)
    max_message_chars: int = 600
    max_turns: int = 12


class AdvisorChatRequest(_Base):
    """One visitor question. Sent as ``text/plain`` JSON — see ``routes/advisor.py``."""

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=64)
    locale: Literal["tr", "en"] = "tr"


class AdvisorStatus(_Base):
    available: bool
    #: ``disabled`` | ``no_key`` | ``no_sdk`` | null.
    reason: str | None = None
    model: str
    web_search: bool


class AdvisorBudget(_Base):
    #: The cap in force for this property today (the agent's, bounded by the operator's max).
    daily_budget_usd: float
    #: The operator's default, which applies when the agent sets no cap of their own.
    default_usd: float = 2.0
    operator_max_usd: float
    spent_today_usd: float
    global_daily_budget_usd: float
    global_spent_today_usd: float


class AdvisorUsageModel(_Base):
    model: str
    calls: int
    cost_usd: float


class AdvisorUsageDay(_Base):
    date: str
    cost_usd: float
    questions: int


class AdvisorUsageSummary(_Base):
    """The property's assistant bill over the last 30 Istanbul days. No conversation content."""

    today_usd: float = 0.0
    last_7_days_usd: float = 0.0
    last_30_days_usd: float = 0.0
    all_time_usd: float = 0.0
    conversations_30d: int = 0
    questions_30d: int = 0
    calls_30d: int = 0
    web_searches_30d: int = 0
    avg_cost_per_question_usd: float | None = None
    avg_call_latency_ms: int | None = None
    by_model: list[AdvisorUsageModel] = Field(default_factory=list)
    daily: list[AdvisorUsageDay] = Field(default_factory=list)


class AdvisorRateRow(_Base):
    """One Turkish cost parameter the calculators use, shown to the agent with its source."""

    key: str
    label: str
    value: str
    source: AdvisorSourceRef | None = None


class InvestmentState(_Base):
    """The dashboard panel: the agent's figures, the card they produce, whether visitors see
    the chat (and what is stopping them), what it has cost, and the rates it assumes."""

    inputs: InvestmentInputs
    card: AdvisorCard | None = None
    status: AdvisorStatus
    visible_to_visitors: bool
    #: What keeps the chat from visitors: ``not_enabled`` | ``no_price`` | ``not_published`` |
    #: ``disabled`` | ``no_key`` | ``no_sdk`` | ``budget``. Empty when they see it.
    blockers: list[str] = Field(default_factory=list)
    usage: AdvisorUsageSummary
    budget: AdvisorBudget
    rates: list[AdvisorRateRow] = Field(default_factory=list)
    rates_as_of: str
    #: Rate keys whose source is more than a year old.
    stale_rates: list[str] = Field(default_factory=list)
