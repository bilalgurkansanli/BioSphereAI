/**
 * BioSphere AI — shared API contract.
 *
 * This file is the single source of truth for the shape of everything that crosses the
 * FastAPI <-> Next.js boundary. The Python side mirrors it in
 * `engine/biosphere/api/schemas.py`; if you change one, change both.
 *
 * Angle convention (identical to the engine, see engine/biosphere/pano/types.py):
 *   yaw   — radians in (-pi, pi],  0 = panorama centre column, positive turns right
 *   pitch — radians in [-pi/2, pi/2], positive looks up
 */

export type QualityMode = "fast" | "balanced" | "studio";
/**
 * `none` is a room created without a capture — no job, no panorama, ready on creation.
 * `walk` is a room created by the 3D walkthrough upload: a walkthrough video and, once built,
 * a 3D area; no panorama unless one is added later.
 */
export type CaptureKind = "video" | "photos" | "none" | "walk";
/** `stitch` builds one scene's panorama; `walkthrough` builds the property's 3D areas. */
export type JobKind = "stitch" | "walkthrough";
export type SweepPattern = "single_row" | "multi_row" | "unknown";

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

/** Ordered pipeline stages. The UI renders these as a progress rail. */
export const JOB_STAGES = [
  "upload",
  "probe",
  "extract",
  "select",
  "features",
  "match",
  "bundle",
  "exposure",
  "warp",
  "seam",
  "blend",
  "refine",
  "poles",
  "levels",
  "done",
] as const;
/** Ordered stages of a walkthrough (3D) job; a different rail from the stitch. */
export const WALK_STAGES = [
  "upload",
  "extract",
  "features",
  "sfm",
  "register",
  "train",
  "clean",
  "export",
  "done",
] as const;
export type WalkStage = (typeof WALK_STAGES)[number];
export type JobStage = (typeof JOB_STAGES)[number] | WalkStage;

export type SceneStatus = "pending" | "processing" | "ready" | "failed";

// ---------------------------------------------------------------------------------
// Core resources
// ---------------------------------------------------------------------------------

export interface Property {
  id: string;
  title: string;
  description: string | null;
  address: string | null;
  public_token: string;
  published: boolean;
  cover_url: string | null;
  scene_count: number;
  views: number;
  created_at: string;
  updated_at: string;
}

export interface PanoramaLevels {
  /** Tiny blurred equirect shown instantly while the big one streams in. */
  preview_url: string;
  sd_url: string | null;
  hd_url: string | null;
  uhd_url: string | null;
  width: number;
  height: number;
  /** Cube faces, present only when the scene was processed in studio mode. */
  cube_urls: Record<"px" | "nx" | "py" | "ny" | "pz" | "nz", string> | null;
}

export interface CoverageGap {
  yaw_start_deg: number;
  yaw_end_deg: number;
  pitch_start_deg: number;
  pitch_end_deg: number;
}

export interface Coverage {
  covered_fraction: number;
  horizon_fraction: number;
  yaw_span_deg: number;
  pitch_min_deg: number;
  pitch_max_deg: number;
  zenith_covered: boolean;
  nadir_covered: boolean;
  gaps: CoverageGap[];
  /** Equirect heat-map PNG highlighting the areas the user should re-shoot. */
  map_url: string | null;
}

export interface StitchMetrics {
  n_frames_extracted: number;
  n_keyframes: number;
  n_pairs: number;
  n_loop_closures: number;
  mean_reprojection_error_px: number;
  max_reprojection_error_px: number;
  /**
   * Inlier-weighted median over the pairs that actually span a quarter turn or more.
   * A plain median across every loop-flagged pair is robust but not monotone in real
   * drift, because most flagged pairs sit only a few keyframes apart and see a fraction
   * of a loop's worth of error. Above 2.0 the panorama shows misaligned detail where the
   * sweep came back round.
   */
  loop_closure_residual_deg: number;
  /** Worst single loop pair — diagnostic only; one weak match can spike it harmlessly. */
  loop_closure_max_deg: number;
  /** How many pairs genuinely span the loop. Often just one. */
  loop_closure_span_pairs: number;
  /**
   * Parallax over every verified match after the solve: the median, the 90th percentile
   * and the fraction beyond 6 px. A hand-held sweep that pivoted about the body rather
   * than the lens reads tens of pixels at p90; the renderer's mesh alignment absorbs
   * most of it, and what remains ghosts where near objects overlap far ones.
   */
  parallax_median_px?: number;
  parallax_p90_px?: number;
  parallax_fraction_over_6px?: number;
  focal_px: number;
  hfov_deg: number;
  seam_gradient_energy: number;
  exposure_spread: number;
  sweep_pattern: SweepPattern;
  stage_seconds: Record<string, number>;
  warnings: string[];
  /** Which phone shot this, and what its lens implies. Null for an upload with no capture
   *  sidecar — a video from somewhere else, or one made before the app recorded this. */
  device?: DeviceInfo | null;
}

/**
 * The capture device's camera, as the engine identified it.
 *
 * A browser cannot read a camera's field of view, so this is a lookup: the capture app
 * writes down which phone it is (the model string on Android, the screen fingerprint on
 * iOS) and the engine looks the lens up in its own table. `hfov_deg` is therefore what the
 * HARDWARE says, which is a different claim from `StitchMetrics.hfov_deg`, what the solve
 * measured from the imagery. The two disagreeing is the diagnosis, not an error.
 */
export interface DeviceInfo {
  id: string;
  brand: string;
  model: string;
  tier: "flagship" | "upper-mid" | "mid" | "budget";
  /** How the phone was named: an exact model string, a screen fingerprint, its brand, or
   *  nothing at all. Anything below "ua-model" or "ios-fingerprint" is a family, not a phone. */
  source: "ua-model" | "ios-fingerprint" | "brand" | "generic" | "none";
  lens: "main" | "ultrawide" | "unknown";
  focal_px: number | null;
  hfov_deg: number | null;
  /** Relative one-sigma uncertainty of the lens figure; the solve is fenced at two of them. */
  sigma_fraction: number;
  ois: boolean;
  /** True only when EVERY model that could produce this fingerprint has the scanner. The
   *  scanner is unreadable from a browser either way; see the measure tool. */
  lidar: boolean;
  /** Models that share the fingerprint, when the answer is a family. */
  candidates: string[];
  notes: string[];
}

export interface InitialView {
  yaw: number;
  pitch: number;
  /** Vertical field of view in degrees. */
  fov: number;
}

/**
 * An ordinary photograph attached to a room.
 *
 * Independent of the panorama: a room may have both, either or neither. `url` is a media path
 * like every other asset, so `absoluteUrl` resolves it the same way.
 */
export interface ScenePhoto {
  id: string;
  scene_id: string;
  url: string;
  width: number;
  height: number;
  caption: string | null;
  order: number;
  created_at: string;
}

/**
 * One derived version of a room's photograph: a decorating style, or the bare room.
 *
 * One type for both, because the UI shows them in one rail. `kind` says which question
 * produced it, and exactly one of `protected_fraction` (restyle: how much of the frame is
 * still the property) and `removed_fraction` (empty: how much is now a reconstruction) is
 * meaningful.
 */
export interface ScenePhotoVariant {
  photo_id: string;
  kind: "style" | "empty";
  /** A `RestyleStyleId`, or `"empty"`. */
  variant: string;
  url: string;
  original_url: string;
  protected_fraction: number | null;
  removed_fraction: number | null;
  /** What the segmenter named in the room. Both tools work from this same list. */
  objects: string[];
  elapsed_ms: number;
  cached: boolean;
  model_backed: boolean;
  /** Restyle only: which take of the style this is; 0 is the default one. */
  variation?: number;
}

export interface ScenePhotoRestyleRequest {
  style: RestyleStyleId;
  window_dressing?: boolean;
  refresh?: boolean;
  /** Another take of the style: each number is its own, repeatable, result. */
  variation?: number;
}

export interface ScenePhotoEmptyRequest {
  window_dressing?: boolean;
  refresh?: boolean;
}

export interface CreateRoomRequest {
  name: string;
}

export interface UpdateScenePhotoRequest {
  caption?: string | null;
  order?: number;
}

/**
 * `portal` is a `link` that knows it is standing on the floor.
 *
 * Both navigate and both need a `target_scene_id`; the difference is entirely in how the
 * player draws them. A link is a tag on a wall — screen space, same size wherever it is. A
 * portal is a ring lying ON the floor where its ray meets it, foreshortened and shrinking with
 * distance, which is what makes "you can stand there" something the picture says rather than
 * something a label claims.
 */
export type HotspotType = "info" | "link" | "media" | "portal";

/** The types that navigate, and therefore need a target that exists. */
export const NAVIGATION_TYPES = ["link", "portal"] as const;

export interface Hotspot {
  id: string;
  scene_id: string;
  type: HotspotType;
  yaw: number;
  pitch: number;
  title: string;
  body: string | null;
  icon: string | null;
  /** Set when the type navigates; the scene it goes to. */
  target_scene_id: string | null;
  media_url: string | null;
  accent: string | null;
  created_at: string;
}

export interface Scene {
  id: string;
  property_id: string;
  name: string;
  status: SceneStatus;
  capture_kind: CaptureKind;
  quality_mode: QualityMode;
  order: number;
  panorama: PanoramaLevels | null;
  coverage: Coverage | null;
  metrics: StitchMetrics | null;
  initial_view: InitialView;
  hotspots: Hotspot[];
  /** The room's ordinary photographs, in display order. */
  photos: ScenePhoto[];
  /** The room's 3D area from the property's walkthrough, or null when the room has none. */
  walkthrough: SceneWalk | null;
  active_job_id: string | null;
  error: ApiError | null;
  created_at: string;
}

/** A room's slice of the property's 3D walkthrough: enough to show a tile and a link. */
export interface SceneWalk {
  status: "pending" | "processing" | "ready" | "failed";
  url: string | null;
  thumbnail_url: string | null;
  splat_count: number;
  bytes: number;
  /** False when the room could not be aligned to the general walkthrough. */
  placed: boolean;
  error: string | null;
}

// ---------------------------------------------------------------------------------
// 3D walkthrough
// ---------------------------------------------------------------------------------

/**
 * One streamable area of the property: the general walkthrough or a room.
 *
 * Every number is in the PROPERTY FRAME: right-handed, +X right, +Y DOWN, +Z forward, floor at
 * y = 0, metres. A Y-up renderer rotates the whole thing 180 degrees about X, i.e.
 * (x, y, z) -> (x, -y, -z).
 */
export interface WalkArea {
  key: string;
  name: string;
  floor_index: number;
  url: string | null;
  bytes: number;
  splat_count: number;
  thumbnail_url: string | null;
  env_pano_url: string | null;
  /** Camera centres along the walk, thinned to at most 64. */
  cameras: number[][];
  centroid: number[];
  entry: { position: number[]; forward: number[] };
  bbox: { min: number[]; max: number[] };
  placed: boolean;
  /**
   * The general walkthrough already covers this area's own viewpoints, so it is a second
   * filming of the same space rather than a new one. Listed and openable by name; never
   * chosen by position, because swapping into it mid-step looks like being teleported.
   */
  redundant?: boolean;
  warnings: string[];
  failed: string | null;
}

export interface WalkFloorPlanHotspot {
  key: string;
  name: string;
  x_pct: number;
  y_pct: number;
  floor_index: number;
}

export interface WalkFloorPlan {
  url: string | null;
  width: number;
  height: number;
  /** World -> pixel: u = (x - x0) * px_per_m, v = (z0 - z) * px_per_m. */
  x0: number;
  z0: number;
  px_per_m: number;
  hotspots: WalkFloorPlanHotspot[];
}

export interface WalkManifest {
  version: number;
  frame: Record<string, unknown>;
  base: WalkArea;
  rooms: WalkArea[];
  floor_plan: WalkFloorPlan | null;
  metrics: Record<string, unknown>;
  warnings: string[];
}

export interface Walkthrough {
  id: string;
  property_id: string;
  status: SceneStatus;
  quality_mode: QualityMode;
  room_scene_ids: string[];
  manifest: WalkManifest | null;
  metrics: Record<string, unknown> | null;
  error: ApiError | null;
  active_job_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface WalkthroughJobResponse {
  walkthrough: Walkthrough;
  job: Job;
}

/** One room of a walkthrough upload: its name and which uploaded files are its video. */
export interface WalkthroughRoomSpec {
  name: string;
  scene_id?: string | null;
  floor_index?: number;
  filenames: string[];
}

/** The `spec` form field of `POST /evler/{id}/walkthrough`; files travel as `files[]`. */
export interface CreateWalkthroughSpec {
  quality_mode: QualityMode | "";
  base_filenames: string[];
  rooms: WalkthroughRoomSpec[];
}

export interface PropertyDetail extends Property {
  scenes: Scene[];
  /** The property's 3D walkthrough, or null when none was uploaded. */
  walkthrough: Walkthrough | null;
}

// ---------------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------------

export interface ApiError {
  code: string;
  message: string;
  /** Turkish user-facing copy; the UI prefers this when locale is `tr`. */
  message_tr: string;
  hint?: string | null;
}

export interface Job {
  id: string;
  property_id: string;
  /** Null for a property-level job (`kind === "walkthrough"`). */
  scene_id: string | null;
  kind: JobKind;
  status: JobStatus;
  stage: JobStage;
  /** Overall completion, 0..1. */
  progress: number;
  /** Completion of the current stage only, 0..1. */
  stage_progress: number;
  message: string;
  message_tr: string;
  eta_seconds: number | null;
  metrics: Partial<StitchMetrics> | null;
  error: ApiError | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

/** Payload of each `message` event on `GET /jobs/{id}/events` (text/event-stream). */
export type JobEvent =
  | { type: "progress"; job: Job }
  | { type: "log"; level: "info" | "warn" | "error"; message: string; at: string }
  | { type: "done"; job: Job; scene: Scene | null; walkthrough?: Walkthrough | null }
  | { type: "error"; job: Job; error: ApiError };

// ---------------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------------

export interface CreatePropertyRequest {
  title: string;
  description?: string;
  address?: string;
}

export interface UpdatePropertyRequest {
  title?: string;
  description?: string;
  address?: string;
  published?: boolean;
  cover_scene_id?: string;
}

export interface CreateSceneRequest {
  name: string;
  quality_mode: QualityMode;
  capture_kind: CaptureKind;
}

export interface UpdateSceneRequest {
  name?: string;
  order?: number;
  initial_view?: InitialView;
}

export interface CreateHotspotRequest {
  type: HotspotType;
  yaw: number;
  pitch: number;
  title: string;
  body?: string;
  icon?: string;
  target_scene_id?: string;
  media_url?: string;
  accent?: string;
}

export type UpdateHotspotRequest = Partial<CreateHotspotRequest>;

// ---------------------------------------------------------------------------------
// AI endpoints
// ---------------------------------------------------------------------------------

export interface HideFurnitureRequest {
  scene_id: string;
  /** Closed polygon in equirect pixel coordinates of the `hd` level. Empty = auto-detect. */
  polygon: Array<[number, number]>;
  strength?: number;
}

export interface HideFurnitureResponse {
  scene_id: string;
  image_url: string;
  mask_url: string;
  removed_objects: Array<{ label: string; confidence: number; area_px: number }>;
  elapsed_ms: number;
  /** True when a real inpainting model ran; false when the deterministic fallback did. */
  model_backed: boolean;
}

export interface EmptyRoomRequest {
  scene_id: string;
  /** How readily a pixel stops counting as part of the room, 0..1. */
  strength?: number;
  /** Ignore the cached result and recompute. */
  refresh?: boolean;
}

export interface EmptyRoomResponse {
  scene_id: string;
  /**
   * The staged panorama as its own ladder. `uhd_url` is always null — the emptied image is
   * capped at 4K because a third of its pixels are synthesised; see the engine's
   * `ai/empty_room._STAGED_MAX_WIDTH`.
   */
  levels: PanoramaLevels;
  mask_url: string | null;
  regions: Array<{ label: string; confidence: number; area_px: number }>;
  /**
   * Solid-angle weighted fraction of the sphere that was replaced. The honest headline:
   * this is classical CV, and a furnishing the colour of the wall behind it is not found.
   */
  removed_fraction: number;
  strength: number;
  elapsed_ms: number;
  /** True when the answer came off disk rather than out of the algorithm. */
  cached: boolean;
  model_backed: boolean;
}

export interface MeasureRequest {
  scene_id: string;
  /** Two or more points on the sphere, in radians. */
  points: Array<{ yaw: number; pitch: number }>;
  /**
   * Metric anchor. A single camera cannot recover absolute scale, so the caller must
   * supply one known length (e.g. a door leaf at 200 cm) or accept relative units.
   */
  reference?: { yaw_a: number; pitch_a: number; yaw_b: number; pitch_b: number; length_cm: number };
  camera_height_cm?: number;
}

export interface MeasureResponse {
  scene_id: string;
  segments: Array<{
    from: number;
    to: number;
    length_cm: number | null;
    length_relative: number;
    /** Why this segment has no metric length: a point at or above the horizon never meets
     *  the floor plane, so there is nothing to intersect. English, from the engine. */
    hint?: string | null;
  }>;
  total_cm: number | null;
  bbox_cm: { width: number; depth: number; height: number } | null;
  area_m2: number | null;
  scale_source: "reference" | "camera_height" | "relative_only";
  confidence: number;
  model_backed: boolean;
}

// ---------------------------------------------------------------------------------
// Listing copy
// ---------------------------------------------------------------------------------

/** The four voices. The engine owns what each one asks the writer for. */
export type ListingTone = "professional" | "warm" | "concise" | "premium";

export interface ListingRoomFact {
  scene_id: string;
  name: string;
  ready: boolean;
  has_panorama: boolean;
  photo_count: number;
  has_walkthrough: boolean;
  /** Measured floor area in m², or null when the room has never been measured. */
  area_m2: number | null;
  /** Share of directions where a wall was found rather than interpolated. */
  measured_fraction: number;
}

/**
 * What the copy was written FROM.
 *
 * Shown beside the text, not hidden behind it: the useful line for an agent about to publish
 * is which numbers went into the paragraph — and, when there are none, that measuring the
 * rooms on the plan first would produce a better one.
 */
export interface ListingFacts {
  address: string | null;
  room_count: number;
  measured_room_count: number;
  /** Sum of the MEASURED rooms only. Not the flat's floor area; hallways are not in it. */
  total_area_m2: number | null;
  ceiling_height_m: number | null;
  camera_height_m: number;
  has_walkthrough: boolean;
  rooms: ListingRoomFact[];
  /** Fingerprint of these facts; the cached copy is the answer to exactly this one. */
  digest: string;
}

export interface ListingCopy {
  title: string;
  description: string;
  highlights: string[];
  alt_titles: string[];
  locale: Locale;
  tone: ListingTone;
  /** False when no model wrote this — the engine's deterministic composer did. Not an error. */
  model_backed: boolean;
  model: string | null;
  /** How many room pictures the writer was shown. Zero on the offline path. */
  images_used: number;
  elapsed_ms: number;
  cached: boolean;
  /**
   * Tokens and money for the call that produced this; null on the offline path, which costs
   * nothing. Kept in the cache, so a copy served from disk still shows what it cost the day
   * it was written — `cached` is what says today's press did not pay it again.
   */
  usage: ListingUsage | null;
  facts: ListingFacts;
  /**
   * Claims in the text the facts and the pictures do not carry (`ungrounded_area:120 m²`,
   * `mentions_price`, `unsupported_claim:metro`), the portal rules it breaks
   * (`contact_phone`, `contact_email`, `contact_link` — sahibinden pulls listings that carry
   * those), plus why no model answered. The text is never edited on the user's behalf, so
   * these are shown next to it.
   */
  warnings: string[];
}

export interface ListingUsage {
  input_tokens: number;
  output_tokens: number;
  /** Null for a model with no published rate in the engine's table, never a guess. */
  cost_usd: number | null;
}

export interface ListingState {
  /** Whether a model would write it. False still generates — deterministically. */
  available: boolean;
  model: string | null;
  /** `no_key` | `no_sdk` | null. */
  reason: string | null;
  /** The last generation, when the property has not changed since it was written. */
  latest: ListingCopy | null;
}

export interface GenerateListingRequest {
  locale?: Locale;
  tone?: ListingTone;
  /** A free instruction from the agent; additive to the tone, outranked by the truth rules. */
  instructions?: string;
  /** Write it again even though nothing changed. */
  refresh?: boolean;
}

// ---------------------------------------------------------------------------------
// Virtual staging (restyle)
// ---------------------------------------------------------------------------------

/** Style ids are the engine's; the UI never invents one. */
export type RestyleStyleId = "modern" | "iskandinav" | "rustik" | "klasik" | "loft";

export interface RestyleStyle {
  id: RestyleStyleId;
  label_tr: string;
  label_en: string;
}

export interface RestyleStylesResponse {
  styles: RestyleStyle[];
  /** Reported without loading a model: the weights and a GPU are present on the server. */
  available: boolean;
}

/**
 * An uploaded photograph, addressed by the hash of its own bytes.
 *
 * There is no row behind this, which is why the same file uploaded twice comes back with the
 * same `photo_id` and whatever styles were already rendered for it.
 */
export interface RestylePhoto {
  photo_id: string;
  original_url: string;
  width: number;
  height: number;
  ready_styles: RestyleStyleId[];
}

export interface RestyleRequest {
  photo_id: string;
  style: RestyleStyleId;
  /** Protect curtains and blinds as well as the window itself. */
  window_dressing?: boolean;
  refresh?: boolean;
  /** Another take of the style: each number is its own, repeatable, result. */
  variation?: number;
}

export interface RestyleResponse {
  photo_id: string;
  style: RestyleStyleId;
  image_url: string;
  original_url: string;
  /**
   * Share of the frame composited back from the original — windows, doors, fittings and the
   * view outside. Worth showing: it is the measure of how much of the picture is still the
   * property rather than a proposal about it.
   */
  protected_fraction: number;
  elapsed_ms: number;
  cached: boolean;
  model_backed: boolean;
  variation?: number;
}

export interface RestyleSceneRequest {
  scene_id: string;
  style: RestyleStyleId;
  window_dressing?: boolean;
  refresh?: boolean;
}

export interface RestyleSceneResponse {
  scene_id: string;
  style: RestyleStyleId;
  levels: PanoramaLevels;
  protected_fraction: number;
  elapsed_ms: number;
  cached: boolean;
  model_backed: boolean;
}

// ---------------------------------------------------------------------------------
// Floor plan
// ---------------------------------------------------------------------------------

/**
 * How a room got where it is on the plan.
 *
 * `anchor`    the room the plan is built around; it defines the frame.
 * `solved`    both rooms marked the doorway between them, so position AND rotation are
 *             geometry rather than guesswork.
 * `estimated` only one side marked it. The room really is through that door on that
 *             bearing; which way it is turned is a guess, and the drawing says so.
 * `placed`    somebody put it there by hand. Outranks the three above and is drawn as a
 *             solid line even when the shape is a placeholder — a person asserting where a
 *             room is is not the software guessing.
 * `unlinked`  no doorway reaches it and nobody has placed it. It is listed, never drawn.
 */
export type PlanPlacement = "anchor" | "solved" | "estimated" | "placed" | "unlinked";

/** Where somebody put a room, in the plan's own metres and radians. */
export interface PlanRoomPlacement {
  x: number;
  z: number;
  rotation: number;
}

export interface PlanRoom {
  scene_id: string;
  name: string;
  placement: PlanPlacement;
  /** Position of the camera that shot this room, in plan metres. */
  x: number;
  z: number;
  /** Rotation from the room's own frame to the plan's, radians. */
  rotation: number;
  /** Already in plan coordinates. Empty when the room has no measured shape. */
  polygon: Array<[number, number]>;
  /** Radius of the placeholder disc, metres, used when `polygon` is empty. */
  radius: number;
  area_m2: number;
  /** True when the room has a finished panorama, i.e. somebody can stand in it. */
  ready: boolean;
  /** Share of directions where a wall was found rather than interpolated. */
  measured_fraction: number;
}

export interface PlanDoor {
  from_scene_id: string;
  to_scene_id: string;
  x: number;
  z: number;
  /** True when both rooms marked it — the case that fixes the far room's rotation. */
  solved: boolean;
}

export interface FloorPlan {
  property_id: string;
  rooms: PlanRoom[];
  doors: PlanDoor[];
  /**
   * The height every metre on this drawing is scaled by. A single panorama cannot measure
   * it; the engine assumes the same 1.6 m the floor rings assume, and the plan says so
   * rather than presenting the scale as measured.
   */
  camera_height_m: number;
  /** Rooms with a panorama and no outline yet — what a build would work on. */
  pending_scene_ids: string[];
  /** `[min_x, min_z, max_x, max_z]` of everything drawn. */
  bounds: [number, number, number, number];
  warnings: string[];
  /** False when the deployment has no segmenter, so building would measure nothing. */
  can_measure: boolean;
}

export interface BuildPlanRequest {
  /** Re-measure rooms that already have an outline. */
  refresh?: boolean;
}

export interface UpdatePlanRequest {
  /**
   * Rooms to move, by scene. `null` gives that room back to the solver, which is a different
   * thing from leaving it out — leaving it out changes nothing about it.
   */
  placements?: Record<string, PlanRoomPlacement | null>;
  /** Give the whole property back to the solver, before applying `placements`. */
  reset?: boolean;
}

// ---------------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------------

export interface Paginated<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface HealthResponse {
  status: "ok";
  version: string;
  ffmpeg: boolean;
  opencv: string;
  gpu: string | null;
  media_root_free_gb: number;
  stalled_jobs?: string[];
  ai?: Record<string, unknown>;
  /** Whether the 3D walkthrough engine (trainer, SOG writer, SfM) is on this server. */
  walkthrough?: { available: boolean; trainer: string | null; sog_writer: boolean; sfm: boolean };
}

export type Locale = "tr" | "en";

// ── Accounts ────────────────────────────────────────────────────────────────────────
//
// Mirrors `engine/biosphere/api/routes/auth.py`. The session token appears exactly once, in
// the response to a sign-in; after that the client holds it and the server holds only its
// digest.

export interface AuthUser {
  id: string;
  email: string;
  name: string;
  avatar_url: string | null;
  role: string;
  /** Whether this account can sign in with a password, as opposed to only through Google. */
  has_password: boolean;
  has_google: boolean;
  created_at: string;
  /** Opened without signing in, in the engine's local mode: there is nothing to sign out of. */
  local?: boolean;
}

export interface AuthSession {
  user: AuthUser;
  token: string;
  expires_at: string;
}

export interface RegisterRequest {
  email: string;
  name: string;
  password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

// ── Investment assistant ─────────────────────────────────────────────────────────────
//
// Mirrors the "Investment assistant" section of `engine/biosphere/api/schemas.py`; the agent's
// figures are `engine/biosphere/invest/inputs.py`. Money is in Turkish lira, percentages are
// percentages (4.88, not 0.0488), and the card's values are raw numbers the client formats.

export type RentStatus = "expected" | "actual";
export type EnergyClass = "A" | "B" | "C" | "D" | "E" | "F" | "G";

/** The agent's figures for one property. Everything but the opt-in may be empty. */
export interface InvestmentInputs {
  /** The agent's opt-in. Off by default; the chat also needs `asking_price_tl`. */
  enabled: boolean;
  asking_price_tl?: number | null;
  monthly_rent_tl?: number | null;
  /** `actual`: let at this rent today. `expected`: the agent's estimate. */
  rent_status?: RentStatus | null;
  aidat_monthly_tl?: number | null;
  emlak_vergisi_yearly_tl?: number | null;
  dask_yearly_tl?: number | null;
  il?: string | null;
  ilce?: string | null;
  mahalle?: string | null;
  gross_m2?: number | null;
  net_m2?: number | null;
  room_layout?: string | null;
  building_age_years?: number | null;
  floor?: number | null;
  total_floors?: number | null;
  heating?: string | null;
  energy_class?: EnergyClass | null;
  sale_kind?: "second_hand" | "new_build" | null;
  in_site?: boolean | null;
  /** Buyer-side commission before KDV; null uses the legal ceiling, 0 means none. */
  commission_pct?: number | null;
  /** Facts the agent asserts for the assistant ("metroya 5 dk"). Never shown to visitors. */
  notes?: string | null;
  daily_budget_usd?: number | null;
}

export interface AdvisorSourceRef {
  title: string;
  url: string;
  date: string;
}

/** The deterministic figures card — computed by the engine, no model involved. */
export interface AdvisorCard {
  price_tl: number;
  location: string | null;
  /** What the agent did not enter: `rent`, `area`. */
  missing: string[];
  purchase: {
    total_costs_tl: number;
    total_outlay_tl: number;
    costs_pct_of_price: number;
    commission_pct: number;
    commission_basis: string;
  };
  rent: { monthly_tl: number; status: RentStatus } | null;
  income: {
    gross_pct: number;
    net_pct: number;
    net_annual_income_tl: number;
    gross_payback_years: number;
    net_payback_years: number | null;
    assumptions: {
      vacancy_months: number;
      maintenance_pct_of_rent: number;
      emlak_vergisi_tl: number;
      emlak_vergisi_basis: string;
      dask_tl: number;
      dask_basis: string;
      aidat_monthly_tl: number | null;
    };
  } | null;
  per_m2: { tl: number; basis: "net" | "brut"; area_m2: number } | null;
  mortgage: {
    monthly_payment_tl: number;
    loan_tl: number;
    down_payment_pct: number;
    down_payment_tl: number;
    term_months: number;
    monthly_rate_pct: number;
    ltv_cap_pct: number;
    rate_source: AdvisorSourceRef | null;
    rent_covers_payment_pct: number | null;
  };
}

/** What the player asks on mount. `enabled: false` carries nothing else. */
export interface AdvisorPublicState {
  enabled: boolean;
  chat_available: boolean;
  /** Why the chat is off while the card is on: `disabled` | `no_key` | `no_sdk` | `budget`. */
  reason: string | null;
  card: AdvisorCard | null;
  suggestions: string[];
  max_message_chars: number;
  max_turns: number;
}

export interface AdvisorChatRequest {
  message: string;
  conversation_id?: string | null;
  locale?: Locale;
}

/** One server-sent event of an answer. See `engine/biosphere/invest/advisor.py:run_turn`. */
export type AdvisorEvent =
  | { type: "meta"; conversation_id: string; turn: number; max_turns: number }
  | { type: "round"; n: number }
  | { type: "status"; phase: "thinking" | "searching" | "computing" | "writing"; query?: string; tool?: string }
  | { type: "text"; delta: string }
  | { type: "retract" }
  | { type: "cite"; url: string; title: string; date?: string | null }
  | { type: "calc"; tool: string; title: string; rows: [string, string][]; error?: boolean }
  | { type: "discard" }
  | { type: "notice"; code: string; message: string }
  | { type: "error"; code: string; message: string }
  | { type: "done"; turns_left: number; full: boolean };

export interface AdvisorUsageSummary {
  today_usd: number;
  last_7_days_usd: number;
  last_30_days_usd: number;
  all_time_usd: number;
  conversations_30d: number;
  questions_30d: number;
  calls_30d: number;
  web_searches_30d: number;
  avg_cost_per_question_usd: number | null;
  avg_call_latency_ms: number | null;
  by_model: { model: string; calls: number; cost_usd: number }[];
  daily: { date: string; cost_usd: number; questions: number }[];
}

export interface AdvisorRateRow {
  key: string;
  label: string;
  value: string;
  source: AdvisorSourceRef | null;
}

/** The dashboard panel's state. */
export interface InvestmentState {
  inputs: InvestmentInputs;
  card: AdvisorCard | null;
  status: { available: boolean; reason: string | null; model: string; web_search: boolean };
  visible_to_visitors: boolean;
  /** `not_enabled` | `no_price` | `not_published` | `disabled` | `no_key` | `no_sdk` | `budget`. */
  blockers: string[];
  usage: AdvisorUsageSummary;
  budget: {
    daily_budget_usd: number;
    /** The operator's default cap, in force when the agent sets none. */
    default_usd: number;
    operator_max_usd: number;
    spent_today_usd: number;
    global_daily_budget_usd: number;
    global_spent_today_usd: number;
  };
  rates: AdvisorRateRow[];
  rates_as_of: string;
  stale_rates: string[];
}

/** `GET /settings/ai`: whether an Anthropic key is set and where it comes from — never the key. */
export interface AiSettings {
  configured: boolean;
  source: "dashboard" | "env" | "none";
  /** `sk-ant-…` plus the last four characters. */
  hint: string;
  can_edit: boolean;
  listing_model: string;
  advisor_model: string;
}
