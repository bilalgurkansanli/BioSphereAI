/**
 * Public surface of the 360 player, plus the small internal contracts its parts share.
 *
 * The `PlayerProps` / `PlayerHandle` pair is imported by the dashboard's scene editor, so
 * treat it as an API: additive changes only.
 */

import type * as THREE from "three";
import type { FloorPlan, Hotspot, InitialView, Locale, Scene } from "@/lib/types";

// ---------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------

export interface PlayerProps {
  scenes: Scene[];
  initialSceneId?: string;
  locale?: Locale;
  /** Editor mode: clicking the sphere reports a direction instead of doing nothing. */
  editable?: boolean;
  onSphereClick?: (p: { yaw: number; pitch: number }) => void;
  onHotspotClick?: (h: Hotspot) => void;
  onHotspotDragEnd?: (h: Hotspot, p: { yaw: number; pitch: number }) => void;
  onSceneChange?: (sceneId: string) => void;
  onViewChange?: (v: InitialView) => void;
  /** Draft hotspots not yet persisted; rendered on top of `scene.hotspots`. */
  hotspotOverrides?: Hotspot[];
  /** Hide the toolbar/room switcher — used when the editor supplies its own. */
  showChrome?: boolean;
  /** The property also has a 3D walkthrough; the toolbar offers the way there. */
  walkAvailable?: boolean;
  onOpenWalk?: () => void;
  /**
   * Show the bottom-left tool rail (furniture removal, and whatever joins it).
   *
   * OFF BY DEFAULT, AND THAT IS A SECURITY DECISION AS MUCH AS A PRODUCT ONE. The tools
   * call `/api/v1/ai/*`, which is unauthenticated compute behind the *restricted* CORS
   * policy — only the public read routes and `/media` answer arbitrary origins. A player
   * framed on a third-party listing portal therefore could not use them even if they were
   * shown, and showing a button that cannot work is worse than not showing it. First-party
   * surfaces opt in.
   */
  tools?: boolean;
  /**
   * The property's embed token, when this player sits on somebody else's page.
   *
   * Exists so MEASURING can be offered where the rest of the rail cannot. The other tools
   * are unauthenticated GPU compute behind the restricted CORS policy, which is why `tools`
   * is off by default; measuring is neither — it is a ray meeting the floor plane, costs the
   * server microseconds, reads no pixels, and has its own public route. Passing the token
   * turns the tool on and points it at that route. A listing-page visitor asking "how wide is
   * this room" is exactly the question a virtual tour should answer.
   */
  measureToken?: string | null;
  /**
   * The property seen from above, if it has been measured.
   *
   * PASSED IN RATHER THAN FETCHED, because the two hosts of this player reach the plan by
   * different routes and neither should be the player's business: the embed is a server
   * component and already has the public payload in hand, while the dashboard holds the
   * private one and is also the only surface that may rebuild it. A player that fetched for
   * itself would have to know which of those it was inside — the same question `tools`
   * exists to keep out of here.
   *
   * `null` or a plan with nothing drawn means no button appears at all. MEASURING is not
   * offered here and never will be: it is a GPU pass behind the restricted CORS policy, so a
   * player framed on a listing portal could not run it, and the dashboard has its own panel
   * for it. The player reads the plan; it does not make one.
   */
  plan?: FloorPlan | null;
  autoRotate?: boolean;
  /**
   * Allow the investment assistant (the name is from the days it was announced, not built).
   *
   * On by default. The segment still appears only where it can answer: on a player that has
   * the listing's embed token (`measureToken`) and whose agent opted in and entered a price —
   * the engine says which (see `InvestmentChat`). A host that would rather not offer it on a
   * particular page turns it off here.
   */
  investmentSoon?: boolean;
  className?: string;
}

export interface PlayerHandle {
  lookAt(view: Partial<InitialView>, opts?: { animate?: boolean }): void;
  getView(): InitialView;
  goToScene(sceneId: string): void;
  enterVr(): Promise<void>;
  toggleFullscreen(): void;
  screenshot(): string | null;
}

// ---------------------------------------------------------------------------------
// Progressive texture ladder
// ---------------------------------------------------------------------------------

export type QualityRung = "preview" | "sd" | "hd" | "uhd";

/** Ascending. Index order is load order; the ladder never skips downwards. */
export const RUNG_ORDER: readonly QualityRung[] = ["preview", "sd", "hd", "uhd"] as const;

/** Long edge in pixels, used for the capability cap and the quality readout. */
export const RUNG_WIDTH: Record<QualityRung, number> = {
  preview: 512,
  sd: 2048,
  hd: 4096,
  uhd: 8192,
};

export const RUNG_LABEL: Record<QualityRung, string> = {
  preview: "LD",
  sd: "SD",
  hd: "HD",
  uhd: "8K",
};

export interface TextureStatus {
  /** Highest rung currently on screen, or null before the first paint. */
  rung: QualityRung | null;
  /** Best rung this device may load — hardware and memory capped. */
  maxRung: QualityRung;
  /** Highest rung this scene actually publishes a URL for. */
  bestAvailable: QualityRung | null;
  /** 0..1 for the rung currently streaming. */
  progress: number;
  /** True once anything at all is on screen. */
  ready: boolean;
  loading: boolean;
}

export const IDLE_TEXTURE_STATUS: TextureStatus = {
  rung: null,
  maxRung: "hd",
  bestAvailable: null,
  progress: 0,
  ready: false,
  loading: true,
};

// ---------------------------------------------------------------------------------
// Internal wiring between the HTML overlays and the r3f scene
// ---------------------------------------------------------------------------------

export interface LookControlsApi {
  getView(): InitialView;
  lookAt(view: Partial<InitialView>, opts?: { animate?: boolean; durationMs?: number }): void;
  /** Cancels inertia and any running `lookAt`, e.g. before a scene swap. */
  stop(): void;
  /** Marks the user as active so idle auto-rotate backs off. */
  poke(): void;
  /**
   * Transient field-of-view offset in degrees, layered on top of the user's own zoom so
   * the doorway transition can pinch the lens without discarding where they zoomed to.
   */
  setFovOffset(deg: number): void;
  /** Converts a client-space point to a sphere direction via the live camera. */
  directionAt(clientX: number, clientY: number): { yaw: number; pitch: number } | null;
  /** True while a pointer drag is in progress — suppresses click-through. */
  isDragging(): boolean;
}

/** Filled in from inside the `<Canvas>` so the HTML chrome can reach the renderer. */
export interface StageRuntime {
  camera: THREE.PerspectiveCamera | null;
  gl: THREE.WebGLRenderer | null;
  scene: THREE.Scene | null;
  controls: LookControlsApi | null;
  forceHighestQuality: (() => void) | null;
}

export function createStageRuntime(): StageRuntime {
  return { camera: null, gl: null, scene: null, controls: null, forceHighestQuality: null };
}

/**
 * One projected hotspot. The projector mutates the DOM through these refs every frame;
 * the `last*` fields exist so it only touches attributes that actually changed.
 */
export interface HotspotNode {
  root: HTMLElement;
  marker: HTMLElement | null;
  card: HTMLElement | null;
  yaw: number;
  pitch: number;
  /** Set while the user is dragging this tag in editable mode; overrides yaw/pitch. */
  dragYaw: number | null;
  dragPitch: number | null;
  /** The persisted yaw at drag start, so we can tell when the parent has caught up. */
  baseYaw: number;
  lastVisible: boolean | null;
  lastSide: string;
  lastVSide: string;
  lastOpacity: number;
  lastZ: number;
}

export type HotspotRegistry = Map<string, HotspotNode>;

/** Motion + fade timings, shared so the shader, the veil and the chrome agree. */
export const TIMING = {
  /** Cross-fade when a higher-resolution rung finishes loading. */
  rungFadeMs: 450,
  /** Doorway transition: dim out, swap, bloom back. */
  sceneOutMs: 180,
  sceneInMs: 320,
  /** Field-of-view pinch during the doorway transition, in degrees. */
  sceneFovDip: 8,
  /** Longest we will hold the black frame waiting for the next room's first texture. */
  sceneSwapTimeoutMs: 900,
  /**
   * Walking through a floor ring: how long the camera takes to reach the doorway.
   *
   * Longer than the list-driven swap above (180 ms) because this one is meant to be WATCHED —
   * it is the difference between choosing a room and going to it.
   *
   * Was 760 ms, which was not enough. Only the part before the veil is actually seen, so at the
   * old veil timing the whole of the motion a viewer got was 418 ms — under half a second, in
   * which the doorway grew 1.5x. That reads as a snap zoom followed by a cut, not as walking.
   * 1150 ms with the veil held later gives 828 ms of visible travel and 2.4x of growth.
   */
  portalTravelMs: 1150,
  /**
   * Fraction of the journey that is over before the veil starts closing.
   *
   * The single most important number here, because everything after it is spent behind black
   * and might as well not have been animated.
   */
  portalVeilAt: 0.72,
  /** No pointer, wheel or key input for this long before auto-rotate drifts. */
  idleMs: 4000,
  /** Auto-hide delay for the toolbar. */
  chromeHideMs: 3000,
} as const;

export const FOV_MIN = 30;
export const FOV_MAX = 100;
export const FOV_DEFAULT = 75;

/** Sphere radius. Large enough that near-plane clipping is impossible at any fov. */
export const SPHERE_RADIUS = 100;

/**
 * How far a portal journey travels, as a fraction of the sphere's radius.
 *
 * What matters is not the distance but how much of it is SEEN. At 0.58 the visible part of the
 * walk grew the doorway 1.5x, which the eye files as a zoom; at 0.72 it is 2.4x, which it files
 * as approach. The rest of the journey is behind the veil and only has to not look wrong.
 *
 * It has to stay well clear of 1.0 or the camera would pass through the sphere and the room
 * would turn inside out, and the texture does smear as it is magnified — but the smear arrives
 * with the black, which is the one place it costs nothing.
 */
export const PORTAL_DOLLY = 0.72;

export type { Hotspot, InitialView, Locale, Scene };
