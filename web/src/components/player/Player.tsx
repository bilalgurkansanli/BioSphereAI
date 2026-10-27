"use client";

/**
 * The 360 player.
 *
 * Shape of the thing: a `<Canvas>` holding exactly one object (the panorama sphere), a
 * stack of HTML overlays above it, and a single frame loop that owns the camera and then
 * writes every overlay's position by hand. React state is used for what changes rarely —
 * which room, which card is open, whether the toolbar is showing — and never for what
 * changes every frame.
 *
 * The one structural subtlety: hotspot projection has to happen inside the r3f frame
 * loop, immediately after the camera pose is written, but the hotspots themselves are DOM
 * nodes living outside the canvas. `PlayerStage` bridges the two by running the
 * projection from the controls' `onFrame` callback against a registry of element refs.
 */

import {
  Suspense,
  forwardRef,
  lazy,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { Canvas, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { absoluteUrl, api } from "@/lib/api";
import { t } from "@/lib/i18n";
import { cn, degrees } from "@/lib/utils";
import type { Hotspot, InitialView, PanoramaLevels, RestyleStyle, RestyleStyleId, Scene } from "@/lib/types";
import { PanoSphere } from "./PanoSphere";
import { hasDrawing, planHeadingDeg } from "./planGeometry";
import { PortalRings } from "./PortalRings";
import { hitTest, tagLift, unliftPitch } from "./portalGeometry";
import { PortalTransit, beginTransit } from "./PortalTransit";
import type { Transit } from "./PortalTransit";
import { RoomBar } from "./RoomBar";
import { HotspotLayer, projectHotspots } from "./HotspotLayer";
import {
  MeasureTool,
  createMeasureRegistry,
  projectMeasure,
  useMeasure,
  type MeasureRegistry,
} from "./MeasureTool";
import { LoadingVeil } from "./LoadingVeil";
import { PlayerChrome } from "./PlayerChrome";
import { PlayerDock } from "./PlayerDock";
import { usePanoVr } from "@/components/vr/usePanoVr";
import type { PanoVrActions, PanoVrCopy } from "@/components/vr/pano/panoVrWorld";
import { useLookControls } from "./useLookControls";
import { useEmptyRoom } from "./useEmptyRoom";
import { useSceneRestyle } from "./useSceneRestyle";
import { useProgressiveTexture } from "./useProgressiveTexture";
import {
  FOV_DEFAULT,
  IDLE_TEXTURE_STATUS,
  TIMING,
  createStageRuntime,
  type HotspotRegistry,
  type LookControlsApi,
  type PlayerHandle,
  type PlayerProps,
  type StageRuntime,
  type TextureStatus,
} from "./types";

/**
 * Everything VR draws or shows, as separate chunks: the headset scene inside the canvas, and the
 * guide and overlays above it. Neither is fetched until the VR button exists, and neither ever
 * is on a desktop without a headset — see `components/vr/usePanoVr.ts`.
 */
const VrPanoStage = lazy(() => import("@/components/vr/pano/VrPanoStage"));
const VrOverlays = lazy(() => import("@/components/vr/ui/VrOverlays"));

/**
 * Where the camera looks while walking through a ring, in radians.
 *
 * The ring is on the floor, so the click points sharply down; walking with that gaze would
 * march the viewer along staring at the tiles. Levelling completely is wrong too — it reads as
 * the camera snapping upright and breaks the link to the thing that was clicked. Seven degrees
 * down is a person walking towards a doorway they have just looked at.
 */
const PORTAL_GAZE = -0.12;

/** Peak opacity of the doorway dim. Not quite 1, so the room never fully disappears. */
const DIM_PEAK = 0.94;

/**
 * A stable empty list, so "no rings while measuring" is not a new array identity every render.
 * `PortalRings` builds meshes from what it is handed; a fresh `[]` each time would have it
 * tearing them down and rebuilding them once per render for as long as the tool is on.
 */
const NO_PORTALS: Hotspot[] = [];

// ---------------------------------------------------------------------------------
// In-canvas stage
// ---------------------------------------------------------------------------------

interface PlayerStageProps {
  panorama: PanoramaLevels | null;
  sceneKey: string;
  preloadUrls: string[];
  initialView: InitialView;
  container: HTMLElement | null;
  runtime: React.RefObject<StageRuntime>;
  registry: React.RefObject<HotspotRegistry>;
  /** The measure tool's marks and lines, projected in the same pass as the tags. */
  measureRegistry: React.RefObject<MeasureRegistry>;
  portals: Hotspot[];
  readyScenes: ReadonlySet<string>;
  onPortalHover: (portal: Hotspot | null) => void;
  transit: React.RefObject<Transit | null>;
  compassRef: React.RefObject<HTMLDivElement | null>;
  /** The plan's heading cone, and the rotation taking this room's yaw into the plan's frame. */
  headingRef: React.RefObject<SVGGElement | null>;
  planRotation: React.RefObject<number | null>;
  controlsRef: React.RefObject<LookControlsApi | null>;
  autoRotate: boolean;
  enabled: boolean;
  onStatus: (status: TextureStatus) => void;
  onTap: (p: { yaw: number; pitch: number }) => void;
  onViewSettled?: (view: InitialView) => void;
  onRequestFullscreen: () => void;
  onRequestReset: () => void;
  /** Covered pitch band of the scene, in radians; the look controls will not leave it. */
  pitchLimits: { min: number; max: number };
  /** Presenting to a headset: the texture ladder may climb to 8K (see `useProgressiveTexture`). */
  headset: boolean;
  /**
   * The group holding the sphere and its rings. Identity outside a headset; in one, VR moves it
   * to the head and turns it (`components/vr/pano/panoVrWorld.ts`).
   */
  content: React.RefObject<THREE.Group | null>;
}

function PlayerStage(props: PlayerStageProps) {
  const {
    panorama,
    sceneKey,
    preloadUrls,
    initialView,
    container,
    runtime,
    registry,
    measureRegistry,
    portals,
    readyScenes,
    onPortalHover,
    transit,
    compassRef,
    headingRef,
    planRotation,
    controlsRef,
    autoRotate,
    enabled,
    onStatus,
    onTap,
    onViewSettled,
    onRequestFullscreen,
    onRequestReset,
    headset,
    content,
  } = props;

  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);

  const { texture, forceHighest } = useProgressiveTexture({ panorama, sceneKey, preloadUrls, onStatus, headset });

  useEffect(() => {
    const store = runtime.current;
    store.gl = gl;
    store.scene = scene;
    store.camera = camera instanceof THREE.PerspectiveCamera ? camera : null;
    store.forceHighestQuality = forceHighest;
    // Declared up front so a session request later does not have to reconfigure the
    // renderer mid-gesture; r3f keeps this in sync with `isPresenting` afterwards.
    gl.xr.enabled = true;
    return () => {
      store.gl = null;
      store.scene = null;
      store.camera = null;
      store.forceHighestQuality = null;
    };
  }, [gl, scene, camera, forceHighest, runtime]);

  const sizeRef = useRef(size);
  useEffect(() => {
    sizeRef.current = size;
  }, [size]);

  const onFrame = useCallback(
    (view: Readonly<InitialView>, activeCamera: THREE.PerspectiveCamera) => {
      const { width, height } = sizeRef.current;
      projectHotspots(registry.current, activeCamera, width, height);
      // Same pass, same camera, same reason: a measurement drawn from a stale pose lags the
      // photograph by a frame, and a line that lags is a line that is not on the floor.
      projectMeasure(measureRegistry.current, activeCamera, width, height);
      const dial = compassRef.current;
      if (dial) dial.style.transform = `rotate(${(-degrees(view.yaw)).toFixed(2)}deg)`;
      // The plan's heading cone, written the same way and for the same reason: this runs at
      // the frame rate, and a `setState` here would re-render the canvas with it.
      const cone = headingRef.current;
      const rotation = planRotation.current;
      if (cone && rotation !== null) {
        cone.setAttribute("transform", `rotate(${planHeadingDeg({ rotation }, view.yaw).toFixed(2)})`);
      }
    },
    [registry, measureRegistry, compassRef, headingRef, planRotation],
  );

  useLookControls({
    target: container,
    initialView,
    enabled,
    autoRotate,
    onFrame,
    onViewSettled,
    onTap,
    onRequestFullscreen,
    onRequestReset,
    apiRef: controlsRef,
    pitchLimits: props.pitchLimits,
  });

  return (
    <>
      <group ref={content}>
        <PanoSphere texture={texture} cutKey={sceneKey} />
        <PortalRings
          portals={portals}
          readyScenes={readyScenes}
          travelling={transit}
          onHoverChange={onPortalHover}
        />
      </group>
      <PortalTransit transit={transit} />
    </>
  );
}

// ---------------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------------

export const Player = forwardRef<PlayerHandle, PlayerProps>(function Player(props, ref) {
  const {
    scenes,
    initialSceneId,
    locale = "tr",
    editable = false,
    onSphereClick,
    onHotspotClick,
    onHotspotDragEnd,
    onSceneChange,
    onViewChange,
    hotspotOverrides,
    showChrome = true,
    tools = false,
  measureToken = null,
    plan = null,
    autoRotate: autoRotateProp = false,
    investmentSoon = true,
    className,
  } = props;

  const playable = useMemo(() => scenes.filter((s) => s.panorama !== null), [scenes]);

  const [container, setContainer] = useState<HTMLElement | null>(null);
  // The canvas is mounted only after hydration. A WebGL context has nothing to say
  // during SSR, and gating it here removes an entire class of hydration mismatch.
  const [mounted, setMounted] = useState(false);
  const [sceneId, setSceneId] = useState<string | null>(
    () => initialSceneId ?? playable[0]?.id ?? scenes[0]?.id ?? null,
  );
  const [status, setStatus] = useState<TextureStatus>(IDLE_TEXTURE_STATUS);
  const [everPainted, setEverPainted] = useState(false);
  const [dim, setDim] = useState(0);
  const [dimDuration, setDimDuration] = useState<number>(TIMING.sceneOutMs);
  const [autoRotate, setAutoRotate] = useState(autoRotateProp);
  const [chromeVisible, setChromeVisible] = useState(true);
  const [roomsOpen, setRoomsOpen] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [planOpen, setPlanOpen] = useState(false);

  const runtime = useRef<StageRuntime>(createStageRuntime());
  const registry = useRef<HotspotRegistry>(new Map());
  const measureRegistry = useRef<MeasureRegistry>(createMeasureRegistry());
  const controlsRef = useRef<LookControlsApi | null>(null);
  const compassRef = useRef<HTMLDivElement | null>(null);
  const headingRef = useRef<SVGGElement | null>(null);
  /**
   * The rotation taking this room's yaw into the plan's frame, or null when the room is not
   * on the plan. A ref rather than state: the frame loop reads it and must not cause renders.
   */
  const planRotation = useRef<number | null>(null);
  const pendingReveal = useRef(false);
  const transitRef = useRef<Transit | null>(null);
  /** The sphere and its rings as one group, which a headset moves to the head and turns. */
  const contentRef = useRef<THREE.Group | null>(null);
  const transitionTimers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const sceneIdRef = useRef(sceneId);
  const playableRef = useRef(playable);

  // These mirrors exist so the imperative handle and the transition machinery can read
  // current values without being rebuilt on every render. Updated after commit, and
  // declared above every effect that reads them so they are never a render behind.
  useEffect(() => {
    sceneIdRef.current = sceneId;
    playableRef.current = playable;
  }, [sceneId, playable]);

  // "Embedded" means framed by somebody else's page — which is exactly when the small
  // BioSphere mark should appear, and exactly when it should not inside our own dashboard.
  const [embedded, setEmbedded] = useState(false);

  useEffect(() => {
    setMounted(true);
    try {
      setEmbedded(window.self !== window.top);
    } catch {
      // Cross-origin access to window.top throws, which is itself proof of framing.
      setEmbedded(true);
    }
  }, []);
  useEffect(() => setAutoRotate(autoRotateProp), [autoRotateProp]);
  useEffect(() => {
    if (status.ready) setEverPainted(true);
  }, [status.ready]);

  const currentScene: Scene | null = useMemo(
    () => playable.find((s) => s.id === sceneId) ?? playable[0] ?? null,
    [playable, sceneId],
  );

  /** Whether there is a drawing worth opening at all; see `planGeometry.hasDrawing`. */
  const planAvailable = useMemo(() => hasDrawing(plan), [plan]);

  /**
   * Whether the plan is on screen right now.
   *
   * Derived rather than kept in step with an effect: a property can stop having a drawing while
   * the plan is open (a room deleted, a rebuild that measured nothing), and a `planOpen` that
   * had to be *reset* would leave the toolbar hidden behind a plan that is no longer there
   * for however long that reset took to land.
   */
  const planShowing = planOpen && planAvailable;

  useEffect(() => {
    const room = plan?.rooms.find(
      (candidate) => candidate.scene_id === sceneId && candidate.placement !== "unlinked",
    );
    planRotation.current = room ? room.rotation : null;
  }, [plan, sceneId]);

  /**
   * Point the cone the moment the plan appears, rather than waiting for the next frame.
   *
   * The same lesson as the portal hover pill: something written only by a running loop is
   * unplaced for as long as it takes that loop to come round, and an arrow pointing at the
   * default direction for even one frame reads as an arrow pointing the wrong way.
   */
  useEffect(() => {
    if (!planShowing) return;
    const cone = headingRef.current;
    const rotation = planRotation.current;
    const view = controlsRef.current?.getView();
    if (cone && rotation !== null && view) {
      cone.setAttribute("transform", `rotate(${planHeadingDeg({ rotation }, view.yaw).toFixed(2)})`);
    }
  }, [planShowing, sceneId, plan]);

  const forceQuality = useCallback(() => runtime.current.forceHighestQuality?.(), []);

  /**
   * The furniture toggle. It substitutes a second ladder for this scene's own; nothing
   * about the scene changes, so turning it off is a state flip and a reload comes back to
   * the real room.
   */
  const emptyRoomCopy = useMemo(
    () => ({ empty: t("player.empty.none", locale), failed: t("player.empty.failed", locale) }),
    [locale],
  );
  const emptyRoom = useEmptyRoom(currentScene?.id ?? null, locale, emptyRoomCopy);

  /**
   * The style rail's state. Same substitution as the furniture toggle, five answers instead
   * of one — see `useSceneRestyle`.
   */
  const decorCopy = useMemo(() => ({ failed: t("player.decor.failed", locale) }), [locale]);
  const restyle = useSceneRestyle(currentScene?.id ?? null, locale, decorCopy);

  /**
   * The measure tool. Unlike the two above it changes NOTHING about the picture — it draws on
   * top of it — so it does not take part in the "these two are alternatives" dance below, and
   * it can be left on while the furniture is hidden or the room is staged. That combination is
   * useful rather than confusing: the wall the sofa has to fit against is easier to mark once
   * the old sofa is out of the way, and the geometry is the panorama's regardless of which
   * ladder is currently painted on the sphere.
   */
  const measureCopy = useMemo(() => ({ failed: t("player.measure.failed", locale) }), [locale]);
  const measure = useMeasure(currentScene?.id ?? null, measureCopy, measureToken);
  // Pulled out as values so `handleTap` can depend on the two things it actually uses. The
  // state object itself is rebuilt every render, so depending on it would rebuild the tap
  // handler every render too — and that handler is re-registered on the container by
  // `useLookControls` each time its identity changes.
  const measuring = measure.active;
  const addMeasureMark = measure.add;

  /**
   * The style catalogue, fetched once and ONLY when the tool rail is on.
   *
   * `tools` is off by default and that is a security decision (see `PlayerProps`): a player
   * framed on a third-party portal is outside the restricted CORS policy and could not call
   * these routes anyway. Asking for the catalogue there would be a request that always fails,
   * once per embed, and a console error on somebody else's page.
   *
   * An empty list is also the honest answer to "this deployment has no GPU": the rail simply
   * has no staging button, rather than one that fails when pressed.
   */
  const [decorStyles, setDecorStyles] = useState<RestyleStyle[]>([]);
  useEffect(() => {
    if (!tools) return;
    let cancelled = false;
    void api.restyle
      .styles()
      .then((body) => {
        if (!cancelled && body.available) setDecorStyles(body.styles);
      })
      .catch(() => {
        /* No catalogue means no button. Nothing to report to someone looking at a room. */
      });
    return () => {
      cancelled = true;
    };
  }, [tools]);

  // The two tools are ALTERNATIVES, not layers: each substitutes its own ladder for the
  // scene's, so having both on would mean one silently winning. Turning either on turns the
  // other off, here rather than inside the hooks — neither of them should have to know the
  // other exists.
  const toggleEmptyRoom = useCallback(() => {
    if (restyle.activeStyle) restyle.select(null);
    emptyRoom.toggle();
    // Paired with the swap, in the same commit: without it the texture ladder restarts at
    // the 512 px preview and the toggle reads as the picture going soft rather than as the
    // furniture leaving. See `useEmptyRoom`'s header.
    forceQuality();
  }, [emptyRoom, restyle, forceQuality]);

  const selectStyle = useCallback(
    (style: RestyleStyleId | null) => {
      if (style !== null && emptyRoom.active) emptyRoom.toggle();
      restyle.select(style);
      forceQuality();
    },
    [emptyRoom, restyle, forceQuality],
  );

  // Absolute URLs as primitives, so the loader's effect keys off the strings rather than
  // an object identity the caller may recreate on every render.
  const levels = restyle.levels ?? emptyRoom.levels ?? currentScene?.panorama ?? null;
  const previewUrl = absoluteUrl(levels?.preview_url);
  const sdUrl = absoluteUrl(levels?.sd_url);
  const hdUrl = absoluteUrl(levels?.hd_url);
  const uhdUrl = absoluteUrl(levels?.uhd_url);
  const panoWidth = levels?.width ?? 0;
  const panoHeight = levels?.height ?? 0;

  const panorama = useMemo<PanoramaLevels | null>(
    () =>
      previewUrl
        ? {
            preview_url: previewUrl,
            sd_url: sdUrl,
            hd_url: hdUrl,
            uhd_url: uhdUrl,
            width: panoWidth,
            height: panoHeight,
            cube_urls: null,
          }
        : null,
    [previewUrl, sdUrl, hdUrl, uhdUrl, panoWidth, panoHeight],
  );

  const sceneNames = useMemo(() => new Map(scenes.map((s) => [s.id, s.name])), [scenes]);

  const hotspots = useMemo<Hotspot[]>(() => {
    if (!currentScene) return [];
    const merged = new Map(currentScene.hotspots.map((h) => [h.id, h]));
    for (const draft of hotspotOverrides ?? []) {
      if (draft.scene_id && draft.scene_id !== currentScene.id) continue;
      merged.set(draft.id, draft);
    }
    return [...merged.values()];
  }, [currentScene, hotspotOverrides]);

  /**
   * The two presentations of a hotspot, split once.
   *
   * A portal is drawn as a ring on the floor, so in the PLAYER it must not also appear as a
   * tag — one thing, two markers, is how a room ends up looking like a control panel. In the
   * EDITOR it stays in the tag list, because that is where placing, dragging and retargeting
   * live and none of that works against a mesh.
   */
  const portals = useMemo(
    () => hotspots.filter((h) => h.type === "portal" && h.target_scene_id),
    [hotspots],
  );
  const tags = useMemo(
    () =>
      editable
        ? // In the editor a portal keeps its marker, because that marker is the only way to
          // select it and the only way to drag it somewhere else. It is lifted off the ring so
          // that it stops swallowing the clicks meant for the ring — see `tagLift`.
          hotspots.map((h) =>
            h.type === "portal" ? { ...h, pitch: h.pitch + tagLift(h.pitch) } : h,
          )
        : hotspots.filter((h) => h.type !== "portal"),
    [hotspots, editable],
  );

  /** The rooms a portal can actually take somebody to. */
  const readyScenes = useMemo(() => new Set(playable.map((scene) => scene.id)), [playable]);

  /** Which ring the pointer is over, so the destination can be named before the click. */
  const [hoveredPortal, setHoveredPortal] = useState<Hotspot | null>(null);
  const hoverPillRef = useRef<HTMLDivElement>(null);

  /**
   * The room the viewer has walked into, when there is no panorama on the other side.
   *
   * Held rather than timed out: the viewer is standing in the dark and a message that expired
   * would leave them in an unexplained black rectangle with no way back.
   */
  const [stranded, setStranded] = useState<{ name: string } | null>(null);

  /**
   * The destination label follows the POINTER, not the ring.
   *
   * Anchoring it to the ring would be more precise and is the wrong trade: the ring is on the
   * floor, so its label would sit in the middle of the floor with nothing to relate it to, and
   * on a ring seen almost edge-on it would land on top of the ring itself. At the cursor it is
   * where the eye already is.
   *
   * TRACKED ALWAYS, PLACED ON APPEARANCE. Listening only while a ring is hovered reads as the
   * cheaper version of the same thing and is not: by the time the hover re-renders, the pointer
   * has already arrived and stopped, so the next move event can be a long time coming - or never,
   * if the viewer holds still - and until then the pill sits at the container's origin, which
   * after its own -100% offset is above the top edge. Measured on a real hover: the label was in
   * the DOM with the right text and `transform: none`, i.e. invisible. So the position is kept
   * whether or not anything is hovered, and the pill places itself the moment it mounts.
   *
   * Written straight to the element. This fires on every pointermove; a `setState` here would
   * re-render the whole player, canvas included, at the pointer's sample rate.
   */
  const pointerRef = useRef<{ x: number; y: number } | null>(null);
  const placePill = useCallback((x: number, y: number) => {
    const pill = hoverPillRef.current;
    if (pill) pill.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
  }, []);

  useEffect(() => {
    if (!container) return;
    const move = (event: PointerEvent) => {
      const box = container.getBoundingClientRect();
      const x = event.clientX - box.left;
      const y = event.clientY - box.top;
      pointerRef.current = { x, y };
      placePill(x, y);
    };
    container.addEventListener("pointermove", move);
    return () => container.removeEventListener("pointermove", move);
  }, [container, placePill]);

  /** A portal's destination, named. Falls back to the tag's own title for a room since gone. */
  const hoveredPortalName = useMemo(() => {
    if (!hoveredPortal) return null;
    const target = hoveredPortal.target_scene_id
      ? scenes.find((scene) => scene.id === hoveredPortal.target_scene_id)
      : null;
    return target?.name ?? hoveredPortal.title ?? null;
  }, [hoveredPortal, scenes]);

  // Rooms one step away are the ones about to be needed: neighbours in the property order,
  // plus anywhere a link tag in this room points.
  const preloadUrls = useMemo(() => {
    if (!currentScene) return [];
    const index = playable.findIndex((s) => s.id === currentScene.id);
    const candidates = new Set<string>();
    for (const offset of [1, -1]) {
      const neighbour = playable[index + offset];
      if (neighbour) candidates.add(neighbour.id);
    }
    for (const hotspot of currentScene.hotspots) {
      if (hotspot.type === "link" && hotspot.target_scene_id) candidates.add(hotspot.target_scene_id);
    }
    candidates.delete(currentScene.id);

    const urls: string[] = [];
    for (const id of candidates) {
      const target = playable.find((s) => s.id === id);
      const url = absoluteUrl(target?.panorama?.sd_url ?? target?.panorama?.preview_url);
      if (url) urls.push(url);
    }
    return urls;
  }, [currentScene, playable]);

  // ---------------------------------------------------------------------------------
  // Scene transitions
  // ---------------------------------------------------------------------------------

  const clearTransition = useCallback(() => {
    for (const timer of transitionTimers.current) clearTimeout(timer);
    transitionTimers.current = [];
    // Abandoning a walk mid-stride leaves the camera off-centre; the driver parks it back at
    // the sphere's middle on the next frame, which is the only place a panorama is undistorted.
    transitRef.current = null;
  }, []);


  const reveal = useCallback(() => {
    pendingReveal.current = false;
    setDimDuration(TIMING.sceneInMs);
    setDim(0);
    controlsRef.current?.setFovOffset(0);
  }, []);

  /**
   * Put the next room on screen. Assumes the screen is already dark.
   *
   * Shared by the two ways of arriving — the quick dim of a room picked off a list, and the
   * walk through a floor ring — because everything after the black frame is identical, and
   * because a second copy is how the two would drift apart.
   */
  const swapTo = useCallback(
    (id: string, fromYaw?: number) => {
      const target = playableRef.current.find((scene) => scene.id === id);
      if (!target) return;
      setSceneId(id);
      onSceneChange?.(id);
      controlsRef.current?.lookAt(
        fromYaw === undefined
          ? target.initial_view
          : { yaw: fromYaw, pitch: 0, fov: target.initial_view.fov },
        { animate: false },
      );
      pendingReveal.current = true;
      // Held black waits for the new room's first texture, but never for long: a stalled
      // request must not leave the viewer staring at a dark rectangle.
      transitionTimers.current.push(setTimeout(reveal, TIMING.sceneSwapTimeoutMs));
    },
    [onSceneChange, reveal],
  );

  /**
   * Walk to another room.
   *
   * Dim, pinch the lens in, swap, then bloom back out — the sequence reads as stepping
   * through a doorway rather than as a slideshow cut. `fromYaw` carries the direction the
   * user was travelling when they clicked a link tag, so they arrive facing onward
   * instead of being spun to face an arbitrary compass bearing.
   */
  const goToScene = useCallback(
    (id: string, fromYaw?: number) => {
      if (id === sceneIdRef.current) return;
      const target = playableRef.current.find((s) => s.id === id);
      if (!target) return;

      clearTransition();
      setRoomsOpen(false);
      controlsRef.current?.stop();
      controlsRef.current?.setFovOffset(-TIMING.sceneFovDip);
      setDimDuration(TIMING.sceneOutMs);
      setDim(DIM_PEAK);

      transitionTimers.current.push(setTimeout(() => swapTo(id, fromYaw), TIMING.sceneOutMs));
    },
    [clearTransition, swapTo],
  );

  /**
   * Walk through a floor ring.
   *
   * The ring is a place, so the click has to be a journey: the camera turns to the doorway and
   * moves towards it, the room growing around it, and only then does the picture give way. See
   * `PortalTransit` for why a camera pinned to the centre of a sphere can travel at all.
   *
   * ARRIVING IN THE DARK IS A REAL DESTINATION. When the room beyond has not been shot yet
   * there is nothing to swap in — and the walk still happens, ending in black. That is not a
   * failure state dressed up: it is what is actually through that door today, and it keeps the
   * ring behaving like a ring while the property is half shot. Earlier this said so with a floating
   * message and no movement, which made the ring look like a button that had refused.
   */
  const travelToPortal = useCallback(
    (portal: Hotspot) => {
      const targetId = portal.target_scene_id;
      if (!targetId || targetId === sceneIdRef.current) return;
      const ready = readyScenes.has(targetId);
      const room = scenes.find((scene) => scene.id === targetId);

      clearTransition();
      setRoomsOpen(false);
      controlsRef.current?.stop();

      const arrive = () => {
        transitRef.current = null;
        if (ready) swapTo(targetId, portal.yaw);
        else setStranded({ name: room?.name ?? portal.title });
      };

      // Somebody who has asked not to be moved gets the destination, not the trip.
      if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
        setDimDuration(TIMING.sceneOutMs);
        setDim(1);
        transitionTimers.current.push(setTimeout(arrive, TIMING.sceneOutMs));
        return;
      }

      // Lift the gaze off the floor and square up to the doorway, overlapping the walk so it
      // reads as one movement instead of "turn, stop, then go". Not all the way to level: a
      // little of the ring's own depression survives, or the camera would seem to snap upright.
      controlsRef.current?.lookAt(
        { yaw: portal.yaw, pitch: Math.max(portal.pitch * 0.35, PORTAL_GAZE) },
        { animate: true, durationMs: Math.round(TIMING.portalTravelMs * 0.4) },
      );

      transitRef.current = beginTransit(portal.yaw, TIMING.portalTravelMs, arrive);

      // The veil closes over the tail of the walk, so the doorway is watched growing before the
      // picture goes rather than being cut away at the start of the move.
      const holdMs = Math.round(TIMING.portalTravelMs * TIMING.portalVeilAt);
      transitionTimers.current.push(
        setTimeout(() => {
          setDimDuration(TIMING.portalTravelMs - holdMs);
          setDim(1);
        }, holdMs),
      );
    },
    [clearTransition, readyScenes, scenes, swapTo],
  );

  /** Back out of a room that turned out to be dark. */
  const leaveTheDark = useCallback(() => {
    setStranded(null);
    reveal();
  }, [reveal]);

  useEffect(() => {
    if (pendingReveal.current && status.ready) reveal();
  }, [status.ready, reveal]);

  useEffect(() => clearTransition, [clearTransition]);

  // Let a parent drive the room by changing `initialSceneId`.
  useEffect(() => {
    if (initialSceneId && initialSceneId !== sceneIdRef.current) goToScene(initialSceneId);
  }, [initialSceneId, goToScene]);

  // ---------------------------------------------------------------------------------
  // Fullscreen, VR, screenshot
  // ---------------------------------------------------------------------------------

  const toggleFullscreen = useCallback(() => {
    if (typeof document === "undefined") return;
    if (document.fullscreenElement) {
      void document.exitFullscreen().catch(() => undefined);
    } else {
      void container?.requestFullscreen?.().catch(() => undefined);
    }
  }, [container]);

  useEffect(() => {
    if (typeof document === "undefined") return;
    const sync = () => setFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", sync);
    return () => document.removeEventListener("fullscreenchange", sync);
  }, []);

  /**
   * VR: a headset over WebXR, or a phone in a VR box. Its button is a dock segment, and only
   * exists where one of the two can actually work (`components/vr/capabilities.ts`).
   */
  const vrMessages = useMemo(
    () => ({ denied: t("vr.denied", locale), noSensor: t("vr.noSensor", locale), failed: t("vr.failed", locale) }),
    [locale],
  );
  const getRenderer = useCallback(() => runtime.current.gl, []);
  const vrHandoffParams = useCallback(() => {
    const view = controlsRef.current?.getView();
    return { scene: sceneIdRef.current, yaw: view?.yaw ?? null, pitch: view?.pitch ?? null };
  }, []);
  const vr = usePanoVr({ renderer: getRenderer, container, messages: vrMessages, handoffParams: vrHandoffParams });
  const { enter: vrEnter, exit: vrExit } = vr;

  const enterVr = useCallback(() => {
    // From the dark room, a headset would open on a black sphere with nothing to look at.
    if (stranded) leaveTheDark();
    vrEnter();
  }, [stranded, leaveTheDark, vrEnter]);

  /**
   * What the headset can do: walk through a ring or a link (the same doorway as the 2D player,
   * minus the dolly — see `panoVrWorld`), step through the rooms from the menu, and leave.
   */
  const vrActions = useMemo<PanoVrActions>(
    () => ({
      navigate: (sceneId, fromYaw) => goToScene(sceneId, fromYaw),
      step: (delta) => {
        const list = playableRef.current;
        if (list.length < 2) return;
        const index = Math.max(0, list.findIndex((scene) => scene.id === sceneIdRef.current));
        goToScene(list[(index + delta + list.length) % list.length].id);
      },
      exit: () => vrExit(),
    }),
    [goToScene, vrExit],
  );
  const vrCopy = useMemo<PanoVrCopy>(
    () => ({
      exit: t("vr.exit", locale),
      pending: t("vr.pending", locale),
      closeCard: t("vr.card.close", locale),
      position: (index, count) => `${index} / ${count}`,
    }),
    [locale],
  );
  /** Every tag but the portals — those are the rings. Unlike `tags`, never lifted for the editor. */
  const vrTags = useMemo(() => hotspots.filter((h) => h.type !== "portal"), [hotspots]);

  const screenshot = useCallback((): string | null => {
    const { gl, scene, camera } = runtime.current;
    if (!gl || !scene || !camera) return null;
    try {
      // The drawing buffer is cleared after compositing unless `preserveDrawingBuffer` is
      // set, and that flag costs a full-frame copy every frame forever. Re-rendering just
      // before the read is the same result for the price of one frame.
      gl.render(scene, camera);
      return gl.domElement.toDataURL("image/jpeg", 0.92);
    } catch {
      // Tainted canvas: a texture host that did not send CORS headers. Nothing to do.
      return null;
    }
  }, []);

  const recentre = useCallback(() => {
    controlsRef.current?.lookAt({ yaw: 0, pitch: 0 }, { animate: true });
  }, []);

  const resetView = useCallback(() => {
    const view = playableRef.current.find((s) => s.id === sceneIdRef.current)?.initial_view;
    controlsRef.current?.lookAt(view ?? { yaw: 0, pitch: 0, fov: FOV_DEFAULT }, { animate: true });
  }, []);

  useImperativeHandle(
    ref,
    (): PlayerHandle => ({
      lookAt: (view, opts) => controlsRef.current?.lookAt(view, opts),
      getView: () =>
        controlsRef.current?.getView() ??
        playableRef.current.find((s) => s.id === sceneIdRef.current)?.initial_view ?? {
          yaw: 0,
          pitch: 0,
          fov: FOV_DEFAULT,
        },
      goToScene: (id) => goToScene(id),
      enterVr: async () => enterVr(),
      toggleFullscreen,
      screenshot,
    }),
    [goToScene, enterVr, toggleFullscreen, screenshot],
  );

  // ---------------------------------------------------------------------------------
  // Chrome visibility
  // ---------------------------------------------------------------------------------

  const chromeState = useRef({ overUi: false, inside: false, roomsOpen: false, fine: false });
  useEffect(() => {
    chromeState.current.roomsOpen = roomsOpen;
  }, [roomsOpen]);

  useEffect(() => {
    if (!container) return;
    chromeState.current.fine = window.matchMedia?.("(pointer: fine)").matches ?? false;

    let timer: ReturnType<typeof setTimeout> | undefined;
    const arm = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const s = chromeState.current;
        // A mouse resting inside the frame is still "using" the player; a finger that
        // stopped moving three seconds ago is not.
        if (s.overUi || s.roomsOpen || (s.fine && s.inside)) {
          arm();
          return;
        }
        setChromeVisible(false);
      }, TIMING.chromeHideMs);
    };

    const bump = () => {
      setChromeVisible(true);
      setShowHint(false);
      arm();
    };

    const onOver = (event: PointerEvent) => {
      chromeState.current.inside = true;
      chromeState.current.overUi = event.target instanceof Element && Boolean(event.target.closest("[data-player-ui]"));
      bump();
    };
    const onLeave = () => {
      chromeState.current.inside = false;
      chromeState.current.overUi = false;
      arm();
    };

    container.addEventListener("pointermove", onOver);
    container.addEventListener("pointerdown", onOver);
    container.addEventListener("keydown", bump);
    container.addEventListener("pointerleave", onLeave);
    arm();

    return () => {
      clearTimeout(timer);
      container.removeEventListener("pointermove", onOver);
      container.removeEventListener("pointerdown", onOver);
      container.removeEventListener("keydown", bump);
      container.removeEventListener("pointerleave", onLeave);
    };
  }, [container]);

  // The nudge appears once the first panorama is up, and only until the user touches it.
  useEffect(() => {
    if (!everPainted || !showChrome) return;
    setShowHint(true);
    const timer = setTimeout(() => setShowHint(false), 5200);
    return () => clearTimeout(timer);
  }, [everPainted, showChrome]);

  // ---------------------------------------------------------------------------------
  // Interaction wiring
  // ---------------------------------------------------------------------------------

  const handleTap = useCallback(
    (point: { yaw: number; pitch: number }) => {
      setRoomsOpen(false);
      // BEFORE EVERYTHING, including the portal hit test. While the measure tool is on, a tap
      // on the picture means one thing and one thing only: put a mark there. Letting a tap that
      // happened to land inside a floor ring walk the viewer into the next room instead would
      // throw away the marks they had already placed (marks belong to the room they were made
      // in — see `useMeasure`) and leave them somewhere else entirely, for a gesture they meant
      // as "measure to here". The rings are also taken off the floor for the duration, below,
      // so there is nothing there to aim at by mistake in the first place.
      if (measuring) {
        addMeasureMark(point);
        return;
      }
      // BEFORE the editor's own "place a tag here", because in the editor a portal has to
      // stay clickable as a portal — otherwise the only way to test a property you are building is
      // to publish it. It used to read `editable ? null : hitTest(...)`, which did the exact
      // opposite of the sentence above it: in the editor every click on a ring fell through to
      // "add a tag here", so the ring looked broken to the person who had just placed it.
      const portal = hitTest(portals, point.yaw, point.pitch);
      if (portal) {
        setHoveredPortal(null);
        travelToPortal(portal);
        return;
      }
      if (editable) onSphereClick?.(point);
    },
    [editable, measuring, addMeasureMark, onSphereClick, portals, travelToPortal],
  );

  const handleDragEnd = useCallback(
    (hotspot: Hotspot, position: { yaw: number; pitch: number }) => {
      onHotspotDragEnd?.(
        hotspot,
        hotspot.type === "portal"
          ? { ...position, pitch: unliftPitch(position.pitch) }
          : position,
      );
    },
    [onHotspotDragEnd],
  );

  const handleNavigate = useCallback(
    (hotspot: Hotspot) => {
      if (!hotspot.target_scene_id) return;
      goToScene(hotspot.target_scene_id, hotspot.yaw);
    },
    [goToScene],
  );

  /**
   * The bottom-left dock's visibility, which is NOT the toolbar's.
   *
   * Tying it to `chromeVisible` was wrong in the place it matters most. The scene editor
   * passes `showChrome={false}` because it supplies its own header — so there is no toolbar
   * on screen to explain the rail disappearing, and three seconds after the page settled
   * the only control the feature has simply vanished. It looked like it had never rendered.
   *
   * So: when the host has suppressed the player's chrome, the host is driving the interface
   * and the dock stays put. When the player owns its chrome, the dock follows it — the
   * photograph should be unobstructed while someone is just looking — except while a tool is
   * working or has something to say, because a control must not leave mid-thought.
   *
   * ONE RULE FOR THE WHOLE CORNER, deliberately: this is a superset of `chromeVisible`, so
   * the plan button it now also governs never leaves earlier than it used to, and the rows
   * above the bar can never fade out from under it and drop everything by their own height.
   */
  const dockVisible =
    !showChrome ||
    chromeVisible ||
    emptyRoom.phase === "working" ||
    emptyRoom.message !== null ||
    restyle.pendingStyle !== null ||
    restyle.message !== null ||
    // Measuring is a mode, and a mode must never hide its own off switch. The panel stays put
    // regardless — it is the readout — but the bar carries the button that turns the mode off,
    // and three seconds of not touching anything is exactly what happens while somebody looks
    // at the number they just took.
    measure.active;

  /**
   * The floor rings, taken up while somebody is measuring.
   *
   * `handleTap` already refuses to travel in that state, so this is not what makes the tool
   * safe — it is what makes it legible. A ring is a bright disc lying on the floor, drawn in
   * the same accent as the measurement marks, in the middle of the surface being marked: it
   * both clutters the thing being measured and looks like a control that has stopped
   * responding. Better to not be there. They come straight back when the tool is switched off.
   */
  const floorRings = measure.active ? NO_PORTALS : portals;

  const initialView = currentScene?.initial_view ?? { yaw: 0, pitch: 0, fov: FOV_DEFAULT };
  // Keep the viewer inside what was photographed. Above and below the covered band the
  // canvas is a synthesised cap and the brand mark; nothing there is worth looking at, and
  // the cap's boundary is the one place the fill is visibly invented. A few degrees of
  // margin so the cornice or the skirting at the edge of the band is still reachable.
  const pitchLimits = useMemo(() => {
    const cov = currentScene?.coverage ?? null;
    const marginDeg = 6;
    const lo = cov && !cov.nadir_covered ? Math.max(-89, cov.pitch_min_deg - marginDeg) : -89;
    const hi = cov && !cov.zenith_covered ? Math.min(89, cov.pitch_max_deg + marginDeg) : 89;
    return { min: (lo * Math.PI) / 180, max: (hi * Math.PI) / 180 };
  }, [currentScene?.coverage]);

  return (
    <div
      ref={setContainer}
      tabIndex={0}
      role="application"
      aria-label={t("player.drag", locale)}
      className={cn(
        "no-tap relative isolate h-full w-full overflow-hidden bg-canvas",
        "focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-bio",
        // The one cue a mouse user gets that the ring is a control and not decoration.
        hoveredPortal ? "cursor-pointer" : "",
        className,
        // The phone headset needs the whole screen. Where element fullscreen exists it has it;
        // on an iPhone, which has none, the player covers the viewport itself. Last, so it wins
        // over whatever size the host gave the player.
        (vr.cardboard.phase === "starting" || vr.cardboard.phase === "active") &&
          !vr.cardboard.fullscreen &&
          "fixed inset-0 z-[2147483000] h-[100dvh] w-screen",
        vr.mode === "cardboard" && "touch-none",
      )}
    >
      {mounted && panorama ? (
        <Canvas
          // Colour fidelity: no tone mapping, sRGB out. Anything else grades a photograph
          // that was already graded by the camera.
          gl={{
            toneMapping: THREE.NoToneMapping,
            outputColorSpace: THREE.SRGBColorSpace,
            antialias: true,
            powerPreference: "high-performance",
            alpha: false,
          }}
          camera={{ fov: initialView.fov, near: 0.1, far: 1000, position: [0, 0, 0] }}
          // 3x phone panels cost 2.25x the fill rate of 2x for no visible gain here.
          dpr={[1, 2]}
          className="absolute inset-0"
        >
          <PlayerStage
            panorama={panorama}
            sceneKey={currentScene?.id ?? "none"}
            preloadUrls={preloadUrls}
            initialView={initialView}
            pitchLimits={pitchLimits}
            container={container}
            runtime={runtime}
            registry={registry}
            measureRegistry={measureRegistry}
            // Drawn in the EDITOR too, and that is the point of the editor: the agent is
            // placing a mark on a floor they cannot walk on, so the only way to know it landed
            // where they meant is to see it land. The tag stays as well — it is the handle you
            // drag — so a portal being edited shows both, which reads as one thing with a grip.
            portals={vr.mode === "off" ? floorRings : NO_PORTALS}
            readyScenes={readyScenes}
            onPortalHover={setHoveredPortal}
            transit={transitRef}
            compassRef={compassRef}
            headingRef={headingRef}
            planRotation={planRotation}
            controlsRef={controlsRef}
            autoRotate={autoRotate && vr.mode === "off"}
            enabled={vr.mode === "off"}
            headset={vr.mode === "xr"}
            content={contentRef}
            onStatus={setStatus}
            onTap={handleTap}
            onViewSettled={onViewChange}
            onRequestFullscreen={toggleFullscreen}
            onRequestReset={resetView}
          />
          {vr.mode !== "off" && currentScene ? (
            <Suspense fallback={null}>
              <VrPanoStage
                mode={vr.mode}
                content={contentRef}
                controls={controlsRef}
                sceneKey={currentScene.id}
                room={{
                  name: currentScene.name,
                  index: Math.max(0, playable.findIndex((scene) => scene.id === currentScene.id)),
                  count: playable.length,
                }}
                // The 2D rings step aside in VR (`NO_PORTALS` above): the headset draws its
                // own, at the panorama's depth, with a gaze dwell on them.
                portals={portals}
                readyScenes={readyScenes}
                sceneNames={sceneNames}
                tags={vrTags}
                dim={dim}
                dimDurationMs={dimDuration}
                copy={vrCopy}
                actions={vrActions}
                cardboard={
                  vr.mode === "cardboard"
                    ? { feed: vr.cardboard.feed, lens: vr.cardboard.lens, debug: vr.debug, selectRef: vr.cardboard.selectRef }
                    : null
                }
              />
            </Suspense>
          ) : null}
        </Canvas>
      ) : null}

      {/* The rooms, by name, for a property whose floor has no rings in it. See `RoomBar`. */}
      {vr.mode === "off" && showChrome ? (
        <RoomBar
          scenes={playable}
          currentSceneId={currentScene?.id ?? null}
          visible={chromeVisible}
          label={t("player.rooms", locale)}
          onSelect={(id) => goToScene(id)}
        />
      ) : null}

      {/* Where the ring under the pointer goes. `pointer-events-none` throughout: this must
          never be the thing the click lands on. */}
      {hoveredPortalName && vr.mode === "off" ? (
        <div
          // A ref callback rather than a layout effect: it runs at mount, before paint, and
          // costs no server-render warning.
          ref={(node) => {
            hoverPillRef.current = node;
            const at = pointerRef.current;
            if (node && at) placePill(at.x, at.y);
          }}
          className="pointer-events-none absolute top-0 left-0 z-30"
        >
          <span
            data-player-ui
            className="glass-strong absolute -translate-x-1/2 -translate-y-[calc(100%+18px)]
                       rounded-full px-2.5 py-1 text-[12px] font-medium whitespace-nowrap text-ink
                       shadow-[var(--shadow-lift)]"
          >
            {hoveredPortalName}
          </span>
        </div>
      ) : null}

      {/* Tags belong to the room you are standing in. Once the walk has ended in the dark
          there is no room to pin them to, and in the editor they were left floating over the
          black like buttons on nothing.

          TAKEN DOWN WHILE MEASURING, TOO, and for a plainer reason: a tag marker is a 26 px
          circle with `pointer-events: auto` sitting on the floor. Left up, every tap that
          landed on one would open a card instead of placing a mark — silently, since the mark
          the user was aiming at simply never appears — and in the editor a press on one would
          start a drag against the same gesture. Neither is fixable from here: `HotspotLayer`'s
          `editable` prop governs dragging, not hit-testing. Unmounting is a supported path
          (see the remount note in `HotspotLayer.register`) and it is the only one that makes
          the whole picture tappable, which is what a measure tool needs the picture to be. */}
      {vr.mode === "off" && !stranded && !measure.active ? (
        <HotspotLayer
          hotspots={tags}
          sceneNames={sceneNames}
          editable={editable}
          registry={registry}
          controls={controlsRef}
          onHotspotClick={onHotspotClick}
          onHotspotDragEnd={handleDragEnd}
          onNavigate={handleNavigate}
        />
      ) : null}

      {/* Doorway dim. Sits above the canvas and below the chrome so the toolbar stays
          legible through the transition instead of blinking with the room. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-10 bg-canvas transition-opacity ease-[var(--ease-out-expo)]"
        style={{ opacity: dim, transitionDuration: `${dimDuration}ms` }}
      />

      {showChrome && vr.mode === "off" ? (
        <PlayerChrome
          scenes={playable}
          currentSceneId={currentScene?.id ?? null}
          locale={locale}
          visible={chromeVisible}
          compassRef={compassRef}
          roomsOpen={roomsOpen}
          onRoomsOpenChange={setRoomsOpen}
          onSelectScene={(id) => goToScene(id)}
          quality={status}
          onForceQuality={forceQuality}
          walkAvailable={props.walkAvailable}
          onOpenWalk={props.onOpenWalk}
          fullscreen={fullscreen}
          onFullscreen={toggleFullscreen}
          onRecentre={recentre}
          embedded={embedded}
          showHint={showHint && chromeVisible}
        />
      ) : null}

      {/* THE BOTTOM-LEFT CORNER, ALL OF IT, IN ONE BAR. Two components used to anchor
          themselves here independently, which drew the plan pill and the measure pill at
          identical coordinates on any listing page that had a floor plan — see `PlayerDock`
          for that, and for why the fix is one row rather than a tidier stack.

          The AI tools are FIRST-PARTY ONLY: they are unauthenticated GPU compute behind the
          restricted CORS policy, so an embedded player is handed none and its bar starts at
          the plan. Measuring is a ray meeting the floor plane and has its own public route,
          so it is offered wherever there is a token for it. */}
      {vr.mode === "off" && currentScene ? (
        <PlayerDock
          locale={locale}
          visible={dockVisible}
          // GATED ON THE DRAWING, NOT ON THE CHROME. `showChrome` was in this condition too,
          // which quietly meant "no plan in the scene editor" — and the editor is the one place
          // where knowing which room is next door while looking at this one has a job to do,
          // since clicking a room on the plan is also how you get to its editor. What the plan
          // must stay gated on is `planAvailable`: a property nobody has measured has a button
          // that opens an empty box, which is worse than no button.
          plan={
            plan && planAvailable
              ? {
                  plan,
                  currentSceneId: currentScene.id,
                  open: planShowing,
                  onOpenChange: setPlanOpen,
                  onSelectRoom: goToScene,
                  headingRef,
                }
              : undefined
          }
          // OFFERED EVERYWHERE THE PLAYER IS, the editor included. It was gated on `showChrome`
          // for one turn on the theory that a room turning under a hotspot being placed is not
          // what an editor is for — which is true of the drift RUNNING and says nothing about
          // the switch being there. The editor is where an agent actually looks at the room
          // they just stitched, so gating it hid the control from the one surface that gets
          // used daily; it is off until pressed, and pressing it is the agent's business.
          autoRotate={{ value: autoRotate, onChange: setAutoRotate }}
          measure={
            tools || measureToken ? { active: measure.active, onToggle: measure.toggle } : undefined
          }
          // The embed token is what the assistant's public routes are keyed on; the dashboard's
          // player has none, so it shows no assistant (the agent has the panel on the property page).
          investmentToken={investmentSoon ? measureToken : null}
          vr={vr.route ? { active: vr.cardboard.phase !== "off", onClick: enterVr } : undefined}
          tools={
            tools
              ? {
                  emptyRoom: {
                    active: emptyRoom.active,
                    phase: emptyRoom.phase,
                    removedFraction: emptyRoom.result?.removed_fraction ?? null,
                    modelBacked: emptyRoom.result?.model_backed ?? false,
                    message: emptyRoom.message,
                    onToggle: toggleEmptyRoom,
                  },
                  decor:
                    decorStyles.length > 0
                      ? {
                          styles: decorStyles,
                          activeStyle: restyle.activeStyle,
                          pendingStyle: restyle.pendingStyle,
                          phase: restyle.phase,
                          message: restyle.message,
                          onSelect: selectStyle,
                        }
                      : undefined,
                }
              : undefined
          }
        />
      ) : null}

      {/* The marks, the lines and the panel. Gated on `tools` with the rest of the rail, and
          for the same reason `PlayerProps.tools` gives: `/ai/measure` sits behind the engine's
          restricted CORS policy, so a player framed on a third-party listing portal could not
          call it even with the button on screen. Mounted after the doorway dim so that its own
          stacking works out — see the note on the overlay in `MeasureTool`. */}
      {(tools || measureToken) && vr.mode === "off" && currentScene ? (
        <MeasureTool locale={locale} state={measure} registry={measureRegistry} />
      ) : null}

      {vr.cardboard.phase !== "off" || vr.handoff || vr.prompt ? (
        <Suspense fallback={null}>
          <VrOverlays
            locale={locale}
            kind="pano"
            cardboard={vr.cardboard}
            handoff={vr.handoff ? { url: vr.handoff, onClose: vr.closeHandoff } : null}
            prompt={vr.prompt ? { onStart: enterVr, onDismiss: vr.dismissPrompt } : null}
          />
        </Suspense>
      ) : null}

      {/* Arrived, and the room is dark. Above the doorway veil, below the toolbar, which stays
          reachable so this is never the only way out. */}
      {stranded && vr.mode === "off" ? (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-black px-6 text-center">
          <p className="text-[15px] font-medium text-white">{stranded.name}</p>
          <p className="max-w-[26rem] text-[13px] text-white/55">
            {t("player.portal.pending", locale)}
          </p>
          <button
            type="button"
            data-player-ui
            onClick={leaveTheDark}
            className="glass-strong mt-1 rounded-full px-4 py-2 text-[12px] font-medium text-ink
                       transition-transform hover:scale-[1.03] active:scale-[0.98]"
          >
            {t("player.portal.back", locale)}
          </button>
        </div>
      ) : null}

      {vr.message ? (
        <div className="pointer-events-none absolute inset-x-0 top-4 z-40 flex justify-center px-4">
          <span className="glass-strong animate-rise rounded-full px-3.5 py-2 text-[12px] text-ink">{vr.message}</span>
        </div>
      ) : null}

      {panorama ? (
        <LoadingVeil visible={!everPainted} progress={status.progress} label={t("player.loading", locale)} />
      ) : (
        <div className="absolute inset-0 z-30 grid place-items-center bg-canvas px-6 text-center">
          <div className="flex flex-col items-center gap-2">
            <span className="animate-breathe bio-gradient h-1.5 w-10 rounded-full" />
            <p className="text-[13px] tracking-tight text-ink-muted">{t("player.empty", locale)}</p>
          </div>
        </div>
      )}
    </div>
  );
});

export type { PlayerHandle, PlayerProps } from "./types";
