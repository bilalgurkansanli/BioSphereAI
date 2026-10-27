"use client";

/**
 * The PlayCanvas side of the 3D walkthrough: one engine instance, the splat areas, the
 * camera and everything that moves it.
 *
 * React owns the chrome; this hook owns the frame loop. The two meet through a small
 * imperative API (`WalkEngineApi`) and a handful of state callbacks that fire rarely
 * (which area is active, how much has loaded), never per frame.
 *
 * COORDINATES. The manifest is in the engine frame (+Y down); PlayCanvas is +Y up. Every
 * splat entity is rotated 180 degrees about X and every position that crosses the boundary
 * goes through `toViewer` / `toEngine` in `walkMath.ts`. Inside this file everything is the
 * viewer frame.
 *
 * STREAMING. The base area (the general walkthrough) is loaded first and shown wherever no
 * room is loaded; the start room next; the rest in the background one by one. Which area is
 * drawn is decided per frame by where the camera stands (`areaAt`), the same bbox swap
 * SplatTour's viewer does.
 */

import { useEffect, useRef } from "react";
import type * as PC from "playcanvas";
import { absoluteUrl } from "@/lib/api";
import type { WalkArea, WalkManifest } from "@/lib/types";
import {
  AREA_OVERLAP_S,
  EYE_HEIGHT,
  areaAt,
  clampToBox,
  constrainToCorridor,
  corridorOf,
  defaultFov,
  drawableAreas,
  SCALE_MAX,
  entryPose,
  forwardOf,
  lookAlong,
  nextRenderScale,
  pathLength,
  pathOf,
  splinePoint,
  tourOrder,
  union,
  updateAreaSwitch,
  viewerBox,
  yawDelta,
  type Box,
  type Corridor,
  type Vec3,
} from "./walkMath";
import { walkDeviceTypes, type WalkVrHost } from "@/components/vr/walk/host";

export type WalkMode = "walk" | "drone";

export interface WalkEngineApi {
  goToArea(key: string): void;
  teleport(position: Vec3, yaw?: number): void;
  walkTo(position: Vec3): void;
  setMode(mode: WalkMode): void;
  setAutoTour(on: boolean): void;
  pickAndWalk(clientX: number, clientY: number): Promise<boolean>;
  resetFov(): void;
  screenshot(): string | null;
  getPose(): { position: Vec3; yaw: number; pitch: number };
  /** The headset and phone-headset modes' handle on the engine (`components/vr/walk`). */
  vr: WalkVrHost;
}

export interface WalkEngineEvents {
  onProgress(loaded: number, total: number, ready: boolean): void;
  onArea(key: string): void;
  onError(message: string): void;
  onAutoTourEnd(): void;
  /** Called every frame with the camera position, for the plan marker (no React state!). */
  onFrame?(position: Vec3, yaw: number): void;
  onUserInput?(): void;
}

export interface UseWalkEngineOptions {
  manifest: WalkManifest;
  startKey: string | null;
  container: React.RefObject<HTMLDivElement | null>;
  canvas: React.RefObject<HTMLCanvasElement | null>;
  fade: React.RefObject<HTMLDivElement | null>;
  events: WalkEngineEvents;
  enabled: boolean;
}

const FOV_DEFAULT = 70;
/** How brightly the environment panorama shows through the splats. See the sphere below. */
const ENV_INTENSITY = 0.35;
const FOV_MIN = 30;
const FOV_MAX = 95;
const WALK_SPEED = 1.7; // m/s, "guided tour pace"
const RUN_SPEED = 3.2;
const DRONE_SPEED = 2.6;
const LOOK_SENS = 0.0042; // radians per pixel
const PITCH_LIMIT = Math.PI / 2 - 0.05;
const TOUR_SPEED = 0.75; // m/s along the spline
const FADE_MS = 380;

interface AreaSlot {
  area: WalkArea;
  asset: PC.Asset | null;
  entity: PC.Entity | null;
  state: "idle" | "loading" | "ready" | "failed";
  loaded: number;
}

/** Ease-out cubic. */
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

export function useWalkEngine(opts: UseWalkEngineOptions): React.RefObject<WalkEngineApi | null> {
  const api = useRef<WalkEngineApi | null>(null);
  // The callbacks, kept current for the engine loop without re-creating it. Written after
  // render rather than during it: the engine only reads them from its own callbacks.
  const eventsRef = useRef(opts.events);
  useEffect(() => {
    eventsRef.current = opts.events;
  });
  const { manifest, startKey, container, canvas, fade, enabled } = opts;

  useEffect(() => {
    if (!enabled) return;
    const canvasEl = canvas.current;
    const containerEl = container.current;
    if (!canvasEl || !containerEl) return;
    let disposed = false;
    let app: PC.Application | null = null;
    const cleanups: (() => void)[] = [];

    (async () => {
      let pc: typeof PC;
      try {
        pc = await import("playcanvas");
      } catch (err) {
        eventsRef.current.onError(`engine import failed: ${String(err)}`);
        return;
      }
      if (disposed) return;

      // ---------------------------------------------------------------- engine ----
      // WebGPU first, WebGL2 as the fallback. It is not a preference but the difference
      // between two renderers: on WebGPU the engine sorts splats on the GPU and can cull by
      // contribution (opacity times projected area), which is exactly the overdraw a splat
      // scene drowns in; on WebGL it sorts on the CPU and draws everything. The device must
      // be created BEFORE anything asks the canvas for a context - a canvas that has handed
      // out a WebGL context can never give a WebGPU one - so there is no probe here.
      const deviceOptions = { antialias: false, alpha: false, powerPreference: "high-performance" as const };
      let device: PC.GraphicsDevice | null = null;
      try {
        // WebGPU first — except where VR can present, which PlayCanvas only does on WebGL2
        // in today's headset browsers. See `walkDeviceTypes`.
        device = await pc.createGraphicsDevice(canvasEl, { deviceTypes: await walkDeviceTypes(), ...deviceOptions });
      } catch {
        device = null;
      }
      if (disposed) {
        device?.destroy?.();
        return;
      }
      if (!device) {
        eventsRef.current.onError("webgl2");
        return;
      }
      app = new pc.Application(canvasEl, { graphicsDevice: device });
      const a = app;
      a.setCanvasFillMode(pc.FILLMODE_NONE);
      a.setCanvasResolution(pc.RESOLUTION_AUTO);
      const dpr = window.devicePixelRatio || 1;
      const scaleCeiling = Math.min(dpr, SCALE_MAX);  // see nextRenderScale
      let renderScale = scaleCeiling;
      a.graphicsDevice.maxPixelRatio = renderScale;
      a.scene.gsplat.enableIds = true;
      // The engine's own culling defaults are left alone. Raising them looked like free
      // performance and is not: a Brush model of a real room has a MEDIAN splat opacity of
      // 0.046, so an 0.02 forward alpha clip cuts away most of the footprint of half the
      // splats, and a 3 px minimum removes the fine detail at the low render scales this
      // viewer falls back to. Pixels (the render scale) and contribution culling below are
      // the levers that cost nothing visible.
      // Only the GPU-sort renderer reads these; on WebGL they are ignored rather than wrong.
      // Culling hardest at the edge of the eye's attention buys the most frames for the least
      // noticed detail, which is the trade a moving viewer wants.
      a.scene.gsplat.foveationStrength = 2;
      a.scene.ambientLight = new pc.Color(0.15, 0.15, 0.15);

      const camera = new pc.Entity("camera");
      camera.addComponent("camera", {
        fov: FOV_DEFAULT,
        nearClip: 0.04,
        farClip: 400,
        clearColor: new pc.Color(0.02, 0.027, 0.039),
        toneMapping: pc.TONEMAP_NONE,
      });
      a.root.addChild(camera);

      const resize = () => {
        const w = Math.max(1, Math.floor(containerEl.clientWidth));
        const h = Math.max(1, Math.floor(containerEl.clientHeight));
        a.graphicsDevice.maxPixelRatio = renderScale;
        a.resizeCanvas(w, h);
      };
      resize();
      const ro = new ResizeObserver(resize);
      ro.observe(containerEl);
      cleanups.push(() => ro.disconnect());

      // ---------------------------------------------------------------- state ----
      const perf = { ms: 0, since: 0, elapsed: 0, report: null as unknown };
      const areas = drawableAreas(manifest);
      const slots = new Map<string, AreaSlot>();
      for (const area of areas) slots.set(area.key, { area, asset: null, entity: null, state: "idle", loaded: 0 });
      const placedBoxes: Box[] = areas.filter((x) => x.placed).map(viewerBox);
      const worldBox: Box = union(placedBoxes) ?? { min: [-20, -1, -20], max: [20, 4, 20] };
      // Walk mode stays within a step of where the phone actually stood (see PATH_RADIUS_M):
      // every placed area's path for the shared frame, an unplaced area's own path in its own.
      const placedCorridor: Corridor = corridorOf(areas.filter((x) => x.placed));
      const totalBytes = Math.max(
        areas.reduce((n, x) => n + (x.bytes || 0), 0),
        1,
      );

      // The opening field of view follows the canvas shape (see HFOV_OPEN_MAX) until the
      // visitor zooms; after that it is theirs.
      let fovTouched = false;
      const openFov = () => defaultFov(containerEl.clientWidth / Math.max(containerEl.clientHeight, 1), FOV_DEFAULT);
      const pose = { position: [0, EYE_HEIGHT, 0] as Vec3, yaw: 0, pitch: 0, fov: openFov() };
      const fovObserver = new ResizeObserver(() => {
        if (!fovTouched) pose.fov = openFov();
      });
      fovObserver.observe(containerEl);
      cleanups.push(() => fovObserver.disconnect());
      let mode: WalkMode = "walk";
      let activeKey: string | null = null;
      const switchState = { pending: null as string | null, held: 0 };
      let fading: { key: string; left: number } | null = null;
      let currentBox: Box = worldBox;
      let currentCorridor: Corridor = placedCorridor;
      /** An unplaced area the visitor opened by name (or the tour is in): see chooseArea. */
      let pinnedKey: string | null = null;
      const keys = new Set<string>();
      const look = { vx: 0, vy: 0 }; // inertia, rad/s
      let dragging = false;
      let pointerId: number | null = null;
      let last = { x: 0, y: 0, t: 0 };
      const touches = new Map<number, { x: number; y: number }>();
      let pinchDist = 0;
      // `elapsed` is advanced by the frame step, never read off the clock: a walk-to that
      // measured wall time would fast-forward through a hitch and put the visitor metres
      // away in one frame, which is precisely what being teleported looks like. Driven by
      // the step, a hitch stretches the walk instead of skipping part of it.
      let walkAnim: { from: Vec3; to: Vec3; elapsed: number; ms: number } | null = null;
      let tour: { keys: string[]; index: number; path: Vec3[]; length: number; t: number; yawBias: number; area: WalkArea | null } | null = null;
      let lastTap = 0;

      const readyOverall = () => (slots.get(manifest.base.key)?.state === "ready") || [...slots.values()].some((s) => s.state === "ready");
      const reportProgress = () => {
        let loaded = 0;
        for (const s of slots.values()) loaded += s.state === "ready" ? s.area.bytes : Math.min(s.loaded, s.area.bytes || s.loaded);
        eventsRef.current.onProgress(loaded, totalBytes, readyOverall());
      };

      // ---------------------------------------------------------------- environment ----
      // A big inverted sphere with the area's environment panorama, centred on the camera:
      // the background wherever the splat has nothing (windows, the doorway behind you).
      //
      // IT MUST BE FAINT. A splat cloud is not opaque - a fifth to a quarter of a typical
      // frame is see-through - so whatever is on this sphere is blended into every pixel of
      // the scene, not only into the holes. At full strength the panorama (which is a blurry
      // 1024-px render of the same room) washed the walkthrough into an unreadable smear:
      // the first real capture looked broken for exactly this reason, while the same model
      // rendered offline looked like a flat. Dim, it reads as depth behind a window and
      // disappears everywhere else.
      const envEntity = new pc.Entity("env");
      const envMaterial = new pc.StandardMaterial();
      envMaterial.useLighting = false;
      envMaterial.useSkybox = false;
      envMaterial.emissive = new pc.Color(0.03, 0.035, 0.045);
      envMaterial.diffuse = new pc.Color(0, 0, 0);
      envMaterial.cull = pc.CULLFACE_FRONT;
      envMaterial.depthWrite = false;
      envMaterial.depthTest = false;
      envMaterial.update();
      // On the SKYBOX layer, which the engine draws after opaque geometry and before the
      // blended pass the splats live in. On the default world layer this sphere painted
      // OVER the whole walkthrough - an opaque mesh with no depth to sort against wins the
      // pixel - and every delivered tour looked like a blurry panorama instead of a room.
      envEntity.addComponent("render", { type: "sphere", material: envMaterial, layers: [pc.LAYERID_SKYBOX] });
      envEntity.setLocalScale(160, 160, 160);
      // The equirect was rendered in the engine frame (+Y down); the same half turn about X
      // that the splats get puts its horizon and its zenith where they belong.
      envEntity.setEulerAngles(180, 0, 0);
      a.root.addChild(envEntity);
      const envTextures = new Map<string, PC.Asset>();
      const applyEnv = (area: WalkArea) => {
        const url = absoluteUrl(area.env_pano_url);
        if (!url) return;
        let asset = envTextures.get(area.key);
        if (!asset) {
          asset = new pc.Asset(`env-${area.key}`, "texture", { url });
          a.assets.add(asset);
          envTextures.set(area.key, asset);
          asset.ready((loaded) => {
            const tex = loaded.resource as PC.Texture;
            tex.addressU = pc.ADDRESS_REPEAT;
            envMaterial.emissiveMap = tex;
            envMaterial.emissive = new pc.Color(1, 1, 1);
            envMaterial.emissiveIntensity = ENV_INTENSITY;
            envMaterial.update();
          });
          a.assets.load(asset);
        } else if (asset.resource) {
          envMaterial.emissiveMap = asset.resource as PC.Texture;
          envMaterial.emissive = new pc.Color(1, 1, 1);
          envMaterial.emissiveIntensity = ENV_INTENSITY;
          envMaterial.update();
        }
      };

      // ---------------------------------------------------------------- areas ----
      const loadArea = (key: string): Promise<void> =>
        new Promise((resolve) => {
          const slot = slots.get(key);
          if (!slot || slot.state !== "idle" || !slot.area.url) return resolve();
          slot.state = "loading";
          const asset = new pc.Asset(key, "gsplat", { url: absoluteUrl(slot.area.url) as string });
          slot.asset = asset;
          asset.on("progress", (received: number, total: number) => {
            slot.loaded = received;
            if (total && !slot.area.bytes) slot.area.bytes = total;
            reportProgress();
          });
          asset.once("load", () => {
            if (disposed) return resolve();
            const entity = new pc.Entity(key);
            entity.addComponent("gsplat", { asset });
            entity.setEulerAngles(180, 0, 0);
            entity.enabled = false;
            a.root.addChild(entity);
            slot.entity = entity;
            slot.state = "ready";
            slot.loaded = slot.area.bytes;
            reportProgress();
            resolve();
          });
          asset.once("error", (err: unknown) => {
            slot.state = "failed";
            eventsRef.current.onError(`${key}: ${String(err)}`);
            reportProgress();
            resolve();
          });
          a.assets.add(asset);
          a.assets.load(asset);
        });

      /**
       * Draw `key` instead of whatever is drawn now.
       *
       * The area being left keeps drawing for a moment unless `immediate`. Every placed area
       * stands in the same coordinate frame, so for that moment the visitor sees both
       * reconstructions of the same room laid over each other and then one of them stops -
       * a dissolve, where a bare swap is the whole room changing between two frames. A
       * teleport passes `immediate`: it already has a fade to black over it, and there the
       * two clouds are of different rooms.
       */
      const setActive = (key: string, immediate = false) => {
        if (key === activeKey) return;
        const leaving = activeKey;
        activeKey = key;
        for (const s of slots.values()) if (s.entity) s.entity.enabled = s.area.key === key;
        fading = null;
        if (!immediate && leaving && slots.get(leaving)?.entity) {
          slots.get(leaving)!.entity!.enabled = true;
          fading = { key: leaving, left: AREA_OVERLAP_S };
        }
        const area = slots.get(key)?.area;
        if (area) {
          applyEnv(area);
          currentBox = area.placed && area.key !== manifest.base.key ? worldBox : area.placed ? worldBox : viewerBox(area);
          currentCorridor = area.placed ? placedCorridor : corridorOf([area]);
          eventsRef.current.onArea(key);
        }
      };

      /**
       * The area to draw for the camera's position, restricted to what has loaded.
       *
       * `sticky` keeps the area in use unless another is clearly nearer; it is off after a
       * teleport, where the area in use says nothing about where the visitor now stands.
       */
      const chooseArea = (sticky = true) => {
        // An unplaced area stands in its own frame, so position can never choose it (areaAt
        // leaves it out on purpose). Opened by name it is pinned, or opening it would put the
        // visitor at its coordinates inside the base model instead.
        if (pinnedKey) {
          const pinned = slots.get(pinnedKey);
          if (pinned && pinned.state === "ready") return pinnedKey;
        }
        const want = areaAt(manifest, pose.position, sticky ? activeKey : null);
        const slot = slots.get(want.key);
        if (slot && slot.state === "ready") return want.key;
        // Fall back to the base while the room streams in; to whatever is loaded otherwise.
        const base = slots.get(manifest.base.key);
        if (base && base.state === "ready") return manifest.base.key;
        const any = [...slots.values()].find((s) => s.state === "ready");
        return any ? any.area.key : null;
      };

      // ---------------------------------------------------------------- camera ----
      const applyPose = () => {
        camera.setPosition(pose.position[0], pose.position[1], pose.position[2]);
        camera.setEulerAngles((pose.pitch * 180) / Math.PI, (pose.yaw * 180) / Math.PI, 0);
        if (camera.camera && Math.abs(camera.camera.fov - pose.fov) > 0.01) camera.camera.fov = pose.fov;
        envEntity.setPosition(pose.position[0], pose.position[1], pose.position[2]);
      };

      const fadeTo = (opacity: number) => {
        const el = fade.current;
        if (!el) return;
        el.style.transition = `opacity ${FADE_MS}ms ease`;
        el.style.opacity = String(opacity);
      };

      const teleport = (position: Vec3, yaw?: number, withFade = true) => {
        const go = () => {
          const target: Vec3 = [position[0], mode === "walk" ? EYE_HEIGHT : position[1], position[2]];
          pose.position = mode === "walk" ? constrainToCorridor(target, currentCorridor) : target;
          if (yaw !== undefined) {
            pose.yaw = yaw;
            pose.pitch = 0;
          }
          look.vx = 0;
          look.vy = 0;
          walkAnim = null;
          applyPose();
          switchState.pending = null;
          switchState.held = 0;
          const key = chooseArea(false);
          if (key) setActive(key, true);
        };
        if (!withFade) return go();
        fadeTo(1);
        window.setTimeout(() => {
          go();
          window.setTimeout(() => fadeTo(0), 60);
        }, FADE_MS);
      };

      const walkTo = (target: Vec3) => {
        const to = constrainToCorridor(clampToBox([target[0], EYE_HEIGHT, target[2]], currentBox), currentCorridor);
        walkAnim = { from: [...pose.position] as Vec3, to, elapsed: 0, ms: 700 + 250 * Math.hypot(to[0] - pose.position[0], to[2] - pose.position[2]) };
      };

      // ---------------------------------------------------------------- picking ----
      const picker = new pc.Picker(a, 1, 1, true);
      const pickAndWalk = async (clientX: number, clientY: number): Promise<boolean> => {
        if (!camera.camera) return false;
        const rect = canvasEl.getBoundingClientRect();
        const scale = 0.25;
        const w = Math.max(1, Math.floor(rect.width * scale));
        const h = Math.max(1, Math.floor(rect.height * scale));
        picker.resize(w, h);
        try {
          picker.prepare(camera.camera, a.scene);
          const x = (clientX - rect.left) * scale;
          const y = (clientY - rect.top) * scale;
          const hit = await picker.getWorldPointAsync(x, y);
          if (!hit) return false;
          // Stand short of the surface that was hit, not on it: a wall is a place to look
          // at, the floor in front of it is the place to stand.
          const dx = hit.x - pose.position[0];
          const dz = hit.z - pose.position[2];
          const dist = Math.hypot(dx, dz);
          const back = Math.min(0.7, dist * 0.35);
          const k = dist > 1e-6 ? (dist - back) / dist : 0;
          walkTo([pose.position[0] + dx * k, EYE_HEIGHT, pose.position[2] + dz * k]);
          return true;
        } catch {
          return false;
        }
      };

      // ---------------------------------------------------------------- auto tour ----
      const startTour = () => {
        // The walk replayed as it was filmed, part by part (see tourOrder), then the rooms.
        tour = { keys: tourOrder(manifest), index: -1, path: [], length: 0, t: 0, yawBias: 0, area: null };
        nextTourLeg();
      };
      const nextTourLeg = () => {
        if (!tour) return;
        tour.index += 1;
        if (tour.index >= tour.keys.length) {
          tour = null;
          eventsRef.current.onAutoTourEnd();
          return;
        }
        const area = slots.get(tour.keys[tour.index])?.area ?? manifest.base;
        const path = pathOf(area);
        tour.path = path.length >= 2 ? path : [entryPose(area).position, entryPose(area).position];
        tour.length = Math.max(pathLength(tour.path), 0.5);
        tour.t = 0;
        tour.yawBias = 0;
        tour.area = area;
        pinnedKey = area.placed ? null : area.key;
        currentBox = area.placed ? worldBox : viewerBox(area);
        currentCorridor = area.placed ? placedCorridor : corridorOf([area]);
        const start = splinePoint(tour.path, 0);
        const ahead = splinePoint(tour.path, 0.02);
        teleport(start, lookAlong(area, 0)?.yaw ?? Math.atan2(-(ahead[0] - start[0]), -(ahead[2] - start[2])));
        void loadArea(area.key);
      };

      // ---------------------------------------------------------------- input ----
      const isUi = (target: EventTarget | null) => !!(target as HTMLElement | null)?.closest?.("[data-walk-ui]");
      const userInput = () => {
        if (tour) {
          tour = null;
          eventsRef.current.onAutoTourEnd();
        }
        eventsRef.current.onUserInput?.();
      };

      const onPointerDown = (e: PointerEvent) => {
        if (isUi(e.target)) return;
        if (e.pointerType === "touch") {
          touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
          if (touches.size === 2) {
            const [p, q] = [...touches.values()];
            pinchDist = Math.hypot(p.x - q.x, p.y - q.y);
          }
          const now = performance.now();
          if (touches.size === 1 && now - lastTap < 320) {
            void pickAndWalk(e.clientX, e.clientY);
            lastTap = 0;
          } else {
            lastTap = now;
          }
          userInput();
          return;
        }
        if (e.button !== 0) return;
        dragging = true;
        pointerId = e.pointerId;
        last = { x: e.clientX, y: e.clientY, t: performance.now() };
        look.vx = 0;
        look.vy = 0;
        containerEl.setPointerCapture?.(e.pointerId);
        userInput();
      };
      const onPointerMove = (e: PointerEvent) => {
        if (e.pointerType === "touch") {
          const prev = touches.get(e.pointerId);
          if (!prev) return;
          const cur = { x: e.clientX, y: e.clientY };
          touches.set(e.pointerId, cur);
          if (touches.size === 1) {
            pose.yaw -= (cur.x - prev.x) * LOOK_SENS * 1.4;
            pose.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, pose.pitch - (cur.y - prev.y) * LOOK_SENS * 1.4));
          } else if (touches.size === 2) {
            // Two fingers: drag to walk, pinch to zoom.
            const [p, q] = [...touches.values()];
            const d = Math.hypot(p.x - q.x, p.y - q.y);
            if (pinchDist > 0) {
              pose.fov = Math.max(FOV_MIN, Math.min(FOV_MAX, pose.fov - (d - pinchDist) * 0.08));
              fovTouched = true;
            }
            pinchDist = d;
            const f = forwardOf(pose.yaw);
            const right: Vec3 = [-f[2], 0, f[0]];
            const dy = (cur.y - prev.y) * 0.012;
            const dx = (cur.x - prev.x) * 0.012;
            moveBy([-(f[0] * dy) + right[0] * -dx, 0, -(f[2] * dy) + right[2] * -dx]);
          }
          return;
        }
        if (!dragging || e.pointerId !== pointerId) return;
        const now = performance.now();
        const dx = e.clientX - last.x;
        const dy = e.clientY - last.y;
        const dt = Math.max((now - last.t) / 1000, 1e-3);
        pose.yaw -= dx * LOOK_SENS;
        pose.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, pose.pitch - dy * LOOK_SENS));
        look.vx = (-dx * LOOK_SENS) / dt;
        look.vy = (-dy * LOOK_SENS) / dt;
        last = { x: e.clientX, y: e.clientY, t: now };
      };
      const onPointerUp = (e: PointerEvent) => {
        if (e.pointerType === "touch") {
          touches.delete(e.pointerId);
          pinchDist = 0;
          return;
        }
        if (e.pointerId !== pointerId) return;
        dragging = false;
        pointerId = null;
        if (performance.now() - last.t > 60) {
          look.vx = 0;
          look.vy = 0;
        }
      };
      const onDblClick = (e: MouseEvent) => {
        if (isUi(e.target)) return;
        userInput();
        void pickAndWalk(e.clientX, e.clientY);
      };
      const onWheel = (e: WheelEvent) => {
        if (isUi(e.target)) return;
        e.preventDefault();
        pose.fov = Math.max(FOV_MIN, Math.min(FOV_MAX, pose.fov + Math.sign(e.deltaY) * 3));
        fovTouched = true;
        userInput();
      };
      const onKeyDown = (e: KeyboardEvent) => {
        if (isUi(e.target) && (e.target as HTMLElement).tagName === "INPUT") return;
        const k = e.key.toLowerCase();
        if (["w", "a", "s", "d", "q", "e", "shift", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(k)) {
          keys.add(k);
          e.preventDefault();
          userInput();
        }
      };
      const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
      const onBlur = () => keys.clear();
      containerEl.addEventListener("pointerdown", onPointerDown);
      containerEl.addEventListener("pointermove", onPointerMove);
      containerEl.addEventListener("pointerup", onPointerUp);
      containerEl.addEventListener("pointercancel", onPointerUp);
      containerEl.addEventListener("dblclick", onDblClick);
      containerEl.addEventListener("wheel", onWheel, { passive: false });
      containerEl.addEventListener("keydown", onKeyDown);
      containerEl.addEventListener("keyup", onKeyUp);
      window.addEventListener("blur", onBlur);
      cleanups.push(() => {
        containerEl.removeEventListener("pointerdown", onPointerDown);
        containerEl.removeEventListener("pointermove", onPointerMove);
        containerEl.removeEventListener("pointerup", onPointerUp);
        containerEl.removeEventListener("pointercancel", onPointerUp);
        containerEl.removeEventListener("dblclick", onDblClick);
        containerEl.removeEventListener("wheel", onWheel);
        containerEl.removeEventListener("keydown", onKeyDown);
        containerEl.removeEventListener("keyup", onKeyUp);
        window.removeEventListener("blur", onBlur);
      });

      const moveBy = (delta: Vec3) => {
        const next: Vec3 = [pose.position[0] + delta[0], pose.position[1] + delta[1], pose.position[2] + delta[2]];
        if (mode === "walk") next[1] = EYE_HEIGHT;
        const box = mode === "walk" ? currentBox : { min: [currentBox.min[0] - 2, -1, currentBox.min[2] - 2] as Vec3, max: [currentBox.max[0] + 2, currentBox.max[1] + 6, currentBox.max[2] + 2] as Vec3 };
        const clamped = clampToBox(next, box, mode === "walk" ? 0.25 : 0);
        pose.position = mode === "walk" ? constrainToCorridor(clamped, currentCorridor) : clamped;
      };

      // ---------------------------------------------------------------- VR ----
      // What the headset modes need from in here; while one runs (`driven`) the loop below
      // leaves the camera to it and only follows the visitor. See `components/vr/walk/host.ts`.
      const vrHost: WalkVrHost = {
        pc,
        app: a,
        camera,
        env: envEntity,
        pose,
        manifest,
        driven: false,
        activeKey: () => activeKey,
        relocated: () => {
          switchState.pending = null;
          switchState.held = 0;
          const key = chooseArea(false);
          if (key) setActive(key, true);
        },
        stopTour: () => {
          if (!tour) return;
          tour = null;
          eventsRef.current.onAutoTourEnd();
        },
        fade: fadeTo,
      };

      // ---------------------------------------------------------------- frame loop ----
      a.on("update", (dt: number) => {
        if (disposed) return;
        if (vrHost.driven) {
          // A headset owns the camera: no input, walk-to, tour or render-scale steering here.
          // Streaming and the area swap still follow the head, which the VR mode writes to `pose`.
          const swapVr = updateAreaSwitch(switchState, chooseArea(), activeKey, dt);
          if (swapVr) setActive(swapVr);
          if (fading) {
            fading.left -= dt;
            if (fading.left <= 0) {
              const left = slots.get(fading.key)?.entity;
              if (left && fading.key !== activeKey) left.enabled = false;
              fading = null;
            }
          }
          eventsRef.current.onFrame?.(pose.position, pose.yaw);
          return;
        }
        const step = Math.min(dt, 0.05);
        // Inertia after a flick.
        if (!dragging && (Math.abs(look.vx) > 1e-3 || Math.abs(look.vy) > 1e-3)) {
          pose.yaw += look.vx * step;
          pose.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, pose.pitch + look.vy * step));
          const decay = Math.exp(-6 * step);
          look.vx *= decay;
          look.vy *= decay;
        }
        // Keys.
        if (keys.size) {
          const run = keys.has("shift");
          const speed = (mode === "walk" ? (run ? RUN_SPEED : WALK_SPEED) : DRONE_SPEED * (run ? 1.8 : 1)) * step;
          const f = forwardOf(pose.yaw);
          const fly: Vec3 = mode === "drone" ? [f[0] * Math.cos(pose.pitch), Math.sin(pose.pitch), f[2] * Math.cos(pose.pitch)] : f;
          const right: Vec3 = [-f[2], 0, f[0]];
          const d: Vec3 = [0, 0, 0];
          if (keys.has("w") || keys.has("arrowup")) for (let i = 0; i < 3; i += 1) d[i] += fly[i];
          if (keys.has("s") || keys.has("arrowdown")) for (let i = 0; i < 3; i += 1) d[i] -= fly[i];
          if (keys.has("d") || keys.has("arrowright")) for (let i = 0; i < 3; i += 1) d[i] += right[i];
          if (keys.has("a") || keys.has("arrowleft")) for (let i = 0; i < 3; i += 1) d[i] -= right[i];
          if (mode === "drone") {
            if (keys.has("e")) d[1] += 1;
            if (keys.has("q")) d[1] -= 1;
          }
          const n = Math.hypot(d[0], d[1], d[2]);
          if (n > 1e-6) moveBy([(d[0] / n) * speed, (d[1] / n) * speed, (d[2] / n) * speed]);
          walkAnim = null;
        }
        // Walk-to animation.
        if (walkAnim) {
          walkAnim.elapsed += step * 1000;
          const u = Math.min(1, walkAnim.elapsed / walkAnim.ms);
          const k = easeOut(u);
          pose.position = [
            walkAnim.from[0] + (walkAnim.to[0] - walkAnim.from[0]) * k,
            EYE_HEIGHT,
            walkAnim.from[2] + (walkAnim.to[2] - walkAnim.from[2]) * k,
          ];
          if (u >= 1) walkAnim = null;
        }
        // Auto tour: along the room's own camera path, looking a little ahead.
        if (tour) {
          tour.t += (TOUR_SPEED * step) / tour.length;
          if (tour.t >= 1) {
            nextTourLeg();
          } else {
            const p = splinePoint(tour.path, tour.t);
            const ahead = splinePoint(tour.path, Math.min(1, tour.t + 0.04));
            pose.position = [p[0], EYE_HEIGHT, p[2]];
            // Look where the phone looked (see lookAlong), eased so the hand's shake and the
            // thinned path's corners become a glide; older manifests have no looks and keep
            // the old gaze down the path with a slow sway.
            const film = tour.area ? lookAlong(tour.area, tour.t) : null;
            if (film) {
              pose.yaw += yawDelta(pose.yaw, film.yaw) * Math.min(1, step * 1.6);
              pose.pitch += (Math.max(-0.6, Math.min(0.6, film.pitch)) - pose.pitch) * Math.min(1, step * 1.6);
            } else {
              const dx = ahead[0] - p[0];
              const dz = ahead[2] - p[2];
              if (Math.hypot(dx, dz) > 1e-4) {
                tour.yawBias += step * 0.9;
                const sway = 0.28 * Math.sin(tour.yawBias);
                const target = Math.atan2(-dx, -dz) + sway;
                pose.yaw += yawDelta(pose.yaw, target) * Math.min(1, step * 2.2);
              }
              pose.pitch += (0 - pose.pitch) * Math.min(1, step * 2);
            }
          }
        }
        applyPose();
        const swap = updateAreaSwitch(switchState, chooseArea(), activeKey, dt);
        if (swap) setActive(swap);
        if (fading) {
          fading.left -= dt;
          if (fading.left <= 0) {
            const left = slots.get(fading.key)?.entity;
            if (left && fading.key !== activeKey) left.enabled = false;
            fading = null;
          }
        }
        eventsRef.current.onFrame?.(pose.position, pose.yaw);

        // ---- hold the frame rate ----
        // The first second is loading, not rendering, and would drive the scale to the floor.
        perf.elapsed += dt;
        if (perf.elapsed > 1) {
          const ms = Math.min(dt * 1000, 200);
          perf.ms = perf.ms === 0 ? ms : perf.ms * 0.9 + ms * 0.1;
          perf.since += dt;
          if (perf.since > 0.5) {
            perf.since = 0;
            const before = renderScale;
            renderScale = nextRenderScale(renderScale, perf.ms, scaleCeiling);
            if (Math.abs(renderScale - before) > 0.005) resize();
          }
          perf.report = { fps: Math.round(1000 / Math.max(perf.ms, 1)), frameMs: +perf.ms.toFixed(1), scale: +renderScale.toFixed(2), dpr };
          (window as unknown as { __walkPerf?: unknown }).__walkPerf = perf.report;
        }
      });

      // ---------------------------------------------------------------- start ----
      a.start();
      const startArea = (startKey && slots.get(startKey)?.area) || manifest.base;
      const start = entryPose(startArea);
      pose.position = start.position;
      pose.yaw = start.yaw;
      applyPose();
      applyEnv(startArea);
      reportProgress();

      api.current = {
        goToArea: (key) => {
          const slot = slots.get(key);
          if (!slot) return;
          const p = entryPose(slot.area);
          if (!slot.area.placed) currentBox = viewerBox(slot.area);
          // The entry pose lies on the target's own path; judge it by that path, not the old one.
          currentCorridor = slot.area.placed ? placedCorridor : corridorOf([slot.area]);
          pinnedKey = slot.area.placed ? null : key;
          teleport(p.position, p.yaw);
          void loadArea(key).then(() => {
            const k = chooseArea(false);
            if (k) setActive(k, true);
          });
        },
        teleport: (position, yaw) => {
          // Callers teleport within the shared frame (the plan): leave any unplaced area.
          if (pinnedKey) {
            pinnedKey = null;
            currentBox = worldBox;
            currentCorridor = placedCorridor;
          }
          teleport(position, yaw);
        },
        walkTo: (position) => walkTo(position),
        setMode: (m) => {
          mode = m;
          if (m === "walk") {
            pose.position = [pose.position[0], EYE_HEIGHT, pose.position[2]];
            applyPose();
          }
        },
        setAutoTour: (on) => {
          if (on) startTour();
          else if (tour) {
            tour = null;
          }
        },
        pickAndWalk,
        resetFov: () => {
          fovTouched = false;
          pose.fov = openFov();
        },
        screenshot: () => {
          try {
            a.render();
            return canvasEl.toDataURL("image/jpeg", 0.92);
          } catch {
            return null;
          }
        },
        getPose: () => ({ position: [...pose.position] as Vec3, yaw: pose.yaw, pitch: pose.pitch }),
        vr: vrHost,
      };

      // Base first, the start room next, then everything else quietly.
      const order = [manifest.base.key, ...(startKey && startKey !== manifest.base.key ? [startKey] : []), ...areas.map((x) => x.key)];
      const seen = new Set<string>();
      for (const key of order) {
        if (disposed) return;
        if (seen.has(key)) continue;
        seen.add(key);
        await loadArea(key);
        const k = chooseArea();
        if (k && k !== activeKey) setActive(k);
      }
    })();

    return () => {
      disposed = true;
      api.current = null;
      for (const fn of cleanups) fn();
      try {
        app?.destroy();
      } catch {
        /* the device may already be lost */
      }
    };
  }, [manifest, startKey, container, canvas, fade, enabled]);

  return api;
}
