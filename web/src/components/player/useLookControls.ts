"use client";

/**
 * Hand-written look controls for a camera that sits at the centre of the sphere.
 *
 * OrbitControls is the obvious reflex and the wrong tool: it orbits a target at a
 * distance, and here the distance is zero — the target and the eye are the same point,
 * which is degenerate. What we actually want is a first-person head: two angles, no
 * translation, no roll.
 *
 * Everything below runs off refs and a single `useFrame`. Nothing here calls `setState`,
 * because at 120 Hz on a phone a per-frame React render is the difference between a dial
 * that feels machined and one that feels like a web page.
 */

import { useCallback, useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { angleDelta, clamp, damp, radians } from "@/lib/utils";
import type { InitialView } from "@/lib/types";
import { FOV_DEFAULT, FOV_MAX, FOV_MIN, TIMING, type LookControlsApi } from "./types";

// ---------------------------------------------------------------------------------
// The angle convention, in one place
// ---------------------------------------------------------------------------------

/** Pitch stops just short of the poles; at exactly +/-pi/2 yaw becomes meaningless. */
const PITCH_LIMIT = Math.PI / 2 - 0.01;

/**
 * Widest HORIZONTAL field the player will show, in degrees.
 *
 * `fov` throughout this file is three's vertical field of view, and it is applied
 * regardless of aspect: the 75-degree default is 107 degrees across a 16:9 desktop and
 * 140 on an ultrawide at the maximum, with a 2.9-5.5x rectilinear stretch at the frame
 * edges. Straight cornices bow as the view pans and near objects smear — a "distortion"
 * that is entirely the projection, not the panorama.
 *
 * THIS IS A BACKSTOP FOR ULTRAWIDE SCREENS, NOT A ZOOM LIMIT, and the difference matters:
 * at 100 degrees it WAS the zoom limit. `FOV_MAX` is 100 degrees of vertical field, which
 * on a 16:9 window is 129 degrees across; capping the horizontal at 100 held the vertical
 * at 68, so the last third of the zoom-out range moved the number and changed nothing on
 * screen. Zooming out simply stopped, with no edge of travel to feel. 130 degrees clears
 * the vertical maximum on everything up to about 16:9 — the zoom range is whole again —
 * and still trims the 140-degree view a 21:9 monitor would otherwise get.
 */
const HFOV_MAX_DEG = 130;

export function effectiveFov(fovDeg: number, aspect: number): number {
  if (!Number.isFinite(aspect) || aspect <= 0) return fovDeg;
  const maxVertical = (2 * Math.atan(Math.tan((HFOV_MAX_DEG * Math.PI) / 360) / aspect) * 180) / Math.PI;
  return Math.min(fovDeg, maxVertical);
}

const _euler = new THREE.Euler();
const _ndc = new THREE.Vector2();

/**
 * Unit direction for a (yaw, pitch) pair.
 *
 * yaw 0 points at +X, which is where the panorama's centre column lands on a
 * `SphereGeometry` once the shader flips u. Positive yaw sweeps toward +Z, which is the
 * viewer's right at yaw 0 — matching "positive turns right" in the engine's contract.
 */
export function directionFromAngles(yaw: number, pitch: number, out = new THREE.Vector3()): THREE.Vector3 {
  const cp = Math.cos(pitch);
  return out.set(cp * Math.cos(yaw), Math.sin(pitch), cp * Math.sin(yaw));
}

/** Inverse of {@link directionFromAngles}. The vector need not be normalised. */
export function anglesFromDirection(d: THREE.Vector3): { yaw: number; pitch: number } {
  const len = d.length() || 1;
  return { yaw: Math.atan2(d.z / len, d.x / len), pitch: Math.asin(clamp(d.y / len, -1, 1)) };
}

/**
 * Points a camera along (yaw, pitch) with no roll.
 *
 * YXZ Euler order is what keeps the horizon level: yaw about world Y is applied first,
 * pitch about the already-yawed X second, and Z stays zero. The -(yaw + pi/2) term is the
 * one-time offset that puts yaw 0 on +X instead of the camera's default -Z.
 */
export function orientCamera(camera: THREE.Camera, yaw: number, pitch: number): void {
  camera.quaternion.setFromEuler(_euler.set(pitch, -(yaw + Math.PI / 2), 0, "YXZ"));
}

/** Wraps to the contract's (-pi, pi]. */
export function normaliseYaw(yaw: number): number {
  const m = (((yaw + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return m === -Math.PI ? Math.PI : m;
}

// ---------------------------------------------------------------------------------
// Feel constants — these are the whole product, tune them with care
// ---------------------------------------------------------------------------------

/** Inertia decay. Lower travels further; 2.8 gives roughly a second of glide. */
const INERTIA_LAMBDA = 2.8;
/** Above this the throw feels like a slingshot rather than a heavy dial. */
const MAX_THROW = 5.5; // rad/s
/** Velocity is measured over the tail of the drag, so a pause before release means stop. */
const FLICK_WINDOW_MS = 90;
/** Zoom easing. High enough to feel immediate, low enough to never step. */
const FOV_LAMBDA = 11;
/**
 * Auto-rotate: a full turn in sixty seconds, at full speed within two thirds of one.
 *
 * BOTH NUMBERS WERE TUNED FOR A DRIFT NOBODY ASKED FOR. 1.6 deg/s over a 1400 ms ramp is
 * ambience — it started itself four idle seconds after you stopped touching the player, when
 * by definition you had looked away, and it was meant to be noticed only in the corner of the
 * eye. Integrated, that is 0.5 degrees in the first second and 3.9 in the first three.
 *
 * It is a button now, and the first second after a press is the one second somebody is
 * watching for a change. The shipped 6 deg/s over a 600 ms ramp turns the room 4.2 degrees in
 * that second and 16.2 in three: unmistakably moving, and still slow enough to read a wall by
 * on the way past. The ramp stays because the start should be eased rather than a jump-cut;
 * it is just no longer sized for a viewer who was not looking.
 *
 * The scale to judge this on is the minute: at 6 deg/s a viewer who leaves it running sees
 * the whole room once a minute, which is about how long somebody spends on one panorama.
 * `autoRotateDelta` is the arithmetic, and the tests integrate it and read the degrees.
 */
const AUTO_ROTATE_RATE = radians(6);
const AUTO_ROTATE_RAMP_MS = 600;

/**
 * Yaw the drift adds over `dt` seconds, `idleFor` milliseconds after it became eligible.
 *
 * A function rather than four lines in the frame loop because "how long after the press does
 * the room actually turn" is the whole of what this control promises, and the frame loop
 * needs a WebGL context and a headset check to run. Here it is arithmetic, and a test can
 * integrate it and read the answer in degrees.
 *
 * Negative `idleFor` is the ordinary case, not an error: it is the wait between the last
 * touch and the drift being allowed to start at all.
 */
export function autoRotateDelta(idleFor: number, dt: number): number {
  if (idleFor <= 0) return 0;
  // Ease in rather than snapping to full speed the instant the gate opens.
  const ramp = clamp(idleFor / AUTO_ROTATE_RAMP_MS, 0, 1);
  return AUTO_ROTATE_RATE * dt * (ramp * ramp * (3 - 2 * ramp));
}
const KEY_LOOK_RATE = radians(80); // rad/s
const KEY_ZOOM_RATE = 34; // deg/s
/** Pointer travel in px below which a press counts as a tap, not a drag. */
const TAP_SLOP = 6;

const easeOutExpo = (t: number) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));

interface Sample {
  t: number;
  yaw: number;
  pitch: number;
}

interface Anim {
  yaw0: number;
  pitch0: number;
  fov0: number;
  dYaw: number;
  dPitch: number;
  dFov: number;
  t: number;
  dur: number;
}

interface ControlState {
  yaw: number;
  pitch: number;
  fov: number;
  /** Where the user's own zoom sits; `fovOffset` is layered on for transitions. */
  fovTarget: number;
  fovOffset: number;
  yawVel: number;
  pitchVel: number;
  dragging: boolean;
  pointers: Map<number, { x: number; y: number }>;
  anchor: { x: number; y: number } | null;
  pinchDist: number;
  pinchFov: number;
  travelled: number;
  samples: Sample[];
  lastInputAt: number;
  keys: Set<string>;
  anim: Anim | null;
  /** Set once, on the first frame, so the initial view does not animate in from zero. */
  primed: boolean;
}

export interface LookControlsOptions {
  /** Element that receives pointer, wheel and key input. Usually the player shell. */
  target: HTMLElement | null;
  /** Where to start looking. Applied on the first frame and whenever the identity changes. */
  initialView?: InitialView;
  enabled?: boolean;
  autoRotate?: boolean;
  /** Runs after the camera has been updated for this frame — do DOM projection here. */
  onFrame?: (view: Readonly<InitialView>, camera: THREE.PerspectiveCamera, dt: number) => void;
  /** Throttled; fires when the view has come to rest. */
  onViewSettled?: (view: InitialView) => void;
  /** A press that never became a drag, reported as a sphere direction. */
  onTap?: (p: { yaw: number; pitch: number }, event: PointerEvent) => void;
  onRequestFullscreen?: () => void;
  onRequestReset?: () => void;
  /** Published so the HTML chrome and the imperative handle can drive the camera. */
  apiRef?: React.RefObject<LookControlsApi | null>;
  /**
   * How far up and down the viewer may look, in radians. Defaults to the poles. A scene
   * whose sweep never reached the ceiling passes its covered band here so the synthesised
   * caps stay out of reach.
   */
  pitchLimits?: { min: number; max: number };
}

export function useLookControls(options: LookControlsOptions): LookControlsApi {
  const {
    target,
    initialView,
    enabled = true,
    autoRotate = false,
    onFrame,
    onViewSettled,
    onTap,
    onRequestFullscreen,
    onRequestReset,
    apiRef,
    pitchLimits,
  } = options;

  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const perspective = camera instanceof THREE.PerspectiveCamera ? camera : null;

  const state = useRef<ControlState>({
    yaw: initialView?.yaw ?? 0,
    pitch: initialView?.pitch ?? 0,
    fov: initialView?.fov ?? FOV_DEFAULT,
    fovTarget: initialView?.fov ?? FOV_DEFAULT,
    fovOffset: 0,
    yawVel: 0,
    pitchVel: 0,
    dragging: false,
    pointers: new Map(),
    anchor: null,
    pinchDist: 0,
    pinchFov: FOV_DEFAULT,
    travelled: 0,
    samples: [],
    lastInputAt: 0,
    keys: new Set(),
    anim: null,
    primed: false,
  });

  // Latest callbacks and flags, without re-subscribing listeners on every parent render.
  const handlers = useRef({ onFrame, onViewSettled, onTap, onRequestFullscreen, onRequestReset });
  const flags = useRef({ enabled, autoRotate, reducedMotion: false });
  const limits = useRef({ min: -PITCH_LIMIT, max: PITCH_LIMIT });
  const clampPitch = (p: number) => clamp(p, limits.current.min, limits.current.max);

  // Refreshed after commit rather than during render. A render React discards — StrictMode's
  // double invoke, or an interrupted concurrent render — must not leave the frame loop
  // holding callbacks from a tree that was never mounted.
  useEffect(() => {
    handlers.current = { onFrame, onViewSettled, onTap, onRequestFullscreen, onRequestReset };
    flags.current.enabled = enabled;
    /**
     * SWITCHING IT ON IS THE INPUT THAT STARTS IT.
     *
     * The drift is gated on `idleMs` of nothing happening, which is right when it starts
     * itself — an untouched player should settle before it begins to move. It is wrong when
     * somebody presses the button, because the press does not count as input (the control
     * carries `data-player-ui`, so the pointer handlers skip it) and the drag that preceded
     * it does: press, and the picture stays dead for the remainder of four seconds and then
     * eases in over another one and a half. Nothing on screen distinguishes that from a
     * broken toggle, so the press backdates the idle clock and the ramp starts now.
     */
    if (autoRotate && !flags.current.autoRotate) {
      state.current.lastInputAt = performance.now() - TIMING.idleMs;
    }
    flags.current.autoRotate = autoRotate;
    limits.current = {
      min: clamp(pitchLimits?.min ?? -PITCH_LIMIT, -PITCH_LIMIT, 0),
      max: clamp(pitchLimits?.max ?? PITCH_LIMIT, 0, PITCH_LIMIT),
    };
  });

  const raycaster = useMemo(() => new THREE.Raycaster(), []);
  const settled = useRef({ reported: true });

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => {
      flags.current.reducedMotion = mq.matches;
    };
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  // ---------------------------------------------------------------------------------
  // Imperative API
  // ---------------------------------------------------------------------------------

  const getView = useCallback((): InitialView => {
    const s = state.current;
    return { yaw: normaliseYaw(s.yaw), pitch: s.pitch, fov: s.fov };
  }, []);

  const stop = useCallback(() => {
    const s = state.current;
    s.anim = null;
    s.yawVel = 0;
    s.pitchVel = 0;
  }, []);

  const poke = useCallback(() => {
    state.current.lastInputAt = performance.now();
  }, []);

  const lookAt = useCallback(
    (view: Partial<InitialView>, opts?: { animate?: boolean; durationMs?: number }) => {
      const s = state.current;
      const animate = opts?.animate !== false && !flags.current.reducedMotion;
      const targetYaw = view.yaw ?? s.yaw;
      const targetPitch = clampPitch(view.pitch ?? s.pitch);
      const targetFov = clamp(view.fov ?? s.fovTarget, FOV_MIN, FOV_MAX);

      s.yawVel = 0;
      s.pitchVel = 0;
      s.fovTarget = targetFov;

      if (!animate) {
        s.anim = null;
        s.yaw = normaliseYaw(targetYaw);
        s.pitch = targetPitch;
        s.fov = clamp(targetFov + s.fovOffset, FOV_MIN, FOV_MAX);
        return;
      }

      // angleDelta keeps a turn from 170 deg to -170 deg going the short way (20 deg)
      // instead of unwinding 340 deg the long way round.
      s.anim = {
        yaw0: s.yaw,
        pitch0: s.pitch,
        fov0: s.fov,
        dYaw: angleDelta(targetYaw, s.yaw),
        dPitch: targetPitch - s.pitch,
        dFov: clamp(targetFov + s.fovOffset, FOV_MIN, FOV_MAX) - s.fov,
        t: 0,
        dur: (opts?.durationMs ?? 780) / 1000,
      };
    },
    [],
  );

  const setFovOffset = useCallback((deg: number) => {
    state.current.fovOffset = deg;
  }, []);

  const directionAt = useCallback(
    (clientX: number, clientY: number) => {
      if (!target || !perspective) return null;
      const rect = target.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return null;
      const ndc = _ndc.set(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -(((clientY - rect.top) / rect.height) * 2 - 1),
      );
      // The camera sits at the origin, so the ray direction *is* the sphere direction —
      // no mesh intersection needed, and it stays correct if the sphere is ever hidden.
      raycaster.setFromCamera(ndc, perspective);
      return anglesFromDirection(raycaster.ray.direction);
    },
    [target, perspective, raycaster],
  );

  const isDragging = useCallback(() => state.current.dragging, []);

  const api = useMemo<LookControlsApi>(
    () => ({ getView, lookAt, stop, poke, setFovOffset, directionAt, isDragging }),
    [getView, lookAt, stop, poke, setFovOffset, directionAt, isDragging],
  );

  useEffect(() => {
    if (!apiRef) return;
    apiRef.current = api;
    return () => {
      apiRef.current = null;
    };
  }, [api, apiRef]);

  // ---------------------------------------------------------------------------------
  // Input
  // ---------------------------------------------------------------------------------

  useEffect(() => {
    if (!target) return;

    const s = state.current;
    /** Radians of rotation per pixel of drag, so the world tracks the finger 1:1. */
    const perPixel = () =>
      radians(effectiveFov(s.fov, target.clientWidth / Math.max(1, target.clientHeight))) /
      Math.max(1, target.clientHeight);

    const markInput = () => {
      s.lastInputAt = performance.now();
      s.anim = null;
      settled.current.reported = false;
    };

    const midpoint = () => {
      let x = 0;
      let y = 0;
      for (const p of s.pointers.values()) {
        x += p.x;
        y += p.y;
      }
      const n = s.pointers.size || 1;
      return { x: x / n, y: y / n };
    };

    const spread = () => {
      const pts = [...s.pointers.values()];
      if (pts.length < 2) return 0;
      return Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
    };

    const onPointerDown = (e: PointerEvent) => {
      // Clicks on the toolbar, hotspots or any other overlay must not spin the world.
      if (e.target instanceof Element && e.target.closest("[data-player-ui]")) return;
      if (!flags.current.enabled) return;

      s.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      target.setPointerCapture?.(e.pointerId);
      markInput();
      s.yawVel = 0;
      s.pitchVel = 0;

      if (s.pointers.size === 1) {
        s.dragging = true;
        s.travelled = 0;
        s.samples = [{ t: performance.now(), yaw: s.yaw, pitch: s.pitch }];
      }
      if (s.pointers.size === 2) {
        s.pinchDist = spread();
        s.pinchFov = s.fovTarget;
      }
      s.anchor = midpoint();
    };

    const onPointerMove = (e: PointerEvent) => {
      const tracked = s.pointers.get(e.pointerId);
      if (!tracked || !flags.current.enabled) return;
      // A `pointerup` delivered outside the window (or swallowed by devtools) would
      // otherwise leave the drag latched, so every subsequent mouse move keeps spinning
      // the view. No buttons held means the gesture is over, whatever we were told.
      if (e.pointerType === "mouse" && e.buttons === 0) {
        onPointerUp(e);
        return;
      }
      tracked.x = e.clientX;
      tracked.y = e.clientY;

      const mid = midpoint();
      if (s.anchor) {
        const dx = mid.x - s.anchor.x;
        const dy = mid.y - s.anchor.y;
        const k = perPixel();
        // Drag right pushes the world right, so the camera turns left: yaw decreases.
        s.yaw -= dx * k;
        s.pitch = clampPitch(s.pitch + dy * k);
        s.travelled += Math.hypot(dx, dy);
      }
      s.anchor = mid;

      if (s.pointers.size >= 2 && s.pinchDist > 0) {
        const dist = spread();
        if (dist > 0) s.fovTarget = clamp(s.pinchFov * (s.pinchDist / dist), FOV_MIN, FOV_MAX);
      }

      const now = performance.now();
      s.samples.push({ t: now, yaw: s.yaw, pitch: s.pitch });
      if (s.samples.length > 12) s.samples.shift();
      s.lastInputAt = now;
    };

    const releaseInertia = () => {
      const now = performance.now();
      const cutoff = now - FLICK_WINDOW_MS;
      // Only samples from the tail of the gesture count. Holding still for a moment
      // before lifting therefore parks the view instead of throwing it.
      const anchor = s.samples.find((p) => p.t >= cutoff);
      if (anchor && now - anchor.t > 8) {
        const dt = (now - anchor.t) / 1000;
        s.yawVel = clamp(angleDelta(s.yaw, anchor.yaw) / dt, -MAX_THROW, MAX_THROW);
        s.pitchVel = clamp((s.pitch - anchor.pitch) / dt, -MAX_THROW, MAX_THROW);
      } else {
        s.yawVel = 0;
        s.pitchVel = 0;
      }
      s.samples = [];
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!s.pointers.has(e.pointerId)) return;
      s.pointers.delete(e.pointerId);
      target.releasePointerCapture?.(e.pointerId);

      if (s.pointers.size === 0) {
        const wasTap = s.travelled < TAP_SLOP;
        s.dragging = false;
        s.anchor = null;
        s.pinchDist = 0;
        if (wasTap) {
          s.yawVel = 0;
          s.pitchVel = 0;
          const dir = api.directionAt(e.clientX, e.clientY);
          if (dir) handlers.current.onTap?.(dir, e);
        } else {
          releaseInertia();
        }
      } else {
        // Lifting one finger of a pinch must not make the remaining one jump.
        s.anchor = midpoint();
        s.pinchDist = spread();
        s.pinchFov = s.fovTarget;
      }
      s.lastInputAt = performance.now();
    };

    const onPointerCancel = (e: PointerEvent) => {
      s.pointers.delete(e.pointerId);
      if (s.pointers.size === 0) {
        s.dragging = false;
        s.anchor = null;
        s.pinchDist = 0;
        s.samples = [];
      }
    };

    const onWheel = (e: WheelEvent) => {
      if (!flags.current.enabled) return;
      if (e.target instanceof Element && e.target.closest("[data-player-ui]")) return;
      e.preventDefault();
      markInput();
      // deltaMode 1 is lines, 2 is pages; normalise both to something pixel-ish.
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 100 : 1;
      s.fovTarget = clamp(s.fovTarget + e.deltaY * unit * 0.045, FOV_MIN, FOV_MAX);
    };

    const KEY_ALIASES: Record<string, string> = {
      arrowup: "up",
      w: "up",
      arrowdown: "down",
      s: "down",
      arrowleft: "left",
      a: "left",
      arrowright: "right",
      d: "right",
      "+": "in",
      "=": "in",
      "-": "out",
      _: "out",
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (!flags.current.enabled) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key.toLowerCase();

      if (key === "f") {
        e.preventDefault();
        handlers.current.onRequestFullscreen?.();
        return;
      }
      if (key === "r") {
        e.preventDefault();
        handlers.current.onRequestReset?.();
        return;
      }
      const alias = KEY_ALIASES[key];
      if (!alias) return;
      e.preventDefault();
      s.keys.add(alias);
      markInput();
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const alias = KEY_ALIASES[e.key.toLowerCase()];
      if (alias) s.keys.delete(alias);
    };

    // Losing focus mid-keypress would otherwise leave the view spinning forever.
    const onBlur = () => s.keys.clear();

    target.addEventListener("pointerdown", onPointerDown);
    target.addEventListener("pointermove", onPointerMove);
    target.addEventListener("pointerup", onPointerUp);
    target.addEventListener("pointercancel", onPointerCancel);
    // Fires whenever capture ends for any reason, including the ones that do not also
    // produce a pointerup — the belt to the pointercancel braces.
    target.addEventListener("lostpointercapture", onPointerCancel);
    target.addEventListener("wheel", onWheel, { passive: false });
    target.addEventListener("keydown", onKeyDown);
    target.addEventListener("keyup", onKeyUp);
    target.addEventListener("blur", onBlur);

    return () => {
      target.removeEventListener("pointerdown", onPointerDown);
      target.removeEventListener("pointermove", onPointerMove);
      target.removeEventListener("pointerup", onPointerUp);
      target.removeEventListener("pointercancel", onPointerCancel);
      target.removeEventListener("lostpointercapture", onPointerCancel);
      target.removeEventListener("wheel", onWheel);
      target.removeEventListener("keydown", onKeyDown);
      target.removeEventListener("keyup", onKeyUp);
      target.removeEventListener("blur", onBlur);
      s.keys.clear();
      s.pointers.clear();
      s.dragging = false;
    };
  }, [target, api]);

  // ---------------------------------------------------------------------------------
  // Frame loop
  // ---------------------------------------------------------------------------------

  useFrame((_, rawDelta) => {
    if (!perspective) return;
    const s = state.current;
    // A backgrounded tab resumes with a huge delta; clamping stops a jump-cut.
    const dt = clamp(rawDelta, 0, 1 / 20);
    const now = performance.now();

    if (!s.primed) {
      s.primed = true;
      if (initialView) {
        s.yaw = normaliseYaw(initialView.yaw);
        s.pitch = clampPitch(initialView.pitch);
        s.fov = clamp(initialView.fov, FOV_MIN, FOV_MAX);
        s.fovTarget = s.fov;
      }
      s.lastInputAt = now;
    }

    // 1. Held keys.
    if (s.keys.size > 0) {
      const rate = KEY_LOOK_RATE * dt * (s.fov / FOV_DEFAULT);
      if (s.keys.has("left")) s.yaw -= rate;
      if (s.keys.has("right")) s.yaw += rate;
      if (s.keys.has("up")) s.pitch += rate;
      if (s.keys.has("down")) s.pitch -= rate;
      if (s.keys.has("in")) s.fovTarget = clamp(s.fovTarget - KEY_ZOOM_RATE * dt, FOV_MIN, FOV_MAX);
      if (s.keys.has("out")) s.fovTarget = clamp(s.fovTarget + KEY_ZOOM_RATE * dt, FOV_MIN, FOV_MAX);
      s.lastInputAt = now;
      settled.current.reported = false;
    }

    // 2. Scripted move, 3. inertia — mutually exclusive.
    if (s.anim) {
      s.anim.t += dt;
      const e = easeOutExpo(clamp(s.anim.t / s.anim.dur, 0, 1));
      s.yaw = s.anim.yaw0 + s.anim.dYaw * e;
      s.pitch = s.anim.pitch0 + s.anim.dPitch * e;
      s.fov = s.anim.fov0 + s.anim.dFov * e;
      if (s.anim.t >= s.anim.dur) s.anim = null;
    } else if (!s.dragging) {
      if (s.yawVel !== 0 || s.pitchVel !== 0) {
        s.yaw += s.yawVel * dt;
        s.pitch += s.pitchVel * dt;
        s.yawVel = damp(s.yawVel, 0, INERTIA_LAMBDA, dt);
        s.pitchVel = damp(s.pitchVel, 0, INERTIA_LAMBDA, dt);
        // Below a fiftieth of a degree per second the glide is over; snap it dead so the
        // frame loop can go quiet instead of dithering forever.
        if (Math.abs(s.yawVel) < 3e-4) s.yawVel = 0;
        if (Math.abs(s.pitchVel) < 3e-4) s.pitchVel = 0;
      }

      if (flags.current.autoRotate && !flags.current.reducedMotion && s.yawVel === 0) {
        s.yaw += autoRotateDelta(now - s.lastInputAt - TIMING.idleMs, dt);
      }
    }

    // 4. Limits. Hitting the pole kills vertical momentum so it does not "stick".
    const clampedPitch = clampPitch(s.pitch);
    if (clampedPitch !== s.pitch) {
      s.pitch = clampedPitch;
      s.pitchVel = 0;
    }
    s.yaw = normaliseYaw(s.yaw);

    // 5. Zoom, always eased — a stepped fov reads as a bug.
    const wantFov = clamp(s.fovTarget + s.fovOffset, FOV_MIN, FOV_MAX);
    if (!s.anim && Math.abs(s.fov - wantFov) > 1e-3) {
      s.fov = damp(s.fov, wantFov, FOV_LAMBDA, dt);
      if (Math.abs(s.fov - wantFov) <= 1e-3) s.fov = wantFov;
    }
    const shownFov = effectiveFov(s.fov, perspective.aspect);
    if (Math.abs(perspective.fov - shownFov) > 1e-4) {
      perspective.fov = shownFov;
      perspective.updateProjectionMatrix();
    }

    // 6. Point the head — unless a headset owns the pose, in which case three overwrites
    //    the camera transform from the XR frame and our angles are only a fallback.
    if (!gl.xr?.isPresenting) orientCamera(perspective, s.yaw, s.pitch);
    // The projection in step 7 runs before the renderer's own matrix pass, so refresh
    // both matrices here; a stale matrixWorldInverse puts every hotspot a frame behind.
    perspective.updateMatrixWorld();
    perspective.matrixWorldInverse.copy(perspective.matrixWorld).invert();

    // 7. Everything that reads the camera — hotspot projection, compass — runs here so
    //    it can never be a frame behind the pose it is drawn against.
    handlers.current.onFrame?.({ yaw: s.yaw, pitch: s.pitch, fov: s.fov }, perspective, dt);

    // 8. Report a resting view, once, to whoever is persisting it.
    const moving = s.dragging || s.anim !== null || s.yawVel !== 0 || s.pitchVel !== 0;
    if (moving) {
      settled.current.reported = false;
    } else if (!settled.current.reported && now - s.lastInputAt > 260) {
      settled.current.reported = true;
      handlers.current.onViewSettled?.({ yaw: s.yaw, pitch: s.pitch, fov: s.fov });
    }
  });

  return api;
}
