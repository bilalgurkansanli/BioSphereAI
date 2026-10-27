"use client";

/**
 * The phone-in-a-box session: everything around the stereo picture that the browser makes
 * difficult. Shared by the 360 player and the 3D walkthrough, which differ only in what they
 * draw; neither engine is imported here.
 *
 * ── ONE GESTURE, SPENT CAREFULLY ──────────────────────────────────────────────────────────
 * iOS hands out motion sensors only through `DeviceOrientationEvent.requestPermission()`, and
 * only when it is called from inside a user gesture; Android hands out fullscreen the same way.
 * An `await` before either call can drop the gesture (Safari is strict about it), so `start`
 * makes both calls synchronously, first thing, and only then waits for their answers. That is
 * also why the first-time guide exists as a separate step: its "Başla" button is a second,
 * fresh gesture, so the visitor can read what is about to happen before the prompt appears.
 *
 * ── EVERY WAY OUT ─────────────────────────────────────────────────────────────────────────
 * The phone is in a plastic box on somebody's face; they cannot see a close button. So:
 * a tap anywhere brings up the exit bar (or selects what the gaze is on, which is what the
 * Cardboard's own button means), two fingers anywhere leave at once, the back button and the
 * iOS back swipe leave (a history entry is pushed for exactly that), dropping out of
 * fullscreen leaves (Android's back does that first), Escape leaves on a desktop, and the
 * panel at the visitor's feet has an exit button that works by looking at it.
 *
 * ── WHEN THE SENSORS NEVER SPEAK ──────────────────────────────────────────────────────────
 * Permission granted is not the same as events arriving: a phone with no gyroscope, a desktop
 * browser that defines the event and never fires it, a cross-origin frame that was denied the
 * sensor. After `SENSOR_TIMEOUT_MS` of silence the head follows a finger drag instead, and the
 * visitor is told why, rather than being left in a frozen split screen.
 */

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { DEFAULT_LENS, LENS_PROFILES, parseLensOverride, type LensId, type LensProfile } from "./lens";
import { quatFromDeviceOrientation, quatFromView, readScreenAngle, type Quat } from "./orientation";

export interface OrientationFeed {
  /** The latest head rotation, written by the sensor or the drag; read once per frame. */
  q: Quat;
  /** False until the first sample. */
  valid: boolean;
  /** `performance.now()` of the latest sample. */
  stamp: number;
  source: "sensor" | "drag";
}

export type CardboardPhase = "off" | "guide" | "starting" | "active";

export interface CardboardMessages {
  denied: string;
  noSensor: string;
}

export interface UseCardboardOptions {
  /**
   * The element taken fullscreen, and the one taps and drags are read from. A getter, read
   * when it is needed: the 360 player holds its element in state, the walkthrough in a ref.
   */
  container: () => HTMLElement | null;
  /** `?vr=cardboard`: drag stands in for the sensors from the start. */
  debug: boolean;
  lensOverride: string | null;
  messages: CardboardMessages;
}

export interface CardboardController {
  phase: CardboardPhase;
  active: boolean;
  /** The phone is upright; the split view waits behind a "turn your phone" card. */
  portrait: boolean;
  /** The exit bar is showing. */
  hud: boolean;
  /** Real element fullscreen is on. When it is not (iPhone), the host covers the viewport itself. */
  fullscreen: boolean;
  lensId: LensId;
  lens: LensProfile;
  setLensId: (id: LensId) => void;
  message: string | null;
  feed: React.RefObject<OrientationFeed>;
  /**
   * Filled in by whichever renderer is drawing: select what the gaze is on. Returns whether
   * anything was there, so a tap on nothing can bring up the exit bar instead.
   */
  selectRef: React.RefObject<(() => boolean) | null>;
  /** The VR button: the guide the first time, straight in after that. Call from a gesture. */
  request: () => void;
  /** The guide's start button. Call from a gesture. */
  start: () => void;
  stop: () => void;
  cancelGuide: () => void;
  showHud: () => void;
}

const SEEN_KEY = "biosphere.vr.guideSeen";
const LENS_KEY = "biosphere.vr.lens";
/** Long enough for a slow sensor's first event, short enough not to feel like a hang. */
const SENSOR_TIMEOUT_MS = 1500;
const HUD_MS = 4200;
const MESSAGE_MS = 6500;
/** Pointer travel below which a press is a tap, as in the 2D player. */
const TAP_SLOP = 8;
/** Drag look speed for the stand-in head, radians per CSS pixel. */
const DRAG_RATE = 0.0045;
const PITCH_LIMIT = Math.PI / 2 - 0.02;

function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* A private window or a blocked store: the guide simply shows again next time. */
  }
}

type PermissionCtor = typeof DeviceOrientationEvent & { requestPermission?: () => Promise<"granted" | "denied" | string> };

/** The lens picked last time. Nothing on the server render depends on it, so no mismatch. */
function storedLens(): LensId {
  if (typeof window === "undefined") return DEFAULT_LENS;
  const stored = readStorage(LENS_KEY);
  return stored === "cardboard" || stored === "vrbox" ? stored : DEFAULT_LENS;
}

const PORTRAIT_QUERY = "(orientation: portrait)";
function subscribePortrait(onChange: () => void): () => void {
  if (typeof window === "undefined" || !window.matchMedia) return () => undefined;
  const query = window.matchMedia(PORTRAIT_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}
const readPortrait = () => window.matchMedia?.(PORTRAIT_QUERY).matches ?? false;
const serverPortrait = () => false;
type LockableOrientation = ScreenOrientation & { lock?: (orientation: string) => Promise<void>; unlock?: () => void };

export function useCardboard(options: UseCardboardOptions): CardboardController {
  const { container: getContainer, debug, lensOverride, messages } = options;
  const [phase, setPhase] = useState<CardboardPhase>("off");
  const [hud, setHud] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [lensId, setLensIdState] = useState<LensId>(storedLens);
  // Portrait matters only while the split view is up; the desktop debug mode never asks.
  const portraitNow = useSyncExternalStore(subscribePortrait, readPortrait, serverPortrait);
  const [message, setMessage] = useState<string | null>(null);

  const feed = useRef<OrientationFeed>({ q: [0, 0, 0, 1], valid: false, stamp: 0, source: debug ? "drag" : "sensor" });
  const selectRef = useRef<(() => boolean) | null>(null);
  const phaseRef = useRef<CardboardPhase>("off");
  const enteredFullscreen = useRef(false);
  const pushedHistory = useRef(false);
  const hudTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  // A typed-in override (`?vrlens=`) is applied on top of the picked profile.
  const setLensId = useCallback((id: LensId) => {
    setLensIdState(id);
    writeStorage(LENS_KEY, id);
  }, []);
  // Memoised: the renderer rebuilds its distortion meshes when this identity changes.
  const lens = useMemo(() => parseLensOverride(lensOverride, LENS_PROFILES[lensId]), [lensOverride, lensId]);

  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(() => setMessage(null), MESSAGE_MS);
    return () => clearTimeout(timer);
  }, [message]);

  const showHud = useCallback(() => {
    setHud(true);
    if (hudTimer.current) clearTimeout(hudTimer.current);
    hudTimer.current = setTimeout(() => setHud(false), HUD_MS);
  }, []);

  // ---------------------------------------------------------------------------------
  // Leaving
  // ---------------------------------------------------------------------------------

  const stop = useCallback((fromHistory = false) => {
    if (phaseRef.current === "off") return;
    phaseRef.current = "off";
    setPhase("off");
    setHud(false);
    feed.current.valid = false;
    const orientation = (typeof screen !== "undefined" ? screen.orientation : undefined) as LockableOrientation | undefined;
    try {
      orientation?.unlock?.();
    } catch {
      /* Not locked, or not lockable: nothing to undo. */
    }
    if (enteredFullscreen.current && typeof document !== "undefined" && document.fullscreenElement) {
      void document.exitFullscreen().catch(() => undefined);
    }
    enteredFullscreen.current = false;
    setFullscreen(false);
    if (pushedHistory.current) {
      pushedHistory.current = false;
      // Our own entry is still on top unless the visitor's back button is why we are leaving.
      if (!fromHistory && (window.history.state as { biosphereVr?: boolean } | null)?.biosphereVr) window.history.back();
    }
  }, []);

  // ---------------------------------------------------------------------------------
  // Entering
  // ---------------------------------------------------------------------------------

  const start = useCallback(() => {
    if (typeof window === "undefined") return;
    if (phaseRef.current === "starting" || phaseRef.current === "active") return;

    // Both gesture-gated calls first, before any await. See the header.
    const ctor = window.DeviceOrientationEvent as PermissionCtor | undefined;
    const permission: Promise<string> =
      !debug && typeof ctor?.requestPermission === "function"
        ? ctor.requestPermission().catch(() => "denied")
        : Promise.resolve("granted");
    const container = getContainer();
    const fullscreenRequest: Promise<boolean> =
      container?.requestFullscreen && !document.fullscreenElement
        ? container.requestFullscreen({ navigationUI: "hide" }).then(
            () => true,
            () => false,
          )
        : Promise.resolve(false);

    phaseRef.current = "starting";
    setPhase("starting");
    setMessage(null);

    void Promise.all([permission, fullscreenRequest]).then(([granted, isFullscreen]) => {
      if (phaseRef.current !== "starting") return;
      enteredFullscreen.current = isFullscreen;
      setFullscreen(isFullscreen);
      if (granted !== "granted") {
        setMessage(messages.denied);
        stop();
        return;
      }
      writeStorage(SEEN_KEY, "1");
      feed.current = { q: quatFromView(-Math.PI / 2, 0), valid: debug, stamp: performance.now(), source: debug ? "drag" : "sensor" };
      // Landscape is a request, not a guarantee: only Android in fullscreen honours it. The
      // portrait card covers everyone else.
      if (isFullscreen) {
        const orientation = screen.orientation as LockableOrientation | undefined;
        void orientation?.lock?.("landscape")?.catch(() => undefined);
      }
      try {
        window.history.pushState({ ...(window.history.state ?? {}), biosphereVr: true }, "");
        pushedHistory.current = true;
      } catch {
        pushedHistory.current = false;
      }
      phaseRef.current = "active";
      setPhase("active");
    });
  }, [getContainer, debug, messages.denied, stop]);

  const request = useCallback(() => {
    if (readStorage(SEEN_KEY) === "1") start();
    else setPhase("guide");
  }, [start]);

  const cancelGuide = useCallback(() => {
    if (phaseRef.current === "guide") setPhase("off");
  }, []);

  // ---------------------------------------------------------------------------------
  // While active
  // ---------------------------------------------------------------------------------

  const active = phase === "active";

  // Head tracking from the sensors, with the drag fallback when they stay silent.
  useEffect(() => {
    if (!active || debug) return;
    const onOrientation = (event: DeviceOrientationEvent) => {
      if (event.alpha === null || event.beta === null || event.gamma === null) return;
      const f = feed.current;
      f.q = quatFromDeviceOrientation(event.alpha, event.beta, event.gamma, readScreenAngle());
      f.valid = true;
      f.stamp = performance.now();
      f.source = "sensor";
    };
    window.addEventListener("deviceorientation", onOrientation);
    const watchdog = setTimeout(() => {
      if (feed.current.source === "sensor" && !feed.current.valid) {
        feed.current = { q: quatFromView(-Math.PI / 2, 0), valid: true, stamp: performance.now(), source: "drag" };
        setMessage(messages.noSensor);
      }
    }, SENSOR_TIMEOUT_MS);
    return () => {
      clearTimeout(watchdog);
      window.removeEventListener("deviceorientation", onOrientation);
    };
  }, [active, debug, messages.noSensor]);

  // Taps, two-finger exit and the drag head, all on the player itself.
  //
  // Taken in the CAPTURE phase and stopped there. The viewer underneath has input handlers of
  // its own on the same element — the walkthrough's engine turns the camera on a drag, walks on
  // two fingers and walks to a double-tapped point — and in the box every one of those would
  // fire on a tap meant for the gaze: a double tap there queued a walk that played out after the
  // visitor left VR. Only the VR interface's own buttons (`data-vr-ui`) get their events through.
  useEffect(() => {
    const container = getContainer();
    if (!active || !container) return;
    let look = { yaw: -Math.PI / 2, pitch: 0 };
    let press: { id: number; x: number; y: number; lastX: number; lastY: number; travelled: number } | null = null;
    const touches = new Set<number>();

    const isUi = (target: EventTarget | null) => target instanceof Element && Boolean(target.closest("[data-vr-ui]"));
    /** Keeps an event from the viewer underneath; true when it was ours to handle. */
    const claim = (event: Event): boolean => {
      if (isUi(event.target)) return false;
      event.stopImmediatePropagation();
      return true;
    };

    const onDown = (event: PointerEvent) => {
      if (!claim(event)) return;
      if (event.pointerType === "touch") {
        touches.add(event.pointerId);
        if (touches.size >= 2) {
          stop();
          return;
        }
      }
      press = { id: event.pointerId, x: event.clientX, y: event.clientY, lastX: event.clientX, lastY: event.clientY, travelled: 0 };
      container.setPointerCapture?.(event.pointerId);
    };
    const onMove = (event: PointerEvent) => {
      if (!claim(event)) return;
      if (!press || event.pointerId !== press.id) return;
      const dx = event.clientX - press.lastX;
      const dy = event.clientY - press.lastY;
      press.lastX = event.clientX;
      press.lastY = event.clientY;
      press.travelled += Math.hypot(dx, dy);
      const f = feed.current;
      if (f.source !== "drag") return;
      // The 2D player's convention: dragging pulls the world, so the head turns the other way.
      look = {
        yaw: look.yaw - dx * DRAG_RATE,
        pitch: Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, look.pitch + dy * DRAG_RATE)),
      };
      f.q = quatFromView(look.yaw, look.pitch);
      f.valid = true;
      f.stamp = performance.now();
    };
    const onUp = (event: PointerEvent) => {
      touches.delete(event.pointerId);
      if (!claim(event)) return;
      if (!press || event.pointerId !== press.id) return;
      const wasTap = press.travelled < TAP_SLOP;
      press = null;
      container.releasePointerCapture?.(event.pointerId);
      if (!wasTap) return;
      // The Cardboard's button is a screen tap: it means "this one", if the gaze is on one.
      if (selectRef.current?.()) return;
      showHud();
    };
    const onCancel = (event: PointerEvent) => {
      touches.delete(event.pointerId);
      if (press?.id === event.pointerId) press = null;
      claim(event);
    };
    // Double-click, wheel and long-press menus mean nothing in the box and something below it.
    const onOther = (event: Event) => {
      if (claim(event) && event.type === "contextmenu") event.preventDefault();
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") stop();
    };

    const capture = { capture: true } as const;
    container.addEventListener("pointerdown", onDown, capture);
    container.addEventListener("pointermove", onMove, capture);
    container.addEventListener("pointerup", onUp, capture);
    container.addEventListener("pointercancel", onCancel, capture);
    for (const type of ["dblclick", "wheel", "contextmenu"]) container.addEventListener(type, onOther, capture);
    window.addEventListener("keydown", onKey);
    return () => {
      container.removeEventListener("pointerdown", onDown, capture);
      container.removeEventListener("pointermove", onMove, capture);
      container.removeEventListener("pointerup", onUp, capture);
      container.removeEventListener("pointercancel", onCancel, capture);
      for (const type of ["dblclick", "wheel", "contextmenu"]) container.removeEventListener(type, onOther, capture);
      window.removeEventListener("keydown", onKey);
    };
  }, [active, getContainer, stop, showHud]);

  // Back button, back swipe, and falling out of fullscreen all mean "take me out".
  useEffect(() => {
    if (!active) return;
    const onPop = () => {
      if (!(window.history.state as { biosphereVr?: boolean } | null)?.biosphereVr) stop(true);
    };
    const onFullscreen = () => {
      setFullscreen(Boolean(document.fullscreenElement));
      if (enteredFullscreen.current && !document.fullscreenElement) {
        enteredFullscreen.current = false;
        stop();
      }
    };
    window.addEventListener("popstate", onPop);
    document.addEventListener("fullscreenchange", onFullscreen);
    return () => {
      window.removeEventListener("popstate", onPop);
      document.removeEventListener("fullscreenchange", onFullscreen);
    };
  }, [active, stop]);

  // Keep the screen awake: a VR box hides the phone, and a dimmed screen in a headset is a
  // visitor taking the phone out to find out why the room went dark.
  useEffect(() => {
    if (!active || typeof navigator === "undefined" || !("wakeLock" in navigator)) return;
    let sentinel: WakeLockSentinel | null = null;
    let live = true;
    const acquire = () => {
      navigator.wakeLock
        .request("screen")
        .then((lock) => {
          if (live) sentinel = lock;
          else void lock.release();
        })
        .catch(() => undefined);
    };
    const onVisibility = () => {
      if (document.visibilityState === "visible") acquire();
    };
    acquire();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      live = false;
      document.removeEventListener("visibilitychange", onVisibility);
      void sentinel?.release().catch(() => undefined);
    };
  }, [active]);

  useEffect(
    () => () => {
      if (hudTimer.current) clearTimeout(hudTimer.current);
      stop();
    },
    [stop],
  );

  const leave = useCallback(() => stop(), [stop]);

  return {
    phase,
    active,
    // The split view needs the phone on its side.
    portrait: active && !debug && portraitNow,
    hud,
    fullscreen,
    lensId,
    lens,
    setLensId,
    message,
    feed,
    selectRef,
    request,
    start,
    stop: leave,
    cancelGuide,
    showHud,
  };
}
