"use client";

/**
 * Which kind of VR this browser can do — and therefore whether the VR button is shown at all.
 *
 * ── THREE ROUTES, OR NONE ─────────────────────────────────────────────────────────────────
 * - `xr`: WebXR `immersive-vr` works here. Meta Quest Browser, Safari on Vision Pro, desktop
 *   Chrome with a PC headset. The real thing, and always preferred.
 * - `cardboard`: a phone with motion sensors and no WebXR — every iPhone (Safari has no WebXR
 *   at all), and Android phones, since Chrome dropped its own Cardboard runtime. Split screen,
 *   lens correction, head tracking from `deviceorientation`.
 * - `handoff`: VR would work, but not inside THIS frame. A listing portal that embedded the
 *   tour without `allow="xr-spatial-tracking"` blocks WebXR outright; an iPhone has no element
 *   fullscreen, so Cardboard inside a 560-pixel iframe is a thumbnail in a box; a cross-origin
 *   frame without `gyroscope` permission never receives a single orientation event. The button
 *   then offers the same room in a new tab, where none of that applies.
 * - none: a desktop without a headset. Side-by-side on a monitor, with nothing tracking the
 *   head, is a novelty rather than a feature, so the button is not shown.
 *
 * The decision is a pure function of a description of the environment (`chooseVrRoute`), so
 * every row of that table is a test rather than a guess about somebody's phone.
 *
 * ── THE DEBUG SWITCH ──────────────────────────────────────────────────────────────────────
 * `?vr=cardboard` forces the Cardboard route on any machine and replaces the motion sensors
 * with mouse-drag. It exists so the whole phone path — guide, split view, lens correction,
 * gaze dwell, exit — can be looked at on a desktop. It is harmless in production: it changes
 * nothing for anyone who did not type it.
 */

import { useEffect, useState } from "react";

export type VrRoute = "xr" | "cardboard" | "handoff";

export interface VrQuery {
  /** `?vr=cardboard` — force the phone-headset path, mouse-drag for the head. */
  debugCardboard: boolean;
  /** `?vr=1` — arrived from an embed that could not start VR itself; offer it at once. */
  handoff: boolean;
  /** `?vrlens=k1,k2,ipdMm,screenMm,fovDeg` — see `lens.parseLensOverride`. */
  lensOverride: string | null;
}

export function parseVrQuery(search: string): VrQuery {
  const params = new URLSearchParams(search);
  const vr = params.get("vr");
  return {
    debugCardboard: vr === "cardboard",
    handoff: vr === "1" || vr === "true",
    lensOverride: params.get("vrlens"),
  };
}

/**
 * A browser that runs inside a headset. Only consulted when WebXR is BLOCKED (a policy refused
 * it), to decide whether "open in a new tab" would lead anywhere: on a desktop the same refusal
 * means nothing, and offering a new tab there would be a button to a dead end.
 */
export function isHeadsetBrowser(ua: string): boolean {
  return /OculusBrowser|Quest|PicoBrowser|Pico Neo|Pico 4|Wolvic/i.test(ua) || /\bVR\b/.test(ua);
}

/** iPhone and iPad, including the iPad that reports itself as a Mac. */
export function isAppleTouch(ua: string, maxTouchPoints: number): boolean {
  return /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && maxTouchPoints > 1);
}

/**
 * The XR framebuffer scale, relative to the headset's own recommendation.
 *
 * PANORAMA: 1. A textured sphere costs nothing to draw, so the 360 asks for the runtime's
 * recommended resolution and turns foveation OFF — fixed foveation blurs the periphery of a
 * photograph, which is exactly where the eye goes when it glances without turning the head.
 *
 * SPLATS: fill rate is everything (`walkMath.nextRenderScale`), and a headset draws two eyes
 * at well above a phone's resolution. Starting points by device, refined at run time by the
 * walkthrough's own frame-time loop (`walk/walkXr.ts`):
 *   Quest 3 / 3S / Pro — 0.8 (XR2 Gen 2 class; 2064×2208 per eye at 1.0)
 *   Quest 2            — 0.65 (XR2; 1832×1920 at 1.0)
 *   Quest 1            — 0.5
 *   Pico 4 / Neo 3     — 0.7
 *   anything else      — 0.7, except a "Macintosh" WebXR browser, which in practice is
 *                        Vision Pro: its panels are ~3600 px per eye, so 0.5.
 */
export function xrFramebufferScale(ua: string, content: "panorama" | "splats"): number {
  if (content === "panorama") return 1;
  if (/Quest 3|Quest Pro/i.test(ua)) return 0.8;
  if (/Quest 2/i.test(ua)) return 0.65;
  if (/Quest/i.test(ua)) return 0.5;
  if (/Pico/i.test(ua)) return 0.7;
  if (/Macintosh/.test(ua)) return 0.5;
  return 0.7;
}

/** Everything `chooseVrRoute` looks at, measured once in the browser. */
export interface VrEnvironment {
  ua: string;
  /** Framed by another page. */
  embedded: boolean;
  /** `immersive-vr` as reported by WebXR, or `blocked` when a permissions policy refused it. */
  xr: "supported" | "unsupported" | "blocked" | "absent";
  /** Touch-first and a phone-sized screen: the only device that fits a VR box. */
  phone: boolean;
  /** `DeviceOrientationEvent` exists at all. */
  orientationEvents: boolean;
  /** Element fullscreen is available to THIS document (false on iPhone, false in a frame without allowfullscreen). */
  fullscreen: boolean;
  /** The motion sensors are not refused by a permissions policy (Chrome reports this; others assume yes). */
  sensorsAllowed: boolean;
  apple: boolean;
  debugCardboard: boolean;
}

export function chooseVrRoute(env: VrEnvironment): VrRoute | null {
  if (env.xr === "supported") return "xr";
  if (env.xr === "blocked" && isHeadsetBrowser(env.ua)) return "handoff";
  if (env.debugCardboard) return "cardboard";
  if (!env.phone || !env.orientationEvents) return null;
  // Inside somebody else's page the phone path needs three things the frame may not have:
  // fullscreen (an iPhone never has it), the sensors (a cross-origin frame needs them granted),
  // and on iOS a permission prompt that a cross-origin frame cannot reliably raise.
  if (env.embedded && (!env.fullscreen || env.apple || !env.sensorsAllowed)) return "handoff";
  return "cardboard";
}

/** The width under which a touchscreen is a phone rather than a tablet, CSS px (short side). */
const PHONE_SHORT_SIDE = 600;

type PolicyDocument = Document & { featurePolicy?: { allowsFeature(feature: string): boolean } };

function policyAllows(feature: string): boolean | null {
  try {
    const policy = (document as PolicyDocument).featurePolicy;
    return policy?.allowsFeature ? policy.allowsFeature(feature) : null;
  } catch {
    return null;
  }
}

function isFramed(): boolean {
  try {
    return window.self !== window.top;
  } catch {
    return true;
  }
}

/** Measures the environment. `xr` is filled in asynchronously by `probeXr`. */
export function readEnvironment(query: VrQuery): VrEnvironment {
  const ua = navigator.userAgent;
  const coarse = window.matchMedia?.("(pointer: coarse)").matches ?? false;
  const hover = window.matchMedia?.("(hover: hover)").matches ?? false;
  const short = Math.min(window.screen?.width ?? 9999, window.screen?.height ?? 9999);
  const doc = document as Document & { webkitFullscreenEnabled?: boolean };
  return {
    ua,
    embedded: isFramed(),
    xr: "absent",
    phone: coarse && !hover && short < PHONE_SHORT_SIDE,
    orientationEvents: typeof window.DeviceOrientationEvent !== "undefined",
    fullscreen: Boolean(doc.fullscreenEnabled ?? doc.webkitFullscreenEnabled),
    sensorsAllowed: (policyAllows("gyroscope") ?? true) && (policyAllows("accelerometer") ?? true),
    apple: isAppleTouch(ua, navigator.maxTouchPoints ?? 0),
    debugCardboard: query.debugCardboard,
  };
}

export async function probeXr(): Promise<VrEnvironment["xr"]> {
  const xr = navigator.xr;
  if (!xr?.isSessionSupported) return "absent";
  // A frame without `xr-spatial-tracking` cannot use WebXR at all; Chrome says so up front,
  // and asking anyway only produces a SecurityError.
  if (policyAllows("xr-spatial-tracking") === false) return "blocked";
  try {
    return (await xr.isSessionSupported("immersive-vr")) ? "supported" : "unsupported";
  } catch (error) {
    return error instanceof DOMException && error.name === "SecurityError" ? "blocked" : "unsupported";
  }
}

export interface VrSupport {
  ready: boolean;
  route: VrRoute | null;
  env: VrEnvironment | null;
  query: VrQuery;
}

const NO_QUERY: VrQuery = { debugCardboard: false, handoff: false, lensOverride: null };

/**
 * The route, measured after mount. Re-measured when a headset is plugged in or taken off
 * (`devicechange`), so a PC headset connected after the page loaded still gets its button.
 */
export function useVrSupport(): VrSupport {
  const [support, setSupport] = useState<VrSupport>({ ready: false, route: null, env: null, query: NO_QUERY });

  useEffect(() => {
    if (typeof window === "undefined") return;
    let live = true;
    const query = parseVrQuery(window.location.search);
    const measure = async () => {
      const env = readEnvironment(query);
      env.xr = await probeXr();
      if (live) setSupport({ ready: true, route: chooseVrRoute(env), env, query });
    };
    void measure();
    const xr = navigator.xr;
    const onDeviceChange = () => void measure();
    xr?.addEventListener?.("devicechange", onDeviceChange);
    return () => {
      live = false;
      xr?.removeEventListener?.("devicechange", onDeviceChange);
    };
  }, []);

  return support;
}

/**
 * The same tour in a new tab, at the same room and view, asking to enter VR.
 *
 * Built from this document's own URL — inside an embed that IS the embed page — so it works
 * for both the 360 (`scene`, `yaw`, `pitch`) and the walkthrough (`mode=walk`, `scene` naming
 * the area), and keeps whatever else the portal put on the link.
 */
export function handoffUrl(href: string, params: Record<string, string | number | null | undefined>): string {
  const url = new URL(href);
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") url.searchParams.delete(key);
    else url.searchParams.set(key, typeof value === "number" ? String(Number(value.toFixed(4))) : value);
  }
  url.searchParams.set("vr", "1");
  return url.toString();
}
