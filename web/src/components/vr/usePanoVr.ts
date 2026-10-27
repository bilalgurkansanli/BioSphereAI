"use client";

/**
 * The 360 player's VR button, end to end: which route it takes, starting and ending a WebXR
 * session on the player's own renderer, the phone headset (`useCardboard`), and the new-tab
 * handoff for a frame that cannot do either.
 *
 * This file is in the player's main bundle and is kept small for that reason: it imports no
 * scene code. What is drawn in the headset (`pano/VrPanoStage`) and the guide and overlays
 * (`ui/VrOverlays`) are separate chunks, fetched when the button first becomes available so a
 * press never waits on the network — and never at all on a desktop without a headset.
 *
 * ── STARTING A SESSION ────────────────────────────────────────────────────────────────────
 * `requestSession` must be called inside the click, so it is the first thing `enter` does.
 * The renderer is configured only once the session exists:
 *
 * - `local` reference space. A panorama has no floor to stand on and no room to walk; `local`
 *   is the one every immersive session grants, and it never triggers a floor-calibration
 *   prompt.
 * - `layers` is requested as an optional feature. three uses a projection layer whenever the
 *   browser exposes `XRWebGLBinding.createProjectionLayer`, and on Quest that call throws unless
 *   the session was granted `layers` — so without the feature the session starts and then dies
 *   inside `setSession`. three's own VRButton requests it for the same reason.
 * - Framebuffer scale 1 and foveation OFF (`xrFramebufferScale`): a sphere is free to draw,
 *   and fixed foveation blurs exactly the periphery a glancing eye looks at.
 * - Tone mapping stays `NoToneMapping` and output stays sRGB: the renderer is the player's own,
 *   configured for a photograph, and XR changes nothing about that.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type * as THREE from "three";
import { handoffUrl, useVrSupport, xrFramebufferScale, type VrRoute } from "./capabilities";
import { useCardboard, type CardboardController } from "./useCardboard";

export type VrMode = "off" | "xr" | "cardboard";

export interface PanoVrMessages {
  denied: string;
  noSensor: string;
  failed: string;
}

export interface UsePanoVrOptions {
  renderer: () => THREE.WebGLRenderer | null;
  container: HTMLElement | null;
  messages: PanoVrMessages;
  /** Where the visitor is now, for the new tab: `scene`, `yaw`, `pitch`. */
  handoffParams: () => Record<string, string | number | null>;
}

export interface PanoVrController {
  ready: boolean;
  route: VrRoute | null;
  mode: VrMode;
  cardboard: CardboardController;
  debug: boolean;
  message: string | null;
  /** The new-tab URL while the handoff sheet is open. */
  handoff: string | null;
  /** Arrived with `?vr=1`: a big "enter VR" button, because starting needs a tap. */
  prompt: boolean;
  enter: () => void;
  exit: () => void;
  closeHandoff: () => void;
  dismissPrompt: () => void;
}

const MESSAGE_MS = 6500;

export function usePanoVr(options: UsePanoVrOptions): PanoVrController {
  const { renderer, container, messages, handoffParams } = options;
  const support = useVrSupport();
  const getContainer = useCallback(() => container, [container]);
  const cardboard = useCardboard({
    container: getContainer,
    debug: support.query.debugCardboard,
    lensOverride: support.query.lensOverride,
    messages: { denied: messages.denied, noSensor: messages.noSensor },
  });
  const { request: requestCardboard, stop: stopCardboard } = cardboard;

  const [xrOn, setXrOn] = useState(false);
  const [xrMessage, setXrMessage] = useState<string | null>(null);
  const [handoff, setHandoff] = useState<string | null>(null);
  const [promptDismissed, setPromptDismissed] = useState(false);
  const session = useRef<XRSession | null>(null);
  const starting = useRef(false);

  useEffect(() => {
    if (!xrMessage) return;
    const timer = setTimeout(() => setXrMessage(null), MESSAGE_MS);
    return () => clearTimeout(timer);
  }, [xrMessage]);

  // Warm the chunks once there is a button to press, so the press never waits on them.
  useEffect(() => {
    if (support.route !== "xr" && support.route !== "cardboard") return;
    const load = () => {
      void import("./pano/VrPanoStage");
      void import("./ui/VrOverlays");
    };
    const idle = (window as Window & { requestIdleCallback?: (cb: () => void) => number }).requestIdleCallback;
    if (idle) idle(load);
    else setTimeout(load, 1200);
  }, [support.route]);

  const openHandoff = useCallback(() => {
    setHandoff(handoffUrl(window.location.href, { ...handoffParams(), mode: "pano" }));
  }, [handoffParams]);

  const startXr = useCallback(() => {
    const xr = typeof navigator !== "undefined" ? navigator.xr : undefined;
    const gl = renderer();
    if (!xr || !gl) {
      setXrMessage(messages.failed);
      return;
    }
    if (session.current || starting.current) return;
    starting.current = true;

    const fail = (error: unknown) => {
      starting.current = false;
      const name = error instanceof DOMException ? error.name : "";
      if (name === "SecurityError" && support.env?.embedded) openHandoff();
      else setXrMessage(messages.failed);
    };

    let request: Promise<XRSession>;
    try {
      // First thing in the click: see the header.
      request = xr.requestSession("immersive-vr", { optionalFeatures: ["local-floor", "hand-tracking", "layers"] });
    } catch (error) {
      fail(error);
      return;
    }

    void request.then(async (granted) => {
      session.current = granted;
      granted.addEventListener(
        "end",
        () => {
          session.current = null;
          setXrOn(false);
        },
        { once: true },
      );
      try {
        gl.xr.enabled = true;
        gl.xr.setReferenceSpaceType("local");
        gl.xr.setFramebufferScaleFactor(xrFramebufferScale(navigator.userAgent, "panorama"));
        gl.xr.setFoveation(0);
        setXrOn(true);
        await gl.xr.setSession(granted);
        starting.current = false;
      } catch (error) {
        session.current = null;
        setXrOn(false);
        void granted.end().catch(() => undefined);
        fail(error);
      }
    }, fail);
  }, [renderer, messages.failed, support.env, openHandoff]);

  const enter = useCallback(() => {
    setPromptDismissed(true);
    switch (support.route) {
      case "xr":
        startXr();
        break;
      case "cardboard":
        requestCardboard();
        break;
      case "handoff":
        openHandoff();
        break;
      default:
        break;
    }
  }, [support.route, startXr, requestCardboard, openHandoff]);

  const exit = useCallback(() => {
    if (session.current) void session.current.end().catch(() => undefined);
    stopCardboard();
  }, [stopCardboard]);

  // A session left running by an unmount (a route change mid-VR) would keep the headset dark.
  useEffect(
    () => () => {
      if (session.current) void session.current.end().catch(() => undefined);
    },
    [],
  );

  const mode: VrMode = xrOn ? "xr" : cardboard.active ? "cardboard" : "off";
  const prompt =
    support.query.handoff && !promptDismissed && mode === "off" && cardboard.phase === "off" && (support.route === "xr" || support.route === "cardboard");

  return {
    ready: support.ready,
    route: support.route,
    mode,
    cardboard,
    debug: support.query.debugCardboard,
    message: xrMessage ?? cardboard.message,
    handoff,
    prompt,
    enter,
    exit,
    closeHandoff: () => setHandoff(null),
    dismissPrompt: () => setPromptDismissed(true),
  };
}
