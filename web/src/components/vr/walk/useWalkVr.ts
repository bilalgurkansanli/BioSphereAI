"use client";

/**
 * The walkthrough's VR button: the same three routes as the 360 (`capabilities.ts`) — a WebXR
 * headset, a phone in a VR box, or a new tab where the frame refuses both.
 *
 * Main-bundle code, and small for it: the locomotion, the eye rendering and the interface are
 * the lazy `WalkVrRuntime`, fetched once the button exists so a press never waits on it. What
 * has to be here is the start of a headset session (`xrSession.ts`), because it must happen
 * inside the click.
 */

import { useCallback, useEffect, useState } from "react";
import type { WalkEngineApi } from "@/components/walk/useWalkEngine";
import { handoffUrl, useVrSupport, type VrRoute } from "../capabilities";
import { useCardboard, type CardboardController } from "../useCardboard";
import { endWalkXr, startWalkXr, walkXrAvailable, type WalkXrSession } from "./xrSession";

export type { WalkXrSession } from "./xrSession";

export type WalkVrMode = "off" | "xr" | "cardboard";

export interface WalkVrMessages {
  denied: string;
  noSensor: string;
  failed: string;
  /** The engine came up on a device that cannot present VR (WebGPU without XR binding). */
  unavailable: string;
}

export interface UseWalkVrOptions {
  engine: React.RefObject<WalkEngineApi | null>;
  container: () => HTMLElement | null;
  messages: WalkVrMessages;
}

export interface WalkVrController {
  route: VrRoute | null;
  mode: WalkVrMode;
  cardboard: CardboardController;
  debug: boolean;
  message: string | null;
  handoff: string | null;
  prompt: boolean;
  /** The running headset session's rig and space, for the runtime. */
  xr: WalkXrSession | null;
  enter: () => void;
  exit: () => void;
  closeHandoff: () => void;
  dismissPrompt: () => void;
}

const MESSAGE_MS = 6500;

export function useWalkVr({ engine, container, messages }: UseWalkVrOptions): WalkVrController {
  const support = useVrSupport();
  const cardboard = useCardboard({
    container,
    debug: support.query.debugCardboard,
    lensOverride: support.query.lensOverride,
    messages: { denied: messages.denied, noSensor: messages.noSensor },
  });
  const { request: requestCardboard, stop: stopCardboard } = cardboard;
  const [xr, setXr] = useState<WalkXrSession | null>(null);
  const [starting, setStarting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [handoff, setHandoff] = useState<string | null>(null);
  const [promptDismissed, setPromptDismissed] = useState(false);

  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(() => setMessage(null), MESSAGE_MS);
    return () => clearTimeout(timer);
  }, [message]);

  useEffect(() => {
    if (support.route !== "xr" && support.route !== "cardboard") return;
    const load = () => {
      void import("./WalkVrRuntime");
      void import("../ui/VrOverlays");
    };
    const idle = (window as Window & { requestIdleCallback?: (cb: () => void) => number }).requestIdleCallback;
    if (idle) idle(load);
    else setTimeout(load, 1200);
  }, [support.route]);

  const openHandoff = useCallback(() => {
    const area = engine.current?.vr.activeKey() ?? null;
    setHandoff(handoffUrl(window.location.href, { mode: "walk", scene: area, yaw: null, pitch: null }));
  }, [engine]);

  const startXr = useCallback(() => {
    const host = engine.current?.vr;
    if (!host) {
      setMessage(messages.failed);
      return;
    }
    if (!walkXrAvailable(host)) {
      setMessage(messages.unavailable);
      return;
    }
    if (starting) return;
    setStarting(true);
    const asked = startWalkXr(host, {
      started: (session) => {
        setStarting(false);
        setXr(session);
      },
      failed: (error) => {
        setStarting(false);
        if (error instanceof DOMException && error.name === "SecurityError" && support.env?.embedded) openHandoff();
        else setMessage(messages.failed);
      },
      ended: () => setXr(null),
    });
    if (!asked) setStarting(false);
  }, [engine, starting, messages.failed, messages.unavailable, support.env, openHandoff]);

  const enter = useCallback(() => {
    setPromptDismissed(true);
    if (support.route === "xr") startXr();
    else if (support.route === "cardboard") requestCardboard();
    else if (support.route === "handoff") openHandoff();
  }, [support.route, startXr, requestCardboard, openHandoff]);

  const exit = useCallback(() => {
    const host = engine.current?.vr;
    if (host) endWalkXr(host);
    stopCardboard();
  }, [engine, stopCardboard]);

  const mode: WalkVrMode = xr ? "xr" : cardboard.active ? "cardboard" : "off";
  const prompt =
    support.query.handoff && !promptDismissed && mode === "off" && cardboard.phase === "off" && (support.route === "xr" || support.route === "cardboard");

  return {
    route: support.route,
    mode,
    cardboard,
    debug: support.query.debugCardboard,
    message: message ?? cardboard.message,
    handoff,
    prompt,
    xr,
    enter,
    exit,
    closeHandoff: () => setHandoff(null),
    dismissPrompt: () => setPromptDismissed(true),
  };
}
