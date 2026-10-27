"use client";

/**
 * The walkthrough's VR modes, attached to the running engine for as long as they are mounted.
 * Lazy-loaded (see `useWalkVr`), renders nothing to the DOM: everything it shows is inside the
 * PlayCanvas scene, drawn by `WalkXr` (a headset) or `WalkCardboard` (a phone in a box).
 */

import { useEffect, useRef } from "react";
import type { WalkEngineApi } from "@/components/walk/useWalkEngine";
import type { LensProfile } from "../lens";
import type { OrientationFeed } from "../useCardboard";
import type { WalkXrSession } from "./useWalkVr";
import { WalkCardboard } from "./walkCardboard";
import { WalkXr, type WalkVrCopy } from "./walkXr";

export interface WalkVrRuntimeProps {
  mode: "xr" | "cardboard";
  engine: React.RefObject<WalkEngineApi | null>;
  xr: WalkXrSession | null;
  feed: React.RefObject<OrientationFeed>;
  lens: LensProfile;
  debug: boolean;
  selectRef: React.RefObject<(() => boolean) | null>;
  copy: WalkVrCopy;
  onExit: () => void;
}

export default function WalkVrRuntime({ mode, engine, xr, feed, lens, debug, selectRef, copy, onExit }: WalkVrRuntimeProps) {
  const exit = useRef(onExit);
  useEffect(() => {
    exit.current = onExit;
  });
  const cardboard = useRef<WalkCardboard | null>(null);
  // The lens at attach time; later changes are pushed to the running renderer below.
  const lensAtStart = useRef(lens);

  useEffect(() => {
    const host = engine.current?.vr;
    if (!host) return;
    if (mode === "xr") {
      if (!xr) return;
      const controller = new WalkXr(host, xr.rig, { space: xr.space, scale: xr.scale, copy, onExit: () => exit.current() });
      controller.attach();
      return () => controller.detach();
    }
    const controller = new WalkCardboard(host, { feed, lens: lensAtStart.current, debug, copy, onExit: () => exit.current() });
    controller.attach();
    cardboard.current = controller;
    selectRef.current = () => controller.select();
    return () => {
      selectRef.current = null;
      cardboard.current = null;
      controller.detach();
    };
  }, [mode, engine, xr, feed, debug, selectRef, copy]);

  useEffect(() => {
    lensAtStart.current = lens;
    cardboard.current?.setLens(lens);
  }, [lens]);

  return null;
}
