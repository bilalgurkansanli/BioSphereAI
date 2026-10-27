"use client";

/**
 * Which decorating style the player is currently showing, if any.
 *
 * The sibling of `useEmptyRoom`, and deliberately the same shape: nothing here edits the
 * scene. The engine writes each restyled panorama into its own directory beside the original
 * and hands back a second ladder; this hook holds which one is on screen. Turning it off is
 * instant and lossless, and a reload comes back to the real room — which is the only honest
 * default for a tool whose answer is a proposal rather than a fact about the property.
 *
 * ── WHERE IT DIFFERS FROM `useEmptyRoom`, AND WHY ─────────────────────────────────────
 * Emptying is a TOGGLE with one answer. Staging is a CHOICE among five, so the state is a
 * style id rather than a boolean and the cache is keyed by `scene + style` rather than by
 * scene. Picking a second style while the first is on screen is not "off then on": the old
 * one stays visible until the new one lands, because a two-minute flash of the unfurnished
 * room in between would look like a failure.
 *
 * ── TWO MINUTES IS THE HONEST NUMBER AND THE UI HAS TO CARRY IT ───────────────────────
 * A sphere is sixteen diffusion passes (`ai/restyle_pano`), so the first ask for a style is
 * minutes, not seconds. Everything after it is off the engine's disk cache and instant. That
 * shape — very slow once, then free — is why the result is held here per style for the life
 * of the page as well: stepping into another room and back must not re-ask, and neither must
 * flicking between two styles the user has already waited for.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { BioSphereError, api } from "@/lib/api";
import type { PanoramaLevels, RestyleSceneResponse, RestyleStyleId } from "@/lib/types";

export type SceneRestylePhase = "idle" | "working" | "ready" | "failed";

export interface SceneRestyleState {
  /** The styled ladder to show, or null when the real room is on screen. */
  levels: PanoramaLevels | null;
  /** Which style is showing. Null means the original. */
  activeStyle: RestyleStyleId | null;
  /** Which style is being rendered right now, if any. */
  pendingStyle: RestyleStyleId | null;
  phase: SceneRestylePhase;
  result: RestyleSceneResponse | null;
  message: string | null;
  /** Pass null to go back to the original. */
  select: (style: RestyleStyleId | null) => void;
}

/** `scene + style`, which is what the engine caches on too. */
function cacheKey(sceneId: string, style: RestyleStyleId): string {
  return `${sceneId}:${style}`;
}

export function useSceneRestyle(
  sceneId: string | null,
  locale: "tr" | "en",
  copy: { failed: string },
): SceneRestyleState {
  // Keyed by scene, like `useEmptyRoom`'s `shownFor`, so walking into another room drops back
  // to that room's own photograph without an effect that re-renders the WebGL canvas.
  const [shown, setShown] = useState<{ scene: string; style: RestyleStyleId } | null>(null);
  const [run, setRun] = useState<{ scene: string; style: RestyleStyleId } | null>(null);
  const [failure, setFailure] = useState<{ scene: string; message: string } | null>(null);
  const [results, setResults] = useState<Record<string, RestyleSceneResponse>>({});

  const inFlight = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      inFlight.current?.abort();
      inFlight.current = null;
    },
    [],
  );

  const activeStyle = sceneId && shown?.scene === sceneId ? shown.style : null;
  const pendingStyle = sceneId && run?.scene === sceneId ? run.style : null;
  const message = sceneId && failure?.scene === sceneId ? failure.message : null;
  const current = sceneId && activeStyle ? (results[cacheKey(sceneId, activeStyle)] ?? null) : null;

  const phase: SceneRestylePhase = pendingStyle
    ? "working"
    : message
      ? "failed"
      : activeStyle
        ? "ready"
        : "idle";

  const select = useCallback(
    (style: RestyleStyleId | null) => {
      if (!sceneId) return;
      setFailure(null);

      if (style === null) {
        setShown(null);
        setRun(null);
        inFlight.current?.abort();
        inFlight.current = null;
        return;
      }
      if (activeStyle === style) return;

      const key = cacheKey(sceneId, style);
      if (results[key]) {
        setShown({ scene: sceneId, style });
        setRun(null);
        return;
      }

      const controller = new AbortController();
      inFlight.current?.abort();
      inFlight.current = controller;
      setRun({ scene: sceneId, style });

      void api.restyle
        .scene({ scene_id: sceneId, style }, controller.signal)
        .then((response) => {
          if (controller.signal.aborted) return;
          setResults((previous) => ({ ...previous, [key]: response }));
          setShown({ scene: sceneId, style });
          setRun(null);
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return;
          setRun(null);
          setFailure({
            scene: sceneId,
            message: error instanceof BioSphereError ? error.localised(locale) : copy.failed,
          });
        })
        .finally(() => {
          if (inFlight.current === controller) inFlight.current = null;
        });
    },
    [sceneId, activeStyle, results, locale, copy.failed],
  );

  return {
    levels: current ? current.levels : null,
    activeStyle,
    pendingStyle,
    phase,
    result: current,
    message,
    select,
  };
}
