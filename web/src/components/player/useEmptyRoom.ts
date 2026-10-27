"use client";

/**
 * The furniture toggle's state, and the one non-obvious thing about it: it is a *view*, not
 * an edit.
 *
 * Nothing here changes the scene. The engine writes the emptied panorama into its own
 * directory beside the original and hands back a second ladder; this hook holds which of the
 * two the player is currently showing. Turning it off is therefore instant and lossless, and
 * a reload comes back to the real room — which is the right default for a tool whose whole
 * job is to answer "what would this look like empty?".
 *
 * ── WHY THE RESULT IS CACHED HERE AS WELL AS ON THE SERVER ────────────────────────────
 * The engine caches per (scene, strength) and answers a warm request off disk, but that is
 * still a round trip, and this is a toggle: the first thing anyone does with it is press it
 * twice. Keeping the resolved ladder per scene id means the second press is a state change
 * and nothing else, and stepping into another room and back does not re-ask.
 *
 * ── WHY IT DOES NOT WALK THE LADDER AGAIN ─────────────────────────────────────────────
 * `useProgressiveTexture` re-runs when the panorama it is given changes, and left alone it
 * would start again from the 512 px preview — so a toggle would blur the room and then
 * sharpen it, which reads as a quality drop rather than as furniture disappearing. The
 * caller pairs the swap with `forceHighestQuality()`, which makes the loader jump straight
 * to the best rung; `PanoSphere` then cross-fades between two full-resolution textures and
 * the furniture simply dissolves.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { BioSphereError, api } from "@/lib/api";
import type { EmptyRoomResponse, PanoramaLevels } from "@/lib/types";

export type EmptyRoomPhase = "idle" | "working" | "ready" | "empty" | "failed";

export interface EmptyRoomState {
  /** The staged ladder to show, or null when the real room is on screen. */
  levels: PanoramaLevels | null;
  active: boolean;
  phase: EmptyRoomPhase;
  /** Set once a run has finished, so the UI can be honest about how much it found. */
  result: EmptyRoomResponse | null;
  /** Localised, ready to render. Null unless `phase === "failed"` or `"empty"`. */
  message: string | null;
  toggle: () => void;
}

/** The engine's own code for "there is nothing here that separates from the room". */
const NOTHING_FOUND = "nothing_to_remove";

/** What a run is doing, and which scene it belongs to. */
interface Run {
  scene: string;
  phase: Exclude<EmptyRoomPhase, "idle" | "ready">;
  message: string | null;
}

export function useEmptyRoom(
  sceneId: string | null,
  locale: "tr" | "en",
  copy: { empty: string; failed: string },
): EmptyRoomState {
  // Everything below is DERIVED FROM THE SCENE ID rather than reset by an effect when it
  // changes. Walking into another room has to drop back to the real panorama — carrying
  // "furniture hidden" into a room that was never emptied would light the toggle with
  // nothing behind it — and an effect that calls setState to achieve that renders twice on
  // every room change, in a component that also owns a WebGL canvas. Storing which scene a
  // piece of state belongs to costs one string and makes the reset free.
  const [shownFor, setShownFor] = useState<string | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  // Keyed by scene id and held in state rather than in a ref: what is on screen depends on
  // it, so it is exactly the kind of value React has to be told about.
  const [results, setResults] = useState<Record<string, EmptyRoomResponse>>({});

  const inFlight = useRef<AbortController | null>(null);

  useEffect(
    () => () => {
      inFlight.current?.abort();
      inFlight.current = null;
    },
    [],
  );

  const cached = sceneId ? (results[sceneId] ?? null) : null;
  const active = sceneId !== null && shownFor === sceneId && cached !== null;
  const current = run && run.scene === sceneId ? run : null;
  const phase: EmptyRoomPhase = current?.phase ?? (cached ? "ready" : "idle");
  const message = current?.message ?? null;

  const toggle = useCallback(() => {
    if (!sceneId) return;

    if (active) {
      setShownFor(null);
      setRun(null);
      return;
    }

    if (cached) {
      setShownFor(sceneId);
      setRun(null);
      return;
    }

    if (phase === "working") return;
    const controller = new AbortController();
    inFlight.current?.abort();
    inFlight.current = controller;
    setRun({ scene: sceneId, phase: "working", message: null });

    void api.ai
      .emptyRoom({ scene_id: sceneId }, controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return;
        setResults((previous) => ({ ...previous, [sceneId]: response }));
        setShownFor(sceneId);
        setRun(null);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const nothing = error instanceof BioSphereError && error.detail.code === NOTHING_FOUND;
        setRun({
          scene: sceneId,
          phase: nothing ? "empty" : "failed",
          message:
            error instanceof BioSphereError
              ? nothing
                ? copy.empty
                : error.localised(locale)
              : copy.failed,
        });
      })
      .finally(() => {
        if (inFlight.current === controller) inFlight.current = null;
      });
  }, [sceneId, active, cached, phase, locale, copy.empty, copy.failed]);

  return {
    levels: active && cached ? cached.levels : null,
    active,
    phase,
    result: cached,
    message,
    toggle,
  };
}
