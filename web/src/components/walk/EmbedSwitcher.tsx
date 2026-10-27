"use client";

/**
 * The embed's two worlds under one URL: the 3D walkthrough and the 360 player.
 *
 * A property can have both. The walkthrough is the richer one and opens first when it is ready;
 * the 360 is one click away in the walkthrough's title strip, and the walkthrough one click
 * away in the player's toolbar. Only one lives at a time: two WebGL contexts on a phone is
 * one too many.
 */

import { useState } from "react";
import { Player } from "@/components/player/Player";
import type { FloorPlan, Locale, Scene, WalkManifest } from "@/lib/types";
import { WalkViewer } from "./WalkViewer";

export type EmbedMode = "walk" | "pano";

export interface EmbedSwitcherProps {
  title: string;
  locale: Locale;
  initialMode: EmbedMode;
  manifest: WalkManifest | null;
  startKey: string | null;
  scenes: Scene[];
  initialSceneId: string | null;
  plan: FloorPlan | null;
  autoRotate: boolean;
  /** The embed token, so the 360 player can offer measuring through the public route. */
  token: string;
}

export function EmbedSwitcher(props: EmbedSwitcherProps) {
  const { title, locale, manifest, startKey, scenes, initialSceneId, plan, autoRotate, token } = props;
  const canWalk = manifest !== null;
  const canPano = scenes.length > 0;
  const [mode, setMode] = useState<EmbedMode>(props.initialMode === "walk" && canWalk ? "walk" : "pano");

  if (mode === "walk" && manifest) {
    return (
      <WalkViewer
        manifest={manifest}
        title={title}
        locale={locale}
        startKey={startKey}
        embedded
        onSwitchTo360={canPano ? () => setMode("pano") : undefined}
        className="h-full w-full"
      />
    );
  }
  return (
    <Player
      scenes={scenes}
      initialSceneId={initialSceneId ?? undefined}
      locale={locale}
      showChrome
      plan={plan}
      autoRotate={autoRotate}
      className="h-full w-full"
      walkAvailable={canWalk}
      onOpenWalk={canWalk ? () => setMode("walk") : undefined}
      measureToken={token}
    />
  );
}
