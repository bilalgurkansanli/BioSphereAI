"use client";

/**
 * The VR button on the player's bottom-left bar.
 *
 * A dock segment rather than a toolbar icon (where it used to be): the toolbar's controls
 * answer "show me the picture differently" and answer it at once, while VR, like the drift and
 * the measure tool beside it, puts the player into a state that lasts. Labelled like its
 * neighbours — "VR" on the bar, the full "VR gözlükle gez" as its accessible name — because a
 * lone glyph in a row of words reads as a different kind of thing and gets skipped.
 *
 * It only exists where VR can actually work (`capabilities.ts` decides); there is no disabled
 * state to explain, because a desktop without a headset never sees it.
 */

import { Glasses } from "lucide-react";
import { DockButton } from "@/components/player/DockButton";
import { t } from "@/lib/i18n";
import type { Locale } from "@/lib/types";

export interface VrDockSegmentProps {
  locale: Locale;
  /** The guide is open or a session is starting. */
  active: boolean;
  onClick: () => void;
}

export function VrDockSegment({ locale, active, onClick }: VrDockSegmentProps) {
  return (
    <DockButton label={t("player.vr", locale)} caption={t("vr.caption", locale)} active={active} onClick={onClick}>
      <Glasses size={15} aria-hidden />
    </DockButton>
  );
}
