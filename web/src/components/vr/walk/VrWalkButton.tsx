"use client";

/**
 * The VR button on the walkthrough's control cluster. The same round, outlined shape as the
 * cluster's own buttons (`WalkViewer`'s `RailButton`), so it reads as one more control of the
 * same kind; shown only where VR can work (`capabilities.ts`).
 */

import { Glasses } from "lucide-react";
import { t } from "@/lib/i18n";
import type { Locale } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface VrWalkButtonProps {
  locale: Locale;
  active: boolean;
  onClick: () => void;
}

export function VrWalkButton({ locale, active, onClick }: VrWalkButtonProps) {
  const label = t("player.vr", locale);
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        "grid h-8 w-8 place-items-center rounded-full border transition",
        active ? "border-bio/50 bg-bio/12 text-bio" : "border-line text-ink-muted hover:border-line-strong hover:bg-white/[0.05] hover:text-ink",
      )}
    >
      <Glasses size={14} aria-hidden />
    </button>
  );
}
