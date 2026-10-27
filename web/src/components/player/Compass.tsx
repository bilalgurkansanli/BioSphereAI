"use client";

/**
 * Yaw readout. The dial is rotated by writing a transform to a ref every frame — routing
 * this through React state would re-render the whole toolbar 120 times a second to move
 * one tick mark.
 */

import { cn } from "@/lib/utils";

export interface CompassProps {
  /** Player writes `rotate(-yaw)` onto this element each frame. */
  dialRef: React.RefObject<HTMLDivElement | null>;
  /** Snaps the view back to the panorama's centre column. */
  onRecentre?: () => void;
  label: string;
  className?: string;
}

export function Compass({ dialRef, onRecentre, label, className }: CompassProps) {
  return (
    <button
      type="button"
      data-player-ui
      onClick={onRecentre}
      aria-label={label}
      title={label}
      className={cn(
        "relative grid h-8 w-8 shrink-0 place-items-center rounded-full border border-line",
        "transition-colors duration-200 ease-[var(--ease-out-expo)]",
        "hover:border-line-strong hover:bg-white/[0.04] active:bg-white/[0.07]",
        className,
      )}
    >
      {/* Rotating ring: ticks belong to the world, so they move as the view turns. */}
      <div ref={dialRef} className="absolute inset-0 will-change-transform" style={{ transform: "rotate(0deg)" }}>
        <svg viewBox="0 0 32 32" className="h-full w-full" aria-hidden focusable="false">
          <line x1="16" y1="3.5" x2="16" y2="7.5" stroke="var(--bio)" strokeWidth="1.75" strokeLinecap="round" />
          <line x1="28.5" y1="16" x2="26" y2="16" stroke="var(--ink-faint)" strokeWidth="1.25" strokeLinecap="round" />
          <line x1="16" y1="28.5" x2="16" y2="26" stroke="var(--ink-faint)" strokeWidth="1.25" strokeLinecap="round" />
          <line x1="3.5" y1="16" x2="6" y2="16" stroke="var(--ink-faint)" strokeWidth="1.25" strokeLinecap="round" />
        </svg>
      </div>

      {/* Fixed marker: this is where you are looking, always straight up. */}
      <svg viewBox="0 0 32 32" className="absolute inset-0 h-full w-full" aria-hidden focusable="false">
        <path d="M16 11.5 L12.9 18.4 L16 16.8 L19.1 18.4 Z" fill="var(--ink)" opacity="0.92" />
      </svg>
    </button>
  );
}
