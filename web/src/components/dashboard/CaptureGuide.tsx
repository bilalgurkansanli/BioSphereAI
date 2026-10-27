"use client";

import { motion } from "motion/react";
import { ui } from "./copy";
import { useLocale } from "@/lib/hooks/useLocale";
import { t, type StringKey } from "@/lib/i18n";
import { cn } from "@/lib/utils";

const STEPS: StringKey[] = ["capture.guide.1", "capture.guide.2", "capture.guide.3", "capture.guide.4"];

/** One sweep of the room, matched by the arc, the cone and the marker dot. */
const SWEEP_SECONDS = 5;

/**
 * Top-down schematic of the capture: the phone stays put in the middle of the room and its
 * view cone sweeps a full circle. Drawn as SVG so it stays crisp at any size, animated
 * through motion so `MotionConfig reducedMotion="user"` can stop it.
 */
function SweepDiagram({ className }: { className?: string }) {
  const loop = { duration: SWEEP_SECONDS, repeat: Infinity, ease: "linear" as const };

  return (
    <svg
      viewBox="0 0 200 184"
      className={cn("w-full", className)}
      role="img"
      aria-label={ui("capture.phone.label")}
      fill="none"
    >
      {/* Room outline, seen from above, with a door gap on the left wall. */}
      <path
        d="M34 24h132a8 8 0 0 1 8 8v120a8 8 0 0 1-8 8H34a8 8 0 0 1-8-8v-38m0-26V32a8 8 0 0 1 8-8"
        stroke="var(--line-strong)"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      {/* Furniture, so the circle reads as a room rather than a diagram. */}
      <rect x="40" y="34" width="34" height="13" rx="3" stroke="var(--line)" />
      <rect x="150" y="60" width="14" height="40" rx="3" stroke="var(--line)" />
      <rect x="60" y="140" width="46" height="12" rx="3" stroke="var(--line)" />

      {/* The path the view sweeps. */}
      <circle cx="100" cy="92" r="52" stroke="var(--line)" strokeDasharray="2.5 6" />

      <motion.circle
        cx="100"
        cy="92"
        r="52"
        stroke="var(--bio)"
        strokeWidth="2.5"
        strokeLinecap="round"
        transform="rotate(-90 100 92)"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={loop}
      />

      <motion.g
        style={{ transformBox: "view-box", transformOrigin: "100px 92px" }}
        initial={{ rotate: 0 }}
        animate={{ rotate: 360 }}
        transition={loop}
      >
        <path d="M100 92 L81.3 45.6 A50 50 0 0 1 118.7 45.6 Z" fill="color-mix(in oklab, var(--bio) 20%, transparent)" />
        <circle cx="100" cy="40" r="3.5" fill="var(--bio)" />
        <rect x="94.5" y="83" width="11" height="18" rx="2.5" fill="var(--surface-3)" stroke="var(--line-strong)" />
        <path d="M97 87.5h6" stroke="var(--bio)" strokeWidth="1.4" strokeLinecap="round" />
      </motion.g>
    </svg>
  );
}

export function CaptureGuide({ className }: { className?: string }) {
  const { locale } = useLocale();

  return (
    <div className={cn("grid gap-8 md:grid-cols-[minmax(0,1fr)_minmax(0,240px)] md:items-center", className)}>
      <ol className="space-y-4">
        {STEPS.map((key, index) => (
          <li key={key} className="flex gap-3.5">
            <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-full border border-line bg-surface-2 font-mono text-[11px] text-bio">
              {index + 1}
            </span>
            <p className="text-[13px] leading-relaxed text-ink-muted">{t(key, locale)}</p>
          </li>
        ))}
      </ol>

      <figure className="mx-auto w-full max-w-[240px]">
        <div className="rounded-lg border border-line bg-canvas-raised p-3">
          <SweepDiagram />
        </div>
        <figcaption className="mt-2.5 text-center text-[11px] text-ink-faint">{ui("capture.phone.label", locale)}</figcaption>
      </figure>
    </div>
  );
}
