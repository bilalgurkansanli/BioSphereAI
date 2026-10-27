"use client";

/**
 * What the user looks at for the second before the first panorama lands.
 *
 * A spinner would be an admission that we have no idea how long this takes. We do: the
 * bytes are being streamed with a content-length, so the bar is genuinely determinate,
 * and the wireframe globe says what is arriving rather than that something is happening.
 */

import { AnimatePresence, motion } from "motion/react";
import { clamp } from "@/lib/utils";

export interface LoadingVeilProps {
  visible: boolean;
  /** 0..1. Determinate whenever the server sends a content-length, which it does. */
  progress: number;
  label: string;
}

export function LoadingVeil({ visible, progress, label }: LoadingVeilProps) {
  const pct = clamp(progress, 0, 1);

  return (
    <AnimatePresence>
      {visible ? (
        <motion.div
          key="veil"
          initial={{ opacity: 1 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.42, ease: [0.16, 1, 0.3, 1] }}
          className="pointer-events-none absolute inset-0 z-30 grid place-items-center bg-canvas"
        >
          <div className="flex flex-col items-center gap-6 px-8">
            <div className="animate-breathe relative">
              <svg
                viewBox="0 0 120 120"
                className="h-[104px] w-[104px] sm:h-[128px] sm:w-[128px]"
                aria-hidden
                focusable="false"
              >
                <defs>
                  <linearGradient id="veil-stroke" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="var(--bio)" stopOpacity="0.85" />
                    <stop offset="55%" stopColor="var(--sky)" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="var(--violet)" stopOpacity="0.22" />
                  </linearGradient>
                </defs>
                <g fill="none" stroke="url(#veil-stroke)" strokeWidth="0.9">
                  <circle cx="60" cy="60" r="46" />
                  {/* Longitudes as ellipses of decreasing width — a sphere's wireframe
                      in projection, not a globe icon. */}
                  <ellipse cx="60" cy="60" rx="46" ry="46" />
                  <ellipse cx="60" cy="60" rx="30" ry="46" />
                  <ellipse cx="60" cy="60" rx="13" ry="46" />
                  {/* Latitudes, spaced by equal angle so they crowd toward the poles. */}
                  <ellipse cx="60" cy="60" rx="46" ry="0.6" />
                  <ellipse cx="60" cy="36.9" rx="39.8" ry="0.6" />
                  <ellipse cx="60" cy="83.1" rx="39.8" ry="0.6" />
                  <ellipse cx="60" cy="19.5" rx="23" ry="0.6" />
                  <ellipse cx="60" cy="100.5" rx="23" ry="0.6" />
                </g>
              </svg>
            </div>

            <div className="flex w-[min(15rem,60vw)] flex-col items-center gap-2.5">
              <span className="text-[12px] font-medium tracking-[0.14em] text-ink-muted uppercase">{label}</span>
              <div className="h-px w-full overflow-hidden rounded-full bg-line">
                <div
                  className="bio-gradient h-full origin-left rounded-full transition-[width] duration-300 ease-[var(--ease-out-expo)]"
                  style={{ width: `${Math.max(pct * 100, 3)}%` }}
                />
              </div>
            </div>
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
