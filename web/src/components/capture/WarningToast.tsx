"use client";

/**
 * WarningToast — one correction, never a stack.
 *
 * A person mid-rotation cannot triage. Builder C ranks by WARNING_PRIORITY and P passes
 * `warnings[0]`; this component renders exactly what it is given and never queues, stacks
 * or reorders.
 *
 * Four properties the brief asks for, and how each is paid for:
 *
 *  UNMISSABLE — severity is carried by three different border TREATMENTS (hairline /
 *  dashed / corner brackets) and by nine distinct icon SILHOUETTES, so the message survives
 *  greyscale, colour blindness, glare and a 200 ms glance. Colour is the last channel, not
 *  the first.
 *
 *  NON-BLOCKING — `pointer-events-none` throughout, and nothing here can interrupt the
 *  recording. Per contract decision D7 a warning is a takeover layer, not a state: it never
 *  costs the row its earned progress.
 *
 *  HONEST — the card says only what P's copy says. It never invents a severity, and an
 *  `info` warning is explicitly denied a haptic, because vibrating for furniture is how a
 *  user learns to ignore the vibration that matters.
 *
 *  NEVER NAGGING — the haptic fires once per warning OCCURRENCE, keyed on `code` + `since`.
 *  Builder C hands back a fresh object every tick; keying on identity would buzz sixty times
 *  a second. A genuinely new occurrence has a new `since` and does buzz again, which is the
 *  honest behaviour.
 *
 * Haptics are Android-only — iOS Safari ships no vibration API — so they are strictly an
 * additional channel and never the sole carrier of anything.
 */

import { useEffect } from "react";
import { Aperture, Compass, Moon, Orbit, Snail, Sun } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Warning, WarningIcon, WarningToastProps } from "@/lib/capture/types";

const TOAST_CSS = `
@keyframes bsw-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }
.bsw-pulse { animation: bsw-pulse 0.85s ease-in-out infinite; }
@keyframes bsw-in { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: none; } }
.bsw-in { animation: bsw-in 260ms var(--ease-out-expo) both; }
`;

// -------------------------------------------------------------------------------------
// Silhouettes
// -------------------------------------------------------------------------------------

/**
 * Three icons are hand-drawn rather than borrowed. `foot-crossed`, `level` and `rotate` all
 * have to be readable as a shape at 22 px in sunlight, which a thin generic glyph is not;
 * these are drawn to be distinguishable from each other in silhouette alone.
 */
function FootCrossedIcon() {
  return (
    <svg viewBox="0 0 24 24" width={22} height={22} fill="none" aria-hidden focusable="false">
      <g stroke="currentColor" strokeWidth={1.9} strokeLinecap="round" strokeLinejoin="round">
        <rect x={4} y={3.5} width={5} height={8.5} rx={2.5} />
        <circle cx={6.5} cy={14.8} r={1.9} />
        <rect x={14} y={7} width={5} height={8.5} rx={2.5} />
        <circle cx={16.5} cy={18.3} r={1.9} />
      </g>
      <line x1={2.6} y1={21.4} x2={21.4} y2={2.6} stroke="currentColor" strokeWidth={2.6} strokeLinecap="round" />
    </svg>
  );
}

function LevelIcon() {
  // Tilted on purpose: a level vial drawn level would be the "all good" icon.
  return (
    <svg viewBox="0 0 24 24" width={22} height={22} fill="none" aria-hidden focusable="false">
      <g
        stroke="currentColor"
        strokeWidth={1.9}
        strokeLinecap="round"
        transform="rotate(-14 12 12)"
      >
        <rect x={2} y={8.5} width={20} height={7} rx={3.4} />
        <line x1={9.4} y1={9.2} x2={9.4} y2={14.8} />
        <line x1={14.6} y1={9.2} x2={14.6} y2={14.8} />
        <circle cx={16.6} cy={12} r={1.7} fill="currentColor" stroke="none" />
      </g>
    </svg>
  );
}

function RotateIcon() {
  return (
    <svg viewBox="0 0 24 24" width={22} height={22} fill="none" aria-hidden focusable="false">
      <g stroke="currentColor" strokeWidth={1.9} strokeLinecap="round" strokeLinejoin="round">
        <rect x={8.5} y={2} width={7} height={12} rx={1.8} />
        <line x1={10.8} y1={4.4} x2={13.2} y2={4.4} />
        <path d="M4.4 16.2a8.4 8.4 0 0 0 15.2 0" />
        <path d="M4.4 16.2 6.9 15.1M4.4 16.2 5.6 18.7" />
      </g>
    </svg>
  );
}

function IconFor({ icon }: { icon: WarningIcon }) {
  switch (icon) {
    case "foot-crossed":
      return <FootCrossedIcon />;
    case "snail":
      return <Snail size={22} strokeWidth={1.9} aria-hidden />;
    case "sun":
      return <Sun size={22} strokeWidth={1.9} aria-hidden />;
    case "moon":
      return <Moon size={22} strokeWidth={1.9} aria-hidden />;
    case "orbit":
      return <Orbit size={22} strokeWidth={1.9} aria-hidden />;
    case "lens":
      return <Aperture size={22} strokeWidth={1.9} aria-hidden />;
    case "compass":
      return <Compass size={22} strokeWidth={1.9} aria-hidden />;
    case "level":
      return <LevelIcon />;
    case "rotate":
      return <RotateIcon />;
  }
}

// -------------------------------------------------------------------------------------
// Haptics
// -------------------------------------------------------------------------------------

/**
 * `navigator.vibrate` is absent on every iPhone, can be present but silently ignored inside
 * an iframe or without a prior user gesture, and is specified to throw on a malformed
 * pattern. It is a bonus channel; it must never be able to take the capture screen down.
 */
function tryVibrate(pattern: readonly number[]): void {
  if (typeof navigator === "undefined" || typeof navigator.vibrate !== "function") return;
  if (pattern.length === 0) return;
  try {
    navigator.vibrate(pattern.slice());
  } catch {
    // Ignored on purpose: a blocked or unsupported vibration is not an error worth surfacing.
  }
}

// -------------------------------------------------------------------------------------
// Component
// -------------------------------------------------------------------------------------

export function WarningToast({ warning, title, body, reducedMotion, className }: WarningToastProps) {
  if (!warning) return null;
  return (
    // `since` distinguishes a fresh occurrence of a condition from the same one still
    // holding, so the key remounts the card exactly when — and only when — something new has
    // happened. That single fact drives both the entrance animation and the haptic, with no
    // state, no timer and no cross-render bookkeeping to get out of step.
    <ToastCard
      key={`${warning.code}:${warning.since}`}
      warning={warning}
      title={title}
      body={body}
      reducedMotion={reducedMotion}
      className={className}
    />
  );
}

interface ToastCardProps {
  warning: Warning;
  title: string;
  body: string | null;
  reducedMotion: boolean;
  className?: string;
}

function ToastCard({ warning, title, body, reducedMotion, className }: ToastCardProps) {
  useEffect(() => {
    // `info` is furniture and gets no haptic: a buzz for a three-degree tilt teaches the user
    // to ignore the buzz that means they are walking.
    if (warning.severity !== "info") tryVibrate(warning.haptic);
    // Mount IS the occurrence — see the key above. An empty dependency list is the point.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const severity = warning.severity;
  const accent = severity === "stop" ? "var(--danger)" : severity === "warn" ? "var(--amber)" : "var(--bio)";

  return (
    /* No exit animation, deliberately. A correction that vanishes the instant the condition
       clears is the fastest possible confirmation that the user just fixed it; a card that
       lingers for 200 ms of politeness is 200 ms of stale instruction while they are turning.
       It also means no held copy, no dismissal race and no ghost card. */
    <div className={cn("pointer-events-none w-full max-w-sm", !reducedMotion && "bsw-in", className)}>
      <style>{TOAST_CSS}</style>

      <div
        // A stop is assertive because the user is not looking at the screen; everything else
        // is polite so it cannot interrupt the row's own announcement.
        role={severity === "stop" ? "alert" : "status"}
        aria-live={severity === "stop" ? "assertive" : "polite"}
        className={cn(
          "relative flex items-start gap-3 rounded-2xl px-3.5 py-3",
          severity === "stop" ? "glass-strong" : "glass",
        )}
        style={{
          // Border TREATMENT is the primary channel: 1 px continuous, 2 px dashed, or 3 px
          // solid under corner brackets. Three different weights and patterns, all legible
          // with no colour at all. Whole pixels on purpose — a 2.5 px border computes down
          // to 2 px on a 1x display, which would have collapsed `stop` onto `warn`'s weight.
          border: `${severity === "stop" ? 3 : severity === "warn" ? 2 : 1}px ${severity === "warn" ? "dashed" : "solid"} ${accent}`,
        }}
      >
        {severity === "stop" ? (
          // Corner brackets as real boxes rather than an SVG: path data has no `calc()`, and
          // the card's height is set by however many lines P's copy wraps to.
          <span
            aria-hidden
            className={cn("pointer-events-none absolute -inset-0.5", !reducedMotion && "bsw-pulse")}
          >
            <span className="absolute left-0 top-0 size-5 rounded-tl-2xl border-l-4 border-t-4" style={{ borderColor: "var(--danger)" }} />
            <span className="absolute right-0 top-0 size-5 rounded-tr-2xl border-r-4 border-t-4" style={{ borderColor: "var(--danger)" }} />
            <span
              className="absolute bottom-0 left-0 size-5 rounded-bl-2xl border-b-4 border-l-4"
              style={{ borderColor: "var(--danger)" }}
            />
            <span
              className="absolute bottom-0 right-0 size-5 rounded-br-2xl border-b-4 border-r-4"
              style={{ borderColor: "var(--danger)" }}
            />
          </span>
        ) : null}

        <span className="mt-px shrink-0" style={{ color: accent }}>
          <IconFor icon={warning.icon} />
        </span>

        <div className="min-w-0 flex-1">
          <p className="text-[14px] font-semibold leading-tight tracking-tight text-ink">{title}</p>
          {body ? <p className="mt-0.5 text-[12px] leading-snug text-ink-muted">{body}</p> : null}
        </div>
      </div>
    </div>
  );
}

export default WarningToast;
