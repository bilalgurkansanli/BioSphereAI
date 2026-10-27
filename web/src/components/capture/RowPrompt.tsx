"use client";

/**
 * RowPrompt — "you finished the circle; now do it again, looking up."
 *
 * This is the hardest sentence in the product, and the measured cost of getting it wrong is
 * exact: the real capture reached only +45.6 deg of elevation, all three of its coverage
 * gaps sat in the +25..+40 deg band, and it came in at 0.74 covered instead of ~0.90. Users
 * skip the UP row. Everything here is arranged around not letting them.
 *
 * THE ARGUMENT THE COMPONENT MAKES, in the order the eye takes it:
 *
 * 1. THE RAIL SAYS THE JOB WAS ALWAYS THREE CIRCLES. It is on screen from the first second
 *    of the LEVEL row, already showing three segments. When the second one lights up, that
 *    is a plan advancing, not a new demand appearing. A three-segment rail that appeared
 *    only at the transition would read as "and now there's more".
 *
 * 2. THE DIAGRAM SAYS IT IS THE SAME CIRCLE, HIGHER. Three rings at the real ROW_PLAN
 *    elevations, drawn as orbits of one sphere. On a row change a ghost of the ring just
 *    completed physically travels to the new one. Nobody has to be told the new row is the
 *    same motion; they watch it move.
 *
 * 3. THE ARC ON THE COMPLETED RING STAYS ON SCREEN. Earned progress is never cleared away
 *    to make room for the next instruction — the reason to do the third circle is visibly
 *    the two already done.
 *
 * Two words of copy carry the rest, and P has already localised them.
 */

import { useEffect, useRef, useState } from "react";
import { ArrowUp, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { DEFAULT_LENS, ROW_IDS, type RowId, type RowPromptProps } from "@/lib/capture/types";
import { rowTargetPitchDeg } from "@/lib/capture/coverage";

const DEG = Math.PI / 180;

/**
 * Scoped, opacity-only attention pulse.
 *
 * Deliberately NOT the shared `.animate-breathe` utility. That one animates `transform`,
 * and globals.css carries an explicit warning that a running CSS animation outranks an
 * inline style — so it would silently flatten the phone glyph's rotation and the nudge
 * arrow's, which is the bug that already cost the player's hotspot cards once. Animating
 * opacity alone cannot collide with a transform, and it works on SVG children, where a
 * `scale()` with no transform-box would scale about the viewBox origin and fling the
 * element off screen.
 */
const ROW_PROMPT_CSS = `
@keyframes bsr-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.34; } }
.bsr-pulse { animation: bsr-pulse 1.6s ease-in-out infinite; }
@keyframes bsr-ghost {
  from { transform: translateY(var(--bsr-from, 0px)); opacity: 0.95; }
  55% { opacity: 0.85; }
  to { transform: translateY(0); opacity: 0; }
}
.bsr-ghost { animation: bsr-ghost 1100ms var(--ease-out-expo) both; }
`;

// Diagram frame. A 32 deg viewing tilt separates the three rings clearly while keeping
// enough foreshortening that they still read as orbits rather than as stacked hoops.
const DX = 40;
const DY = 33;
const DR = 27;
const TILT = 32 * DEG;
const COS_TILT = Math.cos(TILT);
const SIN_TILT = Math.sin(TILT);

interface RingGeom {
  rx: number;
  ry: number;
  cy: number;
}

/**
 * Rounds a coordinate before it reaches an SVG attribute.
 *
 * NOT cosmetic. `Math.sin`/`Math.cos` are implementation-defined in the last ulp, and the
 * V8 that server-renders this is not the same build as the V8 in the phone's browser. Left
 * raw, the difference shows up as a React hydration mismatch on first paint. Three decimals
 * on this viewBox is far below a pixel at any size the diagram is drawn.
 */
function r3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

/** Orthographic projection of the latitude circle at `elevDeg`, as a screen ellipse. */
function ringGeometry(elevDeg: number): RingGeom {
  const e = elevDeg * DEG;
  return {
    rx: r3(DR * Math.cos(e)),
    ry: r3(DR * Math.cos(e) * SIN_TILT),
    cy: r3(DY - DR * Math.sin(e) * COS_TILT),
  };
}

/**
 * Polyline along a ring, from the point facing the viewer, sweeping the way the user is
 * asked to turn. psi = 0 is the near side; increasing psi moves right across the front,
 * which seen from above is clockwise — the contract's yaw sense.
 */
function ringArc(elevDeg: number, fraction: number, step = 7): string {
  const clamped = Math.max(0, Math.min(1, fraction));
  if (clamped <= 0) return "";
  const { rx, ry, cy } = ringGeometry(elevDeg);
  const total = clamped * 360;
  const n = Math.max(2, Math.ceil(total / step));
  let d = "";
  for (let i = 0; i <= n; i++) {
    const psi = ((total * i) / n) * DEG;
    const x = DX + rx * Math.sin(psi);
    const y = cy + ry * Math.cos(psi);
    d += `${i === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
  }
  return d;
}

/** Where the leading edge of the arc sits, so the phone glyph rides it. */
function ringPoint(elevDeg: number, fraction: number): [number, number] {
  const { rx, ry, cy } = ringGeometry(elevDeg);
  const psi = Math.max(0, Math.min(1, fraction)) * 360 * DEG;
  return [r3(DX + rx * Math.sin(psi)), r3(cy + ry * Math.cos(psi))];
}

export function RowPrompt({ row, rowIndex, rowFractions, title, cue, stuck, reducedMotion, lens, className }: RowPromptProps) {
  // From the live lattice, never from the ROW_PLAN literals: those are the 9:16 reference,
  // and on a 4:3 frame the real rows sit 6 deg lower. A ring drawn where the station is not
  // is worse than no ring.
  const pitchOf = (id: RowId) => rowTargetPitchDeg(id, lens ?? DEFAULT_LENS);
  /**
   * The row transition owns its own choreography rather than taking a prop for it: P simply
   * changes `row`, and everything that has to happen at that moment happens here. One less
   * thing for the flow to sequence, and it cannot get out of step with the rail.
   */
  const [intro, setIntro] = useState<{ from: RowId } | null>(null);
  const previousRow = useRef<RowId>(row);

  useEffect(() => {
    if (previousRow.current === row) return;
    const from = previousRow.current;
    previousRow.current = row;
    setIntro({ from });
    const timer = setTimeout(() => setIntro(null), 1900);
    return () => clearTimeout(timer);
  }, [row]);

  const introducing = intro !== null;
  const activePitch = pitchOf(row);
  const activeFraction = Math.max(0, Math.min(1, rowFractions[row] ?? 0));
  const [phoneX, phoneY] = ringPoint(activePitch, activeFraction);

  const ghostFromCy = intro ? ringGeometry(pitchOf(intro.from)).cy : 0;
  const ghostToCy = ringGeometry(activePitch).cy;

  // Up leans back, down leans forward, level keeps turning: one glyph, three rotations.
  const nudgeRotation = row === "up" ? 0 : row === "down" ? 180 : 90;

  return (
    <div
      className={cn("glass-strong w-full max-w-sm rounded-2xl px-3.5 pb-3 pt-2.5", className)}
      style={{
        transform: introducing && !reducedMotion ? "scale(1.035)" : "scale(1)",
        transition: reducedMotion ? undefined : "transform 620ms var(--ease-spring)",
        willChange: "transform",
      }}
    >
      <style>{ROW_PROMPT_CSS}</style>

      {/* ---- The plan, always three segments ------------------------------------------ */}
      <div className="flex items-center gap-2">
        <div className="flex flex-1 gap-1.5">
          {ROW_IDS.map((id) => {
            const fraction = Math.max(0, Math.min(1, rowFractions[id] ?? 0));
            const done = fraction >= 1;
            const isActive = id === row;
            return (
              <div
                key={id}
                className={cn(
                  "relative h-1.5 flex-1 overflow-hidden rounded-full bg-surface-3",
                  // Ring, not colour: the active segment is the one wearing a halo.
                  isActive && "ring-1 ring-bio/70",
                )}
              >
                {/* A FULL-WIDTH BAR THAT IS SCALED, NOT A BAR WHOSE WIDTH CHANGES.
                    `width` is a layout property: every frame of the 420 ms transition would
                    run layout, paint and raster on the main thread — the thread that is also
                    driving the guidance loop while a camera preview streams and the encoder
                    runs. A station fills roughly two dozen times per row, so that is two
                    dozen quarter-second layout storms per row, each landing exactly when the
                    user is being told they got something right. `scaleX` is compositor-only
                    and moves the same pixels. `transform-origin` keeps it growing from the
                    left, and the parent already clips with `overflow-hidden`. */}
                <span
                  className={cn("absolute inset-y-0 left-0 w-full origin-left rounded-full", done ? "bg-success" : "bg-bio")}
                  style={{
                    transform: `scaleX(${fraction})`,
                    transition: reducedMotion ? undefined : "transform 420ms var(--ease-out-expo)",
                  }}
                />
              </div>
            );
          })}
        </div>

        {/* Numerals carry the same message with no colour and no shape involved. */}
        <span className="shrink-0 font-mono text-[10px] leading-none text-ink-faint tabular-nums">
          {Math.min(3, Math.max(1, Math.round(rowIndex)))}/3
        </span>
      </div>

      <div className="mt-2.5 flex items-center gap-3">
        {/* ---- Three orbits of one sphere --------------------------------------------- */}
        <svg viewBox="0 0 80 66" className="h-[58px] w-[70px] shrink-0" aria-hidden focusable="false">
          {/* Every ring's full circle, faint: the shape of the whole job. */}
          {ROW_IDS.map((id) => {
            const g = ringGeometry(pitchOf(id));
            return (
              <ellipse
                key={id}
                cx={DX}
                cy={g.cy}
                rx={g.rx}
                ry={g.ry}
                fill="none"
                stroke="var(--line-strong)"
                strokeWidth={id === row ? 1 : 0.8}
                strokeDasharray={id === row ? "2.5 2.5" : undefined}
              />
            );
          })}

          {/* Progress, drawn for every row. Work already done is never cleared to make room
              for the next instruction — it is the argument for doing the next one. */}
          {ROW_IDS.map((id) => {
            const fraction = Math.max(0, Math.min(1, rowFractions[id] ?? 0));
            if (fraction <= 0) return null;
            const isActive = id === row;
            return (
              <path
                key={id}
                d={ringArc(pitchOf(id), fraction)}
                fill="none"
                stroke={fraction >= 1 ? "var(--success)" : "var(--bio)"}
                strokeWidth={isActive ? 2.4 : 1.6}
                strokeOpacity={isActive ? 1 : 0.55}
                strokeLinecap="round"
              />
            );
          })}

          {/* The ghost: the ring just finished, physically moving to the ring now being asked
              for. This is the whole explanation, and it uses no words.

              Driven by a CSS animation with the travel distance passed in as a custom
              property, NOT by flipping a state flag across two rAF callbacks. The flag
              version silently did nothing whenever frames were throttled — a backgrounded
              tab, a phone call arriving mid-capture — leaving the ghost parked at the old
              row until it timed out. A keyframe animation starts on its own.

              Suppressed entirely under reduced motion: travel IS the message here, so a
              static ghost would be a decoration with nothing to say. The rail advancing and
              the phone glyph's new tilt still carry it. */}
          {intro && !reducedMotion ? (
            <g
              className="bsr-ghost"
              style={{ "--bsr-from": `${r3(ghostFromCy - ghostToCy)}px` } as React.CSSProperties}
            >
              <ellipse
                cx={DX}
                cy={ghostToCy}
                rx={ringGeometry(activePitch).rx}
                ry={ringGeometry(activePitch).ry}
                fill="none"
                stroke="var(--bio-soft)"
                strokeWidth={2}
              />
            </g>
          ) : null}

          {/* The phone, riding the leading edge of the active row, tilted to that row's real
              elevation so the posture being asked for is visible rather than described. */}
          <g transform={`translate(${phoneX.toFixed(2)} ${phoneY.toFixed(2)}) rotate(${-activePitch})`}>
            <rect x={-3.1} y={-5.2} width={6.2} height={10.4} rx={1.5} fill="var(--surface-3)" stroke="var(--ink)" strokeWidth={0.9} />
            <rect x={-1.9} y={-3.9} width={3.8} height={6.2} rx={0.6} fill="var(--bio)" opacity={0.9} />
          </g>

          {stuck ? (
            <circle
              cx={phoneX}
              cy={phoneY}
              r={10}
              fill="none"
              stroke="var(--amber)"
              strokeWidth={1.4}
              strokeDasharray="3 3"
              className={reducedMotion ? undefined : "bsr-pulse"}
            />
          ) : null}
        </svg>

        {/* ---- The words -------------------------------------------------------------- */}
        <div className="min-w-0 flex-1" role="status" aria-live="polite">
          <p className="flex items-center gap-1.5 text-[15px] font-semibold leading-tight tracking-tight text-ink">
            {/* Shape redundancy for "this row is finished". */}
            {activeFraction >= 1 ? <Check size={15} className="shrink-0 text-success" strokeWidth={2.6} /> : null}
            <span className="truncate">{title}</span>
          </p>
          <p
            className={cn(
              "mt-0.5 flex items-center gap-1 text-[12px] leading-snug",
              stuck ? "font-medium text-amber" : "text-ink-muted",
            )}
          >
            {stuck ? (
              // Rotation and pulse live on different elements on purpose; see ROW_PROMPT_CSS.
              <span className={cn("inline-flex shrink-0", !reducedMotion && "bsr-pulse")}>
                <span className="inline-flex" style={{ transform: `rotate(${nudgeRotation}deg)` }}>
                  <ArrowUp size={13} strokeWidth={2.6} />
                </span>
              </span>
            ) : null}
            <span className="min-w-0 truncate">{cue}</span>
          </p>
        </div>
      </div>
    </div>
  );
}

export default RowPrompt;
