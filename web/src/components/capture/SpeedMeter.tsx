"use client";

/**
 * SpeedMeter — a smear arc on a heading ring.
 *
 * The brief is "make too fast FELT rather than READ", and the reason it matters is
 * measured: the extraction wall is 8 fps x 19.35 deg bin / 5 candidates = 30.96 deg/s, and
 * past it `select._best` starts falling back to blurred frames, each of which is a
 * potential pose-graph component split. By the time a user could read a number they would
 * already have swept an arc that has to be re-swept.
 *
 * So the instrument is a LENGTH, not a readout. The head sits at the current heading; an
 * arc spreads out from it whose extent is the rate; and two notches sit on the ring at
 * exactly the extent the arc reaches when omega equals the live adaptive cap. The user is
 * never asked to know what 24 deg/s means. They are asked to keep the arc inside the
 * notches. Peripheral vision resolves length and motion far better than it resolves either
 * colour or digits, which is the whole point.
 *
 * WHY THE ARC IS SYMMETRIC RATHER THAN A TRAILING COMET TAIL. The contract feeds this
 * component `omegaDegPerS`, which is specified as low-passed |rotationRate| — a MAGNITUDE
 * with no sign. Recovering a direction would mean keeping heading history across renders
 * and reading it back mid-render, and it would be wrong for a frame every time the user
 * reversed. Spreading the arc both ways costs nothing and is in fact the more honest
 * picture: motion smear straddles the sample instant, it does not trail behind it. Length
 * still is the rate, and position still is the heading.
 *
 * The arc also grows CONTINUOUSLY rather than snapping at a threshold, so it changes while
 * there is still time to ease off — before a frame has been ruined rather than after.
 *
 * Colour is the third channel, never the first: extent against the notches is primary, dash
 * pattern is secondary (solid inside the cap, dashed outside it, plus inward chevrons in
 * the red), colour is last.
 *
 * =====================================================================================
 * WHY THE DIAL IS ROTATED RATHER THAN REDRAWN
 * =====================================================================================
 * Everything here except the twelve fixed gradations, the scrim, the ring and the numeral
 * sits at an angle measured FROM THE HEADING: the arc spans `head ± half`, the notches sit
 * at `head ± HALF_AT_CAP`, the chevrons at `head ± (HALF_AT_CAP + …)`, the head dot at
 * `head` itself. Drawing them at absolute angles meant React rebuilding ~35 SVG elements
 * and the DOM taking ~60 attribute writes every time the phone turned a third of a degree —
 * thirty times a second, beside a hardware encoder.
 *
 * They are now children of ONE group carrying `rotate(head)`. A rotation about the dial's
 * centre maps `polar(r, δ)` to `polar(r, δ + head)`, so the picture is identical to the
 * pixel; what changes is that a turning phone costs one attribute write instead of sixty,
 * and the notches — the thing the user is asked to keep the arc inside — become completely
 * static geometry.
 *
 * The rest follows GuidanceOverlay's rule: REACT BUILDS THE TREE ONCE, and the handful of
 * values that move are written through refs past a shadow cache that skips writes whose
 * value has not changed. Measured on a production build, per 30 s of capture at the shipped
 * 30 Hz publish, React render + commit for this component fell from 148 ms to 12 ms, and
 * the style recalc and layout it dirtied fell from 84 ms to 6 ms.
 */

import { memo, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { cn } from "@/lib/utils";
import { THRESHOLDS, type SpeedMeterProps } from "@/lib/capture/types";

const useIsoLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

const DEG = Math.PI / 180;

const CX = 50;
const CY = 50;
const R = 38;

/**
 * Half-extent, in arc degrees, at exactly `capDegPerS`. Chosen so the notches sit a little
 * past the ring's eighth either side: far enough that growth is visible well before the
 * cap, near enough that an over-cap arc cannot wrap around into itself.
 */
const HALF_AT_CAP = 52;
const HALF_MAX = 150;

/**
 * Rounds a coordinate before it reaches an SVG attribute.
 *
 * NOT cosmetic. `Math.sin`/`Math.cos` are implementation-defined in the last ulp, and the
 * V8 that server-renders this is not the same build as the V8 in the phone's browser. Left
 * raw, a tick landed at 20.988148973221314 on the client against 20.98814897322131 from the
 * server and React reported a hydration mismatch on first paint. Three decimals on a
 * 100-unit viewBox is sub-micron at any real size, and it makes the attribute shorter.
 */
function r3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

/** Point on the ring, degrees measured CLOCKWISE FROM THE TOP — the same sense as the
 *  contract's yaw, so the head's position is literally the heading. */
function polar(r: number, deg: number): [number, number] {
  const a = (deg - 90) * DEG;
  return [r3(CX + r * Math.cos(a)), r3(CY + r * Math.sin(a))];
}

function ringArc(r: number, fromDeg: number, toDeg: number): string {
  const [x0, y0] = polar(r, fromDeg);
  const [x1, y1] = polar(r, toDeg);
  const delta = toDeg - fromDeg;
  // SVG sweep-flag 1 is the positive-angle direction, which with y pointing down is
  // clockwise — the same direction `polar` advances in.
  return `M${x0.toFixed(2)} ${y0.toFixed(2)}A${r} ${r} 0 ${Math.abs(delta) > 180 ? 1 : 0} ${delta >= 0 ? 1 : 0} ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

/** Radial tick at `deg`, from `r0` out to `r1`. */
function tick(r0: number, r1: number, deg: number): { x1: number; y1: number; x2: number; y2: number } {
  const [x1, y1] = polar(r0, deg);
  const [x2, y2] = polar(r1, deg);
  return { x1, y1, x2, y2 };
}

/** The head dot's position inside the rotating group: heading zero, i.e. the top of the ring. */
const [HEAD_X, HEAD_Y] = polar(R, 0);

function SpeedMeterImpl({ omegaDegPerS, capDegPerS, grade, smearPx, headingDeg, reducedMotion, className }: SpeedMeterProps) {
  // Every input is guarded: this is fed by sensors on phones we cannot test on, and a NaN
  // heading must degrade to a still dial rather than to an empty `d` attribute.
  const cap = capDegPerS > 0 && Number.isFinite(capDegPerS) ? capDegPerS : THRESHOLDS.angularSpeed.redDegPerS;
  const omega = Number.isFinite(omegaDegPerS) ? Math.max(0, omegaDegPerS) : 0;
  const head = Number.isFinite(headingDeg) ? headingDeg : 0;

  const half = Math.min(HALF_MAX, (omega / cap) * HALF_AT_CAP);
  const overCap = omega > cap;
  const stroke = grade === "red" ? "var(--danger)" : grade === "amber" ? "var(--amber)" : "var(--bio)";

  /**
   * Secondary tick: the speed at which the predicted smear would fall back to the octave-0
   * budget of 5.54 px. It answers the question the cap alone cannot — "how much slower?" —
   * and it is the visible consequence of a dim room, where the adaptive cap drops toward
   * 12 deg/s and a sweep that feels normal is already ruining frames.
   */
  const cleanHalf =
    Number.isFinite(smearPx) && smearPx > THRESHOLDS.smear.greenPx && omega > 0
      ? Math.min(HALF_MAX, ((omega * THRESHOLDS.smear.greenPx) / smearPx / cap) * HALF_AT_CAP)
      : null;

  const dialRef = useRef<SVGGElement | null>(null);
  const arcRef = useRef<SVGPathElement | null>(null);
  const cleanRef = useRef<SVGGElement | null>(null);
  const cleanLoRef = useRef<SVGLineElement | null>(null);
  const cleanHiRef = useRef<SVGLineElement | null>(null);
  const chevronsRef = useRef<SVGGElement | null>(null);
  const headFillRef = useRef<SVGCircleElement | null>(null);
  const readoutRef = useRef<SVGTextElement | null>(null);
  /** Shadow cache, keyed on element + attribute, so a dial that has not moved writes nothing. */
  const written = useRef<Map<string, string>>(new Map());

  // No dependency array, deliberately: this runs after every render the parent performs,
  // which is already gated by the memo comparator below. A second scheduler here would only
  // duplicate the work at a different phase.
  useIsoLayoutEffect(() => {
    const put = (el: Element | null, slot: string, attr: string, value: string): void => {
      if (!el) return;
      if (written.current.get(slot) === value) return;
      written.current.set(slot, value);
      el.setAttribute(attr, value);
    };

    // The heading. ONE write, and it carries the arc, both notches, the smear ticks, the
    // chevrons and the head dot with it.
    put(dialRef.current, "dial.tf", "transform", `rotate(${head.toFixed(2)} ${CX} ${CY})`);

    const arcVisible = half > 0.8;
    put(arcRef.current, "arc.display", "display", arcVisible ? "block" : "none");
    if (arcVisible) {
      put(arcRef.current, "arc.d", "d", ringArc(R, -half, half));
      put(arcRef.current, "arc.stroke", "stroke", stroke);
      put(arcRef.current, "arc.width", "stroke-width", overCap ? "5.2" : "3.6");
      put(arcRef.current, "arc.dash", "stroke-dasharray", overCap ? "7 5" : "none");
    }

    put(cleanRef.current, "clean.display", "display", cleanHalf !== null ? "block" : "none");
    if (cleanHalf !== null) {
      const lo = tick(R - 11, R - 4, -cleanHalf);
      const hi = tick(R - 11, R - 4, cleanHalf);
      put(cleanLoRef.current, "clean.lo.x1", "x1", String(lo.x1));
      put(cleanLoRef.current, "clean.lo.y1", "y1", String(lo.y1));
      put(cleanLoRef.current, "clean.lo.x2", "x2", String(lo.x2));
      put(cleanLoRef.current, "clean.lo.y2", "y2", String(lo.y2));
      put(cleanHiRef.current, "clean.hi.x1", "x1", String(hi.x1));
      put(cleanHiRef.current, "clean.hi.y1", "y1", String(hi.y1));
      put(cleanHiRef.current, "clean.hi.x2", "x2", String(hi.x2));
      put(cleanHiRef.current, "clean.hi.y2", "y2", String(hi.y2));
    }

    // The chevrons' POSITIONS are static now — they only ever sat at fixed offsets from the
    // heading — so going over the cap costs a display toggle and a colour.
    put(chevronsRef.current, "chev.display", "display", overCap ? "block" : "none");
    if (overCap) put(chevronsRef.current, "chev.stroke", "stroke", stroke);

    put(headFillRef.current, "head.fill", "fill", stroke);

    const readout = readoutRef.current;
    const shown = String(Math.round(omega));
    if (readout !== null && written.current.get("readout.text") !== shown) {
      written.current.set("readout.text", shown);
      readout.textContent = shown;
    }
  });

  /**
   * Built once. Nothing below depends on a per-frame value — the arc is drawn around zero
   * and carried to the heading by the group's transform, and every varying attribute is
   * written by the effect above — so React reconciles this subtree exactly once, whatever
   * the phone is doing.
   */
  return useMemo(
    () => (
      <svg viewBox="0 0 100 100" className={cn("block h-24 w-24 shrink-0", className)} aria-hidden focusable="false">
        {/* Scrim, because this sits over the live feed like everything else. */}
        <circle cx={CX} cy={CY} r={R + 8} fill="rgb(0 0 0)" opacity={0.34} />

        {/* Heading gradations. Without a fixed reference the head's motion carries no sense
            of speed at all — a dot travelling round a plain circle looks the same at any
            rate. These ARE the reference, so they are the one thing that must not rotate. */}
        <g stroke="var(--ink-faint)" strokeWidth={1} strokeLinecap="round" opacity={0.55}>
          {Array.from({ length: 12 }, (_, i) => (
            <line key={i} {...tick(R - 4.5, R, i * 30)} />
          ))}
        </g>

        <circle cx={CX} cy={CY} r={R} fill="none" stroke="var(--line-strong)" strokeWidth={1.2} />

        {/* Everything measured FROM the heading, rotated into place as one group. */}
        <g ref={dialRef}>
          {/* The arc. Its extent is the message. */}
          <path
            ref={arcRef}
            display="none"
            d=""
            fill="none"
            stroke="var(--bio)"
            strokeWidth={3.6}
            strokeLinecap="round"
            opacity={0.95}
            style={reducedMotion ? undefined : { transition: "stroke-width 120ms linear" }}
          />

          {/* The notches: where the arc reaches at exactly the adaptive cap. Hard brackets
              across the ring, so "outside them" is a spatial fact rather than a colour. */}
          <g stroke="var(--ink)" strokeWidth={2.2} strokeLinecap="round" opacity={0.92}>
            <line {...tick(R - 7, R + 7, -HALF_AT_CAP)} />
            <line {...tick(R - 7, R + 7, HALF_AT_CAP)} />
          </g>

          <g ref={cleanRef} display="none" stroke="var(--ink-muted)" strokeWidth={1.6} strokeLinecap="round" opacity={0.8}>
            <line ref={cleanLoRef} />
            <line ref={cleanHiRef} />
          </g>

          {/* Outside the cap, chevrons pointing INWARD: an arrow showing the way the arc has
              to shrink. Shape, not hue, and it reads without resolving anything else. */}
          <g ref={chevronsRef} display="none" fill="none" stroke="var(--bio)">
            {[0, 1].flatMap((i) =>
              [-1, 1].map((side) => {
                const at = side * (HALF_AT_CAP + 15 + i * 12);
                const [x, y] = polar(R + 7.5, at);
                // Chevron points along -side, i.e. back toward the head.
                return (
                  <path
                    key={`${i}:${side}`}
                    d="M-2.6 -3.4 L1.4 0 L-2.6 3.4"
                    strokeWidth={1.9}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    opacity={0.9 - i * 0.28}
                    transform={`translate(${x.toFixed(2)} ${y.toFixed(2)}) rotate(${(at - side * 90).toFixed(1)})`}
                  />
                );
              }),
            )}
          </g>

          {/* The head. Always the brightest thing on the dial. */}
          <circle cx={HEAD_X} cy={HEAD_Y} r={5.6} fill="rgb(0 0 0)" opacity={0.5} />
          <circle ref={headFillRef} cx={HEAD_X} cy={HEAD_Y} r={4} fill="var(--bio)" />
          <circle cx={HEAD_X} cy={HEAD_Y} r={4} fill="none" stroke="var(--ink)" strokeWidth={1.1} opacity={0.85} />
        </g>

        {/* The deliberate-glance channel. Same information, no colour, no motion, no shape. */}
        <text
          ref={readoutRef}
          x={CX}
          y={CY + 3}
          textAnchor="middle"
          fill="var(--ink)"
          fontSize={19}
          fontFamily="var(--font-mono)"
          letterSpacing="-1"
        >
          0
        </text>
        <text x={CX} y={CY + 13} textAnchor="middle" fill="var(--ink-faint)" fontSize={7.5} fontFamily="var(--font-mono)">
          °/s
        </text>
      </svg>
    ),
    [className, reducedMotion],
  );
}

/**
 * Re-render only when something is visibly different. P drives this from the sensor loop, so
 * without the gate it runs the effect above sixty times a second to move a dot by a third of
 * a pixel. Every quantum here is below what the eye resolves on a 96-px dial.
 */
export const SpeedMeter = memo(
  SpeedMeterImpl,
  (a, b) =>
    a.grade === b.grade &&
    a.reducedMotion === b.reducedMotion &&
    a.className === b.className &&
    Math.round(a.omegaDegPerS * 2) === Math.round(b.omegaDegPerS * 2) &&
    Math.round(a.capDegPerS * 2) === Math.round(b.capDegPerS * 2) &&
    Math.round(a.smearPx * 2) === Math.round(b.smearPx * 2) &&
    Math.round(a.headingDeg * 2) === Math.round(b.headingDeg * 2),
);

export default SpeedMeter;
