"use client";

/**
 * CoverageDome — the wordless picture of what has been photographed.
 *
 * =====================================================================================
 * WHY SVG AND NOT A THREE.JS SCENE
 * =====================================================================================
 * The obvious instinct is a small r3f scene: it is a sphere, three.js draws spheres. I
 * rejected it, and the reasons are specific to this component sitting on top of a live
 * `<video>` that is simultaneously being encoded at 20 Mbps (contract decision D6).
 *
 * 1. A SECOND WEBGL CONTEXT COMPETES WITH THE THING WE ARE HERE TO PROTECT. During the
 *    1.2 s row punctuation the camera is still streaming and `MediaRecorder` is still
 *    encoding — the contract makes recording continuous across rows precisely so the
 *    pitch transitions stay in the footage. A WebGL surface composited over a playing
 *    video pushes the video off the hardware overlay path on a good number of Android
 *    devices, and GPU memory pressure with a camera stream live is exactly when
 *    `webglcontextlost` fires. A decoration that can cost frames from the capture is not
 *    a decoration worth having. Dropped frames become holes and holes become synthesised
 *    mush — measured fact 4.
 *
 * 2. IT BUYS NOTHING GEOMETRICALLY. What we need is an orthographic projection of a unit
 *    sphere: no perspective, no lighting, no depth sort beyond a `z > 0` test on the near
 *    hemisphere. That is the ~30 lines of trigonometry in `projectLattice` below. Pulling
 *    `three` + `@react-three/fiber` onto the one route that must boot on a phone with one
 *    bar of signal, to avoid thirty lines of arithmetic, is a bad trade.
 *
 * 3. THEME TOKENS AND TEXT. Every stroke here is `var(--bio)` / `var(--line-strong)` and
 *    follows `data-theme` for free. In WebGL the colours would have to be read out with
 *    `getComputedStyle` and re-uploaded on a theme change, and the percentage numeral —
 *    the colour-blind and greyscale fallback for the whole component — would be a texture.
 *
 * 4. 2D CANVAS WAS THE RUNNER-UP and loses on the same two points: DPR management, and
 *    colours that must be resolved out of CSS by hand.
 *
 * COST CONTROL, because SVG is not free either:
 *   - Covered cells are RUN-LENGTH MERGED along each latitude band before any geometry is
 *     emitted. A swept row is contiguous, so 72 columns typically collapse to one or two
 *     quad strips. The whole dome is ONE `<path>` element with many subpaths.
 *   - The `punctuation` variant NEVER animates its geometry. It projects once and flies in
 *     with a GPU-composited `transform`/`opacity` transition, so zero JavaScript runs per
 *     frame while the recorder is live. Only the `review` variant re-projects, and by then
 *     the camera is off.
 *   - The review auto-rotate re-projects at ~30 Hz rather than 60. A 24-second revolution
 *     moves 0.5 deg between 30 Hz frames; the extra 30 rebuilds a second would only
 *     produce garbage for the collector.
 *   - Geometry is written with `setAttribute` from a ref, never through React state, so a
 *     rotating dome causes no reconciliation at all.
 */

import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { rowTargetPitchDeg } from "@/lib/capture/coverage";
import {
  COMPLETENESS,
  DEFAULT_LENS,
  THRESHOLDS,
  type CoverageDomeProps,
  type RowId,
  type Degrees,
} from "@/lib/capture/types";

// `useLayoutEffect` warns when React renders this on the server. Everything it does is
// DOM mutation, so on the server the no-op `useEffect` is the correct substitute.
const useIsoLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

const DEG = Math.PI / 180;

/** viewBox is a fixed 100x100 square; `size` only scales it. */
const CX = 50;
const CY = 43.5;
const R = 33;

/**
 * The globe is tilted 20 deg so the north polar cap is on the near hemisphere. Without a
 * tilt the UP row projects onto the silhouette edge-on and reads as a line — and the UP
 * row is the one users skip, so it is the one that must be legible (measured fact 2).
 */
const TILT_DEG = 20;
const COS_TILT = Math.cos(TILT_DEG * DEG);
const SIN_TILT = Math.sin(TILT_DEG * DEG);

/** Near-hemisphere test. A quad whose corners are closer to the limb than this is edge-on
 *  and contributes nothing but z-fighting flicker as the globe turns. */
const NEAR_EPS = 0.035;

/** Degrees per second for the review dome's idle revolution: one turn in 24 s. */
const SPIN_DEG_PER_S = 15;
const SPIN_FRAME_MS = 33;

/** Shortest string that still lands sub-pixel at 240 px: one decimal on a 100-unit box. */
function n1(v: number): string {
  return String(Math.round(v * 10) / 10);
}

// -------------------------------------------------------------------------------------
// Projection
// -------------------------------------------------------------------------------------

/**
 * Projects the whole (yawCells+1) x (pitchCells+1) boundary lattice at once, into a flat
 * `[sx, sy, z]` triple per vertex. Corners are shared between four cells, so projecting
 * boundaries instead of cells is a 4x saving before a single quad is emitted.
 *
 * Sphere frame matches the contract's angle conventions: yaw increases clockwise seen from
 * above, elevation positive above the horizon, elevation -90 at pitch index 0.
 */
function projectLattice(
  yawCells: number,
  pitchCells: number,
  viewYawDeg: number,
  sinPsi: Float64Array,
  cosPsi: Float64Array,
  out: Float32Array,
): void {
  const yawStep = 360 / yawCells;
  const pitchStep = 180 / pitchCells;

  // The yaw ring is identical for every latitude, so the trig runs 73 times, not 2701.
  for (let i = 0; i <= yawCells; i++) {
    const psi = (i * yawStep - viewYawDeg) * DEG;
    sinPsi[i] = Math.sin(psi);
    cosPsi[i] = Math.cos(psi);
  }

  const stride = yawCells + 1;
  for (let j = 0; j <= pitchCells; j++) {
    const elev = (-90 + j * pitchStep) * DEG;
    const sinE = Math.sin(elev);
    const cosE = Math.cos(elev);
    // Tilt about the screen-horizontal axis: y' = y cos t - z sin t, z' = y sin t + z cos t.
    const yTilt = sinE * COS_TILT;
    const zTilt = sinE * SIN_TILT;
    for (let i = 0; i <= yawCells; i++) {
      const x = cosE * sinPsi[i];
      const zFlat = cosE * cosPsi[i];
      const k = (j * stride + i) * 3;
      out[k] = CX + R * x;
      out[k + 1] = CY - R * (yTilt - zFlat * SIN_TILT);
      out[k + 2] = zTilt + zFlat * COS_TILT;
    }
  }
}

/**
 * Run-length merged fill for every covered cell on the near hemisphere.
 *
 * Adjacent quad strips abut exactly, so the caller pairs the fill with a hairline stroke of
 * the same colour to close the antialiasing seams between them.
 */
function coveredPath(
  covered: Uint8Array,
  yawCells: number,
  pitchCells: number,
  v: Float32Array,
): string {
  const stride = yawCells + 1;
  let d = "";

  for (let j = 0; j < pitchCells; j++) {
    const rowBase = j * yawCells;
    const lowRow = j * stride;
    const highRow = (j + 1) * stride;
    let runStart = -1;

    // One extra iteration flushes a run that reaches the last column.
    for (let i = 0; i <= yawCells; i++) {
      const drawable =
        i < yawCells &&
        covered[rowBase + i] !== 0 &&
        v[(lowRow + i) * 3 + 2] > NEAR_EPS &&
        v[(lowRow + i + 1) * 3 + 2] > NEAR_EPS &&
        v[(highRow + i) * 3 + 2] > NEAR_EPS &&
        v[(highRow + i + 1) * 3 + 2] > NEAR_EPS;

      if (drawable) {
        if (runStart < 0) runStart = i;
        continue;
      }
      if (runStart < 0) continue;

      // Out along the lower latitude, back along the upper one. Both are arcs, so every
      // cell boundary in between has to be sampled or the band bows inward.
      for (let k = runStart; k <= i; k++) {
        const p = (lowRow + k) * 3;
        d += `${k === runStart ? "M" : "L"}${n1(v[p])} ${n1(v[p + 1])}`;
      }
      for (let k = i; k >= runStart; k--) {
        const p = (highRow + k) * 3;
        d += `L${n1(v[p])} ${n1(v[p + 1])}`;
      }
      d += "Z";
      runStart = -1;
    }
  }
  return d;
}

/**
 * Visible half-arc of a latitude circle, in radians of yaw either side of the meridian
 * facing the viewer.
 *
 * z' = sin(e)sin(t) + cos(e)cos(psi)cos(t) > 0  <=>  cos(psi) > -tan(e)tan(t)
 * Returns 0 when the whole ring is hidden and PI when the tilt has brought all of it onto
 * the near hemisphere, which is what happens to the UP row's neighbourhood.
 */
function latitudeHalfSpan(elevDeg: Degrees): number {
  const e = elevDeg * DEG;
  const cosE = Math.cos(e);
  if (Math.abs(cosE) < 1e-6) return 0;
  const c = -(Math.sin(e) * SIN_TILT) / (cosE * COS_TILT);
  if (c <= -1) return Math.PI;
  if (c >= 1) return 0;
  return Math.acos(c);
}

/**
 * Polyline along a latitude circle, clipped to the near hemisphere.
 *
 * NOTE — no `viewYawDeg` parameter, and that is not an omission. Spinning the globe about
 * its polar axis maps every latitude circle onto itself, so their projections are invariant
 * under the review dome's rotation. The graticule and the next-row band are therefore drawn
 * ONCE and left alone; only the covered fill and the meridians are rebuilt per frame.
 */
function latitudePath(elevDeg: Degrees, samples = 40): string {
  const half = latitudeHalfSpan(elevDeg);
  if (half <= 0) return "";
  const e = elevDeg * DEG;
  const sinE = Math.sin(e);
  const cosE = Math.cos(e);
  const yTilt = sinE * COS_TILT;
  // Pull in off the exact limb: a vertex sitting at z' = 0 flickers as the globe turns.
  const span = half * 0.998;
  let d = "";
  for (let s = 0; s <= samples; s++) {
    const psi = -span + (2 * span * s) / samples;
    const x = CX + R * cosE * Math.sin(psi);
    const y = CY - R * (yTilt - cosE * Math.cos(psi) * SIN_TILT);
    d += `${s === 0 ? "M" : "L"}${n1(x)} ${n1(y)}`;
  }
  return d;
}

/** Closed band between two latitudes, both clipped to the narrower of the two spans. */
function bandPath(elevLoDeg: Degrees, elevHiDeg: Degrees, samples = 36): string {
  const half = Math.min(latitudeHalfSpan(elevLoDeg), latitudeHalfSpan(elevHiDeg));
  if (half <= 0) return "";
  const span = half * 0.998;

  const edge = (elevDeg: Degrees, forward: boolean, first: boolean): string => {
    const e = elevDeg * DEG;
    const sinE = Math.sin(e);
    const cosE = Math.cos(e);
    const yTilt = sinE * COS_TILT;
    let out = "";
    for (let s = 0; s <= samples; s++) {
      const t = forward ? s : samples - s;
      const psi = -span + (2 * span * t) / samples;
      const x = CX + R * cosE * Math.sin(psi);
      const y = CY - R * (yTilt - cosE * Math.cos(psi) * SIN_TILT);
      out += `${first && s === 0 ? "M" : "L"}${n1(x)} ${n1(y)}`;
    }
    return out;
  };

  return `${edge(elevLoDeg, true, true)}${edge(elevHiDeg, false, false)}Z`;
}

/** Front-facing meridians, drawn only on the large review dome. */
function meridianPath(viewYawDeg: number, everyDeg = 30): string {
  let d = "";
  for (let yaw = 0; yaw < 360; yaw += everyDeg) {
    const psi = (yaw - viewYawDeg) * DEG;
    const sinP = Math.sin(psi);
    const cosP = Math.cos(psi);
    let started = false;
    for (let s = 0; s <= 24; s++) {
      const e = (-90 + (180 * s) / 24) * DEG;
      const cosE = Math.cos(e);
      const sinE = Math.sin(e);
      const zFlat = cosE * cosP;
      const z = sinE * SIN_TILT + zFlat * COS_TILT;
      if (z <= NEAR_EPS) {
        started = false;
        continue;
      }
      const x = CX + R * cosE * sinP;
      const y = CY - R * (sinE * COS_TILT - zFlat * SIN_TILT);
      d += `${started ? "L" : "M"}${n1(x)} ${n1(y)}`;
      started = true;
    }
  }
  return d;
}

// -------------------------------------------------------------------------------------
// Component
// -------------------------------------------------------------------------------------

export function CoverageDome({
  covered,
  yawCells,
  pitchCells,
  coveredFraction,
  nextRow,
  size,
  variant,
  yawDeg,
  reducedMotion,
  label,
  lens,
  className,
}: CoverageDomeProps) {
  // See RowPromptProps.lens: the drawn latitudes must be the ones the stations are on.
  const pitchOf = (id: RowId) => rowTargetPitchDeg(id, lens ?? DEFAULT_LENS);
  // Stripped to fragment-safe characters: React's generated ids have carried colons and,
  // more recently, guillemets, and this one is used inside a `url(#...)` reference.
  const gradientId = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const coveredRef = useRef<SVGPathElement | null>(null);
  const rowsRef = useRef<SVGPathElement | null>(null);
  const equatorRef = useRef<SVGPathElement | null>(null);
  const capsRef = useRef<SVGPathElement | null>(null);
  const meridiansRef = useRef<SVGPathElement | null>(null);
  const bandFillRef = useRef<SVGPathElement | null>(null);
  const bandEdgeRef = useRef<SVGPathElement | null>(null);

  // Scratch buffers, reallocated only when the grid shape changes.
  const scratch = useRef<{ cells: number; v: Float32Array; sin: Float64Array; cos: Float64Array } | null>(null);
  // Survives the effect restarting on a prop change, so the spin never jumps.
  const spinRef = useRef(0);

  const [entered, setEntered] = useState(false);

  const gridOk =
    Number.isInteger(yawCells) &&
    Number.isInteger(pitchCells) &&
    yawCells >= 4 &&
    pitchCells >= 2 &&
    yawCells <= 720 &&
    pitchCells <= 360;
  // A grid of the wrong length is a bug upstream, but a dome that throws takes the whole
  // capture screen with it. Draw the graticule and no fill, and let the numeral carry it.
  const maskOk = gridOk && covered.length === yawCells * pitchCells;

  const detailed = variant === "review";
  // Static yaw wins over the idle spin: `yawDeg` is how the review dome follows a scrubbed
  // thumbnail, and a dome that drifts away from the frame being scrubbed is a lie.
  const spins = detailed && yawDeg === undefined && !reducedMotion;

  // ---- Spin-invariant geometry: the graticule and the next-row band. Drawn once. -------
  useIsoLayoutEffect(() => {
    // Row latitudes come from ROW_PLAN, never from a literal: `rowSeparationDeg` moves them
    // on a phone whose vFOV is not 69 deg, and a dome drawn at a different pitch from the
    // one the user is being asked to sweep would be actively misleading.
    rowsRef.current?.setAttribute(
      "d",
      `${latitudePath(pitchOf("up"), 36)}${latitudePath(pitchOf("down"), 36)}`,
    );
    equatorRef.current?.setAttribute("d", latitudePath(pitchOf("level"), 44));

    // +/-60 deg is poles._CAP_DEG: everything outside it may be synthesised, so the dotted
    // line is an honest "beyond here we invent it" boundary.
    if (detailed) capsRef.current?.setAttribute("d", `${latitudePath(60, 30)}${latitudePath(-60, 30)}`);

    if (nextRow) {
      const centre = pitchOf(nextRow);
      const half = THRESHOLDS.rowPitch.amberDeg;
      bandFillRef.current?.setAttribute("d", bandPath(centre - half, centre + half));
      bandEdgeRef.current?.setAttribute("d", `${latitudePath(centre - half, 34)}${latitudePath(centre + half, 34)}`);
    } else {
      bandFillRef.current?.setAttribute("d", "");
      bandEdgeRef.current?.setAttribute("d", "");
    }
  }, [nextRow, detailed]);

  // ---- Spin-dependent geometry: the covered fill and the meridians. --------------------
  useIsoLayoutEffect(() => {
    if (!gridOk) return;

    const verts = (yawCells + 1) * (pitchCells + 1);
    if (!scratch.current || scratch.current.cells !== verts) {
      scratch.current = {
        cells: verts,
        v: new Float32Array(verts * 3),
        sin: new Float64Array(yawCells + 1),
        cos: new Float64Array(yawCells + 1),
      };
    }
    const buf = scratch.current;

    const draw = (viewYaw: number) => {
      projectLattice(yawCells, pitchCells, viewYaw, buf.sin, buf.cos, buf.v);
      coveredRef.current?.setAttribute("d", maskOk ? coveredPath(covered, yawCells, pitchCells, buf.v) : "");
      if (detailed) meridiansRef.current?.setAttribute("d", meridianPath(viewYaw));
    };

    if (!spins) {
      draw(yawDeg ?? spinRef.current);
      return;
    }

    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const tick = (now: number) => {
      acc += now - last;
      last = now;
      // Throttled to ~30 Hz: at 15 deg/s the globe moves 0.5 deg between redraws, below the
      // eye's threshold for a shape this small, and it halves the path-string churn.
      if (acc >= SPIN_FRAME_MS) {
        spinRef.current = (spinRef.current + (SPIN_DEG_PER_S * acc) / 1000) % 360;
        acc = 0;
        draw(spinRef.current);
      }
      raf = requestAnimationFrame(tick);
    };
    draw(spinRef.current);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // `coveredFraction` is a dependency on purpose: Builder C stamps into the SAME
    // Uint8Array, so the mask's identity never changes and depending on it alone would
    // freeze the dome. The fraction moves whenever the mask does.
  }, [covered, coveredFraction, yawCells, pitchCells, yawDeg, spins, detailed, gridOk, maskOk]);

  // The punctuation dome flies in once. A CSS transition on transform/opacity is composited
  // off the main thread, so this costs nothing while MediaRecorder is running.
  useEffect(() => {
    const raf = requestAnimationFrame(() => setEntered(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  const pct = Math.round(Math.min(1, Math.max(0, coveredFraction)) * 100);
  const ceiling = COMPLETENESS.coveredFractionCeiling;
  const target = COMPLETENESS.coveredFraction;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      role="img"
      aria-label={label}
      className={cn("block shrink-0", className)}
      style={{
        opacity: entered ? 1 : 0,
        transform: entered || reducedMotion ? "scale(1)" : "scale(0.82)",
        transition: reducedMotion
          ? "opacity 180ms linear"
          : "opacity 320ms ease-out, transform 620ms var(--ease-spring)",
        willChange: "transform, opacity",
      }}
    >
      <defs>
        <radialGradient id={`${gradientId}-ball`} cx="34%" cy="26%" r="86%">
          <stop offset="0%" stopColor="var(--surface-3)" stopOpacity="0.98" />
          <stop offset="62%" stopColor="var(--surface)" stopOpacity="0.94" />
          <stop offset="100%" stopColor="var(--canvas)" stopOpacity="0.98" />
        </radialGradient>
      </defs>

      {/* The unphotographed sphere. Everything drawn after this is what the user earned. */}
      <circle cx={CX} cy={CY} r={R} fill={`url(#${gradientId}-ball)`} />

      {detailed ? (
        <>
          <path ref={meridiansRef} d="" fill="none" stroke="var(--line)" strokeWidth="0.45" />
          <path
            ref={capsRef}
            d=""
            fill="none"
            stroke="var(--ink-faint)"
            strokeWidth="0.5"
            strokeDasharray="0.7 1.7"
            strokeLinecap="round"
            opacity="0.8"
          />
        </>
      ) : null}

      {/* Covered cells. Fill plus a hairline stroke of the same colour: abutting quad strips
          otherwise show antialiasing seams between the merged runs. */}
      <path
        ref={coveredRef}
        d=""
        fill="var(--bio)"
        fillOpacity="0.44"
        stroke="var(--bio)"
        strokeWidth="0.4"
        strokeOpacity="0.44"
        strokeLinejoin="round"
      />

      {/* Registration: the three row latitudes, level emphasised. Kept above the fill so the
          plan stays readable once coverage is dense. */}
      <path ref={rowsRef} d="" fill="none" stroke="var(--line-strong)" strokeWidth="0.55" strokeLinecap="round" />
      <path ref={equatorRef} d="" fill="none" stroke="var(--ink-faint)" strokeWidth="0.85" strokeLinecap="round" />

      {/* Where to go next, as a dashed band rather than a line: dash pattern and position
          both carry the message, so it survives greyscale and a colour-blind viewer. */}
      <path ref={bandFillRef} d="" fill="var(--amber)" fillOpacity="0.15" />
      <path
        ref={bandEdgeRef}
        d=""
        fill="none"
        stroke="var(--amber)"
        strokeWidth="0.95"
        strokeDasharray="2.6 2"
        strokeLinecap="round"
      />

      {/* Limb last, so no fill bleeds over the silhouette. */}
      <circle cx={CX} cy={CY} r={R} fill="none" stroke="var(--line-strong)" strokeWidth="0.7" />

      {/* The numeral is the fallback channel: it says the same thing as the shading with no
          colour, no shape and no animation involved. It is also the same number the server
          reports, because the phone and the engine share the 72x36 lattice. */}
      <text
        x={CX}
        y={detailed ? 87 : 90}
        textAnchor="middle"
        fill="var(--ink)"
        fontSize={detailed ? 9.5 : 11}
        fontFamily="var(--font-mono)"
        letterSpacing="-0.4"
      >
        {pct}%
      </text>

      {detailed ? (
        <>
          {/* Scaled against the PHYSICAL ceiling of the three-row plan (0.8949), not against
              100%. Grading a perfect capture as 89% of a bar it can never fill would read as
              failure; contract decision D4 makes the same point about the 0.88 gate. */}
          <rect x={26} y={92} width={48} height={2.6} rx={1.3} fill="var(--surface-3)" />
          <rect
            x={26}
            y={92}
            // Rounded before it reaches the attribute: a raw float here differs in the last
            // ulp between the server's V8 and the phone's, which React reports as a
            // hydration mismatch.
            width={Math.round(Math.max(0, Math.min(1, coveredFraction / ceiling)) * 48000) / 1000}
            height={2.6}
            rx={1.3}
            fill={coveredFraction >= target ? "var(--bio)" : "var(--amber)"}
          />
          <line
            x1={Math.round((26 + (target / ceiling) * 48) * 1000) / 1000}
            y1={90.6}
            x2={Math.round((26 + (target / ceiling) * 48) * 1000) / 1000}
            y2={96}
            stroke="var(--ink)"
            strokeWidth="0.8"
            strokeLinecap="round"
          />
        </>
      ) : null}
    </svg>
  );
}

export default CoverageDome;
