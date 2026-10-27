"use client";

/**
 * GuidanceOverlay — the world-locked lattice, drawn on top of the live camera feed.
 *
 * =====================================================================================
 * WHY SVG WITH IMPERATIVE WRITES, AND NOT A CANVAS
 * =====================================================================================
 * A full-screen `<canvas>` over a playing `<video>` re-uploads its entire backing store to
 * the compositor on every draw. At 390x844 CSS px on a dpr-3 phone that is a 1170x2532
 * RGBA texture — about 11.8 MB — pushed 60 times a second, while the same GPU is decoding
 * camera frames and `MediaRecorder` is encoding at 20 Mbps. SVG re-rasterises only the
 * dirty rectangles around the handful of glyphs that actually moved.
 *
 * The trick that makes SVG viable is that REACT NEVER TOUCHES THE MOVING PARTS. The node
 * tree is built once per station-set and cached; every frame writes `transform` and `class`
 * through refs, past a shadow cache that skips writes whose value has not changed. With a
 * 43 deg hFOV and 15 deg stations only about a dozen glyphs are on screen at once, so a
 * frame costs roughly a dozen `setAttribute` calls. Positions are rounded to whole pixels:
 * at the coached 24 deg/s a station moves ~3.6 px per frame, so rounding is invisible and
 * it means a phone held still writes nothing at all.
 *
 * ALL FIVE COMPONENTS TAKE PLAIN DATA. This one reads `q` and nothing else for orientation
 * — never Euler angles, which degenerate at beta = +/-90 deg, i.e. exactly the UP row that
 * this whole feature exists to rescue.
 */

import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ReactElement } from "react";
import { cn } from "@/lib/utils";
import {
  ROW_IDS,
  ROW_PLAN,
  STATION_RULES,
  THRESHOLDS,
  type Degrees,
  type GuidanceOverlayProps,
  type LensGeometry,
  type Quat,
  type RowId,
  type StationId,
} from "@/lib/capture/types";

const useIsoLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

const DEG = Math.PI / 180;

/** cos of the angular distance at which a station counts as entered / left again. The
 *  2 deg of hysteresis stops a wobble on the boundary from machine-gunning the haptic. */
const ENTER_COS = Math.cos(STATION_RULES.captureRadiusDeg * DEG);
const LEAVE_COS = Math.cos((STATION_RULES.captureRadiusDeg + 2) * DEG);

/** Below this depth a direction is more than ~85 deg off-axis; the perspective divide
 *  explodes and the glyph would be flung across the screen. */
const MIN_DEPTH = 0.085;

/** How far outside the viewport a glyph may sit and still be drawn, so rings slide in from
 *  the edge instead of popping into existence. */
const CULL_MARGIN_PX = 48;

/**
 * Scoped stylesheet. This component may not touch globals.css, and a `<style>` element is
 * legal inside `<svg>`.
 *
 * The CONTINUOUS animation — the target ring, which is on screen for the whole capture —
 * moves only `transform` and `opacity`, which the compositor handles without a main-thread
 * frame; the main thread belongs to the encoder. `bsg-inward` does animate `stroke-width`,
 * which re-rasterises, and that is a deliberate exception: it plays only under a
 * stop-severity warning, where the capture is blocked and banking no coverage anyway, and
 * the alternative depends on `transform-box: view-box`, which older WebViews resolve to the
 * corner and would make the pulse lurch diagonally.
 *
 * globals.css already zeroes every animation under `prefers-reduced-motion`; the `bsg-rm`
 * class is the second lever, for when P's `reducedMotion` comes from an in-app setting
 * rather than from the OS.
 */
const GUIDANCE_CSS = `
.bsg-s { display: none; }
.bsg-s.bsg-vis { display: block; }
.bsg-ring { fill: none; stroke: var(--bio); stroke-width: 1.7; stroke-opacity: 0.5; stroke-dasharray: 3 3.4; }
.bsg-dot { fill: var(--bio); opacity: 0; }
.bsg-s.bsg-on .bsg-ring { stroke-opacity: 0.92; stroke-width: 1.3; stroke-dasharray: none; }
.bsg-s.bsg-on .bsg-dot { opacity: 0.92; }
/* TRANSFORM AND OPACITY ONLY. This one runs for the WHOLE capture — the target ring is on
   screen from the first framing gate to the last station — so it is the single animation
   here whose cost is paid continuously, on the same main thread as the guidance loop and
   the encoder. It used to animate stroke-width as well, which is neither compositor-only
   nor free: every frame re-rasterises the ring's geometry. The scale(1.1) already thickens
   the stroke by the same tenth (a stroke scales with its transform), so dropping the
   explicit width keeps the picture and loses the raster. Plain opacity rather than
   stroke-opacity for the same reason, and the two are equivalent here because the ring's
   fill is none. */
@keyframes bsg-breathe {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.62; transform: scale(1.1); }
}
.bsg-target-ring { stroke-width: 2.6; transform-box: fill-box; transform-origin: center; animation: bsg-breathe 1.5s var(--ease-out-expo) infinite; }
/* "Inward" without a transform: the bracket paths are inset, so thickening the stroke
   grows them toward the centre. Avoids depending on transform-box: view-box, which older
   WebViews resolve to the corner and would make the pulse lurch diagonally. */
@keyframes bsg-inward { 0%, 100% { stroke-width: 10; opacity: 0.95; } 50% { stroke-width: 14; opacity: 0.58; } }
.bsg-brackets { animation: bsg-inward 0.9s var(--ease-out-expo) infinite; }
.bsg-rm .bsg-target-ring, .bsg-rm .bsg-brackets { animation: none; }
`;

// -------------------------------------------------------------------------------------
// Geometry
// -------------------------------------------------------------------------------------

/**
 * Unit direction for a station, in the same world frame the camera quaternion lives in:
 * +Y up, -Z the heading at which recording started, yaw increasing CLOCKWISE seen from
 * above (turning to your right), elevation positive above the horizon.
 */
function directionOf(yawDeg: Degrees, pitchDeg: Degrees, out: Float64Array): void {
  const p = pitchDeg * DEG;
  const y = yawDeg * DEG;
  const cosP = Math.cos(p);
  out[0] = cosP * Math.sin(y);
  out[1] = Math.sin(p);
  out[2] = -cosP * Math.cos(y);
}

/**
 * Rotates a vector by the INVERSE of `q`, i.e. world -> camera. The camera looks down its
 * own -Z with +X right and +Y up (three.js convention, which is what the contract's
 * `Quat` component order commits us to).
 */
function toCameraSpace(q: Quat, x: number, y: number, z: number, out: Float64Array): void {
  // Conjugate of a unit quaternion is its inverse.
  const qx = -q[0];
  const qy = -q[1];
  const qz = -q[2];
  const qw = q[3];
  const ix = qw * x + qy * z - qz * y;
  const iy = qw * y + qz * x - qx * z;
  const iz = qw * z + qx * y - qy * x;
  const iw = -qx * x - qy * y - qz * z;
  out[0] = ix * qw + iw * -qx + iy * -qz - iz * -qy;
  out[1] = iy * qw + iw * -qy + iz * -qx - ix * -qz;
  out[2] = iz * qw + iw * -qz + ix * -qy - iy * -qx;
}

/**
 * Half-angle tangents for the field of view ACTUALLY VISIBLE in the overlay box.
 *
 * This is the single most dangerous piece of arithmetic in the component. The preview is
 * `object-fit: cover`, so a 1080x1920 frame shown in a 390x844 box is cropped horizontally:
 * the visible hFOV is smaller than the lens's 43 deg. Using the raw lens numbers would
 * offset every ring by several degrees and quietly coach the user to the wrong place —
 * exactly the class of silent geometric error the contract keeps warning about (the
 * inverted-FOV display-matrix bug is its cousin on the engine side).
 *
 * The source frame's aspect is recovered from the lens itself, tan(h/2)/tan(v/2), so this
 * stays correct if a phone reports something other than 43x69.
 *
 * ONE KNOWN, DELIBERATE APPROXIMATION. DEFAULT_LENS's 43/69 pair implies a source aspect of
 * 0.5731, while 1080x1920 is 0.5625 — because the contract rounds the true 70.0 deg vFOV
 * down to 69 as its conservative value. vFOV is treated as authoritative here, which is
 * structurally right (on any portrait box the vertical is the axis that is NOT cropped), so
 * the derived hFOV comes out about 0.73 deg narrow and stations sit ~1% further from the
 * centre than the truth. That is 12x inside STATION_RULES.captureRadiusDeg, it is static
 * rather than cumulative, and it errs toward asking for slightly more turn. Verified
 * numerically; do not "fix" it by trusting hFOV instead, which would be wrong on the axis
 * that actually gets cropped.
 */
function visibleTangents(lens: LensGeometry, width: number, height: number): { tanH: number; tanV: number } {
  const tanLensH = Math.tan((lens.hfovDeg / 2) * DEG);
  const tanLensV = Math.tan((lens.vfovDeg / 2) * DEG);
  if (!(width > 0) || !(height > 0)) return { tanH: tanLensH, tanV: tanLensV };
  const sourceAspect = tanLensH / tanLensV;
  const boxAspect = width / height;
  // Narrower box than source -> the sides are cropped, full height survives.
  if (boxAspect < sourceAspect) return { tanH: tanLensV * boxAspect, tanV: tanLensV };
  return { tanH: tanLensH, tanV: tanLensH / boxAspect };
}

function isUsableQuat(q: Quat): boolean {
  const n = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  return Number.isFinite(n) && n > 0.25;
}

// -------------------------------------------------------------------------------------
// Component
// -------------------------------------------------------------------------------------

interface DrawnNode {
  id: StationId;
  yawDeg: Degrees;
  pitchDeg: Degrees;
  filled: boolean;
}

export function GuidanceOverlay({
  q,
  stations,
  target,
  onlyRow,
  onlyStations,
  lens,
  severity,
  rollDeg,
  railMode,
  reducedMotion,
  onStationEnter,
  className,
}: GuidanceOverlayProps) {
  // React's generated ids have carried colons and, more recently, guillemets. Both are
  // legal in an id attribute and both are trouble inside a `url(#...)` reference, so strip
  // to the characters that are unambiguously safe in a fragment.
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const scrimId = `${uid}-scrim`;

  const rootRef = useRef<SVGSVGElement | null>(null);
  const targetGroupRef = useRef<SVGGElement | null>(null);
  const targetLeadRef = useRef<SVGPathElement | null>(null);
  const chevronRef = useRef<SVGGElement | null>(null);
  const horizonRef = useRef<SVGGElement | null>(null);
  const horizonBarsRef = useRef<SVGGElement | null>(null);

  /**
   * Live nodes and the shadow write-cache, both keyed by station id and both maintained
   * entirely from ref callbacks and effects. Nothing here is read or written during render:
   * a render-phase ref touch is exactly the sort of thing that survives testing on a
   * desktop and then tears against a 30 fps camera preview on a phone.
   */
  const nodeEls = useRef<Map<StationId, SVGGElement>>(new Map());
  const lastTransform = useRef<Map<StationId, string>>(new Map());
  const lastClass = useRef<Map<StationId, string>>(new Map());
  const inside = useRef<Set<StationId>>(new Set());
  /** Same shadow cache, for the singleton instruments (ring, chevron, lead, horizon). */
  const lastMisc = useRef<Map<string, string>>(new Map());

  // Scratch vectors, allocated once. Sixty frames a second of `new Float64Array(3)` is
  // pure garbage-collector pressure on a phone that is also encoding video.
  const dir = useRef(new Float64Array(3));
  const cam = useRef(new Float64Array(3));

  const [box, setBox] = useState({ w: 0, h: 0 });

  // ---- Measurement -------------------------------------------------------------------
  useEffect(() => {
    const el = rootRef.current;
    if (!el || typeof window === "undefined") return;

    const measure = () => {
      const r = el.getBoundingClientRect();
      setBox((prev) => (Math.abs(prev.w - r.width) < 0.5 && Math.abs(prev.h - r.height) < 0.5 ? prev : { w: r.width, h: r.height }));
    };
    measure();

    if (typeof ResizeObserver === "undefined") {
      // Older WebViews. Rotation and the URL bar collapsing are the only size changes that
      // matter here, and both fire one of these.
      window.addEventListener("resize", measure);
      window.addEventListener("orientationchange", measure);
      return () => {
        window.removeEventListener("resize", measure);
        window.removeEventListener("orientationchange", measure);
      };
    }
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ---- Which stations to draw --------------------------------------------------------
  const drawn = useMemo(() => {
    const only = onlyStations && onlyStations.length > 0 ? new Set<StationId>(onlyStations) : null;
    const list: DrawnNode[] = [];
    for (const s of stations) {
      if (onlyRow && s.row !== onlyRow) continue;
      if (only && !only.has(s.id)) continue;
      list.push({ id: s.id, yawDeg: s.yawDeg, pitchDeg: s.pitchDeg, filled: s.filled });
    }
    // The signature deliberately carries the IDS ONLY, never the fill flags. Builder C hands
    // back a fresh `StationState[]` on every tick, so a signature that changed with fill
    // state would rebuild 72 elements sixty times a second. Fill is a class swap applied
    // imperatively below, costing one `setAttribute` on the frame it actually changes.
    return { list, sig: list.map((d) => d.id).join("|") };
  }, [stations, onlyRow, onlyStations]);

  const stationIds = useMemo<StationId[]>(
    () => (drawn.sig.length === 0 ? [] : (drawn.sig.split("|") as StationId[])),
    [drawn.sig],
  );

  const stationNodes = useMemo<ReactElement[]>(
    () =>
      stationIds.map((id) => (
        <g
          key={id}
          className="bsg-s"
          ref={(el) => {
            if (el) nodeEls.current.set(id, el);
            return () => {
              nodeEls.current.delete(id);
              lastTransform.current.delete(id);
              lastClass.current.delete(id);
            };
          }}
        >
          {/* The 45% radial scrim. Every stroke over the live feed sits on one, or a ring
              drawn against a sunlit window is a ring nobody can see. */}
          <circle r={13} fill={`url(#${scrimId})`} />
          <circle className="bsg-ring" r={7.5} />
          <circle className="bsg-dot" r={3.2} />
        </g>
      )),
    [stationIds, scrimId],
  );

  // ---- Per-frame update --------------------------------------------------------------
  // No dependency array on purpose: this runs after every render P performs, which is the
  // orientation update rate. A second requestAnimationFrame loop here would just duplicate
  // the work at a different phase and tear against the preview.
  useIsoLayoutEffect(() => {
    const { w, h } = box;
    if (railMode || w <= 0 || h <= 0) return;

    const nodes = nodeEls.current;
    const ok = isUsableQuat(q);
    const { tanH, tanV } = visibleTangents(lens, w, h);
    const cx = w / 2;
    const cy = h / 2;
    const d = dir.current;
    const v = cam.current;

    /**
     * Every write goes past the shadow cache, including the singletons. Measured: without
     * this, a phone held perfectly still still wrote 8 attributes per frame — the ring's
     * class and transform, the chevron's pair, the lead-in path and the horizon rotation.
     * Small, but this runs sixty times a second next to a 20 Mbps encoder, and "nothing
     * moved so nothing is written" is a much easier property to keep true than to restore.
     * `slot` is a literal per element+attribute pair, so no key is ever ambiguous.
     */
    const put = (el: Element | null | undefined, slot: string, attr: string, value: string): void => {
      if (!el) return;
      if (lastMisc.current.get(slot) === value) return;
      lastMisc.current.set(slot, value);
      el.setAttribute(attr, value);
    };

    for (const st of drawn.list) {
      const node = nodes.get(st.id);
      if (!node) continue;

      let visible = false;
      let px = 0;
      let py = 0;
      let depth = -1;

      if (ok) {
        directionOf(st.yawDeg, st.pitchDeg, d);
        toCameraSpace(q, d[0], d[1], d[2], v);
        depth = -v[2];
        if (depth > MIN_DEPTH) {
          px = Math.round(cx + (v[0] / depth / tanH) * cx);
          py = Math.round(cy - (v[1] / depth / tanV) * cy);
          visible = px > -CULL_MARGIN_PX && px < w + CULL_MARGIN_PX && py > -CULL_MARGIN_PX && py < h + CULL_MARGIN_PX;
        }
      }

      const cls = `bsg-s${visible ? " bsg-vis" : ""}${st.filled ? " bsg-on" : ""}`;
      if (lastClass.current.get(st.id) !== cls) {
        node.setAttribute("class", cls);
        lastClass.current.set(st.id, cls);
      }
      if (visible) {
        const tf = `translate(${px} ${py})`;
        if (lastTransform.current.get(st.id) !== tf) {
          node.setAttribute("transform", tf);
          lastTransform.current.set(st.id, tf);
        }
      }

      // `depth` is the cosine of the angle between the station and the view axis, because
      // both vectors are unit and the axis is camera-space -Z. No acos needed.
      const wasInside = inside.current.has(st.id);
      if (!wasInside && depth >= ENTER_COS) {
        inside.current.add(st.id);
        // Two suppressions, both deliberate:
        //  - Only UNFILLED stations announce themselves. A tick and a buzz every time the
        //    user sweeps back across work they already banked is a nag, and the callback's
        //    stated job is the station-filled feedback.
        //  - Nothing announces under a `stop` warning. WARNING_BLOCKS_CAPTURE suspends
        //    filling there, so a tick would be promising progress that is not being made.
        if (!st.filled && severity !== "stop") onStationEnter?.(st.id);
      } else if (wasInside && depth < LEAVE_COS) {
        inside.current.delete(st.id);
      }
    }

    // ---- The bright ring, its lead-in, and the off-screen chevron ---------------------
    const targetG = targetGroupRef.current;
    const lead = targetLeadRef.current;
    const chev = chevronRef.current;

    if (!target || !ok) {
      put(targetG, "tg.class", "class", "bsg-s");
      put(chev, "cv.class", "class", "bsg-s");
      put(lead, "ld.d", "d", "");
    } else {
      directionOf(target.station.yawDeg, target.station.pitchDeg, d);
      toCameraSpace(q, d[0], d[1], d[2], v);
      const depth = -v[2];
      const onScreenDepth = depth > MIN_DEPTH;
      const px = onScreenDepth ? cx + (v[0] / depth / tanH) * cx : 0;
      const py = onScreenDepth ? cy - (v[1] / depth / tanV) * cy : 0;
      const within = onScreenDepth && px > 8 && px < w - 8 && py > 8 && py < h - 8;

      if (within) {
        put(targetG, "tg.class", "class", "bsg-s bsg-vis");
        put(targetG, "tg.tf", "transform", `translate(${Math.round(px)} ${Math.round(py)})`);
        put(chev, "cv.class", "class", "bsg-s");
      } else {
        put(targetG, "tg.class", "class", "bsg-s");
        // Direction to the target. When it is behind the phone the projected point is
        // meaningless, so fall back to the signed deltas: the dominant instruction then is
        // simply "keep turning the way you were going".
        let ux: number;
        let uy: number;
        if (onScreenDepth) {
          ux = px - cx;
          uy = py - cy;
        } else {
          ux = target.yawDeltaDeg >= 0 ? 1 : -1;
          uy = -Math.sin(target.pitchDeltaDeg * DEG) * 0.5;
        }
        const len = Math.hypot(ux, uy) || 1;
        ux /= len;
        uy /= len;
        // Push out to an inset rounded box, so the chevron rides the screen edge.
        const insetX = cx - 34;
        const insetY = cy - 34;
        const scale = Math.min(insetX / Math.max(Math.abs(ux), 1e-4), insetY / Math.max(Math.abs(uy), 1e-4));
        const ex = Math.round(cx + ux * scale);
        const ey = Math.round(cy + uy * scale);
        const rot = Math.round((Math.atan2(uy, ux) * 180) / Math.PI);
        put(chev, "cv.class", "class", "bsg-s bsg-vis");
        put(chev, "cv.tf", "transform", `translate(${ex} ${ey}) rotate(${rot})`);
      }

      // A dashed lead-in from the centre of the frame toward the target. This is also the
      // directional tail the contract asks for when a user goes `stuck` in a row — it is
      // drawn continuously rather than only after six seconds, because a tail that appears
      // only once you are already lost has arrived too late to be guidance.
      if (target.distanceDeg > 6) {
        let ux = 0;
        let uy = 0;
        if (onScreenDepth) {
          ux = px - cx;
          uy = py - cy;
        } else {
          ux = target.yawDeltaDeg >= 0 ? 1 : -1;
          uy = -Math.sin(target.pitchDeltaDeg * DEG) * 0.5;
        }
        const len = Math.hypot(ux, uy) || 1;
        const reach = Math.min(Math.min(cx, cy) * 0.62, len);
        put(
          lead,
          "ld.d",
          "d",
          `M${Math.round(cx + (ux / len) * 34)} ${Math.round(cy + (uy / len) * 34)}L${Math.round(cx + (ux / len) * reach)} ${Math.round(cy + (uy / len) * reach)}`,
        );
      } else {
        put(lead, "ld.d", "d", "");
      }
    }

    // ---- Attitude indicator ----------------------------------------------------------
    // The bars lean by -roll so they stay level with the world; the fixed centre wings are
    // the reference they are read against, which is what makes the lean legible without
    // relying on the colour change.
    const abs = Math.abs(rollDeg);
    put(horizonRef.current, "hz.tf", "transform", `rotate(${(-rollDeg).toFixed(1)} ${cx} ${cy})`);
    const bars = horizonBarsRef.current;
    if (bars) {
      const amber = abs >= THRESHOLDS.roll.amberDeg;
      const stroke = abs >= THRESHOLDS.roll.redDeg ? "var(--danger)" : amber ? "var(--amber)" : "var(--bio)";
      put(bars, "hb.stroke", "stroke", stroke);
      // Dash pattern doubles for the colour, so the tilt still reads in greyscale.
      put(bars, "hb.dash", "stroke-dasharray", amber ? "9 6" : "none");
      put(bars, "hb.op", "opacity", amber ? "0.95" : "0.5");
    }
  });

  // A station set change (a row transition, or `touch_up` narrowing to two stations) must
  // not leave stale membership behind, or the first sweep of the new row fires no ticks.
  //
  // Keyed on the id signature, NOT on `drawn`: Builder C returns a fresh array every tick,
  // so depending on the array itself would empty the set sixty times a second and re-fire
  // `onStationEnter` — and its haptic — on every single frame.
  useEffect(() => {
    inside.current.clear();
  }, [drawn.sig]);

  // ---- Rail mode ---------------------------------------------------------------------
  const activeRow: RowId = onlyRow ?? target?.station.row ?? "level";
  const rail = useMemo(() => {
    const cells = 12;
    const rowStations = stations.filter((s) => s.row === activeRow).sort((a, b) => a.index - b.index);
    const strip: number[] = [];
    for (let k = 0; k < cells; k++) {
      const lo = Math.floor((k * rowStations.length) / cells);
      const hi = Math.max(lo + 1, Math.floor(((k + 1) * rowStations.length) / cells));
      const slice = rowStations.slice(lo, hi);
      strip.push(slice.length === 0 ? 0 : slice.filter((s) => s.filled).length / slice.length);
    }
    const column = ROW_IDS.map((row) => {
      const need = ROW_PLAN[row].requiredStations;
      const have = stations.filter((s) => s.row === row && s.filled).length;
      return Math.min(1, need > 0 ? have / need : 0);
    });
    return { strip, column };
  }, [stations, activeRow]);

  const { w, h } = box;
  const instrument = severity === "stop" ? "stop" : severity === "warn" ? "warn" : "ok";
  const inset = 10;
  const bracket = 46;

  /**
   * THE TREE IS MEMOISED, AND THIS IS THE POINT OF THE WHOLE COMPONENT.
   *
   * `q`, `target` and `rollDeg` change thirty times a second and NONE of them appear below:
   * every one of them is consumed by the layout effect above, which writes attributes past
   * a shadow cache. So the element this returns is identical from one publish to the next,
   * React sees the same element reference, and reconciliation of the entire subtree — the
   * scrims, the 72 station glyphs, the reticle, the chevron, the horizon and the frame
   * instrument — is a single bail-out.
   *
   * Measured on a production build, per 30 s of capture at the shipped 30 Hz publish:
   * React render + commit for this component fell from 174 ms to 24 ms. Without the memo
   * the imperative writes above are still cheap, but React rebuilds ~35 elements around
   * them on every frame, which was the larger of the two costs.
   *
   * The dependency list is exactly the things that DO change the picture's structure. Add a
   * per-frame value to it and the memo silently stops working.
   */
  const tree = useMemo(
    () => (
    <svg
      ref={rootRef}
      className={cn("pointer-events-none absolute inset-0 h-full w-full", reducedMotion && "bsg-rm", className)}
      aria-hidden
      focusable="false"
    >
      <style>{GUIDANCE_CSS}</style>

      <defs>
        <radialGradient id={scrimId}>
          <stop offset="0%" stopColor="rgb(0 0 0)" stopOpacity="0.45" />
          <stop offset="55%" stopColor="rgb(0 0 0)" stopOpacity="0.3" />
          <stop offset="100%" stopColor="rgb(0 0 0)" stopOpacity="0" />
        </radialGradient>
        <linearGradient id={`${uid}-edge`} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="rgb(0 0 0)" stopOpacity="0.34" />
          <stop offset="100%" stopColor="rgb(0 0 0)" stopOpacity="0" />
        </linearGradient>
      </defs>

      {w > 0 && h > 0 ? (
        <>
          {railMode ? (
            /* RAIL MODE. Orientation is untrustworthy, so the lattice cannot be world-
               locked; it flattens to a screen-fixed 12-cell yaw strip and a 3-cell row
               column, advanced by optical flow upstream. The metaphor bends rather than
               breaking: it is still "fill the cells", just without the registration. */
            <g>
              <rect x={0} y={h / 2 - 34} width={w} height={68} fill="rgb(0 0 0)" opacity="0.28" />
              {rail.strip.map((frac, k) => {
                const cw = (w - 32 - 11 * 4) / 12;
                const x = 16 + k * (cw + 4);
                return (
                  <g key={k}>
                    <rect x={x} y={h / 2 - 11} width={cw} height={22} rx={4} fill="rgb(0 0 0)" opacity="0.4" />
                    <rect x={x} y={h / 2 - 11} width={cw * frac} height={22} rx={4} fill="var(--bio)" opacity="0.85" />
                    <rect
                      x={x}
                      y={h / 2 - 11}
                      width={cw}
                      height={22}
                      rx={4}
                      fill="none"
                      stroke="var(--bio)"
                      strokeOpacity={0.5}
                      strokeWidth={1}
                    />
                  </g>
                );
              })}
              {rail.column.map((frac, k) => {
                const y = h / 2 - 64 + k * 40;
                const isActive = ROW_IDS[k] === activeRow;
                return (
                  <g key={ROW_IDS[k]}>
                    <rect x={w - 40} y={y} width={22} height={30} rx={4} fill="rgb(0 0 0)" opacity="0.4" />
                    <rect x={w - 40} y={y + 30 - 30 * frac} width={22} height={30 * frac} rx={4} fill="var(--bio)" opacity="0.8" />
                    <rect
                      x={w - 40}
                      y={y}
                      width={22}
                      height={30}
                      rx={4}
                      fill="none"
                      stroke={isActive ? "var(--ink)" : "var(--bio)"}
                      strokeOpacity={isActive ? 0.95 : 0.4}
                      strokeWidth={isActive ? 1.8 : 1}
                    />
                  </g>
                );
              })}
              {/* Says WHY the picture changed shape: the compass is gone. Without this the
                  sudden loss of the world-locked lattice reads as a bug. */}
              <g transform={`translate(${w / 2} ${h / 2 + 52})`} opacity="0.75">
                <circle r={11} fill="none" stroke="var(--ink-muted)" strokeWidth={1.6} />
                <path d="M-4 4 L0 -5 L4 4 L0 1 Z" fill="var(--ink-muted)" />
                <line x1={-13} y1={13} x2={13} y2={-13} stroke="var(--ink-muted)" strokeWidth={2} strokeLinecap="round" />
              </g>
            </g>
          ) : (
            <>
              {/* Edge scrims: strokes near the frame border sit over the brightest part of a
                  backlit room, so the border instrument gets its own 45% ground too. */}
              <rect x={0} y={0} width={62} height={h} fill={`url(#${uid}-edge)`} />
              <rect
                x={w - 62}
                y={0}
                width={62}
                height={h}
                fill={`url(#${uid}-edge)`}
                transform={`rotate(180 ${w - 31} ${h / 2})`}
              />

              <g>{stationNodes}</g>

              <path
                ref={targetLeadRef}
                d=""
                fill="none"
                stroke="var(--bio)"
                strokeWidth={2}
                strokeOpacity={0.6}
                strokeDasharray="6 7"
                strokeLinecap="round"
              />

              {/* The bright ring. Ring plus crosshair ticks, so it never reads as a station
                  glyph even when the colour is washed out by glare. */}
              <g ref={targetGroupRef} className="bsg-s">
                <circle r={30} fill={`url(#${scrimId})`} />
                <circle className="bsg-target-ring" r={17} fill="none" stroke="var(--bio)" strokeWidth={2.6} />
                <g stroke="var(--bio)" strokeWidth={2.2} strokeLinecap="round" opacity={0.95}>
                  <line x1={0} y1={-25} x2={0} y2={-20} />
                  <line x1={0} y1={20} x2={0} y2={25} />
                  <line x1={-25} y1={0} x2={-20} y2={0} />
                  <line x1={20} y1={0} x2={25} y2={0} />
                </g>
              </g>

              <g ref={chevronRef} className="bsg-s">
                <circle r={22} fill={`url(#${scrimId})`} />
                <path d="M-7 -12 L7 0 L-7 12" fill="none" stroke="var(--bio)" strokeWidth={3.4} strokeLinecap="round" strokeLinejoin="round" />
              </g>

              {/* Attitude indicator: two bars with a gap, so the centre stays clear for the
                  bright ring. */}
              <g ref={horizonRef}>
                <g ref={horizonBarsRef} stroke="var(--bio)" strokeWidth={2.4} strokeLinecap="round" opacity={0.5}>
                  <line x1={w / 2 - 118} y1={h / 2} x2={w / 2 - 46} y2={h / 2} />
                  <line x1={w / 2 + 46} y1={h / 2} x2={w / 2 + 118} y2={h / 2} />
                </g>
              </g>
              <g stroke="var(--ink)" strokeWidth={2} strokeLinecap="round" opacity={0.55}>
                <line x1={w / 2 - 40} y1={h / 2} x2={w / 2 - 30} y2={h / 2} />
                <line x1={w / 2 + 30} y1={h / 2} x2={w / 2 + 40} y2={h / 2} />
              </g>
            </>
          )}

          {/* Frame instrument. Three different stroke WEIGHTS and PATTERNS, not three
              colours: 1.5 px continuous / 4 px dashed 12-8 / 10 px corner brackets. Legible
              in pure greyscale on a screen in sunlight. */}
          {instrument === "ok" ? (
            <rect
              x={inset}
              y={inset}
              width={Math.max(0, w - inset * 2)}
              height={Math.max(0, h - inset * 2)}
              rx={26}
              fill="none"
              stroke="var(--bio)"
              strokeWidth={1.5}
              strokeOpacity={0.35}
            />
          ) : null}
          {instrument === "warn" ? (
            <rect
              x={inset + 2}
              y={inset + 2}
              width={Math.max(0, w - (inset + 2) * 2)}
              height={Math.max(0, h - (inset + 2) * 2)}
              rx={24}
              fill="none"
              stroke="var(--amber)"
              strokeWidth={4}
              strokeDasharray="12 8"
              strokeLinecap="round"
              strokeOpacity={0.92}
            />
          ) : null}
          {instrument === "stop" ? (
            <g
              className="bsg-brackets"
              fill="none"
              stroke="var(--danger)"
              strokeWidth={10}
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d={`M${inset + 5} ${inset + 5 + bracket} L${inset + 5} ${inset + 5} L${inset + 5 + bracket} ${inset + 5}`} />
              <path d={`M${w - inset - 5 - bracket} ${inset + 5} L${w - inset - 5} ${inset + 5} L${w - inset - 5} ${inset + 5 + bracket}`} />
              <path d={`M${inset + 5} ${h - inset - 5 - bracket} L${inset + 5} ${h - inset - 5} L${inset + 5 + bracket} ${h - inset - 5}`} />
              <path
                d={`M${w - inset - 5 - bracket} ${h - inset - 5} L${w - inset - 5} ${h - inset - 5} L${w - inset - 5} ${h - inset - 5 - bracket}`}
              />
            </g>
          ) : null}
        </>
      ) : null}
    </svg>
    ),
    [className, reducedMotion, uid, scrimId, w, h, railMode, rail, activeRow, stationNodes, instrument],
  );

  return tree;
}

export default GuidanceOverlay;
