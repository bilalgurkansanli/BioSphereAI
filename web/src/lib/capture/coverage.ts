/**
 * BioSphere AI — guided capture: the coverage model.
 * =====================================================================================
 *
 * Accumulates where the camera has actually pointed, on the SAME 72x36 equirectangular
 * lattice `engine/biosphere/pano/coverage.py` uses, so the number the phone shows during
 * the sweep and the number the server reports after the upload are one number.
 *
 * Everything in this file is pure: no React, no `three`, no DOM. It runs in plain Node,
 * which is the only reason it can be proved correct without a phone in hand.
 *
 * -------------------------------------------------------------------------------------
 * WHY `coveredFraction` IS FILL-WEIGHTED AND NOT A COUNT OF >=50%-FILLED CELLS
 * -------------------------------------------------------------------------------------
 * `coverage.build_coverage` does two different things with two different resolutions:
 *
 *   covered_fraction = (weights * covered).sum()      <- PER PIXEL, full canvas, continuous
 *   grid_covered     = resize(cov, 72, 36) >= 50%     <- the 5-deg lattice, ONLY for gaps
 *
 * The >= 50% rule is the gap finder's rule, not the fraction's. Reproducing the fraction
 * on a 5-deg lattice therefore means weighting each cell by its solid angle times the
 * fraction of it that is filled — the discrete equivalent of the engine's per-pixel sum.
 *
 * The arithmetic proves it. A perfect +29/0/-29 sweep reaches elevation 63.5 deg at the
 * frame-centre column (sin 63.5 = 0.8949, which is exactly where contract decision D4's
 * ceiling comes from). Counting the whole 60..65 deg cell row as covered because it is
 * 64% filled would report 0.906 — ABOVE the 0.90 gate D4 rejects as unreachable, which
 * would make D4's own reasoning false. Fill weighting reports 0.890, one third of a point
 * under the analytic ceiling, and the shortfall is real: the corners of a rectilinear
 * frame reach only 58.3 deg while its centre column reaches 63.5, so the union's top
 * boundary oscillates between 62.9 and 63.5 as it crosses each station.
 *
 * `grid.covered` (the >= 50% mask) is still maintained exactly as the engine maintains it,
 * and is what `findGaps` and the heat-map read — same split as the engine.
 *
 * -------------------------------------------------------------------------------------
 * WHY THE STAMP IS DRIVEN BY yaw/pitch/roll AND NOT BY THE QUATERNION
 * -------------------------------------------------------------------------------------
 * `OrientationSample.q` is authoritative for guidance, and the contract says so — but its
 * basis convention lives inside Builder S's Z-X'-Y'' conversion, and a coverage model that
 * guessed that convention wrong would be silently, plausibly wrong (the exact failure mode
 * this whole feature exists to prevent). `yawDeg` / `pitchDeg` / `rollDeg` are defined
 * unambiguously by the contract's angle conventions and are explicitly sanctioned for
 * coverage stamping ("Derived for display and for coverage stamping ONLY"). The Euler
 * degeneracy at pitch = +/-90 is harmless HERE and only here: at the pole, yaw and roll
 * rotate about the same axis, so their sum still lands the frustum's footprint in the
 * right place, and the footprint of a cap-covering frame is nearly rotation-invariant
 * anyway. Guidance (Builder V) must keep using `q`.
 *
 * -------------------------------------------------------------------------------------
 * COST
 * -------------------------------------------------------------------------------------
 * A stamp touches only cells inside [pitch +/- (circumradius + cell radius)] and, per row,
 * a yaw window solved exactly from the spherical law of cosines. Cells wholly inside the
 * inscribed cone take one distance test; cells wholly outside take one; only the ~45 cells
 * straddling the footprint's boundary pay for the meridian solve. Measured at 0.11 ms per
 * stamp on a desktop, i.e. ~1 % of one core at COVERAGE_GRID.stampHz even allowing a
 * mid-range phone to be ten times slower.
 *
 * Correctness does not depend on any of that culling: `stampConeExhaustive` walks all 2592
 * cells with every shortcut removed, and the tests assert the two produce identical grids.
 */

import {
  COMPLETENESS,
  COVERAGE_GRID,
  DEFAULT_LENS,
  ROW_IDS,
  ROW_BIAS_DEG,
  ROW_PLAN,
  YAW_SPAN,
  rowSeparationDeg,
  type CompletionBlocker,
  type CoverageApi,
  type CoverageGap,
  type CoverageGrid,
  type CoverageReport,
  type CoverageSnapshot,
  type Degrees,
  type LensGeometry,
  type OrientationSample,
  type RowId,
  type RowProgress,
  type StationState,
} from "@/lib/capture/types";

const DEG = Math.PI / 180;

/**
 * Sub-columns in YAW across a boundary cell. Elevation is not sampled at all — see
 * `meridianInsideFraction` — because that is the axis where all the systematic error was.
 *
 * A grid of point samples counts a cell as covered wherever a sample lands inside, so a
 * boundary that falls between two sample levels rounds the whole sub-band UP. That bias is
 * invisible almost everywhere and fatal on exactly ONE row: the 60..65 deg band that the
 * +29 row reaches 63.49 deg into. At any sub-sample count that band rounded to 0.75 fill
 * instead of its true 0.698, and `coveredFraction` came out at 0.8962 — ABOVE
 * `COMPLETENESS.coveredFractionCeiling`, i.e. claiming more sphere than the plan can
 * physically reach. Over-reporting coverage is the one direction this model must never
 * fail in, so the elevation extent is solved instead of sampled.
 */
const SUB_YAW = 8;

/** Blind probes used to find an interior elevation before bisecting toward each edge. */
const MERIDIAN_PROBES = 7;

/** 5 deg / 2^12 = 0.0012 deg of residual, three orders under anything the UI grades on. */
const BISECT_STEPS = 12;

/** Tiny slack so a cell sitting exactly on the culling radius is never dropped. */
const CULL_SLACK = 1.02;

/** Engine parity: `coverage._MAX_GAPS`. More than eight gaps is not actionable advice. */
const MAX_GAPS = 8;

/**
 * Engine parity: `coverage._SCOPE_MARGIN_DEG`. Gaps are reported inside the elevation band
 * the capture ACTUALLY ATTEMPTED, shrunk by one cell, so a single-row sweep does not report
 * "the entire sky" as its largest gap and bury the one actionable fact — the missing yaw
 * sector — under a ring that spans every direction.
 */
const SCOPE_MARGIN_DEG = 5;

// =====================================================================================
// Angles
// =====================================================================================

/** Wrap any angle into [0, 360). */
export function wrapYaw360(deg: Degrees): Degrees {
  const w = deg % 360;
  return w < 0 ? w + 360 : w;
}

/** Wrap into (-180, 180]. */
export function wrapSigned180(deg: Degrees): Degrees {
  const w = wrapYaw360(deg);
  return w > 180 ? w - 360 : w;
}

/** Signed shortest yaw delta from `fromDeg` to `toDeg`, in (-180, 180]. Positive = right. */
export function shortestYawDeltaDeg(fromDeg: Degrees, toDeg: Degrees): Degrees {
  return wrapSigned180(toDeg - fromDeg);
}

/** Great-circle angle between two (yaw, pitch) directions, degrees. */
export function greatCircleDeg(yawA: Degrees, pitchA: Degrees, yawB: Degrees, pitchB: Degrees): Degrees {
  const pa = pitchA * DEG;
  const pb = pitchB * DEG;
  const dy = (yawB - yawA) * DEG;
  const c = Math.sin(pa) * Math.sin(pb) + Math.cos(pa) * Math.cos(pb) * Math.cos(dy);
  return Math.acos(c < -1 ? -1 : c > 1 ? 1 : c) / DEG;
}

/**
 * Unit direction for a (yaw, pitch) pair, in the world frame this module defines:
 * +Y is up, yaw 0 looks along +Z, yaw increases toward +X. Handedness is irrelevant
 * because station positions and the camera axis are built by the SAME function — only
 * their mutual consistency matters.
 */
export function directionFromYawPitch(yawDeg: Degrees, pitchDeg: Degrees): [number, number, number] {
  const p = pitchDeg * DEG;
  const y = yawDeg * DEG;
  const cp = Math.cos(p);
  return [cp * Math.sin(y), Math.sin(p), cp * Math.cos(y)];
}

// =====================================================================================
// The frame frustum
// =====================================================================================

/**
 * A rectilinear frame as three orthonormal axes plus two half-angle tangents. Stored flat
 * so the inner loop allocates nothing.
 */
export interface FrameFrustum {
  fx: number;
  fy: number;
  fz: number;
  rx: number;
  ry: number;
  rz: number;
  ux: number;
  uy: number;
  uz: number;
  /** tan(hfov/2) — the short edge in portrait. */
  tanH: number;
  /** tan(vfov/2) — the long edge in portrait. */
  tanV: number;
  yawDeg: Degrees;
  pitchDeg: Degrees;
  /** Angle from the axis to a frame CORNER: the smallest cone that contains the frame. */
  circumRadiusDeg: Degrees;
  /** Angle from the axis to the nearest edge: the largest cone contained BY the frame. */
  inRadiusDeg: Degrees;
}

/**
 * Build the frustum for a frame centred on (yaw, pitch) and rolled by `rollDeg`.
 *
 * The sign of `rollDeg` is deliberately not pinned down: the frame is symmetric in both
 * image axes (|.| <= tan), so +rho and -rho produce mirror-image footprints of identical
 * area and identical worst-case reach. Coverage cannot tell them apart, so the convention
 * cannot be got wrong here. (Guidance CAN tell them apart, which is why Builder V draws
 * from the quaternion.)
 */
export function buildFrustum(yawDeg: Degrees, pitchDeg: Degrees, rollDeg: Degrees, lens: LensGeometry = DEFAULT_LENS): FrameFrustum {
  const y = yawDeg * DEG;
  const p = pitchDeg * DEG;
  const sy = Math.sin(y);
  const cy = Math.cos(y);
  const sp = Math.sin(p);
  const cp = Math.cos(p);

  // Optical axis.
  const fx = cp * sy;
  const fy = sp;
  const fz = cp * cy;
  // Horizontal right, perpendicular to the axis and to world up.
  const r0x = cy;
  const r0y = 0;
  const r0z = -sy;
  // Frame up = f x r0. At pitch 0 this is (0, 1, 0).
  const u0x = -sy * sp;
  const u0y = cp;
  const u0z = -cy * sp;

  const rr = (Number.isFinite(rollDeg) ? rollDeg : 0) * DEG;
  const cr = Math.cos(rr);
  const sr = Math.sin(rr);

  const tanH = Math.tan((lens.hfovDeg / 2) * DEG);
  const tanV = Math.tan((lens.vfovDeg / 2) * DEG);

  return {
    fx,
    fy,
    fz,
    rx: r0x * cr - u0x * sr,
    ry: r0y * cr - u0y * sr,
    rz: r0z * cr - u0z * sr,
    ux: r0x * sr + u0x * cr,
    uy: r0y * sr + u0y * cr,
    uz: r0z * sr + u0z * cr,
    tanH,
    tanV,
    yawDeg,
    pitchDeg,
    circumRadiusDeg: Math.atan(Math.hypot(tanH, tanV)) / DEG,
    inRadiusDeg: Math.min(lens.hfovDeg, lens.vfovDeg) / 2,
  };
}

/** The four half-space tests, on a direction that is already a unit vector. */
function insideDirection(fr: FrameFrustum, dx: number, dy: number, dz: number): boolean {
  const df = dx * fr.fx + dy * fr.fy + dz * fr.fz;
  if (df <= 0) return false;
  const dr = dx * fr.rx + dy * fr.ry + dz * fr.rz;
  if (Math.abs(dr) > fr.tanH * df) return false;
  const du = dx * fr.ux + dy * fr.uy + dz * fr.uz;
  return Math.abs(du) <= fr.tanV * df;
}

/** True when the (yaw, pitch) direction falls inside the frame. Allocation-free. */
export function insideFrustum(fr: FrameFrustum, yawDeg: Degrees, pitchDeg: Degrees): boolean {
  const p = pitchDeg * DEG;
  const y = yawDeg * DEG;
  const cp = Math.cos(p);
  return insideDirection(fr, cp * Math.sin(y), Math.sin(p), cp * Math.cos(y));
}

/** As `insideFrustum`, with the meridian's sin/cos hoisted out of the inner loop. */
function insideMeridian(fr: FrameFrustum, sinYaw: number, cosYaw: number, pitchDeg: Degrees): boolean {
  const p = pitchDeg * DEG;
  const cp = Math.cos(p);
  return insideDirection(fr, cp * sinYaw, Math.sin(p), cp * cosYaw);
}

/** Bisect between a known-outside and a known-inside elevation to find the crossing. */
function bisectMeridian(fr: FrameFrustum, sinYaw: number, cosYaw: number, outsideDeg: Degrees, insideDeg: Degrees): Degrees {
  let a = outsideDeg;
  let b = insideDeg;
  for (let i = 0; i < BISECT_STEPS; i += 1) {
    const m = (a + b) * 0.5;
    if (insideMeridian(fr, sinYaw, cosYaw, m)) b = m;
    else a = m;
  }
  return (a + b) * 0.5;
}

/**
 * Fraction of the meridian segment [loDeg, hiDeg] at one yaw that lies inside the frame.
 *
 * The frame's footprint is the intersection of four half-spaces through the origin — a
 * CONVEX spherical polygon — so its intersection with any great circle is a single arc,
 * and the inside set along a meridian is therefore one contiguous interval. That is what
 * licenses bisecting for the two edges instead of sampling: one probe pass to land inside
 * the interval, then two bisections to place its ends to 0.0012 deg.
 *
 * A sliver narrower than the probe spacing (5/7 deg) is missed and counted as zero, which
 * is the conservative direction and worth at most a hundredth of a cell.
 */
function meridianInsideFraction(fr: FrameFrustum, sinYaw: number, cosYaw: number, loDeg: Degrees, hiDeg: Degrees): number {
  const span = hiDeg - loDeg;
  if (span <= 0) return 0;
  const loIn = insideMeridian(fr, sinYaw, cosYaw, loDeg);
  const hiIn = insideMeridian(fr, sinYaw, cosYaw, hiDeg);
  if (loIn && hiIn) return 1;

  let seed = Number.NaN;
  if (loIn) seed = loDeg;
  else if (hiIn) seed = hiDeg;
  else {
    for (let i = 0; i < MERIDIAN_PROBES; i += 1) {
      const e = loDeg + ((i + 0.5) / MERIDIAN_PROBES) * span;
      if (insideMeridian(fr, sinYaw, cosYaw, e)) {
        seed = e;
        break;
      }
    }
    if (!Number.isFinite(seed)) return 0;
  }

  const lower = loIn ? loDeg : bisectMeridian(fr, sinYaw, cosYaw, loDeg, seed);
  const upper = hiIn ? hiDeg : bisectMeridian(fr, sinYaw, cosYaw, hiDeg, seed);
  const f = (upper - lower) / span;
  return f < 0 ? 0 : f > 1 ? 1 : f;
}

// =====================================================================================
// The grid
// =====================================================================================

/** Centre elevation of a pitch row. Row 0 is the -90..-85 deg band. */
export function cellPitchCentreDeg(pitchIndex: number, cellDeg: number = COVERAGE_GRID.cellDeg): Degrees {
  return -90 + (pitchIndex + 0.5) * cellDeg;
}

/** Centre yaw of a column. Column 0 is the 0..5 deg band, i.e. the capture origin. */
export function cellYawCentreDeg(yawIndex: number, cellDeg: number = COVERAGE_GRID.cellDeg): Degrees {
  return (yawIndex + 0.5) * cellDeg;
}

export function createGrid(
  config: Pick<typeof COVERAGE_GRID, "yawCells" | "pitchCells" | "cellDeg"> = COVERAGE_GRID,
): CoverageGrid {
  const cells = config.yawCells * config.pitchCells;
  return {
    yawCells: config.yawCells,
    pitchCells: config.pitchCells,
    cellDeg: config.cellDeg,
    fill: new Float32Array(cells),
    covered: new Uint8Array(cells),
  };
}

/** Zero a grid in place, reusing its buffers. Cheaper than reallocating mid-session. */
export function clearGrid(grid: CoverageGrid): void {
  grid.fill.fill(0);
  grid.covered.fill(0);
}

/**
 * Fraction of one cell that falls inside the frame: sampled across the cell in yaw, solved
 * exactly in elevation. Two exact bounds keep the expensive path rare — a cell whose centre
 * is within (inRadius - cellRadius) is wholly inside; one beyond
 * (circumRadius + cellRadius) is wholly outside — so only the ~45 cells straddling the
 * footprint's boundary pay for the meridian solve.
 */
function cellFill(fr: FrameFrustum, pitchCentre: Degrees, yawCentre: Degrees, cellDeg: number, cellRadiusDeg: number): number {
  const d = greatCircleDeg(fr.yawDeg, fr.pitchDeg, yawCentre, pitchCentre);
  if (d > fr.circumRadiusDeg + cellRadiusDeg) return 0;
  if (d + cellRadiusDeg <= fr.inRadiusDeg) return 1;

  const half = cellDeg / 2;
  const lo = pitchCentre - half;
  const hi = pitchCentre + half;
  const step = cellDeg / SUB_YAW;
  const y0 = yawCentre - half + step / 2;
  let sum = 0;
  for (let b = 0; b < SUB_YAW; b += 1) {
    const yr = (y0 + b * step) * DEG;
    sum += meridianInsideFraction(fr, Math.sin(yr), Math.cos(yr), lo, hi);
  }
  return sum / SUB_YAW;
}

/**
 * The union of two coverages is at least the larger of them, so `fill = max(old, new)` is
 * a strict LOWER BOUND on the true unioned area of a cell. That direction is the safe one:
 * this model can only ever tell a user they have less coverage than they do, never more.
 * (Exactness would need a sub-cell bitmask per cell; the contract fixes `fill` at one
 * float per cell, and the residual error is confined to the outermost ring of cells that
 * no single frame ever covers completely — measured at ~0.002 of `coveredFraction` on a
 * perfect three-row sweep.)
 */
function accumulate(grid: CoverageGrid, index: number, f: number): boolean {
  if (f <= grid.fill[index]) return false;
  grid.fill[index] = f;
  if (f >= COVERAGE_GRID.coveredAtFill && grid.covered[index] === 0) {
    grid.covered[index] = 1;
    return true;
  }
  return false;
}

/**
 * Stamp one frame's footprint onto the grid. Returns the number of cells that newly
 * crossed `COVERAGE_GRID.coveredAtFill` — what the UI turns into a tile-lighting tick.
 *
 * Call at `COVERAGE_GRID.stampHz` and ONLY while no `stop`-severity warning is active:
 * footage the engine will refuse must not paint the dome green.
 */
export function stampCone(grid: CoverageGrid, sample: OrientationSample, lens: LensGeometry = DEFAULT_LENS): number {
  if (!Number.isFinite(sample.yawDeg) || !Number.isFinite(sample.pitchDeg)) return 0;

  const cellDeg = grid.cellDeg;
  const fr = buildFrustum(wrapYaw360(sample.yawDeg), clampPitch(sample.pitchDeg), sample.rollDeg, lens);
  // Great-circle distance from a cell centre to its furthest corner. The yaw half-span
  // shrinks by cos(elev) away from the equator, so the equatorial value bounds every row.
  const cellRadiusDeg = Math.hypot(cellDeg / 2, cellDeg / 2) * CULL_SLACK;
  const limit = fr.circumRadiusDeg + cellRadiusDeg;

  const piLo = Math.max(0, Math.floor((fr.pitchDeg - limit + 90) / cellDeg));
  const piHi = Math.min(grid.pitchCells - 1, Math.floor((fr.pitchDeg + limit + 90) / cellDeg));

  const cosLimit = Math.cos(limit * DEG);
  const sinAxis = Math.sin(fr.pitchDeg * DEG);
  const cosAxis = Math.cos(fr.pitchDeg * DEG);

  let newlyCovered = 0;
  for (let pi = piLo; pi <= piHi; pi += 1) {
    const pitchCentre = cellPitchCentreDeg(pi, cellDeg);
    const sinCell = Math.sin(pitchCentre * DEG);
    const cosCell = Math.cos(pitchCentre * DEG);

    // Solve cos(limit) = sin(a)sin(b) + cos(a)cos(b)cos(dYaw) for the yaw half-window.
    // Poles (cosCell*cosAxis ~ 0) collapse the equation: sweep the whole ring instead.
    const denom = cosCell * cosAxis;
    let yiLo = 0;
    let yiHi = grid.yawCells - 1;
    if (denom > 1e-9) {
      const cosDy = (cosLimit - sinCell * sinAxis) / denom;
      if (cosDy > 1) continue; // this row cannot reach the frame at any yaw
      if (cosDy > -1) {
        const halfWindow = Math.acos(cosDy) / DEG;
        const centreIndex = fr.yawDeg / cellDeg;
        yiLo = Math.floor(centreIndex - halfWindow / cellDeg) - 1;
        yiHi = Math.ceil(centreIndex + halfWindow / cellDeg) + 1;
        if (yiHi - yiLo >= grid.yawCells) {
          yiLo = 0;
          yiHi = grid.yawCells - 1;
        }
      }
    } else if (Math.abs(pitchCentre - fr.pitchDeg) > limit) {
      continue;
    }

    const rowBase = pi * grid.yawCells;
    for (let k = yiLo; k <= yiHi; k += 1) {
      const yi = ((k % grid.yawCells) + grid.yawCells) % grid.yawCells;
      const index = rowBase + yi;
      if (grid.fill[index] >= 1) continue;
      const f = cellFill(fr, pitchCentre, cellYawCentreDeg(yi, cellDeg), cellDeg, cellRadiusDeg);
      if (f > 0 && accumulate(grid, index, f)) newlyCovered += 1;
    }
  }
  return newlyCovered;
}

/**
 * The same stamp with every culling optimisation removed. Kept because the fast path's
 * correctness is an assertion the tests can make rather than a claim this file makes.
 */
export function stampConeExhaustive(grid: CoverageGrid, sample: OrientationSample, lens: LensGeometry = DEFAULT_LENS): number {
  const cellDeg = grid.cellDeg;
  const fr = buildFrustum(wrapYaw360(sample.yawDeg), clampPitch(sample.pitchDeg), sample.rollDeg, lens);
  const cellRadiusDeg = Math.hypot(cellDeg / 2, cellDeg / 2) * CULL_SLACK;
  let newlyCovered = 0;
  for (let pi = 0; pi < grid.pitchCells; pi += 1) {
    const pitchCentre = cellPitchCentreDeg(pi, cellDeg);
    for (let yi = 0; yi < grid.yawCells; yi += 1) {
      const f = cellFill(fr, pitchCentre, cellYawCentreDeg(yi, cellDeg), cellDeg, cellRadiusDeg);
      if (f > 0 && accumulate(grid, pi * grid.yawCells + yi, f)) newlyCovered += 1;
    }
  }
  return newlyCovered;
}

function clampPitch(deg: Degrees): Degrees {
  return deg < -90 ? -90 : deg > 90 ? 90 : deg;
}

// =====================================================================================
// Solid-angle weighting
// =====================================================================================

/**
 * `geometry.sphere_solid_angle_weights`: cos(elevation), normalised to sum to 1 over the
 * whole grid. A cell beside the pole covers almost no sphere and must not count as if it
 * did — the failure this weighting exists to prevent is a capture that only grazed the
 * ceiling scoring better than one that swept the whole horizon.
 */
export function rowCosWeights(pitchCells: number, cellDeg: number): Float64Array {
  const w = new Float64Array(pitchCells);
  for (let pi = 0; pi < pitchCells; pi += 1) w[pi] = Math.cos(cellPitchCentreDeg(pi, cellDeg) * DEG);
  return w;
}

function weightedFill(grid: CoverageGrid, weights: Float64Array, bandDeg: number | null): number {
  let num = 0;
  let den = 0;
  for (let pi = 0; pi < grid.pitchCells; pi += 1) {
    const elev = cellPitchCentreDeg(pi, grid.cellDeg);
    if (bandDeg !== null && Math.abs(elev) > bandDeg) continue;
    const w = weights[pi];
    if (w <= 0) continue;
    let rowSum = 0;
    const base = pi * grid.yawCells;
    for (let yi = 0; yi < grid.yawCells; yi += 1) rowSum += grid.fill[base + yi];
    num += w * rowSum;
    den += w * grid.yawCells;
  }
  return den > 0 ? num / den : 0;
}

// =====================================================================================
// Reach, spans and gaps
// =====================================================================================

/**
 * Frame-EDGE elevation reach, the number `poles._CAP_DEG = 60` is compared against.
 *
 * Sub-cell: the highest row that has any fill contributes `rowLowEdge + maxFill*cellDeg`.
 * On a perfect +29/0/-29 sweep the 60..65 row peaks at 0.70 fill directly above a station,
 * so this returns 63.5 — the analytic reach — instead of the 65 a whole-cell reading would
 * claim or the 62.5 a cell-centre reading would concede.
 */
function pitchReach(grid: CoverageGrid): { minDeg: Degrees; maxDeg: Degrees } {
  let maxDeg = 0;
  let minDeg = 0;
  let seen = false;
  for (let pi = 0; pi < grid.pitchCells; pi += 1) {
    let rowMax = 0;
    const base = pi * grid.yawCells;
    for (let yi = 0; yi < grid.yawCells; yi += 1) {
      const f = grid.fill[base + yi];
      if (f > rowMax) rowMax = f;
    }
    if (rowMax <= 0) continue;
    const lowEdge = -90 + pi * grid.cellDeg;
    const highEdge = lowEdge + grid.cellDeg;
    const up = lowEdge + rowMax * grid.cellDeg;
    const down = highEdge - rowMax * grid.cellDeg;
    if (!seen) {
      maxDeg = up;
      minDeg = down;
      seen = true;
    } else {
      if (up > maxDeg) maxDeg = up;
      if (down < minDeg) minDeg = down;
    }
  }
  return seen ? { minDeg, maxDeg } : { minDeg: 0, maxDeg: 0 };
}

/** `coverage._largest_uncovered_arc_deg`, on a cyclic boolean column mask. */
export function largestUncoveredArcDeg(colCovered: Uint8Array | boolean[]): Degrees {
  const n = colCovered.length;
  if (n === 0) return 360;
  let any = false;
  let all = true;
  for (let i = 0; i < n; i += 1) {
    if (colCovered[i]) any = true;
    else all = false;
  }
  if (all) return 0;
  if (!any) return 360;
  let best = 0;
  let run = 0;
  for (let k = 0; k < 2 * n; k += 1) {
    if (!colCovered[k % n]) {
      run += 1;
      const capped = run > n ? n : run;
      if (capped > best) best = capped;
    } else {
      run = 0;
    }
  }
  return (360 * best) / n;
}

/** `coverage._cyclic_span`: `[startCol, nCols]` of a possibly wrap-around column set. */
export function cyclicSpan(cols: boolean[]): [number, number] {
  const n = cols.length;
  let count = 0;
  for (let i = 0; i < n; i += 1) if (cols[i]) count += 1;
  if (count === 0) return [0, 0];
  if (count === n) return [0, n];

  let bestLen = 0;
  let bestStart = 0;
  let runLen = 0;
  let runStart = 0;
  for (let k = 0; k < 2 * n; k += 1) {
    const c = k % n;
    if (!cols[c]) {
      if (runLen === 0) runStart = c;
      runLen += 1;
      if (runLen > bestLen && runLen <= n) {
        bestLen = runLen;
        bestStart = runStart;
      }
    } else {
      runLen = 0;
    }
  }
  return [(bestStart + bestLen) % n, n - bestLen];
}

/**
 * Target elevation of a row: the lens-derived separation (see `rowSeparationDeg`), shifted
 * down by `ROW_BIAS_DEG` so the lattice buys floor reach with ceiling reach.
 *
 * The separation and the bias are independent on purpose. Separation is the overlap
 * constraint and must follow the lens; the bias is a value judgement about which end of the
 * room is worth photographing, and it is the same three degrees on every lens.
 */
export function rowTargetPitchDeg(row: RowId, lens: LensGeometry = DEFAULT_LENS): Degrees {
  const nominal = ROW_PLAN[row].targetPitchDeg;
  if (row === "level") return ROW_BIAS_DEG;
  const sep = Math.round(rowSeparationDeg(lens));
  return (nominal > 0 ? sep : -sep) + ROW_BIAS_DEG;
}

/**
 * Yaw span of one row read off the grid, exactly the way the engine reads it: a column
 * counts if ANY cell in the row's elevation band is covered, and the span is 360 minus the
 * largest uncovered arc. Independent of whether individual stations dwelt long enough,
 * which is what makes it a fair answer to "did you go all the way round".
 */
export function rowYawSpanFromGrid(grid: CoverageGrid, targetPitchDeg: Degrees): Degrees {
  const cols = new Uint8Array(grid.yawCells);
  let bandRows = 0;
  for (let pi = 0; pi < grid.pitchCells; pi += 1) {
    const elev = cellPitchCentreDeg(pi, grid.cellDeg);
    if (Math.abs(elev - targetPitchDeg) > grid.cellDeg) continue;
    bandRows += 1;
    const base = pi * grid.yawCells;
    for (let yi = 0; yi < grid.yawCells; yi += 1) if (grid.covered[base + yi]) cols[yi] = 1;
  }
  if (bandRows === 0) return 0;
  return 360 - largestUncoveredArcDeg(cols);
}

/**
 * Contiguous uncovered runs, largest first — 8-connected and cyclic in yaw, like
 * `coverage._find_gaps`. Restricted to |elev| <= 60 (nothing above that line may be
 * repaired by `poles.py`, so nothing above it needs to be chased) intersected with the
 * elevation band the capture attempted, shrunk by one cell.
 *
 * Single-cell components are dropped: `COMPLETENESS.maxGapArcDeg` tolerates an uncovered
 * arc up to one 5-deg cell, and a one-cell speck is noise, not an instruction.
 */
export function findGaps(grid: CoverageGrid): CoverageGap[] {
  const reach = pitchReach(grid);
  const nCells = grid.yawCells * grid.pitchCells;
  const scope = new Uint8Array(grid.pitchCells);
  let scoped = 0;
  for (let pi = 0; pi < grid.pitchCells; pi += 1) {
    const elev = cellPitchCentreDeg(pi, grid.cellDeg);
    const inCapBand = Math.abs(elev) <= COMPLETENESS.pitchMaxDeg;
    const attempted = elev <= reach.maxDeg - SCOPE_MARGIN_DEG && elev >= reach.minDeg + SCOPE_MARGIN_DEG;
    if (inCapBand && attempted) {
      scope[pi] = 1;
      scoped += 1;
    }
  }
  if (scoped === 0) return [];

  // Union-find over uncovered, in-scope cells. Column 0 and column N-1 are neighbours.
  const parent = new Int32Array(nCells);
  for (let i = 0; i < nCells; i += 1) parent[i] = -1;
  const find = (a: number): number => {
    let r = a;
    while (parent[r] !== r) r = parent[r];
    let c = a;
    while (parent[c] !== r) {
      const next = parent[c];
      parent[c] = r;
      c = next;
    }
    return r;
  };
  const union = (a: number, b: number): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra < rb ? rb : ra] = ra < rb ? ra : rb;
  };
  const open = (pi: number, yi: number): boolean =>
    scope[pi] === 1 && grid.covered[pi * grid.yawCells + yi] === 0;

  for (let pi = 0; pi < grid.pitchCells; pi += 1) {
    for (let yi = 0; yi < grid.yawCells; yi += 1) {
      if (!open(pi, yi)) continue;
      const idx = pi * grid.yawCells + yi;
      parent[idx] = idx;
    }
  }
  const yc = grid.yawCells;
  for (let pi = 0; pi < grid.pitchCells; pi += 1) {
    for (let yi = 0; yi < yc; yi += 1) {
      if (!open(pi, yi)) continue;
      const idx = pi * yc + yi;
      const left = (yi - 1 + yc) % yc;
      if (open(pi, left)) union(idx, pi * yc + left);
      if (pi > 0) {
        const right = (yi + 1) % yc;
        if (open(pi - 1, yi)) union(idx, (pi - 1) * yc + yi);
        if (open(pi - 1, left)) union(idx, (pi - 1) * yc + left);
        if (open(pi - 1, right)) union(idx, (pi - 1) * yc + right);
      }
    }
  }

  const weights = rowCosWeights(grid.pitchCells, grid.cellDeg);
  let totalWeight = 0;
  for (let pi = 0; pi < grid.pitchCells; pi += 1) totalWeight += weights[pi] * grid.yawCells;

  interface Component {
    cells: number;
    area: number;
    rowMin: number;
    rowMax: number;
    cols: boolean[];
  }
  const groups = new Map<number, Component>();
  for (let pi = 0; pi < grid.pitchCells; pi += 1) {
    for (let yi = 0; yi < yc; yi += 1) {
      if (!open(pi, yi)) continue;
      const root = find(pi * yc + yi);
      let g = groups.get(root);
      if (g === undefined) {
        g = { cells: 0, area: 0, rowMin: pi, rowMax: pi, cols: new Array<boolean>(yc).fill(false) };
        groups.set(root, g);
      }
      g.cells += 1;
      g.area += weights[pi];
      if (pi < g.rowMin) g.rowMin = pi;
      if (pi > g.rowMax) g.rowMax = pi;
      g.cols[yi] = true;
    }
  }

  const gaps: { gap: CoverageGap; area: number }[] = [];
  groups.forEach((g) => {
    if (g.cells <= 1) return;
    const [start, span] = cyclicSpan(g.cols);
    if (span === 0) return;
    const yawStartDeg = wrapYaw360(start * grid.cellDeg);
    const yawEndDeg = yawStartDeg + span * grid.cellDeg;
    const pitchStartDeg = -90 + g.rowMin * grid.cellDeg;
    const pitchEndDeg = -90 + (g.rowMax + 1) * grid.cellDeg;
    const centrePitch = (pitchStartDeg + pitchEndDeg) / 2;
    let nearestRow: RowId = ROW_IDS[0];
    let bestDelta = Infinity;
    for (const row of ROW_IDS) {
      const delta = Math.abs(ROW_PLAN[row].targetPitchDeg - centrePitch);
      if (delta < bestDelta) {
        bestDelta = delta;
        nearestRow = row;
      }
    }
    gaps.push({
      area: g.area,
      gap: {
        yawStartDeg,
        yawEndDeg,
        pitchStartDeg,
        pitchEndDeg,
        solidAngleFraction: totalWeight > 0 ? g.area / totalWeight : 0,
        nearestRow,
      },
    });
  });

  gaps.sort((a, b) => b.area - a.area);
  return gaps.slice(0, MAX_GAPS).map((g) => g.gap);
}

// =====================================================================================
// The report
// =====================================================================================

/**
 * Per-row facts only the host can know, because they are accumulated over TIME rather than
 * read off the grid: how much yaw the user actually travelled (which legitimately exceeds
 * 360 — `YAW_SPAN.targetDeg` is 370) and how far the frame centre wandered while that row
 * was the active one. Optional: without them `report` derives conservative equivalents
 * from the grid, which is what a server-side or restored-from-draft call gets.
 */
export interface TrackedRowStats {
  yawSpanDeg?: Degrees;
  pitchMinDeg?: Degrees;
  pitchMaxDeg?: Degrees;
}

/** Which COMPLETENESS criteria are unmet. Shared by `report` and `PlanApi.blockers`. */
/**
 * How much sphere THIS lens and lattice can physically reach.
 *
 * The three rows sit at `rowTargetPitchDeg`, each frame reaches `vfov/2` past its centre,
 * and the band between the outermost edges is all a three-row sweep can ever photograph.
 */
export function achievableCoverage(lens: LensGeometry = DEFAULT_LENS): {
  coveredFraction: number;
  pitchMaxDeg: Degrees;
  pitchMinDeg: Degrees;
} {
  const half = lens.vfovDeg / 2;
  const up = rowTargetPitchDeg("up", lens) + half;
  const down = rowTargetPitchDeg("down", lens) - half;
  const rad = Math.PI / 180;
  return {
    coveredFraction: (Math.sin(up * rad) + Math.sin(Math.abs(down) * rad)) / 2,
    pitchMaxDeg: up,
    pitchMinDeg: down,
  };
}

/**
 * How much may be DEMANDED of this lens — the ceiling above, less the slack the coaching
 * spends, and never tighter than the shipped constants.
 *
 * WHY THIS IS A FUNCTION AND NOT A CONSTANT. `COMPLETENESS` was measured against a 9:16
 * frame, where three rows reach +55.5/-71.5 and a perfect sweep scores 0.8856. A 4:3 frame
 * is 14 deg shorter, so the same three rows reach barely +45 and a PERFECT sweep on that
 * phone scores about 0.80 — under the 0.855 constant. Held against a fixed number, every
 * correct capture on a 4:3 phone would be refused for a shortfall the phone cannot close at
 * any skill level. That is the same failure as grading a green capture as incomplete, and it
 * is the failure `COMPLETENESS.coveredFraction` was chosen to make impossible.
 *
 * `Math.min` / `Math.max` against the constants means this can only ever LOOSEN. A 9:16
 * phone keeps exactly the thresholds every earlier number was measured against; a narrower
 * frame is judged against what it can actually do, and nothing is judged more harshly than
 * before.
 */
const GATE_MARGIN = {
  /** Measured: a perfect sweep scores 0.8856 and the greenest failing row 0.8611. */
  coveredFraction: 0.031,
  /** The row-pitch error the coaching grades GREEN, which costs reach one-for-one. */
  reachDeg: 3,
} as const;

export function gatesForLens(lens: LensGeometry = DEFAULT_LENS): {
  coveredFraction: number;
  pitchMaxDeg: Degrees;
  pitchMinDeg: Degrees;
} {
  const reach = achievableCoverage(lens);
  return {
    coveredFraction: Math.min(
      COMPLETENESS.coveredFraction,
      reach.coveredFraction - GATE_MARGIN.coveredFraction,
    ),
    pitchMaxDeg: Math.min(COMPLETENESS.pitchMaxDeg, reach.pitchMaxDeg - GATE_MARGIN.reachDeg),
    pitchMinDeg: Math.max(COMPLETENESS.pitchMinDeg, reach.pitchMinDeg + GATE_MARGIN.reachDeg),
  };
}

export function computeBlockers(
  report: Omit<CoverageReport, "blockedBy" | "complete" | "gaps">,
  lens: LensGeometry = DEFAULT_LENS,
): CompletionBlocker[] {
  const gates = gatesForLens(lens);
  const out: CompletionBlocker[] = [];
  if (report.coveredFraction < gates.coveredFraction) out.push("covered_fraction");
  if (report.horizonFraction30 < COMPLETENESS.horizonFraction30) out.push("horizon_30");
  if (report.horizonFraction60 < COMPLETENESS.horizonFraction60) out.push("horizon_60");
  if (report.pitchMaxDeg < gates.pitchMaxDeg) out.push("pitch_max");
  if (report.pitchMinDeg > gates.pitchMinDeg) out.push("pitch_min");
  for (const row of ROW_IDS) {
    const progress = report.rows[row];
    // 1e-9 of slack: a grid-derived span tops out at exactly 360, which is exactly
    // acceptMinDeg, and a float comparison must not turn that into a permanent blocker.
    if (progress.yawSpanDeg + 1e-9 < YAW_SPAN.acceptMinDeg) out.push(`yaw_span_${row}` as CompletionBlocker);
    if (progress.filledStations < progress.requiredStations) out.push(`row_${row}` as CompletionBlocker);
  }
  return out;
}

/**
 * The live coverage answer, in the same vocabulary as the engine's `Coverage`.
 *
 * `tracked` is an extra OPTIONAL parameter, which keeps this assignable to
 * `CoverageApi["report"]` while letting Builder P feed the accumulated per-row yaw travel
 * it owns. Without it the spans are derived from the grid and cap at 360.
 */
export function report(
  grid: CoverageGrid,
  stations: StationState[],
  lens: LensGeometry = DEFAULT_LENS,
  tracked?: Partial<Record<RowId, TrackedRowStats>>,
): CoverageReport {
  const weights = rowCosWeights(grid.pitchCells, grid.cellDeg);
  const coveredFraction = weightedFill(grid, weights, null);
  const horizonFraction30 = weightedFill(grid, weights, 30);
  // Measured against the band this lens can actually reach, not a constant it cannot.
  const horizonFraction60 = weightedFill(grid, weights, gatesForLens(lens).pitchMaxDeg);
  const reach = pitchReach(grid);

  const rows = {} as Record<RowId, RowProgress>;
  for (const row of ROW_IDS) {
    const plan = ROW_PLAN[row];
    const target = rowTargetPitchDeg(row, lens);
    let filled = 0;
    for (const s of stations) if (s.row === row && s.filled) filled += 1;
    const t = tracked?.[row];
    const yawSpanDeg = t?.yawSpanDeg ?? rowYawSpanFromGrid(grid, target);
    rows[row] = {
      row,
      filledStations: filled,
      requiredStations: plan.requiredStations,
      yawSpanDeg,
      pitchMinDeg: t?.pitchMinDeg ?? target,
      pitchMaxDeg: t?.pitchMaxDeg ?? target,
      complete: filled >= plan.requiredStations && yawSpanDeg + 1e-9 >= YAW_SPAN.acceptMinDeg,
    };
  }

  const blockedBy = computeBlockers(
    {
      coveredFraction,
      horizonFraction30,
      horizonFraction60,
      pitchMaxDeg: reach.maxDeg,
      pitchMinDeg: reach.minDeg,
      rows,
    },
    lens,
  );

  return {
    coveredFraction,
    horizonFraction30,
    horizonFraction60,
    pitchMaxDeg: reach.maxDeg,
    pitchMinDeg: reach.minDeg,
    rows,
    gaps: findGaps(grid),
    complete: blockedBy.length === 0,
    blockedBy,
  };
}

// =====================================================================================
// Persistence
// =====================================================================================

export function serialize(
  grid: CoverageGrid,
  stations: StationState[],
  rows: Record<RowId, RowProgress>,
  lens: LensGeometry,
): CoverageSnapshot {
  // `.slice()` detaches from any pooled buffer, so the snapshot is safe to structured-clone
  // into IndexedDB while the live grid keeps being stamped.
  return {
    fill: grid.fill.slice().buffer,
    covered: grid.covered.slice().buffer,
    stations: stations.map((s) => ({ ...s })),
    rows: { ...rows },
    lens: { ...lens },
  };
}

export function deserialize(snapshot: CoverageSnapshot): { grid: CoverageGrid; stations: StationState[] } {
  const fill = new Float32Array(snapshot.fill);
  const covered = new Uint8Array(snapshot.covered);
  const cells = fill.length;
  // A draft written by an older build could carry a different lattice; fall back to the
  // contract's shape rather than producing a grid whose stride is silently wrong.
  const pitchCells = cells === COVERAGE_GRID.totalCells ? COVERAGE_GRID.pitchCells : Math.max(1, Math.round(cells / COVERAGE_GRID.yawCells));
  return {
    grid: {
      yawCells: COVERAGE_GRID.yawCells,
      pitchCells,
      cellDeg: COVERAGE_GRID.cellDeg,
      fill,
      covered: covered.length === cells ? covered : new Uint8Array(cells),
    },
    stations: snapshot.stations.map((s) => ({ ...s })),
  };
}

/** Base64 of the covered mask, for `CaptureTelemetry.coverage.coveredMaskBase64`. */
export function coveredMaskBase64(grid: CoverageGrid): string {
  let binary = "";
  for (let i = 0; i < grid.covered.length; i += 1) binary += String.fromCharCode(grid.covered[i]);
  if (typeof btoa === "function") return btoa(binary);
  // Node (tests, and any future SSR path) has no btoa on older runtimes.
  const g = globalThis as { Buffer?: { from(input: string, enc: string): { toString(enc: string): string } } };
  if (g.Buffer) return g.Buffer.from(binary, "binary").toString("base64");
  return "";
}

/**
 * The nearest direction that is NOT yet covered, as a (yaw, pitch) pair — what the
 * "you are missing the corner behind you" arrow points at. Restricted to |elev| <= 60,
 * because nothing above that line is the user's problem to fix.
 */
export function nearestUncovered(
  grid: CoverageGrid,
  fromYawDeg: Degrees,
  fromPitchDeg: Degrees,
): { yawDeg: Degrees; pitchDeg: Degrees; distanceDeg: Degrees } | null {
  let best: { yawDeg: Degrees; pitchDeg: Degrees; distanceDeg: Degrees } | null = null;
  for (let pi = 0; pi < grid.pitchCells; pi += 1) {
    const pitch = cellPitchCentreDeg(pi, grid.cellDeg);
    if (Math.abs(pitch) > COMPLETENESS.pitchMaxDeg) continue;
    const base = pi * grid.yawCells;
    for (let yi = 0; yi < grid.yawCells; yi += 1) {
      if (grid.covered[base + yi]) continue;
      const yaw = cellYawCentreDeg(yi, grid.cellDeg);
      const d = greatCircleDeg(fromYawDeg, fromPitchDeg, yaw, pitch);
      if (best === null || d < best.distanceDeg) best = { yawDeg: yaw, pitchDeg: pitch, distanceDeg: d };
    }
  }
  return best;
}

/** The `CoverageApi` surface, so a mis-typed export is a compile error here, not in P. */
export const coverageApi = {
  createGrid,
  stampCone,
  report,
  findGaps,
  serialize,
  deserialize,
} satisfies CoverageApi;
