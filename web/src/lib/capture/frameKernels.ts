/**
 * FRAME KERNELS — the pixel maths, lifted out of the React hook so a worker can run it.
 * =====================================================================================
 *
 * Every function below is COPIED VERBATIM from `useCamera.ts`. Not one constant, threshold
 * or loop bound has been changed: `__tests__/sensors.test.ts` pins `analyseFrameLuma` to
 * 136.6 at mid-grey and `quadrantFlow` to a 6-px synthetic shift, and those assertions must
 * keep holding through `useCamera`'s re-export. The move is purely about WHERE the code can
 * be loaded, not what it computes.
 *
 * ── WHY THIS FILE EXISTS AT ALL ───────────────────────────────────────────────────────
 * `useCamera` called these from `requestVideoFrameCallback`, which meant `drawImage` +
 * `getImageData` ran ON THE MAIN THREAD 15 times a second (10 Hz luma, 5 Hz flow) while a
 * hardware H.264 encoder was mid-frame. `getImageData` is a synchronous GPU->CPU readback:
 * it cannot return until the GPU has flushed everything queued ahead of it, so the cost is
 * not the pixels. `scripts/bench-frame-kernels.mjs` runs exactly these functions over
 * buffers of exactly the shipped geometry, with no canvas involved: 0.15 ms for one
 * `analyseFrameLuma` over 64x114, 0.34 ms for one `flowStep` over 192x256, 3.4 ms per
 * SECOND of capture for both clocks together — 0.11 ms of an average 33 ms frame. Everything
 * the phone spends beyond that is the readback, which is why moving the arithmetic changes
 * nothing and moving the READBACK is the whole fix. `frameAnalysis.ts` can only do that if
 * the arithmetic lives in a module with no React and no `document` in it.
 *
 * ── WHAT MAY AND MAY NOT BE IMPORTED HERE ─────────────────────────────────────────────
 * `./types` is imported with `import type` ONLY. That keeps the erased worker bundle down to
 * this file plus `frameAnalysis.worker.ts`; a value import of `types.ts` would drag its
 * ~100 KB of contract constants into the worker chunk for the sake of two integers that the
 * host already knows and can put in the message instead.
 */

import type { FlowSample, LensGeometry, Millis } from "./types";

const DEG = Math.PI / 180;

/** sRGB 8-bit -> linear, precomputed once. */
const LINEARIZE = (() => {
  const t = new Float32Array(256);
  for (let i = 0; i < 256; i += 1) {
    const c = i / 255;
    t[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  return t;
})();

export interface LumaStats {
  /** Mean CIE L*, rescaled 0-255 the way OpenCV scales its L channel. */
  meanLuma: number;
  /** Fraction of pixels sitting on 0 or 255 — extract._clipping_penalty's input. */
  clippedFraction: number;
}

/**
 * Mean L* and clipped fraction over an RGBA buffer.
 *
 * L* rather than a plain grey average because that is what the engine measures: the two real
 * videos came in at 163 and 144, the synthetics at 117-132, and validate.MIN_MEAN_BRIGHTNESS
 * = 25 is quoted on the same scale. A luma average would read systematically brighter in
 * dark rooms and would move the block threshold without anyone noticing.
 */
export function analyseFrameLuma(rgba: Uint8ClampedArray): LumaStats {
  const pixels = rgba.length >> 2;
  if (pixels === 0) return { meanLuma: 0, clippedFraction: 0 };

  let sumL = 0;
  let clipped = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    const r = rgba[i];
    const g = rgba[i + 1];
    const b = rgba[i + 2];
    const y = 0.2126 * LINEARIZE[r] + 0.7152 * LINEARIZE[g] + 0.0722 * LINEARIZE[b];
    const f = y > 0.008856 ? Math.cbrt(y) : 7.787 * y + 16 / 116;
    sumL += (116 * f - 16) * 2.55;

    const grey = 0.299 * r + 0.587 * g + 0.114 * b;
    if (grey >= 254.5 || grey <= 0.5) clipped += 1;
  }

  return { meanLuma: sumL / pixels, clippedFraction: clipped / pixels };
}

/** Zero-mean profile of one quadrant along an axis, for the flow correlator. */
export function profiles(
  grey: Float32Array,
  canvasW: number,
  qx: number,
  qy: number,
  qw: number,
  qh: number,
): { cols: Float32Array; rows: Float32Array } {
  const cols = new Float32Array(qw);
  const rows = new Float32Array(qh);
  for (let y = 0; y < qh; y += 1) {
    let rowSum = 0;
    const base = (qy + y) * canvasW + qx;
    for (let x = 0; x < qw; x += 1) {
      const v = grey[base + x];
      cols[x] += v;
      rowSum += v;
    }
    rows[y] = rowSum / qw;
  }
  for (let x = 0; x < qw; x += 1) cols[x] /= qh;
  return { cols, rows };
}

export interface LagResult {
  shift: number;
  peak: number;
  valid: boolean;
}

/**
 * Normalised cross-correlation of two 1-D profiles over +/- maxLag, with a parabolic
 * sub-pixel refinement of the peak.
 *
 * This is a projection correlator, not the FFT phase correlation the contract's prose
 * describes, and the difference is deliberate: an FFT would drag a transform implementation
 * into a hot path for no gain here. The statistic that matters is the DIFFERENCE between the
 * four quadrants' shifts, both methods estimate the same shift for the small displacements
 * involved (a 24 deg/s sweep moves ~4.8 deg, ~68 px at 640 width, between 5 Hz samples), and
 * the thresholds are quoted in px at 640 width, which transfers unchanged.
 */
export function bestLag(a: Float32Array, b: Float32Array, maxLag: number): LagResult {
  const n = a.length;
  let meanA = 0;
  for (let i = 0; i < n; i += 1) meanA += a[i];
  meanA /= n;

  let varA = 0;
  for (let i = 0; i < n; i += 1) varA += (a[i] - meanA) ** 2;
  // A flat wall or a black frame carries no shift information at all. Saying so is the
  // whole point of FlowSample.valid.
  if (Math.sqrt(varA / n) < 1.5) return { shift: 0, peak: 0, valid: false };

  let bestScore = -Infinity;
  let bestIndex = 0;
  const scores = new Float32Array(2 * maxLag + 1);

  for (let lag = -maxLag; lag <= maxLag; lag += 1) {
    const from = Math.max(0, -lag);
    const to = Math.min(n, n - lag);
    const count = to - from;
    if (count < n / 2) continue;

    let sa = 0;
    let sb = 0;
    let saa = 0;
    let sbb = 0;
    let sab = 0;
    for (let i = from; i < to; i += 1) {
      const va = a[i];
      const vb = b[i + lag];
      sa += va;
      sb += vb;
      saa += va * va;
      sbb += vb * vb;
      sab += va * vb;
    }
    const cov = sab / count - (sa / count) * (sb / count);
    const sda = Math.sqrt(Math.max(1e-6, saa / count - (sa / count) ** 2));
    const sdb = Math.sqrt(Math.max(1e-6, sbb / count - (sb / count) ** 2));
    const score = cov / (sda * sdb);
    scores[lag + maxLag] = score;
    if (score > bestScore) {
      bestScore = score;
      bestIndex = lag + maxLag;
    }
  }

  if (bestScore < 0.4) return { shift: 0, peak: bestScore, valid: false };

  let shift = bestIndex - maxLag;
  if (bestIndex > 0 && bestIndex < scores.length - 1) {
    const l = scores[bestIndex - 1];
    const c = scores[bestIndex];
    const r = scores[bestIndex + 1];
    const denom = l - 2 * c + r;
    if (Math.abs(denom) > 1e-6) shift += (0.5 * (l - r)) / denom;
  }
  return { shift, peak: bestScore, valid: true };
}

export interface FlowFrame {
  /** Grey plane of the whole downscaled frame, row-major, `canvasW * canvasH` long. */
  grey: Float32Array;
  canvasW: number;
  canvasH: number;
}

/**
 * Quadrant optical-flow divergence.
 *
 * A pure rotation moves all four quadrants together. A translation moves them by different
 * amounts, because each quadrant sees a different depth — which is precisely the differential
 * the engine's rotational-purity residual measures after the fact. This is the only walking
 * signal that survives when the motion sensors are denied.
 *
 * `frameW`/`frameH` are the DECODED frame dimensions, needed to normalise a shift measured on
 * the small canvas back to the contract's 640-px-wide reference.
 */
export function quadrantFlow(
  previous: FlowFrame,
  current: FlowFrame,
  frameW: number,
  frameH: number,
  pxPerDegAt640: number,
  t: Millis,
): FlowSample {
  const { canvasW, canvasH } = current;
  const qw = canvasW >> 1;
  const qh = canvasH >> 1;

  // A shift of one canvas pixel is (frameW / canvasW) frame pixels, which is
  // (frameW / canvasW) * (640 / frameW) = 640 / canvasW pixels at the reference width. The
  // vertical factor differs whenever the canvas aspect is not the frame aspect.
  const sx = 640 / canvasW;
  const sy = (frameH * 640) / (canvasH * Math.max(1, frameW));

  const shifts: { x: number; y: number }[] = [];
  for (let qi = 0; qi < 4; qi += 1) {
    const qx = (qi & 1) * qw;
    const qy = (qi >> 1) * qh;
    const p = profiles(previous.grey, canvasW, qx, qy, qw, qh);
    const c = profiles(current.grey, canvasW, qx, qy, qw, qh);
    const dx = bestLag(p.cols, c.cols, Math.min(24, qw >> 2));
    const dy = bestLag(p.rows, c.rows, Math.min(16, qh >> 3));
    if (dx.valid && dy.valid) shifts.push({ x: dx.shift * sx, y: dy.shift * sy });
  }

  if (shifts.length < 2) return { t, divergencePx: 0, meanShiftPx: 0, yawDeltaDeg: 0, valid: false };

  let divergence = 0;
  for (let i = 0; i < shifts.length; i += 1) {
    for (let j = i + 1; j < shifts.length; j += 1) {
      divergence = Math.max(divergence, Math.hypot(shifts[i].x - shifts[j].x, shifts[i].y - shifts[j].y));
    }
  }

  let meanX = 0;
  for (const s of shifts) meanX += s.x;
  meanX /= shifts.length;

  // Content sliding LEFT means the camera turned RIGHT, and yaw increases clockwise.
  return { t, divergencePx: divergence, meanShiftPx: meanX, yawDeltaDeg: -meanX / pxPerDegAt640, valid: true };
}

/** Focal-derived px per degree at the engine's 640-px reference width. */
export function pxPerDegAt640(lens: LensGeometry): number {
  return (320 / Math.tan((lens.hfovDeg / 2) * DEG)) * DEG;
}

/**
 * RGBA -> the Float32 grey plane `quadrantFlow` reads, written into a caller-owned buffer.
 *
 * The destination is passed in rather than returned because the caller ping-pongs two pooled
 * planes: `quadrantFlow` compares this frame against the previous one, so exactly two are
 * ever live, and allocating one per sample was 192 x 256 x 4 bytes = 196 KB five times a
 * second — roughly a megabyte a second of pure garbage handed to a collector on a phone that
 * is simultaneously encoding 1080p. Lifted here so the worker and the synchronous fallback
 * cannot drift apart on the coefficients.
 */
export function toGreyPlane(rgba: Uint8ClampedArray, out: Float32Array): void {
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
    out[p] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
  }
}

/**
 * The two grey planes `quadrantFlow` needs, grown but never shrunk.
 *
 * Keyed on the cell count so a lens or orientation change reallocates once rather than
 * reusing a plane of the wrong shape, and clears `previous` when it does — a plane of the
 * wrong geometry correlated against one of the right geometry produces a confident,
 * meaningless divergence, which is exactly the kind of number that stops a capture for no
 * reason.
 */
export interface FlowPool {
  cells: number;
  planes: [Float32Array, Float32Array];
  next: 0 | 1;
  previous: FlowFrame | null;
}

export function createFlowPool(): FlowPool {
  return { cells: 0, planes: [new Float32Array(0), new Float32Array(0)], next: 0, previous: null };
}

/**
 * One flow measurement: fold RGBA into the next pooled plane, correlate against the last one.
 * Returns null on the first frame of a run or after a geometry change, when there is no
 * previous plane to compare against.
 */
export function flowStep(
  pool: FlowPool,
  rgba: Uint8ClampedArray,
  canvasW: number,
  canvasH: number,
  frameW: number,
  frameH: number,
  pxPerDeg: number,
  t: Millis,
): FlowSample | null {
  const cells = canvasW * canvasH;
  if (pool.cells !== cells) {
    pool.cells = cells;
    pool.planes = [new Float32Array(cells), new Float32Array(cells)];
    pool.next = 0;
    pool.previous = null;
  }
  const grey = pool.planes[pool.next];
  pool.next = pool.next === 0 ? 1 : 0;
  toGreyPlane(rgba, grey);

  const frame: FlowFrame = { grey, canvasW, canvasH };
  const previous = pool.previous;
  pool.previous = frame;
  return previous ? quadrantFlow(previous, frame, frameW, frameH, pxPerDeg, t) : null;
}
