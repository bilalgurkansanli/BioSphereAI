/**
 * HOW MUCH OF THE PER-FRAME ANALYSIS IS ACTUALLY PIXELS?
 * =====================================================================================
 *
 * The guided capture ran `drawImage` + `getImageData` + these kernels 15 times a second on
 * the main thread, and twice the arithmetic was measured, found trivial, and cleared of
 * causing the lag. This script is the third measurement, kept in the tree so nobody has to
 * take the second one on trust: it runs the SHIPPED kernels over buffers of the SHIPPED
 * geometry, with no canvas anywhere, and prints nanoseconds per call.
 *
 * What it can measure:   the arithmetic — L* accumulation, greyscale folding, the four
 *                        quadrant correlations.
 * What it CANNOT measure: `getImageData` itself. There is no GPU in Node, so the readback —
 *                        the synchronous pipeline barrier that makes the browser wait for a
 *                        busy GPU to retire a decode, a composite and an encoder surface
 *                        before it will hand back 29 KB — has no analogue here.
 *
 * That gap is the point. Subtract this script's numbers from the frame budget on the phone
 * and whatever is left is the readback, which is what `frameAnalysis.ts` moves off-thread.
 *
 *   node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON scripts/bench-frame-kernels.mjs
 *
 * (the flag only silences Node's note that `web/package.json` has no `"type": "module"`;
 * adding one would change how Next.js reads the whole package, so the flag is the cheaper
 * side of that trade.)
 *
 * Plain `.mjs` on purpose: Node 22.18+/24 strips the types out of the imported `.ts` module
 * with no build step, and `.mjs` is outside `tsconfig.json`'s `include`, so a benchmark
 * script cannot break `npx tsc --noEmit`.
 */

import { analyseFrameLuma, createFlowPool, flowStep, quadrantFlow, toGreyPlane } from "../src/lib/capture/frameKernels.ts";

// The two geometries the capture actually uses. Quality: 64 on the short edge with the
// aspect kept, which is 64x114 for a portrait 1080x1920 preview. Flow: 2x2 quadrants of the
// contract's 96x128.
const QUALITY_W = 64;
const QUALITY_H = Math.round((64 * 1920) / 1080);
const FLOW_W = 192;
const FLOW_H = 256;

/** The 10 Hz and 5 Hz clocks in useCamera. Used to turn ns/call into ms per second live. */
const QUALITY_HZ = 10;
const FLOW_HZ = 5;

/** Hash noise, so the correlator has real structure to lock onto rather than a flat wall. */
function noise(x, y) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return (s - Math.floor(s)) * 200 + 20;
}

function rgba(width, height, shiftX) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const v = noise(x - shiftX, y);
      const i = (y * width + x) * 4;
      data[i] = v;
      data[i + 1] = v * 0.98;
      data[i + 2] = v * 1.02;
      data[i + 3] = 255;
    }
  }
  return data;
}

function greyPlane(width, height, shiftX) {
  const plane = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) plane[y * width + x] = noise(x - shiftX, y);
  return { grey: plane, canvasW: width, canvasH: height };
}

const rows = [];

/**
 * Median of per-batch means rather than one long average: a single JIT deopt or a GC pause
 * inside one batch would otherwise be smeared across every call and reported as the cost of
 * the arithmetic. `hz` is the rate the capture calls this at, or 0 for a component that is
 * only part of another row.
 */
function bench(name, hz, iterations, fn) {
  for (let i = 0; i < Math.min(iterations, 500); i += 1) fn(i);

  const batches = 9;
  const means = [];
  for (let b = 0; b < batches; b += 1) {
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < iterations; i += 1) fn(i);
    means.push(Number(process.hrtime.bigint() - t0) / iterations);
  }
  means.sort((a, b) => a - b);
  const ns = means[batches >> 1];
  rows.push({ name, ns, best: means[0], worst: means[batches - 1], hz });
}

// -------------------------------------------------------------------------------------

const qualityBuffer = rgba(QUALITY_W, QUALITY_H, 0);
bench(`analyseFrameLuma  ${QUALITY_W}x${QUALITY_H} RGBA`, QUALITY_HZ, 20000, () => {
  analyseFrameLuma(qualityBuffer);
});

const flowBuffer = rgba(FLOW_W, FLOW_H, 0);
const scratch = new Float32Array(FLOW_W * FLOW_H);
bench(`toGreyPlane       ${FLOW_W}x${FLOW_H} RGBA`, 0, 20000, () => {
  toGreyPlane(flowBuffer, scratch);
});

const previous = greyPlane(FLOW_W, FLOW_H, 0);
const current = greyPlane(FLOW_W, FLOW_H, 6);
bench(`quadrantFlow      ${FLOW_W}x${FLOW_H} grey`, 0, 4000, () => {
  quadrantFlow(previous, current, 1080, 1920, 14.18, 0);
});

// The whole 5 Hz body as the worker runs it: fold RGBA into a pooled plane, then correlate.
// Two alternating buffers, because correlating a frame against itself is not the work.
const pool = createFlowPool();
const flowBuffers = [rgba(FLOW_W, FLOW_H, 0), rgba(FLOW_W, FLOW_H, 6)];
bench(`flowStep (fold + correlate)`, FLOW_HZ, 4000, (i) => {
  flowStep(pool, flowBuffers[i & 1], FLOW_W, FLOW_H, 1080, 1920, 14.18, i * 200);
});

// -------------------------------------------------------------------------------------

const pad = (s, n) => String(s).padEnd(n);
const num = (n, d = 0) => n.toFixed(d).padStart(9);

console.log(`node ${process.version}  ${process.platform}/${process.arch}\n`);
console.log(`${pad("kernel", 34)}${pad("ns/call", 11)}${pad("best", 11)}${pad("worst", 11)}ms per second of capture`);
console.log("-".repeat(34 + 11 + 11 + 11 + 24));

let liveMs = 0;
for (const row of rows) {
  const perSecond = row.hz > 0 ? (row.ns * row.hz) / 1e6 : 0;
  liveMs += perSecond;
  console.log(
    `${pad(row.name, 34)}${num(row.ns)}  ${num(row.best)}  ${num(row.worst)}  ${row.hz > 0 ? `${perSecond.toFixed(3)} ms  (@ ${row.hz} Hz)` : "—  (inside flowStep)"}`,
  );
}

console.log("-".repeat(34 + 11 + 11 + 11 + 24));
console.log(`\nWhole 15 Hz analysis budget, arithmetic only: ${liveMs.toFixed(3)} ms per second of capture.`);
console.log(`That is ${((liveMs / 1000) * 100).toFixed(3)}% of one core, and ${(liveMs / 30).toFixed(4)} ms of an average 33.3 ms frame.`);
console.log("Everything the phone spends beyond this is the drawImage/getImageData readback, which is what moves to the worker.");
