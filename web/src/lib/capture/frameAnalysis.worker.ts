/**
 * FRAME ANALYSIS WORKER — where `getImageData` is allowed to block.
 * =====================================================================================
 *
 * The main thread never calls `getImageData` again. It calls `createImageBitmap`, which is
 * asynchronous and hands back a GPU-side handle, transfers that handle here, and this thread
 * pays the GPU->CPU readback. The readback still costs exactly what it always cost; it now
 * blocks a thread that owns nothing — no compositor, no `requestVideoFrameCallback`, no
 * `deviceorientation`, no MediaRecorder pump — instead of stalling all four of them 15 times
 * a second.
 *
 * ── WHAT THIS FILE MAY IMPORT ─────────────────────────────────────────────────────────
 * `./frameKernels` and nothing else at runtime. `./types` appears only through
 * `import type`, which is erased, so the worker chunk stays at the arithmetic and does not
 * pull the ~100 KB capture contract across the thread boundary for two integers. Everything
 * the kernels need that lives on the main thread — canvas dimensions, decoded frame size,
 * px/deg — rides in the message.
 *
 * ── THE TWO CANVASES ARE SEPARATE ON PURPOSE ──────────────────────────────────────────
 * Quality runs at 10 Hz on a ~64-px-short-edge bitmap; flow runs at 5 Hz on a fixed
 * 192 x 256. Sharing one canvas would resize it 15 times a second, and every resize drops
 * the backing store and reallocates it. Two canvases, each resized only when the incoming
 * bitmap's geometry actually changes.
 *
 * ── FAILURE IS REPORTED, NOT SWALLOWED ────────────────────────────────────────────────
 * If `OffscreenCanvas` cannot give a 2D context, this posts `unavailable` once and stops.
 * `frameAnalysis.ts` then tears the worker down and reverts to the synchronous path, which
 * is slow but correct — the alternative is a capture that silently records no brightness,
 * no clipping and no walking signal at all, and the detectors reading `null` forever looks
 * exactly like a calm, well-lit room.
 */

import { analyseFrameLuma, createFlowPool, flowStep } from "./frameKernels";
import type { FlowSample, Millis } from "./types";

// -------------------------------------------------------------------------------------
// Wire format
// -------------------------------------------------------------------------------------

/** 10 Hz luma/clipping. The bitmap arrives pre-resized by `createImageBitmap`. */
export interface QualityRequest {
  kind: "quality";
  /** Echoed back so a reply that outlived a `reset` can be recognised and dropped. */
  seq: number;
  bitmap: ImageBitmap;
  /**
   * The geometry the analysis must run at — REQUESTED, not whatever the bitmap came back as.
   * `ImageBitmapOptions.resizeWidth/resizeHeight` are a request, and an engine that ignores
   * them hands back a full 1080x1920 handle; sizing the surface from `bitmap.width` would
   * then turn a 29 KB readback into an 8 MB one on exactly the phones least able to afford
   * it. Drawing into a canvas of the requested size makes an ignored hint cost one extra
   * scale on this thread instead.
   */
  width: number;
  height: number;
}

/** 5 Hz quadrant flow. Everything the normalisation needs travels with the frame. */
export interface FlowRequest {
  kind: "flow";
  seq: number;
  bitmap: ImageBitmap;
  /** As `QualityRequest.width/height`: the requested geometry, never the bitmap's own. */
  width: number;
  height: number;
  /** DECODED frame dimensions, not the bitmap's — the 640-px normalisation is keyed on them. */
  frameW: number;
  frameH: number;
  pxPerDeg: number;
  t: Millis;
}

/**
 * Drop the previous flow plane.
 *
 * Sent when analysis is paused or the stream changes. Without it the first frame after a
 * pause is correlated against whatever the camera was looking at before it, which produces
 * a large, confident and entirely fictional divergence — and divergence over 7 px for 800 ms
 * is a hard stop shown to the user.
 */
export interface ResetRequest {
  kind: "reset";
}

export type AnalysisRequest = QualityRequest | FlowRequest | ResetRequest;

export interface QualityResponse {
  kind: "quality";
  seq: number;
  meanLuma: number;
  clippedFraction: number;
}

export interface FlowResponse {
  kind: "flow";
  seq: number;
  /** Null on the first frame of a run and after a reset: nothing to correlate against yet. */
  sample: FlowSample | null;
}

/** Posted at most once. The host treats it as "this worker will never produce a number". */
export interface UnavailableResponse {
  kind: "unavailable";
  reason: string;
}

export type AnalysisResponse = QualityResponse | FlowResponse | UnavailableResponse;

// -------------------------------------------------------------------------------------
// Scope
// -------------------------------------------------------------------------------------

/**
 * `self` types as `Window` under `lib: ["dom"]`, whose `postMessage` demands a targetOrigin.
 * The project cannot add `lib: ["webworker"]` — it would delete `HTMLVideoElement` and
 * friends from every other file — so the worker scope is described locally and cast once.
 */
interface WorkerScope {
  postMessage(message: AnalysisResponse): void;
  addEventListener(type: "message", listener: (event: MessageEvent<AnalysisRequest>) => void): void;
}

const scope = self as unknown as WorkerScope;

// -------------------------------------------------------------------------------------
// Canvases
// -------------------------------------------------------------------------------------

interface Surface {
  canvas: OffscreenCanvas;
  ctx: OffscreenCanvasRenderingContext2D;
}

let qualitySurface: Surface | null = null;
let flowSurface: Surface | null = null;
let reportedUnavailable = false;

const pool = createFlowPool();

/**
 * `willReadFrequently` is the whole reason this is cheap here: it asks the implementation for
 * a CPU-backed surface, so `getImageData` is a memcpy out of system memory rather than a
 * round trip that has to synchronise with the GPU. On the main thread that hint helped and
 * the stall remained, because the DRAW still had to reach a surface the compositor shares.
 */
function surfaceFor(existing: Surface | null, width: number, height: number): Surface | null {
  if (existing && existing.canvas.width === width && existing.canvas.height === height) return existing;

  const canvas = existing?.canvas ?? new OffscreenCanvas(width, height);
  canvas.width = width;
  canvas.height = height;
  const ctx = existing?.ctx ?? canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    if (!reportedUnavailable) {
      reportedUnavailable = true;
      scope.postMessage({ kind: "unavailable", reason: "no-2d-context" });
    }
    return null;
  }
  return { canvas, ctx };
}

/** Pixels out of a bitmap, and the bitmap closed — an unclosed ImageBitmap holds GPU memory. */
function readPixels(surface: Surface, bitmap: ImageBitmap): Uint8ClampedArray {
  const { canvas, ctx } = surface;
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return ctx.getImageData(0, 0, canvas.width, canvas.height).data;
}

// -------------------------------------------------------------------------------------
// Pump
// -------------------------------------------------------------------------------------

scope.addEventListener("message", (event) => {
  const request = event.data;

  if (request.kind === "reset") {
    pool.previous = null;
    return;
  }

  if (request.kind === "quality") {
    const surface = surfaceFor(qualitySurface, request.width, request.height);
    if (!surface) {
      request.bitmap.close();
      return;
    }
    qualitySurface = surface;
    const stats = analyseFrameLuma(readPixels(surface, request.bitmap));
    scope.postMessage({ kind: "quality", seq: request.seq, meanLuma: stats.meanLuma, clippedFraction: stats.clippedFraction });
    return;
  }

  const surface = surfaceFor(flowSurface, request.width, request.height);
  if (!surface) {
    request.bitmap.close();
    return;
  }
  flowSurface = surface;
  const width = surface.canvas.width;
  const height = surface.canvas.height;
  const sample = flowStep(
    pool,
    readPixels(surface, request.bitmap),
    width,
    height,
    request.frameW,
    request.frameH,
    request.pxPerDeg,
    request.t,
  );
  scope.postMessage({ kind: "flow", seq: request.seq, sample });
});
