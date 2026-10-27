"use client";

/**
 * FRAME ANALYSIS HOST — the readback the capture loop stopped paying for.
 * =====================================================================================
 *
 * ── THE FAILURE ───────────────────────────────────────────────────────────────────────
 * `useCamera` ran both image kernels inside `requestVideoFrameCallback`:
 *
 *     analyseQuality  10 Hz   drawImage(video -> ~64x114)   then getImageData
 *     analyseFlow      5 Hz   drawImage(video -> 192x256)   then getImageData
 *
 * Fifteen `getImageData` calls a second, on the main thread, on a mid-range Android that is
 * at that moment decoding 1080x1920 @ 30 fps, encoding H.264 at 12 Mbps in hardware, and
 * compositing a full-screen SVG overlay over the video.
 *
 * The pixel count was measured and dismissed twice, correctly. `scripts/bench-frame-kernels.mjs`
 * makes that third: the same kernels over the same buffers, no canvas anywhere, cost 0.15 ms
 * per quality call and 0.34 ms per flow call — 3.4 ms per SECOND of capture for both clocks,
 * 0.11 ms of an average 33 ms frame. Even ten times slower on the phone that is a rounding
 * error. What was never measured is the part that is not arithmetic.
 * `getImageData` is a SYNCHRONOUS GPU->CPU readback: it cannot return until everything the
 * GPU has queued ahead of it has retired and the pixels have been copied back into system
 * memory. Nothing in that sentence is proportional to the size of the rectangle asked for.
 * The video frame being decoded, the overlay being composited and the encoder's own surface
 * are all queued ahead of it, so each call is a pipeline barrier: fifteen a second, on the
 * one thread that also has to service the video-frame callback, `deviceorientation` at
 * 60 Hz, MediaRecorder's data pump and every paint.
 *
 * ── THE FIX ───────────────────────────────────────────────────────────────────────────
 * `createImageBitmap(video, { resizeWidth, resizeHeight })` does the same downscale, returns
 * a PROMISE, and never blocks the caller. The resulting handle is transferred (not copied)
 * to `frameAnalysis.worker.ts`, which owns an OffscreenCanvas and pays the readback on a
 * thread that owns no compositor, no encoder pump and no event loop anyone is waiting on.
 * The main thread's per-frame cost becomes one `createImageBitmap` call and one
 * `postMessage` of a transferable.
 *
 * ── DROP, NEVER QUEUE ─────────────────────────────────────────────────────────────────
 * At most one bitmap is in flight per analysis kind. If the previous one has not come back
 * — because the worker is behind, or because `createImageBitmap` is still resolving — the
 * due frame is DROPPED. Queueing would convert a slow worker into an unbounded backlog of
 * GPU handles and would deliver measurements describing a room the phone stopped pointing at
 * seconds ago; the detectors time-stamp everything and a late `smear` or `divergence` reading
 * is not merely useless, it stops a capture that is going fine. Skipping a sample only lowers
 * the effective rate, which the detectors already tolerate because a phone can drop frames.
 *
 * ── DEGRADING ─────────────────────────────────────────────────────────────────────────
 * `Worker`, `OffscreenCanvas` and `createImageBitmap` are all feature-detected, and any of
 * four things demotes the analyser to the original synchronous main-thread path for the rest
 * of the session: a missing global, a constructor that throws, an `error`/`messageerror` from
 * the worker, or no reply at all to the first submission within `WORKER_PROOF_MS`. The last
 * one exists because a worker whose module URL fails to resolve under a misconfigured asset
 * prefix goes quiet rather than throwing, and a silent analyser reads to the detectors as a
 * calm, well-lit, perfectly still room.
 */

import {
  analyseFrameLuma,
  createFlowPool,
  flowStep,
  type FlowPool,
} from "./frameKernels";
import type { AnalysisRequest, AnalysisResponse } from "./frameAnalysis.worker";
import {
  RECORDING,
  THRESHOLDS,
  estimateExposureSeconds,
  predictedSmearPx,
  type FlowSample,
  type FrameQualitySample,
  type Millis,
} from "./types";

/**
 * How long the worker has to answer its first submission before it is written off.
 *
 * IT MUST BE GENEROUS, AND IT IS FREE TO BE, because the synchronous path keeps running
 * until the worker has proved itself once (see `submit`).
 *
 * It was one second, on the reasoning that ten missed samples is cheap. That was wrong about
 * where the time goes: the worker chunks are NOT in the build manifest and are therefore not
 * preloaded with the page — `new Worker()` is the first thing that asks for them. The
 * bootstrap has to land before it can request the worker chunk and the Turbopack runtime,
 * which is at least two sequential round trips to the origin, and the capture is opened from
 * a tunnel whose hostname changes on every run, so that cache is cold every single time.
 * Add worker start-up, `OffscreenCanvas` creation and the first readback and a second is not
 * a deadline, it is a coin toss.
 *
 * Losing that toss is expensive and silent: `demote` terminates the worker for the rest of
 * the stream's life, so the phone spends the whole capture on exactly the synchronous
 * GPU->CPU readback the offload exists to remove — and reports nothing.
 */
const WORKER_PROOF_MS = 8000;

/** Short edge of the quality bitmap, matching the canvas `analyseQuality` used. */
const QUALITY_SHORT_EDGE_PX = 64;

/**
 * `low` and not `medium`/`high`: `drawImage` into a smaller canvas is itself a bilinear
 * downsample, so `low` is the setting that reproduces the numbers the thresholds were
 * calibrated against. A better filter would quietly raise `meanLuma` in textured rooms and
 * lower `clippedFraction`, moving the block thresholds without anyone editing them.
 */
const RESIZE_QUALITY: ResizeQuality = "low";

// -------------------------------------------------------------------------------------
// Injection surfaces
// -------------------------------------------------------------------------------------

/** Whatever `drawImage` and `createImageBitmap` both accept; the hook passes its video. */
export type FrameSource = CanvasImageSource;

/** The slice of `Worker` this module uses. Replaced wholesale in tests. */
export interface AnalysisWorkerLike {
  postMessage(message: AnalysisRequest, transfer: Transferable[]): void;
  addEventListener(type: "message", listener: (event: MessageEvent<AnalysisResponse>) => void): void;
  addEventListener(type: "error" | "messageerror", listener: () => void): void;
  terminate(): void;
}

export interface AnalysisContextLike {
  drawImage(source: FrameSource, dx: number, dy: number, dw: number, dh: number): void;
  getImageData(sx: number, sy: number, sw: number, sh: number): { data: Uint8ClampedArray };
}

export interface AnalysisCanvasLike {
  width: number;
  height: number;
  getContext(contextId: "2d", options?: { willReadFrequently?: boolean }): AnalysisContextLike | null;
}

/** "worker" until something proves the worker unusable; never recovers within a session. */
export type FrameAnalyserBackend = "worker" | "main-thread";

export interface FrameAnalyserOptions {
  /**
   * Builds the worker. Omitted, the real one is constructed from the module URL — which is
   * the shape Turbopack rewrites into a bundled worker entrypoint, and the only shape it
   * rewrites: the `new URL(..., import.meta.url)` argument must be a literal, so this call
   * cannot be factored into a helper that takes the path. Explicit `null` means "this
   * platform has none", which is how a test exercises the fallback without having to unset
   * globals it does not own.
   */
  createWorker?: (() => AnalysisWorkerLike) | null;
  /** Defaults to the global `createImageBitmap`; explicit `null` means the platform has none. */
  createBitmap?: ((source: FrameSource, options: ImageBitmapOptions) => Promise<ImageBitmap>) | null;
  /** Backing store for the synchronous fallback. Defaults to a pooled `<canvas>`. */
  createCanvas?: () => AnalysisCanvasLike | null;
  /** Fired once, with the reason, if the worker path is abandoned. Telemetry hangs off this. */
  onDemote?: (reason: string) => void;
}

/** Everything the loop knows this frame that the kernels cannot work out for themselves. */
export interface FrameSubmitOptions {
  t: Millis;
  /** The 10 Hz quality clock is due. */
  quality: boolean;
  /** The 5 Hz flow clock is due. */
  flow: boolean;
  /** `VideoFrameCallbackMetadata.mediaTime`; keys the sample to the recorded file. */
  mediaTime: number;
  /** Decoded frame dimensions — `video.videoWidth/videoHeight`, never `getSettings()`. */
  frameW: number;
  frameH: number;
  /** Frames actually delivered over the fps window, as counted by the caller. */
  measuredFps: number;
  /**
   * `track.getSettings().exposureTime / 10000` on Android, 0 where the platform does not
   * report it. Sampled at SUBMIT time, not at reply time, so the composed sample describes
   * the same instant the pixels do.
   */
  reportedExposureS: number;
  /** |rotationRate| now, for the smear prediction. */
  omegaDegPerS: number;
  /** `pxPerDegAt640(lens)`, hoisted out of the loop by the caller. */
  pxPerDeg: number;
}

export interface FrameAnalyser {
  submit(source: FrameSource, options: FrameSubmitOptions): void;
  readQuality(): FrameQualitySample | null;
  readFlow(): FlowSample | null;
  /**
   * Pausing also resets the worker's flow history. The first frame after a pause has nothing
   * legitimate to correlate against, and correlating it against the pre-pause frame produces
   * a large, confident, fictional divergence — which is a hard stop shown to the user.
   */
  setEnabled(enabled: boolean): void;
  dispose(): void;
  readonly backend: FrameAnalyserBackend;
  /** Due frames skipped because the previous submission of that kind was still in flight. */
  readonly droppedFrames: number;
}

// -------------------------------------------------------------------------------------
// Sizing
// -------------------------------------------------------------------------------------

/**
 * The quality bitmap's geometry: 64 on the short edge, aspect preserved. Copied from the
 * `ensureCanvas` call `analyseQuality` made, so a portrait 1080x1920 preview still lands on
 * 64x114 and `meanLuma` keeps the scale the 163/144 field measurements were taken on.
 */
export function qualityBitmapSize(frameW: number, frameH: number): { width: number; height: number } {
  return frameW >= frameH
    ? { width: Math.round((QUALITY_SHORT_EDGE_PX * frameW) / frameH), height: QUALITY_SHORT_EDGE_PX }
    : { width: QUALITY_SHORT_EDGE_PX, height: Math.round((QUALITY_SHORT_EDGE_PX * frameH) / frameW) };
}

/** 2x2 quadrants of exactly the contract's 96x128, so the px thresholds keep their meaning. */
const FLOW_W = THRESHOLDS.flowDivergence.quadrantWidthPx * 2;
const FLOW_H = THRESHOLDS.flowDivergence.quadrantHeightPx * 2;

// -------------------------------------------------------------------------------------
// Host
// -------------------------------------------------------------------------------------

export function createFrameAnalyser(options: FrameAnalyserOptions = {}): FrameAnalyser {
  const createBitmap =
    options.createBitmap !== undefined
      ? options.createBitmap
      : typeof createImageBitmap === "function"
        ? (source: FrameSource, o: ImageBitmapOptions) => createImageBitmap(source, o)
        : null;

  let backend: FrameAnalyserBackend = "main-thread";
  let worker: AnalysisWorkerLike | null = null;
  let droppedFrames = 0;
  let enabled = true;
  let disposed = false;

  let quality: FrameQualitySample | null = null;
  let flow: FlowSample | null = null;

  // In-flight is one flag per KIND, held from the moment `createImageBitmap` is called until
  // the worker's reply lands (or the attempt fails). The two kinds are independent: a slow
  // flow correlation must not also silence the brightness meter, which is what a single
  // shared gate would do at the 2:1 rate ratio between them.
  let qualityInFlight = false;
  let flowInFlight = false;
  let seq = 0;

  /** Sampled at submit time; the worker returns pixels, the main thread owns the rest. */
  let pendingQuality: { t: Millis; mediaTime: number; measuredFps: number; reportedExposureS: number; omegaDegPerS: number } | null = null;

  /** Null until the first submission; the worker's grace period is measured from it. */
  let firstSubmitAt: Millis | null = null;
  let sawReply = false;

  // Synchronous fallback state. Allocated only if it is ever used.
  const flowPool: FlowPool = createFlowPool();
  let qualityCanvas: AnalysisCanvasLike | null = null;
  let flowCanvas: AnalysisCanvasLike | null = null;

  const createCanvas =
    options.createCanvas ??
    (typeof document !== "undefined" ? () => document.createElement("canvas") as unknown as AnalysisCanvasLike : () => null);

  const composeQuality = (
    context: { t: Millis; mediaTime: number; measuredFps: number; reportedExposureS: number; omegaDegPerS: number },
    meanLuma: number,
    clippedFraction: number,
  ): FrameQualitySample => {
    const reported = context.reportedExposureS;
    const exposureS = reported > 0 ? reported : estimateExposureSeconds(context.measuredFps || RECORDING.targetFps, meanLuma);
    return {
      t: context.t,
      mediaTime: context.mediaTime,
      meanLuma,
      clippedFraction,
      exposureS,
      exposureSource: reported > 0 ? "reported" : "estimated",
      measuredFps: context.measuredFps,
      smearPx: predictedSmearPx(context.omegaDegPerS, exposureS),
    };
  };

  const demote = (reason: string) => {
    if (backend === "main-thread") return;
    backend = "main-thread";
    // The in-flight gates are released along with the worker: nothing is coming back to
    // clear them, and a stuck gate is an analyser that never measures anything again.
    qualityInFlight = false;
    flowInFlight = false;
    pendingQuality = null;
    flowPool.previous = null;
    try {
      worker?.terminate();
    } catch {
      // A worker that already died throws on terminate in some builds. Nothing to recover.
    }
    worker = null;
    options.onDemote?.(reason);
  };

  const onMessage = (event: MessageEvent<AnalysisResponse>) => {
    if (disposed) return;
    sawReply = true;
    const message = event.data;

    if (message.kind === "unavailable") {
      demote(message.reason);
      return;
    }
    if (message.kind === "quality") {
      qualityInFlight = false;
      const context = pendingQuality;
      pendingQuality = null;
      // A reply that outlived `setEnabled(false)` describes a frame from before the pause.
      if (context && enabled) quality = composeQuality(context, message.meanLuma, message.clippedFraction);
      return;
    }
    flowInFlight = false;
    if (message.sample && enabled) flow = message.sample;
  };

  // Feature detection, up front and complete. `OffscreenCanvas` is checked on the window
  // because there is no way to ask the worker before building it, and every engine that
  // ships one ships the other.
  const workerFactory =
    options.createWorker !== undefined
      ? options.createWorker
      : typeof Worker !== "undefined" && typeof OffscreenCanvas !== "undefined"
        ? () => new Worker(new URL("./frameAnalysis.worker.ts", import.meta.url), { type: "module" }) as unknown as AnalysisWorkerLike
        : null;

  if (workerFactory && createBitmap) {
    try {
      worker = workerFactory();
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", () => demote("worker-error"));
      worker.addEventListener("messageerror", () => demote("worker-messageerror"));
      backend = "worker";
    } catch (cause) {
      worker = null;
      backend = "main-thread";
      options.onDemote?.(cause instanceof Error ? `worker-construct:${cause.name}` : "worker-construct");
    }
  } else {
    // Both are named when both are absent. A single reason would have telemetry claiming an
    // engine lacks workers when what it actually lacks is `createImageBitmap`, and the two
    // point at different browsers.
    const missing: string[] = [];
    if (!workerFactory) missing.push("no-worker");
    if (!createBitmap) missing.push("no-createImageBitmap");
    options.onDemote?.(missing.join("+"));
  }

  // -----------------------------------------------------------------------------------
  // Synchronous fallback — the exact path `useCamera` used to run inline
  // -----------------------------------------------------------------------------------

  const contextFor = (canvas: AnalysisCanvasLike | null, width: number, height: number): { canvas: AnalysisCanvasLike; ctx: AnalysisContextLike } | null => {
    const target = canvas ?? createCanvas();
    if (!target) return null;
    if (target.width !== width) target.width = width;
    if (target.height !== height) target.height = height;
    const ctx = target.getContext("2d", { willReadFrequently: true });
    return ctx ? { canvas: target, ctx } : null;
  };

  const submitSync = (source: FrameSource, o: FrameSubmitOptions) => {
    if (o.quality) {
      const size = qualityBitmapSize(o.frameW, o.frameH);
      const surface = contextFor(qualityCanvas, size.width, size.height);
      if (surface) {
        qualityCanvas = surface.canvas;
        surface.ctx.drawImage(source, 0, 0, size.width, size.height);
        const stats = analyseFrameLuma(surface.ctx.getImageData(0, 0, size.width, size.height).data);
        quality = composeQuality(o, stats.meanLuma, stats.clippedFraction);
      }
    }
    if (o.flow) {
      const surface = contextFor(flowCanvas, FLOW_W, FLOW_H);
      if (surface) {
        flowCanvas = surface.canvas;
        surface.ctx.drawImage(source, 0, 0, FLOW_W, FLOW_H);
        const sample = flowStep(
          flowPool,
          surface.ctx.getImageData(0, 0, FLOW_W, FLOW_H).data,
          FLOW_W,
          FLOW_H,
          o.frameW,
          o.frameH,
          o.pxPerDeg,
          o.t,
        );
        if (sample) flow = sample;
      }
    }
  };

  // -----------------------------------------------------------------------------------
  // Worker path
  // -----------------------------------------------------------------------------------

  const send = (request: AnalysisRequest, bitmap: ImageBitmap) => {
    const active = worker;
    // `!enabled` matters as much as `!disposed`: a bitmap that resolves AFTER the pause would
    // otherwise reach the worker behind the `reset` and re-arm `pool.previous` with a
    // pre-pause frame — which is the fictional divergence the reset exists to prevent,
    // reintroduced by the race the reset was meant to close.
    if (!active || disposed || !enabled) {
      bitmap.close();
      return false;
    }
    try {
      active.postMessage(request, [bitmap]);
      return true;
    } catch (cause) {
      bitmap.close();
      demote(cause instanceof Error ? `postMessage:${cause.name}` : "postMessage");
      return false;
    }
  };

  const submitWorker = (source: FrameSource, o: FrameSubmitOptions) => {
    if (!createBitmap) return;

    if (o.quality) {
      if (qualityInFlight) {
        droppedFrames += 1;
      } else {
        const size = qualityBitmapSize(o.frameW, o.frameH);
        qualityInFlight = true;
        pendingQuality = { t: o.t, mediaTime: o.mediaTime, measuredFps: o.measuredFps, reportedExposureS: o.reportedExposureS, omegaDegPerS: o.omegaDegPerS };
        const id = (seq += 1);
        createBitmap(source, { resizeWidth: size.width, resizeHeight: size.height, resizeQuality: RESIZE_QUALITY }).then(
          (bitmap) => {
            if (!send({ kind: "quality", seq: id, bitmap, width: size.width, height: size.height }, bitmap)) {
              qualityInFlight = false;
              pendingQuality = null;
            }
          },
          () => {
            // `createImageBitmap` rejects on a video with no current frame — during a lens
            // switch, or between `play()` and the first decode. Release the gate and wait for
            // the next tick rather than treating it as a worker failure.
            qualityInFlight = false;
            pendingQuality = null;
          },
        );
      }
    }

    if (o.flow) {
      if (flowInFlight) {
        droppedFrames += 1;
      } else {
        flowInFlight = true;
        const id = (seq += 1);
        createBitmap(source, { resizeWidth: FLOW_W, resizeHeight: FLOW_H, resizeQuality: RESIZE_QUALITY }).then(
          (bitmap) => {
            const request: AnalysisRequest = {
              kind: "flow",
              seq: id,
              bitmap,
              width: FLOW_W,
              height: FLOW_H,
              frameW: o.frameW,
              frameH: o.frameH,
              pxPerDeg: o.pxPerDeg,
              t: o.t,
            };
            if (!send(request, bitmap)) flowInFlight = false;
          },
          () => {
            flowInFlight = false;
          },
        );
      }
    }
  };

  return {
    submit(source: FrameSource, o: FrameSubmitOptions) {
      if (disposed || !enabled) return;
      if (!o.quality && !o.flow) return;
      if (o.frameW <= 0 || o.frameH <= 0) return;

      if (backend === "worker") {
        // A worker that never answers is indistinguishable from one that is merely slow until
        // a deadline says otherwise, and the loop's own clock is already in hand.
        if (firstSubmitAt === null) firstSubmitAt = o.t;
        else if (!sawReply && o.t - firstSubmitAt > WORKER_PROOF_MS) demote("worker-silent");
      }

      if (backend === "worker") {
        submitWorker(source, o);
        // BOTH, until the worker has answered once. This is the pre-offload cost, paid only
        // during the proof window, and it is what lets that window be long enough for a cold
        // chunk fetch: the detectors never go blind waiting for a worker that is merely slow,
        // so nothing forces the deadline to be short. `sawReply` flips on the first reply and
        // the doubled work stops there.
        if (!sawReply) submitSync(source, o);
      } else {
        submitSync(source, o);
      }
    },

    readQuality: () => quality,
    readFlow: () => flow,

    setEnabled(next: boolean) {
      if (enabled === next) return;
      enabled = next;
      flowPool.previous = null;
      if (!next) {
        // The gates are released here rather than waiting for replies that are about to be
        // discarded: resuming must not have to wait out a stale in-flight frame.
        qualityInFlight = false;
        flowInFlight = false;
        pendingQuality = null;
        try {
          worker?.postMessage({ kind: "reset" }, []);
        } catch {
          // A dead worker is demoted by its own error event; nothing to add here.
        }
      }
    },

    dispose() {
      disposed = true;
      try {
        worker?.terminate();
      } catch {
        // See `demote`.
      }
      worker = null;
      quality = null;
      flow = null;
      flowPool.previous = null;
      qualityCanvas = null;
      flowCanvas = null;
    },

    get backend() {
      return backend;
    },

    get droppedFrames() {
      return droppedFrames;
    },
  };
}
