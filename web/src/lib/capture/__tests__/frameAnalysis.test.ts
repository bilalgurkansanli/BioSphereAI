/**
 * THE ANALYSER'S TWO PROMISES.
 *
 * 1. IT NEVER GOES SILENT. `readQuality()` and `readFlow()` feed `detectors.ts`; a `null`
 *    quality sample does not read as "unknown" downstream, it reads as a calm, well-lit,
 *    perfectly still room, and the whole clipping/brightness/smear half of the coaching
 *    quietly stops existing. Every way the worker can fail — missing globals, a constructor
 *    that throws, an `error` event, a worker that never answers — must land on the
 *    synchronous path still producing numbers.
 *
 * 2. IT DROPS RATHER THAN QUEUES. One bitmap in flight per kind. Queueing turns a worker
 *    that is 200 ms behind into an unbounded backlog of GPU handles delivering measurements
 *    of a room the phone stopped pointing at — and a late `divergencePx` over 7 is a hard
 *    stop shown to a user who is doing nothing wrong.
 *
 * The fakes below stand in for the platform: `createImageBitmap` is a deferred promise so a
 * test can hold a frame mid-flight, and the worker only replies when told to.
 */

import { describe, expect, it, vi } from "vitest";

import {
  createFrameAnalyser,
  qualityBitmapSize,
  type AnalysisCanvasLike,
  type AnalysisContextLike,
  type AnalysisWorkerLike,
  type FrameSource,
  type FrameSubmitOptions,
} from "@/lib/capture/frameAnalysis";
import type { AnalysisRequest, AnalysisResponse } from "@/lib/capture/frameAnalysis.worker";
import { analyseFrameLuma } from "@/lib/capture/frameKernels";
import { THRESHOLDS } from "@/lib/capture/types";

// -------------------------------------------------------------------------------------
// Fakes
// -------------------------------------------------------------------------------------

const FLOW_W = THRESHOLDS.flowDivergence.quadrantWidthPx * 2;
const FLOW_H = THRESHOLDS.flowDivergence.quadrantHeightPx * 2;

/** The same hash noise `sensors.test.ts` correlates against, so a shift is recoverable. */
const noise = (x: number, y: number): number => {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return (s - Math.floor(s)) * 200 + 20;
};

function rgbaOf(width: number, height: number, shiftX: number): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const v = Math.round(noise(x - shiftX, y));
      const i = (y * width + x) * 4;
      data[i] = v;
      data[i + 1] = v;
      data[i + 2] = v;
      data[i + 3] = 255;
    }
  }
  return data;
}

/**
 * A canvas whose `getImageData` synthesises the scene rather than replaying a draw. The
 * fallback path is being tested for its plumbing — which buffer reaches which kernel — not
 * for the browser's rasteriser.
 */
class FakeCanvas implements AnalysisCanvasLike {
  width = 0;
  height = 0;
  shiftX = 0;
  draws = 0;
  reads = 0;

  getContext(): AnalysisContextLike {
    return {
      drawImage: () => {
        this.draws += 1;
      },
      getImageData: (_sx: number, _sy: number, sw: number, sh: number) => {
        this.reads += 1;
        return { data: rgbaOf(sw, sh, this.shiftX) };
      },
    };
  }
}

class FakeWorker implements AnalysisWorkerLike {
  readonly posted: AnalysisRequest[] = [];
  terminated = false;
  private readonly handlers = new Map<string, (event: MessageEvent<AnalysisResponse>) => void>();

  postMessage(request: AnalysisRequest): void {
    this.posted.push(request);
  }

  addEventListener(type: "message", listener: (event: MessageEvent<AnalysisResponse>) => void): void;
  addEventListener(type: "error" | "messageerror", listener: () => void): void;
  addEventListener(type: string, listener: (event: MessageEvent<AnalysisResponse>) => void): void {
    this.handlers.set(type, listener);
  }

  terminate(): void {
    this.terminated = true;
  }

  reply(response: AnalysisResponse): void {
    this.handlers.get("message")?.({ data: response } as MessageEvent<AnalysisResponse>);
  }

  /** The bitmaps this worker was handed, in order, so a test can assert on their geometry. */
  bitmaps(): ImageBitmap[] {
    return this.posted.filter((r): r is Extract<AnalysisRequest, { bitmap: ImageBitmap }> => "bitmap" in r).map((r) => r.bitmap);
  }

  crash(): void {
    this.handlers.get("error")?.({} as MessageEvent<AnalysisResponse>);
  }
}

interface Deferred {
  resolve: () => void;
  reject: () => void;
}

/**
 * `createImageBitmap` that hands back a stub of exactly the requested geometry and, in
 * `deferred` mode, does not resolve until a test says so — which is how "still in flight"
 * is held open long enough to submit a second frame into it.
 */
function fakeBitmaps(deferred: boolean) {
  const pending: Deferred[] = [];
  let closed = 0;
  const create = (_source: FrameSource, options: ImageBitmapOptions) =>
    new Promise<ImageBitmap>((resolve, reject) => {
      const bitmap = {
        width: options.resizeWidth ?? 0,
        height: options.resizeHeight ?? 0,
        close: () => {
          closed += 1;
        },
      } as unknown as ImageBitmap;
      if (!deferred) {
        resolve(bitmap);
        return;
      }
      pending.push({ resolve: () => resolve(bitmap), reject: () => reject(new Error("no current frame")) });
    });
  return { create, pending, closedCount: () => closed };
}

const flush = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

const source = {} as unknown as FrameSource;

function submission(overrides: Partial<FrameSubmitOptions> = {}): FrameSubmitOptions {
  return {
    t: 0,
    quality: true,
    flow: false,
    mediaTime: 0,
    frameW: 1080,
    frameH: 1920,
    measuredFps: 30,
    reportedExposureS: 0,
    omegaDegPerS: 20,
    pxPerDeg: 14.18,
    ...overrides,
  };
}

// -------------------------------------------------------------------------------------
// Fallback
// -------------------------------------------------------------------------------------

describe("frame analyser — falling back to the main thread", () => {
  it("computes on the main thread when the platform has no Worker, and says so", () => {
    const canvas = new FakeCanvas();
    const onDemote = vi.fn();
    const analyser = createFrameAnalyser({
      createWorker: null,
      createBitmap: fakeBitmaps(false).create,
      createCanvas: () => canvas,
      onDemote,
    });

    expect(analyser.backend).toBe("main-thread");
    expect(onDemote).toHaveBeenCalledWith("no-worker");

    analyser.submit(source, submission({ t: 100, mediaTime: 3.5 }));

    const sample = analyser.readQuality();
    expect(sample).not.toBeNull();
    // The canvas is sized by the same rule `analyseQuality` used: 64 on the short edge.
    expect(canvas.width).toBe(64);
    expect(canvas.height).toBe(qualityBitmapSize(1080, 1920).height);
    expect(sample?.meanLuma).toBeCloseTo(analyseFrameLuma(rgbaOf(canvas.width, canvas.height, 0)).meanLuma, 6);
    expect(sample?.t).toBe(100);
    expect(sample?.mediaTime).toBe(3.5);
    // No platform exposureTime, so the sample must admit the number was inferred.
    expect(sample?.exposureSource).toBe("estimated");
    expect(sample?.smearPx).toBeGreaterThan(0);
  });

  it("reports a reported exposure as reported rather than re-estimating it", () => {
    const analyser = createFrameAnalyser({ createWorker: null, createBitmap: null, createCanvas: () => new FakeCanvas() });
    analyser.submit(source, submission({ reportedExposureS: 0.008 }));
    expect(analyser.readQuality()?.exposureS).toBeCloseTo(0.008, 9);
    expect(analyser.readQuality()?.exposureSource).toBe("reported");
  });

  it("falls back when createImageBitmap is missing even though a Worker exists", () => {
    const onDemote = vi.fn();
    const analyser = createFrameAnalyser({
      createWorker: () => new FakeWorker(),
      createBitmap: null,
      createCanvas: () => new FakeCanvas(),
      onDemote,
    });
    expect(analyser.backend).toBe("main-thread");
    expect(onDemote).toHaveBeenCalledWith("no-createImageBitmap");
    analyser.submit(source, submission());
    expect(analyser.readQuality()).not.toBeNull();
  });

  it("falls back when the worker constructor throws", () => {
    const onDemote = vi.fn();
    const analyser = createFrameAnalyser({
      createWorker: () => {
        throw new TypeError("Failed to construct 'Worker'");
      },
      createBitmap: fakeBitmaps(false).create,
      createCanvas: () => new FakeCanvas(),
      onDemote,
    });
    expect(analyser.backend).toBe("main-thread");
    expect(onDemote).toHaveBeenCalledWith("worker-construct:TypeError");
    analyser.submit(source, submission());
    expect(analyser.readQuality()).not.toBeNull();
  });

  it("falls back mid-capture on a worker error, and keeps producing samples afterwards", async () => {
    const worker = new FakeWorker();
    const canvas = new FakeCanvas();
    const onDemote = vi.fn();
    const analyser = createFrameAnalyser({
      createWorker: () => worker,
      createBitmap: fakeBitmaps(false).create,
      createCanvas: () => canvas,
      onDemote,
    });
    expect(analyser.backend).toBe("worker");

    analyser.submit(source, submission({ t: 0 }));
    await flush();
    worker.crash();

    expect(analyser.backend).toBe("main-thread");
    expect(onDemote).toHaveBeenCalledWith("worker-error");
    expect(worker.terminated).toBe(true);

    analyser.submit(source, submission({ t: 100 }));
    expect(analyser.readQuality()?.t).toBe(100);
  });

  it("writes off a worker that never answers, on the loop's own clock", async () => {
    const worker = new FakeWorker();
    const onDemote = vi.fn();
    const analyser = createFrameAnalyser({
      createWorker: () => worker,
      createBitmap: fakeBitmaps(false).create,
      createCanvas: () => new FakeCanvas(),
      onDemote,
    });

    analyser.submit(source, submission({ t: 0 }));
    await flush();
    expect(analyser.backend).toBe("worker");

    // Still inside the grace period, and the grace is long on purpose: the worker chunks are
    // not preloaded, so silence here is a cold module fetch over the network, not a failure.
    analyser.submit(source, submission({ t: 5000 }));
    expect(analyser.backend).toBe("worker");
    // ...and nothing is blind while it waits — the synchronous path runs alongside until the
    // worker has answered once, which is what makes a generous deadline affordable.
    expect(analyser.readQuality()?.t).toBe(5000);

    analyser.submit(source, submission({ t: 8500 }));
    expect(analyser.backend).toBe("main-thread");
    expect(onDemote).toHaveBeenCalledWith("worker-silent");
    // The demotion must also serve the frame it was triggered by, not skip it.
    expect(analyser.readQuality()?.t).toBe(8500);
  });

  it("recovers the same flow shift on the fallback path as the kernels do", () => {
    const canvas = new FakeCanvas();
    const analyser = createFrameAnalyser({ createWorker: null, createBitmap: null, createCanvas: () => canvas });

    analyser.submit(source, submission({ quality: false, flow: true, t: 0 }));
    // The first frame has nothing to correlate against; a sample here would be invented.
    expect(analyser.readFlow()).toBeNull();

    canvas.shiftX = 6;
    analyser.submit(source, submission({ quality: false, flow: true, t: 200 }));

    const flow = analyser.readFlow();
    expect(flow?.valid).toBe(true);
    expect(canvas.width).toBe(FLOW_W);
    expect(canvas.height).toBe(FLOW_H);
    // 6 canvas px * (640 / 192) = 20 px at the contract's reference width.
    expect(flow?.meanShiftPx).toBeCloseTo(20, 0);
  });
});

// -------------------------------------------------------------------------------------
// Drop, never queue
// -------------------------------------------------------------------------------------

describe("frame analyser — one bitmap in flight per kind", () => {
  it("drops a due quality frame while the previous one is still resolving", async () => {
    const worker = new FakeWorker();
    const bitmaps = fakeBitmaps(true);
    const analyser = createFrameAnalyser({ createWorker: () => worker, createBitmap: bitmaps.create, createCanvas: () => new FakeCanvas() });

    analyser.submit(source, submission({ t: 0 }));
    expect(bitmaps.pending).toHaveLength(1);

    // The bitmap has not even resolved yet, let alone been analysed. The gate holds here or
    // it holds nowhere.
    analyser.submit(source, submission({ t: 100 }));
    analyser.submit(source, submission({ t: 200 }));
    expect(bitmaps.pending).toHaveLength(1);
    expect(analyser.droppedFrames).toBe(2);

    bitmaps.pending[0].resolve();
    await flush();
    expect(worker.posted).toHaveLength(1);
  });

  it("drops a due quality frame while the worker still owes a reply", async () => {
    const worker = new FakeWorker();
    const bitmaps = fakeBitmaps(false);
    const analyser = createFrameAnalyser({ createWorker: () => worker, createBitmap: bitmaps.create, createCanvas: () => new FakeCanvas() });

    analyser.submit(source, submission({ t: 0 }));
    await flush();
    expect(worker.posted).toHaveLength(1);

    analyser.submit(source, submission({ t: 100 }));
    await flush();
    expect(worker.posted).toHaveLength(1);
    expect(analyser.droppedFrames).toBe(1);

    worker.reply({ kind: "quality", seq: 1, meanLuma: 140, clippedFraction: 0.02 });
    analyser.submit(source, submission({ t: 200 }));
    await flush();
    expect(worker.posted).toHaveLength(2);
    expect(analyser.droppedFrames).toBe(1);
  });

  it("keeps the two kinds independent — a stalled flow must not silence the light meter", async () => {
    const worker = new FakeWorker();
    const bitmaps = fakeBitmaps(false);
    const analyser = createFrameAnalyser({ createWorker: () => worker, createBitmap: bitmaps.create, createCanvas: () => new FakeCanvas() });

    analyser.submit(source, submission({ t: 0, quality: true, flow: true }));
    await flush();
    expect(worker.posted.map((r) => r.kind)).toEqual(["quality", "flow"]);

    // Flow never answers; quality does. Two more quality frames must still get through.
    worker.reply({ kind: "quality", seq: 1, meanLuma: 150, clippedFraction: 0 });
    analyser.submit(source, submission({ t: 100, quality: true, flow: true }));
    await flush();
    worker.reply({ kind: "quality", seq: 3, meanLuma: 150, clippedFraction: 0 });
    analyser.submit(source, submission({ t: 200, quality: true, flow: true }));
    await flush();

    expect(worker.posted.filter((r) => r.kind === "quality")).toHaveLength(3);
    expect(worker.posted.filter((r) => r.kind === "flow")).toHaveLength(1);
    expect(analyser.droppedFrames).toBe(2);
  });

  it("submits bitmaps already resized to the analysis geometry, not full frames", async () => {
    const worker = new FakeWorker();
    const analyser = createFrameAnalyser({
      createWorker: () => worker,
      createBitmap: fakeBitmaps(false).create,
      createCanvas: () => new FakeCanvas(),
    });

    analyser.submit(source, submission({ t: 0, quality: true, flow: true }));
    await flush();

    const [quality, flow] = worker.bitmaps();
    expect({ w: quality.width, h: quality.height }).toEqual({ w: 64, h: qualityBitmapSize(1080, 1920).height });
    expect({ w: flow.width, h: flow.height }).toEqual({ w: FLOW_W, h: FLOW_H });

    // The REQUESTED geometry travels alongside, because `resizeWidth` is only a hint: an
    // engine that ignores it hands back a full 1080x1920 handle, and a worker that sized its
    // canvas from the bitmap would then read back 8 MB per frame instead of 29 KB.
    const requested = worker.posted.map((r) => ("width" in r ? { w: r.width, h: r.height } : null));
    expect(requested).toEqual([
      { w: 64, h: qualityBitmapSize(1080, 1920).height },
      { w: FLOW_W, h: FLOW_H },
    ]);
  });

  it("releases the gate when createImageBitmap rejects, instead of wedging shut", async () => {
    const worker = new FakeWorker();
    const bitmaps = fakeBitmaps(true);
    const analyser = createFrameAnalyser({ createWorker: () => worker, createBitmap: bitmaps.create, createCanvas: () => new FakeCanvas() });

    // A video between `play()` and its first decoded frame rejects here. It is a normal
    // moment in the lens hand-off, not a reason to stop measuring for the rest of the run.
    analyser.submit(source, submission({ t: 0 }));
    bitmaps.pending[0].reject();
    await flush();

    analyser.submit(source, submission({ t: 100 }));
    bitmaps.pending[1].resolve();
    await flush();
    expect(worker.posted).toHaveLength(1);
  });

  it("pausing resets the worker's flow history and discards replies in flight", async () => {
    const worker = new FakeWorker();
    const analyser = createFrameAnalyser({
      createWorker: () => worker,
      createBitmap: fakeBitmaps(false).create,
      createCanvas: () => new FakeCanvas(),
    });

    analyser.submit(source, submission({ t: 0 }));
    await flush();
    analyser.setEnabled(false);
    expect(worker.posted.some((r) => r.kind === "reset")).toBe(true);

    // The reply describes a frame from before the pause. Publishing it would date-stamp a
    // pre-pause room onto the moment the user resumed — so what matters is that this reply
    // is DISCARDED, not that nothing has ever been measured. The synchronous path runs
    // alongside the worker until it proves itself, so a real sample legitimately exists here.
    const before = analyser.readQuality()?.meanLuma;
    worker.reply({ kind: "quality", seq: 1, meanLuma: 200, clippedFraction: 0 });
    expect(analyser.readQuality()?.meanLuma).toBe(before);
    expect(analyser.readQuality()?.meanLuma).not.toBe(200);

    analyser.submit(source, submission({ t: 100 }));
    await flush();
    expect(worker.posted.filter((r) => r.kind === "quality")).toHaveLength(1);

    analyser.setEnabled(true);
    analyser.submit(source, submission({ t: 200 }));
    await flush();
    expect(worker.posted.filter((r) => r.kind === "quality")).toHaveLength(2);
  });

  it("a bitmap that resolves after a pause never overtakes the reset", async () => {
    const worker = new FakeWorker();
    const bitmaps = fakeBitmaps(true);
    const analyser = createFrameAnalyser({ createWorker: () => worker, createBitmap: bitmaps.create, createCanvas: () => new FakeCanvas() });

    analyser.submit(source, submission({ t: 0, quality: false, flow: true }));
    analyser.setEnabled(false);

    // Arriving now, this frame would land BEHIND the reset and re-arm the worker's flow
    // history with a pre-pause room. It has to be dropped and its GPU handle closed.
    bitmaps.pending[0].resolve();
    await flush();

    expect(worker.posted.filter((r) => r.kind === "flow")).toHaveLength(0);
    expect(bitmaps.closedCount()).toBe(1);
  });

  it("dispose terminates the worker and stops accepting frames", async () => {
    const worker = new FakeWorker();
    const analyser = createFrameAnalyser({
      createWorker: () => worker,
      createBitmap: fakeBitmaps(false).create,
      createCanvas: () => new FakeCanvas(),
    });

    analyser.dispose();
    analyser.submit(source, submission({ t: 0 }));
    await flush();

    expect(worker.terminated).toBe(true);
    expect(worker.posted).toHaveLength(0);
    expect(analyser.readQuality()).toBeNull();
  });
});
