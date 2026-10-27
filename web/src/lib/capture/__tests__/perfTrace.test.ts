/**
 * The summariser is the only part of `perfTrace.ts` that can be wrong quietly.
 *
 * A tracer whose percentiles are off does not fail — it reports a plausible number and sends
 * the next round of fixes at the wrong target, which is exactly the situation this module was
 * built to end. So the maths is pinned against inputs whose answers are arithmetic rather
 * than empirical, including the two mistakes that are easy to make here: sorting a plain
 * array lexicographically (which orders 100 before 9), and interpolating a percentile so
 * that a bimodal distribution reports a duration that never occurred.
 */

import { describe, expect, it } from "vitest";
import {
  type PerfStageSamples,
  type PerfTraceRaw,
  percentileOf,
  summarise,
  summaryToText,
} from "@/lib/capture/perfTrace";

/** 1..100 inclusive, so nearest-rank on `Math.round((n - 1) * p)` has a hand-checkable answer. */
const ONE_TO_100 = Array.from({ length: 100 }, (_, i) => i + 1);

function makeRaw(overrides: Partial<PerfTraceRaw> = {}): PerfTraceRaw {
  return {
    label: "test",
    startedAtMs: 0,
    endedAtMs: 10_000,
    wallClockIso: "2026-08-30T00:00:00.000Z",
    frameIntervals: [],
    frameCount: 0,
    frameDropped: 0,
    longTaskSupported: true,
    longTaskCount: 0,
    longTaskTotalMs: 0,
    longTaskMaxMs: 0,
    stages: [],
    stagesDropped: 0,
    memory: null,
    cameraSettings: null,
    notes: {},
    device: {
      userAgent: "test",
      devicePixelRatio: 2,
      hardwareConcurrency: 8,
      screenWidth: 412,
      screenHeight: 915,
      deviceMemoryGb: 4,
    },
    ...overrides,
  };
}

function stage(name: string, values: number[], dropped = 0): PerfStageSamples {
  return { name, values, dropped };
}

describe("percentileOf", () => {
  it("is nearest-rank over 1..100", () => {
    // index = round((n - 1) * p): 0.5 -> round(49.5) = 50 -> value 51.
    expect(percentileOf(ONE_TO_100, 0.5)).toBe(51);
    // 0.95 -> round(94.05) = 94 -> value 95.
    expect(percentileOf(ONE_TO_100, 0.95)).toBe(95);
    // 0.99 -> round(98.01) = 98 -> value 99.
    expect(percentileOf(ONE_TO_100, 0.99)).toBe(99);
    expect(percentileOf(ONE_TO_100, 0)).toBe(1);
    expect(percentileOf(ONE_TO_100, 1)).toBe(100);
  });

  it("sorts numerically, not lexicographically", () => {
    // Lexicographic order puts 100 before 9 and 1000 before 2, so a wrong sort reports a
    // p95 of 9 here instead of 1000.
    const values = [1000, 9, 100, 2, 30];
    expect(percentileOf(values, 1)).toBe(1000);
    expect(percentileOf(values, 0.5)).toBe(30);
  });

  it("does not interpolate between the two modes of a bimodal sample", () => {
    // 90 cheap frames and 10 stalled ones: the shape a GPU readback stall actually makes.
    const bimodal = [...Array.from({ length: 90 }, () => 4), ...Array.from({ length: 10 }, () => 180)];
    // Every reported value must be one that was actually observed.
    for (const p of [0.5, 0.95, 0.99]) expect([4, 180]).toContain(percentileOf(bimodal, p));
    expect(percentileOf(bimodal, 0.5)).toBe(4);
    expect(percentileOf(bimodal, 0.95)).toBe(180);
  });

  it("ignores non-finite samples and survives an empty series", () => {
    expect(percentileOf([], 0.5)).toBe(0);
    expect(percentileOf([NaN, Infinity, 5, 7], 1)).toBe(7);
    expect(percentileOf([NaN, Infinity, 5, 7], 0)).toBe(5);
  });

  it("reads a Float32Array, which is what the ring buffer hands over", () => {
    expect(percentileOf(Float32Array.from(ONE_TO_100), 0.95)).toBe(95);
  });
});

describe("summarise", () => {
  it("derives frame percentiles, real fps and hitches from a known series", () => {
    // 600 frames over 10 s: 594 at a clean 16 ms, 6 hitches at 120 ms.
    const intervals = [
      ...Array.from({ length: 594 }, () => 16),
      ...Array.from({ length: 6 }, () => 120),
    ];
    const s = summarise(makeRaw({ frameIntervals: intervals, frameCount: 600 }));

    expect(s.frames.count).toBe(600);
    // Real fps is frames per wall-clock second, NOT 1000 / p50 — the difference between the
    // two is precisely the lag the user is reporting.
    expect(s.frames.fps).toBe(60);
    expect(s.frames.p50Ms).toBe(16);
    expect(s.frames.p95Ms).toBe(16);
    // 600 samples: index round(599 * 0.99) = 593 is the last 16 ms sample; the hitches start
    // one index later, which is the point of pinning p99 on a series with exactly 1% tail.
    expect(s.frames.p99Ms).toBe(16);
    expect(s.frames.worstMs).toBe(120);
    expect(s.frames.hitches).toBe(6);
    expect(s.frames.windowed).toBe(false);
  });

  it("flags a windowed trace, because a p99 over a truncated window is a different claim", () => {
    const s = summarise(makeRaw({ frameIntervals: [16, 16], frameCount: 5000, frameDropped: 904 }));
    expect(s.frames.windowed).toBe(true);
    expect(s.frames.count).toBe(5000);
  });

  it("splits long-task total from blocking time at the 50 ms threshold", () => {
    // Four tasks summing to 400 ms: blocking time is 400 - 4 * 50 = 200 ms.
    const s = summarise(
      makeRaw({ longTaskCount: 4, longTaskTotalMs: 400, longTaskMaxMs: 180, endedAtMs: 2000 }),
    );
    expect(s.longTasks.totalMs).toBe(400);
    expect(s.longTasks.blockingMs).toBe(200);
    expect(s.longTasks.maxMs).toBe(180);
    expect(s.longTasks.blockedFraction).toBe(0.2);
  });

  it("never reports negative blocking time when every task is barely over the threshold", () => {
    const s = summarise(makeRaw({ longTaskCount: 3, longTaskTotalMs: 120 }));
    expect(s.longTasks.blockingMs).toBe(0);
  });

  it("reports the absence of the long-task API rather than a plausible zero", () => {
    const s = summarise(makeRaw({ longTaskSupported: false }));
    expect(s.longTasks.supported).toBe(false);
    expect(s.longTasks.count).toBe(0);
  });

  it("gives each stage its own percentiles, total and real cadence", () => {
    const s = summarise(
      makeRaw({
        endedAtMs: 10_000,
        stages: [
          stage("analyseQuality", ONE_TO_100),
          stage("analyseFlow", [2, 2, 2, 2, 90]),
        ],
      }),
    );

    const quality = s.stages.analyseQuality;
    expect(quality.count).toBe(100);
    expect(quality.p50Ms).toBe(51);
    expect(quality.p95Ms).toBe(95);
    expect(quality.maxMs).toBe(100);
    expect(quality.totalMs).toBe(5050);
    expect(quality.hz).toBe(10);

    const flow = s.stages.analyseFlow;
    expect(flow.p50Ms).toBe(2);
    expect(flow.maxMs).toBe(90);
    expect(flow.hz).toBe(0.5);
  });

  it("counts overwritten stage samples in `count` but not in the percentiles", () => {
    // The ring holds the last three; twelve more were overwritten. The cadence must still
    // reflect all fifteen calls, or a stage that overflowed looks like it stopped running.
    const s = summarise(makeRaw({ endedAtMs: 1000, stages: [stage("stamp", [1, 2, 3], 12)] }));
    expect(s.stages.stamp.count).toBe(15);
    expect(s.stages.stamp.hz).toBe(15);
    expect(s.stages.stamp.p50Ms).toBe(2);
  });

  it("returns zeroes rather than NaN for a trace that recorded nothing", () => {
    const s = summarise(makeRaw({ startedAtMs: 500, endedAtMs: 500 }));
    expect(s.durationMs).toBe(0);
    expect(s.frames.fps).toBe(0);
    expect(s.frames.p50Ms).toBe(0);
    expect(s.longTasks.blockedFraction).toBe(0);
    expect(Object.keys(s.stages)).toHaveLength(0);
  });

  it("passes device, camera and heap facts through untouched", () => {
    const s = summarise(
      makeRaw({
        memory: { usedMb: 120.5, totalMb: 180, limitMb: 512 },
        cameraSettings: { width: 1080, height: 1920, frameRate: 30 },
      }),
    );
    expect(s.camera).toEqual({ width: 1080, height: 1920, frameRate: 30 });
    expect(s.memory?.usedMb).toBe(120.5);
    expect(s.device.hardwareConcurrency).toBe(8);
    expect(s.v).toBe(1);
  });
});

describe("summaryToText", () => {
  it("renders every stage and survives the fields the phone may not supply", () => {
    const text = summaryToText(
      summarise(
        makeRaw({
          frameIntervals: [16, 16, 33],
          frameCount: 3,
          stages: [stage("analyseFlow", [2, 90])],
          longTaskCount: 2,
          longTaskTotalMs: 300,
          longTaskMaxMs: 200,
        }),
      ),
    );
    expect(text).toContain("analyseFlow");
    expect(text).toContain("fps");
    // No camera and no heap on this trace: those lines are omitted, not rendered as "null".
    expect(text).not.toContain("null");
    expect(text).not.toContain("undefined");
  });

  it("says so when the browser has no long-task observer", () => {
    expect(summaryToText(summarise(makeRaw({ longTaskSupported: false })))).toContain("unsupported");
  });
});
