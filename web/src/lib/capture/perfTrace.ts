"use client";

/**
 * PERF TRACE — the profiler we cannot attach to the user's phone.
 * =====================================================================================
 *
 * ── WHY THIS EXISTS ──────────────────────────────────────────────────────────────────
 * Two rounds of lag fixes have shipped against a mid-range Android 10 / Chrome 151 handset
 * that nobody here can plug into DevTools, and both rounds were reasoned rather than
 * measured. Every number the fixes were argued from — "the coverage maths is 0.25 ms",
 * "React reconciliation was 348 ms of 900 ticks" — came off a desktop, where there is no
 * hardware H.264 encoder running at 12 Mbps and no thermal ceiling. The user has now said
 * three times that the capture is unusable. The gap this closes is that the phone measures
 * itself and posts the result, with the user doing nothing beyond finishing the capture.
 *
 * ── WHAT IT COSTS TO RUN ─────────────────────────────────────────────────────────────
 * A tracer that perturbs the thing it measures is worse than no tracer. Everything on the
 * hot path here is a write into a preallocated `Float32Array` behind a modulo: no object is
 * allocated per sample, so the GC never runs because of this module, and no percentile,
 * sort, string or JSON is touched until `summary()` is called on the cold path after the
 * recording stops. The one cost that is genuinely added is a bare `requestAnimationFrame`
 * loop whose body is a subtraction and two array writes — that is deliberate, because a
 * frame cadence measured from inside the caller's own gated loop only ever reports the
 * gate, and the number we need is whether the compositor is being starved.
 *
 * ── WHAT THE ANSWER LOOKS LIKE ───────────────────────────────────────────────────────
 * If the prime suspect is right — `analyseQuality` at 10 Hz plus `analyseFlow` at 5 Hz, each
 * a `drawImage` + `getImageData` pair, i.e. 15 synchronous GPU→CPU readbacks per second
 * while a hardware encoder holds the pipeline — the shape in the summary is unmistakable:
 * the long-task count sits near 15/s, the stage p95 for the readback stages dwarfs their
 * p50 (a stall waits on whatever the GPU happened to be doing, so it is bimodal), and the
 * frame-interval p99 lands at a multiple of the p50 while real fps collapses. If instead
 * the frame intervals are flat and the long tasks are few, the readbacks are exonerated and
 * the cost is somewhere nobody has looked yet. Either way the next round is not a guess.
 *
 * ── FAILURE MODES THIS MODULE MUST NOT HAVE ──────────────────────────────────────────
 * It must never throw into the capture loop and never block it. `sendBeacon` is used ahead
 * of `fetch` for exactly that reason: the browser takes the payload and the call returns,
 * and it survives the page being torn down, which `fetch` without `keepalive` does not.
 * `longtask` is absent on Safari and on some Android WebViews, so it is feature-checked
 * rather than assumed, and its absence is reported in the summary instead of being silently
 * indistinguishable from "no long tasks happened".
 */

import type { Millis } from "./types";

// -------------------------------------------------------------------------------------
// Capacities
// -------------------------------------------------------------------------------------

/**
 * 4096 frame intervals is ~68 s at 60 Hz and ~136 s at 30 Hz, which covers a whole capture
 * on the device we care about. The ring overwrites its oldest sample rather than growing,
 * so a trace that outlives its window describes the LAST 68 s — stated in the summary as
 * `frames.windowed`, because a p99 over a truncated window is a different claim.
 */
const FRAME_CAPACITY = 4096;

/** 1024 samples is ~100 s of a 10 Hz stage, and 16 KB of Float32 per stage. */
const STAGE_CAPACITY = 1024;

/**
 * A name typo, or a name built from a variable, would otherwise allocate a fresh ring
 * forever. Sixteen is more distinct stages than the loop has.
 */
const MAX_STAGES = 16;

/**
 * The browser's own definition of a long task. Kept as a constant because the summary
 * reports both the raw sum and the blocking time beyond this threshold, and confusing the
 * two is how a "1200 ms blocked" number becomes unfalsifiable.
 */
const LONG_TASK_THRESHOLD_MS = 50;

/** A frame interval past this is a visible hitch at any refresh rate worth shipping to. */
const HITCH_MS = 50;

// -------------------------------------------------------------------------------------
// Public shapes
// -------------------------------------------------------------------------------------

export interface PerfDeviceFacts {
  userAgent: string;
  devicePixelRatio: number;
  hardwareConcurrency: number;
  screenWidth: number;
  screenHeight: number;
  /** `navigator.deviceMemory`, in GiB. Chrome-only, and quantised to 0.25/0.5/1/2/4/8. */
  deviceMemoryGb: number | null;
}

export interface PerfMemoryFacts {
  usedMb: number;
  totalMb: number;
  limitMb: number;
}

export interface PerfStageSamples {
  name: string;
  /** Durations in ms. `Float32Array` in production, a plain array in the tests. */
  values: ArrayLike<number>;
  /** Samples the ring overwrote, i.e. how much of the stage's history is missing. */
  dropped: number;
}

/**
 * The raw counters, exactly as the recorder holds them. Separated from the summary so the
 * percentile maths is a pure function of plain data and can be tested with no DOM at all.
 */
export interface PerfTraceRaw {
  label: string;
  startedAtMs: Millis;
  endedAtMs: Millis;
  wallClockIso: string;
  frameIntervals: ArrayLike<number>;
  /** Total frames seen, which exceeds `frameIntervals.length` once the ring has wrapped. */
  frameCount: number;
  frameDropped: number;
  longTaskSupported: boolean;
  longTaskCount: number;
  longTaskTotalMs: number;
  longTaskMaxMs: number;
  stages: readonly PerfStageSamples[];
  stagesDropped: number;
  memory: PerfMemoryFacts | null;
  cameraSettings: Record<string, unknown> | null;
  /** Free-form facts the caller recorded once each; see `PerfTrace.note`. */
  notes: Record<string, string>;
  device: PerfDeviceFacts;
}

export interface PerfFrameSummary {
  count: number;
  /** Frames divided by wall-clock seconds — the number the user actually experiences. */
  fps: number;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  worstMs: number;
  /** Frames slower than 50 ms. One per second is a visible stutter. */
  hitches: number;
  /** True when the ring wrapped, so the percentiles describe a window, not the capture. */
  windowed: boolean;
}

export interface PerfLongTaskSummary {
  supported: boolean;
  count: number;
  /** Sum of the full task durations. */
  totalMs: number;
  /** Sum of `duration - 50` — the browser's own "total blocking time". */
  blockingMs: number;
  maxMs: number;
  /** `totalMs` as a share of wall clock. Above ~0.5 the main thread has nothing left. */
  blockedFraction: number;
}

export interface PerfStageSummary {
  count: number;
  p50Ms: number;
  p95Ms: number;
  maxMs: number;
  totalMs: number;
  /** Calls per second across the trace, so a stage's real cadence is checkable. */
  hz: number;
}

export interface PerfSummary {
  /** Bumped when a field changes meaning, so an old trace in the log is never misread. */
  v: 1;
  label: string;
  at: string;
  durationMs: number;
  frames: PerfFrameSummary;
  longTasks: PerfLongTaskSummary;
  stages: Record<string, PerfStageSummary>;
  /** Stage names refused past the 16-name cap — non-zero means a caller is building names
   * dynamically and the per-stage numbers below are incomplete. */
  stagesDropped: number;
  memory: PerfMemoryFacts | null;
  camera: Record<string, unknown> | null;
  notes: Record<string, string>;
  device: PerfDeviceFacts;
}

export interface PerfTraceOptions {
  /** Distinguishes traces in the log — the capture phase, a build tag, anything. */
  label?: string;
  /** Where `flush()` posts. Without one, `flush()` is a no-op that still returns a summary. */
  endpoint?: string;
  /**
   * Off by default. This is a diagnostic, not a product feature: the page turns it on for
   * `?debug=1` or for a build that is chasing the lag, and a disabled trace costs one
   * boolean test per call because every method returns early.
   */
  enabled?: boolean;
  /** Post automatically when the page is hidden or unloaded, which is how a trace survives
   * the user simply closing the tab instead of finishing the flow. */
  flushOnHide?: boolean;
  now?: () => Millis;
}

export interface PerfTrace {
  readonly enabled: boolean;
  /** Begin sampling. Idempotent — a second call does not install a second rAF loop. */
  start(): void;
  /** Stop sampling and release the rAF loop and the observer. Idempotent. */
  stop(): void;
  /** Record one named stage duration, in ms. Hot path: no allocation, no sort, no string. */
  stage(name: string, durationMs: number): void;
  /** Time `fn` and record it under `name`, returning whatever `fn` returned. */
  measure<T>(name: string, fn: () => T): T;
  /** Capture the camera track's settings once — the resolution and frame rate the phone
   * actually gave us, which is the denominator for every number above. */
  attachTrack(track: MediaStreamTrack | null | undefined): void;
  /**
   * Record one fact, first write wins.
   *
   * For the things that are not durations and not device capabilities but decide the whole
   * argument — above all WHICH mimeType the recorder actually landed on. Chrome for Android
   * has no software H.264 in the MediaRecorder path, so `video/mp4` proves a hardware encode
   * and `video/webm` proves libvpx on the CPU. Three rounds of analysis argued about that
   * from a comment instead of from the device.
   */
  note(key: string, value: string): void;
  /** Cold path. Sorts copies of the rings; never call this from the loop. */
  summary(): PerfSummary;
  /** Human-readable, for the DebugHud on the phone itself. */
  toText(): string;
  /** Post the summary. Returns the summary it posted, or null when disabled. Never throws. */
  flush(endpoint?: string): PerfSummary | null;
}

// -------------------------------------------------------------------------------------
// Numeric helpers
// -------------------------------------------------------------------------------------

/** Copied verbatim from `telemetry.ts` so both payloads round identically. */
const round = (value: number, digits: number): number => {
  if (!Number.isFinite(value)) return 0;
  const f = 10 ** digits;
  return Math.round(value * f) / f;
};

/**
 * Nearest-rank on an ascending array, copied verbatim from `telemetry.ts`.
 *
 * Deliberately NOT interpolated. An interpolated p99 of a bimodal distribution — which is
 * exactly what a GPU readback stall produces, a cheap mode and a stalled mode with nothing
 * in between — invents a value that never occurred, and the whole point of this trace is to
 * report durations the phone actually spent.
 */
function percentileOfSorted(sorted: ArrayLike<number>, p: number): number {
  if (sorted.length === 0) return 0;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * p)));
  return sorted[index];
}

/**
 * Ascending copy with the non-finite samples dropped.
 *
 * `Float64Array.prototype.sort` is numeric by default; `Array.prototype.sort` is
 * lexicographic, which silently orders 100 before 9 and has produced wrong percentiles in
 * this codebase before.
 */
function sortedFinite(values: ArrayLike<number>): Float64Array {
  let n = 0;
  for (let i = 0; i < values.length; i += 1) if (Number.isFinite(values[i])) n += 1;
  const out = new Float64Array(n);
  let w = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (Number.isFinite(v)) {
      out[w] = v;
      w += 1;
    }
  }
  out.sort();
  return out;
}

/** Exposed for the tests and for any caller that already holds a sample array. */
export function percentileOf(values: ArrayLike<number>, p: number): number {
  return percentileOfSorted(sortedFinite(values), p);
}

// -------------------------------------------------------------------------------------
// Ring buffer
// -------------------------------------------------------------------------------------

/**
 * One preallocated `Float32Array` per series. Float32 rather than Float64 because these are
 * millisecond durations from a clock that is itself coarsened to 100 µs by cross-origin
 * isolation policy — the precision Float32 loses is precision the samples never had.
 */
class Ring {
  readonly values: Float32Array;
  count = 0;
  private write = 0;

  constructor(capacity: number) {
    this.values = new Float32Array(capacity);
  }

  push(value: number): void {
    this.values[this.write] = value;
    this.write = (this.write + 1) % this.values.length;
    this.count += 1;
  }

  /** How many samples the ring overwrote. */
  get dropped(): number {
    return Math.max(0, this.count - this.values.length);
  }

  /** The live samples, oldest-first order not preserved — percentiles do not care. */
  snapshot(): Float32Array {
    return this.count >= this.values.length ? this.values : this.values.subarray(0, this.count);
  }
}

// -------------------------------------------------------------------------------------
// The summariser — pure, so it is testable without a DOM
// -------------------------------------------------------------------------------------

export function summarise(raw: PerfTraceRaw): PerfSummary {
  const durationMs = Math.max(0, raw.endedAtMs - raw.startedAtMs);
  const seconds = durationMs / 1000;

  const frames = sortedFinite(raw.frameIntervals);
  let hitches = 0;
  for (let i = frames.length - 1; i >= 0; i -= 1) {
    if (frames[i] <= HITCH_MS) break;
    hitches += 1;
  }

  const stages: Record<string, PerfStageSummary> = {};
  for (const stage of raw.stages) {
    const sorted = sortedFinite(stage.values);
    let total = 0;
    for (let i = 0; i < sorted.length; i += 1) total += sorted[i];
    stages[stage.name] = {
      count: sorted.length + stage.dropped,
      p50Ms: round(percentileOfSorted(sorted, 0.5), 3),
      p95Ms: round(percentileOfSorted(sorted, 0.95), 3),
      maxMs: round(sorted.length > 0 ? sorted[sorted.length - 1] : 0, 3),
      totalMs: round(total, 1),
      hz: round(seconds > 0 ? (sorted.length + stage.dropped) / seconds : 0, 2),
    };
  }

  const blocking = Math.max(0, raw.longTaskTotalMs - raw.longTaskCount * LONG_TASK_THRESHOLD_MS);

  return {
    v: 1,
    label: raw.label,
    at: raw.wallClockIso,
    durationMs: round(durationMs, 0),
    frames: {
      count: raw.frameCount,
      fps: round(seconds > 0 ? raw.frameCount / seconds : 0, 2),
      p50Ms: round(percentileOfSorted(frames, 0.5), 2),
      p95Ms: round(percentileOfSorted(frames, 0.95), 2),
      p99Ms: round(percentileOfSorted(frames, 0.99), 2),
      worstMs: round(frames.length > 0 ? frames[frames.length - 1] : 0, 2),
      hitches,
      windowed: raw.frameDropped > 0,
    },
    longTasks: {
      supported: raw.longTaskSupported,
      count: raw.longTaskCount,
      totalMs: round(raw.longTaskTotalMs, 1),
      blockingMs: round(blocking, 1),
      maxMs: round(raw.longTaskMaxMs, 1),
      blockedFraction: round(durationMs > 0 ? raw.longTaskTotalMs / durationMs : 0, 3),
    },
    stages,
    stagesDropped: raw.stagesDropped,
    memory: raw.memory,
    camera: raw.cameraSettings,
    notes: raw.notes,
    device: raw.device,
  };
}

/** The DebugHud renders this straight into a `<pre>`; it is also what a screenshot carries. */
export function summaryToText(s: PerfSummary): string {
  const lines: string[] = [];
  lines.push(`trace ${s.label} · ${round(s.durationMs / 1000, 1)}s`);
  lines.push(
    `fps ${s.frames.fps.toFixed(1)} · frame p50 ${s.frames.p50Ms.toFixed(1)} p95 ${s.frames.p95Ms.toFixed(
      1,
    )} p99 ${s.frames.p99Ms.toFixed(1)} worst ${s.frames.worstMs.toFixed(0)}ms`,
  );
  lines.push(
    `hitches ${s.frames.hitches}${s.frames.windowed ? " (windowed)" : ""} · long ${
      s.longTasks.supported ? `${s.longTasks.count}x ${s.longTasks.totalMs.toFixed(0)}ms (${Math.round(
        s.longTasks.blockedFraction * 100,
      )}% blocked, max ${s.longTasks.maxMs.toFixed(0)}ms)` : "unsupported"
    }`,
  );
  for (const [name, st] of Object.entries(s.stages)) {
    lines.push(
      `  ${name} ${st.count}x @${st.hz.toFixed(1)}Hz · p50 ${st.p50Ms.toFixed(2)} p95 ${st.p95Ms.toFixed(
        2,
      )} max ${st.maxMs.toFixed(1)} · ${st.totalMs.toFixed(0)}ms total`,
    );
  }
  if (s.memory) lines.push(`heap ${s.memory.usedMb.toFixed(0)}/${s.memory.limitMb.toFixed(0)} MB`);
  if (s.camera) {
    const c = s.camera;
    lines.push(`cam ${String(c.width ?? "?")}x${String(c.height ?? "?")} @${String(c.frameRate ?? "?")}`);
  }
  lines.push(
    `dev ${s.device.hardwareConcurrency}c · dpr ${s.device.devicePixelRatio} · ${s.device.screenWidth}x${s.device.screenHeight}${
      s.device.deviceMemoryGb === null ? "" : ` · ${s.device.deviceMemoryGb}GB`
    }`,
  );
  return lines.join("\n");
}

// -------------------------------------------------------------------------------------
// Environment probes
// -------------------------------------------------------------------------------------

/** `performance.memory` is non-standard and Chrome-only, hence the local shape. */
interface ChromeMemory {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
}

const MB = 1024 * 1024;

function readMemory(): PerfMemoryFacts | null {
  try {
    if (typeof performance === "undefined") return null;
    const mem = (performance as Performance & { memory?: ChromeMemory }).memory;
    if (!mem || typeof mem.usedJSHeapSize !== "number") return null;
    return {
      usedMb: round(mem.usedJSHeapSize / MB, 1),
      totalMb: round(mem.totalJSHeapSize / MB, 1),
      limitMb: round(mem.jsHeapSizeLimit / MB, 1),
    };
  } catch {
    return null;
  }
}

function readDevice(): PerfDeviceFacts {
  const blank: PerfDeviceFacts = {
    userAgent: "",
    devicePixelRatio: 0,
    hardwareConcurrency: 0,
    screenWidth: 0,
    screenHeight: 0,
    deviceMemoryGb: null,
  };
  try {
    if (typeof navigator === "undefined" || typeof window === "undefined") return blank;
    const nav = navigator as Navigator & { deviceMemory?: number };
    return {
      userAgent: nav.userAgent ?? "",
      devicePixelRatio: round(window.devicePixelRatio ?? 0, 3),
      hardwareConcurrency: nav.hardwareConcurrency ?? 0,
      screenWidth: typeof screen !== "undefined" ? screen.width : 0,
      screenHeight: typeof screen !== "undefined" ? screen.height : 0,
      deviceMemoryGb: typeof nav.deviceMemory === "number" ? nav.deviceMemory : null,
    };
  } catch {
    return blank;
  }
}

/**
 * `longtask` is absent on Safari and on several Android WebViews. Reading
 * `supportedEntryTypes` is the only check that does not require constructing an observer and
 * catching the `TypeError` that `observe()` throws for an unknown type — and that throw is
 * not consistent across engines, so it cannot be relied on as the probe.
 */
function longTaskSupported(): boolean {
  try {
    if (typeof PerformanceObserver === "undefined") return false;
    const types = PerformanceObserver.supportedEntryTypes;
    return Array.isArray(types) && types.includes("longtask");
  } catch {
    return false;
  }
}

// -------------------------------------------------------------------------------------
// The recorder
// -------------------------------------------------------------------------------------

const DISABLED_SUMMARY_LABEL = "disabled";

class Recorder implements PerfTrace {
  readonly enabled: boolean;

  private readonly label: string;
  private readonly endpoint: string | null;
  private readonly flushOnHide: boolean;
  private readonly now: () => Millis;

  private readonly frames = new Ring(FRAME_CAPACITY);
  private readonly stageRings = new Map<string, Ring>();
  private stagesDropped = 0;

  private longTasks = 0;
  private longTaskTotalMs = 0;
  private longTaskMaxMs = 0;
  private readonly longTaskOk: boolean;
  private observer: PerformanceObserver | null = null;

  private rafHandle = 0;
  private lastFrameAt = 0;
  private running = false;
  private startedAtMs = 0;
  private endedAtMs = 0;
  private startedWallClock = new Date(0);

  private cameraSettings: Record<string, unknown> | null = null;
  private readonly notes: Record<string, string> = {};
  private hideHandler: (() => void) | null = null;

  constructor(options: PerfTraceOptions) {
    this.enabled = options.enabled ?? false;
    this.label = options.label ?? "capture";
    this.endpoint = options.endpoint ?? null;
    this.flushOnHide = options.flushOnHide ?? true;
    this.now =
      options.now ??
      (() => (typeof performance !== "undefined" ? performance.now() : Date.now()));
    this.longTaskOk = this.enabled && longTaskSupported();
  }

  start(): void {
    if (!this.enabled || this.running) return;
    try {
      this.running = true;
      this.startedAtMs = this.now();
      this.endedAtMs = this.startedAtMs;
      this.startedWallClock = new Date();
      this.lastFrameAt = 0;

      if (this.longTaskOk) {
        // `buffered: true` hands us the long tasks that happened during page load and
        // camera start-up, which is where the first stall lives and where no loop of ours
        // was running to observe it.
        this.observer = new PerformanceObserver((list) => this.onLongTasks(list));
        this.observer.observe({ type: "longtask", buffered: true } as PerformanceObserverInit);
      }
      this.scheduleFrame();
      if (this.flushOnHide) this.installHideFlush();
    } catch {
      // A tracer that takes the page down with it has negative value. Whatever failed here
      // simply means fewer numbers in the summary.
      this.running = false;
    }
  }

  stop(): void {
    if (!this.enabled) return;
    try {
      this.running = false;
      this.endedAtMs = this.now();
      if (this.rafHandle !== 0 && typeof cancelAnimationFrame === "function") {
        cancelAnimationFrame(this.rafHandle);
      }
      this.rafHandle = 0;
      this.observer?.disconnect();
      this.observer = null;
      this.removeHideFlush();
    } catch {
      this.observer = null;
    }
  }

  /**
   * Wrapped in a try/catch despite the body being three arithmetic operations and a typed
   * array write. The rule this module lives under is that it can never throw into the
   * capture loop, and a rule that holds only for the code as currently written is not a
   * rule. V8 has not penalised try/catch since TurboFan, so this costs nothing measurable.
   */
  stage(name: string, durationMs: number): void {
    if (!this.enabled) return;
    try {
      if (!Number.isFinite(durationMs)) return;
      let ring = this.stageRings.get(name);
      if (ring === undefined) {
        if (this.stageRings.size >= MAX_STAGES) {
          this.stagesDropped += 1;
          return;
        }
        ring = new Ring(STAGE_CAPACITY);
        this.stageRings.set(name, ring);
      }
      ring.push(durationMs);
    } catch {
      /* a lost sample is not worth a thrown error */
    }
  }

  measure<T>(name: string, fn: () => T): T {
    if (!this.enabled) return fn();
    const t0 = this.now();
    try {
      return fn();
    } finally {
      // In `finally` so a stage that threw still reports the time it burnt before throwing —
      // an exception thrown out of a readback is itself slow, and hiding that duration would
      // make the crashing case look cheap.
      this.stage(name, this.now() - t0);
    }
  }

  note(key: string, value: string): void {
    // First write wins and the key set is capped, for the same reason the stage names are:
    // a caller building keys in a loop would otherwise turn a fixed-size diagnostic into an
    // unbounded one, inside a capture.
    if (!this.enabled || key in this.notes || Object.keys(this.notes).length >= 16) return;
    this.notes[key] = String(value).slice(0, 200);
  }

  attachTrack(track: MediaStreamTrack | null | undefined): void {
    if (!this.enabled || this.cameraSettings !== null || !track) return;
    try {
      this.cameraSettings = { ...(track.getSettings() as Record<string, unknown>) };
    } catch {
      this.cameraSettings = null;
    }
  }

  raw(): PerfTraceRaw {
    const stages: PerfStageSamples[] = [];
    for (const [name, ring] of this.stageRings) {
      stages.push({ name, values: ring.snapshot(), dropped: ring.dropped });
    }
    return {
      label: this.label,
      startedAtMs: this.startedAtMs,
      endedAtMs: this.running ? this.now() : this.endedAtMs,
      wallClockIso: this.startedWallClock.toISOString(),
      frameIntervals: this.frames.snapshot(),
      frameCount: this.frames.count,
      frameDropped: this.frames.dropped,
      longTaskSupported: this.longTaskOk,
      longTaskCount: this.longTasks,
      longTaskTotalMs: this.longTaskTotalMs,
      longTaskMaxMs: this.longTaskMaxMs,
      stages,
      stagesDropped: this.stagesDropped,
      memory: readMemory(),
      cameraSettings: this.cameraSettings,
      notes: { ...this.notes },
      device: readDevice(),
    };
  }

  summary(): PerfSummary {
    return summarise(this.raw());
  }

  toText(): string {
    if (!this.enabled) return "trace off";
    try {
      return summaryToText(this.summary());
    } catch {
      return "trace unavailable";
    }
  }

  flush(endpoint?: string): PerfSummary | null {
    if (!this.enabled) return null;
    let payload: PerfSummary;
    try {
      payload = this.summary();
    } catch {
      return null;
    }
    const url = endpoint ?? this.endpoint;
    if (!url) return payload;
    try {
      const body = JSON.stringify(payload);
      // `sendBeacon` hands the bytes to the browser's own queue and returns immediately —
      // it does not need the document to survive, which is the whole reason it exists and
      // the reason a plain `fetch` loses exactly the traces we most want (the ones from a
      // session the user gave up on and closed).
      const nav = typeof navigator !== "undefined" ? navigator : null;
      if (nav && typeof nav.sendBeacon === "function") {
        const blob = new Blob([body], { type: "application/json" });
        if (nav.sendBeacon(url, blob)) return payload;
      }
      if (typeof fetch === "function") {
        // `keepalive` is the fetch equivalent, capped by the spec at 64 KB of body — a
        // summary is well under 4 KB. The promise is deliberately dropped on the floor:
        // awaiting it inside a capture is the blocking this module must not do.
        void fetch(url, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body,
          keepalive: true,
          // The endpoint is same-site in development and its reply is a 204 nobody reads.
          credentials: "omit",
        }).catch(() => undefined);
      }
    } catch {
      /* a trace that failed to ship must still not break the capture */
    }
    return payload;
  }

  // -- internals ---------------------------------------------------------------------

  private scheduleFrame(): void {
    if (typeof requestAnimationFrame !== "function") return;
    this.rafHandle = requestAnimationFrame((t) => this.onFrame(t));
  }

  /**
   * The only per-frame work in this module: one subtraction, one typed-array write, one
   * modulo. The timestamp comes from rAF's own argument rather than from `performance.now()`
   * so the sample is the frame's presentation time and not the time the callback happened
   * to be reached after everything else queued ahead of it.
   */
  private onFrame(timestamp: number): void {
    if (!this.running) return;
    if (this.lastFrameAt !== 0) this.frames.push(timestamp - this.lastFrameAt);
    this.lastFrameAt = timestamp;
    this.scheduleFrame();
  }

  private onLongTasks(list: PerformanceObserverEntryList): void {
    try {
      for (const entry of list.getEntries()) {
        const d = entry.duration;
        if (!Number.isFinite(d)) continue;
        this.longTasks += 1;
        this.longTaskTotalMs += d;
        if (d > this.longTaskMaxMs) this.longTaskMaxMs = d;
      }
    } catch {
      /* the observer callback runs on the main thread; it may not take the page down */
    }
  }

  /**
   * `visibilitychange` to hidden is the ONLY event Android Chrome guarantees before the tab
   * is discarded — `unload` and `beforeunload` are not fired on mobile when the user swipes
   * the app away, which is precisely how a lagging capture ends.
   */
  private installHideFlush(): void {
    if (typeof document === "undefined" || this.hideHandler !== null) return;
    this.hideHandler = () => {
      if (document.visibilityState === "hidden") this.flush();
    };
    document.addEventListener("visibilitychange", this.hideHandler);
    if (typeof window !== "undefined") window.addEventListener("pagehide", this.hideHandler);
  }

  private removeHideFlush(): void {
    if (typeof document === "undefined" || this.hideHandler === null) return;
    document.removeEventListener("visibilitychange", this.hideHandler);
    if (typeof window !== "undefined") window.removeEventListener("pagehide", this.hideHandler);
    this.hideHandler = null;
  }
}

/**
 * A disabled trace is a real object rather than `null`, so the caller never guards a call
 * site. Every method returns immediately on the `enabled` test.
 */
export function createPerfTrace(options: PerfTraceOptions = {}): PerfTrace {
  return new Recorder(options);
}

/**
 * The shared recorder.
 *
 * A module singleton rather than a prop, because the two places that must feed the SAME
 * trace are `useCamera`'s `requestVideoFrameCallback` (where the suspect readbacks live) and
 * the page that starts and ends the recording — and threading an option down through
 * `UseCameraOptions` would put a diagnostic into a production signature, where it would
 * outlive the investigation. `perfTrace()` returns a disabled no-op until `initPerfTrace`
 * runs, so a call site added to the loop is safe even on a build that never enables tracing.
 */
let shared: PerfTrace = new Recorder({ enabled: false });

/** Replace the shared recorder. Stops the previous one, so a remount cannot leak a rAF loop. */
export function initPerfTrace(options: PerfTraceOptions = {}): PerfTrace {
  shared.stop();
  shared = new Recorder(options);
  return shared;
}

export function perfTrace(): PerfTrace {
  return shared;
}

/** True when the URL carries `?debug=1` or `?trace=1`, which is how the phone opts in. */
export function traceRequested(search?: string): boolean {
  try {
    const query = search ?? (typeof location !== "undefined" ? location.search : "");
    if (!query) return false;
    const params = new URLSearchParams(query);
    return params.get("debug") === "1" || params.get("trace") === "1";
  } catch {
    return false;
  }
}

/**
 * `summarise` on an empty trace, so a caller can render the HUD before `start()` without a
 * null branch. The label makes an accidentally-shipped empty trace obvious in the log.
 */
export function emptySummary(): PerfSummary {
  return summarise({
    label: DISABLED_SUMMARY_LABEL,
    startedAtMs: 0,
    endedAtMs: 0,
    wallClockIso: new Date(0).toISOString(),
    frameIntervals: [],
    frameCount: 0,
    frameDropped: 0,
    longTaskSupported: false,
    longTaskCount: 0,
    longTaskTotalMs: 0,
    longTaskMaxMs: 0,
    stages: [],
    stagesDropped: 0,
    memory: null,
    cameraSettings: null,
    notes: {},
    device: readDevice(),
  });
}
