import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createRecorder, type MediaRecorderLike, type RecorderController } from "@/lib/capture/recorder";
import { RECORDING, type RecordedSegment } from "@/lib/capture/types";

/**
 * THE RECORDER'S DELIVERY CONTRACT.
 *
 * `finishSeal` hands a sealed segment to `onSegment` and to the promise its internal caller
 * is awaiting, and to nothing else. Every one of the recorder's OWN reasons for stopping —
 * the 120 s ceiling, a track ending because a call came in, an encoder error — takes the
 * first path only. So an unwired `onSegment` is not a missing notification: it is the whole
 * recording on the floor, with `state` reading a tidy "stopped" and nothing on screen
 * changed. These tests exist because that is a silent failure, and a silent failure that
 * costs a user a minute of turning in a stranger's living room.
 */

// -------------------------------------------------------------------------------------
// Fakes
// -------------------------------------------------------------------------------------

class FakeMediaRecorder implements MediaRecorderLike {
  state = "inactive";
  readonly mimeType: string;
  private data: ((chunk: Blob) => void) | null = null;
  private stopped: (() => void) | null = null;
  private errored: ((error: unknown) => void) | null = null;
  /** Set to make `start()` throw, the way iOS does after `isTypeSupported` lied. */
  static failNextStart = false;
  /** Set to make `stop()` throw, the way a UA does when it already made the recorder
   *  inactive — the state the stall watchdog is most likely to find it in. */
  static failNextStop = false;
  /** Every `requestData()` the controller made — the stall watchdog's probe is one. */
  dataRequests = 0;

  constructor(mimeType: string) {
    this.mimeType = mimeType;
  }

  start(): void {
    if (FakeMediaRecorder.failNextStart) {
      FakeMediaRecorder.failNextStart = false;
      throw new Error("start() threw");
    }
    this.state = "recording";
  }

  stop(): void {
    if (FakeMediaRecorder.failNextStop) {
      FakeMediaRecorder.failNextStop = false;
      throw new Error("stop() on an inactive recorder");
    }
    this.state = "inactive";
    // Real MediaRecorders fire `stop` asynchronously; queueing it is what makes the
    // "seal, then immediately reopen" path a genuine re-entrancy test rather than a
    // straight-line call.
    queueMicrotask(() => this.stopped?.());
  }

  /** Real recorders expose this; a silent encoder simply does not answer it. */
  requestData(): void {
    this.dataRequests += 1;
  }

  pause(): void {
    this.state = "paused";
  }

  resume(): void {
    this.state = "recording";
  }

  onData(cb: (chunk: Blob) => void): void {
    this.data = cb;
  }

  onStop(cb: () => void): void {
    this.stopped = cb;
  }

  onError(cb: (error: unknown) => void): void {
    this.errored = cb;
  }

  /** Test hook: deliver one timeslice's worth of bytes. */
  emit(bytes = 1024): void {
    this.data?.(new Blob([new Uint8Array(bytes)], { type: this.mimeType }));
  }

  /** Test hook: the encoder fell over. */
  fail(): void {
    this.errored?.(new Error("encoder died"));
  }
}

interface FakeTrack {
  readyState: string;
  listeners: Map<string, Set<() => void>>;
  addEventListener: (type: string, cb: () => void) => void;
  removeEventListener: (type: string, cb: () => void) => void;
  end: () => void;
}

function fakeTrack(): FakeTrack {
  const listeners = new Map<string, Set<() => void>>();
  return {
    readyState: "live",
    listeners,
    addEventListener(type, cb) {
      const set = listeners.get(type) ?? new Set<() => void>();
      set.add(cb);
      listeners.set(type, set);
    },
    removeEventListener(type, cb) {
      listeners.get(type)?.delete(cb);
    },
    end() {
      this.readyState = "ended";
      for (const cb of listeners.get("ended") ?? []) cb();
    },
  };
}

function fakeStream(track: FakeTrack): MediaStream {
  const stream = {
    getTracks: () => [track],
    getVideoTracks: () => [track],
  };
  return stream as unknown as MediaStream;
}

interface Harness {
  controller: RecorderController;
  segments: RecordedSegment[];
  chunks: number;
  errors: number;
  track: FakeTrack;
  live: () => FakeMediaRecorder;
  now: (t: number) => void;
}

function harness(options: { dimensions?: () => { width: number; height: number } | null } = {}): Harness {
  const built: FakeMediaRecorder[] = [];
  const track = fakeTrack();
  let clock = 0;

  const state = {
    segments: [] as RecordedSegment[],
    chunks: 0,
    errors: 0,
  };

  const controller = createRecorder({
    stream: fakeStream(track),
    dimensions: options.dimensions ?? (() => ({ width: 1080, height: 1920 })),
    onSegment: (segment) => {
      state.segments.push(segment);
    },
    onChunk: () => {
      state.chunks += 1;
    },
    onError: () => {
      state.errors += 1;
    },
    factory: (_stream, opts) => {
      const recorder = new FakeMediaRecorder(opts.mimeType ?? "video/mp4");
      built.push(recorder);
      return recorder;
    },
    isTypeSupported: (mime) => mime === RECORDING.mimeLadder[0],
    now: () => clock,
  });

  return {
    controller,
    get segments() {
      return state.segments;
    },
    get chunks() {
      return state.chunks;
    },
    get errors() {
      return state.errors;
    },
    track,
    live: () => built[built.length - 1],
    now: (t: number) => {
      clock = t;
    },
  };
}

// -------------------------------------------------------------------------------------

beforeEach(() => {
  vi.useFakeTimers();
  FakeMediaRecorder.failNextStart = false;
});

afterEach(() => {
  vi.useRealTimers();
});

describe("segment delivery", () => {
  it("delivers a sealed segment through onSegment, exactly once", async () => {
    const h = harness();
    await h.controller.start();
    h.live().emit();
    h.now(5000);

    const returned = await h.controller.stop("user_finish");
    await vi.advanceTimersByTimeAsync(0);

    expect(h.segments).toHaveLength(1);
    // Same object down both paths: a caller that pushes the return value AND collects
    // through the hook would double every segment in the upload.
    expect(returned).toBe(h.segments[0]);
    expect(h.segments[0].endedBy).toBe("user_finish");
    expect(h.segments[0].durationMs).toBe(5000);
  });

  it("carries the decoder's dimensions, so the sidecar is not 0x0", () => {
    // The field the engine is documented to assert the decoded orientation against and
    // "reject loudly rather than solve 69 deg across the short edge". With `dimensions`
    // unwired it is 0, and `portrait` then falls through to its `true` default — an
    // assertion against nothing.
    const h = harness();
    void h.controller.start();
    h.live().emit();
    h.now(2000);
    void h.controller.stop("user_finish");
    return vi.advanceTimersByTimeAsync(0).then(() => {
      expect(h.segments[0].width).toBe(1080);
      expect(h.segments[0].height).toBe(1920);
      expect(h.segments[0].height > h.segments[0].width).toBe(true);
    });
  });

  it("reports 0x0 honestly when the decoder has not spoken yet", async () => {
    // Not a failure to fix here — the camera genuinely does not know the dimensions until
    // the element reports them. The caller's job is to prefer a segment that DOES know;
    // `plan.recordedFrameSize` is tested for exactly that.
    const h = harness({ dimensions: () => null });
    await h.controller.start();
    h.live().emit();
    await h.controller.stop("user_finish");
    await vi.advanceTimersByTimeAsync(0);
    expect(h.segments[0].width).toBe(0);
  });

  it("reports every chunk as it lands, for the live byte count", async () => {
    const h = harness();
    await h.controller.start();
    h.live().emit();
    h.live().emit();
    h.live().emit();
    expect(h.chunks).toBe(3);
    expect(h.controller.chunkCount).toBe(3);
  });

  it("seals nothing at all when no bytes ever arrived", async () => {
    const h = harness();
    await h.controller.start();
    const sealed = await h.controller.stop("user_finish");
    await vi.advanceTimersByTimeAsync(0);
    expect(sealed).toBeNull();
    expect(h.segments).toHaveLength(0);
  });
});

/**
 * Run the live recorder for `durationMs`, delivering a chunk every timeslice the way a real
 * MediaRecorder does. The last `size` wins, so a caller can assert on the sealed blob.
 */
async function runFor(h: ReturnType<typeof harness>, durationMs: number, size = 1024) {
  const step = RECORDING.timesliceMs;
  for (let t = step; t <= durationMs; t += step) {
    h.now(t);
    h.live().emit(size);
    await vi.advanceTimersByTimeAsync(step);
  }
}

describe("the 120-second per-segment ceiling", () => {
  it("hands the segment over instead of dropping it on the floor", async () => {
    const h = harness();
    await h.controller.start();
    // Fed on the recorder's real cadence rather than one chunk and a jump to the ceiling:
    // the stall watchdog is armed the whole time a capture is running, and a two-minute
    // silence IS the failure it exists to catch.
    await runFor(h, RECORDING.maxDurationMs, 4096);
    await vi.advanceTimersByTimeAsync(0);

    expect(h.segments).toHaveLength(1);
    expect(h.segments[0].endedBy).toBe("max_duration");
    // Everything recorded up to the ceiling, not just the last chunk.
    expect(h.segments[0].blob.size).toBe(4096 * (RECORDING.maxDurationMs / RECORDING.timesliceMs));
  });

  it("lets the caller reopen from inside the seal, so the capture crosses the boundary", async () => {
    // This is the shape `CaptureStage` uses: `onSegment` pushes and immediately calls
    // `start()` again. The seal must have finished tearing down the old recorder before the
    // hook fires, or the fresh one is clobbered by the tail of `finishSeal`.
    const h = harness();
    const collected: RecordedSegment[] = [];
    h.controller.configure({
      onSegment: (segment) => {
        collected.push(segment);
        if (segment.endedBy === "max_duration") void h.controller.start();
      },
    });

    await h.controller.start();
    await runFor(h, RECORDING.maxDurationMs);
    await vi.advanceTimersByTimeAsync(0);

    expect(collected).toHaveLength(1);
    expect(h.controller.state).toBe("recording");

    // ...and the second segment is a real one, with its own bytes and its own ceiling.
    h.live().emit(2048);
    h.now(RECORDING.maxDurationMs + 3000);
    await h.controller.stop("user_finish");
    await vi.advanceTimersByTimeAsync(0);
    expect(collected).toHaveLength(2);
    expect(collected[1].blob.size).toBe(2048);
    expect(collected[1].endedBy).toBe("user_finish");
  });
});

describe("an incoming call", () => {
  it("seals the partial segment as `interrupted` when the track ends", async () => {
    const h = harness();
    await h.controller.start();
    h.live().emit(8192);
    h.now(30_000);

    h.track.end();
    await vi.advanceTimersByTimeAsync(0);

    expect(h.segments).toHaveLength(1);
    expect(h.segments[0].endedBy).toBe("interrupted");
    expect(h.segments[0].durationMs).toBe(30_000);
    expect(h.segments[0].blob.size).toBe(8192);
  });

  it("stops listening to a track it has already sealed against", async () => {
    const h = harness();
    await h.controller.start();
    h.live().emit();
    h.track.end();
    await vi.advanceTimersByTimeAsync(0);
    expect(h.segments).toHaveLength(1);

    // A second `ended` on the same track (Safari has fired duplicates) must not seal twice.
    h.track.end();
    await vi.advanceTimersByTimeAsync(0);
    expect(h.segments).toHaveLength(1);
  });
  it("abandons a codec that constructs but never emits, and keeps recording on the next one", async () => {
    // The exact Android 10 / Chrome 151 failure: `isTypeSupported` says yes to everything,
    // every combination constructs happily, and the explicit ones emit nothing. Without a
    // watchdog the user turns through all three rows, reaches 91% coverage and is told at
    // the very end that there is no recording to upload. That is what happened on a real
    // phone, and it cost a real minute of turning.
    vi.useFakeTimers();
    const built: Array<{ mimeType?: string; videoBitsPerSecond?: number }> = [];
    let latest: FakeMediaRecorder | null = null;
    const track = fakeTrack();

    const controller = createRecorder({
      stream: fakeStream(track),
      isTypeSupported: () => true,
      factory: (_s, opts) => {
        built.push({ ...opts });
        const rec = new FakeMediaRecorder(opts?.mimeType ?? "video/mp4");
        latest = rec;
        return rec;
      },
    });

    await controller.start();
    expect(built.length).toBe(1);
    expect(built[0].mimeType).toBeDefined();

    // Not one byte arrives. The watchdog retires that combination and opens the next.
    await vi.advanceTimersByTimeAsync(2600);
    expect(built.length).toBe(2);
    expect(controller.state).toBe("recording");

    // The new one works, and the session carries on with no seal and no error.
    latest!.emit(4096);
    expect(controller.chunkCount).toBe(1);

    // And it stays put: a combination that is producing data is never retired.
    await vi.advanceTimersByTimeAsync(3000);
    expect(built.length).toBe(2);

    vi.useRealTimers();
  });

});

describe("an encoder that stops halfway", () => {
  it("does NOT split the capture because the phone was briefly busy", async () => {
    // THE REGRESSION THIS PINS. The stall fuse was first a flat 4000 ms — four timeslices —
    // and it fired on a real capture: the phone was loaded at the start of recording, one
    // chunk arrived late, the watchdog decided the encoder had died and sealed a 4-second
    // segment. The user filmed for two minutes and got two files, the first four seconds
    // long, and the upload was then refused for it. `dataavailable` carries no timing
    // guarantee under load, so a gap of several seconds is jank, not death.
    const h = harness();
    const errors: unknown[] = [];
    h.controller.configure({ onError: (error) => errors.push(error) });

    await h.controller.start();
    await runFor(h, 3000, 4096);

    // Six seconds of silence — well past the old fuse — then the chunks resume.
    h.now(3000 + 6000);
    await vi.advanceTimersByTimeAsync(6000);
    h.live().emit(4096);
    await vi.advanceTimersByTimeAsync(0);

    expect(errors).toHaveLength(0);
    expect(h.segments).toHaveLength(0);
    expect(h.controller.state).toBe("recording");
  });

  it("asks the encoder for data before concluding it is dead", async () => {
    const h = harness();
    const errors: unknown[] = [];
    h.controller.configure({ onError: (error) => errors.push(error) });

    await h.controller.start();
    await runFor(h, 3000, 4096);

    // Just past the real fuse, but inside the grace the probe buys: it must ask, not seal.
    h.now(3000 + 21_000);
    await vi.advanceTimersByTimeAsync(21_000);
    expect(h.segments).toHaveLength(0);
    expect(h.live().dataRequests).toBe(1);

    // ...and a live encoder answers the probe, which re-arms the watch.
    h.live().emit(4096);
    await vi.advanceTimersByTimeAsync(0);
    expect(errors).toHaveLength(0);
    expect(h.controller.state).toBe("recording");
  });

  it("seals what it has when the encoder never answers", async () => {
    const h = harness();
    const errors: unknown[] = [];
    h.controller.configure({ onError: (error) => errors.push(error) });

    await h.controller.start();
    await runFor(h, 8000, 4096);
    expect(h.controller.state).toBe("recording");

    // Silence through the fuse AND through the grace the probe buys.
    h.now(8000 + 40_000);
    await vi.advanceTimersByTimeAsync(40_000);
    await vi.advanceTimersByTimeAsync(0);

    expect(errors).toHaveLength(1);
    expect(h.segments).toHaveLength(1);
    // The footage that did arrive is kept — the stage opens the next segment on the spot.
    expect(h.segments[0].blob.size).toBe(4096 * (8000 / RECORDING.timesliceMs));
  });

  it("leaves a healthy recording alone for as long as the chunks keep coming", async () => {
    const h = harness();
    const errors: unknown[] = [];
    h.controller.configure({ onError: (error) => errors.push(error) });

    await h.controller.start();
    await runFor(h, 30_000, 2048);

    expect(errors).toHaveLength(0);
    expect(h.controller.state).toBe("recording");
    expect(h.segments).toHaveLength(0);
  });
});

describe("a stop() that throws", () => {
  it("does not latch the controller into never sealing again", async () => {
    // THE BLOCKER THIS PINS. `stop()` on a recorder the UA has already made inactive throws
    // synchronously. The catch used to call `finishSeal()` from inside the Promise executor,
    // so its `sealing = null` was overwritten by the outer `sealing = <promise>` assignment
    // the moment the executor returned. `stop()`'s own `if (sealing) return sealing;` then
    // short-circuited EVERY later seal: the recorder silently refused to close another
    // segment and the capture could never be uploaded.
    //
    // It was unreachable until the stall watchdog started calling `stop()` at exactly the
    // moment the recorder is most likely to be inactive.
    const h = harness();
    await h.controller.start();
    await runFor(h, 3000, 4096);

    FakeMediaRecorder.failNextStop = true;
    const first = await h.controller.stop("error");
    await vi.advanceTimersByTimeAsync(0);
    expect(first).not.toBeNull();
    expect(h.segments).toHaveLength(1);
    expect(h.controller.state).toBe("stopped");

    // ...and the controller is still usable: a second capture starts, records and seals.
    await h.controller.start();
    await runFor(h, 2000, 2048);
    const second = await h.controller.stop("user_finish");
    await vi.advanceTimersByTimeAsync(0);

    expect(second).not.toBeNull();
    expect(h.segments).toHaveLength(2);
    expect(h.segments[1].endedBy).toBe("user_finish");
    expect(h.segments[1].blob.size).toBe(2048 * (2000 / RECORDING.timesliceMs));
  });
});

describe("a pause in the middle of a capture", () => {
  it("does not disarm the stall watch for the rest of the recording", async () => {
    // A portrait rotation and the paused sheet both pause the recorder. Left armed across
    // the pause, the stall fuse expired, its callback saw `state !== "recording"` and
    // returned WITHOUT re-arming — so the encoder-death protection added this session was
    // silently dead for the remainder of the capture.
    const h = harness();
    const errors: unknown[] = [];
    h.controller.configure({ onError: (error) => errors.push(error) });

    await h.controller.start();
    await runFor(h, 3000, 4096);

    h.controller.pause();
    h.now(3000 + 40_000);
    await vi.advanceTimersByTimeAsync(40_000);
    // A pause is a legitimate silence: nothing is sealed and nothing is reported.
    expect(errors).toHaveLength(0);
    expect(h.segments).toHaveLength(0);

    h.controller.resume();
    expect(h.controller.state).toBe("recording");

    // ...and the watch is live again: a genuine stall after resuming is still caught.
    h.now(43_000 + 40_000);
    await vi.advanceTimersByTimeAsync(40_000);
    await vi.advanceTimersByTimeAsync(0);
    expect(errors).toHaveLength(1);
    expect(h.segments).toHaveLength(1);
  });
});

describe("the codec that actually recorded", () => {
  it("is reused for the next segment instead of walking the ladder again", async () => {
    // MEASURED ON A REAL CAPTURE: segment 0 recorded 876 frames as hardware H.264 and
    // segment 1 came back as software VP9 — same phone, same stream, one segment boundary
    // apart. `provenCombo` was written only by `preflight`, so a codec that had just
    // delivered a whole segment was forgotten the moment it sealed. The software encoder
    // then managed 6.3 fps of the 30 it was asked for, and that motion blur is what the
    // solver had to find features in.
    const h = harness();
    await h.controller.start();
    await runFor(h, 3000, 4096);
    const first = h.controller.mimeType;
    expect(first).not.toBeNull();

    await h.controller.stop("max_duration");
    await vi.advanceTimersByTimeAsync(0);

    await h.controller.start();
    await runFor(h, 1000, 4096);
    expect(h.controller.mimeType).toBe(first);
  });
});
