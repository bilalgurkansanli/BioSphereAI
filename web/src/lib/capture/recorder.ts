"use client";

/**
 * RECORDER — the one artefact the whole feature exists to produce.
 * =====================================================================================
 *
 * Four rules, each of them paid for by a measured failure:
 *
 *   1. ALWAYS start(timeslice). Without a timeslice everything buffers into a single Blob
 *      delivered at stop(), and if stop() never cleanly fires — an incoming call ends the
 *      track on iOS — the entire capture is gone. With a 1 s timeslice the loss is bounded
 *      at one second and every chunk can be flushed to IndexedDB as it lands.
 *   2. ALWAYS set videoBitsPerSecond. Chrome's default is 2.5 Mbps and does NOT scale with
 *      resolution, so a forgotten bitrate produces 1080p at roughly the bit budget of the
 *      measured WhatsApp re-encode (478x850 at 1816 kbps) that made a panorama unusable —
 *      with no error anywhere. 20 Mbps is the engine's 0.25 bpp floor at 1080x1920x30 plus
 *      margin; the ACHIEVED rate is measured from the bytes that actually arrived and is
 *      reported, never assumed.
 *   3. NEVER TRUST isTypeSupported(). It returning true has historically not guaranteed that
 *      construction or start() succeeds on iOS, so both are wrapped and the mime ladder is
 *      walked on throw. H.264 in MP4 is preferred on both platforms: it is the one format
 *      both produce, both hardware-encode, and the engine's ffmpeg path already accepts. VP9
 *      is software-encoded on phones and will thermally throttle over a 90 s capture.
 *   4. PROVE IT BEFORE THE USER INVESTS TIME. `preflight()` records ~200 ms into a throwaway
 *      recorder during the framing step and checks a non-empty Blob actually arrives.
 *
 * Recording is CONTINUOUS across rows. The pitch transitions between rows are the footage
 * that ties the rows together in the solver, so nothing here stops and restarts between them;
 * a segment is sealed only by a finish, an interruption, a rotation, an error, or the
 * 120 s ceiling.
 */

import { useCallback, useEffect, useState } from "react";
import type { Millis, RecordedSegment, RecorderApi } from "./types";
import { RECORDING } from "./types";
import { nowMs } from "./useOrientation";

// -------------------------------------------------------------------------------------
// Injectable media layer
// -------------------------------------------------------------------------------------

/**
 * The slice of MediaRecorder this module uses, expressed with callback registrars rather
 * than assignable `on*` properties so a fake can implement it without fighting the DOM's
 * event types. The default adapter below wraps the real thing.
 */
export interface MediaRecorderLike {
  readonly state: string;
  readonly mimeType: string;
  start: (timesliceMs?: number) => void;
  stop: () => void;
  pause: () => void;
  resume: () => void;
  /** Optional: some builds hold the first buffer until it is asked for explicitly. */
  requestData?: () => void;
  onData: (cb: (chunk: Blob) => void) => void;
  onStop: (cb: () => void) => void;
  onError: (cb: (error: unknown) => void) => void;
}

export interface RecorderConstructionOptions {
  mimeType?: string;
  videoBitsPerSecond?: number;
}

export type MediaRecorderFactory = (stream: MediaStream, options: RecorderConstructionOptions) => MediaRecorderLike;

const defaultFactory: MediaRecorderFactory = (stream, options) => {
  const recorder = new MediaRecorder(stream, options);
  return {
    get state() {
      return recorder.state;
    },
    get mimeType() {
      return recorder.mimeType;
    },
    start: (timesliceMs) => recorder.start(timesliceMs),
    stop: () => recorder.stop(),
    pause: () => recorder.pause(),
    resume: () => recorder.resume(),
    requestData: () => recorder.requestData(),
    onData: (cb) => {
      recorder.ondataavailable = (event: BlobEvent) => cb(event.data);
    },
    onStop: (cb) => {
      recorder.onstop = () => cb();
    },
    onError: (cb) => {
      recorder.onerror = (event: Event) => cb(event);
    },
  };
};

const defaultIsTypeSupported = (mime: string): boolean => {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") return false;
  try {
    return MediaRecorder.isTypeSupported(mime);
  } catch {
    return false;
  }
};

// -------------------------------------------------------------------------------------
// Controller
// -------------------------------------------------------------------------------------

export interface RecorderOptions {
  /** Live camera stream. May be null until the camera opens; `start()` refuses until then. */
  stream: MediaStream | null;
  /** Decoder-truth dimensions for the sealed segment, from `video.videoWidth/videoHeight`. */
  dimensions?: () => { width: number; height: number } | null;
  /** Every chunk as it arrives, for the 5 s IndexedDB draft flush. */
  onChunk?: (chunk: Blob, index: number) => void;
  /** A segment was sealed — including by an interruption this module handled on its own. */
  onSegment?: (segment: RecordedSegment) => void;
  onError?: (error: unknown) => void;
  /**
   * The silent-encoder watchdog tore a dead recorder down and is starting a fresh one
   * INSIDE the same segment. Every clock that was anchored to the first start is now wrong
   * by the restart: measured on a real capture, the telemetry track ran 7.2-7.75 s ahead of
   * the media for the whole first segment because nothing re-anchored it.
   */
  onRestart?: () => void;
  bitrateBps?: number;
  timesliceMs?: number;
  maxDurationMs?: number;
  factory?: MediaRecorderFactory;
  isTypeSupported?: (mime: string) => boolean;
  now?: () => Millis;
  /** How long one preflight rung may wait for its first byte. */
  trialMs?: number;
  /** Total time preflight may spend searching before it gives up and lets capture start. */
  preflightBudgetMs?: number;
}

export interface RecorderController extends RecorderApi {
  /** The stream arrives after the controller does — the camera opens on a click. */
  setStream: (stream: MediaStream | null) => void;
  /**
   * Re-point the late-bound parts (stream, dimensions, callbacks) without rebuilding the
   * recorder. A React caller passes fresh closures on every render; rebuilding mid-capture
   * would drop the chunks accumulated so far.
   */
  configure: (next: Partial<RecorderOptions>) => void;
  /** Chunks of the live segment, in arrival order. Not a copy: do not mutate. */
  chunks: () => Blob[];
  /** True when the platform honoured pause(); false means a landscape tilt keeps recording. */
  pauseSupported: () => boolean;
  /** Measured bitrate is below RECORDING.minAcceptableBitrateBps. */
  bitrateShortfall: () => boolean;
  subscribe: (listener: () => void) => () => void;
  dispose: () => void;
}

export function createRecorder(options: RecorderOptions): RecorderController {
  const factory = options.factory ?? defaultFactory;
  const isTypeSupported = options.isTypeSupported ?? defaultIsTypeSupported;
  const clock = options.now ?? nowMs;
  const timesliceMs = options.timesliceMs ?? RECORDING.timesliceMs;
  const maxDurationMs = options.maxDurationMs ?? RECORDING.maxDurationMs;
  const bitrateBps = options.bitrateBps ?? RECORDING.targetBitrateBps;

  let stream = options.stream;
  // Callbacks are read through this holder, never captured, so `configure` can replace them.
  let hooks: Pick<RecorderOptions, "dimensions" | "onChunk" | "onSegment" | "onError" | "onRestart"> = {
    dimensions: options.dimensions,
    onChunk: options.onChunk,
    onSegment: options.onSegment,
    onError: options.onError,
    onRestart: options.onRestart,
  };
  let recorder: MediaRecorderLike | null = null;
  let state: RecorderApi["state"] = "idle";
  let mimeType: string | null = null;
  let chunks: Blob[] = [];
  let bytes = 0;
  // Null rather than 0: an injected clock (and, on some platforms, performance.now()) can
  // legitimately read 0 at the moment recording starts, and `startedAt === 0` as a sentinel
  // would silently seal every segment with a duration of zero.
  let startedAt: Millis | null = null;
  let pausedAt: Millis | null = null;
  let pausedTotalMs = 0;
  let pauseWorks = true;
  let maxTimer: ReturnType<typeof setTimeout> | null = null;
  let sealing: Promise<RecordedSegment | null> | null = null;
  let sealReason: RecordedSegment["endedBy"] = "user_finish";
  let onStopResolve: ((segment: RecordedSegment | null) => void) | null = null;
  let endedListenerCleanup: (() => void) | null = null;

  const listeners = new Set<() => void>();
  const notify = () => {
    for (const listener of listeners) listener();
  };

  const elapsedMs = (): number => {
    if (startedAt === null) return 0;
    const frozen = pausedAt === null ? 0 : clock() - pausedAt;
    return Math.max(0, clock() - startedAt - pausedTotalMs - frozen);
  };

  const measuredBitrate = (): number => {
    const seconds = elapsedMs() / 1000;
    return seconds > 0.5 ? (bytes * 8) / seconds : 0;
  };

  /**
   * Every combination worth trying, in descending order of what we would prefer.
   *
   * Constructing successfully is NOT evidence that a combination works. Measured on a
   * Xiaomi/Android 10 Chrome 151 at 1920x1080: `isTypeSupported` returned true for all
   * eight codecs in the ladder, every one of them constructed without throwing, and every
   * one produced a **zero-byte** recording — while `new MediaRecorder(stream)` with no
   * options at all produced 25 KB in 400 ms. So the ladder has to vary the bitrate too,
   * and the only trustworthy test is whether bytes actually arrive.
   */
  type Combo = { mimeType?: string; bitrate?: number };

  const combos = (): Combo[] => {
    const supported = RECORDING.mimeLadder.filter((mime) => isTypeSupported(mime));
    const first = supported[0];
    const second = supported.find((m) => m !== first && m.split(";")[0] !== first?.split(";")[0]);

    // Deliberately short. The full cross-product of codecs and bitrates is 36 rungs, and
    // every rejected rung costs a quarter of a second of the user's time before the
    // capture can start — nine seconds on a device where nothing works at all. These seven
    // cover the axes that actually carry information: the preferred codec, a different
    // container in case that specific codec is broken, no codec at all, and at each step a
    // descending bitrate, because an encoder that cannot sustain the rate may emit nothing
    // rather than clamp.
    const ladder: Combo[] = [];
    if (first) {
      ladder.push({ mimeType: first, bitrate: bitrateBps });
      ladder.push({ mimeType: first, bitrate: Math.round(bitrateBps * 0.4) });
      ladder.push({ mimeType: first });
    }
    if (second) {
      ladder.push({ mimeType: second, bitrate: bitrateBps });
      ladder.push({ mimeType: second });
    }
    // The last two rungs are the ones that rescued the Android device above, and they must
    // always be present — including when the platform reports no supported codec at all.
    ladder.push({ bitrate: bitrateBps });
    ladder.push({});

    const seen = new Set<string>();
    return ladder.filter((c) => {
      const key = `${c.mimeType ?? "-"}|${c.bitrate ?? "-"}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  };

  /** Build a recorder for one combination. Returns null when the constructor throws. */
  const build = (target: MediaStream, combo: Combo) => {
    try {
      const opts: { mimeType?: string; videoBitsPerSecond?: number } = {};
      if (combo.mimeType) opts.mimeType = combo.mimeType;
      if (combo.bitrate) opts.videoBitsPerSecond = combo.bitrate;
      const built = factory(target, opts);
      return { recorder: built, mimeType: combo.mimeType ?? built.mimeType ?? null };
    } catch {
      return null;
    }
  };

  /**
   * The combination `preflight` proved actually emits bytes on this device. Pinned so the
   * real capture does not repeat the search — and so it cannot pick a different, untested
   * combination than the one that was verified.
   */
  let provenCombo: Combo | null = null;

  const construct = (target: MediaStream): { recorder: MediaRecorderLike; mimeType: string | null } | null => {
    if (provenCombo && !deadCombos.has(comboKey(provenCombo))) {
      const proven = build(target, provenCombo);
      if (proven) {
        activeCombo = provenCombo;
        return proven;
      }
    }
    for (const combo of combos()) {
      if (deadCombos.has(comboKey(combo))) continue;
      const built = build(target, combo);
      if (built) {
        activeCombo = combo;
        return built;
      }
    }
    return null;
  };

  const seal = (reason: RecordedSegment["endedBy"]): RecordedSegment | null => {
    if (chunks.length === 0) return null;
    const type = mimeType ?? recorder?.mimeType ?? "video/mp4";
    const durationMs = elapsedMs();
    const blob = new Blob(chunks, { type });
    const size = hooks.dimensions?.() ?? null;
    const segment: RecordedSegment = {
      blob,
      mimeType: type,
      durationMs,
      width: size?.width ?? 0,
      height: size?.height ?? 0,
      bitrateBps: durationMs > 0 ? (blob.size * 8) / (durationMs / 1000) : 0,
      startedAt: startedAt ?? 0,
      endedBy: reason,
    };
    return segment;
  };

  const clearMaxTimer = () => {
    if (maxTimer !== null) {
      clearTimeout(maxTimer);
      maxTimer = null;
    }
  };

  const finishSeal = () => {
    clearMaxTimer();
    clearNoDataWatchdog();
    endedListenerCleanup?.();
    endedListenerCleanup = null;
    const segment = seal(sealReason);
    state = "stopped";
    recorder = null;
    chunks = [];
    bytes = 0;
    startedAt = null;
    pausedAt = null;
    pausedTotalMs = 0;
    sealing = null;
    if (segment) hooks.onSegment?.(segment);
    onStopResolve?.(segment);
    onStopResolve = null;
    notify();
  };

  /**
   * The silent-encoder watchdog — the difference between losing a capture and not.
   *
   * `construct()` returns the first combination that *constructs*, which is not the same as
   * the first that *works*. Measured on Android 10 / Chrome 151: every codec in the ladder
   * constructed without complaint and emitted zero bytes, and the user turned through all
   * three rows, reached 91 % coverage, and was told at the very end that there was nothing
   * to upload. A minute of their effort, thrown away by a failure nothing was watching for.
   *
   * So recording is now verified rather than assumed: if no byte arrives within a couple of
   * seconds of starting, the combination is declared dead, retired, and the next one is
   * tried — silently, while the user keeps turning. They lose a second or two of footage at
   * the very start of the capture instead of all of it at the end.
   */
  const NO_DATA_TIMEOUT_MS = 2500;
  /**
   * And once bytes ARE flowing, the same watch stays armed on a much longer fuse.
   *
   * WHY IT EXISTS: `onData` used to CLEAR the watchdog on chunk one and nothing re-armed it,
   * so the watch covered only the first 2.5 seconds. An encoder that dies later — a phone
   * thermally throttling under a hardware H.264 encode, or a tab suspended and resumed —
   * then produces nothing for the rest of the sweep while `state` still reads "recording",
   * the clock keeps counting and the lattice keeps filling. The user turns through the whole
   * room and ends up with a two-second file.
   *
   * WHY IT IS THIS LONG, WHICH IS THE PART THAT WAS GOT WRONG. It was first set to a flat
   * 4000 ms — four timeslices — and that fired on a real capture: the phone was busy at the
   * start of recording, one chunk was more than four seconds late, the watchdog decided the
   * encoder had died, and it SEALED A 4-SECOND SEGMENT and opened a new one. The user filmed
   * for two minutes and got two files, the first of which was 4 seconds long. A watchdog
   * that trips on the jank it was meant to survive manufactures exactly the failure it was
   * added to prevent.
   *
   * `dataavailable` carries no timing guarantee whatsoever under load — the spec lets the UA
   * deliver a timesliced blob whenever it can — so any fuse close to the timeslice is a
   * jank detector wearing an encoder-failure costume. Twenty timeslices is far outside
   * anything a delivery hiccup produces and still catches a dead encoder inside half a
   * minute, against a capture ceiling of ten. Losing twenty seconds of detection latency
   * costs nothing; a spurious seal costs a segment and, until the engine's duration floor
   * was moved onto the capture total, the whole upload.
   */
  const STALL_TIMEOUT_MS = Math.max(20_000, timesliceMs * 20);
  let noDataTimer: ReturnType<typeof setTimeout> | null = null;
  /** True once this silence has already been probed with `requestData()`. */
  let stallProbed = false;
  /** Combinations already proven silent in THIS session, so the retry cannot loop on them. */
  const deadCombos = new Set<string>();
  let activeCombo: Combo | null = null;

  const comboKey = (c: Combo) => `${c.mimeType ?? "-"}|${c.bitrate ?? "-"}`;

  const clearNoDataWatchdog = () => {
    if (noDataTimer === null) return;
    clearTimeout(noDataTimer);
    noDataTimer = null;
  };

  const armNoDataWatchdog = (afterMs: number = NO_DATA_TIMEOUT_MS) => {
    clearNoDataWatchdog();
    noDataTimer = setTimeout(() => {
      noDataTimer = null;
      if (state !== "recording") return;
      // A stall AFTER bytes have arrived is a different failure from a silent codec. This
      // combination demonstrably works, so retiring the rung would be wrong; the encoder
      // simply stopped. Seal what exists — the stage opens the next segment on the spot, and
      // a capture is legitimately several segments — rather than throwing the footage away.
      if (bytes > 0) {
        // ASK BEFORE GIVING UP. Some builds hold the buffered frames until they are asked
        // for explicitly — the same behaviour `preflight` already works around — so a long
        // silence is not proof the encoder is dead, only that nothing has been flushed.
        // `requestData()` costs one call and, if the encoder is alive, produces a chunk
        // which re-arms this watch through `onData`. Only a second silence after being
        // asked directly is treated as a failure. Sealing a live segment is expensive: it
        // splits the capture, and the split is what the user sees.
        if (!stallProbed && typeof recorder?.requestData === "function") {
          stallProbed = true;
          try {
            recorder.requestData();
          } catch {
            /* not supported here; fall through to the seal on the next expiry */
          }
          armNoDataWatchdog(Math.max(4000, timesliceMs * 4));
          return;
        }
        hooks.onError?.(new Error("recorder: the encoder stopped emitting data"));
        void stop("error");
        return;
      }
      if (activeCombo) {
        deadCombos.add(comboKey(activeCombo));
        // A combination that produced nothing must never be pinned as proven.
        if (provenCombo && comboKey(provenCombo) === comboKey(activeCombo)) provenCombo = null;
      }
      const remaining = combos().filter((c) => !deadCombos.has(comboKey(c)));
      if (remaining.length === 0) {
        hooks.onError?.(new Error("recorder: every codec produced an empty stream"));
        void stop("error");
        return;
      }
      // Tear the dead recorder down without sealing — there is nothing in it to seal — and
      // start again on the next rung.
      //
      // Detach its callbacks FIRST. `stop()` fires `onstop` asynchronously on a real
      // MediaRecorder, so a handler left attached lands after the replacement is already
      // recording and seals the new session out from under it — turning the rescue into a
      // second, subtler way to lose the capture.
      const dead = recorder;
      if (dead) {
        dead.onData(() => undefined);
        dead.onStop(() => undefined);
        dead.onError(() => undefined);
      }
      try {
        dead?.stop();
      } catch {
        /* already gone */
      }
      recorder = null;
      state = "idle";
      startedAt = null;
      chunks = [];
      bytes = 0;
      notify();
      hooks.onRestart?.();
      void start();
    }, afterMs);
  };

  const start = async (): Promise<void> => {
    if (state === "recording" || state === "paused") return;
    const target = stream;
    if (!target) {
      state = "error";
      hooks.onError?.(new Error("recorder: no stream"));
      notify();
      return;
    }

    const built = construct(target);
    if (!built) {
      state = "error";
      hooks.onError?.(new Error("recorder: no usable mimeType"));
      notify();
      return;
    }

    recorder = built.recorder;
    mimeType = built.mimeType;
    chunks = [];
    bytes = 0;
    pausedTotalMs = 0;
    pausedAt = null;

    recorder.onData((chunk) => {
      if (chunk.size === 0) return;
      // Re-armed on the longer fuse, not cleared. See STALL_TIMEOUT_MS.
      stallProbed = false;
      armNoDataWatchdog(STALL_TIMEOUT_MS);

      // A COMBINATION THAT HAS ACTUALLY RECORDED IS PROVEN. `provenCombo` used to be written
      // only by `preflight`, so a codec that had just delivered a real segment was forgotten
      // the moment that segment sealed, and the next `start()` walked the ladder again from
      // the top. Measured on a real capture: segment 0 recorded 876 frames as hardware H.264
      // and segment 1 came back as software VP9 — the same phone, the same stream, one
      // segment boundary apart. The software encoder then delivered 6.3 fps of the 30 it was
      // asked for, which is what put motion blur into the frames the solver had to match.
      //
      // A first real chunk is stronger evidence than a 1.5 s preflight trial, so it pins the
      // same field. The watchdog still clears it if the combination later goes silent.
      if (activeCombo && provenCombo !== activeCombo) provenCombo = activeCombo;
      chunks.push(chunk);
      bytes += chunk.size;
      hooks.onChunk?.(chunk, chunks.length - 1);
      notify();
    });
    recorder.onStop(() => finishSeal());
    recorder.onError((error) => {
      hooks.onError?.(error);
      sealReason = "error";
      // The recorder may or may not emit `stop` after an error; seal from here if it does not.
      if (state === "recording" || state === "paused") {
        state = "error";
        try {
          recorder?.stop();
        } catch {
          finishSeal();
        }
      }
      notify();
    });

    try {
      // The timeslice is not optional. See rule 1.
      recorder.start(timesliceMs);
    } catch {
      recorder = null;
      state = "error";
      hooks.onError?.(new Error("recorder: start() threw"));
      notify();
      return;
    }

    startedAt = clock();
    state = "recording";
    stallProbed = false;
    armNoDataWatchdog();

    // An incoming call ends the track on iOS. The recorder frequently does NOT fire an error
    // for that, so the track is watched directly and the segment is sealed as `interrupted`
    // — which is exactly the case that makes a scene legitimately span several files.
    const tracks = target.getVideoTracks();
    const onEnded = () => {
      void stop("interrupted");
    };
    for (const track of tracks) track.addEventListener("ended", onEnded);
    endedListenerCleanup = () => {
      for (const track of tracks) track.removeEventListener("ended", onEnded);
    };

    clearMaxTimer();
    maxTimer = setTimeout(() => {
      void stop("max_duration");
    }, maxDurationMs);

    notify();
  };

  const stop = (reason: RecordedSegment["endedBy"]): Promise<RecordedSegment | null> => {
    if (sealing) return sealing;
    if (!recorder) {
      // Nothing running, but chunks may still be held from a recorder that died on its own.
      const segment = seal(reason);
      chunks = [];
      bytes = 0;
      state = segment ? "stopped" : state;
      if (segment) hooks.onSegment?.(segment);
      notify();
      return Promise.resolve(segment);
    }

    sealReason = reason;
    // THE PROMISE IS BUILT AND ASSIGNED BEFORE `stop()` IS ATTEMPTED, and the order is
    // load-bearing.
    //
    // `stop()` on a recorder the UA has already made inactive throws SYNCHRONOUSLY. When
    // that catch called `finishSeal()` from inside the Promise executor, the executor ran to
    // completion first — so `finishSeal`'s own `sealing = null` was immediately overwritten
    // by this assignment, leaving `sealing` holding an already-resolved promise for ever.
    // From then on the guard at the top of `stop()` short-circuits EVERY later seal: the
    // recorder silently refuses to close another segment, and the capture cannot be
    // uploaded at all.
    //
    // It was unreachable until today. The stall watchdog added this session calls
    // `stop("error")` exactly when the recorder is most likely already inactive — which is
    // exactly when `stop()` throws — so a latent latch became a live one.
    let settle: (segment: RecordedSegment | null) => void = () => undefined;
    const pending = new Promise<RecordedSegment | null>((resolve) => {
      settle = resolve;
    });
    onStopResolve = settle;
    sealing = pending;
    {
      try {
        recorder?.stop();
      } catch {
        finishSeal();
        return pending;
      }
      // Safari has been known to swallow `stop` when the track died first. Seal anyway rather
      // than leaving the caller awaiting a promise that never settles with a capture inside it.
      //
      // Detach the dying recorder first, for the same reason the watchdog does. Sealing here
      // means the browser was late, not that it was finished: a `dataavailable` or a `stop`
      // arriving after this timer would otherwise run against the NEXT session's `chunks` —
      // pushing the previous segment's final chunk into the new one, or calling `finishSeal`
      // a second time and sealing a recording that had only just begun. `stop()` on a
      // recorder that has already stopped throws, and that throw is the reason for the catch.
      setTimeout(() => {
        if (!onStopResolve) return;
        const late = recorder;
        if (late) {
          late.onData(() => undefined);
          late.onStop(() => undefined);
          late.onError(() => undefined);
        }
        finishSeal();
      }, 2000);
    }
    return pending;
  };

  const pause = () => {
    if (state !== "recording" || !recorder) return;
    try {
      recorder.pause();
      if (recorder.state !== "paused") throw new Error("pause ignored");
      pausedAt = clock();
      state = "paused";
      // A pause is a legitimate silence, so the stall watch must not count it — and it must
      // not simply expire either. Left armed it fires during the pause, sees `state` is not
      // "recording", returns without re-arming, and the protection added this session is
      // then dead for the rest of the capture. A rotation or the paused sheet would silently
      // disarm the very thing that catches a dead encoder.
      clearNoDataWatchdog();
    } catch {
      // No pause support (older Safari). Keeping the recording running is strictly better
      // than sealing a segment: the contract says a rotation must never end a capture, and
      // the engine drops the landscape frames anyway.
      pauseWorks = false;
    }
    notify();
  };

  const resume = () => {
    if (state !== "paused" || !recorder) return;
    try {
      recorder.resume();
      if (pausedAt !== null) pausedTotalMs += clock() - pausedAt;
      pausedAt = null;
      state = "recording";
      // Re-armed on the post-first-byte fuse if bytes have already flowed, on the cold one
      // if they have not: a capture paused before its first chunk is still an unproven codec.
      stallProbed = false;
      armNoDataWatchdog(bytes > 0 ? STALL_TIMEOUT_MS : NO_DATA_TIMEOUT_MS);
    } catch {
      pauseWorks = false;
    }
    notify();
  };

  /**
   * Record with one combination and report how many bytes actually arrived.
   *
   * The window is long on purpose. Measured on Android 10 / Chrome 151 at 1920x1080, the
   * same combination produced 25 KB on one run and nothing on the next within a 400 ms
   * window: the hardware encoder's first chunk simply does not always arrive that fast, so
   * a short trial is a coin flip rather than a capability test. An impatient preflight is
   * what made a phone that records perfectly well be told it cannot record at all.
   */
  const TRIAL_MS = options.trialMs ?? 1500;
  const PREFLIGHT_BUDGET_MS = options.preflightBudgetMs ?? 3200;

  const trial = (target: MediaStream, combo: Combo): Promise<{ bytes: number; mimeType: string | null }> => {
    const built = build(target, combo);
    if (!built) return Promise.resolve({ bytes: 0, mimeType: null });

    return new Promise((resolve) => {
      let size = 0;
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        try {
          built.recorder.stop();
        } catch {
          /* already stopped */
        }
        resolve({ bytes: size, mimeType: built.mimeType });
      };

      built.recorder.onData((chunk) => {
        size += chunk.size;
        // The moment any byte arrives the question is answered; do not keep the user
        // waiting out the rest of the window.
        if (size > 0) finish();
      });
      built.recorder.onStop(() => {
        if (settled) return;
        settled = true;
        resolve({ bytes: size, mimeType: built.mimeType });
      });
      built.recorder.onError(() => finish());

      try {
        built.recorder.start(250);
      } catch {
        resolve({ bytes: 0, mimeType: null });
        return;
      }
      // Ask for a chunk explicitly rather than only waiting for the timeslice: some builds
      // hold the first buffer until asked.
      setTimeout(() => {
        if (settled) return;
        try {
          built.recorder.requestData?.();
        } catch {
          /* not supported; the timeslice is the fallback */
        }
      }, TRIAL_MS * 0.6);
      setTimeout(finish, TRIAL_MS);
    });
  };

  /**
   * Find a combination that actually records, and pin it.
   *
   * This walks the whole matrix rather than testing only the first rung, because a device
   * can claim to support every codec, construct every one of them without complaint, and
   * still write zero bytes for all of them — measured on Android 10 / Chrome 151, where
   * only "no options at all" produced data. Testing one rung and giving up is what made
   * that phone report "this browser cannot record video" while it demonstrably could.
   *
   * Runs during framing, before the user has invested a minute of turning. The cost is
   * ~250 ms per rejected rung, and it stops at the first that works.
   */
  const preflight = async (): Promise<{ ok: boolean; mimeType: string | null }> => {
    const target = stream;
    if (!target) return { ok: false, mimeType: null };

    // Bounded by wall clock, not by rung count. The caller no longer *blocks* on a failed
    // preflight, so spending ten seconds proving a negative would only make the user wait
    // before being allowed to start anyway.
    const deadline = clock() + PREFLIGHT_BUDGET_MS;
    for (const combo of combos()) {
      const { bytes: got, mimeType: got_mime } = await trial(target, combo);
      if (got > 0) {
        provenCombo = combo;
        return { ok: true, mimeType: got_mime };
      }
      if (clock() >= deadline) break;
    }
    provenCombo = null;
    return { ok: false, mimeType: null };
  };

  return {
    get state() {
      return state;
    },
    get mimeType() {
      return mimeType;
    },
    get chunkCount() {
      return chunks.length;
    },
    get elapsedMs() {
      return elapsedMs();
    },
    get bitrateBps() {
      return measuredBitrate();
    },
    start,
    pause,
    resume,
    stop,
    preflight,
    setStream: (next: MediaStream | null) => {
      stream = next;
      // A different stream is a different encoder pipeline: a combination that was silent
      // on the old one deserves a fresh chance rather than being retired for the session.
      deadCombos.clear();
    },
    configure: (next: Partial<RecorderOptions>) => {
      if ("stream" in next) stream = next.stream ?? null;
      hooks = {
        onRestart: next.onRestart ?? hooks.onRestart,
        dimensions: next.dimensions ?? hooks.dimensions,
        onChunk: next.onChunk ?? hooks.onChunk,
        onSegment: next.onSegment ?? hooks.onSegment,
        onError: next.onError ?? hooks.onError,
      };
    },
    chunks: () => chunks,
    pauseSupported: () => pauseWorks,
    bitrateShortfall: () => {
      const measured = measuredBitrate();
      return measured > 0 && measured < RECORDING.minAcceptableBitrateBps;
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    dispose: () => {
      clearMaxTimer();
      endedListenerCleanup?.();
      endedListenerCleanup = null;
      if (recorder && (state === "recording" || state === "paused")) {
        try {
          recorder.stop();
        } catch {
          // Already gone; the chunks are still in hand and the caller owns them.
        }
      }
      listeners.clear();
      recorder = null;
      state = "idle";
    },
  };
}

// -------------------------------------------------------------------------------------
// React wrapper
// -------------------------------------------------------------------------------------

export interface UseRecorderResult extends RecorderApi {
  pauseSupported: boolean;
  bitrateShortfall: boolean;
  /** Chunks of the live segment, for the draft flush. */
  chunks: () => Blob[];
  /**
   * The controller itself, for hot paths. `elapsedMs` and `bitrateBps` on this object are
   * snapshots taken at the last re-render (one per timeslice, i.e. 1 Hz); a smooth on-screen
   * timer should read `controller.elapsedMs` from a rAF loop instead of re-rendering at 60 Hz.
   */
  controller: RecorderController;
}

/**
 * Thin React mirror over `createRecorder`.
 *
 * The controller holds the truth and every field below is a live getter onto it, so a caller
 * that keeps the returned object in a ref (`useLatest`) reads the real state the instant it
 * changes rather than one render later — which matters, because the page checks
 * `recorder.state !== "recording"` immediately after awaiting `start()`. The React state here
 * exists only to force the re-render.
 *
 * Accepts the stream on its own for the common case, or the full options object.
 */
export function useRecorder(input: RecorderOptions | MediaStream | null): UseRecorderResult {
  const options: RecorderOptions = normaliseRecorderInput(input);

  // Built once, in state rather than a ref: a ref cannot be lazily initialised during render,
  // and rebuilding the recorder mid-capture would drop every chunk accumulated so far.
  const [controller] = useState(() => createRecorder(options));

  // Fresh closures arrive on every render; they are handed to the controller from an effect,
  // which is the only place a mutation like this is legal.
  useEffect(() => {
    controller.configure(options);
  });

  const [, bumpVersion] = useState(0);

  useEffect(() => {
    const unsubscribe = controller.subscribe(() => bumpVersion((n) => n + 1));
    return unsubscribe;
  }, [controller]);

  useEffect(() => () => controller.dispose(), [controller]);

  const start = useCallback(() => controller.start(), [controller]);
  const stop = useCallback((reason: RecordedSegment["endedBy"]) => controller.stop(reason), [controller]);
  const pause = useCallback(() => controller.pause(), [controller]);
  const resume = useCallback(() => controller.resume(), [controller]);
  const preflight = useCallback(() => controller.preflight(), [controller]);
  const chunks = useCallback(() => controller.chunks(), [controller]);

  return {
    get state() {
      return controller.state;
    },
    get mimeType() {
      return controller.mimeType;
    },
    get chunkCount() {
      return controller.chunkCount;
    },
    get elapsedMs() {
      return controller.elapsedMs;
    },
    get bitrateBps() {
      return controller.bitrateBps;
    },
    get pauseSupported() {
      return controller.pauseSupported();
    },
    get bitrateShortfall() {
      return controller.bitrateShortfall();
    },
    start,
    pause,
    resume,
    stop,
    preflight,
    chunks,
    controller,
  };
}

/** `useRecorder(stream)` is the common case; the full options object is the general one. */
function normaliseRecorderInput(input: RecorderOptions | MediaStream | null): RecorderOptions {
  if (input === null) return { stream: null };
  // Structural rather than `instanceof MediaStream`, which does not exist during SSR.
  if (typeof (input as MediaStream).getTracks === "function") return { stream: input as MediaStream };
  return input as RecorderOptions;
}
