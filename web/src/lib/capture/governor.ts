/**
 * BioSphere AI — guided capture: the load governor.
 * =====================================================================================
 *
 * Measures the page's own frame rate and sheds capture work until it is smooth again.
 *
 * Pure: no React, no DOM, no `three`, and — deliberately — no timer of its own. The caller
 * hands it every animation-frame timestamp and reads back a plan. A governor that owned a
 * `setInterval` would be one more thing waking a phone that is already late.
 *
 * -------------------------------------------------------------------------------------
 * WHY THIS EXISTS AT ALL
 * -------------------------------------------------------------------------------------
 * Two rounds of targeted optimisation shipped against this capture and the phone is still
 * unusable. Each round picked a suspect, measured it ON DESKTOP, and fixed it: the loop was
 * gated to 33 Hz, the hot state left React, the backdrop filters came off the live camera,
 * the thumbnails dropped to 480 px, the sensor hooks stopped mirroring into state. Every one
 * of those was a real cost and none of them was THE cost, because the machine the numbers
 * came from was not the machine that is slow. A mid-range Android decoding 1080x1920 @ 30,
 * running a hardware H.264 encode at 12 Mbps, compositing a full-screen SVG over the video
 * and servicing a 33 Hz JS loop does not have a single hot spot to find — it has a budget it
 * is over, and the shape of the overrun differs per device, per thermal state and per frame.
 *
 * So this module does not know what is expensive. It only knows whether the phone is
 * keeping up, and it turns things down until it is. Whatever the real cost turns out to be,
 * it is downstream of one of the six knobs in `LoadPlan`, and the governor will find the
 * setting that fits inside the budget without anyone naming the culprit first.
 *
 * -------------------------------------------------------------------------------------
 * THE RULE THAT SHAPES EVERY NUMBER BELOW
 * -------------------------------------------------------------------------------------
 * A tier that stops the capture from working is not a tier, it is a bug. The capture has
 * to come back with: 22 of 24 stations per row, >= 360 deg of yaw span, a coverage grid with
 * no holes at 5 deg, and an orientation track the reconstruction can align to the video.
 * Every value in `LOAD_PLANS` is justified against those four, and the floors — 5 Hz stamps,
 * 15 Hz publish, 12 drawn glyphs — are where the next step down would start costing one of
 * them. Below those floors the honest answer is not a lower tier, it is a capture this
 * device cannot do.
 */

import type { Millis } from "@/lib/capture/types";

// ── The plan ───────────────────────────────────────────────────────────────────────────

export type LoadTier = 0 | 1 | 2 | 3 | 4;

export interface LoadPlan {
  tier: LoadTier;
  /** Quadrant phase correlation in `useCamera`. Tier >= 1 turns it off. */
  opticalFlow: boolean;
  /** Frame-statistics rate in `useCamera`: luma, clipping, exposure, predicted smear. */
  qualityHz: number;
  /** Hot-store publish rate, which bounds how far the world-locked overlay lags the room. */
  publishHz: number;
  /** Coverage-cone stamp rate, which bounds how finely the sphere can be filled. */
  stampHz: number;
  /** How many station glyphs `GuidanceOverlay` may draw, nearest the view axis first. */
  latticeStations: number;
  /** Tier >= 3 stops grabbing review stills. */
  thumbnails: boolean;
}

/**
 * The five settings, and what each step actually buys.
 *
 * Indexed by tier, frozen, and returned BY IDENTITY: a consumer can compare the plan it was
 * handed against the one it used last with `!==` and skip its own reconfiguration entirely.
 * That matters for `useCamera`, whose analysis effect tears down the video-frame callback —
 * and with it the previous flow frame — if it is rebuilt.
 *
 * ── TIER 1 — the two synchronous readbacks become one ────────────────────────────────
 * `opticalFlow` off. `analyseFlow` runs 5 times a second and each run is a `drawImage`
 * followed by `getImageData` on a 192x256 canvas plus a 49,152-cell greyscale loop. The
 * pixel COUNT is trivial, which is why earlier passes dismissed it; the cost is that
 * `getImageData` is a synchronous GPU->CPU readback that forces the pipeline to drain while
 * a hardware encoder is mid-frame. Together with `analyseQuality` at 10 Hz that is fifteen
 * pipeline stalls a second, and this halves them at the cost of nothing the capture must
 * measure: flow feeds only `flowDivergence`, an advisory that says "you are orbiting rather
 * than pivoting". No row completes or fails on it, no station fills from it, and the same
 * geometry is recovered offline by the reconstruction anyway. It is the largest saving
 * available for the smallest loss, so it is the first thing to go and it goes alone.
 *
 * ── TIER 2 — nothing the user can see changes ─────────────────────────────────────────
 * `qualityHz` 10 -> 4. The quality sample feeds the dark, clipped, over-exposed and smear
 * detectors, all of which require a condition to HOLD before they fire — none of them
 * triggers on a single sample. Four samples a second still fills every detector window.
 * `measuredFps` is unaffected: it is derived from `frameTimes`, which is pushed on every
 * video frame regardless of this rate.
 * `publishHz` 30 -> 24. At the coached 24 deg/s the world-locked overlay now trails the
 * room by 1.0 deg instead of 0.8 deg. The bright ring's capture radius is several degrees
 * and the station lattice is 15 deg apart, so this is a fourteenth of a station spacing.
 * `latticeStations` 72 -> 32. The overlay keeps a `<g>` per station and writes a transform
 * to each one on every publish. Thirty-two is more than the active row holds (24), so every
 * glyph that could be on screen still is; what stops being written are the two rows the
 * user is not being coached through.
 *
 * ── TIER 3 — the review strip goes ────────────────────────────────────────────────────
 * `thumbnails` off. Each grab is a `drawImage` to 480 px plus a `toBlob` JPEG encode, on a
 * phone already running a hardware encode. They exist only for the review strip, which
 * renders correctly with zero stills, and nothing in the verdict, the coverage report or
 * the upload reads them.
 * `latticeStations` 32 -> 16. The 43 deg frustum holds about a dozen glyphs at once; 16
 * keeps every visible one plus a margin for the ones sliding in at the edge.
 *
 * ── TIER 4 — the floor ────────────────────────────────────────────────────────────────
 * `stampHz` 10 -> 5. THIS IS THE MEASUREMENT THAT BOUNDS THE TIER. At the coached 24 deg/s
 * a 10 Hz stamp advances 2.4 deg between cones and a 5 Hz stamp advances 4.8 deg, both
 * under the 5 deg grid cell, so the cone track stays contiguous and no cell is skipped. At
 * 4 Hz it would advance 6.0 deg — wider than a cell — and the grid would grow holes the
 * verdict reads as missed coverage on a capture that was physically perfect. Five is
 * therefore not a taste, it is the last rate that preserves the coverage guarantee. The
 * telemetry orientation track rides the same clock and halves with it; 5 Hz is still twice
 * the 2 Hz the pose alignment needs.
 * `publishHz` 24 -> 15: 1.6 deg of overlay lag at the coached rate. Perceptible as softness
 * in the lattice, not as a ring in the wrong place. Below 15 the lattice reads as sliding
 * against the room, which is the exact complaint this module exists to answer, so 15 is the
 * floor rather than a step on the way down.
 * `qualityHz` 4 -> 2 and `latticeStations` 12: twelve is what the frustum holds, so this is
 * the last cut that removes only glyphs the user cannot see.
 */
export const LOAD_PLANS: readonly LoadPlan[] = Object.freeze([
  Object.freeze({ tier: 0, opticalFlow: true, qualityHz: 10, publishHz: 30, stampHz: 10, latticeStations: Infinity, thumbnails: true }),
  Object.freeze({ tier: 1, opticalFlow: false, qualityHz: 10, publishHz: 30, stampHz: 10, latticeStations: Infinity, thumbnails: true }),
  Object.freeze({ tier: 2, opticalFlow: false, qualityHz: 4, publishHz: 24, stampHz: 10, latticeStations: 32, thumbnails: true }),
  Object.freeze({ tier: 3, opticalFlow: false, qualityHz: 4, publishHz: 24, stampHz: 10, latticeStations: 16, thumbnails: false }),
  Object.freeze({ tier: 4, opticalFlow: false, qualityHz: 2, publishHz: 15, stampHz: 5, latticeStations: 12, thumbnails: false }),
]) as readonly LoadPlan[];

export function planForTier(tier: LoadTier): LoadPlan {
  return LOAD_PLANS[tier];
}

/** Publish cadence in milliseconds. */
export function publishIntervalMs(plan: LoadPlan): number {
  return 1000 / plan.publishHz;
}

/**
 * How often the loop body may run — deliberately 3 ms UNDER the publish interval, so a tick
 * is never skipped merely because it arrived a millisecond early on a 60 Hz display. That
 * off-by-one is why the shipped constants are 33 and 30 rather than 33 and 33.
 */
export function processIntervalMs(plan: LoadPlan): number {
  return 1000 / plan.publishHz - 3;
}

// ── The measurement ────────────────────────────────────────────────────────────────────

/**
 * Two seconds of history. Long enough that the median is not swayed by one bad frame, short
 * enough that a phone which recovers (the encoder settles, the thermal throttle lifts) is
 * seen to have recovered inside the 8 s restore clock rather than after it.
 */
const WINDOW_MS = 2000;

/** 2 s at 120 Hz is 240 intervals; the ring is sized once and never reallocated. */
const CAPACITY = 256;

/**
 * No verdict before this many intervals. Six is 100 ms on a healthy phone and 300 ms on the
 * 20 fps case this is meant to catch — early enough to act, late enough that a median means
 * something.
 */
const MIN_SAMPLES = 6;

/**
 * The sort behind the median runs at most this often. `frame()` is called on EVERY animation
 * frame, and a governor that sorted 240 numbers 120 times a second would be a plausible
 * candidate for the lag it was hired to measure. Ten evaluations a second is four times
 * finer than the 1.5 s escalation clock can use.
 */
const EVALUATE_INTERVAL_MS = 100;

/**
 * An interval longer than this is DROPPED, not recorded.
 *
 * A backgrounded tab, a permission dialog, an incoming call or an attached debugger produces
 * one multi-second gap. That is not jank and it is not the phone's fault, but a single 3000
 * ms sample sits in the window for two seconds and drags the jank ratio to a tenth of the
 * window on its own. The cut-off is 1 s rather than something tighter precisely so that the
 * 300 ms stalls the median is designed to absorb still reach it and still get absorbed.
 */
const STALL_IGNORE_MS = 1000;

/** The fastest interval any of this is willing to believe: 144 Hz. */
const MIN_NOMINAL_MS = 1000 / 144;

/**
 * THE CLAMP THAT MAKES THE INFERENCE SAFE.
 *
 * The display's nominal frame time is inferred from the fastest intervals observed, because
 * plenty of these phones are 90 or 120 Hz and some throttle rAF to 30 — assuming 60 would
 * either flag a healthy 120 Hz panel or excuse a 30 Hz one. But inference alone has a hole:
 * a phone that has been slow since its very first frame observes nothing fast, infers its
 * own sickness as the baseline, and reports itself healthy forever. That is the failure mode
 * that let two rounds of fixes ship without helping. Capping the inferred nominal at 30 Hz
 * means the governor will accept a slow DISPLAY but never a slow PAGE.
 */
const MAX_NOMINAL_MS = 1000 / 30;

/** The nominal comes off the 10th percentile, not the minimum: one lucky 4 ms frame is not a refresh rate. */
const NOMINAL_QUANTILE = 0.1;

/** 20% of headroom over nominal before a phone is called slow. */
const SLOW_FACTOR = 1.2;

/**
 * The absolute floor the target can never fall below, whatever the display does.
 *
 * A 120 Hz phone rendering at 60 is perfectly smooth for this capture, and shedding work
 * from it because it is "half its nominal" would be pure damage. The overlay is world-locked
 * to a room the user is turning through at 24 deg/s; 30 fps is where that stops reading as
 * locked. So badness is an ABSOLUTE test at 30 fps, which the nominal may raise for a
 * genuinely slow display but may never lower.
 */
const FLOOR_INTERVAL_MS = 1000 / 30;

/** An interval past 1.5x nominal is a dropped frame, which is what the eye actually notices. */
const JANK_FACTOR = 1.5;

/** A quarter of the window dropping frames is visible stutter even when the median is fine. */
const JANK_LIMIT = 0.25;

/**
 * Health is not merely the absence of badness — it is 20% inside it. Without the gap a phone
 * sitting exactly on the threshold alternates between escalating and restoring forever, and
 * the flicker of the overlay changing density is more objectionable than either tier.
 */
const HEALTHY_FACTOR = 0.8;

/** Sustained badness before a step down. Long enough to outlast one rough second, short enough to act inside a row. */
const ESCALATE_AFTER_MS = 1500;

/** Median past twice the target is under 15 fps: the ladder would take 6 s to reach tier 4 and the user has already given up. */
const PANIC_FACTOR = 2;
const PANIC_AFTER_MS = 700;

/**
 * Sustained health before a step UP, and it is deliberately more than five times the
 * escalation clock. Shedding is cheap and invisible; restoring is a bet that the phone can
 * take the work back, and losing that bet costs another 1.5 s of stutter. Eight seconds is
 * also longer than the thermal and auto-exposure transients that make a phone briefly look
 * recovered when it is not.
 */
const RECOVER_AFTER_MS = 8000;

/**
 * Three arrivals at a tier and it becomes the floor for the rest of the capture.
 *
 * A device that keeps falling back to tier 2 is not going to stop falling back to tier 2,
 * and the round trip costs 1.5 s of stutter on the way down plus a visible change in overlay
 * density on the way up, twice per cycle. Pinning is a FLOOR, not a freeze: escalating past
 * a pinned tier stays available, because the whole point is to end up smooth.
 */
const PIN_AFTER_VISITS = 3;

// ── The meter ──────────────────────────────────────────────────────────────────────────

export interface FrameMeter {
  /** Real frames per second, from the mean of the kept intervals. */
  fps: number;
  /** The median interval, which is the number the tier decision is actually made on. */
  medianIntervalMs: number;
  /** Fraction of the window past 1.5x nominal. */
  jankRatio: number;
  /** The display's inferred frame time, clamped into [144 Hz, 30 Hz]. */
  nominalIntervalMs: number;
  /** The interval the median has to stay under to count as healthy. */
  targetIntervalMs: number;
  tier: LoadTier;
  /** The lowest tier still reachable; equals `tier` once this device has been pinned. */
  floorTier: LoadTier;
  /** Why the tier last moved, or why it has not. */
  reason: string;
  /** Timestamp of the last tier change, so the HUD can show how long this tier has held. */
  changedAt: Millis;
  /** Intervals currently in the window. Under `MIN_SAMPLES` no verdict has been reached yet. */
  samples: number;
}

export interface Governor {
  /**
   * Feed one animation-frame timestamp and read the plan back. Cheap on every call: a ring
   * write and two comparisons, with the sort and the tier decision throttled to 10 Hz.
   */
  frame(now: Millis): LoadPlan;
  /** The current plan without feeding a timestamp — for consumers that are not the rAF driver. */
  plan(): LoadPlan;
  /** Diagnostics. Allocates, so call it from the `?debug=1` path only. */
  meter(): FrameMeter;
}

export function createGovernor(): Governor {
  // Ring buffer. Parallel arrays rather than objects, because this is written 120 times a
  // second on the phones that need it least and allocating there would be absurd.
  const intervals = new Float64Array(CAPACITY);
  const stamps = new Float64Array(CAPACITY);
  const scratch = new Float64Array(CAPACITY);
  let start = 0;
  let count = 0;
  let sum = 0;

  let previousAt: number | null = null;
  let lastEvaluatedAt = -Infinity;

  let nominalMs = MAX_NOMINAL_MS;
  let medianMs = 0;
  let jankRatio = 0;
  let targetMs = FLOOR_INTERVAL_MS;
  let fps = 0;

  let tier: LoadTier = 0;
  let floorTier: LoadTier = 0;
  let reason = "no verdict yet";
  let changedAt = 0;
  const visits = [1, 0, 0, 0, 0];

  let badSince: number | null = null;
  let goodSince: number | null = null;

  const push = (at: number, interval: number): void => {
    if (count === CAPACITY) {
      sum -= intervals[start];
      start = (start + 1) % CAPACITY;
      count -= 1;
    }
    const slot = (start + count) % CAPACITY;
    intervals[slot] = interval;
    stamps[slot] = at;
    sum += interval;
    count += 1;
  };

  const trim = (at: number): void => {
    const cutoff = at - WINDOW_MS;
    while (count > 0 && stamps[start] < cutoff) {
      sum -= intervals[start];
      start = (start + 1) % CAPACITY;
      count -= 1;
    }
  };

  /** Copy the live window out of the ring into a flat sorted view. One sort per evaluation. */
  const sortWindow = (): Float64Array => {
    for (let i = 0; i < count; i += 1) scratch[i] = intervals[(start + i) % CAPACITY];
    const view = scratch.subarray(0, count);
    view.sort();
    return view;
  };

  const setTier = (next: LoadTier, why: string, at: number): void => {
    tier = next;
    reason = why;
    changedAt = at;
    badSince = null;
    goodSince = null;
    visits[next] += 1;
    if (visits[next] >= PIN_AFTER_VISITS && next > floorTier) {
      floorTier = next;
      reason = `${why}; pinned at tier ${next} after ${visits[next]} visits`;
    }
  };

  const evaluate = (at: number): void => {
    lastEvaluatedAt = at;
    if (count < MIN_SAMPLES) {
      badSince = null;
      goodSince = null;
      return;
    }

    const sorted = sortWindow();
    medianMs = quantile(sorted, count, 0.5);
    fps = sum > 0 ? (count * 1000) / sum : 0;

    // The nominal only ever tightens. Evidence that the display CAN go faster is permanent;
    // evidence that it is currently slow is exactly what the tier decision is for, and
    // letting the baseline drift up to meet it would silence the instrument.
    const observed = quantile(sorted, count, NOMINAL_QUANTILE);
    const clamped = Math.min(Math.max(observed, MIN_NOMINAL_MS), MAX_NOMINAL_MS);
    if (clamped < nominalMs) nominalMs = clamped;

    const jankAbove = nominalMs * JANK_FACTOR;
    let janky = 0;
    for (let i = 0; i < count; i += 1) if (sorted[i] > jankAbove) janky += 1;
    jankRatio = janky / count;

    targetMs = Math.max(nominalMs * SLOW_FACTOR, FLOOR_INTERVAL_MS);

    // TWO INDEPENDENT BADNESS SIGNALS, because neither catches the other's case.
    //
    // The median catches a uniformly slow phone — 50 ms every frame — which a jank ratio
    // measured against its own inferred nominal would call perfect. The jank ratio catches a
    // phone alternating 16 and 50 ms, whose median lands on 16 or on 50 depending on nothing
    // more than the parity of the sample count, and which is the single most unpleasant
    // thing a world-locked overlay can do.
    const slow = medianMs > targetMs;
    const stuttering = jankRatio > JANK_LIMIT;
    const bad = slow || stuttering;
    const healthy = medianMs <= targetMs * HEALTHY_FACTOR && jankRatio <= JANK_LIMIT * HEALTHY_FACTOR;

    if (bad) {
      goodSince = null;
      badSince ??= at;
      if (tier >= 4) {
        reason = `tier 4 and still ${fps.toFixed(0)}fps — nothing left to shed`;
        return;
      }
      const held = at - badSince;
      const panicking = medianMs > targetMs * PANIC_FACTOR;
      if (panicking && held >= PANIC_AFTER_MS) {
        const next = tierOf(tier + 2);
        setTier(next, `shed x2: ${fps.toFixed(0)}fps, median ${medianMs.toFixed(0)}ms vs ${targetMs.toFixed(0)}ms target`, at);
        return;
      }
      if (held >= ESCALATE_AFTER_MS) {
        const next = tierOf(tier + 1);
        const cause = slow ? `median ${medianMs.toFixed(0)}ms over ${targetMs.toFixed(0)}ms` : `jank ${(jankRatio * 100).toFixed(0)}%`;
        setTier(next, `shed: ${cause} held ${(held / 1000).toFixed(1)}s`, at);
      }
      return;
    }

    badSince = null;
    if (!healthy) {
      // The hysteresis band: not bad enough to shed, not good enough to restore. Both clocks
      // stay stopped, which is the whole reason the band exists.
      goodSince = null;
      return;
    }

    goodSince ??= at;
    if (tier === floorTier) {
      goodSince = null;
      if (tier > 0) reason = `holding tier ${tier}: pinned floor`;
      return;
    }
    if (at - goodSince >= RECOVER_AFTER_MS) {
      // ONE TIER AT A TIME, always. Restoring two at once is a bet that the phone can take
      // back work it has already failed at twice, and the 1.5 s of stutter it costs to lose
      // that bet is worse than the tier.
      setTier(tierOf(tier - 1), `restore: ${(RECOVER_AFTER_MS / 1000).toFixed(0)}s at ${fps.toFixed(0)}fps`, at);
    }
  };

  return {
    frame(now: Millis): LoadPlan {
      if (previousAt !== null) {
        const interval = now - previousAt;
        // A non-monotonic timestamp is a caller mixing clocks; ignoring it beats poisoning
        // the window with a negative interval that would read as an impossibly fast display
        // and drag the inferred nominal to the floor for the rest of the capture.
        if (interval > 0 && interval <= STALL_IGNORE_MS) push(now, interval);
      }
      previousAt = now;
      trim(now);
      if (now - lastEvaluatedAt >= EVALUATE_INTERVAL_MS) evaluate(now);
      return LOAD_PLANS[tier];
    },

    plan(): LoadPlan {
      return LOAD_PLANS[tier];
    },

    meter(): FrameMeter {
      return {
        fps,
        medianIntervalMs: medianMs,
        jankRatio,
        nominalIntervalMs: nominalMs,
        targetIntervalMs: targetMs,
        tier,
        floorTier,
        reason,
        changedAt,
        samples: count,
      };
    },
  };
}

// ── Helpers ────────────────────────────────────────────────────────────────────────────

function tierOf(n: number): LoadTier {
  const clamped = Math.min(4, Math.max(0, Math.round(n)));
  return clamped as LoadTier;
}

/** Nearest-rank quantile over an ascending view. No interpolation: these are frame times, not statistics. */
function quantile(sorted: Float64Array, n: number, q: number): number {
  if (n === 0) return 0;
  const index = Math.min(n - 1, Math.max(0, Math.round(q * (n - 1))));
  return sorted[index];
}
