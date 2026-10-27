/**
 * The load governor, driven off a fake clock.
 *
 * Every test here is a frame stream and nothing else: the governor has no timer, no DOM and
 * no React, so a test IS the phone. The streams are the four the capture actually meets — a
 * healthy 60 Hz panel, a phone stuck at 20 fps, a phone that alternates, and a phone that
 * recovers — plus the two the earlier analysis got wrong on desktop: a display that is not
 * 60 Hz, and a multi-second gap that is not jank.
 */

import { describe, expect, it } from "vitest";

import {
  LOAD_PLANS,
  createGovernor,
  planForTier,
  processIntervalMs,
  publishIntervalMs,
  type Governor,
} from "@/lib/capture/governor";

const MS_60HZ = 1000 / 60;
const MS_120HZ = 1000 / 120;
const MS_30HZ = 1000 / 30;
const MS_20FPS = 50;
const MS_8FPS = 125;

/** Feed a constant-interval stream for a fixed span. Returns the timestamp to resume from. */
function driveFor(g: Governor, from: number, durationMs: number, intervalMs: number): number {
  let t = from;
  const end = from + durationMs;
  while (t <= end) {
    g.frame(t);
    t += intervalMs;
  }
  return t;
}

/**
 * Feed a stream until `done()`, and FAIL rather than hang if it never happens. The limit is
 * an assertion in its own right — "restores within a minute" is part of what is being tested.
 */
function driveUntil(g: Governor, from: number, intervalMs: number, done: () => boolean, limitMs: number, what: string): number {
  let t = from;
  const deadline = from + limitMs;
  while (t <= deadline) {
    g.frame(t);
    if (done()) return t;
    t += intervalMs;
  }
  throw new Error(`${what} did not happen within ${limitMs} ms (tier ${g.plan().tier}, ${g.meter().reason})`);
}

describe("LOAD_PLANS", () => {
  it("never sheds past what the capture still has to be able to measure", () => {
    for (const plan of LOAD_PLANS) {
      // A 5 deg cell at the coached 24 deg/s: 5 Hz advances 4.8 deg per stamp, 4 Hz would
      // advance 6.0 deg and start skipping cells the verdict then reads as missed coverage.
      expect(plan.stampHz).toBeGreaterThanOrEqual(5);
      expect(24 / plan.stampHz).toBeLessThan(5);
      // Below 15 Hz the world-locked lattice reads as sliding against the room, which is the
      // complaint this module exists to answer rather than to cause.
      expect(plan.publishHz).toBeGreaterThanOrEqual(15);
      // The 43 deg frustum holds about a dozen glyphs; fewer and the lattice ends in mid-air.
      expect(plan.latticeStations).toBeGreaterThanOrEqual(12);
      expect(plan.qualityHz).toBeGreaterThanOrEqual(2);
    }
  });

  it("only ever turns knobs down as the tier rises", () => {
    for (let i = 1; i < LOAD_PLANS.length; i += 1) {
      const previous = LOAD_PLANS[i - 1];
      const plan = LOAD_PLANS[i];
      expect(plan.tier).toBe(i);
      expect(plan.qualityHz).toBeLessThanOrEqual(previous.qualityHz);
      expect(plan.publishHz).toBeLessThanOrEqual(previous.publishHz);
      expect(plan.stampHz).toBeLessThanOrEqual(previous.stampHz);
      expect(plan.latticeStations).toBeLessThanOrEqual(previous.latticeStations);
      expect(Number(plan.opticalFlow)).toBeLessThanOrEqual(Number(previous.opticalFlow));
      expect(Number(plan.thumbnails)).toBeLessThanOrEqual(Number(previous.thumbnails));
    }
  });

  it("turns the second GPU readback off first and alone", () => {
    expect(LOAD_PLANS[0].opticalFlow).toBe(true);
    expect(LOAD_PLANS[1].opticalFlow).toBe(false);
    // Tier 1 changes nothing else, so a phone that only needed the readbacks halved pays
    // nothing the user can see.
    expect(LOAD_PLANS[1].qualityHz).toBe(LOAD_PLANS[0].qualityHz);
    expect(LOAD_PLANS[1].publishHz).toBe(LOAD_PLANS[0].publishHz);
    expect(LOAD_PLANS[1].thumbnails).toBe(true);
  });

  it("stops grabbing review stills from tier 3", () => {
    expect(LOAD_PLANS[2].thumbnails).toBe(true);
    expect(LOAD_PLANS[3].thumbnails).toBe(false);
    expect(LOAD_PLANS[4].thumbnails).toBe(false);
  });

  it("derives the loop cadence with the same off-by-three the shipped constants use", () => {
    expect(publishIntervalMs(planForTier(0))).toBeCloseTo(33.33, 1);
    expect(processIntervalMs(planForTier(0))).toBeCloseTo(30.33, 1);
    expect(processIntervalMs(planForTier(4))).toBeCloseTo(63.67, 1);
  });
});

describe("a healthy phone", () => {
  it("stays at tier 0 through a smooth 60 Hz minute", () => {
    const g = createGovernor();
    driveFor(g, 0, 60_000, MS_60HZ);
    const meter = g.meter();
    expect(meter.tier).toBe(0);
    expect(meter.floorTier).toBe(0);
    expect(meter.fps).toBeCloseTo(60, 0);
    expect(meter.medianIntervalMs).toBeCloseTo(MS_60HZ, 1);
    expect(meter.jankRatio).toBe(0);
    expect(g.plan()).toBe(LOAD_PLANS[0]);
  });

  it("infers 120 Hz rather than assuming 60, and does not shed from a fast panel", () => {
    const g = createGovernor();
    driveFor(g, 0, 20_000, MS_120HZ);
    const meter = g.meter();
    expect(meter.nominalIntervalMs).toBeCloseTo(MS_120HZ, 1);
    expect(meter.fps).toBeCloseTo(120, 0);
    expect(meter.tier).toBe(0);
    // The target never drops below the absolute 30 fps floor: a 120 Hz phone rendering at 60
    // is smooth for this capture, and shedding work from it would be pure damage.
    expect(meter.targetIntervalMs).toBeCloseTo(1000 / 30, 1);
  });

  it("does not condemn a browser that throttles rAF to 30", () => {
    const g = createGovernor();
    driveFor(g, 0, 20_000, MS_30HZ);
    expect(g.plan().tier).toBe(0);
    expect(g.meter().nominalIntervalMs).toBeCloseTo(MS_30HZ, 1);
  });

  it("absorbs one 300 ms stall instead of treating it as a verdict", () => {
    const g = createGovernor();
    let t = driveFor(g, 0, 4000, MS_60HZ);
    t += 300;
    g.frame(t);
    t = driveFor(g, t + MS_60HZ, 4000, MS_60HZ);
    const meter = g.meter();
    expect(meter.tier).toBe(0);
    expect(meter.medianIntervalMs).toBeCloseTo(MS_60HZ, 1);
  });

  it("drops a backgrounded gap entirely rather than poisoning the window with it", () => {
    const g = createGovernor();
    let t = driveFor(g, 0, 3000, MS_60HZ);
    // Five seconds with the tab hidden, a permission sheet up or a call in progress. Not jank,
    // and one sample of it would sit in a 2 s window dragging the jank ratio on its own.
    t += 5000;
    g.frame(t);
    driveFor(g, t + MS_60HZ, 3000, MS_60HZ);
    const meter = g.meter();
    expect(meter.tier).toBe(0);
    expect(meter.fps).toBeCloseTo(60, 0);
    expect(meter.jankRatio).toBe(0);
  });
});

describe("a phone that cannot keep up", () => {
  it("holds tier 0 for the first second and a half, then sheds", () => {
    const g = createGovernor();
    // The first verdict lands once the window holds six intervals — 300 ms at 20 fps — and
    // the escalation clock runs 1.5 s from there.
    const t = driveFor(g, 0, 1500, MS_20FPS);
    expect(g.plan().tier).toBe(0);
    driveFor(g, t, 700, MS_20FPS);
    expect(g.plan().tier).toBe(1);
    expect(g.meter().reason).toMatch(/shed/);
  });

  it("walks all the way down while the stream stays bad, one step per 1.5 s", () => {
    const g = createGovernor();
    driveFor(g, 0, 12_000, MS_20FPS);
    expect(g.plan().tier).toBe(4);
    expect(g.meter().reason).toMatch(/nothing left to shed/);
  });

  it("skips two tiers at once when the phone is under half the target rate", () => {
    const g = createGovernor();
    // 8 fps. Climbing the ladder one rung per 1.5 s would take six seconds to reach the
    // bottom, by which time the user has already put the phone down.
    driveFor(g, 0, 1700, MS_8FPS);
    expect(g.plan().tier).toBeGreaterThanOrEqual(2);
    expect(g.meter().reason).toMatch(/shed x2/);
    driveFor(g, 1700, 2000, MS_8FPS);
    expect(g.plan().tier).toBe(4);
  });

  it("sheds on a phone that alternates 16 and 50 ms even though its median looks survivable", () => {
    const g = createGovernor();
    let t = 0;
    // The jank ratio is the signal here: half the frames land past 1.5x nominal, which is
    // the most unpleasant thing a world-locked overlay can do and the one case a median
    // alone reports as either perfect or terrible depending on the parity of the sample count.
    for (let i = 0; i < 200; i += 1) {
      g.frame(t);
      t += i % 2 === 0 ? MS_60HZ : MS_20FPS;
    }
    expect(g.meter().jankRatio).toBeGreaterThan(0.25);
    expect(g.plan().tier).toBeGreaterThanOrEqual(1);
  });
});

describe("hysteresis", () => {
  it("does not flap when good and bad stretches alternate faster than the restore clock", () => {
    const g = createGovernor();
    let t = 0;
    const seen: number[] = [];
    // Three seconds bad, three seconds good, repeatedly. Each bad stretch is long enough to
    // shed; no good stretch is anywhere near the 8 s needed to take work back.
    for (let cycle = 0; cycle < 5; cycle += 1) {
      t = driveFor(g, t, 3000, MS_20FPS);
      seen.push(g.plan().tier);
      t = driveFor(g, t, 3000, MS_60HZ);
      seen.push(g.plan().tier);
    }
    for (let i = 1; i < seen.length; i += 1) expect(seen[i]).toBeGreaterThanOrEqual(seen[i - 1]);
    expect(seen[seen.length - 1]).toBeGreaterThan(0);
  });

  it("restores one tier at a time once the stream genuinely recovers", () => {
    const g = createGovernor();
    let t = driveFor(g, 0, 6000, MS_20FPS);
    const shed = g.plan().tier;
    expect(shed).toBeGreaterThanOrEqual(3);

    // Eight seconds of health buys exactly one tier back, never two.
    for (let step = shed - 1; step >= 0; step -= 1) {
      t = driveUntil(g, t, MS_60HZ, () => g.plan().tier === step, 20_000, `restore to tier ${step}`);
      expect(g.plan().tier).toBe(step);
      expect(g.meter().reason).toMatch(/restore/);
    }
    expect(g.plan().tier).toBe(0);
  });

  it("takes at least the restore clock to give a tier back", () => {
    const g = createGovernor();
    let t = driveFor(g, 0, 2200, MS_20FPS);
    expect(g.plan().tier).toBe(1);
    // The window still holds two seconds of bad frames when the good stream starts, so health
    // cannot even begin to accrue before that flushes.
    t = driveFor(g, t, 7000, MS_60HZ);
    expect(g.plan().tier).toBe(1);
    driveFor(g, t, 6000, MS_60HZ);
    expect(g.plan().tier).toBe(0);
  });
});

describe("pinning", () => {
  it("stops trying once a device has fallen back to the same tier three times", () => {
    const g = createGovernor();
    let t = 0;

    for (let cycle = 0; cycle < 3; cycle += 1) {
      t = driveUntil(g, t, MS_20FPS, () => g.plan().tier === 1, 10_000, `shed to tier 1 on cycle ${cycle}`);
      if (cycle === 2) break;
      t = driveUntil(g, t, MS_60HZ, () => g.plan().tier === 0, 30_000, `restore to tier 0 on cycle ${cycle}`);
    }

    expect(g.meter().floorTier).toBe(1);
    expect(g.meter().reason).toMatch(/pinned/);

    // Half a minute of a perfectly healthy stream, and it stays shed. A device that keeps
    // falling back is not going to get better, and the round trip is worse than the tier.
    driveFor(g, t, 30_000, MS_60HZ);
    expect(g.plan().tier).toBe(1);
    expect(g.meter().floorTier).toBe(1);
  });

  it("still lets a pinned device shed further", () => {
    const g = createGovernor();
    let t = 0;
    for (let cycle = 0; cycle < 3; cycle += 1) {
      t = driveUntil(g, t, MS_20FPS, () => g.plan().tier === 1, 10_000, `shed to tier 1 on cycle ${cycle}`);
      if (cycle === 2) break;
      t = driveUntil(g, t, MS_60HZ, () => g.plan().tier === 0, 30_000, `restore to tier 0 on cycle ${cycle}`);
    }
    expect(g.meter().floorTier).toBe(1);
    // Pinning is a floor, not a freeze: the point is still to end up smooth.
    driveFor(g, t, 8000, MS_20FPS);
    expect(g.plan().tier).toBeGreaterThan(1);
  });
});

describe("the meter", () => {
  it("reports nothing until the window holds enough intervals to mean anything", () => {
    const g = createGovernor();
    driveFor(g, 0, 40, MS_60HZ);
    const meter = g.meter();
    expect(meter.samples).toBeLessThan(6);
    expect(meter.tier).toBe(0);
    expect(meter.reason).toBe("no verdict yet");
  });

  it("hands back the same plan object for the same tier, so consumers can skip on identity", () => {
    const g = createGovernor();
    driveFor(g, 0, 1000, MS_60HZ);
    expect(g.plan()).toBe(g.plan());
    expect(g.plan()).toBe(planForTier(0));
  });

  it("survives a caller that hands it a non-monotonic timestamp", () => {
    const g = createGovernor();
    driveFor(g, 0, 2000, MS_60HZ);
    const before = g.meter().nominalIntervalMs;
    g.frame(500);
    g.frame(600);
    // A negative interval would read as an impossibly fast display and drag the inferred
    // nominal to the floor for the rest of the capture.
    expect(g.meter().nominalIntervalMs).toBeCloseTo(before, 3);
  });
});
