/**
 * Gaze-and-hold, driven with synthetic time at 60 frames a second. Each case is a thing a
 * head does: rest on a ring, wobble off its rim, glance across, keep staring after it fired.
 */

import { describe, expect, it } from "vitest";
import { DWELL_DEFAULTS, createDwell, resetDwell, stepDwell, type DwellState } from "../dwell";

const FRAME = 1000 / 60;

/** Runs `frames` steps with the given hover, returning every id that fired. */
function hold(state: DwellState, hovered: string | null, frames: number, clock: { now: number }): string[] {
  const fired: string[] = [];
  for (let i = 0; i < frames; i += 1) {
    clock.now += FRAME;
    const step = stepDwell(state, hovered, clock.now);
    if (step.fired) fired.push(step.fired);
  }
  return fired;
}

describe("dwell", () => {
  it("fires once the gaze has rested for the dwell time, not before", () => {
    const state = createDwell();
    const first = 10_000;
    // The first sighting starts the clock; it does not itself count as dwell time.
    expect(stepDwell(state, "ring", first).progress).toBe(0);
    expect(stepDwell(state, "ring", first + DWELL_DEFAULTS.dwellMs - 1).fired).toBeNull();
    expect(stepDwell(state, "ring", first + DWELL_DEFAULTS.dwellMs).fired).toBe("ring");
  });

  it("fires within a frame of the dwell time at 60 Hz", () => {
    const state = createDwell();
    const clock = { now: 10_000 };
    const frames = Math.ceil(DWELL_DEFAULTS.dwellMs / FRAME);
    expect(hold(state, "ring", frames - 1, clock)).toEqual([]);
    expect(hold(state, "ring", 3, clock)).toEqual(["ring"]);
  });

  it("reports progress that rises steadily to 1", () => {
    const state = createDwell();
    let now = 0;
    let last = -1;
    stepDwell(state, "ring", now);
    for (let i = 0; i < 120; i += 1) {
      now += FRAME;
      const { progress, fired } = stepDwell(state, "ring", now);
      if (fired) break;
      expect(progress).toBeGreaterThan(last);
      expect(progress).toBeLessThan(1);
      last = progress;
    }
    expect(last).toBeGreaterThan(0.95);
  });

  it("survives the tremor at a ring's rim", () => {
    const state = createDwell();
    const clock = { now: 0 };
    hold(state, "ring", 45, clock); // 0.75 s in
    hold(state, null, 6, clock); // 100 ms off the rim
    const fired = hold(state, "ring", 50, clock);
    expect(fired).toEqual(["ring"]);
  });

  it("does not count the time spent off the rim", () => {
    const state = createDwell();
    const clock = { now: 0 };
    hold(state, "ring", 30, clock);
    const before = stepDwell(state, "ring", (clock.now += FRAME)).progress;
    hold(state, null, 12, clock); // 200 ms away, inside the grace period
    const after = stepDwell(state, "ring", (clock.now += FRAME)).progress;
    // One frame of progress, not twelve.
    expect(after - before).toBeLessThan((2.5 * FRAME) / DWELL_DEFAULTS.dwellMs);
  });

  it("starts over after a real look away", () => {
    const state = createDwell();
    const clock = { now: 0 };
    hold(state, "ring", 70, clock); // nearly there
    hold(state, null, 30, clock); // half a second away
    expect(hold(state, "ring", 70, clock)).toEqual([]);
  });

  it("starts the neighbour from zero when the gaze slides across", () => {
    const state = createDwell();
    const clock = { now: 0 };
    hold(state, "a", 80, clock);
    const step = stepDwell(state, "b", (clock.now += FRAME));
    expect(step.progress).toBe(0);
    expect(hold(state, "b", 80, clock)).toEqual([]);
  });

  it("never fires twice for one long stare", () => {
    const state = createDwell();
    const clock = { now: 0 };
    expect(hold(state, "ring", 600, clock)).toEqual(["ring"]);
  });

  it("re-arms after the gaze leaves and comes back", () => {
    const state = createDwell();
    const clock = { now: 0 };
    expect(hold(state, "ring", 100, clock)).toEqual(["ring"]);
    hold(state, null, 60, clock);
    expect(hold(state, "ring", 100, clock)).toEqual(["ring"]);
  });

  it("refuses everything through the doorway's cooldown, then counts from its end", () => {
    const state = createDwell();
    stepDwell(state, "ring", 0);
    const firedAt = DWELL_DEFAULTS.dwellMs;
    expect(stepDwell(state, "ring", firedAt).fired).toBe("ring");
    const cooldownEnd = firedAt + DWELL_DEFAULTS.cooldownMs;
    // The next room appears with another ring right where the visitor is looking.
    for (let t = firedAt + FRAME; t < cooldownEnd; t += FRAME) {
      expect(stepDwell(state, "next-ring", t)).toEqual({ progress: 0, fired: null });
    }
    // It was seen all through the cooldown, but only the time after it counts.
    expect(stepDwell(state, "next-ring", cooldownEnd + DWELL_DEFAULTS.dwellMs - 1).fired).toBeNull();
    expect(stepDwell(state, "next-ring", cooldownEnd + DWELL_DEFAULTS.dwellMs).fired).toBe("next-ring");
  });

  it("does not refire the same ring when the gaze wobbles during the cooldown", () => {
    const state = createDwell();
    const clock = { now: 0 };
    expect(hold(state, "ring", 100, clock)).toEqual(["ring"]);
    hold(state, null, 5, clock); // a wobble inside the grace period
    expect(hold(state, "ring", 300, clock)).toEqual([]);
  });

  it("can be reset by a room change, with its own cooldown", () => {
    const state = createDwell();
    const clock = { now: 0 };
    hold(state, "ring", 60, clock);
    const resetAt = clock.now;
    resetDwell(state, resetAt, 500);
    expect(hold(state, "ring", 20, clock)).toEqual([]);
    // The ring has been looked at since the reset, but counts only from the cooldown's end.
    expect(stepDwell(state, "ring", resetAt + 500 + DWELL_DEFAULTS.dwellMs - 1).fired).toBeNull();
    expect(stepDwell(state, "ring", resetAt + 500 + DWELL_DEFAULTS.dwellMs).fired).toBe("ring");
  });

  it("honours a per-target dwell time", () => {
    const state = createDwell();
    let now = 0;
    const slow = { ...DWELL_DEFAULTS, dwellMs: 2200 };
    stepDwell(state, "exit", now, slow);
    now += 1600;
    expect(stepDwell(state, "exit", now, slow).fired).toBeNull();
    now += 700;
    expect(stepDwell(state, "exit", now, slow).fired).toBe("exit");
  });
});
