import { describe, expect, it } from "vitest";

import { autoRotateDelta, effectiveFov } from "../useLookControls";
import { FOV_DEFAULT, FOV_MAX, FOV_MIN, TIMING } from "../types";

/** Vertical field of view, in degrees, that spans `hfov` across a window of `aspect`. */
const verticalFor = (hfov: number, aspect: number) =>
  (2 * Math.atan(Math.tan((hfov * Math.PI) / 360) / aspect) * 180) / Math.PI;

/** Horizontal field, in degrees, that a vertical `fov` spans on a window of `aspect`. */
const horizontalFor = (fov: number, aspect: number) =>
  (2 * Math.atan(Math.tan((fov * Math.PI) / 360) * aspect) * 180) / Math.PI;

const ASPECTS: Array<[string, number]> = [
  ["phone portrait", 390 / 844],
  ["tablet portrait", 3 / 4],
  ["square", 1],
  ["laptop 16:10", 16 / 10],
  ["desktop 16:9", 16 / 9],
];

describe("zoom range", () => {
  it("reaches the widest zoom on every ordinary screen", () => {
    // The bug this pins: a horizontal cap of 100 degrees held a 16:9 window at 68 degrees
    // of vertical field, so the last third of the zoom-out travel changed the number and
    // nothing on screen — the view simply stopped widening with no edge to feel.
    for (const [name, aspect] of ASPECTS) {
      expect(effectiveFov(FOV_MAX, aspect), `${name} cannot reach the zoom-out limit`).toBeCloseTo(
        FOV_MAX,
        3,
      );
    }
  });

  it("still trims an ultrawide monitor's stretch", () => {
    const ultrawide = 21 / 9;
    const shown = effectiveFov(FOV_MAX, ultrawide);
    expect(shown).toBeLessThan(FOV_MAX);
    expect(horizontalFor(shown, ultrawide)).toBeLessThanOrEqual(131);
  });

  it("never widens what the caller asked for", () => {
    for (const [, aspect] of ASPECTS.concat([["ultrawide", 21 / 9]])) {
      for (const fov of [FOV_MIN, FOV_DEFAULT, FOV_MAX]) {
        expect(effectiveFov(fov, aspect)).toBeLessThanOrEqual(fov + 1e-9);
      }
    }
  });

  it("passes a degenerate aspect through untouched", () => {
    expect(effectiveFov(FOV_DEFAULT, 0)).toBe(FOV_DEFAULT);
    expect(effectiveFov(FOV_DEFAULT, Number.NaN)).toBe(FOV_DEFAULT);
  });

  it("keeps the zoom-in end usable at both extremes", () => {
    // A phone in portrait sees a lot vertically; the tight end must still be a real
    // magnification rather than a full-frame view.
    expect(horizontalFor(FOV_MIN, 390 / 844)).toBeLessThan(20);
    expect(horizontalFor(FOV_MIN, 16 / 9)).toBeLessThan(60);
    expect(verticalFor(130, 16 / 9)).toBeGreaterThan(FOV_MAX);
  });
});

/**
 * Integrates the drift at 60 Hz and reports how far the room has turned, in degrees.
 *
 * `sinceInputMs` is the wall-clock age of the last thing the visitor did. Pressing the toggle
 * backdates that to `TIMING.idleMs` (see the `autoRotate` transition in `useLookControls`),
 * which is what "start now" means in this file's arithmetic.
 */
function degreesTurned(forMs: number, sinceInputMs: number): number {
  const steps = Math.round((forMs / 1000) * 60);
  const step = forMs / steps;
  let yaw = 0;
  for (let i = 0; i < steps; i++) {
    yaw += autoRotateDelta(sinceInputMs + i * step - TIMING.idleMs, step / 1000);
  }
  return (yaw * 180) / Math.PI;
}

describe("auto-rotate", () => {
  it("turns the room visibly in the first second after the switch is thrown", () => {
    // The press backdates the idle clock, so the ramp starts now rather than four seconds
    // from now. The numbers this pins are the whole reason the control was retimed: the old
    // 1.6 deg/s over a 1400 ms ramp gave half a degree in the first second, which on a
    // photograph of a room is nothing at all, and a toggle whose effect you cannot see is a
    // toggle that looks broken.
    expect(degreesTurned(1000, TIMING.idleMs)).toBeGreaterThan(4);
    expect(degreesTurned(3000, TIMING.idleMs)).toBeGreaterThan(15);
    // And still calm: a full revolution is a minute, not seconds.
    expect(degreesTurned(3000, TIMING.idleMs)).toBeLessThan(22);
  });

  it("eases in rather than snapping to full speed", () => {
    expect(degreesTurned(300, TIMING.idleMs)).toBeLessThan(
      degreesTurned(300, TIMING.idleMs + 2000) / 2,
    );
    // Full rate once the ramp is done: six degrees a second, a turn in sixty.
    expect(degreesTurned(1000, TIMING.idleMs + 2000)).toBeCloseTo(6, 1);
  });

  it("stays still until the idle gate opens", () => {
    // A player nobody has switched it on for has just been touched; it must not creep.
    expect(degreesTurned(TIMING.idleMs, 0)).toBe(0);
    expect(autoRotateDelta(0, 1)).toBe(0);
    expect(autoRotateDelta(-1000, 1)).toBe(0);
  });
});
