/**
 * The four decisions the measure tool makes on its own, measured.
 *
 * Everything else it does is the engine's answer rendered back — but these are ours, and each
 * one is a place where a wrong line of code produces a number that LOOKS right. A tolerance
 * that collapsed to zero, or a comma that turned into a decimal point on a Turkish screen,
 * would both ship as "the measurement is a bit off" rather than as a crash.
 */

import { describe, expect, it } from "vitest";
import {
  CAMERA_HEIGHT_MAX_CM,
  CAMERA_HEIGHT_MIN_CM,
  DEFAULT_CAMERA_HEIGHT_CM,
  clampCameraHeight,
  formatLength,
  formatMetres,
  markFault,
  toleranceCm,
} from "../MeasureTool";

const rad = (deg: number) => (deg * Math.PI) / 180;

describe("markFault — why a point cannot reach the floor", () => {
  it("calls a point above the horizon what the engine calls it", () => {
    expect(markFault(rad(12))).toBe("horizon");
  });

  it("counts the horizon itself as above it, exactly as `pitch >= 0` does server-side", () => {
    expect(markFault(0)).toBe("horizon");
  });

  it("calls the half-degree sliver below the horizon grazing, not usable", () => {
    // The engine refuses anything shallower than 0.5 degrees of depression: at that angle the
    // floor hit is already 115 camera-heights away.
    expect(markFault(rad(-0.2))).toBe("grazing");
    expect(markFault(rad(-0.5))).toBe("grazing");
  });

  it("accepts an ordinary mark on the floor a couple of metres away", () => {
    // 155 cm up, about 2.2 m out: 35 degrees down.
    expect(markFault(rad(-35))).toBeNull();
    // And the shallowest thing the engine will still answer.
    expect(markFault(rad(-0.6))).toBeNull();
  });
});

describe("toleranceCm — the ± band", () => {
  it("never claims better than 5 per cent, however confident the engine is", () => {
    // 0.98 is the engine's own ceiling on `confidence`; nothing can score higher.
    expect(toleranceCm(300, 0.98)).toBeGreaterThanOrEqual(300 * 0.05);
    expect(toleranceCm(300, 1)).toBeCloseTo(15, 6);
  });

  it("opens up to 35 per cent when the engine does not trust the marks at all", () => {
    expect(toleranceCm(300, 0)).toBeCloseTo(105, 6);
  });

  it("moves in a straight line between those two bounds", () => {
    expect(toleranceCm(100, 0.5)).toBeCloseTo(20, 6);
  });

  it("clamps a confidence outside 0..1 rather than inverting the band", () => {
    expect(toleranceCm(100, 5)).toBeCloseTo(5, 6);
    expect(toleranceCm(100, -5)).toBeCloseTo(35, 6);
  });

  it("scales with the length, so a short segment gets a short band", () => {
    expect(toleranceCm(50, 0.8)).toBeCloseTo(toleranceCm(100, 0.8) / 2, 6);
  });
});

describe("formatMetres — the same notation as the floor plan", () => {
  it("writes a Turkish decimal with a comma", () => {
    expect(formatMetres(342, "tr")).toBe("3,42");
  });

  it("writes an English decimal with a point", () => {
    expect(formatMetres(342, "en")).toBe("3.42");
  });

  it("keeps two decimals even when they are zeros, so a column of numbers lines up", () => {
    expect(formatMetres(300, "tr")).toBe("3,00");
    expect(formatMetres(5, "en")).toBe("0.05");
  });
});

describe("formatLength — digits, and what they are worth", () => {
  it("prints the band beside the value", () => {
    // 342 cm at the engine's best confidence: 5.6 % of it is 19 cm.
    expect(formatLength(342, 0.98, "tr")).toBe("3,42 ± 0,19 m");
    expect(formatLength(342, 0.98, "en")).toBe("3.42 ± 0.19 m");
  });

  it("says nothing about a band when there is no confidence to derive one from", () => {
    expect(formatLength(342, null, "en")).toBe("3.42 m");
  });

  it("drops a band that would round to zero rather than printing '± 0,00 m'", () => {
    // 8 cm at full trust is a 4 mm band — below half a centimetre, where printing it would
    // claim an exactness the assumed camera height cannot support in either direction.
    expect(formatLength(8, 1, "tr")).toBe("0,08 m");
  });
});

describe("clampCameraHeight — the range the assumption can be true over", () => {
  it("leaves an ordinary hand-held height alone", () => {
    expect(clampCameraHeight(DEFAULT_CAMERA_HEIGHT_CM)).toBe(155);
    expect(clampCameraHeight(132)).toBe(132);
  });

  it("pulls a mistyped order of magnitude back to the edge instead of trusting it", () => {
    expect(clampCameraHeight(15)).toBe(CAMERA_HEIGHT_MIN_CM);
    expect(clampCameraHeight(1550)).toBe(CAMERA_HEIGHT_MAX_CM);
  });

  it("falls back to the default rather than propagating a NaN through every length", () => {
    expect(clampCameraHeight(Number.NaN)).toBe(DEFAULT_CAMERA_HEIGHT_CM);
  });

  it("rounds to whole centimetres — the assumption is not worth a millimetre", () => {
    expect(clampCameraHeight(155.7)).toBe(156);
  });
});
