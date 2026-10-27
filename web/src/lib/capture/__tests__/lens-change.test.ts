import { describe, expect, it } from "vitest";

import { lensReallyChanged, type LensSignature } from "@/lib/capture/useCamera";

/**
 * THE BUG THIS FILE EXISTS FOR, AND IT MADE THE CAPTURE IMPOSSIBLE ON IPHONE.
 *
 * The lens watchdog compared a joined STRING, so a wobble in any field read as a lens
 * switch. `lens_changed` is severity `stop`: it blocks the capture, dwell stops
 * accumulating, and NO reticle ever fills. The user aims at a ring, the ring reports it is
 * on target, and nothing happens — for the rest of the session, because the flag latched and
 * was never cleared.
 *
 * Android hid it: there zoom already sits at its minimum and the number never moves. iOS
 * hands back one virtual rear camera whose reported zoom drifts, so it tripped at once.
 */

const base: LensSignature = { deviceId: "cam-a", groupId: "grp-1", zoom: 1.0, shape: 0.5625 };
const with_ = (over: Partial<LensSignature>): LensSignature => ({ ...base, ...over });

describe("lensReallyChanged", () => {
  it("ignores the zoom drift a virtual camera reports", () => {
    for (const z of [0.98, 1.0, 1.02, 1.1, 1.2]) {
      expect(lensReallyChanged(base, with_({ zoom: z }))).toBe(false);
    }
  });

  it("catches the jump between the wide and the ultra-wide lens", () => {
    // Every phone that has both puts a factor of two between them.
    expect(lensReallyChanged(base, with_({ zoom: 0.5 }))).toBe(true);
    expect(lensReallyChanged(base, with_({ zoom: 2.0 }))).toBe(true);
  });

  it("catches a different camera outright", () => {
    expect(lensReallyChanged(base, with_({ deviceId: "cam-b" }))).toBe(true);
    expect(lensReallyChanged(base, with_({ groupId: "grp-2" }))).toBe(true);
  });

  it("catches a mode of a different shape, and forgives rounding", () => {
    expect(lensReallyChanged(base, with_({ shape: 0.75 }))).toBe(true); // 9:16 -> 3:4
    expect(lensReallyChanged(base, with_({ shape: 0.5629 }))).toBe(false);
  });

  it("says nothing when the platform reports nothing", () => {
    // iOS often omits zoom entirely, and a missing field must not read as a change — that is
    // the difference between a warning and a capture that cannot be completed.
    const blank: LensSignature = { deviceId: "", groupId: "", zoom: 0, shape: 0 };
    expect(lensReallyChanged(blank, blank)).toBe(false);
    expect(lensReallyChanged(base, with_({ zoom: 0 }))).toBe(false);
    expect(lensReallyChanged(base, with_({ shape: 0 }))).toBe(false);
    expect(lensReallyChanged(base, with_({ deviceId: "" }))).toBe(false);
  });

  it("is symmetric — which way the zoom moved is not the question", () => {
    const a = with_({ zoom: 1.0 });
    const b = with_({ zoom: 2.4 });
    expect(lensReallyChanged(a, b)).toBe(lensReallyChanged(b, a));
  });
});
