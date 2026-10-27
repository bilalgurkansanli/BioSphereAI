import { describe, expect, it } from "vitest";
import { buildConstraintLadder } from "@/lib/capture/useCamera";
import { RECORDING } from "@/lib/capture/types";

/**
 * TWO SHIPPED REGRESSIONS, IN OPPOSITE DIRECTIONS. Both came from writing the requirement
 * as a ceiling when it is a floor.
 *
 *   1. Capping BOTH edges at 1080x1920 asks for 9:16. On a 3:4 sensor delivering 1728x2304
 *      the only way there is to throw away a quarter of the width, so `crop-and-scale`
 *      cropped and the horizontal field of view fell from 43 deg to 32.9 — "the camera is
 *      zoomed in", and underneath it a lattice built on 43 deg quietly invalidated.
 *
 *   2. Dropping the height cap and keeping `width: {max: 1080}` capped the LONG edge instead,
 *      because camera modes are landscape. The UA scaled 1920x1080 to 1080x608 and the track
 *      arrived as 608x1080: 0.66 MP, on which the solver could not link 23 of 87 keyframes.
 *
 * The rule is therefore: a MINIMUM on both edges, no maximum anywhere, no aspect ratio.
 */
describe("the camera constraint ladder", () => {
  const short = RECORDING.minShortEdgePx;
  const ladder = buildConstraintLadder("dev-1", short);
  const videoOf = (c: MediaStreamConstraints) =>
    c.video as Record<string, { ideal?: number; max?: number; min?: number } | undefined>;

  it("never pins an aspect ratio", () => {
    for (const rung of ladder) {
      const v = videoOf(rung);
      if (!v.width && !v.height) continue;
      // An `ideal` on BOTH edges is a shape, and a shape is something to crop towards.
      expect(v.width?.ideal !== undefined && v.height?.ideal !== undefined).toBe(false);
      expect(v.aspectRatio).toBeUndefined();
    }
  });

  it("puts a floor under BOTH edges, which is the only way to say 'short edge'", () => {
    const v = videoOf(ladder[0]);
    expect(v.width?.min).toBe(short);
    expect(v.height?.min).toBe(short);
  });

  it("caps neither edge — a ceiling is what cropped, and then what downscaled", () => {
    for (const rung of ladder) {
      const v = videoOf(rung);
      expect(v.width?.max).toBeUndefined();
      expect(v.height?.max).toBeUndefined();
    }
  });

  it("does not make the frame rate a hard floor, which a dim room cannot meet", () => {
    // An over-constrained request throws rather than degrading, and the ladder would then
    // fall all the way to the rung with no geometry at all — losing the resolution floor
    // to save a frame rate no sensor can deliver at 58 ms of exposure.
    for (const rung of ladder) {
      expect(videoOf(rung).frameRate?.min).toBeUndefined();
    }
  });

  it("offers a lower floor before giving up on geometry entirely", () => {
    const floors = ladder
      .map((rung) => videoOf(rung).width?.min)
      .filter((min): min is number => typeof min === "number");
    expect(new Set(floors).size).toBeGreaterThan(1);
    expect(Math.min(...floors)).toBeLessThan(short);
  });

  it("keeps a last rung with no geometry at all, for a phone that refuses the rest", () => {
    const last = videoOf(ladder[ladder.length - 1]);
    expect(last.width).toBeUndefined();
    expect(last.height).toBeUndefined();
  });
});
