import { describe, expect, it } from "vitest";
import { BLANK_WALL_SCORE, BLANK_WALL_SUSTAIN_MS, latch, textureScore } from "../WalkCaptureGuards";

function image(width: number, height: number, grey: (x: number, y: number) => number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const g = grey(x, y);
      const i = (y * width + x) * 4;
      out[i] = g;
      out[i + 1] = g;
      out[i + 2] = g;
      out[i + 3] = 255;
    }
  }
  return out;
}

describe("textureScore", () => {
  it("scores a bare wall near zero, even with sensor noise", () => {
    let seed = 7;
    const noise = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5) * 6;
    const wall = image(96, 168, () => 200 + noise());
    expect(textureScore(wall, 96, 168)).toBeLessThan(BLANK_WALL_SCORE);
  });

  it("scores a room with edges and detail well above the warning line", () => {
    const room = image(96, 168, (x, y) => ((Math.floor(x / 5) + Math.floor(y / 7)) % 2 ? 60 : 190));
    expect(textureScore(room, 96, 168)).toBeGreaterThan(0.5);
  });

  it("counts only the cells that have detail", () => {
    // Left half detailed, right half flat.
    const half = image(96, 96, (x, y) => (x < 48 ? ((x + y) % 2 ? 40 : 220) : 128));
    expect(textureScore(half, 96, 96)).toBeCloseTo(0.5, 6);
  });
});

describe("latch", () => {
  it("fires only after the condition has held for the sustain time, and resets at once", () => {
    const s = { since: null as number | null };
    expect(latch(s, true, 0, BLANK_WALL_SUSTAIN_MS)).toBe(false);
    expect(latch(s, true, BLANK_WALL_SUSTAIN_MS - 1, BLANK_WALL_SUSTAIN_MS)).toBe(false);
    expect(latch(s, true, BLANK_WALL_SUSTAIN_MS, BLANK_WALL_SUSTAIN_MS)).toBe(true);
    expect(latch(s, false, BLANK_WALL_SUSTAIN_MS + 10, BLANK_WALL_SUSTAIN_MS)).toBe(false);
    expect(latch(s, true, BLANK_WALL_SUSTAIN_MS + 20, BLANK_WALL_SUSTAIN_MS)).toBe(false);
  });
});
