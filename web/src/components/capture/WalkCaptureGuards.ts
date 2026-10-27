/**
 * Live checks for a 3D-walkthrough recording, as pure functions (vitest runs them under Node).
 *
 * The walk's 3D build breaks where two frames a third of a second apart share nothing the
 * matcher can hold on to. On the first real capture (82 s, an empty white flat) that happened
 * five times, and every one was one of two things the phone can see coming:
 *
 *   - a turn too fast for the exposure: the frames around 11 s, 60 s and 72 s are smeared
 *     (Laplacian variance ~1 against ~37 for a kept frame). The solved parts of that walk
 *     turned at 22 deg/s median, but 28 % of the time above 40 deg/s and 10 % above 60;
 *     the breaks themselves were whip pans of 90-220 deg/s. The turning-speed meter covers it.
 *   - a bare wall filling the frame: around 50.7 s six consecutive frames score 0.01-0.11 on
 *     the measure below, where the walk's median frame scores 0.56. Nothing warned about it.
 *
 * `textureScore` is that measure: the fraction of 8x8 cells of a 96-px-wide grey thumbnail
 * whose grey levels spread by more than 6. It tracks what the pipeline's feature detector
 * finds (correlation 0.67 with ALIKED keypoints per frame on the same walk) and costs one
 * 96x170 readback. On a capture that rebuilt cleanly (the berlin bench flat) 1.8 % of frames
 * score under 0.15, on the real walk 2.0 % - so a sustained low score is a moment, not a
 * flat, and the warning fires only for the moments that broke the walk.
 */

/** Width of the thumbnail the score is computed on, pixels. */
export const TEXTURE_THUMB_WIDTH = 96;
/** Cell size and grey-level spread that count a cell as "has detail". */
export const TEXTURE_CELL = 8;
export const TEXTURE_SPREAD = 6;
/** Below this score the view is a bare surface (see the module comment for the numbers). */
export const BLANK_WALL_SCORE = 0.15;
/** ...and it must stay there this long before the warning shows: a pan across a wall is fine. */
export const BLANK_WALL_SUSTAIN_MS = 500;

/**
 * Fraction (0..1) of TEXTURE_CELL-sized cells whose grey-level standard deviation exceeds
 * TEXTURE_SPREAD, on an RGBA buffer of `width` x `height` (already thumbnail-sized).
 */
export function textureScore(rgba: ArrayLike<number>, width: number, height: number): number {
  const c = TEXTURE_CELL;
  const cols = Math.floor(width / c);
  const rows = Math.floor(height / c);
  if (cols === 0 || rows === 0) return 0;
  let detailed = 0;
  for (let cy = 0; cy < rows; cy += 1) {
    for (let cx = 0; cx < cols; cx += 1) {
      let sum = 0;
      let sq = 0;
      for (let y = cy * c; y < cy * c + c; y += 1) {
        for (let x = cx * c; x < cx * c + c; x += 1) {
          const i = (y * width + x) * 4;
          const g = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
          sum += g;
          sq += g * g;
        }
      }
      const n = c * c;
      const mean = sum / n;
      const sd = Math.sqrt(Math.max(sq / n - mean * mean, 0));
      if (sd > TEXTURE_SPREAD) detailed += 1;
    }
  }
  return detailed / (cols * rows);
}

/** State of a sustain-latched warning: when the condition started holding (ms), or null. */
export interface Latch {
  since: number | null;
}

/** True once `condition` has held for `sustainMs`; resets the moment it stops holding. */
export function latch(state: Latch, condition: boolean, now: number, sustainMs: number): boolean {
  if (!condition) {
    state.since = null;
    return false;
  }
  if (state.since === null) state.since = now;
  return now - state.since >= sustainMs;
}
