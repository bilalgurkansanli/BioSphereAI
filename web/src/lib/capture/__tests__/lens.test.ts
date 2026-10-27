import { describe, expect, it } from "vitest";

import { COMPLETENESS, DEFAULT_LENS, lensForFrame, ROW_IDS, ROW_PLAN, rowSeparationDeg, vfovForFrame } from "../types";
import { createGrid, gatesForLens, greatCircleDeg, report, rowTargetPitchDeg, stampCone } from "../coverage";
import { buildStations } from "../plan";
import { makeConfig, sweep } from "./synthetic";

/**
 * THE REGRESSION THIS FILE EXISTS FOR.
 *
 * `DEFAULT_LENS.vfovDeg = 69` is the vertical field of view of a 9:16 frame, and it was
 * right for exactly as long as the capture delivered 9:16. Fixing the resolution ladder
 * changed the delivered frame to 4:3 (1080x1440) and nothing noticed: the lattice went on
 * separating its rows by 29 deg as though each frame were 69 deg tall, when it was 55-61.
 * Cross-row overlap fell from the designed 0.55 to 0.494, and rows that share too little
 * cannot be held rigid against each other — which arrives in the delivered panorama as
 * straight lines bending across a row boundary.
 */

const overlapAt = (vfovDeg: number, hfovDeg: number, separationDeg: number): number =>
  (vfovDeg - separationDeg - (hfovDeg / 2) * Math.tan((5 * Math.PI) / 180)) / vfovDeg;

describe("vfovForFrame", () => {
  it("reproduces the 69 deg constant for the 9:16 frame it was written for", () => {
    expect(vfovForFrame(43, 1080, 1920)).toBeCloseTo(70, 0);
    expect(vfovForFrame(43, 608, 1080)).toBeCloseTo(70, 0);
  });

  it("gives a materially shorter frame for 4:3, which is what went unnoticed", () => {
    expect(vfovForFrame(43, 1080, 1440)).toBeCloseTo(55.4, 1);
    // 14 deg below the constant the whole lattice was derived from.
    expect(DEFAULT_LENS.vfovDeg - vfovForFrame(43, 1080, 1440)).toBeGreaterThan(13);
  });

  it("does not care which way round the edges arrive", () => {
    expect(lensForFrame(1080, 1440).vfovDeg).toBeCloseTo(lensForFrame(1440, 1080).vfovDeg, 6);
  });

  it("falls back to the constant rather than returning nonsense for a degenerate frame", () => {
    expect(vfovForFrame(43, 0, 1440)).toBe(DEFAULT_LENS.vfovDeg);
    expect(vfovForFrame(43, 1080, 0)).toBe(DEFAULT_LENS.vfovDeg);
  });

  it("keeps the assumed hfov, which the browser cannot report", () => {
    expect(lensForFrame(1080, 1440).hfovDeg).toBe(DEFAULT_LENS.hfovDeg);
    expect(lensForFrame(1080, 1440).source).toBe("track-settings");
  });
});

describe("rowSeparationDeg", () => {
  it("is unchanged for the 9:16 lens every earlier number was measured against", () => {
    expect(Math.round(rowSeparationDeg(DEFAULT_LENS))).toBe(29);
  });

  it("closes the rows up on a 4:3 frame instead of pretending it is 9:16", () => {
    expect(Math.round(rowSeparationDeg(lensForFrame(1080, 1440)))).toBe(23);
  });

  it("holds the engine's 0.55 design overlap on every frame shape", () => {
    for (const [w, h] of [
      [1080, 1920],
      [1080, 1440],
      [1080, 1350],
      [608, 1080],
    ] as const) {
      const lens = lensForFrame(w, h);
      const overlap = overlapAt(lens.vfovDeg, lens.hfovDeg, rowSeparationDeg(lens));
      // 1e-9 of slack: the separation is derived from 0.45 = 1 - 0.55, so the round trip
      // lands on the design point to within floating-point noise, by construction.
      expect(overlap).toBeGreaterThanOrEqual(0.55 - 1e-9);
    }
  });

  it("never widens the rows to protect pole reach — that priority was backwards", () => {
    // The old floor was `60 - vfov/2`, and it OUTRANKED the overlap term: on a 61-deg frame
    // it pushed the separation UP to 29.5, spending overlap to keep the top row reaching
    // poles._CAP_DEG. A bent room is a defect; a synthesised ceiling is a trade the product
    // has already chosen to make.
    const narrow = { ...DEFAULT_LENS, vfovDeg: 61 };
    const oldFloor = 60 - narrow.vfovDeg / 2; // 29.5
    expect(rowSeparationDeg(narrow)).toBeLessThan(oldFloor);
    expect(overlapAt(narrow.vfovDeg, narrow.hfovDeg, rowSeparationDeg(narrow))).toBeGreaterThanOrEqual(0.55 - 1e-9);
  });

  it("still refuses to exceed the 31 deg ceiling on a very tall frame", () => {
    expect(rowSeparationDeg({ ...DEFAULT_LENS, vfovDeg: 100 })).toBe(31);
  });
});

// =====================================================================================

describe("a correct capture on a 4:3 phone", () => {
  /**
   * THE FAILURE THIS PREVENTS. `COMPLETENESS` was measured on a 9:16 frame, where three
   * rows reach +55.5/-71.5 and a perfect sweep scores 0.8856. A 4:3 frame is 14 deg
   * shorter, so the same three rows reach barely +45 and a PERFECT sweep on that phone
   * scores about 0.80 — under the 0.855 constant. Judged against a fixed number, every
   * correct capture on a 4:3 phone would have been refused for a shortfall the phone cannot
   * close at any skill level.
   */
  const lens = lensForFrame(1080, 1440);
  const samples = sweep({ lens });
  const grid = createGrid();
  for (const s of samples) stampCone(grid, s, lens);
  const stations = buildStations(makeConfig({ lens }));
  for (const station of stations) {
    for (const s of samples) {
      if (greatCircleDeg(s.yawDeg, s.pitchDeg, station.yawDeg, station.pitchDeg) <= 9) {
        station.filled = true;
        station.filledAt = s.t;
        break;
      }
    }
  }
  const r = report(grid, stations, lens);

  it("cannot reach what a 9:16 frame reaches — this is physics, not technique", () => {
    expect(r.pitchMaxDeg).toBeLessThan(COMPLETENESS.pitchMaxDeg);
    expect(r.coveredFraction).toBeLessThan(COMPLETENESS.coveredFraction);
  });

  it("is nevertheless graded complete, because it did everything it could", () => {
    expect(r.blockedBy).toEqual([]);
    expect(r.complete).toBe(true);
  });

  it("is still held to a real standard — a sweep that skips a sector fails", () => {
    const bad = sweep({ lens, skipSectors: [[120, 180]] });
    const badGrid = createGrid();
    for (const s of bad) stampCone(badGrid, s, lens);
    expect(report(badGrid, buildStations(makeConfig({ lens })), lens).complete).toBe(false);
  });

  it("loosens the gates and never tightens them", () => {
    const wide = gatesForLens(DEFAULT_LENS);
    const narrow = gatesForLens(lens);
    expect(narrow.coveredFraction).toBeLessThan(wide.coveredFraction);
    expect(narrow.pitchMaxDeg).toBeLessThan(wide.pitchMaxDeg);
    // Never TIGHTER than the shipped constants, on any lens. `pitchMaxDeg` comes out looser
    // even on 9:16: the constant was set at the 55.5 the plan nominally reaches, with no
    // room for the 3 deg of row-pitch error the coaching grades GREEN — so a green capture
    // could be blocked by `pitch_max` for a shortfall the screen had just approved.
    expect(wide.coveredFraction).toBeLessThanOrEqual(COMPLETENESS.coveredFraction);
    expect(wide.pitchMaxDeg).toBeLessThanOrEqual(COMPLETENESS.pitchMaxDeg);
    expect(wide.pitchMinDeg).toBeGreaterThanOrEqual(COMPLETENESS.pitchMinDeg);
  });
});

// =====================================================================================

describe("the drawn guidance and the real lattice", () => {
  /**
   * THE HAZARD THIS PINS. `ROW_PLAN[row].targetPitchDeg` are the 9:16 reference literals.
   * The stations are built from `rowTargetPitchDeg(row, lens)`, which follows the frame
   * shape. On a 4:3 phone those two answers are 6 deg apart — so a guidance ring drawn from
   * the literal sits six degrees away from the station it is supposed to be showing, which
   * is worse than drawing no ring at all. Both must come from the same function.
   */
  const lens = lensForFrame(1080, 1440);

  it("disagree by enough to mislead, which is why the literal must never be drawn", () => {
    for (const row of ["up", "down"] as const) {
      const drift = Math.abs(ROW_PLAN[row].targetPitchDeg - rowTargetPitchDeg(row, lens));
      expect(drift).toBeGreaterThanOrEqual(5);
    }
  });

  it("agree exactly once both come from rowTargetPitchDeg", () => {
    const stations = buildStations(makeConfig({ lens }));
    for (const row of ROW_IDS) {
      const drawn = rowTargetPitchDeg(row, lens);
      const station = stations.find((s) => s.row === row);
      expect(station?.pitchDeg).toBe(drawn);
    }
  });

  it("still agree on the 9:16 lens, where the literals happen to be right", () => {
    for (const row of ROW_IDS) {
      expect(rowTargetPitchDeg(row, DEFAULT_LENS)).toBe(ROW_PLAN[row].targetPitchDeg);
    }
  });
});
