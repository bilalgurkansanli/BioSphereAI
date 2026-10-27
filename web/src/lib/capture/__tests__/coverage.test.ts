import { describe, expect, it } from "vitest";

import {
  COMPLETENESS,
  ROW_BIAS_DEG,
  COVERAGE_GRID,
  DEFAULT_LENS,
  ROW_IDS,
  THRESHOLDS,
  type LensGeometry,
  type OrientationSample,
  type StationState,
} from "@/lib/capture/types";
import {
  buildFrustum,
  cellPitchCentreDeg,
  cellYawCentreDeg,
  clearGrid,
  coveredMaskBase64,
  createGrid,
  cyclicSpan,
  deserialize,
  directionFromYawPitch,
  findGaps,
  greatCircleDeg,
  insideFrustum,
  largestUncoveredArcDeg,
  nearestUncovered,
  report,
  rowCosWeights,
  rowTargetPitchDeg,
  rowYawSpanFromGrid,
  serialize,
  shortestYawDeltaDeg,
  stampCone,
  stampConeExhaustive,
  wrapSigned180,
  wrapYaw360,
} from "@/lib/capture/coverage";
import { buildStations } from "@/lib/capture/plan";
import {
  bandSphereFraction,
  frustumSphereFraction,
  makeConfig,
  makeOrientation,
  perfectSweep,
  singleRowSweep,
  sweep,
  sweepWithGap,
} from "./synthetic";

/** Solid-angle-weighted fill of a whole grid — the same sum `report` reports. */
function weightedFill(grid: ReturnType<typeof createGrid>): number {
  const w = rowCosWeights(grid.pitchCells, grid.cellDeg);
  let num = 0;
  let den = 0;
  for (let pi = 0; pi < grid.pitchCells; pi += 1) {
    for (let yi = 0; yi < grid.yawCells; yi += 1) num += w[pi] * grid.fill[pi * grid.yawCells + yi];
    den += w[pi] * grid.yawCells;
  }
  return num / den;
}

function stampAll(samples: OrientationSample[], lens: LensGeometry = DEFAULT_LENS) {
  const grid = createGrid();
  for (const s of samples) stampCone(grid, s, lens);
  return grid;
}

function filledStations(samples: OrientationSample[]): StationState[] {
  // A station is filled when the axis passed within the capture radius. The dwell clock is
  // `plan.updateStations`' business; here we only need coverage's view of "was it reached".
  const stations = buildStations(makeConfig());
  for (const station of stations) {
    for (const s of samples) {
      if (greatCircleDeg(s.yawDeg, s.pitchDeg, station.yawDeg, station.pitchDeg) <= 9) {
        station.filled = true;
        station.filledAt = s.t;
        break;
      }
    }
  }
  return stations;
}

// =====================================================================================

describe("angle primitives", () => {
  it("wraps yaw into [0, 360)", () => {
    expect(wrapYaw360(0)).toBe(0);
    expect(wrapYaw360(360)).toBe(0);
    expect(wrapYaw360(-1)).toBeCloseTo(359, 10);
    expect(wrapYaw360(725)).toBeCloseTo(5, 10);
  });

  it("wraps signed into (-180, 180]", () => {
    expect(wrapSigned180(190)).toBeCloseTo(-170, 10);
    expect(wrapSigned180(-190)).toBeCloseTo(170, 10);
    expect(wrapSigned180(180)).toBeCloseTo(180, 10);
  });

  it("takes the short way round the 0/360 seam", () => {
    expect(shortestYawDeltaDeg(359, 1)).toBeCloseTo(2, 10);
    expect(shortestYawDeltaDeg(1, 359)).toBeCloseTo(-2, 10);
    expect(shortestYawDeltaDeg(10, 200)).toBeCloseTo(-170, 10);
  });

  it("measures great-circle distance", () => {
    expect(greatCircleDeg(0, 0, 0, 0)).toBeCloseTo(0, 10);
    expect(greatCircleDeg(0, 0, 90, 0)).toBeCloseTo(90, 10);
    expect(greatCircleDeg(0, 0, 0, 90)).toBeCloseTo(90, 10);
    // 45 deg of yaw at the equator is 45 deg of arc; 45 deg of yaw at 60 deg elevation is
    // much less. This is the property the cos(elev) weighting exists to respect.
    expect(greatCircleDeg(0, 60, 45, 60)).toBeLessThan(45);
    expect(greatCircleDeg(0, 60, 45, 60)).toBeCloseTo(22.02, 1);
  });
});

describe("the hemisphere convention, pinned against the engine", () => {
  /**
   * `engine/biosphere/pano/geometry.py:309-333` decomposes a world->camera rotation with
   *
   *     d = R.T @ [0, 0, 1]
   *     yaw   = atan2(d[0], d[2])
   *     pitch = asin(-d[1])
   *
   * i.e. world **+Y is DOWN**, yaw increases toward +X, and positive pitch tips the optical
   * axis up. `coverage.directionFromYawPitch` builds the same directions with **+Y UP**, so
   * the two differ by a single Y flip and nothing else.
   *
   * Nothing today passes a direction across that boundary — `coverage.ts`,
   * `GuidanceOverlay.tsx` and the engine each build both their camera axis and their target
   * directions with their own one function, so every frustum test and great-circle distance
   * is an exact isometry inside each module. That is why the sign is correct despite three
   * modules using three bases. It is also why it is FRAGILE: the property holds by nobody
   * ever crossing the boundary, not by anything checking. These tests are the check.
   */
  const engineYawPitch = (d: readonly [number, number, number]) => {
    // The same direction expressed in the engine's Y-down world frame.
    const e = [d[0], -d[1], d[2]] as const;
    return {
      yawDeg: wrapYaw360((Math.atan2(e[0], e[2]) * 180) / Math.PI),
      pitchDeg: (Math.asin(Math.max(-1, Math.min(1, -e[1]))) * 180) / Math.PI,
    };
  };

  it("round-trips every (yaw, pitch) through the engine's own decomposition", () => {
    for (const yaw of [0, 5, 90, 137, 180, 271, 355]) {
      for (const pitch of [-89, -63.5, -29, 0, 29, 63.5, 89]) {
        const decoded = engineYawPitch(directionFromYawPitch(yaw, pitch));
        expect(decoded.pitchDeg).toBeCloseTo(pitch, 9);
        // Yaw is undefined at the poles, where cos(pitch) collapses the direction onto Y.
        if (Math.abs(pitch) < 89.5) expect(decoded.yawDeg).toBeCloseTo(wrapYaw360(yaw), 9);
      }
    }
  });

  it("agrees with the frustum's own optical axis, so the two can never disagree", () => {
    for (const [yaw, pitch] of [
      [0, 0],
      [137, 29],
      [312, -29],
      [45, 84],
    ]) {
      const fr = buildFrustum(yaw, pitch, 0, DEFAULT_LENS);
      const d = directionFromYawPitch(yaw, pitch);
      expect(fr.fx).toBeCloseTo(d[0], 12);
      expect(fr.fy).toBeCloseTo(d[1], 12);
      expect(fr.fz).toBeCloseTo(d[2], 12);
    }
  });

  it("puts POSITIVE pitch in the UP half of the grid, which is the whole point of the up row", () => {
    // The +25..+40 deg band is where all three measured coverage gaps sat. A hemisphere flip
    // here would paint the up row into the floor and report the capture as complete.
    const up = createGrid();
    stampCone(up, makeOrientation(0, 0, 29), DEFAULT_LENS);
    const down = createGrid();
    stampCone(down, makeOrientation(0, 0, -29), DEFAULT_LENS);

    const band = (grid: ReturnType<typeof createGrid>) => {
      let lo = Infinity;
      let hi = -Infinity;
      for (let pi = 0; pi < grid.pitchCells; pi += 1) {
        for (let yi = 0; yi < grid.yawCells; yi += 1) {
          if (grid.fill[pi * grid.yawCells + yi] <= 0) continue;
          const elev = cellPitchCentreDeg(pi, grid.cellDeg);
          lo = Math.min(lo, elev);
          hi = Math.max(hi, elev);
        }
      }
      return { lo, hi };
    };

    // 29 + 69/2 = 63.5 at the frame's centre column, so the up row touches the 60..65 band
    // and reaches no lower than about -5. Its mirror does the exact opposite.
    expect(band(up).hi).toBeCloseTo(62.5, 6);
    expect(band(down).lo).toBeCloseTo(-62.5, 6);
    expect(band(up).hi).toBeCloseTo(-band(down).lo, 6);
    expect(band(up).lo).toBeCloseTo(-band(down).hi, 6);
    expect(band(up).hi).toBeGreaterThan(COMPLETENESS.pitchMaxDeg);
    expect(band(down).lo).toBeLessThan(COMPLETENESS.pitchMinDeg);
  });

  it("indexes pitch from the nadir up, which is the OPPOSITE of the engine's row order", () => {
    // Not a bug, and worth pinning so nobody "fixes" one to match the other: the engine's
    // grid row 0 is +87.5 (`cell_elev = 90 - (i + 0.5) * 5` in coverage.py) and this one's is
    // -87.5. The two grids are never compared cell for cell — only `coveredFraction`, which
    // is a solid-angle-weighted sum and is invariant to row order — and `coveredMaskBase64`
    // ships this module's own convention with its own reader.
    expect(cellPitchCentreDeg(0)).toBeCloseTo(-87.5, 9);
    expect(cellPitchCentreDeg(35)).toBeCloseTo(87.5, 9);
    expect(cellPitchCentreDeg(0)).toBeLessThan(cellPitchCentreDeg(35));
  });

  it("keeps great-circle distance an isometry of the direction it builds", () => {
    // The property that lets `cellFill` cull by `greatCircleDeg` and then test membership
    // with dot products, without the two ever disagreeing about where a cell is.
    const dot = (a: readonly [number, number, number], b: readonly [number, number, number]) =>
      a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for (const [ya, pa, yb, pb] of [
      [0, 0, 90, 0],
      [10, 29, 350, -29],
      [137, 63.5, 200, -12],
    ]) {
      const angle = (Math.acos(Math.max(-1, Math.min(1, dot(directionFromYawPitch(ya, pa), directionFromYawPitch(yb, pb))))) * 180) / Math.PI;
      expect(angle).toBeCloseTo(greatCircleDeg(ya, pa, yb, pb), 9);
    }
  });
});

describe("frustum", () => {
  it("contains its own axis and excludes everything behind it", () => {
    const fr = buildFrustum(0, 0, 0, DEFAULT_LENS);
    expect(insideFrustum(fr, 0, 0)).toBe(true);
    expect(insideFrustum(fr, 180, 0)).toBe(false);
    expect(insideFrustum(fr, 90, 0)).toBe(false);
  });

  it("reaches hfov/2 across the short edge and vfov/2 down the long one", () => {
    const fr = buildFrustum(0, 0, 0, DEFAULT_LENS);
    expect(insideFrustum(fr, DEFAULT_LENS.hfovDeg / 2 - 0.1, 0)).toBe(true);
    expect(insideFrustum(fr, DEFAULT_LENS.hfovDeg / 2 + 0.1, 0)).toBe(false);
    expect(insideFrustum(fr, 0, DEFAULT_LENS.vfovDeg / 2 - 0.1)).toBe(true);
    expect(insideFrustum(fr, 0, DEFAULT_LENS.vfovDeg / 2 + 0.1)).toBe(false);
  });

  it("puts the centre column of an UP-row frame at exactly 55.5 deg — the whole plan", () => {
    // 21 + 69/2 = 55.5 on the ROW_BIAS_DEG = -8 lattice, and that reach is what
    // COMPLETENESS.pitchMaxDeg was retuned to. Built from the plan, not from a literal, so
    // moving the bias again moves this with it.
    const fr = buildFrustum(0, rowTargetPitchDeg("up"), 0, DEFAULT_LENS);
    expect(insideFrustum(fr, 0, 55.4)).toBe(true);
    expect(insideFrustum(fr, 0, 55.6)).toBe(false);
    expect(insideFrustum(fr, 0, COMPLETENESS.pitchMaxDeg)).toBe(true);
  });

  it("does not reach as high at the frame corners as at its centre column", () => {
    // The corner of a rectilinear 43x69 frame reaches 8 deg less elevation than its centre
    // column, at world yaw ~36. That shortfall is exactly why the measured
    // `coveredFraction` lands a hair under the analytic band fraction.
    const up = rowTargetPitchDeg("up");
    const fr = buildFrustum(0, up, 0, DEFAULT_LENS);
    const maxElevationAtYaw = (yawDeg: number): number => {
      let best = -90;
      for (let e = 0; e <= 90; e += 0.02) if (insideFrustum(fr, yawDeg, e)) best = e;
      return best;
    };
    // Found rather than hard-coded: the yaw a frame still reaches depends on where it is
    // pitched, so a literal here would have to move every time ROW_BIAS_DEG does.
    let edgeYaw = 0;
    for (let y = 0; y <= 89; y += 0.5) if (maxElevationAtYaw(y) > -90) edgeYaw = y;
    expect(maxElevationAtYaw(0)).toBeCloseTo(up + DEFAULT_LENS.vfovDeg / 2, 1);
    expect(edgeYaw).toBeGreaterThan(DEFAULT_LENS.hfovDeg / 2);
    // The claim is about the frame's own shape: corners reach materially lower than the
    // centre column, which is why measured coverage lands under the analytic band fraction.
    // 3.8 deg at this pitch; the shortfall shrinks as the frame is aimed nearer the horizon.
    expect(maxElevationAtYaw(edgeYaw)).toBeLessThan(maxElevationAtYaw(0) - 3);
  });

  it("is unaffected in area by the sign of roll", () => {
    const positive = createGrid();
    const negative = createGrid();
    stampCone(positive, makeOrientation(0, 90, 10, { rollDeg: 7 }));
    stampCone(negative, makeOrientation(0, 90, 10, { rollDeg: -7 }));
    expect(weightedFill(positive)).toBeCloseTo(weightedFill(negative), 4);
  });
});

describe("stampCone against the analytic solid angle", () => {
  it("integrates one frame to 4 asin(sin(hfov/2) sin(vfov/2)) / 4pi", () => {
    const grid = createGrid();
    stampCone(grid, makeOrientation(0, 180, 0), DEFAULT_LENS);
    const expected = frustumSphereFraction(DEFAULT_LENS.hfovDeg, DEFAULT_LENS.vfovDeg);
    expect(expected).toBeCloseTo(0.066561, 5);
    expect(weightedFill(grid)).toBeCloseTo(expected, 2);
  });

  it("integrates to the same solid angle at any elevation — the cos(elev) weighting works", () => {
    const expected = frustumSphereFraction(DEFAULT_LENS.hfovDeg, DEFAULT_LENS.vfovDeg);
    for (const pitch of [0, 29, -29, 55]) {
      const grid = createGrid();
      stampCone(grid, makeOrientation(0, 40, pitch), DEFAULT_LENS);
      expect(weightedFill(grid)).toBeCloseTo(expected, 2);
    }
  });

  it("integrates a narrow square frame correctly too", () => {
    const square: LensGeometry = { ...DEFAULT_LENS, hfovDeg: 40, vfovDeg: 40 };
    const grid = createGrid();
    stampCone(grid, makeOrientation(0, 200, -12), square);
    expect(weightedFill(grid)).toBeCloseTo(frustumSphereFraction(40, 40), 2);
  });

  it("agrees cell for cell with the un-culled reference implementation", () => {
    for (const [yaw, pitch, roll] of [
      [0, 0, 0],
      [137, 29, 4],
      [312, -29, -6],
      [45, 84, 0],
      [45, -88, 12],
    ]) {
      const fast = createGrid();
      const slow = createGrid();
      const sample = makeOrientation(0, yaw, pitch, { rollDeg: roll });
      stampCone(fast, sample);
      stampConeExhaustive(slow, sample);
      expect(Array.from(fast.fill)).toEqual(Array.from(slow.fill));
      expect(Array.from(fast.covered)).toEqual(Array.from(slow.covered));
    }
  });

  it("reports how many cells newly crossed the 50% line, and never double-counts", () => {
    const grid = createGrid();
    const sample = makeOrientation(0, 0, 0);
    const first = stampCone(grid, sample);
    expect(first).toBeGreaterThan(0);
    expect(stampCone(grid, sample)).toBe(0);
  });

  it("is monotone: fill never decreases", () => {
    const grid = createGrid();
    stampCone(grid, makeOrientation(0, 0, 0));
    const before = Array.from(grid.fill);
    stampCone(grid, makeOrientation(100, 180, 0));
    for (let i = 0; i < before.length; i += 1) expect(grid.fill[i]).toBeGreaterThanOrEqual(before[i]);
  });

  it("wraps across the 0/360 seam without leaving a stripe", () => {
    const grid = createGrid();
    stampCone(grid, makeOrientation(0, 0, 0));
    // Column 0 (yaw 0-5) and column 71 (yaw 355-360) both sit inside a frame aimed at 0.
    expect(grid.covered[Math.floor(grid.pitchCells / 2) * grid.yawCells + 0]).toBe(1);
    expect(grid.covered[Math.floor(grid.pitchCells / 2) * grid.yawCells + grid.yawCells - 1]).toBe(1);
  });
});

describe("the grid", () => {
  it("matches the memory budget the contract quotes", () => {
    const grid = createGrid();
    expect(grid.fill.length).toBe(COVERAGE_GRID.totalCells);
    expect(grid.covered.length).toBe(COVERAGE_GRID.totalCells);
    expect(grid.fill.byteLength + grid.covered.byteLength).toBe(COVERAGE_GRID.bytes);
  });

  it("indexes row-major with pitch 0 at -90 and yaw 0 at the capture origin", () => {
    expect(cellPitchCentreDeg(0)).toBeCloseTo(-87.5, 10);
    expect(cellPitchCentreDeg(35)).toBeCloseTo(87.5, 10);
    expect(cellYawCentreDeg(0)).toBeCloseTo(2.5, 10);
    expect(cellYawCentreDeg(71)).toBeCloseTo(357.5, 10);
  });

  it("clears in place", () => {
    const grid = createGrid();
    stampCone(grid, makeOrientation(0, 0, 0));
    clearGrid(grid);
    expect(weightedFill(grid)).toBe(0);
  });
});

// =====================================================================================

describe("a perfect three-row sweep", () => {
  const samples = perfectSweep();
  const grid = stampAll(samples);
  const stations = filledStations(samples);
  const r = report(grid, stations, DEFAULT_LENS);

  it("reproduces the contract's 0.8856 physical ceiling", () => {
    // ROW_BIAS_DEG puts the lattice at +21/-8/-37, so the photographed band is -71.5..+55.5.
    // Asymmetric, and worth 0.009 less than the symmetric +/-63.5 plan — the price of the
    // five degrees of floor the bias buys.
    expect(bandSphereFraction(55.5, 71.5)).toBeCloseTo(COMPLETENESS.coveredFractionCeiling, 2);
    expect(r.coveredFraction).toBeGreaterThan(0.885);
    expect(r.coveredFraction).toBeLessThanOrEqual(COMPLETENESS.coveredFractionCeiling);
    expect(r.coveredFraction).toBeCloseTo(COMPLETENESS.coveredFractionCeiling, 2);
  });

  it("clears the gate, which is the point of not setting it at 0.90", () => {
    expect(r.coveredFraction).toBeGreaterThanOrEqual(COMPLETENESS.coveredFraction);
  });

  it("reaches the frame-edge elevation the pole filler needs", () => {
    expect(r.pitchMaxDeg).toBeGreaterThanOrEqual(COMPLETENESS.pitchMaxDeg);
    expect(r.pitchMinDeg).toBeLessThanOrEqual(COMPLETENESS.pitchMinDeg);
    // Up reach 21 + vfov/2 = 55.5, five degrees under poles._CAP_DEG = 60: the ring above
    // it is closed by fill_uncovered's equirect pass instead of the conformal cap. Down
    // reach gains those five degrees, which is what ROW_BIAS_DEG was moved for.
    expect(r.pitchMaxDeg).toBeCloseTo(55.5, 0);
    expect(r.pitchMinDeg).toBeCloseTo(-71.5, 0);
  });

  it("covers the horizon bands the viewer actually looks at", () => {
    expect(r.horizonFraction30).toBeGreaterThanOrEqual(COMPLETENESS.horizonFraction30);
    expect(r.horizonFraction60).toBeGreaterThanOrEqual(COMPLETENESS.horizonFraction60);
  });

  it("goes all the way round in every row", () => {
    for (const row of ROW_IDS) {
      expect(r.rows[row].yawSpanDeg).toBeCloseTo(360, 6);
      expect(r.rows[row].complete).toBe(true);
    }
  });

  it("leaves no gaps and no blockers", () => {
    expect(r.gaps).toEqual([]);
    expect(r.blockedBy).toEqual([]);
    expect(r.complete).toBe(true);
  });
});

describe("a level-row-only sweep", () => {
  const samples = singleRowSweep();
  const grid = stampAll(samples);
  const r = report(grid, filledStations(samples), DEFAULT_LENS);

  it("cannot physically pass 0.59 of the sphere — sin(vfov/2)", () => {
    expect(r.coveredFraction).toBeCloseTo(bandSphereFraction(26.5, 42.5), 1);
    expect(r.coveredFraction).toBeLessThan(0.6);
  });

  it("leaves an unfillable ring: reach 26.5 is 33.5 deg short of poles._CAP_DEG", () => {
    expect(r.pitchMaxDeg).toBeCloseTo(26.5, 0);
    expect(r.pitchMaxDeg).toBeLessThan(COMPLETENESS.pitchMaxDeg);
    expect(r.blockedBy).toContain("pitch_max");
    expect(r.blockedBy).toContain("pitch_min");
    expect(r.blockedBy).toContain("covered_fraction");
    expect(r.complete).toBe(false);
  });

  it("no longer covers the horizon band on its own — ROW_BIAS_DEG moved it off the equator", () => {
    // At bias -8 the level row spans -42.5..+26.5, so the top of the |elev| <= 30 band it
    // used to own outright is now the up row's job. Worth pinning: it is the first thing
    // that breaks if the bias is pushed further, and the band is what the viewer looks at.
    expect(r.horizonFraction30).toBeCloseTo(0.946, 2);
    expect(r.horizonFraction30).toBeLessThan(COMPLETENESS.horizonFraction30);
    expect(r.rows.level.yawSpanDeg).toBeCloseTo(360, 6);
    expect(r.rows.up.filledStations).toBe(0);
    expect(r.blockedBy).toContain("row_up");
    expect(r.blockedBy).toContain("row_down");
  });

  it("does not report the whole missing sky as a gap", () => {
    // coverage.py's _SCOPE_MARGIN_DEG rule: gaps live inside the band the capture
    // ATTEMPTED. Reporting a ring spanning every direction would bury the actionable fact.
    for (const gap of r.gaps) {
      expect(gap.pitchEndDeg).toBeLessThanOrEqual(35);
      expect(gap.pitchStartDeg).toBeGreaterThanOrEqual(-35);
    }
  });
});

describe("a sweep that skips a sector", () => {
  const samples = sweepWithGap(120, 180);
  const grid = stampAll(samples);
  const r = report(grid, filledStations(samples), DEFAULT_LENS);

  it("is found by findGaps directly, in the same order the report carries", () => {
    const direct = findGaps(grid);
    expect(direct).toEqual(r.gaps);
    // Largest first, by solid angle — "the corner behind you" before a two-cell speck.
    for (let i = 1; i < direct.length; i += 1) {
      expect(direct[i - 1].solidAngleFraction).toBeGreaterThanOrEqual(direct[i].solidAngleFraction);
    }
    // Single-cell specks are not instructions; COMPLETENESS.maxGapArcDeg tolerates one cell.
    for (const gap of direct) {
      const cells = ((gap.yawEndDeg - gap.yawStartDeg) / 5) * ((gap.pitchEndDeg - gap.pitchStartDeg) / 5);
      expect(cells).toBeGreaterThan(1);
    }
  });

  it("finds nothing to chase on a perfect sweep", () => {
    expect(findGaps(stampAll(perfectSweep()))).toEqual([]);
  });

  it("finds the missing sector and puts it first", () => {
    expect(r.gaps.length).toBeGreaterThan(0);
    const largest = r.gaps[0];
    // A 43-deg-wide frame still reaches 21.5 deg into the skipped sector from each side,
    // so the hole is the middle ~17 deg of it.
    expect(largest.yawStartDeg).toBeGreaterThanOrEqual(120);
    expect(largest.yawEndDeg).toBeLessThanOrEqual(180);
    expect(largest.solidAngleFraction).toBeGreaterThan(0);
  });

  it("names a row that would fix it", () => {
    expect(ROW_IDS).toContain(r.gaps[0].nearestRow);
  });

  it("refuses to call the capture complete", () => {
    expect(r.complete).toBe(false);
    expect(r.blockedBy.length).toBeGreaterThan(0);
  });

  it("shows the hole in every row's yaw span", () => {
    for (const row of ROW_IDS) expect(r.rows[row].yawSpanDeg).toBeLessThan(360);
  });

  it("points the nearest-uncovered arrow into the hole", () => {
    const nearest = nearestUncovered(grid, 150, 0);
    expect(nearest).not.toBeNull();
    if (nearest !== null) {
      expect(nearest.yawDeg).toBeGreaterThan(120);
      expect(nearest.yawDeg).toBeLessThan(180);
    }
  });
});

describe("a sweep that stops short", () => {
  it("reports the un-swept arc as missing yaw span", () => {
    const samples = sweep({ yawSpanDeg: 200 });
    const grid = stampAll(samples);
    const r = report(grid, filledStations(samples), DEFAULT_LENS);
    // 200 deg of travel plus 21.5 deg of half-frame at each end ~= 243 deg of arc.
    expect(r.rows.level.yawSpanDeg).toBeGreaterThan(200);
    expect(r.rows.level.yawSpanDeg).toBeLessThan(280);
    expect(r.blockedBy).toContain("yaw_span_level");
    expect(r.complete).toBe(false);
  });
});

// =====================================================================================

describe("cyclic helpers, mirrored from coverage.py", () => {
  it("finds the largest uncovered arc across the seam", () => {
    const cols = new Uint8Array(72).fill(1);
    cols[0] = 0;
    cols[1] = 0;
    cols[71] = 0;
    // Three consecutive missing columns, wrapping: 3/72 of the circle.
    expect(largestUncoveredArcDeg(cols)).toBeCloseTo(15, 6);
  });

  it("returns 0 for a full ring and 360 for an empty one", () => {
    expect(largestUncoveredArcDeg(new Uint8Array(72).fill(1))).toBe(0);
    expect(largestUncoveredArcDeg(new Uint8Array(72))).toBe(360);
  });

  it("spans a wrap-around column set", () => {
    const cols = new Array<boolean>(8).fill(false);
    cols[7] = true;
    cols[0] = true;
    cols[1] = true;
    expect(cyclicSpan(cols)).toEqual([7, 3]);
  });

  it("derives a row's yaw span from the grid the way the engine does", () => {
    const grid = createGrid();
    for (let yaw = 0; yaw < 200; yaw += 5) stampCone(grid, makeOrientation(0, yaw, 0));
    const span = rowYawSpanFromGrid(grid, 0);
    expect(span).toBeGreaterThan(200);
    expect(span).toBeLessThan(260);
  });
});

describe("row geometry", () => {
  it("puts the default lens at +21 / -8 / -37", () => {
    expect(rowTargetPitchDeg("level")).toBe(ROW_BIAS_DEG);
    expect(rowTargetPitchDeg("up")).toBe(21);
    expect(rowTargetPitchDeg("down")).toBe(-37);
  });

  it("keeps the separation the overlap arithmetic needs while biasing the whole lattice", () => {
    // The bias must not eat into cross-row overlap: it shifts all three rows together, so
    // the gap between them is still exactly rowSeparationDeg.
    expect(rowTargetPitchDeg("up") - rowTargetPitchDeg("level")).toBe(29);
    expect(rowTargetPitchDeg("level") - rowTargetPitchDeg("down")).toBe(29);
  });

  it("recomputes separation for a narrower lens instead of hard-coding 29", () => {
    const narrow: LensGeometry = { ...DEFAULT_LENS, vfovDeg: 66 };
    // 28 of separation, still biased 8 down: the two are independent by design.
    expect(rowTargetPitchDeg("up", narrow)).toBe(20);
    expect(rowTargetPitchDeg("up", narrow) - rowTargetPitchDeg("level", narrow)).toBe(28);
    // The admissible window is only [27, 30]; 29 on a 66-deg lens would push cross-row
    // overlap under the engine's 0.55 design point.
    expect(rowTargetPitchDeg("up", narrow) - rowTargetPitchDeg("level", narrow)).toBeLessThan(29);
  });
});

describe("persistence", () => {
  it("round-trips a grid through a snapshot without sharing buffers", () => {
    const grid = stampAll(sweep({ rows: ["level"] }));
    const stations = buildStations(makeConfig());
    stations[0].filled = true;
    const rows = report(grid, stations).rows;
    const snapshot = serialize(grid, stations, rows, DEFAULT_LENS);
    const restored = deserialize(snapshot);

    expect(Array.from(restored.grid.fill)).toEqual(Array.from(grid.fill));
    expect(Array.from(restored.grid.covered)).toEqual(Array.from(grid.covered));
    expect(restored.stations[0].filled).toBe(true);

    // Mutating the live grid after serialising must not touch the snapshot.
    stampCone(grid, makeOrientation(0, 0, 80));
    const second = deserialize(snapshot);
    expect(Array.from(second.grid.fill)).toEqual(Array.from(restored.grid.fill));
  });

  it("encodes the covered mask for the telemetry sidecar", () => {
    const grid = stampAll(sweep({ rows: ["level"] }));
    const encoded = coveredMaskBase64(grid);
    expect(encoded.length).toBeGreaterThan(0);
    expect(Buffer.from(encoded, "base64").length).toBe(COVERAGE_GRID.totalCells);
  });
});

describe("degenerate input", () => {
  it("ignores a NaN sample rather than poisoning the grid", () => {
    const grid = createGrid();
    stampCone(grid, makeOrientation(0, Number.NaN, 0));
    expect(weightedFill(grid)).toBe(0);
  });

  it("survives a sample pointed straight up, where Euler degenerates", () => {
    const grid = createGrid();
    expect(() => stampCone(grid, makeOrientation(0, 0, 90))).not.toThrow();
    expect(grid.covered[(grid.pitchCells - 1) * grid.yawCells]).toBe(1);
  });

  it("reports zeroes for an empty grid instead of NaN", () => {
    const r = report(createGrid(), [], DEFAULT_LENS);
    expect(r.coveredFraction).toBe(0);
    expect(r.horizonFraction30).toBe(0);
    expect(r.pitchMaxDeg).toBe(0);
    expect(Number.isNaN(r.coveredFraction)).toBe(false);
    expect(r.complete).toBe(false);
  });
});

// =====================================================================================
// The gate against what the coaching certifies
// =====================================================================================
//
// The capture screen and the review screen must agree. The screen grades a row GREEN while
// its elevation is within `THRESHOLDS.rowPitch.greenDeg`, does not even warn until
// `amberDeg`, and auto-advances each row on station count alone. If the coverage gate is
// stricter than that, a capture the app praised all the way through is refused at the end,
// with nothing on the review naming an error the user was never told about.
//
// This is the "hala çekmediğin yerler var" report, and it is what the 0.88 -> 0.868 change
// exists to close. The numbers below are measured against the shipped coverage code.

describe("the coverage gate and the coaching", () => {
  const coveredAt = (rowBiasDeg: number, pitchWobbleDeg = 0) => {
    const samples = sweep({ rowBiasDeg, pitchWobbleDeg });
    return report(stampAll(samples), filledStations(samples), DEFAULT_LENS);
  };

  it("passes a row held as far off as the screen is willing to call green", () => {
    const green = coveredAt(THRESHOLDS.rowPitch.greenDeg);
    expect(green.coveredFraction).toBeGreaterThanOrEqual(COMPLETENESS.coveredFraction);
  });

  it("still passes a green row carried on an ordinary unsteady hand", () => {
    const wobbly = coveredAt(THRESHOLDS.rowPitch.greenDeg, 3);
    expect(wobbly.coveredFraction).toBeGreaterThanOrEqual(COMPLETENESS.coveredFraction);
  });

  it("loses roughly 0.008 of the sphere per degree the rows are held low", () => {
    const slope = coveredAt(0).coveredFraction - coveredAt(1).coveredFraction;
    expect(slope).toBeCloseTo(0.00806, 3);
  });

  it("still refuses a row far enough off that the pole filler cannot repair it", () => {
    const bad = coveredAt(6);
    expect(bad.coveredFraction).toBeLessThan(COMPLETENESS.coveredFraction);
    expect(bad.complete).toBe(false);
  });
});
