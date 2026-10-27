import { describe, expect, it } from "vitest";

import {
  COMPLETENESS,
  DEFAULT_LENS,
  DURATION_BUDGET,
  RECORDING,
  ROW_IDS,
  ROW_PLAN,
  STATION_RULES,
  THRESHOLDS,
  YAW_SPAN,
  type CaptureTelemetry,
  type CoverageReport,
  type DetectorReadout,
  type RowId,
  type StationState,
  ROW_BIAS_DEG,
} from "@/lib/capture/types";
import { createGrid, report, stampCone, rowTargetPitchDeg, greatCircleDeg } from "@/lib/capture/coverage";
import {
  CAPTURE_COUNTDOWN_MS,
  MAX_CAPTURE_MS,
  PIVOT_MEDIAN_LIMIT_M,
  PIVOT_P90_LIMIT_M,
  accumulateYaw,
  blockers,
  buildBonusStations,
  armRow,
  buildStations,
  captureClock,
  forgetStationClock,
  framingDecision,
  isRowComplete,
  isStalled,
  nextRow,
  nextTarget,
  stationsForState,
  orbiting,
  railRowProgress,
  recordedFrameSize,
  rowDurationSeconds,
  rowFraction,
  rowFractions,
  rowIndex,
  rowProgress,
  stationId,
  totalDurationSeconds,
  updateStations,
  verdict,
  WALKING_TIME_FRACTION,
  type FramingInput,
} from "@/lib/capture/plan";
import { makeConfig, makeOrientation, perfectSweep, singleRowSweep, sweep, sweepWithGap } from "./synthetic";

const config = makeConfig();

/** Aim at a station and hold, in `stepMs` ticks, for `holdMs`. Returns the ids that filled. */
/** The level row is at ROW_BIAS_DEG, not on the equator: dwelling ON it means pitching there. */
const LEVEL = rowTargetPitchDeg("level");

function dwell(stations: StationState[], yawDeg: number, pitchDeg: number, startAt: number, holdMs: number, stepMs = 50, blocked = false) {
  const filled: string[] = [];
  for (let t = startAt; t <= startAt + holdMs; t += stepMs) {
    filled.push(...updateStations(stations, makeOrientation(t, yawDeg, pitchDeg), t, blocked));
  }
  return filled;
}

/** Run a whole synthetic sweep through the station machinery at a realistic tick rate. */
function runSweep(samples: ReturnType<typeof perfectSweep>, stations: StationState[]) {
  for (const s of samples) updateStations(stations, s, s.t, false);
  return stations;
}

// =====================================================================================

describe("the station lattice", () => {
  const stations = buildStations(config);

  it("is three rows of 24 at 15-deg spacing", () => {
    expect(stations).toHaveLength(72);
    for (const row of ROW_IDS) {
      const inRow = stations.filter((s) => s.row === row);
      expect(inRow).toHaveLength(ROW_PLAN[row].stationCount);
      expect(new Set(inRow.map((s) => s.yawDeg)).size).toBe(24);
    }
  });

  it("spaces stations exactly 3 coverage cells apart, so both share one lattice", () => {
    expect(ROW_PLAN.level.stationSpacingDeg).toBe(15);
    expect(ROW_PLAN.level.stationSpacingDeg % 5).toBe(0);
    const level = stations.filter((s) => s.row === "level").sort((a, b) => a.index - b.index);
    for (let i = 0; i < level.length; i += 1) expect(level[i].yawDeg).toBeCloseTo(i * 15, 10);
  });

  it("reads row elevations from the plan, never from a literal", () => {
    expect(stations.find((s) => s.row === "level")?.pitchDeg).toBe(ROW_BIAS_DEG);
    expect(stations.find((s) => s.row === "up")?.pitchDeg).toBe(21);
    expect(stations.find((s) => s.row === "down")?.pitchDeg).toBe(-37);
    const narrow = buildStations(makeConfig({ lens: { ...DEFAULT_LENS, vfovDeg: 66 } }));
    expect(narrow.find((s) => s.row === "up")?.pitchDeg).toBe(20);
  });

  it("leaves no dead band between adjacent stations", () => {
    // Worst case: 7.5 deg of yaw plus the 5 deg amber pitch tolerance = 9.01 deg, which is
    // why captureRadiusDeg is 9.
    expect(Math.hypot(15 / 2, THRESHOLDS.rowPitch.amberDeg)).toBeCloseTo(9.01, 2);
    expect(STATION_RULES.captureRadiusDeg).toBe(9);
  });

  it("locks every row that is not its turn", () => {
    // THE REPORT THIS PINS: "after eye level it started me from below, then carried on
    // upward — it got confusing." Two mechanisms, one symptom.
    //
    // Every row used to be handed to `updateStations` and `nextTarget` at all times. The
    // frame is 69 deg tall, so tilting between rows banked stations 29 deg away and a row
    // arrived already part done; and `nextTarget` only PREFERS the active row, so once the
    // nearby level stations were filled the bright ring crossed into a row the overlay was
    // not drawing. The reticle pointed at something invisible, in a row nobody had asked for.
    const all = buildStations(config);
    const rows = {
      level: all.filter((s) => s.row === "level"),
      up: all.filter((s) => s.row === "up"),
      down: all.filter((s) => s.row === "down"),
    };

    for (const [state, expected] of [
      ["row_level", "level"],
      ["row_up", "up"],
      ["row_down", "down"],
    ] as const) {
      const offered = stationsForState(state, rows, all);
      expect(offered).toHaveLength(ROW_PLAN[expected].stationCount);
      expect(offered.every((s) => s.row === expected)).toBe(true);
      // The SAME array each time, not a fresh filter: `updateStations` keys its dwell clock
      // on identity, and a new array per tick holds every station at zero dwell for ever.
      expect(stationsForState(state, rows, all)).toBe(offered);
    }

    // States with no row of their own still see the whole lattice — touch-up picks its own
    // set, and the review needs every station to report against.
    expect(stationsForState("touch_up", rows, all)).toBe(all);
    expect(stationsForState("review", rows, all)).toBe(all);
  });

  it("cannot fill or aim at another row while a row is being swept", () => {
    const all = buildStations(config);
    const rows = {
      level: all.filter((s) => s.row === "level"),
      up: all.filter((s) => s.row === "up"),
      down: all.filter((s) => s.row === "down"),
    };
    const levelOnly = stationsForState("row_level", rows, all);

    // Look straight at an UP station for far longer than the dwell requires, while the
    // machine says the level row is active.
    const upStation = rows.up[4];
    for (let t = 0; t <= 3000; t += 50) {
      updateStations(levelOnly, makeOrientation(t, upStation.yawDeg, upStation.pitchDeg), t, false);
    }
    expect(all.filter((s) => s.row === "up" && s.filled)).toHaveLength(0);

    // ...and the contrast that shows the lock is load-bearing rather than decorative: hand
    // the same samples the WHOLE lattice, which is what used to happen, and the up station
    // fills while the user is still being asked for the level row.
    const unlocked = buildStations(config);
    for (let t = 0; t <= 3000; t += 50) {
      updateStations(unlocked, makeOrientation(t, upStation.yawDeg, upStation.pitchDeg), t, false);
    }
    expect(unlocked.filter((s) => s.row === "up" && s.filled).length).toBeGreaterThan(0);

    // ...and the bright ring can only ever name a level station.
    const target = nextTarget(levelOnly, makeOrientation(4000, upStation.yawDeg, upStation.pitchDeg), "level");
    expect(target?.station.row).toBe("level");
  });

  it("gives every station a stable id", () => {
    expect(stationId("up", 7)).toBe("up:7");
    expect(new Set(stations.map((s) => s.id)).size).toBe(72);
  });

  it("builds the four opt-in pole reticles", () => {
    const poles = buildBonusStations();
    expect(poles).toHaveLength(4);
    expect(poles.filter((p) => p.pitchDeg === 90)).toHaveLength(2);
    expect(poles.filter((p) => p.pitchDeg === -90)).toHaveLength(2);
  });
});

describe("station dwell", () => {
  it("fills only after STATION_RULES.dwellMs inside the capture radius", () => {
    const stations = buildStations(config);
    expect(dwell(stations, 0, LEVEL, 0, STATION_RULES.dwellMs - 100)).toHaveLength(0);
    expect(dwell(stations, 0, LEVEL, 1000, STATION_RULES.dwellMs + 100).length).toBeGreaterThan(0);
    expect(stations.find((s) => s.id === "level:0")?.filled).toBe(true);
  });

  it("cannot be banked by a whip-pan", () => {
    const stations = buildStations(config);
    // One tick per station, 20 ms apart: 24 stations crossed in half a second.
    for (let i = 0; i < 24; i += 1) {
      updateStations(stations, makeOrientation(i * 20, i * 15, 0), i * 20, false);
    }
    expect(stations.filter((s) => s.filled)).toHaveLength(0);
  });

  it("resets dwell when the axis leaves the radius", () => {
    const stations = buildStations(config);
    dwell(stations, 0, LEVEL, 0, 200);
    expect(stations.find((s) => s.id === "level:0")?.dwellMs).toBeGreaterThan(0);
    updateStations(stations, makeOrientation(300, 90, 0), 300, false);
    expect(stations.find((s) => s.id === "level:0")?.dwellMs).toBe(0);
  });

  it("refuses a station the phone reached from the wrong elevation", () => {
    // `captureRadiusDeg` is 9.01, derived as sqrt(7.5^2 + 5^2): 7.5 deg of yaw slack plus
    // 5 deg of pitch. The great-circle test alone would let the whole 9 go into pitch, so a
    // row held 8 deg low fills every ring while `pitch_max` and `coveredFraction` fail on
    // the same capture. A filled ring has to mean the elevation the gates measure.
    const stations = buildStations(config);
    const upStation = stations.find((s) => s.id === "up:0");
    const upPitch = rowTargetPitchDeg("up");
    expect(upStation?.pitchDeg).toBe(upPitch);

    dwell(stations, 0, upPitch - THRESHOLDS.rowPitch.amberDeg - 3, 0, 2000);
    expect(upStation?.filled).toBe(false);
  });

  it("keeps filling through a whole sweep carried on a wobbling hand", () => {
    // THE REGRESSION THIS PINS, and it is the one that would have cost a real capture.
    //
    // The off-row pitch test was first folded into the branch that ZEROES `dwellMs`. That
    // turned the 250 ms dwell into a demand for 250 ms of CONTINUOUSLY in-band pitch inside
    // a station's 750 ms yaw window, and a hand carries about 6 deg of wobble at around
    // 1.4 Hz, so a single out-of-band sample erased the clock. Measured: ONE station of 72
    // filled on a perfectly aimed sweep. Nothing on screen said why — `row_pitch` needs
    // 500 ms of sustained error to latch and the excursions are 130-200 ms — so the user
    // would have turned through the room watching nothing light up.
    //
    // Off-row now PAUSES the clock instead of resetting it.
    const stations = buildStations(config);
    let t = 0;
    const stepMs = 33.3;
    for (let yaw = 0; yaw <= 370; yaw += (24 * stepMs) / 1000) {
      const wobble = 6 * Math.sin(2 * Math.PI * 1.4 * (t / 1000));
      updateStations(stations, makeOrientation(t, yaw % 360, LEVEL + wobble), t, false);
      t += stepMs;
    }
    const level = stations.filter((s) => s.row === "level" && s.filled).length;
    expect(level).toBe(ROW_PLAN.level.stationCount);
  });

  it("still refuses a row genuinely held off the plan, so the screen shows the problem", () => {
    // The rule has to keep its point: a ring that lights while the elevation is wrong is a
    // ring that lies, and the review would then refuse a capture the screen had certified.
    const stations = buildStations(config);
    let t = 0;
    const stepMs = 33.3;
    for (let yaw = 0; yaw <= 370; yaw += (24 * stepMs) / 1000) {
      // The up row held nine degrees low, with a steady hand.
      updateStations(stations, makeOrientation(t, yaw % 360, rowTargetPitchDeg("up") - 9), t, false);
      t += stepMs;
    }
    expect(stations.filter((s) => s.row === "up" && s.filled)).toHaveLength(0);
  });

  it("still fills through the pitch wobble an ordinary hand carries", () => {
    // The tolerance is the amber rung, not the green one: a hand wanders about 6 deg, and a
    // row that can never fill is worse than a row that fills a degree low.
    const stations = buildStations(config);
    for (let t = 0; t <= 2000; t += 50) {
      const wobble = 2.5 * Math.sin((t / 250) * Math.PI);
      updateStations(stations, makeOrientation(t, 0, rowTargetPitchDeg("up") + wobble), t, false);
    }
    expect(stations.find((s) => s.id === "up:0")?.filled).toBe(true);
  });

  it("exempts the pole reticles, where every direction is the row", () => {
    const poles = buildBonusStations();
    const zenith = poles.find((s) => s.pitchDeg > 80);
    expect(zenith).toBeDefined();
    for (let t = 0; t <= 2000; t += 50) {
      updateStations(poles, makeOrientation(t, zenith!.yawDeg, 84), t, false);
    }
    expect(zenith?.filled).toBe(true);
  });

  it("refuses to bank a station while a stop-severity warning is active", () => {
    const stations = buildStations(config);
    expect(dwell(stations, 0, LEVEL, 0, 2000, 50, true)).toHaveLength(0);
    expect(stations.find((s) => s.id === "level:0")?.filled).toBe(false);
    // ...and starts from zero once it clears, not from where it was.
    expect(stations.find((s) => s.id === "level:0")?.dwellMs).toBe(0);
  });

  it("clamps dt so a backgrounded tab cannot fill a station on the first tick back", () => {
    const stations = buildStations(config);
    updateStations(stations, makeOrientation(0, 0, 0), 0, false);
    // 30 seconds later, one tick. A naive dt would award 30 000 ms of dwell.
    updateStations(stations, makeOrientation(30_000, 0, 0), 30_000, false);
    expect(stations.find((s) => s.id === "level:0")?.filled).toBe(false);
    expect(stations.find((s) => s.id === "level:0")?.dwellMs).toBeLessThanOrEqual(STATION_RULES.dwellMs);
  });

  it("ignores a NaN heading rather than filling everything", () => {
    const stations = buildStations(config);
    updateStations(stations, makeOrientation(0, 0, 0), 0, false);
    const ids = updateStations(stations, makeOrientation(500, Number.NaN, 0), 500, false);
    expect(ids).toHaveLength(0);
  });

  it("forgets its clock on demand, so a retake cannot inherit a stale dt", () => {
    const stations = buildStations(config);
    updateStations(stations, makeOrientation(0, 0, 0), 0, false);
    forgetStationClock(stations);
    updateStations(stations, makeOrientation(10_000, 0, 0), 10_000, false);
    expect(stations.find((s) => s.id === "level:0")?.dwellMs).toBe(0);
  });
});

describe("a coached sweep fills the lattice", () => {
  it("fills at least the required stations in every row at 24 deg/s", () => {
    const stations = runSweep(perfectSweep(), buildStations(config));
    for (const row of ROW_IDS) {
      const filled = stations.filter((s) => s.row === row && s.filled).length;
      expect(filled).toBeGreaterThanOrEqual(ROW_PLAN[row].requiredStations);
    }
  });

  it("leaves the skipped sector's stations empty", () => {
    const stations = runSweep(sweepWithGap(120, 180), buildStations(config));
    const inSector = stations.filter((s) => s.row === "level" && s.yawDeg >= 125 && s.yawDeg <= 175);
    expect(inSector.length).toBeGreaterThan(0);
    for (const s of inSector) expect(s.filled).toBe(false);
  });

  it("fills nothing in the rows a single-row capture never attempted", () => {
    const stations = runSweep(singleRowSweep(), buildStations(config));
    expect(stations.filter((s) => s.row === "up" && s.filled)).toHaveLength(0);
    expect(stations.filter((s) => s.row === "down" && s.filled)).toHaveLength(0);
  });
});

describe("the bright ring", () => {
  const stations = buildStations(config);

  it("prefers the active row over a geometrically nearer station elsewhere", () => {
    // Pointing at (0, 14) is 14 deg from level:0 and 15 deg from up:0 — but the UP row is
    // the active one, so the ring must not invite the user back down to level.
    const target = nextTarget(stations, makeOrientation(0, 0, 14), "up", DEFAULT_LENS);
    expect(target?.station.row).toBe("up");
  });

  it("falls back to any row once the active row is done", () => {
    const done = buildStations(config).map((s) => (s.row === "level" ? { ...s, filled: true } : s));
    const target = nextTarget(done, makeOrientation(0, 0, 0), "level", DEFAULT_LENS);
    expect(target).not.toBeNull();
    expect(target?.station.row).not.toBe("level");
  });

  it("returns null when everything is filled", () => {
    const done = buildStations(config).map((s) => ({ ...s, filled: true }));
    expect(nextTarget(done, makeOrientation(0, 0, 0), "level", DEFAULT_LENS)).toBeNull();
  });

  it("signs the yaw delta so positive means turn right, across the seam", () => {
    const target = nextTarget(stations, makeOrientation(0, 359, 0), "level", DEFAULT_LENS);
    expect(target?.station.id).toBe("level:0");
    expect(target?.yawDeltaDeg).toBeCloseTo(1, 6);

    const back = nextTarget(stations, makeOrientation(0, 1, 0), "level", DEFAULT_LENS);
    expect(back?.yawDeltaDeg).toBeLessThanOrEqual(0);
  });

  it("uses the real frustum for onScreen, not a bounding box", () => {
    const near = nextTarget(stations, makeOrientation(0, 2, 0), "level", DEFAULT_LENS);
    expect(near?.onScreen).toBe(true);

    // Leave exactly one station open, at yaw 30. Standing at yaw 0 it is 30 deg away:
    // inside vfov/2 (34.5) but well outside hfov/2 (21.5), so it is off the short edge of
    // a portrait frame and the edge chevron has to appear. A bounding-box test that only
    // compared against vfov would call this on-screen and drop the chevron.
    const oneOpen = buildStations(config).map((s) => ({ ...s, filled: s.id !== "level:2" }));
    const far = nextTarget(oneOpen, makeOrientation(0, 0, ROW_BIAS_DEG), "level", DEFAULT_LENS);
    expect(far?.station.id).toBe("level:2");
    // A 30-deg yaw gap measured along the level row, which ROW_BIAS_DEG moved off the
    // equator: a parallel of latitude is shorter than a great circle, so the angular
    // distance is 29.70, not 30. Still between hfov/2 and vfov/2, which is the point.
    expect(far?.distanceDeg).toBeCloseTo(29.70, 2);
    expect(far?.distanceDeg).toBeLessThan(DEFAULT_LENS.vfovDeg / 2);
    expect(far?.distanceDeg).toBeGreaterThan(DEFAULT_LENS.hfovDeg / 2);
    expect(far?.onScreen).toBe(false);
  });

  it("always sends the user clockwise, even when the nearer gap is behind them", () => {
    // The complaint this fixes: the ring pointed LEFT on the level row and RIGHT on the row
    // after it, because "nearest unfilled station" is symmetric and simply follows whichever
    // gap happens to be closer. A guide that reverses between rows has to be re-read
    // mid-turn, and a row swept twice in opposite directions stitches badly.
    const open = new Set(["level:22", "level:4"]); // 330 deg behind, 60 deg ahead
    const stations = buildStations(config).map((s) => ({ ...s, filled: !open.has(s.id) }));
    const target = nextTarget(stations, makeOrientation(0, 0, LEVEL), "level", DEFAULT_LENS);
    expect(target?.station.yawDeg).toBe(60);
    // 30 deg behind is genuinely nearer than 60 ahead — and is deliberately not chosen.
    expect(target?.distanceDeg).toBeGreaterThan(
      greatCircleDeg(0, LEVEL, 330, LEVEL),
    );
  });

  it("does not send the user a whole turn back for a station they just passed", () => {
    // The clockwise rule needs a tolerance or it bites on its own: drift two degrees past an
    // unfilled ring and its clockwise distance flips from 0 to 358.
    const open = new Set(["level:1", "level:2"]); // 15 deg (just passed) and 30 deg ahead
    const stations = buildStations(config).map((s) => ({ ...s, filled: !open.has(s.id) }));
    const target = nextTarget(stations, makeOrientation(0, 17, LEVEL), "level", DEFAULT_LENS);
    expect(target?.station.yawDeg).toBe(15);
  });

  it("is deterministic when two stations are exactly equidistant", () => {
    const midway = makeOrientation(0, 7.5, 0);
    const a = nextTarget(stations, midway, "level", DEFAULT_LENS);
    const b = nextTarget(stations, midway, "level", DEFAULT_LENS);
    expect(a?.station.id).toBe(b?.station.id);
  });
});

describe("accumulateYaw", () => {
  it("crosses the 0/360 seam without a 358-degree jump", () => {
    expect(accumulateYaw(359, 1, 0)).toBeCloseTo(2, 6);
    expect(accumulateYaw(1, 359, 0)).toBeCloseTo(2, 6);
    expect(accumulateYaw(0, 359.5, 0)).toBeCloseTo(0.5, 6);
  });

  it("never decreases, so a reversal cannot cost earned progress", () => {
    let span = 370;
    for (const [from, to] of [[10, 5] as const, [5, 0] as const, [0, 355] as const]) {
      const next = accumulateYaw(from, to, span);
      expect(next).toBeGreaterThanOrEqual(span);
      span = next;
    }
    expect(span).toBeCloseTo(385, 6);
  });

  it("ignores sensor hash below the deadband", () => {
    let span = 0;
    for (let i = 0; i < 1000; i += 1) span = accumulateYaw(0, i % 2 === 0 ? 0.1 : -0.1, span);
    expect(span).toBe(0);
  });

  it("ignores a discontinuity: a compass re-reference is not a turn", () => {
    expect(accumulateYaw(0, 120, 0)).toBe(0);
    expect(accumulateYaw(0, 44, 0)).toBeCloseTo(44, 6);
  });

  it("accumulates a full coached row to the 370-degree target", () => {
    let span = 0;
    let previous = 0;
    for (let i = 1; i <= 155; i += 1) {
      const current = (i * 2.4) % 360;
      span = accumulateYaw(previous, current, span);
      previous = current;
    }
    expect(span).toBeGreaterThanOrEqual(YAW_SPAN.acceptMinDeg);
    expect(span).toBeLessThanOrEqual(YAW_SPAN.acceptMaxDeg);
  });

  it("survives NaN without poisoning the running total", () => {
    expect(accumulateYaw(Number.NaN, 10, 42)).toBe(42);
    expect(accumulateYaw(10, Number.NaN, 42)).toBe(42);
  });
});

describe("row progress", () => {
  it("needs both gates: 22 stations AND a closed circle", () => {
    const stations = buildStations(config);
    for (const s of stations) if (s.row === "level" && s.index < 22) s.filled = true;

    expect(isRowComplete(rowProgress(stations, "level", 359, 0, 0))).toBe(false);
    expect(isRowComplete(rowProgress(stations, "level", 370, 0, 0))).toBe(true);

    const twentyOne = buildStations(config);
    for (const s of twentyOne) if (s.row === "level" && s.index < 21) s.filled = true;
    expect(isRowComplete(rowProgress(twentyOne, "level", 370, 0, 0))).toBe(false);
  });

  it("reports the minimum of the two gates, so the rail never lies", () => {
    const stations = buildStations(config);
    for (const s of stations) if (s.row === "level" && s.index < 22) s.filled = true;
    // Stations at 100%, span at 27%. A mean would read 63% and promise an exit that is
    // nowhere near.
    expect(rowFraction(rowProgress(stations, "level", 100, 0, 0))).toBeCloseTo(100 / 370, 4);
  });

  it("clamps to 1 rather than overshooting on the 370-degree target", () => {
    const stations = buildStations(config);
    for (const s of stations) if (s.row === "level") s.filled = true;
    expect(rowFraction(rowProgress(stations, "level", 400, 0, 0))).toBe(1);
  });

  it("orders the rows level, up, down — up second because it is the row users skip", () => {
    expect(ROW_IDS).toEqual(["level", "up", "down"]);
    expect(rowIndex("level")).toBe(1);
    expect(rowIndex("up")).toBe(2);
    expect(rowIndex("down")).toBe(3);
    expect(nextRow("level")).toBe("up");
    expect(nextRow("up")).toBe("down");
    expect(nextRow("down")).toBeNull();
  });

  it("exposes per-row fractions in the shape RowPrompt wants", () => {
    const stations = buildStations(config);
    const rows = {} as Record<RowId, ReturnType<typeof rowProgress>>;
    for (const row of ROW_IDS) rows[row] = rowProgress(stations, row, 0, 0, 0);
    const fractions = rowFractions(rows);
    expect(Object.keys(fractions).sort()).toEqual([...ROW_IDS].sort());
    for (const row of ROW_IDS) expect(fractions[row]).toBe(0);
  });
});

describe("stalls", () => {
  it("does not nag before STUCK_AFTER_MS and does after", () => {
    expect(isStalled(1000, 1000 + 5999)).toBe(false);
    expect(isStalled(1000, 1000 + 6000)).toBe(true);
  });

  it("says nothing at all before the first station fills", () => {
    expect(isStalled(null, 999_999)).toBe(false);
  });
});

describe("pacing", () => {
  it("budgets ~15.4 s per row and ~50 s in total at the coached 24 deg/s", () => {
    expect(rowDurationSeconds("level")).toBeCloseTo(DURATION_BUDGET.perRowSeconds, 1);
    expect(totalDurationSeconds()).toBeCloseTo(DURATION_BUDGET.totalSeconds, 0);
  });

  it("stretches when a dim room forces the adaptive cap down", () => {
    expect(totalDurationSeconds(12)).toBeGreaterThan(totalDurationSeconds(24));
  });
});

// =====================================================================================

describe("blockers", () => {
  it("names every unmet criterion and nothing else", () => {
    const samples = singleRowSweep();
    const grid = createGrid();
    for (const s of samples) stampCone(grid, s);
    const r = report(grid, buildStations(config));
    expect(blockers(r)).toEqual(r.blockedBy);
    expect(blockers(r)).toContain("pitch_max");
    // horizon_30 IS unmet here now, and naming it is the point of this test. At
    // ROW_BIAS_DEG = -8 the level row spans -42.5..+26.5, so it no longer covers the
    // |elev| <= 30 band by itself the way it did from the equator — the up row finishes the
    // top of it. A level-only sweep is therefore short of the horizon band as well.
    expect(blockers(r)).toContain("horizon_30");
  });

  it("is empty for a capture that meets COMPLETENESS", () => {
    const grid = createGrid();
    for (const s of perfectSweep()) stampCone(grid, s);
    const stations = runSweep(perfectSweep(), buildStations(config));
    expect(blockers(report(grid, stations))).toEqual([]);
  });
});

// =====================================================================================

function goodTelemetry(): Pick<CaptureTelemetry, "translation" | "image"> {
  return {
    translation: {
      medianPivotRadiusM: 0.18,
      p90PivotRadiusM: 0.3,
      medianDriftMPerS: 0.05,
      maxDriftMPerS: 0.12,
      walkingFraction: 0,
      motionGateRan: true,
    },
    image: { meanLuma: 150, minLuma: 120, meanClippedFraction: 0.01, maxClippedFraction: 0.03, fractionOverSmearBudget: 0 },
  };
}

function completeReport(): CoverageReport {
  const grid = createGrid();
  for (const s of perfectSweep()) stampCone(grid, s);
  return report(grid, runSweep(perfectSweep(), buildStations(config)));
}

describe("verdict", () => {
  it("mirrors validate.py's ladder: short, dark, blurry, walking, then coverage", () => {
    const r = completeReport();
    expect(verdict(r, goodTelemetry())).toBe("great");
    expect(verdict(r, goodTelemetry(), DURATION_BUDGET.minSeconds * 1000 - 1)).toBe("too_short");
    expect(verdict(r, { ...goodTelemetry(), image: { ...goodTelemetry().image, meanLuma: 30 } })).toBe("too_dark");
    expect(verdict(r, { ...goodTelemetry(), image: { ...goodTelemetry().image, fractionOverSmearBudget: 0.5 } })).toBe("too_blurry");
    expect(verdict(r, { ...goodTelemetry(), translation: { ...goodTelemetry().translation, walkingFraction: 0.4 } })).toBe("walking");
  });

  it("does not call a capture walking on one peak the detector never latched", () => {
    // The drift latch needs `stopSustainMs` (1500 ms) above the stop line before it grades
    // red, because a push-off or a shift of grip crosses that line for a few hundred
    // milliseconds and then the high-pass takes it back. `walkingFraction` counts latched
    // samples; `maxDriftMPerS` is a raw peak with no duration behind it.
    //
    // Real numbers, from a 219 s capture with three complete rows: 2.5% latched, median
    // drift 0.043 m/s, one peak at 0.4028. The engine solved it with a 0.76 degree loop
    // residual — not a walked capture.
    const onePeak = {
      ...goodTelemetry(),
      translation: {
        ...goodTelemetry().translation,
        motionGateRan: true,
        walkingFraction: 0.0251,
        medianDriftMPerS: 0.043,
        maxDriftMPerS: 0.4028,
      },
    };
    expect(onePeak.translation.maxDriftMPerS).toBeGreaterThan(THRESHOLDS.translationDrift.stopMPerS);
    expect(verdict(completeReport(), onePeak)).toBe("great");

    // Sustained drift still fails, which is the whole point of the gate.
    const sustained = {
      ...onePeak,
      translation: { ...onePeak.translation, walkingFraction: WALKING_TIME_FRACTION },
    };
    expect(verdict(completeReport(), sustained)).toBe("walking");
  });

  it("does not refuse a 4:3 capture for a shortfall its lens cannot close", () => {
    // THE BUG THIS PINS, from a real capture (vivo Y22s, 1080x1440): every station of every
    // row filled, no gaps, horizon at 100%, pitch reach +56/-75 — and 0.8314 of the sphere,
    // which is all three rows of a 4:3 frame can reach. `computeBlockers` passed it, because
    // `gatesForLens` adapts the threshold to what the lens can do. `verdict` then failed it
    // anyway, by testing `coveredFraction` against the raw 0.855 constant a second time.
    //
    // The user's report was exact: "I completed all the circles and it still says 82% of the
    // sphere, shoot it again". There was nothing left for them to do.
    const r = completeReport();
    const shortOfTheRawConstant: CoverageReport = {
      ...r,
      coveredFraction: 0.8314,
      blockedBy: [],
    };
    expect(shortOfTheRawConstant.coveredFraction).toBeLessThan(COMPLETENESS.coveredFraction);
    expect(verdict(shortOfTheRawConstant, goodTelemetry())).toBe("great");

    // And the gate still closes when the blockers say it should.
    expect(verdict({ ...shortOfTheRawConstant, blockedBy: ["covered_fraction"] }, goodTelemetry())).toBe("gaps");
  });

  it("puts dark ahead of blurry, exactly as the engine does", () => {
    const bad = {
      translation: goodTelemetry().translation,
      image: { ...goodTelemetry().image, meanLuma: 20, fractionOverSmearBudget: 0.9 },
    };
    expect(verdict(completeReport(), bad)).toBe("too_dark");
  });

  it("calls an incomplete capture `gaps`, not `great`", () => {
    const grid = createGrid();
    for (const s of sweepWithGap(120, 180)) stampCone(grid, s);
    const r = report(grid, runSweep(sweepWithGap(120, 180), buildStations(config)));
    expect(verdict(r, goodTelemetry())).toBe("gaps");
    expect(r.coveredFraction).toBeLessThan(COMPLETENESS.coveredFraction);
  });

  it("calls a capture that never went round `too_short`, whatever the clock says", () => {
    const grid = createGrid();
    for (const s of sweep({ yawSpanDeg: 60 })) stampCone(grid, s);
    const r = report(grid, buildStations(config));
    for (const row of ROW_IDS) expect(r.rows[row].yawSpanDeg).toBeLessThan(YAW_SPAN.rowCountedMinDeg);
    expect(verdict(r, goodTelemetry(), 30_000)).toBe("too_short");
  });

  it("still reaches a walking verdict when the inertial gate never ran", () => {
    // Absence of evidence is not evidence: the camera-side signal alone must be able to
    // condemn a capture.
    const flowOnly = {
      ...goodTelemetry(),
      translation: { ...goodTelemetry().translation, motionGateRan: false, walkingFraction: 0.6, maxDriftMPerS: 0 },
    };
    expect(verdict(completeReport(), flowOnly)).toBe("walking");
  });
});

// =====================================================================================

/** A capture with perfect coverage that was ORBITED rather than pivoted. */
function orbitTelemetry(medianM: number, p90M = medianM): Pick<CaptureTelemetry, "translation" | "image"> {
  const base = goodTelemetry();
  return { ...base, translation: { ...base.translation, medianPivotRadiusM: medianM, p90PivotRadiusM: p90M } };
}

describe("the pivot rung", () => {
  it("refuses to call an orbit `great`, however complete its coverage is", () => {
    const r = completeReport();
    // 0.35 m is the arm's-length swing that measures ~3.6 px of rotational residual on the
    // engine's own ladder — the real capture whose ceiling beams visibly bend. Coverage,
    // stations and yaw span are all perfect here, which is exactly why nothing else catches
    // it.
    expect(verdict(r, orbitTelemetry(0.35))).toBe("gaps");
    expect(verdict(r, orbitTelemetry(1.2))).toBe("gaps");
    expect(verdict(r, orbitTelemetry(0.35))).not.toBe("great");
  });

  it("passes a phone genuinely turning about itself", () => {
    const r = completeReport();
    expect(verdict(r, orbitTelemetry(0.18, 0.3))).toBe("great");
    expect(verdict(r, orbitTelemetry(PIVOT_MEDIAN_LIMIT_M, PIVOT_P90_LIMIT_M))).toBe("great");
  });

  it("catches a capture that is mostly tight but swings wide on part of the turn", () => {
    const r = completeReport();
    expect(orbiting({ medianPivotRadiusM: 0.15, p90PivotRadiusM: 0.6 })).toBe(true);
    expect(verdict(r, orbitTelemetry(0.15, 0.6))).toBe("gaps");
  });

  it("treats a radius that was never measurable as no evidence, not as evidence", () => {
    // Below the gate speed for the whole capture, or no accelerometer at all. Silence must
    // not be read as an orbit — that would fail every gyro-less phone on a guess.
    expect(orbiting({ medianPivotRadiusM: null, p90PivotRadiusM: null })).toBe(false);
    const r = completeReport();
    const base = goodTelemetry();
    const blind = { ...base, translation: { ...base.translation, medianPivotRadiusM: null, p90PivotRadiusM: null } };
    expect(verdict(r, blind)).toBe("great");
  });

  it("sits BELOW walking in the ladder, so the stricter diagnosis still wins", () => {
    const orbitAndWalk = {
      ...orbitTelemetry(1.2),
      translation: { ...orbitTelemetry(1.2).translation, walkingFraction: 0.4 },
    };
    expect(verdict(completeReport(), orbitAndWalk)).toBe("walking");
  });

  it("agrees with the predicate the review screen shows its copy from", () => {
    // One test, two consumers: if these ever diverged the phone would grade a capture on
    // one rule and explain it with another.
    for (const [median, p90] of [
      [0.1, 0.2],
      [0.35, 0.35],
      [0.15, 0.6],
      [1.2, 1.4],
    ] as const) {
      const graded = verdict(completeReport(), orbitTelemetry(median, p90));
      expect(orbiting({ medianPivotRadiusM: median, p90PivotRadiusM: p90 })).toBe(graded === "gaps");
    }
  });
});

describe("a verdict with no coverage measurement at all (RAIL MODE)", () => {
  /** Nothing ever stamped the sphere: every cell empty, every row span zero. */
  function unmeasured(): CoverageReport {
    return report(createGrid(), buildStations(config));
  }

  it("never says `great` and never says `too_short` about a sweep it did not watch", () => {
    const graded = verdict(unmeasured(), goodTelemetry(), 45_000, { coverageMeasured: false });
    expect(graded).toBe("gaps");
    expect(graded).not.toBe("great");
    expect(graded).not.toBe("too_short");
  });

  it("still applies every gate that does NOT need a compass", () => {
    const dark = { ...goodTelemetry(), image: { ...goodTelemetry().image, meanLuma: 20 } };
    expect(verdict(unmeasured(), dark, 45_000, { coverageMeasured: false })).toBe("too_dark");

    const walked = { ...goodTelemetry(), translation: { ...goodTelemetry().translation, walkingFraction: 0.4 } };
    expect(verdict(unmeasured(), walked, 45_000, { coverageMeasured: false })).toBe("walking");

    // The duration gate is a clock, not a compass, so it still bites.
    expect(verdict(unmeasured(), goodTelemetry(), 3000, { coverageMeasured: false })).toBe("too_short");
  });

  it("is the reason the same report would otherwise read as a capture that missed everything", () => {
    // Same inputs, only the flag differs. This is the bug the flag exists to prevent.
    expect(verdict(unmeasured(), goodTelemetry(), 45_000)).toBe("too_short");
  });
});

// =====================================================================================

describe("the framing gate", () => {
  /** A readout that clears every gate: still, level, well lit, running at rate. */
  function goodReadout(overrides: Partial<DetectorReadout> = {}): DetectorReadout {
    return {
      omegaDegPerS: 1,
      speedCapDegPerS: THRESHOLDS.angularSpeed.greenMaxDegPerS,
      speedGrade: "green",
      smearPx: 0,
      smearGrade: "green",
      pivotRadiusM: null,
      pivotGrade: "green",
      translationMPerS: 0,
      stepBandRmsMs2: 0,
      flowDivergencePx: 0,
      walkingGrade: "green",
      rollDeg: 0,
      rollGrade: "green",
      rowPitchErrorDeg: 0,
      rowPitchGrade: "green",
      meanLuma: 120,
      clippedFraction: 0,
      exposureGrade: "green",
      measuredFps: 30,
      motionGateDegraded: false,
      ...overrides,
    };
  }

  function input(overrides: Partial<FramingInput> = {}): FramingInput {
    return {
      now: 0,
      readout: goodReadout(),
      pitchDeg: 0,
      yawDeg: 0,
      brightestYawDeg: null,
      portraitBlocked: false,
      enteredAt: 0,
      steadySince: null,
      gatesSince: null,
      ...overrides,
    };
  }

  /** Drive the gate from `from` to `to` in `stepMs` ticks, carrying the clocks. */
  function run(base: Partial<FramingInput>, fromMs: number, toMs: number, stepMs = 100) {
    let steadySince: number | null = null;
    let gatesSince: number | null = null;
    let last = framingDecision(input({ ...base, now: fromMs }));
    for (let t = fromMs; t <= toMs; t += stepMs) {
      last = framingDecision(input({ ...base, now: t, steadySince, gatesSince }));
      steadySince = last.steadySince;
      gatesSince = last.gatesSince;
      if (last.start) return { decision: last, startedAt: t };
    }
    return { decision: last, startedAt: null };
  }

  it("starts recording by itself once every gate has held for the countdown", () => {
    const { decision, startedAt } = run({}, 0, 20_000);
    expect(decision.start).toBe(true);
    expect(startedAt).not.toBeNull();
    // steadyHoldMs (1200) then countdownMs (3000), and not a tick before.
    expect(startedAt).toBeGreaterThanOrEqual(THRESHOLDS.framing.steadyHoldMs + THRESHOLDS.framing.countdownMs);
  });

  it("REFUSES TO START IN LANDSCAPE, even though every number it can read says go", () => {
    // The exact probe behind the finding: the roll and pitch reaching this function have
    // already been un-rotated by `screen.orientation.angle` upstream, so a phone lying
    // perfectly level in landscape presents roll 0 and pitch 0 and passes `level`, `hold`,
    // `dark`, `bright` and `fps`. Nothing in the readout can catch it.
    const readout = goodReadout({ rollDeg: 0 });
    expect(framingDecision(input({ readout, pitchDeg: 0, portraitBlocked: false, now: 10_000, steadySince: 0, gatesSince: 0 })).start).toBe(true);

    const { decision, startedAt } = run({ readout, pitchDeg: 0, portraitBlocked: true }, 0, 30_000);
    expect(startedAt).toBeNull();
    expect(decision.start).toBe(false);
    expect(decision.gate).toBe("level");
    // And the escape hatch is withheld too: "Yine de başla" must not punch through into a
    // landscape recording either.
    expect(decision.canForceStart).toBe(false);
  });

  it("costs the countdown when the phone is rotated mid-count, rather than pausing it", () => {
    // Two and a half seconds into a three-second count, a wrist twist.
    let steadySince: number | null = null;
    let gatesSince: number | null = null;
    for (let t = 0; t <= 3800; t += 100) {
      const d = framingDecision(input({ now: t, steadySince, gatesSince }));
      steadySince = d.steadySince;
      gatesSince = d.gatesSince;
      expect(d.start).toBe(false);
      if (t >= 3700) break;
    }
    expect(gatesSince).not.toBeNull();

    const rotated = framingDecision(input({ now: 3800, portraitBlocked: true, steadySince, gatesSince }));
    expect(rotated.gatesSince).toBeNull();
    expect(rotated.steadySince).toBeNull();
    expect(rotated.countdownSeconds).toBeNull();

    // Back upright: the full hold and the full countdown are owed again.
    const { startedAt } = run({}, 3900, 12_000);
    expect(startedAt).not.toBeNull();
    expect((startedAt ?? 0) - 3900).toBeGreaterThanOrEqual(THRESHOLDS.framing.steadyHoldMs + THRESHOLDS.framing.countdownMs);
  });

  it("names the gate that is actually holding it up", () => {
    const held = { now: 5000, steadySince: 0, gatesSince: null };
    expect(framingDecision(input({ ...held, readout: goodReadout({ omegaDegPerS: 40 }) })).gate).toBe("hold");
    expect(framingDecision(input({ ...held, readout: goodReadout({ rollDeg: 30 }) })).gate).toBe("level");
    expect(framingDecision(input({ ...held, pitchDeg: 40 })).gate).toBe("level");
    expect(framingDecision(input({ ...held, readout: goodReadout({ meanLuma: 10 }) })).gate).toBe("dark");
    expect(framingDecision(input({ ...held, readout: goodReadout({ clippedFraction: 0.9 }) })).gate).toBe("bright");
    expect(framingDecision(input({ ...held, readout: goodReadout({ measuredFps: 10 }) })).gate).toBe("fps");
    expect(framingDecision(input({ ...held })).gate).toBeNull();
  });

  it("offers the force-start escape hatch once, and only once, the gate has held too long", () => {
    const stuck = { readout: goodReadout({ meanLuma: 5 }), steadySince: 0, gatesSince: null };
    expect(framingDecision(input({ ...stuck, now: THRESHOLDS.framing.forceStartAfterMs - 1 })).canForceStart).toBe(false);
    expect(framingDecision(input({ ...stuck, now: THRESHOLDS.framing.forceStartAfterMs + 1 })).canForceStart).toBe(true);
  });

  it("does not block on a rate it has not measured yet", () => {
    expect(framingDecision(input({ now: 5000, steadySince: 0, readout: goodReadout({ measuredFps: 0 }) })).gate).toBeNull();
  });

  it("holds rather than starting when there is no readout at all", () => {
    const d = framingDecision(input({ now: 20_000, readout: null, steadySince: 0, gatesSince: 0 }));
    expect(d.start).toBe(false);
    expect(d.gate).toBe("hold");
  });
});

// =====================================================================================

describe("RAIL MODE row progression", () => {
  it("asks for exactly the turn the coached guide asks for", () => {
    for (const row of ROW_IDS) {
      const target = rowDurationSeconds(row) * 1000;
      expect(railRowProgress(target - 1, row).complete).toBe(false);
      expect(railRowProgress(target, row).complete).toBe(true);
    }
  });

  it("counts down in whole seconds and never below zero", () => {
    expect(railRowProgress(0, "level").remainingSeconds).toBe(Math.ceil(rowDurationSeconds("level")));
    expect(railRowProgress(1_000_000, "level").remainingSeconds).toBe(0);
    expect(railRowProgress(-5, "level").remainingSeconds).toBe(Math.ceil(rowDurationSeconds("level")));
  });

  it("reports a monotone fraction clamped to 1", () => {
    const half = railRowProgress((rowDurationSeconds("up") * 1000) / 2, "up");
    expect(half.fraction).toBeCloseTo(0.5, 3);
    expect(railRowProgress(1_000_000, "up").fraction).toBe(1);
  });
});

// =====================================================================================

describe("the capture ceiling", () => {
  it("does NOT end a capture at the per-segment 120 s ceiling", () => {
    // The bug this replaces: `RECORDING.maxDurationMs` is a segment limit, and treating it
    // as the capture limit silently threw away every take longer than two minutes.
    expect(captureClock(120_000).exhausted).toBe(false);
    expect(MAX_CAPTURE_MS).toBeGreaterThan(120_000);
  });

  it("stays inside what the engine will actually accept", () => {
    // The ceiling is set by the UPLOAD, not by the capture, so it is pinned against the
    // server's limit rather than left to drift. `Settings.max_upload_bytes` is 1536 MB
    // (engine/biosphere/core/config.py); a client that lets the user shoot more than the
    // server will take fails at the worst possible moment — after the room is filmed and
    // the progress bar has reached the end.
    const ENGINE_MAX_UPLOAD_BYTES = 1536 * 1024 * 1024;
    const worstCaseBytes = (MAX_CAPTURE_MS / 1000) * (RECORDING.targetBitrateBps / 8);
    expect(worstCaseBytes).toBeLessThan(ENGINE_MAX_UPLOAD_BYTES);
    // ...and with real headroom, because a WebM's cluster headers repeat per segment.
    expect(worstCaseBytes * 1.25).toBeLessThan(ENGINE_MAX_UPLOAD_BYTES);
  });

  it("gives a capture more room than the coached plan needs", () => {
    // 50 s is the coached budget. A first-timer learning the motion, correcting a row and
    // going round again is not unusual at five or six minutes, and the ceiling that ends
    // the capture must sit above that rather than inside it.
    expect(MAX_CAPTURE_MS).toBeGreaterThanOrEqual(600_000);
    expect(totalDurationSeconds() * 1000).toBeLessThan(MAX_CAPTURE_MS / 5);
  });

  it("counts up until the last minute, then counts down", () => {
    expect(captureClock(10_000).counting).toBe(false);
    expect(captureClock(MAX_CAPTURE_MS - CAPTURE_COUNTDOWN_MS - 1).counting).toBe(false);
    expect(captureClock(MAX_CAPTURE_MS - CAPTURE_COUNTDOWN_MS).counting).toBe(true);
    expect(captureClock(MAX_CAPTURE_MS - 5000).remainingMs).toBe(5000);
  });

  it("is exhausted exactly at the ceiling and never reports negative time", () => {
    expect(captureClock(MAX_CAPTURE_MS - 1).exhausted).toBe(false);
    expect(captureClock(MAX_CAPTURE_MS).exhausted).toBe(true);
    expect(captureClock(MAX_CAPTURE_MS + 60_000).remainingMs).toBe(0);
  });

  it("survives a garbage clock", () => {
    expect(captureClock(Number.NaN).recordedMs).toBe(0);
    expect(captureClock(-1).recordedMs).toBe(0);
  });
});

describe("recordedFrameSize", () => {
  const portrait = { width: 1080, height: 1920 };

  it("never lets a zero win, which a `??` chain would", () => {
    // `last?.width ?? camera?.width ?? 0` returns 0 here, because 0 is not nullish — and the
    // sidecar then ships 0x0 with `portrait` falling through to its `true` default.
    const segments = [portrait, { width: 0, height: 0 }];
    expect(recordedFrameSize(segments, null)).toEqual(portrait);
  });

  it("prefers the most recent segment that actually stated its dimensions", () => {
    expect(recordedFrameSize([{ width: 720, height: 1280 }, portrait], null)).toEqual(portrait);
  });

  it("falls back to the camera only when no segment knows, and to 0x0 only when nothing does", () => {
    expect(recordedFrameSize([{ width: 0, height: 0 }], portrait)).toEqual(portrait);
    expect(recordedFrameSize([], { width: 0, height: 0 })).toEqual({ width: 0, height: 0 });
    expect(recordedFrameSize([], null)).toEqual({ width: 0, height: 0 });
  });

  it("keeps portrait derivable, which is the whole point of the field", () => {
    const size = recordedFrameSize([portrait], null);
    expect(size.height > size.width).toBe(true);
  });
});

// =====================================================================================

describe("the settling pause at a row boundary", () => {
  /** Aim at the first ring of `row`, tick for `holdMs`, return the last verdict and fills. */
  function enterRow(row: RowId, holdMs: number, stepMs = 50) {
    const stations = buildStations(config);
    const first = stations.filter((s) => s.row === row).sort((a, b) => a.index - b.index)[0];
    let since: number | null = null;
    let last = armRow(row, stations, null, null, 0);
    const filled: string[] = [];
    for (let t = 0; t <= holdMs; t += stepMs) {
      const sample = makeOrientation(t, first.yawDeg, first.pitchDeg);
      last = armRow(row, stations, sample, since, t);
      since = last.since;
      filled.push(...updateStations(stations, sample, t, last.blocking));
    }
    return { arm: last, filled, first, stations };
  }

  it("blocks the first ring of the up row for two seconds", () => {
    const early = enterRow("up", STATION_RULES.rowArmMs - 200);
    expect(early.arm.blocking).toBe(true);
    expect(early.filled).toHaveLength(0);
  });

  it("blocks the first ring of the down row for two seconds", () => {
    const early = enterRow("down", STATION_RULES.rowArmMs - 200);
    expect(early.arm.blocking).toBe(true);
    expect(early.filled).toHaveLength(0);
  });

  it("fills once the hold is served", () => {
    const done = enterRow("up", STATION_RULES.rowArmMs + STATION_RULES.dwellMs + 200);
    expect(done.arm.blocking).toBe(false);
    expect(done.filled).toContain(done.first.id);
  });

  it("counts down whole seconds and never shows a zero", () => {
    const seen = new Set<number>();
    const stations = buildStations(config);
    const first = stations.filter((s) => s.row === "up")[0];
    let since: number | null = null;
    for (let t = 0; t < STATION_RULES.rowArmMs; t += 100) {
      const arm = armRow("up", stations, makeOrientation(t, first.yawDeg, first.pitchDeg), since, t);
      since = arm.since;
      if (arm.secondsLeft !== null) seen.add(arm.secondsLeft);
    }
    expect([...seen].sort()).toEqual([1, 2]);
  });

  it("leaves the level row alone — the user says that entry already feels right", () => {
    const stations = buildStations(config);
    const first = stations.filter((s) => s.row === "level")[0];
    const arm = armRow("level", stations, makeOrientation(0, first.yawDeg, first.pitchDeg), null, 0);
    expect(arm.blocking).toBe(false);
    expect(arm.secondsLeft).toBeNull();
  });

  it("does not charge a toll on the way: nothing is blocked until a ring is reached", () => {
    const stations = buildStations(config);
    const first = stations.filter((s) => s.row === "up")[0];
    // Away in PITCH, not yaw: the row is 24 stations at 15 deg, so the worst yaw distance
    // to the nearest one is 7.5 deg and the 9-deg capture radius covers every yaw there is.
    // Leaving the row means tilting off it, which is also how a user actually leaves.
    const away = makeOrientation(0, first.yawDeg, first.pitchDeg + 30);
    const arm = armRow("up", stations, away, null, 0);
    expect(arm.blocking).toBe(false);
    expect(arm.since).toBeNull();
  });

  it("restarts the hold if the user swings off the ring", () => {
    const stations = buildStations(config);
    const first = stations.filter((s) => s.row === "up")[0];
    let since: number | null = null;
    for (let t = 0; t < 1500; t += 100) {
      since = armRow("up", stations, makeOrientation(t, first.yawDeg, first.pitchDeg), since, t).since;
    }
    // Swing away: the clock is dropped. Off in pitch — see above, yaw cannot leave the row.
    since = armRow("up", stations, makeOrientation(1500, first.yawDeg, first.pitchDeg + 30), since, 1500).since;
    expect(since).toBeNull();
    // Coming back starts a fresh two seconds rather than inheriting the 1.5 already served.
    const back = armRow("up", stations, makeOrientation(1600, first.yawDeg, first.pitchDeg), since, 1600);
    expect(back.blocking).toBe(true);
    expect(back.secondsLeft).toBe(2);
  });

  it("stops asking once the row is under way", () => {
    const stations = buildStations(config);
    const rowStations = stations.filter((s) => s.row === "up").sort((a, b) => a.index - b.index);
    rowStations[0].filled = true;
    const second = rowStations[1];
    const arm = armRow("up", stations, makeOrientation(0, second.yawDeg, second.pitchDeg), null, 0);
    expect(arm.blocking).toBe(false);
    expect(arm.secondsLeft).toBeNull();
  });
});
