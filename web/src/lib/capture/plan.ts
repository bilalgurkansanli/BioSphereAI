/**
 * BioSphere AI — guided capture: the live plan.
 * =====================================================================================
 *
 * Turns the contract's rows and thresholds into the thing the screen shows: which row is
 * active, where the bright ring goes next, how far through the user is, and — the part
 * that has to feel earned rather than arbitrary — when a row is allowed to end.
 *
 * Pure: no React, no `three`, no DOM.
 *
 * -------------------------------------------------------------------------------------
 * WHAT "EARNED" MEANS HERE, PRECISELY
 * -------------------------------------------------------------------------------------
 * A row advances on TWO independent conditions, and both are the engine's, not a designer's:
 *
 *   1. `filledStations >= requiredStations` (22 of 24). Stations are a lattice of aiming
 *      reticles at 15-deg spacing; 15 deg is the tightest keyframe bin any quality preset
 *      asks for, so a filled station is a guarantee that the extractor has a candidate
 *      frame in that bin. Two may be missed — the pose graph carries `neighbour_radius = 3`
 *      and loop closure — but a third hole is a chain break, and
 *      `solve.MIN_COMPONENT_FRACTION = 0.60` refuses a graph that breaks too often.
 *
 *   2. `yawSpanDeg >= YAW_SPAN.acceptMinDeg` (360). The wrap-around loop-closure pair is
 *      (last keyframe, first keyframe); it only verifies at ordinary overlap when the span
 *      lands in [340.6, 379.4]. `targetYawSpanDeg` 370 is the coached target, 360 the gate.
 *
 * Neither can be gamed by the other. Twenty-two stations without going round is impossible
 * (they are 15 deg apart). Going round without filling them is possible — a whip-pan — and
 * that is exactly what condition 1 catches. Rocking on the spot inflates travelled yaw and
 * is caught by condition 1 too.
 */

import {
  BONUS_POLE_STATIONS,
  COMPLETENESS,
  DEFAULT_LENS,
  DURATION_BUDGET,
  ROW_ID_BY_STATE,
  ROW_IDS,
  ROW_PLAN,
  STATION_RULES,
  STUCK_AFTER_MS,
  THRESHOLDS,
  YAW_SPAN,
  type CaptureConfig,
  type CaptureState,
  type CaptureTelemetry,
  type CaptureVerdict,
  type CompletionBlocker,
  type CoverageReport,
  type Degrees,
  type DetectorReadout,
  type LensGeometry,
  type Millis,
  type OrientationSample,
  type PlanApi,
  type RecordedSegment,
  type RowId,
  type RowProgress,
  type Station,
  type StationId,
  type StationState,
  type StationTarget,
} from "@/lib/capture/types";
import { computeBlockers, greatCircleDeg, insideFrustum, buildFrustum, rowTargetPitchDeg, shortestYawDeltaDeg, wrapYaw360 } from "@/lib/capture/coverage";

export { rowTargetPitchDeg };

/**
 * Largest dwell credit a single tick may award.
 *
 * `updateStations` is handed `now` and derives dt from the previous call. A backgrounded
 * tab, a dropped call or a debugger pause makes that gap arbitrarily large, and without
 * this clamp the first tick after returning would bank every station the axis happens to
 * be pointing at — footage that was never recorded.
 *
 * HALF of `STATION_RULES.dwellMs`, not all of it. Clamping at the full 250 ms still lets a
 * single tick take a station from 0 to filled, because the fill test is `>=`; at 125 ms two
 * separate ticks are always required, so a station can only be earned by the axis being
 * observed on the target twice. Legitimate ticks are unaffected: the coverage stamp runs at
 * 10 Hz (100 ms) and the sensors at 20-60 Hz.
 */
const MAX_DWELL_STEP_MS = STATION_RULES.dwellMs / 2;

/**
 * Yaw jitter below this is sensor noise, not travel. `accumulateYaw` would otherwise let a
 * phone resting on a table accrue a full circle out of magnetometer hash.
 */
const YAW_DEADBAND_DEG = 0.25;

/**
 * A single yaw step larger than this is a discontinuity — a compass re-reference, a
 * heading fix landing, or a dropped frame — not a human turning. At the coached 24 deg/s
 * and a 10 Hz stamp a real step is 2.4 deg; 45 deg is nearly twenty times that.
 */
const YAW_MAX_STEP_DEG = 45;

/**
 * `updateStations` mutates its array in place (it returns the ids that filled, not a new
 * array), so the previous tick's clock can live keyed on that array's identity. A WeakMap
 * keeps this module free of module-level mutable state: two concurrent sessions, or a test
 * running twenty synthetic sweeps, never see each other's clocks.
 */
const lastTickAt = new WeakMap<StationState[], Millis>();

// =====================================================================================
// Stations
// =====================================================================================

/** `${row}:${index}` — stable across a session and across a draft restore. */
export function stationId(row: RowId, index: number): StationId {
  return `${row}:${index}`;
}

/**
 * The full lattice: three rows of 24 reticles at 15-deg spacing.
 *
 * Row elevations come from `rowTargetPitchDeg`, i.e. from `rowSeparationDeg(lens)`, NOT
 * from a literal. On the default 69-deg lens that is 29 — identical to `ROW_PLAN` — but on
 * a narrower lens the admissible window is only [27, 30] and hard-coding 29 would quietly
 * push cross-row overlap under the engine's 0.55 design point.
 */
export function buildStations(config: CaptureConfig): StationState[] {
  const out: StationState[] = [];
  for (const row of ROW_IDS) {
    const plan = config.rows[row];
    const pitchDeg = rowTargetPitchDeg(row, config.lens);
    for (let index = 0; index < plan.stationCount; index += 1) {
      out.push({
        id: stationId(row, index),
        row,
        index,
        yawDeg: wrapYaw360(index * plan.stationSpacingDeg),
        pitchDeg,
        filled: false,
        filledAt: null,
        dwellMs: 0,
      });
    }
  }
  return out;
}

/** The four opt-in pole reticles. Never a debt — `poles.py` synthesises these acceptably. */
export function buildBonusStations(): StationState[] {
  return BONUS_POLE_STATIONS.map((b) => {
    const [row, indexText] = b.id.split(":");
    return {
      id: b.id,
      row: row as RowId,
      index: Number(indexText),
      yawDeg: 0,
      pitchDeg: b.pitchDeg,
      filled: false,
      filledAt: null,
      dwellMs: 0,
    };
  });
}

/**
 * Advance dwell, fill stations, return the ids that filled on THIS tick.
 *
 * `blocked` is true while a `stop`-severity warning is active. Dwell is not merely paused
 * then, it is RESET: the seconds a user spent aiming while walking produced footage the
 * engine refuses, and letting that time count toward a station would bank exactly the
 * frames the whole feature exists to prevent. Warnings carry hysteresis precisely so this
 * reset cannot be triggered by a flicker.
 */
export function updateStations(stations: StationState[], sample: OrientationSample, now: Millis, blocked: boolean): StationId[] {
  const previous = lastTickAt.get(stations);
  lastTickAt.set(stations, now);
  const filledNow: StationId[] = [];

  if (!Number.isFinite(sample.yawDeg) || !Number.isFinite(sample.pitchDeg)) return filledNow;

  const raw = previous === undefined ? 0 : now - previous;
  const dt = raw <= 0 ? 0 : Math.min(raw, MAX_DWELL_STEP_MS);

  for (const station of stations) {
    if (station.filled) continue;
    const distance = greatCircleDeg(sample.yawDeg, sample.pitchDeg, station.yawDeg, station.pitchDeg);
    // A STATION WHOSE FILL DOES NOT IMPLY THE ELEVATION THE GATES MEASURE IS A STATION THAT
    // LIES. `captureRadiusDeg` is 9.01, derived at types.ts as sqrt(7.5^2 + 5^2) — a 7.5-deg
    // YAW budget (half the station spacing) and a 5-deg PITCH budget — but the great-circle
    // test throws that decomposition away and lets the whole 9 deg go into pitch. Measured
    // against the shipped code, a row held 8 deg low still fills 24 of 24 while
    // `coveredFraction` reads 0.824 and `pitch_max` 55.49: the screen certifies every
    // direction and the review then says directions are missing, with nothing naming which.
    //
    // The radius keeps its yaw job; the pitch component is held to the `amberDeg` it was
    // derived from. NOT `greenDeg` (3): a hand carries about 6 deg of ordinary pitch wobble,
    // and at green a perfectly aimed sweep fails to fill 16/22 of the level row — a capture
    // that can never end is strictly worse than one that ends a degree low.
    //
    // The +/-90 pole reticles are exempt: there every direction is the row.
    if (distance > STATION_RULES.captureRadiusDeg || blocked) {
      station.dwellMs = 0;
      continue;
    }

    // OFF-ROW PITCH PAUSES THE CLOCK. IT MUST NOT RESET IT, and the difference between those
    // two is the difference between a capture that works and one that cannot be finished.
    //
    // Resetting is right for LEAVING the station — the user swung away and the dwell they
    // had earned there is no longer evidence of anything. But a hand carries roughly 6 deg
    // of ordinary pitch wobble at around 1.4 Hz, so folding the pitch test into the reset
    // made `dwellMs` a demand for 250 ms of CONTINUOUSLY in-band pitch inside a station's
    // 750 ms yaw window. A single out-of-band sample erased the clock. Simulated against the
    // shipped coverage code with zero row bias and a 6 deg oscillation: ONE station of 72
    // filled, while `coveredFraction` read 0.9358 — the highest coverage in the whole study.
    // Nothing on screen said why, because `row_pitch` needs 500 ms of sustained error to
    // latch and the excursions are 130-200 ms. Pausing instead keeps the rule's intent — the
    // time credited is only time spent actually looking where the gates measure — without
    // making a steady hand a requirement no hand meets.
    const offRow =
      Math.abs(station.pitchDeg) <= COMPLETENESS.pitchMaxDeg &&
      Math.abs(sample.pitchDeg - station.pitchDeg) > THRESHOLDS.rowPitch.amberDeg;
    if (offRow) continue;

    station.dwellMs += dt;
    if (station.dwellMs >= STATION_RULES.dwellMs) {
      station.filled = true;
      station.filledAt = now;
      station.dwellMs = STATION_RULES.dwellMs;
      filledNow.push(station.id);
    }
  }
  return filledNow;
}

/**
 * The settling pause on the first ring of a new row.
 *
 * WHY A ROW BOUNDARY IS NOT LIKE THE REST OF A ROW. Mid-row, the phone is already moving
 * along the line it has been following and `dwellMs` of 250 is proof enough. Getting to the
 * up or down row means tilting through ~29 degrees first, so the first ring is arrived at
 * on the tail of a deliberate re-aim: the hand has not damped out, and 250 ms of it banks
 * the station off the blurriest frames in the row. That is the "rushed first frame" a real
 * capture reported at both row changes.
 *
 * So the first ring of `up` and `down` asks for `rowArmMs` of continuous aim before it may
 * fill, counted down on screen. Leaving the ring resets the clock, because a hold that was
 * interrupted is not the hold being asked for. `level` is exempt: it starts from standing
 * still with the framing countdown already behind it.
 *
 * Pure, and reset-safe -- the caller owns `since` and hands it back each tick.
 */
export interface RowArm {
  /** Stations must not fill yet. Feeds `updateStations`' `blocked`. */
  blocking: boolean;
  /** Whole seconds still to hold, for the on-screen number. Null when nothing is arming. */
  secondsLeft: number | null;
  /** Carry back next tick. */
  since: Millis | null;
}

const NOT_ARMING: RowArm = { blocking: false, secondsLeft: null, since: null };

export function armRow(
  row: RowId | null,
  stations: readonly StationState[],
  sample: OrientationSample | null,
  since: Millis | null,
  now: Millis,
): RowArm {
  // Only the two rows that are ENTERED BY TILTING, and only before the row has begun: once
  // any ring of it has filled the sweep is under way and every later ring is mid-row.
  if (row !== "up" && row !== "down") return NOT_ARMING;

  let arrivedAtRing = false;
  for (const station of stations) {
    if (station.row !== row) continue;
    if (station.filled) return NOT_ARMING;
    if (arrivedAtRing || sample === null) continue;
    if (!Number.isFinite(sample.yawDeg) || !Number.isFinite(sample.pitchDeg)) continue;
    const distance = greatCircleDeg(sample.yawDeg, sample.pitchDeg, station.yawDeg, station.pitchDeg);
    if (distance <= STATION_RULES.captureRadiusDeg) arrivedAtRing = true;
  }

  // Not on a ring yet: nothing to block and no clock to run. The countdown is a reward for
  // arriving, not a toll charged on the way.
  if (!arrivedAtRing) return NOT_ARMING;

  const started = since ?? now;
  const held = now - started;
  if (held >= STATION_RULES.rowArmMs) {
    // Armed. `since` is kept so a fill that has not landed yet cannot re-arm the row.
    return { blocking: false, secondsLeft: null, since: started };
  }
  return {
    blocking: true,
    secondsLeft: Math.max(1, Math.ceil((STATION_RULES.rowArmMs - held) / 1000)),
    since: started,
  };
}


/**
 * The stations a given capture state is allowed to touch. A ROW THAT IS NOT ITS TURN IS
 * LOCKED — it cannot fill and the bright ring cannot point at it.
 *
 * This is what makes the capture *guided* rather than merely instrumented. Without it two
 * things happened at once, and together they are what made the rows feel out of order:
 *
 *   - The up and down rows filled while the level row was still being swept. The frame is
 *     69 deg tall, so tilting between rows or glancing at the ceiling banked stations 29 deg
 *     away; by the time their row was announced it was already part done and the guidance
 *     picked up somewhere in the middle of it.
 *   - `nextTarget` only PREFERS the active row. Once the nearby level stations were filled it
 *     fell through to a station in another row — which the overlay does not draw, because it
 *     is scoped to the active row. The reticle then pointed at nothing the user could see.
 *
 * `rows` must be arrays built ONCE and mutated in place, not fresh filters: `updateStations`
 * keys its dwell clock on array identity, so a new array each tick would hold every station
 * at zero dwell for ever.
 */
export function stationsForState(
  state: CaptureState,
  rows: Readonly<Record<RowId, StationState[]>>,
  all: StationState[],
): StationState[] {
  const row = (ROW_ID_BY_STATE as Partial<Record<CaptureState, RowId>>)[state];
  return row ? rows[row] : all;
}

/** Drop the dwell clock for an array — call on reset so a stale dt cannot leak across takes. */
export function forgetStationClock(stations: StationState[]): void {
  lastTickAt.delete(stations);
}

/**
 * The bright ring: the nearest unfilled station.
 *
 * Preference is strictly hierarchical — every unfilled station in the active row outranks
 * every station in any other row — so the ring never invites the user to abandon a row
 * halfway. Ties break on index, which makes the choice deterministic and therefore
 * testable; without that a user standing exactly between two stations would see the ring
 * flip on sensor noise.
 */
/**
 * How much closer a rival station must be before the reticle abandons the one it is already
 * pointing at. Six degrees is under half the 15 deg station spacing, so the target still
 * changes the moment the operator genuinely passes one — it just cannot be flipped by noise.
 */
const TARGET_STICKY_MARGIN_DEG = 6;

/**
 * How far behind the current heading a station may sit and still count as "ahead".
 *
 * The reticle ranks candidates CLOCKWISE, not by absolute proximity, so the guide always
 * sends the user the same way round. Without a tolerance that rule bites the moment the
 * operator drifts two degrees past an unfilled station: its clockwise distance flips from 0
 * to 358 and the ring would ask for a complete extra turn to come back to it. Eight degrees
 * is comfortably inside the 15-deg station spacing, so a just-passed station is still the
 * nearest thing to aim at while a genuinely skipped one is left for the next lap.
 */
const BACKWARD_TOLERANCE_DEG = 8;

/**
 * Ranking distance to a station, measured the way the sweep actually travels.
 *
 * ALWAYS CLOCKWISE. The old rule was "nearest unfilled station", which is symmetric, so at
 * the start of a row the ring pointed whichever way the first gap happened to fall — left
 * on the level row, right on the row after it. A guide that reverses between rows is a
 * guide the user has to re-read every time, and re-reading it mid-turn is how a row gets
 * swept twice in opposite directions and stitches badly.
 */
function clockwiseRankDeg(fromYawDeg: Degrees, toYawDeg: Degrees): number {
  const ahead = ((toYawDeg - fromYawDeg) % 360 + 360) % 360;
  return ahead > 360 - BACKWARD_TOLERANCE_DEG ? 360 - ahead : ahead;
}

export function nextTarget(
  stations: StationState[],
  sample: OrientationSample,
  activeRow: RowId | null,
  lens: LensGeometry = DEFAULT_LENS,
  /**
   * The station the reticle is currently on, if any.
   *
   * Without this the choice is recomputed from scratch 30 times a second, and an operator
   * standing between two stations — which, at 15 deg spacing, is most of the time — sees the
   * target flick between them on every tremor. On a real phone the dots "kept moving around"
   * for exactly this reason: nothing was moving except the tie-break.
   */
  previousId?: StationId | null,
): StationTarget | null {
  if (!Number.isFinite(sample.yawDeg) || !Number.isFinite(sample.pitchDeg)) return null;

  let best: StationState | null = null;
  let bestRank = Infinity;
  let bestInRow = false;

  // Inside the active row the ranking is clockwise, so the guide never reverses on the
  // user. Off-row — the touch-up pass, where there is no sweep direction to be consistent
  // with — it stays the plain angular distance.
  const rankOf = (station: StationState, inRow: boolean): number =>
    inRow
      ? clockwiseRankDeg(sample.yawDeg, station.yawDeg)
      : greatCircleDeg(sample.yawDeg, sample.pitchDeg, station.yawDeg, station.pitchDeg);

  for (const station of stations) {
    if (station.filled) continue;
    const inRow = activeRow !== null && station.row === activeRow;
    if (bestInRow && !inRow) continue;
    const rank = rankOf(station, inRow);
    const better = (inRow && !bestInRow) || rank < bestRank - 1e-9;
    if (best === null || better) {
      best = station;
      bestRank = rank;
      bestInRow = inRow;
    }
  }
  if (best === null) return null;

  // Stay on the current target unless the new pick is decisively closer. Only ever applied
  // to a station that is still unfilled and still in the active row, so this can never hold
  // the reticle on something the operator has already completed or left behind.
  if (previousId != null && best.id !== previousId) {
    const held = stations.find((s) => s.id === previousId && !s.filled);
    const heldEligible = held !== undefined && (activeRow === null || held.row === activeRow || !bestInRow);
    if (held !== undefined && heldEligible) {
      const heldInRow = activeRow === null || held.row === activeRow;
      const heldRank = rankOf(held, heldInRow && activeRow !== null);
      if (heldRank - bestRank < TARGET_STICKY_MARGIN_DEG) {
        best = held;
        bestRank = heldRank;
      }
    }
  }

  const frustum = buildFrustum(sample.yawDeg, sample.pitchDeg, sample.rollDeg, lens);
  const station: Station = {
    id: best.id,
    row: best.row,
    index: best.index,
    yawDeg: best.yawDeg,
    pitchDeg: best.pitchDeg,
  };
  return {
    station,
    // The TRUE angular distance, not the clockwise rank that chose it: `onScreen`, the edge
    // chevron and the tests all mean "how far away is it", not "how far round to it".
    distanceDeg: greatCircleDeg(sample.yawDeg, sample.pitchDeg, best.yawDeg, best.pitchDeg),
    yawDeltaDeg: shortestYawDeltaDeg(sample.yawDeg, best.yawDeg),
    pitchDeltaDeg: best.pitchDeg - sample.pitchDeg,
    // The real frustum, not a bounding box: a station 30 deg off in yaw is off-screen even
    // though 30 < vfov/2, and the edge chevron has to appear for it.
    onScreen: insideFrustum(frustum, best.yawDeg, best.pitchDeg),
  };
}

// =====================================================================================
// Row progress
// =====================================================================================

export function rowProgress(
  stations: StationState[],
  row: RowId,
  yawSpanDeg: Degrees,
  pitchMinDeg: Degrees,
  pitchMaxDeg: Degrees,
): RowProgress {
  const plan = ROW_PLAN[row];
  let filledStations = 0;
  for (const s of stations) if (s.row === row && s.filled) filledStations += 1;
  return {
    row,
    filledStations,
    requiredStations: plan.requiredStations,
    yawSpanDeg,
    pitchMinDeg,
    pitchMaxDeg,
    complete: filledStations >= plan.requiredStations && yawSpanDeg + 1e-9 >= YAW_SPAN.acceptMinDeg,
  };
}

/**
 * 0..1 for the three-segment rail at the top of the screen.
 *
 * Deliberately the MINIMUM of the two gates rather than their average: a rail that reads
 * 90% while a hard condition sits at 60% is a rail that lies, and the user finds out only
 * when the row refuses to end. `targetYawSpanDeg` (370) drives the arc rather than the
 * 360 gate, so the last ten degrees of overshoot are visibly asked for.
 */
export function rowFraction(progress: RowProgress): number {
  const plan = ROW_PLAN[progress.row];
  const byStations = plan.requiredStations > 0 ? progress.filledStations / plan.requiredStations : 1;
  const bySpan = plan.targetYawSpanDeg > 0 ? progress.yawSpanDeg / plan.targetYawSpanDeg : 1;
  return clamp01(Math.min(byStations, bySpan));
}

/** Per-row fractions in `ROW_IDS` order, the shape `RowPromptProps.rowFractions` wants. */
export function rowFractions(rows: Record<RowId, RowProgress>): Record<RowId, number> {
  const out = {} as Record<RowId, number>;
  for (const row of ROW_IDS) out[row] = rowFraction(rows[row]);
  return out;
}

/** True when this row has earned its exit. */
export function isRowComplete(progress: RowProgress): boolean {
  return progress.complete;
}

/** 1-based position of a row in the capture order, for `RowPromptProps.rowIndex`. */
export function rowIndex(row: RowId): number {
  return ROW_IDS.indexOf(row) + 1;
}

/** The row after this one, or null when `down` is finished. */
export function nextRow(row: RowId): RowId | null {
  const i = ROW_IDS.indexOf(row);
  return i >= 0 && i + 1 < ROW_IDS.length ? ROW_IDS[i + 1] : null;
}

// =====================================================================================
// Yaw accumulation
// =====================================================================================

/**
 * Accumulate travelled yaw across the 0/360 seam.
 *
 * TOTAL ABSOLUTE TRAVEL, not net displacement, and therefore MONOTONE NON-DECREASING.
 * That property is the whole point: a user who sweeps 370 deg and then turns back 90 deg
 * to re-cover a dim corner must not watch the row un-complete itself. Net displacement
 * would do exactly that, and "a warning must never cost earned progress" (decision D7) is
 * the same principle one level up.
 *
 * The two filters that make it honest:
 *   - a `YAW_DEADBAND_DEG` floor, so magnetometer hash on a phone lying still accrues
 *     nothing;
 *   - a `YAW_MAX_STEP_DEG` ceiling, so a compass re-reference or a seam glitch contributes
 *     nothing rather than a spurious 180 deg.
 *
 * Inflation by rocking on the spot is possible in principle and harmless in practice: a
 * row also needs 22 of its 24 stations, and rocking fills two.
 */
export function accumulateYaw(previousDeg: Degrees, currentDeg: Degrees, spanSoFarDeg: Degrees): Degrees {
  if (!Number.isFinite(previousDeg) || !Number.isFinite(currentDeg)) return spanSoFarDeg;
  const base = Number.isFinite(spanSoFarDeg) ? spanSoFarDeg : 0;
  const step = Math.abs(shortestYawDeltaDeg(previousDeg, currentDeg));
  if (step < YAW_DEADBAND_DEG || step > YAW_MAX_STEP_DEG) return base;
  return base + step;
}

// =====================================================================================
// Completion, stalls and the verdict
// =====================================================================================

/** Which criteria are blocking `done`. Same function `report()` uses, so they cannot drift. */
export function blockers(report: CoverageReport): CompletionBlocker[] {
  return computeBlockers(report);
}

/**
 * True when a row has gone `STUCK_AFTER_MS` without a station filling. Six seconds is long
 * enough not to nag and short enough to rescue the user stuck at eye level in the UP row —
 * the row every measured capture skipped — before they give up on it.
 */
export function isStalled(lastFillAt: Millis | null, now: Millis, afterMs: number = STUCK_AFTER_MS): boolean {
  if (lastFillAt === null) return false;
  return now - lastFillAt >= afterMs;
}

/**
 * Fraction of frames over the octave-1 smear budget past which the capture is called
 * blurry rather than merely degraded. Above 11.08 px the matcher runs on octave 2 alone —
 * 4x fewer keypoints against a 3.0 px RANSAC gate — so a third of the capture in that
 * state is a graph that will not hold together.
 */
export const BLURRY_FRAME_FRACTION = 0.35;

/**
 * Fraction of recorded time with the walking gate tripped past which the capture is called
 * walking. The engine measures 21.17 px of rotational residual on a real walkthrough and
 * refuses it outright; a tenth of a capture spent walking is already several keyframes
 * that will not link.
 */
export const WALKING_TIME_FRACTION = 0.1;

/**
 * PIVOT RADIUS — the rung the ladder was missing, and the one that decides whether the
 * ceiling beams come out straight.
 *
 * `r = |a_radial| / omega^2` is the distance between the optical centre and the axis the
 * phone is actually turning about. Zero is a phone spinning on its own axis; 0.35 m is a
 * phone held at arm's length being swung around the body, and 1.2 m is a walk around a
 * coffee table dressed up as a turn. On the engine's own arm-to-residual ladder those are
 * ~3.6 px and 5.37 px of rotational residual against a 10 px reject — i.e. the measured
 * bedroom capture whose ceiling visibly bends, and half way to outright refusal.
 *
 * Coverage cannot see any of this: an orbit sweeps exactly the same directions as a pivot
 * and fills every station on the way round. Without this rung a competent user with their
 * elbows out is told "bu çekimden temiz bir panorama çıkar" — which is the one thing this
 * screen must never say when it is not true.
 *
 * The limits are the contract's own (`THRESHOLDS.pivotRadius`, whose comment reserves
 * "good" for r <= 0.20 m and calls r ~ 0.35 m degraded): the MEDIAN must stay inside the
 * green band, and the p90 must stay inside the band the contract calls degraded, so a
 * capture that is mostly tight but swings wide on a quarter of the turn is still caught.
 */
export const PIVOT_MEDIAN_LIMIT_M = THRESHOLDS.pivotRadius.greenM;
export const PIVOT_P90_LIMIT_M = 0.35;

/**
 * True when the phone was orbiting the user rather than turning about itself.
 *
 * Exported because the review screen needs the same answer the verdict used — a `gaps`
 * verdict that came from an orbit needs pivot copy, not "%N of the sphere is covered", and
 * two independent re-derivations of that test is exactly how a phone and a server end up
 * telling two different stories.
 *
 * A null radius means the gate never engaged (the phone never turned fast enough for
 * `r = a/omega^2` to be meaningful, or there was no accelerometer at all). That is absence
 * of evidence and is deliberately NOT treated as evidence of an orbit.
 */
export function orbiting(translation: Pick<CaptureTelemetry["translation"], "medianPivotRadiusM" | "p90PivotRadiusM">): boolean {
  const { medianPivotRadiusM: median, p90PivotRadiusM: p90 } = translation;
  const medianWide = median !== null && Number.isFinite(median) && median > PIVOT_MEDIAN_LIMIT_M;
  const tailWide = p90 !== null && Number.isFinite(p90) && p90 > PIVOT_P90_LIMIT_M;
  return medianWide || tailWide;
}

/** Extra facts the caller knows and a `CoverageReport` cannot express. */
export interface VerdictOptions {
  /**
   * FALSE in RAIL MODE: there was no orientation sensor, so nothing painted the sphere and
   * `report.coveredFraction` is 0 because it was never measured — not because the room was
   * missed. A verdict must never turn "we could not look" into "you did not cover it", and
   * it must never call such a capture `great` either.
   */
  coverageMeasured?: boolean;
}

/**
 * The phone's own grade, in the same ladder and the same order
 * `engine/biosphere/pano/validate.py` uses, so the user hears ONE story from their phone
 * and from the server:
 *
 *   too_short -> too_dark -> too_blurry -> walking -> orbit -> (gaps | great)
 *
 * `durationMs` and `options` are extra OPTIONAL parameters — optional, so this stays
 * assignable to `PlanApi["verdict"]` — because `MIN_VIDEO_DURATION_S = 6.0` is a duration
 * gate and "was coverage measurable at all" is a sensor fact, and nothing in a
 * `CoverageReport` can stand in for either honestly.
 */
export function verdict(
  report: CoverageReport,
  telemetry: Pick<CaptureTelemetry, "translation" | "image">,
  durationMs?: number,
  options?: VerdictOptions,
): CaptureVerdict {
  if (durationMs !== undefined && durationMs < DURATION_BUDGET.minSeconds * 1000) return "too_short";

  const { translation, image } = telemetry;

  // validate.py rejects on MEAN brightness; `blockMeanL` (40) is 1.6x the engine's own
  // uncalibrated cliff of 25, so the phone refuses slightly before the server would.
  if (image.meanLuma > 0 && image.meanLuma < THRESHOLDS.brightness.blockMeanL) return "too_dark";

  // "Featureless AND unmatchable" is the engine's rule; the phone's equivalent of that
  // conjunction is a capture that spent most of its frames over the octave-1 smear budget.
  if (image.fractionOverSmearBudget >= BLURRY_FRAME_FRACTION) return "too_blurry";

  // `walkingFraction` IS the drift verdict, and the peak is not allowed a second vote.
  //
  // This used to also fail a capture whose `maxDriftMPerS` touched the stop line even once.
  // That reads as caution and is in fact a way of undoing the detector's own judgement:
  // `walkingGrade` only goes red once the speed estimate has held above the stop line for
  // `translationDrift.stopSustainMs` (1500 ms) — the latch exists precisely because a
  // push-off, a shift of grip, or a step taken to keep balance puts the estimate over the
  // line for a few hundred milliseconds and then the high-pass takes it away again.
  // `walkingFraction` counts the latched samples; `maxDriftMPerS` is a raw per-sample peak
  // with no duration behind it at all.
  //
  // Measured on a real capture (vivo Y22s, 219 s, three complete rows): walkingFraction
  // 0.0251 — the detector called 2.5% of it walking — median drift 0.043 m/s, and one peak
  // at 0.4028 m/s. Failed as "walking". The engine then solved it with a 0.76 degree loop
  // residual and 85% coverage, which is not what a walked capture looks like.
  if (translation.walkingFraction >= WALKING_TIME_FRACTION) return "walking";

  // A capture that never went round at all reads as too_short even if the clock says
  // otherwise: `select._ROW_MIN_YAW_SPAN_DEG` would not count a single one of its rows.
  // Skipped when yaw was never measured — in rail mode every row's span is 0 by definition
  // and calling that "too short" would be a measurement the phone never took.
  const coverageMeasured = options?.coverageMeasured !== false;
  if (coverageMeasured) {
    let countedRows = 0;
    for (const row of ROW_IDS) if (report.rows[row].yawSpanDeg >= YAW_SPAN.rowCountedMinDeg) countedRows += 1;
    if (countedRows === 0) return "too_short";
  }

  // Between walking and coverage: an orbit passes every coverage gate there is, so this has
  // to sit above them or it never fires.
  if (orbiting(translation)) return "gaps";

  // Nothing measured the sphere, so nothing may claim it. `gaps` is the honest floor: the
  // capture is unverified, and the review copy says exactly that.
  if (!coverageMeasured) return "gaps";

  // `blockedBy` IS the coverage verdict, and it is the only one allowed to speak here.
  //
  // This line used to also test `coveredFraction` against the raw `COMPLETENESS` constant,
  // which quietly cancelled the whole point of `coverage.gatesForLens`: that function exists
  // because the constants were measured on a 9:16 frame where three rows reach +55/-71 and a
  // perfect sweep scores 0.886, while a 4:3 frame is 14 degrees shorter and a PERFECT sweep
  // on it scores about 0.80. Testing the raw 0.855 alongside the adapted gate meant a 4:3
  // phone could fill every station of every row, leave no gap, hold the horizon at 100% —
  // and still be told to shoot it again for a shortfall no amount of skill could close.
  //
  // Measured on a real capture that triggered exactly that (vivo Y22s, 1080x1440): rows
  // level/up/down all 1.00, zero gaps, horizon 1.00 and 0.9997, pitch reach +56/-75, and
  // 0.8314 of the sphere — refused, because 0.8314 < 0.855. `computeBlockers` had already
  // passed it.
  if (report.blockedBy.length > 0) return "gaps";
  return "great";
}

// =====================================================================================
// Pacing
// =====================================================================================

/**
 * Seconds a row should take at the coached pace, for the "about 15 seconds" copy and for
 * the fallback progress bar in RAIL MODE, where there is no compass to measure against.
 */
export function rowDurationSeconds(row: RowId, speedDegPerS: number = THRESHOLDS.angularSpeed.greenMaxDegPerS): number {
  const span = ROW_PLAN[row].targetYawSpanDeg;
  const speed = speedDegPerS > 0 ? speedDegPerS : THRESHOLDS.angularSpeed.greenMaxDegPerS;
  return span / speed;
}

/** Total capture budget, rows plus transitions. ~50 s at the coached 24 deg/s. */
export function totalDurationSeconds(speedDegPerS: number = THRESHOLDS.angularSpeed.greenMaxDegPerS): number {
  let total = 0;
  for (const row of ROW_IDS) total += rowDurationSeconds(row, speedDegPerS);
  return total + DURATION_BUDGET.rowTransitionSeconds * (ROW_IDS.length - 1);
}

/**
 * THE TOTAL CAPTURE CEILING, which is NOT `RECORDING.maxDurationMs`.
 *
 * `RECORDING.maxDurationMs` (120 s) is a SEGMENT ceiling — the contract's own words for it
 * are "bounds the blast radius of a MediaRecorder failure or a backgrounding event". A
 * segment that hits it is sealed and a fresh one opened; the capture continues and nothing
 * is lost. Treating it as the ceiling on the whole capture is what silently threw away
 * every take longer than two minutes, and two minutes is inside the range a first-timer
 * spends on three rows, not outside it: 50 s is the coached budget, ~150 s is a measured
 * novice re-aiming for missed stations, and a touch-up pass lands on top of that.
 *
 * 600 s is twelve times the coached budget and, at `RECORDING.targetBitrateBps` of 12 Mbps,
 * about 900 MB. THE UPLOAD IS WHAT SETS THIS NUMBER, not the capture: past the point where
 * the file cannot be sent, every extra second is footage the user shot and cannot use. So
 * this constant and `Settings.max_upload_bytes` on the engine move together, and the engine's
 * is the larger of the two — a client ceiling the server would refuse is the same failure at
 * the same moment as having no ceiling at all.
 *
 * It was 240 s, which is the coached 50 s plus room for a novice re-aiming and a touch-up
 * pass. That turned out to be short of what a real first capture takes: a person learning
 * the motion, correcting a row, and going round again is not unusual at five or six minutes.
 * The countdown is the part that made it feel tighter still — the clock flips to counting
 * down for the last minute, which reads as a one-minute limit rather than as a warning.
 *
 * Reaching it ends the capture into `paused`, where every sealed second is still in hand.
 */
export const MAX_CAPTURE_MS = 600_000;

/** How long before the ceiling the on-screen clock switches to counting down. */
export const CAPTURE_COUNTDOWN_MS = 60_000;

export interface CaptureClock {
  recordedMs: number;
  remainingMs: number;
  /** Show the countdown rather than the elapsed time. */
  counting: boolean;
  /** The capture must end now; nothing recorded past here would upload. */
  exhausted: boolean;
}

/** Pure: the recording clock the bottom chrome shows and the ceiling the loop enforces. */
export function captureClock(recordedMs: number, ceilingMs: number = MAX_CAPTURE_MS): CaptureClock {
  const elapsed = Number.isFinite(recordedMs) && recordedMs > 0 ? recordedMs : 0;
  const remainingMs = Math.max(0, ceilingMs - elapsed);
  return {
    recordedMs: elapsed,
    remainingMs,
    counting: remainingMs <= CAPTURE_COUNTDOWN_MS,
    exhausted: remainingMs <= 0,
  };
}

// =====================================================================================
// RAIL MODE — the timed fallback, when there is no compass to measure against
// =====================================================================================

export interface RailRowProgress {
  /** 0..1 through this row, by the clock alone. */
  fraction: number;
  /** Whole seconds left in this row, for the counter. Never negative. */
  remainingSeconds: number;
  complete: boolean;
}

/**
 * Row progress with no orientation sensor at all.
 *
 * With `deviceorientation` denied there is no yaw, so there are no stations, no coverage
 * stamps and no span — the two conditions a row normally EARNS its exit with cannot even
 * be evaluated. The honest fallback is the one `PermissionGate` already promises the user
 * ("halkalar yerine sayaç görürsünüz"): the coached duration for the row, counted down in
 * front of them, with the copy saying plainly that this is a timer and not a measurement.
 *
 * The target is `rowDurationSeconds` — the SAME arithmetic the coached pace is derived
 * from (370 deg at 24 deg/s), so the timed guide asks for exactly the turn the guided one
 * does. Anything longer would be an invented number, and anything shorter would coach a
 * whip-pan.
 */
export function railRowProgress(
  elapsedMs: number,
  row: RowId,
  speedDegPerS: number = THRESHOLDS.angularSpeed.greenMaxDegPerS,
): RailRowProgress {
  const targetMs = rowDurationSeconds(row, speedDegPerS) * 1000;
  const elapsed = Number.isFinite(elapsedMs) && elapsedMs > 0 ? elapsedMs : 0;
  const remainingMs = Math.max(0, targetMs - elapsed);
  return {
    fraction: clamp01(targetMs > 0 ? elapsed / targetMs : 1),
    remainingSeconds: Math.ceil(remainingMs / 1000),
    complete: remainingMs <= 0,
  };
}

// =====================================================================================
// Framing gates
// =====================================================================================

/** Which gate is holding the countdown, or null when all of them are clear. */
export type FramingGate = "hold" | "level" | "dark" | "bright" | "fps" | null;

export interface FramingInput {
  now: Millis;
  readout: DetectorReadout | null;
  /** Frame-centre elevation, or null when there is no orientation sample. */
  pitchDeg: Degrees | null;
  yawDeg: Degrees | null;
  /** Brightest yaw seen so far while framing, or null before the first sample. */
  brightestYawDeg: Degrees | null;
  /** True while the phone is not upright. NOTHING may start recording in this state. */
  portraitBlocked: boolean;
  /** When `framing` was entered, for the force-start escape hatch. */
  enteredAt: Millis;
  /** Running gate clocks, carried by the caller between ticks. */
  steadySince: Millis | null;
  gatesSince: Millis | null;
}

export interface FramingDecision {
  gate: FramingGate;
  steadySince: Millis | null;
  gatesSince: Millis | null;
  /** Whole seconds left on the 3-2-1, or null when no countdown is running. */
  countdownSeconds: number | null;
  canForceStart: boolean;
  /** All gates held for `countdownMs`: recording starts itself. */
  start: boolean;
}

/**
 * STEADY, LEVEL and LIGHT, then a three-second countdown that recording starts itself at
 * the end of. Every threshold is `THRESHOLDS.framing`, deliberately looser than the live
 * warnings: this gate exists to catch a hopeless start, not to grade a good one.
 *
 * PORTRAIT IS THE FIRST GATE, and it is the reason this function is pure and tested.
 * `readout.rollDeg` comes from a quaternion that has ALREADY been un-rotated by
 * `screen.orientation.angle`, so a phone held perfectly level in landscape reads roll 0 and
 * pitch 0 — every gate passes, the countdown runs invisibly underneath the "turn it
 * upright" prompt, and `row_level` begins recording a file whose field of view is inverted:
 * 69 deg across the SHORT edge. There is no roll or pitch test that can catch this, because
 * by the time the numbers reach here the rotation has been divided out. The only thing that
 * knows is `portraitBlocked`, so it is checked first and it also resets both gate clocks —
 * a rotation must cost the countdown, not merely pause it.
 */
export function framingDecision(input: FramingInput): FramingDecision {
  const framing = THRESHOLDS.framing;

  if (input.portraitBlocked) {
    // Not "hold": the phone IS steady, it is simply sideways. `level` is the true failing
    // gate, and force-start is withheld — an expert must not be able to punch through into
    // a landscape recording either.
    return {
      gate: "level",
      steadySince: null,
      gatesSince: null,
      countdownSeconds: null,
      canForceStart: false,
      start: false,
    };
  }

  // An expert must never be trapped behind a gate that will not clear.
  const canForceStart = input.now - input.enteredAt > framing.forceStartAfterMs;
  const readout = input.readout;
  if (!readout) {
    return {
      gate: "hold",
      steadySince: input.steadySince,
      gatesSince: null,
      countdownSeconds: null,
      canForceStart,
      start: false,
    };
  }

  const steadySince =
    readout.omegaDegPerS <= framing.steadyMaxDegPerS ? (input.steadySince ?? input.now) : null;

  const steady = steadySince !== null && input.now - steadySince >= framing.steadyHoldMs;
  const level =
    Math.abs(readout.rollDeg) <= framing.levelMaxRollDeg && Math.abs(input.pitchDeg ?? 0) <= framing.levelMaxPitchDeg;
  const litEnough = readout.meanLuma >= framing.minMedianLuma;
  const unclipped = readout.clippedFraction <= framing.maxClippedFraction;
  // `measuredFps` is 0 until the first window closes; an unmeasured rate must not block.
  const fastEnough = readout.measuredFps <= 0 || readout.measuredFps >= THRESHOLDS.frameRate.minFps;

  // iPhones expose NO exposure lock — exposureMode and exposureTime are absent from
  // getCapabilities() — so refusing to start pointed at the brightest thing in the room is
  // the only defence against the measured failure where the first and last keyframes blew
  // out on a window and failed to link.
  const facingLight =
    input.yawDeg !== null &&
    input.brightestYawDeg !== null &&
    Math.abs(shortestYawDeltaDeg(input.brightestYawDeg, input.yawDeg)) < framing.awayFromLightDeg &&
    !unclipped;

  const gate: FramingGate = !steady
    ? "hold"
    : !level
      ? "level"
      : !litEnough
        ? "dark"
        : facingLight || !unclipped
          ? "bright"
          : !fastEnough
            ? "fps"
            : null;

  if (gate !== null) {
    return { gate, steadySince, gatesSince: null, countdownSeconds: null, canForceStart, start: false };
  }

  const gatesSince = input.gatesSince ?? input.now;
  const held = input.now - gatesSince;
  if (held >= framing.countdownMs) {
    return { gate: null, steadySince, gatesSince: null, countdownSeconds: null, canForceStart, start: true };
  }
  return {
    gate: null,
    steadySince,
    gatesSince,
    countdownSeconds: Math.max(0, Math.ceil((framing.countdownMs - held) / 1000)),
    canForceStart,
    start: false,
  };
}

// =====================================================================================
// Recorded geometry
// =====================================================================================

/**
 * The pixel geometry the sidecar reports, picked so a zero can never win.
 *
 * `last?.width ?? camera?.width ?? 0` looks right and is not: `0` is not nullish, so a
 * segment sealed before the decoder stated its dimensions poisons the chain and the upload
 * ships `0x0` with `portrait` falling through to its `true` default. The engine is
 * documented to assert the decoded orientation against these two numbers and "reject
 * loudly rather than solve 69 deg across the short edge" — it cannot assert against 0x0.
 *
 * The LAST segment carrying real dimensions wins over the camera, because iOS records
 * sensor-native landscape with a rotation tag and only what actually came out of
 * MediaRecorder is worth putting in the sidecar.
 */
export function recordedFrameSize(
  segments: readonly Pick<RecordedSegment, "width" | "height">[],
  fallback: { width: number; height: number } | null,
): { width: number; height: number } {
  for (let i = segments.length - 1; i >= 0; i -= 1) {
    const segment = segments[i];
    if (segment.width > 0 && segment.height > 0) return { width: segment.width, height: segment.height };
  }
  if (fallback && fallback.width > 0 && fallback.height > 0) return { width: fallback.width, height: fallback.height };
  return { width: 0, height: 0 };
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** The `PlanApi` surface, so a mis-typed export is a compile error here, not in P. */
export const planApi = {
  buildStations,
  updateStations,
  nextTarget,
  rowProgress,
  accumulateYaw,
  blockers,
  verdict,
} satisfies PlanApi;
