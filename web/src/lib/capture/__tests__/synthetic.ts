/**
 * Synthetic sensor streams.
 * =====================================================================================
 *
 * The panorama engine is verified against a synthetic ground-truth harness — a sphere
 * rendered from a known pose set, so "the stitch is correct" is an assertion rather than
 * an opinion. This is the capture UI's equivalent: a perfect three-row sweep, a too-fast
 * sweep, a walking capture, a single-row capture and a sweep with a missing sector, all
 * generated from closed-form motion, so "the coverage model is correct" and "the walking
 * detector fires" are assertions too.
 *
 * Not a test file (no `.test.ts` suffix), so vitest does not collect it as a suite.
 */

import {
  DEFAULT_LENS,
  ROW_IDS,
  THRESHOLDS,
  type AccelSource,
  type CaptureConfig,
  type Degrees,
  type FlowSample,
  type FrameQualitySample,
  type Millis,
  type MotionSample,
  type OrientationSample,
  type Quat,
  type RowId,
  type SensorFrame,
  COMPLETENESS,
  RECORDING,
  ROW_PLAN,
  STATION_RULES,
  YAW_SPAN,
} from "@/lib/capture/types";
import type { LensGeometry } from "@/lib/capture/types";
import { rowTargetPitchDeg } from "@/lib/capture/coverage";

const DEG = Math.PI / 180;

// =====================================================================================
// Orientation
// =====================================================================================

/**
 * Quaternion for the basis `useOrientation.viewAnglesFromQuat` decodes: local +X is the
 * frame's right edge, local +Y its top edge, and the optical axis looks along local **-Z**.
 *
 * This used to build the mirror-image basis (optical axis along +Z, and a roll of the
 * opposite sign), on the argument that no module under test read `q`. That stopped being
 * true the moment the walking gate started rotating linear acceleration into the world frame
 * with it, and a fixture whose quaternion disagrees with the yaw/pitch/roll printed beside it
 * is worse than no quaternion at all. `sensors.test.ts` now asserts the round trip.
 */
export function quatFromYawPitchRoll(yawDeg: Degrees, pitchDeg: Degrees, rollDeg: Degrees): Quat {
  const y = yawDeg * DEG;
  const p = pitchDeg * DEG;
  const r = rollDeg * DEG;
  const sy = Math.sin(y);
  const cy = Math.cos(y);
  const sp = Math.sin(p);
  const cp = Math.cos(p);
  const cr = Math.cos(r);
  const sr = Math.sin(r);

  // Optical axis for (yaw, pitch) in the frame `viewAnglesFromQuat` reads:
  // yaw = atan2(f.x, -f.z), pitch = asin(f.y), so yaw 0 looks along -Z.
  const f = [cp * sy, sp, -cp * cy];
  // Right and up at zero roll. r0 x u0 = -f, which is what makes (r0, u0, -f) right-handed.
  const r0 = [cy, 0, sy];
  const u0 = [-sy * sp, cp, cy * sp];
  // Roll rotates the pair about the optical axis. The sign is fixed by
  // roll = atan2(right.y, up.y): right.y = cos(pitch) sin(roll), up.y = cos(pitch) cos(roll).
  const rv = [r0[0] * cr + u0[0] * sr, r0[1] * cr + u0[1] * sr, r0[2] * cr + u0[2] * sr];
  const uv = [-r0[0] * sr + u0[0] * cr, -r0[1] * sr + u0[1] * cr, -r0[2] * sr + u0[2] * cr];

  const m00 = rv[0];
  const m10 = rv[1];
  const m20 = rv[2];
  const m01 = uv[0];
  const m11 = uv[1];
  const m21 = uv[2];
  const m02 = -f[0];
  const m12 = -f[1];
  const m22 = -f[2];

  const trace = m00 + m11 + m22;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    return [(m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s];
  }
  if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    return [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  }
  if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    return [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s];
  }
  const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
  return [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s];
}

export interface OrientationOptions {
  rollDeg?: Degrees;
  headingAccuracyDeg?: number | null;
  reference?: OrientationSample["reference"];
  screenAngleDeg?: Degrees;
}

export function makeOrientation(t: Millis, yawDeg: Degrees, pitchDeg: Degrees, options: OrientationOptions = {}): OrientationSample {
  const rollDeg = options.rollDeg ?? 0;
  const wrapped = ((yawDeg % 360) + 360) % 360;
  return {
    t,
    q: quatFromYawPitchRoll(wrapped, pitchDeg, rollDeg),
    yawDeg: wrapped,
    pitchDeg,
    rollDeg,
    reference: options.reference ?? "absolute",
    headingAccuracyDeg: options.headingAccuracyDeg ?? 10,
    screenAngleDeg: options.screenAngleDeg ?? 0,
  };
}

// =====================================================================================
// Sweeps
// =====================================================================================

export interface SweepOptions {
  /** Degrees per second the phone turns. The coached pace is 24. */
  speedDegPerS?: number;
  /** Samples per second. `COVERAGE_GRID.stampHz` is 10. */
  sampleHz?: number;
  /** Total yaw travelled per row. The contract's target is 370. */
  yawSpanDeg?: Degrees;
  /** Rows to sweep, in order. */
  rows?: readonly RowId[];
  /** Constant roll, degrees. */
  rollDeg?: Degrees;
  /** Yaw sectors [start, end) that the phone skips entirely, in degrees from the origin. */
  skipSectors?: readonly (readonly [Degrees, Degrees])[];
  /** Starting timestamp. */
  startAt?: Millis;
  /** The lens the sweep aims with; the lattice closes up on a shorter frame. */
  lens?: LensGeometry;
  /** Elevation wobble amplitude within a row, degrees. */
  pitchWobbleDeg?: Degrees;
  /**
   * Constant elevation error applied to the non-level rows, degrees, signed towards the
   * horizon. This is the single most common real capture error — the up row held a few
   * degrees low because looking further up is uncomfortable — and it is what the coverage
   * gate is calibrated against.
   */
  rowBiasDeg?: Degrees;
}

/**
 * A sweep as the sequence of orientation samples a phone would report. Rows run in
 * `ROW_IDS` order (level, up, down — level first sets the loop origin, up second because
 * it is the row users skip, down last because it ends looking at the floor).
 */
export function sweep(options: SweepOptions = {}): OrientationSample[] {
  const speed = options.speedDegPerS ?? THRESHOLDS.angularSpeed.greenMaxDegPerS;
  const hz = options.sampleHz ?? 10;
  const span = options.yawSpanDeg ?? YAW_SPAN.targetDeg;
  const rows = options.rows ?? ROW_IDS;
  const rollDeg = options.rollDeg ?? 0;
  const wobble = options.pitchWobbleDeg ?? 0;
  const bias = options.rowBiasDeg ?? 0;
  // The lattice a sweep aims at depends on the lens: a 4:3 frame closes the rows up.
  const lens = options.lens ?? DEFAULT_LENS;
  const dtMs = 1000 / hz;
  const stepDeg = speed / hz;

  const out: OrientationSample[] = [];
  let t = options.startAt ?? 0;
  for (const row of rows) {
    const aimed = rowTargetPitchDeg(row, lens);
    // Signed towards the horizon, so one number describes the symmetric case: the up row
    // held low and the down row held high, which is what a tiring wrist actually does.
    const pitch = aimed - Math.sign(aimed) * bias;
    const steps = Math.ceil(span / stepDeg);
    for (let i = 0; i <= steps; i += 1) {
      const yaw = Math.min(span, i * stepDeg);
      const wrapped = yaw % 360;
      const skipped = (options.skipSectors ?? []).some(([a, b]) => wrapped >= a && wrapped < b);
      if (!skipped) {
        out.push(makeOrientation(t, wrapped, pitch + wobble * Math.sin((i / 8) * Math.PI), { rollDeg }));
      }
      t += dtMs;
    }
    // Two seconds of row transition, during which the phone pitches to the next row.
    t += 2000;
  }
  return out;
}

/** A perfect three-row sweep at the coached pace. The reference capture. */
export function perfectSweep(): OrientationSample[] {
  return sweep();
}

/** Level row only. Reach is +/-34.5 deg, an unfillable 25.5-deg ring at each pole. */
export function singleRowSweep(): OrientationSample[] {
  return sweep({ rows: ["level"] });
}

/** A three-row sweep with a 60-degree sector never visited. */
export function sweepWithGap(startDeg: Degrees = 120, endDeg: Degrees = 180): OrientationSample[] {
  return sweep({ skipSectors: [[startDeg, endDeg]] });
}

// =====================================================================================
// Motion
// =====================================================================================

/**
 * Every motion fixture carries a NOISE FLOOR, because a detector that only works on a
 * noiseless signal works on no phone at all. 2 mg RMS is the quiet end of the consumer MEMS
 * range, and the 8 Hz term is hand tremor: it sits an octave above the 1-3 Hz step band and
 * two and a half above the 0.3 Hz translation high-pass, so an estimator that lets it into
 * either statistic is measuring the wrong thing.
 *
 * Deterministic, so a red test is reproducible. A plain LCG is plenty: nothing here needs
 * statistical quality, only that the sequence be uncorrelated with any signal frequency.
 */
const NOISE_FLOOR_MS2 = 0.02;
const TREMOR_MS2 = 0.05;

function makeNoise(seed = 0x9e3779b9): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    return (state / 0x100000000 - 0.5) * 2 * NOISE_FLOOR_MS2;
  };
}

/** Hand tremor plus sensor noise, in the device frame. The floor under every fixture. */
function sensorFloor(seconds: number, noise: () => number): { x: number; y: number; z: number } {
  return {
    x: TREMOR_MS2 * Math.sin(2 * Math.PI * 8 * seconds) + noise(),
    y: TREMOR_MS2 * 0.8 * Math.sin(2 * Math.PI * 9.3 * seconds + 1) + noise(),
    z: TREMOR_MS2 * Math.sin(2 * Math.PI * 7.4 * seconds + 2) + noise(),
  };
}

export interface TurnOptions {
  /** Angular speed, deg/s. */
  omegaDegPerS?: number;
  /** Distance from the rotation axis to the lens, metres. 0.35 is a typical hand-held arc. */
  pivotRadiusM?: number;
  sampleHz?: number;
  durationMs?: number;
  startAt?: Millis;
  accelSource?: AccelSource;
  /** Extra body-sway amplitude, m/s^2, on top of the tremor and noise floor. */
  noiseMs2?: number;
  /** Set null to model a phone with no gyroscope: `rotationRate` absent, omega reported 0. */
  gyroscope?: boolean;
}

/**
 * Turning on the spot. The lens orbits the body at `pivotRadiusM`, so the only linear
 * acceleration is centripetal — constant in the DEVICE frame (it points along the optical
 * axis, toward the pivot) and therefore pure DC there, which is exactly why the 1-3 Hz
 * step-cadence detector must ignore it. In the world frame the same vector rotates once per
 * turn, i.e. at 0.067 Hz for the coached 24 deg/s, which is what the translation high-pass
 * has to reject.
 */
export function turningMotion(options: TurnOptions = {}): MotionSample[] {
  const omega = options.omegaDegPerS ?? THRESHOLDS.angularSpeed.greenMaxDegPerS;
  const radius = options.pivotRadiusM ?? 0.35;
  const hz = options.sampleHz ?? 60;
  const duration = options.durationMs ?? 4000;
  const source = options.accelSource ?? "native";
  const sway = options.noiseMs2 ?? 0;
  const gyroscope = options.gyroscope ?? true;
  const dtMs = 1000 / hz;
  const omegaRad = omega * DEG;
  const centripetal = omegaRad * omegaRad * radius;
  const noise = makeNoise();

  const out: MotionSample[] = [];
  const start = options.startAt ?? 0;
  for (let t = start; t <= start + duration; t += dtMs) {
    const s = (t - start) / 1000;
    const floor = sensorFloor(s, noise);
    const jitter = sway * Math.sin(2 * Math.PI * 0.37 * s);
    const accel =
      source === "none"
        ? null
        : { x: floor.x + jitter, y: floor.y + jitter * 0.5, z: -centripetal + floor.z + jitter };
    out.push({
      t,
      rotationRate: gyroscope ? { x: 0, y: omega, z: 0 } : null,
      omegaDegPerS: gyroscope ? omega : 0,
      linearAccel: accel,
      accelMagnitude: accel === null ? 0 : Math.hypot(accel.x, accel.y, accel.z),
      gravity: { x: 0, y: -9.81, z: 0 },
      accelSource: source,
      intervalMs: dtMs,
    });
  }
  return out;
}

/** Standing still with the phone in hand. Tremor and sensor noise, and nothing else. */
export function standingMotion(options: TurnOptions = {}): MotionSample[] {
  return turningMotion({ omegaDegPerS: 0, pivotRadiusM: 0, ...options });
}

export interface WalkOptions extends TurnOptions {
  /** Step cadence in Hz. Humans walk at 1.5-2.5. */
  cadenceHz?: number;
  /**
   * Peak footfall acceleration on the dominant axis, m/s^2. The contract quotes 1-3 m/s^2
   * peak-to-peak for a walk, i.e. 0.5-1.5 peak; 1.2 is mid-range for a phone held out in
   * front of the body, where the arm damps some of what the torso sees.
   */
  stepAmplitudeMs2?: number;
  /** Walking speed, m/s. The engine measures 13.70 px of residual at 0.8. */
  speedMPerS?: number;
  /** Milliseconds of standing still before the walk starts. */
  restLeadMs?: number;
  /**
   * Seconds the body takes to reach `speedMPerS` from rest. 0 models a walk ALREADY UNDER
   * WAY when the fixture starts, which is the hard case: a constant velocity produces no
   * acceleration whatsoever, so nothing but the footfall cadence is left to detect.
   */
  accelerateSeconds?: number;
}

/**
 * Walking through the rooms — the capture the pipeline refuses outright at 21.2 px of
 * rotational residual.
 *
 * WHAT THIS FIXTURE USED TO DO, AND WHY IT WAS A LIE. It injected
 * `speed * 2*pi*0.5 * cos(2*pi*0.5*t)` — a +/-2.5 m/s^2 oscillation at 0.5 Hz, described as
 * the body "speeding up and slowing down within each stride". A walking body does no such
 * thing: 2.5 m/s^2 is a quarter of gravity, sustained, alternating twice a second. It was
 * there because it is exactly what the old leaky double integration needed in order to
 * report anything, and with it the walking test passed against a signal no human produces.
 * Removing it turns a green test red until the detector can see a real gait, which is the
 * point of having the test at all.
 *
 * What remains is what a walk actually is: periodic footfall energy in the 1-3 Hz band, plus
 * (unless `accelerateSeconds` is 0) the one-off acceleration from rest to travel speed.
 */
export function walkingMotion(options: WalkOptions = {}): MotionSample[] {
  const omega = options.omegaDegPerS ?? 18;
  const hz = options.sampleHz ?? 60;
  const duration = options.durationMs ?? 6000;
  const cadence = options.cadenceHz ?? 2;
  const amplitude = options.stepAmplitudeMs2 ?? 1.2;
  const speed = options.speedMPerS ?? 0.8;
  const restLead = (options.restLeadMs ?? 3000) / 1000;
  const rampSeconds = options.accelerateSeconds ?? 0;
  const gyroscope = options.gyroscope ?? true;
  const dtMs = 1000 / hz;
  const noise = makeNoise();

  const out: MotionSample[] = [];
  const start = options.startAt ?? 0;
  for (let t = start; t <= start + duration; t += dtMs) {
    const s = (t - start) / 1000;
    const floor = sensorFloor(s, noise);
    const walked = s - restLead;
    if (walked < 0) {
      out.push(motionSample(t, floor, omega, dtMs, gyroscope));
      continue;
    }
    // Footfall, three axes at the phase offsets a hand-held phone sees: mostly vertical,
    // with fore-aft and lateral components a quarter and three quarters of it.
    const theta = 2 * Math.PI * cadence * walked;
    const envelope = Math.min(1, walked / (1 / cadence));
    const stepX = envelope * amplitude * 0.42 * Math.sin(theta);
    const stepY = envelope * amplitude * Math.sin(theta + 0.7);
    const stepZ = envelope * amplitude * 0.75 * Math.sin(theta + 1.4);
    // Half a sine of push, which integrates to exactly `speed`. Zero when the walk was
    // already under way before the fixture began.
    const ramp =
      rampSeconds > 0 && walked < rampSeconds
        ? ((speed * Math.PI) / (2 * rampSeconds)) * Math.sin((Math.PI * walked) / rampSeconds)
        : 0;
    const accel = { x: floor.x + stepX + ramp, y: floor.y + stepY, z: floor.z + stepZ };
    out.push(motionSample(t, accel, omega, dtMs, gyroscope));
  }
  return out;
}

export interface GlideOptions extends TurnOptions {
  /** Push-off acceleration, m/s^2. */
  pushMs2?: number;
  /** How long the push and the brake each last, seconds. */
  pushSeconds?: number;
  /** How long the body coasts at constant speed between them, seconds. */
  coastSeconds?: number;
  restLeadMs?: number;
}

/**
 * The silent glide: a user pushing off, coasting past a coffee table and braking, with no
 * footfalls at all. Sidestepping, shuffling on a rug, or rolling on an office chair.
 *
 * This is the case that separates an honest translation statistic from a decorative one. The
 * step-cadence detector is blind to it by construction — there is no cadence — and the
 * coasting phase is inertially INVISIBLE, because a constant velocity produces no
 * acceleration. All that exists to be found is the push, and a statistic that throws the
 * push away (as a leaky double integration does within a second) can never see this at all.
 */
export function glidingMotion(options: GlideOptions = {}): MotionSample[] {
  const omega = options.omegaDegPerS ?? 12;
  const hz = options.sampleHz ?? 60;
  const duration = options.durationMs ?? 12_000;
  const push = options.pushMs2 ?? 0.7;
  const pushSeconds = options.pushSeconds ?? 1.2;
  const coastSeconds = options.coastSeconds ?? 2.3;
  const restLead = (options.restLeadMs ?? 3000) / 1000;
  const gyroscope = options.gyroscope ?? true;
  const dtMs = 1000 / hz;
  const noise = makeNoise();

  const out: MotionSample[] = [];
  const start = options.startAt ?? 0;
  for (let t = start; t <= start + duration; t += dtMs) {
    const s = (t - start) / 1000;
    const floor = sensorFloor(s, noise);
    const moved = s - restLead;
    let a = 0;
    if (moved >= 0 && moved < pushSeconds) a = push;
    else if (moved >= pushSeconds + coastSeconds && moved < pushSeconds + coastSeconds + pushSeconds) a = -push;
    const accel = { x: floor.x + a, y: floor.y, z: floor.z };
    out.push(motionSample(t, accel, omega, dtMs, gyroscope));
  }
  return out;
}

function motionSample(
  t: Millis,
  accel: { x: number; y: number; z: number },
  omegaDegPerS: number,
  intervalMs: number,
  gyroscope: boolean,
): MotionSample {
  return {
    t,
    rotationRate: gyroscope ? { x: 0, y: omegaDegPerS, z: 0 } : null,
    omegaDegPerS: gyroscope ? omegaDegPerS : 0,
    linearAccel: accel,
    accelMagnitude: Math.hypot(accel.x, accel.y, accel.z),
    gravity: { x: 0, y: -9.81, z: 0 },
    accelSource: "native",
    intervalMs,
  };
}

/**
 * A sensor stream that DIES: real samples up to `freezeAtMs`, then the same last sample
 * handed back at every tick for the rest of the run, with its own timestamp frozen where the
 * stream stopped. This is exactly what `motionRef.current.read()` does after iOS suspends
 * `devicemotion` on a background/foreground round trip, and it is indistinguishable from a
 * perfectly still user unless the consumer checks the sample's age.
 */
export function frozenAfter<T extends { t: Millis }>(samples: T[], freezeAtMs: Millis): T[] {
  const out: T[] = [];
  let frozen: T | null = null;
  for (const s of samples) {
    if (s.t <= freezeAtMs) {
      out.push(s);
      frozen = s;
    } else if (frozen !== null) {
      out.push(frozen);
    }
  }
  return out;
}

// =====================================================================================
// Image quality and flow
// =====================================================================================

export interface QualityOptions {
  meanLuma?: number;
  clippedFraction?: number;
  exposureS?: number;
  measuredFps?: number;
  omegaDegPerS?: number;
}

export function makeQuality(t: Millis, options: QualityOptions = {}): FrameQualitySample {
  const exposureS = options.exposureS ?? 1 / 120;
  const omega = options.omegaDegPerS ?? THRESHOLDS.angularSpeed.greenMaxDegPerS;
  return {
    t,
    mediaTime: t / 1000,
    meanLuma: options.meanLuma ?? 150,
    clippedFraction: options.clippedFraction ?? 0.01,
    exposureS,
    exposureSource: "reported",
    measuredFps: options.measuredFps ?? RECORDING.targetFps,
    smearPx: omega * exposureS * THRESHOLDS.smear.pxPerDegAt1080,
  };
}

export function makeFlow(t: Millis, divergencePx: number, meanShiftPx = 6): FlowSample {
  return {
    t,
    divergencePx,
    meanShiftPx,
    yawDeltaDeg: meanShiftPx * 0.06,
    valid: true,
  };
}

// =====================================================================================
// Frames and config
// =====================================================================================

export interface FrameOptions {
  orientation?: OrientationSample | null;
  motion?: MotionSample | null;
  quality?: FrameQualitySample | null;
  flow?: FlowSample | null;
  recording?: boolean;
  activeRow?: RowId | null;
}

export function makeFrame(t: Millis, options: FrameOptions = {}): SensorFrame {
  return {
    t,
    // `=== undefined` rather than `??`: an explicit `null` means "there is no orientation"
    // / "no row is being coached", and `??` would silently replace both with a default.
    orientation: options.orientation === undefined ? makeOrientation(t, 0, 0) : options.orientation,
    motion: options.motion === undefined ? null : options.motion,
    quality: options.quality === undefined ? null : options.quality,
    flow: options.flow === undefined ? null : options.flow,
    recording: options.recording ?? true,
    activeRow: options.activeRow === undefined ? "level" : options.activeRow,
  };
}

/** Sampling interval of a stream, from its first strictly increasing gap. */
function inferIntervalMs(samples: readonly { t: Millis }[]): number {
  for (let i = 1; i < samples.length; i += 1) {
    const gap = samples[i].t - samples[i - 1].t;
    if (gap > 0) return gap;
  }
  return 1000 / 60;
}

/**
 * Drive a detector with a stream of motion samples, pairing each with an orientation and a
 * quality sample so every detector sees a complete frame.
 *
 * The FRAME clock is derived from the stream's cadence rather than read off each sample,
 * because the host tick advances whether or not the sensor does. Feed this a `frozenAfter`
 * stream and it reproduces what a dead `devicemotion` really looks like from inside the rAF
 * loop: `frame.t` marching on while `frame.motion.t` stands still.
 */
export function framesFromMotion(motion: MotionSample[], options: FrameOptions & QualityOptions = {}): SensorFrame[] {
  const dtMs = inferIntervalMs(motion);
  const start = motion.length > 0 ? motion[0].t : 0;
  return motion.map((m, i) => {
    const t = start + i * dtMs;
    return makeFrame(t, {
      ...options,
      motion: m,
      // The level row sits at ROW_BIAS_DEG, not on the equator, so a synthetic frame that
      // means "well executed" has to be pitched there — at 0 it is 8 deg off its own row and
      // every detector rightly says so.
      orientation:
        options.orientation ?? makeOrientation(t, ((m.omegaDegPerS * t) / 1000) % 360, rowTargetPitchDeg("level")),
      quality: options.quality ?? makeQuality(t, { ...options, omegaDegPerS: m.omegaDegPerS }),
    });
  });
}

export function makeConfig(overrides: Partial<CaptureConfig> = {}): CaptureConfig {
  return {
    lens: DEFAULT_LENS,
    rows: ROW_PLAN,
    completeness: COMPLETENESS,
    yawSpan: YAW_SPAN,
    station: STATION_RULES,
    recording: RECORDING,
    thresholds: THRESHOLDS,
    qualityMode: "balanced",
    captureKind: "video",
    voiceGuide: false,
    sounds: false,
    haptics: true,
    locale: "tr",
    ...overrides,
  };
}

// =====================================================================================
// Analytic references
// =====================================================================================

/**
 * Exact solid angle of a rectangular pyramid with half-angles a and b, as a fraction of
 * the sphere: Omega = 4 asin(sin a sin b), and the sphere is 4 pi.
 * This is what a single frame's footprint MUST integrate to, at any pitch.
 */
export function frustumSphereFraction(hfovDeg: Degrees, vfovDeg: Degrees): number {
  const a = (hfovDeg / 2) * DEG;
  const b = (vfovDeg / 2) * DEG;
  return (4 * Math.asin(Math.sin(a) * Math.sin(b))) / (4 * Math.PI);
}

/** Covered fraction of a band that reaches `reachDeg` above and below the horizon. */
export function bandSphereFraction(upReachDeg: Degrees, downReachDeg: Degrees = upReachDeg): number {
  // Solid angle of the band between two elevations, as a fraction of the sphere. Written
  // for two reaches rather than one because ROW_BIAS_DEG makes the lattice asymmetric: the
  // band runs from -66.5 to +60.5, not +/-63.5, and sin() of one reach no longer describes it.
  return (Math.sin(upReachDeg * DEG) + Math.sin(downReachDeg * DEG)) / 2;
}
