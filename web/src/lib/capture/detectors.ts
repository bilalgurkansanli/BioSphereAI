/**
 * BioSphere AI — guided capture: the live detectors.
 * =====================================================================================
 *
 * Pure functions from sensor samples to a continuous readout and, above threshold, to
 * exactly one ranked warning. No React, no `three`, no DOM.
 *
 * -------------------------------------------------------------------------------------
 * HYSTERESIS IS NOT POLISH HERE, IT IS THE FEATURE
 * -------------------------------------------------------------------------------------
 * Every threshold in `THRESHOLDS` sits in the middle of the range a human turning on the
 * spot actually occupies. A bare comparison would flicker several times per second on a
 * perfectly good capture, and a user who has seen a warning blink and clear three times
 * will correctly conclude the warning means nothing — at which point the one that matters
 * costs them the capture. So every detector is a two-level latch:
 *
 *   enter  value crosses `enter` and STAYS across for `enterMs`
 *   exit   value falls back past `exit` (= 0.85 * enter for above-is-bad detectors,
 *          1/0.85 * enter for below-is-bad ones) and stays there for `EXIT_SUSTAIN_MS`
 *
 * The 15% band and the 700 ms clear are deliberately asymmetric with the enter sustains
 * (500-1500 ms): warnings should appear promptly and leave reluctantly, because the state
 * they describe — a room that is too dark, a body that is walking — does not actually
 * oscillate at 2 Hz. `Warning.since` reports when the condition FIRST held, not when the
 * latch fired, so the UI can age a warning honestly.
 *
 * -------------------------------------------------------------------------------------
 * THE MOTION GATE FAILS SAFE, NEVER OPEN — AND IT GATES ON AGE, NOT ON PRESENCE
 * -------------------------------------------------------------------------------------
 * MDN is explicit that `DeviceMotionEvent.acceleration` is simply ABSENT on hardware that
 * cannot remove gravity. Absence of evidence of walking must never be read as evidence of
 * standing still — that is precisely how the 21.2 px walkthrough gets through.
 *
 * The subtler version of the same trap, and the one that actually shipped: `SensorFrame`
 * carries whatever the host's `read()` handed back, and every sensor hook in this codebase
 * returns its LAST sample forever. A `devicemotion` stream that dies — iOS after a
 * background/foreground round trip, Android throttling a hidden tab — therefore keeps
 * presenting one frozen sample, which reads as a perfectly still user: drift 0, band 0,
 * `walkingGrade: "green"`, and telemetry that cheerfully claims the gate ran. So every
 * signal here is gated on the AGE of the sample that carries it (`GATE_STALE_MS` for
 * motion, `ORIENTATION_STALE_MS` for orientation), never on its presence. A stale sample is
 * treated exactly like a missing one.
 *
 * When the inertial gate has no fresh evidence, `motionGateDegraded` goes true, the
 * camera-side flow detector's stop threshold TIGHTENS from 7 px to
 * `MOTION_GATE_FALLBACK.flowStopPxWhenDegraded` (5 px), and `summary()` reports
 * `motionGateRan: false` so the server can weight the upload accordingly.
 *
 * -------------------------------------------------------------------------------------
 * WHAT THE WALKING GATE CAN AND CANNOT SEE
 * -------------------------------------------------------------------------------------
 * A constant velocity produces no acceleration. It is not that this estimator is bad at
 * recovering a steady glide — NO inertial estimator can, at any price, because there is
 * nothing in the signal. Every honest inertial walking statistic therefore measures one of
 * two things: the acceleration transient that STARTS a translation, or the periodic
 * footfall energy that accompanies one. This module measures both, and the camera-side flow
 * divergence covers the third case (steady translation already under way) because parallax
 * is a geometric fact rather than an inertial one.
 *
 *   1. `translationMPerS` — the windowed peak of the WORLD-FRAME speed obtained by
 *      high-passing linear acceleration at `translationDrift.highPassHz` and integrating it
 *      ONCE with a 5 s leak. The previous shape (leaky DOUBLE integration, net displacement
 *      over the window) cannot work: both integrators leaked at the same 0.3 Hz corner, so
 *      position decayed to 15 % inside the 1 s window and a realistic 0.8 m/s walk measured
 *      0.068 m/s against a 0.30 m/s stop line. Integrating once and holding the peak keeps
 *      the walk-off transient, which is the only part of a walk an accelerometer ever sees.
 *   2. `stepBandRmsMs2` — the 1-3 Hz band RMS, which is what actually catches steady gait.
 *   3. `flowDivergencePx` — the camera, which catches what neither of the above can.
 *
 * Measured on the fixtures in `__tests__/synthetic.ts`, with the 2 mg noise floor, the hand
 * tremor and the standing-still lead-in a real capture has. `old drift` is the previous
 * leaky double integration run over the identical streams; stop lines are 0.30 m/s and
 * 0.70 m/s^2.
 *
 *                        old drift    new peak speed   band RMS   caught by
 *   standing still          0.004          0.02          0.02     (nothing, correctly)
 *   turn 24 deg/s r=0.35    0.004          0.03          0.04     (nothing, correctly)
 *   turn 45 deg/s r=0.50    0.004          0.21          0.15     (nothing, correctly)
 *   realistic steady gait   0.010          0.12          1.09     step cadence
 *   walk-off                0.093          0.41          1.10     both
 *   silent glide            0.078          0.32          0.16     peak speed ONLY
 *
 * The old drift statistic reached the 0.30 stop line on none of them, and the glide reached
 * neither line, i.e. no warning of any kind. Note also the division of labour: a steady gait
 * is a cadence phenomenon and a glide is a transient one, and neither detector can see the
 * other's case. Note finally the third row — the new statistic is fifty times more sensitive
 * than the old one and STILL says nothing about a fast turn with the elbows out, which is
 * the `pivot_radius` detector's business and not this one's.
 *
 * THE ESTIMATOR IS CLOCKED ON THE SENSOR, NOT ON THE HOST TICK. `push` runs at the rAF
 * rate, which is 120 Hz on ProMotion and 20 Hz on a struggling Android, while
 * `devicemotion` runs at its own rate entirely. Integrating the same acceleration twice
 * because the host ticked twice, or aliasing a 3 Hz band because it ticked once, are both
 * silent. So `createTranslationEstimator` advances only when the MOTION SAMPLE's own
 * timestamp advances, and takes its dt from that.
 *
 * -------------------------------------------------------------------------------------
 * WHY `speedGrade` AND `smearGrade` ARE NOT THE SAME NUMBER
 * -------------------------------------------------------------------------------------
 * They measure two different walls and only coincide in an average room.
 *   speedGrade  is the EXTRACTION wall: `extract_fps * bin_deg / 5` = 30.96 deg/s. Fixed,
 *               because it is a property of the pipeline, not of the light.
 *   smearGrade  is the ADAPTIVE wall: smear = omega * t_exposure * 23.93 px/deg against
 *               SIFT's octave-1 budget of 11.08 px. This IS the adaptive cap
 *               `0.463 / t_exposure` expressed in pixels, so grading it separately gives
 *               the dim-room half-speed coaching without inventing a second speed number.
 * `too_fast` therefore fires on the fixed wall and `smear` on the adaptive one; they carry
 * different copy and different icons and never contradict each other.
 *
 * Both need an angular rate, and a phone with no gyroscope has none. `MotionSample` reports
 * `omegaDegPerS: 0` there, which used to be graded a confident green at any speed: `too_fast`
 * and `smear` could never fire and the framing "hold steady" gate passed instantly while the
 * phone was being waved. So omega is a LADDER — real gyro, then successive orientation
 * quaternions, then nothing — and "nothing" is graded amber rather than green, because a
 * zero that was never measured is not a zero.
 *
 * -------------------------------------------------------------------------------------
 * `stop` MEANS "THIS FRAME CARRIES NO USABLE INFORMATION", NOT "SOMETHING IS UNTIDY"
 * -------------------------------------------------------------------------------------
 * `WARNING_COPY` gives each code its ordinary severity. Three codes are ESCALATED to `stop`
 * when their own contract-declared block condition holds, because at that point the frames
 * arriving are not merely poor, they are empty — and `stop` is the only signal this module
 * has that suspends station fill and `stampCone`:
 *
 *   too_dark    below `brightness.blockMeanL` (40)  — a finger over the lens reads ~5 L*.
 *   glare       above `clipping.blockFraction` (0.15)
 *   track_lost  no fresh orientation at all
 *
 * Without this, ten seconds of a finger over the lens during a fifty-second sweep paints the
 * dome green, fills stations, drops the flow walking-signal (`FlowSample.valid` goes false),
 * leaves the whole-capture mean luma around 120 so the verdict never notices, and tells the
 * user "Çekim tam." Those two thresholds exist in the contract with the word `block` in their
 * names and previously had no code path to one.
 *
 * `rankWarnings` therefore orders by SEVERITY first and `WARNING_PRIORITY` second, so the
 * invariant Builder V relies on — `warnings[0]` is a `stop` if and only if any `stop` is
 * active — survives a code being escalated out of its table position.
 */

import {
  MOTION_GATE_FALLBACK,
  STUCK_AFTER_MS,
  THRESHOLDS,
  WARNING_COPY,
  WARNING_PRIORITY,
  adaptiveSpeedCapDegPerS,
  predictedSmearPx,
  type CaptureConfig,
  type CaptureTelemetry,
  type DetectorInstance,
  type DetectorReadout,
  type DetectorsApi,
  type Degrees,
  type Grade,
  type Millis,
  type OrientationSample,
  type Quat,
  type RowId,
  type SensorFrame,
  type Vec3,
  type Warning,
  type WarningCode,
  type WarningSeverity,
} from "@/lib/capture/types";
import { rowTargetPitchDeg } from "@/lib/capture/coverage";

// =====================================================================================
// Tunables that the contract does not fix, each with its reason
// =====================================================================================

/** How far a value must fall back below an above-is-bad threshold before the latch clears. */
const HYSTERESIS = 0.85;

/** How long it must stay there. Longer than any enter sustain except walking's, on purpose. */
const EXIT_SUSTAIN_MS = 700;

/** dt is clamped into this band: a backgrounded tab must not integrate a 30-second step. */
const MIN_DT_S = 0.001;
const MAX_DT_S = 0.2;

/**
 * No FRESH linear acceleration for this long and the inertial walking gate is declared
 * degraded. Two seconds is ~40 missed samples at the Android worst case of 20 Hz — far
 * beyond a dropped event, comfortably short of a user noticing.
 *
 * Applied to `SensorFrame.motion.t`, not to the presence of `frame.motion`: the host's
 * `read()` hands back the last sample forever, so a dead stream is indistinguishable from a
 * still user unless the age is checked.
 */
const GATE_STALE_MS = 2000;

/**
 * The same rule for orientation, but ten times tighter. Both platforms deliver
 * `deviceorientation` at ~60 Hz, so 500 ms is already 30 missed events, and a frozen
 * attitude is more expensive than a frozen accelerometer: `updateStations` keeps dwelling on
 * whatever station the dead sample happens to sit on and banks it, and `stampCone` keeps
 * painting a direction the camera is no longer pointing.
 */
/**
 * 1500 ms, not 500.
 *
 * A stale orientation sample is supposed to mean "the sensor died" — a real condition that
 * must suspend coverage, because a frozen attitude keeps dwelling on one station and paints
 * the sphere in a direction the camera is no longer pointing.
 *
 * But `deviceorientation` is delivered on the main thread, so anything that janks that
 * thread — a hardware encoder starting, a garbage collection, a layout — starves it just as
 * effectively as a dead sensor. At 500 ms that produced a vicious loop on a real phone: the
 * capture stuttered, orientation went "stale", `track_lost` escalated to a blocking warning,
 * the capture froze with a card over it, and the user stopped turning. The lag was
 * manufacturing its own error.
 *
 * A genuinely dead sensor stays dead for far longer than a second and a half, so this still
 * catches the case it exists for, and stops punishing a phone that merely hiccupped.
 */
const ORIENTATION_STALE_MS = 1500;

/**
 * Anti-windup on the velocity integration, seconds. NOT a signal filter — the 0.3 Hz
 * high-pass upstream is what removes accelerometer bias, and this only bounds the random
 * walk that any integration performs on zero-mean noise. That walk settles at
 * sigma * sqrt(tau * dt / 2), so 5 s of leak on a 2 mg noise floor at 60 Hz is 4 mm/s per
 * axis — measured at 0.02 m/s for a hand-held phone standing still once tremor is included,
 * two orders under the 0.30 m/s stop line — while a body's ~1 s acceleration to walking pace
 * is retained essentially whole.
 *
 * Nothing worse happens if it is too long, because a real deceleration is itself an
 * acceleration and drives the estimate back to zero on its own; the leak only matters for
 * the residue.
 */
const VELOCITY_LEAK_TAU_S = 5;

/**
 * How long the peak of the speed estimate is held, ms.
 *
 * This is the "how long do we keep believing you are still moving after the last evidence"
 * window, and it has to exceed `translationDrift.stopSustainMs` (1500) or a real but brief
 * excursion could never latch at all: a push-off puts the estimate over the stop line for a
 * few hundred milliseconds and then the high-pass takes it away again. 2 s at the 0.30 m/s
 * stop line is 0.6 m of travel, which is past any argument about whether it was parallax.
 */
const TRANSLATION_HOLD_MS = 2 * THRESHOLDS.translationDrift.windowMs;

/** A sensor clock that jumps backwards further than this is a new clock, not a late sample. */
const CLOCK_BACKSTEP_RESET_MS = 1000;

/** Minimum gated samples before a pivot-radius estimate is honest enough to show. */
const MIN_PIVOT_SAMPLES = 12;

/** Enter sustains that `THRESHOLDS` does not name explicitly. */
const DARK_SUSTAIN_MS = 500;
const TRACK_LOST_SUSTAIN_MS = 1000;

/** `webkitCompassAccuracy` past this (or the -1 "uncalibrated" sentinel) is not a heading. */
const MAX_HEADING_ACCURACY_DEG = 25;

/** Quantile store cap. 4096 samples at 60 Hz is 68 s; beyond that it decimates. */
const QUANTILE_CAP = 4096;

const DEG = Math.PI / 180;

/**
 * Age of a sample, ms, with `Infinity` for one that does not exist and 0 for one stamped in
 * the future. The future case is not hypothetical: `performance.now()` and a sensor event's
 * own timestamp can disagree by a frame, and the safe reading of a skewed clock is that the
 * sample is current — treating it as infinitely stale would disable the gate on a healthy
 * phone, which is the fail-open direction.
 */
function ageOf(now: Millis, at: Millis | undefined): number {
  if (at === undefined || !Number.isFinite(at)) return Number.POSITIVE_INFINITY;
  const age = now - at;
  return age > 0 ? age : 0;
}

// =====================================================================================
// Small pure signal-processing pieces
// =====================================================================================

/** One-pole low pass with a time constant in seconds. `y += (x - y) * (1 - e^-dt/tau)`. */
class OnePole {
  private y = 0;
  private primed = false;

  constructor(private readonly tauSeconds: number) {}

  push(x: number, dtSeconds: number): number {
    if (!Number.isFinite(x)) return this.y;
    if (!this.primed) {
      this.y = x;
      this.primed = true;
      return this.y;
    }
    const tau = this.tauSeconds > 0 ? this.tauSeconds : 1e-6;
    const a = 1 - Math.exp(-dtSeconds / tau);
    this.y += (x - this.y) * a;
    return this.y;
  }

  get value(): number {
    return this.y;
  }

  reset(): void {
    this.y = 0;
    this.primed = false;
  }
}

/** One-pole high-pass, the standard `y = a*(y' + x - x')` form, corner in Hz. */
class HighPass {
  private lastIn = 0;
  private lastOut = 0;
  private primed = false;

  constructor(private readonly cornerHz: number) {}

  push(value: number, dtSeconds: number): number {
    if (!Number.isFinite(value)) return this.lastOut;
    if (!this.primed) {
      // Prime on the first sample so the filter never sees a step from zero and injects a
      // phantom impulse into the integrator behind it.
      this.primed = true;
      this.lastIn = value;
      this.lastOut = 0;
      return 0;
    }
    const tau = 1 / (2 * Math.PI * Math.max(this.cornerHz, 1e-6));
    const a = tau / (tau + dtSeconds);
    const out = a * (this.lastOut + value - this.lastIn);
    this.lastIn = value;
    this.lastOut = out;
    return out;
  }

  reset(): void {
    this.primed = false;
    this.lastIn = 0;
    this.lastOut = 0;
  }
}

/**
 * RBJ cookbook band-pass biquad, constant 0 dB peak gain, redesigned when the delivered
 * sample rate moves materially.
 *
 * A difference-of-low-passes is three lines shorter and is the obvious way to build a 1-3 Hz
 * band, but its peak gain is 0.5, not 1: `THRESHOLDS.stepCadence.warnRmsMs2` / `stopRmsMs2`
 * are quoted as the RMS of the WALKING SIGNAL (1-3 m/s^2 peak-to-peak -> 0.35-1.06 RMS), not
 * of a filter's output, so an uncorrected difference filter needs twice the real walking
 * energy before it stops a capture. That is the fail-OPEN direction in the one detector
 * allowed to abort. f0 is the geometric centre sqrt(1*3) = 1.732 Hz and Q = f0/bandwidth =
 * 0.866, which puts -3 dB at exactly 1.0 and 3.0 Hz.
 */
class BandPass {
  private b0 = 0;
  private b2 = 0;
  private a1 = 0;
  private a2 = 0;
  private x1 = 0;
  private x2 = 0;
  private y1 = 0;
  private y2 = 0;
  private designedFs = 0;

  constructor(
    private readonly lowHz: number,
    private readonly highHz: number,
  ) {}

  private design(fs: number): void {
    const f0 = Math.sqrt(this.lowHz * this.highHz);
    const q = f0 / (this.highHz - this.lowHz);
    const w0 = (2 * Math.PI * f0) / fs;
    const alpha = Math.sin(w0) / (2 * q);
    const a0 = 1 + alpha;
    this.b0 = alpha / a0;
    this.b2 = -alpha / a0;
    this.a1 = (-2 * Math.cos(w0)) / a0;
    this.a2 = (1 - alpha) / a0;
    this.designedFs = fs;
  }

  push(value: number, fs: number): number {
    // Below 4x the top of the band the biquad is not trustworthy. Report nothing rather than
    // a number the caller would compare against a calibrated threshold.
    if (!Number.isFinite(value) || !Number.isFinite(fs) || fs < 4 * this.highHz) return 0;
    // Sample jitter must not re-tune the filter every sample, but a phone that drops from
    // 60 Hz to 20 Hz must not keep a stale design either.
    if (this.designedFs === 0 || Math.abs(fs - this.designedFs) / this.designedFs > 0.1) this.design(fs);

    const out = this.b0 * value + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1;
    this.x1 = value;
    this.y2 = this.y1;
    this.y1 = out;
    return out;
  }

  reset(): void {
    this.x1 = this.x2 = this.y1 = this.y2 = 0;
    this.designedFs = 0;
  }
}

/** A time-bounded ring of scalars, with a running sum of squares for a cheap RMS. */
class TimeWindow {
  private times: number[] = [];
  private values: number[] = [];
  private sumSq = 0;

  constructor(private readonly windowMs: number) {}

  push(t: Millis, v: number): void {
    if (!Number.isFinite(v)) return;
    this.times.push(t);
    this.values.push(v);
    this.sumSq += v * v;
    const cutoff = t - this.windowMs;
    let drop = 0;
    while (drop < this.times.length && this.times[drop] < cutoff) {
      this.sumSq -= this.values[drop] * this.values[drop];
      drop += 1;
    }
    if (drop > 0) {
      this.times.splice(0, drop);
      this.values.splice(0, drop);
    }
  }

  get count(): number {
    return this.values.length;
  }

  rms(): number {
    if (this.values.length === 0) return 0;
    const mean = this.sumSq / this.values.length;
    return mean > 0 ? Math.sqrt(mean) : 0;
  }

  mean(): number {
    if (this.values.length === 0) return 0;
    let s = 0;
    for (let i = 0; i < this.values.length; i += 1) s += this.values[i];
    return s / this.values.length;
  }

  /** Largest value still inside the window. O(n) over a window that never exceeds ~250. */
  max(): number {
    let m = 0;
    for (let i = 0; i < this.values.length; i += 1) if (this.values[i] > m) m = this.values[i];
    return m;
  }

  /** Time from the oldest sample still inside the window to the newest, ms. */
  spanMs(): number {
    if (this.times.length < 2) return 0;
    return this.times[this.times.length - 1] - this.times[0];
  }

  reset(): void {
    this.times = [];
    this.values = [];
    this.sumSq = 0;
  }
}

// =====================================================================================
// Attitude helpers
//
// Deliberately duplicated from `useOrientation.ts` rather than imported. That module is a
// React hook file, and `vitest.config.mts` states as a property of this one that it runs
// under plain Node with no React in the graph; a runtime import would quietly make that
// false. The duplication is fifteen lines of arithmetic with no state, and
// `__tests__/sensors.test.ts` pins it against `useOrientation`'s own implementation so the
// two cannot drift apart.
// =====================================================================================

/** Rotate a vector by a quaternion. Port of THREE.Vector3.applyQuaternion. */
function rotateByQuat(q: Quat, v: Vec3): Vec3 {
  const [qx, qy, qz, qw] = q;
  const tx = 2 * (qy * v.z - qz * v.y);
  const ty = 2 * (qz * v.x - qx * v.z);
  const tz = 2 * (qx * v.y - qy * v.x);
  return {
    x: v.x + qw * tx + qy * tz - qz * ty,
    y: v.y + qw * ty + qz * tx - qx * tz,
    z: v.z + qw * tz + qx * ty - qy * tx,
  };
}

/** Unsigned rotation angle between two orientations, degrees. */
function quatAngleBetweenDeg(a: Quat, b: Quat): Degrees {
  const dot = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return (2 * Math.acos(Math.min(1, dot)) * 180) / Math.PI;
}

/**
 * ACCELEROMETER-frame vector (x right, y top edge, z out of the screen) to the world frame
 * the camera quaternion lives in.
 *
 * `quatFromDeviceOrientation` applies the screen spin on the LOCAL side, so the camera
 * quaternion already contains it and the accelerometer axes do not. Undoing it is a plain
 * rotation about the device's own z axis by +`screenAngleDeg`, after which the camera
 * quaternion carries the vector the rest of the way. Equivalent to
 * `rotateVec(deviceFrameQuat(q, angle), v)` and asserted to be, in sensors.test.ts.
 */
export function deviceToWorld(q: Quat, screenAngleDeg: Degrees, v: Vec3): Vec3 {
  const a = (Number.isFinite(screenAngleDeg) ? screenAngleDeg : 0) * DEG;
  const c = Math.cos(a);
  const s = Math.sin(a);
  return rotateByQuat(q, { x: v.x * c - v.y * s, y: v.x * s + v.y * c, z: v.z });
}

// =====================================================================================
// The translation estimator — ONE implementation, two homes
//
// `useMotion.ts` runs an instance at the native `devicemotion` rate and exposes it as
// `readTranslation()`; the detector runs its own over whatever `SensorFrame.motion` carries,
// so the walking gate works whether or not the host remembers to wire the better-sampled
// one in. Both are this code, so the two can never disagree about what the numbers mean.
// =====================================================================================

/** One reading for the translation estimator. */
export interface TranslationInput {
  /** The SENSOR's own timestamp. The estimator is clocked on this, never on the host tick. */
  t: Millis;
  /** Device-frame linear acceleration, m/s^2, gravity already removed by the platform. */
  accel: Vec3;
  /** Camera quaternion and `screen.orientation.angle`, when an attitude is available. */
  attitude: { q: Quat; screenAngleDeg: Degrees } | null;
}

/** The two windowed statistics, plus an honest account of whether they ran at all. */
export interface TranslationEstimate {
  /**
   * Windowed peak of the estimated WORLD-FRAME speed, m/s. Graded against
   * `THRESHOLDS.translationDrift`, whose 0.15 / 0.30 lines come from validate.py's own
   * ladder (a 0.35 m arm at 24 deg/s traces 0.146 m/s of chord and measures 2.34 px, which
   * is accepted; walking at 0.80 m/s measures 13.70 px, which is refused).
   *
   * NOT odometry, and never to be shown as a distance. It is the acceleration transient
   * that starts a translation, integrated once and held; a translation already under way at
   * constant speed leaves no inertial trace at all and is the flow detector's job.
   */
  driftMPerS: number;
  /** RMS of the 1-3 Hz band-passed acceleration VECTOR, m/s^2. `THRESHOLDS.stepCadence`. */
  stepBandRmsMs2: number;
  /** FALSE means no usable acceleration ever arrived and neither number was measured. */
  gateRan: boolean;
  /** True when the integration ran in a fixed world frame instead of the tumbling device one. */
  worldFrame: boolean;
  /** Seconds of history behind the numbers. Under ~1 s they are still warming up. */
  windowSeconds: number;
  /** Timestamp of the newest sample behind the numbers; -Infinity before the first. */
  at: Millis;
}

export interface TranslationEstimator {
  push: (input: TranslationInput) => TranslationEstimate;
  read: () => TranslationEstimate;
  reset: () => void;
}

export function createTranslationEstimator(): TranslationEstimator {
  const drift = THRESHOLDS.translationDrift;
  const step = THRESHOLDS.stepCadence;

  const hp = [new HighPass(drift.highPassHz), new HighPass(drift.highPassHz), new HighPass(drift.highPassHz)];
  const band = [
    new BandPass(step.bandLowHz, step.bandHighHz),
    new BandPass(step.bandLowHz, step.bandHighHz),
    new BandPass(step.bandLowHz, step.bandHighHz),
  ];
  const bandWindow = new TimeWindow(step.windowMs);
  const speedWindow = new TimeWindow(TRANSLATION_HOLD_MS);

  let vx = 0;
  let vy = 0;
  let vz = 0;
  let lastT: Millis | null = null;
  let rateHz = 0;
  let worldFrame = false;
  let gateRan = false;
  let at = Number.NEGATIVE_INFINITY;

  function read(): TranslationEstimate {
    return {
      driftMPerS: speedWindow.max(),
      stepBandRmsMs2: bandWindow.rms(),
      gateRan,
      worldFrame,
      windowSeconds: speedWindow.spanMs() / 1000,
      at,
    };
  }

  function reset(): void {
    for (const f of hp) f.reset();
    for (const f of band) f.reset();
    bandWindow.reset();
    speedWindow.reset();
    vx = vy = vz = 0;
    lastT = null;
    rateHz = 0;
    worldFrame = false;
    gateRan = false;
    at = Number.NEGATIVE_INFINITY;
  }

  function push(input: TranslationInput): TranslationEstimate {
    const t = input.t;
    const a = input.accel;
    if (!Number.isFinite(t) || !Number.isFinite(a.x) || !Number.isFinite(a.y) || !Number.isFinite(a.z)) return read();

    if (lastT !== null) {
      // A host that ticks faster than the sensor hands back the SAME sample repeatedly.
      // Integrating it again would inflate every statistic in proportion to the tick rate.
      if (t <= lastT - CLOCK_BACKSTEP_RESET_MS) reset();
      else if (t <= lastT) return read();
    }

    const dtRaw = lastT === null ? 0 : (t - lastT) / 1000;
    const dt = Math.min(MAX_DT_S, Math.max(MIN_DT_S, dtRaw || 1 / 60));
    if (dtRaw > 0) {
      // 2 s time constant on the rate estimate: fast enough to follow a phone dropping from
      // 60 Hz to 20 Hz, slow enough that per-sample jitter never re-tunes the biquad.
      const alpha = 1 - Math.exp(-dt / 2);
      rateHz = rateHz === 0 ? 1 / dtRaw : rateHz + (1 / dtRaw - rateHz) * alpha;
    }
    lastT = t;
    at = t;
    gateRan = true;

    // Integrate in a fixed frame whenever an attitude is known. In the device frame a walk
    // that curves partially cancels itself, which UNDERSTATES the drift — the wrong direction
    // to be wrong in for the one detector allowed to abort a capture. It also matters for the
    // bias: gravity leakage is DC in the world frame and a slow oscillation in the device
    // frame, and only the DC version is removed cleanly by the high-pass.
    let wx = a.x;
    let wy = a.y;
    let wz = a.z;
    worldFrame = false;
    if (input.attitude !== null) {
      const w = deviceToWorld(input.attitude.q, input.attitude.screenAngleDeg, a);
      wx = w.x;
      wy = w.y;
      wz = w.z;
      worldFrame = true;
    }

    const leak = Math.exp(-dt / VELOCITY_LEAK_TAU_S);
    vx = vx * leak + hp[0].push(wx, dt) * dt;
    vy = vy * leak + hp[1].push(wy, dt) * dt;
    vz = vz * leak + hp[2].push(wz, dt) * dt;
    speedWindow.push(t, Math.hypot(vx, vy, vz));

    // Step cadence stays in the DEVICE frame on purpose: footfall energy is carried by the
    // magnitude of the band-passed vector, which is invariant to how the phone is held, and
    // skipping the rotation keeps this signal alive when orientation has been denied.
    //
    // FILTER FIRST, THEN TAKE THE MAGNITUDE — never the other way round. The contract phrases
    // this as band-passing |linear acceleration|, and band-passing the magnitude is exactly
    // wrong: |sin(2 pi f t)| is a RECTIFIED wave whose fundamental sits at 2f, so a 2 Hz gait
    // (the dead centre of human cadence) is moved to 4 Hz and lands outside the band it is
    // being measured in. Measured on the realistic-gait fixture: 1.09 m/s^2 filtering per
    // axis, 0.18 filtering the magnitude — the difference between stopping a walking capture
    // and waving it through.
    const fs = rateHz > 0 ? rateHz : 60;
    const bx = band[0].push(a.x, fs);
    const by = band[1].push(a.y, fs);
    const bz = band[2].push(a.z, fs);
    bandWindow.push(t, Math.hypot(bx, by, bz));

    return read();
  }

  return { push, read, reset };
}

/** Decimating sample store, so a 120 s capture cannot grow an unbounded array. */
class Quantiles {
  private data: number[] = [];
  private stride = 1;
  private seen = 0;

  push(v: number): void {
    if (!Number.isFinite(v)) return;
    this.seen += 1;
    if (this.seen % this.stride !== 0) return;
    this.data.push(v);
    if (this.data.length > QUANTILE_CAP) {
      const kept: number[] = [];
      for (let i = 0; i < this.data.length; i += 2) kept.push(this.data[i]);
      this.data = kept;
      this.stride *= 2;
    }
  }

  get count(): number {
    return this.data.length;
  }

  quantile(q: number): number | null {
    if (this.data.length === 0) return null;
    const sorted = this.data.slice().sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
    return sorted[idx];
  }

  max(): number {
    let m = 0;
    for (let i = 0; i < this.data.length; i += 1) if (this.data[i] > m) m = this.data[i];
    return m;
  }

  mean(): number {
    if (this.data.length === 0) return 0;
    let s = 0;
    for (let i = 0; i < this.data.length; i += 1) s += this.data[i];
    return s / this.data.length;
  }

  reset(): void {
    this.data = [];
    this.stride = 1;
    this.seen = 0;
  }
}

/**
 * Two-level hysteresis latch. `enterHeld` and `exitHeld` are booleans the caller computes
 * from its own thresholds, which keeps every threshold comparison visible at its call site
 * rather than buried in a shared helper.
 */
class Latch {
  private candidateSince: Millis | null = null;
  private clearSince: Millis | null = null;
  private activeSince: Millis | null = null;

  update(now: Millis, enterHeld: boolean, exitHeld: boolean, enterMs: number, exitMs: number = EXIT_SUSTAIN_MS): boolean {
    if (this.activeSince === null) {
      if (!enterHeld) {
        this.candidateSince = null;
        return false;
      }
      if (this.candidateSince === null) this.candidateSince = now;
      if (now - this.candidateSince >= enterMs) {
        this.activeSince = this.candidateSince;
        this.clearSince = null;
      }
      return this.activeSince !== null;
    }
    if (!exitHeld) {
      this.clearSince = null;
      return true;
    }
    if (this.clearSince === null) this.clearSince = now;
    if (now - this.clearSince >= exitMs) {
      this.activeSince = null;
      this.candidateSince = null;
      this.clearSince = null;
      return false;
    }
    return true;
  }

  /** When the condition FIRST held continuously, not when the latch fired. */
  get since(): Millis | null {
    return this.activeSince;
  }

  get active(): boolean {
    return this.activeSince !== null;
  }

  reset(): void {
    this.candidateSince = null;
    this.clearSince = null;
    this.activeSince = null;
  }
}

// =====================================================================================
// Grading
// =====================================================================================

/** Lower is better: green at or below `green`, amber at or below `amber`, red above. */
export function gradeAscending(value: number, green: number, amber: number): Grade {
  if (!Number.isFinite(value)) return "green";
  if (value <= green) return "green";
  if (value <= amber) return "amber";
  return "red";
}

/** Higher is better: green at or above `green`, amber at or above `amber`, red below. */
export function gradeDescending(value: number, green: number, amber: number): Grade {
  if (!Number.isFinite(value)) return "green";
  if (value >= green) return "green";
  if (value >= amber) return "amber";
  return "red";
}

/** Hoisted out of `worstGrade`: that runs three times per sensor frame, i.e. 180 Hz. */
const GRADE_RANK: Readonly<Record<Grade, number>> = { green: 0, amber: 1, red: 2 };

/** The worse of two grades. */
export function worstGrade(a: Grade, b: Grade): Grade {
  return GRADE_RANK[a] >= GRADE_RANK[b] ? a : b;
}

// =====================================================================================
// Warnings
// =====================================================================================

const WARNING_UNIT: Readonly<Record<WarningCode, Warning["unit"]>> = {
  walking: "m/s",
  too_fast: "deg/s",
  smear: "px",
  pivot_radius: "m",
  roll: "deg",
  row_pitch: "deg",
  too_dark: "L*",
  glare: "fraction",
  low_fps: "fps",
  lens_changed: "fraction",
  landscape: "deg",
  track_lost: "deg",
};

/**
 * `titleKey`, `bodyKey`, `voiceKey` and `icon` always come from `WARNING_COPY`, so the
 * mapping lives in exactly one place and Builder P's `copy.ts` cannot drift from what this
 * module emits. So does `severity`, unless `escalateTo` is given.
 *
 * `escalateTo` exists for the three codes whose contract carries a SECOND, harder threshold
 * that the copy table has no way to express — `brightness.blockMeanL`, `clipping.
 * blockFraction`, and orientation going absent altogether. Past those the frames arriving
 * carry no usable information, and `stop` is the only severity that suspends station fill
 * and coverage stamping. It is only ever used to escalate, never to soften: a `stop` in the
 * table stays a `stop`.
 */
export function buildWarning(
  code: WarningCode,
  since: Millis,
  value: number,
  threshold: number,
  haptics: boolean,
  escalateTo?: WarningSeverity,
): Warning {
  const meta = WARNING_COPY[code];
  const severity =
    escalateTo !== undefined && SEVERITY_RANK[escalateTo] > SEVERITY_RANK[meta.severity] ? escalateTo : meta.severity;
  const haptic = !haptics ? [] : severity === "stop" ? [80, 60, 80] : severity === "warn" ? [24, 40, 24] : [];
  return {
    code,
    severity,
    since,
    value,
    threshold,
    unit: WARNING_UNIT[code],
    titleKey: meta.title,
    bodyKey: meta.body,
    voiceKey: meta.voice,
    icon: meta.icon,
    haptic,
  };
}

/** Built once: `rankWarnings` runs on every sensor frame. */
const PRIORITY_RANK: ReadonlyMap<WarningCode, number> = new Map(WARNING_PRIORITY.map((code, i) => [code, i]));

/** Worst first, so a comparator can put every `stop` above every `warn` above every `info`. */
const SEVERITY_RANK: Readonly<Record<WarningSeverity, number>> = { info: 0, warn: 1, stop: 2 };

/**
 * Sort by SEVERITY first, `WARNING_PRIORITY` second. Only `[0]` is ever rendered — a person
 * mid-turn cannot triage.
 *
 * `WARNING_PRIORITY` already lists all three table-level `stop` codes above every `warn`, so
 * for an un-escalated set this is exactly the priority order and nothing changes. The
 * severity key exists because `too_dark`, `glare` and `track_lost` can be escalated out of
 * their table positions (see `buildWarning`), and Builder V's contract — "warnings[0] is a
 * stop if and only if any stop is active" — has to survive that. Without it, a `warn`-level
 * `too_fast` at priority 3 would hide a `stop`-level `too_dark` at priority 7 and the host,
 * which reads severity off `warnings[0]`, would carry on painting the dome from black frames.
 */
export function rankWarnings(warnings: Warning[]): Warning[] {
  const priority = (w: Warning): number => PRIORITY_RANK.get(w.code) ?? WARNING_PRIORITY.length;
  return warnings.slice().sort((a, b) => SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity] || priority(a) - priority(b));
}

/** True when `severity` should suspend station capture — `stop` only, by contract. */
export function blocksCapture(warnings: Warning[]): boolean {
  return warnings.some((w) => w.severity === "stop");
}

// =====================================================================================
// The detector
// =====================================================================================

/** External conditions the detector cannot see in a `SensorFrame`, pushed in by the host. */
export interface ExternalFlags {
  /** `screen.orientation.angle` left portrait. Recording is PAUSED, never stopped. */
  landscape?: boolean;
  /** `CameraApi.configurationChanged`: zoom or the active physical lens moved mid-capture. */
  lensChanged?: boolean;
}

/**
 * A `DetectorInstance` plus the two inputs the contract's `SensorFrame` does not carry.
 * Structurally assignable to `DetectorInstance`, so Builder P may hold it as either.
 */
export interface CaptureDetector extends DetectorInstance {
  /** Latched immediately; both conditions are binary facts, not noisy measurements. */
  setExternalFlags: (flags: ExternalFlags) => void;
  /**
   * OPTIONAL, and the detector is correct without it: hand in
   * `MotionApiExt.readTranslation()` once per tick and the walking statistics come from an
   * estimator that saw EVERY `devicemotion` sample instead of the subset that happened to
   * survive the rAF tick. Same code either way (`createTranslationEstimator`), so the two can
   * only differ in how well they were sampled — which is exactly why the better-sampled one
   * wins when it is fresh. A stale or absent one falls back, silently and safely, to the
   * detector's own.
   */
  setTranslation: (estimate: TranslationEstimate | null) => void;
  /** Feed the tick of every station fill so the stall coach knows the row is moving. */
  noteStationFilled: (now: Millis) => void;
  /** True when nothing has filled for `STUCK_AFTER_MS`. Drives `RowPromptProps.stuck`. */
  isStalled: (now: Millis, afterMs?: number) => boolean;
}

/** Sample ages, measured once per tick and shared by `measure` and `evaluate`. */
interface SampleAges {
  motionMs: number;
  orientationMs: number;
}

const EMPTY_READOUT: DetectorReadout = {
  omegaDegPerS: 0,
  speedCapDegPerS: THRESHOLDS.angularSpeed.adaptiveMaxDegPerS,
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
  meanLuma: 0,
  clippedFraction: 0,
  exposureGrade: "green",
  measuredFps: 0,
  motionGateDegraded: true,
};

class Detector implements CaptureDetector {
  private readonly omegaLp = new OnePole(THRESHOLDS.angularSpeed.lowPassSeconds);
  private readonly translation = createTranslationEstimator();
  private readonly radialAccel = new TimeWindow(THRESHOLDS.pivotRadius.windowMs);
  private readonly omegaSq = new TimeWindow(THRESHOLDS.pivotRadius.windowMs);
  private readonly fpsWindow = new TimeWindow(THRESHOLDS.frameRate.windowMs);

  private readonly latches: Record<string, Latch> = {
    too_fast: new Latch(),
    smear: new Latch(),
    pivot_radius: new Latch(),
    drift: new Latch(),
    cadence: new Latch(),
    flow: new Latch(),
    roll: new Latch(),
    row_pitch: new Latch(),
    too_dark: new Latch(),
    // The block latches are separate from the warn ones so that "dim" and "no picture at
    // all" can enter and clear on their own thresholds instead of sharing one.
    too_dark_block: new Latch(),
    glare: new Latch(),
    glare_block: new Latch(),
    low_fps: new Latch(),
    track_lost: new Latch(),
  };

  private readout: DetectorReadout = { ...EMPTY_READOUT };
  private lastT: Millis | null = null;
  private gateSamples = 0;
  private lastStationFillAt: Millis | null = null;
  private external: ExternalFlags = {};
  private landscapeSince: Millis | null = null;
  private lensChangedSince: Millis | null = null;
  private activeWarnings: Warning[] = [];
  private supplied: TranslationEstimate | null = null;

  // Orientation-derived angular rate, for hardware with no gyroscope.
  private lastQuat: Quat | null = null;
  private lastQuatT: Millis | null = null;
  private derivedOmegaDegPerS = 0;

  // Telemetry aggregates, accumulated only while `frame.recording` is true.
  private recordedTicks = 0;
  private walkingTicks = 0;
  private overSpeedTicks = 0;
  private omegaSamples = new Quantiles();
  private pivotSamples = new Quantiles();
  private driftSamples = new Quantiles();
  private lumaSum = 0;
  private lumaCount = 0;
  private lumaMin = Infinity;
  private clipSum = 0;
  private clipMax = 0;
  private smearFrames = 0;
  private overSmearFrames = 0;

  constructor(private readonly config: CaptureConfig) {}

  setExternalFlags(flags: ExternalFlags): void {
    if (flags.landscape !== undefined) {
      this.external.landscape = flags.landscape;
      if (!flags.landscape) this.landscapeSince = null;
    }
    if (flags.lensChanged !== undefined) {
      this.external.lensChanged = flags.lensChanged;
      if (!flags.lensChanged) this.lensChangedSince = null;
    }
  }

  setTranslation(estimate: TranslationEstimate | null): void {
    this.supplied = estimate;
  }

  noteStationFilled(now: Millis): void {
    this.lastStationFillAt = now;
  }

  isStalled(now: Millis, afterMs: number = STUCK_AFTER_MS): boolean {
    if (this.lastStationFillAt === null) return false;
    return now - this.lastStationFillAt >= afterMs;
  }

  push(frame: SensorFrame): DetectorReadout {
    const now = Number.isFinite(frame.t) ? frame.t : (this.lastT ?? 0);
    const rawDt = this.lastT === null ? 0 : (now - this.lastT) / 1000;
    const dt = Math.min(MAX_DT_S, Math.max(MIN_DT_S, rawDt));
    this.lastT = now;

    // AGE, not presence. Every sensor hook in this codebase returns its last sample forever,
    // so `frame.motion !== null` says nothing about whether the stream is alive. A negative
    // age (a sample stamped in the future by a clock that stepped) is clamped to 0 rather
    // than treated as infinitely stale, because the safe reading of a skewed clock is that
    // the sample is current.
    const ages: SampleAges = {
      motionMs: ageOf(now, frame.motion?.t),
      orientationMs: ageOf(now, frame.orientation?.t),
    };

    const readout = this.measure(frame, now, dt, ages);
    this.readout = readout;
    this.activeWarnings = this.evaluate(frame, readout, now, ages);
    if (frame.recording) this.accumulate(frame, readout);
    return readout;
  }

  warnings(): Warning[] {
    return this.activeWarnings;
  }

  summary(): Pick<CaptureTelemetry, "translation" | "image"> & { sweep: Partial<CaptureTelemetry["sweep"]> } {
    const ticks = this.recordedTicks > 0 ? this.recordedTicks : 1;
    return {
      translation: {
        medianPivotRadiusM: this.pivotSamples.count >= MIN_PIVOT_SAMPLES ? this.pivotSamples.quantile(0.5) : null,
        p90PivotRadiusM: this.pivotSamples.count >= MIN_PIVOT_SAMPLES ? this.pivotSamples.quantile(0.9) : null,
        medianDriftMPerS: this.driftSamples.quantile(0.5) ?? 0,
        maxDriftMPerS: this.driftSamples.max(),
        walkingFraction: this.walkingTicks / ticks,
        // FALSE means the inertial gate never ran. Absence of evidence, not evidence.
        motionGateRan: this.gateSamples > 0,
      },
      image: {
        meanLuma: this.lumaCount > 0 ? this.lumaSum / this.lumaCount : 0,
        minLuma: Number.isFinite(this.lumaMin) ? this.lumaMin : 0,
        meanClippedFraction: this.lumaCount > 0 ? this.clipSum / this.lumaCount : 0,
        maxClippedFraction: this.clipMax,
        fractionOverSmearBudget: this.smearFrames > 0 ? this.overSmearFrames / this.smearFrames : 0,
      },
      sweep: {
        meanAngularVelocityDegPerS: this.omegaSamples.mean(),
        p95AngularVelocityDegPerS: this.omegaSamples.quantile(0.95) ?? 0,
        fractionOverSpeedCap: this.overSpeedTicks / ticks,
      },
    };
  }

  reset(): void {
    this.omegaLp.reset();
    this.translation.reset();
    this.radialAccel.reset();
    this.omegaSq.reset();
    this.fpsWindow.reset();
    for (const key of Object.keys(this.latches)) this.latches[key].reset();
    this.readout = { ...EMPTY_READOUT };
    this.lastT = null;
    this.supplied = null;
    this.lastQuat = null;
    this.lastQuatT = null;
    this.derivedOmegaDegPerS = 0;
    this.gateSamples = 0;
    this.lastStationFillAt = null;
    this.external = {};
    this.landscapeSince = null;
    this.lensChangedSince = null;
    this.activeWarnings = [];
    this.recordedTicks = 0;
    this.walkingTicks = 0;
    this.overSpeedTicks = 0;
    this.omegaSamples.reset();
    this.pivotSamples.reset();
    this.driftSamples.reset();
    this.lumaSum = 0;
    this.lumaCount = 0;
    this.lumaMin = Infinity;
    this.clipSum = 0;
    this.clipMax = 0;
    this.smearFrames = 0;
    this.overSmearFrames = 0;
  }

  /** The last readout, for hosts that tick faster than they render. */
  get last(): DetectorReadout {
    return this.readout;
  }

  // ---------------------------------------------------------------------------------

  private measure(frame: SensorFrame, now: Millis, dt: number, ages: SampleAges): DetectorReadout {
    const speed = THRESHOLDS.angularSpeed;
    const quality = frame.quality;
    // A stale sample is not a sample. Everything below reads these, never `frame.motion` and
    // `frame.orientation` directly, so there is exactly one place the staleness rule lives.
    const motion = ages.motionMs <= GATE_STALE_MS ? frame.motion : null;
    const orientation = ages.orientationMs <= ORIENTATION_STALE_MS ? frame.orientation : null;

    // --- angular speed -------------------------------------------------------------
    // A LADDER, and the order matters. `MotionSample.omegaDegPerS` is 0 on a phone with no
    // gyroscope, which used to be graded a confident green at any speed: `too_fast` and
    // `smear` could never fire and the framing "hold steady" gate passed instantly while the
    // phone was being waved. Successive orientation quaternions carry the same information
    // (that is what `quatAngleBetweenDeg` over dt IS), so the second rung is a real
    // measurement rather than a guess — and it is kept warm even while the gyro is being
    // used, so a hand-over mid-capture does not measure one huge step across the gap.
    const derived = orientation !== null ? this.deriveOmega(orientation) : null;
    let rawOmega = 0;
    let omegaMeasured = false;
    if (motion !== null && motion.rotationRate !== null && Number.isFinite(motion.omegaDegPerS)) {
      rawOmega = motion.omegaDegPerS;
      omegaMeasured = true;
    } else if (derived !== null) {
      rawOmega = derived;
      omegaMeasured = true;
    }
    // With nothing to measure, the low-pass is left alone rather than fed a fabricated zero
    // that would then have to decay back out of it once a real rate arrived.
    const omegaDegPerS = omegaMeasured ? this.omegaLp.push(rawOmega, dt) : 0;

    // --- exposure, smear, and the adaptive cap --------------------------------------
    // No quality sample yet means no measured exposure; 1/60 s is the plausible indoor
    // value the contract's own blur arithmetic is quoted at, and it is the pessimistic
    // choice (it under-estimates the safe speed, which costs seconds rather than a capture).
    const exposureS = quality !== null && quality.exposureS > 0 ? quality.exposureS : 1 / 60;
    const speedCapDegPerS = adaptiveSpeedCapDegPerS(exposureS);
    // Recomputed from OUR omega rather than taken from `quality.smearPx`, so the smear meter
    // and the speed meter can never disagree about how fast the phone is moving.
    const smearPx = omegaMeasured ? predictedSmearPx(omegaDegPerS, exposureS) : (quality?.smearPx ?? 0);

    // --- the inertial motion gate ----------------------------------------------------
    const accel = motion !== null && motion.accelSource !== "none" ? motion.linearAccel : null;
    if (accel !== null && motion !== null) {
      this.translation.push({
        t: motion.t,
        accel,
        // Only a FRESH attitude may be used to fix the frame: rotating today's acceleration
        // by a two-second-old quaternion produces a world frame that is confidently wrong,
        // which is worse than the tumbling device frame it replaces.
        attitude: orientation !== null ? { q: orientation.q, screenAngleDeg: orientation.screenAngleDeg } : null,
      });
    }
    // The host's estimator (`MotionApiExt.readTranslation()`) saw every `devicemotion`
    // sample; ours saw whichever ones survived the rAF tick. Same code, so prefer theirs
    // whenever it is fresh, and fall back without a seam when it is not wired at all.
    const own = this.translation.read();
    const supplied = this.supplied;
    const estimate = supplied !== null && supplied.gateRan && ageOf(now, supplied.at) <= GATE_STALE_MS ? supplied : own;
    const gateRan = estimate.gateRan && ageOf(now, estimate.at) <= GATE_STALE_MS;
    const motionGateDegraded = !gateRan;
    if (gateRan) this.gateSamples += 1;

    // Never a number that was not measured: with no fresh evidence these report 0 AND
    // `motionGateDegraded`, and every consumer is required to read the pair together.
    const translationMPerS = gateRan ? estimate.driftMPerS : 0;
    const stepBandRmsMs2 = gateRan ? estimate.stepBandRmsMs2 : 0;

    // --- pivot radius ---------------------------------------------------------------
    // r = |a_radial| / omega^2. The radial (centripetal) direction is the optical axis, and
    // the rear camera looks along the device's z axis, so |a.z| is the component wanted.
    // Gated to omega > 15 deg/s: below that omega^2 is small enough that accelerometer
    // noise (1-3 mg RMS) dominates and the quotient is meaningless, not merely imprecise.
    let pivotRadiusM = this.readout.pivotRadiusM;
    if (accel !== null && omegaMeasured && rawOmega > THRESHOLDS.pivotRadius.gateOmegaDegPerS) {
      const omegaRad = rawOmega * DEG;
      this.radialAccel.push(now, Math.abs(accel.z));
      this.omegaSq.push(now, omegaRad * omegaRad);
      const meanOmegaSq = this.omegaSq.mean();
      pivotRadiusM = this.omegaSq.count >= MIN_PIVOT_SAMPLES && meanOmegaSq > 1e-6 ? this.radialAccel.mean() / meanOmegaSq : null;
    } else if (!omegaMeasured || rawOmega <= THRESHOLDS.pivotRadius.gateOmegaDegPerS) {
      pivotRadiusM = null;
    }

    // --- flow ------------------------------------------------------------------------
    // KNOWN HOLE, and the contract's `DetectorReadout` has no field to express it: an
    // invalid flow sample (too dark or too uniform for phase correlation to mean anything)
    // reports 0, which grades green — so the camera-side walking signal disappears at
    // exactly the moment the frames stop carrying information. The specific case that costs
    // a capture, a lens covered or pointed into the sun, is caught by the `too_dark` and
    // `glare` block escalations below, which suspend stamping outright. What remains
    // uncovered is a low-texture wall with the inertial gate also degraded, where nothing on
    // the phone can see a translation; `motionGateRan: false` is what tells the server so.
    const flowDivergencePx = frame.flow !== null && frame.flow.valid ? frame.flow.divergencePx : 0;

    // --- image -----------------------------------------------------------------------
    const meanLuma = quality !== null ? quality.meanLuma : this.readout.meanLuma;
    const clippedFraction = quality !== null ? quality.clippedFraction : this.readout.clippedFraction;
    let measuredFps = this.readout.measuredFps;
    if (quality !== null && Number.isFinite(quality.measuredFps)) {
      this.fpsWindow.push(now, quality.measuredFps);
      measuredFps = this.fpsWindow.mean();
    }

    // --- attitude --------------------------------------------------------------------
    // Roll and row pitch are DISPLAY quantities — the horizon bar and the pitch coach — not
    // gates, so a stale sample holds its last value rather than snapping the horizon level,
    // and only a genuinely absent one reports 0. Either way `track_lost` is what tells the
    // user the track is gone, and it escalates to `stop` while it is.
    const rollDeg =
      orientation !== null && Number.isFinite(orientation.rollDeg)
        ? orientation.rollDeg
        : frame.orientation !== null
          ? this.readout.rollDeg
          : 0;
    const rowTarget: Degrees = frame.activeRow !== null ? rowTargetPitchDeg(frame.activeRow as RowId, this.config.lens) : 0;
    const rowPitchErrorDeg =
      frame.activeRow === null
        ? 0
        : orientation !== null
          ? orientation.pitchDeg - rowTarget
          : frame.orientation !== null
            ? this.readout.rowPitchErrorDeg
            : 0;

    const walkingGrade = worstGrade(
      worstGrade(
        gradeAscending(translationMPerS, THRESHOLDS.translationDrift.warnMPerS, THRESHOLDS.translationDrift.stopMPerS),
        gradeAscending(stepBandRmsMs2, THRESHOLDS.stepCadence.warnRmsMs2, THRESHOLDS.stepCadence.stopRmsMs2),
      ),
      gradeAscending(flowDivergencePx, THRESHOLDS.flowDivergence.warnPx, this.flowStopPx(motionGateDegraded)),
    );

    const exposureGrade = worstGrade(
      gradeDescending(meanLuma, THRESHOLDS.brightness.warnMeanL, THRESHOLDS.brightness.blockMeanL),
      gradeAscending(clippedFraction, THRESHOLDS.clipping.warnFraction, THRESHOLDS.clipping.blockFraction),
    );

    return {
      omegaDegPerS,
      speedCapDegPerS,
      // AMBER, not green, when there is no rate to grade. A phone with neither a gyroscope
      // nor a usable attitude cannot tell a still hand from a whip-pan, and a green dot
      // beside a 0 deg/s it never measured is the lie that let finding 6 through. Amber is
      // the honest reading of "unknown" in a three-colour vocabulary that has no fourth
      // state, and it also keeps the framing gate from passing on an unmeasured 0.
      speedGrade: omegaMeasured ? gradeAscending(omegaDegPerS, speed.greenMaxDegPerS, speed.redDegPerS) : "amber",
      smearPx,
      smearGrade: omegaMeasured ? gradeAscending(smearPx, THRESHOLDS.smear.greenPx, THRESHOLDS.smear.amberPx) : "amber",
      pivotRadiusM,
      pivotGrade: pivotRadiusM === null ? "green" : gradeAscending(pivotRadiusM, THRESHOLDS.pivotRadius.greenM, THRESHOLDS.pivotRadius.amberM),
      translationMPerS,
      stepBandRmsMs2,
      flowDivergencePx,
      walkingGrade,
      rollDeg,
      rollGrade: gradeAscending(Math.abs(rollDeg), THRESHOLDS.roll.greenDeg, THRESHOLDS.roll.amberDeg),
      rowPitchErrorDeg,
      rowPitchGrade: gradeAscending(Math.abs(rowPitchErrorDeg), THRESHOLDS.rowPitch.greenDeg, THRESHOLDS.rowPitch.amberDeg),
      meanLuma,
      clippedFraction,
      exposureGrade,
      measuredFps,
      motionGateDegraded,
    };
  }

  /** With no inertial gate the camera-side stop threshold tightens from 7 px to 5 px. */
  private flowStopPx(degraded: boolean): number {
    return degraded ? MOTION_GATE_FALLBACK.flowStopPxWhenDegraded : THRESHOLDS.flowDivergence.stopPx;
  }

  /**
   * Angular rate from successive orientation quaternions, deg/s, or null before the second
   * distinct sample. Kept warm on every fresh orientation sample whether or not a gyroscope
   * is present, so `lastQuat` is never a stale anchor.
   *
   * The last value is HELD across a repeated sample rather than reported as zero: the host
   * ticks faster than the sensor, and a rate that alternated between the truth and 0 at the
   * beat frequency would halve the speed meter and flicker every grade attached to it.
   */
  private deriveOmega(sample: OrientationSample): number | null {
    const previous = this.lastQuat;
    const previousT = this.lastQuatT;
    if (previous === null || previousT === null) {
      this.lastQuat = sample.q;
      this.lastQuatT = sample.t;
      return null;
    }
    if (!(sample.t > previousT)) return this.derivedOmegaDegPerS;
    const seconds = (sample.t - previousT) / 1000;
    this.lastQuat = sample.q;
    this.lastQuatT = sample.t;
    if (seconds <= 0) return this.derivedOmegaDegPerS;
    const rate = quatAngleBetweenDeg(previous, sample.q) / seconds;
    this.derivedOmegaDegPerS = Number.isFinite(rate) ? rate : 0;
    return this.derivedOmegaDegPerS;
  }

  private evaluate(frame: SensorFrame, r: DetectorReadout, now: Millis, ages: SampleAges): Warning[] {
    const out: Warning[] = [];
    const haptics = this.config.haptics;
    const speed = THRESHOLDS.angularSpeed;

    // --- landscape and lens change: binary facts, no sustain, no hysteresis -----------
    if (this.external.landscape === true) {
      if (this.landscapeSince === null) this.landscapeSince = now;
      out.push(buildWarning("landscape", this.landscapeSince, 90, 0, haptics));
    }
    if (this.external.lensChanged === true) {
      if (this.lensChangedSince === null) this.lensChangedSince = now;
      out.push(buildWarning("lens_changed", this.lensChangedSince, 1, 0, haptics));
    }

    // --- walking: three independent signals, any one of which latches a `stop` --------
    const stopPx = this.flowStopPx(r.motionGateDegraded);
    const drift = THRESHOLDS.translationDrift;
    const cadence = THRESHOLDS.stepCadence;
    const driftActive = this.latches.drift.update(
      now,
      r.translationMPerS > drift.stopMPerS,
      r.translationMPerS < drift.stopMPerS * HYSTERESIS,
      drift.stopSustainMs,
    );
    const cadenceActive = this.latches.cadence.update(
      now,
      r.stepBandRmsMs2 > cadence.stopRmsMs2,
      r.stepBandRmsMs2 < cadence.stopRmsMs2 * HYSTERESIS,
      drift.stopSustainMs,
    );
    const flowActive = this.latches.flow.update(
      now,
      r.flowDivergencePx > stopPx,
      r.flowDivergencePx < stopPx * HYSTERESIS,
      THRESHOLDS.flowDivergence.sustainMs,
    );
    if (driftActive || cadenceActive || flowActive) {
      // Report whichever signal is furthest past its own threshold, so the debug overlay
      // names the evidence rather than an arbitrary one of three.
      const candidates: { since: Millis; value: number; threshold: number; excess: number }[] = [];
      if (driftActive && this.latches.drift.since !== null) {
        candidates.push({ since: this.latches.drift.since, value: r.translationMPerS, threshold: drift.stopMPerS, excess: r.translationMPerS / drift.stopMPerS });
      }
      if (cadenceActive && this.latches.cadence.since !== null) {
        candidates.push({ since: this.latches.cadence.since, value: r.stepBandRmsMs2, threshold: cadence.stopRmsMs2, excess: r.stepBandRmsMs2 / cadence.stopRmsMs2 });
      }
      if (flowActive && this.latches.flow.since !== null) {
        candidates.push({ since: this.latches.flow.since, value: r.flowDivergencePx, threshold: stopPx, excess: r.flowDivergencePx / stopPx });
      }
      candidates.sort((a, b) => b.excess - a.excess);
      const top = candidates[0];
      if (top !== undefined) out.push(buildWarning("walking", top.since, top.value, top.threshold, haptics));
    }

    // --- too fast: the fixed extraction wall -----------------------------------------
    if (
      this.latches.too_fast.update(
        now,
        r.omegaDegPerS > speed.redDegPerS,
        r.omegaDegPerS < speed.redDegPerS * HYSTERESIS,
        speed.redSustainMs,
      ) &&
      this.latches.too_fast.since !== null
    ) {
      out.push(buildWarning("too_fast", this.latches.too_fast.since, r.omegaDegPerS, speed.redDegPerS, haptics));
    }

    // --- smear: the adaptive wall, i.e. "this room is dim, halve your speed" ----------
    if (
      this.latches.smear.update(
        now,
        r.smearPx > THRESHOLDS.smear.amberPx,
        r.smearPx < THRESHOLDS.smear.amberPx * HYSTERESIS,
        speed.redSustainMs,
      ) &&
      this.latches.smear.since !== null
    ) {
      out.push(buildWarning("smear", this.latches.smear.since, r.smearPx, THRESHOLDS.smear.amberPx, haptics));
    }

    // --- pivot radius: a quality slope, so it warns and never blocks ------------------
    const pivot = r.pivotRadiusM;
    if (
      this.latches.pivot_radius.update(
        now,
        pivot !== null && pivot > THRESHOLDS.pivotRadius.amberM,
        pivot === null || pivot < THRESHOLDS.pivotRadius.amberM * HYSTERESIS,
        THRESHOLDS.pivotRadius.windowMs,
      ) &&
      this.latches.pivot_radius.since !== null
    ) {
      out.push(buildWarning("pivot_radius", this.latches.pivot_radius.since, pivot ?? 0, THRESHOLDS.pivotRadius.amberM, haptics));
    }

    // --- low fps ---------------------------------------------------------------------
    if (
      this.latches.low_fps.update(
        now,
        r.measuredFps > 0 && r.measuredFps < THRESHOLDS.frameRate.minFps,
        r.measuredFps === 0 || r.measuredFps > THRESHOLDS.frameRate.minFps / HYSTERESIS,
        THRESHOLDS.frameRate.windowMs,
      ) &&
      this.latches.low_fps.since !== null
    ) {
      out.push(buildWarning("low_fps", this.latches.low_fps.since, r.measuredFps, THRESHOLDS.frameRate.minFps, haptics));
    }

    // --- glare: 0.05 is exactly where extract._clipping_penalty starts docking quality --
    // Two latches on one code. 0.05 docks a frame's quality score; 0.15 is
    // `clipping.blockFraction`, past which the frame is a white rectangle carrying no
    // features to match — so it escalates to `stop` and coverage stops accumulating from it.
    const glareWarn = this.latches.glare.update(
      now,
      r.clippedFraction > THRESHOLDS.clipping.warnFraction,
      r.clippedFraction < THRESHOLDS.clipping.warnFraction * HYSTERESIS,
      THRESHOLDS.clipping.sustainMs,
    );
    const glareBlock = this.latches.glare_block.update(
      now,
      r.clippedFraction > THRESHOLDS.clipping.blockFraction,
      r.clippedFraction < THRESHOLDS.clipping.blockFraction * HYSTERESIS,
      THRESHOLDS.clipping.sustainMs,
    );
    const glareSince = this.latches.glare.since ?? this.latches.glare_block.since;
    if ((glareWarn || glareBlock) && glareSince !== null) {
      out.push(
        buildWarning(
          "glare",
          glareSince,
          r.clippedFraction,
          glareBlock ? THRESHOLDS.clipping.blockFraction : THRESHOLDS.clipping.warnFraction,
          haptics,
          glareBlock ? "stop" : undefined,
        ),
      );
    }

    // --- too dark --------------------------------------------------------------------
    // Same two-level shape. 80 L* is "this room is dim"; 40 L* is `brightness.blockMeanL`,
    // 1.6x the engine's own MIN_MEAN_BRIGHTNESS cliff, and a finger over the lens reads
    // about 5. A capture cannot be allowed to bank stations and paint the dome from frames
    // that carry no picture — the whole-capture mean luma stays around 120 through ten such
    // seconds, so nothing downstream would ever notice.
    const darkWarn = this.latches.too_dark.update(
      now,
      r.meanLuma > 0 && r.meanLuma < THRESHOLDS.brightness.warnMeanL,
      r.meanLuma === 0 || r.meanLuma > THRESHOLDS.brightness.warnMeanL / HYSTERESIS,
      DARK_SUSTAIN_MS,
    );
    const darkBlock = this.latches.too_dark_block.update(
      now,
      r.meanLuma > 0 && r.meanLuma < THRESHOLDS.brightness.blockMeanL,
      r.meanLuma === 0 || r.meanLuma > THRESHOLDS.brightness.blockMeanL / HYSTERESIS,
      DARK_SUSTAIN_MS,
    );
    const darkSince = this.latches.too_dark.since ?? this.latches.too_dark_block.since;
    if ((darkWarn || darkBlock) && darkSince !== null) {
      out.push(
        buildWarning(
          "too_dark",
          darkSince,
          r.meanLuma,
          darkBlock ? THRESHOLDS.brightness.blockMeanL : THRESHOLDS.brightness.warnMeanL,
          haptics,
          darkBlock ? "stop" : undefined,
        ),
      );
    }

    // --- track lost ------------------------------------------------------------------
    // Two different failures under one code, and only one of them is a stop.
    //
    // NO FRESH ATTITUDE (absent, or frozen for longer than ORIENTATION_STALE_MS) means there
    // is no direction to attribute a frame to. Stamping the dome and dwelling on stations
    // from a dead compass invents coverage in directions the camera never pointed, and
    // over-reporting coverage is the one direction `coverage.ts` says the model must never
    // fail in. So: `stop`.
    //
    // A POOR EARTH-REFERENCED HEADING is only a problem when the heading is what yaw is
    // built from. Since `useOrientation` now drives yaw from the relative gyro track, this
    // only applies to a sample that says `reference: "absolute"`, and it stays a `warn`.
    const orientationMissing = frame.orientation === null || ages.orientationMs > ORIENTATION_STALE_MS;
    const accuracy = frame.orientation?.headingAccuracyDeg ?? null;
    const headingBad =
      !orientationMissing &&
      frame.orientation !== null &&
      frame.orientation.reference === "absolute" &&
      accuracy !== null &&
      (accuracy < 0 || accuracy > MAX_HEADING_ACCURACY_DEG);
    const trackBad = orientationMissing || headingBad;
    if (
      this.latches.track_lost.update(now, trackBad, !trackBad, TRACK_LOST_SUSTAIN_MS) &&
      this.latches.track_lost.since !== null
    ) {
      out.push(
        buildWarning(
          "track_lost",
          this.latches.track_lost.since,
          accuracy ?? -1,
          MAX_HEADING_ACCURACY_DEG,
          haptics,
          orientationMissing ? "stop" : undefined,
        ),
      );
    }

    // --- row pitch: only meaningful while a row is being coached ----------------------
    const rowPitchError = Math.abs(r.rowPitchErrorDeg);
    if (
      this.latches.row_pitch.update(
        now,
        frame.activeRow !== null && rowPitchError > THRESHOLDS.rowPitch.amberDeg,
        frame.activeRow === null || rowPitchError < THRESHOLDS.rowPitch.amberDeg * HYSTERESIS,
        THRESHOLDS.roll.redSustainMs,
      ) &&
      this.latches.row_pitch.since !== null
    ) {
      out.push(buildWarning("row_pitch", this.latches.row_pitch.since, r.rowPitchErrorDeg, THRESHOLDS.rowPitch.amberDeg, haptics));
    }

    // --- roll ------------------------------------------------------------------------
    const roll = Math.abs(r.rollDeg);
    if (
      this.latches.roll.update(
        now,
        roll > THRESHOLDS.roll.amberDeg,
        roll < THRESHOLDS.roll.amberDeg * HYSTERESIS,
        THRESHOLDS.roll.redSustainMs,
      ) &&
      this.latches.roll.since !== null
    ) {
      out.push(buildWarning("roll", this.latches.roll.since, r.rollDeg, THRESHOLDS.roll.amberDeg, haptics));
    }

    return rankWarnings(out);
  }

  private accumulate(frame: SensorFrame, r: DetectorReadout): void {
    this.recordedTicks += 1;
    this.omegaSamples.push(r.omegaDegPerS);
    if (r.omegaDegPerS > THRESHOLDS.angularSpeed.redDegPerS) this.overSpeedTicks += 1;
    if (this.activeWarnings.some((w) => w.code === "walking")) this.walkingTicks += 1;
    if (r.pivotRadiusM !== null) this.pivotSamples.push(r.pivotRadiusM);
    if (!r.motionGateDegraded) this.driftSamples.push(r.translationMPerS);
    if (frame.quality !== null) {
      this.lumaSum += frame.quality.meanLuma;
      this.lumaCount += 1;
      if (frame.quality.meanLuma < this.lumaMin) this.lumaMin = frame.quality.meanLuma;
      this.clipSum += frame.quality.clippedFraction;
      if (frame.quality.clippedFraction > this.clipMax) this.clipMax = frame.quality.clippedFraction;
      this.smearFrames += 1;
      if (r.smearPx > THRESHOLDS.smear.amberPx) this.overSmearFrames += 1;
    }
  }
}

export function create(config: CaptureConfig): CaptureDetector {
  return new Detector(config);
}

/** The `DetectorsApi` surface, so a mis-typed export is a compile error here, not in P. */
export const detectorsApi = { create } satisfies DetectorsApi;
