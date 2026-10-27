"use client";

/**
 * MOTION — the translation gate, which is the single most valuable detector in the feature.
 * =====================================================================================
 *
 * Measured on real captures: rotational-purity residual is 0.98 px on a nodal tripod,
 * 2.3-2.4 px hand-held and careful, 3.6 px on the bedroom turn whose ceiling beams visibly
 * bend, and 21.2 px walking, which the pipeline refuses outright. Nothing else the phone can
 * measure separates a good capture from a ruined one so cleanly.
 *
 * WHAT THIS MODULE DOES NOT CLAIM
 * -------------------------------------------------------------------------------------
 * It does not measure metres. Double-integrating a consumer accelerometer diverges within
 * about a second: bias of a few mg integrates to tens of centimetres of phantom
 * displacement per second, and no amount of filtering fixes that. So there is no odometry
 * here. There are two WINDOWED PATTERN STATISTICS, both compared against a threshold and
 * then thrown away, and BOTH ARE COMPUTED BY `detectors.createTranslationEstimator` — the
 * same code the live walking gate runs, so the number this hook publishes and the number
 * that stops a capture can never mean two different things. See that function for the
 * derivation of each; the short version:
 *
 *   1. `driftMPerS` — windowed peak of the world-frame speed from a SINGLE leaky
 *      integration of high-passed linear acceleration. Thresholds (warn 0.15, stop 0.30 m/s)
 *      come from validate.py's own ladder: a 0.35 m arm at 24 deg/s traces 0.146 m/s of
 *      chord and measures 2.34 px, which is accepted; walking at 0.80 m/s measures 13.70 px,
 *      which is refused.
 *      WHERE IT FAILS, and this is physics rather than implementation: a translation already
 *      under way at CONSTANT velocity produces no acceleration at all and is inertially
 *      invisible. Only the transient that starts one can be seen. A phone in a car or a lift
 *      reads as walking for a few seconds and then stops doing so.
 *
 *   2. `stepBandRmsMs2` — RMS of the 1.0-3.0 Hz band-passed acceleration VECTOR over a 2 s
 *      window. Human walking cadence is 1.5-2.5 Hz with roughly 1-3 m/s^2 peak-to-peak, i.e.
 *      0.35-1.06 m/s^2 RMS; turning on the spot puts almost no energy in that band whatever
 *      the rotation rate. Independent of the integration and therefore of its drift, which
 *      is exactly why the contract asks for both.
 *      WHERE IT FAILS: it cannot see a smooth glide (no footfalls at all), and it will fire
 *      on someone tapping the phone or shivering. It is a walking detector, not a
 *      translation detector.
 *
 * This module used to band-pass `|linearAccel|` — the MAGNITUDE — through one biquad. That
 * is the one arrangement guaranteed not to work: rectification puts a 2 Hz gait's
 * fundamental at 4 Hz, outside the band being measured. Measured on the realistic-gait
 * fixture it read 0.18 m/s^2 against a 0.70 stop line where the correct per-axis filtering
 * reads 1.09. It has never been called by anything, which is the only reason it never cost a
 * capture.
 *
 * FAIL SAFE
 * -------------------------------------------------------------------------------------
 * MDN is explicit that `DeviceMotionEvent.acceleration` is simply ABSENT when the hardware
 * cannot remove gravity. Absence of evidence of walking must never read as evidence of
 * standing still, so `gateAvailable` goes false, the readout is flagged degraded, and the
 * camera-side flow detector takes over with a tightened threshold. Nothing here ever
 * silently reports 0 m/s because it had nothing to measure — and `staleMs()` exists so a
 * consumer can tell a live stream from a dead one, because `read()` returns the last sample
 * forever and a frozen sample looks exactly like a user standing perfectly still.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { AccelSource, Millis, MotionApi, MotionSample, OrientationSample, Vec3 } from "./types";
import { MOTION_GATE_FALLBACK } from "./types";
import { createTranslationEstimator, type TranslationEstimate } from "./detectors";
import { deviceFrameQuat, deviceGravityDirection, nowMs, quatAngleDeg } from "./useOrientation";

export type { TranslationEstimate } from "./detectors";

// -------------------------------------------------------------------------------------
// Small DSP pieces. All rate-aware: DeviceMotionEvent runs at ~60 Hz on iOS and anywhere
// between 20 and 60 Hz on Android, so a filter written as a per-sample coefficient would
// have a different corner frequency on every phone.
// -------------------------------------------------------------------------------------

/** One-pole low-pass with a corner in Hz, evaluated at the actual dt of each sample. */
function lowPass(previous: number, value: number, cornerHz: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cornerHz);
  const a = 1 - Math.exp(-dt / tau);
  return previous + (value - previous) * a;
}


// -------------------------------------------------------------------------------------
// The tracker
// -------------------------------------------------------------------------------------

/** One raw reading, in the shape both platforms reduce to. */
export interface MotionReading {
  t?: Millis;
  /** Degrees per second, in DeviceMotionEvent's own alpha/beta/gamma naming. */
  rotationRate?: { alpha: number | null; beta: number | null; gamma: number | null } | null;
  /** m/s^2 with gravity already removed by the platform. Absent on gyro-less hardware. */
  acceleration?: { x: number | null; y: number | null; z: number | null } | null;
  accelerationIncludingGravity?: { x: number | null; y: number | null; z: number | null } | null;
  /** `event.interval`, ms. */
  interval?: number;
}

export interface MotionTrackerOptions {
  /**
   * Corner frequency of the gravity estimator, Hz. Only used on the `derived` path.
   *
   * The contract's MOTION_GATE_FALLBACK.gravityLowPassAlpha = 0.8 is the coefficient from
   * Android's own low-pass example, whose nominal delivery rate is a few Hz — there it is a
   * ~0.2 Hz corner. Applied literally per-sample at the 60 Hz DeviceMotionEvent rate the
   * same 0.8 becomes a 2.1 Hz corner, which would put the gravity tracker straight through
   * the 1-3 Hz walking band this module exists to detect and quietly subtract the evidence
   * away. So the constant is honoured as a FREQUENCY, not as a per-sample coefficient, and
   * 0.3 Hz is chosen to match THRESHOLDS.translationDrift.highPassHz — the same signal, the
   * same idea, one corner.
   */
  gravityCornerHz?: number;
  /**
   * Live orientation, when available. Two uses, both worth having:
   *   - the linear acceleration is rotated into the world frame before integration, so a
   *     walk that curves does not partially cancel itself in a tumbling device frame;
   *   - the platform's accelerometer sign convention is resolved (iOS reports
   *     accelerationIncludingGravity inverted relative to Android), so the reported gravity
   *     vector always points DOWN.
   *
   * READ FRESH ON EVERY SAMPLE, so pass an accessor that reads a ref rather than a value
   * captured in a closure. `useMotion` wraps whatever it is given in exactly such an
   * indirection, so a caller that hands the hook a new arrow function on every render still
   * gets live orientation instead of the one that existed at mount.
   */
  orientation?: () => OrientationSample | null;
  /**
   * On a device with no gyroscope, derive the angular rate from successive orientation
   * quaternions. `rotationRate` stays null — that is a fact about the hardware, and
   * `hasGyroscope` keeps reporting it — but `omegaDegPerS` becomes a real measurement instead
   * of a constant 0, which would grade every whip-pan green and pass the framing "hold
   * steady" gate instantly while the phone was being waved. Default true; set false for a
   * strict reading of the sample contract.
   */
  deriveOmegaFromOrientation?: boolean;
  now?: () => Millis;
}

/** Where `MotionSample.omegaDegPerS` came from, and therefore how much to trust it. */
export type OmegaSource =
  /** `DeviceMotionEvent.rotationRate`. */
  | "gyroscope"
  /** Reconstructed from successive orientation quaternions on gyro-less hardware. */
  | "orientation"
  /** Neither. `omegaDegPerS` is 0 because nothing measured it, NOT because nothing moved. */
  | "none";

export interface MotionTracker {
  push: (reading: MotionReading) => MotionSample | null;
  /**
   * Late-bind the orientation accessor, replacing whatever `MotionTrackerOptions.orientation`
   * supplied. The tracker is built once and lives for the whole session, so a hook that
   * captured the caller's accessor at construction would keep the first one forever — which
   * silently works when it closes over a ref and silently does not when it closes over
   * state. `useMotion` calls this from an effect on every render, so the hook's `orientation`
   * option behaves like any other prop.
   */
  setOrientation: (source: (() => OrientationSample | null) | null) => void;
  read: () => MotionSample | null;
  readTranslation: () => TranslationEstimate;
  readAccelSource: () => AccelSource;
  /** How `omegaDegPerS` was obtained. `"none"` means it is a placeholder, not a measurement. */
  readOmegaSource: () => OmegaSource;
  /** True once the platform's accelerometer sign has been resolved against orientation. */
  gravitySignResolved: () => boolean;
  staleMs: (at?: Millis) => number;
  reset: () => void;
}

export function createMotionTracker(options: MotionTrackerOptions = {}): MotionTracker {
  const gravityCornerHz = options.gravityCornerHz ?? 0.3;
  const clock = options.now ?? nowMs;
  let orientationOf = options.orientation ?? null;
  const deriveOmega = options.deriveOmegaFromOrientation ?? true;

  // One shared implementation with the live walking gate. See detectors.ts.
  const translation = createTranslationEstimator();

  let gravity: Vec3 | null = null;
  let lastT: Millis | null = null;
  let accelSource: AccelSource = "none";
  let omegaSource: OmegaSource = "none";
  let sample: MotionSample | null = null;

  /** The platform's accelerometer sign, resolved once against orientation and then kept. */
  let signResolved = false;
  let signFlip = 1;

  // Orientation-derived angular rate, for gyro-less hardware.
  let lastQuatT: Millis | null = null;
  let lastQuat: OrientationSample["q"] | null = null;
  let derivedOmegaDegPerS = 0;

  function finite(v: { x: number | null; y: number | null; z: number | null } | null | undefined): Vec3 | null {
    if (!v) return null;
    const { x, y, z } = v;
    if (x === null || y === null || z === null) return null;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
    return { x, y, z };
  }

  function push(reading: MotionReading): MotionSample | null {
    const t = reading.t ?? clock();
    const dtRaw = lastT === null ? 0 : (t - lastT) / 1000;
    // A backgrounded tab returns with a multi-second gap; integrating across it would inject
    // metres of phantom displacement. Clamp, and let the window flush naturally.
    const dt = Math.min(0.2, Math.max(1 / 200, dtRaw || 1 / 60));
    lastT = t;

    const rr = reading.rotationRate;
    // DeviceMotionEvent names its axes alpha/beta/gamma; the contract's Vec3 is x = beta,
    // y = gamma, z = alpha.
    const rotationRate: Vec3 | null =
      rr && rr.alpha !== null && rr.beta !== null && rr.gamma !== null && Number.isFinite(rr.alpha) && Number.isFinite(rr.beta) && Number.isFinite(rr.gamma)
        ? { x: rr.beta, y: rr.gamma, z: rr.alpha }
        : null;

    const orientation = orientationOf?.() ?? null;

    // The quaternion track is kept warm whether or not a gyroscope exists, so `lastQuat` is
    // never a stale anchor and the derived rate is available the instant it is needed.
    if (orientation !== null) {
      if (lastQuat !== null && lastQuatT !== null && orientation.t > lastQuatT) {
        const seconds = (orientation.t - lastQuatT) / 1000;
        const rate = quatAngleDeg(lastQuat, orientation.q) / seconds;
        // HOLD the last value across a repeated orientation sample rather than reporting 0:
        // `devicemotion` and `deviceorientation` are independent streams, so at 60 Hz each
        // roughly half the motion events see no new quaternion, and a rate that alternated
        // between the truth and zero would halve the speed meter.
        if (Number.isFinite(rate)) derivedOmegaDegPerS = rate;
      }
      if (lastQuatT === null || orientation.t > lastQuatT) {
        lastQuat = orientation.q;
        lastQuatT = orientation.t;
      }
    }

    let omegaDegPerS = 0;
    if (rotationRate) {
      omegaDegPerS = Math.hypot(rotationRate.x, rotationRate.y, rotationRate.z);
      omegaSource = "gyroscope";
    } else if (deriveOmega && lastQuat !== null && lastQuatT !== null) {
      omegaDegPerS = derivedOmegaDegPerS;
      omegaSource = "orientation";
    } else {
      omegaSource = "none";
    }

    const native = finite(reading.acceleration);
    const withGravity = finite(reading.accelerationIncludingGravity);

    let linearAccel: Vec3 | null = null;
    if (native && (native.x !== 0 || native.y !== 0 || native.z !== 0 || withGravity === null)) {
      accelSource = "native";
      linearAccel = native;
      if (withGravity) {
        // Even on the native path the gravity vector is worth having (roll, absolute pitch),
        // and it comes for free as the difference.
        const g = { x: withGravity.x - native.x, y: withGravity.y - native.y, z: withGravity.z - native.z };
        gravity = gravity === null ? g : {
          x: lowPass(gravity.x, g.x, gravityCornerHz, dt),
          y: lowPass(gravity.y, g.y, gravityCornerHz, dt),
          z: lowPass(gravity.z, g.z, gravityCornerHz, dt),
        };
      }
    } else if (withGravity) {
      accelSource = "derived";
      gravity = gravity === null ? withGravity : {
        x: lowPass(gravity.x, withGravity.x, gravityCornerHz, dt),
        y: lowPass(gravity.y, withGravity.y, gravityCornerHz, dt),
        z: lowPass(gravity.z, withGravity.z, gravityCornerHz, dt),
      };
      linearAccel = { x: withGravity.x - gravity.x, y: withGravity.y - gravity.y, z: withGravity.z - gravity.z };
    } else {
      accelSource = "none";
      linearAccel = null;
    }

    // Resolve the platform's accelerometer sign ONCE, against orientation. iOS reports
    // accelerationIncludingGravity inverted relative to Android/the spec, and there is no
    // way to tell from the numbers alone which convention a phone uses. With orientation we
    // know where down is, so the ambiguity is a single dot product. Sticky, because it is a
    // property of the platform, not of the moment.
    if (!signResolved && gravity && orientation) {
      const expectedDown = deviceGravityDirection(deviceFrameQuat(orientation.q, orientation.screenAngleDeg));
      const dot = gravity.x * expectedDown.x + gravity.y * expectedDown.y + gravity.z * expectedDown.z;
      if (Math.abs(dot) > 4.9) {
        signFlip = dot >= 0 ? 1 : -1;
        signResolved = true;
      }
    }
    const reportedGravity = gravity ? { x: gravity.x * signFlip, y: gravity.y * signFlip, z: gravity.z * signFlip } : null;

    // All of the DSP lives in `detectors.createTranslationEstimator`, including the choice of
    // frame: this hook's only job is to hand it the freshest attitude it has.
    if (linearAccel) {
      translation.push({
        t,
        accel: linearAccel,
        attitude: orientation ? { q: orientation.q, screenAngleDeg: orientation.screenAngleDeg } : null,
      });
    }

    sample = {
      t,
      rotationRate,
      omegaDegPerS,
      linearAccel,
      accelMagnitude: linearAccel ? Math.hypot(linearAccel.x, linearAccel.y, linearAccel.z) : 0,
      gravity: reportedGravity,
      accelSource,
      intervalMs: typeof reading.interval === "number" && Number.isFinite(reading.interval) ? reading.interval : dt * 1000,
    };
    return sample;
  }

  function readTranslation(): TranslationEstimate {
    const estimate = translation.read();
    // `accelSource` can go back to "none" mid-session (a stream that stops delivering
    // `acceleration`), and past that point the windows hold numbers nothing is measuring.
    return accelSource === "none" ? { ...estimate, gateRan: false } : estimate;
  }

  return {
    push,
    setOrientation: (source) => {
      orientationOf = source;
    },
    read: () => sample,
    readTranslation,
    readAccelSource: () => accelSource,
    readOmegaSource: () => omegaSource,
    gravitySignResolved: () => signResolved,
    staleMs: (at?: Millis) => (sample === null ? Number.POSITIVE_INFINITY : (at ?? clock()) - sample.t),
    reset: () => {
      translation.reset();
      gravity = null;
      lastT = null;
      accelSource = "none";
      omegaSource = "none";
      sample = null;
      lastQuat = null;
      lastQuatT = null;
      derivedOmegaDegPerS = 0;
      // The sign convention is a property of the device, so it deliberately survives a reset.
    },
  };
}

// -------------------------------------------------------------------------------------
// The hook
// -------------------------------------------------------------------------------------

type MotionEventCtor = typeof DeviceMotionEvent & { requestPermission?: () => Promise<string> };

export type MotionPermissionOutcome = "granted" | "denied" | "not-required";

export interface UseMotionOptions extends MotionTrackerOptions {
  /** Injected sample source; receives a push and returns an unsubscribe. Replaces `window`. */
  source?: (push: (reading: MotionReading) => void) => () => void;
  autoStart?: boolean;
  /**
   * How often the React `sample`, `accelSource` and `omegaSource` fields refresh. `read()`
   * is always current and never involves React. Zero means never — see the same option on
   * `useOrientation` for why a consumer with its own loop wants that. Default 4.
   */
  stateHz?: number;
}

export interface MotionApiExt extends MotionApi {
  /**
   * The two windowed statistics, computed at the NATIVE sensor rate. This is why they exist
   * here as well as inside the detector: the step band tops out at 3 Hz and the detector is
   * pushed on the rAF tick, which is an independent clock — every motion sample that arrives
   * between two frames is simply never seen. Hand this to
   * `CaptureDetector.setTranslation()` once per tick and the walking gate grades a stream
   * nothing was dropped from.
   */
  readTranslation: () => TranslationEstimate;
  /**
   * Milliseconds since the last sample; Infinity before the first.
   *
   * NOT decoration. `read()` returns the last sample forever, so a `devicemotion` stream
   * that dies presents as a perfectly still user until somebody checks this.
   */
  staleMs: () => number;
  permission: "unknown" | MotionPermissionOutcome;
  /** True when a gyroscope is present (`rotationRate` was delivered). */
  hasGyroscope: boolean;
  /** How `sample.omegaDegPerS` was obtained. `"none"` means it is not a measurement. */
  omegaSource: OmegaSource;
  start: () => void;
}

export function useMotion(options: UseMotionOptions = {}): MotionApiExt {
  const { source, autoStart = true, stateHz = 4, gravityCornerHz, orientation, deriveOmegaFromOrientation, now } = options;

  // Late-bound values live in refs SYNCED IN AN EFFECT, never written during render. The
  // listener effect must not re-run on a fresh closure: restarting it would flush the 2 s
  // step-cadence window and the 1 s drift window, which is exactly the evidence the walking
  // gate needs.
  const sourceRef = useRef(source);
  const stateHzRef = useRef(stateHz);
  const clockRef = useRef(now ?? nowMs);
  const lastStateAtRef = useRef(0);

  // Held in state, not a ref: a ref cannot be lazily initialised during render.
  const [tracker] = useState(() => createMotionTracker({ gravityCornerHz, deriveOmegaFromOrientation, now }));

  useEffect(() => {
    sourceRef.current = source;
    stateHzRef.current = stateHz;
    clockRef.current = now ?? nowMs;
    // The tracker is built ONCE and lives for the session, so the orientation accessor is
    // late-bound here rather than captured in the initialiser above. A caller writing the
    // natural `useMotion({ orientation: () => orientationRef.current.read() })` hands over a
    // fresh arrow on every render; without this it would be the very first one forever,
    // which happens to work when it closes over a ref and silently does not otherwise.
    tracker.setOrientation(orientation ?? null);
  });

  const [sample, setSample] = useState<MotionSample | null>(null);
  const [accelSource, setAccelSource] = useState<AccelSource>("none");
  const [hasGyroscope, setHasGyroscope] = useState(false);
  const [omegaSource, setOmegaSource] = useState<OmegaSource>("none");
  const [permission, setPermission] = useState<"unknown" | MotionPermissionOutcome>("unknown");
  const [attached, setAttached] = useState(autoStart);
  /** Mirrors `hasGyroscope` so the latch can be read without a render-scoped dependency. */
  const hasGyroscopeRef = useRef(false);

  const publish = useCallback(
    (next: MotionSample | null) => {
      if (!next) return;
      // Latched: this only ever goes false->true, so setting it on every one of ~60 events a
      // second is 59 wasted scheduler trips per second. Same reason as `live` in
      // `useOrientation`.
      if (next.rotationRate && !hasGyroscopeRef.current) {
        hasGyroscopeRef.current = true;
        setHasGyroscope(true);
      }
      const hz = stateHzRef.current;
      const interval = hz > 0 ? 1000 / hz : Number.POSITIVE_INFINITY;
      if (next.t - lastStateAtRef.current >= interval) {
        lastStateAtRef.current = next.t;
        setSample(next);
        setAccelSource(next.accelSource);
        setOmegaSource(tracker.readOmegaSource());
      }
    },
    [tracker],
  );

  useEffect(() => {
    if (!attached || typeof window === "undefined") return;

    const inject = sourceRef.current;
    const clock = clockRef.current;
    let disposed = false;

    const feed = (reading: MotionReading) => {
      if (disposed) return;
      publish(tracker.push(reading));
    };

    if (inject) {
      const detach = inject(feed);
      return () => {
        disposed = true;
        detach();
      };
    }

    if (!("DeviceMotionEvent" in window)) return;

    const onMotion = (event: DeviceMotionEvent) => {
      feed({
        t: clock(),
        rotationRate: event.rotationRate,
        acceleration: event.acceleration,
        accelerationIncludingGravity: event.accelerationIncludingGravity,
        interval: event.interval,
      });
    };

    window.addEventListener("devicemotion", onMotion);
    return () => {
      disposed = true;
      window.removeEventListener("devicemotion", onMotion);
    };
  }, [attached, publish, tracker]);

  const request = useCallback(async (): Promise<MotionPermissionOutcome> => {
    if (typeof window === "undefined" || !("DeviceMotionEvent" in window)) {
      setPermission("not-required");
      return "not-required";
    }
    const ctor = window.DeviceMotionEvent as MotionEventCtor;
    if (typeof ctor.requestPermission !== "function") {
      setPermission("not-required");
      setAttached(true);
      return "not-required";
    }
    try {
      // Same rule as orientation: this call must be the first thing after the click, because
      // iOS spends the transient activation on the first await.
      const pending = ctor.requestPermission();
      const result = await pending;
      const outcome: MotionPermissionOutcome = result === "granted" ? "granted" : "denied";
      setPermission(outcome);
      if (outcome === "granted") setAttached(true);
      return outcome;
    } catch {
      setPermission("denied");
      return "denied";
    }
  }, []);

  const stop = useCallback(() => setAttached(false), []);
  const start = useCallback(() => setAttached(true), []);
  const read = useCallback(() => tracker.read(), [tracker]);
  const readTranslation = useCallback(() => tracker.readTranslation(), [tracker]);
  const staleMs = useCallback(() => tracker.staleMs(clockRef.current()), [tracker]);

  return {
    sample,
    read,
    accelSource,
    // FAIL SAFE: false the moment there is nothing to measure with. The detectors must read
    // this as "the inertial gate did not run", never as "the user is standing still".
    gateAvailable: accelSource !== "none",
    request,
    stop,
    readTranslation,
    staleMs,
    permission,
    hasGyroscope,
    omegaSource,
    start,
  };
}

/** Re-exported so callers can see which fallback constant shaped the derived path. */
export const GRAVITY_FALLBACK_ALPHA = MOTION_GATE_FALLBACK.gravityLowPassAlpha;
