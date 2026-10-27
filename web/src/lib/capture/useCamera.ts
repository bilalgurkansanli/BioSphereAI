"use client";

/**
 * CAMERA — open the right lens, prove it, and report what actually happened.
 * =====================================================================================
 *
 * Three failures this module exists to prevent, all of them silent:
 *
 *   1. THE WRONG LENS. On a multi-lens iPhone `facingMode:'environment'` resolves to a
 *      VIRTUAL device the OS switches between wide, ultra-wide and telephoto at will. A
 *      mid-capture switch from 26mm to 13mm changes the field of view the solver assumes and
 *      does not look like a failure — it looks like a subtly wrong panorama, and
 *      rotation.py's [30, 110] hfov clamp is far too loose to catch it. The two-phase open
 *      below picks a physical rear wide by exact deviceId, and then VERIFIES the choice by
 *      reading the deviceId back out of `getSettings()`; a mismatch is recorded as
 *      `virtualLensOnly` rather than assumed away.
 *   2. SILENT DOWN-RESOLUTION. `ideal` constraints never throw; they hand back the nearest
 *      supported mode. 1080 on the short edge is the minimum that fills an 8K equirect
 *      master without upsampling (1080/43 = 25.1 px/deg against the master's 22.8), so
 *      anything less is refused rather than recorded.
 *   3. LYING ABOUT LOCKS. Every control is feature-detected from `getCapabilities()`, applied
 *      in its own try/catch, and then read back from `getSettings()`. `appliedControls`
 *      reports what stuck, never what was asked for. iOS has no exposure control at all —
 *      exposureMode, exposureTime and exposureCompensation are simply absent — so the blown
 *      first keyframe measured on the real capture cannot be fixed by any API call there,
 *      only by the framing gate that starts the user away from the window.
 *
 * The two image kernels (`analyseFrameLuma`, `quadrantFlow`) are exported as pure functions
 * over pixel buffers so they can be tested in plain Node with a synthetic frame, and the
 * whole media layer can be driven by an injected `mediaDevices`.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import type {
  CameraApi,
  CameraControlPlan,
  CameraError,
  FlowSample,
  FrameQualitySample,
  GuidedQualityMode,
  LensGeometry,
  Millis,
} from "./types";
import { CAMERA_CONTROLS, DEFAULT_LENS, LENS_SELECTION, RECORDING, THRESHOLDS, SHUTTER_CAP, type ShutterCapRecord } from "./types";
import { createFrameAnalyser, type FrameAnalyser } from "./frameAnalysis";
import type { LoadPlan } from "./governor";
import { nowMs } from "./useOrientation";

const DEG = Math.PI / 180;

/** Coverage stamps at 10 Hz; the image statistics ride the same clock. */
const COVERAGE_STAMP_HZ = 10;

/** Long edge of a review-strip still. The strip renders them at 54x96 CSS px. */
const THUMBNAIL_LONG_EDGE_PX = 480;

/** In-app webviews: no getUserMedia worth having, and no way to fix it from inside. */
const IN_APP_BROWSER_UA = /FBAN|FBAV|FB_IAB|Instagram|LinkedInApp|Line\/|MicroMessenger|WeChat|Snapchat|TikTok|Pinterest|GSA\//i;

// -------------------------------------------------------------------------------------
// Pure kernels
// -------------------------------------------------------------------------------------

/** sRGB 8-bit -> linear, precomputed once. */
const LINEARIZE = (() => {
  const t = new Float32Array(256);
  for (let i = 0; i < 256; i += 1) {
    const c = i / 255;
    t[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  return t;
})();

export interface LumaStats {
  /** Mean CIE L*, rescaled 0-255 the way OpenCV scales its L channel. */
  meanLuma: number;
  /** Fraction of pixels sitting on 0 or 255 — extract._clipping_penalty's input. */
  clippedFraction: number;
}

/**
 * Mean L* and clipped fraction over an RGBA buffer.
 *
 * L* rather than a plain grey average because that is what the engine measures: the two real
 * videos came in at 163 and 144, the synthetics at 117-132, and validate.MIN_MEAN_BRIGHTNESS
 * = 25 is quoted on the same scale. A luma average would read systematically brighter in
 * dark rooms and would move the block threshold without anyone noticing.
 */
export function analyseFrameLuma(rgba: Uint8ClampedArray): LumaStats {
  const pixels = rgba.length >> 2;
  if (pixels === 0) return { meanLuma: 0, clippedFraction: 0 };

  let sumL = 0;
  let clipped = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    const r = rgba[i];
    const g = rgba[i + 1];
    const b = rgba[i + 2];
    const y = 0.2126 * LINEARIZE[r] + 0.7152 * LINEARIZE[g] + 0.0722 * LINEARIZE[b];
    const f = y > 0.008856 ? Math.cbrt(y) : 7.787 * y + 16 / 116;
    sumL += (116 * f - 16) * 2.55;

    const grey = 0.299 * r + 0.587 * g + 0.114 * b;
    if (grey >= 254.5 || grey <= 0.5) clipped += 1;
  }

  return { meanLuma: sumL / pixels, clippedFraction: clipped / pixels };
}

/** Zero-mean profile of one quadrant along an axis, for the flow correlator. */
function profiles(grey: Float32Array, canvasW: number, qx: number, qy: number, qw: number, qh: number): { cols: Float32Array; rows: Float32Array } {
  const cols = new Float32Array(qw);
  const rows = new Float32Array(qh);
  for (let y = 0; y < qh; y += 1) {
    let rowSum = 0;
    const base = (qy + y) * canvasW + qx;
    for (let x = 0; x < qw; x += 1) {
      const v = grey[base + x];
      cols[x] += v;
      rowSum += v;
    }
    rows[y] = rowSum / qw;
  }
  for (let x = 0; x < qw; x += 1) cols[x] /= qh;
  return { cols, rows };
}

interface LagResult {
  shift: number;
  peak: number;
  valid: boolean;
}

/**
 * Normalised cross-correlation of two 1-D profiles over +/- maxLag, with a parabolic
 * sub-pixel refinement of the peak.
 *
 * This is a projection correlator, not the FFT phase correlation the contract's prose
 * describes, and the difference is deliberate: an FFT would drag a transform implementation
 * into a hot path for no gain here. The statistic that matters is the DIFFERENCE between the
 * four quadrants' shifts, both methods estimate the same shift for the small displacements
 * involved (a 24 deg/s sweep moves ~4.8 deg, ~68 px at 640 width, between 5 Hz samples), and
 * the thresholds are quoted in px at 640 width, which transfers unchanged.
 */
function bestLag(a: Float32Array, b: Float32Array, maxLag: number): LagResult {
  const n = a.length;
  let meanA = 0;
  for (let i = 0; i < n; i += 1) meanA += a[i];
  meanA /= n;

  let varA = 0;
  for (let i = 0; i < n; i += 1) varA += (a[i] - meanA) ** 2;
  // A flat wall or a black frame carries no shift information at all. Saying so is the
  // whole point of FlowSample.valid.
  if (Math.sqrt(varA / n) < 1.5) return { shift: 0, peak: 0, valid: false };

  let bestScore = -Infinity;
  let bestIndex = 0;
  const scores = new Float32Array(2 * maxLag + 1);

  for (let lag = -maxLag; lag <= maxLag; lag += 1) {
    const from = Math.max(0, -lag);
    const to = Math.min(n, n - lag);
    const count = to - from;
    if (count < n / 2) continue;

    let sa = 0;
    let sb = 0;
    let saa = 0;
    let sbb = 0;
    let sab = 0;
    for (let i = from; i < to; i += 1) {
      const va = a[i];
      const vb = b[i + lag];
      sa += va;
      sb += vb;
      saa += va * va;
      sbb += vb * vb;
      sab += va * vb;
    }
    const cov = sab / count - (sa / count) * (sb / count);
    const sda = Math.sqrt(Math.max(1e-6, saa / count - (sa / count) ** 2));
    const sdb = Math.sqrt(Math.max(1e-6, sbb / count - (sb / count) ** 2));
    const score = cov / (sda * sdb);
    scores[lag + maxLag] = score;
    if (score > bestScore) {
      bestScore = score;
      bestIndex = lag + maxLag;
    }
  }

  if (bestScore < 0.4) return { shift: 0, peak: bestScore, valid: false };

  let shift = bestIndex - maxLag;
  if (bestIndex > 0 && bestIndex < scores.length - 1) {
    const l = scores[bestIndex - 1];
    const c = scores[bestIndex];
    const r = scores[bestIndex + 1];
    const denom = l - 2 * c + r;
    if (Math.abs(denom) > 1e-6) shift += (0.5 * (l - r)) / denom;
  }
  return { shift, peak: bestScore, valid: true };
}

export interface FlowFrame {
  /** Grey plane of the whole downscaled frame, row-major, `canvasW * canvasH` long. */
  grey: Float32Array;
  canvasW: number;
  canvasH: number;
}

/**
 * Quadrant optical-flow divergence.
 *
 * A pure rotation moves all four quadrants together. A translation moves them by different
 * amounts, because each quadrant sees a different depth — which is precisely the differential
 * the engine's rotational-purity residual measures after the fact. This is the only walking
 * signal that survives when the motion sensors are denied.
 *
 * `frameW`/`frameH` are the DECODED frame dimensions, needed to normalise a shift measured on
 * the small canvas back to the contract's 640-px-wide reference.
 */
export function quadrantFlow(
  previous: FlowFrame,
  current: FlowFrame,
  frameW: number,
  frameH: number,
  pxPerDegAt640: number,
  t: Millis,
): FlowSample {
  const { canvasW, canvasH } = current;
  const qw = canvasW >> 1;
  const qh = canvasH >> 1;

  // A shift of one canvas pixel is (frameW / canvasW) frame pixels, which is
  // (frameW / canvasW) * (640 / frameW) = 640 / canvasW pixels at the reference width. The
  // vertical factor differs whenever the canvas aspect is not the frame aspect.
  const sx = 640 / canvasW;
  const sy = (frameH * 640) / (canvasH * Math.max(1, frameW));

  const shifts: { x: number; y: number }[] = [];
  for (let qi = 0; qi < 4; qi += 1) {
    const qx = (qi & 1) * qw;
    const qy = (qi >> 1) * qh;
    const p = profiles(previous.grey, canvasW, qx, qy, qw, qh);
    const c = profiles(current.grey, canvasW, qx, qy, qw, qh);
    const dx = bestLag(p.cols, c.cols, Math.min(24, qw >> 2));
    const dy = bestLag(p.rows, c.rows, Math.min(16, qh >> 3));
    if (dx.valid && dy.valid) shifts.push({ x: dx.shift * sx, y: dy.shift * sy });
  }

  if (shifts.length < 2) return { t, divergencePx: 0, meanShiftPx: 0, yawDeltaDeg: 0, valid: false };

  let divergence = 0;
  for (let i = 0; i < shifts.length; i += 1) {
    for (let j = i + 1; j < shifts.length; j += 1) {
      divergence = Math.max(divergence, Math.hypot(shifts[i].x - shifts[j].x, shifts[i].y - shifts[j].y));
    }
  }

  let meanX = 0;
  for (const s of shifts) meanX += s.x;
  meanX /= shifts.length;

  // Content sliding LEFT means the camera turned RIGHT, and yaw increases clockwise.
  return { t, divergencePx: divergence, meanShiftPx: meanX, yawDeltaDeg: -meanX / pxPerDegAt640, valid: true };
}

/** Focal-derived px per degree at the engine's 640-px reference width. */
export function pxPerDegAt640(lens: LensGeometry): number {
  return (320 / Math.tan((lens.hfovDeg / 2) * DEG)) * DEG;
}

// -------------------------------------------------------------------------------------
// Device selection
// -------------------------------------------------------------------------------------

export interface RearLensPick {
  deviceId: string | null;
  label: string;
  /**
   * True when we could NOT prove a physical rear wide was chosen: no labels, no candidate,
   * or (checked after opening) the browser handed back a different deviceId than requested.
   */
  virtual: boolean;
  /** How many rear-facing candidates survived the filter; >1 means the pick is a convention. */
  candidateCount: number;
}

/**
 * Pick the physical rear wide from an enumerated device list.
 *
 * `LENS_SELECTION.reject` deliberately does NOT contain /wide/: several Android OEMs label
 * the MAIN lens "Wide Angle", and rejecting that would leave no candidate at all. "Ultra
 * wide" is caught by /ultra/, which is the case that actually matters. On iOS the surviving
 * candidate is "Back Camera", which is the 26mm wide; the auto-switching virtual devices are
 * named "Back Dual Wide Camera" / "Back Triple Camera" and are removed by /dual|triple/.
 *
 * When several candidates survive (Android often exposes every rear sensor with an
 * indistinguishable "camera2 N, facing back" label), the first in enumeration order is taken:
 * camera id 0 is the main sensor by Android convention. That is a convention, not a
 * guarantee, so `candidateCount` travels with the pick and lands in telemetry.
 */
export function pickRearCamera(devices: MediaDeviceInfo[], preference: LensPreference = "wide"): RearLensPick {
  const inputs = devices.filter((d) => d.kind === "videoinput");
  const labelled = inputs.filter((d) => d.label.trim().length > 0);
  if (labelled.length === 0) {
    return { deviceId: null, label: "", virtual: true, candidateCount: 0 };
  }

  if (preference === "ultrawide") {
    // The 3D walkthrough WANTS the 13 mm lens: more of the room in every frame is more
    // overlap for the solver and fewer frames for the trainer. iOS names it ("Back Ultra
    // Wide Camera"); Android mostly does not, and there the zoom floor in `applyControls`
    // selects it through the logical camera instead.
    const ultra = labelled.filter((d) => LENS_SELECTION.accept.test(d.label) && /ultra/i.test(d.label) && !/front|depth|macro/i.test(d.label));
    if (ultra.length > 0) {
      return { deviceId: ultra[0].deviceId, label: ultra[0].label, virtual: false, candidateCount: ultra.length };
    }
  }

  const candidates = labelled.filter((d) => LENS_SELECTION.accept.test(d.label) && !LENS_SELECTION.reject.test(d.label));
  if (candidates.length === 0) {
    return { deviceId: null, label: "", virtual: true, candidateCount: 0 };
  }

  const chosen = candidates[0];
  return { deviceId: chosen.deviceId, label: chosen.label, virtual: false, candidateCount: candidates.length };
}

// -------------------------------------------------------------------------------------
// Hook
// -------------------------------------------------------------------------------------

/** The slice of `navigator.mediaDevices` this module uses; injectable for tests. */
export interface MediaDevicesLike {
  getUserMedia: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
  enumerateDevices: () => Promise<MediaDeviceInfo[]>;
}

/** Which rear lens to pin: the 26 mm wide (panoramas) or the 13 mm ultra-wide (walkthroughs). */
export type LensPreference = "wide" | "ultrawide";

export interface UseCameraOptions {
  /** Studio asks for 2160 on the short edge; balanced asks for 1080 and keeps the fps. */
  qualityMode?: GuidedQualityMode;
  lens?: LensGeometry;
  lensPreference?: LensPreference;
  /** Overrides of `CAMERA_CONTROLS` for a capture with different needs (a walkthrough locks white balance). */
  controls?: Partial<CameraControlPlan>;
  /**
   * The governor's current plan, read fresh each frame. Gates the analysis clocks from the
   * same place the rest of the loop is gated, so a phone that is shedding load stops paying
   * for optical flow here too. Passed as a getter, never as a value: a value would join the
   * dependency array below and every tier change would tear down the video-frame callback
   * and throw away the flow history with it.
   */
  loadPlan?: () => LoadPlan;
  /** Live |rotationRate| for the smear prediction carried in every FrameQualitySample. */
  omegaDegPerS?: () => number;
  /** Injected media layer. Replaces `navigator.mediaDevices` entirely. */
  mediaDevices?: MediaDevicesLike;
  /** Image statistics rate. 10 Hz matches the coverage stamp rate. */
  qualityHz?: number;
  onLensChanged?: () => void;
  onTrackEnded?: () => void;
  now?: () => Millis;
}

export interface CameraApiExt extends CameraApi {
  /** DOMException name behind `error`; "NotReadableError" is the camera-in-use case. */
  errorName: string | null;
  /** True once a stream is live and its short edge clears RECORDING.minShortEdgePx. */
  ready: boolean;
  /** How many rear candidates the label filter left. >1 means the pick was a convention. */
  rearCandidateCount: number;
  /** Torch capability, feature-detected. iOS has had it since 17.x despite the support tables. */
  torchAvailable: boolean;
  /** What the shutter cap did (Android), or null before it ran. */
  shutterCap: ShutterCapRecord | null;
}

interface QualityState {
  sample: FrameQualitySample | null;
  frameTimes: number[];
}

export function useCamera(options: UseCameraOptions = {}): CameraApiExt {
  const {
    qualityMode = "balanced",
    lens = DEFAULT_LENS,
    lensPreference = "wide",
    controls,
    loadPlan,
    omegaDegPerS,
    mediaDevices,
    qualityHz = COVERAGE_STAMP_HZ,
    onLensChanged,
    onTrackEnded,
    now,
  } = options;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const stillCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const qualityRef = useRef<QualityState>({ sample: null, frameTimes: [] });
  /** Owns the worker, the bitmaps in flight and the synchronous fallback. */
  const analyserRef = useRef<FrameAnalyser | null>(null);
  /** Why the worker path was abandoned, if it was. Surfaced by the debug HUD and the trace. */
  const analysisDemotedRef = useRef<string | null>(null);
  const lastQualityAtRef = useRef(0);
  const lastFlowAtRef = useRef(0);
  const signatureRef = useRef<LensSignature | null>(null);
  /** True while `applyControls` is mid-flight: it changes zoom on purpose, and the watchdog
   *  polling in the middle of that would read our own control as the OS switching lenses. */
  const applyingControlsRef = useRef(false);
  const openingRef = useRef(false);

  // Late-bound callbacks and the clock live in refs SYNCED IN AN EFFECT — never written during
  // render. Rebuilding the frame loop on every parent render would cancel the video-frame
  // callback and throw away the previous flow frame, which is half of every divergence
  // measurement.
  const clockRef = useRef(now ?? nowMs);
  const omegaRef = useRef(omegaDegPerS);
  const onLensChangedRef = useRef(onLensChanged);
  const onTrackEndedRef = useRef(onTrackEnded);
  const loadPlanRef = useRef(loadPlan);
  useEffect(() => {
    clockRef.current = now ?? nowMs;
    omegaRef.current = omegaDegPerS;
    onLensChangedRef.current = onLensChanged;
    onTrackEndedRef.current = onTrackEnded;
    loadPlanRef.current = loadPlan;
  });

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const [label, setLabel] = useState<string | null>(null);
  const [virtualLensOnly, setVirtualLensOnly] = useState(false);
  const [rearCandidateCount, setRearCandidateCount] = useState(0);
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [appliedControls, setAppliedControls] = useState<Partial<Record<keyof CameraControlPlan, boolean>>>({});
  const [shutterCap, setShutterCap] = useState<ShutterCapRecord | null>(null);
  /** Latched: a genuine lens switch invalidates every pose fitted before it, so the capture
   *  must stop rather than quietly carry on. Cleared on `close`, which is what a retake goes
   *  through -- without that, one trip ended the session for good. */
  const [configurationChanged, setConfigurationChanged] = useState(false);
  const [torchAvailable, setTorchAvailable] = useState(false);
  const [error, setError] = useState<CameraError | null>(null);
  const [errorName, setErrorName] = useState<string | null>(null);

  // ---------------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------------

  const close = useCallback(() => {
    const current = streamRef.current;
    streamRef.current = null;
    if (current) for (const track of current.getTracks()) track.stop();
    const video = videoRef.current;
    if (video) {
      try {
        video.srcObject = null;
      } catch {
        // Safari throws if the element is already torn down. Nothing to recover.
      }
    }
    qualityRef.current = { sample: null, frameTimes: [] };
    signatureRef.current = null;
    setConfigurationChanged(false);
    setStream(null);
    setSize(null);
  }, []);

  const open = useCallback(async () => {
    if (openingRef.current) return;
    openingRef.current = true;
    setError(null);
    setErrorName(null);

    try {
      if (typeof window === "undefined") return;
      if (typeof navigator !== "undefined" && IN_APP_BROWSER_UA.test(navigator.userAgent)) {
        setError("in-app-browser");
        return;
      }
      if (!window.isSecureContext) {
        setError("insecure-context");
        return;
      }
      const devices: MediaDevicesLike | undefined =
        mediaDevices ??
        (typeof navigator !== "undefined" && navigator.mediaDevices?.getUserMedia
          ? {
              getUserMedia: (c) => navigator.mediaDevices.getUserMedia(c),
              enumerateDevices: () => navigator.mediaDevices.enumerateDevices(),
            }
          : undefined);
      if (!devices) {
        setError("no-device");
        return;
      }

      close();

      // PHASE 1 — a throwaway grant, purely to unlock device labels. Labels and real
      // deviceIds appear only after a grant, and without labels there is no way to tell the
      // wide from the ultra-wide.
      let pick: RearLensPick = { deviceId: null, label: "", virtual: true, candidateCount: 0 };
      try {
        const probe = await devices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
        try {
          pick = pickRearCamera(await devices.enumerateDevices(), lensPreference);
        } finally {
          for (const track of probe.getTracks()) track.stop();
        }
      } catch (cause) {
        setError(mapCameraError(cause));
        setErrorName(errorNameOf(cause));
        return;
      }

      // PHASE 2 — reopen on the physical rear wide by exact deviceId. deviceIds are
      // regenerated on every page load, so nothing here is ever persisted.
      const shortEdge = qualityMode === "studio" ? RECORDING.preferredShortEdgePx : RECORDING.minShortEdgePx;
      const ladder = buildConstraintLadder(pick.deviceId, shortEdge);

      let opened: MediaStream | null = null;
      let lastError: unknown = null;
      for (const constraints of ladder) {
        try {
          opened = await devices.getUserMedia(constraints);
          break;
        } catch (cause) {
          lastError = cause;
        }
      }
      if (!opened) {
        setError(mapCameraError(lastError));
        setErrorName(errorNameOf(lastError));
        return;
      }

      streamRef.current = opened;
      setStream(opened);

      const track = opened.getVideoTracks()[0] ?? null;
      const live = (track?.getSettings() ?? {}) as MediaTrackSettings & Record<string, unknown>;
      setSettings({ ...live });
      setLabel(track?.label ?? pick.label ?? null);
      setRearCandidateCount(pick.candidateCount);
      signatureRef.current = lensSignature(live);

      // The verification that matters: Apple forum 776460 documents deviceId being silently
      // ignored on a virtual device. If what came back is not what was asked for, we did not
      // pin a lens, and the upload must say so.
      const honoured = pick.deviceId !== null && typeof live.deviceId === "string" && live.deviceId === pick.deviceId;
      const onVirtualLens = pick.virtual || !honoured;
      setVirtualLensOnly(onVirtualLens);

      const measured = await attachToVideo(videoRef, opened);
      if (measured) setSize(measured);

      if (measured && Math.min(measured.width, measured.height) < RECORDING.minShortEdgePx) {
        // `ideal` never throws, so a phone that cannot do 1080 hands back 720 with no error
        // anywhere. Ask once with a hard minimum to find out whether a better mode exists.
        const rescued = await retryWithMinimum(devices, pick.deviceId);
        if (rescued) {
          for (const t of opened.getTracks()) t.stop();
          streamRef.current = rescued;
          setStream(rescued);
          const rescuedSize = await attachToVideo(videoRef, rescued);
          if (rescuedSize) setSize(rescuedSize);
          const rescuedTrack = rescued.getVideoTracks()[0];
          if (rescuedTrack) setSettings({ ...(rescuedTrack.getSettings() as Record<string, unknown>) });
        } else {
          setError("resolution-too-low");
        }
      }

      const activeTrack = streamRef.current?.getVideoTracks()[0] ?? null;
      if (activeTrack) {
        const caps = (activeTrack.getCapabilities?.() ?? {}) as Record<string, unknown>;
        setTorchAvailable(Array.isArray(caps.torch) ? caps.torch.includes(true) : Boolean(caps.torch));
        // Let auto-exposure and auto-white-balance settle before freezing them, or the whole
        // capture inherits whatever the sensor guessed in its first frame.
        applyingControlsRef.current = true;
        window.setTimeout(() => {
          // `!onVirtualLens` is what lets the zoom go below 1: on a virtual device that
          // would be the OS switching to the ultra-wide mid-capture. See `pinZoomToMin`.
          void applyControls(activeTrack, !onVirtualLens, { ...CAMERA_CONTROLS, ...controls, zoomFloor: lensPreference === "ultrawide" ? "min" : 1 })
            .then(async (applied) => {
              const after = activeTrack.getSettings() as MediaTrackSettings & Record<string, unknown>;
              setSettings({ ...after });
              // RE-BASELINE. The signature hashes zoom and frame shape, and `applyControls` has
              // just changed zoom on purpose -- `pinZoomToMin` is the whole point of it. Left
              // alone, the watchdog reads our own control as the OS switching lenses, latches
              // `configurationChanged`, and the `lens_changed` warning it raises is severity
              // `stop`: it blocks the capture, so dwell never accumulates and NO reticle ever
              // fills. On Android this hid, because there zoom was already at its minimum and
              // the value did not move. On iPhone it does, and the capture is dead on arrival.
              signatureRef.current = lensSignature(after);
              // The shutter cap runs AFTER the zoom is pinned (the lens it meters for is the
              // lens that records) and while the watchdog is still muted.
              if ((controls?.capExposure ?? CAMERA_CONTROLS.capExposure) && activeTrack.readyState === "live") {
                const cap = await applyShutterCap(activeTrack, videoRef.current);
                setShutterCap(cap);
                applied.capExposure = cap.applied;
                setSettings({ ...(activeTrack.getSettings() as Record<string, unknown>) });
              }
              setAppliedControls(applied);
            })
            .finally(() => {
              applyingControlsRef.current = false;
            });
        }, controls?.settleMs ?? CAMERA_CONTROLS.settleMs);
      }
    } finally {
      openingRef.current = false;
    }
  }, [close, controls, lensPreference, mediaDevices, qualityMode]);

  // Release everything on unmount: a live track keeps the sensor hot and the LED on.
  useEffect(() => close, [close]);

  // ---------------------------------------------------------------------------------
  // Per-frame analysis
  // ---------------------------------------------------------------------------------

  useEffect(() => {
    if (!stream || typeof window === "undefined") return;

    const clock = clockRef.current;
    const flowIntervalMs = 1000 / THRESHOLDS.flowDivergence.rateHz;
    const qualityIntervalMs = 1000 / Math.max(1, qualityHz);
    const pxPerDeg = pxPerDegAt640(lens);

    // Built per stream, not per mount: a lens change rebuilds this effect, and the worker's
    // flow history is keyed to the geometry it was measuring. Carrying it across would
    // correlate a 1080-wide frame against a 720-wide one and report a divergence the room
    // never had, which is a hard stop shown to the user.
    const analyser = createFrameAnalyser({
      // Why the offload gave up, recorded where it can actually be read. Without this the
      // worker path can be abandoned for any of eight reasons and the phone spends the whole
      // capture on the synchronous readback with nothing anywhere saying so — which is
      // indistinguishable, from the outside, from the offload simply not having helped.
      onDemote: (reason) => {
        analysisDemotedRef.current = reason;
      },
    });
    analyserRef.current = analyser;
    let disposed = false;
    let rvfcHandle = 0;
    let rafHandle = 0;
    let waitHandle = 0;
    let video: HTMLVideoElement | null = null;

    // ONE SUBMISSION PER DUE CLOCK, AND NOT ONE PIXEL READ ON THIS THREAD.
    //
    // What used to live here was `drawImage(video, ...)` into a small 2D canvas followed by
    // `getImageData`, ten and five times a second. The destination was tiny, which is why
    // two earlier passes costed it and cleared it — but the destination was never the cost.
    // A 2D canvas under Blink's `minimumAccelerated2dCanvasSize` (257x256) is CPU-backed no
    // matter what flags it is given, and a CPU-backed destination cannot take
    // `PaintCanvasVideoRenderer`'s accelerated path: `drawImage` from a texture-backed
    // Android camera frame falls through to `ReadbackTextureBackedFrameToMemorySync`, which
    // flushes the GL command buffer, blocks this thread on an IPC round trip to the GPU
    // process, and copies the WHOLE 1080x1920 surface into system memory before scaling it.
    // It runs inside `requestVideoFrameCallback`, i.e. in the rendering steps immediately
    // before the frame is committed, and the command-buffer drain makes the hardware H.264
    // encoder's in-flight surface complete first. The encoder and the compositor end up
    // queued behind a luma histogram.
    //
    // `createImageBitmap(video, { resizeWidth, resizeHeight })` does the same downscale
    // asynchronously and off this thread, and the result is transferred to a worker that
    // owns the OffscreenCanvas and the kernels. The barrier is not made smaller; it leaves
    // the main thread entirely. `frameAnalysis.ts` keeps the old synchronous path as the
    // fallback for browsers with no worker, no OffscreenCanvas or no createImageBitmap,
    // where a CPU canvas read frequently is exactly the right thing to do.
    const onFrame = (mediaTime: number) => {
      if (disposed || !video) return;
      const t = clock();
      const state = qualityRef.current;
      state.frameTimes.push(t);
      const windowStart = t - THRESHOLDS.frameRate.windowMs;
      while (state.frameTimes.length > 0 && state.frameTimes[0] < windowStart) state.frameTimes.shift();

      // The governor's plan, read fresh rather than captured: this closure outlives every
      // tier change by design.
      const plan = loadPlanRef.current?.() ?? null;
      const qualityDue = t - lastQualityAtRef.current >= (plan ? 1000 / plan.qualityHz : qualityIntervalMs);
      const flowDue = (plan === null || plan.opticalFlow) && t - lastFlowAtRef.current >= flowIntervalMs;
      if (!qualityDue && !flowDue) return;
      if (qualityDue) lastQualityAtRef.current = t;
      if (flowDue) lastFlowAtRef.current = t;

      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (vw === 0 || vh === 0) return;

      const times = state.frameTimes;
      const spanSeconds = times.length > 1 ? (times[times.length - 1] - times[0]) / 1000 : 0;
      const measuredFps = spanSeconds > 0.2 ? (times.length - 1) / spanSeconds : 0;

      // Android reports exposureTime in 100-microsecond units; iOS does not report it at all
      // and it has to be inferred from the delivered frame rate and the scene brightness.
      // Sampled HERE rather than when the worker replies, so the composed sample describes
      // the same instant its pixels do.
      const track = streamRef.current?.getVideoTracks()[0];
      const trackSettings = (track?.getSettings() ?? {}) as Record<string, unknown>;
      const reportedExposureS = typeof trackSettings.exposureTime === "number" ? trackSettings.exposureTime / 10000 : 0;

      analyserRef.current?.submit(video, {
        t,
        quality: qualityDue,
        flow: flowDue,
        mediaTime,
        frameW: vw,
        frameH: vh,
        measuredFps,
        reportedExposureS,
        omegaDegPerS: omegaRef.current?.() ?? 0,
        pxPerDeg,
      });
    };

    type RvfcVideo = HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
      cancelVideoFrameCallback?: (handle: number) => void;
    };
    let rvfcVideo: RvfcVideo | null = null;

    const begin = (element: HTMLVideoElement) => {
      video = element;
      // The element can mount AFTER the stream opens — the page shows the preview only from
      // the framing step onward — so the attachment is (re)made here rather than assumed.
      if (element.srcObject !== stream) {
        element.srcObject = stream;
        element.muted = true;
        element.playsInline = true;
        void element.play().catch(() => undefined);
      }

      rvfcVideo = element as RvfcVideo;
      if (typeof rvfcVideo.requestVideoFrameCallback === "function") {
        const step = (_now: number, meta: { mediaTime: number }) => {
          onFrame(meta.mediaTime);
          if (!disposed && rvfcVideo && typeof rvfcVideo.requestVideoFrameCallback === "function") {
            rvfcHandle = rvfcVideo.requestVideoFrameCallback(step);
          }
        };
        rvfcHandle = rvfcVideo.requestVideoFrameCallback(step);
      } else {
        // No requestVideoFrameCallback (pre-15.4 Safari, Firefox). rAF cannot see dropped
        // frames, so `measuredFps` degrades to the paint rate — reported as measured anyway,
        // because it still bounds the delivered rate from above.
        const step = () => {
          onFrame(element.currentTime);
          if (!disposed) rafHandle = window.requestAnimationFrame(step);
        };
        rafHandle = window.requestAnimationFrame(step);
      }
    };

    const waitForElement = () => {
      if (disposed) return;
      const element = videoRef.current;
      if (!element) {
        waitHandle = window.requestAnimationFrame(waitForElement);
        return;
      }
      begin(element);
    };
    waitForElement();

    return () => {
      disposed = true;
      if (waitHandle) window.cancelAnimationFrame(waitHandle);
      if (rvfcHandle && rvfcVideo && typeof rvfcVideo.cancelVideoFrameCallback === "function") rvfcVideo.cancelVideoFrameCallback(rvfcHandle);
      if (rafHandle) window.cancelAnimationFrame(rafHandle);
      analyser.dispose();
      if (analyserRef.current === analyser) analyserRef.current = null;
    };
  }, [stream, lens, qualityHz]);

  // An incoming call ends the track on iOS: the recorder seals its segment and the page
  // decides whether to reopen. It is never a permission problem, and it is the reason a scene
  // can legitimately consist of several files.
  useEffect(() => {
    if (!stream) return;
    const tracks = stream.getVideoTracks();
    const onEnded = () => onTrackEndedRef.current?.();
    for (const track of tracks) track.addEventListener("ended", onEnded);
    return () => {
      for (const track of tracks) track.removeEventListener("ended", onEnded);
    };
  }, [stream]);

  // ---------------------------------------------------------------------------------
  // Silent-lens-switch watchdog and rotation
  // ---------------------------------------------------------------------------------

  useEffect(() => {
    if (!stream || typeof window === "undefined") return;
    const timer = window.setInterval(() => {
      const track = streamRef.current?.getVideoTracks()[0];
      if (!track) return;
      const live = track.getSettings() as MediaTrackSettings & Record<string, unknown>;
      if (applyingControlsRef.current) return;
      const signature = lensSignature(live);
      if (signatureRef.current !== null && lensReallyChanged(signatureRef.current, signature)) {
        signatureRef.current = signature;
        setSettings({ ...live });
        setConfigurationChanged(true);
        onLensChangedRef.current?.();
      }
    }, LENS_SELECTION.configurationPollMs);
    return () => window.clearInterval(timer);
  }, [stream]);

  useEffect(() => {
    if (!stream || typeof window === "undefined") return;
    let disposed = false;
    let waitHandle = 0;
    let detach: (() => void) | null = null;

    const attach = (video: HTMLVideoElement) => {
      // THE STREAM ITSELF. This effect used to bind only the dimension listeners below,
      // which meant the preview element was mounted with no source at all whenever it
      // arrived after `open()` had already run — and the two-phase lens selection
      // guarantees that, because phase one's track is stopped and replaced. The result on
      // a real phone was a completely black screen with the guidance drawn over it: the
      // capture worked, the user simply could not see the room they were filming.
      //
      // `srcObject` is a DOM property rather than an attribute, so React cannot restore it
      // on re-render either; it has to be set here, every time the stream or the element
      // changes.
      if (video.srcObject !== stream) {
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        void video.play().catch(() => {
          // Autoplay refusal is survivable: the element starts as soon as it is visible,
          // and the framing gate will not pass until dimensions arrive either way.
        });
      }

      // Rotation changes what the decoder hands us; the stream itself must survive it.
      const refresh = () => {
        if (video.videoWidth <= 0 || video.videoHeight <= 0) return;
        setSize({ width: video.videoWidth, height: video.videoHeight });
        // Re-checked here because the decoder is the only honest source of the dimensions,
        // and it may only speak once the element has mounted.
        const short = Math.min(video.videoWidth, video.videoHeight);
        setError((current) => {
          if (short < RECORDING.minShortEdgePx) return "resolution-too-low";
          return current === "resolution-too-low" ? null : current;
        });
      };
      refresh();
      video.addEventListener("resize", refresh);
      video.addEventListener("loadedmetadata", refresh);
      window.addEventListener("orientationchange", refresh);
      window.screen?.orientation?.addEventListener?.("change", refresh);
      detach = () => {
        video.removeEventListener("resize", refresh);
        video.removeEventListener("loadedmetadata", refresh);
        window.removeEventListener("orientationchange", refresh);
        window.screen?.orientation?.removeEventListener?.("change", refresh);
      };
    };

    const waitForElement = () => {
      if (disposed) return;
      const element = videoRef.current;
      if (!element) {
        waitHandle = window.requestAnimationFrame(waitForElement);
        return;
      }
      attach(element);
    };
    waitForElement();

    return () => {
      disposed = true;
      if (waitHandle) window.cancelAnimationFrame(waitHandle);
      detach?.();
    };
  }, [stream]);

  // ---------------------------------------------------------------------------------
  // Readers
  // ---------------------------------------------------------------------------------

  /** "worker" or "main-thread" — the debug HUD's proof the offload actually took. */
  const analysisBackend = useCallback(() => analyserRef.current?.backend ?? "main-thread", []);
  /** Null while the worker is healthy; otherwise the reason it was written off. */
  const analysisDemotedReason = useCallback(() => analysisDemotedRef.current, []);
  const readQuality = useCallback(() => analyserRef.current?.readQuality() ?? null, []);
  const readFlow = useCallback(() => analyserRef.current?.readFlow() ?? null, []);

  /**
   * A still for the review strip.
   *
   * Sized to what the strip actually shows, not to the sensor. The images are rendered at
   * 54x96 CSS px (`ReviewStep.tsx`), i.e. 162x288 device pixels on a dpr-3 phone, so a full
   * 1080x1920 grab was blitting and JPEG-encoding forty times the pixels anyone would ever
   * see — synchronously, inside the capture loop, while the encoder is running.
   *
   * The canvas is pooled through `ensureCanvas` for the same reason every other canvas in
   * this file is: a fresh 2 Mpx backing store per grab is a megabyte of garbage handed to a
   * collector that is already being asked not to pause during a recording.
   */
  const grabStill = useCallback(async (): Promise<Blob | null> => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || typeof document === "undefined") return null;
    const scale = Math.min(1, THUMBNAIL_LONG_EDGE_PX / Math.max(video.videoWidth, video.videoHeight));
    const canvas = ensureCanvas(
      stillCanvasRef,
      Math.max(1, Math.round(video.videoWidth * scale)),
      Math.max(1, Math.round(video.videoHeight * scale)),
    );
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    if (typeof canvas.toBlob !== "function") return null;
    return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.85));
  }, []);

  const setTorch = useCallback(async (on: boolean): Promise<boolean> => {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track) return false;
    try {
      await track.applyConstraints({ advanced: [{ torch: on }] } as unknown as MediaTrackConstraints);
      return (track.getSettings() as Record<string, unknown>).torch === on;
    } catch {
      return false;
    }
  }, []);

  const ready = stream !== null && size !== null && Math.min(size.width, size.height) >= RECORDING.minShortEdgePx && error === null;

  return {
    stream,
    videoRef,
    size,
    label,
    virtualLensOnly,
    settings,
    appliedControls,
    shutterCap,
    configurationChanged,
    error,
    open,
    close,
    readQuality,
    readFlow,
    analysisBackend,
    analysisDemotedReason,
    grabStill,
    setTorch,
    errorName,
    ready,
    rearCandidateCount,
    torchAvailable,
  };
}

// -------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------

function ensureCanvas(ref: RefObject<HTMLCanvasElement | null>, width: number, height: number): HTMLCanvasElement {
  let canvas = ref.current;
  if (!canvas) {
    canvas = document.createElement("canvas");
    ref.current = canvas;
  }
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  return canvas;
}

/**
 * Constraint ladder, most specific first. Every rung uses `ideal` for the geometry because an
 * `exact` resolution throws on phones that could otherwise have delivered something usable;
 * the resolution floor is enforced afterwards against the decoder's own dimensions.
 */
export function buildConstraintLadder(deviceId: string | null, shortEdgePx: number): MediaStreamConstraints[] {
  // A FLOOR ON BOTH EDGES. NO CEILING ON EITHER, AND NO SHAPE.
  //
  // The requirement is one sentence — the SHORT edge must carry at least `shortEdgePx` —
  // and the spec has no way to say "short edge". Saying it as a minimum on both edges is
  // exact: whichever dimension the UA calls width, both are at or above the floor, and that
  // is true in either orientation. It implies no aspect ratio, so there is nothing to crop
  // towards, and it has no maximum, so there is nothing to downscale towards either.
  //
  // TWO REGRESSIONS LIVE HERE, IN OPPOSITE DIRECTIONS, AND BOTH WERE SHIPPED.
  //
  //   1. `width: {ideal: 1080, max: 1080}` WITH `height: {ideal: 1920, max: 1920}` asks for
  //      9:16. On a 3:4 sensor delivering 1728x2304 the only way to reach 9:16 is to throw
  //      away a quarter of the width, so `crop-and-scale` cropped and the horizontal field
  //      of view fell from 43 deg to 32.9. On screen: "the camera is zoomed in".
  //
  //   2. Removing the height cap and keeping `width: {max: 1080}` looked like the fix and
  //      was worse. `width` in a constraint is matched against the SOURCE mode, and phone
  //      camera modes are landscape: 1920x1080. Capping "width" at 1080 therefore capped
  //      the LONG edge, the UA scaled 1920x1080 down to 1080x608, and the track arrived
  //      rotated as 608x1080 — a 0.66 MP video, measured on a real capture, where the
  //      solver then found so few features that 23 of 87 keyframes could not be linked.
  //
  // Both come from trying to express the requirement as a ceiling. It is a floor. The
  // encoder-load worry that motivated the ceiling is real but belongs to the recorder's
  // bitrate, not to the camera's mode: a mode larger than asked for costs a downscale in
  // the encoder, while a mode smaller than asked for costs the capture.
  const geometry = (floorPx: number): MediaTrackConstraints => ({
    width: { min: floorPx },
    height: { min: floorPx },
    // No `min` here. A hard frame-rate floor is unsatisfiable in a dim room at any
    // resolution, and an over-constrained request does not degrade — it throws, and the
    // ladder falls to a rung with no geometry at all.
    frameRate: { ideal: RECORDING.targetFps },
    // Cropping is what produced regression 1; forbid it outright. With only minimums there
    // is no target shape to resample towards, so nothing is given up by saying so.
    ...({ resizeMode: "none" } as Record<string, unknown>),
  });

  // The floor, then a rung for a phone that genuinely has no 1080 mode, then no geometry.
  const floors = [shortEdgePx, Math.round((shortEdgePx * 2) / 3)];
  const ladder: MediaStreamConstraints[] = [];
  for (const floor of floors) {
    const g = geometry(floor);
    if (deviceId) {
      ladder.push({ video: { ...g, deviceId: { exact: deviceId } }, audio: false });
      ladder.push({ video: { ...g, deviceId: { ideal: deviceId } }, audio: false });
    }
    ladder.push({ video: { ...g, facingMode: { ideal: "environment" } }, audio: false });
  }
  ladder.push({ video: { facingMode: { ideal: "environment" } }, audio: false });
  return ladder;
}

/** One hard attempt at the 1080 floor, to tell "phone cannot" from "browser did not bother". */
async function retryWithMinimum(devices: MediaDevicesLike, deviceId: string | null): Promise<MediaStream | null> {
  const video: MediaTrackConstraints = {
    // Both edges, not a 16:9 pair. `height: {min: 1920}` was a shape in disguise and it is
    // what made this retry fail on a phone whose only large mode is 4:3.
    width: { min: RECORDING.minShortEdgePx },
    height: { min: RECORDING.minShortEdgePx },
    frameRate: { ideal: RECORDING.targetFps },
    ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: "environment" } }),
  };
  try {
    return await devices.getUserMedia({ video, audio: false });
  } catch {
    return null;
  }
}

/**
 * Attach a stream to the preview element and wait for the DECODER to state its dimensions.
 *
 * `getSettings()` width/height is an unordered pair on iOS — the track is sensor-native
 * landscape with a rotation applied — so the only trustworthy source for what the file will
 * contain is `video.videoWidth/videoHeight`.
 */
async function attachToVideo(
  ref: RefObject<HTMLVideoElement | null>,
  stream: MediaStream,
): Promise<{ width: number; height: number } | null> {
  const video = await waitForElement(ref);
  if (!video) return null;

  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  try {
    await video.play();
  } catch {
    // Autoplay can still be refused; the element plays as soon as it is visible and the
    // dimensions arrive either way.
  }

  if (video.videoWidth > 0 && video.videoHeight > 0) return { width: video.videoWidth, height: video.videoHeight };

  return new Promise((resolve) => {
    const done = () => {
      video.removeEventListener("loadedmetadata", done);
      resolve(video.videoWidth > 0 ? { width: video.videoWidth, height: video.videoHeight } : null);
    };
    video.addEventListener("loadedmetadata", done);
    window.setTimeout(done, 4000);
  });
}

/** The element may mount a tick after `open()` is called from a click handler. */
async function waitForElement(ref: RefObject<HTMLVideoElement | null>): Promise<HTMLVideoElement | null> {
  for (let i = 0; i < 30; i += 1) {
    if (ref.current) return ref.current;
    await new Promise<void>((resolve) => {
      if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => resolve());
      else setTimeout(resolve, 16);
    });
  }
  return ref.current;
}

/**
 * Everything that identifies WHICH lens is live, and nothing that merely identifies how the
 * phone is being held: width and height swap on rotation, and a rotation is not a lens change.
 */
export interface LensSignature {
  deviceId: string;
  groupId: string;
  /** 0 when the platform does not report it, which iOS often does not. */
  zoom: number;
  /** short/long edge; 0 when unknown. */
  shape: number;
}

function lensSignature(settings: MediaTrackSettings & Record<string, unknown>): LensSignature {
  const w = typeof settings.width === "number" ? settings.width : 0;
  const h = typeof settings.height === "number" ? settings.height : 0;
  return {
    deviceId: typeof settings.deviceId === "string" ? settings.deviceId : "",
    groupId: typeof settings.groupId === "string" ? settings.groupId : "",
    zoom: typeof settings.zoom === "number" && settings.zoom > 0 ? settings.zoom : 0,
    shape: w > 0 && h > 0 ? Math.min(w, h) / Math.max(w, h) : 0,
  };
}

/** A zoom move smaller than this is the platform breathing, not a lens change. Wide to
 *  ultra-wide is a factor of two on every phone that has both; 1.25 sits clear of both. */
const LENS_ZOOM_RATIO = 1.25;
/** 9:16 is 0.5625 and 3:4 is 0.75, so anything under 0.02 is rounding, not a mode change. */
const LENS_SHAPE_EPS = 0.02;

/**
 * Did the OPTICS actually change, or is the platform just talking?
 *
 * THIS EXISTS BECAUSE A STRING COMPARE MADE THE CAPTURE IMPOSSIBLE ON IPHONE. The old
 * signature was a joined string, so any wobble in any field read as a lens switch — and the
 * `lens_changed` warning it raises is severity `stop`, which blocks the capture, which stops
 * dwell accumulating, which means NO reticle ever fills. The user aims at a ring, the ring
 * says it is on target, and nothing happens, for the rest of the session.
 *
 * Android never showed it: there zoom is already at its minimum and the number does not move.
 * iOS hands back one VIRTUAL rear camera and reports a zoom that drifts, so on iPhone the
 * comparison tripped almost immediately.
 *
 * What actually invalidates the poses already fitted is a different camera: a new `deviceId`
 * or `groupId`, a jump between the wide and ultra-wide lens, or a switch to a mode of a
 * different shape. Each of those is checked on its own terms, with a threshold that clears
 * the noise by a wide margin.
 */
export function lensReallyChanged(before: LensSignature, after: LensSignature): boolean {
  if (before.deviceId !== "" && after.deviceId !== "" && before.deviceId !== after.deviceId) return true;
  if (before.groupId !== "" && after.groupId !== "" && before.groupId !== after.groupId) return true;
  if (before.zoom > 0 && after.zoom > 0) {
    const ratio = Math.max(before.zoom, after.zoom) / Math.min(before.zoom, after.zoom);
    if (ratio >= LENS_ZOOM_RATIO) return true;
  }
  if (before.shape > 0 && after.shape > 0 && Math.abs(before.shape - after.shape) > LENS_SHAPE_EPS) {
    return true;
  }
  return false;
}

/**
 * Feature-detect, apply, then READ BACK. Each control is independent so one rejection cannot
 * take the others down with it, and the returned map is what `getSettings()` confirms — not
 * what was requested. Android grants all four; iOS grants white balance only.
 */
/** `CAMERA_CONTROLS` plus where the zoom may go: `1` (the wide) or `"min"` (the widest lens behind the control). */
type AppliedControlPlan = CameraControlPlan & { zoomFloor?: number | "min" };

async function applyControls(
  track: MediaStreamTrack,
  physicalLens: boolean,
  plan: AppliedControlPlan = CAMERA_CONTROLS,
): Promise<Partial<Record<keyof CameraControlPlan, boolean>>> {
  const caps = (track.getCapabilities?.() ?? {}) as Record<string, unknown>;
  const applied: Partial<Record<keyof CameraControlPlan, boolean>> = {};

  const modeList = (key: string): string[] => (Array.isArray(caps[key]) ? (caps[key] as string[]) : []);
  const settingsNow = () => track.getSettings() as Record<string, unknown>;

  const tryConstraint = async (set: Record<string, unknown>): Promise<boolean> => {
    try {
      await track.applyConstraints({ advanced: [set] } as unknown as MediaTrackConstraints);
      return true;
    } catch {
      return false;
    }
  };

  // Exposure. "manual" holds the current exposure outright; "single-shot" runs auto-exposure
  // once and then freezes, which is the closer analogue of an AE lock where manual is refused.
  if (plan.lockExposure) {
    const modes = modeList("exposureMode");
    let ok = false;
    if (modes.includes("manual")) ok = await tryConstraint({ exposureMode: "manual" });
    if (!ok && modes.includes("single-shot")) ok = await tryConstraint({ exposureMode: "single-shot" });
    applied.lockExposure = ok && settingsNow().exposureMode !== "continuous";
  }

  if (plan.lockWhiteBalance) {
    const modes = modeList("whiteBalanceMode");
    let ok = false;
    if (modes.includes("manual")) ok = await tryConstraint({ whiteBalanceMode: "manual" });
    if (!ok && modes.includes("single-shot")) ok = await tryConstraint({ whiteBalanceMode: "single-shot" });
    applied.lockWhiteBalance = ok && settingsNow().whiteBalanceMode !== "continuous";
  }

  // Focus. Everything in a room panorama is effectively at infinity, so manual focus is
  // pinned to the largest focusDistance the lens reports rather than left wherever autofocus
  // last hunted to.
  if (plan.lockFocus) {
    const modes = modeList("focusMode");
    let ok = false;
    if (modes.includes("manual")) {
      const range = caps.focusDistance as { max?: number } | undefined;
      const set: Record<string, unknown> = { focusMode: "manual" };
      if (typeof range?.max === "number") set.focusDistance = range.max;
      ok = await tryConstraint(set);
    }
    if (!ok && modes.includes("single-shot")) ok = await tryConstraint({ focusMode: "single-shot" });
    applied.lockFocus = ok && settingsNow().focusMode !== "continuous";
  }

  // Zoom. Pinned to 1.0 rather than to the capability minimum: on Android phones that expose
  // an ultra-wide through the same logical camera the minimum is 0.5, and pinning there would
  // select exactly the 13mm lens this whole module works to avoid. Where 1.0 is out of range
  // the minimum is the only thing left.
  if (plan.pinZoomToMin) {
    const zoom = caps.zoom as { min?: number; max?: number } | undefined;
    if (zoom && typeof zoom.min === "number" && typeof zoom.max === "number") {
      // THE WIDEST THE PINNED LENS OFFERS — which is not always 1.
      //
      // This used to clamp at `max(zoom.min, 1)`, so a phone reporting `zoom.min = 0.6`
      // was held at 1.0 and the capture threw away field of view the lens was willing to
      // give. The clamp was not arbitrary: on a VIRTUAL device — the one iOS hands back for
      // `facingMode: environment`, which the OS switches between wide, ultra-wide and
      // telephoto at will — anything under 1 is the operating system's way of saying
      // "switch to the ultra-wide", and a 26mm-to-13mm change mid-capture poisons every
      // pose the solver fits.
      //
      // But that risk belongs to the virtual device alone. When the two-phase open has
      // pinned a PHYSICAL rear lens by exact deviceId, there is no second lens behind the
      // zoom control to switch to, and `zoom.min` is simply that lens at its widest. So the
      // floor of 1 applies only where it is earned.
      // The floor is 1.0 EVERYWHERE, physical lens or not. The reasoning above assumed a
      // deviceId that round-trips names one sensor; on Android it names a CameraManager
      // camera, which on Pixel and Samsung phones is a LOGICAL device whose zoom range starts
      // at 0.5-0.6 -- and pinning there selects the ultra-wide for the whole capture, with
      // a field of view the solver models as a plain 26 mm pinhole. A range that starts
      // below 1 is itself the evidence of a second lens behind the control.
      // ...unless the caller asked for the widest lens on purpose: the 3D walkthrough
      // wants the ultra-wide, and on Android the zoom floor is the only way to choose it.
      const floor = plan.zoomFloor === "min" ? zoom.min : (plan.zoomFloor ?? 1);
      const target = Math.min(zoom.max, Math.max(zoom.min, floor));
      void physicalLens;
      const ok = await tryConstraint({ zoom: target });
      applied.pinZoomToMin = ok && Math.abs(Number(settingsNow().zoom ?? target) - target) < 1e-3;
    } else {
      applied.pinZoomToMin = false;
    }
  }

  applied.offerTorch = Array.isArray(caps.torch) ? caps.torch.includes(true) : Boolean(caps.torch);
  return applied;
}

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

/** Mean luma of the live preview, averaged over a few frames. Null when unmeasurable. */
async function probeLuma(video: HTMLVideoElement | null, samples: number, gapMs: number): Promise<number | null> {
  if (!video || video.videoWidth === 0 || typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = 48;
  canvas.height = 64;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  let acc = 0;
  let n = 0;
  for (let i = 0; i < samples; i += 1) {
    if (i > 0) await sleep(gapMs);
    try {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const d = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let sum = 0;
      for (let k = 0; k < d.length; k += 4) sum += 0.299 * d[k] + 0.587 * d[k + 1] + 0.114 * d[k + 2];
      acc += sum / (d.length / 4);
      n += 1;
    } catch {
      return null;
    }
  }
  return n > 0 ? acc / n : null;
}

/**
 * Trade shutter time for gain at the exposure the phone itself metered. See `SHUTTER_CAP`
 * in `types.ts` for the argument and the three failed attempts it answers. Every exit
 * that is not a verified success puts the sensor back on continuous auto-exposure.
 */
async function applyShutterCap(track: MediaStreamTrack, video: HTMLVideoElement | null): Promise<ShutterCapRecord> {
  const caps = (track.getCapabilities?.() ?? {}) as Record<string, unknown>;
  const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const range = (key: string): { min: number; max: number } | null => {
    const r = caps[key] as { min?: unknown; max?: unknown } | undefined;
    const lo = num(r?.min);
    const hi = num(r?.max);
    return lo !== null && hi !== null && hi > lo ? { min: lo, max: hi } : null;
  };
  const before = track.getSettings() as Record<string, unknown>;
  const tAuto = num(before.exposureTime); // Android: units of 100 microseconds
  const isoAuto = num(before.iso);
  const record: ShutterCapRecord = {
    applied: false,
    reason: "unsupported",
    exposureBeforeS: tAuto !== null ? tAuto * 1e-4 : null,
    exposureAfterS: null,
    isoBefore: isoAuto,
    isoAfter: null,
    lumaBefore: null,
    lumaAfter: null,
  };
  const modes = Array.isArray(caps.exposureMode) ? (caps.exposureMode as string[]) : [];
  const expRange = range("exposureTime");
  const isoRange = range("iso");
  if (!modes.includes("manual") || !expRange || !isoRange || tAuto === null || isoAuto === null || tAuto <= 0 || isoAuto <= 0) {
    return record;
  }
  const maxUnits = SHUTTER_CAP.maxExposureS * 1e4;
  const preferredUnits = SHUTTER_CAP.preferredExposureS * 1e4;
  if (tAuto <= maxUnits) {
    return { ...record, reason: "fast_enough" };
  }
  let tNew = Math.min(Math.max(preferredUnits, expRange.min), Math.min(maxUnits, tAuto));
  let isoNew = (isoAuto * tAuto) / tNew;
  if (isoNew > isoRange.max) {
    // Not enough gain headroom for the preferred time: take the longest time the ceiling
    // allows, if it is still meaningfully shorter than what auto-exposure chose.
    tNew = Math.min(tAuto, Math.max(expRange.min, (isoAuto * tAuto) / isoRange.max));
    isoNew = isoRange.max;
    if (tNew >= 0.9 * tAuto) return { ...record, reason: "iso_ceiling" };
  }
  const apply = async (set: Record<string, unknown>): Promise<boolean> => {
    try {
      await track.applyConstraints({ advanced: [set] } as unknown as MediaTrackConstraints);
      return true;
    } catch {
      return false;
    }
  };
  const revert = async (reason: string, lumaAfter: number | null): Promise<ShutterCapRecord> => {
    await apply({ exposureMode: "continuous" });
    return { ...record, reason, lumaAfter };
  };

  const lumaBefore = await probeLuma(video, 3, 60);
  record.lumaBefore = lumaBefore;
  if (lumaBefore === null) return { ...record, reason: "no_luma_probe" };

  const ok = await apply({ exposureMode: "manual", exposureTime: Math.round(tNew), iso: Math.round(isoNew) });
  const after = track.getSettings() as Record<string, unknown>;
  const tGot = num(after.exposureTime);
  const isoGot = num(after.iso);
  const within = (got: number | null, want: number) => got !== null && Math.abs(got - want) <= SHUTTER_CAP.readbackTolerance * want;
  if (!ok || after.exposureMode === "continuous" || !within(tGot, tNew) || !within(isoGot, isoNew)) {
    return revert("readback", null);
  }
  record.exposureAfterS = tGot !== null ? tGot * 1e-4 : null;
  record.isoAfter = isoGot;

  // Closed loop on the picture itself, in third-stop steps, toward the auto baseline.
  await sleep(SHUTTER_CAP.settleMs);
  let iso = isoGot ?? isoNew;
  let luma = await probeLuma(video, 2, 60);
  for (let step = 0; step < SHUTTER_CAP.maxSteps; step += 1) {
    if (luma === null) return revert("no_luma_probe", null);
    const ev = Math.log2(Math.max(luma, 1) / Math.max(lumaBefore, 1));
    if (Math.abs(ev) <= SHUTTER_CAP.toleranceEv) break;
    const factor = Math.pow(2, -Math.max(-1 / 3, Math.min(1 / 3, ev)));
    iso = Math.min(isoRange.max, Math.max(isoRange.min, iso * factor));
    if (!(await apply({ iso: Math.round(iso) }))) return revert("servo_apply", luma);
    await sleep(SHUTTER_CAP.stepMs);
    luma = await probeLuma(video, 2, 60);
  }
  if (luma === null) return revert("no_luma_probe", null);
  const finalEv = Math.log2(Math.max(luma, 1) / Math.max(lumaBefore, 1));
  if (Math.abs(finalEv) > SHUTTER_CAP.abortEv) return revert("servo_failed", luma);
  const settled = track.getSettings() as Record<string, unknown>;
  return {
    ...record,
    applied: true,
    reason: "ok",
    exposureAfterS: (num(settled.exposureTime) ?? tGot ?? tNew) * 1e-4,
    isoAfter: num(settled.iso) ?? iso,
    lumaAfter: luma,
  };
}

function errorNameOf(cause: unknown): string | null {
  if (cause instanceof DOMException) return cause.name;
  if (cause && typeof cause === "object" && typeof (cause as { name?: unknown }).name === "string") return (cause as { name: string }).name;
  return null;
}

function mapCameraError(cause: unknown): CameraError {
  switch (errorNameOf(cause)) {
    case "NotAllowedError":
    case "PermissionDeniedError":
    case "SecurityError":
      return "denied";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "no-device";
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      return "resolution-too-low";
    default:
      // NotReadableError / TrackStartError land here: the camera exists but another app holds
      // it. `errorName` carries the detail so the page can say "close the other app".
      return "unknown";
  }
}
