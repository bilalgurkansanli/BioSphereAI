"use client";

/**
 * CaptureStage — the machine, the loop and every platform interrupt.
 *
 * This is the only component that knows about all four builders at once: Builder S's
 * sensors, Builder C's pure coverage/plan/detector functions, Builder V's plain-data
 * visuals, and the contract's state table. Everything below is wiring plus the three things
 * a browser will not do for you: keep the screen awake, stay in portrait, and survive an
 * incoming call.
 *
 * ── WHY THE LOOP LOOKS LIKE THIS ──────────────────────────────────────────────────────
 * The camera preview and the guidance lattice have to stay smooth for a minute straight on
 * a phone that is simultaneously H.264-encoding 1080p at 20 Mbps. So:
 *
 *  - Sensors are read through their `read()` accessors inside ONE rAF loop. Nothing in the
 *    hot path goes through React state, because a `setState` per `deviceorientation` event
 *    is 60 renders a second of a tree that contains a <video>.
 *  - The loop publishes one small "hot" object at ~30 Hz through `hotStore`, NOT through
 *    `useState`. While it was a state hook, every publish re-rendered this entire component
 *    — the overlay, the prompt, the dial, the toast, the clock and every conditional branch
 *    around them — and React reconciliation measured larger than the detectors, the cone
 *    stamp and the 2592-cell report combined. See `hotStore.ts` for the numbers. Each
 *    consumer now subscribes to the slice it draws with its own notion of "changed", so a
 *    publish wakes only what actually moved, and this component re-renders on real state
 *    changes alone.
 *  - `stations` is published only when a station actually fills, so the 72-ring array keeps
 *    a stable identity between fills.
 *  - The <video> lives in a memoised child whose props never change, so React never touches
 *    the element that is painting the camera.
 *  - Expensive work is rate-limited to what it is worth: coverage stamping at
 *    COVERAGE_GRID.stampHz (10 Hz — a 24 deg/s sweep moves 2.4 deg between stamps), the
 *    2592-cell solid-angle report at 2 Hz, frame statistics at 10 Hz, optical flow at 5 Hz.
 *
 * ── WHAT IS DELIBERATELY NOT A STATE ──────────────────────────────────────────────────
 * Per contract decision D7, `correction` and `rotate_portrait` are not states. A warning is
 * `activeWarning` and never costs earned progress; a rotation sets `portraitBlocked` and
 * PAUSES the recorder rather than ending it. Recording is likewise continuous across the
 * three rows (RECORDING_STATES): the pitch transitions between rows are the footage that
 * ties the rows together in the solver, so the recorder is never stopped between them.
 */

import { Check, Download, Loader2, Pause, Play, RotateCcw, Smartphone, TriangleAlert } from "lucide-react";
import { memo, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type RefObject } from "react";
import { DebugHud } from "./DebugHud";
import { CoverageDome } from "./CoverageDome";
import { GuidanceOverlay } from "./GuidanceOverlay";
import { PermissionGate } from "./PermissionGate";
import { ReviewStep } from "./ReviewStep";
import { RowPrompt } from "./RowPrompt";
import { SpeedMeter } from "./SpeedMeter";
import { WarningToast } from "./WarningToast";
import { Button } from "@/components/ui/Button";
import { Switch } from "@/components/ui/Switch";
import { API_BASE, BioSphereError, api } from "@/lib/api";
import { cc, rowCueKey, rowTitleKey, rowVoiceKey } from "@/lib/capture/copy";
import * as Coverage from "@/lib/capture/coverage";
import * as Detectors from "@/lib/capture/detectors";
import * as Plan from "@/lib/capture/plan";
import { UploadResume } from "@/lib/capture/uploadResume";
import { createTelemetry, type TelemetryApiExt } from "@/lib/capture/telemetry";
import { createHotStore, tupleEqual, useHotSelector, type HotStore } from "@/lib/capture/hotStore";
import { initPerfTrace, perfTrace, traceRequested } from "@/lib/capture/perfTrace";
import {
  createGovernor,
  planForTier,
  processIntervalMs,
  publishIntervalMs,
  type Governor,
  type LoadPlan,
} from "@/lib/capture/governor";
import { useCamera } from "@/lib/capture/useCamera";
import { useMotion } from "@/lib/capture/useMotion";
import { readScreenAngleDeg, useOrientation } from "@/lib/capture/useOrientation";
import { useRecorder } from "@/lib/capture/recorder";
import {
  deviceCoachingNote,
  readModelHints,
  recommendedQualityMode,
  resolvePhoneLens,
  type DeviceModelHints,
} from "@/lib/capture/device";
import {
  BONUS_POLE_STATIONS,
  COMPLETENESS,
  COVERAGE_GRID,
  DEFAULT_LENS,
  lensForFrame,
  HAPTICS,
  RECORDING,
  RECORDING_STATES,
  ROW_ID_BY_STATE,
  ROW_IDS,
  ROW_PLAN,
  ROW_STATES,
  STATION_RULES,
  STUCK_AFTER_MS,
  THRESHOLDS,
  UPLOAD_POLICY,
  WARNING_BLOCKS_CAPTURE,
  YAW_SPAN,
  initialCaptureState,
  nextCaptureState,
  type CaptureConfig,
  type CaptureCopyKey,
  type CaptureEventType,
  type CaptureGuardName,
  type CaptureResult,
  type CaptureStageProps,
  type CaptureState,
  type CaptureVerdict,
  type CoverageGrid,
  type CoverageReport,
  type DetectorReadout,
  type FlowSample,
  type FrameQualitySample,
  type GuidedQualityMode,
  type LensGeometry,
  type OrientationSample,
  type PermissionStatus,
  type Quat,
  type RecordedSegment,
  type RowId,
  type SensorFrame,
  type StationId,
  type StationState,
  type StationTarget,
  type TelemetryApi,
  type UploadPlan,
  type Warning,
  type WarningSeverity,
} from "@/lib/capture/types";
import { useLocale } from "@/lib/hooks/useLocale";
import type { Locale } from "@/lib/types";
import { cn, formatBytes } from "@/lib/utils";

// ── Cadences ───────────────────────────────────────────────────────────────────────────

/** Hot-state publish interval; see the loop note in the file header. */
const PUBLISH_INTERVAL_MS = 33;
/**
 * How often the loop does any work at all. Slightly under the publish interval so a tick
 * is never skipped merely because it arrived a millisecond early on a 60 Hz display.
 */
const PROCESS_INTERVAL_MS = 30;
/** The 2592-cell solid-angle report is far too heavy for a per-frame budget. */
const REPORT_INTERVAL_MS = 500;
/** Frame statistics come off a canvas draw; the detectors do not need them faster. */
const QUALITY_INTERVAL_MS = 100;
/** THRESHOLDS.flowDivergence.rateHz. */
const FLOW_INTERVAL_MS = 1000 / THRESHOLDS.flowDivergence.rateHz;
/** Row punctuation: the dome flies in, then the next row's cue takes over. */
/**
 * How long the row-transition card stays up.
 *
 * 1200 ms was chosen when the card was a wordless dome — long enough to register a shape.
 * It now names the next row and the move that starts it, and a sentence the user cannot
 * finish reading is worse than no sentence. `DURATION_BUDGET.rowTransitionSeconds` is 2, so
 * this still fits inside the plan's own allowance for the transition, and the recorder keeps
 * rolling throughout: the tilt between rows is the footage that ties them together.
 */
const PUNCTUATION_MS = 1800;
/** After a rotation the phone is still swinging; give auto-exposure and the wrist a beat. */
const ROTATION_SETTLE_MS = 1500;
/** Sustained walking is the only condition allowed to interrupt a capture. */
const WALKING_PAUSE_MS = 6000;
/** `sweep_check` holds for this long so the verdict does not appear out of nowhere. */
const SWEEP_CHECK_MS = 2000;
/** A still every 45 deg of the level row, for the review strip. */
const THUMBNAIL_EVERY_DEG = 45;
const THUMBNAIL_MAX = 8;
/** How far the phone may be rolled and still count towards a rolled pole frame. */
const POLE_ROLL_TOLERANCE_DEG = 35;
/**
 * How many MediaRecorder failures a capture may absorb before the flow stops re-opening
 * segments and hands the user the paused sheet instead. Three is enough to ride out the
 * transient encoder errors a backgrounded tab or a thermal throttle produces, and few
 * enough that a genuinely dead recorder cannot loop behind a screen that looks healthy.
 */
const MAX_RECORDER_FAULTS = 3;
/**
 * No orientation sample within this long of the stage going live means there is no usable
 * compass, whatever the permission said — a granted-but-absent sensor is the same
 * experience as a denied one, and both must land in RAIL MODE rather than on a dead screen.
 */
const RAIL_FALLBACK_MS = 4000;
/** RAIL MODE has no yaw, so review stills are taken on a clock instead of every 45 deg. */
const RAIL_THUMBNAIL_EVERY_MS = 2500;
/**
 * How long a recording state will wait for the camera to come back before it stops
 * pretending. An incoming call costs a full `getUserMedia` reopen — two phases, a probe and
 * a constrained ladder — which lands well inside this on every device measured; past it the
 * honest move is the paused sheet, where everything sealed so far is still in hand.
 */
const CAMERA_REOPEN_GRACE_MS = 8000;
/** Gap between saving one recorded segment and the next; see `download`. */
const DOWNLOAD_STAGGER_MS = 400;

const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

/** States in which the camera, the sensors and the rAF loop are all live. */
const LIVE_STATES = new Set<CaptureState>([
  "framing",
  "row_level",
  "row_up",
  "row_down",
  "touch_up",
  "paused",
  "bonus_poles",
  "sweep_check",
]);

/** The gate decision itself is pure and lives in `plan.ts`, where it is unit-tested. */
type FramingGate = Plan.FramingGate;

const FRAMING_COPY: Readonly<Record<Exclude<FramingGate, null>, CaptureCopyKey>> = {
  hold: "capture.frame.hold",
  level: "capture.frame.level",
  dark: "capture.frame.dark",
  bright: "capture.frame.bright",
  fps: "capture.frame.slowfps",
};

// ── Loop-local mutable state ───────────────────────────────────────────────────────────

interface HotState {
  q: Quat;
  headingDeg: number;
  rollDeg: number;
  target: StationTarget | null;
  readout: DetectorReadout | null;
  warning: Warning | null;
  severity: WarningSeverity | null;
  rowFractions: Record<RowId, number>;
  stuck: boolean;
  /**
   * TOTAL recorded time — every sealed segment plus the live one — not the live recorder's
   * own clock. The live clock resets at every segment boundary, and a capture is now
   * routinely several segments (the 120 s per-segment ceiling, an incoming call), so the
   * recorder's `elapsedMs` alone would show a user two minutes into their sweep a timer
   * reading four seconds.
   */
  recordedMs: number;
  /** RAIL MODE only: whole seconds left in this row on the timed guide. */
  railSecondsLeft: number | null;
}

const INITIAL_HOT: HotState = {
  q: IDENTITY_QUAT,
  headingDeg: 0,
  rollDeg: 0,
  target: null,
  readout: null,
  warning: null,
  severity: null,
  rowFractions: { level: 0, up: 0, down: 0 },
  stuck: false,
  recordedMs: 0,
  railSecondsLeft: null,
};

/** Per-row sweep bookkeeping only P can do: it needs the window in which a row was ACTIVE. */
interface RowTrack {
  lastYawDeg: number | null;
  spanDeg: number;
  pitchMinDeg: number;
  pitchMaxDeg: number;
  seen: boolean;
}

const emptyRowTrack = (): RowTrack => ({ lastYawDeg: null, spanDeg: 0, pitchMinDeg: 0, pitchMaxDeg: 0, seen: false });

interface LoopState {
  /** Gate for the whole tick body; see the throttle at the top of the loop. */
  lastProcess: number;
  /** Sticky target id, so the reticle cannot be flipped by sensor noise. */
  lastTargetId?: StationId | null;
  debugFps?: number;
  debugFrames?: number;
  debugSince?: number;
  debugWorst?: number;
  lastPublish: number;
  lastStamp: number;
  lastReport: number;
  lastQuality: number;
  lastFlow: number;
  quality: FrameQualitySample | null;
  flow: FlowSample | null;
  steadySince: number | null;
  gatesSince: number | null;
  framingEnteredAt: number;
  brightestYaw: number | null;
  brightestLuma: number;
  walkingSince: number | null;
  lastFillAt: number;
  lastWarningCode: string | null;
  /** When the first ring of the current up/down row was reached. See `Plan.armRow`. */
  rowArmSince: number | null;
  thumbnailYaw: number | null;
  thumbnailAt: number;
  thumbnailBusy: boolean;
  /** RAIL MODE: which row the timed clock belongs to, and how much of it has been RECORDED. */
  railRow: CaptureState | null;
  railElapsedMs: number;
  railLastTick: number;
  /** Cells the rail lattice has painted, so it only republishes when the picture changes. */
  railFilled: number;
}

const emptyLoopState = (): LoopState => ({
  lastProcess: 0,
  lastPublish: 0,
  lastStamp: 0,
  lastReport: 0,
  lastQuality: 0,
  lastFlow: 0,
  quality: null,
  flow: null,
  steadySince: null,
  gatesSince: null,
  framingEnteredAt: 0,
  brightestYaw: null,
  brightestLuma: 0,
  walkingSince: null,
  lastFillAt: 0,
  lastWarningCode: null,
  rowArmSince: null,
  thumbnailYaw: null,
  thumbnailAt: 0,
  thumbnailBusy: false,
  railRow: null,
  railElapsedMs: 0,
  railLastTick: 0,
  railFilled: -1,
});

/**
 * Largest slice of RAIL MODE row time a single tick may bank.
 *
 * The same argument as `MAX_DWELL_STEP_MS` in the plan, for the same reason: the rail clock
 * measures RECORDED time, and a rotation, a backgrounded tab or a dropped call would
 * otherwise hand the first tick back a gap of arbitrary size — completing a row out of
 * seconds during which the camera recorded nothing at all.
 */
const RAIL_MAX_STEP_MS = 250;

// =======================================================================================
// Component
// =======================================================================================

export function CaptureStage({
  propertyId,
  defaultSceneName,
  defaultQualityMode = "balanced",
  onUseExistingVideo,
  onHandoff,
}: CaptureStageProps) {
  const { locale } = useLocale();
  const reducedMotion = useReducedMotion();

  const [qualityMode, setQualityMode] = useState<GuidedQualityMode>(defaultQualityMode);
  /** Set the moment the user picks a preset, so the device suggestion never overrides them. */
  const qualityTouchedRef = useRef(false);
  const [sceneName, setSceneName] = useState(defaultSceneName ?? "");
  const [voiceGuide, setVoiceGuide] = useState(false);
  const [sounds, setSounds] = useState(true);

  const config = useMemo<CaptureConfig>(
    () => ({
      lens: DEFAULT_LENS,
      rows: ROW_PLAN,
      completeness: COMPLETENESS,
      // The shipped constants, not copies of them: a literal here would widen to `number`
      // and stop matching `typeof YAW_SPAN`, and worse, would let the two drift.
      yawSpan: YAW_SPAN,
      station: STATION_RULES,
      recording: RECORDING,
      thresholds: THRESHOLDS,
      qualityMode,
      captureKind: "video",
      voiceGuide,
      sounds,
      haptics: true,
      locale,
    }),
    [qualityMode, voiceGuide, sounds, locale],
  );

  // ── Sensors, camera, recorder ───────────────────────────────────────────────────────

  /**
   * Declared BEFORE the sensor hooks on purpose. React runs a component's effect cleanups
   * in the order the effects were created, so this flips false ahead of `useRecorder`'s own
   * dispose — which seals one last segment on unmount, and whose `onSegment` would
   * otherwise open a fresh recorder on a stream the teardown is in the middle of stopping.
   */
  const mountedRef = useRef(true);
  useEffect(
    () => () => {
      mountedRef.current = false;
    },
    [],
  );

  // `stateHz: 0` — no React mirror at all. This component reads both sensors through
  // `orientationRef.current.read()` and `motionRef.current.read()` inside its own loop and
  // never touches the hooks' `sample`, `reference`, `headingAccuracyDeg` or source fields.
  // At the default 4 Hz the two hooks were re-rendering this whole 2800-line component eight
  // times a second, every second of every capture, to publish values nothing on screen was
  // showing — and each publish is a guaranteed change, because the trackers allocate a fresh
  // sample object per push, so React's same-value bail-out never fired.
  const orientation = useOrientation({ stateHz: 0 });
  const motion = useMotion({ stateHz: 0 });
  /**
   * THE APP MEASURES ITSELF AND SHEDS WORK UNTIL IT IS SMOOTH.
   *
   * Two rounds of targeted optimisation shipped against a phone nobody could profile, and
   * both missed. This is the answer that does not require knowing the cause: the governor
   * watches the real animation-frame interval and turns the loop's own cadences down a tier
   * at a time until the frames land on schedule. Whatever is expensive on a given device —
   * a codec, a compositor, a browser version nobody has — it gets turned down.
   *
   * Lazily built rather than `useRef(createGovernor())`, which would allocate its ring
   * buffers on every render. The local binding is what the tick closes over: `.current`
   * does not stay narrowed across a function boundary.
   */
  const governorRef = useRef<Governor | null>(null);
  const governor: Governor = (governorRef.current ??= createGovernor());
  const planRef = useRef<LoadPlan>(planForTier(0));
  /** Read by `useCamera`'s frame loop, which must not be rebuilt when the tier changes. */
  const loadPlan = useCallback(() => planRef.current, []);

  const camera = useCamera({
    // An incoming call ends the track on iOS. `useCamera` reopens the stream from here; the
    // recorder seals its partial segment on the same event and the effect below opens the
    // next one, so the call costs a join in the footage rather than the capture.
    onTrackEnded: () => {
      if (!mountedRef.current) return;
      void cameraRef.current.open();
    },
    // The live smear prediction carried in every FrameQualitySample. Without it
    // `quality.smearPx` is a constant 0, so the sidecar reports zero blur evidence for a
    // capture the phone's own detector graded `too_blurry` — the phone and the server
    // reading two different stories off the same minute.
    omegaDegPerS: () => readoutRef.current?.omegaDegPerS ?? 0,
    // A stable getter, so a tier change never rebuilds the video-frame callback.
    loadPlan,
  });
  const cameraRef = useLatest(camera);

  const recorder = useRecorder({
    stream: camera.stream,
    // Decoder truth for the sealed segment. Unwired this returns undefined, every
    // `RecordedSegment` carries 0x0, and the sidecar ships the one field the engine is
    // meant to reject an inverted-FOV file on as a zero.
    dimensions: () => cameraRef.current.size ?? null,
    // THE delivery path for a finished segment. `finishSeal` hands the segment to this
    // callback and to nothing else, so an unwired `onSegment` is a capture on the floor.
    onSegment: (segment) => segmentSealedRef.current(segment),
    onChunk: () => chunkArrivedRef.current(),
    onError: () => {
      recorderFaultsRef.current += 1;
    },
    // A watchdog restart is a new encoder with a new first frame; the track's media
    // clock has to follow it or the whole segment's orientation samples are offset.
    onRestart: () => telemetryRef.current?.restartSegment(),
  });

  const orientationRef = useLatest(orientation);
  const motionRef = useLatest(motion);
  const recorderRef = useLatest(recorder);
  const configRef = useLatest(config);
  const localeRef = useLatest(locale);

/**
 * The rectangle the user can actually see, which on iOS is not the one `fixed inset-0` gets.
 *
 * `position: fixed` resolves against the LAYOUT viewport. Safari keeps that at the size the
 * page had with its toolbars hidden, and it does not shrink when the user pinches. So the
 * stage was laid out wider and taller than the glass: the recording clock ran off the left,
 * the finish button off the right, and the bottom row sat behind the toolbar. `visualViewport`
 * is the only thing that reports what is genuinely on screen, offset included, so the stage
 * is positioned from that and follows every toolbar slide and pinch.
 *
 * Returns null where the API is missing, and the `.stage-viewport` class then carries the
 * `100dvh` fallback on its own.
 */
function useVisibleRect(): { width: number; height: number; left: number; top: number } | null {
  const [rect, setRect] = useState<{ width: number; height: number; left: number; top: number } | null>(null);
  useEffect(() => {
    const vv = typeof window === "undefined" ? undefined : window.visualViewport;
    if (!vv) return;
    let frame = 0;
    const read = () => {
      cancelAnimationFrame(frame);
      // Coalesced: iOS fires resize and scroll together while the toolbar animates, and
      // laying the stage out on every one of them fights the compositor mid-capture.
      frame = requestAnimationFrame(() =>
        setRect({ width: vv.width, height: vv.height, left: vv.offsetLeft, top: vv.offsetTop }),
      );
    };
    read();
    vv.addEventListener("resize", read);
    vv.addEventListener("scroll", read);
    return () => {
      cancelAnimationFrame(frame);
      vv.removeEventListener("resize", read);
      vv.removeEventListener("scroll", read);
    };
  }, []);
  return rect;
}

  /** Lens geometry for the frame actually delivered.
   *
   *  This used to relabel the source as `track-settings` while keeping DEFAULT_LENS's 43x69
   *  — claiming to have measured what it had in fact assumed. 69 deg is the vfov of a 9:16
   *  frame; the moment the camera handed back 4:3 the number was 8 deg wrong and every
   *  threshold derived from it went with it. `lensForFrame` keeps the assumed hfov, which
   *  the browser cannot report, and takes the shape from the track, which it can. */
  const lens = useMemo<LensGeometry>(
    () => (camera.size ? lensForFrame(camera.size.width, camera.size.height) : DEFAULT_LENS),
    [camera.size],
  );
  // Which phone this is. Read once, off the critical path: the engine looks the model up
  // in its own lens table and bounds the focal search with it (see capture/device.ts).
  // A capture never waits on this; if it resolves late the sidecar simply carries less.
  const modelHintsRef = useRef<DeviceModelHints | null>(null);
  const [phoneLens, setPhoneLens] = useState<ReturnType<typeof resolvePhoneLens> | null>(null);
  useEffect(() => {
    let live = true;
    void readModelHints().then((hints) => {
      if (!live) return;
      modelHintsRef.current = hints;
      telemetryRef.current?.setModelHints(hints);
      const resolved = resolvePhoneLens(hints);
      setPhoneLens(resolved);
      // The preset follows the stabiliser, not the megapixels: STUDIO's extra resolution and
      // seam search are only worth their three-times cost if the frames are sharp enough to
      // hold them, and without OIS an indoor shutter smears every one of them. Only ever a
      // DEFAULT — the picker is still there and the user's own choice always wins, which is
      // why this only fires while nothing has been chosen yet.
      const suggested = recommendedQualityMode(resolved.lens);
      if (suggested !== defaultQualityMode && !qualityTouchedRef.current) setQualityMode(suggested);
    });
    return () => {
      live = false;
    };
  }, [defaultQualityMode]);

  const visible = useVisibleRect();
  const lensRef = useLatest(lens);

  // ── Machine ─────────────────────────────────────────────────────────────────────────

  // `hasDraft` is pinned false; see the guard block below for why.
  const [state, setState] = useState<CaptureState>(() => initialCaptureState(false));
  const stateRef = useRef(state);
  const previousStateRef = useRef<CaptureState>(state);
  const resumeTargetRef = useRef<CaptureState | null>(null);

  const [permissions, setPermissions] = useState<PermissionStatus>({
    camera: "unknown",
    motion: "unknown",
    inAppBrowser: false,
    wakeLock: false,
    portrait: true,
  });
  const [railMode, setRailMode] = useState(false);
  const [portraitBlocked, setPortraitBlocked] = useState(false);
  const portraitBlockedRef = useLatest(portraitBlocked);

  // ── Live model ──────────────────────────────────────────────────────────────────────

  const gridRef = useRef<CoverageGrid | null>(null);
  const stationsRef = useRef<StationState[]>([]);
  /**
   * The same station objects, indexed by row, in ARRAYS WHOSE IDENTITY NEVER CHANGES.
   *
   * A guided capture has to guide: while the level row is being swept, the up and down rows
   * are locked — they cannot fill and the bright ring cannot point at them. That means
   * handing `updateStations` and `nextTarget` one row's stations rather than all seventy-two.
   *
   * It must not be a `.filter()` at the call site. `updateStations` keys its dwell clock on
   * the array's identity through a WeakMap, so a fresh array every tick would make `dt` zero
   * every tick and NO station would ever fill. These are built once per session, hold the
   * same objects by reference, and are mutated in place exactly as the full array is.
   */
  const rowStationsRef = useRef<Record<RowId, StationState[]>>({ level: [], up: [], down: [] });
  const bonusRef = useRef<StationState[]>([]);
  // The extended detector, not the bare `DetectorInstance`: `setExternalFlags` carries the
  // two facts `SensorFrame` cannot (landscape, lens change) and `isStalled` is the stall
  // clock that drives `RowPromptProps.stuck`.
  const detectorRef = useRef<Detectors.CaptureDetector | null>(null);
  // The extended telemetry too: `setRecordingContext`, `setCoveredMask` and `noteLensChange`
  // are the three things only the page can know, and none of them are on the bare
  // `TelemetryApi` the contract declares.
  const telemetryRef = useRef<TelemetryApiExt | null>(null);
  const rowTrackRef = useRef<Record<RowId, RowTrack>>({
    level: emptyRowTrack(),
    up: emptyRowTrack(),
    down: emptyRowTrack(),
  });
  const segmentsRef = useRef<RecordedSegment[]>([]);
  const thumbnailsRef = useRef<Blob[]>([]);
  const touchUpIdsRef = useRef<StationId[] | null>(null);
  const loopRef = useRef<LoopState>(emptyLoopState());
  const recordingStartedRef = useRef(false);
  /** The drawing-only lattice RAIL MODE fills from its clock; never fed to coverage. */
  const railStationsRef = useRef<StationState[]>([]);
  /** Latest detector readout, for `useCamera`'s smear prediction. */
  const readoutRef = useRef<DetectorReadout | null>(null);
  /** How many times MediaRecorder has failed in this session. Bounds the restart loop. */
  const recorderFaultsRef = useRef(0);
  /**
   * `screen.orientation.angle` as it was WHEN RECORDING STARTED, plus any non-zero angle
   * seen while it ran. Sampling this at `sweep_check` — as the field's first author did —
   * reads it after the phone is back in portrait and therefore always reports 0, which is
   * the one value that cannot be evidence of the inverted-FOV file it exists to catch.
   */
  const recordedScreenAngleRef = useRef(0);
  /**
   * Late-bound recorder callbacks. The recorder is constructed above `beginSegment`, so its
   * hooks reach the handlers through these; both are pointed at the real functions from an
   * effect, never written during render.
   */
  const segmentSealedRef = useRef<(segment: RecordedSegment) => void>(() => undefined);
  const chunkArrivedRef = useRef<() => void>(() => undefined);

  /**
   * The 30 Hz channel. NOT a ref read during render — a ref read during render is a value
   * React was never told changed, which is exactly how a lattice ends up one row behind the
   * row the user is being coached through. This is a subscribable store: every consumer is
   * told, and each one decides for itself what counts as news.
   */
  // A lazy `useState` initialiser rather than a ref, because the store is created exactly
  // once and then never replaced — and reading a ref during render is the thing this store
  // exists to make unnecessary.
  const [hotStore] = useState<HotStore<HotState>>(() => createHotStore(INITIAL_HOT));

  const [stationsView, setStationsView] = useState<StationState[]>([]);
  const [touchUpIds, setTouchUpIds] = useState<StationId[] | null>(null);
  const [coveredView, setCoveredView] = useState<Uint8Array | null>(null);
  const [recordedBytes, setRecordedBytes] = useState(0);
  const [coverageReport, setCoverageReport] = useState<CoverageReport | null>(null);
  const [result, setResult] = useState<CaptureResult | null>(null);
  const [framingGate, setFramingGate] = useState<FramingGate>("hold");
  const [countdown, setCountdown] = useState<number | null>(null);
  // Separate from `countdown`, which belongs to the pre-recording framing gate and is shown
  // inside a card that covers the viewfinder. This one has to sit OVER a live sweep, because
  // what it is asking for is that the user keeps aiming at the ring underneath it.
  const [rowArmCountdown, setRowArmCountdown] = useState<number | null>(null);
  const rowArmCountdownRef = useRef<number | null>(null);
  const [canForceStart, setCanForceStart] = useState(false);
  const [preflightFailed, setPreflightFailed] = useState(false);
  /** `?debug=1` turns on the on-device HUD; a ref so the loop never re-renders for it. */
  const debugRef = useRef(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    debugRef.current = new URLSearchParams(window.location.search).get("debug") === "1";
  }, []);
  const [punctuationRow, setPunctuationRow] = useState<RowId | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  /** True once `MAX_CAPTURE_MS` of footage exists: the capture must end, not merely pause. */
  const [budgetExhausted, setBudgetExhausted] = useState(false);

  const coverageRef = useLatest(coverageReport);
  const resultRef = useLatest(result);
  const sceneNameRef = useLatest(sceneName);
  const railModeRef = useLatest(railMode);

  // ── Guards and dispatch ─────────────────────────────────────────────────────────────

  const guardsRef = useRef<Partial<Record<CaptureGuardName, boolean>>>({});
  guardsRef.current = {
    // Pinned off, deliberately. `CaptureDraft.chunks` needs the raw timesliced Blob chunks,
    // and `RecorderApi` exposes only `chunkCount` — so P cannot build a resumable draft
    // through the contract as written. Offering a "resume" that silently drops the footage
    // would be worse than not offering one, so this is reported upstream rather than faked.
    hasDraft: false,
    motionGranted: permissions.motion === "granted" || permissions.motion === "not-required",
    // RAIL MODE has no station lattice to aim at and no gap list worth trusting — the
    // report's "gaps" there are simply everywhere the phone could not measure. Offering
    // "Eksikleri tamamla" would hand the user a touch-up pass with nothing on screen.
    //
    // OFF RAIL THE SAME TRAP IS ONE STEP FURTHER IN. `gapStations` can only name a reticle
    // that is still EMPTY, so a report whose every station is filled hands the touch-up pass
    // an empty target set: no ring appears, `allTargetsFilled` returns false for ever, and
    // ROW_COMPLETE can never fire. The user taps "fill the gaps", the recorder opens a fresh
    // segment, and they are left looking at a live camera with nothing to aim at. That state
    // is reachable — a couple of degrees of row bias fills all 72 stations while
    // `coveredFraction` still refuses — so the button asks the lattice, not just the report.
    hasGaps:
      !railMode &&
      stationsRef.current.some((station) => !station.filled) &&
      ((coverageReport?.gaps.length ?? 0) > 0 || coverageReport?.complete === false),
    coverageComplete: !railMode && coverageReport?.complete === true,
  };

  const send = useCallback((event: CaptureEventType) => {
    const current = stateRef.current;
    const next = nextCaptureState(current, event, guardsRef.current, resumeTargetRef.current);
    // An illegal event is ignored — never thrown, never toasted. The loop fires events from
    // sensor data, and a stray FINISH landing mid-transition must not end a capture.
    if (!next || next === current) return;
    // Any route into `paused` is what `@resume` has to come back to, whether the user hit
    // Bitir, backgrounded the app, or walked for six seconds.
    if (next === "paused") resumeTargetRef.current = current;
    previousStateRef.current = current;
    stateRef.current = next;
    setState(next);
  }, []);

  /**
   * The row currently being coached. In `touch_up` and `bonus_poles` it follows the bright
   * ring, which the loop computes from the previous frame's target — one publish of lag on
   * entry, converging immediately afterwards.
   *
   * Selected rather than read off a published object, so the ROW is what this component
   * subscribes to and not the target: the target is a fresh object thirty times a second,
   * the row it belongs to changes a handful of times in a whole capture.
   */
  const targetRow = useHotSelector(hotStore, selectTargetRow);
  const activeRow: RowId | null = useMemo(() => {
    if ((ROW_STATES as readonly string[]).includes(state)) {
      return ROW_ID_BY_STATE[state as (typeof ROW_STATES)[number]];
    }
    if (state === "touch_up" || state === "bonus_poles") return targetRow;
    return null;
  }, [state, targetRow]);

  // ── Feedback ────────────────────────────────────────────────────────────────────────

  const feedback = useFeedback(sounds);
  const speak = useCallback(
    (key: CaptureCopyKey) => {
      if (!configRef.current.voiceGuide) return;
      speakLine(cc(key, localeRef.current), localeRef.current);
    },
    [configRef, localeRef],
  );

  // ── Permissions ─────────────────────────────────────────────────────────────────────

  // In-app webviews (Instagram, Facebook, LinkedIn, Line, WeChat) cannot reach getUserMedia
  // at all, so the interstitial has to come BEFORE anything is requested: a denied prompt in
  // those browsers cannot be re-raised without a reload. Read as an external store so the
  // very first client render is already correct — a frame of the permission ask, shown to
  // someone inside Instagram, is a frame that invites a tap that can never succeed.
  const inAppBrowser = useSyncExternalStore(subscribeNothing, readInAppBrowser, () => false);
  const permissionStatus: PermissionStatus = useMemo(
    () => ({ ...permissions, inAppBrowser }),
    [permissions, inAppBrowser],
  );

  const requestPermissions = useCallback(() => {
    // SYNCHRONOUS, and in this order. iOS consumes the click's transient activation at the
    // first `await`, and `DeviceOrientationEvent.requestPermission()` throws NotAllowedError
    // without one. Both sensor requests are therefore STARTED before anything is awaited;
    // the camera, which needs no activation, is opened in the continuation.
    const motionRequest = motionRef.current.request();
    const orientationRequest = orientationRef.current.request();

    void (async () => {
      const [motionResult, orientationResult] = await Promise.all([motionRequest, orientationRequest]);
      const oriented = orientationResult === "granted" || orientationResult === "not-required";
      const inertial = motionResult === "granted" || motionResult === "not-required";

      setPermissions((prev) => ({
        ...prev,
        motion: oriented ? (orientationResult === "not-required" ? "not-required" : "granted") : "denied",
      }));
      // RAIL MODE. Losing orientation is not a reason to stop: every camera-derived meter
      // still works and the lattice flattens to a horizon strip advanced by optical flow.
      setRailMode(!oriented);
      // A denied motion stream costs the inertial walking gate; the quadrant-flow detector
      // takes over and telemetry records `motionGateRan: false`. Absence of evidence of
      // walking must never read as evidence of standing still.
      if (!inertial && oriented) setPermissions((prev) => ({ ...prev, motion: "denied" }));

      await cameraRef.current.open();
    })();
  }, [cameraRef, motionRef, orientationRef]);

  /**
   * "Tekrar dene" from the camera-denied wall.
   *
   * `PermissionGate` chooses its face from `status` alone — a denied camera renders the wall
   * whichever of the two permission states the machine is in. So returning the machine to
   * `permissions` is NOT enough on its own: `permissions.camera` is still "denied", the wall
   * renders again, and the retry button is a dead end with no route back to the ask.
   *
   * The copy on that wall reads "turn the permission on in the address bar, then try again",
   * so the honest behaviour is to actually re-attempt rather than just re-render the ask.
   * `send` is synchronous (it writes `stateRef` before returning) and neither call awaits
   * before `requestPermission()`, so iOS's transient activation survives into the retry.
   *
   * The status is NOT optimistically cleared here: `camera.open()` resets its own error
   * first, so a failed retry re-publishes "denied" as a fresh value and the entry effect
   * below fires again. Clearing it would only buy a flash of the ask before the same wall.
   */
  const retryPermissions = useCallback(() => {
    send("RETRY_PERMISSIONS");
    requestPermissions();
  }, [requestPermissions, send]);

  // The camera's own state decides the transition: reading `camera.error` inside the async
  // continuation above would race React's render.
  useEffect(() => {
    if (stateRef.current !== "permissions") return;
    if (camera.error) {
      setPermissions((prev) => ({ ...prev, camera: camera.error === "denied" ? "denied" : "unavailable" }));
      send("PERMISSIONS_DENIED_CAMERA");
    } else if (camera.stream) {
      setPermissions((prev) => ({ ...prev, camera: "granted" }));
      // Both events land on `stance`; the distinction is kept so the event log says which
      // path was taken.
      send(permissions.motion === "denied" ? "PERMISSIONS_DENIED_MOTION" : "PERMISSIONS_GRANTED");
    }
  }, [camera.error, camera.stream, permissions.motion, send]);

  // ── Session lifecycle ───────────────────────────────────────────────────────────────

  const resetSession = useCallback(() => {
    gridRef.current = Coverage.createGrid();
    stationsRef.current = Plan.buildStations(configRef.current);
    rowStationsRef.current = {
      level: stationsRef.current.filter((station) => station.row === "level"),
      up: stationsRef.current.filter((station) => station.row === "up"),
      down: stationsRef.current.filter((station) => station.row === "down"),
    };
    bonusRef.current = BONUS_POLE_STATIONS.map((bonus) => {
      const [row, index] = bonus.id.split(":");
      return {
        id: bonus.id,
        row: row as RowId,
        index: Number(index),
        // Yaw is degenerate at a pole; these stations are reached by pitch alone.
        yawDeg: 0,
        pitchDeg: bonus.pitchDeg,
        filled: false,
        filledAt: null,
        dwellMs: 0,
      };
    });
    detectorRef.current = Detectors.create(configRef.current);
    // The camera is passed as a GETTER: this runs at `stance`, before the two-phase lens
    // pick has necessarily settled, and the label and track settings the sidecar needs are
    // the ones true at `build()` time.
    // The LIVE lens, not the config's default: `lensForFrame` takes the frame shape from
    // the track, so on the shipped 4:3 capture the sidecar's vfov is the real 55 deg
    // rather than the 69 deg of a 9:16 frame nobody recorded.
    telemetryRef.current = createTelemetry({ lens: lensRef.current }, () => cameraRef.current);
    telemetryRef.current.setModelHints(modelHintsRef.current);
    rowTrackRef.current = { level: emptyRowTrack(), up: emptyRowTrack(), down: emptyRowTrack() };
    segmentsRef.current = [];
    thumbnailsRef.current = [];
    touchUpIdsRef.current = null;
    loopRef.current = emptyLoopState();
    // A second lattice, drawn only. RAIL MODE fills this one from its clock so the strip in
    // `GuidanceOverlay` moves; `stationsRef` stays honest and empty, because nothing
    // measured where the camera pointed.
    railStationsRef.current = Plan.buildStations(configRef.current);
    recorderFaultsRef.current = 0;
    recordedScreenAngleRef.current = 0;
    readoutRef.current = null;
    setTouchUpIds(null);
    setCoveredView(null);
    setRecordedBytes(0);
    setBudgetExhausted(false);
    recordingStartedRef.current = false;
    setStationsView(stationsRef.current.slice());
    setCoverageReport(null);
    setResult(null);
    hotStore.publish(INITIAL_HOT);
  }, [cameraRef, configRef, hotStore]);

  /**
   * Opens the next recorder segment and tells telemetry where it sits in the media.
   *
   * A scene is legitimately several segments: an incoming call ends the track on iOS, and
   * `touch_up` and `bonus_poles` each add a pass after the main sweep was sealed. Every
   * segment is anchored by its offset so the 10 Hz orientation track stays aligned to the
   * engine's own extracted frames across the joins.
   */
  const beginSegment = useCallback(async () => {
    const priorMs = segmentsRef.current.reduce((total, segment) => total + segment.durationMs, 0);
    telemetryRef.current?.markSegment(segmentsRef.current.length, priorMs / 1000);
    recordingStartedRef.current = true;
    // The angle the file is actually being written at. Read here, not at `sweep_check`,
    // where the phone is back in portrait and the answer is always 0.
    const angle = orientationRef.current.read()?.screenAngleDeg ?? readScreenAngleDeg();
    if (angle !== 0 || recordedScreenAngleRef.current === 0) recordedScreenAngleRef.current = angle;
    await recorderRef.current.start();
  }, [orientationRef, recorderRef]);

  /**
   * Is there a stream that could actually be recorded right now?
   *
   * `camera.stream` being non-null is not the same question: an incoming call ENDS the
   * tracks and leaves the object in place, so a recorder started on it produces nothing and
   * throws. This is the difference between "wait for the camera to come back" and "the
   * recorder is broken".
   */
  const canRecord = useCallback(() => {
    const stream = cameraRef.current.stream;
    return stream !== null && stream.getVideoTracks().some((track) => track.readyState === "live");
  }, [cameraRef]);

  /**
   * A segment has been sealed — by the machine, or by the recorder acting on its own.
   *
   * THIS IS THE ONLY PLACE A SEGMENT IS COLLECTED. `finishSeal` delivers it here and to the
   * promise its internal caller is awaiting, and nowhere else; with this unwired the 120 s
   * per-segment ceiling and a dropped track both dropped the entire recording on the floor
   * while the screen carried on showing a healthy capture.
   *
   * A seal the RECORDER initiated is not the end of the capture. The machine is still in a
   * recording state, the user is still turning, so the next segment opens immediately and
   * the join costs a frame instead of the minute. The machine's own seals — finishing a
   * capture, a retake — are left alone: they are already going somewhere else.
   */
  const handleSegment = useCallback(
    (segment: RecordedSegment) => {
      segmentsRef.current.push(segment);
      setRecordedBytes(segmentsRef.current.reduce((total, s) => total + s.blob.size, 0));

      const selfInitiated =
        segment.endedBy === "max_duration" || segment.endedBy === "interrupted" || segment.endedBy === "error";
      if (!selfInitiated || !mountedRef.current) return;
      if (!RECORDING_STATES.includes(stateRef.current) || portraitBlockedRef.current) return;
      // A recorder that cannot stay alive must not be restarted forever behind a screen that
      // looks fine. Hand the user the paused sheet instead: everything sealed so far is in
      // `segmentsRef` and finishing from there uploads all of it.
      if (recorderFaultsRef.current > MAX_RECORDER_FAULTS) {
        send("PAUSE");
        return;
      }
      // An incoming call ends the TRACK, so this seal usually arrives while the stream is
      // already dead. Starting on it would only burn the fault budget; `useCamera` is
      // reopening in parallel and the continuity effect below picks it up the moment a live
      // stream exists again.
      if (!canRecord()) return;
      void beginSegment();
    },
    [beginSegment, canRecord, portraitBlockedRef, send],
  );

  /** Live byte count, the one thing chunk-level access buys without a draft store. */
  const handleChunk = useCallback(() => {
    const sealed = segmentsRef.current.reduce((total, s) => total + s.blob.size, 0);
    const live = recorderRef.current.chunks().reduce((total, chunk) => total + chunk.size, 0);
    setRecordedBytes(sealed + live);
  }, [recorderRef]);

  // Pointed at the current closures after every render, never during one. The recorder is
  // constructed above these definitions and reaches them through the two refs.
  useEffect(() => {
    segmentSealedRef.current = handleSegment;
    chunkArrivedRef.current = handleChunk;
  });

  const buildReport = useCallback((): CoverageReport | null => {
    const grid = gridRef.current;
    if (!grid) return null;

    // Per-row yaw span and elevation extremes are only knowable from the window in which a
    // row was ACTIVE — that window is P's bookkeeping, so it is handed to C rather than
    // re-derived from the grid, where a row's own contribution is no longer separable.
    const tracked: Partial<Record<RowId, Coverage.TrackedRowStats>> = {};
    for (const row of ROW_IDS) {
      const track = rowTrackRef.current[row];
      tracked[row] = {
        yawSpanDeg: track.spanDeg,
        // `undefined` rather than 0 when a row was never entered: 0 would read as "the
        // camera looked at the horizon in that row", which is a claim we cannot make.
        pitchMinDeg: track.seen ? track.pitchMinDeg : undefined,
        pitchMaxDeg: track.seen ? track.pitchMaxDeg : undefined,
      };
    }

    return Coverage.report(grid, stationsRef.current, lensRef.current, tracked);
  }, [lensRef]);

  // ── The loop ────────────────────────────────────────────────────────────────────────

  const isLive = LIVE_STATES.has(state);

  /**
   * Tell the stylesheet the camera is live.
   *
   * Two things below `[data-live-camera]` in globals.css cost real frames only while a
   * video is playing underneath them — the frosted panels' `backdrop-filter` and the
   * full-viewport grain layer — and cost nothing anywhere else. Both are global (one is
   * `body::before`), so the switch has to be on `body` rather than on a React subtree.
   *
   * Removed on cleanup rather than toggled, so a capture abandoned by navigation cannot
   * leave the rest of the app painted flat.
   */
  /**
   * THE PHONE REPORTS WHAT ACTUALLY HAPPENED, so nobody has to guess again.
   *
   * Three rounds of performance work have shipped against a device nobody can profile, and
   * every theory behind them was later refuted — the readback flag, the dev build, the
   * software-encoder story. Each was plausible and none was measured on the phone. This
   * closes that hole: frame intervals, long tasks, the stage timings the loop reports, and
   * the two facts that settle the encoder argument outright — the mimeType the recorder
   * actually landed on, and the geometry the camera actually gave us.
   *
   * The trace posts itself to the engine, so the user does nothing and reads nothing. It is
   * off unless `?debug=1` asked for it, and disabled it costs one boolean test per call.
   */
  useEffect(() => {
    if (!isLive || typeof window === "undefined") return;
    const trace = initPerfTrace({
      label: "capture",
      enabled: traceRequested(),
      endpoint: `${API_BASE}/api/v1/telemetry/capture`,
      flushOnHide: true,
    });
    if (!trace.enabled) return;
    trace.start();
    return () => {
      trace.stop();
      trace.flush();
    };
  }, [isLive]);

  // The two facts that settle the encoder argument, recorded as soon as they exist. The
  // codec is the decisive one: MediaRecorder on Chrome for Android has no software H.264, so
  // an `video/mp4` mimeType proves a hardware encode and a `video/webm` one proves libvpx on
  // the CPU. Three rounds of analysis argued about this from a stale comment.
  useEffect(() => {
    const trace = perfTrace();
    if (!trace.enabled) return;
    trace.attachTrack(camera.stream?.getVideoTracks()[0] ?? null);
  }, [camera.stream]);

  useEffect(() => {
    const trace = perfTrace();
    if (!trace.enabled || !recorder.mimeType) return;
    trace.note("mimeType", recorder.mimeType);
  }, [recorder.mimeType]);

  // Where the per-frame analysis actually ran, and why it moved if it did. Without this a
  // capture that silently fell back to the synchronous readback is indistinguishable from
  // one where the offload simply did not help.
  useEffect(() => {
    if (!isLive) return;
    const trace = perfTrace();
    if (!trace.enabled) return;
    const settle = window.setTimeout(() => {
      trace.note("analysis", camera.analysisBackend());
      const reason = camera.analysisDemotedReason();
      if (reason) trace.note("analysisDemoted", reason);
    }, 12_000);
    return () => window.clearTimeout(settle);
  }, [isLive, camera]);

  useEffect(() => {
    if (!isLive || typeof document === "undefined") return;
    document.body.dataset.liveCamera = "1";
    return () => {
      delete document.body.dataset.liveCamera;
    };
  }, [isLive]);

  useEffect(() => {
    if (!isLive || typeof window === "undefined") return;
    let frame = 0;

    const tick = () => {
      frame = window.requestAnimationFrame(tick);

      const now = performance.now();
      const loop = loopRef.current;

      // Before the gate below, deliberately. Past it the governor could never observe
      // anything faster than the loop's own process rate, so it could not tell a healthy
      // 60 fps phone from one barely holding 34 — the same reason the frame counter sits
      // here rather than in the diagnostics block.
      const plan = governor.frame(now);
      planRef.current = plan;

      // FRAME RATE IS COUNTED HERE, BEFORE THE GATE BELOW.
      //
      // rAF fires at the display's rate; everything after the gate runs at 33 Hz by design.
      // Counting after it produced a "fps" reading structurally incapable of exceeding 33 —
      // so a phone painting a healthy 60 and a phone barely holding 34 both reported the
      // same number, and the one instrument meant to reveal jank was blind to it.
      if (debugRef.current) loop.debugFrames = (loop.debugFrames ?? 0) + 1;

      // Everything below runs at the PUBLISH cadence, not at the display's.
      //
      // The individual samplers were already throttled, but the work between them —
      // stamping the coverage cone, updating the station lattice, running every detector,
      // evaluating row completion — ran on every animation frame, i.e. 60 times a second
      // on top of a live camera preview, a guidance overlay and a hardware encoder. A real
      // mid-range Android phone stuttered visibly. Half of that work was being thrown away
      // regardless, because nothing downstream is published faster than 30 Hz.
      //
      // At the coached 24 deg/s the view moves 0.8 deg between 30 Hz ticks — a fifth of a
      // 5 deg coverage cell — so nothing is missed by looking twice as rarely.
      if (now - loop.lastProcess < processIntervalMs(plan)) return;
      loop.lastProcess = now;
      // Diagnostics for `?debug=1`. A plain property write on an object that already
      // exists: no allocation, no React, nothing that could itself become the slowness it
      // is meant to measure. `tickStart` is read again at the end of the tick.
      const tickStart = now;
      const current = stateRef.current;
      const cam = cameraRef.current;
      const rec = recorderRef.current;

      // 1. Gather. Frame statistics and optical flow each walk a downscaled canvas, and
      //    neither means anything faster than its own sampling rate.
      if (now - loop.lastQuality >= 1000 / plan.qualityHz) {
        loop.quality = cam.readQuality();
        loop.lastQuality = now;
      }
      if (plan.opticalFlow) {
        if (now - loop.lastFlow >= FLOW_INTERVAL_MS) {
          loop.flow = cam.readFlow();
          loop.lastFlow = now;
        }
      } else if (loop.flow !== null) {
        // Cleared, not left standing. `loop.flow` is handed straight to `SensorFrame.flow`,
        // so a stale sample would let the divergence detector keep asserting a measurement
        // that stopped being taken — inventing a warning out of a tier change.
        loop.flow = null;
      }

      const sample = orientationRef.current.read();
      const recording = rec.state === "recording";
      // The coached row comes from the MACHINE, not from the published `activeRow`, which is
      // one render behind at a transition. That single frame is enough to attribute the
      // first samples of `row_up` to `level` — and, worse, to re-fire ROW_COMPLETE against
      // the row that just finished and skip a row entirely.
      // In `touch_up` and `bonus_poles` the row follows the bright ring, read straight out
      // of the store rather than off a rendered value: the loop must not depend on React
      // having committed anything, and the last publish IS the previous frame's target — the
      // same one-publish lag as before, with no render in the path.
      //
      // EVERY OTHER non-row state is null, and that is load-bearing. `framing`, `paused` and
      // `sweep_check` have no row being coached, and `sensorFrame.activeRow` is what gates
      // the `row_pitch` warning: handing those states the last target's row would coach a
      // pitch at a user who is not being asked to sweep anything, which is precisely the
      // "the fix invents a new warning" failure this pass exists to remove.
      const rowNow: RowId | null = (ROW_STATES as readonly string[]).includes(current)
        ? ROW_ID_BY_STATE[current as (typeof ROW_STATES)[number]]
        : current === "touch_up" || current === "bonus_poles"
          ? (hotStore.get().target?.station.row ?? null)
          : null;
      const sensorFrame: SensorFrame = {
        t: now,
        orientation: sample,
        motion: motionRef.current.read(),
        quality: loop.quality,
        flow: loop.flow,
        recording,
        activeRow: rowNow,
      };

      // 2. Detect. Only warnings[0] is ever shown: a person mid-rotation cannot triage.
      const detector = detectorRef.current;
      // The two facts a SensorFrame cannot carry. Both are binary and latched immediately:
      // a lens that switched mid-capture poisons every pose, and rotation.py's [30, 110]
      // hfov clamp will not catch a jump to ultrawide.
      detector?.setExternalFlags({ landscape: portraitBlockedRef.current, lensChanged: cam.configurationChanged });
      if (cam.configurationChanged) telemetryRef.current?.noteLensChange();
      const readout = detector ? detector.push(sensorFrame) : null;
      readoutRef.current = readout;
      const warning = detector ? (detector.warnings()[0] ?? null) : null;
      const blocked = warning ? WARNING_BLOCKS_CAPTURE.includes(warning.severity) : false;

      if (warning && warning.code !== loop.lastWarningCode) {
        feedback.haptic(warning.severity === "stop" ? HAPTICS.stop : HAPTICS.warn);
        if (warning.voiceKey) speak(warning.voiceKey);
        loop.lastWarningCode = warning.code;
      } else if (!warning) {
        loop.lastWarningCode = null;
      }

      // Walking is the only detector allowed to interrupt, and only once it has held for
      // six seconds — one stumble must not cost the row.
      if (warning?.code === "walking") {
        loop.walkingSince ??= now;
        if (now - loop.walkingSince > WALKING_PAUSE_MS && RECORDING_STATES.includes(current)) {
          loop.walkingSince = null;
          send("PAUSE");
        }
      } else {
        loop.walkingSince = null;
      }

      // 3. Framing gates. No tap starts the recording: a tap at t=0 jolts the phone, which
      //    is exactly how the measured capture lost its first keyframe to a blown frame.
      //    The decision is pure and lives in `plan.ts`; only the plumbing is here.
      if (current === "framing") {
        // Remember the brightest direction seen while framing so the start heading can be
        // pushed away from it.
        if (sample && readout && readout.meanLuma > loop.brightestLuma) {
          loop.brightestLuma = readout.meanLuma;
          loop.brightestYaw = sample.yawDeg;
        }
        const decision = Plan.framingDecision({
          now,
          readout,
          pitchDeg: sample?.pitchDeg ?? null,
          yawDeg: sample?.yawDeg ?? null,
          brightestYawDeg: loop.brightestYaw,
          // The gate that no roll or pitch test can catch: both have already been
          // un-rotated by `screen.orientation.angle` upstream, so a phone lying level in
          // landscape reads 0/0 and passes everything.
          portraitBlocked: portraitBlockedRef.current,
          enteredAt: loop.framingEnteredAt,
          steadySince: loop.steadySince,
          gatesSince: loop.gatesSince,
        });
        loop.steadySince = decision.steadySince;
        loop.gatesSince = decision.gatesSince;
        setFramingGate(decision.gate);
        setCountdown(decision.countdownSeconds);
        setCanForceStart(decision.canForceStart);
        if (decision.start) send("GATES_PASSED");
      }

      // The angle the file is being written at, latched while it is being written. Zero is
      // the answer at `sweep_check` no matter what happened during the sweep.
      if (recording && sample && sample.screenAngleDeg !== 0) {
        recordedScreenAngleRef.current = sample.screenAngleDeg;
      }

      // 4. Stations, coverage and telemetry — only while the recorder is actually running.
      if (sample && recording) {
        if (current === "bonus_poles") applyPoleRollGate(bonusRef.current, sample.rollDeg);
        const filling = fillTargets(current, stationsRef.current, bonusRef.current, rowStationsRef.current);
        // The settling pause at a row boundary. It gates the FILL only: the recorder keeps
        // running and the dome keeps being stamped, because two seconds of the phone held
        // steady on the ring is the best footage in the row, not footage to throw away.
        // Only the claim "this station is covered" waits for the hand to stop moving.
        const arm =
          current === "touch_up" || current === "bonus_poles"
            ? { blocking: false, secondsLeft: null, since: null }
            : Plan.armRow(rowNow, filling, sample, loop.rowArmSince, now);
        loop.rowArmSince = arm.since;
        if (arm.secondsLeft !== rowArmCountdownRef.current) {
          rowArmCountdownRef.current = arm.secondsLeft;
          setRowArmCountdown(arm.secondsLeft);
        }
        const justFilled = Plan.updateStations(filling, sample, now, blocked || arm.blocking);
        if (justFilled.length > 0) {
          loop.lastFillAt = now;
          // The detector owns the stall clock, so the "you are stuck" coach and the
          // warnings agree about when the row stopped moving.
          detector?.noteStationFilled(now);
          feedback.tick();
          feedback.haptic(HAPTICS.stationFilled);
          // Publish whichever set is actually on screen: the pole stations are a separate
          // array, and republishing the row stations here would blank the bonus pass.
          setStationsView((current === "bonus_poles" ? bonusRef.current : stationsRef.current).slice());
        }

        if (now - loop.lastStamp >= 1000 / plan.stampHz) {
          loop.lastStamp = now;
          // A blocked capture must not bank coverage either, or the user paints the sphere
          // with footage the engine will refuse.
          if (!blocked && gridRef.current) Coverage.stampCone(gridRef.current, sample, lensRef.current);
          telemetryRef.current?.push(sensorFrame, readout);
        }

        trackRow(sample, rowNow, rowTrackRef.current);
        maybeGrabThumbnail(current, now, loop, sample.yawDeg, cam, thumbnailsRef, plan.thumbnails);
      }

      // 4b. RAIL MODE. No orientation sample ever arrives, so none of the above runs: no
      //     station can fill, no cone is stamped and no row can ever earn its exit. The
      //     fallback is the one the permission screen promised — the coached duration for
      //     the row, counted down on screen, with the copy saying plainly that it is a
      //     timer and not a measurement. Coverage stays honestly empty and the verdict is
      //     told so, rather than being handed a zero it would read as "you missed it all".
      let railSecondsLeft: number | null = null;
      if (railModeRef.current && recording && (ROW_STATES as readonly string[]).includes(current)) {
        if (loop.railRow !== current) {
          loop.railRow = current;
          loop.railElapsedMs = 0;
          loop.railLastTick = now;
        }
        // Clamped, and only while `recording`: the counter measures footage, not wall clock.
        loop.railElapsedMs += Math.min(Math.max(0, now - loop.railLastTick), RAIL_MAX_STEP_MS);
        loop.railLastTick = now;
        const rail = Plan.railRowProgress(loop.railElapsedMs, rowNow ?? "level");
        railSecondsLeft = rail.remainingSeconds;
        // Paint the drawing-only lattice so the strip advances with the clock.
        const filled = paintRailLattice(railStationsRef.current, rowNow ?? "level", rail.fraction);
        if (filled !== loop.railFilled) {
          loop.railFilled = filled;
          feedback.tick();
          setStationsView(railStationsRef.current.slice());
        }
        maybeGrabThumbnail(current, now, loop, null, cam, thumbnailsRef, plan.thumbnails);
        if (rail.complete) send("ROW_COMPLETE");
      }

      // 5. Progress and completion.
      if (now - loop.lastReport >= REPORT_INTERVAL_MS) {
        loop.lastReport = now;
        const report = buildReport();
        if (report) {
          setCoverageReport(report);
          if (recording && RECORDING_STATES.includes(current)) {
            const done =
              current === "touch_up" || current === "bonus_poles"
                ? allTargetsFilled(current, stationsRef.current, bonusRef.current, touchUpIdsRef.current)
                : rowNow !== null && report.rows[rowNow].complete;
            if (done) send("ROW_COMPLETE");
          }
        }
      }

      // TOTAL recorded time, across every sealed segment plus the live one. The recorder's
      // own clock restarts at each segment boundary, so it is not a capture length — and a
      // capture is now routinely several segments. Past the ceiling nothing more would
      // upload, so the capture ENDS rather than quietly growing a file nobody can send.
      const recordedMs = segmentsRef.current.reduce((total, s) => total + s.durationMs, 0) + rec.elapsedMs;
      const clock = Plan.captureClock(recordedMs);
      if (recording && clock.exhausted && RECORDING_STATES.includes(current)) {
        setBudgetExhausted(true);
        send("FINISH");
      }

      // 6. Publish.
      if (now - loop.lastPublish >= publishIntervalMs(plan)) {
        loop.lastPublish = now;
        const ring = ringTargets(current, stationsRef.current, bonusRef.current, touchUpIdsRef.current, sample?.rollDeg ?? 0, rowStationsRef.current);
        const chosen = sample ? Plan.nextTarget(ring, sample, rowNow, lensRef.current, loop.lastTargetId ?? null) : null;
        loop.lastTargetId = chosen?.station.id ?? null;
        hotStore.publish({
          q: sample?.q ?? IDENTITY_QUAT,
          headingDeg: sample?.yawDeg ?? 0,
          rollDeg: sample?.rollDeg ?? 0,
          target: chosen,
          readout,
          warning,
          severity: warning?.severity ?? null,
          // In rail mode the rail is a clock, so the three-segment bar has to be one too —
          // reading it off an empty station lattice would peg all three rows at zero for
          // the whole capture.
          rowFractions: railModeRef.current ? rowFractions(railStationsRef.current) : rowFractions(stationsRef.current),
          stuck: recording && detector !== null && detector.isStalled(now, STUCK_AFTER_MS),
          recordedMs,
          railSecondsLeft,
        });
      }

      // 7. Diagnostics. Cheap enough to leave on: two subtractions and one object write.
      if (debugRef.current && typeof window !== "undefined") {
        const spent = performance.now() - tickStart;
        loop.debugWorst = Math.max(loop.debugWorst ?? 0, spent);
        if (now - (loop.debugSince ?? 0) >= 1000) {
          loop.debugFps = loop.debugFrames;
          loop.debugFrames = 0;
          loop.debugSince = now;
          loop.debugWorst = spent;
        }
        // `meter()` allocates, so it is built only inside this branch and never in the hot
        // path. Its fps and the counter's will legitimately disagree across a background
        // gap — the governor drops stalls over a second, the counter does not.
        const meter = governor.meter();
        window.__biosphereDebug = {
          fps: loop.debugFps ?? 0,
          tier: meter.tier,
          floorTier: meter.floorTier,
          medianMs: meter.medianIntervalMs,
          jankRatio: meter.jankRatio,
          governorReason: meter.reason,
          analysis: cam.analysisBackend(),
          demoted: cam.analysisDemotedReason(),
          // The two facts that settle the encoder question on the device rather than in a
          // comment. `video/mp4` on Chrome for Android can only be the hardware encoder;
          // `video/webm` can only be libvpx on the CPU.
          codec: rec.mimeType ?? "-",
          camW: cam.size?.width ?? 0,
          camH: cam.size?.height ?? 0,
          // The number that decides whether a rotating capture can be sharp at all. Over
          // ~12 ms the smear exceeds what the feature detector survives, at any sweep speed.
          exposureMs: (loop.quality?.exposureS ?? 0) * 1000,
          iso: typeof cam.settings?.iso === "number" ? cam.settings.iso : 0,
          lumaMean: loop.quality?.meanLuma ?? 0,
          tickMs: spent,
          worstTickMs: loop.debugWorst ?? 0,
          orientationAgeMs: sample ? now - sample.t : 9999,
          motionAgeMs: readout?.motionGateDegraded ? -1 : 0,
          omegaDegPerS: readout?.omegaDegPerS ?? 0,
          recorderState: rec.state,
          recordedBytes: rec.chunkCount,
          warning: warning?.code ?? null,
          severity: warning?.severity ?? null,
          blocked,
          state: current,
        };
      }
    };

    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [
    isLive,
    // Referentially stable — built once through a lazy ref — so naming it here satisfies the
    // linter without ever rebuilding the loop.
    governor,
    buildReport,
    cameraRef,
    feedback,
    hotStore,
    lensRef,
    motionRef,
    orientationRef,
    portraitBlockedRef,
    railModeRef,
    recorderRef,
    send,
    speak,
  ]);

  /**
   * RAIL MODE, second entry: the sensor was granted and never spoke.
   *
   * A permission grant is not a sensor. Desktop browsers, a few Android builds and any
   * device whose compass has been disabled at the OS level report "granted" and then emit
   * nothing at all — and the resulting screen is identical to the denied case the user was
   * warned about: no rings, no coverage, no row that can ever end. Four seconds of silence
   * once the stage is live is the honest cut-off. One-way: a lattice that materialises
   * mid-row after the user has been following a counter is worse than one consistent guide.
   */
  useEffect(() => {
    if (!isLive || railMode) return;
    const timer = setTimeout(() => {
      if (orientationRef.current.read() === null) setRailMode(true);
    }, RAIL_FALLBACK_MS);
    return () => clearTimeout(timer);
  }, [isLive, railMode, orientationRef]);

  // ── Upload ──────────────────────────────────────────────────────────────────────────

  const abortRef = useRef<AbortController | null>(null);
  /** Survives a failed upload so "retry" sends only the parts the server has not got. It
   *  checks the files' names and sizes itself, so a re-shoot can never inherit stale parts. */
  const uploadResumeRef = useRef<UploadResume>(new UploadResume());

  const upload = useCallback(async () => {
    const built = resultRef.current;
    if (!built || built.segments.length === 0) {
      setUploadError(localeRef.current === "tr" ? "Yüklenecek kayıt bulunamadı." : "There is no recording to upload.");
      send("UPLOAD_FAILED");
      return;
    }

    setUploadProgress(0);
    setUploadError(null);

    const controller = new AbortController();
    abortRef.current = controller;

    const extension = built.segments[0].mimeType.includes("mp4") ? "mp4" : "webm";
    const files: File[] = built.segments.map(
      (segment, index) => new File([segment.blob], `capture-${index + 1}.${extension}`, { type: segment.mimeType }),
    );
    // The `motion.json` sidecar rides the existing `files[]` array, so `api.ts` is untouched.
    const sidecar = telemetryRef.current?.toFile(built.telemetry);
    if (sidecar) files.push(sidecar);

    const connection = connectionInfo();
    const plan: UploadPlan = {
      propertyId,
      files,
      name: sceneNameRef.current.trim() || (localeRef.current === "tr" ? "Sahne" : "Scene"),
      qualityMode: built.qualityMode,
      captureKind: built.captureKind,
      estimatedBytes: files.reduce((total, file) => total + file.size, 0),
      effectiveType: connection.effectiveType,
      saveData: connection.saveData,
    };

    try {
      const created = await api.scenes.create(
        plan.propertyId,
        plan.files,
        { name: plan.name, quality_mode: plan.qualityMode, capture_kind: plan.captureKind },
        setUploadProgress,
        controller.signal,
        uploadResumeRef.current,
      );
      send("UPLOAD_OK");
      onHandoff({ sceneId: created.scene.id, jobId: created.job.id });
    } catch (cause) {
      // UPLOAD_POLICY.retainDraftUntilAck — the blobs are NOT released on failure, so a
      // retry never means a re-shoot.
      setUploadError(
        cause instanceof BioSphereError
          ? cause.localised(localeRef.current)
          : localeRef.current === "tr"
            ? "Yükleme tamamlanamadı."
            : "The upload did not finish.",
      );
      send("UPLOAD_FAILED");
    } finally {
      abortRef.current = null;
    }
  }, [localeRef, onHandoff, resultRef, sceneNameRef, send, propertyId]);

  const uploadRef = useLatest(upload);

  // ── State entry effects ─────────────────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;

    const enter = async () => {
      switch (state) {
        case "stance": {
          resetSession();
          break;
        }

        case "framing": {
          // `framing` is reached three ways: from `stance` (fresh), from `paused` via
          // RESTART, and from `review` via RETAKE. The last two arrive with a live recorder
          // and a half-painted sphere, so entry always discards and rebuilds. Anything else
          // would silently mix two captures into one scene.
          const recorderState = recorderRef.current.state;
          if (recorderState === "recording" || recorderState === "paused") {
            await recorderRef.current.stop("interrupted");
          }
          resetSession();
          setFramingGate("hold");
          setCountdown(null);
          rowArmCountdownRef.current = null;
          setRowArmCountdown(null);
          setCanForceStart(false);
          loopRef.current.framingEnteredAt = performance.now();

          // Try to prove the recorder produces bytes before the user invests a minute —
          // `isTypeSupported()` returning true has never guaranteed that `start()` works.
          //
          // But a failed probe is NOT proof that the device cannot record, and it must not
          // be treated as such. Measured on Android 10 / Chrome 151: the identical
          // combination produced 25 KB on one attempt and nothing on the next, because the
          // hardware encoder's first chunk does not reliably arrive inside a short window.
          // Blocking on that turned a phone which records perfectly well away at the door
          // with "this browser cannot record video" — the worst kind of wrong, because the
          // user has no way to disagree with it.
          //
          // So the hard block is reserved for the one case we can be certain about:
          // MediaRecorder not existing at all. Anything else proceeds, and a capture that
          // genuinely cannot record will fail visibly a few seconds later with the footage
          // it did or did not gather — a far cheaper mistake than refusing to start.
          const preflight = await recorderRef.current.preflight();
          const recorderMissing = typeof window !== "undefined" && typeof window.MediaRecorder === "undefined";
          if (!cancelled) setPreflightFailed(recorderMissing);
          if (!cancelled && !preflight.ok && !recorderMissing) {
            console.warn(
              "[capture] recorder preflight inconclusive: no bytes arrived during the probe. " +
                "Starting anyway rather than refusing a device that may well record.",
            );
          }
          break;
        }

        case "row_level": {
          if (!recordingStartedRef.current) {
            // Yaw 0 is the loop-closure origin, zeroed at the instant recording begins.
            orientationRef.current.setOrigin();
            await beginSegment();
          }
          speak(rowVoiceKey("level"));
          break;
        }

        case "row_up":
        case "row_down": {
          // Deliberately NOT restarting the recorder — the pitch transition between rows is
          // the footage that links them in the solver.
          const row = ROW_ID_BY_STATE[state];
          // Snapshot the mask for the punctuation dome. A copy, not the live Uint8Array:
          // the loop keeps stamping into the grid, and the 1.2 s dome should show the
          // sphere as it stood when the row closed. In rail mode nothing stamped it, and a
          // celebratory dome of an empty sphere reads as a row that achieved nothing — so
          // the haptic and the spoken cue still fire, and the dome does not.
          if (gridRef.current && !railModeRef.current) setCoveredView(new Uint8Array(gridRef.current.covered));
          setPunctuationRow(row);
          feedback.haptic(HAPTICS.rowComplete);
          speak(rowVoiceKey(row));
          break;
        }

        case "touch_up": {
          touchUpIdsRef.current = gapStations(coverageRef.current, stationsRef.current);
          // Minutes may have passed on the review screen. Without this, the first tick of the
          // touch-up pass would hand `updateStations` a dt of "since the last row", clamped
          // to half a dwell, and credit aim the user never held.
          Plan.forgetStationClock(stationsRef.current);
          setTouchUpIds(touchUpIdsRef.current);
          setStationsView(stationsRef.current.slice());
          // A second segment for the same scene. `api.scenes.create` already takes a File[],
          // so both reach the engine in one upload.
          if (recorderRef.current.state !== "recording") await beginSegment();
          speak("capture.say.gap");
          break;
        }

        case "bonus_poles": {
          for (const station of bonusRef.current) {
            station.filled = false;
            station.filledAt = null;
            station.dwellMs = 0;
          }
          Plan.forgetStationClock(bonusRef.current);
          setStationsView(bonusRef.current.slice());
          if (recorderRef.current.state !== "recording") await beginSegment();
          break;
        }

        case "paused": {
          recorderRef.current.pause();
          break;
        }

        case "sweep_check": {
          const startedAt = performance.now();
          // The sealed segment arrives through `onSegment`, which has already pushed it by
          // the time this resolves — `finishSeal` calls the hook before it settles the
          // promise. Pushing the return value here as well would double every segment.
          await recorderRef.current.stop(
            previousStateRef.current === "row_down" ? "row_complete" : "user_finish",
          );
          recordingStartedRef.current = false;
          setRecordedBytes(segmentsRef.current.reduce((total, segment) => total + segment.blob.size, 0));

          // The geometry the engine asserts the decoded orientation against. Taken from the
          // sealed segments rather than from the live camera: iOS records sensor-native
          // landscape and applies a rotation tag, so only what actually came out of
          // MediaRecorder is worth putting in the sidecar. `recordedFrameSize` picks the
          // last POSITIVE pair — a `??` chain cannot, because 0 is not nullish and a
          // segment sealed before the decoder spoke would poison every fallback behind it.
          const last = segmentsRef.current[segmentsRef.current.length - 1];
          const size = Plan.recordedFrameSize(segmentsRef.current, cameraRef.current.size);
          telemetryRef.current?.setRecordingContext({
            // Latched while the recording was actually running; by now the phone is back in
            // portrait and reading the live angle would always report 0.
            screenAngleDeg: recordedScreenAngleRef.current,
            requestedBitrateBps: RECORDING.targetBitrateBps,
            mimeType: last?.mimeType ?? recorderRef.current.mimeType,
            width: size.width,
            height: size.height,
          });
          // The mask the server re-renders its heat-map from, and the one the review dome
          // draws — the phone's "94%" and the server's "94%" are the same 2592 cells.
          if (gridRef.current) telemetryRef.current?.setCoveredMask(gridRef.current.covered);

          const report = buildReport();
          if (report && !cancelled) {
            setCoverageReport(report);
            const built = finalise(report, segmentsRef.current, detectorRef.current, telemetryRef.current, {
              thumbnails: thumbnailsRef.current,
              qualityMode: configRef.current.qualityMode,
              // In rail mode nothing ever stamped the sphere. The verdict has to be told
              // that, or it reads a coverage of zero as a capture that missed everything.
              coverageMeasured: !railModeRef.current,
            });
            if (built) setResult(built);
          }

          // Hold the veil for a full beat even when the work took 40 ms: a verdict that
          // appears instantly reads as a guess rather than a measurement.
          await wait(Math.max(0, SWEEP_CHECK_MS - (performance.now() - startedAt)));
          if (!cancelled) send("CHECK_DONE");
          break;
        }

        case "review": {
          speak("capture.say.done");
          break;
        }

        case "uploading": {
          await uploadRef.current();
          break;
        }

        case "exit_to_upload": {
          onUseExistingVideo();
          break;
        }

        default:
          break;
      }
    };

    void enter();
    return () => {
      cancelled = true;
    };
    // `state` is the entry trigger, and everything else is reached through refs on purpose:
    // an entry action must run exactly once per entry, not again because a sensor
    // re-rendered the component.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  /**
   * Segment continuity — the BACKSTOP. `handleSegment` re-opens the recorder the instant a
   * self-initiated seal lands, with no gap at all; this effect exists for the cases where
   * no seal ever arrives, and it is a state invariant rather than an event handler: while
   * the machine is in a recording state and the phone is upright, the recorder is running.
   *
   * Three ways it can stop from underneath us:
   *
   *   - PAUSE (the user, a rotation, backgrounding, six seconds of walking) left it paused;
   *     resuming is enough, and the segment continues unbroken.
   *   - the track ENDED. An incoming call does exactly this on iOS. There is no resuming
   *     from that, so the partial segment is sealed and a new one opened. Because a scene is
   *     already a `File[]`, the call costs a join in the footage and nothing else — where
   *     before it cost the entire capture.
   *   - `start()` itself failed, or a seal raced the state the machine was moving into.
   *
   * It deliberately does NOT key on `state === "error"`. That was the original condition
   * and it can never be true here: every path into `error` ends in `finishSeal`, which
   * overwrites the state with `"stopped"` before anyone can observe it — so the branch was
   * dead for every self-initiated stop, which is to say for every case it was written for.
   */
  useEffect(() => {
    if (!RECORDING_STATES.includes(state) || portraitBlocked || !recordingStartedRef.current) return;

    if (recorder.state === "paused") {
      recorderRef.current.resume();
      return;
    }
    if (recorder.state === "recording") return;
    if (recorderFaultsRef.current > MAX_RECORDER_FAULTS) return;

    // The camera is still reopening after the call. `camera.stream` is a dependency, so
    // this effect runs again the moment a live one lands; the timer is the promise that the
    // user is not left turning in front of a frozen preview if it never does.
    if (!canRecord()) {
      const giveUp = setTimeout(() => {
        if (!canRecord() && RECORDING_STATES.includes(stateRef.current)) send("PAUSE");
      }, CAMERA_REOPEN_GRACE_MS);
      return () => clearTimeout(giveUp);
    }

    let cancelled = false;
    // Read through a call, not a property: `recorder.state` is a live getter onto the
    // controller, and TypeScript would otherwise narrow the first read and keep that
    // narrowing across the awaits — during which the value is precisely what changes.
    const running = () => recorderRef.current.state === "recording";
    void (async () => {
      // Seals and delivers (through `onSegment`) whatever chunks survived; a no-op when the
      // recorder died with nothing in hand.
      await recorderRef.current.stop("interrupted");
      if (cancelled || running()) return;
      await beginSegment();
      // Still not running: the recorder is genuinely gone. Stop turning the user in front
      // of a screen that looks fine and show them the paused sheet — every second sealed
      // so far is in `segmentsRef`, and finishing from there uploads all of it.
      if (!cancelled && !running()) send("PAUSE");
    })();
    return () => {
      cancelled = true;
    };
  }, [state, portraitBlocked, recorder.state, camera.stream, canRecord, recorderRef, beginSegment, send]);

  useEffect(() => {
    if (!punctuationRow) return;
    const timer = setTimeout(() => setPunctuationRow(null), PUNCTUATION_MS);
    return () => clearTimeout(timer);
  }, [punctuationRow]);

  // ── Platform interrupts ─────────────────────────────────────────────────────────────

  // Portrait. There is NO API that can force it — `screen.orientation.lock()` does not exist
  // on iOS Safari at any version — so it is a UI gate. On rotation the recorder is PAUSED,
  // never stopped: iOS can change the output dimensions mid-stream, and stopping would cost
  // the whole capture for a two-second wrist twist.
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const query = window.matchMedia("(orientation: portrait)");
    let settleTimer = 0;

    const apply = () => {
      const portrait = query.matches;
      setPortraitBlocked(!portrait);
      setPermissions((prev) => (prev.portrait === portrait ? prev : { ...prev, portrait }));

      if (!portrait) {
        if (recorderRef.current.state === "recording") recorderRef.current.pause();
        return;
      }
      if (recorderRef.current.state !== "paused" || !RECORDING_STATES.includes(stateRef.current)) return;
      window.clearTimeout(settleTimer);
      settleTimer = window.setTimeout(() => {
        if (recorderRef.current.state === "paused" && RECORDING_STATES.includes(stateRef.current)) {
          recorderRef.current.resume();
        }
      }, ROTATION_SETTLE_MS);
    };

    apply();
    query.addEventListener("change", apply);
    return () => {
      window.clearTimeout(settleTimer);
      query.removeEventListener("change", apply);
    };
  }, [recorderRef]);

  // Wake lock. An active camera stream does NOT reliably hold the screen awake, and the
  // platform silently drops the lock on backgrounding, so it is re-acquired on return.
  const wakeRef = useRef<WakeLockSentinel | null>(null);

  const acquireWakeLock = useCallback(async () => {
    if (typeof navigator === "undefined" || !("wakeLock" in navigator)) return;
    try {
      wakeRef.current = await navigator.wakeLock.request("screen");
      setPermissions((prev) => ({ ...prev, wakeLock: true }));
    } catch {
      // Refused, unsupported, or the document was hidden. A real degradation, never fatal.
      setPermissions((prev) => ({ ...prev, wakeLock: false }));
    }
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") return;
    const onVisibility = () => {
      if (document.visibilityState === "hidden") {
        if (RECORDING_STATES.includes(stateRef.current)) send("PAUSE");
      } else if (LIVE_STATES.has(stateRef.current)) {
        void acquireWakeLock();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [acquireWakeLock, send]);

  useEffect(() => {
    if (!isLive) return;
    void acquireWakeLock();
    return () => {
      void wakeRef.current?.release().catch(() => undefined);
      wakeRef.current = null;
    };
  }, [isLive, acquireWakeLock]);

  // Teardown: tracks stopped, listeners gone, upload aborted, speech silenced.
  useEffect(
    () => () => {
      abortRef.current?.abort();
      orientationRef.current.stop();
      motionRef.current.stop();
      cameraRef.current.close();
      void wakeRef.current?.release().catch(() => undefined);
      cancelSpeech();
    },
    [cameraRef, motionRef, orientationRef],
  );

  // ── Actions ─────────────────────────────────────────────────────────────────────────

  /** Fullscreen, the orientation lock and the AudioContext all need a user gesture, so they
   *  ride the "Hazırım" tap — the last deliberate press before the camera takes over. */
  const enterImmersive = useCallback(() => {
    feedback.unlock();
    if (typeof document === "undefined") return;

    const element = document.documentElement;
    if (typeof element.requestFullscreen === "function" && !document.fullscreenElement) {
      void element.requestFullscreen().catch(() => undefined);
    }
    // `lock()` is absent from lib.dom and from every iPhone, hence a structural type plus a
    // runtime check rather than a cast to `any`.
    const orientationApi = screen.orientation as ScreenOrientation & {
      lock?: (orientation: "portrait") => Promise<void>;
    };
    if (typeof orientationApi?.lock === "function") void orientationApi.lock("portrait").catch(() => undefined);
  }, [feedback]);

  /**
   * The escape hatch: keep the footage when the upload will not go through.
   *
   * EVERY segment, not just the first. A split capture is precisely the one this matters
   * for — the recorder only seals early when the encoder stalls or a call arrives, which is
   * the same bad afternoon in which the upload also fails — and saving one piece of a
   * three-piece capture while telling the user their video is safe is worse than saving
   * nothing, because they will delete the draft believing they have it.
   *
   * The clicks are staggered: a browser that sees several synchronous downloads from one
   * gesture keeps the first and silently drops the rest.
   */
  const download = useCallback(() => {
    const segments = segmentsRef.current;
    if (segments.length === 0 || typeof URL === "undefined") return;
    const stamp = Date.now();

    segments.forEach((segment, index) => {
      const suffix = segments.length === 1 ? "" : `-${index + 1}`;
      const extension = segment.mimeType.includes("mp4") ? "mp4" : "webm";
      window.setTimeout(() => {
        const url = URL.createObjectURL(segment.blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `biosphere-${stamp}${suffix}.${extension}`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        // Revoked later, not in this turn of the event loop: Safari aborts a download whose
        // object URL disappears immediately.
        window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
      }, index * DOWNLOAD_STAGGER_MS);
    });
  }, []);

  // ── Render ──────────────────────────────────────────────────────────────────────────

  if (state === "intro" || state === "resume_draft") {
    return (
      <IntroScreen
        locale={locale}
        draft={state === "resume_draft"}
        voiceGuide={voiceGuide}
        sounds={sounds}
        onVoiceGuide={setVoiceGuide}
        onSounds={setSounds}
        onStart={() => send("START")}
        onUseExistingVideo={() => send("USE_EXISTING_VIDEO")}
      />
    );
  }

  if (state === "permissions" || state === "permission_denied") {
    return (
      <PermissionGate
        status={permissionStatus}
        onRequest={requestPermissions}
        onUseExistingVideo={() => send("USE_EXISTING_VIDEO")}
        onRetry={retryPermissions}
        locale={locale}
      />
    );
  }

  if (state === "stance") {
    return (
      <StanceScreen
        locale={locale}
        onReady={() => {
          enterImmersive();
          send("READY");
        }}
        onBack={() => send("BACK")}
      />
    );
  }

  if (state === "review") {
    // A review with no result can only mean `finalise` found no detector or telemetry, which
    // is a bug rather than a user state — show the measuring veil instead of a broken page.
    return result && coverageReport ? (
      <ReviewStep
        result={result}
        coverage={coverageReport}
        thumbnails={result.thumbnails}
        sceneName={sceneName}
        onSceneNameChange={setSceneName}
        qualityMode={qualityMode}
        onQualityModeChange={(mode) => {
          // From here on the device suggestion stays quiet: a preset the user chose is a
          // decision, and a phone's spec sheet does not get to overrule it.
          qualityTouchedRef.current = true;
          setQualityMode(mode);
        }}
        deviceNote={deviceCoachingNote(phoneLens?.lens ?? null, phoneLens?.lidar ?? "no", locale)}
        onFillGaps={() => send("FILL_GAPS")}
        onAccept={() => send("ACCEPT")}
        onRetake={() => send("RETAKE")}
        onWantPoles={() => send("WANT_POLES")}
        onDownload={download}
        locale={locale}
        // RAIL MODE painted no sphere, so every coverage number on that screen would be a
        // measurement nobody took. The review says so instead of showing 0%.
        coverageMeasured={!railMode}
        // The machine's own guard, handed over rather than re-derived. `FILL_GAPS` is
        // rejected outright when `hasGaps` is false, so a review that offered the button on
        // a different rule would be offering one that does nothing when tapped.
        canFillGaps={guardsRef.current.hasGaps === true}
      />
    ) : (
      <Veil locale={locale} />
    );
  }

  if (state === "uploading" || state === "upload_failed" || state === "handoff") {
    return (
      <UploadScreen
        locale={locale}
        failed={state === "upload_failed"}
        progress={uploadProgress}
        error={uploadError}
        bytes={recordedBytes}
        onRetry={() => send("RETRY_UPLOAD")}
        onDownload={download}
        onBack={() => send("BACK")}
      />
    );
  }

  if (state === "exit_to_upload") return <Veil locale={locale} />;

  // ── The live stage ──────────────────────────────────────────────────────────────────

  const row = activeRow ?? "level";
  // WHY THE ROW HAS NOT ENDED, said out loud. `RowProgress.complete` needs the rings AND a
  // full turn; when the rings are done and the turn is not, the screen used to go quiet and
  // the user was left to guess. See `capture.cue.closeLoop`.
  const rowNowProgress = activeRow !== null ? (coverageReport?.rows[activeRow] ?? null) : null;
  const loopStillOpen =
    rowNowProgress !== null &&
    rowNowProgress.filledStations >= rowNowProgress.requiredStations &&
    rowNowProgress.yawSpanDeg < YAW_SPAN.acceptMinDeg;

  return (
    <div
      className="no-tap stage-viewport fixed inset-x-0 top-0 overflow-hidden bg-black"
      style={visible ? { width: visible.width, height: visible.height, left: visible.left, top: visible.top } : undefined}
    >
      <CameraFeed videoRef={camera.videoRef} />
      {debugRef.current ? <DebugHud /> : null}

      <HotGuidance
        store={hotStore}
        stations={stationsView}
        onlyRow={state === "touch_up" ? null : activeRow}
        onlyStations={state === "touch_up" ? touchUpIds : null}
        lens={lens}
        railMode={railMode}
        reducedMotion={reducedMotion}
      />

      {/* The row-boundary settling pause. Deliberately NOT a card: it has to leave the ring
          and the viewfinder visible, because the instruction is "keep aiming at that". */}
      {rowArmCountdown !== null ? (
        <div className="pointer-events-none absolute inset-0 z-20 grid place-items-center">
          <div className="flex flex-col items-center rounded-lg bg-canvas/55 px-6 py-4 backdrop-blur-[2px]">
            <p className="font-mono text-[54px] leading-none font-semibold text-bio tabular-nums">
              {rowArmCountdown}
            </p>
            <p className="mt-1.5 text-[13px] font-medium text-ink">{cc("capture.row.settle", locale)}</p>
          </div>
        </div>
      ) : null}

      {/* Chrome sits on its own scrim so pale type never has to survive a white wall. */}
      <div className="pointer-events-none absolute inset-x-0 top-0 bg-[linear-gradient(180deg,rgb(0_0_0/0.55),transparent)] pb-10">
        <div className="pointer-events-auto px-4 pt-[max(env(safe-area-inset-top),0.75rem)]">
          <HotRowPrompt
            store={hotStore}
            row={row}
            rowIndex={ROW_IDS.indexOf(row) + 1}
            title={cc(rowTitleKey(row), locale)}
            cue={cc(
              state === "touch_up"
                ? "capture.resume.aim"
                : loopStillOpen
                  ? "capture.cue.closeLoop"
                  : rowCueKey(row),
              locale,
            )}
            reducedMotion={reducedMotion}
            lens={lens}
          />
          {railMode && RECORDING_STATES.includes(state) ? <HotRailCounter store={hotStore} locale={locale} /> : null}
        </div>
      </div>

      <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-[linear-gradient(0deg,rgb(0_0_0/0.6),transparent)] pt-12">
        <div className="pointer-events-auto px-4 pb-[max(env(safe-area-inset-bottom),0.75rem)]">
          {RECORDING_STATES.includes(state) ? <HotRecordingClock store={hotStore} bytes={recordedBytes} locale={locale} /> : null}
          <div className="flex items-center gap-3">
            <HotSpeedMeter store={hotStore} reducedMotion={reducedMotion} />
            {RECORDING_STATES.includes(state) ? (
              <button
                type="button"
                // `bonus_poles` has no FINISH transition — the poles are optional, so leaving
                // them is SKIP_POLES. Sending FINISH there would be a legal no-op and would
                // strand the user behind two frames they never wanted.
                onClick={() => send(state === "bonus_poles" ? "SKIP_POLES" : "FINISH")}
                className="glass-strong rounded-full px-4 py-2.5 text-[13px] font-semibold text-ink"
              >
                {cc(state === "bonus_poles" ? "capture.bonus.skip" : "capture.finish", locale)}
              </button>
            ) : null}
          </div>
        </div>
      </div>

      {/*
        The toast MUST be positioned from here. Its own root is `position: static`, and every
        other child of this `fixed inset-0` stage is absolutely positioned — so an unwrapped
        toast is the only in-flow child, lands in the container's top-left corner, and is
        painted UNDERNEATH the top scrim and the row prompt by CSS paint order. The single
        most important card in the product — the red "Yürümeyin" stop — was arriving dimmed
        and overlapped. Above the bottom chrome, below the modal layers.
        The rotate prompt owns the screen while it is up, so the toast stands down.
      */}
      {/* 10rem clears the bottom chrome — the 6rem speed meter, the clock line above it and
          the safe-area inset below — on every viewport, without hard-coding a device. */}
      <div className="pointer-events-none absolute inset-x-0 bottom-[calc(env(safe-area-inset-bottom,0px)+10rem)] z-20 flex justify-center px-4">
        <HotWarningToast store={hotStore} suppressed={portraitBlocked} locale={locale} reducedMotion={reducedMotion} />
      </div>

      {punctuationRow && coveredView ? (
        <div className="pointer-events-none absolute inset-0 grid place-items-center">
          {/* THE TRANSITION HAS TO SAY WHICH ROW IS NEXT, IN WORDS.
              It used to be this dome and nothing else: a wireframe band for 1.2 s, from
              which the user was expected to infer that the upper row had begun. The report
              was "after eye level it started me from below, then carried on upward — it got
              confusing". A guide that shows a diagram instead of naming the next move is an
              instrument, not a guide. */}
          <div className="flex flex-col items-center gap-3">
            <CoverageDome
              covered={coveredView}
              yawCells={COVERAGE_GRID.yawCells}
              pitchCells={COVERAGE_GRID.pitchCells}
              coveredFraction={coverageReport?.coveredFraction ?? 0}
              nextRow={punctuationRow}
              size={96}
              variant="punctuation"
              reducedMotion={reducedMotion}
              label={cc("capture.row.done", locale)}
              lens={lens}
            />
            <div className="glass-strong rounded-2xl px-4 py-3 text-center">
              <p className="text-[12px] font-medium uppercase tracking-wide text-success">
                {cc("capture.row.done", locale)}
              </p>
              <p className="mt-1 text-[17px] font-semibold tracking-tight text-ink">
                {cc("capture.row.next", locale, { row: cc(rowTitleKey(punctuationRow), locale) })}
              </p>
              <p className="mt-1 text-[13px] leading-relaxed text-ink-muted">
                {cc(punctuationRow === "up" ? "capture.row.nextUp" : "capture.row.nextDown", locale)}
              </p>
            </div>
          </div>
        </div>
      ) : null}

      {state === "framing" ? (
        <FramingCard
          locale={locale}
          gate={framingGate}
          countdown={countdown}
          canForce={canForceStart}
          preflightFailed={preflightFailed}
          onForce={() => send("FORCE_START")}
          onUseExistingVideo={() => send("USE_EXISTING_VIDEO")}
        />
      ) : null}

      {state === "paused" ? (
        <PausedSheet
          locale={locale}
          coverage={coverageReport}
          coverageMeasured={!railMode}
          budgetExhausted={budgetExhausted}
          onResume={() => send("RESUME")}
          onFinish={() => send("FINISH")}
          onRestart={() => send("RESTART")}
        />
      ) : null}

      {state === "sweep_check" ? <Veil locale={locale} /> : null}
      {portraitBlocked ? <RotatePrompt locale={locale} /> : null}
    </div>
  );
}

// =======================================================================================
// Hot subscribers
// =======================================================================================
//
// One component per moving thing on screen, each subscribing to the slice it draws with its
// own definition of "changed". Nothing here re-renders `CaptureStage`; a publish reaches
// exactly the components whose picture is different, and a phone held perfectly still
// re-renders nothing at all, because `useOrientation` hands back the same sample object and
// every selected value compares identical.

type CaptureHotStore = HotStore<HotState>;

/** The bright ring's ROW. Changes a handful of times per capture; the target itself is a
 *  fresh object thirty times a second, which is why the row is what gets selected. */
function selectTargetRow(hot: HotState): RowId | null {
  return hot.target?.station.row ?? null;
}

interface GuidanceSlice {
  q: Quat;
  target: StationTarget | null;
  severity: WarningSeverity | null;
  rollDeg: number;
}

function selectGuidance(hot: HotState): GuidanceSlice {
  return { q: hot.q, target: hot.target, severity: hot.severity, rollDeg: hot.rollDeg };
}

/**
 * Exact comparison, deliberately — no rounding.
 *
 * Every field here is derived from ONE `OrientationSample`, so a phone that has not moved
 * produces bit-identical numbers and the whole lattice skips its frame. A phone that HAS
 * moved must redraw, and quantising would only decide how far it may drift before the rings
 * catch up. `GuidanceOverlay` already rounds to whole pixels at the point of writing, which
 * is the right place for that decision because it is the only place that knows the scale.
 */
function guidanceEqual(a: GuidanceSlice, b: GuidanceSlice): boolean {
  return (
    Object.is(a.q, b.q) &&
    a.severity === b.severity &&
    Object.is(a.rollDeg, b.rollDeg) &&
    sameTarget(a.target, b.target)
  );
}

function sameTarget(a: StationTarget | null, b: StationTarget | null): boolean {
  if (a === b) return true;
  if (a === null || b === null) return false;
  return (
    a.station.id === b.station.id &&
    Object.is(a.distanceDeg, b.distanceDeg) &&
    Object.is(a.yawDeltaDeg, b.yawDeltaDeg) &&
    Object.is(a.pitchDeltaDeg, b.pitchDeltaDeg) &&
    a.onScreen === b.onScreen
  );
}

const HotGuidance = memo(function HotGuidance({
  store,
  stations,
  onlyRow,
  onlyStations,
  lens,
  railMode,
  reducedMotion,
}: {
  store: CaptureHotStore;
  stations: StationState[];
  onlyRow: RowId | null;
  onlyStations: StationId[] | null;
  lens: LensGeometry;
  railMode: boolean;
  reducedMotion: boolean;
}) {
  const view = useHotSelector(store, selectGuidance, guidanceEqual);
  return (
    <GuidanceOverlay
      q={view.q}
      stations={stations}
      target={view.target}
      onlyRow={onlyRow}
      onlyStations={onlyStations}
      lens={lens}
      severity={view.severity}
      rollDeg={view.rollDeg}
      railMode={railMode}
      reducedMotion={reducedMotion}
      className="absolute inset-0"
    />
  );
});

type SpeedSlice = readonly [number, number, DetectorReadout["speedGrade"], number, number];

function selectSpeed(hot: HotState): SpeedSlice {
  const r = hot.readout;
  return [
    r?.omegaDegPerS ?? 0,
    r?.speedCapDegPerS ?? THRESHOLDS.angularSpeed.greenMaxDegPerS,
    r?.speedGrade ?? "green",
    r?.smearPx ?? 0,
    hot.headingDeg,
  ];
}

const HotSpeedMeter = memo(function HotSpeedMeter({ store, reducedMotion }: { store: CaptureHotStore; reducedMotion: boolean }) {
  const [omegaDegPerS, capDegPerS, grade, smearPx, headingDeg] = useHotSelector(store, selectSpeed, tupleEqual);
  return (
    <SpeedMeter
      omegaDegPerS={omegaDegPerS}
      capDegPerS={capDegPerS}
      grade={grade}
      smearPx={smearPx}
      headingDeg={headingDeg}
      reducedMotion={reducedMotion}
      className="flex-1"
    />
  );
});

type RowPromptSlice = readonly [number, number, number, boolean];

function selectRowPrompt(hot: HotState): RowPromptSlice {
  return [hot.rowFractions.level, hot.rowFractions.up, hot.rowFractions.down, hot.stuck];
}

/**
 * The rail advances when a station FILLS — a couple of dozen times in a whole capture — and
 * the stuck flag flips a handful of times more. Selecting the three fractions as numbers
 * rather than taking the freshly-allocated `rowFractions` object is the difference between
 * that and thirty renders a second of three polyline arcs.
 */
const HotRowPrompt = memo(function HotRowPrompt({
  store,
  row,
  rowIndex,
  title,
  cue,
  reducedMotion,
  lens,
}: {
  store: CaptureHotStore;
  row: RowId;
  rowIndex: number;
  title: string;
  cue: string;
  reducedMotion: boolean;
  lens: LensGeometry;
}) {
  const slice = useHotSelector(store, selectRowPrompt, tupleEqual);
  const fractions = useMemo<Record<RowId, number>>(() => ({ level: slice[0], up: slice[1], down: slice[2] }), [slice]);
  return (
    <RowPrompt
      row={row}
      rowIndex={rowIndex}
      rowFractions={fractions}
      title={title}
      cue={cue}
      stuck={slice[3]}
      reducedMotion={reducedMotion}
      lens={lens}
    />
  );
});

function selectWarning(hot: HotState): Warning | null {
  return hot.warning;
}

/**
 * A held warning is ONE occurrence, however many objects the detector allocates for it.
 *
 * `buildWarning` returns a fresh object every tick, so identity comparison would re-render
 * the toast thirty times a second for a card whose text has not changed. `since` is the
 * latch's start instant and is stable for as long as the condition holds, so `code + since`
 * is exactly "the same occurrence" — and a genuinely new occurrence has a new `since` and
 * does re-render, which is the behaviour the toast's own haptic keys on.
 */
function warningEqual(a: Warning | null, b: Warning | null): boolean {
  if (a === b) return true;
  if (a === null || b === null) return false;
  return a.code === b.code && a.since === b.since && a.severity === b.severity;
}

const HotWarningToast = memo(function HotWarningToast({
  store,
  suppressed,
  locale,
  reducedMotion,
}: {
  store: CaptureHotStore;
  suppressed: boolean;
  locale: Locale;
  reducedMotion: boolean;
}) {
  const warning = useHotSelector(store, selectWarning, warningEqual);
  return (
    <WarningToast
      warning={suppressed ? null : warning}
      title={warning ? cc(warning.titleKey, locale) : ""}
      body={warning?.bodyKey ? cc(warning.bodyKey, locale) : null}
      reducedMotion={reducedMotion}
    />
  );
});

function selectRecordedMs(hot: HotState): number {
  return hot.recordedMs;
}

/** The clock shows whole seconds, so it is told about whole seconds. */
function sameSecond(a: number, b: number): boolean {
  return Math.floor(a / 1000) === Math.floor(b / 1000);
}

const HotRecordingClock = memo(function HotRecordingClock({
  store,
  bytes,
  locale,
}: {
  store: CaptureHotStore;
  bytes: number;
  locale: Locale;
}) {
  const recordedMs = useHotSelector(store, selectRecordedMs, sameSecond);
  return <RecordingClock recordedMs={recordedMs} bytes={bytes} locale={locale} />;
});

function selectRailSeconds(hot: HotState): number | null {
  return hot.railSecondsLeft;
}

const HotRailCounter = memo(function HotRailCounter({ store, locale }: { store: CaptureHotStore; locale: Locale }) {
  const secondsLeft = useHotSelector(store, selectRailSeconds);
  return <RailCounter locale={locale} secondsLeft={secondsLeft} />;
});

// =======================================================================================
// Loop helpers — pure, so the rAF body stays readable
// =======================================================================================

/**
 * RAIL MODE's drawing lattice, painted from the clock.
 *
 * Mutated in place and returned as a COUNT rather than a new array, for the same two
 * reasons the real lattice is: `GuidanceOverlay` keys its node map on station id and a
 * fresh array every frame would thrash it, and the caller only wants to republish when the
 * picture actually changed. Nothing here ever reaches `Coverage.report` or the telemetry —
 * these stations record how long the user has been turning, not where they pointed.
 */
function paintRailLattice(stations: StationState[], row: RowId, fraction: number): number {
  const order = ROW_IDS.indexOf(row);
  let filled = 0;
  for (const station of stations) {
    const stationOrder = ROW_IDS.indexOf(station.row);
    const count = ROW_PLAN[station.row].stationCount;
    const upTo =
      stationOrder < order ? count : stationOrder > order ? 0 : Math.floor(Math.min(1, Math.max(0, fraction)) * count);
    station.filled = station.index < upTo;
    if (station.filled) filled += 1;
  }
  return filled;
}

/**
 * The array handed to `Plan.updateStations`, and the reason it is never a filtered copy.
 *
 * `updateStations` keys its dwell clock on the ARRAY ITSELF (a WeakMap of array -> last tick).
 * Handing it a freshly filtered array each frame restarts that clock every frame, so `dt` is
 * always 0, dwell never reaches `STATION_RULES.dwellMs`, and NO STATION EVER FILLS. The two
 * stable arrays are the row set and the pole set; everything else is a drawing concern.
 *
 * In `touch_up` this also means a station outside the requested set still gets credit if the
 * user happens to cross it — which is correct: the coverage is real, and completion is gated
 * on the requested set separately.
 */
/**
 * Which stations may FILL right now. A ROW THAT IS NOT ITS TURN IS LOCKED.
 *
 * This used to be every station in every row, all the time, and that is what made the guided
 * capture stop guiding. Two things went wrong at once:
 *
 *   - The up and down rows filled while the user was still sweeping the level row. The frame
 *     is 69 deg tall, so simply tilting between rows, or glancing up at the ceiling, banked
 *     stations 29 deg away. By the time their row arrived it was already part done, and the
 *     guidance started somewhere in the middle of it.
 *   - `nextTarget` prefers the active row but falls back to any unfilled station, so once the
 *     nearby level stations were done the bright ring jumped to a station in ANOTHER row —
 *     which `GuidanceOverlay` does not draw, because it is scoped to `onlyRow`. The reticle
 *     pointed at something invisible, in a row the user had not been asked for yet.
 *
 * Locking is the whole point of a guided capture: one row at a time, in the order the prompt
 * announces, and nothing else on screen or in the arithmetic.
 *
 * `rowStations` are stable arrays holding the same objects — see the ref's own note. Filtering
 * here would reset `updateStations`' dwell clock on every tick.
 */
function fillTargets(
  state: CaptureState,
  rows: StationState[],
  bonus: StationState[],
  rowStations: Record<RowId, StationState[]>,
): StationState[] {
  if (state === "bonus_poles") return bonus;
  return Plan.stationsForState(state, rowStations, rows);
}

/**
 * The pole roll gate, applied by zeroing dwell rather than by withholding stations.
 *
 * One frame at +90 covers an inscribed 21.5-deg cap; two rolled 90 deg apart cover 26.5. The
 * roll IS the second frame's whole purpose, and a station's great-circle distance cannot
 * express it. Zeroing dwell each tick keeps the array identity stable (see `fillTargets`)
 * while making it impossible for an unrolled frame to bank the rolled station.
 */
function applyPoleRollGate(bonus: StationState[], rollDeg: number): void {
  for (const station of bonus) {
    const required = BONUS_POLE_STATIONS.find((entry) => entry.id === station.id)?.rollDeg ?? 0;
    if (Math.abs(shortestAngle(rollDeg - required)) > POLE_ROLL_TOLERANCE_DEG) station.dwellMs = 0;
  }
}

/** What the bright ring may point at. Filtering is safe here — `nextTarget` is pure. */
function ringTargets(
  state: CaptureState,
  rows: StationState[],
  bonus: StationState[],
  touchUpIds: StationId[] | null,
  rollDeg: number,
  rowStations: Record<RowId, StationState[]>,
): StationState[] {
  if (state === "bonus_poles") {
    return bonus.filter((station) => {
      const required = BONUS_POLE_STATIONS.find((entry) => entry.id === station.id)?.rollDeg ?? 0;
      return Math.abs(shortestAngle(rollDeg - required)) <= POLE_ROLL_TOLERANCE_DEG;
    });
  }
  if (state === "touch_up" && touchUpIds && touchUpIds.length > 0) {
    const wanted = new Set<StationId>(touchUpIds);
    return rows.filter((station) => wanted.has(station.id));
  }
  // The same lock as `fillTargets`, and it has to be the same or the ring points at a
  // station the overlay is not drawing. `nextTarget` only PREFERS the active row; left with
  // the whole lattice it will happily cross into one the user has not been asked for.
  return Plan.stationsForState(state, rowStations, rows);
}

function allTargetsFilled(
  state: CaptureState,
  rows: StationState[],
  bonus: StationState[],
  touchUpIds: StationId[] | null,
): boolean {
  if (state === "bonus_poles") return bonus.length > 0 && bonus.every((station) => station.filled);
  if (!touchUpIds || touchUpIds.length === 0) return false;
  const wanted = new Set<StationId>(touchUpIds);
  return rows.filter((station) => wanted.has(station.id)).every((station) => station.filled);
}

/** Unwrapped yaw span and elevation extremes for the row currently being swept. */
function trackRow(sample: OrientationSample, row: RowId | null, tracks: Record<RowId, RowTrack>): void {
  if (!row) return;
  const track = tracks[row];
  if (track.lastYawDeg !== null) {
    // `accumulateYaw` owns the 0/360 seam and short reversals; P only supplies the pair.
    track.spanDeg = Plan.accumulateYaw(track.lastYawDeg, sample.yawDeg, track.spanDeg);
  }
  track.lastYawDeg = sample.yawDeg;

  if (!track.seen) {
    track.seen = true;
    track.pitchMinDeg = sample.pitchDeg;
    track.pitchMaxDeg = sample.pitchDeg;
  } else {
    track.pitchMinDeg = Math.min(track.pitchMinDeg, sample.pitchDeg);
    track.pitchMaxDeg = Math.max(track.pitchMaxDeg, sample.pitchDeg);
  }
}

/**
 * A still every 45 deg of the level row — enough for a review strip, cheap enough to ignore.
 *
 * `yawDeg` is null in RAIL MODE, where there is no heading to space them by; the strip then
 * falls back to a clock. An empty review strip is a review screen that cannot show the user
 * a single frame of what they just shot, which is the one thing it exists for.
 */
function maybeGrabThumbnail(
  state: CaptureState,
  now: number,
  loop: LoopState,
  yawDeg: number | null,
  camera: { grabStill: () => Promise<Blob | null> },
  store: RefObject<Blob[]>,
  enabled: boolean,
): void {
  // `enabled` is the governor's: a still costs a bitmap and a JPEG encode, and a phone that
  // is already dropping frames should spend that on the capture rather than on the review
  // strip. The strip degrades to whatever was grabbed before the tier changed.
  if (!enabled || state !== "row_level" || loop.thumbnailBusy || store.current.length >= THUMBNAIL_MAX) return;
  if (yawDeg === null) {
    if (now - loop.thumbnailAt < RAIL_THUMBNAIL_EVERY_MS) return;
  } else if (loop.thumbnailYaw !== null && Math.abs(shortestAngle(yawDeg - loop.thumbnailYaw)) < THUMBNAIL_EVERY_DEG) {
    return;
  }
  loop.thumbnailYaw = yawDeg;
  loop.thumbnailAt = now;
  loop.thumbnailBusy = true;
  void camera
    .grabStill()
    .then((blob) => {
      if (blob) store.current.push(blob);
    })
    .catch(() => undefined)
    .finally(() => {
      loop.thumbnailBusy = false;
    });
}

function rowFractions(stations: StationState[]): Record<RowId, number> {
  const out: Record<RowId, number> = { level: 0, up: 0, down: 0 };
  for (const row of ROW_IDS) {
    let filled = 0;
    for (const station of stations) if (station.row === row && station.filled) filled++;
    out[row] = Math.min(1, filled / ROW_PLAN[row].requiredStations);
  }
  return out;
}

const wrap360 = (deg: number): number => ((deg % 360) + 360) % 360;
const shortestAngle = (deg: number): number => wrap360(deg + 180) - 180;

/** The stations a touch-up pass should revisit: the unfilled ones sitting inside a gap. */
function gapStations(report: CoverageReport | null, stations: StationState[]): StationId[] {
  const unfilled = stations.filter((station) => !station.filled);
  if (!report || report.gaps.length === 0) return unfilled.map((station) => station.id);

  const wanted = new Set<StationId>();
  for (const gap of report.gaps) {
    const spacing = ROW_PLAN[gap.nearestRow].stationSpacingDeg;
    // One station of slack either side: a gap edge rarely lands on a station centre.
    const from = gap.yawStartDeg - spacing;
    const width = wrap360(gap.yawEndDeg + spacing - from);
    for (const station of unfilled) {
      if (station.row !== gap.nearestRow) continue;
      if (wrap360(station.yawDeg - from) <= width) wanted.add(station.id);
    }
  }
  // A report whose gaps map to no station would strand the user on an empty screen.
  return wanted.size > 0 ? [...wanted] : unfilled.map((station) => station.id);
}

/**
 * Builds the CaptureResult that the review screen grades and the upload sends.
 *
 * The verdict is entirely Builder C's, including `too_short`: `PlanApi.verdict` takes the
 * recorded duration as a third argument precisely because nothing in a `CoverageReport` can
 * stand in for validate.MIN_VIDEO_DURATION_S = 6.0 honestly. P supplies the duration and
 * makes no judgement of its own — one verdict ladder, in one place.
 */
function finalise(
  report: CoverageReport,
  segments: RecordedSegment[],
  detector: Detectors.CaptureDetector | null,
  telemetry: TelemetryApi | null,
  extras: { thumbnails: Blob[]; qualityMode: GuidedQualityMode; coverageMeasured: boolean },
): CaptureResult | null {
  if (!detector || !telemetry) return null;

  const totalMs = segments.reduce((total, segment) => total + segment.durationMs, 0);
  const verdict: CaptureVerdict = Plan.verdict(report, detector.summary(), totalMs, {
    coverageMeasured: extras.coverageMeasured,
  });

  return {
    segments: segments.slice(),
    telemetry: telemetry.build(report, segments, verdict),
    thumbnails: extras.thumbnails.slice(),
    verdict,
    qualityMode: extras.qualityMode,
    captureKind: "video",
  };
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function connectionInfo(): { effectiveType: string | null; saveData: boolean } {
  if (typeof navigator === "undefined") return { effectiveType: null, saveData: false };
  // `connection` is not in lib.dom and is absent on every iPhone, so it is read
  // structurally and treated as a hint rather than a fact.
  const connection = (navigator as Navigator & { connection?: { effectiveType?: string; saveData?: boolean } })
    .connection;
  return { effectiveType: connection?.effectiveType ?? null, saveData: connection?.saveData === true };
}

// =======================================================================================
// Small hooks
// =======================================================================================

/** Keeps a ref pointed at the latest render's value, for use inside long-lived closures. */
function useLatest<T>(value: T): RefObject<T> {
  const ref = useRef(value);
  ref.current = value;
  return ref;
}

/**
 * `prefers-reduced-motion`, read through `useSyncExternalStore` rather than a state-in-effect
 * pair: the media query IS the external store, the server snapshot is `false`, and the value
 * is correct on the very first client render instead of one cascading render later.
 */
const REDUCED_MOTION = "(prefers-reduced-motion: reduce)";

function subscribeReducedMotion(onChange: () => void): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return () => undefined;
  const query = window.matchMedia(REDUCED_MOTION);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

function readReducedMotion(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  return window.matchMedia(REDUCED_MOTION).matches;
}

function useReducedMotion(): boolean {
  return useSyncExternalStore(subscribeReducedMotion, readReducedMotion, () => false);
}

/** Nothing to subscribe to: neither the user agent nor `mediaDevices` changes in a session. */
function subscribeNothing(): () => void {
  return () => undefined;
}

function readInAppBrowser(): boolean {
  if (typeof navigator === "undefined" || !navigator.mediaDevices) return true;
  return /FBAN|FBAV|FB_IAB|Instagram|LinkedInApp|Line\/|MicroMessenger|Twitter/i.test(navigator.userAgent);
}

interface Feedback {
  unlock: () => void;
  tick: () => void;
  haptic: (pattern: readonly number[]) => void;
}

/**
 * Sound and haptics. Both are secondary channels by design — iOS Safari has no vibration
 * API at all — so neither is ever the only carrier of a message.
 */
function useFeedback(enabled: boolean): Feedback {
  const contextRef = useRef<AudioContext | null>(null);
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;

  const unlock = useCallback(() => {
    if (typeof window === "undefined" || typeof AudioContext === "undefined") return;
    try {
      // Created inside a user gesture, the only moment iOS lets an AudioContext start in
      // the "running" state.
      contextRef.current ??= new AudioContext();
      void contextRef.current.resume().catch(() => undefined);
    } catch {
      contextRef.current = null;
    }
  }, []);

  const tick = useCallback(() => {
    const context = contextRef.current;
    if (!enabledRef.current || !context || context.state !== "running") return;
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.frequency.value = 880;
    gain.gain.setValueAtTime(0.0001, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.08, context.currentTime + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.09);
    oscillator.connect(gain).connect(context.destination);
    oscillator.start();
    oscillator.stop(context.currentTime + 0.1);
  }, []);

  const haptic = useCallback((pattern: readonly number[]) => {
    if (typeof navigator === "undefined" || typeof navigator.vibrate !== "function" || pattern.length === 0) return;
    try {
      navigator.vibrate([...pattern]);
    } catch {
      // Some Androids throw while the page is not visible. Nothing to recover from.
    }
  }, []);

  return useMemo(() => ({ unlock, tick, haptic }), [unlock, tick, haptic]);
}

function speakLine(text: string, locale: Locale): void {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  try {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = locale === "tr" ? "tr-TR" : "en-GB";
    utterance.rate = 1.05;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  } catch {
    // A missing Turkish voice is common. Silence is acceptable: the visual channel was
    // designed to work alone.
  }
}

function cancelSpeech(): void {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  try {
    window.speechSynthesis.cancel();
  } catch {
    // Nothing to recover from.
  }
}

// =======================================================================================
// Screens
// =======================================================================================

/** The preview element, isolated so the 30 Hz publish loop can never re-render it. */
const CameraFeed = memo(function CameraFeed({ videoRef }: { videoRef: RefObject<HTMLVideoElement | null> }) {
  return (
    <video
      ref={videoRef}
      playsInline
      muted
      autoPlay
      // `object-cover` crops nothing for a portrait stream in a portrait viewport; it only
      // guards against a device handing back a slightly different aspect ratio.
      className="absolute inset-0 size-full object-cover"
    />
  );
});

/* The modal layers all carry z-30: the warning toast is positioned at z-20 so it can clear
   the chrome, and a sheet that owns the screen has to sit above it rather than behind it. */
function Veil({ locale }: { locale: Locale }) {
  return (
    <div className="absolute inset-0 z-30 grid place-items-center bg-canvas/85 backdrop-blur-sm">
      <p className="flex items-center gap-2.5 text-[15px] text-ink">
        <Loader2 className="size-4 animate-spin text-bio" aria-hidden />
        {cc("capture.check.working", locale)}
      </p>
    </div>
  );
}

function IntroScreen({
  locale,
  draft,
  voiceGuide,
  sounds,
  onVoiceGuide,
  onSounds,
  onStart,
  onUseExistingVideo,
}: {
  locale: Locale;
  draft: boolean;
  voiceGuide: boolean;
  sounds: boolean;
  onVoiceGuide: (value: boolean) => void;
  onSounds: (value: boolean) => void;
  onStart: () => void;
  onUseExistingVideo: () => void;
}) {
  return (
    <Sheet>
      <h1 className="text-[24px] leading-tight font-semibold tracking-tight text-ink">
        {cc(draft ? "capture.draft.title" : "capture.intro.title", locale)}
      </h1>
      <p className="mt-2.5 text-[14px] leading-relaxed text-ink-muted">
        {cc(draft ? "capture.draft.body" : "capture.intro.body", locale)}
      </p>

      <div className="mt-6 space-y-3 border-y border-line py-5">
        <Switch
          checked={voiceGuide}
          onChange={onVoiceGuide}
          label={cc("capture.voice", locale)}
          ariaLabel={cc("capture.voice", locale)}
        />
        <Switch
          checked={sounds}
          onChange={onSounds}
          label={cc("capture.sound", locale)}
          ariaLabel={cc("capture.sound", locale)}
        />
      </div>

      <Button variant="primary" size="lg" className="mt-6 w-full" onClick={onStart}>
        {cc("capture.intro.start", locale)}
      </Button>
      <QuietButton onClick={onUseExistingVideo}>{cc("capture.intro.have", locale)}</QuietButton>
    </Sheet>
  );
}

function StanceScreen({ locale, onReady, onBack }: { locale: Locale; onReady: () => void; onBack: () => void }) {
  return (
    <Sheet>
      <h1 className="text-[22px] leading-tight font-semibold tracking-tight text-ink">
        {cc("capture.stand.title", locale)}
      </h1>

      <StanceDiagram className="mx-auto mt-5 w-full max-w-[200px]" />

      <ol className="mt-5 space-y-3.5">
        {(["capture.stand.1", "capture.stand.2", "capture.stand.3"] as const).map((key, index) => (
          <li key={key} className="flex gap-3">
            <span className="mt-px grid size-6 shrink-0 place-items-center rounded-full border border-line bg-surface-2 font-mono text-[11px] text-bio">
              {index + 1}
            </span>
            <p className="text-[13.5px] leading-relaxed text-ink-muted">{cc(key, locale)}</p>
          </li>
        ))}
      </ol>

      <Button variant="primary" size="lg" className="mt-7 w-full" onClick={onReady}>
        {cc("capture.stand.ready", locale)}
      </Button>
      <QuietButton onClick={onBack}>{locale === "tr" ? "Geri" : "Back"}</QuietButton>
    </Sheet>
  );
}

/**
 * A top-down stance figure: feet planted, the phone turning on the spot. Drawn here rather
 * than reused from `dashboard/CaptureGuide.tsx`, whose `SweepDiagram` is module-private —
 * exporting it would mean editing a file this builder does not own.
 */
function StanceDiagram({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 200 150" className={className} role="img" aria-label="" fill="none">
      <circle cx="100" cy="78" r="54" stroke="var(--line)" strokeDasharray="2.5 6" />
      <circle
        cx="100"
        cy="78"
        r="54"
        stroke="var(--bio)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeDasharray="230 110"
        transform="rotate(-100 100 78)"
      />
      {/* The feet are the fixed point of the entire capture. */}
      <ellipse cx="92" cy="86" rx="5" ry="9" stroke="var(--ink-faint)" strokeWidth="1.3" />
      <ellipse cx="108" cy="86" rx="5" ry="9" stroke="var(--ink-faint)" strokeWidth="1.3" />
      <path d="M100 66 L78 26 A46 46 0 0 1 122 26 Z" fill="color-mix(in oklab, var(--bio) 18%, transparent)" />
      <rect x="94" y="58" width="12" height="19" rx="2.5" fill="var(--surface-3)" stroke="var(--line-strong)" />
      <path d="M96.5 62.5h7" stroke="var(--bio)" strokeWidth="1.4" strokeLinecap="round" />
      <path d="M148 78a48 48 0 0 1-8 26" stroke="var(--bio)" strokeWidth="1.5" strokeLinecap="round" />
      <path
        d="m137 100 3.5 5 5.5-2.5"
        stroke="var(--bio)"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function FramingCard({
  locale,
  gate,
  countdown,
  canForce,
  preflightFailed,
  onForce,
  onUseExistingVideo,
}: {
  locale: Locale;
  gate: FramingGate;
  countdown: number | null;
  canForce: boolean;
  preflightFailed: boolean;
  onForce: () => void;
  onUseExistingVideo: () => void;
}) {
  if (preflightFailed) {
    return (
      <div className="absolute inset-0 z-30 grid place-items-center bg-canvas/90 px-6 backdrop-blur-sm">
        <div className="glass max-w-[380px] rounded-lg p-6 text-center">
          <TriangleAlert className="mx-auto size-6 text-danger" aria-hidden />
          <p className="mt-3 text-[14px] leading-relaxed text-ink">
            {locale === "tr"
              ? "Bu tarayıcı video kaydedemiyor. Sayfayı Safari veya Chrome'da açın, ya da hazır bir video yükleyin."
              : "This browser cannot record video. Open the page in Safari or Chrome, or upload a video you already have."}
          </p>
          <Button variant="secondary" size="lg" className="mt-5 w-full" onClick={onUseExistingVideo}>
            {cc("capture.intro.have", locale)}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="pointer-events-none absolute inset-0 z-30 grid place-items-center px-6">
      <div className="glass-strong pointer-events-auto flex w-full max-w-[320px] flex-col items-center rounded-lg px-5 py-6 text-center">
        {countdown !== null ? (
          <>
            <p className="font-mono text-[54px] leading-none font-semibold text-bio tabular-nums">{countdown}</p>
            <p className="mt-2 text-[13px] text-ink-muted">{cc("capture.frame.starting", locale)}</p>
          </>
        ) : (
          <>
            <p className="text-[17px] font-semibold tracking-tight text-ink">
              {cc(gate ? FRAMING_COPY[gate] : "capture.frame.turnhere", locale)}
            </p>
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-muted">
              {locale === "tr"
                ? "Üç ölçüt de sağlanınca kayıt kendiliğinden başlar."
                : "Recording starts by itself once all three checks pass."}
            </p>
          </>
        )}

        {canForce && countdown === null ? (
          <Button variant="ghost" size="sm" className="mt-4" onClick={onForce}>
            {cc("capture.frame.force", locale)}
          </Button>
        ) : null}
      </div>
    </div>
  );
}

function PausedSheet({
  locale,
  coverage,
  coverageMeasured,
  budgetExhausted,
  onResume,
  onFinish,
  onRestart,
}: {
  locale: Locale;
  coverage: CoverageReport | null;
  coverageMeasured: boolean;
  budgetExhausted: boolean;
  onResume: () => void;
  onFinish: () => void;
  onRestart: () => void;
}) {
  // "Şimdi bitirilsin mi?" is only honest if it can say what finishing costs, so the count
  // is the actual number of stations still owed across the three rows. In rail mode nothing
  // counted stations, so there is no honest number and the block is not shown at all.
  const remaining =
    coverage && coverageMeasured
      ? ROW_IDS.reduce(
          (total, row) => total + Math.max(0, coverage.rows[row].requiredStations - coverage.rows[row].filledStations),
          0,
        )
      : 0;

  return (
    <div className="absolute inset-0 grid place-items-end bg-canvas/70 backdrop-blur-sm">
      <div className="glass-strong animate-rise w-full rounded-t-lg px-5 pt-6 pb-[max(env(safe-area-inset-bottom),1.25rem)]">
        <div className="mx-auto max-w-[420px]">
          <div className="flex items-start gap-3">
            <span className="grid size-9 shrink-0 place-items-center rounded-full border border-line bg-surface-3 text-ink-muted">
              <Pause className="size-4" aria-hidden />
            </span>
            <div className="min-w-0">
              <h2 className="text-[16px] font-semibold tracking-tight text-ink">{cc("capture.pause.title", locale)}</h2>
              <p className="mt-1 text-[13px] leading-relaxed text-ink-muted">{cc("capture.pause.body", locale)}</p>
            </div>
          </div>

          {budgetExhausted ? (
            // Resuming is deliberately not offered: the loop would end the capture again
            // within a frame, which is a button that does nothing. Every second recorded is
            // sealed and finishing from here uploads all of it.
            <div className="mt-4 rounded-md border border-amber/25 bg-amber/5 p-3.5">
              <p className="text-[13px] font-medium text-ink">
                {locale === "tr" ? "Kayıt süresi doldu" : "The recording time is up"}
              </p>
              <p className="mt-1 text-[12.5px] leading-relaxed text-ink-muted">
                {locale === "tr"
                  ? `Bu çekim ${Math.round(Plan.MAX_CAPTURE_MS / 60_000)} dakikaya ulaştı — daha uzunu yüklenemiyor. Çektiğiniz her saniye duruyor; şimdi bitirip yükleyebilirsiniz.`
                  : `This capture reached ${Math.round(Plan.MAX_CAPTURE_MS / 60_000)} minutes, which is as much as can be uploaded. Every second you shot is safe — finish here and send it.`}
              </p>
            </div>
          ) : remaining > 0 ? (
            <div className="mt-4 rounded-md border border-amber/25 bg-amber/5 p-3.5">
              <p className="text-[13px] font-medium text-ink">{cc("capture.early.title", locale)}</p>
              <p className="mt-1 text-[12.5px] leading-relaxed text-ink-muted">
                {cc("capture.early.body", locale, { n: remaining })}
              </p>
            </div>
          ) : null}

          <div className="mt-5 space-y-2.5">
            {budgetExhausted ? (
              <Button variant="primary" size="lg" className="w-full" onClick={onFinish}>
                <Check className="size-4" aria-hidden />
                {cc("capture.pause.finish", locale)}
              </Button>
            ) : (
              <Button variant="primary" size="lg" className="w-full" onClick={onResume}>
                <Play className="size-4" aria-hidden />
                {cc("capture.pause.resume", locale)}
              </Button>
            )}
            <div className="flex gap-2.5">
              {budgetExhausted ? null : (
                <Button variant="secondary" className="flex-1" onClick={onFinish}>
                  <Check className="size-4" aria-hidden />
                  {cc("capture.pause.finish", locale)}
                </Button>
              )}
              <Button variant="ghost" className="flex-1" onClick={onRestart}>
                <RotateCcw className="size-4" aria-hidden />
                {cc("capture.pause.restart", locale)}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * RAIL MODE's counter — the thing `PermissionGate` promises the user when the motion
 * sensor is refused ("halkalar yerine sayaç görürsünüz") and which, until now, did not
 * exist: the screen simply showed a frozen lattice and a row that could never end.
 *
 * It says what it is. A timer cannot tell anyone which directions they actually covered,
 * and copy that implied otherwise would be the same lie as a green verdict on a capture
 * nothing measured.
 */
function RailCounter({ locale, secondsLeft }: { locale: Locale; secondsLeft: number | null }) {
  return (
    <div className="glass mt-2 flex items-center gap-3 rounded-lg px-3.5 py-2.5">
      <p className="font-mono text-[26px] leading-none font-semibold text-bio tabular-nums">
        {secondsLeft === null ? "—" : secondsLeft}
        <span className="ml-0.5 text-[13px] text-ink-faint">{locale === "tr" ? "sn" : "s"}</span>
      </p>
      <p className="min-w-0 text-[11.5px] leading-snug text-ink-muted">
        {locale === "tr"
          ? "Yön sensörü kapalı: bu sayaç süreyle ilerler. Hangi yönleri çektiğinizi ölçemiyoruz — yavaş ve düzenli dönün, sayaç bitince sıra değişir."
          : "No orientation sensor: this counter runs on time alone. We cannot measure which directions you covered — turn slowly and evenly, and the row changes when it reaches zero."}
      </p>
    </div>
  );
}

/**
 * The recording clock, and the reason it exists: a capture now has a real ceiling, and a
 * ceiling nobody can see is a ceiling that arrives as a surprise. It counts UP for most of
 * the capture — the useful number while you are turning is how long you have been at it —
 * and switches to counting DOWN over the last minute, in amber, so the end is never news.
 */
function RecordingClock({ recordedMs, bytes, locale }: { recordedMs: number; bytes: number; locale: Locale }) {
  const clock = Plan.captureClock(recordedMs);
  const shown = clock.counting ? clock.remainingMs : clock.recordedMs;
  const seconds = Math.max(0, Math.round(shown / 1000));

  return (
    <div className="mb-2 flex items-baseline gap-2">
      <span
        className={cn(
          "font-mono text-[12px] font-medium tabular-nums",
          clock.counting ? "text-amber" : "text-ink-muted",
        )}
      >
        {clock.counting ? "−" : ""}
        {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, "0")}
      </span>
      <span className="font-mono text-[11px] text-ink-faint tabular-nums">{formatBytes(bytes)}</span>
      {clock.counting ? (
        <span className="text-[11px] text-amber">
          {locale === "tr" ? "kayıt süresi bitiyor" : "recording time is running out"}
        </span>
      ) : null}
    </div>
  );
}

function RotatePrompt({ locale }: { locale: Locale }) {
  return (
    <div className="absolute inset-0 z-30 grid place-items-center bg-canvas/92 px-8 text-center backdrop-blur-sm">
      <div>
        <Smartphone className="mx-auto size-8 rotate-90 text-bio" strokeWidth={1.5} aria-hidden />
        <h2 className="mt-4 text-[18px] font-semibold tracking-tight text-ink">{cc("capture.rotate.title", locale)}</h2>
        <p className="mt-2 text-[13.5px] leading-relaxed text-ink-muted">{cc("capture.rotate.body", locale)}</p>
      </div>
    </div>
  );
}

function UploadScreen({
  locale,
  failed,
  progress,
  error,
  bytes,
  onRetry,
  onDownload,
  onBack,
}: {
  locale: Locale;
  failed: boolean;
  progress: number;
  error: string | null;
  bytes: number;
  onRetry: () => void;
  onDownload: () => void;
  onBack: () => void;
}) {
  const connection = connectionInfo();
  const slow =
    connection.saveData ||
    (connection.effectiveType !== null && UPLOAD_POLICY.slowEffectiveTypes.includes(connection.effectiveType));

  return (
    <Sheet>
      {failed ? (
        <>
          <TriangleAlert className="size-6 text-danger" aria-hidden />
          <h1 className="mt-3 text-[19px] font-semibold tracking-tight text-ink">
            {locale === "tr" ? "Yükleme tamamlanamadı" : "The upload did not finish"}
          </h1>
          <p className="mt-2 text-[13.5px] leading-relaxed text-ink-muted">
            {error ??
              (locale === "tr"
                ? "Video telefonunuzda duruyor; tekrar denemek yeniden çekmek anlamına gelmiyor."
                : "The video is still on your phone; retrying does not mean re-shooting.")}
          </p>
          <div className="mt-6 space-y-2.5">
            <Button variant="primary" size="lg" className="w-full" onClick={onRetry}>
              <RotateCcw className="size-4" aria-hidden />
              {cc("capture.upload.retry", locale)}
            </Button>
            {/* The escape hatch belongs on the screen that needs it. "The video is still on
                your phone" is true only while this tab lives, and if the engine is simply
                unreachable — wrong host, phone on mobile data — Retry fails identically
                every time. Making the user find their way back to review to save it is
                asking them to guess, with 150 MB of their afternoon riding on the answer. */}
            <Button variant="secondary" size="lg" className="w-full" onClick={onDownload}>
              <Download className="size-4" aria-hidden />
              {cc("capture.save.local", locale)}
            </Button>
            <Button variant="ghost" size="lg" className="w-full" onClick={onBack}>
              {locale === "tr" ? "Geri" : "Back"}
            </Button>
          </div>
        </>
      ) : (
        <>
          <h1 className="text-[19px] font-semibold tracking-tight text-ink">{cc("capture.upload.title", locale)}</h1>
          <p className="mt-1.5 font-mono text-[13px] text-ink-muted tabular-nums">
            {Math.round(progress * 100)}% · {formatBytes(bytes)}
          </p>
          <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
            <div
              className="h-full rounded-full bg-[linear-gradient(90deg,var(--bio),var(--bio-soft))] transition-[width] duration-300 ease-out-expo"
              style={{ width: `${Math.round(progress * 100)}%` }}
              aria-hidden
            />
          </div>
          {slow ? (
            <p className="mt-4 text-[12.5px] leading-relaxed text-amber">
              {cc("capture.upload.slowlink", locale, { n: Math.max(1, Math.round(bytes / 1_000_000)) })}
            </p>
          ) : null}
          <p className="mt-4 flex items-center gap-2 text-[12px] text-ink-faint">
            <Loader2 className="size-3.5 animate-spin" aria-hidden />
            {locale === "tr" ? "Bu sayfayı açık bırakın." : "Keep this page open."}
          </p>
        </>
      )}
    </Sheet>
  );
}

function Sheet({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid-bg flex min-h-[100dvh] w-full items-center justify-center px-5 py-10">
      <div className="animate-rise glass w-full max-w-[420px] rounded-lg p-6 sm:p-7">{children}</div>
    </div>
  );
}

function QuietButton({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="mt-3 w-full py-2 text-[13px] text-ink-faint transition-colors hover:text-ink-muted"
    >
      {children}
    </button>
  );
}
