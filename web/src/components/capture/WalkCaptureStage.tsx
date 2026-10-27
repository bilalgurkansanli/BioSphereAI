"use client";

/**
 * Recording one AREA of a 3D walkthrough on the phone: the general walkthrough, or a room.
 *
 * Deliberately not `CaptureStage`. That component is a machine built to enforce a rotation
 * about one point (stations, rows, coverage rings, a walking detector that stops the
 * capture); a walkthrough is the opposite motion, and threading a mode through three
 * thousand lines that assume the other would have left both worse. What is shared is the
 * plumbing under it: `useCamera` (now with an ultra-wide preference), `useRecorder`
 * (segments, watchdogs, the mime ladder) and `useMotion` (the turning-speed coach).
 *
 * The screens follow SplatTour's recording guide: rules, then a live preview with the pass
 * being walked (rooms: perimeter one way, perimeter the other way, cross, details; the
 * general walkthrough: one continuous pass), a turning-speed meter that warns on fast
 * pans, then a review with duration and size before the upload.
 */

import { Camera, Check, ChevronRight, RotateCcw, Square, TriangleAlert, Video } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ui, type UiKey } from "@/components/dashboard/copy";
import { Button } from "@/components/ui/Button";
import { useCamera } from "@/lib/capture/useCamera";
import { useMotion } from "@/lib/capture/useMotion";
import { useRecorder } from "@/lib/capture/recorder";
import type { RecordedSegment } from "@/lib/capture/types";
import type { Locale, QualityMode } from "@/lib/types";
import { cn } from "@/lib/utils";
import { BLANK_WALL_SCORE, BLANK_WALL_SUSTAIN_MS, TEXTURE_THUMB_WIDTH, latch, textureScore, type Latch } from "./WalkCaptureGuards";

export interface WalkArea {
  id: string;
  kind: "general" | "room";
  name: string;
}

export interface WalkCaptureStageProps {
  area: WalkArea;
  locale: Locale;
  quality: QualityMode;
  /** The recording, as files ready for the walkthrough upload. */
  onRecorded: (files: File[], seconds: number) => void;
  onCancel: () => void;
}

type Phase = "intro" | "opening" | "ready" | "recording" | "review";

/** Suggested length of each room pass, seconds; a small room finishes early by tapping "next". */
const PASS_SECONDS = [50, 50, 35, 45];
const MIN_SECONDS = 60;
/** |rotationRate| above which a pan blurs the frames the solver needs (deg/s). */
const TURN_WARN_DEG_PER_S = 40;
const TURN_WARN_SUSTAIN_MS = 600;
// White balance is the one lock every platform honours and the one drift the trainer
// cannot undo; exposure is handled by the shutter cap where the phone allows it.
const WALK_CONTROLS = { lockWhiteBalance: true } as const;
/** How often the preview is scored for detail (see WalkCaptureGuards). */
const TEXTURE_EVERY_MS = 300;
/** The bare-wall warning; kept with its guard rather than in the dashboard copy. */
const BLANK_WALL_COPY = {
  title: { tr: "Boş duvar", en: "Bare wall" },
  body: {
    tr: "Telefonu odaya çevirin: köşe, kapı ya da pencere görünsün. Düz duvarda 3B kayıt izini kaybeder.",
    en: "Point the phone back into the room so a corner, a door or a window shows. On a bare wall the 3D recording loses track.",
  },
} as const;

export function WalkCaptureStage({ area, locale, quality, onRecorded, onCancel }: WalkCaptureStageProps) {
  const [phase, setPhase] = useState<Phase>("intro");
  // Bumped to (re)open the camera: by the rules screen's button, and by a track that ended
  // mid-walk (the recorder seals its segment on the same event; the next opens on the new
  // stream, so the loss is a join in the footage, not the capture).
  const [attempt, setAttempt] = useState(0);
  const [pass, setPass] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [tooFast, setTooFast] = useState(false);
  const [landscape, setLandscape] = useState(false);
  const [blankWall, setBlankWall] = useState(false);
  const thumbRef = useRef<HTMLCanvasElement | null>(null);
  const [segments, setSegments] = useState<RecordedSegment[]>([]);
  const segmentsRef = useRef<RecordedSegment[]>([]);
  const recordingRef = useRef(false);
  const startedAtRef = useRef(0);
  const accumulatedMsRef = useRef(0);

  const camera = useCamera({
    qualityMode: quality === "studio" ? "studio" : "balanced",
    lensPreference: "ultrawide",
    controls: WALK_CONTROLS,
    onTrackEnded: () => setAttempt((a) => a + 1),
  });
  // Plain values for render; `camera` itself carries a ref (its video element) and the
  // compiler rightly refuses to see it read during render.
  const { videoRef, stream: cameraStream, label: cameraLabel, virtualLensOnly, ready: cameraReady, error: cameraError } = camera;
  // "Latest" refs for the callbacks below, written after render (never during it).
  const cameraRef = useRef(camera);
  useEffect(() => {
    cameraRef.current = camera;
  });

  const recorder = useRecorder({
    stream: cameraStream,
    dimensions: () => cameraRef.current.size ?? null,
    onSegment: (segment) => {
      segmentsRef.current = [...segmentsRef.current, segment];
      setSegments(segmentsRef.current);
      accumulatedMsRef.current += segment.durationMs;
      // The recorder seals a segment at its own ceiling; the walk goes on in the next one.
      if (recordingRef.current && segment.endedBy === "max_duration") {
        startedAtRef.current = 0; // the clock loop restamps it on its next tick
        void recorderRef.current.start();
      }
    },
  });
  const recorderRef = useRef(recorder);
  useEffect(() => {
    recorderRef.current = recorder;
  });

  const motion = useMotion({ stateHz: 0 });
  const motionRef = useRef(motion);
  useEffect(() => {
    motionRef.current = motion;
  });

  // Open the camera when the user leaves the rules screen; close it on unmount.
  useEffect(() => {
    if (attempt === 0 || (phase !== "opening" && phase !== "ready" && phase !== "recording")) return;
    let cancelled = false;
    void cameraRef.current.open().then(() => {
      // The hook's error state lands with its own render; a failed open shows as the rules
      // screen again through `shown` below, so only success moves the phase here.
      if (!cancelled) setPhase((p) => (p === "opening" ? "ready" : p));
    });
    return () => {
      cancelled = true;
    };
    // `phase` is read, not depended on: a phase change must not reopen the camera.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt]);
  useEffect(() => () => cameraRef.current.close(), []);
  // A camera that failed to open sends the user back to the rules with the reason.
  const shown: Phase = (phase === "opening" || phase === "ready") && cameraError ? "intro" : phase;

  // Orientation, turning speed and how much detail the view holds, from one loop at ~10 Hz.
  useEffect(() => {
    if (phase !== "ready" && phase !== "recording") return;
    let fastSince = 0;
    let lastTexture = 0;
    const blank: Latch = { since: null };
    const id = window.setInterval(() => {
      setLandscape(window.matchMedia?.("(orientation: landscape)").matches ?? false);
      const omega = motionRef.current.read()?.omegaDegPerS ?? 0;
      const now = performance.now();
      if (now - lastTexture >= TEXTURE_EVERY_MS) {
        lastTexture = now;
        const score = previewTexture(videoRef.current, thumbRef);
        setBlankWall(latch(blank, score !== null && score < BLANK_WALL_SCORE, now, BLANK_WALL_SUSTAIN_MS));
      }
      if (recordingRef.current && !startedAtRef.current) startedAtRef.current = now;
      if (omega > TURN_WARN_DEG_PER_S) {
        if (!fastSince) fastSince = now;
        if (now - fastSince > TURN_WARN_SUSTAIN_MS) setTooFast(true);
      } else {
        fastSince = 0;
        setTooFast(false);
      }
      if (recordingRef.current) setElapsedMs(accumulatedMsRef.current + (now - startedAtRef.current));
    }, 100);
    return () => window.clearInterval(id);
    // `videoRef` is a ref: read inside the loop, never a reason to restart it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  const start = useCallback(async () => {
    segmentsRef.current = [];
    setSegments([]);
    accumulatedMsRef.current = 0;
    setPass(0);
    recordingRef.current = true;
    startedAtRef.current = 0; // stamped by the clock loop
    await recorderRef.current.start();
    if (recorderRef.current.state !== "recording") {
      recordingRef.current = false;
      return;
    }
    setPhase("recording");
  }, []);

  const finish = useCallback(async () => {
    recordingRef.current = false;
    await recorderRef.current.stop("user_finish");
    setPhase("review");
  }, []);

  const retake = useCallback(() => {
    segmentsRef.current = [];
    setSegments([]);
    accumulatedMsRef.current = 0;
    setElapsedMs(0);
    setPhase("ready");
  }, []);

  const totalMs = useMemo(() => segments.reduce((n, s) => n + s.durationMs, 0), [segments]);
  const totalBytes = useMemo(() => segments.reduce((n, s) => n + s.blob.size, 0), [segments]);

  const accept = () => {
    const ext = segments[0]?.mimeType.includes("mp4") ? "mp4" : "webm";
    const slug = area.kind === "general" ? "general" : area.name.replace(/[^\p{L}\p{N}]+/gu, "_").slice(0, 40) || area.id;
    const files = segments.map((s, i) => new File([s.blob], `${slug}-${i + 1}.${ext}`, { type: s.mimeType }));
    onRecorded(files, totalMs / 1000);
  };

  const passKey = (n: number, cue = false): UiKey => `walkcap.pass.${n + 1}${cue ? ".cue" : ""}` as UiKey;
  const t = (key: UiKey) => ui(key, locale);

  // ---------------------------------------------------------------- screens ----
  if (shown === "intro" || shown === "opening") {
    return (
      <div className="mx-auto flex min-h-[100dvh] w-full max-w-md flex-col justify-center gap-5 px-5 py-8">
        <div>
          <p className="text-[10px] font-semibold tracking-[0.18em] text-ink-faint uppercase">{t("walkcap.title")}</p>
          <h1 className="mt-1 text-[20px] font-semibold tracking-tight text-ink">
            {area.kind === "general" ? t("walkcap.plan.general") : area.name}
          </h1>
          <p className="mt-1 text-[12px] text-ink-muted">{area.kind === "general" ? t("walkcap.plan.general.hint") : t("walkcap.plan.room.hint")}</p>
        </div>
        <ol className="space-y-2.5 text-[13.5px] leading-relaxed text-ink">
          {(["walkcap.rules.1", "walkcap.rules.2", "walkcap.rules.3", "walkcap.rules.4"] as UiKey[]).map((k, i) => (
            <li key={k} className="flex gap-3">
              <span className="grid size-6 shrink-0 place-items-center rounded-full border border-bio/40 bg-bio/10 font-mono text-[11px] text-bio">{i + 1}</span>
              <span>{t(k)}</span>
            </li>
          ))}
        </ol>
        {cameraError ? (
          <p className="flex items-center gap-2 text-[12px] text-danger">
            <TriangleAlert className="size-4" aria-hidden />
            {t("walkcap.cameraError")} ({cameraError})
          </p>
        ) : null}
        <div className="flex gap-2">
          <Button variant="ghost" onClick={onCancel}>
            {ui("common.back", locale)}
          </Button>
          <Button
            variant="primary"
            size="lg"
            className="flex-1"
            loading={shown === "opening"}
            onClick={() => {
              setAttempt((a) => a + 1);
              setPhase("opening");
            }}
          >
            <Camera className="size-4" aria-hidden />
            {ui("common.continue", locale)}
          </Button>
        </div>
      </div>
    );
  }

  if (shown === "review") {
    const short = totalMs < MIN_SECONDS * 1000;
    return (
      <div className="mx-auto flex min-h-[100dvh] w-full max-w-md flex-col justify-center gap-5 px-5 py-8">
        <div className="glass rounded-lg p-5">
          <div className="flex items-center gap-3">
            <span className="grid size-10 place-items-center rounded-full border border-success/30 bg-success/10 text-success">
              <Check className="size-5" aria-hidden />
            </span>
            <div>
              <h1 className="text-[17px] font-semibold tracking-tight text-ink">{t("walkcap.review.title")}</h1>
              <p className="text-[12px] text-ink-muted">{area.kind === "general" ? t("walkcap.plan.general") : area.name}</p>
            </div>
          </div>
          <dl className="mt-4 grid grid-cols-3 gap-3 text-center">
            <Stat label={t("walkcap.elapsed")} value={fmt(totalMs)} />
            <Stat label="MB" value={(totalBytes / 1e6).toFixed(0)} />
            <Stat label={ui("walk.stat.areas", locale)} value={String(segments.length)} />
          </dl>
          {short ? <p className="mt-3 text-[12px] text-amber">{t("walkcap.review.short")}</p> : null}
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={retake}>
            <RotateCcw className="size-4" aria-hidden />
            {t("walkcap.review.retake")}
          </Button>
          <Button variant="primary" size="lg" className="flex-1" disabled={segments.length === 0} onClick={accept}>
            {t("walkcap.review.upload")}
            <ChevronRight className="size-4" aria-hidden />
          </Button>
        </div>
      </div>
    );
  }

  // ready / recording: the live preview with the coaching on top.
  const recording = shown === "recording";
  const isRoom = area.kind === "room";
  const passTarget = isRoom ? PASS_SECONDS[pass] * 1000 : 0;
  const passStart = isRoom ? PASS_SECONDS.slice(0, pass).reduce((n, s) => n + s * 1000, 0) : 0;
  const passFraction = isRoom ? Math.min(1, Math.max(0, (elapsedMs - passStart) / Math.max(passTarget, 1))) : 0;

  return (
    <div data-live-camera className="relative h-[100dvh] w-full overflow-hidden bg-black">
      <video ref={videoRef} playsInline muted autoPlay className="absolute inset-0 h-full w-full object-cover" />

      {/* Top: what to do now. */}
      <div className="absolute inset-x-0 top-0 z-10 bg-gradient-to-b from-black/70 to-transparent px-4 pt-[max(0.75rem,env(safe-area-inset-top))] pb-10">
        <p className="text-[10px] font-semibold tracking-[0.18em] text-white/60 uppercase">
          {area.kind === "general" ? t("walkcap.plan.general") : area.name}
        </p>
        {isRoom ? (
          <>
            <p className="mt-1 text-[16px] font-semibold text-white">{t(passKey(pass))}</p>
            <p className="mt-0.5 text-[12.5px] leading-snug text-white/80">{t(passKey(pass, true))}</p>
            <div className="mt-2 flex gap-1">
              {PASS_SECONDS.map((_, i) => (
                <span key={i} className="h-1 flex-1 overflow-hidden rounded-full bg-white/20">
                  <span className="bio-gradient block h-full" style={{ width: `${i < pass ? 100 : i === pass ? passFraction * 100 : 0}%` }} />
                </span>
              ))}
            </div>
          </>
        ) : (
          <p className="mt-1 text-[13px] leading-snug text-white/85">{t("walkcap.general.cue")}</p>
        )}
      </div>

      {/* Warnings. */}
      {landscape ? (
        <div className="absolute inset-x-4 top-1/3 z-20 rounded-lg border border-amber/40 bg-black/70 p-4 text-center text-[13px] text-amber">{t("walkcap.landscape")}</div>
      ) : tooFast ? (
        <div className="absolute inset-x-4 top-1/3 z-20 rounded-lg border border-danger/40 bg-black/70 p-4 text-center">
          <p className="text-[14px] font-semibold text-danger">{t("walkcap.tooFast")}</p>
          <p className="mt-1 text-[12px] text-white/80">{t("walkcap.tooFast.body")}</p>
        </div>
      ) : blankWall ? (
        <div className="absolute inset-x-4 top-1/3 z-20 rounded-lg border border-amber/40 bg-black/70 p-4 text-center">
          <p className="text-[14px] font-semibold text-amber">{BLANK_WALL_COPY.title[locale]}</p>
          <p className="mt-1 text-[12px] text-white/80">{BLANK_WALL_COPY.body[locale]}</p>
        </div>
      ) : null}

      {/* Bottom: clock and the controls. */}
      <div className="absolute inset-x-0 bottom-0 z-10 bg-gradient-to-t from-black/75 to-transparent px-4 pt-10 pb-[max(1rem,env(safe-area-inset-bottom))]">
        <div className="mb-3 flex items-center justify-between text-white/80">
          <span className={cn("flex items-center gap-2 font-mono text-[13px] tabular-nums", recording ? "text-white" : "")}>
            <span className={cn("size-2 rounded-full", recording ? "animate-breathe bg-danger" : "bg-white/40")} aria-hidden />
            {fmt(elapsedMs)}
          </span>
          <span className="text-[11px]">
            {cameraLabel ? cameraLabel.replace(/camera/i, "").trim() : ""}
            {virtualLensOnly ? ` · ${t("walkcap.setup.lens.fallback")}` : ` · ${t("walkcap.setup.lens")}`}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {!recording ? (
            <>
              <Button variant="ghost" className="text-white" onClick={onCancel}>
                {ui("common.back", locale)}
              </Button>
              <Button variant="primary" size="lg" className="flex-1" disabled={!cameraReady || landscape} onClick={() => void start()}>
                <Video className="size-4" aria-hidden />
                {t("walkcap.record")}
              </Button>
            </>
          ) : (
            <>
              {isRoom && pass < PASS_SECONDS.length - 1 ? (
                <Button variant="secondary" size="lg" className="flex-1" onClick={() => setPass((p) => Math.min(p + 1, PASS_SECONDS.length - 1))}>
                  {t("walkcap.nextPass")}
                  <ChevronRight className="size-4" aria-hidden />
                </Button>
              ) : null}
              <Button variant="danger" size="lg" className={isRoom && pass < PASS_SECONDS.length - 1 ? "" : "flex-1"} onClick={() => void finish()}>
                <Square className="size-4" aria-hidden />
                {t("walkcap.stop")}
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-line bg-surface-2/40 px-2 py-2">
      <dt className="text-[10px] tracking-[0.12em] text-ink-faint uppercase">{label}</dt>
      <dd className="mt-0.5 font-mono text-[15px] text-ink tabular-nums">{value}</dd>
    </div>
  );
}

function fmt(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/**
 * The preview's detail score (WalkCaptureGuards.textureScore) on a 96-px-wide thumbnail, or
 * null while the video has no frame. One drawImage and one small readback per call.
 */
function previewTexture(video: HTMLVideoElement | null, canvasRef: { current: HTMLCanvasElement | null }): number | null {
  if (!video || video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) return null;
  const w = TEXTURE_THUMB_WIDTH;
  const h = Math.max(8, Math.round((video.videoHeight * w) / video.videoWidth));
  const canvas = canvasRef.current ?? document.createElement("canvas");
  canvasRef.current = canvas;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  try {
    ctx.drawImage(video, 0, 0, w, h);
    return textureScore(ctx.getImageData(0, 0, w, h).data, w, h);
  } catch {
    return null; // a tainted or lost frame says nothing about the wall
  }
}
