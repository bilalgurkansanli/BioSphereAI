"use client";

/**
 * The 3D walkthrough viewer: a splat renderer with the chrome a visitor needs.
 *
 * Layout, top to bottom: a title strip (and the way back to the 360 view when the property has
 * one); the world; a bottom rail with the floor plan tile and a thumbnail per room; a
 * cluster of controls on the right (auto tour, walk / free flight, controls help,
 * fullscreen). Overlays: a loading veil with real byte progress, the floor plan with
 * clickable hotspots and a you-are-here mark, the controls legend.
 *
 * Every interactive overlay carries `data-walk-ui`; the engine's input handlers skip
 * events that start on one, which is what keeps a click on a thumbnail from also turning
 * the camera.
 */

import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import {
  Compass,
  Footprints,
  Keyboard,
  Map as MapIcon,
  Maximize,
  Minimize,
  Pause,
  Play,
  RotateCcw,
  X,
} from "lucide-react";
import { absoluteUrl } from "@/lib/api";
import { t } from "@/lib/i18n";
import type { Locale, WalkManifest } from "@/lib/types";
import { cn, formatBytes } from "@/lib/utils";
import { useWalkEngine, type WalkMode } from "./useWalkEngine";
import { drawableAreas, planFraction, planPoint, type Vec3 } from "./walkMath";
import { useWalkVr } from "@/components/vr/walk/useWalkVr";
import { VrWalkButton } from "@/components/vr/walk/VrWalkButton";
import type { WalkVrCopy } from "@/components/vr/walk/walkXr";

/** VR, fetched only where it can run: see `components/vr/walk/useWalkVr.ts`. */
const WalkVrRuntime = lazy(() => import("@/components/vr/walk/WalkVrRuntime"));
const VrOverlays = lazy(() => import("@/components/vr/ui/VrOverlays"));

export interface WalkViewerProps {
  manifest: WalkManifest;
  title?: string;
  locale?: Locale;
  /** Area key to open in; the general walkthrough's entrance otherwise. */
  startKey?: string | null;
  /** Shown top-left; the caller decides what "back to 360" means. */
  onSwitchTo360?: () => void;
  embedded?: boolean;
  className?: string;
}

const CHROME_HIDE_MS = 3500;

const subscribeNever = () => () => undefined;
const readCoarsePointer = () => window.matchMedia?.("(pointer: coarse)").matches ?? false;
const readPerfFlag = () => new URLSearchParams(window.location.search).get("perf") === "1";

export function WalkViewer({ manifest, title, locale = "tr", startKey = null, onSwitchTo360, embedded, className }: WalkViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fadeRef = useRef<HTMLDivElement | null>(null);
  const markerRef = useRef<HTMLDivElement | null>(null);
  const showPerf = useSyncExternalStore(subscribeNever, readPerfFlag, () => false);
  const [progress, setProgress] = useState({ loaded: 0, total: 1, ready: false });
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [mode, setMode] = useState<WalkMode>("walk");
  const [touring, setTouring] = useState(false);
  const [planOpen, setPlanOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [unsupported, setUnsupported] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [chromeVisible, setChromeVisible] = useState(true);
  const hideTimer = useRef<number | null>(null);
  // Client-only facts, read through an external-store subscription rather than a mount
  // effect: the server render sees the fallbacks, the first client render the truth.
  const mounted = useSyncExternalStore(subscribeNever, () => true, () => false);
  const coarse = useSyncExternalStore(subscribeNever, readCoarsePointer, () => false);

  const poke = useCallback(() => {
    setChromeVisible(true);
    if (hideTimer.current) window.clearTimeout(hideTimer.current);
    hideTimer.current = window.setTimeout(() => setChromeVisible(false), CHROME_HIDE_MS);
  }, []);
  useEffect(() => {
    // The chrome starts visible; only the hide timer needs arming here.
    hideTimer.current = window.setTimeout(() => setChromeVisible(false), CHROME_HIDE_MS);
    return () => {
      if (hideTimer.current) window.clearTimeout(hideTimer.current);
    };
  }, []);

  const plan = manifest.floor_plan;
  const events = useMemo(
    () => ({
      onProgress: (loaded: number, total: number, ready: boolean) => setProgress({ loaded, total, ready }),
      onArea: (key: string) => setActiveKey(key),
      onError: (message: string) => {
        if (message === "webgl2") setUnsupported(true);
        else setError(message);
      },
      onAutoTourEnd: () => setTouring(false),
      onUserInput: () => {
        setTouring(false);
        poke();
      },
      onFrame: (position: Vec3, yaw: number) => {
        const el = markerRef.current;
        if (!el || !plan) return;
        const f = planFraction(plan, position);
        el.style.left = `${(f.x * 100).toFixed(3)}%`;
        el.style.top = `${(f.y * 100).toFixed(3)}%`;
        // The cone points where the visitor looks. yaw 0 looks along -Z, which is UP the plan.
        el.style.transform = `translate(-50%, -50%) rotate(${(-yaw * 180) / Math.PI}deg)`;
      },
    }),
    [plan, poke],
  );

  const engine = useWalkEngine({
    manifest,
    startKey,
    container: containerRef,
    canvas: canvasRef,
    fade: fadeRef,
    events,
    enabled: mounted && !unsupported,
  });

  const areas = useMemo(() => drawableAreas(manifest), [manifest]);
  const rooms = useMemo(() => manifest.rooms.filter((r) => !r.failed && r.url), [manifest]);
  // The base is listed too: an unplaced part opened by name is pinned until another area is
  // opened by name, and the general walkthrough is the way back to it.
  const railAreas = areas;

  // VR: a headset, or a phone in a VR box. In the box the 2D chrome steps aside entirely.
  const getVrContainer = useCallback(() => containerRef.current, []);
  const vrMessages = useMemo(
    () => ({
      denied: t("vr.denied", locale),
      noSensor: t("vr.noSensor", locale),
      failed: t("vr.failed", locale),
      unavailable: t("vr.walk.unavailable", locale),
    }),
    [locale],
  );
  const vr = useWalkVr({ engine, container: getVrContainer, messages: vrMessages });
  const vrCopy = useMemo<WalkVrCopy>(
    () => ({
      exit: t("vr.exit", locale),
      smooth: t("vr.walk.smooth", locale),
      areaName: (key) => {
        const area = areas.find((a) => a.key === key);
        if (!area) return title ?? t("walk.title", locale);
        return area.key === manifest.base.key && rooms.length ? t("walk.general", locale) : area.name;
      },
    }),
    [locale, areas, rooms, manifest.base.key, title],
  );
  const inCardboard = vr.cardboard.phase === "starting" || vr.cardboard.phase === "active";

  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void el.requestFullscreen?.();
  }, []);
  useEffect(() => {
    const onChange = () => setFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const selectMode = (m: WalkMode) => {
    setMode(m);
    engine.current?.setMode(m);
    poke();
  };
  const toggleTour = () => {
    const next = !touring;
    setTouring(next);
    engine.current?.setAutoTour(next);
    poke();
  };
  const goTo = (key: string) => {
    setPlanOpen(false);
    setTouring(false);
    engine.current?.setAutoTour(false);
    engine.current?.goToArea(key);
    poke();
  };
  const onPlanClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!plan) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setPlanOpen(false);
    engine.current?.teleport(planPoint(plan, x, y));
  };

  const percent = Math.round(Math.min(1, progress.loaded / Math.max(progress.total, 1)) * 100);
  const activeArea = areas.find((a) => a.key === activeKey) ?? null;

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      role="application"
      aria-label={t("walk.title", locale)}
      onPointerMove={poke}
      className={cn(
        "no-tap relative isolate h-full w-full overflow-hidden bg-canvas select-none outline-none",
        "focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-bio",
        className,
        // A phone in a VR box needs the whole screen; where element fullscreen does not exist
        // (iPhone) the viewer covers the viewport itself.
        inCardboard && !vr.cardboard.fullscreen && "fixed inset-0 z-[2147483000] h-[100dvh] w-screen",
      )}
    >
      <canvas ref={canvasRef} className="absolute inset-0 block h-full w-full touch-none" />
      {showPerf ? <PerfBadge /> : null}

      {/* Teleport fade. */}
      <div ref={fadeRef} className="pointer-events-none absolute inset-0 z-20 bg-canvas opacity-0" />

      {/* Title strip. */}
      <div
        data-walk-ui
        className={cn(
          "absolute inset-x-0 top-0 z-30 flex items-start justify-between gap-3 px-3 pt-[max(0.75rem,env(safe-area-inset-top))] transition-opacity duration-300",
          chromeVisible ? "opacity-100" : "opacity-0",
          inCardboard && "hidden",
        )}
      >
        <div className="glass-strong flex max-w-[70%] items-center gap-2 rounded-full py-1.5 pr-3.5 pl-3">
          <span className="h-2 w-2 shrink-0 rounded-full bg-bio shadow-[var(--shadow-glow)]" />
          <span className="truncate text-[12px] font-medium tracking-tight text-ink">{title ?? t("walk.title", locale)}</span>
          {activeArea ? (
            <span className="truncate text-[11px] text-ink-muted">· {activeArea.key === manifest.base.key && rooms.length ? t("walk.general", locale) : activeArea.name}</span>
          ) : null}
        </div>
        {onSwitchTo360 ? (
          <button
            type="button"
            onClick={onSwitchTo360}
            className="glass-strong flex h-8 items-center gap-1.5 rounded-full border border-line px-3 text-[12px] font-medium text-ink-muted transition hover:border-line-strong hover:text-ink"
          >
            <RotateCcw size={13} />
            {t("walk.to360", locale)}
          </button>
        ) : null}
      </div>

      {/* Bottom rail: plan tile + rooms. */}
      <div
        data-walk-ui
        className={cn(
          "absolute inset-x-0 bottom-0 z-30 flex items-end justify-between gap-3 px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] transition-opacity duration-300",
          chromeVisible ? "opacity-100" : "opacity-0 pointer-events-none",
          inCardboard && "hidden",
        )}
      >
        <div className="glass-strong flex max-w-[calc(100%-11rem)] gap-1.5 overflow-x-auto rounded-[var(--radius-lg)] p-1.5 [scrollbar-width:none]">
          {plan?.url ? (
            <RailTile
              label={t("walk.plan", locale)}
              image={absoluteUrl(plan.url)}
              active={planOpen}
              onClick={() => {
                setPlanOpen((v) => !v);
                poke();
              }}
              icon={<MapIcon size={12} />}
            />
          ) : null}
          {railAreas.map((a) => (
            <RailTile
              key={a.key}
              label={a.key === manifest.base.key && rooms.length ? t("walk.general", locale) : a.name}
              image={absoluteUrl(a.thumbnail_url)}
              active={activeKey === a.key}
              muted={!a.placed}
              title={!a.placed ? t("walk.unplaced", locale) : undefined}
              onClick={() => goTo(a.key)}
            />
          ))}
        </div>

        <div className="glass-strong flex items-center gap-1 rounded-full p-1.5 shadow-[var(--shadow-lift)]">
          <RailButton label={touring ? t("walk.autotour.pause", locale) : t("walk.autotour", locale)} active={touring} onClick={toggleTour}>
            {touring ? <Pause size={14} /> : <Play size={14} />}
          </RailButton>
          <div className="mx-0.5 h-5 w-px bg-line" />
          <RailButton label={t("walk.mode.walk", locale)} active={mode === "walk"} onClick={() => selectMode("walk")}>
            <Footprints size={14} />
          </RailButton>
          <RailButton label={t("walk.mode.drone", locale)} active={mode === "drone"} onClick={() => selectMode("drone")}>
            <Compass size={14} />
          </RailButton>
          <div className="mx-0.5 h-5 w-px bg-line" />
          <RailButton label={t("walk.controls", locale)} active={helpOpen} onClick={() => setHelpOpen((v) => !v)}>
            <Keyboard size={14} />
          </RailButton>
          {vr.route ? <VrWalkButton locale={locale} active={vr.mode !== "off" || vr.cardboard.phase !== "off"} onClick={vr.enter} /> : null}
          <RailButton label={t("walk.fullscreen", locale)} onClick={toggleFullscreen}>
            {fullscreen ? <Minimize size={14} /> : <Maximize size={14} />}
          </RailButton>
        </div>
      </div>

      {/* Floor plan overlay. */}
      {planOpen && plan?.url ? (
        <div data-walk-ui className="absolute inset-0 z-40 flex flex-col items-center justify-center bg-canvas/85 p-4 backdrop-blur-md sm:p-8">
          <button
            type="button"
            aria-label={t("walk.plan.close", locale)}
            onClick={() => setPlanOpen(false)}
            className="absolute top-3 right-3 grid h-9 w-9 place-items-center rounded-full border border-line bg-surface text-ink-muted hover:text-ink"
          >
            <X size={15} />
          </button>
          <div
            className="relative max-h-[78vh] max-w-full cursor-crosshair overflow-hidden rounded-[var(--radius-md)] border border-line-strong bg-black shadow-[var(--shadow-lift)]"
            style={{ aspectRatio: `${plan.width} / ${Math.max(plan.height, 1)}` }}
            onClick={onPlanClick}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={absoluteUrl(plan.url) ?? undefined} alt="" className="block max-h-[78vh] w-auto max-w-full" draggable={false} />
            {plan.hotspots.map((h) => (
              <button
                key={h.key}
                type="button"
                title={h.name}
                onClick={(e) => {
                  e.stopPropagation();
                  goTo(h.key);
                }}
                className={cn(
                  "absolute -translate-x-1/2 -translate-y-1/2 rounded-full border-2 px-2 py-0.5 text-[10px] font-medium tracking-wide uppercase shadow-[var(--shadow-glow)] transition",
                  activeKey === h.key ? "border-bio bg-bio/25 text-bio" : "border-amber/80 bg-amber/20 text-amber hover:bg-amber/35",
                )}
                style={{ left: `${h.x_pct * 100}%`, top: `${h.y_pct * 100}%` }}
              >
                {h.name}
              </button>
            ))}
            <div ref={markerRef} className="pointer-events-none absolute left-1/2 top-1/2" style={{ transform: "translate(-50%,-50%)" }}>
              <svg width="34" height="34" viewBox="-17 -17 34 34" className="drop-shadow">
                <path d="M0 0 L-9 -15 A17 17 0 0 1 9 -15 Z" fill="rgba(23,226,197,0.35)" />
                <circle r="4" fill="#17e2c5" stroke="#05070a" strokeWidth="1.5" />
              </svg>
            </div>
          </div>
          <p className="mt-3 text-[11px] font-medium tracking-[0.14em] text-ink-muted uppercase">{t("walk.plan.hint", locale)}</p>
        </div>
      ) : null}

      {/* Controls legend. */}
      {helpOpen ? (
        <div data-walk-ui className="absolute inset-0 z-40 grid place-items-center bg-canvas/70 p-4 backdrop-blur-sm" onClick={() => setHelpOpen(false)}>
          <div className="glass-strong w-full max-w-sm rounded-[var(--radius-lg)] p-5 text-[12px] text-ink" onClick={(e) => e.stopPropagation()}>
            <div className="mb-3 flex items-center justify-between">
              <span className="text-[10px] font-semibold tracking-[0.18em] text-ink-muted uppercase">{t("walk.controls", locale)}</span>
              <button type="button" aria-label={t("walk.plan.close", locale)} onClick={() => setHelpOpen(false)} className="text-ink-muted hover:text-ink">
                <X size={14} />
              </button>
            </div>
            {coarse ? (
              <ul className="space-y-2 text-ink-muted">
                <li>{t("walk.controls.touchLook", locale)}</li>
                <li>{t("walk.controls.tapWalk", locale)}</li>
              </ul>
            ) : (
              <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2">
                <Key keys={["W", "A", "S", "D"]} />
                <dd className="self-center text-ink-muted">{t("walk.controls.move", locale)}</dd>
                <Key keys={["Sürükle"]} />
                <dd className="self-center text-ink-muted">{t("walk.controls.look", locale)}</dd>
                <Key keys={["Tekerlek"]} />
                <dd className="self-center text-ink-muted">{t("walk.controls.zoom", locale)}</dd>
                <Key keys={["Çift tık"]} />
                <dd className="self-center text-ink-muted">{t("walk.controls.walkTo", locale)}</dd>
                <Key keys={["Q", "E"]} />
                <dd className="self-center text-ink-muted">{t("walk.mode.drone", locale)} ↑↓</dd>
              </dl>
            )}
          </div>
        </div>
      ) : null}

      {/* Loading veil. */}
      {!progress.ready && !unsupported ? (
        <div className="pointer-events-none absolute inset-0 z-30 grid place-items-center bg-canvas">
          <div className="w-64 text-center">
            <div className="mx-auto mb-4 h-10 w-10 animate-spin-slow rounded-full border border-line border-t-bio" />
            <p className="text-[12px] font-medium tracking-[0.14em] text-ink-muted uppercase">{t("walk.loading", locale)}</p>
            <div className="mt-3 h-px w-full bg-line">
              <div className="bio-gradient h-px" style={{ width: `${Math.max(3, percent)}%` }} />
            </div>
            <p className="mt-2 font-mono text-[11px] text-ink-faint">
              {formatBytes(progress.loaded)} / {formatBytes(progress.total)}
            </p>
            {error ? <p className="mt-2 text-[11px] text-danger">{error}</p> : null}
          </div>
        </div>
      ) : null}
      {unsupported ? (
        <div className="absolute inset-0 z-30 grid place-items-center bg-canvas p-6 text-center text-[13px] text-ink-muted">{t("walk.unsupported", locale)}</div>
      ) : null}

      {vr.mode !== "off" ? (
        <Suspense fallback={null}>
          <WalkVrRuntime
            mode={vr.mode}
            engine={engine}
            xr={vr.xr}
            feed={vr.cardboard.feed}
            lens={vr.cardboard.lens}
            debug={vr.debug}
            selectRef={vr.cardboard.selectRef}
            copy={vrCopy}
            onExit={vr.exit}
          />
        </Suspense>
      ) : null}
      {vr.cardboard.phase !== "off" || vr.handoff || vr.prompt ? (
        <Suspense fallback={null}>
          <VrOverlays
            locale={locale}
            kind="walk"
            cardboard={vr.cardboard}
            handoff={vr.handoff ? { url: vr.handoff, onClose: vr.closeHandoff } : null}
            prompt={vr.prompt ? { onStart: vr.enter, onDismiss: vr.dismissPrompt } : null}
          />
        </Suspense>
      ) : null}
      {vr.message ? (
        <div className="pointer-events-none absolute inset-x-0 top-14 z-50 flex justify-center px-4">
          <span className="glass-strong animate-rise rounded-full px-3.5 py-2 text-[12px] text-ink">{vr.message}</span>
        </div>
      ) : null}

      {embedded && !inCardboard ? (
        <a
          href={process.env.NEXT_PUBLIC_SITE_URL ?? "/"}
          target="_blank"
          rel="noreferrer"
          data-walk-ui
          className={cn("absolute top-[max(0.75rem,env(safe-area-inset-top))] left-1/2 z-30 -translate-x-1/2 text-[10px] font-semibold tracking-[0.18em] text-ink-faint uppercase transition-opacity", chromeVisible ? "opacity-100" : "opacity-0")}
        >
          BioSphere AI
        </a>
      ) : null}
    </div>
  );
}

/**
 * A frame-rate readout, shown only with `?perf=1`.
 *
 * A splat viewer that stutters is a report nobody can act on: the same property is smooth on one
 * machine and unusable on the next, and the only thing that separates them is how many pixels
 * the GPU is asked for. The engine already measures itself to steer its render scale, so this
 * puts the numbers on screen - frames, milliseconds, and the scale it settled at.
 */
function PerfBadge() {
  const [perf, setPerf] = useState<{ fps: number; frameMs: number; scale: number; dpr: number } | null>(null);
  useEffect(() => {
    const id = window.setInterval(() => {
      const p = (window as unknown as { __walkPerf?: { fps: number; frameMs: number; scale: number; dpr: number } }).__walkPerf;
      setPerf(p ?? null);
    }, 500);
    return () => window.clearInterval(id);
  }, []);
  if (!perf) return null;
  return (
    <div className="pointer-events-none absolute top-2 right-2 z-40 rounded bg-canvas/80 px-2 py-1 font-mono text-[11px] text-ink-muted tabular-nums">
      {perf.fps} fps · {perf.frameMs} ms · x{perf.scale} (dpr {perf.dpr})
    </div>
  );
}


function RailTile({
  label,
  image,
  active,
  muted,
  title,
  icon,
  onClick,
}: {
  label: string;
  image: string | null;
  active?: boolean;
  muted?: boolean;
  title?: string;
  icon?: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title ?? label}
      aria-label={label}
      aria-pressed={active}
      className={cn(
        "group flex w-[92px] shrink-0 flex-col gap-1 rounded-[var(--radius-sm)] border p-1 text-left transition",
        active ? "border-bio/60 bg-bio/10 ring-glow" : "border-transparent hover:border-line-strong hover:bg-white/[0.04]",
        muted ? "opacity-60" : "",
      )}
    >
      <div className="relative aspect-[16/10] w-full overflow-hidden rounded-[6px] bg-surface-2">
        {image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={image} alt="" className="h-full w-full object-cover transition group-hover:scale-105" draggable={false} />
        ) : (
          <div className="grid h-full w-full place-items-center text-ink-faint">{icon}</div>
        )}
        {icon && image ? <span className="absolute top-1 left-1 grid h-4 w-4 place-items-center rounded bg-canvas/70 text-ink">{icon}</span> : null}
      </div>
      <span className={cn("truncate px-0.5 text-[10px] font-medium tracking-[0.08em] uppercase", active ? "text-bio" : "text-ink-muted")}>{label}</span>
    </button>
  );
}

function RailButton({ label, active, onClick, children }: { label: string; active?: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        "grid h-8 w-8 place-items-center rounded-full border transition",
        active ? "border-bio/50 bg-bio/12 text-bio" : "border-line text-ink-muted hover:border-line-strong hover:bg-white/[0.05] hover:text-ink",
      )}
    >
      {children}
    </button>
  );
}

function Key({ keys }: { keys: string[] }) {
  return (
    <dt className="flex gap-1">
      {keys.map((k) => (
        <kbd key={k} className="rounded border border-line-strong bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-ink">
          {k}
        </kbd>
      ))}
    </dt>
  );
}
