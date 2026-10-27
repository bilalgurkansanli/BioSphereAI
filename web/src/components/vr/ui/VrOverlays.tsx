"use client";

/**
 * The page-side of VR: the first-time guide for the phone headset, the "turn your phone" card,
 * the exit bar, the new-tab sheet for a frame that cannot do VR, and the "enter VR" prompt a
 * handed-off tab opens with. Lazy-loaded with the rest of the VR code; shared by the 360 player
 * and the walkthrough.
 *
 * Everything here carries `data-player-ui`, `data-walk-ui` and `data-vr-ui`: the three input
 * systems underneath (the 2D look controls, the walkthrough's engine, the Cardboard tap
 * handler) each skip events that start on their own marker, and a press on "Başla" must not
 * also spin the room or select whatever the gaze happened to rest on.
 *
 * ── THE GUIDE IS A GESTURE, NOT A FORMALITY ───────────────────────────────────────────────
 * iOS only grants the motion sensors from inside a tap, and shows its own prompt when asked.
 * The guide's "Başla" is that tap: the visitor reads what is about to happen — including that
 * a permission prompt is coming — before the prompt appears, instead of being asked for their
 * motion sensors by a button that said "VR". It shows once; after that the VR button goes
 * straight in.
 */

import { useEffect, useSyncExternalStore } from "react";
import { ExternalLink, Glasses, Smartphone, X } from "lucide-react";
import { t } from "@/lib/i18n";
import type { Locale } from "@/lib/types";
import { cn } from "@/lib/utils";
import type { LensId } from "../lens";
import type { CardboardPhase } from "../useCardboard";

const UI_ATTRS = { "data-player-ui": true, "data-walk-ui": true, "data-vr-ui": true } as const;

/** iOS 13+ raises a motion-sensor prompt; the guide says so before it appears. */
const subscribeNever = () => () => undefined;
const readAsksPermission = () =>
  typeof (window.DeviceOrientationEvent as (typeof DeviceOrientationEvent & { requestPermission?: unknown }) | undefined)
    ?.requestPermission === "function";
const serverAsksPermission = () => false;

export interface VrOverlaysProps {
  locale: Locale;
  /** Which experience the guide describes. */
  kind: "pano" | "walk";
  cardboard: {
    phase: CardboardPhase;
    portrait: boolean;
    hud: boolean;
    lensId: LensId;
    setLensId: (id: LensId) => void;
    start: () => void;
    stop: () => void;
    cancelGuide: () => void;
  };
  handoff: { url: string; onClose: () => void } | null;
  prompt: { onStart: () => void; onDismiss: () => void } | null;
}

export default function VrOverlays({ locale, kind, cardboard, handoff, prompt }: VrOverlaysProps) {
  const { phase } = cardboard;
  const asksPermission = useSyncExternalStore(subscribeNever, readAsksPermission, serverAsksPermission);

  return (
    <>
      {phase === "guide" ? (
        <Sheet onDismiss={cardboard.cancelGuide}>
          <div className="mb-4 flex items-center gap-2.5">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-bio/40 bg-bio/10 text-bio">
              <Glasses size={17} aria-hidden />
            </span>
            <h2 className="text-[15px] font-semibold tracking-tight text-ink">{t("vr.guide.title", locale)}</h2>
          </div>
          <ol className="space-y-2.5 text-[13px] leading-relaxed text-ink-muted">
            {[t("vr.guide.step1", locale), t(kind === "walk" ? "vr.guide.walkStep2" : "vr.guide.step2", locale), t("vr.guide.step3", locale)].map(
              (step, index) => (
                <li key={step} className="flex gap-2.5">
                  <span className="mt-[1px] grid h-5 w-5 shrink-0 place-items-center rounded-full border border-line-strong text-[11px] font-semibold text-ink tabular-nums">
                    {index + 1}
                  </span>
                  <span>{step}</span>
                </li>
              ),
            )}
          </ol>

          <div className="mt-4 flex items-center justify-between gap-3">
            <span className="text-[11px] font-semibold tracking-[0.14em] text-ink-faint uppercase">{t("vr.guide.lens", locale)}</span>
            <div role="radiogroup" aria-label={t("vr.guide.lens", locale)} className="flex gap-1 rounded-full border border-line p-1">
              {(["vrbox", "cardboard"] as const).map((id) => (
                <button
                  key={id}
                  type="button"
                  role="radio"
                  aria-checked={cardboard.lensId === id}
                  onClick={() => cardboard.setLensId(id)}
                  className={cn(
                    "rounded-full px-3 py-1 text-[12px] font-medium transition-colors",
                    cardboard.lensId === id ? "bg-bio/15 text-bio" : "text-ink-muted hover:text-ink",
                  )}
                >
                  {t(id === "vrbox" ? "vr.guide.vrbox" : "vr.guide.cardboard", locale)}
                </button>
              ))}
            </div>
          </div>

          {asksPermission ? <p className="mt-3 text-[12px] leading-relaxed text-amber">{t("vr.guide.permission", locale)}</p> : null}

          <div className="mt-5 flex justify-end gap-2">
            <button
              type="button"
              onClick={cardboard.cancelGuide}
              className="rounded-full border border-line px-4 py-2 text-[12px] font-medium text-ink-muted transition-colors hover:border-line-strong hover:text-ink"
            >
              {t("vr.guide.cancel", locale)}
            </button>
            <button
              type="button"
              onClick={cardboard.start}
              className="rounded-full bg-bio px-5 py-2 text-[12px] font-semibold text-canvas transition-transform hover:scale-[1.02] active:scale-[0.98]"
            >
              {t("vr.guide.start", locale)}
            </button>
          </div>
        </Sheet>
      ) : null}

      {phase === "starting" ? (
        <div className="pointer-events-none absolute inset-x-0 top-4 z-50 flex justify-center px-4">
          <span className="glass-strong animate-rise rounded-full px-3.5 py-2 text-[12px] text-ink">{t("vr.starting", locale)}</span>
        </div>
      ) : null}

      {phase === "active" ? (
        <>
          {/* Where the phone's middle goes: a hairline between the eyes, for lining the phone
              up in the box. Thicker would land inside the lens barrel and read as a defect. */}
          <span aria-hidden className="pointer-events-none absolute inset-y-0 left-1/2 z-40 w-px -translate-x-1/2 bg-white/25" />

          {cardboard.hud ? (
            <div {...UI_ATTRS} className="absolute inset-x-0 top-3 z-50 flex justify-center px-3">
              <div className="glass-strong animate-rise flex items-center gap-1.5 rounded-full p-1.5 shadow-[var(--shadow-lift)]">
                <button
                  type="button"
                  onClick={cardboard.stop}
                  className="flex h-9 items-center gap-1.5 rounded-full border border-danger/40 px-3.5 text-[12px] font-semibold text-danger transition-colors hover:bg-danger/10"
                >
                  <X size={14} aria-hidden />
                  {t("vr.exit", locale)}
                </button>
                <span aria-hidden className="h-5 w-px bg-line" />
                {(["vrbox", "cardboard"] as const).map((id) => (
                  <button
                    key={id}
                    type="button"
                    aria-pressed={cardboard.lensId === id}
                    onClick={() => cardboard.setLensId(id)}
                    className={cn(
                      "h-9 rounded-full px-3 text-[12px] font-medium transition-colors",
                      cardboard.lensId === id ? "bg-bio/15 text-bio" : "text-ink-muted hover:text-ink",
                    )}
                  >
                    {t(id === "vrbox" ? "vr.guide.vrbox" : "vr.guide.cardboard", locale)}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {cardboard.portrait ? (
            <div {...UI_ATTRS} className="absolute inset-0 z-50 flex flex-col items-center justify-center gap-4 bg-black px-8 text-center">
              <style>{`@keyframes vr-turn { 0%, 20% { transform: rotate(0deg); } 60%, 100% { transform: rotate(-90deg); } }`}</style>
              <Smartphone size={44} aria-hidden className="text-bio" style={{ animation: "vr-turn 1.8s var(--ease-out-expo) infinite alternate" }} />
              <div>
                <p className="text-[15px] font-semibold text-white">{t("vr.rotate.title", locale)}</p>
                <p className="mt-1.5 max-w-[22rem] text-[13px] leading-relaxed text-white/60">{t("vr.rotate.body", locale)}</p>
              </div>
              <button
                type="button"
                onClick={cardboard.stop}
                className="mt-2 flex items-center gap-1.5 rounded-full border border-white/20 px-4 py-2 text-[12px] font-medium text-white/80"
              >
                <X size={13} aria-hidden />
                {t("vr.exit", locale)}
              </button>
            </div>
          ) : null}
        </>
      ) : null}

      {handoff ? (
        <Sheet onDismiss={handoff.onClose}>
          <h2 className="text-[15px] font-semibold tracking-tight text-ink">{t("vr.handoff.title", locale)}</h2>
          <p className="mt-2 text-[13px] leading-relaxed text-ink-muted">{t("vr.handoff.body", locale)}</p>
          <div className="mt-5 flex justify-end gap-2">
            <button
              type="button"
              onClick={handoff.onClose}
              className="rounded-full border border-line px-4 py-2 text-[12px] font-medium text-ink-muted transition-colors hover:border-line-strong hover:text-ink"
            >
              {t("vr.guide.cancel", locale)}
            </button>
            <a
              href={handoff.url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={handoff.onClose}
              className="flex items-center gap-1.5 rounded-full bg-bio px-5 py-2 text-[12px] font-semibold text-canvas transition-transform hover:scale-[1.02] active:scale-[0.98]"
            >
              <ExternalLink size={13} aria-hidden />
              {t("vr.handoff.open", locale)}
            </a>
          </div>
        </Sheet>
      ) : null}

      {prompt ? (
        <Sheet onDismiss={prompt.onDismiss}>
          <div className="flex flex-col items-center gap-4 py-2 text-center">
            <span className="grid h-14 w-14 place-items-center rounded-full border border-bio/40 bg-bio/10 text-bio">
              <Glasses size={26} aria-hidden />
            </span>
            <h2 className="text-[15px] font-semibold tracking-tight text-ink">{t("vr.prompt.title", locale)}</h2>
            <button
              type="button"
              onClick={prompt.onStart}
              className="w-full rounded-full bg-bio px-5 py-3 text-[14px] font-semibold text-canvas transition-transform hover:scale-[1.01] active:scale-[0.98]"
            >
              {t("vr.prompt.start", locale)}
            </button>
            <button type="button" onClick={prompt.onDismiss} className="text-[12px] font-medium text-ink-muted hover:text-ink">
              {t("vr.prompt.later", locale)}
            </button>
          </div>
        </Sheet>
      ) : null}
    </>
  );
}

/** A centred card over a dimmed player; Escape and a click outside both dismiss it. */
function Sheet({ children, onDismiss }: { children: React.ReactNode; onDismiss: () => void }) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onDismiss();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onDismiss]);

  return (
    <div
      {...UI_ATTRS}
      className="absolute inset-0 z-50 grid place-items-center bg-canvas/75 p-4 backdrop-blur-md"
      onClick={(event) => {
        if (event.target === event.currentTarget) onDismiss();
      }}
    >
      <div role="dialog" aria-modal="true" className="glass-strong animate-rise w-full max-w-sm rounded-[var(--radius-lg)] p-5 shadow-[var(--shadow-lift)]">
        {children}
      </div>
    </div>
  );
}
