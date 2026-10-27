"use client";

/**
 * The toolbar.
 *
 * Everything here is an interruption of a photograph, so the rules are: one row, glass,
 * no fills except on the control that is currently doing something, and out of the way
 * the moment nobody is touching it. Labels are icons with accessible names rather than
 * text, except the two places where the text is the information — which room you are in,
 * and which resolution you are looking at.
 *
 * Every control here answers "show me the picture differently", and answers it at once.
 * Auto-rotate did not: it is a behaviour that keeps running after the press, so it moved to
 * `PlayerDock` with the other two controls that put the player into a state. VR followed it
 * there for the same reason (`components/vr/VrDockSegment`).
 */

import { AnimatePresence, motion } from "motion/react";
import { Footprints, Maximize, Minimize, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { t } from "@/lib/i18n";
import type { Locale, Scene } from "@/lib/types";
import { Compass } from "./Compass";
import { RoomSwitcher } from "./RoomSwitcher";
import { RUNG_LABEL, type TextureStatus } from "./types";

const EASE = [0.16, 1, 0.3, 1] as const;

interface ToolButtonProps {
  label: string;
  active?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

function ToolButton({ label, active, onClick, children }: ToolButtonProps) {
  return (
    <button
      type="button"
      data-player-ui
      onClick={onClick}
      aria-label={label}
      aria-pressed={active}
      title={label}
      className={cn(
        "grid h-8 w-8 shrink-0 place-items-center rounded-full border",
        "transition-all duration-200 ease-[var(--ease-out-expo)] active:scale-95",
        active
          ? "border-bio/50 bg-bio/12 text-bio"
          : "border-line text-ink-muted hover:border-line-strong hover:bg-white/[0.05] hover:text-ink",
      )}
    >
      {children}
    </button>
  );
}

export interface PlayerChromeProps {
  scenes: Scene[];
  currentSceneId: string | null;
  locale: Locale;
  visible: boolean;
  /** The compass dial element, written to every frame by the player's frame loop. */
  compassRef: React.RefObject<HTMLDivElement | null>;
  roomsOpen: boolean;
  onRoomsOpenChange: (open: boolean) => void;
  onSelectScene: (sceneId: string) => void;
  quality: TextureStatus;
  onForceQuality: () => void;
  /** The property has a 3D walkthrough too; `onOpenWalk` swaps the player for it. */
  walkAvailable?: boolean;
  onOpenWalk?: () => void;
  fullscreen: boolean;
  onFullscreen: () => void;
  onRecentre: () => void;
  /** Shows the small BioSphere mark; only meaningful inside a third-party page. */
  embedded: boolean;
  /** The one-time "drag to look around" nudge. */
  showHint: boolean;
}

export function PlayerChrome(props: PlayerChromeProps) {
  const {
    scenes,
    currentSceneId,
    locale,
    visible,
    compassRef,
    roomsOpen,
    onRoomsOpenChange,
    onSelectScene,
    quality,
    onForceQuality,
    walkAvailable,
    onOpenWalk,
    fullscreen,
    onFullscreen,
    onRecentre,
    embedded,
    showHint,
  } = props;

  // Only offer the upgrade when there is genuinely a better rung this device can take.
  const canUpgrade =
    quality.bestAvailable !== null &&
    quality.rung !== null &&
    quality.rung !== quality.bestAvailable &&
    quality.rung !== quality.maxRung &&
    !quality.loading;

  return (
    <>
      <AnimatePresence>
        {showHint ? (
          <motion.div
            key="hint"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.5, ease: EASE }}
            className="pointer-events-none absolute inset-x-0 bottom-[5.25rem] z-20 flex justify-center"
          >
            <span className="glass rounded-full px-3 py-1.5 text-[11px] tracking-tight text-ink-muted">
              {t("player.drag", locale)}
            </span>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <AnimatePresence>
        {visible ? (
          <motion.div
            key="chrome"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 14 }}
            transition={{ duration: 0.32, ease: EASE }}
            className={cn(
              "absolute inset-x-0 bottom-0 z-20 flex justify-center",
              // Keep clear of the home indicator and the notch on phones.
              "px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]",
            )}
          >
            <div className="relative max-w-full">
              <div
                data-player-ui
                className={cn(
                  "glass-strong flex max-w-full items-center gap-1.5 rounded-full",
                  "p-1.5 shadow-[var(--shadow-lift)]",
                )}
              >
                <RoomSwitcher
                  scenes={scenes}
                  currentSceneId={currentSceneId}
                  open={roomsOpen}
                  onOpenChange={onRoomsOpenChange}
                  onSelect={onSelectScene}
                  label={t("player.rooms", locale)}
                />

                {scenes.length > 1 ? <span aria-hidden className="h-5 w-px shrink-0 bg-line" /> : null}

                <Compass dialRef={compassRef} onRecentre={onRecentre} label={t("player.rooms", locale)} />

                <button
                  type="button"
                  data-player-ui
                  onClick={onForceQuality}
                  disabled={!canUpgrade}
                  aria-label={t("player.quality", locale)}
                  title={t("player.quality", locale)}
                  className={cn(
                    "flex h-8 shrink-0 items-center gap-1 rounded-full border px-2.5",
                    "text-[11px] font-semibold tracking-[0.06em] tabular-nums",
                    "transition-all duration-200 ease-[var(--ease-out-expo)]",
                    canUpgrade
                      ? "border-line text-ink-muted hover:border-bio/50 hover:bg-bio/10 hover:text-bio active:scale-95"
                      : "cursor-default border-line text-ink-faint",
                  )}
                >
                  {quality.loading ? (
                    <span aria-hidden className="animate-breathe h-1.5 w-1.5 rounded-full bg-bio" />
                  ) : canUpgrade ? (
                    <Sparkles size={12} />
                  ) : null}
                  {quality.rung ? RUNG_LABEL[quality.rung] : "—"}
                </button>

                {walkAvailable && onOpenWalk ? (
                  <ToolButton label={t("walk.to3d", locale)} onClick={onOpenWalk}>
                    <Footprints size={14} />
                  </ToolButton>
                ) : null}

                <ToolButton label={t("player.fullscreen", locale)} active={fullscreen} onClick={onFullscreen}>
                  {fullscreen ? <Minimize size={14} /> : <Maximize size={14} />}
                </ToolButton>

                {embedded ? (
                  <>
                    <span aria-hidden className="h-5 w-px shrink-0 bg-line" />
                    <a
                      data-player-ui
                      href={process.env.NEXT_PUBLIC_SITE_URL ?? "/"}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={cn(
                        "hidden shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 sm:flex",
                        "text-[10px] font-semibold tracking-[0.1em] uppercase",
                        "text-ink-faint transition-colors duration-200 hover:text-ink",
                      )}
                    >
                      <span aria-hidden className="bio-gradient h-1.5 w-1.5 rounded-full" />
                      {t("app.name", locale)}
                    </a>
                  </>
                ) : null}
              </div>
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </>
  );
}
