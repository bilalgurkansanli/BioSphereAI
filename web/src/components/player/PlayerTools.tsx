"use client";

/**
 * The two tools that rewrite the photograph: empty the room, and restyle it.
 *
 * ── WHY THEY ARE NOT ON THE TOOLBAR ───────────────────────────────────────────────────
 * `PlayerChrome` is the *viewing* controls: which room, which way is north, how sharp, how
 * big. Every one of them is instant, reversible and about the picture as it is. These two
 * change the picture itself, take a minute to do it, and have something to say about the
 * result. Mixing them would make the toolbar mean two things — and the toolbar is already
 * the widest element on a phone.
 *
 * ── WHERE THEY SIT IS NOT THIS FILE'S BUSINESS ────────────────────────────────────────
 * This used to anchor itself to the bottom-left corner, which was fine until something else
 * did too. `PlayerDock` owns that corner now, and this exports the pieces it arranges:
 *
 *   `ToolSegments`  the buttons, which go on the dock's one bar
 *   `ToolPanels`    what the tools have to SAY, which stacks above it
 *
 * The split is by lifetime, not by taste. A button is always the same size and belongs in a
 * row with the other buttons; a sentence about what a run managed, or a picker of five
 * styles, is 20 to 300 px tall depending on the answer and would make that row jump.
 *
 * ── THEY SHARE THE BAR WITH THE OTHER THREE, AND THAT IS A REVERSAL ───────────────────
 * They had their own row above the bar on the first pass, on the reasoning that a control
 * which rewrites a photograph over a minute is not a peer of one that turns a drift on. That
 * reasoning is sound and the result still looked bad: two stacked glass slabs of different
 * widths in a corner, which is the thing a corner should never be. Stacking is what a viewer
 * reads as a hierarchy, so it was announcing a distinction nobody needed while costing the
 * layout its only straight line. One row, in one bar, with a hairline where the meaning
 * changes — the distinction survives, at the price it is actually worth.
 *
 * ── HONESTY IS PART OF THE CONTROL ────────────────────────────────────────────────────
 * The removal is classical computer vision: it finds furnishings by how far they sit from
 * the colour of the wall or floor behind them, so a wooden chest on a wooden floor can
 * survive it. The button therefore reports what it actually did — how much of the sphere it
 * replaced — rather than presenting an empty room as a finished fact. A user who can see
 * that a chair stayed and is told nothing concludes the feature is broken; the same user
 * told "some furnishings the colour of the surface behind them may remain" concludes it has
 * limits, which is true and far less damaging.
 *
 * ── AND WHY THE MEASURE BUTTON IS NOT ONE OF THEM ─────────────────────────────────────
 * These two are ALTERNATIVES — each replaces the panorama with something a model made, so
 * they sit together behind one beta label. Measuring replaces nothing: it draws on top of
 * the photograph, and it is exact projective geometry rather than a model (`model_backed` is
 * `false` on that endpoint's own response). Putting it beside them would put it under the beta
 * label, which would be a claim about it that is not true; putting it there WITHOUT the label
 * would make the label look like it applied to whichever button happened to be nearest. So it
 * sits on the far side of the dock's hairline, with the other controls that only ever draw ON
 * the picture, and its panel carries its own honesty line about the assumed camera height —
 * the thing that actually limits it.
 */

import { AnimatePresence, motion } from "motion/react";
import { Loader2, Palette, Sofa, X } from "lucide-react";
import { BetaTag } from "@/components/ui/BetaTag";
import { cn } from "@/lib/utils";
import { t } from "@/lib/i18n";
import type { Locale, RestyleStyle, RestyleStyleId } from "@/lib/types";
import { DockButton } from "./DockButton";
import type { EmptyRoomPhase } from "./useEmptyRoom";
import type { SceneRestylePhase } from "./useSceneRestyle";

const EASE = [0.16, 1, 0.3, 1] as const;

/** Below this share of the sphere the result is not worth calling an emptied room. */
const THIN_RESULT = 0.02;

export interface PlayerToolsProps {
  locale: Locale;
  emptyRoom: {
    active: boolean;
    phase: EmptyRoomPhase;
    removedFraction: number | null;
    /** True when a scene model drew the mask, rather than the colour fallback. */
    modelBacked: boolean;
    message: string | null;
    onToggle: () => void;
  };
  /** Absent when the deployment cannot stage, which is how the dock stays honest. */
  decor?: {
    styles: RestyleStyle[];
    activeStyle: RestyleStyleId | null;
    pendingStyle: RestyleStyleId | null;
    phase: SceneRestylePhase;
    message: string | null;
    onSelect: (style: RestyleStyleId | null) => void;
  };
  /**
   * The style picker's state, held by the dock: the button that opens it is on the bar and
   * the picker itself is an overlay above it, so neither half can own it.
   */
  pickerOpen: boolean;
  onPickerOpenChange: (open: boolean) => void;
}

/**
 * What the tools managed, in their own words — shown once a run has landed.
 *
 * The caveat belongs to the COLOUR path alone. That one finds furnishings by how far they sit
 * from the surface behind them, so a wooden chest on a wooden floor survives it and the user
 * deserves to be told. A scene model does not have that failure, and repeating the warning
 * under a room that is visibly empty would read as a product that does not know what it just
 * did.
 */
function toolNote({ locale, emptyRoom, decor }: PlayerToolsProps): string | null {
  if (decor?.message) return decor.message;
  if (emptyRoom.message) return emptyRoom.message;
  // While it works, say how long: the first emptying of a room is a minute or two of
  // segmentation and rebuilding, and a spinner alone for that long reads as a hang.
  if (emptyRoom.phase === "working") return t("player.empty.slow", locale);
  if (!emptyRoom.active || emptyRoom.removedFraction === null) return null;
  if (emptyRoom.removedFraction < THIN_RESULT) return t("player.empty.none", locale);
  return emptyRoom.modelBacked ? null : t("player.empty.partial", locale);
}

/**
 * The beta label and the two buttons, as bare segments for the dock's bar.
 *
 * A fragment rather than a wrapper: the bar is one flex row and these have to be its direct
 * children, or the gap between "Boşalt" and "Dekor" stops being the gap between "Ölç" and
 * the drift and the row acquires a seam that means nothing.
 */
export function ToolSegments(props: PlayerToolsProps) {
  const { locale, emptyRoom, decor, pickerOpen, onPickerOpenChange } = props;
  const busy = emptyRoom.phase === "working";

  return (
    <>
      {/* The tag sits WITH the pair rather than on either one: both replace the photograph
          with something a model made, and a label on only one would imply the other is
          finished. */}
      <BetaTag title={t("player.beta", locale)} data-player-ui className="shrink-0" />

      <DockButton
        label={
          busy
            ? t("player.empty.working", locale)
            : emptyRoom.active
              ? t("player.empty.off", locale)
              : t("player.empty.on", locale)
        }
        active={emptyRoom.active}
        busy={busy}
        onClick={emptyRoom.onToggle}
      >
        {busy ? (
          <Loader2 size={14} className="animate-spin" aria-hidden />
        ) : (
          <Sofa size={14} aria-hidden />
        )}
      </DockButton>

      {decor ? (
        <DockButton
          label={
            decor.pendingStyle
              ? t("player.decor.working", locale)
              : decor.activeStyle
                ? styleLabel(decor.styles, decor.activeStyle, locale)
                : t("player.decor.on", locale)
          }
          active={decor.activeStyle !== null}
          busy={decor.pendingStyle !== null}
          expanded={pickerOpen}
          onClick={() => onPickerOpenChange(!pickerOpen)}
        >
          {decor.pendingStyle ? (
            <Loader2 size={14} className="animate-spin" aria-hidden />
          ) : (
            <Palette size={14} aria-hidden />
          )}
        </DockButton>
      ) : null}
    </>
  );
}

/** The note and the style picker: everything that is taller than a button. */
export function ToolPanels(props: PlayerToolsProps) {
  const { locale, emptyRoom, decor, pickerOpen, onPickerOpenChange } = props;
  const note = toolNote(props);

  return (
    <>
      <AnimatePresence>
        {note ? (
          <motion.p
            key={note}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 6 }}
            transition={{ duration: 0.28, ease: EASE }}
            data-player-ui
            className={cn(
              "glass pointer-events-auto max-w-[min(20rem,70vw)] rounded-lg px-2.5 py-1.5",
              "text-[11px] leading-snug tracking-tight",
              emptyRoom.phase === "failed" ? "text-amber" : "text-ink-muted",
            )}
          >
            {note}
          </motion.p>
        ) : null}
      </AnimatePresence>

      <AnimatePresence>
        {decor && pickerOpen ? (
          <StylePicker locale={locale} decor={decor} onClose={() => onPickerOpenChange(false)} />
        ) : null}
      </AnimatePresence>
    </>
  );
}

function styleLabel(styles: RestyleStyle[], id: RestyleStyleId, locale: Locale): string {
  const found = styles.find((style) => style.id === id);
  if (!found) return id;
  return locale === "tr" ? found.label_tr : found.label_en;
}

interface StylePickerProps {
  locale: Locale;
  decor: NonNullable<PlayerToolsProps["decor"]>;
  onClose: () => void;
}

/**
 * The five styles, plus the way back.
 *
 * IT SAYS HOW LONG THIS TAKES, BEFORE THE CLICK. A sphere is sixteen diffusion passes and the
 * first ask for a style is minutes. A spinner that appears after the click and then sits there
 * for two of them is indistinguishable from a hang, and a user who cancels a hang has paid the
 * whole cost for nothing. So the warning lives on the picker rather than in the busy state: it
 * is information needed in order to decide, not an apology afterwards.
 */
function StylePicker({ locale, decor, onClose }: StylePickerProps) {
  return (
    <motion.div
      key="styles"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
      transition={{ duration: 0.26, ease: EASE }}
      data-player-ui
      role="listbox"
      aria-label={t("player.decor", locale)}
      style={{ transformOrigin: "bottom left" }}
      className="glass-strong pointer-events-auto flex max-w-[min(22rem,80vw)] flex-col gap-2
                 rounded-[var(--radius-lg)] p-2.5 shadow-[var(--shadow-lift)]"
    >
      <div className="flex items-center justify-between gap-3 pl-1">
        <span className="text-[11px] font-medium tracking-wide text-ink-faint uppercase">
          {t("player.decor", locale)}
        </span>
        <button
          type="button"
          onClick={onClose}
          aria-label={t("player.decor", locale)}
          className="grid size-6 place-items-center rounded-full text-ink-faint hover:bg-white/[0.06] hover:text-ink"
        >
          <X size={13} aria-hidden />
        </button>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {decor.styles.map((style) => {
          const selected = decor.activeStyle === style.id;
          const pending = decor.pendingStyle === style.id;
          return (
            <button
              key={style.id}
              type="button"
              role="option"
              aria-selected={selected}
              disabled={decor.pendingStyle !== null}
              onClick={() => decor.onSelect(style.id)}
              className={cn(
                "flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[12px] font-medium",
                "transition-all duration-150",
                decor.pendingStyle !== null && !pending ? "opacity-45" : "",
                selected
                  ? "border-bio/50 bg-bio/12 text-bio"
                  : "border-line text-ink-muted hover:border-line-strong hover:text-ink",
              )}
            >
              {pending ? <Loader2 size={12} className="animate-spin" aria-hidden /> : null}
              {locale === "tr" ? style.label_tr : style.label_en}
            </button>
          );
        })}
      </div>

      {decor.activeStyle ? (
        <button
          type="button"
          onClick={() => decor.onSelect(null)}
          className="self-start rounded-full px-3 py-1 text-[11px] text-ink-faint hover:text-ink"
        >
          {t("player.decor.off", locale)}
        </button>
      ) : (
        <p className="px-1 text-[11px] leading-snug text-ink-faint">{t("player.decor.slow", locale)}</p>
      )}
    </motion.div>
  );
}
