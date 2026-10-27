"use client";

/**
 * The bottom-left corner, and the single thing that owns it.
 *
 * ── WHY THIS EXISTS AT ALL ────────────────────────────────────────────────────────────
 * Two separate components used to anchor themselves here — the plan dock and the tool rail —
 * each with its own `absolute bottom-0 left-0`. That was safe only while they could never
 * appear together, which stopped being true the moment an embedded player was allowed to
 * measure: a listing page with a floor plan drew the "Kuş bakışı" pill and the "Ölç" pill at
 * exactly the same coordinates, one over the other. Nothing warns you about that — both
 * elements are positioned, both are correct on their own, and the corner is simply wrong.
 *
 * So the corner gets one owner. Everything that lives here is laid out by this file, in one
 * bar, with one visibility rule, and a new control is a new segment rather than a new element
 * competing for the same corner.
 *
 * ── ONE ROW, NOT A STACK ──────────────────────────────────────────────────────────────
 * The first fix un-overlapped them but left two glass slabs stacked in the corner — the AI
 * tools above, the view controls below — of different widths, because their labels are
 * different lengths. That is worse than it sounds. A stack in a corner has no straight edge
 * to rest on: every row starts at the same left margin and ends wherever its text ran out, so
 * the eye reads a ragged staircase rather than a control. And stacking IS a hierarchy, so it
 * was also announcing a distinction between the rows that nobody needed announced.
 *
 * One row. Grouped left to right by what the control does to the player, hairlines where the
 * meaning changes:
 *
 *   [beta] [Boşalt] [Dekor]  │  [Kuş bakışı] [Ölç] [⟳ Döndürmeyi aç]  │  [Yatırım]
 *    rewrites the picture       what a visitor can do with it          the purchase, not the picture
 *
 * It scrolls rather than wraps when it cannot fit. Wrapping would put the corner back to
 * being two rows of unequal width — the exact thing this is not — and the row is reachable
 * either way, whereas a second row on a phone is 44 px of photograph gone for good.
 *
 * ── WHAT STAYS ABOVE IT ───────────────────────────────────────────────────────────────
 * Only the things that are not button-shaped: the plan panel, the style picker, and the
 * sentence a finished run leaves behind. Those are 20 to 300 px tall depending on the answer,
 * so they can never join a row without making it jump.
 *
 * ── AUTO-ROTATE LIVES HERE, NOT ON THE TOOLBAR ────────────────────────────────────────
 * It used to be an unlabelled icon between the compass and the resolution readout, which put
 * a behaviour toggle in a row of instant, reversible view controls and gave it the least
 * findable slot on the screen. It moved here — and then it kept the unlabelled shape for a
 * while, on the theory that a turning picture explains itself. It does, once it is turning;
 * the button has to be findable BEFORE that. So it carries words like its neighbours, and the
 * words are the state rather than the feature's name. It is still off until somebody asks for
 * it: a panorama that starts spinning under a visitor's hand is a thing done to them.
 *
 * The trailing group is the investment assistant on its own: every other segment acts on the
 * picture, and this one answers questions about buying the flat. It appears only on listings
 * whose agent switched it on — see `InvestmentChat`.
 */

import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { Map, RotateCw, Ruler, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";
import { t } from "@/lib/i18n";
import type { FloorPlan, Locale } from "@/lib/types";
import { DockButton } from "./DockButton";
import { InvestmentChat, useInvestmentAvailability } from "./InvestmentChat";
import { PlanView } from "./PlanView";
import { ToolPanels, ToolSegments, type PlayerToolsProps } from "./PlayerTools";
import { VrDockSegment } from "@/components/vr/VrDockSegment";

const EASE = [0.16, 1, 0.3, 1] as const;

export interface PlayerDockProps {
  locale: Locale;
  /**
   * One rule for the whole corner. Separate rules per element would be a layout that jumps:
   * the rail's own rule already outlives the toolbar's (see `Player.dockVisible`), so a
   * segment that left early would shorten the bar under the pointer.
   */
  visible: boolean;
  /** Absent when the property has no drawing, or the host draws its own chrome. */
  plan?: {
    plan: FloorPlan;
    currentSceneId: string | null;
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onSelectRoom: (sceneId: string) => void;
    /** The heading cone, written every frame by the player's own loop. */
    headingRef: React.RefObject<SVGGElement | null>;
  };
  autoRotate?: {
    value: boolean;
    onChange: (value: boolean) => void;
  };
  /** Absent when the deployment or the host cannot offer measuring. */
  measure?: {
    active: boolean;
    onToggle: () => void;
  };
  /** Absent where no VR can work here — a desktop without a headset. See `components/vr`. */
  vr?: {
    active: boolean;
    onClick: () => void;
  };
  /**
   * The listing's embed token, for the investment assistant. The segment appears only when the
   * engine says the agent opted in and entered a price (see `InvestmentChat`); with no token —
   * a player inside the dashboard — there is nothing to ask and no segment.
   */
  investmentToken?: string | null;
  /**
   * The AI tools. First-party only: they are unauthenticated GPU compute behind the
   * restricted CORS policy, so an embedded player is handed none and the bar starts at the
   * plan instead. See `PlayerTools`.
   */
  tools?: Omit<PlayerToolsProps, "locale" | "pickerOpen" | "onPickerOpenChange">;
}

export function PlayerDock(props: PlayerDockProps) {
  const { locale, visible, plan, autoRotate, measure, tools, investmentToken } = props;
  const investment = useInvestmentAvailability(investmentToken ?? null, locale);
  const investmentOn = investment.enabled && Boolean(investmentToken);

  // Which control is expanded. A style picker left permanently open would cover a third of
  // the photograph on a phone, and the photograph is the product. It lives here rather than
  // in `PlayerTools` because the button that opens it is on the bar and the picker itself is
  // an overlay above it — neither half can own the state on its own.
  const [pickerOpen, setPickerOpen] = useState(false);
  const [investOpen, setInvestOpen] = useState(false);

  const planOpen = plan?.open ?? false;
  // A corner with nothing in it should not paint a bar with nothing on it.
  const hasBar = Boolean(tools || plan || measure || autoRotate || investmentOn);
  const toolProps: PlayerToolsProps | null = tools
    ? { ...tools, locale, pickerOpen, onPickerOpenChange: setPickerOpen }
    : null;

  return (
    <div
      className={cn(
        "pointer-events-none absolute left-0 z-30 flex flex-col items-start gap-2",
        // The toolbar is centred on this same edge and, on a phone, there is no screen's
        // width between them — so below `sm` the corner lifts above it instead.
        "bottom-[4.5rem] px-3 sm:bottom-0 sm:pb-[max(0.75rem,env(safe-area-inset-bottom))]",
        "transition-opacity duration-300 ease-[var(--ease-out-expo)]",
        // An open assistant holds the corner up too: the visitor is reading or typing there,
        // and the idle timer that hides the chrome is not watching the keyboard.
        visible || planOpen || (investmentOn && investOpen) ? "opacity-100" : "opacity-0",
      )}
    >
      {toolProps ? <ToolPanels {...toolProps} /> : null}

      <AnimatePresence>
        {investmentOn && investOpen && investment.state && investmentToken ? (
          <InvestmentChat
            locale={locale}
            token={investmentToken}
            state={investment.state}
            onClose={() => setInvestOpen(false)}
          />
        ) : null}
      </AnimatePresence>

      <AnimatePresence>
        {plan && planOpen ? (
          <motion.div
            key="plan-panel"
            data-player-ui
            initial={{ opacity: 0, y: 10, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.99 }}
            transition={{ duration: 0.24, ease: EASE }}
            // The origin is the bar below it, so it grows out of the corner it belongs to.
            style={{ transformOrigin: "bottom left" }}
            className={cn(
              "glass-strong pointer-events-auto relative overflow-hidden rounded-2xl",
              "shadow-[var(--shadow-lift)]",
              "h-[min(21rem,52vh)] w-[min(26rem,calc(100vw-1.5rem))]",
            )}
          >
            <PlanView
              plan={plan.plan}
              currentSceneId={plan.currentSceneId}
              locale={locale}
              open
              onClose={() => plan.onOpenChange(false)}
              onSelectRoom={(sceneId) => {
                plan.onOpenChange(false);
                plan.onSelectRoom(sceneId);
              }}
              headingRef={plan.headingRef}
            />
          </motion.div>
        ) : null}
      </AnimatePresence>

      {hasBar ? (
        <div
          data-player-ui
          className={cn(
            "glass-strong pointer-events-auto flex max-w-[calc(100vw-1.5rem)] items-center gap-1.5",
            "rounded-full p-1.5 shadow-[var(--shadow-lift)]",
            // Scroll, never wrap — see the header. The bar is a rounded pill, so the scrollbar
            // would cut a chord across it; every browser we ship to can hide it.
            "overflow-x-auto [-ms-overflow-style:none] [scrollbar-width:none]",
            "[&::-webkit-scrollbar]:hidden",
          )}
        >
          {toolProps ? <ToolSegments {...toolProps} /> : null}

          {toolProps && (plan || measure || investmentOn) ? <Hairline /> : null}

          {plan ? (
            <DockButton
              label={t("plan.open", locale)}
              active={plan.open}
              expanded={plan.open}
              onClick={() => plan.onOpenChange(!plan.open)}
            >
              <Map size={14} aria-hidden />
            </DockButton>
          ) : null}

          {measure ? (
            <DockButton
              label={measure.active ? t("player.measure.off", locale) : t("player.measure", locale)}
              active={measure.active}
              onClick={measure.onToggle}
            >
              <Ruler size={14} aria-hidden />
            </DockButton>
          ) : null}

          {/* Labelled like everything beside it. It shipped icon-only on the argument that
              the spinning picture is its own label — true once the drift has started, and no
              help at all before it, which is the moment the button has to be found. A lone
              glyph in a row of words also reads as a different KIND of thing, so the eye
              skips it. The words are the state: "Döndürmeyi aç" / "Döndürmeyi durdur". */}
          {autoRotate ? (
            <DockButton
              label={
                autoRotate.value ? t("player.autorotate.off", locale) : t("player.autorotate", locale)
              }
              active={autoRotate.value}
              onClick={() => autoRotate.onChange(!autoRotate.value)}
            >
              <RotateCw size={14} className={cn(autoRotate.value && "animate-spin-slow")} aria-hidden />
            </DockButton>
          ) : null}

          {/* With the drift and the measure tool: a control that puts the player into a state. */}
          {props.vr ? <VrDockSegment locale={locale} active={props.vr.active} onClick={props.vr.onClick} /> : null}

          {investmentOn && (tools || plan || measure || autoRotate || props.vr) ? <Hairline /> : null}

          {/* Last, past the hairline: everything before it acts on the picture, this one
              answers questions about the purchase — a different kind of control. Present only
              where the agent opted in (see `InvestmentChat`). */}
          {investmentOn ? (
            <DockButton
              label={t("player.invest.title", locale)}
              caption={t("player.invest", locale)}
              active={investOpen}
              expanded={investOpen}
              onClick={() => setInvestOpen((open) => !open)}
            >
              <TrendingUp size={14} aria-hidden />
            </DockButton>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/** Where the meaning changes. `shrink-0` because the bar scrolls rather than compressing. */
function Hairline() {
  return <span aria-hidden className="h-5 w-px shrink-0 bg-line" />;
}
