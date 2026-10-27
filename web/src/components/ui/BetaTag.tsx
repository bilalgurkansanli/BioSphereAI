import { cn } from "@/lib/utils";

/**
 * "Beta", on the surfaces where a model produced what the user is looking at.
 *
 * ── WHY IT IS ITS OWN COMPONENT AND NOT A `<Badge>` ───────────────────────────────────
 * `Badge` is a status: ready, failed, published, draft — something about THIS row that will
 * change. This is a statement about the FEATURE, it is the same everywhere it appears, and it
 * has to look the same everywhere or it stops reading as one label and starts reading as five
 * unrelated decorations. One component is how that stays true.
 *
 * It also has to work in the player, which is embedded on third-party sites and must not
 * import the dashboard's copy table — so the word is hard-coded (it is "beta" in both
 * languages the product speaks) and the explanation arrives as a prop from whoever has a
 * locale to hand.
 */
export interface BetaTagProps {
  /** Localised sentence for the tooltip. Optional, but the label means more with it. */
  title?: string;
  className?: string;
}

export function BetaTag({ title, className }: BetaTagProps) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex shrink-0 items-center rounded-full border border-amber/30 bg-amber/10",
        "px-1.5 py-0.5 text-[10px] leading-none font-semibold tracking-wide text-amber uppercase",
        className,
      )}
    >
      beta
    </span>
  );
}
