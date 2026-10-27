import { cn } from "@/lib/utils";

/**
 * "Yakında", on a control that exists but does not work yet.
 *
 * ── WHY IT IS NOT `BetaTag` ───────────────────────────────────────────────────────────
 * They look alike and mean opposite things about whether pressing the thing does anything.
 * Beta says "this runs, and it may be wrong" — the empty-room tool really does empty a room.
 * This says "this does not run yet". Sharing one component would eventually put one word on
 * both, and the first agent to press a beta-labelled button that did nothing would file it as
 * a bug. So: a second component, a different colour, and no way to confuse them.
 *
 * ── AND WHY THE WORD IS A PROP ────────────────────────────────────────────────────────
 * `BetaTag` hard-codes "beta" because it is the same word in both languages this product
 * speaks. "Yakında" is not "soon", so this one has to be told. It ships inside the player,
 * which is embedded on third-party pages and must not import the dashboard's copy table —
 * the caller has the locale, the caller passes the word.
 */
export interface SoonTagProps {
  /** Localised: "yakında" / "soon". */
  label: string;
  className?: string;
}

export function SoonTag({ label, className }: SoonTagProps) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center rounded-full border border-sky/25 bg-sky/10",
        "px-1.5 py-0.5 text-[10px] leading-none font-semibold tracking-wide text-sky uppercase",
        className,
      )}
    >
      {label}
    </span>
  );
}
