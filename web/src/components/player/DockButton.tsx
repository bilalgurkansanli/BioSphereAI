"use client";

/**
 * One segment of the dock's bar.
 *
 * A shared primitive because the bar is now a single row of controls owned by two files —
 * `PlayerDock` puts the plan, the measure switch and the drift on it, `PlayerTools` the two
 * that rewrite the photograph — and the moment those two files each drew their own pill the
 * corner went back to looking like a pile of unrelated things. Same height, same radius, same
 * border, same accent when active: that is what makes a row of buttons read as one control
 * cluster rather than as five buttons that happen to be adjacent.
 *
 * It is deliberately the same shape as `PlayerChrome`'s `ToolButton`. The two bars sit on the
 * same edge of the same photograph and a viewer sees them together; if they differed, the
 * difference would be read as meaning something.
 */

import { cn } from "@/lib/utils";

export interface DockButtonProps {
  /** The accessible name and the tooltip, and the visible text unless `caption` overrides. */
  label: string;
  /**
   * Shorter visible text, where the accessible name has to say more than the bar has room
   * for. The two must not disagree: this is a prefix of the name, never a different word —
   * a screen reader announcing something the sighted user cannot see is a second interface.
   */
  caption?: string;
  /** Icon-only turns the segment into a square. For controls whose state IS the picture. */
  showLabel?: boolean;
  active: boolean;
  /** Mid-run: the press is refused and the cursor says why, but the label stays readable. */
  busy?: boolean;
  /** Set instead of `aria-pressed` where the control opens a panel rather than toggling. */
  expanded?: boolean;
  /**
   * The feature is announced but not built. Dashed and faint, so it is legible as unfinished
   * before anything is read — a solid pill that opens an apology is a bug report waiting to
   * happen. It still presses: the panel it opens is the whole point of the segment.
   */
  soon?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

export function DockButton({
  label,
  caption,
  showLabel = true,
  active,
  busy = false,
  expanded,
  soon = false,
  onClick,
  children,
}: DockButtonProps) {
  return (
    <button
      type="button"
      data-player-ui
      onClick={onClick}
      disabled={busy}
      aria-label={label}
      aria-pressed={expanded === undefined ? active : undefined}
      aria-expanded={expanded}
      title={label}
      className={cn(
        "flex h-8 shrink-0 items-center rounded-full border",
        "text-[12px] font-medium tracking-tight whitespace-nowrap",
        "transition-all duration-200 ease-[var(--ease-out-expo)]",
        showLabel ? "gap-1.5 px-2.5" : "w-8 justify-center",
        busy ? "cursor-wait" : "active:scale-95",
        soon ? "border-dashed" : "",
        active
          ? soon
            ? "border-sky/50 bg-sky/10 text-sky"
            : "border-bio/50 bg-bio/12 text-bio"
          : soon
            ? "border-line text-ink-faint hover:border-sky/40 hover:text-sky"
            : "border-line text-ink-muted hover:border-line-strong hover:bg-white/[0.05] hover:text-ink",
      )}
    >
      {children}
      {showLabel ? <span>{caption ?? label}</span> : null}
    </button>
  );
}
