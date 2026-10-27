import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export type BadgeTone = "neutral" | "bio" | "amber" | "danger" | "success" | "sky";

const TONES: Record<BadgeTone, string> = {
  neutral: "border-line bg-surface-2 text-ink-muted",
  bio: "border-bio/25 bg-bio/10 text-bio",
  amber: "border-amber/25 bg-amber/10 text-amber",
  danger: "border-danger/25 bg-danger/10 text-danger",
  success: "border-success/25 bg-success/10 text-success",
  sky: "border-sky/25 bg-sky/10 text-sky",
};

const DOTS: Record<BadgeTone, string> = {
  neutral: "bg-ink-faint",
  bio: "bg-bio",
  amber: "bg-amber",
  danger: "bg-danger",
  success: "bg-success",
  sky: "bg-sky",
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
  dot?: boolean;
  /** Gently pulses the dot — used for "processing". */
  pulse?: boolean;
}

export function Badge({ tone = "neutral", dot = false, pulse = false, className, children, ...rest }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-medium " +
          "leading-5 tracking-wide whitespace-nowrap",
        TONES[tone],
        className,
      )}
      {...rest}
    >
      {dot ? (
        <span className="relative flex size-1.5">
          {pulse ? (
            <span className={cn("absolute inline-flex size-full rounded-full opacity-70 animate-breathe", DOTS[tone])} />
          ) : null}
          <span className={cn("relative inline-flex size-1.5 rounded-full", DOTS[tone])} />
        </span>
      ) : null}
      {children}
    </span>
  );
}
