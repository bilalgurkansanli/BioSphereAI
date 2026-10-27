import { clamp, cn } from "@/lib/utils";

export type ProgressTone = "bio" | "amber" | "danger" | "success";

const FILLS: Record<ProgressTone, string> = {
  bio: "bg-[linear-gradient(90deg,var(--bio),var(--bio-soft))]",
  amber: "bg-amber",
  danger: "bg-danger",
  success: "bg-success",
};

export interface ProgressProps {
  /** 0..1. Omit for the indeterminate treatment. */
  value?: number;
  tone?: ProgressTone;
  size?: "sm" | "md";
  label: string;
  className?: string;
}

export function Progress({ value, tone = "bio", size = "md", label, className }: ProgressProps) {
  const indeterminate = value == null || !Number.isFinite(value);
  const pct = indeterminate ? 0 : clamp(value, 0, 1) * 100;

  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : Math.round(pct)}
      className={cn(
        "relative w-full overflow-hidden rounded-full bg-surface-3",
        size === "sm" ? "h-1" : "h-1.5",
        className,
      )}
    >
      {indeterminate ? (
        <div
          className="animate-shimmer absolute inset-0 bg-[linear-gradient(90deg,transparent_0%,var(--bio)_50%,transparent_100%)] opacity-70"
          aria-hidden
        />
      ) : (
        <div
          className={cn("h-full rounded-full transition-[width] duration-500 ease-out-expo", FILLS[tone])}
          style={{ width: `${pct}%` }}
          aria-hidden
        />
      )}
    </div>
  );
}
