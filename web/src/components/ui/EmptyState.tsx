import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  body?: string;
  action?: ReactNode;
  secondary?: ReactNode;
  tone?: "neutral" | "danger";
  className?: string;
}

export function EmptyState({
  icon: Icon,
  title,
  body,
  action,
  secondary,
  tone = "neutral",
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "relative isolate flex flex-col items-center justify-center overflow-hidden rounded-lg border border-line " +
          "bg-canvas-raised px-6 py-16 text-center sm:py-20",
        className,
      )}
    >
      {/* The blueprint field turns a dead end into a deliberate-looking surface. */}
      <div className="grid-bg pointer-events-none absolute inset-0 -z-10 opacity-70" aria-hidden />

      <div
        className={cn(
          "mb-5 flex size-14 items-center justify-center rounded-xl border",
          tone === "danger" ? "border-danger/25 bg-danger/10 text-danger" : "border-line bg-surface-2 text-bio",
        )}
      >
        <Icon className="size-6" strokeWidth={1.6} aria-hidden />
      </div>

      <h3 className="text-balance text-lg font-semibold tracking-tight text-ink">{title}</h3>
      {body ? <p className="mt-2 max-w-sm text-pretty text-sm leading-relaxed text-ink-muted">{body}</p> : null}

      {action || secondary ? (
        <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
          {action}
          {secondary}
        </div>
      ) : null}
    </div>
  );
}
