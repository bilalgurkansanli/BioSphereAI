"use client";

import { Loader2 } from "lucide-react";
import { useId } from "react";
import { cn } from "@/lib/utils";

export interface SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  description?: string;
  disabled?: boolean;
  loading?: boolean;
  /** Accessible name when no visible label is rendered. */
  ariaLabel?: string;
  className?: string;
}

export function Switch({
  checked,
  onChange,
  label,
  description,
  disabled = false,
  loading = false,
  ariaLabel,
  className,
}: SwitchProps) {
  const id = useId();
  const control = (
    <button
      type="button"
      role="switch"
      id={id}
      aria-checked={checked}
      aria-label={label ? undefined : ariaLabel}
      aria-labelledby={label ? `${id}-label` : undefined}
      aria-busy={loading || undefined}
      disabled={disabled || loading}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full border transition-colors duration-200",
        checked ? "border-bio/40 bg-bio/80" : "border-line bg-surface-3",
        (disabled || loading) && "cursor-not-allowed opacity-50",
      )}
    >
      <span
        className={cn(
          "grid size-4.5 place-items-center rounded-full bg-canvas shadow-sm transition-transform duration-200 ease-out-expo",
          checked ? "translate-x-[22px]" : "translate-x-[3px]",
        )}
      >
        {loading ? <Loader2 className="size-3 animate-spin text-ink-muted" aria-hidden /> : null}
      </span>
    </button>
  );

  if (!label && !description) return <div className={className}>{control}</div>;

  return (
    <div className={cn("flex items-start justify-between gap-4", className)}>
      <div className="min-w-0">
        {label ? (
          <label id={`${id}-label`} htmlFor={id} className="block cursor-pointer text-[13px] font-medium text-ink">
            {label}
          </label>
        ) : null}
        {description ? <p className="mt-0.5 text-[12px] leading-relaxed text-ink-faint">{description}</p> : null}
      </div>
      {control}
    </div>
  );
}
