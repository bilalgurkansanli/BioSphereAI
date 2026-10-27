import { Loader2 } from "lucide-react";
import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export type ButtonVariant = "primary" | "secondary" | "outline" | "ghost" | "danger";
export type ButtonSize = "sm" | "md" | "lg" | "icon" | "icon-sm";

const BASE =
  "relative inline-flex shrink-0 select-none items-center justify-center gap-2 whitespace-nowrap " +
  "rounded-md font-medium leading-none transition-all duration-150 ease-out-expo " +
  "active:translate-y-px disabled:pointer-events-none disabled:opacity-40";

const VARIANTS: Record<ButtonVariant, string> = {
  // The one saturated surface in the product. Dark ink on it keeps contrast well above AA.
  primary:
    "bg-[linear-gradient(120deg,var(--bio-soft),var(--bio))] text-canvas font-semibold " +
    "shadow-[0_6px_24px_-10px_var(--bio)] hover:brightness-110 hover:shadow-[0_8px_30px_-8px_var(--bio)]",
  secondary: "bg-surface-2 text-ink border border-line hover:bg-surface-3 hover:border-line-strong",
  outline: "border border-line-strong text-ink hover:bg-surface-2",
  ghost: "text-ink-muted hover:text-ink hover:bg-surface-2",
  danger: "border border-danger/30 bg-danger/10 text-danger hover:bg-danger/18 hover:border-danger/50",
};

const SIZES: Record<ButtonSize, string> = {
  sm: "h-8 gap-1.5 px-3 text-[13px]",
  md: "h-10 px-4 text-sm",
  lg: "h-12 px-6 text-[15px]",
  icon: "h-10 w-10 p-0",
  "icon-sm": "h-8 w-8 p-0",
};

export function buttonStyles(
  variant: ButtonVariant = "secondary",
  size: ButtonSize = "md",
  className?: string,
): string {
  return cn(BASE, VARIANTS[variant], SIZES[size], className);
}

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export function Button({
  variant = "secondary",
  size = "md",
  loading = false,
  className,
  children,
  disabled,
  type = "button",
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      className={buttonStyles(variant, size, className)}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? (
        <>
          <Loader2 className="absolute size-4 animate-spin" aria-hidden />
          {/* Keeping the label in flow stops the button from resizing mid-action. */}
          <span className="invisible contents">{children}</span>
        </>
      ) : (
        children
      )}
    </button>
  );
}
