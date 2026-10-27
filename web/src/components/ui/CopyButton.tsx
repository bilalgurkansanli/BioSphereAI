"use client";

import { Check, Copy, X } from "lucide-react";
import { useCopyToClipboard } from "@/lib/hooks/useCopyToClipboard";
import { cn } from "@/lib/utils";
import { Button, type ButtonProps } from "./Button";

export interface CopyButtonProps extends Omit<ButtonProps, "onClick" | "children" | "value"> {
  value: string;
  label: string;
  copiedLabel: string;
  failedLabel?: string;
  /** Renders the icon only; `label` becomes the accessible name. */
  iconOnly?: boolean;
}

export function CopyButton({
  value,
  label,
  copiedLabel,
  failedLabel,
  iconOnly = false,
  variant = "secondary",
  size,
  className,
  ...rest
}: CopyButtonProps) {
  const { copied, failed, copy } = useCopyToClipboard();

  const Icon = copied ? Check : failed ? X : Copy;
  const text = copied ? copiedLabel : failed ? (failedLabel ?? label) : label;

  return (
    <Button
      variant={copied ? "primary" : variant}
      size={size ?? (iconOnly ? "icon-sm" : "sm")}
      onClick={() => void copy(value)}
      aria-label={iconOnly ? text : undefined}
      className={cn("transition-colors", failed && "text-danger", className)}
      {...rest}
    >
      <Icon className="size-3.5 shrink-0" aria-hidden />
      {iconOnly ? null : <span>{text}</span>}
      {/* Announce the outcome without moving focus. */}
      <span className="sr-only" role="status" aria-live="polite">
        {copied ? copiedLabel : ""}
      </span>
    </Button>
  );
}
