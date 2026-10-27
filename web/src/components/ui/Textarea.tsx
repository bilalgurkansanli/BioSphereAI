"use client";

import { useId, type TextareaHTMLAttributes } from "react";
import { cn } from "@/lib/utils";
import { FieldShell, fieldStyles } from "./Input";

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  hint?: string;
  error?: string;
  wrapperClassName?: string;
}

export function Textarea({ label, hint, error, className, wrapperClassName, id, rows = 3, ...rest }: TextareaProps) {
  const autoId = useId();
  const fieldId = id ?? autoId;

  return (
    <FieldShell
      id={fieldId}
      label={label}
      hint={hint}
      error={error}
      required={rest.required}
      className={wrapperClassName}
    >
      <textarea
        id={fieldId}
        rows={rows}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${fieldId}-error` : hint ? `${fieldId}-hint` : undefined}
        className={cn(
          fieldStyles,
          "resize-y py-2.5 leading-relaxed",
          error && "border-danger/60 focus:border-danger focus:ring-danger/20",
          className,
        )}
        {...rest}
      />
    </FieldShell>
  );
}
