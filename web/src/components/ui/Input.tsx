"use client";

import { useId, type InputHTMLAttributes, type ReactNode } from "react";
import { cn } from "@/lib/utils";

export const fieldStyles =
  "w-full rounded-md border border-line bg-surface-2/70 px-3 text-sm text-ink placeholder:text-ink-faint " +
  "transition-colors duration-150 outline-none hover:border-line-strong " +
  "focus:border-bio/60 focus:bg-surface-2 focus:ring-2 focus:ring-bio/20 " +
  "disabled:cursor-not-allowed disabled:opacity-50";

export interface FieldShellProps {
  id: string;
  label?: string;
  hint?: string;
  error?: string;
  required?: boolean;
  trailing?: ReactNode;
  className?: string;
  children: ReactNode;
}

/** Shared label / hint / error scaffolding so every field reports errors identically. */
export function FieldShell({ id, label, hint, error, required, className, children }: FieldShellProps) {
  return (
    <div className={cn("space-y-1.5", className)}>
      {label ? (
        <label htmlFor={id} className="flex items-center gap-1 text-[13px] font-medium text-ink-muted">
          {label}
          {required ? (
            <span className="text-bio" aria-hidden>
              *
            </span>
          ) : null}
        </label>
      ) : null}
      {children}
      {error ? (
        <p id={`${id}-error`} className="text-[12px] leading-relaxed text-danger">
          {error}
        </p>
      ) : hint ? (
        <p id={`${id}-hint`} className="text-[12px] leading-relaxed text-ink-faint">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "size"> {
  label?: string;
  hint?: string;
  error?: string;
  wrapperClassName?: string;
}

export function Input({ label, hint, error, className, wrapperClassName, id, ...rest }: InputProps) {
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
      <input
        id={fieldId}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${fieldId}-error` : hint ? `${fieldId}-hint` : undefined}
        className={cn(fieldStyles, "h-10", error && "border-danger/60 focus:border-danger focus:ring-danger/20", className)}
        {...rest}
      />
    </FieldShell>
  );
}
