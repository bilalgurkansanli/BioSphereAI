"use client";

import { ChevronDown } from "lucide-react";
import { useId, type SelectHTMLAttributes } from "react";
import { cn } from "@/lib/utils";
import { FieldShell, fieldStyles } from "./Input";

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, "children"> {
  label?: string;
  hint?: string;
  error?: string;
  options: SelectOption[];
  placeholder?: string;
  wrapperClassName?: string;
}

/**
 * A native `<select>` on purpose: it inherits the platform's keyboard behaviour, its
 * mobile wheel picker, and screen-reader support for free. Only the chrome is restyled.
 */
export function Select({
  label,
  hint,
  error,
  options,
  placeholder,
  className,
  wrapperClassName,
  id,
  ...rest
}: SelectProps) {
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
      <div className="relative">
        <select
          id={fieldId}
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? `${fieldId}-error` : hint ? `${fieldId}-hint` : undefined}
          className={cn(
            fieldStyles,
            "h-10 cursor-pointer appearance-none pr-9",
            error && "border-danger/60 focus:border-danger focus:ring-danger/20",
            className,
          )}
          {...rest}
        >
          {placeholder ? (
            <option value="" disabled>
              {placeholder}
            </option>
          ) : null}
          {options.map((option) => (
            <option key={option.value} value={option.value} disabled={option.disabled}>
              {option.label}
            </option>
          ))}
        </select>
        <ChevronDown
          className="pointer-events-none absolute top-1/2 right-3 size-4 -translate-y-1/2 text-ink-faint"
          aria-hidden
        />
      </div>
    </FieldShell>
  );
}
