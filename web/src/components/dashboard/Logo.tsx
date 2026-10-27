import { cn } from "@/lib/utils";

/** A sphere with a meridian — the product mark. Inherits `currentColor`. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      className={cn("size-6", className)}
      aria-hidden
    >
      <circle cx="12" cy="12" r="9" />
      <ellipse cx="12" cy="12" rx="4.1" ry="9" />
      <path d="M3.2 9.4h17.6M3.2 14.6h17.6" />
    </svg>
  );
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cn("flex items-center gap-2.5", className)}>
      <span className="grid size-8 shrink-0 place-items-center rounded-sm border border-line bg-surface-2 text-bio">
        <LogoMark className="size-4.5" />
      </span>
      <span className="text-[15px] font-semibold tracking-tight text-ink">BioSphere</span>
    </span>
  );
}
