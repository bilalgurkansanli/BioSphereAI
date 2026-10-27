import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/**
 * A skeleton is only honest if it matches the shape it replaces, so callers size it and
 * this component only supplies the surface treatment.
 */
export function Skeleton({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      aria-hidden
      className={cn(
        "animate-shimmer rounded-sm bg-[linear-gradient(90deg,var(--surface-2)_0%,var(--surface-3)_50%,var(--surface-2)_100%)]",
        className,
      )}
      {...rest}
    />
  );
}

/** The property grid's placeholder: same geometry as a real card, so nothing jumps on load. */
export function SkeletonPropertyCard() {
  return (
    <div className="glass overflow-hidden rounded-lg">
      <Skeleton className="aspect-[16/10] w-full rounded-none" />
      <div className="space-y-3 p-4">
        <Skeleton className="h-4 w-3/5" />
        <Skeleton className="h-3 w-2/5" />
        <div className="flex gap-2 pt-1">
          <Skeleton className="h-5 w-16 rounded-full" />
          <Skeleton className="h-5 w-20 rounded-full" />
        </div>
      </div>
    </div>
  );
}
