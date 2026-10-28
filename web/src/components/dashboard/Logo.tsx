import { cn } from "@/lib/utils";

/**
 * The product mark: a "B" whose bowls are a 360° orbit around the camera standing in the
 * middle of the room. A transparent PNG (`public/brand/logo-mark.png`, built from
 * `docs/brand/logo-source.jpg`), so it carries its own gradient rather than `currentColor`.
 */
export function LogoMark({ className }: { className?: string }) {
  return (
    // Plain <img>: a 512 px static asset shown at icon sizes needs no optimiser round-trip.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src="/brand/logo-mark.png"
      alt=""
      aria-hidden
      draggable={false}
      width={512}
      height={512}
      className={cn("size-6 shrink-0 object-contain select-none", className)}
    />
  );
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cn("flex items-center gap-2.5", className)}>
      <span className="grid size-8 shrink-0 place-items-center rounded-sm border border-line bg-surface-2">
        <LogoMark className="size-5" />
      </span>
      <span className="text-[15px] font-semibold tracking-tight text-ink">BioSphere</span>
    </span>
  );
}
