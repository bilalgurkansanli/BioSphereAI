"use client";

/**
 * The permission step — the only screen between "Başla" and the camera.
 *
 * THE ONE RULE THIS COMPONENT EXISTS TO PROTECT: `onRequest` is called straight out of the
 * click handler with nothing awaited in between. iOS gates
 * `DeviceOrientationEvent.requestPermission()` behind a *transient user activation*: the
 * activation is consumed by the first `await`, so a parent that does
 * `await getUserMedia(); await requestPermission()` silently loses motion access on every
 * iPhone, and the capture falls back to RAIL MODE for no reason at all. Hence a plain
 * `onClick={onRequest}` here and a synchronous request chain in the parent.
 *
 * The screen has three faces, chosen from `status` alone:
 *   1. in-app browser  — no permission can succeed, so nothing is requested;
 *   2. camera denied   — an honest dead end with two ways out;
 *   3. the ask         — what each sensor is for, in one line each.
 * A motion denial is never a dead end: it degrades to a timed guide, which is why it shows
 * as a note inside the ask rather than as its own screen.
 */

import { Camera, CameraOff, Check, Compass, Copy, ExternalLink, RotateCcw, Video } from "lucide-react";
import { useEffect, useState, useSyncExternalStore } from "react";
import { Button } from "@/components/ui/Button";
import { cc } from "@/lib/capture/copy";
import type { PermissionGateProps } from "@/lib/capture/types";
import { cn } from "@/lib/utils";

export function PermissionGate({ status, onRequest, onUseExistingVideo, onRetry, locale }: PermissionGateProps) {
  if (status.inAppBrowser) return <InAppBrowserWall locale={locale} />;
  if (status.camera === "denied" || status.camera === "unavailable") {
    return <CameraDenied locale={locale} onRetry={onRetry} onUseExistingVideo={onUseExistingVideo} />;
  }
  return (
    <Ask
      locale={locale}
      motionDenied={status.motion === "denied"}
      onRequest={onRequest}
      onUseExistingVideo={onUseExistingVideo}
    />
  );
}

// ── The ask ────────────────────────────────────────────────────────────────────────────

function Ask({
  locale,
  motionDenied,
  onRequest,
  onUseExistingVideo,
}: {
  locale: PermissionGateProps["locale"];
  motionDenied: boolean;
  onRequest: () => void;
  onUseExistingVideo: () => void;
}) {
  return (
    <Shell>
      <h1 className="text-[22px] leading-tight font-semibold tracking-tight text-ink">
        {cc("capture.perm.title", locale)}
      </h1>
      <p className="mt-2.5 text-[14px] leading-relaxed text-ink-muted">{cc("capture.perm.body", locale)}</p>

      <ul className="mt-6 space-y-3">
        <Reason
          icon={<Camera className="size-[18px]" strokeWidth={1.7} aria-hidden />}
          title={locale === "tr" ? "Kamera" : "Camera"}
          body={
            locale === "tr"
              ? "Odanın görüntüsünü kaydeder. Görüntü telefonunuzda kalır, siz yükle demeden gitmez."
              : "Records the room. The footage stays on your phone until you choose to upload it."
          }
        />
        <Reason
          icon={<Compass className="size-[18px]" strokeWidth={1.7} aria-hidden />}
          title={locale === "tr" ? "Hareket ve yön" : "Motion & orientation"}
          body={
            locale === "tr"
              ? "Telefonun nereye baktığını söyler — hangi yönleri çektiğinizi bu sayede bilebiliyoruz."
              : "Tells us where the phone is pointing — it is how we know which directions you have covered."
          }
          note={
            motionDenied
              ? // The dictionary line promises the timed guide; this second sentence names
                // its one real cost, which the user is entitled to know BEFORE they spend a
                // minute turning rather than on the review screen afterwards.
                `${cc("capture.perm.motion.denied", locale)} ${
                  locale === "tr"
                    ? "Video yine tam kalitede kaydedilir; sadece kapsamayı ölçemeyiz."
                    : "The video still records at full quality; we simply cannot measure the coverage."
                }`
              : null
          }
        />
      </ul>

      {/* The gesture must reach requestPermission() unbroken: no async wrapper, no confirm
          dialog, no analytics await between the tap and the call. */}
      <Button variant="primary" size="lg" className="mt-7 w-full" onClick={onRequest}>
        {cc("capture.perm.allow", locale)}
      </Button>

      <button
        type="button"
        onClick={onUseExistingVideo}
        className="mt-3 w-full py-2 text-[13px] text-ink-faint transition-colors hover:text-ink-muted"
      >
        {cc("capture.intro.have", locale)}
      </button>
    </Shell>
  );
}

function Reason({
  icon,
  title,
  body,
  note,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
  note?: string | null;
}) {
  return (
    <li className="flex gap-3.5 rounded-md border border-line bg-surface-2/40 p-3.5">
      <span className="mt-0.5 grid size-9 shrink-0 place-items-center rounded-full border border-line bg-surface-3 text-bio">
        {icon}
      </span>
      <div className="min-w-0">
        <p className="text-[13px] font-semibold tracking-tight text-ink">{title}</p>
        <p className="mt-1 text-[12.5px] leading-relaxed text-ink-muted">{body}</p>
        {note ? <p className="mt-2 text-[12px] leading-relaxed text-amber">{note}</p> : null}
      </div>
    </li>
  );
}

// ── Camera denied ──────────────────────────────────────────────────────────────────────

function CameraDenied({
  locale,
  onRetry,
  onUseExistingVideo,
}: {
  locale: PermissionGateProps["locale"];
  onRetry: () => void;
  onUseExistingVideo: () => void;
}) {
  return (
    <Shell>
      <span className="grid size-12 place-items-center rounded-full border border-danger/30 bg-danger/10 text-danger">
        <CameraOff className="size-5" strokeWidth={1.7} aria-hidden />
      </span>
      <h1 className="mt-5 text-[20px] leading-tight font-semibold tracking-tight text-ink">
        {locale === "tr" ? "Kameraya erişilemiyor" : "The camera is unavailable"}
      </h1>
      {/* No pretending: a denied camera cannot be re-prompted from inside the page on
          either platform, so the copy names the exact place the switch lives. */}
      <p className="mt-2.5 text-[14px] leading-relaxed text-ink-muted">{cc("capture.perm.camera.denied", locale)}</p>

      <div className="mt-7 space-y-2.5">
        <Button variant="primary" size="lg" className="w-full" onClick={onRetry}>
          <RotateCcw className="size-4" aria-hidden />
          {locale === "tr" ? "Tekrar dene" : "Try again"}
        </Button>
        <Button variant="secondary" size="lg" className="w-full" onClick={onUseExistingVideo}>
          <Video className="size-4" aria-hidden />
          {cc("capture.intro.have", locale)}
        </Button>
      </div>
    </Shell>
  );
}

// ── In-app browser ─────────────────────────────────────────────────────────────────────

function InAppBrowserWall({ locale }: { locale: PermissionGateProps["locale"] }) {
  const href = useCurrentHref();
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(timer);
  }, [copied]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(href);
      setCopied(true);
    } catch {
      // Clipboard access is refused inside several in-app webviews, which is exactly where
      // this screen appears. The address stays selectable below, so the user can still
      // long-press it — never leave them with only a button that does nothing.
      setCopied(false);
    }
  };

  return (
    <Shell>
      <span className="grid size-12 place-items-center rounded-full border border-amber/30 bg-amber/10 text-amber">
        <ExternalLink className="size-5" strokeWidth={1.7} aria-hidden />
      </span>
      <h1 className="mt-5 text-[20px] leading-tight font-semibold tracking-tight text-ink">
        {cc("capture.perm.inapp", locale)}
      </h1>
      <p className="mt-2.5 text-[14px] leading-relaxed text-ink-muted">{cc("capture.perm.inapp.body", locale)}</p>

      <p className="mt-6 mb-2 text-[12px] font-medium text-ink-faint">
        {locale === "tr" ? "Bu adresi kopyalayın" : "Copy this address"}
      </p>
      <p className="rounded-md border border-line bg-surface-2/60 px-3 py-2.5 font-mono text-[12px] leading-relaxed break-all text-ink-muted select-all">
        {href || "…"}
      </p>
      <Button variant="secondary" size="lg" className="mt-3 w-full" onClick={() => void copy()} disabled={!href}>
        {copied ? <Check className="size-4" aria-hidden /> : <Copy className="size-4" aria-hidden />}
        {copied
          ? locale === "tr"
            ? "Kopyalandı"
            : "Copied"
          : locale === "tr"
            ? "Adresi kopyala"
            : "Copy address"}
      </Button>
    </Shell>
  );
}

/**
 * The current address, as an external store rather than a state-in-effect pair. The page is
 * prerendered on the server, where `location` does not exist, so the server snapshot is the
 * empty string and the real value arrives on the first client render — not one render later,
 * which would show an empty box to the very user who is being asked to copy the link.
 *
 * Nothing subscribes: a full navigation unmounts this component anyway.
 */
function useCurrentHref(): string {
  return useSyncExternalStore(
    () => () => undefined,
    () => window.location.href,
    () => "",
  );
}

// ── Shared frame ───────────────────────────────────────────────────────────────────────

function Shell({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("grid-bg flex min-h-[100dvh] w-full items-center justify-center px-5 py-10", className)}>
      <div className="animate-rise glass w-full max-w-[420px] rounded-lg p-6 sm:p-7">{children}</div>
    </div>
  );
}
