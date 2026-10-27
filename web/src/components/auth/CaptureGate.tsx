"use client";

/**
 * The door of `/cek`: a capture link from a QR code, or a signed-in account.
 *
 * The capture page opens on a DIFFERENT device from the dashboard that drew its QR code, and
 * until capture links existed that device met a sign-in form first. Now the QR carries `k`, a
 * short-lived token the engine honours for one property and only on the routes a capture
 * needs (`api/deps.py`, `_CAPTURE_ROUTES`). This gate decides which of three doors applies:
 *
 *   `?k=` in the address  → adopt it into this tab, strip it from the address bar (a URL gets
 *                           screenshotted and shared; the token should not ride along), and
 *                           render the capture with no sign-in at all.
 *   a session             → exactly the old behaviour: `RequireSession`.
 *   a grant that the engine refused (expired, revoked, property handed to someone else)
 *                         → say so in words a phone holder can act on — scan a new code —
 *                           rather than redirecting somebody who may have no account to a
 *                           sign-in form.
 *
 * WHY NOT `useSearchParams`. Reading it here would make every route under this layout
 * client-render up to a Suspense boundary, and the decision does not need React's view of the
 * URL: it needs `sessionStorage`, which only the browser has. So the snapshot is read from the
 * browser directly through `useSyncExternalStore`, whose server snapshot is "pending" — the
 * same fallback `RequireSession` shows while it asks the engine who you are.
 *
 * WHILE `k` IS STILL IN THE ADDRESS the answer is "pending", never "session". Otherwise the
 * first client render would mount `RequireSession`, find no session and navigate to `/giris`
 * a few milliseconds before the effect below adopted the grant.
 */

import { KeyRound, ScanLine } from "lucide-react";
import Link from "next/link";
import { useEffect, useSyncExternalStore, type ReactNode } from "react";
import { RequireSession, SessionProvider } from "@/components/auth/SessionProvider";
import { buttonStyles } from "@/components/ui/Button";
import { useLocale } from "@/lib/hooks/useLocale";
import {
  CAPTURE_GRANT_EVENT,
  SIGNED_OUT_EVENT,
  adoptCaptureGrantFromAddress,
  captureGateMode,
  type CaptureGateMode,
} from "@/lib/session";

function subscribe(onChange: () => void): () => void {
  window.addEventListener(CAPTURE_GRANT_EVENT, onChange);
  window.addEventListener(SIGNED_OUT_EVENT, onChange);
  return () => {
    window.removeEventListener(CAPTURE_GRANT_EVENT, onChange);
    window.removeEventListener(SIGNED_OUT_EVENT, onChange);
  };
}

export function CaptureGate({ children, fallback }: { children: ReactNode; fallback: ReactNode }) {
  useEffect(() => {
    adoptCaptureGrantFromAddress();
  }, []);

  const mode = useSyncExternalStore<CaptureGateMode>(subscribe, captureGateMode, () => "pending");

  if (mode === "pending") return <>{fallback}</>;
  if (mode === "grant") return <>{children}</>;
  if (mode === "expired") return <ExpiredLink />;
  return (
    <SessionProvider>
      <RequireSession fallback={fallback}>{children}</RequireSession>
    </SessionProvider>
  );
}

function ExpiredLink() {
  const { locale } = useLocale();
  const tr = locale === "tr";
  const next = typeof window !== "undefined" ? `${window.location.pathname}${window.location.search}` : "/cek";
  return (
    <main className="grid min-h-dvh place-items-center px-5">
      <div className="glass w-full max-w-sm rounded-lg p-6 text-center">
        <span className="mx-auto grid size-11 place-items-center rounded-full border border-line bg-surface-3 text-amber">
          <ScanLine className="size-5" strokeWidth={1.7} aria-hidden />
        </span>
        <h1 className="mt-4 text-[17px] font-semibold tracking-tight text-ink">
          {tr ? "Bu çekim bağlantısı artık geçerli değil" : "This capture link no longer works"}
        </h1>
        <p className="mt-2 text-[13.5px] leading-relaxed text-ink-muted">
          {tr
            ? "Bağlantılar 12 saat geçerlidir ve panodan iptal edilebilir. Ev sayfasındaki QR kodunu yeniden okutun ya da hesabınızla giriş yapın."
            : "Links last 12 hours and can be revoked from the dashboard. Scan the QR code on the property page again, or sign in with your account."}
        </p>
        <Link
          href={`/giris?next=${encodeURIComponent(next)}`}
          className={buttonStyles("secondary", "lg", "mt-5 w-full")}
        >
          <KeyRound className="size-4" aria-hidden />
          {tr ? "Giriş yap" : "Sign in"}
        </Link>
      </div>
    </main>
  );
}
