import type { ReactNode } from "react";
import { RequireSession, SessionProvider } from "@/components/auth/SessionProvider";
import { DashboardChrome } from "@/components/dashboard/DashboardChrome";
import { DashboardGateFallback } from "@/components/dashboard/DashboardGateFallback";

/**
 * The shell itself is interactive (locale, health polling, toasts), so it lives in a
 * client component and this layout stays a server component that only composes it.
 *
 * The gate is here rather than in middleware because the credential is a token in
 * `localStorage`, which no server-side check can see. Every page under this layout is
 * therefore client-rendered behind `RequireSession` — and every one of them was already
 * fetching its data from the browser, so nothing was given up by putting the gate here.
 */
export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <SessionProvider>
      <RequireSession fallback={<DashboardGateFallback />}>
        <DashboardChrome>{children}</DashboardChrome>
      </RequireSession>
    </SessionProvider>
  );
}
