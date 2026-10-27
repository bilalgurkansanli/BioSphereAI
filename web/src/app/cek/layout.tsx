import type { ReactNode } from "react";
import { CaptureGate } from "@/components/auth/CaptureGate";
import { DashboardGateFallback } from "@/components/dashboard/DashboardGateFallback";

/**
 * The guided capture uploads footage to a property, and it opens on a DIFFERENT device than
 * the one that made the QR code — that is the whole point of the page.
 *
 * So it has two doors. The QR code the property page draws carries a capture link (`?k=`): a
 * short-lived token the engine honours for that one property and only on the routes a capture
 * uses, so the phone needs no account at all. Without one — a link typed by hand, an old
 * bookmark — the page is behind the same sign-in as the dashboard. `CaptureGate` decides;
 * the engine enforces (`api/deps.py`).
 */
export default function CaptureLayout({ children }: { children: ReactNode }) {
  return <CaptureGate fallback={<DashboardGateFallback />}>{children}</CaptureGate>;
}
