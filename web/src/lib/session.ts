/**
 * Where the session token lives on the client.
 *
 * IT IS IN `localStorage`, AND THAT IS A TRADE-OFF RATHER THAN AN OVERSIGHT. The engine also
 * sets the same token as an httpOnly cookie on every sign-in, which is the stronger option —
 * script on the page cannot read it — but a cookie is only stored when the response is
 * same-origin (or credentialed), and in this project's normal shape the dashboard is served
 * from :3000 while the engine answers on :8008. So the header is what actually carries the
 * session today, and the cookie is there so that a deployment which puts the engine behind
 * the site's own domain gets the httpOnly version with no client change at all.
 *
 * What makes the trade-off acceptable here, and what would end it: this app loads no
 * third-party script — no analytics, no tag manager, no ad pixel — so the XSS surface is
 * the code in this repository. The day a third-party script is added to the dashboard, the
 * same-origin deployment stops being an optimisation and becomes the requirement.
 */

const KEY = "biosphere.session";

/** Fired when the engine rejects the token we hold, so the shell can send the user to sign in. */
export const SIGNED_OUT_EVENT = "biosphere:signed-out";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(KEY);
  } catch {
    // Private mode, or a browser configured to block site data. The user is simply not
    // signed in, which is a state the app already renders.
    return null;
  }
}

export function setToken(token: string): void {
  try {
    window.localStorage.setItem(KEY, token);
  } catch {
    /* Nothing to do: the session lasts as long as the tab. */
  }
}

export function clearToken(): void {
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    /* Already gone, as far as anyone can tell. */
  }
}

/**
 * A capture link's token: what a phone that scanned the dashboard's QR code holds instead of
 * a session.
 *
 * IN `sessionStorage`, NOT `localStorage`, and that is the point. The link acts as the
 * property's owner (on the few routes a capture needs, for one property — the engine
 * enforces that), so it should die with the tab that was opened for the job rather than sit
 * in a phone's browser for months. A reload keeps it; a new tab asks for the QR again.
 */
export interface CaptureGrant {
  token: string;
  propertyId: string;
}

const GRANT_KEY = "biosphere.captureGrant";
/** Remembers, for this tab, that a grant WAS held and the engine refused it. */
const GRANT_EXPIRED_KEY = "biosphere.captureGrantExpired";

/** Fired when a capture grant is adopted or dropped, for the capture page's gate. */
export const CAPTURE_GRANT_EVENT = "biosphere:capture-grant";

export function getCaptureGrant(): CaptureGrant | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(GRANT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<CaptureGrant>;
    return typeof parsed.token === "string" && typeof parsed.propertyId === "string"
      ? { token: parsed.token, propertyId: parsed.propertyId }
      : null;
  } catch {
    return null;
  }
}

export function setCaptureGrant(grant: CaptureGrant): void {
  try {
    window.sessionStorage.setItem(GRANT_KEY, JSON.stringify(grant));
    window.sessionStorage.removeItem(GRANT_EXPIRED_KEY);
  } catch {
    /* Private mode with storage blocked: the grant lives as long as this render. */
  }
  window.dispatchEvent(new CustomEvent(CAPTURE_GRANT_EVENT));
}

export function captureGrantExpired(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.sessionStorage.getItem(GRANT_EXPIRED_KEY) === "1";
  } catch {
    return false;
  }
}

function dropCaptureGrant(): boolean {
  try {
    if (window.sessionStorage.getItem(GRANT_KEY) === null) return false;
    window.sessionStorage.removeItem(GRANT_KEY);
    window.sessionStorage.setItem(GRANT_EXPIRED_KEY, "1");
    return true;
  } catch {
    return false;
  }
}

/** Which door `/cek` should open — see `components/auth/CaptureGate.tsx`. */
export type CaptureGateMode = "pending" | "session" | "grant" | "expired";

/**
 * The gate's decision, read straight from the browser.
 *
 * While `k` is still in the address the answer is `pending`, never `session`: the first render
 * would otherwise mount the sign-in guard and navigate to `/giris` a moment before the grant
 * was adopted. A real session wins over a grant.
 */
export function captureGateMode(): CaptureGateMode {
  if (new URLSearchParams(window.location.search).has("k")) return "pending";
  if (getToken()) return "session";
  if (getCaptureGrant()) return "grant";
  if (captureGrantExpired()) return "expired";
  return "session";
}

/**
 * Move `?k=` from the address bar into this tab's storage. Idempotent.
 *
 * The token leaves the address because an address gets screenshotted, bookmarked and pasted
 * into chats; `sessionStorage` keeps it across a reload of this tab and nowhere else. A
 * mangled value (cut off by a messenger's link preview, say) is dropped rather than adopted.
 */
export function adoptCaptureGrantFromAddress(): void {
  const url = new URL(window.location.href);
  const token = url.searchParams.get("k");
  if (token === null) return;
  const propertyId = url.searchParams.get("property");
  url.searchParams.delete("k");
  window.history.replaceState(window.history.state, "", url.toString());
  if (propertyId && token.length >= 20) {
    setCaptureGrant({ token, propertyId });
  } else {
    window.dispatchEvent(new CustomEvent(CAPTURE_GRANT_EVENT));
  }
}

/**
 * What this tab can do about a capture link for `propertyId`: `session` can mint one;
 * `grant:<token>` already holds one and can only pass it on; `none` can do neither. A string,
 * so `useSyncExternalStore` compares it by value.
 */
export function captureLinkCredentials(propertyId: string | null | undefined): string {
  if (getToken()) return "session";
  const held = getCaptureGrant();
  return held && held.propertyId === propertyId ? `grant:${held.token}` : "none";
}

export function authHeaders(): Record<string, string> {
  // A real session wins: an agent signed in on their phone keeps their full account even if
  // they also scanned a QR code.
  const token = getToken() ?? getCaptureGrant()?.token ?? null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * Announce that the token we were holding is no longer accepted.
 *
 * Deliberately an event rather than a direct `router.push`: this module is imported by the
 * API client, which runs in server components too, and reaching for the router from there
 * is how a build breaks. The shell listens and decides where to go.
 *
 * A refused capture grant is dropped here too (and remembered as expired for this tab), so the
 * capture page can say "scan a new code" instead of sending someone with no account to a
 * sign-in form.
 */
export function announceSignedOut(): void {
  clearToken();
  if (typeof window !== "undefined") {
    const hadGrant = dropCaptureGrant();
    window.dispatchEvent(new CustomEvent(SIGNED_OUT_EVENT));
    if (hadGrant) window.dispatchEvent(new CustomEvent(CAPTURE_GRANT_EVENT));
  }
}
