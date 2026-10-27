"use client";

/**
 * "Telefonla çek" — the desktop → phone handoff.
 *
 * The guided capture only makes sense on the device holding the camera, and the person who
 * plans a property is usually at a laptop while the person shooting it is standing in the flat.
 * This card closes that gap: it renders the capture URL as a QR code so the agent can point
 * their phone at the screen and be in the capture flow one tap later, with the property id
 * already attached.
 *
 * WITH A PROPERTY, THE CODE CARRIES A CAPTURE LINK (`k`): a token the engine honours for this
 * property only, on the routes a capture uses, for twelve hours (`api/deps.py`). The phone that
 * scans it films straight away instead of meeting a sign-in form — it is usually not the
 * agent's own phone that is signed in, and never at the moment the client is waiting. The
 * code is not drawn until the link exists: a QR somebody scans in the 100 ms before the token
 * arrived would be the old, sign-in-first link. If minting fails (engine down, signed out),
 * the plain link is drawn — the old behaviour, still correct, one sign-in slower.
 *
 * The QR is drawn BLACK ON WHITE regardless of theme. Scanners are specified for dark
 * modules on a light ground; inverted codes are an optional extra that many phone cameras
 * simply do not attempt, and a code that fails to scan on the one screen it exists for is
 * worse than an ugly white square on a dark page.
 */

import { Check, Copy, QrCode, Smartphone, TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { Button } from "@/components/ui/Button";
import { api } from "@/lib/api";
import { encodeQr, qrToPath } from "@/lib/capture/qr";
import { CAPTURE_GRANT_EVENT, SIGNED_OUT_EVENT, captureLinkCredentials } from "@/lib/session";
import type { Locale } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface PhoneHandoffProps {
  /** Attaches the capture to an existing property. Omit to let the phone create one. */
  propertyId?: string | null;
  locale: Locale;
  /** Rendered under the code — e.g. "yine de bu cihazda devam et". */
  footer?: React.ReactNode;
  /** Open the phone in 3D-walkthrough mode (`?m=w`) instead of the 360° capture. */
  walk?: boolean;
  className?: string;
}

/** Modules of quiet zone. The standard requires four; scanners get unreliable below that. */
const QUIET_ZONE = 4;

/**
 * A new link is minted this long before the current one lapses. The dashboard is often left
 * open on an office screen all day; the code on it must never be one that dies in the hand of
 * whoever scans it on the way out of the door.
 */
const REMINT_BEFORE_MS = 30 * 60_000;

type Grant = { propertyId: string; token: string } | { propertyId: string; token: null };

function subscribeCredentials(onChange: () => void): () => void {
  window.addEventListener(CAPTURE_GRANT_EVENT, onChange);
  window.addEventListener(SIGNED_OUT_EVENT, onChange);
  return () => {
    window.removeEventListener(CAPTURE_GRANT_EVENT, onChange);
    window.removeEventListener(SIGNED_OUT_EVENT, onChange);
  };
}

/**
 * The capture link for `propertyId`, kept fresh. `undefined` while the first one is minted;
 * `token: null` when there is none to give, in which case the plain link is used.
 *
 * ONLY A SESSION MINTS. A tab that is itself running on a capture link — the link opened on a
 * laptop rather than a phone, which lands on this card — must not try: the engine refuses
 * grants on the mint route, the API client reads that 401 as "your credential is dead", and
 * the very link the visitor arrived with would be thrown away. Such a tab shows the link it
 * holds, which is the same twelve-hour token, unchanged.
 */
function useCaptureLink(propertyId: string | null | undefined): Grant | undefined {
  const [grant, setGrant] = useState<Grant | undefined>(undefined);
  const credentials = useSyncExternalStore(
    subscribeCredentials,
    () => captureLinkCredentials(propertyId),
    () => "pending",
  );

  useEffect(() => {
    if (!propertyId || credentials !== "session") return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const mint = () => {
      api.properties
        .captureLink(propertyId)
        .then((link) => {
          if (!alive) return;
          setGrant({ propertyId, token: link.token });
          const lapse = Date.parse(link.expires_at);
          const wait = Number.isFinite(lapse) ? Math.max(60_000, lapse - Date.now() - REMINT_BEFORE_MS) : 60 * 60_000;
          timer = setTimeout(mint, wait);
        })
        .catch(() => {
          if (alive) setGrant({ propertyId, token: null });
        });
    };
    mint();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [propertyId, credentials]);

  if (!propertyId || credentials === "pending") return undefined;
  if (credentials.startsWith("grant:")) return { propertyId, token: credentials.slice("grant:".length) };
  if (credentials === "none") return { propertyId, token: null };
  // A grant for a previous property is not a grant for this one.
  return grant && grant.propertyId === propertyId ? grant : undefined;
}

export function PhoneHandoff({ propertyId, locale, footer, className, walk = false }: PhoneHandoffProps) {
  // The origin comes from an external store, not from a state-in-effect pair: the page is
  // prerendered on the server where `location` does not exist, and a QR that renders empty
  // for one frame is a QR someone has already pointed a phone at.
  const origin = useSyncExternalStore(
    () => () => undefined,
    () => window.location.origin,
    () => "",
  );
  const [copied, setCopied] = useState(false);
  const grant = useCaptureLink(propertyId);
  const linkToken = grant?.token ?? null;
  // With a property, wait for the link (see the header); without one there is nothing to mint.
  const waiting = Boolean(propertyId) && grant === undefined;

  const href = useMemo(() => {
    if (!origin || waiting) return "";
    const url = new URL("/cek", origin);
    if (propertyId) url.searchParams.set("property", propertyId);
    if (walk) url.searchParams.set("m", "w");
    if (linkToken) url.searchParams.set("k", linkToken);
    return url.toString();
  }, [origin, propertyId, walk, waiting, linkToken]);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(timer);
  }, [copied]);

  const qr = useMemo(() => {
    if (!href) return null;
    try {
      const matrix = encodeQr(href);
      return { path: qrToPath(matrix), extent: matrix.size + QUIET_ZONE * 2 };
    } catch {
      // Only reachable for an absurdly long origin. The link below still works, so the card
      // degrades to "type this address" rather than throwing the page away.
      return null;
    }
  }, [href]);

  // A phone cannot resolve the developer's `localhost`, and the resulting "site can't be
  // reached" looks like a broken product rather than a network fact. Say it up front.
  const loopback = useMemo(() => {
    if (!href) return false;
    try {
      const host = new URL(href).hostname;
      return host === "localhost" || host === "127.0.0.1" || host === "::1" || host.endsWith(".local");
    } catch {
      return false;
    }
  }, [href]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(href);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className={cn("glass rounded-lg p-5 sm:p-6", className)}>
      <div className="flex items-start gap-3">
        <span className="grid size-9 shrink-0 place-items-center rounded-full border border-line bg-surface-3 text-bio">
          <Smartphone className="size-[18px]" strokeWidth={1.7} aria-hidden />
        </span>
        <div className="min-w-0">
          <h3 className="text-sm font-semibold tracking-tight text-ink">
            {locale === "tr" ? "Telefonla çek" : "Shoot with your phone"}
          </h3>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-muted">
            {locale === "tr"
              ? "Kodu telefonunuzun kamerasıyla okutun. Rehberli çekim telefonda açılır, çekim bitince tura kendisi eklenir."
              : "Scan the code with your phone's camera. The guided capture opens there and the room is added to this property when you finish."}
          </p>
        </div>
      </div>

      <div className="mt-5 flex flex-col items-center gap-4 sm:flex-row sm:items-start">
        <div className="shrink-0 rounded-md bg-white p-3 shadow-lift">
          {qr ? (
            <svg
              viewBox={`0 0 ${qr.extent} ${qr.extent}`}
              className="block size-[168px]"
              role="img"
              aria-label={locale === "tr" ? "Çekim bağlantısının QR kodu" : "QR code for the capture link"}
              shapeRendering="crispEdges"
            >
              <rect width={qr.extent} height={qr.extent} fill="#ffffff" />
              <g transform={`translate(${QUIET_ZONE} ${QUIET_ZONE})`}>
                <path d={qr.path} fill="#000000" />
              </g>
            </svg>
          ) : (
            <div className="grid size-[168px] place-items-center text-[#8b98a5]">
              <QrCode className="size-8" strokeWidth={1.4} aria-hidden />
            </div>
          )}
        </div>

        <div className="min-w-0 flex-1">
          <p className="text-[12px] font-medium text-ink-faint">
            {locale === "tr" ? "Ya da bu adresi açın" : "Or open this address"}
          </p>
          <p className="mt-1.5 rounded-md border border-line bg-surface-2/60 px-3 py-2 font-mono text-[11.5px] leading-relaxed break-all text-ink-muted select-all">
            {href || "…"}
          </p>
          <Button variant="secondary" size="sm" className="mt-2.5" onClick={() => void copy()} disabled={!href}>
            {copied ? <Check className="size-3.5" aria-hidden /> : <Copy className="size-3.5" aria-hidden />}
            {copied
              ? locale === "tr"
                ? "Kopyalandı"
                : "Copied"
              : locale === "tr"
                ? "Bağlantıyı kopyala"
                : "Copy link"}
          </Button>

          {linkToken ? (
            <p className="mt-3 text-[12px] leading-relaxed text-ink-faint">
              {locale === "tr"
                ? "Bu bağlantıyla giriş yapmadan çekim yapılır. 12 saat geçerlidir ve yalnızca bu eve video yükleyebilir; ev bilgilerini değiştiremez."
                : "This link films without signing in. It lasts 12 hours and can only upload footage to this property; it cannot change anything else."}
            </p>
          ) : null}

          {loopback ? (
            <p className="mt-3 flex gap-2 text-[12px] leading-relaxed text-amber">
              <TriangleAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden />
              <span>
                {locale === "tr"
                  ? "Bu adres yalnız bu bilgisayarda geçerli. Telefondan açmak için sunucuyu ağ adresiyle yayınlayın ve telefonu aynı Wi-Fi ağına bağlayın."
                  : "This address only resolves on this computer. Serve the app on your network address and put the phone on the same Wi-Fi to open it there."}
              </span>
            </p>
          ) : null}
        </div>
      </div>

      {footer ? <div className="mt-5 border-t border-line pt-4">{footer}</div> : null}
    </div>
  );
}
