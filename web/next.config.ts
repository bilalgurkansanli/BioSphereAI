import type { NextConfig } from "next";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8008";
const apiOrigin = (() => {
  try {
    return new URL(API_BASE).origin;
  } catch {
    return "http://127.0.0.1:8000";
  }
})();

const nextConfig: NextConfig = {
  reactStrictMode: true,

  // The guided capture is opened on a phone, which means the dev server is reached over
  // the LAN rather than on localhost. Next 16 refuses to serve /_next/* dev resources to
  // an origin that is not listed here, and it refuses them with a 403 — so the page's
  // HTML arrives, no JavaScript chunk does, React never hydrates, and every button on the
  // page silently does nothing. That failure looks exactly like a broken component, which
  // is what makes it worth spelling out: it cost an afternoon of chasing the wrong bug.
  //
  // The private address ranges (RFC 1918), only in development; this has no effect on a
  // production build. Next matches `*` against exactly one dot-separated label, so
  // "10.*.*.*" covers 10.0.0.0/8 and nothing outside it. ("172.*.*.*" is wider than
  // 172.16/12 — harmless for a dev server that answers on a LAN.)
  allowedDevOrigins: [
    "192.168.*.*",
    "10.*.*.*",
    "172.*.*.*",
    // Quick Cloudflare tunnels, used to get a real HTTPS origin on a phone. Camera and
    // motion-sensor APIs need a secure context, and a phone on the LAN is not one.
    "*.trycloudflare.com",
  ],

  // Panoramas are served by the engine, not by Next's image optimiser — an 8K
  // equirectangular texture must reach WebGL byte-for-byte, so it is never resized.
  images: {
    remotePatterns: [{ protocol: "http", hostname: "127.0.0.1" }, { protocol: "http", hostname: "localhost" }],
  },

  async headers() {
    return [
      {
        // The embed route exists to be framed by third-party listing portals. Next sends
        // no X-Frame-Options by default; this states the permissive policy explicitly so
        // a reverse proxy adding a restrictive default cannot silently break embeds.
        source: "/embed/:path*",
        headers: [
          { key: "Content-Security-Policy", value: "frame-ancestors *;" },
          { key: "X-Frame-Options", value: "ALLOWALL" },
          { key: "Referrer-Policy", value: "no-referrer-when-downgrade" },
          // VR inside a portal's iframe: WebXR needs `xr-spatial-tracking`, the phone headset
          // needs the motion sensors and fullscreen, and a VR box needs the screen kept awake.
          // A frame only ever gets what BOTH sides grant — the portal's `allow` attribute
          // (which the dashboard's embed snippet carries) and this document's own policy — so
          // this cannot open anything up on its own; it states that this page wants them, so a
          // proxy adding a restrictive default policy cannot quietly take VR away. `self` is
          // enough: the player embeds nothing that needs them in turn.
          {
            key: "Permissions-Policy",
            value:
              "xr-spatial-tracking=(self), accelerometer=(self), gyroscope=(self), magnetometer=(self), fullscreen=(self), screen-wake-lock=(self)",
          },
        ],
      },
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-DNS-Prefetch-Control", value: "on" },
        ],
      },
    ];
  },

  async redirects() {
    // The dashboard's listing unit is a property ("ev"), and its URLs say so. Links to the
    // old English path are already in browser histories, in bookmarks and in QR codes that
    // were printed before the rename, so the old path keeps working — permanently, because
    // it is never coming back.
    const legacy = [{ source: "/tours/:path*", destination: "/evler/:path*", permanent: true }];
    if (process.env.NEXT_PUBLIC_SITE_ONLY !== "1") return legacy;
    // The public project page has no engine behind it: every route that needs one would
    // only show "engine offline". They go to the guide, which explains where the dashboard
    // actually runs — on the visitor's own computer. Temporary, so a later hosted demo can
    // take the paths back.
    const appRoutes = ["/evler", "/new", "/dekor", "/ayarlar", "/cek", "/giris", "/tani", "/embed", "/tours"];
    return appRoutes.flatMap((path) => [
      { source: path, destination: "/rehber#panel", permanent: false },
      { source: `${path}/:rest*`, destination: "/rehber#panel", permanent: false },
    ]);
  },

  async rewrites() {
    // Lets the browser talk to the engine same-origin during development, which removes
    // CORS from the equation entirely while iterating on the dashboard.
    return [{ source: "/engine/:path*", destination: `${apiOrigin}/:path*` }];
  },
};

export default nextConfig;
