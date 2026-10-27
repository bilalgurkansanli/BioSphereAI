/**
 * One HTTPS origin in front of both halves of the stack, for testing the guided capture on a
 * phone over the local network.
 *
 * WHY THIS EXISTS RATHER THAN A TUNNEL.
 *
 * The capture cannot run over plain http://192.168.x.x: getUserMedia, DeviceOrientation,
 * DeviceMotion and MediaRecorder all require a secure context, and a LAN IP over http is not
 * one. That is why the project reached for Cloudflare quick tunnels. They cost more than they
 * look:
 *
 *   - a fresh hostname on every run, so every link goes stale and the bundle has to be
 *     rebuilt because `NEXT_PUBLIC_*` is inlined at build time;
 *   - a body-size ceiling at the edge — measured, 300 MB passes and 520 MB comes back 413
 *     with `Server: cloudflare` — which is well inside what a long capture produces;
 *   - every byte of a ~150 MB upload leaves the building and comes back.
 *
 * A self-signed certificate over the LAN has none of those. The address is stable, the
 * upload is a local network transfer, and there is no third party in the path. The cost is
 * one certificate warning, accepted once per phone. Chrome still reports `isSecureContext`
 * as true after the bypass, which is the only thing the capture APIs care about.
 *
 * ONE ORIGIN, NOT TWO. The engine is reached through `/api` and `/media` on this same host
 * rather than through a second certificate, so the phone trusts one exception instead of two
 * and there is no mixed-content question at all.
 *
 * Everything is piped, never buffered: an upload is streamed straight through, and the job
 * progress stream stays a stream.
 */

import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.LAN_PROXY_PORT ?? 3443);
const WEB = { host: "127.0.0.1", port: Number(process.env.WEB_PORT ?? 3000) };
const ENGINE = { host: "127.0.0.1", port: Number(process.env.ENGINE_PORT ?? 8008) };

/** Paths the engine owns. Everything else is the Next server's. */
const ENGINE_PREFIXES = ["/api/", "/media/", "/docs", "/openapi.json"];

const options = {
  key: fs.readFileSync(path.join(here, "certs", "key.pem")),
  cert: fs.readFileSync(path.join(here, "certs", "cert.pem")),
};

/**
 * What `?diag=1` injects into a page: the phone's own errors and clicks, sent back here.
 *
 * Three questions, and between them they separate every remaining cause of "the button does
 * nothing" on a device with no console:
 *   boot   — did the page's scripts run at all
 *   click  — did a click event reach the button, and was it defaultPrevented
 *   error  — did anything throw, including during hydration, where React swallows the
 *            failure into a client re-render and the page still looks fine
 *
 * `sendBeacon` rather than `fetch`: it survives the page being navigated away or reloaded in
 * the same gesture, which is exactly when the interesting failures happen.
 */
const INJECT = `<script>(function(){
var send=function(kind,detail){try{navigator.sendBeacon('/__diag_log',kind+' | '+detail)}catch(e){}};
window.addEventListener('error',function(e){send('ERROR',(e.message||'')+' @ '+(e.filename||'')+':'+(e.lineno||''))});
window.addEventListener('unhandledrejection',function(e){var r=e.reason;send('REJECT',String(r&&(r.stack||r.message)||r))});
document.addEventListener('click',function(e){var t=e.target&&e.target.closest?e.target.closest('button'):null;
send('CLICK',(t?('button "'+(t.textContent||'').trim().slice(0,30)+'" disabled='+t.disabled):'not-a-button')+' prevented='+e.defaultPrevented)},true);
window.addEventListener('load',function(){send('LOAD','react='+(!!document.querySelector('button')&&Object.keys(document.querySelector('button')).some(function(k){return k.indexOf('__react')===0})))});
send('BOOT',navigator.userAgent);
})()</script>`;

const server = https.createServer(options, (req, res) => {
  const url = req.url ?? "/";

  // The certificate, downloadable. iOS Safari will not hand a page the camera just because
  // the user tapped through a certificate warning: the origin has to be genuinely trusted,
  // which on iOS means installing this file as a profile and switching it on under
  // Settings > General > About > Certificate Trust Settings. Chrome on Android is happy with
  // the warning alone, which is why this was not needed until now.
  if (url === "/cert" || url === "/cert.crt") {
    // NO `content-disposition`. With `attachment` Safari saves the file to Files and the
    // profile installer never runs, so nothing appears under Settings > General > VPN &
    // Device Management and the certificate can never be trusted. Served inline with the
    // certificate MIME type, iOS recognises it and offers to install it instead.
    res.writeHead(200, { "content-type": "application/x-x509-ca-cert" });
    res.end(options.cert);
    return;
  }
  // A self-contained check of the things that can only be wrong ON THE PHONE.
  //
  // The capture is opened on a device with no console, no devtools and no way to report what
  // it saw. When a button does nothing there, the desktop can prove the server is healthy and
  // still be no closer: the failure is a secure-context rule, a missing API on an older
  // Chrome, or a certificate the browser accepted for a navigation and not for a fetch. This
  // page runs those checks in the real origin and prints the answers, which is the difference
  // between one look and an afternoon of guessing.
  //
  // Deliberately dependency-free and served by the proxy rather than the app: if the app is
  // the thing that is broken, a diagnostic inside it tells you nothing.
  if (url === "/diag") {
    res.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
    res.end(`<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{font:14px/1.6 system-ui;margin:16px;background:#0b0f0e;color:#dfe7e5}
b{color:#5eead4}pre{white-space:pre-wrap;word-break:break-all;background:#131a19;padding:10px;border-radius:8px}</style>
<h3>BioSphere LAN diagnostic</h3><pre id=o>running…</pre><script>
const out = [];
const say = (k, v) => { out.push(k.padEnd(22) + ' ' + v); document.getElementById('o').textContent = out.join('\\n'); };
say('origin', location.origin);
say('isSecureContext', String(window.isSecureContext));
say('mediaDevices', String(!!navigator.mediaDevices));
say('getUserMedia', String(!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)));
say('MediaRecorder', String(typeof MediaRecorder !== 'undefined'));
say('DeviceOrientation', String(typeof DeviceOrientationEvent !== 'undefined'));
say('AbortSignal.timeout', String(typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function'));
say('AbortSignal.any', String(typeof AbortSignal !== 'undefined' && typeof AbortSignal.any === 'function'));
say('ua', navigator.userAgent);
(async () => {
  try {
    const r = await fetch('/api/v1/health', { cache: 'no-store' });
    say('GET /api/v1/health', r.status + ' ' + (await r.text()).slice(0, 90));
  } catch (e) { say('GET /api/v1/health', 'THREW ' + e); }
  try {
    const r = await fetch('/api/v1/properties', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify({ title: 'diag ' + Date.now() }),
    });
    const body = await r.text();
    say('POST /api/v1/properties', r.status + ' ' + body.slice(0, 120));
    const id = (body.match(/"id":"([^"]+)"/) || [])[1];
    if (id) { await fetch('/api/v1/properties/' + id, { method: 'DELETE' }); say('cleanup', 'deleted ' + id); }
  } catch (e) { say('POST /api/v1/properties', 'THREW ' + e); }
})();
</script>`);
    return;
  }

  // Where the phone reports what only it can see. See `INJECT` below.
  if (url === "/__diag_log" && req.method === "POST") {
    let body = "";
    req.on("data", (c) => {
      body += c;
    });
    req.on("end", () => {
      console.log(`PHONE ${body.slice(0, 800)}`);
      res.writeHead(204);
      res.end();
    });
    return;
  }

  const toEngine = ENGINE_PREFIXES.some((p) => url === p.replace(/\/$/, "") || url.startsWith(p));
  const target = toEngine ? ENGINE : WEB;

  // One line in, one line out. Without this a request that never arrives and a request that
  // arrived and died look identical from here — which is exactly the hole an upload stuck at
  // 0% fell into: the engine logged nothing, and nothing else was watching the wire.
  const wantsDiag = url.includes("diag=1");
  const started = Date.now();
  const bodyBytes = req.headers["content-length"] ?? "?";
  const expect = req.headers["expect"] ?? "-";
  // Uploads and API calls always; everything else only when asked. The default is quiet
  // because a page load is a hundred chunk requests and the one line that matters would be
  // lost in them -- but "did the phone reach us at all" is unanswerable from a quiet log,
  // and that is exactly the question when a button appears to do nothing. LAN_PROXY_VERBOSE=1
  // turns the page loads back on.
  const interesting =
    req.method !== "GET" || url.startsWith("/api/") || process.env.LAN_PROXY_VERBOSE === "1";
  if (interesting) {
    console.log(`-> ${req.method} ${url} len=${bodyBytes} expect=${expect}`);
  }
  let received = 0;
  req.on("data", (c) => {
    received += c.length;
  });
  req.on("aborted", () => {
    console.log(`!! ABORTED ${req.method} ${url} after ${received} bytes, ${Date.now() - started}ms`);
  });

  const proxied = http.request(
    {
      host: target.host,
      port: target.port,
      method: req.method,
      path: url,
      headers: {
        ...req.headers,
        host: `${target.host}:${target.port}`,
        // The injection below rewrites the HTML, which means reading it. Asking upstream for
        // identity rather than decompressing here keeps that to one concern -- and getting it
        // wrong is not subtle: buffering a gzip body, dropping the content-encoding header
        // and sending the bytes on renders a page of mojibake.
        ...(wantsDiag ? { "accept-encoding": "identity" } : {}),
      },
    },
    (upstream) => {
      if (interesting) {
        console.log(
          `<- ${upstream.statusCode} ${req.method} ${url} sent=${received} ${Date.now() - started}ms`,
        );
      }

      // `?diag=1` on any page turns that page into its own reporter. The phone has no
      // console and no devtools, so a script that throws during hydration is invisible from
      // here: the server sees a healthy page load, the user sees a button that does nothing,
      // and the two observations never meet. Injected rather than added to the app for the
      // same reason `/diag` is served here -- if the app is what is broken, a probe inside it
      // is not evidence.
      //
      // Off unless asked, and the response is buffered only in that case: HTML pages are
      // small, but streaming is the whole point of this proxy and must not regress for
      // ordinary traffic.
      const isHtml = String(upstream.headers["content-type"] ?? "").includes("text/html");
      if (wantsDiag && isHtml) {
        const chunks = [];
        upstream.on("data", (c) => chunks.push(c));
        upstream.on("end", () => {
          const html = Buffer.concat(chunks).toString("utf8").replace("<head>", `<head>${INJECT}`);
          const headers = { ...upstream.headers };
          delete headers["content-length"];
          delete headers["content-encoding"];
          res.writeHead(upstream.statusCode ?? 502, headers);
          res.end(html);
        });
        return;
      }

      res.writeHead(upstream.statusCode ?? 502, upstream.headers);
      upstream.pipe(res);
    },
  );

  proxied.on("error", (err) => {
    console.log(`!! UPSTREAM ${req.method} ${url} after ${received} bytes: ${err.message}`);
    // A dead upstream must not take the proxy down with it — the other half may be fine, and
    // a 502 tells the phone something useful where a socket reset does not.
    if (!res.headersSent) res.writeHead(502, { "content-type": "text/plain; charset=utf-8" });
    res.end(`upstream ${target.host}:${target.port} unreachable: ${err.message}`);
  });

  req.pipe(proxied);
});

/**
 * WebSocket upgrades, without which `next dev` cannot hydrate through this proxy.
 *
 * THIS WAS MISSING AND THE SYMPTOM WAS NOT A BROKEN SOCKET. Without an `upgrade` handler
 * Node hands the request to the ordinary listener above, which forwards it as a plain HTTP
 * GET and then holds the connection open waiting for a response that a WebSocket endpoint is
 * never going to send. Next's HMR client retries, and each retry parks another socket.
 * Chrome allows six per host over HTTP/1.1, so after a handful the pool is gone and the
 * chunk requests that hydration is still waiting for queue behind sockets that will never
 * free — the page renders from the server HTML, CSS works, the button takes a press, and
 * `onClick` does not exist because React never attached. No error is logged anywhere,
 * because nothing failed; a script simply never arrived.
 *
 * MEASURED FROM THE PHONE, which is the only place it shows: `react=false` at `load`, seven
 * clean click events with `prevented=false`, and not one request leaving the browser.
 *
 * Loopback fails fast rather than hanging, so a desktop developer on localhost:3000 never
 * sees any of this. That is what made it a LAN-only bug and this file the right place for it.
 */
server.on("upgrade", (req, clientSocket, head) => {
  const url = req.url ?? "/";
  const toEngine = ENGINE_PREFIXES.some((p) => url === p.replace(/\/$/, "") || url.startsWith(p));
  const target = toEngine ? ENGINE : WEB;

  const proxied = http.request({
    host: target.host,
    port: target.port,
    method: req.method,
    path: url,
    headers: { ...req.headers, host: `${target.host}:${target.port}` },
  });

  proxied.on("upgrade", (upstreamRes, upstreamSocket, upstreamHead) => {
    const statusLine = `HTTP/1.1 ${upstreamRes.statusCode} ${upstreamRes.statusMessage}\r\n`;
    const headers = Object.entries(upstreamRes.headers)
      .flatMap(([k, v]) => (Array.isArray(v) ? v.map((one) => `${k}: ${one}`) : [`${k}: ${v}`]))
      .join("\r\n");
    clientSocket.write(`${statusLine}${headers}\r\n\r\n`);
    if (upstreamHead?.length) clientSocket.write(upstreamHead);
    upstreamSocket.pipe(clientSocket);
    clientSocket.pipe(upstreamSocket);
    const drop = () => {
      upstreamSocket.destroy();
      clientSocket.destroy();
    };
    upstreamSocket.on("error", drop);
    clientSocket.on("error", drop);
  });

  // A dead upstream must close the client's socket rather than leave it hanging, which is
  // the failure this whole handler exists to stop happening one level up.
  proxied.on("error", () => clientSocket.destroy());
  clientSocket.on("error", () => proxied.destroy());
  if (head?.length) proxied.write(head);
  proxied.end();
});

// No timeout on the request body. A capture is hundreds of megabytes over Wi-Fi from a phone
// that may be in another room; Node's default would abort it half way.
server.requestTimeout = 0;
server.headersTimeout = 0;
server.setTimeout(0);

server.listen(PORT, "0.0.0.0", () => {
  console.log(`https proxy on :${PORT}  ->  web ${WEB.port}, engine ${ENGINE.port} (/api, /media)`);
});
