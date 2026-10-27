"use client";

/**
 * Capability probe — deliberately standalone.
 *
 * The guided capture refuses to start on a real phone and we cannot see that phone's
 * console. This page imports nothing from `lib/capture`, so it cannot fail for the same
 * reason the capture does; it just asks the browser what it can actually do and prints
 * the answer big enough to read on a phone or screenshot.
 *
 * Delete once the capture is trusted on real hardware.
 */

import { useCallback, useState } from "react";

const MIME_LADDER = [
  "video/mp4;codecs=avc1.640028",
  "video/mp4;codecs=avc1.42E01E,mp4a.40.2",
  "video/mp4",
  "video/mp4;codecs=h264",
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm;codecs=h264",
  "video/webm",
  "video/x-matroska;codecs=avc1",
];

type Row = { k: string; v: string; ok?: boolean };

export default function ProbePage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [busy, setBusy] = useState(false);

  const run = useCallback(async () => {
    setBusy(true);
    const out: Row[] = [];
    const push = (k: string, v: unknown, ok?: boolean) => out.push({ k, v: String(v), ok });

    push("userAgent", navigator.userAgent);
    push("secureContext", window.isSecureContext, window.isSecureContext);
    push("origin", location.origin);
    push("MediaRecorder", typeof MediaRecorder, typeof MediaRecorder !== "undefined");
    push("mediaDevices", typeof navigator.mediaDevices, typeof navigator.mediaDevices !== "undefined");
    push(
      "DeviceOrientationEvent",
      typeof DeviceOrientationEvent === "undefined" ? "absent" : "present",
      typeof DeviceOrientationEvent !== "undefined",
    );
    push(
      "DeviceMotionEvent",
      typeof DeviceMotionEvent === "undefined" ? "absent" : "present",
      typeof DeviceMotionEvent !== "undefined",
    );
    push(
      "motion needs permission",
      typeof DeviceMotionEvent !== "undefined" &&
        typeof (DeviceMotionEvent as unknown as { requestPermission?: unknown }).requestPermission === "function"
        ? "yes (iOS)"
        : "no",
    );

    if (typeof MediaRecorder !== "undefined" && typeof MediaRecorder.isTypeSupported === "function") {
      for (const m of MIME_LADDER) {
        let ok = false;
        try {
          ok = MediaRecorder.isTypeSupported(m);
        } catch {
          ok = false;
        }
        push(`  ${m}`, ok ? "SUPPORTED" : "no", ok);
      }
    } else {
      push("isTypeSupported", "MISSING — cannot probe codecs", false);
    }

    // Camera + a real 400 ms recording, which is the check the capture page actually fails.
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1080 }, height: { ideal: 1920 } },
        audio: false,
      });
      const track = stream.getVideoTracks()[0];
      const s = track?.getSettings?.() ?? {};
      push("camera", "OPENED", true);
      push("  resolution", `${s.width ?? "?"} x ${s.height ?? "?"}`);
      push("  frameRate", s.frameRate ?? "?");
      push("  facingMode", s.facingMode ?? "?");
      push("  label", track?.label ?? "?");
      push("  short edge >= 1080", Math.min(s.width ?? 0, s.height ?? 0) >= 1080, Math.min(s.width ?? 0, s.height ?? 0) >= 1080);
    } catch (e) {
      push("camera", `FAILED: ${e instanceof Error ? `${e.name}: ${e.message}` : String(e)}`, false);
    }

    if (stream && typeof MediaRecorder !== "undefined") {
      // Isolate the two variables. Every explicit mimeType came back EMPTY on the first
      // device we measured while "no options at all" worked — but that rung changes the
      // bitrate too, so on its own it cannot say which one the encoder objected to.
      const supported = MIME_LADDER.filter((m) => {
        try {
          return MediaRecorder.isTypeSupported(m);
        } catch {
          return false;
        }
      });
      const best = supported[0];
      const matrix: Array<{ label: string; mime?: string; bps?: number }> = [
        { label: "mime + 20Mbps", mime: best, bps: 20_000_000 },
        { label: "mime + 8Mbps", mime: best, bps: 8_000_000 },
        { label: "mime + 4Mbps", mime: best, bps: 4_000_000 },
        { label: "mime, no bitrate", mime: best },
        { label: "no mime + 20Mbps", bps: 20_000_000 },
        { label: "no mime + 8Mbps", bps: 8_000_000 },
        { label: "no options", },
      ];
      push("— isolating mimeType vs bitrate —", `best mime: ${best ?? "none"}`);
      let recorded = false;
      for (const combo of matrix) {
        const c = combo.label;
        try {
          const opts: MediaRecorderOptions = {};
          if (combo.mime) opts.mimeType = combo.mime;
          if (combo.bps) opts.videoBitsPerSecond = combo.bps;
          const rec = new MediaRecorder(stream, Object.keys(opts).length ? opts : undefined);
          const bytes = await new Promise<number>((resolve) => {
            let n = 0;
            const timer = setTimeout(() => {
              try {
                rec.stop();
              } catch {
                resolve(-1);
              }
            }, 400);
            rec.ondataavailable = (ev) => {
              n += ev.data?.size ?? 0;
            };
            rec.onstop = () => {
              clearTimeout(timer);
              resolve(n);
            };
            rec.onerror = () => {
              clearTimeout(timer);
              resolve(-1);
            };
            try {
              rec.start(100);
            } catch (err) {
              clearTimeout(timer);
              push(`  start(${c ?? "no mime"})`, `threw ${err instanceof Error ? err.name : "?"}`, false);
              resolve(-1);
            }
          });
          push(`  ${c}`, bytes > 0 ? `OK ${bytes} bytes` : `EMPTY (${bytes})`, bytes > 0);
          if (bytes > 0) recorded = true;
        } catch (e) {
          push(`  ${c}`, `threw ${e instanceof Error ? e.name : String(e)}`, false);
        }
      }
      push("RECORDING WORKS", recorded ? "YES" : "NO", recorded);
    }

    stream?.getTracks().forEach((t) => t.stop());
    setRows(out);
    setBusy(false);
  }, []);

  return (
    <main className="min-h-[100dvh] bg-canvas px-4 py-6 text-ink">
      <h1 className="text-lg font-semibold">Cihaz testi</h1>
      <p className="mt-1 text-[13px] text-ink-muted">
        Kamera ve kayıt yeteneklerini ölçer. Sonucu ekran görüntüsü olarak paylaşın.
      </p>
      <button
        onClick={() => void run()}
        disabled={busy}
        className="mt-4 h-12 w-full rounded-md bg-[linear-gradient(120deg,var(--bio-soft),var(--bio))] font-semibold text-canvas disabled:opacity-50"
      >
        {busy ? "Ölçülüyor…" : "Testi başlat"}
      </button>

      {rows.length > 0 ? (
        <div className="mt-5 space-y-1 font-mono text-[11.5px] leading-snug">
          {rows.map((r, i) => (
            <div
              key={i}
              className={
                r.ok === true ? "text-success" : r.ok === false ? "text-danger" : "text-ink-muted"
              }
            >
              <span className="text-ink-faint">{r.k}</span>
              {": "}
              <span className="break-all">{r.v}</span>
            </div>
          ))}
        </div>
      ) : null}
    </main>
  );
}
