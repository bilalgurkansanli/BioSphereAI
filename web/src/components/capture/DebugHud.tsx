"use client";

/**
 * On-device diagnostics for the capture, shown only with `?debug=1`.
 *
 * The capture runs on a phone we cannot attach a profiler to, and "it lags and pops up
 * errors" is not something that can be fixed by guessing. The loop publishes a tiny object
 * to `window.__biosphereDebug` on every processed tick — a plain property write, no React,
 * no allocation beyond the numbers themselves — and this reads it a few times a second.
 *
 * It records WARNING HISTORY rather than the current warning, because the interesting ones
 * are exactly those that fire, block the capture and clear again before anyone can read
 * them.
 */

import { useEffect, useRef, useState } from "react";

import { MAX_EXPOSURE_SECONDS, THRESHOLDS } from "@/lib/capture/types";

export interface DebugSnapshot {
  fps: number;
  tier: number;
  floorTier: number;
  medianMs: number;
  jankRatio: number;
  governorReason: string;
  analysis: "worker" | "main-thread";
  demoted: string | null;
  codec: string;
  camW: number;
  camH: number;
  exposureMs: number;
  /** Sensor gain, and the brightness it actually produced. Together these say whether a
      capped exposure was paid for in gain or simply taken out of the picture. */
  iso: number;
  lumaMean: number;
  tickMs: number;
  worstTickMs: number;
  orientationAgeMs: number;
  motionAgeMs: number;
  omegaDegPerS: number;
  recorderState: string;
  /** Chunks delivered so far — zero here is the silent-encoder failure. */
  recordedBytes: number;
  warning: string | null;
  severity: string | null;
  blocked: boolean;
  state: string;
}

declare global {
  interface Window {
    __biosphereDebug?: DebugSnapshot;
  }
}

interface Entry {
  at: number;
  code: string;
  severity: string;
}

export function DebugHud() {
  const [snap, setSnap] = useState<DebugSnapshot | null>(null);
  const [log, setLog] = useState<Entry[]>([]);
  const lastCode = useRef<string | null>(null);
  const started = useRef(Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => {
      const s = window.__biosphereDebug;
      if (!s) return;
      setSnap({ ...s });
      const code = s.warning ?? null;
      if (code !== lastCode.current) {
        lastCode.current = code;
        if (code) {
          setLog((prev) =>
            [{ at: Math.round((Date.now() - started.current) / 100) / 10, code, severity: s.severity ?? "?" }, ...prev].slice(0, 12),
          );
        }
      }
    }, 250);
    return () => window.clearInterval(timer);
  }, []);

  if (!snap) {
    return (
      <div className="pointer-events-none absolute top-2 left-2 z-40 rounded bg-black/70 px-2 py-1 font-mono text-[10px] text-white">
        debug: waiting for the loop
      </div>
    );
  }

  const bad = (v: boolean) => (v ? "text-red-400" : "text-emerald-400");

  return (
    <div className="pointer-events-none absolute top-2 left-2 z-40 max-w-[62vw] rounded bg-black/75 px-2 py-1.5 font-mono text-[10px] leading-snug text-white">
      {/* Real animation frames per second now, not the loop's own 33 Hz gate, so a number
          under 30 means the phone is genuinely dropping frames. */}
      <div className={bad(snap.fps < 30)}>
        fps {snap.fps.toFixed(0)} · tick {snap.tickMs.toFixed(1)}ms · worst {snap.worstTickMs.toFixed(0)}ms
      </div>
      {/* The governor's own view. `tier` is what it has turned the loop down to; `median` is
          the number that decision was made on; `analysis` says whether the per-frame image
          work actually left the main thread on this device. */}
      <div className={bad(snap.tier > 0)}>
        tier {snap.tier}/{snap.floorTier} · med {snap.medianMs.toFixed(1)}ms · jank{" "}
        {(snap.jankRatio * 100).toFixed(0)}% · {snap.analysis === "worker" ? "wkr" : "MAIN"}
      </div>
      {/* mp4 = hardware encoder, webm = libvpx on the CPU. There is no third possibility on
          Chrome for Android, so this one line answers a question three rounds of analysis
          argued about from a stale comment. */}
      <div className={bad(!snap.codec.startsWith("video/mp4"))}>
        {snap.codec.replace("video/", "").split(";")[0]} · {snap.camW}x{snap.camH}
        {snap.demoted ? ` · ${snap.demoted}` : null}
      </div>
      {/* Over ~12 ms a rotating capture cannot be sharp however slowly it is swept, so this
          is the single number that predicts whether the panorama will have visible seams. */}
      {/* iso and luma sit next to exp because the three are one measurement: shortening the
          exposure is only safe when the gain rises to pay for it, and luma is the proof it
          did. A dark preview with exp low and iso flat is the trade going unpaid. */}
      <div className={bad(snap.exposureMs > MAX_EXPOSURE_SECONDS * 1000 * 1.17 || snap.lumaMean < THRESHOLDS.framing.minMedianLuma)}>
        exp {snap.exposureMs.toFixed(1)}ms · iso {snap.iso || "-"} · luma {snap.lumaMean.toFixed(0)}
      </div>
      <div className={bad(snap.orientationAgeMs > 300)}>
        orient {snap.orientationAgeMs.toFixed(0)}ms · motion {snap.motionAgeMs < 0 ? "DEGRADED" : "ok"} · ω {snap.omegaDegPerS.toFixed(0)}°/s
      </div>
      <div className={bad(snap.recordedBytes === 0)}>
        rec {snap.recorderState} · {snap.recordedBytes} chunks · {snap.state}
        {snap.blocked ? " · BLOCKED" : ""}
      </div>
      {log.length > 0 ? (
        <div className="mt-1 border-t border-white/20 pt-1">
          {log.map((e, i) => (
            <div key={i} className={e.severity === "stop" ? "text-red-400" : "text-amber-300"}>
              {e.at.toFixed(1)}s {e.code} [{e.severity}]
            </div>
          ))}
        </div>
      ) : (
        <div className="mt-1 border-t border-white/20 pt-1 text-white/50">no warnings yet</div>
      )}
    </div>
  );
}
