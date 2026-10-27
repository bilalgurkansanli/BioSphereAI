"use client";

import { useCallback, useEffect, useRef } from "react";
import { api } from "@/lib/api";
import { useAppStore, type EngineHealth } from "@/lib/store";
import type { Locale } from "@/lib/types";

/** Below this the engine can still stitch, but an 8K studio job will run out of room. */
const LOW_DISK_GB = 3;

function inspect(
  info: Awaited<ReturnType<typeof api.health>>,
  locale: Locale,
): EngineHealth {
  const issues: string[] = [];

  if (!info.ffmpeg) {
    issues.push(
      locale === "tr"
        ? "FFmpeg bulunamadı — video yüklemeleri işlenemez."
        : "FFmpeg is missing — video uploads cannot be processed.",
    );
  }
  if (info.media_root_free_gb < LOW_DISK_GB) {
    issues.push(
      locale === "tr"
        ? `Disk alanı azalıyor — ${info.media_root_free_gb.toFixed(1)} GB kaldı.`
        : `Low disk space — ${info.media_root_free_gb.toFixed(1)} GB left.`,
    );
  }

  return {
    status: issues.length > 0 ? "degraded" : "online",
    issues,
    info,
    checkedAt: Date.now(),
  };
}

export interface UseEngineHealth extends EngineHealth {
  refresh: () => void;
}

/**
 * Polls `GET /health`. Mount this once (the top bar does); everything else reads the same
 * value straight from the store so a dashboard does not open five parallel health loops.
 */
export function useEngineHealth(intervalMs = 20_000): UseEngineHealth {
  const engine = useAppStore((s) => s.engine);
  const setEngine = useAppStore((s) => s.setEngine);
  const locale = useAppStore((s) => s.locale);

  const localeRef = useRef(locale);
  localeRef.current = locale;

  const aliveRef = useRef(true);

  const check = useCallback(async () => {
    try {
      const info = await api.health();
      if (aliveRef.current) setEngine(inspect(info, localeRef.current));
    } catch {
      if (!aliveRef.current) return;
      setEngine({
        status: "offline",
        issues: [
          localeRef.current === "tr"
            ? "Motora ulaşılamıyor. Terminalden `biosphere serve` çalıştırın."
            : "The engine is unreachable. Run `biosphere serve` in a terminal.",
        ],
        info: null,
        checkedAt: Date.now(),
      });
    }
  }, [setEngine]);

  useEffect(() => {
    aliveRef.current = true;
    void check();

    const id = setInterval(() => void check(), intervalMs);
    // A laptop that slept for an hour should not show an hour-old green pill.
    const onVisible = () => {
      if (document.visibilityState === "visible") void check();
    };
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      aliveRef.current = false;
      clearInterval(id);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [check, intervalMs]);

  return { ...engine, refresh: () => void check() };
}
