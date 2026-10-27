"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface UseCopyToClipboard {
  copied: boolean;
  failed: boolean;
  copy: (text: string) => Promise<boolean>;
}

/**
 * `navigator.clipboard` is unavailable on insecure origins, which is exactly how an agent
 * reaches the dashboard from a phone on the office Wi-Fi, so a legacy fallback stays.
 */
export function useCopyToClipboard(resetAfterMs = 2000): UseCopyToClipboard {
  const [copied, setCopied] = useState(false);
  const [failed, setFailed] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const copy = useCallback(
    async (text: string) => {
      const done = (ok: boolean) => {
        setCopied(ok);
        setFailed(!ok);
        if (timer.current) clearTimeout(timer.current);
        timer.current = setTimeout(() => {
          setCopied(false);
          setFailed(false);
        }, resetAfterMs);
        return ok;
      };

      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(text);
          return done(true);
        }
      } catch {
        // Fall through to the textarea trick.
      }

      try {
        const area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(area);
        return done(ok);
      } catch {
        return done(false);
      }
    },
    [resetAfterMs],
  );

  return { copied, failed, copy };
}
