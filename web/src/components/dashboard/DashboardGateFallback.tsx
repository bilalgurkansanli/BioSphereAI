"use client";

import { LogoMark } from "./Logo";
import { ui } from "./copy";
import { useLocale } from "@/lib/hooks/useLocale";

/**
 * What the dashboard shows while it does not yet know who you are.
 *
 * Not a spinner and not a sign-in prompt: it is on screen for the length of one request to
 * `/auth/me`, and a returning user must not see the word "giriş" flash before their own
 * properties appear. When the answer is "nobody", `RequireSession` navigates away from under
 * this and the sign-in screen takes over.
 */
export function DashboardGateFallback() {
  const { locale } = useLocale();
  return (
    <div className="grid min-h-dvh place-items-center px-6">
      <div className="flex flex-col items-center gap-3 text-center">
        <span className="animate-breathe grid size-11 place-items-center rounded-md border border-line bg-surface-2 text-bio">
          <LogoMark className="size-5" />
        </span>
        <p className="text-[13px] text-ink-faint">{ui("auth.checking", locale)}</p>
      </div>
    </div>
  );
}
