/**
 * Shown when the public token does not resolve to a property.
 *
 * This renders inside a stranger's website, so it apologises quietly and does not offer
 * navigation the visitor cannot use — there is no dashboard behind this frame for them.
 */

import { DEFAULT_LOCALE, t } from "@/lib/i18n";

export default function EmbedNotFound() {
  return (
    <div className="grid h-full w-full place-items-center bg-canvas px-6 text-center">
      <div className="animate-rise flex max-w-sm flex-col items-center gap-3">
        <span aria-hidden className="bio-gradient h-1 w-9 rounded-full opacity-60" />
        <h1 className="text-[15px] font-semibold tracking-tight text-ink">{t("error.generic", DEFAULT_LOCALE)}</h1>
        <p className="text-[13px] leading-relaxed text-ink-muted">{t("app.tagline", DEFAULT_LOCALE)}</p>
        <span className="mt-1 text-[10px] font-semibold tracking-[0.16em] text-ink-faint uppercase">
          {t("app.name", DEFAULT_LOCALE)}
        </span>
      </div>
    </div>
  );
}
