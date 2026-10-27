"use client";

import { useCallback, useEffect } from "react";
import { t, type StringKey } from "@/lib/i18n";
import { hydratePersistedState, useAppStore } from "@/lib/store";
import type { Locale } from "@/lib/types";

export interface UseLocale {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  toggleLocale: () => void;
  /** `t` pre-bound to the active locale. */
  t: (key: StringKey) => string;
}

export function useLocale(): UseLocale {
  const locale = useAppStore((s) => s.locale);
  const setLocale = useAppStore((s) => s.setLocale);
  const toggleLocale = useAppStore((s) => s.toggleLocale);

  useEffect(hydratePersistedState, []);

  const translate = useCallback((key: StringKey) => t(key, locale), [locale]);

  return { locale, setLocale, toggleLocale, t: translate };
}
