"use client";

/**
 * Global UI state — deliberately tiny.
 *
 * Only things that are genuinely app-wide live here: the chosen locale, the engine's
 * health (rendered in the top bar but read from several places), and a single helper that
 * turns a caught value into copy a human can act on. Server state — properties, scenes, jobs —
 * stays in the component that fetches it, so there is never a second source of truth to
 * invalidate.
 */

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { BioSphereError } from "@/lib/api";
import { DEFAULT_LOCALE, t } from "@/lib/i18n";
import type { HealthResponse, Locale } from "@/lib/types";

export type EngineStatus = "checking" | "online" | "degraded" | "offline";

export interface EngineHealth {
  status: EngineStatus;
  /** Why the pill is not green. Empty when everything is fine. */
  issues: string[];
  info: HealthResponse | null;
  checkedAt: number | null;
}

export const INITIAL_ENGINE: EngineHealth = {
  status: "checking",
  issues: [],
  info: null,
  checkedAt: null,
};

interface AppState {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  toggleLocale: () => void;

  engine: EngineHealth;
  setEngine: (engine: EngineHealth) => void;

  /** Never let a raw exception reach the screen. */
  describeError: (error: unknown) => string;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      locale: DEFAULT_LOCALE,
      setLocale: (locale) => set({ locale }),
      toggleLocale: () => set({ locale: get().locale === "tr" ? "en" : "tr" }),

      engine: INITIAL_ENGINE,
      setEngine: (engine) => set({ engine }),

      describeError: (error) => {
        const locale = get().locale;
        if (error instanceof BioSphereError) return error.localised(locale);
        return t("error.generic", locale);
      },
    }),
    {
      name: "biosphere.ui",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ locale: state.locale }),
      /**
       * The server always renders Turkish. Rehydrating during module evaluation would let
       * a persisted `en` reach the first client render and trip hydration, so `useLocale`
       * rehydrates inside an effect instead — one frame of Turkish beats a mismatch.
       */
      skipHydration: true,
    },
  ),
);

let rehydrateStarted = false;

/** Idempotent: many components call `useLocale`, only the first one restores localStorage. */
export function hydratePersistedState(): void {
  if (rehydrateStarted) return;
  rehydrateStarted = true;
  void useAppStore.persist.rehydrate();
}
