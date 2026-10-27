"use client";

import { useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import { SIGNED_OUT_EVENT, clearToken } from "@/lib/session";
import type { AuthUser } from "@/lib/types";

type Status = "checking" | "signed-in" | "signed-out";

interface SessionValue {
  status: Status;
  user: AuthUser | null;
  /** Adopt a session the sign-in form just obtained, without a second round trip. */
  adopt(user: AuthUser): void;
  signOut(): Promise<void>;
}

const SessionContext = createContext<SessionValue | null>(null);

/**
 * Who is signed in, for the whole client tree.
 *
 * The status starts at `checking` rather than `signed-out` and that distinction is the whole
 * design: rendering "you are not signed in" for the half-second it takes to ask the engine
 * would flash a sign-in screen at every returning user on every page load. Guarded areas
 * show nothing until the answer is known.
 *
 * There is no polling. The one event that can invalidate a session behind our back — the
 * engine rejecting the token — already reaches us through `SIGNED_OUT_EVENT`, which the API
 * client fires on any 401.
 */
export function SessionProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status>("checking");
  const [user, setUser] = useState<AuthUser | null>(null);

  useEffect(() => {
    let cancelled = false;

    // ASKED EVEN WITHOUT A TOKEN. In the engine's default local mode (`BIOSPHERE_AUTH_MODE`)
    // there is no sign-in at all: `/auth/me` answers with the installation's owner, and the
    // dashboard opens straight away. In accounts mode the same call answers 401 and the
    // sign-in screen follows, as before.
    void api.auth
      .me()
      .then((me) => {
        if (cancelled) return;
        setUser(me);
        setStatus("signed-in");
      })
      .catch(() => {
        if (cancelled) return;
        // Either the token is dead (the API client has already dropped it) or the engine is
        // down. Both leave the user unable to use the dashboard, and the sign-in screen is
        // where either one is recoverable.
        clearToken();
        setUser(null);
        setStatus("signed-out");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onSignedOut = () => {
      setUser(null);
      setStatus("signed-out");
    };
    window.addEventListener(SIGNED_OUT_EVENT, onSignedOut);
    return () => window.removeEventListener(SIGNED_OUT_EVENT, onSignedOut);
  }, []);

  const adopt = useCallback((next: AuthUser) => {
    setUser(next);
    setStatus("signed-in");
  }, []);

  const signOut = useCallback(async () => {
    await api.auth.logout();
    setUser(null);
    setStatus("signed-out");
  }, []);

  const value = useMemo<SessionValue>(() => ({ status, user, adopt, signOut }), [status, user, adopt, signOut]);
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionValue {
  const value = useContext(SessionContext);
  if (!value) throw new Error("useSession must be used inside <SessionProvider>");
  return value;
}

/**
 * Renders its children only for a signed-in visitor; sends everyone else to the sign-in
 * screen, remembering where they were going.
 *
 * The redirect happens in an effect rather than during render because Next forbids
 * navigating while rendering, and because the answer is not known on the server: the token
 * lives in `localStorage`, which the server has never seen. That is also why the guarded
 * pages are client-rendered rather than protected by middleware — middleware runs where the
 * credential is not.
 */
export function RequireSession({ children, fallback }: { children: ReactNode; fallback: ReactNode }) {
  const { status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status !== "signed-out") return;
    const next = `${window.location.pathname}${window.location.search}`;
    const query = next && next !== "/" ? `?next=${encodeURIComponent(next)}` : "";
    router.replace(`/giris${query}`);
  }, [status, router]);

  if (status === "signed-in") return <>{children}</>;
  return <>{fallback}</>;
}
