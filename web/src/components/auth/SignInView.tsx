"use client";

import { ArrowLeft, Eye, EyeOff, Loader2 } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { useSession } from "./SessionProvider";
import { Button, buttonStyles } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { LogoMark } from "@/components/dashboard/Logo";
import { ui } from "@/components/dashboard/copy";
import { BioSphereError, api } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import { cn } from "@/lib/utils";

type Mode = "signIn" | "signUp";

/** Google's mark, inline — an external image on the sign-in screen would be a third-party
 *  request on every visit, made for a logo. */
function GoogleMark() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" aria-hidden>
      <path
        fill="#4285F4"
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.76h3.57c2.08-1.92 3.28-4.74 3.28-8.09Z"
      />
      <path
        fill="#34A853"
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.76c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23Z"
      />
      <path
        fill="#FBBC05"
        d="M5.84 14.11a6.6 6.6 0 0 1 0-4.22V7.05H2.18a11 11 0 0 0 0 9.9l3.66-2.84Z"
      />
      <path
        fill="#EA4335"
        d="M12 4.75c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 1.46 14.97.5 12 .5A11 11 0 0 0 2.18 7.05l3.66 2.84c.87-2.6 3.3-4.14 6.16-4.14Z"
      />
    </svg>
  );
}

/**
 * The door. One screen with two modes rather than two routes, because the difference between
 * signing in and signing up is one field and a verb, and a person who guessed wrong should
 * not have to find another page to fix it.
 */
export function SignInView() {
  const { locale } = useLocale();
  const router = useRouter();
  const params = useSearchParams();
  const { status, adopt } = useSession();

  const next = params.get("next");
  const googleError = params.get("google");
  const code = params.get("code");

  const [mode, setMode] = useState<Mode>(params.get("mod") === "kayit" ? "signUp" : "signIn");
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [reveal, setReveal] = useState(false);
  // Seeded from the URL rather than set in an effect: arriving with `?code=` means the
  // exchange starts on the first render, and a button that is briefly enabled before that
  // effect runs is a double sign-in waiting to happen.
  const [busy, setBusy] = useState(() => Boolean(params.get("code")));
  const [formError, setFormError] = useState<string | null>(null);
  const [googleEnabled, setGoogleEnabled] = useState(false);

  const destination = useMemo(() => {
    // Only ever a path on this site. A `next` of `https://elsewhere/` would turn the sign-in
    // screen into an open redirect, which is a phishing primitive with our domain on it.
    if (next && next.startsWith("/") && !next.startsWith("//")) return next;
    return "/evler";
  }, [next]);

  useEffect(() => {
    void api.auth
      .googleStatus()
      .then((s) => setGoogleEnabled(s.enabled))
      .catch(() => setGoogleEnabled(false));
  }, []);

  // Derived, not stored. What Google said is a fact about the URL, and copying a fact into
  // state only creates a second place for it to be wrong.
  const googleMessage = useMemo(() => {
    if (!googleError) return null;
    if (googleError === "cancelled") return ui("auth.google.cancelled", locale);
    if (googleError === "closed") return ui("auth.google.closed", locale);
    return ui("auth.google.failed", locale);
  }, [googleError, locale]);

  const error = formError ?? googleMessage;

  // The one-time code Google's callback handed back, traded for a session.
  useEffect(() => {
    if (!code) return;
    let cancelled = false;
    void api.auth
      .exchange(code)
      .then((session) => {
        if (cancelled) return;
        adopt(session.user);
        router.replace(destination);
      })
      .catch((cause) => {
        if (cancelled) return;
        setFormError(cause instanceof BioSphereError ? cause.localised(locale) : ui("auth.google.failed", locale));
        setBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [code, adopt, router, destination, locale]);

  // Already signed in and arriving here by hand or by a stale link.
  useEffect(() => {
    if (status === "signed-in" && !code) router.replace(destination);
  }, [status, code, router, destination]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setFormError(null);
    setBusy(true);
    try {
      const session =
        mode === "signUp"
          ? await api.auth.register({ email: email.trim(), name: name.trim(), password })
          : await api.auth.login({ email: email.trim(), password });
      adopt(session.user);
      router.replace(destination);
    } catch (cause) {
      setFormError(cause instanceof BioSphereError ? cause.localised(locale) : t("error.generic", locale));
      setBusy(false);
    }
  };

  return (
    <main className="relative isolate flex min-h-dvh flex-col overflow-x-clip">
      <div className="grid-bg pointer-events-none absolute inset-x-0 top-0 -z-10 h-[620px]" aria-hidden />
      <div
        className="pointer-events-none absolute top-[-280px] left-1/2 -z-10 size-[640px] -translate-x-1/2 rounded-full
                   bg-[radial-gradient(circle,var(--bio-dim)_0%,transparent_62%)] blur-2xl"
        aria-hidden
      />

      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6">
        <Link href="/" className="flex items-center gap-2.5 text-ink" aria-label={t("app.name", locale)}>
          <span className="grid size-8 place-items-center rounded-sm border border-line bg-surface-2 text-bio">
            <LogoMark className="size-4.5" />
          </span>
          <span className="text-[15px] font-semibold tracking-tight">BioSphere</span>
        </Link>
        <Link href="/" className={buttonStyles("ghost", "sm")}>
          <ArrowLeft className="size-3.5" aria-hidden />
          {ui("auth.backHome", locale)}
        </Link>
      </header>

      <div className="flex flex-1 items-start justify-center px-6 pt-4 pb-20 sm:items-center sm:pt-0">
        <div className="w-full max-w-[420px]">
          <div className="glass rounded-lg p-6 sm:p-8">
            <h1 className="text-[22px] font-semibold tracking-[-0.02em] text-ink">
              {ui(mode === "signIn" ? "auth.signIn.title" : "auth.signUp.title", locale)}
            </h1>
            <p className="mt-2 text-[13.5px] leading-relaxed text-ink-muted">
              {ui(mode === "signIn" ? "auth.signIn.body" : "auth.signUp.body", locale)}
            </p>

            {googleEnabled ? (
              <>
                <a
                  href={api.auth.googleStartUrl()}
                  className={cn(buttonStyles("secondary", "md"), "mt-6 w-full")}
                  // A full navigation, not a fetch: the Google flow is a redirect chain that
                  // has to happen in the address bar for the user to see whose page they are
                  // typing a password into.
                >
                  <GoogleMark />
                  {ui("auth.google", locale)}
                </a>
                <div className="my-5 flex items-center gap-3" aria-hidden>
                  <span className="h-px flex-1 bg-line" />
                  <span className="text-[11px] tracking-wide text-ink-faint uppercase">{ui("auth.or", locale)}</span>
                  <span className="h-px flex-1 bg-line" />
                </div>
              </>
            ) : null}

            <form onSubmit={submit} className={cn("space-y-4", googleEnabled ? "" : "mt-6")}>
              {mode === "signUp" ? (
                <Input
                  label={ui("auth.name", locale)}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  autoComplete="name"
                  required
                  disabled={busy}
                />
              ) : null}

              <Input
                type="email"
                label={ui("auth.email", locale)}
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
                inputMode="email"
                required
                disabled={busy}
              />

              <div className="relative">
                <Input
                  type={reveal ? "text" : "password"}
                  label={ui("auth.password", locale)}
                  hint={mode === "signUp" ? ui("auth.password.hint", locale) : undefined}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete={mode === "signUp" ? "new-password" : "current-password"}
                  required
                  disabled={busy}
                  className="pr-11"
                />
                <button
                  type="button"
                  onClick={() => setReveal((v) => !v)}
                  aria-label={ui(reveal ? "auth.password.hide" : "auth.password.show", locale)}
                  className="absolute top-[26px] right-1 grid size-9 place-items-center rounded-sm text-ink-faint transition-colors hover:text-ink-muted"
                >
                  {reveal ? <EyeOff className="size-4" aria-hidden /> : <Eye className="size-4" aria-hidden />}
                </button>
              </div>

              {error ? (
                <p role="alert" className="rounded-sm border border-danger/30 bg-danger/10 px-3 py-2 text-[13px] text-danger">
                  {error}
                </p>
              ) : null}

              <Button type="submit" variant="primary" size="lg" className="w-full" loading={busy} disabled={busy}>
                {busy ? <Loader2 className="size-4 animate-spin" aria-hidden /> : null}
                {ui(mode === "signIn" ? "auth.signIn" : "auth.signUp", locale)}
              </Button>
            </form>
          </div>

          <p className="mt-5 text-center text-[13px] text-ink-muted">
            {ui(mode === "signIn" ? "auth.noAccount" : "auth.haveAccount", locale)}{" "}
            <button
              type="button"
              onClick={() => {
                setMode(mode === "signIn" ? "signUp" : "signIn");
                setFormError(null);
              }}
              className="font-medium text-bio underline-offset-2 hover:underline"
            >
              {ui(mode === "signIn" ? "auth.signUp" : "auth.signIn", locale)}
            </button>
          </p>
        </div>
      </div>
    </main>
  );
}
