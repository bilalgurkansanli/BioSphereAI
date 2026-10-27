"use client";

import { CheckCircle2, ExternalLink, KeyRound, ShieldCheck, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState, type FormEvent } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { fieldStyles } from "@/components/ui/Input";
import { api, BioSphereError } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import type { AiSettings, Locale } from "@/lib/types";
import { cn } from "@/lib/utils";

const COPY = {
  title: { tr: "Ayarlar", en: "Settings" },
  subtitle: {
    tr: "Bu kurulum yalnızca sizin bilgisayarınızda çalışır. Ayarlar da burada kalır.",
    en: "This installation runs only on your computer, and so do its settings.",
  },
  keyTitle: { tr: "Anthropic API anahtarı", en: "Anthropic API key" },
  keyBody: {
    tr: "İlan metni ve yatırım asistanı Claude ile çalışır. Proje bir anahtarla gelmez: kendi anahtarınızı girersiniz, kullanım kendi hesabınıza yazılır. Anahtar motorun veri klasöründe saklanır, depoya girmez ve tam hâli bir daha gösterilmez.",
    en: "Listing copy and the investment assistant run on Claude. The project ships without a key: you enter your own, and usage is billed to your own account. The key is stored in the engine's data folder, never in the repository, and is never shown in full again.",
  },
  getKey: { tr: "Anahtar al (console.anthropic.com)", en: "Get a key (console.anthropic.com)" },
  placeholder: { tr: "sk-ant-…", en: "sk-ant-…" },
  save: { tr: "Kaydet", en: "Save" },
  replace: { tr: "Değiştir", en: "Replace" },
  check: { tr: "Anahtarı test et", en: "Test the key" },
  remove: { tr: "Anahtarı sil", en: "Remove key" },
  saved: { tr: "Anahtar kaydedildi.", en: "Key saved." },
  removed: { tr: "Anahtar silindi.", en: "Key removed." },
  none: { tr: "Anahtar yok — AI özellikleri kapalı.", en: "No key — the AI features are off." },
  fromDashboard: { tr: "Bu panelden girildi", en: "Entered in this dashboard" },
  fromEnv: { tr: "engine/.env veya ortam değişkeninden", en: "From engine/.env or the environment" },
  envNote: {
    tr: "Buraya girdiğiniz anahtar .env'dekinin önüne geçer.",
    en: "A key entered here takes precedence over the one in .env.",
  },
  ownerOnly: {
    tr: "Anahtarı yalnızca kurulumun sahibi değiştirebilir.",
    en: "Only the installation's owner can change the key.",
  },
  models: { tr: "Modeller", en: "Models" },
  listing: { tr: "İlan metni", en: "Listing copy" },
  advisor: { tr: "Yatırım asistanı", en: "Investment assistant" },
  modelsNote: {
    tr: "Model seçimi engine/.env içindeki BIOSPHERE_LISTING_MODEL ve BIOSPHERE_ADVISOR_MODEL ile değişir.",
    en: "Change models with BIOSPHERE_LISTING_MODEL and BIOSPHERE_ADVISOR_MODEL in engine/.env.",
  },
} satisfies Record<string, Record<Locale, string>>;

const say = (key: keyof typeof COPY, locale: Locale) => COPY[key][locale];

function errorText(error: unknown, locale: Locale): string {
  return error instanceof BioSphereError ? error.localised(locale) : String(error);
}

export function SettingsView() {
  const { locale } = useLocale();
  const [settings, setSettings] = useState<AiSettings | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState<"save" | "check" | "remove" | null>(null);
  const [check, setCheck] = useState<{ ok: boolean; message: string } | null>(null);

  const load = useCallback(() => {
    api.settings
      .ai()
      .then(setSettings)
      .catch((error) => toast.error(errorText(error, locale)));
  }, [locale]);

  useEffect(load, [load]);

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!draft.trim()) return;
    setBusy("save");
    setCheck(null);
    try {
      setSettings(await api.settings.saveAiKey(draft.trim()));
      setDraft("");
      toast.success(say("saved", locale));
    } catch (error) {
      toast.error(errorText(error, locale));
    } finally {
      setBusy(null);
    }
  }

  async function test() {
    setBusy("check");
    try {
      setCheck(await api.settings.checkAiKey());
    } catch (error) {
      setCheck({ ok: false, message: errorText(error, locale) });
    } finally {
      setBusy(null);
    }
  }

  async function remove() {
    setBusy("remove");
    setCheck(null);
    try {
      setSettings(await api.settings.clearAiKey());
      toast.success(say("removed", locale));
    } catch (error) {
      toast.error(errorText(error, locale));
    } finally {
      setBusy(null);
    }
  }

  const status = !settings
    ? null
    : !settings.configured
      ? say("none", locale)
      : `${settings.hint} · ${say(settings.source === "dashboard" ? "fromDashboard" : "fromEnv", locale)}`;

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="mb-7">
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{say("title", locale)}</h1>
        <p className="mt-1 text-[13px] text-ink-muted">{say("subtitle", locale)}</p>
      </div>

      <Card className="space-y-5 p-5 sm:p-6">
        <div className="flex items-start gap-3">
          <span className="grid size-9 shrink-0 place-items-center rounded-md border border-line bg-surface-2 text-bio">
            <KeyRound className="size-4" aria-hidden />
          </span>
          <div className="min-w-0">
            <h2 className="text-[15px] font-semibold text-ink">{say("keyTitle", locale)}</h2>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-muted">{say("keyBody", locale)}</p>
            <a
              href="https://console.anthropic.com/settings/keys"
              target="_blank"
              rel="noreferrer"
              className="mt-2 inline-flex items-center gap-1 text-[13px] font-medium text-bio hover:underline"
            >
              {say("getKey", locale)}
              <ExternalLink className="size-3.5" aria-hidden />
            </a>
          </div>
        </div>

        <div
          role="status"
          className={cn(
            "flex items-center gap-2 rounded-md border px-3 py-2 font-mono text-[12px]",
            settings?.configured ? "border-success/25 text-success" : "border-line text-ink-faint",
          )}
        >
          {settings?.configured ? <ShieldCheck className="size-4 shrink-0" aria-hidden /> : null}
          <span className="truncate">{status ?? "…"}</span>
        </div>

        {settings && !settings.can_edit ? (
          <p className="text-[13px] text-ink-muted">{say("ownerOnly", locale)}</p>
        ) : (
          <form onSubmit={save} className="flex flex-col gap-2 sm:flex-row">
            <label htmlFor="anthropic-key" className="sr-only">
              {say("keyTitle", locale)}
            </label>
            <input
              id="anthropic-key"
              type="password"
              autoComplete="off"
              spellCheck={false}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={say("placeholder", locale)}
              className={cn(fieldStyles, "h-10 font-mono")}
            />
            <Button type="submit" variant="primary" loading={busy === "save"} disabled={!draft.trim()}>
              {say(settings?.source === "dashboard" ? "replace" : "save", locale)}
            </Button>
          </form>
        )}

        {settings?.source === "env" && settings.can_edit ? (
          <p className="text-[12px] text-ink-faint">{say("envNote", locale)}</p>
        ) : null}

        {settings?.configured ? (
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" onClick={test} loading={busy === "check"}>
              {say("check", locale)}
            </Button>
            {settings.source === "dashboard" && settings.can_edit ? (
              <Button size="sm" variant="danger" onClick={remove} loading={busy === "remove"}>
                <Trash2 className="size-3.5" aria-hidden />
                {say("remove", locale)}
              </Button>
            ) : null}
            {check ? (
              <span
                className={cn(
                  "inline-flex items-center gap-1.5 text-[13px]",
                  check.ok ? "text-success" : "text-danger",
                )}
              >
                {check.ok ? <CheckCircle2 className="size-4" aria-hidden /> : null}
                {check.message}
              </span>
            ) : null}
          </div>
        ) : null}

        {settings ? (
          <div className="border-t border-line pt-4">
            <h3 className="text-[13px] font-medium text-ink-muted">{say("models", locale)}</h3>
            <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[13px]">
              <dt className="text-ink-faint">{say("listing", locale)}</dt>
              <dd className="font-mono text-ink">{settings.listing_model}</dd>
              <dt className="text-ink-faint">{say("advisor", locale)}</dt>
              <dd className="font-mono text-ink">{settings.advisor_model}</dd>
            </dl>
            <p className="mt-2 text-[12px] text-ink-faint">{say("modelsNote", locale)}</p>
          </div>
        ) : null}
      </Card>
    </div>
  );
}
