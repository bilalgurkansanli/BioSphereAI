"use client";

/**
 * The listing's headline and description, written from the property and edited before it is kept.
 *
 * ── WHY THIS IS A DRAFT AND NOT AN ANSWER ─────────────────────────────────────────────
 * What comes back is text about somebody's property that somebody is about to publish under
 * their own name. So nothing here saves on its own: the copy lands in two ordinary form
 * fields, the agent edits them, and "Tura kaydet" is the same `PATCH /properties/{id}` the title
 * has always gone through. A panel that wrote straight into the property would be a panel that
 * quietly rewrote a listing every time somebody pressed a button to see what it would say.
 *
 * ── AND WHY THE WARNINGS ARE AS PROMINENT AS THE TEXT ─────────────────────────────────
 * A paragraph that invents a square metre reads exactly as well as one that does not. The
 * engine checks every figure in the answer against the facts it supplied and reports what it
 * cannot account for, and the whole value of that check is lost if it lands in a tooltip.
 * They sit above the text, in the agent's own language, and they say what to do about it.
 *
 * ── THE FACTS STRIP ───────────────────────────────────────────────────────────────────
 * Four numbers under the copy: rooms, measured rooms, total measured area, ceiling. Not
 * decoration — it is the panel's answer to "why is this description so thin". A property whose
 * rooms have never been measured has nothing to say about size, and the fix is one button
 * away on the plan below.
 */

import { AlertTriangle, Check, PencilLine, RotateCcw, Sparkles, Wand2 } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { ui } from "./copy";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { CopyButton } from "@/components/ui/CopyButton";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Skeleton } from "@/components/ui/Skeleton";
import { Textarea } from "@/components/ui/Textarea";
import { BioSphereError, api } from "@/lib/api";
import { useApi } from "@/lib/hooks/useApi";
import type { ListingCopy, ListingState, ListingTone, Locale, PropertyDetail } from "@/lib/types";

const TONES: ListingTone[] = ["professional", "warm", "concise", "premium"];

/**
 * The engine's warning codes, in the agent's language.
 *
 * Kept here rather than in `copy.tsx` because half of them carry a payload — the invented
 * figure, the claim, the HTTP status — and the sentence has to wrap around it. An unknown
 * code falls through to its own text: a warning nobody has translated yet is still a warning,
 * and swallowing it would be the one failure this panel exists to prevent.
 */
const WARNINGS: Record<string, { tr: (d: string) => string; en: (d: string) => string }> = {
  ungrounded_area: {
    tr: (d) => `Metinde ölçülmemiş bir alan geçiyor: “${d}”. Bu sayı turdan gelmiyor.`,
    en: (d) => `The text contains an area nobody measured: “${d}”. That figure is not from the property.`,
  },
  mentions_price: {
    tr: () => "Metinde fiyat geçiyor. Fiyatı sistem bilmiyor; siz doğrulayın.",
    en: () => "The text mentions a price. Nothing here knows one; check it yourself.",
  },
  // The portal rules. Worded as consequences rather than as style notes, because they are:
  // sahibinden does not publish a listing whose text carries these.
  contact_phone: {
    tr: () => "Metinde telefon numarası var. sahibinden ilan açıklamasında numara kabul etmiyor — ilan yayınlanmaz.",
    en: () => "The text contains a phone number. sahibinden does not allow one in the description — the listing will not be published.",
  },
  contact_email: {
    tr: () => "Metinde e-posta adresi var. İlan açıklamasında iletişim bilgisi yasak.",
    en: () => "The text contains an e-mail address. Contact details are not allowed in the description.",
  },
  contact_link: {
    tr: () => "Metinde bağlantı ya da site adresi var. İlan açıklamasında link yasak.",
    en: () => "The text contains a link or a web address. Links are not allowed in the description.",
  },
  unsupported_claim: {
    tr: (d) => `“${d}” hakkındaki ifade turdan ya da fotoğraflardan doğrulanamaz.`,
    en: (d) => `The claim about “${d}” cannot be checked against the property or the pictures.`,
  },
  title_too_long: {
    tr: (d) => `Başlık ${d} karakter; ilan siteleri genellikle 65'te kesiyor.`,
    en: (d) => `The title is ${d} characters; listing portals usually cut at 65.`,
  },
  description_long: {
    tr: (d) => `Açıklama ${d} kelime — istenenden uzun.`,
    en: (d) => `The description is ${d} words — longer than asked for.`,
  },
  description_empty: {
    tr: () => "Model boş bir açıklama döndürdü.",
    en: () => "The model returned an empty description.",
  },
  model_off: {
    tr: (d) =>
      d === "no_sdk"
        ? "Yazım modeli kurulu değil; metin ölçülen bilgilerden kurallı olarak yazıldı."
        : "Model anahtarı tanımlı değil; metin ölçülen bilgilerden kurallı olarak yazıldı.",
    en: (d) =>
      d === "no_sdk"
        ? "The writing model is not installed; this copy was composed from the measured facts."
        : "No model key is configured; this copy was composed from the measured facts.",
  },
  model_unavailable: {
    tr: (d) => `Model yanıt vermedi (${d}); metin ölçülen bilgilerden yazıldı.`,
    en: (d) => `The model did not answer (${d}); this copy was composed from the facts.`,
  },
};

function describeWarning(code: string, locale: Locale): string {
  const at = code.indexOf(":");
  const key = at === -1 ? code : code.slice(0, at);
  const detail = at === -1 ? "" : code.slice(at + 1);
  const entry = WARNINGS[key];
  return entry ? entry[locale](detail) : code;
}

/** A warning about the *text* is the agent's problem; one about the model is the server's. */
function isAboutTheText(code: string): boolean {
  return !code.startsWith("model_off") && !code.startsWith("model_unavailable");
}

export interface ListingPanelProps {
  property: PropertyDetail;
  locale: Locale;
  /** Called after the copy is written into the property, so the page can pick the new title up. */
  onSaved: () => void;
}

export function ListingPanel({ property, locale, onSaved }: ListingPanelProps) {
  const { data: state, initial, setData } = useApi<ListingState>(
    () => api.listing.get(property.id),
    [property.id],
  );

  const [instructions, setInstructions] = useState("");
  const [writing, setWriting] = useState(false);
  const [saving, setSaving] = useState(false);

  const copy: ListingCopy | null = state?.latest ?? null;

  /**
   * The two fields are DERIVED until somebody types in them.
   *
   * The cached copy arrives after the first paint, and a written one arrives seconds after a
   * button, so "put the answer in the box" cannot be an effect that copies state into state:
   * it would either fight the agent's typing or need a flag to stop it. Instead `edited` is
   * null until the first keystroke and the boxes read through to the newest thing the server
   * has — the fresh copy, else the cached one, else the property's own title. Writing again
   * clears it, which is what "write me a different one" means.
   */
  const [edited, setEdited] = useState<{ title: string; description: string } | null>(null);
  const [chosenTone, setChosenTone] = useState<ListingTone | null>(null);

  const title = edited?.title ?? copy?.title ?? property.title;
  const description = edited?.description ?? copy?.description ?? property.description ?? "";
  const tone = chosenTone ?? copy?.tone ?? "professional";
  const edit = (next: Partial<{ title: string; description: string }>) =>
    setEdited({ title, description, ...next });

  const write = async (refresh: boolean) => {
    setWriting(true);
    try {
      const written = await api.listing.generate(property.id, {
        locale,
        tone,
        instructions: instructions.trim() || undefined,
        refresh,
      });
      setData((current) => ({
        available: current?.available ?? written.model_backed,
        model: current?.model ?? written.model,
        reason: current?.reason ?? null,
        latest: written,
      }));
      setEdited(null);
    } catch (cause) {
      toast.error(
        cause instanceof BioSphereError ? cause.localised(locale) : ui("listing.failed", locale),
      );
    } finally {
      setWriting(false);
    }
  };

  const save = async () => {
    setSaving(true);
    try {
      await api.properties.update(property.id, { title: title.trim(), description });
      toast.success(ui("listing.saved", locale));
      onSaved();
    } catch (cause) {
      toast.error(
        cause instanceof BioSphereError ? cause.localised(locale) : ui("listing.failed", locale),
      );
    } finally {
      setSaving(false);
    }
  };

  const textWarnings = useMemo(
    () => (copy?.warnings ?? []).filter(isAboutTheText),
    [copy?.warnings],
  );
  const modelNotes = useMemo(
    () => (copy?.warnings ?? []).filter((w) => !isAboutTheText(w)),
    [copy?.warnings],
  );

  const dirty = title.trim() !== property.title || description !== (property.description ?? "");
  const words = description.trim() ? description.trim().split(/\s+/).length : 0;

  if (initial) {
    return (
      <section className="glass mt-6 rounded-lg p-5">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="mt-4 h-24 w-full" />
      </section>
    );
  }

  return (
    <section aria-label={ui("listing.title", locale)} className="glass mt-6 rounded-lg p-5">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="flex items-center gap-2 text-sm font-semibold tracking-tight text-ink">
            <Sparkles className="size-4 text-bio" aria-hidden />
            {ui("listing.title", locale)}
          </h2>
          <p className="mt-1 max-w-2xl text-[12.5px] leading-relaxed text-ink-muted">
            {ui("listing.body", locale)}
          </p>
        </div>
        {copy ? (
          <div className="flex flex-col items-end gap-1">
            <Badge tone={copy.model_backed ? "bio" : "neutral"} dot>
              {copy.model_backed
                ? `${copy.model ?? ""} · ${copy.images_used} ${ui("listing.images", locale)}`
                : ui("listing.offline", locale)}
            </Badge>
            {/* What it cost, next to what wrote it. The question "bu bize ne kadara mal
                oluyor" is asked while looking at the answer, not in a server log a week
                later — and `cached` is the difference between a bill and a memory of one. */}
            {copy.usage ? (
              <span className="text-[11px] tabular-nums text-ink-faint">
                {copy.usage.cost_usd !== null
                  ? `${copy.cached ? ui("listing.cost.cached", locale) : ui("listing.cost", locale)} $${copy.usage.cost_usd.toFixed(3)}`
                  : ui("listing.cost.unknown", locale)}
                {" · "}
                {Math.round((copy.usage.input_tokens + copy.usage.output_tokens) / 100) / 10}k token
                {" · "}
                {(copy.elapsed_ms / 1000).toFixed(0)} {ui("listing.seconds", locale)}
              </span>
            ) : null}
          </div>
        ) : null}
      </div>

      {/* Controls. The tone is a list rather than a free box on purpose — the engine turns
          each one into a sentence somebody has read — and the instruction is the escape
          hatch beside it, not instead of it. */}
      <div className="grid gap-3 sm:grid-cols-[180px_minmax(0,1fr)_auto] sm:items-end">
        <Select
          label={ui("listing.tone", locale)}
          value={tone}
          onChange={(e) => setChosenTone(e.target.value as ListingTone)}
          options={TONES.map((id) => ({ value: id, label: ui(`listing.tone.${id}`, locale) }))}
        />
        <Input
          label={ui("listing.instructions", locale)}
          placeholder={ui("listing.instructions.placeholder", locale)}
          value={instructions}
          maxLength={500}
          onChange={(e) => setInstructions(e.target.value)}
        />
        <div className="flex gap-2">
          <Button variant="primary" loading={writing} onClick={() => void write(false)}>
            <Wand2 className="size-4" aria-hidden />
            {copy ? ui("listing.rewrite", locale) : ui("listing.write", locale)}
          </Button>
          {copy ? (
            <Button
              variant="secondary"
              size="icon"
              aria-label={ui("listing.again", locale)}
              title={ui("listing.again", locale)}
              loading={writing}
              onClick={() => void write(true)}
            >
              <RotateCcw className="size-4" aria-hidden />
            </Button>
          ) : null}
        </div>
      </div>

      {modelNotes.length > 0 ? (
        <p className="mt-3 text-[12px] leading-relaxed text-ink-faint">
          {modelNotes.map((w) => describeWarning(w, locale)).join(" ")}
        </p>
      ) : null}

      {textWarnings.length > 0 ? (
        <div className="mt-4 rounded-md border border-amber/40 bg-amber/5 p-3.5">
          <p className="flex items-center gap-1.5 text-[12px] font-medium text-amber">
            <AlertTriangle className="size-3.5" aria-hidden />
            {ui("listing.warnings", locale)}
          </p>
          <ul className="mt-1.5 space-y-1 text-[12.5px] leading-relaxed text-ink-muted">
            {textWarnings.map((w) => (
              <li key={w}>{describeWarning(w, locale)}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
        <div className="min-w-0 space-y-3">
          <Input
            label={ui("listing.field.title", locale)}
            value={title}
            maxLength={200}
            hint={`${title.length}/65`}
            onChange={(e) => edit({ title: e.target.value })}
          />

          {copy && copy.alt_titles.length > 0 ? (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[11.5px] text-ink-faint">{ui("listing.alts", locale)}</span>
              {copy.alt_titles.map((alt) => (
                <button
                  key={alt}
                  type="button"
                  onClick={() => edit({ title: alt })}
                  className="rounded-full border border-line px-2.5 py-1 text-[11.5px] text-ink-muted transition-colors hover:border-bio/40 hover:text-ink"
                >
                  <PencilLine className="mr-1 inline size-3" aria-hidden />
                  {alt}
                </button>
              ))}
            </div>
          ) : null}

          <Textarea
            label={ui("listing.field.description", locale)}
            rows={8}
            value={description}
            hint={`${words} ${ui("listing.words", locale)}`}
            onChange={(e) => edit({ description: e.target.value })}
          />

          <div className="flex flex-wrap items-center gap-2">
            <Button variant="primary" loading={saving} disabled={!dirty || !title.trim()} onClick={() => void save()}>
              <Check className="size-4" aria-hidden />
              {ui("listing.save", locale)}
            </Button>
            <CopyButton
              value={`${title}\n\n${description}`}
              label={ui("listing.copy", locale)}
              copiedLabel={ui("listing.copied", locale)}
            />
            {!dirty ? (
              <span className="text-[11.5px] text-ink-faint">{ui("listing.upToDate", locale)}</span>
            ) : null}
          </div>
        </div>

        <aside className="space-y-3">
          {copy && copy.highlights.length > 0 ? (
            <div className="rounded-md border border-line bg-surface-2/40 p-3.5">
              <p className="text-[11.5px] font-medium text-ink-muted">
                {ui("listing.highlights", locale)}
              </p>
              <ul className="mt-2 space-y-1.5 text-[12.5px] text-ink">
                {copy.highlights.map((h) => (
                  <li key={h} className="flex gap-1.5">
                    <span className="text-bio" aria-hidden>
                      ·
                    </span>
                    {h}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {copy ? <FactsStrip copy={copy} locale={locale} /> : null}
        </aside>
      </div>
    </section>
  );
}

/** The numbers the copy was written from, and the sentence to read when there are none. */
function FactsStrip({ copy, locale }: { copy: ListingCopy; locale: Locale }) {
  const f = copy.facts;
  const rows: Array<[string, string]> = [
    [ui("listing.facts.rooms", locale), String(f.room_count)],
    [
      ui("listing.facts.measured", locale),
      `${f.measured_room_count}/${f.room_count}`,
    ],
  ];
  if (f.total_area_m2 !== null) {
    rows.push([ui("listing.facts.area", locale), `~${f.total_area_m2} m²`]);
  }
  if (f.ceiling_height_m !== null) {
    rows.push([ui("listing.facts.ceiling", locale), `~${f.ceiling_height_m} m`]);
  }

  return (
    <div className="rounded-md border border-line bg-surface-2/40 p-3.5">
      <p className="text-[11.5px] font-medium text-ink-muted">{ui("listing.facts", locale)}</p>
      <dl className="mt-2 space-y-1 text-[12px]">
        {rows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-3">
            <dt className="text-ink-faint">{label}</dt>
            <dd className="text-ink">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2.5 text-[11px] leading-relaxed text-ink-faint">
        {f.measured_room_count === 0
          ? ui("listing.facts.unmeasured", locale)
          : ui("listing.facts.note", locale)}
      </p>
    </div>
  );
}
