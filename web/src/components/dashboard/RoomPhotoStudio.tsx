"use client";

/**
 * The two AI tools, on one of a room's photographs.
 *
 * ── WHY IT IS A COMPARISON AND NOT A GALLERY ──────────────────────────────────────────
 * Both tools answer a question the user has to JUDGE — "is that still my room?" for a
 * decorating style, "did that leave a mess?" for the furniture removal — and neither is
 * judgeable from the result alone. The engine is honest about its limits in numbers
 * (`protected_fraction`, `removed_fraction`) and those numbers are shown, but a number is a
 * summary and the eye is the instrument. So the result never appears on its own: it appears
 * wiped across the original, on the same slider `/dekor` uses.
 *
 * That is especially load-bearing for the emptying tool. What was behind the sofa was never
 * photographed, so it is reconstructed, and on a crowded frame the reconstruction is soft —
 * `ai.empty_photo` records exactly how soft, measured. A user who can drag the original back
 * over it decides for themselves whether to publish it; a user handed only the output cannot.
 *
 * ── THE RAIL ASKS FOR ONE THING AT A TIME ─────────────────────────────────────────────
 * The server runs one model pass at a time on one card, so firing all six at once would not be
 * faster — it would be the same wait with the answers arriving in an order nobody chose. The
 * rail therefore requests only what is selected, and remembers everything that comes back.
 */

import { AlertCircle, Download, Loader2, RefreshCw, Sofa, Sparkles } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { roomObject, ui } from "./copy";
import { BeforeAfterSlider } from "@/components/restyle/BeforeAfterSlider";
import { BetaTag } from "@/components/ui/BetaTag";
import { Dialog } from "@/components/ui/Dialog";
import { BioSphereError, absoluteUrl, api } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import type { Locale, RestyleStyle, RestyleStyleId, ScenePhoto, ScenePhotoVariant } from "@/lib/types";
import { cn } from "@/lib/utils";

/** The variant name the engine uses for "the furniture is gone". */
const EMPTY = "empty";

interface StudioState {
  /** Which photograph every other field belongs to. */
  photoId: string | null;
  variants: Record<string, ScenePhotoVariant>;
  selected: string | null;
  pending: string | null;
  failed: string | null;
}

const empty = (photoId: string | null): StudioState => ({
  photoId,
  variants: {},
  selected: null,
  pending: null,
  failed: null,
});

export interface RoomPhotoStudioProps {
  photo: ScenePhoto | null;
  styles: RestyleStyle[];
  onClose: () => void;
}

export function RoomPhotoStudio({ photo, styles, onClose }: RoomPhotoStudioProps) {
  const { locale } = useLocale();

  /**
   * EVERY PIECE OF STATE CARRIES THE PHOTOGRAPH IT BELONGS TO, in one object rather than four.
   *
   * Opening a different photograph has to forget the last one's styles, selection and error,
   * and the obvious way — an effect that clears four `useState`s when `photo` changes — is
   * both a lint error and a real hazard: `setState` called synchronously from an effect is a
   * cascading render, and here it would run in a component that has an image decoding in it.
   * Storing the id alongside makes the reset free, which is the same trick `useEmptyRoom`
   * uses for the scene it belongs to.
   */
  const [state, setState] = useState<StudioState>(() => empty(null));
  const current = photo && state.photoId === photo.id ? state : empty(photo?.id ?? null);
  const { variants, selected, pending, failed } = current;

  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortRef.current?.abort(), []);

  /** What has already been rendered, asked once when a photograph is opened. */
  useEffect(() => {
    if (!photo) return;
    let cancelled = false;
    void api.photos
      .variants(photo.id)
      .then((rows) => {
        if (cancelled) return;
        // Merged rather than assigned: a pass may have finished while this was in flight, and
        // the answer to "what is on disk" must not overwrite the answer to "what I just made".
        setState((previous) => {
          const base = previous.photoId === photo.id ? previous : empty(photo.id);
          return {
            ...base,
            photoId: photo.id,
            variants: { ...Object.fromEntries(rows.map((r) => [r.variant, r])), ...base.variants },
          };
        });
      })
      .catch(() => {
        /* Nothing rendered yet is indistinguishable from a failed lookup here, and both mean
           the same thing to the user: press a button to make one. */
      });
    return () => {
      cancelled = true;
    };
  }, [photo]);

  const request = useCallback(
    async (variant: string) => {
      if (!photo || pending) return;
      if (variants[variant]) {
        setState((previous) => ({ ...previous, selected: variant }));
        return;
      }
      const controller = new AbortController();
      abortRef.current?.abort();
      abortRef.current = controller;
      setState((previous) => ({
        ...(previous.photoId === photo.id ? previous : empty(photo.id)),
        photoId: photo.id,
        pending: variant,
        failed: null,
      }));
      try {
        const row =
          variant === EMPTY
            ? await api.photos.empty(photo.id, {}, controller.signal)
            : await api.photos.restyle(
                photo.id,
                { style: variant as RestyleStyleId },
                controller.signal,
              );
        if (controller.signal.aborted) return;
        setState((previous) => ({
          ...previous,
          variants: { ...previous.variants, [variant]: row },
          selected: variant,
        }));
      } catch (error) {
        if (controller.signal.aborted) return;
        setState((previous) => ({
          ...previous,
          failed:
            error instanceof BioSphereError
              ? error.localised(locale)
              : ui("photos.failed", locale),
        }));
      } finally {
        setState((previous) =>
          previous.pending === variant ? { ...previous, pending: null } : previous,
        );
      }
    },
    [photo, pending, variants, locale],
  );

  /**
   * Another take of the style on screen. Each take is its own repeatable seed on the server and
   * overwrites that style's file, so the take number rides on the URL or the browser would keep
   * showing the previous picture from its cache.
   */
  const retake = useCallback(
    async (variant: string) => {
      if (!photo || pending || variant === EMPTY) return;
      const previous = variants[variant];
      const next = (previous?.variation ?? 0) + 1;
      const controller = new AbortController();
      abortRef.current?.abort();
      abortRef.current = controller;
      setState((current) => ({ ...current, pending: variant, failed: null }));
      try {
        const row = await api.photos.restyle(
          photo.id,
          { style: variant as RestyleStyleId, variation: next },
          controller.signal,
        );
        if (controller.signal.aborted) return;
        setState((current) => ({
          ...current,
          variants: { ...current.variants, [variant]: row },
          selected: variant,
        }));
      } catch (error) {
        if (controller.signal.aborted) return;
        setState((current) => ({
          ...current,
          failed:
            error instanceof BioSphereError ? error.localised(locale) : ui("photos.failed", locale),
        }));
      } finally {
        setState((current) => (current.pending === variant ? { ...current, pending: null } : current));
      }
    },
    [photo, pending, variants, locale],
  );

  if (!photo) return null;

  const originalUrl = absoluteUrl(photo.url);
  const active = selected ? (variants[selected] ?? null) : null;
  const activeUrl = active
    ? (() => {
        const url = absoluteUrl(active.url);
        const take = active.variation ?? 0;
        return url && take > 0 ? `${url}${url.includes("?") ? "&" : "?"}v=${take}` : url;
      })()
    : null;

  return (
    <Dialog
      open
      onClose={onClose}
      title={ui("photos.studio.title", locale)}
      size="lg"
      closeLabel={ui("common.close", locale)}
      footer={
        activeUrl ? (
          <div className="flex items-center gap-2">
            {active && active.kind === "style" ? (
              <button
                type="button"
                onClick={() => void retake(active.variant)}
                disabled={pending !== null}
                title={ui("decor.variation.hint", locale)}
                className="inline-flex h-9 items-center gap-1.5 rounded-md border border-line px-3 text-[13px]
                           font-medium text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink
                           disabled:opacity-45"
              >
                <RefreshCw className="size-3.5" aria-hidden />
                {ui("decor.variation", locale)}
              </button>
            ) : null}
            <a
              href={activeUrl}
              download
              className="inline-flex h-9 items-center gap-1.5 rounded-md border border-line-strong px-3
                         text-[13px] font-medium text-ink transition-colors hover:bg-surface-2"
            >
              <Download className="size-3.5" aria-hidden />
              {ui("decor.download", locale)}
            </a>
          </div>
        ) : null
      }
    >
      <div className="space-y-3">
        {originalUrl ? (
          <BeforeAfterSlider
            beforeSrc={originalUrl}
            afterSrc={activeUrl}
            beforeLabel={ui("photos.studio.original", locale)}
            afterLabel={ui("decor.generated", locale)}
            ariaLabel={ui("decor.slider", locale)}
            aspectRatio={photo.height > 0 ? photo.width / photo.height : 4 / 3}
            pending={
              pending ? (
                <span className="flex items-center gap-2 rounded-full bg-surface-2/90 px-3 py-1.5 text-[13px] text-ink-muted">
                  <Loader2 className="size-3.5 animate-spin" aria-hidden />
                  {ui("photos.studio.working", locale)}
                </span>
              ) : null
            }
          />
        ) : null}

        <div className="flex flex-wrap items-center gap-1.5">
          {/* On the rail rather than in the dialog's title: the title is "edit a photo", which
              is not a beta idea — the six buttons under it are. */}
          <BetaTag title={ui("beta.hint", locale)} className="mr-0.5" />
          <Chip
            label={ui("photos.studio.original", locale)}
            active={selected === null}
            disabled={pending !== null}
            onClick={() => setState((previous) => ({ ...previous, selected: null }))}
          />
          {styles.map((style) => (
            <Chip
              key={style.id}
              label={locale === "tr" ? style.label_tr : style.label_en}
              icon={<Sparkles className="size-3" aria-hidden />}
              active={selected === style.id}
              ready={Boolean(variants[style.id])}
              busy={pending === style.id}
              disabled={pending !== null && pending !== style.id}
              onClick={() => void request(style.id)}
            />
          ))}
          <Chip
            label={ui("photos.studio.empty", locale)}
            icon={<Sofa className="size-3" aria-hidden />}
            active={selected === EMPTY}
            ready={Boolean(variants[EMPTY])}
            busy={pending === EMPTY}
            disabled={pending !== null && pending !== EMPTY}
            onClick={() => void request(EMPTY)}
          />
        </div>

        {/* A FIXED FLOOR UNDER THE NOTE, because the note is the last thing on the dialog and
            the rail is directly above it. Its text changes length as the selection changes —
            three lines for the emptying caveat, one for the original — and without a floor the
            rail slides up and down under the pointer between clicks. Found by clicking on a
            chip that had moved. */}
        <div className="min-h-[3.25rem]">
          <Note active={active} failed={failed} locale={locale} />
        </div>
      </div>
    </Dialog>
  );
}

/** What the engine actually did, in the words a user can act on. */
function Note({
  active,
  failed,
  locale,
}: {
  active: ScenePhotoVariant | null;
  failed: string | null;
  locale: Locale;
}) {
  if (failed) {
    return (
      <p className="flex items-start gap-2 text-[12px] text-danger">
        <AlertCircle className="mt-px size-3.5 shrink-0" aria-hidden />
        {failed}
      </p>
    );
  }
  if (!active) {
    return <p className="text-[12px] text-ink-faint">{ui("decor.disclaimer", locale)}</p>;
  }

  if (active.kind === "empty") {
    const removed = Math.round((active.removed_fraction ?? 0) * 100);
    return (
      <p className="text-[12px] leading-relaxed text-ink-faint">
        {removed === 0
          ? ui("photos.studio.nothing", locale)
          : `${ui("photos.studio.removed", locale).replace("{n}", String(removed))} ${ui("photos.studio.soft", locale)}`}
        {active.objects.length > 0 ? (
          <>
            {" "}
            {ui("photos.studio.found", locale)}:{" "}
            {active.objects.map((name) => roomObject(name, locale)).join(", ")}.
          </>
        ) : null}
      </p>
    );
  }

  const kept = Math.round((active.protected_fraction ?? 0) * 100);
  return (
    <p className="text-[12px] leading-relaxed text-ink-faint">
      {ui("decor.disclaimer", locale)}
      {kept > 0 ? <> {ui("photos.studio.kept", locale).replace("{n}", String(kept))}</> : null}
    </p>
  );
}

interface ChipProps {
  label: string;
  icon?: React.ReactNode;
  active?: boolean;
  ready?: boolean;
  busy?: boolean;
  disabled?: boolean;
  onClick: () => void;
}

function Chip({ label, icon, active, ready, busy, disabled, onClick }: ChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      className={cn(
        "flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[12px] font-medium transition-all",
        disabled ? "opacity-45" : "",
        active
          ? "border-bio/50 bg-bio/12 text-bio"
          : "border-line text-ink-muted hover:border-line-strong hover:text-ink",
      )}
    >
      {busy ? <Loader2 className="size-3 animate-spin" aria-hidden /> : icon}
      {label}
      {/* A dot for "already rendered", so a second visit shows what is free to look at. */}
      {ready && !active && !busy ? (
        <span aria-hidden className="size-1.5 rounded-full bg-bio" />
      ) : null}
    </button>
  );
}
