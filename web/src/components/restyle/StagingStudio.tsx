"use client";

/**
 * Virtual staging, end to end: drop a room photograph, get it back in five styles.
 *
 * ── THE FIVE STYLES ARE ASKED FOR ONE AT A TIME, IN THE USER'S OWN ORDER ──────────────
 * The server renders on one GPU behind one lock, so five parallel requests would not be five
 * times faster — they would be the same forty seconds with the results arriving in whatever
 * order the queue felt like, and the one the user is actually looking at possibly last.
 *
 * So the queue lives here and it is re-ordered by attention: whichever style is selected goes
 * to the front. Click Rustic while Modern is still rendering and Rustic is the next request
 * made, not the fourth. The pass already in flight is left alone — cancelling it would waste
 * the seconds already spent and the server would finish and cache it anyway.
 *
 * ── A RENDER IS NEVER THROWN AWAY ─────────────────────────────────────────────────────
 * Results are held per style for the life of the photograph, and the server caches them on
 * disk against the photo's content hash. Switching between two finished styles is a state
 * change and nothing else; reloading the page and dropping the same file in again finds every
 * style already rendered.
 */

import { AlertCircle, Download, ImagePlus, Loader2, RefreshCw, Sparkles } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import { toast } from "sonner";
import { BeforeAfterSlider } from "./BeforeAfterSlider";
import { ui } from "@/components/dashboard/copy";
import { BetaTag } from "@/components/ui/BetaTag";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { BioSphereError, absoluteUrl, api } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import type { Locale, RestylePhoto, RestyleStyle, RestyleStyleId } from "@/lib/types";
import { cn } from "@/lib/utils";

type StyleState =
  | { phase: "idle" }
  | { phase: "queued" }
  | { phase: "working" }
  | { phase: "ready"; url: string; protectedFraction: number; variation: number }
  | { phase: "failed"; message: string };

/**
 * The image URL for one take of a style. A new take overwrites the same file on the server, so
 * the take number rides on the URL — otherwise the browser shows the old picture from its cache
 * and "another take" appears to do nothing.
 */
function takeUrl(url: string, variation: number): string {
  return variation > 0 ? `${url}${url.includes("?") ? "&" : "?"}v=${variation}` : url;
}

const IDLE: StyleState = { phase: "idle" };

function labelOf(style: RestyleStyle, locale: Locale): string {
  return locale === "tr" ? style.label_tr : style.label_en;
}

export function StagingStudio() {
  const { locale } = useLocale();

  const [catalogue, setCatalogue] = useState<RestyleStyle[] | null>(null);
  const [serverReady, setServerReady] = useState<boolean | null>(null);
  const [photo, setPhoto] = useState<RestylePhoto | null>(null);
  const [uploading, setUploading] = useState(false);
  const [active, setActive] = useState<RestyleStyleId | null>(null);
  const [states, setStates] = useState<Record<string, StyleState>>({});

  /** Guards against a queue from a previous photograph writing into the current one. */
  const runIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  /**
   * Read by the queue loop each iteration, so a click re-orders work already scheduled.
   *
   * Synced in an effect rather than assigned during render: writing a ref while rendering is
   * a real hazard under concurrent rendering (a render that is thrown away still leaves its
   * write behind) and the lint rule is right to refuse it. One tick of staleness costs
   * nothing here — the loop reads this between requests, seconds apart.
   */
  const activeRef = useRef<RestyleStyleId | null>(null);
  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  useEffect(() => {
    let cancelled = false;
    api.restyle
      .styles()
      .then((body) => {
        if (cancelled) return;
        setCatalogue(body.styles);
        setServerReady(body.available);
        setActive((current) => current ?? body.styles[0]?.id ?? null);
      })
      .catch(() => {
        if (!cancelled) setServerReady(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const setStyleState = useCallback((id: string, next: StyleState) => {
    setStates((current) => ({ ...current, [id]: next }));
  }, []);

  /**
   * Render every style that is not done yet, newest attention first.
   *
   * Sequential on purpose — see the module docstring. `pending` is recomputed each turn
   * rather than iterated as a fixed list, which is what lets a click during the run change
   * what happens next.
   */
  const runQueue = useCallback(
    async (target: RestylePhoto, ids: RestyleStyleId[], done: Set<string>) => {
      const runId = ++runIdRef.current;
      const controller = new AbortController();
      abortRef.current?.abort();
      abortRef.current = controller;

      const remaining = new Set(ids.filter((id) => !done.has(id)));
      setStates((current) => {
        const next = { ...current };
        for (const id of remaining) next[id] = { phase: "queued" };
        return next;
      });

      while (remaining.size > 0) {
        if (runIdRef.current !== runId) return;
        const preferred = activeRef.current;
        const id =
          preferred && remaining.has(preferred)
            ? preferred
            : (remaining.values().next().value as RestyleStyleId);
        remaining.delete(id);

        setStyleState(id, { phase: "working" });
        try {
          const result = await api.restyle.render(
            { photo_id: target.photo_id, style: id },
            controller.signal,
          );
          if (runIdRef.current !== runId) return;
          const url = absoluteUrl(result.image_url);
          const variation = result.variation ?? 0;
          setStyleState(
            id,
            url
              ? { phase: "ready", url: takeUrl(url, variation), protectedFraction: result.protected_fraction, variation }
              : { phase: "failed", message: ui("decor.failed", locale) },
          );
        } catch (error) {
          if (controller.signal.aborted || runIdRef.current !== runId) return;
          const message =
            error instanceof BioSphereError ? error.localised(locale) : ui("decor.failed", locale);
          setStyleState(id, { phase: "failed", message });
        }
      }
    },
    [locale, setStyleState],
  );

  /**
   * One style, outside the queue: a retry of a failed style, or another take of a finished one.
   *
   * The queue renders each style once; these are the two ways a user asks for a style AGAIN,
   * and before they existed a failed style showed a retry icon that did nothing when pressed.
   * Each take is its own repeatable seed on the server, so take 2 is take 2 tomorrow too.
   */
  const renderOne = useCallback(
    async (id: RestyleStyleId, variation: number) => {
      if (!photo) return;
      setStyleState(id, { phase: "working" });
      try {
        const result = await api.restyle.render({ photo_id: photo.photo_id, style: id, variation });
        const url = absoluteUrl(result.image_url);
        const take = result.variation ?? variation;
        setStyleState(
          id,
          url
            ? { phase: "ready", url: takeUrl(url, take), protectedFraction: result.protected_fraction, variation: take }
            : { phase: "failed", message: ui("decor.failed", locale) },
        );
      } catch (error) {
        const message =
          error instanceof BioSphereError ? error.localised(locale) : ui("decor.failed", locale);
        setStyleState(id, { phase: "failed", message });
      }
    },
    [photo, locale, setStyleState],
  );

  const select = useCallback(
    (id: RestyleStyleId) => {
      setActive(id);
      if (states[id]?.phase === "failed") void renderOne(id, 0);
    },
    [states, renderOne],
  );

  const accept = useCallback(
    async (file: File) => {
      setUploading(true);
      abortRef.current?.abort();
      runIdRef.current += 1;
      setStates({});
      try {
        const uploaded = await api.restyle.upload(file);
        setPhoto(uploaded);
        // `ready_styles` is deliberately not pre-seeded into state: it says WHICH styles the
        // server already holds but not where they are, and the render call answers a cached
        // style off disk in milliseconds. So every style goes through the one queue rather
        // than through a second path that would have to resolve those URLs its own way.
        void runQueue(
          uploaded,
          (catalogue ?? []).map((s) => s.id),
          new Set(),
        );
      } catch (error) {
        const message =
          error instanceof BioSphereError ? error.localised(locale) : ui("decor.failed", locale);
        toast.error(message);
      } finally {
        setUploading(false);
      }
    },
    [catalogue, locale, runQueue],
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    accept: { "image/jpeg": [".jpg", ".jpeg"], "image/png": [".png"], "image/webp": [".webp"] },
    maxFiles: 1,
    multiple: false,
    noClick: true,
    disabled: uploading,
    onDropAccepted: (files) => files[0] && void accept(files[0]),
    onDropRejected: () => toast.error(ui("decor.notimage", locale)),
  });

  const originalUrl = photo ? absoluteUrl(photo.original_url) : null;
  const activeState = active ? (states[active] ?? IDLE) : IDLE;
  const aspect = photo && photo.height > 0 ? photo.width / photo.height : 4 / 3;

  const activeReady = activeState.phase === "ready" ? activeState : null;
  const protectedPercent = useMemo(
    () => (activeReady ? Math.round(activeReady.protectedFraction * 100) : null),
    [activeReady],
  );

  if (serverReady === false && !photo) {
    return (
      <EmptyState
        icon={Sparkles}
        title={ui("decor.unavailable.title", locale)}
        body={ui("decor.unavailable.body", locale)}
      />
    );
  }

  return (
    <div className="flex flex-col gap-5">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-ink">
            {ui("decor.title", locale)}
            <BetaTag title={ui("beta.hint", locale)} />
          </h1>
          <p className="mt-1 max-w-xl text-[13px] text-ink-muted">{ui("decor.subtitle", locale)}</p>
        </div>
        {photo ? (
          <div className="flex gap-2">
            <Button size="sm" variant="secondary" onClick={open} loading={uploading}>
              <ImagePlus className="size-3.5" aria-hidden />
              {ui("decor.another", locale)}
            </Button>
            {activeReady && active ? (
              <Button
                size="sm"
                variant="secondary"
                title={ui("decor.variation.hint", locale)}
                onClick={() => void renderOne(active, activeReady.variation + 1)}
              >
                <RefreshCw className="size-3.5" aria-hidden />
                {ui("decor.variation", locale)}
              </Button>
            ) : null}
            {activeReady ? (
              <a
                href={activeReady.url}
                download
                className="inline-flex h-8 items-center gap-1.5 rounded-md border border-line-strong px-3
                           text-[13px] font-medium text-ink transition-colors hover:bg-surface-2"
              >
                <Download className="size-3.5" aria-hidden />
                {ui("decor.download", locale)}
              </a>
            ) : null}
          </div>
        ) : null}
      </header>

      <div {...getRootProps()} className="contents">
        <input {...getInputProps()} />

        {!photo ? (
          <div
            className={cn(
              "grid place-items-center rounded-lg border border-dashed px-6 py-20 text-center transition-colors",
              isDragActive ? "border-bio bg-bio-dim/40" : "border-line bg-canvas-raised",
            )}
          >
            <div className="flex flex-col items-center gap-3">
              <span className="grid size-12 place-items-center rounded-full bg-surface-2 text-bio">
                {uploading ? (
                  <Loader2 className="size-5 animate-spin" aria-hidden />
                ) : (
                  <ImagePlus className="size-5" aria-hidden />
                )}
              </span>
              <p className="text-[15px] font-medium text-ink">
                {uploading ? ui("decor.uploading", locale) : ui("decor.drop", locale)}
              </p>
              <p className="text-[12px] text-ink-faint">{ui("decor.drop.hint", locale)}</p>
              {!uploading ? (
                <Button size="sm" variant="outline" onClick={open}>
                  {ui("decor.choose", locale)}
                </Button>
              ) : null}
            </div>
          </div>
        ) : (
          <div className="grid gap-5 lg:grid-cols-[180px_minmax(0,1fr)]">
            <StyleRail
              catalogue={catalogue ?? []}
              states={states}
              active={active}
              locale={locale}
              onSelect={select}
            />

            <div className="flex min-w-0 flex-col gap-3">
              {originalUrl ? (
                <BeforeAfterSlider
                  beforeSrc={originalUrl}
                  afterSrc={activeReady?.url ?? null}
                  beforeLabel={ui("decor.original", locale)}
                  afterLabel={ui("decor.generated", locale)}
                  ariaLabel={ui("decor.slider", locale)}
                  aspectRatio={aspect}
                  pending={<PendingBadge state={activeState} locale={locale} />}
                />
              ) : null}

              <p className="text-[12px] leading-relaxed text-ink-faint">
                {ui("decor.disclaimer", locale)}
                {protectedPercent !== null && protectedPercent > 0 ? (
                  <>
                    {" "}
                    {ui("decor.protected", locale).replace("{n}", String(protectedPercent))}
                  </>
                ) : null}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function PendingBadge({ state, locale }: { state: StyleState; locale: Locale }) {
  if (state.phase === "failed") {
    return (
      <span className="flex items-center gap-2 rounded-full bg-danger/12 px-3 py-1.5 text-[13px] text-danger">
        <AlertCircle className="size-3.5" aria-hidden />
        {state.message}
      </span>
    );
  }
  return (
    <span className="flex items-center gap-2 rounded-full bg-surface-2/90 px-3 py-1.5 text-[13px] text-ink-muted">
      <Loader2 className="size-3.5 animate-spin" aria-hidden />
      {state.phase === "queued" ? ui("decor.queued", locale) : ui("decor.working", locale)}
    </span>
  );
}

interface StyleRailProps {
  catalogue: RestyleStyle[];
  states: Record<string, StyleState>;
  active: RestyleStyleId | null;
  locale: Locale;
  onSelect: (id: RestyleStyleId) => void;
}

/**
 * The style list. A column on a wide screen and a scrolling row on a phone — a five-item
 * vertical rail on a 375 px screen would push the picture, which is the product, below the
 * fold.
 */
function StyleRail({ catalogue, states, active, locale, onSelect }: StyleRailProps) {
  return (
    <div
      role="tablist"
      aria-label={ui("decor.styles", locale)}
      className="flex gap-2 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible lg:pb-0"
    >
      <p className="hidden text-[11px] font-medium tracking-wide text-ink-faint uppercase lg:block">
        {ui("decor.styles", locale)}
      </p>
      {catalogue.map((style) => {
        const state = states[style.id] ?? IDLE;
        const selected = style.id === active;
        return (
          <button
            key={style.id}
            role="tab"
            aria-selected={selected}
            title={state.phase === "failed" ? ui("decor.retry", locale) : undefined}
            onClick={() => onSelect(style.id)}
            className={cn(
              "group flex shrink-0 items-center gap-2 rounded-md border px-3 py-2 text-left transition-all",
              "lg:w-full",
              selected
                ? "border-bio bg-bio-dim/50 text-ink"
                : "border-line bg-canvas-raised text-ink-muted hover:border-line-strong hover:text-ink",
            )}
          >
            <StyleDot state={state} />
            <span className="text-[13px] font-medium whitespace-nowrap">{labelOf(style, locale)}</span>
          </button>
        );
      })}
    </div>
  );
}

function StyleDot({ state }: { state: StyleState }) {
  if (state.phase === "working") {
    return <Loader2 className="size-3.5 shrink-0 animate-spin text-bio" aria-hidden />;
  }
  if (state.phase === "failed") {
    return <RefreshCw className="size-3.5 shrink-0 text-danger" aria-hidden />;
  }
  return (
    <span
      aria-hidden
      className={cn(
        "size-2 shrink-0 rounded-full transition-colors",
        state.phase === "ready" ? "bg-bio" : "bg-ink-faint/40",
      )}
    />
  );
}
