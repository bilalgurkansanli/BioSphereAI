"use client";

/**
 * The investment assistant, in the player's bottom-left corner.
 *
 * ── THE CARD COMES FIRST, AND NEEDS NO MODEL ──────────────────────────────────────────
 * Opening the panel shows the engine's own figures for this flat — gross yield, amortisman,
 * price per m², an example mortgage payment — computed from the agent's numbers with no model
 * call. Most visitors want exactly those four; the chat underneath is for the next question
 * ("bölgedeki kiralar ne kadar?", "%30 peşinatla taksit?"). A deployment without an API key,
 * or a listing whose daily budget is spent, still shows the card and says why the chat is off.
 *
 * ── THE SEGMENT EXISTS ONLY WHERE IT CAN WORK ─────────────────────────────────────────
 * It replaced a "yakında" teaser that was on every player. Now `useInvestmentAvailability`
 * asks the engine once per listing, and the segment appears only when the agent has opted in
 * and entered a price — a visitor never meets a button that opens nothing.
 *
 * ── TYPING MUST NOT DRIVE THE PLAYER ──────────────────────────────────────────────────
 * The player listens for keys on its container — "f" is fullscreen, "w a s d" look around, and
 * it calls `preventDefault`, which would eat those letters out of a question. React's
 * `stopPropagation` runs too late for a native listener on an ancestor, so the panel stops
 * key events natively, at its own root, before they reach the player. Escape is handled there
 * too, which is why it still closes the panel while the cursor is in the text box.
 *
 * ── WHAT THE PANEL PROMISES ──────────────────────────────────────────────────────────
 * One plain line: estimates, not investment advice; and one for privacy: questions go to a
 * hosted model in the USA. Both are the panel's, not the model's, so no answer has to repeat
 * a disclaimer.
 */

import { AnimatePresence, motion } from "motion/react";
import { ArrowUp, ExternalLink, RotateCcw, Square, TrendingUp, X } from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { KpiCard } from "@/components/invest/KpiCard";
import { MarkdownLite } from "@/components/invest/MarkdownLite";
import { fill } from "@/components/invest/format";
import { hostOf } from "@/components/invest/markdown";
import { useAdvisorChat, type ChatTurn } from "@/components/invest/useAdvisorChat";
import { api } from "@/lib/api";
import { t } from "@/lib/i18n";
import type { AdvisorPublicState, Locale } from "@/lib/types";
import { cn } from "@/lib/utils";

const EASE = [0.16, 1, 0.3, 1] as const;

// ---------------------------------------------------------------------------------
// Availability
// ---------------------------------------------------------------------------------

export interface InvestmentAvailability {
  state: AdvisorPublicState | null;
  /** True only when the agent opted in and the listing has a price. */
  enabled: boolean;
  failed: boolean;
  retry: () => void;
}

/**
 * Whether this listing offers the assistant, and its card. One cheap request per listing and
 * locale, no model; `null` token (a player in the dashboard) asks nothing and offers nothing.
 */
export function useInvestmentAvailability(token: string | null, locale: Locale): InvestmentAvailability {
  const [attempt, setAttempt] = useState(0);
  // Each answer is filed under the request it answers, and only the current request's answer
  // is read: a token or locale change shows nothing stale, with no state reset in an effect.
  const key = token ? `${token}|${locale}|${attempt}` : null;
  const [result, setResult] = useState<{ key: string; state: AdvisorPublicState | null; failed: boolean } | null>(null);

  useEffect(() => {
    if (!token || !key) return;
    const controller = new AbortController();
    api.public
      .advisor(token, locale, controller.signal)
      .then((state) => setResult({ key, state, failed: false }))
      .catch(() => {
        if (!controller.signal.aborted) setResult({ key, state: null, failed: true });
      });
    return () => controller.abort();
  }, [token, locale, key]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  const current = result && result.key === key ? result : null;
  return {
    state: current?.state ?? null,
    enabled: Boolean(current?.state?.enabled),
    failed: current?.failed ?? false,
    retry,
  };
}

// ---------------------------------------------------------------------------------
// The panel
// ---------------------------------------------------------------------------------

export interface InvestmentChatProps {
  locale: Locale;
  token: string;
  state: AdvisorPublicState;
  onClose: () => void;
}

export function InvestmentChat({ locale, token, state, onClose }: InvestmentChatProps) {
  const root = useRef<HTMLDivElement | null>(null);
  const scroller = useRef<HTMLDivElement | null>(null);
  const input = useRef<HTMLTextAreaElement | null>(null);
  const [draft, setDraft] = useState("");
  const chat = useAdvisorChat(token, locale, state.max_turns);
  const pinned = useRef(true);

  // Keys stop at the panel (see the header); Escape closes it from anywhere inside.
  useEffect(() => {
    const element = root.current;
    if (!element) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
      event.stopPropagation();
    };
    const stop = (event: Event) => event.stopPropagation();
    element.addEventListener("keydown", onKey);
    element.addEventListener("keyup", stop);
    return () => {
      element.removeEventListener("keydown", onKey);
      element.removeEventListener("keyup", stop);
    };
  }, [onClose]);

  // Escape with the focus outside the panel (on the photograph) closes it too.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Follow the answer as it streams — unless the visitor scrolled up to read something.
  const last = chat.turns[chat.turns.length - 1];
  useLayoutEffect(() => {
    const element = scroller.current;
    if (element && pinned.current) element.scrollTop = element.scrollHeight;
  }, [last?.answer, last?.status, last?.calcs.length, chat.turns.length]);

  const onScroll = () => {
    const element = scroller.current;
    if (!element) return;
    pinned.current = element.scrollHeight - element.scrollTop - element.clientHeight < 48;
  };

  const submit = (text: string) => {
    if (!text.trim() || chat.busy) return;
    pinned.current = true;
    chat.send(text.slice(0, state.max_message_chars));
    setDraft("");
  };

  const offline = !state.chat_available;
  const offlineCopy =
    state.reason === "budget" ? t("player.invest.offline.budget", locale) : t("player.invest.offline.no_key", locale);

  return (
    <motion.div
      key="invest"
      ref={root}
      data-player-ui
      initial={{ opacity: 0, y: 8, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 6, scale: 0.99 }}
      transition={{ duration: 0.24, ease: EASE }}
      style={{ transformOrigin: "bottom left" }}
      role="dialog"
      aria-label={t("player.invest.title", locale)}
      // The height leaves room for the bar under it and the toolbar the corner lifts above on a
      // phone (`PlayerDock`), and `dvh` follows the visible viewport when the keyboard opens.
      className="glass-strong pointer-events-auto flex max-h-[min(calc(100dvh-9.5rem),42rem)] w-[min(27rem,calc(100vw-1.5rem))]
                 flex-col overflow-hidden rounded-[var(--radius-lg)] shadow-[var(--shadow-lift)]"
    >
      <header className="flex items-center justify-between gap-3 border-b border-line px-3.5 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <span className="grid size-6 shrink-0 place-items-center rounded-full border border-line text-bio">
            <TrendingUp size={13} strokeWidth={1.9} aria-hidden />
          </span>
          <span className="truncate text-[13px] font-semibold tracking-tight text-ink">
            {t("player.invest.title", locale)}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {chat.turns.length > 0 ? (
            <button
              type="button"
              onClick={chat.reset}
              aria-label={t("player.invest.newChat", locale)}
              title={t("player.invest.newChat", locale)}
              className="grid size-6 place-items-center rounded-full text-ink-faint transition-colors hover:bg-white/[0.06] hover:text-ink"
            >
              <RotateCcw size={12} aria-hidden />
            </button>
          ) : null}
          <button
            type="button"
            onClick={onClose}
            aria-label={t("player.invest.close", locale)}
            className="grid size-6 place-items-center rounded-full text-ink-faint transition-colors hover:bg-white/[0.06] hover:text-ink"
          >
            <X size={13} aria-hidden />
          </button>
        </div>
      </header>

      <div
        ref={scroller}
        onScroll={onScroll}
        className="min-h-0 flex-1 space-y-3 overflow-y-auto overscroll-contain px-3.5 py-3"
        aria-live="polite"
      >
        {state.card ? <KpiCard card={state.card} locale={locale} /> : null}
        <p className="text-[10.5px] leading-snug text-ink-faint">{t("player.invest.note", locale)}</p>

        {offline ? (
          <p className="rounded-[var(--radius-sm)] border border-line px-2.5 py-2 text-[11.5px] leading-relaxed text-ink-muted">
            {offlineCopy}
          </p>
        ) : chat.turns.length === 0 ? (
          <div className="space-y-2">
            <p className="text-[12px] leading-relaxed text-ink-muted">{t("player.invest.body", locale)}</p>
            {state.suggestions.length ? (
              <div className="flex flex-wrap gap-1.5" aria-label={t("player.invest.suggestions", locale)}>
                {state.suggestions.map((question) => (
                  <button
                    key={question}
                    type="button"
                    onClick={() => submit(question)}
                    className="rounded-full border border-line px-2.5 py-1 text-left text-[11.5px] leading-snug text-ink-muted
                               transition-colors hover:border-bio/40 hover:text-ink"
                  >
                    {question}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        ) : (
          chat.turns.map((turn) => <Turn key={turn.id} turn={turn} locale={locale} />)
        )}
      </div>

      {offline ? null : (
        <footer className="border-t border-line px-3 pb-2.5 pt-2">
          {chat.exhausted ? (
            <div className="flex items-center justify-between gap-2 py-1">
              <p className="text-[11.5px] leading-snug text-ink-muted">{t("player.invest.limitReached", locale)}</p>
              <button
                type="button"
                onClick={chat.reset}
                className="shrink-0 rounded-full border border-line px-2.5 py-1 text-[11.5px] text-ink-muted hover:border-bio/40 hover:text-ink"
              >
                {t("player.invest.newChat", locale)}
              </button>
            </div>
          ) : (
            <form
              className="flex items-end gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                submit(draft);
              }}
            >
              <textarea
                ref={input}
                value={draft}
                rows={1}
                maxLength={state.max_message_chars}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault();
                    submit(draft);
                  }
                }}
                placeholder={t("player.invest.placeholder", locale)}
                aria-label={t("player.invest.placeholder", locale)}
                className="max-h-28 min-h-[2.25rem] flex-1 resize-none rounded-[var(--radius-sm)] border border-line bg-white/[0.04]
                           px-2.5 py-2 text-[12.5px] leading-snug text-ink outline-none placeholder:text-ink-faint
                           focus:border-bio/50 focus:ring-2 focus:ring-bio/15"
              />
              {chat.busy ? (
                <button
                  type="button"
                  onClick={chat.stop}
                  aria-label={t("player.invest.stop", locale)}
                  title={t("player.invest.stop", locale)}
                  className="grid size-9 shrink-0 place-items-center rounded-full border border-line text-ink-muted hover:text-ink"
                >
                  <Square size={12} aria-hidden />
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!draft.trim()}
                  aria-label={t("player.invest.send", locale)}
                  title={t("player.invest.send", locale)}
                  className="grid size-9 shrink-0 place-items-center rounded-full bg-bio text-canvas transition-opacity
                             disabled:opacity-35"
                >
                  <ArrowUp size={15} aria-hidden />
                </button>
              )}
            </form>
          )}
          <div className="mt-1.5 flex items-start justify-between gap-3 text-[10px] leading-snug text-ink-faint">
            <span>{t("player.invest.privacy", locale)}</span>
            {chat.turnsLeft !== null && chat.turnsLeft <= 3 && chat.turns.length > 0 && !chat.exhausted ? (
              <span className="shrink-0">{fill(t("player.invest.turnsLeft", locale), { n: chat.turnsLeft })}</span>
            ) : null}
          </div>
        </footer>
      )}
    </motion.div>
  );
}

// ---------------------------------------------------------------------------------
// One question and its answer
// ---------------------------------------------------------------------------------

function statusLine(turn: ChatTurn, locale: Locale): string | null {
  const status = turn.status;
  if (!status) return null;
  if (status.phase === "searching") {
    return status.query
      ? fill(t("player.invest.status.searchingQuery", locale), { query: status.query })
      : t("player.invest.status.searching", locale);
  }
  if (status.phase === "computing") return t("player.invest.status.computing", locale);
  if (status.phase === "writing") return turn.answer ? null : t("player.invest.status.writing", locale);
  return t("player.invest.status.thinking", locale);
}

/** Sources the answer links in its text, plus those the API cited, without repeats. */
function sourcesOf(turn: ChatTurn): { url: string; label: string }[] {
  const seen = new Map<string, string>();
  for (const match of turn.answer.matchAll(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g)) {
    if (!seen.has(match[2])) seen.set(match[2], match[1]);
  }
  for (const source of turn.sources) {
    if (!seen.has(source.url)) seen.set(source.url, source.title);
  }
  return [...seen].map(([url, label]) => ({ url, label }));
}

function Turn({ turn, locale }: { turn: ChatTurn; locale: Locale }) {
  const status = statusLine(turn, locale);
  const sources = turn.done ? sourcesOf(turn) : [];
  return (
    <div className="space-y-2">
      <p className="ml-auto w-fit max-w-[85%] rounded-[var(--radius-sm)] rounded-br-sm bg-bio/12 px-2.5 py-1.5 text-[12.5px] leading-snug text-ink">
        {turn.question}
      </p>
      {turn.answer ? <MarkdownLite text={turn.answer} /> : null}
      <AnimatePresence initial={false}>
        {status ? (
          <motion.p
            key={status}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-1.5 text-[11.5px] text-ink-faint"
          >
            <span aria-hidden className="animate-breathe inline-block size-1.5 rounded-full bg-bio" />
            {status}
          </motion.p>
        ) : null}
      </AnimatePresence>
      {sources.length ? (
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-[10.5px] text-ink-faint">{t("player.invest.sources", locale)}:</span>
          {sources.map((source) => (
            <a
              key={source.url}
              href={source.url}
              target="_blank"
              rel="noopener noreferrer nofollow"
              title={source.label}
              className="inline-flex max-w-[12rem] items-center gap-1 truncate rounded-full border border-line px-2 py-0.5
                         text-[10.5px] text-ink-muted transition-colors hover:border-sky/40 hover:text-sky"
            >
              <ExternalLink size={9} aria-hidden className="shrink-0" />
              <span className="truncate">{hostOf(source.url)}</span>
            </a>
          ))}
        </div>
      ) : null}
      {turn.done && turn.calcs.some((c) => c.rows.length) ? (
        <details className="group text-[11px] text-ink-faint">
          <summary className="cursor-pointer select-none hover:text-ink-muted">{t("player.invest.calcs", locale)}</summary>
          <div className="mt-1.5 space-y-1.5">
            {turn.calcs
              .filter((c) => c.rows.length)
              .map((calc, index) => (
                <dl key={`${calc.tool}-${index}`} className="rounded-[var(--radius-xs)] border border-line px-2 py-1.5">
                  <dt className="mb-0.5 font-medium text-ink-muted">{calc.title}</dt>
                  {calc.rows.map(([label, value]) => (
                    <dd key={label} className="flex justify-between gap-3">
                      <span>{label}</span>
                      <span className="tabular-nums text-ink-muted">{value}</span>
                    </dd>
                  ))}
                </dl>
              ))}
          </div>
        </details>
      ) : null}
      {turn.notice ? <p className={cn("text-[11.5px] leading-snug text-ink-faint")}>{turn.notice}</p> : null}
      {turn.error ? <p className="text-[11.5px] leading-snug text-amber">{turn.error}</p> : null}
    </div>
  );
}
