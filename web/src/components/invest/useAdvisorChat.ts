"use client";

/**
 * One visitor's conversation with the investment assistant, as React state.
 *
 * The engine streams an answer as events (see `engine/biosphere/invest/advisor.py:run_turn`);
 * this hook folds them into turns the panel draws. Three of the events rewrite what is already
 * on screen, and each exists for a measured reason:
 *
 * `retract`  — text written before a tool call is narration ("Hemen araştırıyorum."), streamed
 *              so the visitor sees life and then withdrawn back to the last `round` mark,
 *              because left in place it ran into the real answer.
 * `discard`  — a refusal: everything this answer showed is withdrawn, as the API requires.
 * `done`     — carries how many questions the conversation has left.
 *
 * The conversation id comes from the engine's first event. When the engine has forgotten it
 * (an hour idle, or a restart — conversations live in its memory only) the next question is
 * refused with `conversation_expired`, and the hook starts a new conversation and asks again,
 * once, so the visitor sees an answer rather than an apology.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { BioSphereError, api } from "@/lib/api";
import { t } from "@/lib/i18n";
import type { AdvisorEvent, Locale } from "@/lib/types";
import { fill } from "./format";

export interface ChatSource {
  url: string;
  title: string;
  date?: string | null;
}

export interface ChatCalc {
  tool: string;
  title: string;
  rows: [string, string][];
  error?: boolean;
}

export interface ChatTurn {
  id: number;
  question: string;
  answer: string;
  /** What the model is doing right now; null once the answer is complete. */
  status: Extract<AdvisorEvent, { type: "status" }> | null;
  sources: ChatSource[];
  calcs: ChatCalc[];
  notice: string | null;
  error: string | null;
  done: boolean;
}

export interface AdvisorChat {
  turns: ChatTurn[];
  busy: boolean;
  turnsLeft: number | null;
  /** The conversation cannot take more questions; a new one can be started. */
  exhausted: boolean;
  send: (question: string) => void;
  stop: () => void;
  reset: () => void;
}

function describeError(cause: unknown, locale: Locale): string {
  if (cause instanceof BioSphereError) {
    if (cause.detail.code === "rate_limited") {
      const seconds = /retry_after=(\d+)/.exec(cause.detail.hint ?? "")?.[1];
      const base = cause.localised(locale);
      return seconds ? `${base} ${fill(t("player.invest.retryIn", locale), { s: seconds })}` : base;
    }
    return cause.localised(locale);
  }
  return locale === "tr" ? "Asistana ulaşılamadı. Tekrar deneyin." : "The assistant could not be reached. Try again.";
}

export function useAdvisorChat(token: string, locale: Locale, maxTurns: number): AdvisorChat {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [busy, setBusy] = useState(false);
  const [turnsLeft, setTurnsLeft] = useState<number | null>(null);
  const [exhausted, setExhausted] = useState(false);
  const conversation = useRef<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const nextId = useRef(1);

  useEffect(() => () => controller.current?.abort(), []);

  const patch = useCallback((id: number, change: (turn: ChatTurn) => ChatTurn) => {
    setTurns((all) => all.map((turn) => (turn.id === id ? change(turn) : turn)));
  }, []);

  const send = useCallback(
    (raw: string) => {
      const question = raw.trim();
      if (!question || busy || exhausted) return;
      const id = nextId.current++;
      setTurns((all) => [
        ...all,
        { id, question, answer: "", status: { type: "status", phase: "thinking" }, sources: [], calcs: [], notice: null, error: null, done: false },
      ]);
      setBusy(true);
      const abort = new AbortController();
      controller.current = abort;
      // Where the current call's text began, so a `retract` can cut back to it.
      let roundStart = 0;
      let answer = "";

      const onEvent = (event: AdvisorEvent) => {
        switch (event.type) {
          case "meta":
            conversation.current = event.conversation_id;
            break;
          case "round":
            roundStart = answer.length;
            break;
          case "status":
            patch(id, (turn) => ({ ...turn, status: event }));
            break;
          case "text":
            answer += event.delta;
            patch(id, (turn) => ({ ...turn, answer, status: { type: "status", phase: "writing" } }));
            break;
          case "retract":
            answer = answer.slice(0, roundStart);
            patch(id, (turn) => ({ ...turn, answer }));
            break;
          case "discard":
            answer = "";
            roundStart = 0;
            patch(id, (turn) => ({ ...turn, answer, sources: [], calcs: [] }));
            break;
          case "cite":
            patch(id, (turn) =>
              turn.sources.some((s) => s.url === event.url)
                ? turn
                : { ...turn, sources: [...turn.sources, { url: event.url, title: event.title, date: event.date }] },
            );
            break;
          case "calc":
            patch(id, (turn) => ({ ...turn, calcs: [...turn.calcs, { tool: event.tool, title: event.title, rows: event.rows, error: event.error }] }));
            break;
          case "notice":
            patch(id, (turn) => ({ ...turn, notice: event.message }));
            break;
          case "error":
            patch(id, (turn) => ({ ...turn, error: event.message, status: null }));
            break;
          case "done":
            setTurnsLeft(event.turns_left);
            if (event.turns_left <= 0 || event.full) setExhausted(true);
            patch(id, (turn) => ({ ...turn, status: null, done: true }));
            break;
        }
      };

      const ask = (conversationId: string | null) =>
        api.public.advisorChat(token, { message: question, conversation_id: conversationId, locale }, onEvent, abort.signal);

      void (async () => {
        try {
          try {
            await ask(conversation.current);
          } catch (cause) {
            const expired = cause instanceof BioSphereError && cause.detail.code === "conversation_expired";
            if (!expired) throw cause;
            conversation.current = null;
            setTurnsLeft(null);
            await ask(null);
          }
        } catch (cause) {
          if (abort.signal.aborted) {
            patch(id, (turn) => ({ ...turn, status: null, notice: t("player.invest.stopped", locale) }));
          } else {
            const limit = cause instanceof BioSphereError && cause.detail.code === "conversation_limit";
            if (limit) setExhausted(true);
            patch(id, (turn) => ({ ...turn, status: null, error: describeError(cause, locale) }));
          }
        } finally {
          patch(id, (turn) => (turn.status ? { ...turn, status: null } : turn));
          setBusy(false);
          if (controller.current === abort) controller.current = null;
        }
      })();
    },
    [busy, exhausted, locale, patch, token],
  );

  const stop = useCallback(() => controller.current?.abort(), []);

  const reset = useCallback(() => {
    controller.current?.abort();
    conversation.current = null;
    setTurns([]);
    setTurnsLeft(null);
    setExhausted(false);
  }, []);

  return { turns, busy, turnsLeft: turnsLeft ?? (turns.length === 0 ? maxTurns : null), exhausted, send, stop, reset };
}
