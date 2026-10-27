"use client";

import { ArrowRight, Ban, Check, Loader2, RotateCcw, TriangleAlert } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { ui } from "./copy";
import { Button, buttonStyles } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { Progress } from "@/components/ui/Progress";
import { api } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import { stageLabel, t } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";
import { JOB_STAGES, WALK_STAGES, type ApiError, type Job, type QualityMode, type Scene, type Walkthrough } from "@/lib/types";
import { cn, formatDuration } from "@/lib/utils";
import Link from "next/link";

type Phase = "running" | "done" | "error" | "cancelled";

export interface JobProgressProps {
  job: Job;
  propertyId: string;
  /** The scene a stitch job belongs to; a walkthrough job has none. */
  sceneId: string | null;
  /** Reused when the agent retries a failed job. */
  qualityMode: QualityMode;
  autoNavigate?: boolean;
  onDone?: (scene: Scene) => void;
  /** Fires instead of `onDone` when a walkthrough job finishes. */
  onWalkthroughDone?: (walkthrough: Walkthrough) => void;
  className?: string;
}

export function JobProgress({
  job,
  propertyId,
  sceneId,
  qualityMode,
  autoNavigate = true,
  onDone,
  onWalkthroughDone,
  className,
}: JobProgressProps) {
  const { locale } = useLocale();
  const router = useRouter();
  const describeError = useAppStore((s) => s.describeError);

  const [jobId, setJobId] = useState(job.id);
  const [current, setCurrent] = useState<Job>(job);
  const [phase, setPhase] = useState<Phase>("running");
  const [failure, setFailure] = useState<ApiError | null>(null);
  const [lastLog, setLastLog] = useState<string | null>(null);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [retrying, setRetrying] = useState(false);

  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;
  const onWalkDoneRef = useRef(onWalkthroughDone);
  onWalkDoneRef.current = onWalkthroughDone;
  const isWalk = current.kind === "walkthrough";
  const stages: readonly string[] = isWalk ? WALK_STAGES : JOB_STAGES;

  useEffect(() => {
    // `subscribe` owns both the EventSource and its polling fallback; the returned
    // unsubscribe is the only thing that closes them, so it must run on every teardown.
    const stop = api.jobs.subscribe(jobId, (event) => {
      switch (event.type) {
        case "progress":
          setCurrent(event.job);
          if (event.job.status === "failed") {
            setPhase("error");
            setFailure(event.job.error);
          } else if (event.job.status === "cancelled") {
            setPhase("cancelled");
          }
          break;
        case "log":
          setLastLog(event.message);
          break;
        case "done":
          setCurrent(event.job);
          setPhase("done");
          if (event.scene) onDoneRef.current?.(event.scene);
          if (event.walkthrough) onWalkDoneRef.current?.(event.walkthrough);
          break;
        case "error":
          setCurrent(event.job);
          setFailure(event.error);
          setPhase("error");
          break;
      }
    });

    return stop;
  }, [jobId]);

  useEffect(() => {
    if (phase !== "done" || !autoNavigate) return;
    // A beat of celebration, then the editor — long enough to register, short enough to
    // never feel like the app is stuck on a success screen.
    const target = sceneId ? `/evler/${propertyId}/scenes/${sceneId}` : `/evler/${propertyId}/walk`;
    const timer = setTimeout(() => router.push(target), 1500);
    return () => clearTimeout(timer);
  }, [phase, autoNavigate, router, propertyId, sceneId]);

  const activeIndex = useMemo(() => {
    const index = stages.indexOf(current.stage);
    return index < 0 ? 0 : index;
  }, [current.stage, stages]);

  const percent = Math.round((phase === "done" ? 1 : current.progress) * 100);
  const message = locale === "tr" ? current.message_tr || current.message : current.message;

  const cancel = async () => {
    setCancelling(true);
    try {
      await api.jobs.cancel(jobId);
      setPhase("cancelled");
      setConfirmCancel(false);
      toast(ui("job.cancelled", locale));
    } catch (cause) {
      toast.error(describeError(cause) || ui("job.cancelFailed", locale));
    } finally {
      setCancelling(false);
    }
  };

  const retry = async () => {
    setRetrying(true);
    try {
      const next = sceneId ? await api.scenes.reprocess(sceneId, qualityMode) : await api.walkthrough.rebuild(propertyId, qualityMode);
      setFailure(null);
      setCurrent(next);
      setPhase("running");
      setJobId(next.id);
      toast(ui("job.retrying", locale));
    } catch (cause) {
      toast.error(describeError(cause));
    } finally {
      setRetrying(false);
    }
  };

  return (
    <div className={cn("glass overflow-hidden rounded-lg", className)}>
      <div className="border-b border-line px-5 py-5 sm:px-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <h2 className="text-base font-semibold tracking-tight text-ink">
              {phase === "done"
                ? ui("job.done.title", locale)
                : phase === "error"
                  ? ui("job.failed.title", locale)
                  : ui(isWalk ? "walk.job.title" : "job.title", locale)}
            </h2>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-muted">
              {phase === "done"
                ? ui("job.done.body", locale)
                : phase === "cancelled"
                  ? ui("job.cancelled", locale)
                  : phase === "error"
                    ? stageLabel(current.stage, locale)
                    : message || ui("job.subtitle", locale)}
            </p>
          </div>

          <div className="text-right">
            <div
              className={cn(
                "font-mono text-3xl leading-none font-semibold tabular-nums",
                phase === "error" ? "text-danger" : phase === "done" ? "text-success" : "text-ink",
              )}
            >
              {percent}
              <span className="text-lg text-ink-faint">%</span>
            </div>
            {phase === "running" ? (
              <p className="mt-1.5 text-[12px] text-ink-faint">
                {t("job.eta", locale)} ·{" "}
                <span className="text-ink-muted tabular-nums">{formatDuration(current.eta_seconds)}</span>
              </p>
            ) : null}
          </div>
        </div>

        <Progress
          className="mt-4"
          value={phase === "error" ? current.progress : phase === "done" ? 1 : current.progress}
          tone={phase === "error" ? "danger" : phase === "done" ? "success" : "bio"}
          label={ui(isWalk ? "walk.job.title" : "job.title", locale)}
        />

        {/* Announced only when the stage name changes, so it never chatters. */}
        <p className="sr-only" aria-live="polite">
          {`${stageLabel(current.stage, locale)} — ${percent}%`}
        </p>
      </div>

      <AnimatePresence mode="wait">
        {phase === "error" ? (
          <motion.div
            key="error"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="px-5 py-5 sm:px-6"
          >
            <div className="flex gap-3 rounded-md border border-danger/25 bg-danger/5 p-4">
              <TriangleAlert className="mt-0.5 size-4 shrink-0 text-danger" aria-hidden />
              <div className="min-w-0 space-y-1.5">
                <p className="text-[13px] leading-relaxed font-medium text-ink">
                  {failure ? (locale === "tr" ? failure.message_tr || failure.message : failure.message) : t("error.generic", locale)}
                </p>
                {failure?.hint ? (
                  <p className="text-[12px] leading-relaxed text-ink-muted">{failure.hint}</p>
                ) : null}
              </div>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <Button variant="primary" loading={retrying} onClick={() => void retry()}>
                <RotateCcw className="size-4" aria-hidden />
                {t("job.retry", locale)}
              </Button>
              <Link href={`/evler/${propertyId}`} className={buttonStyles("ghost", "md")}>
                {ui("editor.back", locale)}
              </Link>
            </div>
          </motion.div>
        ) : phase === "done" ? (
          <motion.div
            key="done"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center gap-4 px-5 py-10 sm:px-6"
          >
            <motion.span
              className="ring-glow relative grid size-16 place-items-center rounded-full border border-success/30 bg-success/10 text-success"
              initial={{ scale: 0.7, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            >
              <Check className="size-7" strokeWidth={2.4} aria-hidden />
              <motion.span
                className="absolute inset-0 rounded-full border border-success/50"
                initial={{ scale: 1, opacity: 0.8 }}
                animate={{ scale: 1.7, opacity: 0 }}
                transition={{ duration: 1.1, ease: "easeOut", repeat: 1 }}
                aria-hidden
              />
            </motion.span>
            <Link href={sceneId ? `/evler/${propertyId}/scenes/${sceneId}` : `/evler/${propertyId}/walk`} className={buttonStyles("primary", "md")}>
              {sceneId ? ui("job.openEditor", locale) : ui("walk.open", locale)}
              <ArrowRight className="size-4" aria-hidden />
            </Link>
          </motion.div>
        ) : (
          <motion.ol
            key="rail"
            aria-label={ui("job.steps", locale)}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="relative px-5 py-4 sm:px-6"
          >
            {/* Spine: a dim full-height rule with a lit section that grows with progress. */}
            <span className="absolute top-6 bottom-6 left-[27px] w-px bg-line sm:left-[31px]" aria-hidden />
            <motion.span
              className="absolute top-6 left-[27px] w-px origin-top bg-bio/70 sm:left-[31px]"
              aria-hidden
              style={{ height: "calc(100% - 3rem)" }}
              initial={{ scaleY: 0 }}
              animate={{ scaleY: activeIndex / (stages.length - 1) }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            />

            {stages.map((stage, index) => {
              const done = index < activeIndex;
              const active = index === activeIndex && phase === "running";
              return (
                <li key={stage} className="relative flex h-8 items-center gap-3">
                  <span
                    className={cn(
                      "z-10 grid size-[15px] shrink-0 place-items-center rounded-full border transition-colors duration-300",
                      done
                        ? "border-bio/60 bg-bio/20 text-bio"
                        : active
                          ? "border-bio bg-canvas text-bio"
                          : "border-line bg-canvas text-transparent",
                    )}
                  >
                    {done ? (
                      <Check className="size-2.5" strokeWidth={3.5} aria-hidden />
                    ) : active ? (
                      <span className="animate-breathe size-1.5 rounded-full bg-bio" aria-hidden />
                    ) : (
                      <span className="size-1 rounded-full bg-ink-faint/40" aria-hidden />
                    )}
                  </span>

                  <span
                    className={cn(
                      "text-[13px] transition-colors duration-300",
                      done ? "text-ink-muted" : active ? "font-medium text-ink" : "text-ink-faint/60",
                    )}
                  >
                    {stageLabel(stage as Job["stage"], locale)}
                  </span>

                  {active ? (
                    <span className="ml-auto flex items-center gap-2">
                      <span className="hidden w-24 sm:block">
                        <Progress value={current.stage_progress} size="sm" label={stageLabel(stage as Job["stage"], locale)} />
                      </span>
                      <Loader2 className="size-3.5 animate-spin text-bio" aria-hidden />
                    </span>
                  ) : null}
                </li>
              );
            })}
          </motion.ol>
        )}
      </AnimatePresence>

      {phase === "running" ? (
        <div className="flex items-center justify-between gap-3 border-t border-line px-5 py-3 sm:px-6">
          <p className="min-w-0 truncate font-mono text-[11px] text-ink-faint">
            {lastLog ?? ui("job.queued", locale)}
          </p>
          <Button variant="ghost" size="sm" onClick={() => setConfirmCancel(true)}>
            <Ban className="size-3.5" aria-hidden />
            {t("job.cancel", locale)}
          </Button>
        </div>
      ) : phase === "cancelled" ? (
        <div className="flex justify-end border-t border-line px-5 py-3 sm:px-6">
          <Link href={`/evler/${propertyId}`} className={buttonStyles("secondary", "sm")}>
            {ui("editor.back", locale)}
          </Link>
        </div>
      ) : null}

      <Dialog
        open={confirmCancel}
        onClose={() => setConfirmCancel(false)}
        title={t("job.cancel", locale)}
        description={ui("job.cancel.confirm", locale)}
        size="sm"
        closeLabel={ui("common.close", locale)}
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirmCancel(false)}>
              {t("editor.cancel", locale)}
            </Button>
            <Button variant="danger" loading={cancelling} onClick={() => void cancel()}>
              {t("job.cancel", locale)}
            </Button>
          </>
        }
      />
    </div>
  );
}
