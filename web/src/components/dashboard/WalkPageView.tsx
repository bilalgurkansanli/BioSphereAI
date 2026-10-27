"use client";

/**
 * The dashboard's own look at a property's 3D walkthrough: the viewer, full height, with the
 * build's numbers underneath for the agent who wants to know what they got.
 */

import { ArrowLeft, RotateCcw } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { toast } from "sonner";
import { ApiErrorState } from "./ErrorState";
import { JobProgress } from "./JobProgress";
import { ui } from "./copy";
import { useCrumbLabel } from "./Breadcrumbs";
import { Button, buttonStyles } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { WalkViewer } from "@/components/walk/WalkViewer";
import { api } from "@/lib/api";
import { useApi } from "@/lib/hooks/useApi";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";
import type { Job, PropertyDetail } from "@/lib/types";
import { Box } from "lucide-react";

export function WalkPageView({ propertyId }: { propertyId: string }) {
  const { locale } = useLocale();
  const describeError = useAppStore((s) => s.describeError);
  const { data: property, loading, error, refetch } = useApi<PropertyDetail>(() => api.properties.get(propertyId), [propertyId]);
  const [rebuilding, setRebuilding] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  useCrumbLabel(`/evler/${propertyId}`, property?.title);
  useCrumbLabel(`/evler/${propertyId}/walk`, t("walk.title", locale));

  if (loading && !property) return <Skeleton className="h-[70vh] w-full rounded-lg" />;
  if (error || !property) return <ApiErrorState error={error} onRetry={refetch} />;

  const walk = property.walkthrough;
  const activeJobId = walk?.active_job_id ?? null;

  const rebuild = async () => {
    setRebuilding(true);
    try {
      const next = await api.walkthrough.rebuild(propertyId, walk?.quality_mode ?? "");
      setJob(next);
    } catch (cause) {
      toast.error(describeError(cause));
    } finally {
      setRebuilding(false);
    }
  };

  const metrics = (walk?.metrics ?? {}) as Record<string, unknown>;
  const seconds = typeof metrics.seconds_total === "number" ? metrics.seconds_total : null;
  const bytes = typeof metrics.bytes_total === "number" ? metrics.bytes_total : null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Link href={`/evler/${propertyId}`} className={buttonStyles("ghost", "sm")}>
            <ArrowLeft className="size-4" aria-hidden />
            {ui("editor.back", locale)}
          </Link>
          <h1 className="text-base font-semibold tracking-tight text-ink">{property.title} · {t("walk.title", locale)}</h1>
        </div>
        {walk && walk.status !== "processing" ? (
          <Button variant="secondary" size="sm" loading={rebuilding} onClick={() => void rebuild()}>
            <RotateCcw className="size-4" aria-hidden />
            {ui("walk.rebuild", locale)}
          </Button>
        ) : null}
      </div>

      {job || (walk && walk.status === "processing" && activeJobId) ? (
        <ActiveWalkJob propertyId={propertyId} job={job} jobId={activeJobId} qualityMode={walk?.quality_mode ?? "balanced"} onDone={() => { setJob(null); void refetch(); }} />
      ) : null}

      {walk?.status === "ready" && walk.manifest ? (
        <div className="stage-viewport overflow-hidden rounded-lg border border-line" style={{ height: "min(78vh, 900px)" }}>
          <WalkViewer manifest={walk.manifest} title={property.title} locale={locale} className="h-full w-full" />
        </div>
      ) : !walk ? (
        <EmptyState icon={Box} title={ui("walk.empty.title", locale)} body={ui("walk.empty.body", locale)} action={<Link href={`/evler/${propertyId}`} className={buttonStyles("primary", "md")}>{ui("editor.back", locale)}</Link>} />
      ) : walk.status === "failed" ? (
        <EmptyState
          icon={Box}
          tone="danger"
          title={ui("walk.failed.title", locale)}
          body={walk.error ? (locale === "tr" ? walk.error.message_tr : walk.error.message) : t("error.generic", locale)}
          action={<Button variant="primary" loading={rebuilding} onClick={() => void rebuild()}>{ui("walk.rebuild", locale)}</Button>}
        />
      ) : null}

      {walk?.status === "ready" && walk.manifest ? (
        <dl className="grid grid-cols-2 gap-3 text-[12px] sm:grid-cols-4">
          <Stat label={ui("walk.stat.areas", locale)} value={String(1 + walk.manifest.rooms.filter((r) => !r.failed).length)} />
          <Stat label={ui("walk.stat.splats", locale)} value={[walk.manifest.base, ...walk.manifest.rooms].reduce((n, a) => n + (a.splat_count || 0), 0).toLocaleString(locale)} />
          <Stat label={ui("walk.stat.size", locale)} value={bytes !== null ? `${(bytes / 1e6).toFixed(1)} MB` : "—"} />
          <Stat label={ui("walk.stat.time", locale)} value={seconds !== null ? `${Math.round(seconds / 60)} dk` : "—"} />
        </dl>
      ) : null}
      {walk?.manifest?.warnings?.length ? (
        <ul className="space-y-1 text-[11px] text-amber">
          {walk.manifest.warnings.map((w) => (
            <li key={w} className="font-mono">{w}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass rounded-md px-4 py-3">
      <dt className="text-[10px] font-semibold tracking-[0.14em] text-ink-faint uppercase">{label}</dt>
      <dd className="mt-1 font-mono text-[15px] text-ink tabular-nums">{value}</dd>
    </div>
  );
}

function ActiveWalkJob({ propertyId, job, jobId, qualityMode, onDone }: { propertyId: string; job: Job | null; jobId: string | null; qualityMode: PropertyDetail["scenes"][number]["quality_mode"]; onDone: () => void }) {
  const { data } = useApi<Job | null>(() => (job ? Promise.resolve(job) : jobId ? api.jobs.get(jobId) : Promise.resolve(null)), [job?.id ?? null, jobId]);
  const current = job ?? data;
  if (!current) return null;
  return <JobProgress job={current} propertyId={propertyId} sceneId={null} qualityMode={qualityMode} autoNavigate={false} onWalkthroughDone={onDone} />;
}
