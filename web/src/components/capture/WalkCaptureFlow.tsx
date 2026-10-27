"use client";

/**
 * The whole 3D-walkthrough visit on a phone: plan the areas, shoot each one, upload it the
 * moment it is shot, and build once at the end.
 *
 * Uploading per area is a memory decision as much as a UX one: a five-room flat is over a
 * gigabyte of video, and a phone browser holding that in blobs while the agent keeps
 * filming does not stay open. So every area leaves the phone before the next is recorded
 * (`defer_build`), and the build is one request at the end.
 */

import { ArrowRight, Box, Check, Plus, Trash2, UploadCloud } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { WalkCaptureStage, type WalkArea } from "./WalkCaptureStage";
import { ui } from "@/components/dashboard/copy";
import { JobProgress } from "@/components/dashboard/JobProgress";
import { QualitySelector } from "@/components/dashboard/QualitySelector";
import { Button, buttonStyles } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Progress } from "@/components/ui/Progress";
import { Switch } from "@/components/ui/Switch";
import { BioSphereError, api } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import type { Job, Locale, QualityMode } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface WalkCaptureFlowProps {
  propertyId: string;
  locale: Locale;
}

type Step = "plan" | "shoot" | "uploading" | "between" | "build" | "building";

let seq = 0;
const roomDraft = (name: string): WalkArea => ({ id: `r${(seq += 1)}`, kind: "room", name });

export function WalkCaptureFlow({ propertyId, locale }: WalkCaptureFlowProps) {
  const describeError = useAppStore((s) => s.describeError);
  const [withGeneral, setWithGeneral] = useState(true);
  const [rooms, setRooms] = useState<WalkArea[]>(() => [roomDraft(ui("walk.room.placeholder", locale).replace("{n}", "1"))]);
  const [quality, setQuality] = useState<QualityMode>("balanced");
  const [step, setStep] = useState<Step>("plan");
  const [index, setIndex] = useState(0);
  const [done, setDone] = useState<Set<string>>(new Set());
  const [progress, setProgress] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortRef.current?.abort(), []);

  const areas: WalkArea[] = [...(withGeneral ? [{ id: "general", kind: "general" as const, name: ui("walkcap.plan.general", locale) }] : []), ...rooms.filter((r) => r.name.trim())];
  const current = areas[index] ?? null;
  const remaining = areas.filter((a) => !done.has(a.id));

  const uploadArea = async (area: WalkArea, files: File[]) => {
    setStep("uploading");
    setProgress(0);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await api.walkthrough.create(
        propertyId,
        area.kind === "general" ? files : [],
        area.kind === "room" ? [{ name: area.name.trim(), files }] : [],
        quality,
        setProgress,
        controller.signal,
        true,
      );
      setDone((d) => new Set([...d, area.id]));
      setStep("between");
    } catch (cause) {
      if (!(cause instanceof BioSphereError && cause.detail.code === "aborted")) toast.error(describeError(cause));
      setStep("between");
    } finally {
      abortRef.current = null;
    }
  };

  const build = async () => {
    try {
      const next = await api.walkthrough.rebuild(propertyId, quality);
      setJob(next);
      setStep("building");
    } catch (cause) {
      toast.error(describeError(cause));
    }
  };

  if (step === "shoot" && current) {
    return (
      <WalkCaptureStage
        key={current.id}
        area={current}
        locale={locale}
        quality={quality}
        onRecorded={(files) => void uploadArea(current, files)}
        onCancel={() => setStep(done.size ? "between" : "plan")}
      />
    );
  }

  if (step === "building" && job) {
    return (
      <main className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6">
        <JobProgress job={job} propertyId={propertyId} sceneId={null} qualityMode={quality} autoNavigate={false} />
        <Link href={`/evler/${propertyId}/walk`} className={cn(buttonStyles("secondary", "md"), "mt-4 w-full")}>
          {ui("walkcap.done.open", locale)}
          <ArrowRight className="size-4" aria-hidden />
        </Link>
      </main>
    );
  }

  return (
    <main className="mx-auto flex min-h-[100dvh] w-full max-w-md flex-col justify-center gap-5 px-5 py-8">
      <div>
        <p className="text-[10px] font-semibold tracking-[0.18em] text-ink-faint uppercase">{ui("walkcap.title", locale)}</p>
        <p className="mt-2 text-[13px] leading-relaxed text-ink-muted">{ui("walkcap.plan.intro", locale)}</p>
      </div>

      {step === "plan" ? (
        <>
          <div className="glass rounded-lg p-4">
            <Switch checked={withGeneral} onChange={setWithGeneral} label={ui("walkcap.plan.general", locale)} description={ui("walkcap.plan.general.hint", locale)} />
            {!withGeneral && rooms.length > 1 ? <p className="mt-2 text-[12px] text-amber">{ui("walk.needsBase", locale)}</p> : null}
            {!withGeneral && rooms.length === 1 ? <p className="mt-2 text-[12px] text-ink-faint">{ui("walkcap.plan.skipGeneral", locale)}</p> : null}
          </div>
          <div className="glass rounded-lg p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[12px] font-semibold tracking-[0.12em] text-ink-muted uppercase">{ui("walkcap.plan.rooms", locale)}</span>
              <Button variant="ghost" size="sm" onClick={() => setRooms((r) => [...r, roomDraft(ui("walk.room.placeholder", locale).replace("{n}", String(r.length + 1)))])}>
                <Plus className="size-3.5" aria-hidden />
                {ui("walkcap.plan.addRoom", locale)}
              </Button>
            </div>
            <p className="mb-3 text-[12px] text-ink-faint">{ui("walkcap.plan.room.hint", locale)}</p>
            <div className="space-y-2">
              {rooms.map((room) => (
                <div key={room.id} className="flex gap-2">
                  <Input value={room.name} onChange={(e) => setRooms((rs) => rs.map((r) => (r.id === room.id ? { ...r, name: e.target.value } : r)))} aria-label={ui("walkcap.plan.rooms", locale)} wrapperClassName="flex-1" />
                  <Button variant="ghost" size="icon" disabled={rooms.length === 1} aria-label={ui("upload.remove", locale)} onClick={() => setRooms((rs) => rs.filter((r) => r.id !== room.id))}>
                    <Trash2 className="size-4" aria-hidden />
                  </Button>
                </div>
              ))}
            </div>
          </div>
          <QualitySelector value={quality} onChange={setQuality} label={ui("wizard.quality", locale)} flavour="walk" />
          <Button variant="primary" size="lg" disabled={areas.length === 0 || (!withGeneral && rooms.length > 1)} onClick={() => { setIndex(0); setStep("shoot"); }}>
            {ui("walkcap.plan.start", locale)}
            <ArrowRight className="size-4" aria-hidden />
          </Button>
        </>
      ) : null}

      {step === "uploading" && current ? (
        <div className="glass rounded-lg p-5">
          <div className="mb-2 flex items-center justify-between text-[12px]">
            <span className="text-ink-muted">{ui("walkcap.uploading", locale)} · {current.name}</span>
            <span className="font-mono text-ink">{Math.round(progress * 100)}%</span>
          </div>
          <Progress value={progress} label={ui("walkcap.uploading", locale)} />
        </div>
      ) : null}

      {step === "between" || step === "build" ? (
        <div className="glass rounded-lg p-5">
          <ul className="space-y-2">
            {areas.map((a) => (
              <li key={a.id} className="flex items-center justify-between text-[13px]">
                <span className={done.has(a.id) ? "text-ink" : "text-ink-muted"}>{a.name}</span>
                {done.has(a.id) ? (
                  <span className="flex items-center gap-1 text-[11px] text-success">
                    <Check className="size-3.5" aria-hidden />
                    {ui("walkcap.uploaded", locale)}
                  </span>
                ) : (
                  <Button size="sm" variant="secondary" onClick={() => { setIndex(areas.findIndex((x) => x.id === a.id)); setStep("shoot"); }}>
                    {ui("walkcap.record", locale)}
                  </Button>
                )}
              </li>
            ))}
          </ul>
          {remaining.length === 0 ? (
            <>
              <p className="mt-4 text-[12px] text-ink-muted">{ui("walkcap.build.hint", locale)}</p>
              <Button variant="primary" size="lg" className="mt-3 w-full" onClick={() => void build()}>
                <Box className="size-4" aria-hidden />
                {ui("walkcap.build", locale)}
              </Button>
            </>
          ) : (
            <Button variant="primary" size="lg" className="mt-4 w-full" onClick={() => { setIndex(areas.findIndex((x) => x.id === remaining[0].id)); setStep("shoot"); }}>
              <UploadCloud className="size-4" aria-hidden />
              {ui("walkcap.next.title", locale)}: {remaining[0].name}
              <ArrowRight className="size-4" aria-hidden />
            </Button>
          )}
        </div>
      ) : null}
    </main>
  );
}
