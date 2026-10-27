"use client";

import { ArrowLeft, ArrowRight, Check, Monitor, Smartphone, UploadCloud } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { CaptureGuide } from "./CaptureGuide";
import { PhoneHandoff } from "@/components/capture/PhoneHandoff";
import { JobProgress } from "./JobProgress";
import { QualitySelector } from "./QualitySelector";
import { UploadDropzone, type MediaSelection } from "./UploadDropzone";
import { ui, type UiKey } from "./copy";
import { Button, buttonStyles } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Progress } from "@/components/ui/Progress";
import { BioSphereError, api } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";
import type { Job, QualityMode, Property } from "@/lib/types";
import { cn } from "@/lib/utils";

const STEPS: Array<{ label: UiKey; desc: UiKey }> = [
  { label: "wizard.step.details", desc: "wizard.step.details.desc" },
  { label: "wizard.step.capture", desc: "wizard.step.capture.desc" },
  { label: "wizard.step.upload", desc: "wizard.step.upload.desc" },
];

export function NewPropertyWizard() {
  const { locale } = useLocale();
  const describeError = useAppStore((s) => s.describeError);

  const [step, setStep] = useState(0);
  const [title, setTitle] = useState("");
  const [address, setAddress] = useState("");
  const [titleError, setTitleError] = useState<string | null>(null);
  const [property, setProperty] = useState<Property | null>(null);
  const [saving, setSaving] = useState(false);

  const [quality, setQuality] = useState<QualityMode>("balanced");
  const [sceneName, setSceneName] = useState("");
  const [selection, setSelection] = useState<MediaSelection | null>(null);
  /** Which way the footage arrives. See the switcher in step three. */
  const [source, setSource] = useState<"computer" | "phone">("computer");

  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const [sceneId, setSceneId] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  // Leaving mid-upload must not keep a 500 MB request alive in the background.
  useEffect(() => () => abortRef.current?.abort(), []);

  const submitDetails = async () => {
    const trimmed = title.trim();
    if (!trimmed) {
      setTitleError(ui("wizard.title.required", locale));
      return;
    }
    setTitleError(null);
    setSaving(true);
    try {
      if (!property) {
        const created = await api.properties.create({ title: trimmed, address: address.trim() || undefined });
        setProperty(created);
        toast.success(ui("wizard.created", locale));
      } else if (property.title !== trimmed || (property.address ?? "") !== address.trim()) {
        // The agent stepped back and edited; keep the draft property rather than making a second.
        const updated = await api.properties.update(property.id, { title: trimmed, address: address.trim() });
        setProperty(updated);
      }
      setStep(1);
    } catch (cause) {
      toast.error(describeError(cause));
    } finally {
      setSaving(false);
    }
  };

  const startUpload = async () => {
    if (!property || !selection) return;
    setUploading(true);
    setUploaded(0);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const result = await api.scenes.create(
        property.id,
        selection.files,
        {
          name: sceneName.trim() || (locale === "tr" ? "Sahne 1" : "Scene 1"),
          quality_mode: quality,
          capture_kind: selection.kind,
        },
        setUploaded,
        controller.signal,
      );
      setSceneId(result.scene.id);
      setJob(result.job);
    } catch (cause) {
      // An abort is the agent's own decision, not an error worth shouting about.
      if (!(cause instanceof BioSphereError && cause.detail.code === "aborted")) {
        toast.error(describeError(cause));
      }
    } finally {
      setUploading(false);
      abortRef.current = null;
    }
  };

  if (job && sceneId && property) {
    return (
      <div className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6">
        <JobProgress job={job} propertyId={property.id} sceneId={sceneId} qualityMode={quality} />
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6">
      <h1 className="text-2xl font-semibold tracking-tight text-ink">{ui("wizard.title", locale)}</h1>

      <ol className="mt-6 mb-8 flex items-center gap-2">
        {STEPS.map((entry, index) => {
          const done = index < step;
          const active = index === step;
          return (
            <li key={entry.label} className={cn("flex items-center gap-3", index < STEPS.length - 1 && "flex-1")}>
              <span
                className={cn(
                  "grid size-7 shrink-0 place-items-center rounded-full border font-mono text-[12px] transition-colors duration-300",
                  done
                    ? "border-bio/50 bg-bio/15 text-bio"
                    : active
                      ? "border-bio bg-bio/10 text-bio"
                      : "border-line bg-surface-2 text-ink-faint",
                )}
              >
                {done ? <Check className="size-3.5" strokeWidth={3} aria-hidden /> : index + 1}
              </span>
              <span className={cn("hidden text-[13px] font-medium sm:block", active ? "text-ink" : "text-ink-faint")}>
                {ui(entry.label, locale)}
              </span>
              {index < STEPS.length - 1 ? (
                <span
                  className={cn("h-px flex-1 transition-colors duration-500", done ? "bg-bio/50" : "bg-line")}
                  aria-hidden
                />
              ) : null}
            </li>
          );
        })}
      </ol>

      <div className="glass rounded-lg p-5 sm:p-6">
        <div className="mb-5">
          <h2 className="text-base font-semibold tracking-tight text-ink">{ui(STEPS[step].label, locale)}</h2>
          <p className="mt-1 text-[13px] text-ink-muted">{ui(STEPS[step].desc, locale)}</p>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            {step === 0 ? (
              <div className="space-y-4">
                <Input
                  label={t("property.title", locale)}
                  placeholder={ui("wizard.title.placeholder", locale)}
                  value={title}
                  required
                  autoFocus
                  error={titleError ?? undefined}
                  onChange={(event) => {
                    setTitle(event.target.value);
                    if (titleError) setTitleError(null);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void submitDetails();
                  }}
                />
                <Input
                  label={`${t("property.address", locale)} · ${ui("common.optional", locale)}`}
                  placeholder={ui("wizard.address.placeholder", locale)}
                  value={address}
                  onChange={(event) => setAddress(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void submitDetails();
                  }}
                />
              </div>
            ) : step === 1 ? (
              <div className="space-y-8">
                <CaptureGuide />
                <div className="border-t border-line pt-6">
                  <h3 className="text-sm font-semibold tracking-tight text-ink">{ui("wizard.quality", locale)}</h3>
                  <p className="mt-1 mb-4 text-[12px] text-ink-muted">{ui("wizard.quality.desc", locale)}</p>
                  <QualitySelector value={quality} onChange={setQuality} label={ui("wizard.quality", locale)} />
                </div>
              </div>
            ) : (
              <div className="space-y-5">
                {/* WHERE THE FOOTAGE COMES FROM, as one either/or rather than two stacked
                    offers. The person filling this in is in one of two places — at a laptop
                    holding a file, or standing in the flat holding the camera — and showing
                    both paths at once asks them to read the one that cannot apply to them.
                    Hidden mid-upload: a switch that would abandon a transfer in progress is
                    not a choice, it is a trap. */}
                {!uploading ? (
                  <div
                    role="group"
                    className="flex w-fit items-center gap-0.5 rounded-full border border-line bg-surface-2/60 p-0.5"
                  >
                    {[
                      { value: "computer" as const, icon: Monitor, label: ui("wizard.source.computer", locale) },
                      { value: "phone" as const, icon: Smartphone, label: ui("wizard.source.phone", locale) },
                    ].map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        aria-pressed={source === option.value}
                        onClick={() => setSource(option.value)}
                        className={cn(
                          "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[12px] font-medium transition-colors",
                          source === option.value ? "bg-surface-3 text-ink" : "text-ink-faint hover:text-ink-muted",
                        )}
                      >
                        <option.icon className="size-3.5" aria-hidden />
                        {option.label}
                      </button>
                    ))}
                  </div>
                ) : null}

                {source === "computer" || uploading ? (
                  <>
                    <Input
                      label={t("scene.name", locale)}
                      placeholder={ui("wizard.scene.placeholder", locale)}
                      value={sceneName}
                      disabled={uploading}
                      onChange={(event) => setSceneName(event.target.value)}
                    />
                    <UploadDropzone selection={selection} onSelect={setSelection} disabled={uploading} />

                    {uploading ? (
                      <div className="rounded-md border border-line bg-surface-2/40 p-4">
                        <div className="mb-2 flex items-center justify-between text-[12px]">
                          <span className="font-medium text-ink">{t("upload.uploading", locale)}</span>
                          <span className="font-mono text-ink-muted tabular-nums">
                            {Math.round(uploaded * 100)}%
                          </span>
                        </div>
                        <Progress value={uploaded} label={t("upload.uploading", locale)} />
                      </div>
                    ) : null}
                  </>
                ) : (
                  // No room-name field on this path: the guided capture asks for it on the
                  // phone, where the person can see which room they are standing in.
                  <PhoneHandoff
                    propertyId={property?.id}
                    locale={locale}
                    className="border-line-strong"
                    footer={
                      <div className="space-y-1.5 text-[12.5px] leading-relaxed text-ink-muted">
                        <p>{ui("wizard.phone.note", locale)}</p>
                        <p className="text-ink-faint">{ui("wizard.phone.signin", locale)}</p>
                      </div>
                    }
                  />
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        <div className="mt-7 flex items-center justify-between gap-3 border-t border-line pt-5">
          <Button
            variant="ghost"
            onClick={() => setStep((s) => Math.max(0, s - 1))}
            disabled={step === 0 || uploading}
            className={cn(step === 0 && "invisible")}
          >
            <ArrowLeft className="size-4" aria-hidden />
            {ui("common.back", locale)}
          </Button>

          {step === 0 ? (
            <Button variant="primary" loading={saving} onClick={() => void submitDetails()}>
              {ui("common.continue", locale)}
              <ArrowRight className="size-4" aria-hidden />
            </Button>
          ) : step === 1 ? (
            <Button variant="primary" onClick={() => setStep(2)}>
              {ui("common.continue", locale)}
              <ArrowRight className="size-4" aria-hidden />
            </Button>
          ) : uploading ? (
            <Button variant="danger" onClick={() => abortRef.current?.abort()}>
              {t("editor.cancel", locale)}
            </Button>
          ) : source === "phone" ? (
            // Nothing to upload from here, so the way out is forward: the property page,
            // where the room appears on its own when the phone finishes.
            <Link
              href={property ? `/evler/${property.id}` : "/evler"}
              className={buttonStyles("primary", "md")}
            >
              {ui("wizard.phone.open", locale)}
              <ArrowRight className="size-4" aria-hidden />
            </Link>
          ) : (
            <Button variant="primary" disabled={!selection} onClick={() => void startUpload()}>
              <UploadCloud className="size-4" aria-hidden />
              {ui("upload.start", locale)}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
