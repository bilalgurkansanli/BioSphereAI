"use client";

/**
 * "Sahne ekle" on a property page: film the room with a phone, or upload a video.
 *
 * THE PHONE COMES FIRST. Before this dialog offered it, the guided capture for an existing
 * property had no door from the property page at all — the QR existed only in the new-property
 * wizard and in the 3D walkthrough card — so adding a room meant uploading whatever video was
 * at hand, and the video at hand is usually one that went through a messenger (the first real
 * tour was a 0.6 MP WhatsApp transcode). The guided capture records the sensor's own 4:3 frame
 * and sends the sidecar that names the phone's lens, which the stitch uses to bound its focal.
 *
 * While the QR is on screen the dialog watches the property for a room the phone created, so
 * the agent at the desk sees "it arrived" without reloading.
 */

import { Smartphone, UploadCloud } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { JobProgress } from "./JobProgress";
import { PhoneHandoff } from "@/components/capture/PhoneHandoff";
import { QualitySelector } from "./QualitySelector";
import { UploadDropzone, type MediaSelection } from "./UploadDropzone";
import { ui } from "./copy";
import { Button } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { Input } from "@/components/ui/Input";
import { Progress } from "@/components/ui/Progress";
import { Tabs, tabPanelProps } from "@/components/ui/Tabs";
import { BioSphereError, api } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";
import type { Job, QualityMode, Scene } from "@/lib/types";

export interface AddSceneDialogProps {
  propertyId: string;
  open: boolean;
  onClose: () => void;
  /** Suggested room name, e.g. "Sahne 3". */
  suggestedName: string;
  onSceneReady: (scene: Scene) => void;
}

/** How often the dialog asks whether the phone has created a room. One cheap GET. */
const ARRIVAL_POLL_MS = 4000;

type Source = "phone" | "file";

export function AddSceneDialog({ propertyId, open, onClose, suggestedName, onSceneReady }: AddSceneDialogProps) {
  const { locale } = useLocale();
  const describeError = useAppStore((s) => s.describeError);
  const tr = locale === "tr";

  const [source, setSource] = useState<Source>("phone");
  const [arrived, setArrived] = useState<Scene | null>(null);
  // The parent passes a fresh arrow on every render; the watcher must not restart for that.
  const onSceneReadyRef = useRef(onSceneReady);
  useEffect(() => {
    onSceneReadyRef.current = onSceneReady;
  }, [onSceneReady]);

  const [name, setName] = useState("");
  const [quality, setQuality] = useState<QualityMode>("balanced");
  const [selection, setSelection] = useState<MediaSelection | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const [sceneId, setSceneId] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortRef.current?.abort(), []);

  // A closed dialog holding a half-finished upload would be a trap on reopen. Reset while
  // rendering the close rather than in an effect after it: the "adjust state when a prop
  // changes" pattern, which costs no second commit of the stale dialog.
  const [wasOpen, setWasOpen] = useState(open);
  if (open !== wasOpen) {
    setWasOpen(open);
    if (!open) {
      setSelection(null);
      setUploaded(0);
      setJob(null);
      setSceneId(null);
      setName("");
      setArrived(null);
    }
  }

  // Watch for the room the phone creates. The first answer is the baseline; any scene id that
  // appears after it came from the QR (or from another tab — either way it is news).
  useEffect(() => {
    if (!open || source !== "phone" || job) return;
    let alive = true;
    let known: Set<string> | null = null;
    const tick = async () => {
      try {
        const detail = await api.properties.get(propertyId);
        if (!alive) return;
        const ids = new Set(detail.scenes.map((scene) => scene.id));
        if (known === null) {
          known = ids;
          return;
        }
        const fresh = [...detail.scenes].reverse().find((scene) => !known?.has(scene.id));
        known = ids;
        if (fresh) {
          setArrived(fresh);
          onSceneReadyRef.current(fresh);
        }
      } catch {
        /* A missed poll is a later "it arrived", not an error worth showing. */
      }
    };
    void tick();
    const timer = setInterval(() => void tick(), ARRIVAL_POLL_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [open, source, job, propertyId]);

  const start = async () => {
    if (!selection) return;
    setUploading(true);
    setUploaded(0);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const result = await api.scenes.create(
        propertyId,
        selection.files,
        { name: name.trim() || suggestedName, quality_mode: quality, capture_kind: selection.kind },
        setUploaded,
        controller.signal,
      );
      setSceneId(result.scene.id);
      setJob(result.job);
    } catch (cause) {
      if (!(cause instanceof BioSphereError && cause.detail.code === "aborted")) {
        toast.error(describeError(cause));
      }
    } finally {
      setUploading(false);
      abortRef.current = null;
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={t("scene.add", locale)}
      description={job ? undefined : ui("wizard.step.upload.desc", locale)}
      size="lg"
      closeLabel={ui("common.close", locale)}
      footer={
        job ? undefined : source === "phone" ? (
          <Button variant="ghost" onClick={onClose}>
            {ui("common.close", locale)}
          </Button>
        ) : uploading ? (
          <Button variant="danger" onClick={() => abortRef.current?.abort()}>
            {t("editor.cancel", locale)}
          </Button>
        ) : (
          <>
            <Button variant="ghost" onClick={onClose}>
              {t("editor.cancel", locale)}
            </Button>
            <Button variant="primary" disabled={!selection} onClick={() => void start()}>
              <UploadCloud className="size-4" aria-hidden />
              {ui("upload.start", locale)}
            </Button>
          </>
        )
      }
    >
      {job && sceneId ? (
        <JobProgress
          job={job}
          propertyId={propertyId}
          sceneId={sceneId}
          qualityMode={quality}
          autoNavigate={false}
          onDone={onSceneReady}
          className="border-0 bg-transparent backdrop-blur-none"
        />
      ) : (
        <div className="space-y-5">
          <Tabs
            idBase="add-scene"
            size="sm"
            value={source}
            onChange={(id) => setSource(id as Source)}
            items={[
              { id: "phone", label: tr ? "Telefonla çek" : "Shoot with a phone", icon: Smartphone },
              { id: "file", label: tr ? "Video yükle" : "Upload a video", icon: UploadCloud },
            ]}
          />
          {source === "phone" ? (
            <div {...tabPanelProps("add-scene", "phone")} className="space-y-4">
              <PhoneHandoff propertyId={propertyId} locale={locale} />
              {arrived ? (
                <p
                  role="status"
                  className="rounded-md border border-bio/30 bg-bio/[0.06] px-3.5 py-2.5 text-[13px] leading-relaxed text-ink"
                >
                  {tr
                    ? `“${arrived.name}” telefondan geldi ve işleniyor. Hazır olunca ev sayfasında görünür.`
                    : `“${arrived.name}” arrived from the phone and is being processed. It appears on the property page when ready.`}
                </p>
              ) : (
                <p className="text-[12.5px] leading-relaxed text-ink-faint">
                  {tr
                    ? "Telefonda çekim bitip yükleme tamamlandığında oda burada görünür; bu pencereyi açık bırakabilirsiniz."
                    : "When the phone finishes and the upload completes, the room shows up here; you can leave this open."}
                </p>
              )}
            </div>
          ) : (
            <div {...tabPanelProps("add-scene", "file")} className="space-y-5">
              <Input
                label={t("scene.name", locale)}
                placeholder={suggestedName}
                value={name}
                disabled={uploading}
                onChange={(event) => setName(event.target.value)}
              />
              <QualitySelector
                value={quality}
                onChange={setQuality}
                label={ui("wizard.quality", locale)}
                className="sm:grid-cols-3"
              />
              <UploadDropzone selection={selection} onSelect={setSelection} disabled={uploading} />
              {uploading ? (
                <div className="rounded-md border border-line bg-surface-2/40 p-4">
                  <div className="mb-2 flex items-center justify-between text-[12px]">
                    <span className="font-medium text-ink">{t("upload.uploading", locale)}</span>
                    <span className="font-mono text-ink-muted tabular-nums">{Math.round(uploaded * 100)}%</span>
                  </div>
                  <Progress value={uploaded} label={t("upload.uploading", locale)} />
                </div>
              ) : null}
            </div>
          )}
        </div>
      )}
    </Dialog>
  );
}
