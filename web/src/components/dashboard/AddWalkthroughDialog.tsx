"use client";

/**
 * Upload the videos of a 3D walkthrough: one general video, one per room.
 *
 * The general video is required as soon as there is more than one room, because it is what
 * places the rooms in the flat; the engine says so too, but a form that lets the user find
 * out after uploading two gigabytes is a bad form. Rooms can be new (a name) or an existing
 * room of the property (its 360 stays, the 3D area joins it).
 */

import { Box, Plus, Trash2, UploadCloud } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { JobProgress } from "./JobProgress";
import { QualitySelector } from "./QualitySelector";
import { ui } from "./copy";
import { Button } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { Input } from "@/components/ui/Input";
import { Progress } from "@/components/ui/Progress";
import { Select } from "@/components/ui/Select";
import { BioSphereError, api } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";
import type { Job, QualityMode, Scene, Walkthrough } from "@/lib/types";
import { cn, formatBytes } from "@/lib/utils";

const ACCEPT = ".mp4,.m4v,.mov,.webm,video/mp4,video/quicktime,video/webm";

interface RoomDraft {
  id: string;
  name: string;
  sceneId: string | null;
  files: File[];
}

export interface AddWalkthroughDialogProps {
  propertyId: string;
  scenes: Scene[];
  open: boolean;
  onClose: () => void;
  onDone: (walkthrough: Walkthrough) => void;
}

let draftSeq = 0;
const newDraft = (name: string): RoomDraft => ({ id: `d${(draftSeq += 1)}`, name, sceneId: null, files: [] });

export function AddWalkthroughDialog({ propertyId, scenes, open, onClose, onDone }: AddWalkthroughDialogProps) {
  const { locale } = useLocale();
  const describeError = useAppStore((s) => s.describeError);
  const [quality, setQuality] = useState<QualityMode>("balanced");
  const [base, setBase] = useState<File[]>([]);
  const [rooms, setRooms] = useState<RoomDraft[]>(() => [newDraft(ui("walk.room.placeholder", "tr").replace("{n}", "1"))]);
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    if (open) return;
    setBase([]);
    setRooms([newDraft(ui("walk.room.placeholder", locale).replace("{n}", "1"))]);
    setUploaded(0);
    setJob(null);
  }, [open, locale]);

  const usedFilenames = useMemo(() => {
    const names = new Map<string, number>();
    for (const f of [...base, ...rooms.flatMap((r) => r.files)]) names.set(f.name, (names.get(f.name) ?? 0) + 1);
    return [...names.entries()].filter(([, n]) => n > 1).map(([name]) => name);
  }, [base, rooms]);
  const totalBytes = useMemo(() => [...base, ...rooms.flatMap((r) => r.files)].reduce((n, f) => n + f.size, 0), [base, rooms]);
  const roomsWithVideo = rooms.filter((r) => r.files.length > 0 && r.name.trim());
  const needsBase = roomsWithVideo.length > 1 && base.length === 0;
  const canStart = !uploading && !needsBase && usedFilenames.length === 0 && (base.length > 0 || roomsWithVideo.length > 0);

  const start = async () => {
    if (!canStart) return;
    setUploading(true);
    setUploaded(0);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const result = await api.walkthrough.create(
        propertyId,
        base,
        roomsWithVideo.map((r) => ({ name: r.name.trim(), scene_id: r.sceneId, files: r.files })),
        quality,
        setUploaded,
        controller.signal,
      );
      setJob(result.job);
    } catch (cause) {
      if (!(cause instanceof BioSphereError && cause.detail.code === "aborted")) toast.error(describeError(cause));
    } finally {
      setUploading(false);
      abortRef.current = null;
    }
  };

  const roomOptions = [
    { value: "", label: ui("walk.room.new", locale) },
    ...scenes.map((s) => ({ value: s.id, label: s.name })),
  ];

  return (
    <Dialog
      open={open}
      onClose={() => {
        abortRef.current?.abort();
        onClose();
      }}
      title={ui("walk.add.title", locale)}
      description={job ? undefined : ui("walk.add.desc", locale)}
      size="lg"
      closeLabel={ui("common.close", locale)}
    >
      {job ? (
        <JobProgress job={job} propertyId={propertyId} sceneId={null} qualityMode={quality} autoNavigate={false} onWalkthroughDone={onDone} className="border-0 bg-transparent backdrop-blur-none" />
      ) : (
        <div className="space-y-5">
          <section>
            <h3 className="mb-1 text-[12px] font-semibold tracking-[0.12em] text-ink-muted uppercase">{ui("walk.base.title", locale)}</h3>
            <p className="mb-2 text-[12px] text-ink-faint">{ui("walk.base.hint", locale)}</p>
            <FilePicker files={base} onChange={setBase} disabled={uploading} label={ui("walk.base.pick", locale)} />
          </section>

          <section>
            <div className="mb-2 flex items-center justify-between">
              <div>
                <h3 className="text-[12px] font-semibold tracking-[0.12em] text-ink-muted uppercase">{ui("walk.rooms.title", locale)}</h3>
                <p className="text-[12px] text-ink-faint">{ui("walk.rooms.hint", locale)}</p>
              </div>
              <Button variant="ghost" size="sm" disabled={uploading} onClick={() => setRooms((r) => [...r, newDraft(ui("walk.room.placeholder", locale).replace("{n}", String(r.length + 1)))])}>
                <Plus className="size-3.5" aria-hidden />
                {ui("walk.rooms.add", locale)}
              </Button>
            </div>
            <div className="space-y-3">
              {rooms.map((room) => (
                <div key={room.id} className="rounded-md border border-line bg-surface-2/40 p-3">
                  <div className="grid gap-2 sm:grid-cols-[1fr_180px_auto]">
                    <Input
                      value={room.name}
                      disabled={uploading || room.sceneId !== null}
                      onChange={(e) => setRooms((rs) => rs.map((r) => (r.id === room.id ? { ...r, name: e.target.value } : r)))}
                      placeholder={t("scene.name", locale)}
                      aria-label={t("scene.name", locale)}
                    />
                    <Select
                      options={roomOptions}
                      value={room.sceneId ?? ""}
                      disabled={uploading}
                      aria-label={ui("walk.room.existing", locale)}
                      onChange={(e) => {
                        const id = e.target.value || null;
                        const scene = scenes.find((s) => s.id === id);
                        setRooms((rs) => rs.map((r) => (r.id === room.id ? { ...r, sceneId: id, name: scene ? scene.name : r.name } : r)));
                      }}
                    />
                    <Button variant="ghost" size="icon" aria-label={ui("upload.remove", locale)} disabled={uploading || rooms.length === 1} onClick={() => setRooms((rs) => rs.filter((r) => r.id !== room.id))}>
                      <Trash2 className="size-4" aria-hidden />
                    </Button>
                  </div>
                  <div className="mt-2">
                    <FilePicker files={room.files} disabled={uploading} label={ui("walk.room.pick", locale)} onChange={(files) => setRooms((rs) => rs.map((r) => (r.id === room.id ? { ...r, files } : r)))} />
                  </div>
                </div>
              ))}
            </div>
          </section>

          <QualitySelector value={quality} onChange={setQuality} label={ui("wizard.quality", locale)} flavour="walk" />

          {needsBase ? <p className="text-[12px] text-amber">{ui("walk.needsBase", locale)}</p> : null}
          {usedFilenames.length ? <p className="text-[12px] text-danger">{ui("walk.duplicateNames", locale)} {usedFilenames.join(", ")}</p> : null}

          {uploading ? (
            <div className="rounded-md border border-line bg-surface-2/40 p-4">
              <div className="mb-2 flex items-center justify-between text-[12px]">
                <span className="text-ink-muted">{t("upload.uploading", locale)} · {formatBytes(totalBytes)}</span>
                <span className="font-mono text-ink">{Math.round(uploaded * 100)}%</span>
              </div>
              <Progress value={uploaded} label={t("upload.uploading", locale)} />
              <div className="mt-3 flex justify-end">
                <Button variant="danger" size="sm" onClick={() => abortRef.current?.abort()}>{t("job.cancel", locale)}</Button>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-between gap-3">
              <p className="text-[12px] text-ink-faint">{totalBytes ? formatBytes(totalBytes) : ""}</p>
              <Button variant="primary" disabled={!canStart} onClick={() => void start()}>
                <UploadCloud className="size-4" aria-hidden />
                {ui("walk.start", locale)}
              </Button>
            </div>
          )}
        </div>
      )}
    </Dialog>
  );
}

function FilePicker({ files, onChange, disabled, label }: { files: File[]; onChange: (files: File[]) => void; disabled?: boolean; label: string }) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  return (
    <div className={cn("flex flex-wrap items-center gap-2", disabled ? "opacity-60" : "")}>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        multiple
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const picked = Array.from(e.target.files ?? []).sort((a, b) => a.name.localeCompare(b.name));
          if (picked.length) onChange(picked);
          e.target.value = "";
        }}
      />
      <Button variant="secondary" size="sm" disabled={disabled} onClick={() => inputRef.current?.click()}>
        <Box className="size-3.5" aria-hidden />
        {label}
      </Button>
      {files.length ? (
        <ul className="flex flex-wrap gap-1.5">
          {files.map((f) => (
            <li key={f.name} className="flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1 text-[11px] text-ink-muted">
              <span className="max-w-[160px] truncate">{f.name}</span>
              <span className="font-mono text-ink-faint">{formatBytes(f.size)}</span>
              {!disabled ? (
                <button type="button" aria-label="remove" className="text-ink-faint hover:text-danger" onClick={() => onChange(files.filter((x) => x !== f))}>
                  ×
                </button>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
