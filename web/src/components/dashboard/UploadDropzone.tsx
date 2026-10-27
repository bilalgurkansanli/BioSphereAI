"use client";

import { Film, ImageIcon, TriangleAlert, UploadCloud, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useDropzone, type Accept, type FileRejection } from "react-dropzone";
import { ui } from "./copy";
import { Button } from "@/components/ui/Button";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import type { CaptureKind } from "@/lib/types";
import { cn, formatBytes, formatDuration } from "@/lib/utils";

/**
 * Matches the limits quoted in `upload.hint`, and the engine's own.
 *
 * `MAX_BYTES` is per FILE and sits under `Settings.max_upload_bytes` (1536 MB) so a
 * multi-segment capture still clears the request as a whole. `MAX_SECONDS` mirrors
 * `plan.MAX_CAPTURE_MS`: the guided capture will now record for ten minutes, so a dropzone
 * that called anything past two "too long" was warning about its own app's output.
 *
 * Both duration checks are advisory — the engine decides — and both are measured across the
 * WHOLE selection, because a capture legitimately arrives in several segments.
 */
const MAX_BYTES = 700 * 1024 * 1024;
const MIN_SECONDS = 5;
const MAX_SECONDS = 600;
/**
 * The engine's limit is per FILE (`Settings.max_video_seconds`, checked in `storage.py` on
 * every file of a capture). The guided capture never trips it — its recorder seals segments far
 * shorter — but a single long video picked from a gallery does, and the engine can only say so
 * after the whole file has been uploaded. Warn before the upload instead.
 */
const MAX_FILE_SECONDS = 300;

const VIDEO_ACCEPT: Accept = {
  "video/mp4": [".mp4", ".m4v"],
  "video/quicktime": [".mov"],
  "video/webm": [".webm"],
};

const PHOTO_ACCEPT: Accept = {
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
  "image/webp": [".webp"],
};

export interface MediaSelection {
  files: File[];
  kind: CaptureKind;
}

export interface UploadDropzoneProps {
  selection: MediaSelection | null;
  onSelect: (selection: MediaSelection | null) => void;
  disabled?: boolean;
  className?: string;
}

export function UploadDropzone({ selection, onSelect, disabled = false, className }: UploadDropzoneProps) {
  const { locale } = useLocale();
  const [kind, setKind] = useState<CaptureKind>("video");
  const [rejection, setRejection] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  /**
   * Seconds per file, keyed by the `File` itself rather than by position.
   *
   * Keying on identity means the map only ever accumulates, so nothing has to be cleared
   * when the selection changes — a stale entry simply stops being looked up. Keying on an
   * index would need a reset, and resetting state from inside the probing effect is both a
   * lint error here and a real flicker: for one frame the selection would have files and no
   * measurements, which reads as "too short".
   */
  const [durationByFile, setDurationByFile] = useState<ReadonlyMap<File, number>>(() => new Map());
  const [durationFailed, setDurationFailed] = useState(false);

  const files = selection?.files ?? [];
  const file = files[0] ?? null;
  const measured = files.map((f) => durationByFile.get(f)).filter((d): d is number => d !== undefined);
  // Only once every file has answered: a partial sum would read as "too short" and warn the
  // user about a capture that is fine, while the second segment is still being read.
  const duration = selection !== null && measured.length === files.length && files.length > 0
    ? measured.reduce((a, b) => a + b, 0)
    : null;

  // The preview URL is created here and revoked here — nothing else may outlive it.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setDurationFailed(false);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // One hidden probe per file. `<video preload="metadata">` reads only the header, so this
  // costs a few kilobytes each rather than decoding anything.
  useEffect(() => {
    if (selection === null || selection.kind !== "video" || typeof document === "undefined") return;
    const urls: string[] = [];
    const probes: HTMLVideoElement[] = [];
    for (const f of selection.files) {
      const url = URL.createObjectURL(f);
      urls.push(url);
      const probe = document.createElement("video");
      probe.preload = "metadata";
      probe.muted = true;
      probe.onloadedmetadata = () => {
        const value = probe.duration;
        if (Number.isFinite(value) && value > 0) {
          setDurationByFile((prev) => new Map(prev).set(f, value));
        } else {
          // A browser-muxed WebM declares no duration; the engine measures it by demuxing.
          // Saying so beats reporting a length that is not there.
          setDurationFailed(true);
        }
      };
      probe.onerror = () => setDurationFailed(true);
      probe.src = url;
      probes.push(probe);
    }
    return () => {
      for (const probe of probes) {
        probe.onloadedmetadata = null;
        probe.onerror = null;
        probe.removeAttribute("src");
      }
      for (const url of urls) URL.revokeObjectURL(url);
    };
  }, [selection]);

  const onDrop = useCallback(
    (accepted: File[], rejections: FileRejection[]) => {
      if (rejections.length > 0) {
        const tooBig = rejections.some((r) => r.errors.some((e) => e.code === "file-too-large"));
        setRejection(tooBig ? ui("upload.tooBig", locale) : ui("upload.rejected", locale));
      } else {
        setRejection(null);
      }
      if (accepted.length === 0) return;
      // ALL of them, video included. A guided capture is legitimately several segments — the
      // recorder seals one at its ceiling, a call ends the track, a stall restarts the
      // encoder — and the "save to my phone" escape hatch writes one file per segment. This
      // used to keep only the first, so a user rescuing a two-minute capture that had been
      // split would upload its four-second opening and nothing else. Order is the order the
      // picker gives, which for `capture-1.webm`, `capture-2.webm` is the recording order.
      onSelect({ files: [...accepted].sort((a, b) => a.name.localeCompare(b.name)), kind });
    },
    [kind, locale, onSelect],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: kind === "video" ? VIDEO_ACCEPT : PHOTO_ACCEPT,
    multiple: true,
    maxSize: MAX_BYTES,
    disabled,
  });

  const switchKind = (next: CaptureKind) => {
    if (next === kind) return;
    setKind(next);
    setRejection(null);
    onSelect(null);
  };

  const tooShort = duration !== null && duration < MIN_SECONDS;
  const tooLong = duration !== null && duration > MAX_SECONDS;
  const fileTooLong = measured.some((seconds) => seconds > MAX_FILE_SECONDS);

  return (
    <div className={cn("space-y-4", className)}>
      <div role="group" className="flex w-fit items-center gap-0.5 rounded-full border border-line bg-surface-2/60 p-0.5">
        {[
          { value: "video" as const, icon: Film, label: "Video" },
          { value: "photos" as const, icon: ImageIcon, label: t("upload.photos", locale) },
        ].map((option) => (
          <button
            key={option.value}
            type="button"
            aria-pressed={kind === option.value}
            disabled={disabled}
            onClick={() => switchKind(option.value)}
            className={cn(
              "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[12px] font-medium transition-colors",
              kind === option.value ? "bg-surface-3 text-ink" : "text-ink-faint hover:text-ink-muted",
            )}
          >
            <option.icon className="size-3.5" aria-hidden />
            {option.label}
          </button>
        ))}
      </div>

      {!selection ? (
        <div
          {...getRootProps({
            className: cn(
              "relative grid cursor-pointer place-items-center rounded-lg border border-dashed px-6 py-12 text-center transition-colors duration-200",
              isDragActive ? "border-bio/60 bg-bio/5" : "border-line-strong bg-surface-2/30 hover:border-bio/40 hover:bg-surface-2/60",
              disabled && "pointer-events-none opacity-50",
            ),
          })}
        >
          <input {...getInputProps()} />
          <div
            className={cn(
              "mb-4 grid size-12 place-items-center rounded-xl border transition-colors",
              isDragActive ? "border-bio/40 bg-bio/10 text-bio" : "border-line bg-surface-2 text-ink-faint",
            )}
          >
            <UploadCloud className="size-5" strokeWidth={1.7} aria-hidden />
          </div>
          <p className="text-sm font-medium text-ink">
            {isDragActive ? ui("upload.release", locale) : t("upload.drop", locale)}
          </p>
          <p className="mt-1 text-[13px] text-ink-muted">{t("upload.or", locale)}</p>
          <p className="mt-4 font-mono text-[11px] text-ink-faint">
            {kind === "video" ? t("upload.hint", locale) : ui("upload.photosHint", locale)}
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-4 rounded-lg border border-line bg-surface-2/40 p-4 sm:flex-row">
          <div className="relative aspect-video w-full shrink-0 overflow-hidden rounded-md border border-line bg-canvas sm:w-44">
            {previewUrl && kind === "video" ? (
              <video
                src={previewUrl}
                preload="metadata"
                muted
                playsInline
                className="size-full object-cover"
              />
            ) : previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={previewUrl} alt="" className="size-full object-cover" />
            ) : null}
          </div>

          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate text-[13px] font-medium text-ink">{file?.name}</p>
                <p className="mt-0.5 font-mono text-[11px] text-ink-faint">
                  {formatBytes(selection.files.reduce((sum, f) => sum + f.size, 0))}
                  {selection.kind === "photos" ? ` · ${selection.files.length} ${ui("upload.photosCount", locale)}` : null}
                  {selection.kind === "video" && selection.files.length > 1
                    ? ` · ${selection.files.length} ${locale === "tr" ? "parça" : "segments"}`
                    : null}
                  {selection.kind === "video" && duration !== null
                    ? ` · ${ui("upload.duration", locale)} ${formatDuration(duration)}`
                    : null}
                  {selection.kind === "video" && duration === null && !durationFailed
                    ? ` · ${ui("upload.reading", locale)}`
                    : null}
                </p>
              </div>
              <Button variant="ghost" size="sm" disabled={disabled} onClick={() => onSelect(null)}>
                <X className="size-3.5" aria-hidden />
                {ui("upload.remove", locale)}
              </Button>
            </div>

            {tooShort || tooLong || fileTooLong || durationFailed ? (
              <p className="mt-3 flex items-start gap-2 rounded-sm border border-amber/20 bg-amber/5 px-3 py-2 text-[12px] leading-relaxed text-amber">
                <TriangleAlert className="mt-px size-3.5 shrink-0" aria-hidden />
                {fileTooLong
                  ? ui("upload.fileTooLong", locale)
                  : tooShort
                  ? ui("upload.tooShort", locale)
                  : tooLong
                    ? ui("upload.tooLong", locale)
                    : ui("upload.durationUnknown", locale)}
              </p>
            ) : null}
          </div>
        </div>
      )}

      {rejection ? (
        <p role="alert" className="text-[12px] text-danger">
          {rejection}
        </p>
      ) : null}
    </div>
  );
}
