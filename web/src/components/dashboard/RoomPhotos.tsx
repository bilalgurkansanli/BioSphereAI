"use client";

/**
 * A room's ordinary photographs: drop them in, caption them, take them out.
 *
 * ── WHY A ROOM HAS THESE AT ALL ───────────────────────────────────────────────────────
 * A listing is not one 360 per room, it is a room. The agent shoots the panorama and also
 * takes four stills of the same space, and a portal wants both. Keeping them on the room —
 * rather than in a separate album beside the property — is what lets anything downstream (an
 * export, a portal upload) ask one question: what have we got for the child's bedroom?
 *
 * ── THE UPLOAD IS OPTIMISTIC ABOUT NOTHING ────────────────────────────────────────────
 * No placeholder tiles appear before the server has the bytes. A photograph that shows up and
 * then vanishes because the file was corrupt is worse than one that takes a second to appear,
 * and the server deliberately accepts a partial batch — four good files out of five — so the
 * only honest source of truth for what was stored is its answer.
 */

import { ImagePlus, Loader2, Trash2, Wand2, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import { toast } from "sonner";
import { RoomPhotoStudio } from "./RoomPhotoStudio";
import { ui } from "./copy";
import { Button } from "@/components/ui/Button";
import { BioSphereError, absoluteUrl, api } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import type { Locale, RestyleStyle, ScenePhoto } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface RoomPhotosProps {
  sceneId: string;
  photos: ScenePhoto[];
  onChange: (photos: ScenePhoto[]) => void;
  className?: string;
}

export function RoomPhotos({ sceneId, photos, onChange, className }: RoomPhotosProps) {
  const { locale } = useLocale();
  const [busy, setBusy] = useState(false);
  const [removing, setRemoving] = useState<string | null>(null);
  const [studio, setStudio] = useState<ScenePhoto | null>(null);

  /**
   * The decorating styles, fetched once and only if the server can actually run them.
   *
   * An empty list is the honest answer to "this box has no GPU": the tiles then carry no
   * wand at all, rather than one that fails when pressed.
   */
  const [styles, setStyles] = useState<RestyleStyle[]>([]);
  useEffect(() => {
    let cancelled = false;
    void api.restyle
      .styles()
      .then((body) => {
        if (!cancelled && body.available) setStyles(body.styles);
      })
      .catch(() => {
        /* No catalogue means no tools. Nothing to report to someone managing photographs. */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const accept = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      setBusy(true);
      try {
        const created = await api.photos.add(sceneId, files);
        onChange([...photos, ...created]);
        const skipped = files.length - created.length;
        if (skipped > 0) toast.warning(ui("photos.skipped", locale).replace("{n}", String(skipped)));
      } catch (error) {
        toast.error(
          error instanceof BioSphereError ? error.localised(locale) : ui("photos.failed", locale),
        );
      } finally {
        setBusy(false);
      }
    },
    [sceneId, photos, onChange, locale],
  );

  const remove = useCallback(
    async (photo: ScenePhoto) => {
      setRemoving(photo.id);
      try {
        await api.photos.remove(photo.id);
        onChange(photos.filter((p) => p.id !== photo.id));
        toast.success(ui("photos.deleted", locale));
      } catch (error) {
        toast.error(
          error instanceof BioSphereError ? error.localised(locale) : ui("photos.failed", locale),
        );
      } finally {
        setRemoving(null);
      }
    },
    [photos, onChange, locale],
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    accept: { "image/jpeg": [".jpg", ".jpeg"], "image/png": [".png"], "image/webp": [".webp"] },
    noClick: true,
    disabled: busy,
    onDropAccepted: (files) => void accept(files),
    onDropRejected: () => toast.error(ui("photos.failed", locale)),
  });

  return (
    <div {...getRootProps()} className={cn("space-y-3 outline-none", className)}>
      <input {...getInputProps()} />

      <div
        className={cn(
          "grid place-items-center rounded-lg border border-dashed px-4 py-8 text-center transition-colors",
          isDragActive ? "border-bio bg-bio-dim/40" : "border-line bg-canvas-raised",
        )}
      >
        <div className="flex flex-col items-center gap-2">
          <span className="grid size-10 place-items-center rounded-full bg-surface-2 text-bio">
            {busy ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : (
              <ImagePlus className="size-4" aria-hidden />
            )}
          </span>
          <p className="text-[13px] font-medium text-ink">
            {busy ? ui("photos.uploading", locale) : ui("photos.drop", locale)}
          </p>
          <p className="text-[11px] text-ink-faint">{ui("photos.hint", locale)}</p>
          {!busy ? (
            <Button size="sm" variant="outline" onClick={open}>
              {ui("photos.choose", locale)}
            </Button>
          ) : null}
        </div>
      </div>

      {photos.length === 0 ? (
        <p className="py-2 text-center text-[12px] text-ink-faint">{ui("photos.empty", locale)}</p>
      ) : (
        <>
          <p className="text-[11px] text-ink-faint">
            {photos.length} {ui("photos.count", locale)}
          </p>
          <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {photos.map((photo) => (
              <PhotoTile
                key={photo.id}
                photo={photo}
                locale={locale}
                removing={removing === photo.id}
                onEdit={styles.length > 0 ? () => setStudio(photo) : undefined}
                onRemove={() => void remove(photo)}
                onCaption={(caption) =>
                  onChange(photos.map((p) => (p.id === photo.id ? { ...p, caption } : p)))
                }
              />
            ))}
          </ul>
        </>
      )}

      <RoomPhotoStudio photo={studio} styles={styles} onClose={() => setStudio(null)} />
    </div>
  );
}

interface PhotoTileProps {
  photo: ScenePhoto;
  locale: Locale;
  removing: boolean;
  /** Absent when the server cannot run the tools; the tile then has no wand. */
  onEdit?: () => void;
  onRemove: () => void;
  onCaption: (caption: string | null) => void;
}

function PhotoTile({ photo, locale, removing, onEdit, onRemove, onCaption }: PhotoTileProps) {
  const url = absoluteUrl(photo.url);
  const [draft, setDraft] = useState(photo.caption ?? "");
  // The last value the server confirmed, so a blur that changed nothing does not spend a
  // request — this fires on every tile of a twenty-photo room otherwise.
  const savedRef = useRef(photo.caption ?? "");

  const commit = useCallback(() => {
    const next = draft.trim();
    if (next === savedRef.current) return;
    savedRef.current = next;
    onCaption(next || null);
    void api.photos
      .update(photo.id, { caption: next || null })
      .catch((error: unknown) =>
        toast.error(
          error instanceof BioSphereError ? error.localised(locale) : ui("photos.failed", locale),
        ),
      );
  }, [draft, photo.id, onCaption, locale]);

  return (
    <li className="group relative overflow-hidden rounded-md border border-line bg-surface-2">
      <div className="relative aspect-[4/3]">
        {url ? (
          // Served by the engine at the size it chose; next/image would proxy and re-encode it.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={url}
            alt={photo.caption ?? ""}
            loading="lazy"
            className="h-full w-full object-cover"
          />
        ) : null}
        {/* Both actions live in one corner rail. Revealed on hover so the grid reads as
            photographs rather than as a row of controls, but focus-visible brings them back
            for anybody arriving by keyboard. */}
        <div className="absolute top-1.5 right-1.5 flex gap-1 opacity-0 transition-opacity
                        group-focus-within:opacity-100 group-hover:opacity-100">
          {onEdit ? (
            <button
              type="button"
              onClick={onEdit}
              aria-label={ui("photos.studio.open", locale)}
              title={ui("photos.studio.open", locale)}
              className="grid size-7 place-items-center rounded-full bg-black/60 text-white/90
                         hover:bg-bio/80 focus-visible:opacity-100"
            >
              <Wand2 className="size-3.5" aria-hidden />
            </button>
          ) : null}
          <button
            type="button"
            onClick={onRemove}
            disabled={removing}
            aria-label={ui("photos.delete", locale)}
            title={ui("photos.delete", locale)}
            className="grid size-7 place-items-center rounded-full bg-black/60 text-white/90
                       hover:bg-danger/80 focus-visible:opacity-100"
          >
            {removing ? (
              <Loader2 className="size-3.5 animate-spin" aria-hidden />
            ) : (
              <Trash2 className="size-3.5" aria-hidden />
            )}
          </button>
        </div>
      </div>
      <div className="relative">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
            if (e.key === "Escape") {
              setDraft(savedRef.current);
              e.currentTarget.blur();
            }
          }}
          placeholder={ui("photos.caption", locale)}
          className="w-full bg-transparent px-2 py-1.5 pr-6 text-[11px] text-ink placeholder:text-ink-faint
                     focus:bg-surface-3 focus:outline-none"
        />
        {draft ? (
          <button
            type="button"
            onClick={() => {
              setDraft("");
              savedRef.current = "";
              onCaption(null);
              void api.photos.update(photo.id, { caption: null }).catch(() => undefined);
            }}
            aria-label={ui("photos.caption", locale)}
            className="absolute top-1/2 right-1 -translate-y-1/2 text-ink-faint hover:text-ink"
          >
            <X className="size-3" aria-hidden />
          </button>
        ) : null}
      </div>
    </li>
  );
}
