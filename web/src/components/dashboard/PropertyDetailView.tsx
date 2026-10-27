"use client";

import {
  Box,
  DoorOpen,
  Eye,
  Images,
  Layers,
  PencilLine,
  Plus,
  RotateCcw,
  Smartphone,
  Tag,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";
import { AddSceneDialog } from "./AddSceneDialog";
import { AddWalkthroughDialog } from "./AddWalkthroughDialog";
import { useCrumbLabel } from "./Breadcrumbs";
import { EmbedPanel } from "./EmbedPanel";
import { ApiErrorState } from "./ErrorState";
import { InvestmentPanel } from "./InvestmentPanel";
import { ListingPanel } from "./ListingPanel";
import { LogoMark } from "./Logo";
import { PlanPanel } from "./PlanPanel";
import { ui } from "./copy";
import { Badge, type BadgeTone } from "@/components/ui/Badge";
import { Button, buttonStyles } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { Input } from "@/components/ui/Input";
import { DropdownMenu } from "@/components/ui/DropdownMenu";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { Switch } from "@/components/ui/Switch";
import { BioSphereError, absoluteUrl, api } from "@/lib/api";
import { useApi } from "@/lib/hooks/useApi";
import { useLocale } from "@/lib/hooks/useLocale";
import { qualityLabel, t, type StringKey } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";
import type { Scene, SceneStatus, PropertyDetail } from "@/lib/types";
import { relativeTime } from "@/lib/utils";

const STATUS: Record<SceneStatus, { tone: BadgeTone; key: StringKey; pulse?: boolean }> = {
  ready: { tone: "bio", key: "scene.ready" },
  processing: { tone: "sky", key: "scene.processing", pulse: true },
  pending: { tone: "neutral", key: "scene.pending" },
  failed: { tone: "danger", key: "scene.failed" },
};

function SceneTile({
  scene,
  propertyId,
  onReprocess,
  onDelete,
}: {
  scene: Scene;
  propertyId: string;
  onReprocess: (scene: Scene) => void;
  onDelete: (scene: Scene) => void;
}) {
  const { locale } = useLocale();
  const router = useRouter();
  // A room that only has a 3D area shows that area's picture and opens the walkthrough.
  const walkOnly = !scene.panorama && scene.walkthrough?.status === "ready";
  const thumb = absoluteUrl(scene.panorama?.preview_url ?? scene.walkthrough?.thumbnail_url);
  const status = STATUS[scene.status];
  const href = walkOnly ? `/evler/${propertyId}/walk?scene=${scene.id}` : `/evler/${propertyId}/scenes/${scene.id}`;

  return (
    <div className="glass group relative flex flex-col rounded-lg transition-all duration-200 ease-out-expo hover:border-line-strong hover:shadow-lift">
      <div className="relative aspect-[16/9] overflow-hidden rounded-t-lg border-b border-line bg-surface-2">
        {thumb ? (
          // The preview is a tiny blurred equirect straight from the engine origin.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={thumb}
            alt=""
            loading="lazy"
            decoding="async"
            className="size-full object-cover transition-transform duration-500 ease-out-expo group-hover:scale-[1.03]"
          />
        ) : (
          <div className="grid size-full place-items-center">
            <LogoMark className="size-8 text-ink-faint/40" />
          </div>
        )}
        <div className="absolute top-2.5 left-2.5 flex gap-1.5">
          <Badge tone={status.tone} dot pulse={status.pulse}>
            {t(status.key, locale)}
          </Badge>
          {scene.walkthrough?.status === "ready" ? (
            <Badge tone="sky">
              <Box className="size-3" aria-hidden />
              3B
            </Badge>
          ) : null}
        </div>
      </div>

      <div className="flex items-center justify-between gap-2 p-3.5">
        <div className="min-w-0">
          <h3 className="truncate text-[14px] font-medium text-ink">
            <Link href={href} className="after:absolute after:inset-0 after:content-['']">
              {scene.name}
            </Link>
          </h3>
          <p className="mt-0.5 flex items-center gap-2 text-[11px] text-ink-faint">
            <span>{qualityLabel(scene.quality_mode, locale)}</span>
            <span className="inline-flex items-center gap-1">
              <Tag className="size-3" aria-hidden />
              {scene.hotspots.length}
            </span>
          </p>
        </div>
        <div className="relative z-10">
          <DropdownMenu
            label={scene.name}
            items={[
              {
                id: "edit",
                label: ui("detail.scene.edit", locale),
                icon: PencilLine,
                onSelect: () => router.push(href),
              },
              {
                id: "reprocess",
                label: t("scene.reprocess", locale),
                icon: RotateCcw,
                onSelect: () => onReprocess(scene),
              },
              {
                id: "delete",
                label: t("editor.delete", locale),
                icon: Trash2,
                danger: true,
                separatorBefore: true,
                onSelect: () => onDelete(scene),
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}

export function PropertyDetailView({ propertyId }: { propertyId: string }) {
  const { locale } = useLocale();
  const router = useRouter();
  const describeError = useAppStore((s) => s.describeError);
  const { data: property, initial, error, refetch, setData } = useApi<PropertyDetail>(() => api.properties.get(propertyId), [propertyId]);

  useCrumbLabel(`/evler/${propertyId}`, property?.title);

  const [adding, setAdding] = useState(false);
  const [addingWalk, setAddingWalk] = useState(false);
  const [addingRoom, setAddingRoom] = useState(false);
  const [roomName, setRoomName] = useState("");
  const [creatingRoom, setCreatingRoom] = useState(false);

  /**
   * Create a room with nothing in it yet.
   *
   * Navigates straight into the editor afterwards rather than leaving the user on the list:
   * an empty room is not the goal, and the only reason to make one is to put photographs in
   * it, which is one tab away in there.
   */
  const createRoom = async () => {
    const name = roomName.trim();
    if (!name || !property) return;
    setCreatingRoom(true);
    try {
      const room = await api.properties.createRoom(property.id, { name });
      setAddingRoom(false);
      setRoomName("");
      toast.success(ui("room.added", locale));
      router.push(`/evler/${property.id}/scenes/${room.id}`);
    } catch (error) {
      toast.error(
        error instanceof BioSphereError ? error.localised(locale) : ui("photos.failed", locale),
      );
    } finally {
      setCreatingRoom(false);
    }
  };
  const [deleteFor, setDeleteFor] = useState<Scene | null>(null);
  const [busy, setBusy] = useState(false);
  const [publishing, setPublishing] = useState(false);

  const togglePublish = async (next: boolean) => {
    if (!property) return;
    setPublishing(true);
    setData((current) => (current ? { ...current, published: next } : current));
    try {
      await api.properties.update(property.id, { published: next });
      toast.success(ui(next ? "detail.published.on" : "detail.published.off", locale));
    } catch (cause) {
      // Roll the switch back so the UI never claims a state the server rejected.
      setData((current) => (current ? { ...current, published: !next } : current));
      toast.error(describeError(cause));
    } finally {
      setPublishing(false);
    }
  };

  const reprocess = async (scene: Scene) => {
    try {
      await api.scenes.reprocess(scene.id, scene.quality_mode);
      toast.success(ui("detail.reprocess.started", locale));
      refetch();
    } catch (cause) {
      toast.error(describeError(cause));
    }
  };

  const deleteScene = async () => {
    if (!deleteFor) return;
    const target = deleteFor;
    setBusy(true);
    try {
      await api.scenes.remove(target.id);
      setData((current) =>
        current
          ? { ...current, scenes: current.scenes.filter((s) => s.id !== target.id), scene_count: current.scene_count - 1 }
          : current,
      );
      toast.success(ui("detail.scene.deleted", locale));
      setDeleteFor(null);
    } catch (cause) {
      toast.error(describeError(cause));
    } finally {
      setBusy(false);
    }
  };

  if (initial) {
    return (
      <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="mt-3 h-4 w-40" />
        <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_336px]">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {Array.from({ length: 3 }, (_, i) => (
              <Skeleton key={i} className="aspect-[16/11] rounded-lg" />
            ))}
          </div>
          <Skeleton className="h-72 rounded-lg" />
        </div>
      </div>
    );
  }

  if (!property) {
    return (
      <div className="mx-auto w-full max-w-3xl px-4 py-16 sm:px-6">
        {error && error.status !== 404 ? (
          <ApiErrorState error={error} onRetry={refetch} />
        ) : (
          <EmptyState
            icon={Images}
            title={ui("detail.notFound.title", locale)}
            body={ui("detail.notFound.body", locale)}
            action={
              <Link href="/evler" className={buttonStyles("secondary", "md")}>
                {t("nav.properties", locale)}
              </Link>
            }
          />
        )}
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="mb-7 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2.5">
            <h1 className="text-2xl font-semibold tracking-tight text-ink">{property.title}</h1>
            <Badge tone={property.published ? "bio" : "neutral"} dot>
              {t(property.published ? "property.published" : "property.draft", locale)}
            </Badge>
          </div>
          {property.address ? <p className="mt-1.5 text-[13px] text-ink-muted">{property.address}</p> : null}
          <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-ink-faint">
            <span className="inline-flex items-center gap-1.5">
              <Layers className="size-3.5" aria-hidden />
              {property.scenes.length} {t("property.scenes", locale)}
            </span>
            <span className="inline-flex items-center gap-1.5">
              <Eye className="size-3.5" aria-hidden />
              {property.views} {t("property.views", locale)}
            </span>
            <span>
              {ui("detail.updated", locale)} {relativeTime(property.updated_at, locale)}
            </span>
          </div>
        </div>

        {/* Two ways to add a room, side by side because they are genuinely alternatives: the
            360 is the product, and a room somebody only photographed is still a room. */}
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={() => setAddingWalk(true)}>
            <Box className="size-4" aria-hidden />
            {ui("walk.add", locale)}
          </Button>
          <Button variant="secondary" onClick={() => setAddingRoom(true)}>
            <DoorOpen className="size-4" aria-hidden />
            {ui("room.add", locale)}
          </Button>
          <Button variant="primary" onClick={() => setAdding(true)}>
            <Plus className="size-4" aria-hidden />
            {t("scene.add", locale)}
          </Button>
        </div>
      </div>

      {/* `min-w-0` on both columns: the embed snippet is one unbreakable line, and without
          it the grid track grows to that line's width and the whole page scrolls sideways. */}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_336px]">
        <section aria-label={ui("detail.scenes", locale)} className="min-w-0">
          {property.scenes.length === 0 ? (
            <EmptyState
              icon={Images}
              title={ui("detail.empty.title", locale)}
              body={ui("detail.empty.body", locale)}
              action={
                <Button variant="primary" onClick={() => setAdding(true)}>
                  <Plus className="size-4" aria-hidden />
                  {t("scene.add", locale)}
                </Button>
              }
            />
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {property.scenes.map((scene) => (
                <SceneTile
                  key={scene.id}
                  scene={scene}
                  propertyId={property.id}
                  onReprocess={(s) => void reprocess(s)}
                  onDelete={setDeleteFor}
                />
              ))}

              <button
                type="button"
                onClick={() => setAdding(true)}
                className="group flex min-h-40 flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line-strong bg-surface-2/20 p-6 text-center transition-colors hover:border-bio/40 hover:bg-surface-2/50"
              >
                <span className="grid size-10 place-items-center rounded-full border border-line bg-surface-2 text-ink-faint transition-colors group-hover:border-bio/40 group-hover:text-bio">
                  <Plus className="size-4" aria-hidden />
                </span>
                <span className="text-[13px] font-medium text-ink">{t("scene.add", locale)}</span>
                <span className="text-[11px] text-ink-faint">{ui("detail.addScene.hint", locale)}</span>
              </button>
            </div>
          )}
        </section>

        <aside className="h-fit min-w-0 space-y-4 lg:sticky lg:top-20">
          <WalkCard property={property} onAdd={() => setAddingWalk(true)} onChanged={() => void refetch()} />

          <div className="glass rounded-lg p-5">
            <h2 className="mb-4 text-sm font-semibold tracking-tight text-ink">{ui("detail.share", locale)}</h2>
            <Switch
              checked={property.published}
              loading={publishing}
              onChange={(next) => void togglePublish(next)}
              label={t("property.published", locale)}
              description={ui("detail.publish.desc", locale)}
            />
          </div>

          <div className="glass rounded-lg p-5">
            <EmbedPanel property={property} />
          </div>
        </aside>
      </div>

      {/* Above the plan, and full width for the same reason: this is a paragraph somebody
          edits, and a textarea in a 336 px column is a paragraph nobody can read back. It
          sits before the plan because the plan is where its missing numbers come from — an
          agent who reads "no room has been measured" scrolls down to the button that fixes
          it. */}
      <ListingPanel property={property} locale={locale} onSaved={() => void refetch()} />

      {/* The investment assistant's figures sit beside the listing copy: both are what the
          agent tells the world about the flat, and both are full-width forms. */}
      <InvestmentPanel property={property} locale={locale} />

      {/* Full width rather than in the sidebar: a plan of a flat squeezed into a 336 px
          column is a plan nobody can read the room names off. */}
      <PlanPanel
        propertyId={property.id}
        locale={locale}
        onOpenRoom={(sceneId) => router.push(`/evler/${property.id}/scenes/${sceneId}`)}
      />

      <AddWalkthroughDialog
        propertyId={property.id}
        scenes={property.scenes}
        open={addingWalk}
        onClose={() => {
          setAddingWalk(false);
          void refetch();
        }}
        onDone={() => void refetch()}
      />

      <Dialog
        open={addingRoom}
        onClose={() => setAddingRoom(false)}
        title={ui("room.add.title", locale)}
        description={ui("room.add.body", locale)}
        size="sm"
        closeLabel={ui("common.close", locale)}
        footer={
          <>
            <Button variant="ghost" onClick={() => setAddingRoom(false)}>
              {ui("common.close", locale)}
            </Button>
            <Button
              variant="primary"
              loading={creatingRoom}
              disabled={roomName.trim().length === 0}
              onClick={() => void createRoom()}
            >
              {ui("room.add", locale)}
            </Button>
          </>
        }
      >
        <label className="block space-y-1.5">
          <span className="text-[12px] font-medium text-ink-muted">
            {ui("room.add.name", locale)}
          </span>
          <Input
            value={roomName}
            autoFocus
            placeholder={ui("room.add.placeholder", locale)}
            onChange={(e) => setRoomName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && roomName.trim()) void createRoom();
            }}
          />
        </label>
      </Dialog>

      <AddSceneDialog
        propertyId={property.id}
        open={adding}
        onClose={() => setAdding(false)}
        suggestedName={locale === "tr" ? `Sahne ${property.scenes.length + 1}` : `Scene ${property.scenes.length + 1}`}
        onSceneReady={() => refetch()}
      />

      <Dialog
        open={deleteFor !== null}
        onClose={() => setDeleteFor(null)}
        title={ui("detail.scene.delete.title", locale)}
        description={ui("detail.scene.delete.confirm", locale)}
        size="sm"
        closeLabel={ui("common.close", locale)}
        footer={
          <>
            <Button variant="ghost" onClick={() => setDeleteFor(null)}>
              {t("editor.cancel", locale)}
            </Button>
            <Button variant="danger" loading={busy} onClick={() => void deleteScene()}>
              {t("editor.delete", locale)}
            </Button>
          </>
        }
      >
        <p className="text-[13px] text-ink-muted">{deleteFor?.name}</p>
      </Dialog>
    </div>
  );
}


/** The 3D walkthrough's status, in the sidebar, with the way in and the way to rebuild it. */
function WalkCard({ property, onAdd, onChanged }: { property: PropertyDetail; onAdd: () => void; onChanged: () => void }) {
  const { locale } = useLocale();
  const describeError = useAppStore((s) => s.describeError);
  const [busy, setBusy] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const walk = property.walkthrough;

  const rebuild = async () => {
    setBusy(true);
    try {
      await api.walkthrough.rebuild(property.id, walk?.quality_mode ?? "");
      onChanged();
    } catch (cause) {
      toast.error(describeError(cause));
    } finally {
      setBusy(false);
    }
  };
  const remove = async () => {
    setBusy(true);
    try {
      await api.walkthrough.remove(property.id);
      setConfirmRemove(false);
      onChanged();
    } catch (cause) {
      toast.error(describeError(cause));
    } finally {
      setBusy(false);
    }
  };

  const tone: BadgeTone = !walk ? "neutral" : walk.status === "ready" ? "bio" : walk.status === "failed" ? "danger" : "sky";
  const rooms = walk?.manifest?.rooms.filter((r) => !r.failed).length ?? 0;
  const bytes = typeof walk?.metrics?.bytes_total === "number" ? (walk.metrics.bytes_total as number) : null;

  return (
    <div className="glass rounded-lg p-5">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold tracking-tight text-ink">{ui("walk.card.title", locale)}</h2>
        {walk ? (
          <Badge tone={tone} dot pulse={walk.status === "processing"}>
            {walk.status === "ready" ? ui("walk.card.ready", locale) : t(STATUS[walk.status].key, locale)}
          </Badge>
        ) : null}
      </div>
      {!walk ? (
        <>
          <p className="text-[13px] leading-relaxed text-ink-muted">{ui("walk.card.none", locale)}</p>
          <Button variant="primary" className="mt-4 w-full" onClick={onAdd}>
            <Box className="size-4" aria-hidden />
            {ui("walk.add", locale)}
          </Button>
          <Link href={`/cek?property=${property.id}&m=w`} className={buttonStyles("secondary", "md") + " mt-2 w-full"}>
            <Smartphone className="size-4" aria-hidden />
            {ui("walk.capturePhone", locale)}
          </Link>
        </>
      ) : walk.status === "processing" ? (
        <>
          <p className="text-[13px] leading-relaxed text-ink-muted">{ui("walk.card.processing", locale)}</p>
          <Link href={`/evler/${property.id}/walk`} className={buttonStyles("secondary", "md") + " mt-4 w-full"}>
            {ui("walk.open", locale)}
          </Link>
        </>
      ) : (
        <>
          {walk.status === "ready" ? (
            <>
              <p className="text-[12px] text-ink-faint">
                {rooms} {t("property.scenes", locale)}
                {bytes !== null ? ` · ${(bytes / 1e6).toFixed(1)} MB` : ""}
              </p>
              {walk.manifest?.warnings?.length ? (
                <details className="mt-2 text-[11.5px] text-ink-muted">
                  <summary className="cursor-pointer text-amber">
                    {ui("walk.card.warnings", locale)} ({walk.manifest.warnings.length})
                  </summary>
                  <ul className="mt-1.5 space-y-1 font-mono text-[10.5px] leading-snug text-ink-faint">
                    {walk.manifest.warnings.slice(0, 8).map((w) => (
                      <li key={w} className="break-words">
                        {w}
                      </li>
                    ))}
                  </ul>
                </details>
              ) : null}
            </>
          ) : (
            <p className="text-[13px] text-danger">
              {ui("walk.card.failed", locale)} {walk.error ? (locale === "tr" ? walk.error.message_tr : walk.error.message) : ""}
            </p>
          )}
          <div className="mt-4 flex flex-col gap-2">
            {walk.status === "ready" ? (
              <Link href={`/evler/${property.id}/walk`} className={buttonStyles("primary", "md") + " w-full"}>
                <Box className="size-4" aria-hidden />
                {ui("walk.open", locale)}
              </Link>
            ) : null}
            <div className="flex gap-2">
              <Button variant="secondary" size="sm" className="flex-1" loading={busy} onClick={() => void rebuild()}>
                <RotateCcw className="size-3.5" aria-hidden />
                {ui("walk.rebuild", locale)}
              </Button>
              <Button variant="secondary" size="sm" className="flex-1" onClick={onAdd}>
                <Plus className="size-3.5" aria-hidden />
                {ui("walk.add", locale)}
              </Button>
              <Link href={`/cek?property=${property.id}&m=w`} className={buttonStyles("secondary", "sm") + " flex-1"} title={ui("walk.capturePhone", locale)}>
                <Smartphone className="size-3.5" aria-hidden />
                {ui("walk.capturePhone.short", locale)}
              </Link>
              <Button variant="ghost" size="icon-sm" aria-label={ui("walk.remove", locale)} onClick={() => setConfirmRemove(true)}>
                <Trash2 className="size-4" aria-hidden />
              </Button>
            </div>
          </div>
        </>
      )}
      <Dialog
        open={confirmRemove}
        onClose={() => setConfirmRemove(false)}
        title={ui("walk.remove", locale)}
        description={ui("walk.remove.confirm", locale)}
        size="sm"
        closeLabel={ui("common.close", locale)}
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirmRemove(false)}>
              {ui("common.close", locale)}
            </Button>
            <Button variant="danger" loading={busy} onClick={() => void remove()}>
              {t("editor.delete", locale)}
            </Button>
          </>
        }
      />
    </div>
  );
}
