"use client";

import { ArrowLeft, Crosshair, Gauge, MapPin, PanelRightClose, PanelRightOpen, Plus, Tag, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { useCrumbLabel } from "./Breadcrumbs";
import { CoveragePanel } from "./CoveragePanel";
import { ApiErrorState } from "./ErrorState";
import { JobProgress } from "./JobProgress";
import { MetricsReport } from "./MetricsReport";
import { RoomPhotos } from "./RoomPhotos";
import { ui } from "./copy";
import { Badge } from "@/components/ui/Badge";
import { Button, buttonStyles } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs, tabPanelProps } from "@/components/ui/Tabs";
import { Textarea } from "@/components/ui/Textarea";
import { Player } from "@/components/player/Player";
import type { PlayerHandle } from "@/components/player/types";
import { api } from "@/lib/api";
import { useApi } from "@/lib/hooks/useApi";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";
import { NAVIGATION_TYPES } from "@/lib/types";
import type {
  CreateHotspotRequest,
  FloorPlan,
  Hotspot,
  HotspotType,
  InitialView,
  Job,
  ScenePhoto,
  PropertyDetail,
} from "@/lib/types";
import { cn, degrees } from "@/lib/utils";

type TabId = "tags" | "photos" | "view" | "coverage" | "report";

const DRAFT_PREFIX = "draft-";

function emptyHotspot(sceneId: string, yaw: number, pitch: number): Hotspot {
  return {
    id: `${DRAFT_PREFIX}${Date.now()}`,
    scene_id: sceneId,
    type: "info",
    yaw,
    pitch,
    title: "",
    body: null,
    icon: null,
    target_scene_id: null,
    media_url: null,
    accent: null,
    created_at: new Date().toISOString(),
  };
}

/** Fetches the job behind a still-processing scene so the editor can show real progress. */
function ActiveJobPanel({ jobId, propertyId, sceneId }: { jobId: string; propertyId: string; sceneId: string }) {
  const { data } = useApi<Job>(() => api.jobs.get(jobId), [jobId]);
  if (!data) return <Skeleton className="h-64 w-full rounded-lg" />;
  return (
    <JobProgress job={data} propertyId={propertyId} sceneId={sceneId} qualityMode="balanced" autoNavigate={false} />
  );
}

export function SceneEditor({ propertyId, sceneId }: { propertyId: string; sceneId: string }) {
  const { locale } = useLocale();
  const describeError = useAppStore((s) => s.describeError);
  const router = useRouter();
  const playerRef = useRef<PlayerHandle>(null);

  const { data: property, initial, error, refetch, setData } = useApi<PropertyDetail>(() => api.properties.get(propertyId), [propertyId]);
  const scene = property?.scenes.find((s) => s.id === sceneId) ?? null;

  /**
   * The floor plan, for the player's own bird's-eye button.
   *
   * READ-ONLY HERE, AND THAT IS THE WHOLE DIFFERENCE from `PlanPanel`. Reading a plan is
   * arithmetic off disk and costs milliseconds; MEASURING one is a scene-parsing pass per
   * room and costs a GPU minute, which is why the engine splits the endpoint. The panel on
   * the property page is where an agent presses "measure" and drags rooms into place. This is
   * where they consult the result while standing in a room — so it asks for the cached copy
   * and nothing else.
   *
   * A failure is an ordinary answer, not an error to report: a property nobody has measured
   * has no drawing, the player finds no drawing and shows no button, and an agent who has not
   * built a plan yet is not owed a toast about it every time they open a room.
   */
  const { data: plan } = useApi<FloorPlan>(() => api.plan.get(propertyId), [propertyId]);

  /**
   * Write a room's photographs back into the cached property.
   *
   * The editor holds the whole `PropertyDetail` because the player needs every room, so an upload
   * has to land in that document rather than in a second list beside it — otherwise the tab
   * and the property would disagree until the next refetch, and that refetch is a network round
   * trip for something already known.
   */
  const setPhotos = useCallback(
    (id: string, photos: ScenePhoto[]) => {
      setData((current) =>
        current
          ? { ...current, scenes: current.scenes.map((s) => (s.id === id ? { ...s, photos } : s)) }
          : current,
      );
    },
    [setData],
  );

  useCrumbLabel(`/evler/${propertyId}`, property?.title);
  useCrumbLabel(`/evler/${propertyId}/scenes/${sceneId}`, scene?.name);

  const [tab, setTab] = useState<TabId>("tags");
  const [panelOpen, setPanelOpen] = useState(true);
  const [hotspots, setHotspots] = useState<Hotspot[]>([]);
  const [draft, setDraft] = useState<Hotspot | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [savingView, setSavingView] = useState(false);
  const [view, setView] = useState<InitialView | null>(null);

  const serverHotspots = scene?.hotspots;
  useEffect(() => {
    setHotspots(serverHotspots ?? []);
  }, [serverHotspots]);

  useEffect(() => {
    if (scene && !view) setView(scene.initial_view);
  }, [scene, view]);

  /**
   * The player renders `hotspotOverrides` on top of each scene's own list, so the scene it
   * receives carries every persisted tag *except* the one being edited — that one is the
   * override. Nothing is ever drawn twice, and an unsaved tag appears the instant it is
   * placed.
   */
  const scenesForPlayer = useMemo(() => {
    if (!property) return [];
    const visible = draft ? hotspots.filter((h) => h.id !== draft.id) : hotspots;
    return property.scenes.map((s) => (s.id === sceneId ? { ...s, hotspots: visible } : s));
  }, [property, hotspots, draft, sceneId]);

  const linkTargets = useMemo(
    () =>
      (property?.scenes ?? [])
        .filter((s) => s.id !== sceneId && s.status === "ready")
        .map((s) => ({ value: s.id, label: s.name })),
    [property, sceneId],
  );

  const handleSphereClick = useCallback(
    ({ yaw, pitch }: { yaw: number; pitch: number }) => {
      setDraft(emptyHotspot(sceneId, yaw, pitch));
      setFormError(null);
      setTab("tags");
      setPanelOpen(true);
    },
    [sceneId],
  );

  const handleViewChange = useCallback((next: InitialView) => {
    // Fires every frame while dragging; only a whole degree of change is worth a render.
    setView((prev) => {
      if (
        prev &&
        Math.round(degrees(prev.yaw)) === Math.round(degrees(next.yaw)) &&
        Math.round(degrees(prev.pitch)) === Math.round(degrees(next.pitch)) &&
        Math.round(prev.fov) === Math.round(next.fov)
      ) {
        return prev;
      }
      return next;
    });
  }, []);

  const persistMove = useCallback(
    async (hotspot: Hotspot, position: { yaw: number; pitch: number }) => {
      const previous = hotspots.find((h) => h.id === hotspot.id) ?? hotspot;
      setHotspots((current) => current.map((h) => (h.id === hotspot.id ? { ...h, ...position } : h)));
      try {
        const saved = await api.hotspots.update(hotspot.id, position);
        setHotspots((current) => current.map((h) => (h.id === saved.id ? saved : h)));
      } catch (cause) {
        setHotspots((current) => current.map((h) => (h.id === previous.id ? previous : h)));
        toast.error(describeError(cause));
      }
    },
    [hotspots, describeError],
  );

  const saveDraft = async () => {
    if (!draft) return;
    const title = draft.title.trim();
    if (!title) {
      setFormError(ui("editor.hotspot.titleRequired", locale));
      return;
    }
    if (draft.type === "link" && !draft.target_scene_id) {
      setFormError(ui("editor.hotspot.targetRequired", locale));
      return;
    }

    const isNew = draft.id.startsWith(DRAFT_PREFIX);
    const optimistic: Hotspot = { ...draft, title };
    const previous = hotspots.find((h) => h.id === draft.id) ?? null;

    setFormError(null);
    setSaving(true);
    setHotspots((current) =>
      isNew ? [...current, optimistic] : current.map((h) => (h.id === optimistic.id ? optimistic : h)),
    );
    setDraft(null);

    const body: CreateHotspotRequest = {
      type: optimistic.type,
      yaw: optimistic.yaw,
      pitch: optimistic.pitch,
      title,
      body: optimistic.body ?? undefined,
      target_scene_id: optimistic.target_scene_id ?? undefined,
      media_url: optimistic.media_url ?? undefined,
    };

    try {
      const saved = isNew ? await api.hotspots.create(sceneId, body) : await api.hotspots.update(optimistic.id, body);
      setHotspots((current) => current.map((h) => (h.id === optimistic.id ? saved : h)));
      toast.success(ui("editor.hotspot.saved", locale));
    } catch (cause) {
      // Put the list back the way the server still sees it, and reopen the form so the
      // agent's typing is not thrown away.
      setHotspots((current) =>
        isNew
          ? current.filter((h) => h.id !== optimistic.id)
          : current.map((h) => (h.id === optimistic.id && previous ? previous : h)),
      );
      setDraft(optimistic);
      toast.error(describeError(cause) || ui("editor.hotspot.saveFailed", locale));
    } finally {
      setSaving(false);
    }
  };

  const deleteHotspot = async (hotspot: Hotspot) => {
    if (hotspot.id.startsWith(DRAFT_PREFIX)) {
      setDraft(null);
      return;
    }
    const index = hotspots.findIndex((h) => h.id === hotspot.id);
    setHotspots((current) => current.filter((h) => h.id !== hotspot.id));
    if (draft?.id === hotspot.id) setDraft(null);

    try {
      await api.hotspots.remove(hotspot.id);
      toast.success(ui("editor.hotspot.deleted", locale));
    } catch (cause) {
      setHotspots((current) => {
        const restored = [...current];
        restored.splice(Math.max(0, index), 0, hotspot);
        return restored;
      });
      toast.error(describeError(cause) || ui("editor.hotspot.deleteFailed", locale));
    }
  };

  const saveInitialView = async () => {
    const next = playerRef.current?.getView();
    if (!next || !scene) return;
    setSavingView(true);
    try {
      const updated = await api.scenes.update(sceneId, { initial_view: next });
      setData((current) =>
        current ? { ...current, scenes: current.scenes.map((s) => (s.id === updated.id ? updated : s)) } : current,
      );
      toast.success(ui("editor.view.saved", locale));
    } catch (cause) {
      toast.error(describeError(cause) || ui("editor.view.failed", locale));
    } finally {
      setSavingView(false);
    }
  };

  if (initial) {
    return (
      <div className="p-4 sm:p-6">
        <Skeleton className="h-[70dvh] w-full rounded-lg" />
      </div>
    );
  }

  if (!scene) {
    return (
      <div className="mx-auto w-full max-w-3xl px-4 py-16 sm:px-6">
        {error && error.status !== 404 ? (
          <ApiErrorState error={error} onRetry={refetch} />
        ) : (
          <EmptyState
            icon={MapPin}
            title={ui("editor.scene.notFound", locale)}
            body={ui("detail.notFound.body", locale)}
            action={
              <Link href={`/evler/${propertyId}`} className={buttonStyles("secondary", "md")}>
                {ui("editor.back", locale)}
              </Link>
            }
          />
        )}
      </div>
    );
  }

  // A ROOM WITH NO 360 IS NOT A ROOM THAT IS STILL PROCESSING, and the guard below cannot
  // tell them apart on its own — both have `panorama === null`. One is waiting for a stitch
  // and the other never will be, so a photo-only room shown the "not ready yet, this page
  // will refresh itself" screen would wait for ever for something nobody started.
  if (scene.capture_kind === "none" && !scene.panorama) {
    return (
      <div className="mx-auto w-full max-w-3xl space-y-5 px-4 py-8 sm:px-6">
        <Link href={`/evler/${propertyId}`} className={buttonStyles("ghost", "sm", "-ml-2")}>
          <ArrowLeft className="size-4" aria-hidden />
          {ui("editor.back", locale)}
        </Link>
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-lg font-semibold tracking-tight text-ink">{scene.name}</h1>
            <Badge tone="neutral">{ui("room.nocapture", locale)}</Badge>
          </div>
          <p className="text-[13px] text-ink-muted">{ui("room.photoOnly.body", locale)}</p>
        </div>
        <RoomPhotos
          sceneId={scene.id}
          photos={scene.photos}
          onChange={(photos) => setPhotos(scene.id, photos)}
        />
      </div>
    );
  }

  if (scene.status !== "ready" || !scene.panorama) {
    return (
      <div className="mx-auto w-full max-w-3xl space-y-5 px-4 py-8 sm:px-6">
        <Link href={`/evler/${propertyId}`} className={buttonStyles("ghost", "sm", "-ml-2")}>
          <ArrowLeft className="size-4" aria-hidden />
          {ui("editor.back", locale)}
        </Link>
        {scene.active_job_id ? (
          <ActiveJobPanel jobId={scene.active_job_id} propertyId={propertyId} sceneId={sceneId} />
        ) : (
          <EmptyState
            icon={Gauge}
            tone={scene.status === "failed" ? "danger" : "neutral"}
            title={ui("editor.notReady.title", locale)}
            body={scene.error ? (locale === "tr" ? scene.error.message_tr : scene.error.message) : ui("editor.notReady.body", locale)}
            action={
              <Button variant="secondary" onClick={refetch}>
                {t("job.retry", locale)}
              </Button>
            }
          />
        )}
      </div>
    );
  }

  // No icons here: four labels plus the collapse button already fill a 380 px panel, and a
  // horizontally scrolling tab strip is worse than a plain one.
  const tabs = [
    { id: "tags", label: ui("editor.tab.tags", locale) },
    {
      id: "photos",
      label: ui("photos.tab", locale),
      // The count is the useful thing to see without opening the tab: an agent checking a
      // property before publishing wants to know which rooms still have none.
      badge: scene && scene.photos.length > 0 ? scene.photos.length : undefined,
    },
    { id: "view", label: ui("editor.tab.view", locale) },
    { id: "coverage", label: ui("editor.tab.coverage", locale) },
    { id: "report", label: ui("editor.tab.report", locale) },
  ];

  return (
    <div className="relative h-[calc(100dvh-7.5rem)] overflow-hidden md:h-[calc(100dvh-3.5rem)]">
      <Player
        ref={playerRef}
        scenes={scenesForPlayer}
        initialSceneId={sceneId}
        locale={locale}
        editable
        showChrome={false}
        // The editor is first-party and same-origin, which is what the AI routes' CORS
        // policy requires; the public embed deliberately does not get these.
        tools
        // The bird's-eye button on the player's own dock. `null` until it loads and for a
        // property with no drawing; the player shows no button in either case.
        plan={plan}
        hotspotOverrides={draft ? [draft] : undefined}
        // Walking a floor ring into a room that HAS been shot moves the player to that room,
        // and the editor has to come too: its sidebar, its tag list and its URL all belong to
        // one scene, so staying put would leave you editing a room you are no longer looking at.
        onSceneChange={(id) => {
          if (id !== sceneId) router.push(`/evler/${propertyId}/scenes/${id}`);
        }}
        onSphereClick={handleSphereClick}
        onHotspotClick={(hotspot) => {
          setDraft({ ...hotspot });
          setFormError(null);
          setTab("tags");
          setPanelOpen(true);
        }}
        onHotspotDragEnd={(hotspot, position) => void persistMove(hotspot, position)}
        onViewChange={handleViewChange}
        className="absolute inset-0"
      />

      <div className="pointer-events-none absolute inset-x-3 top-3 z-20 flex items-start justify-between gap-3">
        <div className="glass-strong pointer-events-auto flex items-center gap-2.5 rounded-md py-1.5 pr-3 pl-1.5">
          <Link
            href={`/evler/${propertyId}`}
            aria-label={ui("editor.back", locale)}
            className="grid size-7 place-items-center rounded-sm text-ink-muted transition-colors hover:bg-surface-3 hover:text-ink"
          >
            <ArrowLeft className="size-4" aria-hidden />
          </Link>
          <span className="max-w-[40vw] truncate text-[13px] font-medium text-ink">{scene.name}</span>
          <Badge tone="bio">{t("scene.ready", locale)}</Badge>
        </div>

        {!panelOpen ? (
          <Button
            variant="secondary"
            size="sm"
            className="pointer-events-auto"
            onClick={() => setPanelOpen(true)}
          >
            <PanelRightOpen className="size-4" aria-hidden />
            {ui("editor.panel.show", locale)}
          </Button>
        ) : null}
      </div>

      {panelOpen ? (
        <aside
          aria-label={ui("editor.inspector", locale)}
          className={cn(
            "glass-strong absolute z-20 flex flex-col overflow-hidden rounded-lg shadow-lift",
            "inset-x-3 bottom-3 max-h-[58%] lg:inset-y-3 lg:right-3 lg:left-auto lg:max-h-none lg:w-[380px]",
          )}
        >
          <div className="flex items-center gap-2 px-3 pt-2">
            <Tabs
              items={tabs}
              value={tab}
              onChange={(id) => setTab(id as TabId)}
              idBase="inspector"
              size="sm"
              className="min-w-0 flex-1 border-b-0"
            />
            <button
              type="button"
              onClick={() => setPanelOpen(false)}
              aria-label={ui("editor.panel.hide", locale)}
              className="grid size-7 shrink-0 place-items-center rounded-sm text-ink-faint transition-colors hover:bg-surface-3 hover:text-ink"
            >
              <PanelRightClose className="size-4" aria-hidden />
            </button>
          </div>
          <div className="h-px bg-line" />

          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {tab === "tags" ? (
              <div {...tabPanelProps("inspector", "tags")} className="space-y-4 outline-none">
                {draft ? (
                  <div className="space-y-3 rounded-md border border-bio/30 bg-bio/5 p-3">
                    <div className="flex items-center justify-between">
                      <h3 className="text-[12px] font-semibold text-ink">
                        {draft.id.startsWith(DRAFT_PREFIX)
                          ? ui("editor.hotspot.new", locale)
                          : ui("editor.hotspot.edit", locale)}
                      </h3>
                      <span className="font-mono text-[10px] text-ink-faint tabular-nums">
                        {Math.round(degrees(draft.yaw))}° / {Math.round(degrees(draft.pitch))}°
                      </span>
                    </div>

                    <Select
                      label={ui("editor.hotspot.type", locale)}
                      value={draft.type}
                      options={[
                        { value: "info", label: ui("editor.hotspot.type.info", locale) },
                        {
                          value: "portal",
                          label: ui("editor.hotspot.type.portal", locale),
                          disabled: linkTargets.length === 0,
                        },
                        {
                          value: "link",
                          label: ui("editor.hotspot.type.link", locale),
                          disabled: linkTargets.length === 0,
                        },
                        { value: "media", label: ui("editor.hotspot.type.media", locale) },
                      ]}
                      onChange={(event) =>
                        setDraft((current) =>
                          current ? { ...current, type: event.target.value as HotspotType } : current,
                        )
                      }
                      hint={
                        NAVIGATION_TYPES.includes(draft.type as (typeof NAVIGATION_TYPES)[number]) &&
                        linkTargets.length === 0
                          ? ui("editor.hotspot.noRooms", locale)
                          : draft.type === "portal"
                            ? draft.pitch >= 0
                              ? // A ring is placed by clicking the FLOOR, and the commonest
                                // mistake is clicking the wall. Said here, where the choice is
                                // made, rather than discovered as a ring that never appears.
                                ui("editor.hotspot.portal.above", locale)
                              : ui("editor.hotspot.portal.hint", locale)
                            : undefined
                      }
                    />

                    <Input
                      label={t("editor.hotspot.title", locale)}
                      value={draft.title}
                      autoFocus
                      error={formError ?? undefined}
                      onChange={(event) =>
                        setDraft((current) => (current ? { ...current, title: event.target.value } : current))
                      }
                    />

                    {NAVIGATION_TYPES.includes(
                      draft.type as (typeof NAVIGATION_TYPES)[number],
                    ) ? (
                      <Select
                        label={ui("editor.hotspot.target", locale)}
                        value={draft.target_scene_id ?? ""}
                        placeholder={ui("editor.hotspot.target.placeholder", locale)}
                        options={linkTargets}
                        onChange={(event) =>
                          setDraft((current) =>
                            current ? { ...current, target_scene_id: event.target.value || null } : current,
                          )
                        }
                      />
                    ) : draft.type === "media" ? (
                      <Input
                        label={ui("editor.hotspot.media", locale)}
                        value={draft.media_url ?? ""}
                        inputMode="url"
                        onChange={(event) =>
                          setDraft((current) => (current ? { ...current, media_url: event.target.value || null } : current))
                        }
                      />
                    ) : null}

                    <Textarea
                      label={t("editor.hotspot.body", locale)}
                      value={draft.body ?? ""}
                      rows={3}
                      onChange={(event) =>
                        setDraft((current) => (current ? { ...current, body: event.target.value || null } : current))
                      }
                    />

                    <div className="flex items-center gap-2">
                      <Button variant="primary" size="sm" loading={saving} onClick={() => void saveDraft()}>
                        {t("editor.save", locale)}
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => setDraft(null)}>
                        {t("editor.cancel", locale)}
                      </Button>
                      {!draft.id.startsWith(DRAFT_PREFIX) ? (
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          aria-label={t("editor.delete", locale)}
                          className="ml-auto text-danger hover:bg-danger/10"
                          onClick={() => void deleteHotspot(draft)}
                        >
                          <Trash2 className="size-3.5" aria-hidden />
                        </Button>
                      ) : null}
                    </div>
                  </div>
                ) : (
                  <p className="flex items-center gap-2 rounded-md border border-dashed border-line-strong px-3 py-2.5 text-[12px] leading-relaxed text-ink-muted">
                    <Plus className="size-3.5 shrink-0 text-bio" aria-hidden />
                    {t("editor.hotspot.add", locale)}
                  </p>
                )}

                <div>
                  <h3 className="mb-2 text-[12px] font-semibold text-ink">
                    {ui("editor.hotspot.list", locale)}{" "}
                    <span className="font-normal text-ink-faint">({hotspots.length})</span>
                  </h3>
                  {hotspots.length === 0 ? (
                    <p className="text-[12px] text-ink-faint">{ui("editor.hotspot.empty", locale)}</p>
                  ) : (
                    <ul className="space-y-1.5">
                      {hotspots.map((hotspot) => (
                        <li key={hotspot.id} className="flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => {
                              playerRef.current?.lookAt(
                                { yaw: hotspot.yaw, pitch: hotspot.pitch },
                                { animate: true },
                              );
                              setDraft({ ...hotspot });
                              setFormError(null);
                            }}
                            title={ui("editor.hotspot.focus", locale)}
                            className="flex min-w-0 flex-1 items-center gap-2 rounded-sm border border-line bg-surface-2/50 px-2.5 py-2 text-left transition-colors hover:border-line-strong hover:bg-surface-2"
                          >
                            <Tag className="size-3 shrink-0 text-bio" aria-hidden />
                            <span className="min-w-0 flex-1 truncate text-[12px] text-ink">{hotspot.title}</span>
                            <span className="shrink-0 font-mono text-[10px] text-ink-faint tabular-nums">
                              {Math.round(degrees(hotspot.yaw))}°
                            </span>
                          </button>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            aria-label={`${t("editor.delete", locale)}: ${hotspot.title}`}
                            className="shrink-0 text-ink-faint hover:text-danger"
                            onClick={() => void deleteHotspot(hotspot)}
                          >
                            <Trash2 className="size-3.5" aria-hidden />
                          </Button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            ) : tab === "photos" ? (
              <div {...tabPanelProps("inspector", "photos")} className="outline-none">
                {scene ? (
                  <RoomPhotos
                    sceneId={scene.id}
                    photos={scene.photos}
                    onChange={(photos) => setPhotos(scene.id, photos)}
                  />
                ) : null}
              </div>
            ) : tab === "view" ? (
              <div {...tabPanelProps("inspector", "view")} className="space-y-4 outline-none">
                <div>
                  <h3 className="text-[12px] font-semibold text-ink">{ui("editor.view.title", locale)}</h3>
                  <p className="mt-1 text-[12px] leading-relaxed text-ink-muted">{ui("editor.view.desc", locale)}</p>
                </div>

                <dl className="grid grid-cols-3 gap-2">
                  {[
                    { label: ui("editor.view.yaw", locale), value: `${Math.round(degrees(view?.yaw ?? 0))}°` },
                    { label: ui("editor.view.pitch", locale), value: `${Math.round(degrees(view?.pitch ?? 0))}°` },
                    { label: ui("editor.view.fov", locale), value: `${Math.round(view?.fov ?? 0)}°` },
                  ].map((item) => (
                    <div key={item.label} className="rounded-sm border border-line bg-surface-2/50 px-2.5 py-2">
                      <dt className="text-[10px] tracking-wide text-ink-faint uppercase">{item.label}</dt>
                      <dd className="mt-0.5 font-mono text-[13px] text-ink tabular-nums">{item.value}</dd>
                    </div>
                  ))}
                </dl>

                <Button variant="primary" size="sm" loading={savingView} onClick={() => void saveInitialView()}>
                  <Crosshair className="size-3.5" aria-hidden />
                  {t("editor.initialView", locale)}
                </Button>
              </div>
            ) : tab === "coverage" ? (
              <div {...tabPanelProps("inspector", "coverage")} className="outline-none">
                <CoveragePanel coverage={scene.coverage} />
              </div>
            ) : (
              <div {...tabPanelProps("inspector", "report")} className="outline-none">
                <MetricsReport metrics={scene.metrics} />
              </div>
            )}
          </div>
        </aside>
      ) : null}
    </div>
  );
}
