"use client";

/**
 * The floor plan, in the panel where it is made — and, when the solver gets it wrong, fixed.
 *
 * ── WHY THE BUILD LIVES HERE AND NOWHERE ELSE ─────────────────────────────────────────
 * Reading the plan is free and measuring it is not: each room is a scene-parsing pass over
 * sixteen perspective windows, seconds on a card and minutes without one. So the engine
 * splits the endpoint in two, and the split shows up here as a rule about who may press
 * what. The agent who has just finished shooting a flat presses "measure" once; every
 * viewer afterwards — in the dashboard, in the player, on a listing portal's page — reads
 * the answer off disk. A plan built on demand would make a stranger's page wait on somebody
 * else's GPU.
 *
 * ── AND WHY IT CAN BE EDITED ──────────────────────────────────────────────────────────
 * The solver is honest about what it does not know: a doorway marked from one side only
 * fixes a room's position and leaves its rotation as a four-way guess, and a room no doorway
 * reaches is not placed at all. Both are the right answer, and neither helps the agent
 * looking at a kitchen that is a quarter turn out. So the automatic layout is a FIRST DRAFT:
 * rooms drag, a selected room turns, and a property with no floor rings anywhere can be
 * assembled entirely by hand. Corrections are stored per scene, so re-measuring a room or
 * shooting the one beyond it does not throw the agent's work away.
 *
 * Every placement round-trips to the server rather than being patched locally, because a
 * placement is NOT LOCAL: the layout is walked again from every placed room, so moving the
 * hallway is supposed to carry what lies beyond it.
 *
 * ── WHAT IT SAYS WHEN THERE IS NOTHING TO DRAW ────────────────────────────────────────
 * Three different nothings, and they need three different sentences, because the fix is
 * different in each case: no room has been measured yet (press the button), the rooms are
 * measured but no doorway links them (drop floor rings, or place them by hand), or this
 * deployment has no segmenter at all (nothing you can do here). Collapsing them into one
 * empty state would leave the agent guessing which of the three they were in.
 */

import { useCallback, useMemo, useState } from "react";
import { Compass, Loader2, Pencil, Plus, RotateCw, Ruler, Undo2 } from "lucide-react";
import { toast } from "sonner";
import { PlanView } from "@/components/player/PlanView";
import {
  drawnRooms,
  hasDrawing,
  parkingSpot,
  unlinkedRooms,
} from "@/components/player/planGeometry";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { BioSphereError, api } from "@/lib/api";
import { useApi } from "@/lib/hooks/useApi";
import { t } from "@/lib/i18n";
import type { FloorPlan, Locale, PlanRoom, UpdatePlanRequest } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface PlanPanelProps {
  propertyId: string;
  locale: Locale;
  /** Opening a room from the plan; the property page routes it to that scene's editor. */
  onOpenRoom?: (sceneId: string) => void;
}

/** Keep a repeatedly turned room's rotation in (-pi, pi] instead of letting it wind up. */
function wrap(angle: number): number {
  return ((angle + Math.PI) % (2 * Math.PI)) - Math.PI;
}

export function PlanPanel({ propertyId, locale, onOpenRoom }: PlanPanelProps) {
  const { data, loading, initial, setData } = useApi<FloorPlan>(() => api.plan.get(propertyId), [propertyId]);
  const [building, setBuilding] = useState(false);
  const [editing, setEditing] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const build = useCallback(async () => {
    setBuilding(true);
    try {
      setData(await api.plan.build(propertyId));
    } catch (cause) {
      toast.error(cause instanceof BioSphereError ? cause.localised(locale) : t("plan.failed", locale));
    } finally {
      setBuilding(false);
    }
  }, [propertyId, locale, setData]);

  /**
   * Every edit goes through here, and every edit takes the whole plan back.
   *
   * Optimism is the wrong shape for this one: the server may legitimately move rooms the
   * caller never touched, so patching a local copy of the room that was dragged would leave
   * the panel drawing a flat the server does not have.
   */
  const patch = useCallback(
    async (body: UpdatePlanRequest) => {
      try {
        setData(await api.plan.place(propertyId, body));
      } catch (cause) {
        toast.error(
          cause instanceof BioSphereError ? cause.localised(locale) : t("plan.saveFailed", locale),
        );
      }
    },
    [propertyId, locale, setData],
  );

  const place = useCallback(
    (room: PlanRoom, next: Partial<{ x: number; z: number; rotation: number }>) =>
      void patch({
        placements: {
          [room.scene_id]: {
            x: next.x ?? room.x,
            z: next.z ?? room.z,
            rotation: wrap(next.rotation ?? room.rotation),
          },
        },
      }),
    [patch],
  );

  const drawn = useMemo(() => (data ? drawnRooms(data) : []), [data]);
  const unlinked = useMemo(() => (data ? unlinkedRooms(data) : []), [data]);

  if (initial && loading) return <Skeleton className="h-[320px] w-full rounded-lg" />;
  if (!data) return null;

  const selected = drawn.find((room) => room.scene_id === selectedId) ?? null;
  const placedCount = drawn.filter((room) => room.placement === "placed").length;
  const measured = drawn.filter((room) => room.polygon.length >= 3);
  const area = measured.reduce((total, room) => total + room.area_m2, 0);
  const pending = data.pending_scene_ids.length;
  const decimal = (value: number) =>
    locale === "tr" ? value.toFixed(1).replace(".", ",") : value.toFixed(1);

  return (
    <div className="glass rounded-lg p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold tracking-tight text-ink">{t("plan.title", locale)}</h2>
          <p className="mt-0.5 text-[11px] text-ink-faint">
            {editing
              ? t("plan.editHint", locale)
              : measured.length > 0
                ? t("plan.summary", locale)
                    .replace("{n}", String(measured.length))
                    .replace("{area}", decimal(area))
                : t("plan.scaleNote", locale).replace(
                    "{h}",
                    locale === "tr"
                      ? data.camera_height_m.toFixed(2).replace(".", ",")
                      : data.camera_height_m.toFixed(2),
                  )}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {hasDrawing(data) ? (
            <Button
              variant={editing ? "primary" : "ghost"}
              onClick={() => {
                setEditing((on) => !on);
                setSelectedId(null);
              }}
            >
              <Pencil className="size-4" aria-hidden />
              {editing ? t("plan.editDone", locale) : t("plan.edit", locale)}
            </Button>
          ) : null}

          {data.can_measure ? (
            <Button variant="ghost" loading={building} disabled={building} onClick={() => void build()}>
              {building ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : (
                <Ruler className="size-4" aria-hidden />
              )}
              {pending > 0
                ? t("plan.measureMore", locale).replace("{n}", String(pending))
                : t("plan.measure", locale)}
            </Button>
          ) : (
            <span className="text-[11px] text-ink-faint">{t("plan.noModel", locale)}</span>
          )}
        </div>
      </div>

      {hasDrawing(data) ? (
        <>
          <div className="relative h-[360px] overflow-hidden rounded-md border border-line bg-canvas">
            <PlanView
              plan={data}
              currentSceneId={null}
              locale={locale}
              open
              onSelectRoom={(sceneId) => onOpenRoom?.(sceneId)}
              headingRef={{ current: null }}
              edit={
                editing
                  ? {
                      selectedId,
                      onSelect: setSelectedId,
                      onMove: (sceneId, position) => {
                        const room = drawn.find((candidate) => candidate.scene_id === sceneId);
                        if (room) place(room, position);
                      },
                    }
                  : undefined
              }
            />
          </div>

          {editing ? (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {selected ? (
                <>
                  <span className="text-[12px] font-medium text-ink">{selected.name}</span>
                  <Button
                    variant="ghost"
                    onClick={() => place(selected, { rotation: selected.rotation + Math.PI / 2 })}
                  >
                    <RotateCw className="size-4" aria-hidden />
                    {t("plan.rotate", locale)}
                  </Button>
                  {selected.placement === "placed" ? (
                    <Button
                      variant="ghost"
                      onClick={() => void patch({ placements: { [selected.scene_id]: null } })}
                    >
                      <Undo2 className="size-4" aria-hidden />
                      {t("plan.resetRoom", locale)}
                    </Button>
                  ) : null}
                </>
              ) : (
                <span className="text-[11px] text-ink-faint">{t("plan.editHint", locale)}</span>
              )}

              {placedCount > 0 ? (
                <button
                  type="button"
                  onClick={() => {
                    setSelectedId(null);
                    void patch({ reset: true });
                  }}
                  className="ml-auto text-[11px] text-ink-faint underline-offset-2 hover:text-ink hover:underline"
                >
                  {`${t("plan.resetAll", locale)} · ${t("plan.placedCount", locale).replace(
                    "{n}",
                    String(placedCount),
                  )}`}
                </button>
              ) : null}
            </div>
          ) : null}

          {unlinked.length > 0 ? (
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="text-[11px] text-ink-faint">{t("plan.unlinked", locale)}</span>
              {unlinked.map((room) => (
                <button
                  key={room.scene_id}
                  type="button"
                  disabled={!editing}
                  title={editing ? t("plan.addRoom", locale) : undefined}
                  onClick={() => place(room, parkingSpot(data, room))}
                  className={cn(
                    "flex items-center gap-1 rounded-full border border-line px-2 py-0.5 text-[11px]",
                    editing
                      ? "cursor-pointer text-ink-muted hover:border-bio/45 hover:text-bio"
                      : "cursor-default text-ink-muted",
                  )}
                >
                  {editing ? <Plus className="size-3" aria-hidden /> : null}
                  {room.name}
                </button>
              ))}
            </div>
          ) : null}

          {!editing && drawn.some((room) => room.placement === "estimated") ? (
            <p className="mt-3 text-[11px] leading-relaxed text-ink-faint">
              {t("plan.needsPortals", locale)}
            </p>
          ) : null}
        </>
      ) : (
        <EmptyState
          icon={Compass}
          title={t("plan.title", locale)}
          body={
            !data.can_measure
              ? t("plan.noModel", locale)
              : pending > 0
                ? t("plan.empty", locale)
                : t("plan.needsPortals", locale)
          }
          action={
            data.can_measure && pending > 0 ? (
              <Button variant="primary" loading={building} onClick={() => void build()}>
                <Ruler className="size-4" aria-hidden />
                {t("plan.measure", locale)}
              </Button>
            ) : undefined
          }
        />
      )}
    </div>
  );
}
