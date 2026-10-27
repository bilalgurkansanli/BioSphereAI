"use client";

/**
 * The flat from above, and where you are standing in it.
 *
 * This component is only the DRAWING. Where it sits is its host's business: `PlayerDock` hangs
 * it off a button in the player's bottom-left corner, and `PlanPanel` lays it into a card on
 * the property page. Both hand it a box and it fills it, which is why neither of them had to
 * fork it when the player's version stopped being a full-frame overlay.
 *
 * ── WHAT THE DRAWING IS ALLOWED TO CLAIM ──────────────────────────────────────────────
 * Three kinds of line, and the difference between them is the honesty of the feature:
 *
 *   solid teal     a wall this room's own panorama measured.
 *   dashed         a room that is on the plan but has no measured shape — the disc is a
 *                  placeholder for "a room, about this big", never a claim about its walls.
 *   amber          a room nobody has shot yet. The same amber the floor ring uses for the
 *                  same fact, from the same constant, so the two cannot drift apart.
 *
 * A room reached through a doorway that only one side marked is drawn with a broken outline
 * whatever its colour: it really is through that door, on that bearing, but which way it is
 * turned is a guess the assembler made and the drawing must not launder into a fact.
 *
 * ── THERE IS NO COMPASS ROSE, AND THAT IS DELIBERATE ──────────────────────────────────
 * Matterport's plan has a north arrow because a Matterport camera knows which way it was
 * pointing. A phone sweep does not: the engine levels the horizon against gravity, which
 * fixes *up* and says nothing about north, so every panorama's yaw origin is wherever the
 * photographer started turning.
 *
 * The phone does have a magnetometer, and `lib/capture/useOrientation.ts` already measured
 * why it is not trusted: beside a fridge or a steel stud wall it keeps reporting its nominal
 * 10-15 degrees of accuracy while the heading is 30-40 degrees out, and it SNAPS as you turn
 * past the anomaly. The capture builds yaw from the relative track for that reason. A north
 * arrow here would put the number that file deliberately refuses in front of a buyer. The
 * plan's own orientation is the reference instead, which is what a viewer actually needs —
 * "the kitchen is behind me", not "the kitchen is north".
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Loader2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { t } from "@/lib/i18n";
import type { FloorPlan, Locale, PlanRoom } from "@/lib/types";
import { PENDING_COLOUR, READY_COLOUR } from "./PortalRings";
import {
  type PlanViewport,
  drawnRooms,
  fitViewport,
  labelPoint,
  polygonPath,
  roomBounds,
  scaleBar,
  screenDeltaToPlan,
  snapDelta,
  toScreen,
  unlinkedRooms,
} from "./planGeometry";

const EASE = [0.16, 1, 0.3, 1] as const;

/**
 * How long the "you are here" cone is, in pixels.
 *
 * Fixed on screen rather than in metres, because it is not a measurement of anything — it is
 * a statement about which way the viewer is facing, and it has to stay readable when a whole
 * flat is fitted into a phone.
 */
const CONE_LENGTH = 34;
const CONE_HALF_ANGLE = 26;

/** One decimal, with the decimal separator the reader uses. */
function metres(value: number, locale: Locale): string {
  const text = value.toFixed(1);
  return locale === "tr" ? text.replace(".", ",") : text;
}

/**
 * Everything the plan needs in order to be edited, or nothing at all.
 *
 * The player never passes this and the dashboard always does, which is the same split
 * `tools` makes on the player itself: a viewer on a listing portal is looking at a drawing,
 * not authoring one. Keeping it in one optional object means the read-only path has no edit
 * code in it to accidentally trigger.
 */
export interface PlanEditing {
  selectedId: string | null;
  onSelect: (sceneId: string | null) => void;
  /** Called once, when a drag ends, with the room's new position in plan metres. */
  onMove: (sceneId: string, position: { x: number; z: number }) => void;
}

export interface PlanViewProps {
  plan: FloorPlan;
  currentSceneId: string | null;
  locale: Locale;
  open: boolean;
  /** Omitted where the plan is the panel rather than an overlay, and there is nothing to close. */
  onClose?: () => void;
  onSelectRoom: (sceneId: string) => void;
  /**
   * The heading cone's group. The player writes a `rotate(...)` onto it from inside the frame
   * loop, the way it writes the compass dial — a `setState` at the pointer's sample rate
   * would re-render the whole player, canvas included, every time somebody moved their head.
   */
  headingRef: React.RefObject<SVGGElement | null>;
  /** Offered only where a build is possible: the dashboard, never a third-party embed. */
  onMeasure?: () => void;
  measuring?: boolean;
  /** Present only where the plan may be authored; see {@link PlanEditing}. */
  edit?: PlanEditing;
}

function roomColour(room: PlanRoom): string {
  return room.ready ? READY_COLOUR : PENDING_COLOUR;
}

function RoomShape({
  room,
  viewport,
  active,
  onSelect,
  label,
  editing,
  selected,
  offset,
  onDragStart,
}: {
  room: PlanRoom;
  viewport: PlanViewport;
  active: boolean;
  onSelect: () => void;
  label: string;
  editing: boolean;
  selected: boolean;
  /** Screen-space preview offset while this room is being dragged. */
  offset: { x: number; y: number } | null;
  onDragStart: (room: PlanRoom, event: React.PointerEvent<SVGGElement>) => void;
}) {
  const colour = roomColour(room);
  const estimated = room.placement === "estimated";
  const [cx, cy] = toScreen(viewport, room.x, room.z);
  const path = polygonPath(viewport, room.polygon as ReadonlyArray<readonly [number, number]>);

  return (
    <g
      role="button"
      tabIndex={editing || room.ready ? 0 : -1}
      aria-label={label}
      aria-current={active ? "true" : undefined}
      aria-disabled={!editing && !room.ready}
      // In edit mode a click SELECTS rather than travels: the agent is arranging the flat,
      // not walking it, and a drag that ended on the room it started on must not also open it.
      onPointerDown={editing ? (event) => onDragStart(room, event) : undefined}
      onClick={editing ? onSelect : room.ready ? onSelect : undefined}
      onKeyDown={(event) => {
        if ((editing || room.ready) && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          onSelect();
        }
      }}
      transform={offset ? `translate(${offset.x.toFixed(1)} ${offset.y.toFixed(1)})` : undefined}
      className={cn(
        "outline-none",
        editing ? "cursor-grab active:cursor-grabbing" : room.ready ? "cursor-pointer" : "cursor-default",
        "[&>.shape]:transition-opacity [&>.shape]:duration-200",
        (editing || room.ready) && "hover:[&>.shape]:opacity-100 focus-visible:[&>.shape]:opacity-100",
      )}
    >
      {path ? (
        <path
          className="shape"
          d={path}
          fill={colour}
          fillOpacity={selected ? 0.22 : active ? 0.16 : 0.07}
          stroke={selected ? "var(--bio)" : colour}
          strokeWidth={selected ? 2.8 : active ? 2.4 : 1.6}
          strokeOpacity={estimated && !selected ? 0.6 : 0.95}
          strokeDasharray={estimated ? "7 5" : undefined}
          strokeLinejoin="round"
        />
      ) : (
        <circle
          className="shape"
          cx={cx}
          cy={cy}
          r={Math.max(room.radius * viewport.scale, 16)}
          fill={colour}
          fillOpacity={selected ? 0.18 : active ? 0.12 : 0.05}
          stroke={selected ? "var(--bio)" : colour}
          strokeWidth={selected ? 2.4 : 1.4}
          strokeOpacity={selected ? 0.95 : 0.5}
          // A placeholder is always dashed, whatever placed it: the outline is not a wall.
          strokeDasharray="5 6"
        />
      )}
    </g>
  );
}

function RoomLabel({
  room,
  viewport,
  active,
  locale,
}: {
  room: PlanRoom;
  viewport: PlanViewport;
  active: boolean;
  locale: Locale;
}) {
  const [cx, cy] = toScreen(viewport, ...labelPoint(room));
  const measured = room.polygon.length >= 3 && room.area_m2 > 0;
  return (
    <g className="pointer-events-none select-none" transform={`translate(${cx.toFixed(1)} ${cy.toFixed(1)})`}>
      <text
        y={measured ? -6 : 0}
        textAnchor="middle"
        className={cn(
          "text-[12px] font-medium",
          active ? "fill-[var(--ink)]" : "fill-[var(--ink-muted)]",
        )}
        style={{ paintOrder: "stroke", stroke: "var(--canvas)", strokeWidth: 3.5 }}
      >
        {room.name}
      </text>
      {measured ? (
        <text
          y={10}
          textAnchor="middle"
          className="fill-[var(--ink-faint)] text-[10px] tabular-nums"
          style={{ paintOrder: "stroke", stroke: "var(--canvas)", strokeWidth: 3 }}
        >
          {`${metres(room.area_m2, locale)} m²`}
        </text>
      ) : !room.ready ? (
        <text
          y={14}
          textAnchor="middle"
          className="fill-[var(--amber)] text-[10px]"
          style={{ paintOrder: "stroke", stroke: "var(--canvas)", strokeWidth: 3 }}
        >
          {t("plan.notShot", locale)}
        </text>
      ) : null}
    </g>
  );
}

export function PlanView(props: PlanViewProps) {
  const {
    plan,
    currentSceneId,
    locale,
    open,
    onClose,
    onSelectRoom,
    headingRef,
    onMeasure,
    measuring,
    edit,
  } = props;

  const [box, setBox] = useState<{ width: number; height: number } | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);

  /**
   * The drawing is fitted to the frame, so the frame has to be measured — on every render as
   * well as on every resize.
   *
   * A ResizeObserver alone is not enough, and the way it fails is quiet: its first reading is
   * whatever the box was when it was attached, and a panel attached mid-layout is zero wide.
   * The plan is then committed at that size and NOTHING EVER CORRECTS IT — the observer has
   * no change to report, because as far as it is concerned nothing changed. Measured in the
   * dashboard: the outline came out four pixels across with a "0.05 m" scale bar, and stayed
   * that way until the window was resized by hand.
   *
   * So: measure in a LAYOUT effect on every render (so a corrected size never reaches the
   * screen as a frame of collapsed drawing), keep the observer for the resizes that arrive
   * without one, and refuse a degenerate reading outright rather than drawing to it.
   */
  const measure = useCallback(() => {
    const node = frameRef.current;
    if (!node) return;
    const width = node.clientWidth;
    const height = node.clientHeight;
    if (width < 8 || height < 8) return;
    setBox((current) =>
      current && current.width === width && current.height === height ? current : { width, height },
    );
  }, []);

  useLayoutEffect(measure);

  useEffect(() => {
    const node = frameRef.current;
    if (!node || !open) return;
    const observer = new ResizeObserver(measure);
    observer.observe(node);

    // A ResizeObserver watching a box that has no size does not always report the moment it
    // gets one — a panel that was laid out while its window was hidden comes back with the
    // right size and no callback, and since nothing re-rendered, the layout effect above
    // never runs either. Measured: the frame read 1097 x 272 in the DOM while the drawing was
    // still waiting for its first good measurement. These two are the events that say "the
    // page you laid out is now actually on screen".
    window.addEventListener("resize", measure);
    document.addEventListener("visibilitychange", measure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
      document.removeEventListener("visibilitychange", measure);
    };
  }, [open, measure]);

  const rooms = useMemo(() => drawnRooms(plan), [plan]);
  const unlinked = useMemo(() => unlinkedRooms(plan), [plan]);
  const viewport = useMemo(
    () => (box ? fitViewport(plan.bounds, box.width, box.height) : null),
    [plan.bounds, box],
  );
  const current = rooms.find((room) => room.scene_id === currentSceneId) ?? null;

  const bar = viewport ? scaleBar(viewport, Math.min(140, viewport.width * 0.28)) : null;

  // Escape closes it, like every other thing that covers the picture.
  const handleKey = useCallback(
    (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose?.();
    },
    [onClose],
  );
  useEffect(() => {
    if (!open || !onClose) return;
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [open, onClose, handleKey]);

  const marker = viewport && current ? toScreen(viewport, current.x, current.z) : null;

  /**
   * The drag in progress, as a SCREEN offset.
   *
   * Screen rather than metres, and a transform rather than recomputed geometry, because the
   * room's polygon already arrives in plan coordinates: translating the group is the whole
   * preview. The authoritative answer comes back from the server on drop, which it has to —
   * moving one room can move the rooms beyond it.
   */
  const [drag, setDrag] = useState<{ id: string; x: number; y: number } | null>(null);
  const dragOrigin = useRef<{ id: string; pointerX: number; pointerY: number } | null>(null);

  const beginDrag = useCallback(
    (room: PlanRoom, event: React.PointerEvent<SVGGElement>) => {
      if (!edit) return;
      event.stopPropagation();
      try {
        // Capture keeps the move and drop events coming to this element even when the
        // pointer leaves the room's own shape, which on a small plan it does immediately.
        // Guarded because a pointer id the element never saw is an exception rather than a
        // no-op, and losing the handler to it would leave a drag that starts and never moves.
        event.currentTarget.setPointerCapture?.(event.pointerId);
      } catch {
        /* Without capture the drag still works while the pointer stays over the plan. */
      }
      dragOrigin.current = { id: room.scene_id, pointerX: event.clientX, pointerY: event.clientY };
      setDrag({ id: room.scene_id, x: 0, y: 0 });
      edit.onSelect(room.scene_id);
    },
    [edit],
  );

  /** The snapped offset for the room under the pointer, in metres and in pixels. */
  const dragTo = useCallback(
    (clientX: number, clientY: number) => {
      const origin = dragOrigin.current;
      if (!origin || !viewport) return null;
      const moving = rooms.find((room) => room.scene_id === origin.id);
      if (!moving) return null;

      const loose = screenDeltaToPlan(viewport, clientX - origin.pointerX, clientY - origin.pointerY);
      const snapped = snapDelta(
        roomBounds(moving),
        rooms.filter((room) => room.scene_id !== origin.id).map(roomBounds),
        loose,
      );
      return {
        room: moving,
        metres: snapped,
        pixels: { x: snapped.x * viewport.scale, y: -snapped.z * viewport.scale },
      };
    },
    [rooms, viewport],
  );

  const moveDrag = useCallback(
    (event: React.PointerEvent<SVGGElement>) => {
      const next = dragTo(event.clientX, event.clientY);
      if (next) setDrag({ id: next.room.scene_id, ...next.pixels });
    },
    [dragTo],
  );

  const endDrag = useCallback(
    (event: React.PointerEvent<SVGGElement>) => {
      const next = dragTo(event.clientX, event.clientY);
      dragOrigin.current = null;
      setDrag(null);
      if (!next || !edit) return;
      // A tap that never moved is a selection, not a placement; committing it would pin a
      // room the agent only wanted to look at, and pinning is what "Otomatiğe dön" undoes.
      if (Math.hypot(next.metres.x, next.metres.z) < 0.02) return;
      edit.onMove(next.room.scene_id, {
        x: next.room.x + next.metres.x,
        z: next.room.z + next.metres.z,
      });
    },
    [dragTo, edit],
  );

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          key="plan"
          data-player-ui
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.24, ease: EASE }}
          className="absolute inset-0 z-40 flex flex-col bg-canvas/92 backdrop-blur-md"
        >
          <div className="flex shrink-0 items-start justify-between gap-3 px-4 pt-[max(0.85rem,env(safe-area-inset-top))]">
            <div className="min-w-0">
              <p className="text-[13px] font-medium tracking-tight text-ink">{t("plan.title", locale)}</p>
              <p className="mt-0.5 text-[11px] leading-snug text-ink-faint">
                {t("plan.scaleNote", locale).replace(
                  "{h}",
                  locale === "tr"
                    ? plan.camera_height_m.toFixed(2).replace(".", ",")
                    : plan.camera_height_m.toFixed(2),
                )}
              </p>
            </div>
            {onClose ? (
              <button
                type="button"
                onClick={onClose}
                aria-label={t("plan.close", locale)}
                className={cn(
                  "grid h-8 w-8 shrink-0 place-items-center rounded-full border border-line",
                  "text-ink-muted transition-colors hover:border-line-strong hover:text-ink",
                )}
              >
                <X size={15} />
              </button>
            ) : null}
          </div>

          <div ref={frameRef} className="relative min-h-0 flex-1">
            {viewport ? (
              <svg
                width={viewport.width}
                height={viewport.height}
                viewBox={`0 0 ${viewport.width} ${viewport.height}`}
                className="absolute inset-0"
                role="img"
                aria-label={t("plan.title", locale)}
                onPointerMove={drag ? moveDrag : undefined}
                onPointerUp={drag ? endDrag : undefined}
                onPointerCancel={drag ? endDrag : undefined}
                // Clicking the paper puts the selection down, which is how every canvas works
                // and how the agent stops the rotate controls following them around.
                onPointerDown={edit && !drag ? () => edit.onSelect(null) : undefined}
              >
                {/* Doorways under the rooms, so a room's own outline stays the strongest line. */}
                {plan.doors.map((door) => {
                  const [dx, dy] = toScreen(viewport, door.x, door.z);
                  return (
                    <circle
                      key={`${door.from_scene_id}-${door.to_scene_id}`}
                      cx={dx}
                      cy={dy}
                      r={5}
                      fill="var(--canvas)"
                      stroke={READY_COLOUR}
                      strokeWidth={1.6}
                      // A doorway only one room marked is the reason its far side is a guess;
                      // the same broken line says so here as on the room it placed.
                      strokeDasharray={door.solved ? undefined : "3 3"}
                      strokeOpacity={0.85}
                    />
                  );
                })}

                {rooms.map((room) => (
                  <RoomShape
                    key={room.scene_id}
                    room={room}
                    viewport={viewport}
                    active={room.scene_id === currentSceneId}
                    onSelect={() =>
                      edit ? edit.onSelect(room.scene_id) : onSelectRoom(room.scene_id)
                    }
                    label={room.name}
                    editing={Boolean(edit)}
                    selected={edit?.selectedId === room.scene_id}
                    offset={drag?.id === room.scene_id ? { x: drag.x, y: drag.y } : null}
                    onDragStart={beginDrag}
                  />
                ))}

                {marker ? (
                  <g transform={`translate(${marker[0].toFixed(1)} ${marker[1].toFixed(1)})`}>
                    {/* Rotated from the frame loop; see `headingRef`. */}
                    <g ref={headingRef}>
                      <path
                        d={`M0 0 L${(
                          CONE_LENGTH * Math.sin((-CONE_HALF_ANGLE * Math.PI) / 180)
                        ).toFixed(1)} ${(-CONE_LENGTH * Math.cos((CONE_HALF_ANGLE * Math.PI) / 180)).toFixed(
                          1,
                        )} A${CONE_LENGTH} ${CONE_LENGTH} 0 0 1 ${(
                          CONE_LENGTH * Math.sin((CONE_HALF_ANGLE * Math.PI) / 180)
                        ).toFixed(1)} ${(-CONE_LENGTH * Math.cos((CONE_HALF_ANGLE * Math.PI) / 180)).toFixed(
                          1,
                        )} Z`}
                        fill="var(--bio)"
                        fillOpacity={0.3}
                        stroke="var(--bio)"
                        strokeOpacity={0.5}
                        strokeWidth={1}
                      />
                    </g>
                    <circle r={5.5} fill="var(--bio)" stroke="var(--canvas)" strokeWidth={2} />
                  </g>
                ) : null}

                {rooms.map((room) => (
                  <RoomLabel
                    key={room.scene_id}
                    room={room}
                    viewport={viewport}
                    active={room.scene_id === currentSceneId}
                    locale={locale}
                  />
                ))}

                {bar ? (
                  <g transform={`translate(18 ${viewport.height - 22})`} className="pointer-events-none">
                    <line
                      x1={0}
                      y1={0}
                      x2={bar.pixels}
                      y2={0}
                      stroke="var(--ink-faint)"
                      strokeWidth={1.5}
                    />
                    <line x1={0} y1={-4} x2={0} y2={4} stroke="var(--ink-faint)" strokeWidth={1.5} />
                    <line
                      x1={bar.pixels}
                      y1={-4}
                      x2={bar.pixels}
                      y2={4}
                      stroke="var(--ink-faint)"
                      strokeWidth={1.5}
                    />
                    <text x={bar.pixels / 2} y={-8} textAnchor="middle" className="fill-[var(--ink-faint)] text-[10px] tabular-nums">
                      {`${bar.metres} m`}
                    </text>
                  </g>
                ) : null}
              </svg>
            ) : null}
          </div>

          <div className="shrink-0 space-y-2 px-4 pb-[max(0.85rem,env(safe-area-inset-bottom))]">
            {unlinked.length > 0 ? (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] text-ink-faint">{t("plan.unlinked", locale)}</span>
                {unlinked.map((room) => (
                  <span
                    key={room.scene_id}
                    className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-muted"
                  >
                    {room.name}
                  </span>
                ))}
              </div>
            ) : null}

            {plan.pending_scene_ids.length > 0 && onMeasure ? (
              <button
                type="button"
                onClick={onMeasure}
                disabled={measuring || !plan.can_measure}
                className={cn(
                  "flex items-center gap-2 rounded-full border px-3 py-1.5 text-[12px] font-medium",
                  "transition-colors duration-200",
                  plan.can_measure
                    ? "border-bio/45 text-bio hover:bg-bio/10"
                    : "cursor-default border-line text-ink-faint",
                )}
              >
                {measuring ? <Loader2 size={13} className="animate-spin" /> : null}
                {plan.can_measure
                  ? t("plan.measureMore", locale).replace("{n}", String(plan.pending_scene_ids.length))
                  : t("plan.noModel", locale)}
              </button>
            ) : null}
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
