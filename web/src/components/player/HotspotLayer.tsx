"use client";

/**
 * Interactive tags, drawn as HTML on top of the canvas.
 *
 * HTML rather than sprites because these are typography — a label rendered into a texture
 * is never as crisp as one the browser lays out, and it cannot be selected, focused or
 * read by a screen reader. The cost is that their positions have to be recomputed by hand
 * every frame, which `projectHotspots` does by writing straight to element refs. Nothing
 * in the hot path touches React state; at 120 Hz a `setState` per tag per frame is the
 * difference between tags that feel welded to the wall and tags that swim.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { ArrowUpRight, Play, Plus, X } from "lucide-react";
import { cn, clamp } from "@/lib/utils";
import type { Hotspot } from "@/lib/types";
import { directionFromAngles } from "./useLookControls";
import type { HotspotNode, HotspotRegistry, LookControlsApi } from "./types";

// ---------------------------------------------------------------------------------
// Per-frame projection
// ---------------------------------------------------------------------------------

const _dir = new THREE.Vector3();
const _forward = new THREE.Vector3();
const _ndc = new THREE.Vector3();

/** Below this dot product with the view axis a tag is behind us and gets hidden outright. */
const FACING_CUTOFF = 0.02;
/** Radial NDC distance where tags start fading, and where they are gone. */
const FADE_START = 0.8;
const FADE_END = 1.25;

/**
 * Positions every registered tag against the current camera. Call once per frame, after
 * the camera pose has been written and its matrices refreshed.
 */
export function projectHotspots(
  registry: HotspotRegistry,
  camera: THREE.PerspectiveCamera,
  width: number,
  height: number,
): void {
  if (registry.size === 0 || width === 0 || height === 0) return;
  camera.getWorldDirection(_forward);

  for (const node of registry.values()) {
    const yaw = node.dragYaw ?? node.yaw;
    const pitch = node.dragPitch ?? node.pitch;
    directionFromAngles(yaw, pitch, _dir);

    const facing = _dir.dot(_forward);
    if (facing <= FACING_CUTOFF) {
      setVisible(node, false);
      continue;
    }

    // The tag's direction is a point one unit in front of the eye, so projecting it is
    // exactly the perspective divide the sphere itself gets.
    _ndc.copy(_dir).project(camera);
    const radial = Math.hypot(_ndc.x, _ndc.y);

    // Two independent fades: one for the frame edge, one for the sliver right at the
    // 90-degree limit where a tag would otherwise blink in and out.
    const edge = clamp((FADE_END - radial) / (FADE_END - FADE_START), 0, 1);
    const grazing = clamp((facing - FACING_CUTOFF) / 0.12, 0, 1);
    const opacity = edge * grazing;

    if (opacity <= 0.012) {
      setVisible(node, false);
      continue;
    }

    const x = (_ndc.x * 0.5 + 0.5) * width;
    const y = (-_ndc.y * 0.5 + 0.5) * height;

    setVisible(node, true);
    node.root.style.transform = `translate3d(${x.toFixed(2)}px, ${y.toFixed(2)}px, 0)`;

    if (Math.abs(opacity - node.lastOpacity) > 0.01) {
      node.root.style.opacity = opacity.toFixed(3);
      node.lastOpacity = opacity;
    }

    // The tag nearest the middle of the frame is the one being looked at; a couple of
    // percent of extra size is enough to say so without turning into a zoom effect.
    if (node.marker) {
      const scale = 0.84 + 0.2 * (1 - clamp(radial, 0, 1));
      node.marker.style.transform = `translate(-50%, -50%) scale(${scale.toFixed(3)})`;
    }

    const z = 40 + Math.round((1 - clamp(radial, 0, 1)) * 40);
    if (z !== node.lastZ) {
      node.root.style.zIndex = String(z);
      node.lastZ = z;
    }

    // Flip the detail card away from whichever edge it would otherwise run off.
    if (node.card) {
      const side = x > width * 0.58 ? "left" : "right";
      const vside = y < height * 0.3 ? "bottom" : y > height * 0.7 ? "top" : "middle";
      if (side !== node.lastSide || vside !== node.lastVSide) {
        node.lastSide = side;
        node.lastVSide = vside;
        node.card.style.transform = cardTransform(side, vside);
      }
    }
  }
}

function setVisible(node: HotspotNode, visible: boolean): void {
  if (node.lastVisible === visible) return;
  node.lastVisible = visible;
  node.root.style.visibility = visible ? "visible" : "hidden";
  node.root.style.pointerEvents = visible ? "auto" : "none";
}

function cardTransform(side: string, vside: string): string {
  const horizontal = side === "left" ? "calc(-100% - 26px)" : "26px";
  const vertical = vside === "top" ? "calc(-100% - 14px)" : vside === "bottom" ? "14px" : "-50%";
  return `translate(${horizontal}, ${vertical})`;
}

// ---------------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------------

const VIDEO_PATTERN = /\.(mp4|webm|mov|m4v)(\?|$)/i;

export interface HotspotLayerProps {
  hotspots: Hotspot[];
  /** Scene id -> room name, so a link tag can name where it goes. */
  sceneNames: Map<string, string>;
  editable: boolean;
  registry: React.RefObject<HotspotRegistry>;
  controls: React.RefObject<LookControlsApi | null>;
  onHotspotClick?: (h: Hotspot) => void;
  onHotspotDragEnd?: (h: Hotspot, p: { yaw: number; pitch: number }) => void;
  onNavigate?: (h: Hotspot) => void;
}

export function HotspotLayer({
  hotspots,
  sceneNames,
  editable,
  registry,
  controls,
  onHotspotClick,
  onHotspotDragEnd,
  onNavigate,
}: HotspotLayerProps) {
  const [openId, setOpenId] = useState<string | null>(null);
  const dragging = useRef<{ id: string; moved: boolean } | null>(null);

  // Tags that leave the list must not leave stale nodes behind, or the projector keeps
  // writing transforms to detached elements forever. A stale `openId` needs no cleanup:
  // the card only renders inside a tag that still exists.
  useEffect(() => {
    const live = new Set(hotspots.map((h) => h.id));
    const store = registry.current;
    for (const id of [...store.keys()]) {
      if (!live.has(id)) store.delete(id);
    }
  }, [hotspots, registry]);

  const register = useCallback(
    (hotspot: Hotspot) => (element: HTMLDivElement | null) => {
      const store = registry.current;
      // Detach is a no-op on purpose. This callback's identity changes every render, so
      // React detaches and re-attaches constantly; deleting here would discard the node —
      // and with it an in-flight drag's yaw override — on any unrelated re-render. Truly
      // departed tags are pruned by the effect above instead.
      if (!element) return;
      const existing = store.get(hotspot.id);
      const node: HotspotNode = existing ?? {
        root: element,
        marker: null,
        card: null,
        yaw: hotspot.yaw,
        pitch: hotspot.pitch,
        dragYaw: null,
        dragPitch: null,
        baseYaw: hotspot.yaw,
        lastVisible: null,
        lastSide: "",
        lastVSide: "",
        lastOpacity: -1,
        lastZ: -1,
      };
      // A DIFFERENT element means every cached "last written" value on this node describes a
      // node that is no longer in the document. The projector's guards then short-circuit
      // against those stale values and never write anything to the element that IS on screen,
      // which keeps whatever it was born with — for a tag, hidden. Surfaced when the player
      // began unmounting this layer for its dark-room state: the tag came back registered,
      // projected every frame, and permanently invisible. Any remount would have done it.
      if (node.root !== element) {
        node.lastVisible = null;
        node.lastOpacity = -1;
        node.lastZ = -1;
        node.lastSide = "";
        node.lastVSide = "";
        node.card = null;
      }
      node.root = element;
      node.yaw = hotspot.yaw;
      node.pitch = hotspot.pitch;
      // Once the parent has persisted a drag, drop the local override so the tag is
      // driven by props again instead of drifting out of sync with the server.
      if (node.dragYaw !== null && hotspot.yaw !== node.baseYaw) {
        node.dragYaw = null;
        node.dragPitch = null;
      }
      node.marker = element.querySelector<HTMLElement>("[data-hotspot-marker]");
      store.set(hotspot.id, node);
    },
    [registry],
  );

  /**
   * The card mounts and unmounts long after its tag does, so it registers itself rather
   * than being discovered from the root. Clearing the cached sides forces the projector
   * to write a fresh anchor on the very next frame instead of waiting for the view to
   * move across a flip boundary.
   */
  const registerCard = useCallback(
    (id: string) => (element: HTMLDivElement | null) => {
      const node = registry.current.get(id);
      if (!node) return;
      node.card = element;
      node.lastSide = "";
      node.lastVSide = "";
    },
    [registry],
  );

  const beginDrag = useCallback(
    (hotspot: Hotspot, event: React.PointerEvent<HTMLButtonElement>) => {
      if (!editable) return;
      const node = registry.current.get(hotspot.id);
      if (!node) return;

      event.preventDefault();
      event.stopPropagation();
      const target = event.currentTarget;
      target.setPointerCapture(event.pointerId);
      dragging.current = { id: hotspot.id, moved: false };
      node.baseYaw = hotspot.yaw;

      const move = (e: PointerEvent) => {
        const dir = controls.current?.directionAt(e.clientX, e.clientY);
        if (!dir) return;
        dragging.current = { id: hotspot.id, moved: true };
        node.dragYaw = dir.yaw;
        node.dragPitch = dir.pitch;
      };

      const finish = (e: PointerEvent) => {
        target.releasePointerCapture?.(e.pointerId);
        target.removeEventListener("pointermove", move);
        target.removeEventListener("pointerup", finish);
        target.removeEventListener("pointercancel", finish);
        const moved = dragging.current?.moved ?? false;
        dragging.current = null;
        if (moved && node.dragYaw !== null && node.dragPitch !== null) {
          onHotspotDragEnd?.(hotspot, { yaw: node.dragYaw, pitch: node.dragPitch });
        }
      };

      target.addEventListener("pointermove", move);
      target.addEventListener("pointerup", finish);
      target.addEventListener("pointercancel", finish);
    },
    [editable, registry, controls, onHotspotDragEnd],
  );

  const activate = useCallback(
    (hotspot: Hotspot) => {
      // A drag that ended on the marker must not also read as a click.
      if (dragging.current?.moved) return;
      onHotspotClick?.(hotspot);
      if (hotspot.type === "link" && !editable) {
        onNavigate?.(hotspot);
        return;
      }
      setOpenId((current) => (current === hotspot.id ? null : hotspot.id));
    },
    [onHotspotClick, onNavigate, editable],
  );

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      {hotspots.map((hotspot) => {
        const accent = hotspot.accent ?? "var(--bio)";
        const targetName = hotspot.target_scene_id ? sceneNames.get(hotspot.target_scene_id) : undefined;
        const label = hotspot.title || targetName || "";
        const open = openId === hotspot.id;
        const hasCard = hotspot.type !== "link" && (Boolean(hotspot.body) || Boolean(hotspot.media_url) || Boolean(hotspot.title));

        return (
          <div
            key={hotspot.id}
            ref={register(hotspot)}
            className="absolute left-0 top-0 h-0 w-0"
            style={{ visibility: "hidden", willChange: "transform, opacity" }}
          >
            <button
              type="button"
              data-player-ui
              data-hotspot-marker
              aria-label={label || hotspot.type}
              aria-expanded={hasCard ? open : undefined}
              onPointerDown={(event) => beginDrag(hotspot, event)}
              onClick={() => activate(hotspot)}
              className={cn(
                "group/hs absolute left-0 top-0 flex origin-center items-center rounded-full",
                "no-tap outline-none",
                editable ? "cursor-move" : "cursor-pointer",
              )}
              style={{ transform: "translate(-50%, -50%)" }}
            >
              <span className="relative grid h-10 w-10 shrink-0 place-items-center">
                {/* Slow outward pulse: enough to catch the eye in peripheral vision,
                    quiet enough to sit in a photograph. Suppressed by the global
                    reduced-motion rule in globals.css. */}
                <span
                  aria-hidden
                  className="absolute inset-1 animate-ping rounded-full opacity-30"
                  style={{ backgroundColor: accent, animationDuration: "2.6s" }}
                />
                <span
                  aria-hidden
                  className="absolute inset-0 rounded-full border transition-transform duration-300 ease-[var(--ease-out-expo)] group-hover/hs:scale-110"
                  style={{ borderColor: accent, opacity: 0.55 }}
                />
                <span
                  className={cn(
                    "glass-strong relative grid h-[26px] w-[26px] place-items-center rounded-full",
                    "transition-all duration-300 ease-[var(--ease-out-expo)]",
                    "group-hover/hs:scale-105 group-focus-visible/hs:scale-105",
                  )}
                  style={{ color: accent, boxShadow: `0 0 18px -6px ${accent}` }}
                >
                  {hotspot.type === "link" ? (
                    <ArrowUpRight size={14} strokeWidth={2.4} />
                  ) : hotspot.type === "media" ? (
                    <Play size={12} strokeWidth={2.4} className="translate-x-px fill-current" />
                  ) : (
                    <Plus size={14} strokeWidth={2.6} />
                  )}
                </span>
              </span>

              {label ? (
                <span
                  className={cn(
                    "glass ml-1 flex max-w-0 items-center overflow-hidden whitespace-nowrap rounded-full",
                    "text-[12px] font-medium tracking-tight text-ink opacity-0",
                    "transition-[max-width,opacity,padding] duration-[380ms] ease-[var(--ease-out-expo)]",
                    "group-hover/hs:max-w-[15rem] group-hover/hs:px-3 group-hover/hs:py-1.5 group-hover/hs:opacity-100",
                    "group-focus-visible/hs:max-w-[15rem] group-focus-visible/hs:px-3 group-focus-visible/hs:py-1.5 group-focus-visible/hs:opacity-100",
                    open && "max-w-[15rem] px-3 py-1.5 opacity-100",
                  )}
                >
                  {hotspot.type === "link" && targetName ? (
                    <>
                      <span className="text-ink-muted">{label}</span>
                      {label !== targetName ? (
                        <span className="ml-1.5 border-l border-line pl-1.5 text-ink">{targetName}</span>
                      ) : null}
                    </>
                  ) : (
                    label
                  )}
                </span>
              ) : null}
            </button>

            {hasCard && open ? (
              // Two elements, deliberately. The outer one owns the anchor transform that
              // `projectHotspots` writes; the inner one owns the entrance animation.
              // They cannot be the same element: a CSS animation outranks inline styles in
              // the cascade, so `animate-rise` — which ends on `transform: none` with
              // fill-mode `both` — would pin the card to identity forever and silently
              // defeat the edge flip, sending cards off-screen near the right edge.
              <div
                data-player-ui
                data-hotspot-card
                ref={registerCard(hotspot.id)}
                className="absolute left-0 top-0 w-[min(19rem,68vw)]"
                style={{ transform: cardTransform("right", "middle") }}
              >
                <div
                  role="dialog"
                  aria-label={hotspot.title}
                  className={cn(
                    "glass-strong animate-rise w-full rounded-[var(--radius-md)]",
                    "p-3.5 shadow-[var(--shadow-lift)]",
                  )}
                >
                <div className="flex items-start justify-between gap-3">
                  <h3 className="text-[13px] font-semibold leading-snug tracking-tight text-ink">{hotspot.title}</h3>
                  <button
                    type="button"
                    data-player-ui
                    onClick={() => setOpenId(null)}
                    className="-m-1 rounded-[var(--radius-xs)] p-1 text-ink-faint transition-colors hover:text-ink focus-visible:text-ink"
                    aria-label="Close"
                  >
                    <X size={14} />
                  </button>
                </div>

                {hotspot.media_url ? (
                  <div className="mt-2.5 overflow-hidden rounded-[var(--radius-sm)] border border-line bg-black/40">
                    {VIDEO_PATTERN.test(hotspot.media_url) ? (
                      <video src={hotspot.media_url} controls playsInline className="block w-full" />
                    ) : (
                      // Deliberately a plain <img>: the engine's media host is outside
                      // next/image's optimiser and these are already sized assets.
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={hotspot.media_url} alt={hotspot.title} className="block w-full" loading="lazy" />
                    )}
                  </div>
                ) : null}

                {hotspot.body ? (
                  <p className="mt-2 text-[12.5px] leading-relaxed text-ink-muted">{hotspot.body}</p>
                ) : null}

                  <span
                    aria-hidden
                    className="mt-3 block h-px w-full"
                    style={{ background: `linear-gradient(90deg, ${accent}, transparent)` }}
                  />
                </div>
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
