"use client";

/**
 * Room navigation.
 *
 * A permanently visible thumbnail strip would be the obvious layout and the wrong one:
 * the panorama is the product, and a ten-room property would give a fifth of a phone screen
 * over to chrome. So the strip is a drawer — collapsed to the current room's name, opened
 * on demand, and closed again the moment a room is picked.
 */

import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Layers } from "lucide-react";
import { absoluteUrl } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { Scene } from "@/lib/types";

export interface RoomSwitcherProps {
  scenes: Scene[];
  currentSceneId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelect: (sceneId: string) => void;
  label: string;
}

export function RoomSwitcher({ scenes, currentSceneId, open, onOpenChange, onSelect, label }: RoomSwitcherProps) {
  const stripRef = useRef<HTMLDivElement>(null);
  const current = scenes.find((s) => s.id === currentSceneId) ?? null;

  // Keep the active thumbnail in view when the drawer opens on a long property.
  useEffect(() => {
    if (!open) return;
    const active = stripRef.current?.querySelector<HTMLElement>("[data-active='true']");
    active?.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
  }, [open, currentSceneId]);

  if (scenes.length <= 1) return null;

  return (
    <>
      <AnimatePresence>
        {open ? (
          <motion.div
            key="strip"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.26, ease: [0.16, 1, 0.3, 1] }}
            data-player-ui
            // Hugs its thumbnails on a short property and only starts scrolling on a long
            // one; a fixed-width drawer holding two rooms reads as a layout bug.
            className="absolute bottom-[calc(100%+10px)] left-1/2 w-max max-w-[min(34rem,calc(100vw-1.5rem))] -translate-x-1/2"
          >
            <div className="glass-strong rounded-[var(--radius-lg)] p-2 shadow-[var(--shadow-lift)]">
              <div
                ref={stripRef}
                role="listbox"
                aria-label={label}
                className="flex gap-2 overflow-x-auto overscroll-x-contain scroll-smooth pb-0.5"
              >
                {scenes.map((scene) => {
                  const active = scene.id === currentSceneId;
                  const thumb = absoluteUrl(scene.panorama?.preview_url);
                  return (
                    <button
                      key={scene.id}
                      type="button"
                      data-player-ui
                      data-active={active}
                      role="option"
                      aria-selected={active}
                      onClick={() => {
                        onSelect(scene.id);
                        onOpenChange(false);
                      }}
                      className={cn(
                        "group relative w-[104px] shrink-0 overflow-hidden rounded-[var(--radius-sm)]",
                        "border transition-all duration-300 ease-[var(--ease-out-expo)]",
                        active
                          ? "ring-glow border-bio/60"
                          : "border-line hover:border-line-strong hover:-translate-y-0.5",
                      )}
                    >
                      <span className="block aspect-[16/9] w-full bg-surface-2">
                        {thumb ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={thumb}
                            alt=""
                            loading="lazy"
                            decoding="async"
                            className={cn(
                              "h-full w-full object-cover transition-[opacity,transform] duration-500 ease-[var(--ease-out-expo)]",
                              active ? "opacity-100" : "opacity-70 group-hover:opacity-100 group-hover:scale-[1.04]",
                            )}
                          />
                        ) : null}
                      </span>
                      <span
                        className={cn(
                          "block truncate px-2 py-1.5 text-left text-[11px] font-medium tracking-tight",
                          active ? "text-ink" : "text-ink-muted group-hover:text-ink",
                        )}
                      >
                        {scene.name}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <button
        type="button"
        data-player-ui
        aria-label={label}
        aria-expanded={open}
        onClick={() => onOpenChange(!open)}
        className={cn(
          "flex h-8 min-w-0 items-center gap-1.5 rounded-full border px-2.5",
          "text-[12px] font-medium tracking-tight transition-colors duration-200 ease-[var(--ease-out-expo)]",
          open
            ? "border-bio/50 bg-bio/10 text-ink"
            : "border-line text-ink-muted hover:border-line-strong hover:bg-white/[0.04] hover:text-ink",
        )}
      >
        <Layers size={14} className="shrink-0" />
        {/* Truncates rather than pushing the rest of the toolbar off a 360px screen. */}
        <span className="max-w-[4.5rem] truncate sm:max-w-[7.5rem]">{current?.name ?? label}</span>
        <span className="shrink-0 text-ink-faint tabular-nums">
          {Math.max(scenes.findIndex((s) => s.id === currentSceneId) + 1, 1)}/{scenes.length}
        </span>
      </button>
    </>
  );
}
