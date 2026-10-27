"use client";

/**
 * Every room in the property, by name, along the top.
 *
 * ── WHY THIS EXISTS WHEN THERE ARE ALREADY TWO WAYS TO CHANGE ROOM ────────────────────
 * The floor rings are the good way: they are where the room is, they say "walk here", and a
 * property built well out of them needs nothing else. But they only work if the person who shot
 * the property PLACED them — in the corridor, in the doorway, at every junction — and that is
 * work, on site, that nobody can be made to do. A property with one panorama per room and no
 * rings between them is not a broken property; it is the ordinary case.
 *
 * So this is the floor that holds when the rings are missing: a flat list of names, one tap
 * each, always visible, no discovery required. `RoomSwitcher` at the bottom is the same idea
 * with thumbnails behind a drawer — better for recognising a room you have seen, worse for
 * finding one you have not, and hidden until you look for it.
 *
 * ── IT IS TEXT, ON PURPOSE ────────────────────────────────────────────────────────────
 * A thumbnail of an empty white room looks like a thumbnail of a different empty white room.
 * The name is the only thing that distinguishes "Çocuk odası" from "Yatak odası" at a glance,
 * which is exactly what somebody who cannot find the kitchen needs.
 */

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import type { Scene } from "@/lib/types";

export interface RoomBarProps {
  scenes: Scene[];
  currentSceneId: string | null;
  visible: boolean;
  label: string;
  onSelect: (sceneId: string) => void;
  className?: string;
}

export function RoomBar({
  scenes,
  currentSceneId,
  visible,
  label,
  onSelect,
  className,
}: RoomBarProps) {
  const stripRef = useRef<HTMLDivElement>(null);

  // Keep the current room in view on a long property, the way the switcher does when it opens.
  useEffect(() => {
    const active = stripRef.current?.querySelector<HTMLElement>("[data-active='true']");
    active?.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
  }, [currentSceneId]);

  // One room is not a property, and a bar naming it would be chrome that says nothing.
  if (scenes.length <= 1) return null;

  return (
    <div
      data-player-ui
      className={cn(
        "pointer-events-none absolute inset-x-0 top-0 z-30 flex justify-center",
        "px-3 pt-[max(0.75rem,env(safe-area-inset-top))]",
        "transition-opacity duration-300 ease-[var(--ease-out-expo)]",
        visible ? "opacity-100" : "pointer-events-none opacity-0",
        className,
      )}
    >
      <div
        ref={stripRef}
        role="tablist"
        aria-label={label}
        className={cn(
          "glass-strong pointer-events-auto flex max-w-full gap-1 overflow-x-auto rounded-full p-1",
          "shadow-[var(--shadow-lift)] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden",
          visible ? "" : "pointer-events-none",
        )}
      >
        {scenes.map((scene) => {
          const active = scene.id === currentSceneId;
          return (
            <button
              key={scene.id}
              type="button"
              role="tab"
              aria-selected={active}
              data-active={active}
              onClick={() => onSelect(scene.id)}
              className={cn(
                "shrink-0 rounded-full px-3 py-1.5 text-[12px] font-medium whitespace-nowrap",
                "transition-colors duration-150",
                active
                  ? "bg-bio/15 text-bio"
                  : "text-ink-muted hover:bg-white/[0.06] hover:text-ink",
              )}
            >
              {scene.name}
            </button>
          );
        })}
      </div>
    </div>
  );
}
