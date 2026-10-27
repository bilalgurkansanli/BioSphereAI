"use client";

import type { LucideIcon } from "lucide-react";
import { motion } from "motion/react";
import { useRef, type ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface TabItem {
  id: string;
  label: string;
  icon?: LucideIcon;
  badge?: ReactNode;
}

export interface TabsProps {
  items: TabItem[];
  value: string;
  onChange: (id: string) => void;
  /** Namespace for the generated tab/panel ids so several tab sets can coexist. */
  idBase: string;
  className?: string;
  size?: "sm" | "md";
}

/** Spread onto the element that renders a tab's content. */
export function tabPanelProps(idBase: string, id: string) {
  return {
    id: `${idBase}-panel-${id}`,
    role: "tabpanel" as const,
    "aria-labelledby": `${idBase}-tab-${id}`,
    tabIndex: 0,
  };
}

export function Tabs({ items, value, onChange, idBase, className, size = "md" }: TabsProps) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);

  const move = (from: number, delta: number) => {
    const next = (from + delta + items.length) % items.length;
    onChange(items[next].id);
    refs.current[next]?.focus();
  };

  return (
    <div
      role="tablist"
      aria-orientation="horizontal"
      className={cn("flex items-center gap-1 border-b border-line", className)}
    >
      {items.map((item, index) => {
        const selected = item.id === value;
        const Icon = item.icon;
        return (
          <button
            key={item.id}
            ref={(node) => {
              refs.current[index] = node;
            }}
            type="button"
            role="tab"
            id={`${idBase}-tab-${item.id}`}
            aria-selected={selected}
            aria-controls={`${idBase}-panel-${item.id}`}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(item.id)}
            onKeyDown={(event) => {
              if (event.key === "ArrowRight") {
                event.preventDefault();
                move(index, 1);
              } else if (event.key === "ArrowLeft") {
                event.preventDefault();
                move(index, -1);
              } else if (event.key === "Home") {
                event.preventDefault();
                move(0, 0);
              } else if (event.key === "End") {
                event.preventDefault();
                move(items.length - 1, 0);
              }
            }}
            className={cn(
              "relative flex items-center gap-1.5 rounded-t-sm font-medium transition-colors duration-150",
              size === "sm" ? "px-2.5 py-2 text-[12px]" : "px-3 py-2.5 text-[13px]",
              selected ? "text-ink" : "text-ink-faint hover:text-ink-muted",
            )}
          >
            {Icon ? <Icon className="size-3.5" aria-hidden /> : null}
            {item.label}
            {item.badge}
            {selected ? (
              <motion.span
                layoutId={`${idBase}-tab-underline`}
                className="absolute inset-x-0 -bottom-px h-px bg-bio"
                transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
              />
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
