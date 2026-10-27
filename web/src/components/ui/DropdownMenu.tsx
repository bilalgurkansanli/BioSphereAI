"use client";

import { MoreHorizontal, type LucideIcon } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface DropdownItem {
  id: string;
  label: string;
  icon?: LucideIcon;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
  /** Draws a hairline above this item. */
  separatorBefore?: boolean;
}

export interface DropdownMenuProps {
  items: DropdownItem[];
  /** Accessible name of the trigger. */
  label: string;
  align?: "start" | "end";
  trigger?: ReactNode;
  /**
   * Static content above the items — an account's name and address, a group label.
   *
   * Needed because `items` renders only the ENABLED ones: a disabled entry is skipped
   * entirely rather than drawn greyed out, so a non-interactive row cannot be expressed as
   * an item. This sits outside the roving-focus list and never takes keyboard focus.
   */
  header?: ReactNode;
  className?: string;
  triggerClassName?: string;
}

export function DropdownMenu({
  items,
  label,
  align = "end",
  trigger,
  header,
  className,
  triggerClassName,
}: DropdownMenuProps) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const itemRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const id = useId();

  const enabled = items.filter((item) => !item.disabled);

  const close = useCallback(
    (returnFocus = true) => {
      setOpen(false);
      if (returnFocus) triggerRef.current?.focus();
    },
    [],
  );

  useEffect(() => {
    if (!open) return;

    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    // A menu anchored to a card must not float away from it while the page scrolls.
    const onScroll = () => setOpen(false);

    document.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("scroll", onScroll, true);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, true);
      window.removeEventListener("scroll", onScroll, true);
    };
  }, [open]);

  useEffect(() => {
    if (open) itemRefs.current[active]?.focus();
  }, [open, active]);

  const openAt = (index: number) => {
    setActive(index);
    setOpen(true);
  };

  return (
    <div ref={rootRef} className={cn("relative", className)}>
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? `${id}-menu` : undefined}
        aria-label={label}
        onClick={() => (open ? close(false) : openAt(0))}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openAt(0);
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            openAt(Math.max(0, enabled.length - 1));
          }
        }}
        className={cn(
          "grid size-8 place-items-center rounded-sm border border-transparent text-ink-faint transition-colors",
          "hover:border-line hover:bg-surface-3 hover:text-ink",
          open && "border-line bg-surface-3 text-ink",
          triggerClassName,
        )}
      >
        {trigger ?? <MoreHorizontal className="size-4" aria-hidden />}
      </button>

      <AnimatePresence>
        {open ? (
          <motion.div
            id={`${id}-menu`}
            role="menu"
            aria-label={label}
            initial={{ opacity: 0, y: -4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.14, ease: [0.16, 1, 0.3, 1] }}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.stopPropagation();
                close();
              } else if (event.key === "ArrowDown") {
                event.preventDefault();
                setActive((i) => (i + 1) % enabled.length);
              } else if (event.key === "ArrowUp") {
                event.preventDefault();
                setActive((i) => (i - 1 + enabled.length) % enabled.length);
              } else if (event.key === "Home") {
                event.preventDefault();
                setActive(0);
              } else if (event.key === "End") {
                event.preventDefault();
                setActive(enabled.length - 1);
              } else if (event.key === "Tab") {
                setOpen(false);
              }
            }}
            className={cn(
              "glass-strong absolute z-40 mt-1.5 min-w-48 origin-top overflow-hidden rounded-md p-1 shadow-lift",
              align === "end" ? "right-0" : "left-0",
            )}
          >
            {header ? (
              <div className="border-b border-line px-2.5 py-2 text-[12px] leading-snug text-ink-muted">
                {header}
              </div>
            ) : null}
            {enabled.map((item, index) => {
              const Icon = item.icon;
              return (
                <div key={item.id}>
                  {item.separatorBefore ? <div className="my-1 h-px bg-line" role="separator" /> : null}
                  <button
                    ref={(node) => {
                      itemRefs.current[index] = node;
                    }}
                    type="button"
                    role="menuitem"
                    tabIndex={index === active ? 0 : -1}
                    onMouseEnter={() => setActive(index)}
                    onClick={() => {
                      close(false);
                      item.onSelect();
                    }}
                    className={cn(
                      "flex w-full items-center gap-2.5 rounded-sm px-2.5 py-2 text-left text-[13px] transition-colors outline-none",
                      item.danger ? "text-danger hover:bg-danger/10" : "text-ink-muted hover:bg-surface-3 hover:text-ink",
                      index === active && (item.danger ? "bg-danger/10" : "bg-surface-3 text-ink"),
                    )}
                  >
                    {Icon ? <Icon className="size-4 shrink-0 opacity-80" aria-hidden /> : null}
                    <span className="truncate">{item.label}</span>
                  </button>
                </div>
              );
            })}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
