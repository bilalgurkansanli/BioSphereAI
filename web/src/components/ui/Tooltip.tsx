"use client";

import { AnimatePresence, motion } from "motion/react";
import { cloneElement, useEffect, useId, useRef, useState, type ReactElement, type ReactNode } from "react";
import { cn } from "@/lib/utils";

export type TooltipSide = "top" | "bottom" | "left" | "right";

const SIDES: Record<TooltipSide, string> = {
  top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
  bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
  left: "right-full top-1/2 -translate-y-1/2 mr-2",
  right: "left-full top-1/2 -translate-y-1/2 ml-2",
};

const OFFSETS: Record<TooltipSide, { x: number; y: number }> = {
  top: { x: 0, y: 4 },
  bottom: { x: 0, y: -4 },
  left: { x: 4, y: 0 },
  right: { x: -4, y: 0 },
};

export interface TooltipProps {
  content: ReactNode;
  side?: TooltipSide;
  delay?: number;
  /** A single element — it receives `aria-describedby` so the tip is announced. */
  children: ReactElement<{ "aria-describedby"?: string }>;
  className?: string;
}

export function Tooltip({ content, side = "top", delay = 120, children, className }: TooltipProps) {
  const [open, setOpen] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const id = useId();

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const show = () => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setOpen(true), delay);
  };
  const hide = () => {
    if (timer.current) clearTimeout(timer.current);
    setOpen(false);
  };

  return (
    <span
      className="relative inline-flex"
      onPointerEnter={show}
      onPointerLeave={hide}
      onFocusCapture={show}
      onBlurCapture={hide}
      // Escape must dismiss a tip that is covering the thing you are trying to read.
      onKeyDown={(event) => {
        if (event.key === "Escape") hide();
      }}
    >
      {cloneElement(children, { "aria-describedby": open ? id : undefined })}
      <AnimatePresence>
        {open ? (
          <motion.span
            role="tooltip"
            id={id}
            initial={{ opacity: 0, ...OFFSETS[side] }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            exit={{ opacity: 0, ...OFFSETS[side] }}
            transition={{ duration: 0.14, ease: [0.16, 1, 0.3, 1] }}
            className={cn(
              "glass-strong pointer-events-none absolute z-50 w-max max-w-64 rounded-sm px-2.5 py-1.5",
              "text-[12px] leading-snug font-medium text-ink shadow-lift",
              SIDES[side],
              className,
            )}
          >
            {content}
          </motion.span>
        ) : null}
      </AnimatePresence>
    </span>
  );
}
