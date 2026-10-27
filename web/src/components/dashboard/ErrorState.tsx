"use client";

import { RotateCcw, TriangleAlert, Unplug } from "lucide-react";
import { ui } from "./copy";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { BioSphereError } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";

/** A dead engine and a failed request need different advice, so they get different copy. */
export function isOffline(error: BioSphereError | null): boolean {
  if (!error) return false;
  return error.status === 0 || error.detail.code === "unreachable" || error.detail.code === "network_error";
}

export interface ApiErrorStateProps {
  error: BioSphereError | null;
  onRetry: () => void;
  /** Overrides the headline for non-connection failures. */
  title?: string;
  className?: string;
}

export function ApiErrorState({ error, onRetry, title, className }: ApiErrorStateProps) {
  const { locale } = useLocale();
  const offline = isOffline(error);

  return (
    <EmptyState
      className={className}
      tone="danger"
      icon={offline ? Unplug : TriangleAlert}
      title={offline ? ui("common.offline.title", locale) : (title ?? t("error.generic", locale))}
      body={offline ? t("error.offline", locale) : (error?.localised(locale) ?? t("error.generic", locale))}
      action={
        <Button variant="secondary" onClick={onRetry}>
          <RotateCcw className="size-4" aria-hidden />
          {t("job.retry", locale)}
        </Button>
      }
      secondary={
        error?.detail.hint && !offline ? (
          <p className="max-w-md font-mono text-[11px] leading-relaxed break-words text-ink-faint">
            {error.detail.hint}
          </p>
        ) : null
      }
    />
  );
}
