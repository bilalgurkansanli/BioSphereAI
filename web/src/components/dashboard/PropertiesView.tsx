"use client";

import { Images, Plus } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { toast } from "sonner";
import { EmbedPanel } from "./EmbedPanel";
import { ApiErrorState } from "./ErrorState";
import { PropertyCard } from "./PropertyCard";
import { ui } from "./copy";
import { Button, buttonStyles } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { EmptyState } from "@/components/ui/EmptyState";
import { Input } from "@/components/ui/Input";
import { SkeletonPropertyCard } from "@/components/ui/Skeleton";
import { api } from "@/lib/api";
import { useApi } from "@/lib/hooks/useApi";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";
import type { Paginated, Property } from "@/lib/types";

export function PropertiesView() {
  const { locale } = useLocale();
  const describeError = useAppStore((s) => s.describeError);
  const { data, initial, error, refetch, setData } = useApi<Paginated<Property>>(() => api.properties.list(60), []);

  const [embedFor, setEmbedFor] = useState<Property | null>(null);
  const [renameFor, setRenameFor] = useState<Property | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleteFor, setDeleteFor] = useState<Property | null>(null);
  const [busy, setBusy] = useState(false);

  const properties = data?.items ?? [];

  const startRename = (property: Property) => {
    setRenameValue(property.title);
    setRenameFor(property);
  };

  const submitRename = async () => {
    if (!renameFor) return;
    const title = renameValue.trim();
    if (!title) return;

    setBusy(true);
    try {
      const updated = await api.properties.update(renameFor.id, { title });
      setData((current) =>
        current ? { ...current, items: current.items.map((x) => (x.id === updated.id ? updated : x)) } : current,
      );
      toast.success(ui("properties.renamed", locale));
      setRenameFor(null);
    } catch (cause) {
      toast.error(describeError(cause));
    } finally {
      setBusy(false);
    }
  };

  const submitDelete = async () => {
    if (!deleteFor) return;
    const target = deleteFor;

    setBusy(true);
    try {
      await api.properties.remove(target.id);
      setData((current) =>
        current
          ? { ...current, items: current.items.filter((x) => x.id !== target.id), total: Math.max(0, current.total - 1) }
          : current,
      );
      toast.success(ui("properties.deleted", locale));
      setDeleteFor(null);
    } catch (cause) {
      toast.error(describeError(cause));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-ink">{t("nav.properties", locale)}</h1>
          <p className="mt-1 text-[13px] text-ink-muted">
            {data && data.total > 0
              ? `${data.total} ${ui(data.total === 1 ? "properties.unit" : "properties.unit.plural", locale)}`
              : ui("properties.subtitle", locale)}
          </p>
        </div>
        <Link href="/new" className={buttonStyles("primary", "md")}>
          <Plus className="size-4" aria-hidden />
          {t("nav.new", locale)}
        </Link>
      </div>

      {initial ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {Array.from({ length: 6 }, (_, i) => (
            <SkeletonPropertyCard key={i} />
          ))}
        </div>
      ) : error && properties.length === 0 ? (
        <ApiErrorState error={error} onRetry={refetch} title={ui("properties.loadFailed", locale)} />
      ) : properties.length === 0 ? (
        <EmptyState
          icon={Images}
          title={t("property.empty.title", locale)}
          body={t("property.empty.body", locale)}
          action={
            <Link href="/new" className={buttonStyles("primary", "md")}>
              <Plus className="size-4" aria-hidden />
              {t("property.create", locale)}
            </Link>
          }
        />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {properties.map((property) => (
            <PropertyCard
              key={property.id}
              property={property}
              onEmbed={setEmbedFor}
              onRename={startRename}
              onDelete={setDeleteFor}
            />
          ))}
        </div>
      )}

      <Dialog
        open={embedFor !== null}
        onClose={() => setEmbedFor(null)}
        title={t("embed.title", locale)}
        description={embedFor?.title}
        size="lg"
        closeLabel={ui("common.close", locale)}
      >
        {embedFor ? <EmbedPanel property={embedFor} bare /> : null}
      </Dialog>

      <Dialog
        open={renameFor !== null}
        onClose={() => setRenameFor(null)}
        title={ui("properties.rename.title", locale)}
        closeLabel={ui("common.close", locale)}
        footer={
          <>
            <Button variant="ghost" onClick={() => setRenameFor(null)}>
              {t("editor.cancel", locale)}
            </Button>
            <Button variant="primary" loading={busy} disabled={!renameValue.trim()} onClick={() => void submitRename()}>
              {t("editor.save", locale)}
            </Button>
          </>
        }
      >
        <Input
          label={t("property.title", locale)}
          value={renameValue}
          autoFocus
          onChange={(event) => setRenameValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void submitRename();
          }}
        />
      </Dialog>

      <Dialog
        open={deleteFor !== null}
        onClose={() => setDeleteFor(null)}
        title={ui("properties.delete.title", locale)}
        description={t("property.delete.confirm", locale)}
        size="sm"
        closeLabel={ui("common.close", locale)}
        footer={
          <>
            <Button variant="ghost" onClick={() => setDeleteFor(null)}>
              {t("editor.cancel", locale)}
            </Button>
            <Button variant="danger" loading={busy} onClick={() => void submitDelete()}>
              {ui("properties.delete", locale)}
            </Button>
          </>
        }
      >
        <p className="text-[13px] text-ink-muted">{deleteFor?.title}</p>
      </Dialog>
    </div>
  );
}
