"use client";

import { Code2, Eye, Layers, PencilLine, SquareArrowOutUpRight, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import type { CSSProperties } from "react";
import { ui } from "./copy";
import { LogoMark } from "./Logo";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { DropdownMenu } from "@/components/ui/DropdownMenu";
import { absoluteUrl } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import type { Property } from "@/lib/types";
import { relativeTime } from "@/lib/utils";

const ACCENTS = ["--bio", "--sky", "--violet", "--amber"] as const;

function hash(seed: string): number {
  let h = 0;
  for (let i = 0; i < seed.length; i += 1) h = (h * 31 + seed.charCodeAt(i)) | 0;
  return Math.abs(h);
}

/**
 * A property with no cover still needs to be recognisable in a grid, so the id seeds a stable
 * two-tone gradient — the same property always gets the same colour.
 */
function coverStyle(seed: string): CSSProperties {
  const h = hash(seed);
  const a = ACCENTS[h % ACCENTS.length];
  const b = ACCENTS[(h + 1 + (h % 2)) % ACCENTS.length];
  const angle = 105 + (h % 70);
  return {
    backgroundImage:
      `linear-gradient(${angle}deg, color-mix(in oklab, var(${a}) 34%, var(--surface)) 0%, ` +
      `color-mix(in oklab, var(${b}) 20%, var(--canvas)) 100%)`,
  };
}

export interface PropertyCardProps {
  property: Property;
  onEmbed: (property: Property) => void;
  onRename: (property: Property) => void;
  onDelete: (property: Property) => void;
}

export function PropertyCard({ property, onEmbed, onRename, onDelete }: PropertyCardProps) {
  const { locale } = useLocale();
  const router = useRouter();
  const cover = absoluteUrl(property.cover_url);

  return (
    <Card interactive className="animate-rise group relative flex flex-col">
      <div className="relative aspect-[16/10] overflow-hidden rounded-t-lg border-b border-line">
        {cover ? (
          // Panoramas are served by the engine on a local origin, which Next's optimiser
          // refuses by default — the raw element keeps the thumbnail working everywhere.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={cover}
            alt=""
            loading="lazy"
            decoding="async"
            className="size-full object-cover transition-transform duration-500 ease-out-expo group-hover:scale-[1.03]"
          />
        ) : (
          <div className="relative size-full" style={coverStyle(property.id)}>
            <div
              className="absolute inset-0 bg-[repeating-linear-gradient(90deg,transparent_0_22px,rgb(255_255_255/0.05)_22px_23px)]"
              aria-hidden
            />
            <div className="absolute inset-0 grid place-items-center">
              <LogoMark className="size-9 opacity-25 grayscale" />
            </div>
            <span className="sr-only">{ui("properties.noCover", locale)}</span>
          </div>
        )}

        <div className="absolute top-2.5 left-2.5">
          {property.published ? (
            <Badge tone="bio" dot>
              {t("property.published", locale)}
            </Badge>
          ) : (
            <Badge tone="neutral" dot>
              {t("property.draft", locale)}
            </Badge>
          )}
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-1 p-4">
        <div className="flex items-start justify-between gap-2">
          <h3 className="min-w-0 text-[15px] leading-snug font-semibold tracking-tight text-ink">
            {/* Stretched link: the whole card is the hit target, the menu sits above it. */}
            <Link href={`/evler/${property.id}`} className="line-clamp-2 after:absolute after:inset-0 after:content-['']">
              {property.title}
            </Link>
          </h3>
          <div className="relative z-10 -mt-1 -mr-1">
            <DropdownMenu
              label={property.title}
              items={[
                {
                  id: "open",
                  label: ui("properties.open", locale),
                  icon: SquareArrowOutUpRight,
                  onSelect: () => router.push(`/evler/${property.id}`),
                },
                { id: "embed", label: ui("properties.embed", locale), icon: Code2, onSelect: () => onEmbed(property) },
                { id: "rename", label: ui("properties.rename", locale), icon: PencilLine, onSelect: () => onRename(property) },
                {
                  id: "delete",
                  label: ui("properties.delete", locale),
                  icon: Trash2,
                  danger: true,
                  separatorBefore: true,
                  onSelect: () => onDelete(property),
                },
              ]}
            />
          </div>
        </div>

        <p className="line-clamp-1 text-[13px] text-ink-faint">{property.address ?? "—"}</p>

        {/* Icons carry the meaning so the row never wraps and every card stays the same
            height; the words stay available to screen readers. */}
        <div className="mt-auto flex items-center gap-3 pt-3 text-[12px] text-ink-faint">
          <span className="inline-flex shrink-0 items-center gap-1.5">
            <Layers className="size-3.5" aria-hidden />
            {property.scene_count}
            <span className="sr-only">{t("property.scenes", locale)}</span>
          </span>
          <span className="inline-flex shrink-0 items-center gap-1.5">
            <Eye className="size-3.5" aria-hidden />
            {property.views}
            <span className="sr-only">{t("property.views", locale)}</span>
          </span>
          <span className="ml-auto truncate">{relativeTime(property.updated_at, locale)}</span>
        </div>
      </div>
    </Card>
  );
}
