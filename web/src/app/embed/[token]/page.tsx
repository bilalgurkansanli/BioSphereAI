/**
 * The public embed: one property, one player, nothing else.
 *
 * A server component on purpose. The property is fetched on this side so a listing portal's
 * crawler gets real OpenGraph tags instead of an empty shell, and so the visitor's first
 * paint is not gated on a client round-trip to an engine that may be on a different host.
 *
 * Deep links matter here: portals paste a URL per listing, sometimes per room, so
 * `?scene`, `?yaw`, `?pitch`, `?fov` and `?autorotate` are all honoured. Rather than
 * driving the player imperatively after mount — which would show the default framing for
 * a frame and then jump — the requested view is folded into the scene data itself, so the
 * very first rendered frame is already the one that was linked to.
 */

import { cache } from "react";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { absoluteUrl, api } from "@/lib/api";
import { DEFAULT_LOCALE, t } from "@/lib/i18n";
import { clamp, radians } from "@/lib/utils";
import type { FloorPlan, InitialView, Scene, PropertyDetail } from "@/lib/types";
import { EmbedSwitcher, type EmbedMode } from "@/components/walk/EmbedSwitcher";

export const dynamic = "force-dynamic";

type SearchParams = Record<string, string | string[] | undefined>;

interface EmbedPageProps {
  params: Promise<{ token: string }>;
  searchParams: Promise<SearchParams>;
}

/** `cache` dedupes the fetch between `generateMetadata` and the page render. */
const loadProperty = cache(async (token: string): Promise<PropertyDetail | null> => {
  try {
    return await api.public.property(token);
  } catch {
    return null;
  }
});

/**
 * The plan, fetched here rather than by the player.
 *
 * Server-side for the same reason the property is: the visitor's first paint should not wait on
 * a client round-trip to an engine that may be on another host. `null` is an ordinary
 * answer — an old engine without the route, a property whose rooms were never measured — and it
 * simply means no plan button, so it is never allowed to fail the page.
 */
const loadPlan = cache(async (token: string): Promise<FloorPlan | null> => {
  try {
    return await api.public.plan(token);
  } catch {
    return null;
  }
});

function first(value: string | string[] | undefined): string | null {
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

/** Accepts radians, or a `123deg`/`123d` suffix, because portals will paste both. */
function parseAngle(raw: string | null, limit: number): number | null {
  if (!raw) return null;
  const degreeForm = /^(-?\d+(?:\.\d+)?)\s*(?:deg|d|°)$/i.exec(raw.trim());
  const value = degreeForm ? radians(Number(degreeForm[1])) : Number(raw);
  if (!Number.isFinite(value)) return null;
  return clamp(value, -limit, limit);
}

export async function generateMetadata({ params }: { params: Promise<{ token: string }> }): Promise<Metadata> {
  const { token } = await params;
  const property = await loadProperty(token);
  const name = t("app.name", DEFAULT_LOCALE);

  if (!property) {
    return { title: { absolute: name }, description: t("app.tagline", DEFAULT_LOCALE) };
  }

  const cover = absoluteUrl(property.cover_url);
  const description = property.description ?? property.address ?? t("app.tagline", DEFAULT_LOCALE);

  // Nested metadata replaces the root's `openGraph` wholesale rather than merging into
  // it, so every field this page cares about is restated here.
  return {
    title: { absolute: `${property.title} · ${name}` },
    description,
    openGraph: {
      type: "website",
      title: property.title,
      description,
      siteName: name,
      images: cover ? [{ url: cover, width: 1200, height: 630, alt: property.title }] : undefined,
    },
    twitter: {
      card: cover ? "summary_large_image" : "summary",
      title: property.title,
      description,
      images: cover ? [cover] : undefined,
    },
  };
}

export default async function EmbedPage({ params, searchParams }: EmbedPageProps) {
  const { token } = await params;
  const query = await searchParams;
  const [property, plan] = await Promise.all([loadProperty(token), loadPlan(token)]);

  if (!property) notFound();

  const playable = property.scenes.filter((scene) => scene.panorama !== null);
  // The 3D walkthrough, when it is built. `?mode=pano` opens the 360 instead; `?scene`
  // names a room in either world (the room's 3D area shares the scene's id).
  const manifest = property.walkthrough?.status === "ready" ? (property.walkthrough.manifest ?? null) : null;
  const requestedMode = first(query.mode);
  const initialMode: EmbedMode = manifest && requestedMode !== "pano" && requestedMode !== "360" ? "walk" : "pano";

  if (playable.length === 0 && !manifest) {
    // Not an error: the property exists, its rooms are still being stitched. Say so plainly
    // rather than showing a 404 to someone who followed a perfectly good link.
    return (
      <div className="grid h-full w-full place-items-center bg-canvas px-6 text-center">
        <div className="animate-rise flex max-w-sm flex-col items-center gap-3">
          <span aria-hidden className="animate-breathe bio-gradient h-1 w-10 rounded-full" />
          <h1 className="text-[15px] font-semibold tracking-tight text-ink">{property.title}</h1>
          <p className="text-[13px] text-ink-muted">{t("player.empty", DEFAULT_LOCALE)}</p>
        </div>
      </div>
    );
  }

  const requestedSceneId = first(query.scene);
  const target = playable.find((scene) => scene.id === requestedSceneId) ?? playable[0] ?? null;
  const walkStart =
    manifest && requestedSceneId && [manifest.base, ...manifest.rooms].some((a) => a.key === requestedSceneId)
      ? requestedSceneId
      : null;

  const yaw = parseAngle(first(query.yaw), Math.PI);
  const pitch = parseAngle(first(query.pitch), Math.PI / 2);
  const fovRaw = Number(first(query.fov));
  const fov = Number.isFinite(fovRaw) && fovRaw > 0 ? clamp(fovRaw, 30, 100) : null;
  const autoRotate = first(query.autorotate) === "1" || first(query.autorotate) === "true";

  const hasViewOverride = yaw !== null || pitch !== null || fov !== null;
  const scenes: Scene[] = hasViewOverride && target
    ? playable.map((scene) =>
        scene.id === target.id
          ? {
              ...scene,
              initial_view: {
                yaw: yaw ?? scene.initial_view.yaw,
                pitch: pitch ?? scene.initial_view.pitch,
                fov: fov ?? scene.initial_view.fov,
              } satisfies InitialView,
            }
          : scene,
      )
    : playable;

  return (
    <main className="h-full w-full">
      <EmbedSwitcher
      token={token}
        title={property.title}
        locale={DEFAULT_LOCALE}
        initialMode={initialMode}
        manifest={manifest}
        startKey={walkStart}
        scenes={scenes}
        initialSceneId={target?.id ?? null}
        plan={plan}
        autoRotate={autoRotate}
      />
    </main>
  );
}
