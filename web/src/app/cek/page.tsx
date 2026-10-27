"use client";

/**
 * /cek — the guided capture route.
 *
 * This page is opened by someone standing in a flat with a phone in their hand, usually
 * from a QR code on a colleague's laptop or a link they sent themselves. It is deliberately
 * OUTSIDE the `(dashboard)` route group, so it gets no top bar, no breadcrumbs and no
 * sidebar: the camera needs the whole viewport, and every pixel of chrome is a pixel the
 * agent is not using to see the room.
 *
 * Three entry conditions, all of which have to work on a cold load with no app state:
 *   1. `/cek?property=<id>`  — attach to an existing property. The common case: the property was made
 *      at a desk, the rooms are shot on site.
 *   2. `/cek`            — no property yet. Ask for a title, create one, then rewrite the URL
 *      so a reload or a backgrounded tab does not create a second property.
 *   3. desktop           — capture makes no sense here, so hand the URL to a phone as a QR
 *      rather than failing at `getUserMedia`.
 *
 * The page owns nothing about capture itself; `CaptureStage` does. What lands here is the
 * property lifecycle either side of it, and the handoff to the dashboard's existing
 * `JobProgress` once a scene has been created.
 *
 * IT IS A LOOP, NOT A SINGLE CAPTURE. An agent visiting a flat shoots every room in one
 * trip, and the panorama for room one takes minutes that they are not going to spend
 * standing in the hallway — nor should they, because the job is queued on the SERVER and
 * survives the page, the phone locking, and the browser being closed. So the screen that
 * follows an upload leads with the next room rather than with a progress bar, and the bar
 * stays underneath for anyone who does want to watch it.
 */

import { ArrowRight, Camera, TriangleAlert } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { CaptureStage } from "@/components/capture/CaptureStage";
import { PhoneHandoff } from "@/components/capture/PhoneHandoff";
import { WalkCaptureFlow } from "@/components/capture/WalkCaptureFlow";
import { JobProgress } from "@/components/dashboard/JobProgress";
import { ui } from "@/components/dashboard/copy";
import { Button, buttonStyles } from "@/components/ui/Button";
import Link from "next/link";
import { Input } from "@/components/ui/Input";
import { BioSphereError, api } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import type { Job, Scene } from "@/lib/types";
import { cn } from "@/lib/utils";

export default function CapturePage() {
  // `useSearchParams` forces client-side rendering up to the nearest Suspense boundary, so
  // the boundary is explicit here rather than left to the framework's default.
  return (
    <Suspense fallback={<Splash />}>
      <CaptureRoute />
    </Suspense>
  );
}

type Handoff = { scene: Scene; job: Job };

function CaptureRoute() {
  const { locale } = useLocale();
  const router = useRouter();
  const searchParams = useSearchParams();

  const [propertyId, setPropertyId] = useState<string | null>(() => searchParams.get("property"));
  // `?m=w` is the 3D walkthrough: the phone is walked through the flat instead of turned
  // on the spot, and a different stage owns the camera. It is a URL parameter, not app
  // state, for the same reason the property id is: the link is what reaches the phone.
  const walkMode = searchParams.get("m") === "w";
  const [propertyTitle, setPropertyTitle] = useState<string | null>(null);
  const [checking, setChecking] = useState<boolean>(() => searchParams.get("property") !== null);
  const [linkBroken, setLinkBroken] = useState(false);
  const [handoff, setHandoff] = useState<Handoff | null>(null);
  const [handoffError, setHandoffError] = useState<string | null>(null);
  const [roomsCaptured, setRoomsCaptured] = useState(0);
  // Remounting the stage is the point, not an accident: `CaptureStage` holds a camera
  // stream, a recorder, coverage rings and a state machine, and room two has to start from
  // the same blank slate room one did. Changing the key throws all of it away.
  const [roomKey, setRoomKey] = useState(0);

  const isPhone = useIsPhone();
  const [overrideDevice, setOverrideDevice] = useState(false);

  // A property id that came in on the URL is verified once, up front. A dead link discovered
  // here costs two seconds; discovered at upload time it costs the whole capture.
  useEffect(() => {
    const incoming = searchParams.get("property");
    // `checking` is initialised from the same parameter, so the no-property case needs no state
    // change at all — and the effect stays free of a synchronous setState.
    if (!incoming) return;
    let alive = true;
    api.properties
      .get(incoming)
      .then((property) => {
        if (!alive) return;
        setPropertyId(property.id);
        setPropertyTitle(property.title);
        setLinkBroken(false);
      })
      .catch(() => {
        if (!alive) return;
        // Includes an unreachable engine, not only a 404 — either way the honest move is to
        // offer a fresh property rather than pretend the capture will land somewhere.
        setLinkBroken(true);
        setPropertyId(null);
      })
      .finally(() => {
        if (alive) setChecking(false);
      });
    return () => {
      alive = false;
    };
  }, [searchParams]);

  const onPropertyCreated = useCallback((id: string, title: string) => {
    setPropertyId(id);
    setPropertyTitle(title);
    setLinkBroken(false);
    // Rewrite the URL rather than pushing: a reload, an incoming call or an accidental
    // back-swipe must reattach to THIS property instead of creating another empty one.
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      url.searchParams.set("property", id);
      window.history.replaceState(null, "", url.toString());
    }
  }, []);

  const useExistingVideo = useCallback(() => {
    // The existing upload paths, untouched: a property detail page has AddSceneDialog, and the
    // wizard covers the case where no property exists yet.
    router.push(propertyId ? `/evler/${propertyId}` : "/new");
  }, [router, propertyId]);

  const onSceneCreated = useCallback(
    ({ sceneId, jobId }: { sceneId: string; jobId: string }) => {
      // `CaptureStage` hands back ids; `JobProgress` wants the resources. Fetching them here
      // keeps the stage free of any dashboard dependency.
      Promise.all([api.scenes.get(sceneId), api.jobs.get(jobId)])
        .then(([scene, job]) => {
          setRoomsCaptured((n) => n + 1);
          setHandoff({ scene, job });
        })
        .catch((cause: unknown) => {
          // The upload succeeded — the scene exists and is being processed. Failing to
          // *display* it is a navigation problem, so say so and offer the property page.
          setHandoffError(
            cause instanceof BioSphereError
              ? cause.localised(locale)
              : locale === "tr"
                ? "Sahne oluşturuldu ama durumu gösterilemedi."
                : "The scene was created but its progress could not be shown.",
          );
        });
    },
    [locale],
  );

  const nextRoom = useCallback(() => {
    setHandoff(null);
    setHandoffError(null);
    setRoomKey((k) => k + 1);
  }, []);

  if (handoff && propertyId) {
    return (
      <main className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6">
        <NextRoomCard locale={locale} propertyId={propertyId} rooms={roomsCaptured} onNextRoom={nextRoom} />
        {/* `autoNavigate` is a desktop behaviour. On a phone mid-visit it would yank the
            agent out of the capture flow and into the editor the moment room one finished,
            which is exactly when they are pointing the camera at room two. */}
        <div className="mt-4">
          <JobProgress
            job={handoff.job}
            propertyId={propertyId}
            sceneId={handoff.scene.id}
            qualityMode={handoff.scene.quality_mode}
            autoNavigate={false}
          />
        </div>
      </main>
    );
  }

  if (handoffError && propertyId) {
    return (
      <Card>
        <TriangleAlert className="size-6 text-amber" aria-hidden />
        <h1 className="mt-3 text-[19px] font-semibold tracking-tight text-ink">
          {locale === "tr" ? "Sahne yüklendi" : "The scene was uploaded"}
        </h1>
        <p className="mt-2 text-[13.5px] leading-relaxed text-ink-muted">{handoffError}</p>
        <Button variant="primary" size="lg" className="mt-6 w-full" onClick={() => router.push(`/evler/${propertyId}`)}>
          {ui("editor.back", locale)}
          <ArrowRight className="size-4" aria-hidden />
        </Button>
      </Card>
    );
  }

  if (checking) return <Splash />;

  // Desktop: the capture flow needs the device that holds the camera. Rather than letting
  // it fail at getUserMedia, move the whole session to a phone with one scan.
  if (!isPhone && !overrideDevice) {
    return (
      <div className="grid-bg flex min-h-[100dvh] w-full items-center justify-center px-5 py-10">
        <div className="w-full max-w-[560px]">
          {propertyId ? (
            <p className="mb-3 text-[13px] text-ink-muted">
              {locale === "tr" ? "Ev" : "Property"}
              {": "}
              <span className="font-medium text-ink">{propertyTitle ?? propertyId}</span>
            </p>
          ) : (
            <div className="glass mb-4 rounded-lg p-5">
              <NewPropertyForm locale={locale} onCreated={onPropertyCreated} broken={linkBroken} compact />
            </div>
          )}
          <PhoneHandoff
            propertyId={propertyId}
            locale={locale}
            walk={walkMode}
            footer={
              <button
                type="button"
                onClick={() => setOverrideDevice(true)}
                className="w-full py-1 text-[13px] text-ink-faint transition-colors hover:text-ink-muted"
              >
                {locale === "tr" ? "Yine de bu cihazda devam et" : "Continue on this device anyway"}
              </button>
            }
          />
        </div>
      </div>
    );
  }

  if (!propertyId) {
    return (
      <Card>
        <NewPropertyForm locale={locale} onCreated={onPropertyCreated} broken={linkBroken} />
      </Card>
    );
  }

  if (walkMode) {
    return <WalkCaptureFlow propertyId={propertyId} locale={locale} />;
  }

  return (
    <CaptureStage
      key={roomKey}
      propertyId={propertyId}
      defaultQualityMode="balanced"
      onUseExistingVideo={useExistingVideo}
      onHandoff={onSceneCreated}
    />
  );
}

// ── Between rooms ──────────────────────────────────────────────────────────────────────

/**
 * What the agent sees the instant an upload lands: the next room, not a progress bar.
 *
 * The claim it makes — that they can walk away — is one the system has to actually honour,
 * and it does: the job runs in the engine's worker, its state lives in the database, and
 * `JobProgress` reattaches by id. Nothing about it depends on this tab surviving.
 */
function NextRoomCard({
  locale,
  propertyId,
  rooms,
  onNextRoom,
}: {
  locale: "tr" | "en";
  propertyId: string;
  rooms: number;
  onNextRoom: () => void;
}) {
  return (
    <div className="glass animate-rise rounded-lg p-5 sm:p-6">
      <div className="flex items-baseline justify-between gap-3">
        <h1 className="text-[19px] font-semibold tracking-tight text-ink">{ui("job.nextRoom", locale)}</h1>
        <p className="shrink-0 font-mono text-[12px] tabular-nums text-ink-faint">
          {ui("job.roomsCaptured", locale)}: {rooms}
        </p>
      </div>
      <p className="mt-2 text-[13.5px] leading-relaxed text-ink-muted">{ui("job.nextRoom.body", locale)}</p>
      <div className="mt-5 flex flex-col gap-2.5 sm:flex-row">
        <Button variant="primary" size="lg" className="w-full sm:flex-1" onClick={onNextRoom}>
          <Camera className="size-4" aria-hidden />
          {ui("job.nextRoom", locale)}
        </Button>
        <Link href={`/evler/${propertyId}`} className={cn(buttonStyles("secondary", "lg"), "w-full sm:w-auto")}>
          {ui("job.openProperty", locale)}
          <ArrowRight className="size-4" aria-hidden />
        </Link>
      </div>
    </div>
  );
}

// ── Property creation ──────────────────────────────────────────────────────────────────────

function NewPropertyForm({
  locale,
  onCreated,
  broken,
  compact = false,
}: {
  locale: "tr" | "en";
  onCreated: (id: string, title: string) => void;
  broken: boolean;
  compact?: boolean;
}) {
  const [title, setTitle] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const create = async () => {
    const trimmed = title.trim();
    if (!trimmed) {
      setError(ui("wizard.title.required", locale));
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const property = await api.properties.create({ title: trimmed });
      onCreated(property.id, property.title);
    } catch (cause) {
      setError(cause instanceof BioSphereError ? cause.localised(locale) : t("error.generic", locale));
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      {broken ? (
        <p className="mb-4 flex gap-2 rounded-md border border-amber/25 bg-amber/5 p-3 text-[12.5px] leading-relaxed text-amber">
          <TriangleAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          <span>
            {locale === "tr"
              ? "Bağlantıdaki tura ulaşılamadı. Yeni bir tur oluşturup çekime devam edebilirsiniz."
              : "The property in that link could not be reached. You can create a new one and carry on."}
          </span>
        </p>
      ) : null}

      {!compact ? (
        <h1 className="text-[22px] leading-tight font-semibold tracking-tight text-ink">{ui("wizard.title", locale)}</h1>
      ) : (
        <h2 className="text-sm font-semibold tracking-tight text-ink">{ui("wizard.title", locale)}</h2>
      )}
      <p className="mt-2 mb-5 text-[13.5px] leading-relaxed text-ink-muted">
        {locale === "tr"
          ? "Çektiğiniz odalar bu turun altında toplanır. Sadece bir ad yeterli."
          : "The rooms you shoot are collected under this property. A name is all it needs."}
      </p>

      <Input
        label={t("property.title", locale)}
        placeholder={ui("wizard.title.placeholder", locale)}
        value={title}
        required
        autoFocus={!compact}
        error={error ?? undefined}
        onChange={(event) => {
          setTitle(event.target.value);
          if (error) setError(null);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") void create();
        }}
      />

      <Button variant="primary" size="lg" className="mt-4 w-full" loading={saving} onClick={() => void create()}>
        {ui("common.continue", locale)}
        <ArrowRight className="size-4" aria-hidden />
      </Button>
    </>
  );
}

// ── Device detection ───────────────────────────────────────────────────────────────────

/**
 * Is this the device that will actually hold the camera?
 *
 * Deliberately capability-based rather than a user-agent sniff: a coarse pointer plus real
 * touch points plus a media-device API is what the capture needs, and an iPad in desk mode
 * or a touchscreen laptop answering "yes" is fine — the handoff card is only a suggestion
 * and always has "continue on this device anyway" under it.
 *
 * Starts `true` so a phone never sees a flash of the desktop QR card while the effect runs;
 * the desktop's one-frame flash of the capture intro is the cheaper of the two.
 */
function readIsPhone(): boolean {
  if (typeof window === "undefined" || typeof navigator === "undefined") return true;
  const coarse = typeof window.matchMedia === "function" ? window.matchMedia("(pointer: coarse)").matches : false;
  return coarse && navigator.maxTouchPoints > 0 && Boolean(navigator.mediaDevices);
}

function subscribeIsPhone(onChange: () => void): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return () => undefined;
  // A tablet in a keyboard case changes pointer type without a reload.
  const query = window.matchMedia("(pointer: coarse)");
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

function useIsPhone(): boolean {
  return useSyncExternalStore(subscribeIsPhone, readIsPhone, () => true);
}

// ── Shell ──────────────────────────────────────────────────────────────────────────────

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid-bg flex min-h-[100dvh] w-full items-center justify-center px-5 py-10">
      <div className="animate-rise glass w-full max-w-[420px] rounded-lg p-6 sm:p-7">{children}</div>
    </div>
  );
}

function Splash() {
  return (
    <div className="grid min-h-[100dvh] w-full place-items-center bg-canvas">
      <span className="size-8 animate-spin rounded-full border-2 border-line border-t-bio" aria-hidden />
      <span className="sr-only">{ui("common.loading")}</span>
    </div>
  );
}
