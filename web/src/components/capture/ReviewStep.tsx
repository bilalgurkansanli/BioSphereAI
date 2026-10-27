"use client";

/**
 * The honest verdict, delivered BEFORE the upload.
 *
 * The engine already knows how to refuse a capture — `validate.py` returns walking /
 * too_dark / too_blurry / too_short with bilingual copy. The problem is that it learns this
 * after a ~125 MB upload and a queue wait, by which point the agent has left the property.
 * This screen says the same thing while they are still standing in the room, using the same
 * vocabulary (`CaptureVerdict` mirrors the validator's verdicts exactly), so the phone and
 * the server never tell two different stories.
 *
 * Consequence for the button order: when coverage is short, "Eksikleri tamamla" is the
 * primary action and uploading is demoted to a plain link. We are not trying to stop anyone
 * — `capture.review.anyway` is always there — but the default has to be the thing that
 * actually works.
 */

import {
  ArrowRight,
  CircleCheck,
  Download,
  Info,
  RotateCcw,
  Sparkles,
  TriangleAlert,
  UploadCloud,
} from "lucide-react";
import { useEffect, useMemo, useSyncExternalStore } from "react";
import { CoverageDome } from "./CoverageDome";
import { ui } from "@/components/dashboard/copy";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { cc, rowTitleKey } from "@/lib/capture/copy";
import { orbiting } from "@/lib/capture/plan";
import {
  COMPLETENESS,
  COVERAGE_GRID,
  ROW_IDS,
  ROW_PLAN,
  type CaptureCopyKey,
  type CaptureVerdict,
  type CoverageGap,
  type GuidedQualityMode,
  type ReviewStepProps,
  type RowId,
} from "@/lib/capture/types";
import { qualityDescription, qualityLabel } from "@/lib/i18n";
import { cn, formatBytes } from "@/lib/utils";

/**
 * Verdict → copy and tone.
 *
 * `too_blurry` and `too_short` have no copy keys of their own in the contract, and inventing
 * two is worse than mapping them onto the closest true statement: blur in a room capture is
 * almost always the low-light story that `review.dark` already tells, and a capture that
 * stopped early is, by definition, a capture with gaps.
 */
const VERDICTS: Readonly<
  Record<CaptureVerdict, { title: CaptureCopyKey; body: CaptureCopyKey; tone: "good" | "warn" | "bad" }>
> = {
  great: { title: "capture.review.great.title", body: "capture.review.great.body", tone: "good" },
  gaps: { title: "capture.review.gaps.title", body: "capture.review.gaps.body", tone: "warn" },
  too_short: { title: "capture.review.gaps.title", body: "capture.review.gaps.body", tone: "warn" },
  walking: { title: "capture.review.walk.title", body: "capture.review.walk.body", tone: "bad" },
  too_dark: { title: "capture.review.dark.title", body: "capture.review.dark.body", tone: "bad" },
  too_blurry: { title: "capture.review.dark.title", body: "capture.review.dark.body", tone: "bad" },
};

/**
 * Two verdicts the contract's copy keys cannot express, written here rather than added to
 * `CaptureCopyKey` — that union lives in `types.ts`, which is the shared contract, and a
 * verdict whose cause is a phone-side measurement is a phone-side string.
 *
 * Both are the same shape as the dictionary entries: a diagnosis for the title and, in the
 * body, the CORRECTION rather than the fault.
 */
const PIVOT_COPY = {
  tr: {
    title: "Telefon kendi ekseninde dönmemiş",
    body:
      "Telefon, gövdenizin çevresinde bir yay çizerek dönmüş. Bu, düz çizgileri — tavan kirişlerini, kapı kenarlarını — panoramada büken tek şeydir. " +
      "Dirseklerinizi gövdenize yaslayıp telefonu göğsünüzün önünde, size yakın tutarak tekrar çekin: ayaklarınız sabit, dönen sadece siz ve telefon.",
  },
  en: {
    title: "The phone did not turn about itself",
    body:
      "The phone swung in an arc around your body instead of rotating on the spot. That is the one thing that bends straight lines — ceiling beams, door frames — in the finished panorama. " +
      "Shoot it again with your elbows against your body and the phone close in front of your chest: feet planted, only you and the phone turning.",
  },
} as const;

const BLURRY_COPY = {
  tr: {
    title: "Kareler hareket bulanıklığı taşıyor",
    body:
      "Telefon bu ışıkta uzun bir pozlama seçti ve dönüş hızıyla birleşince karelerin yaklaşık yarısı bulanık çıktı. Bulanık kareler panoramada yumuşak duvarlar ve çift kenarlar olarak görünür. " +
      "Daha fazla ışıkla (perdeleri açın, lambaları yakın) ve her halkada bir an durarak, yarı hızda tekrar çekin.",
  },
  en: {
    title: "The frames carry motion blur",
    body:
      "In this light the phone chose a long exposure, and together with the turning speed about half of the frames came out smeared. Smeared frames become soft walls and doubled edges in the panorama. " +
      "Shoot it again with more light (open the curtains, switch the lamps on) and at half the speed, pausing for a beat at each ring.",
  },
} as const;

const UNVERIFIED_COPY = {
  tr: {
    title: "Kapsama ölçülemedi",
    body:
      "Yön sensörü kapalı olduğu için hangi yönleri çektiğinizi ölçemedik — bu, çekimin kötü olduğu anlamına gelmez, sadece kontrol edemediğimiz anlamına gelir. " +
      "Yükleyin: eksik bir yer varsa motor size söyleyecek.",
  },
  en: {
    title: "Coverage could not be measured",
    body:
      "With the orientation sensor off we could not measure which directions you covered — that does not mean the capture is bad, only that we could not check it. " +
      "Upload it: if something is missing, the engine will say so.",
  },
} as const;

/**
 * `coverageMeasured` is an intersection rather than an addition to `ReviewStepProps`: the
 * prop contract lives in `types.ts`, which is shared, and this is a fact only the stage
 * knows about its own session.
 */
export function ReviewStep({
  result,
  coverage,
  thumbnails,
  sceneName,
  onSceneNameChange,
  qualityMode,
  onQualityModeChange,
  deviceNote,
  onFillGaps,
  onAccept,
  onRetake,
  onWantPoles,
  onDownload,
  locale,
  coverageMeasured = true,
  canFillGaps = false,
}: ReviewStepProps & { coverageMeasured?: boolean; canFillGaps?: boolean }) {
  const verdict = VERDICTS[result.verdict];
  // An orbit passes every coverage gate there is, so the verdict it earns is `gaps` — but
  // "%N of the sphere is covered" is the wrong sentence for it and "fill the gaps" is the
  // wrong instruction. The predicate is `plan.ts`'s, the same one the verdict itself used,
  // so the grade and the explanation cannot drift apart.
  //
  // It only ever replaces the `gaps` copy: `walking`, `too_dark` and `too_blurry` are
  // stricter diagnoses that outrank it in the same ladder the engine uses, and each already
  // names its own fix.
  const showPivot = result.verdict === "gaps" && orbiting(result.telemetry.translation);
  const showUnverified = result.verdict === "gaps" && !showPivot && !coverageMeasured;
  const coveredPercent = Math.round(coverage.coveredFraction * 100);
  // The machine's own guard, passed in rather than re-derived here. `FILL_GAPS` is rejected
  // outright when the stage's `hasGaps` is false, so a review that decided for itself would
  // be offering a primary button whose only effect is nothing at all — and the state that
  // makes them disagree is a real one: every station filled, coverage still short.
  const hasGaps = coverageMeasured && canFillGaps;

  /**
   * The specific criteria that are short, in the user's own vocabulary.
   *
   * `blockedBy` is the same list `plan.verdict` consulted, so this can never disagree with
   * the headline above it. Rendering it is what turns "eksik bölgeler var" from an
   * accusation into an instruction.
   */
  const blockerReasons: string[] = coverageMeasured
    ? coverage.blockedBy.map((blocker) => {
        const tr = locale === "tr";
        switch (blocker) {
          case "covered_fraction":
            return tr
              ? `Kürenin %${coveredPercent}'i kapsandı, hedef %${Math.round(COMPLETENESS.coveredFraction * 100)}`
              : `${coveredPercent}% of the sphere covered, target ${Math.round(COMPLETENESS.coveredFraction * 100)}%`;
          case "horizon_30":
          case "horizon_60":
            return tr ? "Göz hizasında kapanmayan bir yön var" : "A direction at eye level is still open";
          case "pitch_max":
            return tr ? "Tavana yeterince yukarı bakılmadı" : "The ceiling was not reached";
          case "pitch_min":
            return tr ? "Zemine yeterince aşağı bakılmadı" : "The floor was not reached";
          case "yaw_span_level":
          case "row_level":
            return tr ? "Göz hizası turu tamamlanmadı" : "The eye-level turn is incomplete";
          case "yaw_span_up":
          case "row_up":
            return tr ? "Yukarı bakan tur tamamlanmadı" : "The upward turn is incomplete";
          case "yaw_span_down":
          case "row_down":
            return tr ? "Aşağı bakan tur tamamlanmadı" : "The downward turn is incomplete";
          default:
            return tr ? "Bir ölçüt eksik kaldı" : "One criterion is short";
        }
      })
    : [];
  // Filling gaps cannot fix a swung capture — going round again with the same arm swings
  // the same arc — so the honest primary action there is to shoot it again.
  // Blur is the same story: filling gaps at the same speed in the same light produces the
  // same smear, so the honest primary action is a slower, brighter re-shoot.
  const showBlurry = result.verdict === "too_blurry";
  const reshootFirst = showPivot || result.verdict === "walking" || showBlurry;

  const bytes = useMemo(() => result.segments.reduce((total, s) => total + s.blob.size, 0), [result.segments]);
  const thumbUrls = useObjectUrls(thumbnails);
  const covered = useCoveredMask(result.telemetry.coverage.coveredMaskBase64);
  // Hoisted, not called inside the JSX: the dome is rendered conditionally, and a hook
  // inside that branch would change the hook order the moment the mask fails to decode.
  const reducedMotion = usePrefersReducedMotion();

  return (
    <div className="min-h-[100dvh] w-full overflow-y-auto bg-canvas pb-[max(env(safe-area-inset-bottom),1rem)]">
      <div className="mx-auto w-full max-w-[520px] px-4 py-6 sm:px-6">
        {/* ── Verdict ─────────────────────────────────────────────────────────────── */}
        <section
          className={cn(
            "animate-rise flex gap-3.5 rounded-lg border p-4",
            verdict.tone === "good"
              ? "border-success/25 bg-success/5"
              : verdict.tone === "warn"
                ? "border-amber/25 bg-amber/5"
                : "border-danger/25 bg-danger/5",
          )}
        >
          <span
            className={cn(
              "mt-0.5 shrink-0",
              verdict.tone === "good" ? "text-success" : verdict.tone === "warn" ? "text-amber" : "text-danger",
            )}
          >
            {verdict.tone === "good" ? (
              <CircleCheck className="size-5" strokeWidth={1.8} aria-hidden />
            ) : (
              <TriangleAlert className="size-5" strokeWidth={1.8} aria-hidden />
            )}
          </span>
          <div className="min-w-0">
            <h1 className="text-[17px] leading-tight font-semibold tracking-tight text-ink">
              {showPivot
                ? PIVOT_COPY[locale].title
                : showBlurry
                  ? BLURRY_COPY[locale].title
                  : showUnverified
                    ? UNVERIFIED_COPY[locale].title
                    : cc(verdict.title, locale)}
            </h1>
            <p className="mt-1.5 text-[13px] leading-relaxed text-ink-muted">
              {showPivot
                ? PIVOT_COPY[locale].body
                : showBlurry
                  ? BLURRY_COPY[locale].body
                  : showUnverified
                    ? UNVERIFIED_COPY[locale].body
                    : cc(verdict.body, locale, { n: coveredPercent })}
            </p>
            {/* Name the criterion, never just "incomplete".
                A user who turned through all three rows, filled every station and covered
                91% of the sphere was told "there are missing areas" with nothing to act on —
                which reads as the product moving the goalposts. If something is genuinely
                short, say which thing, in the same words the coaching used. */}
            {blockerReasons.length > 0 ? (
              <ul className="mt-2.5 space-y-1">
                {blockerReasons.map((reason) => (
                  <li key={reason} className="flex gap-1.5 text-[12.5px] leading-relaxed text-ink-faint">
                    <span aria-hidden>·</span>
                    <span>{reason}</span>
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        </section>

        {/* ── Coverage ────────────────────────────────────────────────────────────── */}
        {!coverageMeasured ? (
          // No sphere was painted, so there is no percentage, no dome and no per-row count
          // that means anything. A "0%" and three empty chips would read as a failed capture
          // rather than an unmeasured one — the difference this whole screen exists to keep.
          <section className="glass mt-4 rounded-lg p-5">
            <p className="text-[12px] font-medium text-ink-faint">{ui("coverage.covered", locale)}</p>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-muted">
              {locale === "tr"
                ? "Bu çekimde ölçülmedi. Yön sensörü olmadan küre haritası çıkarılamıyor; video ve süre bilgisi eksiksiz yüklenir."
                : "Not measured for this capture. Without an orientation sensor there is no sphere map; the video and its timing upload in full."}
            </p>
          </section>
        ) : (
        <section className="glass mt-4 rounded-lg p-5">
          <div className="flex items-center gap-5">
            {covered ? (
              <CoverageDome
                covered={covered}
                yawCells={COVERAGE_GRID.yawCells}
                pitchCells={COVERAGE_GRID.pitchCells}
                coveredFraction={coverage.coveredFraction}
                nextRow={null}
                size={132}
                variant="review"
                reducedMotion={reducedMotion}
                label={ui("coverage.map", locale)}
                className="shrink-0"
              />
            ) : null}

            <div className="min-w-0 flex-1">
              <p className="text-[12px] font-medium text-ink-faint">{ui("coverage.covered", locale)}</p>
              <p className="mt-0.5 font-mono text-[28px] leading-none font-semibold text-ink tabular-nums">
                {coveredPercent}
                <span className="text-[17px] text-ink-faint">%</span>
              </p>
              <CoverageBar fraction={coverage.coveredFraction} locale={locale} />
              {/* A full three-row capture measures 89-91%, because three rows photograph a
                  band and the two polar caps are synthesised. Saying so is the difference
                  between "you covered 91%" reading as a pass and reading as a 9-point
                  failure the user tries, and fails, to close by re-shooting. */}
              <p className="mt-1.5 text-[11px] leading-snug text-ink-faint">
                {ui("coverage.ceilingNote", locale).replace(
                  "{target}",
                  String(Math.round(COMPLETENESS.coveredFraction * 100)),
                )}
              </p>
            </div>
          </div>

          <dl className="mt-5 grid grid-cols-3 gap-3 border-t border-line pt-4">
            {ROW_IDS.map((row) => (
              <RowChip key={row} row={row} coverage={coverage} locale={locale} />
            ))}
          </dl>

          <dl className="mt-3 grid grid-cols-2 gap-3">
            <Stat
              label={ui("coverage.pitchRange", locale)}
              value={`${Math.round(coverage.pitchMinDeg)}° … ${Math.round(coverage.pitchMaxDeg)}°`}
              // The pole filler may only repair above 60°, so the frame-edge reach is the
              // number that decides whether a ring is repairable or simply black.
              ok={coverage.pitchMaxDeg >= COMPLETENESS.pitchMaxDeg && coverage.pitchMinDeg <= COMPLETENESS.pitchMinDeg}
            />
            <Stat
              label={ui("coverage.horizon", locale)}
              value={`${Math.round(coverage.horizonFraction60 * 100)}%`}
              ok={coverage.horizonFraction60 >= COMPLETENESS.horizonFraction60}
            />
          </dl>

          {coverage.gaps.length > 0 ? (
            <div className="mt-4 border-t border-line pt-4">
              <p className="text-[12px] font-medium text-ink-faint">{ui("coverage.gapsTitle", locale)}</p>
              <ul className="mt-2 space-y-1.5">
                {coverage.gaps.slice(0, 4).map((gap, index) => (
                  <li key={index} className="flex items-baseline gap-2 text-[12.5px] text-ink-muted">
                    <span className="size-1.5 shrink-0 rounded-full bg-amber" aria-hidden />
                    <span>{describeGap(gap, locale)}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </section>
        )}

        {/* ── What was captured ───────────────────────────────────────────────────── */}
        {thumbUrls.length > 0 ? (
          <section className="mt-4">
            <p className="mb-2 text-[12px] font-medium text-ink-faint">{cc("capture.review.thumbs", locale)}</p>
            <ul className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-1 sm:mx-0 sm:px-0">
              {thumbUrls.map((url, index) => (
                <li key={url} className="shrink-0">
                  {/* Plain <img>: these are blob: URLs for frames that exist only in this
                      tab, so next/image has nothing to optimise and would only add a proxy
                      hop that cannot resolve a blob. */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={url}
                    alt={`${cc("capture.review.thumbs", locale)} ${index + 1}`}
                    className="h-24 w-[54px] rounded-sm border border-line object-cover"
                  />
                </li>
              ))}
            </ul>
          </section>
        ) : null}

        {/* ── Scene details ───────────────────────────────────────────────────────── */}
        <section className="glass mt-4 space-y-4 rounded-lg p-5">
          <Input
            label={locale === "tr" ? "Oda adı" : "Room name"}
            placeholder={locale === "tr" ? "Örn. Salon" : "e.g. Living room"}
            value={sceneName}
            onChange={(event) => onSceneNameChange(event.target.value)}
          />

          <div>
            <p className="mb-2 text-[13px] font-medium text-ink-muted">
              {locale === "tr" ? "İşleme kalitesi" : "Processing quality"}
            </p>
            <QualityChoice value={qualityMode} onChange={onQualityModeChange} locale={locale} />
            {/* Under the picker rather than beside it: the note explains why the preset
                landed where it did, and on a phone there is no room for it anywhere else. */}
            {deviceNote ? (
              <p className="mt-2 flex items-start gap-1.5 text-[12px] leading-relaxed text-ink-faint">
                <Info size={13} className="mt-0.5 shrink-0" aria-hidden />
                <span>{deviceNote}</span>
              </p>
            ) : null}
          </div>

          <p className="text-[12px] text-ink-faint">
            {formatBytes(bytes)}
            {result.segments.length > 1
              ? ` · ${result.segments.length} ${locale === "tr" ? "parça" : "segments"}`
              : ""}
          </p>
        </section>

        {/* ── Actions ─────────────────────────────────────────────────────────────── */}
        <section className="mt-5 space-y-2.5">
          {reshootFirst ? (
            <>
              <Button variant="primary" size="lg" className="w-full" onClick={onRetake}>
                <RotateCcw className="size-4" aria-hidden />
                {cc("capture.review.retake", locale)}
              </Button>
              <Button variant="secondary" size="lg" className="w-full" onClick={onAccept}>
                <UploadCloud className="size-4" aria-hidden />
                {cc("capture.review.anyway", locale)}
              </Button>
            </>
          ) : hasGaps ? (
            <>
              <Button variant="primary" size="lg" className="w-full" onClick={onFillGaps}>
                {cc("capture.review.fill", locale)}
                <ArrowRight className="size-4" aria-hidden />
              </Button>
              <Button variant="secondary" size="lg" className="w-full" onClick={onAccept}>
                <UploadCloud className="size-4" aria-hidden />
                {cc("capture.review.anyway", locale)}
              </Button>
            </>
          ) : (
            <Button variant="primary" size="lg" className="w-full" onClick={onAccept}>
              <UploadCloud className="size-4" aria-hidden />
              {cc("capture.review.use", locale)}
            </Button>
          )}

          <div className="flex gap-2.5">
            {/* The pole pass is aimed with roll and pitch reticles. Without an orientation
                sensor there is nothing to aim at and the pass could never end, so it is not
                offered rather than offered and broken. */}
            {coverageMeasured ? (
              <Button variant="ghost" className="flex-1" onClick={onWantPoles}>
                <Sparkles className="size-4" aria-hidden />
                {cc("capture.bonus.poles", locale)}
              </Button>
            ) : null}
            {reshootFirst ? null : (
              <Button variant="ghost" className="flex-1" onClick={onRetake}>
                <RotateCcw className="size-4" aria-hidden />
                {cc("capture.review.retake", locale)}
              </Button>
            )}
          </div>

          {/* The recording belongs to the person who shot it. Even if every upload fails,
              their minute of work leaves this screen with them. */}
          <button
            type="button"
            onClick={onDownload}
            className="flex w-full items-center justify-center gap-2 py-2 text-[13px] text-ink-faint transition-colors hover:text-ink-muted"
          >
            <Download className="size-3.5" aria-hidden />
            {cc("capture.save.local", locale)}
          </button>
        </section>
      </div>
    </div>
  );
}

// ── Pieces ─────────────────────────────────────────────────────────────────────────────

function CoverageBar({ fraction, locale }: { fraction: number; locale: ReviewStepProps["locale"] }) {
  // The scale tops out at the PHYSICAL ceiling of the three-row plan, sin(63.5°) = 0.8949,
  // not at 1.0. Drawing a perfect capture as a two-thirds-full bar would be a lie told by
  // the axis: nothing a user can do with this plan reaches 100%.
  const ceiling = COMPLETENESS.coveredFractionCeiling;
  const pct = Math.min(100, (fraction / ceiling) * 100);
  const markerPct = Math.min(100, (COMPLETENESS.coveredFraction / ceiling) * 100);
  const good = fraction >= COMPLETENESS.coveredFraction;

  return (
    <div className="mt-2.5">
      <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
        <div
          className={cn("h-full rounded-full transition-[width] duration-700 ease-out-expo", good ? "bg-success" : "bg-amber")}
          style={{ width: `${pct}%` }}
          aria-hidden
        />
        <span className="absolute inset-y-0 w-px bg-ink/50" style={{ left: `${markerPct}%` }} aria-hidden />
      </div>
      <p className="mt-1.5 text-[11px] text-ink-faint">
        {cc("capture.review.enough", locale)} · {Math.round(COMPLETENESS.coveredFraction * 100)}%
      </p>
    </div>
  );
}

function RowChip({
  row,
  coverage,
  locale,
}: {
  row: RowId;
  coverage: ReviewStepProps["coverage"];
  locale: ReviewStepProps["locale"];
}) {
  const progress = coverage.rows[row];
  const required = progress?.requiredStations || ROW_PLAN[row].requiredStations;
  const filled = progress?.filledStations ?? 0;
  const done = progress?.complete ?? false;

  return (
    <div className="min-w-0">
      <dt className="truncate text-[11px] text-ink-faint">{cc(rowTitleKey(row), locale)}</dt>
      <dd className={cn("mt-0.5 font-mono text-[13px] tabular-nums", done ? "text-success" : "text-ink-muted")}>
        {filled}/{required}
      </dd>
    </div>
  );
}

function Stat({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className="min-w-0">
      <dt className="truncate text-[11px] text-ink-faint">{label}</dt>
      <dd className={cn("mt-0.5 font-mono text-[13px] tabular-nums", ok ? "text-ink-muted" : "text-amber")}>{value}</dd>
    </div>
  );
}

function QualityChoice({
  value,
  onChange,
  locale,
}: {
  value: GuidedQualityMode;
  onChange: (mode: GuidedQualityMode) => void;
  locale: ReviewStepProps["locale"];
}) {
  // Two options, not three. `fast` is excluded by contract decision D5: its 15.05° keyframe
  // bin is narrower than the pace this flow coaches, so a guided capture processed as
  // `fast` would fall back to a blurred keyframe in 4.3% of bins.
  const modes: GuidedQualityMode[] = ["balanced", "studio"];
  return (
    <div role="radiogroup" aria-label={locale === "tr" ? "İşleme kalitesi" : "Processing quality"} className="grid grid-cols-2 gap-2.5">
      {modes.map((mode) => {
        const selected = mode === value;
        return (
          <button
            key={mode}
            type="button"
            role="radio"
            aria-checked={selected}
            onClick={() => onChange(mode)}
            className={cn(
              "rounded-md border p-3 text-left transition-all duration-200 ease-out-expo",
              selected
                ? "border-bio/45 bg-bio/5 shadow-[0_0_0_1px_var(--bio-dim)]"
                : "border-line bg-surface-2/40 hover:border-line-strong",
            )}
          >
            <span className="block text-[13px] font-semibold tracking-tight text-ink">{qualityLabel(mode, locale)}</span>
            <span className="mt-0.5 block font-mono text-[11px] text-ink-faint">{qualityDescription(mode, locale)}</span>
          </button>
        );
      })}
    </div>
  );
}

// ── Helpers ────────────────────────────────────────────────────────────────────────────

/** Human direction for a gap, in the dashboard's existing compass vocabulary. */
function describeGap(gap: CoverageGap, locale: ReviewStepProps["locale"]): string {
  const midYaw = ((gap.yawStartDeg + gap.yawEndDeg) / 2 + 360) % 360;
  const midPitch = (gap.pitchStartDeg + gap.pitchEndDeg) / 2;

  const compass =
    midYaw < 45 || midYaw >= 315
      ? ui("dir.front", locale)
      : midYaw < 135
        ? ui("dir.right", locale)
        : midYaw < 225
          ? ui("dir.back", locale)
          : ui("dir.left", locale);

  const band =
    midPitch > 20
      ? cc("capture.band.ceiling", locale)
      : midPitch < -20
        ? cc("capture.band.floor", locale)
        : ui("dir.level", locale);

  const share = Math.max(1, Math.round(gap.solidAngleFraction * 100));
  return `${compass} · ${band} — %${share}`;
}

/**
 * Blob URLs for the review strip.
 *
 * Created in a memo and revoked by an effect keyed on that memo, rather than created in an
 * effect and pushed into state: the state-in-effect version renders the strip empty for one
 * frame every time, which reads as a flash of missing thumbnails on the screen where the
 * user is deciding whether their capture is any good.
 */
function useObjectUrls(blobs: Blob[]): string[] {
  const urls = useMemo(
    () => (typeof URL === "undefined" ? [] : blobs.map((blob) => URL.createObjectURL(blob))),
    [blobs],
  );

  useEffect(() => () => urls.forEach((url) => URL.revokeObjectURL(url)), [urls]);

  return urls;
}

/** Decodes the 72x36 covered mask the telemetry carries for the server's own heat-map. */
function useCoveredMask(base64: string): Uint8Array | null {
  return useMemo(() => {
    if (!base64 || typeof atob !== "function") return null;
    try {
      const binary = atob(base64);
      if (binary.length !== COVERAGE_GRID.totalCells) return null;
      const out = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
      return out;
    } catch {
      return null;
    }
  }, [base64]);
}

/**
 * `prefers-reduced-motion` as an external store. Duplicated from `CaptureStage` rather than
 * shared, because `CaptureStage` imports this file — extracting it would only trade twelve
 * lines for an import cycle or a module that exists to hold one hook.
 */
const REDUCED_MOTION = "(prefers-reduced-motion: reduce)";

function subscribeReducedMotion(onChange: () => void): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return () => undefined;
  const query = window.matchMedia(REDUCED_MOTION);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

function readReducedMotion(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  return window.matchMedia(REDUCED_MOTION).matches;
}

function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(subscribeReducedMotion, readReducedMotion, () => false);
}
