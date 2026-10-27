/**
 * Naming the phone, so the engine can look its lens up.
 *
 * The engine models the capture as a rotation about the lens, and self-calibrates the
 * focal length from the imagery. That measurement is biased LONG on a hand-held sweep and
 * only long — the arm's tangential motion adds flow in the same direction as the rotation,
 * and a rotation-only model absorbs it by lengthening the focal. Measured on two captures
 * of one bedroom from one phone: 45.2 and 51.7 degrees across, from a lens that is 53.
 * A focal that is too long bows straight lines outward and shrinks the room.
 *
 * The lens itself is a constant of the hardware. If the engine knows WHICH phone this was
 * it can look the number up and bound the search — see `biosphere/pano/devices.py`, which
 * holds the table and does the looking-up. This module's whole job is to write down enough
 * for that lookup to succeed, and to be honest when it cannot.
 *
 * What is actually available in a browser
 * ---------------------------------------
 * * `navigator.userAgentData.getHighEntropyValues(["model"])` — Chromium on Android, and
 *   nothing else. Returns the model as the vendor set it: "SM-S911B", "Pixel 8", "V2206".
 *   The user-agent string is useless: Chrome froze it at "Android 10; K" in 2022, which is
 *   exactly what a real capture from a vivo Y22s reports.
 * * On iOS there is no model, from any API. Safari's WebGL renderer has been "Apple GPU"
 *   for every device since iOS 12. What remains is the screen: CSS viewport size in points
 *   plus device pixel ratio put an iPhone in a small family, and the models inside a family
 *   share a main camera. `IOS_FINGERPRINTS` in the engine's table does that matching.
 * * Nothing here is a fingerprint in the tracking sense: it is the same handful of numbers
 *   the page already needs to lay itself out, it never leaves the capture sidecar, and the
 *   sidecar is not served publicly.
 */

import { IOS_FINGERPRINTS, PHONE_LENSES, type PhoneLens } from "./deviceProfiles.generated";

export interface ScreenHints {
  /** CSS pixels — points, not device pixels. `screen`, not the window, so a split view
   *  or a browser chrome does not change the answer. */
  cssWidth: number;
  cssHeight: number;
  devicePixelRatio: number;
}

export interface DeviceModelHints {
  /** Vendor model string from UA client hints. Empty when the platform will not say. */
  uaModel: string;
  /** "vivo", "Apple", "Samsung"… Derived, never guessed from the model string's shape. */
  brand: string;
  platformVersion: string;
  screen: ScreenHints;
  hardwareConcurrency: number;
  deviceMemoryGb: number | null;
  /** Which of the above the browser actually answered, so the engine can weigh them. */
  source: "ua-client-hints" | "screen-only" | "none";
}

interface UADataLike {
  brands?: Array<{ brand: string; version: string }>;
  mobile?: boolean;
  platform?: string;
  getHighEntropyValues?: (hints: string[]) => Promise<Record<string, unknown>>;
}

/** Apple never answers a model query, so the brand has to come from the platform. */
function brandFor(uaModel: string, platform: string, userAgent: string): string {
  const model = uaModel.trim().toLowerCase();
  if (/iphone|ipad|ipod/i.test(userAgent) || platform.toLowerCase() === "ios") return "Apple";
  if (model.startsWith("sm-") || model.startsWith("gt-")) return "Samsung";
  if (model.startsWith("pixel")) return "Google";
  if (model.startsWith("v") && /^v\d{4}/.test(model)) return "vivo";
  for (const brand of ["vivo", "samsung", "xiaomi", "redmi", "poco", "oppo", "realme", "honor", "huawei", "oneplus", "motorola", "nokia", "tecno", "infinix", "google"]) {
    if (model.includes(brand)) return brand === "redmi" || brand === "poco" ? "Xiaomi" : brand.charAt(0).toUpperCase() + brand.slice(1);
  }
  return "";
}

function readScreen(): ScreenHints {
  if (typeof window === "undefined" || typeof screen === "undefined") {
    return { cssWidth: 0, cssHeight: 0, devicePixelRatio: 0 };
  }
  return {
    cssWidth: Math.round(screen.width || 0),
    cssHeight: Math.round(screen.height || 0),
    devicePixelRatio: Math.round((window.devicePixelRatio || 0) * 100) / 100,
  };
}

/**
 * Everything the browser will say about which phone this is.
 *
 * Never throws and never blocks for long: `getHighEntropyValues` is a promise that can be
 * rejected by permissions policy, and a capture must not wait on it. Callers await this
 * once, early, and keep the result.
 */
export async function readModelHints(): Promise<DeviceModelHints> {
  const nav = (typeof navigator !== "undefined" ? navigator : undefined) as
    | (Navigator & { userAgentData?: UADataLike; deviceMemory?: number })
    | undefined;
  const userAgent = nav?.userAgent ?? "";
  const screenHints = readScreen();
  const base: DeviceModelHints = {
    uaModel: "",
    brand: brandFor("", nav?.userAgentData?.platform ?? "", userAgent),
    platformVersion: "",
    screen: screenHints,
    hardwareConcurrency: nav?.hardwareConcurrency ?? 0,
    deviceMemoryGb: typeof nav?.deviceMemory === "number" ? nav.deviceMemory : null,
    source: screenHints.cssWidth > 0 ? "screen-only" : "none",
  };

  const uaData = nav?.userAgentData;
  if (!uaData?.getHighEntropyValues) return base;
  try {
    const high = await uaData.getHighEntropyValues(["model", "platform", "platformVersion"]);
    const uaModel = typeof high.model === "string" ? high.model.trim() : "";
    const platform = typeof high.platform === "string" ? high.platform : (uaData.platform ?? "");
    return {
      ...base,
      uaModel,
      brand: brandFor(uaModel, platform, userAgent),
      platformVersion: typeof high.platformVersion === "string" ? high.platformVersion : "",
      source: uaModel ? "ua-client-hints" : base.source,
    };
  } catch {
    // Permissions policy, an older Chromium, or a privacy extension. The screen still
    // narrows an iPhone down, and an unknown Android is no worse off than before.
    return base;
  }
}


// ---------------------------------------------------------------------------------
// What the phone can do, decided before a frame is recorded
// ---------------------------------------------------------------------------------

/**
 * The lens behind a set of hints, or null when nothing matched.
 *
 * The same lookup the engine does (`biosphere/pano/devices.py`), against the same table,
 * generated from it. The app needs the answer EARLIER than the engine does: optical
 * stabilisation decides how fast a sweep may go, and a preset chosen for a phone that
 * cannot carry it wastes the user's minutes on frames the solve will not use.
 */
export function resolvePhoneLens(hints: DeviceModelHints | null): {
  lens: PhoneLens | null;
  /** iOS families are several phones; "maybe" means the screen cannot tell them apart. */
  lidar: "yes" | "no" | "maybe";
  candidates: string[];
} {
  if (!hints) return { lens: null, lidar: "no", candidates: [] };
  const model = hints.uaModel.trim().toLowerCase();
  if (model) {
    const exact = PHONE_LENSES.find((p) => p.aliases.includes(model));
    if (exact) return { lens: exact, lidar: exact.lidar ? "yes" : "no", candidates: [exact.model] };
    const loose = PHONE_LENSES.find((p) =>
      p.aliases.some((a) => a.length >= 5 && (a.includes(model) || model.includes(a))),
    );
    if (loose) return { lens: loose, lidar: loose.lidar ? "yes" : "no", candidates: [loose.model] };
  }
  const { cssWidth, cssHeight, devicePixelRatio } = hints.screen;
  if (cssWidth > 0 && cssHeight > 0 && devicePixelRatio > 0) {
    const short = Math.min(cssWidth, cssHeight);
    const long = Math.max(cssWidth, cssHeight);
    const hit = IOS_FINGERPRINTS.find(
      (f) =>
        Math.min(f.cssWidth, f.cssHeight) === short &&
        Math.max(f.cssWidth, f.cssHeight) === long &&
        Math.abs(f.dpr - devicePixelRatio) <= 0.05,
    );
    if (hit) {
      // A family, so the "lens" reported here is the family's midpoint. Good enough for
      // coaching, which is all the app does with it; the engine carries the spread too.
      return {
        lens: {
          id: `ios-${hit.cssWidth}x${hit.cssHeight}`,
          brand: "Apple",
          model: hit.models.length === 1 ? hit.models[0] : `iPhone (${hit.models.length} models)`,
          tier: "flagship",
          equivMm: hit.equivMm,
          ois: true,
          lidar: hit.lidar === "yes",
          aliases: [],
        },
        lidar: hit.lidar,
        candidates: [...hit.models],
      };
    }
  }
  return { lens: null, lidar: "no", candidates: [] };
}

/**
 * Which preset this phone can actually carry.
 *
 * STUDIO costs roughly three times BALANCED in engine time and buys its resolution and its
 * seam search back only if the frames are sharp enough to hold them. The thing that decides
 * that on a hand-held sweep is not the megapixel count, it is optical stabilisation: without
 * OIS every frame at an indoor shutter carries motion smear, and 8K of a smeared frame is
 * 8K of smear. So the preset follows the stabiliser and the tier, not the sensor's headline.
 *
 * Deliberately only a DEFAULT. The user may still pick either, and a tripod would make this
 * advice wrong; nothing here is a lock.
 */
export function recommendedQualityMode(lens: PhoneLens | null): "balanced" | "studio" {
  if (!lens) return "balanced";
  return lens.ois && (lens.tier === "flagship" || lens.tier === "upper-mid") ? "studio" : "balanced";
}

/**
 * The one sentence worth saying to the user about their phone, or null.
 *
 * Only ever about something they can act on. "Your phone has no OIS" is actionable -- sweep
 * slower, turn a light on. "Your phone has a LiDAR scanner" is not, on its own, so it is
 * paired with what the browser can and cannot do with it, because a customer who knows the
 * scanner exists will otherwise assume the property used it.
 */
export function deviceCoachingNote(
  lens: PhoneLens | null,
  lidar: "yes" | "no" | "maybe",
  locale: "tr" | "en",
): string | null {
  const tr = locale === "tr";
  if (lens && !lens.ois) {
    return tr
      ? "Bu telefonda optik sabitleme yok: yavaş çevirin ve odayı aydınlatın, netlik doğrudan buna bağlı."
      : "This phone has no optical stabilisation: sweep slowly and light the room, sharpness follows directly from that.";
  }
  if (lidar === "yes") {
    return tr
      ? "Bu modelde LiDAR var, ama tarayıcı ona erişemiyor; ölçüm kamera yüksekliğinden hesaplanır."
      : "This model has LiDAR, but the browser cannot reach it; distances are computed from the camera height.";
  }
  return null;
}
