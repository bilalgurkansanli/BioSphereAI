"use client";

/**
 * Streams a panorama in at increasing resolution, cheapest rung first.
 *
 * The honest constraint this hook exists to respect: an 8192x4096 RGBA texture is 134 MB
 * of VRAM before mipmaps and about 180 MB after. Handing that to a mid-range phone does
 * not degrade gracefully — it drops the WebGL context and the tab dies. So the ladder is
 * capped by three separate limits (hardware `MAX_TEXTURE_SIZE`, reported device memory,
 * and the coarse-pointer/small-screen heuristic that catches phones lying about memory),
 * and every texture we replace is released rather than left for the GC to find.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import type { PanoramaLevels } from "@/lib/types";
import { RUNG_ORDER, RUNG_WIDTH, TIMING, type QualityRung, type TextureStatus } from "./types";

/** Grace period before a superseded texture is released, so the cross-fade can finish. */
const RETIRE_DELAY_MS = TIMING.rungFadeMs + 150;

/** Wait this long after the ladder settles before warming the neighbouring rooms. */
const PRELOAD_IDLE_MS = 2000;

type NavigatorWithMemory = Navigator & { deviceMemory?: number };

function urlForRung(levels: PanoramaLevels, rung: QualityRung): string | null {
  switch (rung) {
    case "preview":
      return levels.preview_url || null;
    case "sd":
      return levels.sd_url;
    case "hd":
      return levels.hd_url;
    case "uhd":
      return levels.uhd_url;
  }
}

/**
 * The largest rung this machine is allowed to load.
 *
 * `MAX_TEXTURE_SIZE` is a hard ceiling — exceeding it is an INVALID_VALUE, not a slow
 * path. The other two limits are policy: a device reporting under 4 GB, or a small
 * touchscreen, gets capped at 4K because the extra pixels are invisible at that screen
 * size and the allocation is what kills the tab.
 *
 * A HEADSET IS NOT A SMALL TOUCHSCREEN, and the 8K rung is the one it needs. A Quest 2 panel
 * resolves about 20 pixels per degree and a Quest 3 about 25; the 4K equirect carries 11.4
 * (4096 / 360) and the 8K 22.8. So in a headset the 4K picture is visibly soft and the 8K one
 * is matched to the panel, and the touchscreen heuristic — which a headset browser can trip,
 * since it reports a coarse pointer and a small "screen" — must not hold it back. The memory
 * limit still applies: 180 MB of texture on a device that says it has under 4 GB is a crash
 * whether or not it is on somebody's face.
 */
function computeMaxRung(renderer: THREE.WebGLRenderer, headset = false): QualityRung {
  const ctx = renderer.getContext();
  const maxSize = ctx.getParameter(ctx.MAX_TEXTURE_SIZE) as number;

  let cap: QualityRung = "preview";
  for (const rung of RUNG_ORDER) {
    if (RUNG_WIDTH[rung] <= maxSize) cap = rung;
  }

  if (typeof window !== "undefined") {
    const memory = (navigator as NavigatorWithMemory).deviceMemory;
    const lowMemory = typeof memory === "number" && memory < 4;
    const coarse = window.matchMedia?.("(pointer: coarse)").matches ?? false;
    const smallScreen = Math.min(window.screen?.width ?? 9999, window.screen?.height ?? 9999) < 820;
    if (lowMemory || (!headset && coarse && smallScreen)) {
      if (RUNG_WIDTH[cap] > RUNG_WIDTH.hd) cap = "hd";
    }
  }

  return cap;
}

interface FetchedImage {
  /** Object URL for the decoded bytes, or the original URL when we streamed nothing. */
  href: string;
  revoke: boolean;
}

/**
 * Fetches the bytes with real progress, then hands them to `TextureLoader` as a blob.
 *
 * `TextureLoader` alone cannot report progress: it builds an `HTMLImageElement`, whose
 * loading is opaque, so its `onProgress` argument never fires. Streaming the response
 * ourselves is the only way to drive a determinate progress bar. If the fetch fails for
 * any reason other than an abort we fall back to letting `TextureLoader` hit the URL
 * directly, so a CORS or streaming quirk costs us the progress bar, not the panorama.
 */
async function fetchWithProgress(
  url: string,
  signal: AbortSignal,
  onProgress: (fraction: number) => void,
): Promise<FetchedImage> {
  try {
    const res = await fetch(url, { signal, mode: "cors", credentials: "omit" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const total = Number(res.headers.get("content-length") ?? 0);
    const reader = res.body?.getReader();
    if (!reader) {
      const blob = await res.blob();
      onProgress(1);
      return { href: URL.createObjectURL(blob), revoke: true };
    }

    const chunks: Uint8Array[] = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        received += value.byteLength;
        // Without a content-length, approach 95% asymptotically rather than lying.
        onProgress(total > 0 ? Math.min(received / total, 1) : 0.95 * (1 - Math.exp(-received / 1.5e6)));
      }
    }

    const merged = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.byteLength;
    }
    onProgress(1);
    const blob = new Blob([merged], { type: res.headers.get("content-type") ?? "image/jpeg" });
    return { href: URL.createObjectURL(blob), revoke: true };
  } catch (error) {
    if (signal.aborted) throw error;
    return { href: url, revoke: false };
  }
}

function configure(texture: THREE.Texture, renderer: THREE.WebGLRenderer): THREE.Texture {
  // Photographic content: decode sRGB on sample so filtering happens in linear light,
  // and let the renderer encode back on output. Anything else grades the photo.
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = true;
  // Repeat on u so the +/-180 column interpolates cyclically instead of clamping into a
  // visible seam; clamp on v because the poles have nothing to wrap to.
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

export interface ProgressiveTextureOptions {
  panorama: PanoramaLevels | null;
  /** Identity of the owning scene; a change means "new room", not "better pixels". */
  sceneKey: string;
  /** Neighbouring rooms' mid-resolution URLs, warmed into the HTTP cache when idle. */
  preloadUrls?: string[];
  onStatus?: (status: TextureStatus) => void;
  /** Presenting to a headset: the 8K rung is allowed where the hardware has room for it. */
  headset?: boolean;
}

export interface ProgressiveTextureResult {
  texture: THREE.Texture | null;
  status: TextureStatus;
  /** Jump straight to the best rung this device allows, skipping the rest of the ladder. */
  forceHighest: () => void;
}

export function useProgressiveTexture(options: ProgressiveTextureOptions): ProgressiveTextureResult {
  const { panorama, sceneKey, preloadUrls, onStatus, headset = false } = options;
  const renderer = useThree((s) => s.gl);

  const [maxRung, setMaxRung] = useState<QualityRung>("hd");
  const [texture, setTexture] = useState<THREE.Texture | null>(null);
  const [status, setStatus] = useState<TextureStatus>({
    rung: null,
    maxRung: "hd",
    bestAvailable: null,
    progress: 0,
    ready: false,
    loading: true,
  });

  const statusRef = useRef(status);
  const onStatusRef = useRef(onStatus);

  // Declared before the ladder effect so it is always the fresher of the two on a commit.
  useEffect(() => {
    onStatusRef.current = onStatus;
  });

  const publish = useCallback((next: Partial<TextureStatus>) => {
    const merged = { ...statusRef.current, ...next };
    statusRef.current = merged;
    setStatus(merged);
    onStatusRef.current?.(merged);
  }, []);

  useEffect(() => {
    const cap = computeMaxRung(renderer, headset);
    setMaxRung(cap);
    publish({ maxRung: cap });
  }, [renderer, publish, headset]);

  /**
   * Which picture, at which rung, is on the sphere right now — so a cap change can climb from
   * it. Keyed by the room AND its URLs: the same room with new URLs (re-processed, refetched) is
   * a new picture and has to climb from the bottom again.
   */
  const painted = useRef<{ key: string; rung: QualityRung } | null>(null);

  /**
   * The texture on the sphere right now. Owned by nobody's teardown: a run that is cancelled
   * releases what it loaded EXCEPT this, and whatever replaces it retires it after the cross-
   * fade. Before, a cancelled run released everything it had loaded, including the picture
   * still on screen, which three then had to upload again on the next frame — invisible while
   * every re-run restarted at the preview, a hitch once a re-run could keep the picture.
   */
  const displayed = useRef<THREE.Texture | null>(null);
  const show = useCallback((next: THREE.Texture | null): THREE.Texture | null => {
    const previous = displayed.current;
    displayed.current = next;
    setTexture(next);
    return previous;
  }, []);

  useEffect(
    () => () => {
      const last = displayed.current;
      displayed.current = null;
      if (last) setTimeout(() => last.dispose(), RETIRE_DELAY_MS);
    },
    [],
  );

  // `forceHighest` flips this; the ladder effect re-runs and goes straight to the top.
  const [forced, setForced] = useState(0);
  const forceHighest = useCallback(() => setForced((n) => n + 1), []);

  const laddderDone = status.loading === false;

  useEffect(() => {
    if (!panorama) {
      const previous = show(null);
      if (previous) setTimeout(() => previous.dispose(), RETIRE_DELAY_MS);
      publish({ rung: null, bestAvailable: null, progress: 0, ready: false, loading: false });
      return;
    }

    const available = RUNG_ORDER.filter((r) => urlForRung(panorama, r) !== null);
    const permitted = available.filter((r) => RUNG_WIDTH[r] <= RUNG_WIDTH[maxRung]);
    const best = permitted.length > 0 ? permitted[permitted.length - 1] : null;

    if (!best) {
      const previous = show(null);
      if (previous) setTimeout(() => previous.dispose(), RETIRE_DELAY_MS);
      publish({ rung: null, bestAvailable: null, progress: 0, ready: false, loading: false });
      return;
    }

    // On a forced upgrade the low rungs are already on screen — do not re-walk them. Likewise
    // when the cap moves under a room already painted (putting on a headset lifts it to 8K):
    // climbing from the preview again would blur the picture the visitor is looking at for the
    // length of three downloads, and a cap that FELL leaves the sharper texture where it is.
    const pictureKey = `${sceneKey}|${RUNG_ORDER.map((r) => urlForRung(panorama, r) ?? "").join("|")}`;
    const paintedHere = painted.current?.key === pictureKey ? painted.current.rung : null;
    const plan =
      forced > 0 && statusRef.current.ready
        ? [best]
        : paintedHere
          ? permitted.filter((r) => RUNG_WIDTH[r] > RUNG_WIDTH[paintedHere])
          : permitted;
    if (plan.length === 0) {
      publish({ loading: false });
      return;
    }

    const controller = new AbortController();
    const owned: THREE.Texture[] = [];
    const timers: ReturnType<typeof setTimeout>[] = [];
    let cancelled = false;

    /** Release later, not now: the shader may still be mixing this texture out. */
    const retire = (tex: THREE.Texture, delay: number) => {
      timers.push(setTimeout(() => tex.dispose(), delay));
    };

    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin("anonymous");

    publish({ bestAvailable: available[available.length - 1] ?? null, loading: true, progress: 0 });

    void (async () => {
      for (const rung of plan) {
        const url = urlForRung(panorama, rung);
        if (!url) continue;

        let source: FetchedImage | null = null;
        try {
          let lastReported = -1;
          source = await fetchWithProgress(url, controller.signal, (fraction) => {
            // The bar is a React state write; 2% granularity keeps it off the hot path.
            if (cancelled || fraction - lastReported < 0.02) return;
            lastReported = fraction;
            publish({ progress: fraction });
          });
          if (cancelled) break;

          const loaded = await loader.loadAsync(source.href);
          if (source.revoke) URL.revokeObjectURL(source.href);
          source = null;

          if (cancelled) {
            loaded.dispose();
            break;
          }

          configure(loaded, renderer);
          owned.push(loaded);
          const superseded = show(loaded);
          painted.current = { key: pictureKey, rung };
          publish({ rung, progress: 1, ready: true, loading: rung !== plan[plan.length - 1] });

          // Retire what this replaced once the cross-fade has finished with it: tracked if this
          // run loaded it (a cancel then releases it at once), untracked if an earlier run did.
          if (superseded && superseded !== loaded) {
            if (owned.includes(superseded)) retire(superseded, RETIRE_DELAY_MS);
            else setTimeout(() => superseded.dispose(), RETIRE_DELAY_MS);
          }
        } catch {
          if (source?.revoke) URL.revokeObjectURL(source.href);
          if (cancelled) break;
          // A rung that 404s or decodes badly is not fatal: keep whatever is on screen
          // and stop climbing rather than blanking a working panorama.
          publish({ loading: false });
          break;
        }
      }
      if (!cancelled) publish({ loading: false });
    })();

    return () => {
      cancelled = true;
      controller.abort();
      for (const timer of timers) clearTimeout(timer);
      // Same grace period on teardown: the next room cross-fades over these. The one still
      // on the sphere is not ours to release — see `displayed`.
      for (const tex of owned) if (tex !== displayed.current) setTimeout(() => tex.dispose(), RETIRE_DELAY_MS);
    };
  }, [panorama, sceneKey, maxRung, forced, renderer, publish, show]);

  // Warm the neighbouring rooms once this one has stopped competing for bandwidth. The
  // bytes land in the HTTP cache, so stepping through a doorway decodes rather than
  // downloads — and nothing is uploaded to the GPU until that room is actually shown.
  const preloadKey = (preloadUrls ?? []).join("|");
  useEffect(() => {
    if (!laddderDone || !preloadKey) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      for (const url of preloadKey.split("|")) {
        if (!url) continue;
        void fetch(url, { signal: controller.signal, mode: "cors", credentials: "omit" })
          .then((res) => res.arrayBuffer())
          .catch(() => undefined);
      }
    }, PRELOAD_IDLE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [preloadKey, laddderDone]);

  return { texture, status, forceHighest };
}
