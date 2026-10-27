/**
 * Typed client for the BioSphere engine API.
 *
 * Every shape here comes from `./types`, which mirrors the FastAPI schemas. Media URLs
 * arrive from the backend as absolute *paths* (`/media/...`) because the engine does not
 * know what host it is reachable on; `absoluteUrl` resolves them against the API origin
 * so the same payload works from the dashboard, from an embed, and from a phone on the
 * local network.
 */

import type {
  AdvisorChatRequest,
  AiSettings,
  AdvisorEvent,
  AdvisorPublicState,
  InvestmentInputs,
  InvestmentState,
  ApiError,
  AuthSession,
  AuthUser,
  LoginRequest,
  RegisterRequest,
  BuildPlanRequest,
  CreateHotspotRequest,
  CreateRoomRequest,
  CreateSceneRequest,
  CreatePropertyRequest,
  EmptyRoomRequest,
  EmptyRoomResponse,
  FloorPlan,
  GenerateListingRequest,
  ListingCopy,
  ListingState,
  UpdatePlanRequest,
  HealthResponse,
  HideFurnitureRequest,
  HideFurnitureResponse,
  Hotspot,
  Job,
  JobEvent,
  MeasureRequest,
  MeasureResponse,
  Paginated,
  QualityMode,
  RestylePhoto,
  RestyleRequest,
  RestyleResponse,
  RestyleSceneRequest,
  RestyleSceneResponse,
  RestyleStylesResponse,
  ScenePhoto,
  ScenePhotoEmptyRequest,
  ScenePhotoRestyleRequest,
  ScenePhotoVariant,
  UpdateScenePhotoRequest,
  Scene,
  Property,
  PropertyDetail,
  UpdateHotspotRequest,
  UpdateSceneRequest,
  UpdatePropertyRequest,
  Walkthrough,
  WalkthroughJobResponse,
} from "./types";

import { announceSignedOut, authHeaders, clearToken, getToken, setToken } from "./session";
import { PART_RETRY_DELAYS_MS, STAGING_LOST_CODES, type UploadResume } from "./capture/uploadResume";

const CONFIGURED_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8008").replace(/\/+$/, "");

const LOOPBACK = /^(localhost|127\.0\.0\.1|\[::1\]|0\.0\.0\.0)$/i;

/**
 * Resolve the engine's address at runtime rather than trusting the build.
 *
 * `NEXT_PUBLIC_API_BASE` is inlined when the bundle is compiled, so a build made for a
 * desktop developer hard-codes a loopback address — and loopback on a phone is the phone.
 * The guided capture is opened on a phone by design (QR handoff, a link sent over chat),
 * which makes that the normal case rather than an edge case: every request would quietly
 * fail against a device that is not running the engine, with no error the user could act
 * on.
 *
 * So when the configured base points at loopback but the page itself is being served from
 * somewhere else, keep the port and scheme and follow the page's own host. Any explicitly
 * configured non-loopback base is respected untouched — that is a deliberate deployment
 * choice and must not be second-guessed.
 */
/**
 * A runtime override for the engine's address, for a developer with two engines.
 *
 * `?api=http://127.0.0.1:8018` on any page pins it (and remembers it in localStorage under
 * `biosphere.apiBase`); `?api=` with an empty value forgets it. Never read on the server,
 * and never shown to a visitor: the embed is a server component and ignores it.
 */
function overriddenBase(): string | null {
  try {
    const key = "biosphere.apiBase";
    const fromQuery = new URLSearchParams(window.location.search).get("api");
    if (fromQuery !== null) {
      if (fromQuery === "") window.localStorage.removeItem(key);
      else window.localStorage.setItem(key, fromQuery);
    }
    const stored = window.localStorage.getItem(key);
    if (!stored) return null;
    return new URL(stored).toString().replace(/\/+$/, "");
  } catch {
    return null;
  }
}

function resolveBase(): string {
  if (typeof window === "undefined") return CONFIGURED_BASE;
  const overridden = overriddenBase();
  if (overridden) return overridden;
  try {
    const configured = new URL(CONFIGURED_BASE);
    if (!LOOPBACK.test(configured.hostname)) return CONFIGURED_BASE;
    if (LOOPBACK.test(window.location.hostname)) return CONFIGURED_BASE;
    // Served over HTTPS from a LAN address: that is `scripts/lan-proxy.mjs`, which puts the
    // engine behind `/api` and `/media` on its own origin so the phone trusts one certificate.
    // Following the host but keeping plain http and the engine's port would be mixed content,
    // blocked before a single request leaves the phone.
    if (window.location.protocol === "https:") return window.location.origin;
    configured.hostname = window.location.hostname;
    return configured.toString().replace(/\/+$/, "");
  } catch {
    return CONFIGURED_BASE;
  }
}

export const API_BASE = resolveBase();

const V1 = `${API_BASE}/api/v1`;

/** Resolve a backend-relative media path against the API origin. */
export function absoluteUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (/^https?:\/\//i.test(path) || path.startsWith("data:") || path.startsWith("blob:")) return path;
  return `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
}

export class BioSphereError extends Error {
  readonly status: number;
  readonly detail: ApiError;

  constructor(status: number, detail: ApiError) {
    super(detail.message);
    this.name = "BioSphereError";
    this.status = status;
    this.detail = detail;
  }

  /** Prefer the Turkish copy when the UI is in Turkish. */
  localised(locale: "tr" | "en"): string {
    return locale === "tr" ? this.detail.message_tr || this.detail.message : this.detail.message;
  }
}

const FALLBACK_ERROR = (status: number): ApiError => ({
  code: "network_error",
  message: `Request failed with status ${status}`,
  message_tr: "Sunucuya ulaşılamadı. Motorun çalıştığından emin olun.",
});

/**
 * How long any single JSON call may take before it is treated as a failure.
 *
 * A FETCH WITH NO DEADLINE IS A LOADING SPINNER WITH NO END. `useApi` only leaves its
 * `initial` state when the promise settles, so a request that hangs rather than fails
 * renders skeleton cards forever: no data, no error, no retry button, and nothing on
 * screen that names the problem. Observed exactly that on the dashboard against a
 * reachable engine -- `fetch` never rejected, so the page sat on placeholders while
 * `curl` answered the same URL in 48 ms.
 *
 * The engine answers these in single-digit milliseconds on a LAN, and the slowest of them
 * measured here (60 properties with their scenes) took 48 ms. Fifteen seconds is three orders
 * of magnitude of headroom and still far inside the patience of someone watching a page
 * that has not painted. Uploads do NOT come through here -- they run on XHR with their
 * own stall detection, which is why this can be short.
 */
const REQUEST_TIMEOUT_MS = 15_000;

/**
 * ...except for the calls that are a computation rather than a lookup.
 *
 * Emptying a room runs a scene-parsing model over sixteen windows and then fills the mask,
 * the first time it is asked for a scene. Fifteen seconds would abort it *while the server is
 * still working*, and the next press would start the same work again — so the one endpoint
 * that can legitimately take longer than a page load says so, and only it.
 *
 * MEASURED ON A REAL 4096-WIDE CAPTURE: 53 s with the classical fill, which is the default.
 * The optional diffusion fill is minutes, because it is a couple of dozen windows at several
 * seconds of GPU each — 120 s was written for "seconds of OpenCV" and cut that path off
 * mid-run, leaving the engine to finish work nobody would ever see. Ten minutes covers both
 * with room for a slower card.
 *
 * IT IS A DEADLINE, NOT A DESIGN. A computation this long does not belong in a request at
 * all: the engine already has a job queue with progress, reconnection and survival across a
 * closed tab, and `/ai/empty-room` should be a job like a stitch is. Until it is, this number
 * is what stands between a working feature and a spinner that gives up first.
 */
const EMPTY_ROOM_TIMEOUT_MS = 600_000;

/** The caller's abort and ours, as one signal. Older engines get the timeout alone. */
function withDeadline(caller: AbortSignal | null | undefined, deadline: AbortSignal): AbortSignal {
  if (!caller) return deadline;
  const any = (AbortSignal as { any?: (s: AbortSignal[]) => AbortSignal }).any;
  return typeof any === "function" ? any([caller, deadline]) : caller;
}

type RequestInitWithDeadline = RequestInit & { timeoutMs?: number };

async function request<T>(path: string, init: RequestInitWithDeadline = {}): Promise<T> {
  let res: Response;
  const { timeoutMs = REQUEST_TIMEOUT_MS, ...rest } = init;
  const deadline = AbortSignal.timeout(timeoutMs);
  try {
    res = await fetch(`${V1}${path}`, {
      ...rest,
      headers: {
        ...(rest.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...authHeaders(),
        ...rest.headers,
      },
      // The cookie the engine also sets is only useful same-origin; sending it costs
      // nothing and is what makes an httpOnly deployment work without touching this file.
      credentials: "include",
      cache: "no-store",
      // Covers the response BODY as well as the connection: headers that arrive and then
      // stall mid-body would otherwise hang `res.text()` below, past every check here.
      signal: withDeadline(rest.signal, deadline),
    });
  } catch (cause) {
    // The caller aborting is not a failure to report -- it is the caller getting what it
    // asked for, and dressing it up as "the engine is unreachable" would be a lie the UI
    // then shows to someone who navigated away.
    if (init.signal?.aborted) throw cause;
    if (deadline.aborted) {
      throw new BioSphereError(0, {
        code: "timeout",
        message: `The engine did not answer within ${timeoutMs / 1000}s.`,
        message_tr: `Motor ${timeoutMs / 1000} saniyede yanıt vermedi.`,
        hint: "The address may be reachable but not answering — check the engine and the proxy.",
      });
    }
    throw new BioSphereError(0, {
      code: "unreachable",
      message: "The engine is not reachable.",
      message_tr: "Motor çalışmıyor gibi görünüyor. `biosphere serve` komutunu çalıştırın.",
      hint: String(cause),
    });
  }

  if (res.status === 204) return undefined as T;

  const text = await res.text();
  const payload = text ? safeJson(text) : null;

  if (!res.ok) {
    const detail =
      payload && typeof payload === "object" && "code" in payload
        ? (payload as ApiError)
        : payload && typeof payload === "object" && "detail" in payload
          ? ((payload as { detail: ApiError }).detail ?? FALLBACK_ERROR(res.status))
          : FALLBACK_ERROR(res.status);
    // A 401 while we were holding a token means the session died — it expired, it was
    // revoked from another device, or the database was reset under us. Dropping it here
    // rather than in each caller is what keeps a dead token from producing a page of
    // identical "sign in to continue" toasts on the next poll.
    if (res.status === 401 && !path.startsWith("/auth/") && getToken()) announceSignedOut();
    throw new BioSphereError(res.status, detail);
  }

  return payload as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

const json = (body: unknown) => JSON.stringify(body);

// ---------------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------------

/**
 * How much of a capture goes in one request before it is sliced instead.
 *
 * THE SINGLE POST IS NOT WRONG, IT JUST HAS A CEILING. A browser must build the whole
 * multipart body before it can send it, and on iOS a 270 MB capture never left the phone:
 * no request reached the proxy, no error reached the page, and the bar sat at 0% under
 * "keep this page open" until the user gave up — with the only copy of the recording in
 * that tab. Android was fine at 40-150 MB, so the proven path keeps everything under this
 * line and only larger captures take the sliced one.
 */
const CHUNK_ABOVE_BYTES = 32 * 1024 * 1024;
/** 8 MB parts: small enough that a phone holds one at a time, large enough that a 270 MB
 *  capture is 34 requests rather than 270. */
const PART_BYTES = 8 * 1024 * 1024;
/** A part is small and idempotent — it is addressed by index — so a flaky link is worth
 *  retrying rather than failing a quarter-gigabyte upload over one dropped chunk. */
const PART_ATTEMPTS = 3;

/** Wait `ms` before a retry; an abort ends the wait at once (the attempt then rejects). */
function pause(ms: number, signal: AbortSignal | undefined): Promise<void> {
  if (!(ms > 0) || signal?.aborted) return Promise.resolve();
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

function newUploadId(): string {
  const bytes = new Uint8Array(12);
  (globalThis.crypto ?? ({} as Crypto)).getRandomValues?.(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** One part, by PUT, with progress and abort. Resolves when the server has it. */
function putPart(
  uploadId: string,
  index: number,
  body: Blob,
  signal: AbortSignal | undefined,
  onBytes: (loadedInThisPart: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `${V1}/properties/uploads/${uploadId}/parts/${index}`);
    xhr.withCredentials = true;
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    for (const [key, value] of Object.entries(authHeaders())) xhr.setRequestHeader(key, value);
    xhr.upload.onprogress = (e) => onBytes(e.loaded);
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300
        ? resolve()
        : reject(new BioSphereError(xhr.status, FALLBACK_ERROR(xhr.status)));
    const fail = () =>
      reject(
        new BioSphereError(0, {
          code: "upload_failed",
          message: `Part ${index} failed.`,
          message_tr: "Yükleme parçası gönderilemedi.",
        }),
      );
    xhr.onerror = fail;
    xhr.upload.onerror = fail;
    xhr.onabort = () =>
      reject(new BioSphereError(0, { code: "aborted", message: "Upload cancelled.", message_tr: "Yükleme iptal edildi." }));
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.send(body);
  });
}

/**
 * Send the capture as parts, then ask the server to assemble it.
 *
 * `Blob.slice` is a view, not a copy, so the phone holds one 8 MB part at a time however
 * long the capture ran. Each part is addressed by index and is therefore idempotent: a
 * dropped chunk is retried rather than costing the whole upload. Progress is the sum of the
 * parts already accepted plus whatever the part in flight has moved, so the bar means
 * "bytes the server has", not "bytes we handed to the browser".
 */
async function uploadInParts(
  propertyId: string,
  files: File[],
  meta: CreateSceneRequest,
  total: number,
  onProgress: ((fraction: number) => void) | undefined,
  signal: AbortSignal | undefined,
  resume?: UploadResume,
): Promise<{ scene: Scene; job: Job }> {
  // With a `resume`, a retry continues the SAME staged upload and skips every part the
  // server already acknowledged; see `lib/capture/uploadResume.ts`.
  const attempt0 = resume ? resume.begin(files, newUploadId) : { uploadId: newUploadId(), skip: new Set<number>() };
  const uploadId = attempt0.uploadId;
  const manifest: { filename: string; parts: number[] }[] = [];
  let settled = 0;
  let index = 0;

  for (const file of files) {
    const parts: number[] = [];
    for (let offset = 0; offset < Math.max(file.size, 1); offset += PART_BYTES) {
      const slice = file.slice(offset, Math.min(offset + PART_BYTES, file.size));
      const at = index;
      if (!attempt0.skip.has(at)) {
        let lastAttemptError: unknown = null;
        let sent = false;
        for (let attempt = 0; attempt < PART_ATTEMPTS && !sent; attempt += 1) {
          if (attempt > 0) await pause(PART_RETRY_DELAYS_MS[Math.min(attempt, PART_RETRY_DELAYS_MS.length - 1)], signal);
          try {
            await putPart(uploadId, at, slice, signal, (loaded) =>
              onProgress?.(Math.min(1, (settled + loaded) / total)),
            );
            sent = true;
            resume?.ack(at);
          } catch (cause) {
            // An abort is the user, not the link: never retry through it.
            if (cause instanceof BioSphereError && cause.detail.code === "aborted") throw cause;
            lastAttemptError = cause;
          }
        }
        if (!sent) throw lastAttemptError;
      }
      settled += slice.size;
      onProgress?.(Math.min(1, settled / total));
      parts.push(at);
      index += 1;
    }
    manifest.push({ filename: file.name, parts });
  }

  let created: { scene: Scene; job: Job };
  try {
    created = await request<{ scene: Scene; job: Job }>(`/properties/${propertyId}/scenes/from-parts`, {
      method: "POST",
      body: JSON.stringify({
        upload_id: uploadId,
        name: meta.name,
        quality_mode: meta.quality_mode,
        capture_kind: meta.capture_kind,
        files: manifest,
      }),
      signal,
    });
  } catch (cause) {
    if (resume && cause instanceof BioSphereError && STAGING_LOST_CODES.includes(cause.detail.code)) {
      // The parts we skipped are gone (expired or swept). Start over once, from scratch,
      // rather than handing the user a retry button that can only fail the same way.
      resume.reset();
      if (attempt0.skip.size > 0) return uploadInParts(propertyId, files, meta, total, onProgress, signal, resume);
    }
    throw cause;
  }
  // Assembled: the server has deleted the staging directory, so there is nothing to resume.
  resume?.reset();
  onProgress?.(1);
  return created;
}

/**
 * Upload the walkthrough videos (general + rooms) as staged parts, then finalise.
 *
 * Same part protocol as a room capture; only the finalise body differs, because one
 * upload carries several areas and each has to be told apart by file name.
 */
async function uploadWalkthroughInParts(
  propertyId: string,
  base: File[],
  rooms: { name: string; scene_id?: string | null; floor_index?: number; files: File[] }[],
  quality_mode: QualityMode | "",
  onProgress: ((fraction: number) => void) | undefined,
  signal: AbortSignal | undefined,
  deferBuild = false,
): Promise<WalkthroughJobResponse> {
  const uploadId = newUploadId();
  const all = [...base, ...rooms.flatMap((r) => r.files)];
  const total = Math.max(
    all.reduce((n, f) => n + f.size, 0),
    1,
  );
  let settled = 0;
  let index = 0;
  const partsOf = new Map<File, { filename: string; parts: number[] }>();
  for (const file of all) {
    const parts: number[] = [];
    for (let offset = 0; offset < Math.max(file.size, 1); offset += PART_BYTES) {
      const slice = file.slice(offset, Math.min(offset + PART_BYTES, file.size));
      const at = index;
      let lastAttemptError: unknown = null;
      let sent = false;
      for (let attempt = 0; attempt < PART_ATTEMPTS && !sent; attempt += 1) {
        try {
          await putPart(uploadId, at, slice, signal, (loaded) =>
            onProgress?.(Math.min(1, (settled + loaded) / total)),
          );
          sent = true;
        } catch (cause) {
          if (cause instanceof BioSphereError && cause.detail.code === "aborted") throw cause;
          lastAttemptError = cause;
        }
      }
      if (!sent) throw lastAttemptError;
      settled += slice.size;
      onProgress?.(Math.min(1, settled / total));
      parts.push(at);
      index += 1;
    }
    partsOf.set(file, { filename: file.name, parts });
  }
  const created = await request<WalkthroughJobResponse>(`/properties/${propertyId}/walkthrough/from-parts`, {
    method: "POST",
    body: JSON.stringify({
      upload_id: uploadId,
      quality_mode,
      defer_build: deferBuild,
      base: base.map((f) => partsOf.get(f)),
      rooms: rooms.map((r) => ({
        name: r.name,
        scene_id: r.scene_id ?? null,
        floor_index: r.floor_index ?? 0,
        files: r.files.map((f) => partsOf.get(f)),
      })),
    }),
    signal,
    timeoutMs: 120_000,
  });
  onProgress?.(1);
  return created;
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  /** Installation settings: the person's own Anthropic key, which is never sent back in full. */
  settings: {
    ai: () => request<AiSettings>("/settings/ai"),
    saveAiKey: (apiKey: string) =>
      request<AiSettings>("/settings/ai", { method: "PUT", body: json({ api_key: apiKey }) }),
    clearAiKey: () => request<AiSettings>("/settings/ai", { method: "DELETE" }),
    checkAiKey: () =>
      request<{ ok: boolean; message: string }>("/settings/ai/check", { method: "POST", timeoutMs: 20_000 }),
  },

  auth: {
    /** Creates the account AND signs it in — there is no separate "now log in" step. */
    register: async (body: RegisterRequest): Promise<AuthSession> => {
      const session = await request<AuthSession>("/auth/register", { method: "POST", body: json(body) });
      setToken(session.token);
      return session;
    },
    login: async (body: LoginRequest): Promise<AuthSession> => {
      const session = await request<AuthSession>("/auth/login", { method: "POST", body: json(body) });
      setToken(session.token);
      return session;
    },
    /** Trades the one-time code Google's callback hands back for a session. */
    exchange: async (code: string): Promise<AuthSession> => {
      const session = await request<AuthSession>("/auth/exchange", { method: "POST", body: json({ code }) });
      setToken(session.token);
      return session;
    },
    me: () => request<AuthUser>("/auth/me"),
    /**
     * Signs out locally even if the engine cannot be reached.
     *
     * The user pressed "çıkış"; the one outcome they must not get is still being signed in
     * because the server was down. The token is dropped either way — and it is only useful
     * against a server that is answering, which by hypothesis this one is not.
     */
    logout: async (): Promise<void> => {
      try {
        await request<void>("/auth/logout", { method: "POST" });
      } finally {
        clearToken();
      }
    },
    googleStatus: () => request<{ enabled: boolean }>("/auth/google/status"),
    /** Where the browser is sent to begin the Google flow. A full navigation, not a fetch. */
    googleStartUrl: () => `${V1}/auth/google/start`,
  },

  walkthrough: {
    get: (propertyId: string) => request<Walkthrough>(`/properties/${propertyId}/walkthrough`),
    rebuild: (propertyId: string, quality_mode: QualityMode | "" = "") =>
      request<Job>(`/properties/${propertyId}/walkthrough/rebuild`, { method: "POST", body: json({ quality_mode }) }),
    remove: (propertyId: string) => request<void>(`/properties/${propertyId}/walkthrough`, { method: "DELETE" }),
    /**
     * Upload the general walkthrough video and the room videos in one session and start the
     * 3D build. Always the parts path: these videos are minutes long.
     */
    create: (
      propertyId: string,
      base: File[],
      rooms: { name: string; scene_id?: string | null; floor_index?: number; files: File[] }[],
      quality_mode: QualityMode | "" = "",
      onProgress?: (fraction: number) => void,
      signal?: AbortSignal,
      /** Store the videos without building; `rebuild` starts the build once every area is in. */
      deferBuild = false,
    ) => uploadWalkthroughInParts(propertyId, base, rooms, quality_mode, onProgress, signal, deferBuild),
  },

  properties: {
    list: (limit = 50, offset = 0) => request<Paginated<Property>>(`/properties?limit=${limit}&offset=${offset}`),
    get: (id: string) => request<PropertyDetail>(`/properties/${id}`),
    create: (body: CreatePropertyRequest) => request<Property>("/properties", { method: "POST", body: json(body) }),
    update: (id: string, body: UpdatePropertyRequest) =>
      request<Property>(`/properties/${id}`, { method: "PATCH", body: json(body) }),
    remove: (id: string) => request<void>(`/properties/${id}`, { method: "DELETE" }),

    /**
     * A room with no capture behind it.
     *
     * The property's unit is a ROOM, and not every room starts with a panorama — an agent may
     * have four photographs of the kitchen and no 360 of it. Such a room is `ready` the
     * moment it exists; photographs are attached afterwards through `scenes.photos`.
     */
    createRoom: (propertyId: string, body: CreateRoomRequest) =>
      request<Scene>(`/properties/${propertyId}/rooms`, { method: "POST", body: json(body) }),

    /**
     * A capture link for the QR code: a token a phone can film THIS property with, without
     * signing in, for twelve hours. The engine limits what it can do (`api/deps.py`,
     * `_CAPTURE_ROUTES`); `revokeCaptureLinks` kills every one the property has handed out.
     */
    captureLink: (propertyId: string) =>
      request<{ token: string; expires_at: string; property_id: string }>(
        `/properties/${propertyId}/capture-link`,
        { method: "POST" },
      ),
    revokeCaptureLinks: (propertyId: string) =>
      request<void>(`/properties/${propertyId}/capture-links`, { method: "DELETE" }),
  },

  scenes: {
    get: (id: string) => request<Scene>(`/scenes/${id}`),
    update: (id: string, body: UpdateSceneRequest) =>
      request<Scene>(`/scenes/${id}`, { method: "PATCH", body: json(body) }),
    remove: (id: string) => request<void>(`/scenes/${id}`, { method: "DELETE" }),
    reprocess: (id: string, quality_mode: CreateSceneRequest["quality_mode"]) =>
      request<Job>(`/scenes/${id}/reprocess`, { method: "POST", body: json({ quality_mode }) }),

    /**
     * Upload with progress. `fetch` cannot report upload progress, so this one endpoint
     * uses XMLHttpRequest — a 500 MB video on a phone tether needs a real progress bar.
     */
    create(
      propertyId: string,
      files: File[],
      meta: CreateSceneRequest,
      onProgress?: (fraction: number) => void,
      signal?: AbortSignal,
      /** Kept by the caller across retries of the same capture; see `uploadResume.ts`. */
      resume?: UploadResume,
    ): Promise<{ scene: Scene; job: Job }> {
      const total = files.reduce((sum, f) => sum + f.size, 0);
      if (total > CHUNK_ABOVE_BYTES) {
        return uploadInParts(propertyId, files, meta, total, onProgress, signal, resume);
      }
      return new Promise((resolve, reject) => {
        const form = new FormData();
        for (const f of files) form.append("files", f, f.name);
        form.append("name", meta.name);
        form.append("quality_mode", meta.quality_mode);
        form.append("capture_kind", meta.capture_kind);

        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${V1}/properties/${propertyId}/scenes`);
        xhr.withCredentials = true;
        for (const [key, value] of Object.entries(authHeaders())) xhr.setRequestHeader(key, value);
        xhr.responseType = "text";

        /**
         * A half-open mobile connection produces neither `load` nor `error` — the socket is
         * simply never answered. Without this the user watches the bar frozen at 62% under
         * the line "keep this page open", which is the one instruction that guarantees they
         * will wait; eventually they kill the tab, and the tab's heap is the only copy of
         * the recording.
         *
         * The progress stream is the only proof the transfer is alive, so the clock is the
         * gap between bytes rather than a total. A 100 s capture is ~150 MB and may honestly
         * take twenty minutes on a phone tether; ninety seconds with nothing moving is not
         * slow, it is dead.
         *
         * Cleared on the UPLOAD's completion, not the response's: `upload.onprogress` stops
         * firing the moment the last byte is sent, while the server is still streaming the
         * file to disk and running ffprobe on it. Policing that window would abort the
         * server for doing its job.
         */
        /**
         * THE DEADLINE IS CHECKED AGAINST THE CLOCK, NOT TRUSTED TO THE TIMER.
         *
         * A phone that locks its screen or switches app freezes the page: `onprogress`
         * stops firing AND the pending timeout does not run. When the user comes back both
         * resume, and a bare `setTimeout` fires immediately for time the transfer never
         * actually spent stalled — aborting a perfectly healthy upload the moment the user
         * looks at it again, which on a 900 MB capture is the worst possible moment.
         *
         * So the timer only ASKS whether the deadline has really passed. If it has not — the
         * page was simply not running — it re-arms for the remainder instead of aborting.
         */
        const STALL_MS = 90_000;
        /**
         * A separate, much shorter deadline for the case where NOT ONE BYTE has moved.
         *
         * 90 seconds is the right patience for a transfer that is under way: a 150 MB capture
         * on a phone tether genuinely crawls, and aborting it would be worse than waiting. It
         * is the wrong patience for a request that never started. Those are different
         * failures with different causes — a refused connection, a certificate the browser
         * will not reuse, a proxy that never answered — and reporting them the same way cost
         * a capture: the bar sat at 0% under "keep this page open", which is the one
         * instruction that guarantees the user waits.
         */
        const NEVER_STARTED_MS = 25_000;
        let stallTimer = 0;
        let stalled = false;
        let everMoved = false;
        let lastProgressAt = Date.now();
        const armStall = (afterMs?: number) => {
          const limit = everMoved ? STALL_MS : NEVER_STARTED_MS;
          window.clearTimeout(stallTimer);
          stallTimer = window.setTimeout(() => {
            const idle = Date.now() - lastProgressAt;
            if (idle < limit) {
              armStall(limit - idle);
              return;
            }
            stalled = true;
            xhr.abort();
          }, afterMs ?? limit);
        };
        const disarmStall = () => window.clearTimeout(stallTimer);

        xhr.upload.onprogress = (e) => {
          if (e.loaded > 0) everMoved = true;
          lastProgressAt = Date.now();
          armStall();
          if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
        };
        xhr.upload.onloadend = disarmStall;
        // Safari raises this without always raising `xhr.onerror`, and an unhandled one
        // leaves the promise pending for ever — the shape of a capture lost in silence.
        xhr.upload.onerror = () => {
          disarmStall();
          reject(
            new BioSphereError(0, {
              code: "upload_failed",
              message: "The upload failed while sending.",
              message_tr: "Yükleme gönderim sırasında başarısız oldu. Kaydınız telefonunuzda duruyor.",
            }),
          );
        };
        xhr.onloadend = disarmStall;
        xhr.onload = () => {
          const payload = safeJson(xhr.responseText);
          if (xhr.status >= 200 && xhr.status < 300) {
            onProgress?.(1);
            resolve(payload as { scene: Scene; job: Job });
          } else {
            const detail =
              payload && typeof payload === "object" && "code" in payload
                ? (payload as ApiError)
                : FALLBACK_ERROR(xhr.status);
            reject(new BioSphereError(xhr.status, detail));
          }
        };
        xhr.onerror = () =>
          reject(
            new BioSphereError(0, {
              code: "upload_failed",
              message: "Upload failed.",
              message_tr: "Yükleme başarısız oldu. Bağlantınızı kontrol edin.",
            }),
          );
        // A stall aborts the request too, but it is not a cancellation and must not read as
        // one: "Yükleme iptal edildi" on the failure screen tells the user they did this.
        xhr.onabort = () =>
          reject(
            stalled
              ? everMoved
                ? new BioSphereError(0, {
                    code: "upload_stalled",
                    message: "The connection stopped responding.",
                    message_tr: "Bağlantı yanıt vermiyor. Kaydınız telefonunuzda duruyor; tekrar deneyin.",
                  })
                : new BioSphereError(0, {
                    // Nothing left the phone at all. That is not a slow network, and telling
                    // the user to wait it out would waste the capture a second time.
                    code: "upload_never_started",
                    message: "The connection to the server was never established.",
                    message_tr:
                      "Sunucuya bağlanılamadı — hiç veri gitmedi. Kaydınız telefonunuzda duruyor. Aynı Wi-Fi'da olduğunuzdan emin olup tekrar deneyin.",
                  })
              : new BioSphereError(0, {
                  code: "aborted",
                  message: "Upload cancelled.",
                  message_tr: "Yükleme iptal edildi.",
                }),
          );
        signal?.addEventListener("abort", () => xhr.abort(), { once: true });
        armStall();
        xhr.send(form);
      });
    },
  },

  photos: {
    /**
     * Attach photographs to a room. Multipart, several at a time.
     *
     * PARTIAL SUCCESS IS THE CONTRACT: with four good files and one corrupt, the server keeps
     * the four and the response says which arrived — refusing all five to punish one would
     * cost the user four uploads. Compare `response.length` against what was sent to tell the
     * user something was skipped.
     */
    add(sceneId: string, files: File[], signal?: AbortSignal): Promise<ScenePhoto[]> {
      const form = new FormData();
      for (const file of files) form.append("files", file, file.name);
      return request<ScenePhoto[]>(`/scenes/${sceneId}/photos`, {
        method: "POST",
        body: form,
        signal,
      });
    },

    update: (photoId: string, body: UpdateScenePhotoRequest) =>
      request<ScenePhoto>(`/scene-photos/${photoId}`, { method: "PATCH", body: json(body) }),

    remove: (photoId: string) => request<void>(`/scene-photos/${photoId}`, { method: "DELETE" }),

    /** Styles and the bare room already rendered for this photo. Asked once when it opens. */
    variants: (photoId: string) =>
      request<ScenePhotoVariant[]>(`/scene-photos/${photoId}/variants`),

    /**
     * The two AI tools, on a room's own photograph.
     *
     * Both get the long deadline for the reason `emptyRoom` does: these are computations, not
     * lookups. Both are also serialised behind one lock on the server — the card does one
     * thing at a time — so several requests in flight are several requests queued.
     */
    restyle: (photoId: string, body: ScenePhotoRestyleRequest, signal?: AbortSignal) =>
      request<ScenePhotoVariant>(`/scene-photos/${photoId}/restyle`, {
        method: "POST",
        body: json(body),
        signal,
        timeoutMs: EMPTY_ROOM_TIMEOUT_MS,
      }),

    empty: (photoId: string, body: ScenePhotoEmptyRequest = {}, signal?: AbortSignal) =>
      request<ScenePhotoVariant>(`/scene-photos/${photoId}/empty`, {
        method: "POST",
        body: json(body),
        signal,
        timeoutMs: EMPTY_ROOM_TIMEOUT_MS,
      }),
  },

  jobs: {
    get: (id: string) => request<Job>(`/jobs/${id}`),
    cancel: (id: string) => request<Job>(`/jobs/${id}/cancel`, { method: "POST" }),

    /**
     * Live progress over SSE. Returns an unsubscribe function.
     *
     * EventSource reconnects on its own, but it cannot report a terminal server error, so
     * a poll fallback kicks in if the stream fails before a terminal event arrives.
     */
    subscribe(
      id: string,
      onEvent: (event: JobEvent) => void,
      onError?: (error: unknown) => void,
    ): () => void {
      let closed = false;
      let poll: ReturnType<typeof setInterval> | null = null;
      // EventSource CANNOT SET A HEADER, which is the one thing this API uses to
      // authenticate. `withCredentials` sends the session cookie instead, and that works
      // wherever the engine answers on the site's own origin. Where it does not — the split
      // :3000 / :8008 development setup — the stream is refused, `onerror` fires, and the
      // poll fallback below takes over with the header on every request. Live progress
      // degrades to a two-second poll there; it does not break.
      const source = new EventSource(`${V1}/jobs/${id}/events`, { withCredentials: true });

      const stop = () => {
        if (closed) return;
        closed = true;
        source.close();
        if (poll) clearInterval(poll);
      };

      source.onmessage = (e) => {
        if (!e.data) return;
        try {
          const parsed = JSON.parse(e.data) as JobEvent;
          onEvent(parsed);
          if (parsed.type === "done" || parsed.type === "error") stop();
        } catch (err) {
          onError?.(err);
        }
      };

      source.onerror = () => {
        if (closed) return;
        // Fall back to polling rather than leaving the UI frozen on a stale frame.
        if (!poll) {
          poll = setInterval(async () => {
            try {
              const job = await api.jobs.get(id);
              onEvent({ type: "progress", job });
              if (job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") {
                stop();
              }
            } catch (err) {
              onError?.(err);
            }
          }, 2000);
        }
      };

      return stop;
    },
  },

  hotspots: {
    create: (sceneId: string, body: CreateHotspotRequest) =>
      request<Hotspot>(`/scenes/${sceneId}/hotspots`, { method: "POST", body: json(body) }),
    update: (id: string, body: UpdateHotspotRequest) =>
      request<Hotspot>(`/hotspots/${id}`, { method: "PATCH", body: json(body) }),
    remove: (id: string) => request<void>(`/hotspots/${id}`, { method: "DELETE" }),
  },

  ai: {
    hideFurniture: (body: HideFurnitureRequest) =>
      request<HideFurnitureResponse>("/ai/hide-furniture", { method: "POST", body: json(body) }),
    measure: (body: MeasureRequest) =>
      request<MeasureResponse>("/ai/measure", { method: "POST", body: json(body) }),

    /**
     * Strip a scene back to its bare room.
     *
     * Seconds of OpenCV on a cold scene and milliseconds on a warm one — the engine caches
     * the result per (scene, strength) — so this gets its own longer deadline rather than
     * the 15 s every other JSON call lives under. A user who has pressed a button and is
     * watching a spinner will wait; a user watching a timeout will not press it again.
     */
    emptyRoom: (body: EmptyRoomRequest, signal?: AbortSignal) =>
      request<EmptyRoomResponse>("/ai/empty-room", {
        method: "POST",
        body: json(body),
        signal,
        timeoutMs: EMPTY_ROOM_TIMEOUT_MS,
      }),
  },

  /**
   * Virtual staging over an ordinary photograph.
   *
   * One style per call, deliberately — see the engine's `routes/restyle.py`. The rail fills in
   * as the answers arrive rather than showing nothing for a minute.
   */
  restyle: {
    styles: () => request<RestyleStylesResponse>("/ai/restyle/styles"),
    photo: (photoId: string) => request<RestylePhoto>(`/ai/restyle/photos/${photoId}`),

    upload(file: File, signal?: AbortSignal): Promise<RestylePhoto> {
      const form = new FormData();
      form.append("file", file, file.name);
      return request<RestylePhoto>("/ai/restyle/photos", { method: "POST", body: form, signal });
    },

    /**
     * Render one style. Gets the long deadline for the same reason `emptyRoom` does: this is a
     * computation, not a lookup, and a user watching a spinner will wait where a user watching
     * a timeout will not press the button again.
     *
     * The server serialises these behind one lock — a card does one pass at a time — so a rail
     * asking for five styles at once is five queued requests, and the last of them waits for
     * the four in front of it.
     */
    render: (body: RestyleRequest, signal?: AbortSignal) =>
      request<RestyleResponse>("/ai/restyle", {
        method: "POST",
        body: json(body),
        signal,
        timeoutMs: EMPTY_ROOM_TIMEOUT_MS,
      }),

    /**
     * The same thing over a whole 360 scene, and MUCH slower: a sphere is sixteen diffusion
     * passes rather than one, so the first ask for a style is minutes and everything after
     * it is off the engine's disk cache. It shares `EMPTY_ROOM_TIMEOUT_MS` because it shares
     * that endpoint's shape and its debt — both are computations wearing a request's clothes,
     * and both belong in the job queue the engine already has.
     */
    scene: (body: RestyleSceneRequest, signal?: AbortSignal) =>
      request<RestyleSceneResponse>("/ai/restyle/scene", {
        method: "POST",
        body: json(body),
        signal,
        timeoutMs: EMPTY_ROOM_TIMEOUT_MS,
      }),
  },

  /**
   * The property seen from above.
   *
   * Two calls with very different costs, and the split is the endpoint's design: `get` only
   * assembles outlines that are already on disk, so it is milliseconds and safe to call on
   * every player mount, while `build` measures the rooms that have not been measured — a
   * scene-parsing pass per room. Hence the long deadline on the second and none on the first.
   */
  plan: {
    get: (propertyId: string, signal?: AbortSignal) =>
      request<FloorPlan>(`/properties/${propertyId}/plan`, { signal }),
    build: (propertyId: string, body: BuildPlanRequest = {}, signal?: AbortSignal) =>
      request<FloorPlan>(`/properties/${propertyId}/plan/build`, {
        method: "POST",
        body: json(body),
        signal,
        timeoutMs: EMPTY_ROOM_TIMEOUT_MS,
      }),

    /**
     * Move rooms by hand. Answers with the WHOLE recomputed plan, not an acknowledgement.
     *
     * That is deliberate on the server's side and the client must honour it: a placement is
     * not local, because the layout is walked again from every placed room, so moving one
     * room can legitimately move the ones beyond it. Patching a local copy of the one room
     * that was dragged would draw a plan the server does not have.
     */
    place: (propertyId: string, body: UpdatePlanRequest, signal?: AbortSignal) =>
      request<FloorPlan>(`/properties/${propertyId}/plan`, { method: "PATCH", body: json(body), signal }),
  },

  /**
   * The listing's headline and description, written from the property.
   *
   * Same shape of split as `plan`, for the same reason: `get` reads the cached copy off the
   * server's disk and calls no model, so the panel can ask for it on mount; `generate` is a
   * round trip to a hosted model with room pictures attached, which is seconds and somebody's
   * money — hence the long deadline, and hence it only ever runs from a button.
   *
   * `generate` never fails for want of a model: with no key configured it answers with the
   * engine's deterministic copy and `model_backed: false`. Read `warnings` before publishing
   * — that is where an invented square metre shows up.
   */
  listing: {
    get: (propertyId: string, signal?: AbortSignal) =>
      request<ListingState>(`/properties/${propertyId}/listing`, { signal }),
    generate: (propertyId: string, body: GenerateListingRequest = {}, signal?: AbortSignal) =>
      request<ListingCopy>(`/properties/${propertyId}/listing`, {
        method: "POST",
        body: json(body),
        signal,
        timeoutMs: EMPTY_ROOM_TIMEOUT_MS,
      }),
  },

  public: {
    property: (token: string) => request<PropertyDetail>(`/public/properties/${token}`),
    plan: (token: string, signal?: AbortSignal) =>
      request<FloorPlan>(`/public/properties/${token}/plan`, { signal }),
    walkthrough: (token: string, signal?: AbortSignal) =>
      request<Walkthrough>(`/public/properties/${token}/walkthrough`, { signal }),
    /**
     * Floor distances, for a viewer on somebody else's listing page.
     *
     * The same arithmetic as `ai.measure` behind a different door. It lives under `/public`
     * because that prefix is the only one answering arbitrary origins, and it is allowed to:
     * measuring reads no pixels and starts no model, it intersects a marked ray with the
     * floor plane, so the cost is microseconds and there is nothing to leak that the
     * panorama itself does not already show. The engine still checks the scene belongs to
     * this token's property.
     */
    /**
     * A CORS simple request (no credentials, body labelled `text/plain`), like the investment
     * chat below: the player runs framed on listing portals, where a credentialed JSON POST is
     * preflighted, refused by the public policy (GET/HEAD only) and never sent. The engine
     * reads the body as JSON whatever its label says.
     */
    measure: (token: string, body: MeasureRequest, signal?: AbortSignal) =>
      publicPost<MeasureResponse>(`/public/properties/${token}/measure`, body, signal),

    /**
     * The investment card and whether to offer the chat. No model runs; the card is the
     * engine's own arithmetic on the agent's figures. `enabled: false` means the agent has not
     * opted in, and the player shows no segment at all.
     */
    advisor: (token: string, locale: "tr" | "en" = "tr", signal?: AbortSignal) =>
      publicGet<AdvisorPublicState>(`/public/properties/${token}/advisor?locale=${locale}`, signal),

    /**
     * Ask the investment assistant one question; the answer arrives as server-sent events and
     * `onEvent` sees each one. Resolves when the answer is complete, rejects with a
     * `BioSphereError` for a refusal at the door (rate limit, daily budget, expired
     * conversation) — those come back as ordinary JSON errors before any streaming starts.
     */
    advisorChat: (
      token: string,
      body: AdvisorChatRequest,
      onEvent: (event: AdvisorEvent) => void,
      signal?: AbortSignal,
    ) => streamAdvisor(`/public/properties/${token}/advisor/chat`, body, onEvent, signal),
  },

  /**
   * The agent's side of the investment assistant: the figures, the opt-in, what it has cost.
   * Reads and writes one property; never calls a model.
   */
  investment: {
    get: (propertyId: string) => request<InvestmentState>(`/properties/${propertyId}/investment`),
    update: (propertyId: string, body: InvestmentInputs) =>
      request<InvestmentState>(`/properties/${propertyId}/investment`, { method: "PUT", body: json(body) }),
  },
};

// ---------------------------------------------------------------------------------
// The investment assistant's public calls
// ---------------------------------------------------------------------------------
//
// NOT through `request()`, and the difference is the whole point. `request()` sends the
// session header and `credentials: "include"`; the public prefix answers ANY origin (the player
// is framed on listing portals) with `Access-Control-Allow-Origin: *`, and a browser refuses to
// hand a credentialed response with a wildcard origin to the page. So these calls carry no
// credentials and no custom headers, and the chat's POST sends its JSON as `text/plain` — a
// CORS "simple request" that needs no preflight, which the public policy (GET/HEAD only) would
// refuse. The engine parses the body as JSON whatever the label says.

async function publicFailure(res: Response): Promise<BioSphereError> {
  const payload = safeJson(await res.text().catch(() => ""));
  const detail =
    payload && typeof payload === "object" && "code" in payload ? (payload as ApiError) : FALLBACK_ERROR(res.status);
  return new BioSphereError(res.status, detail);
}

async function publicGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  const deadline = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(`${V1}${path}`, {
      credentials: "omit",
      cache: "no-store",
      signal: withDeadline(signal, deadline),
    });
  } catch (cause) {
    if (signal?.aborted) throw cause;
    throw new BioSphereError(0, {
      code: "unreachable",
      message: "The engine is not reachable.",
      message_tr: "Sunucuya ulaşılamadı.",
      hint: String(cause),
    });
  }
  if (!res.ok) throw await publicFailure(res);
  return (await res.json()) as T;
}

async function publicPost<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const deadline = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(`${V1}${path}`, {
      method: "POST",
      credentials: "omit",
      cache: "no-store",
      headers: { "Content-Type": "text/plain;charset=UTF-8" },
      body: JSON.stringify(body),
      signal: withDeadline(signal, deadline),
    });
  } catch (cause) {
    if (signal?.aborted) throw cause;
    throw new BioSphereError(0, {
      code: "unreachable",
      message: "The engine is not reachable.",
      message_tr: "Sunucuya ulaşılamadı.",
      hint: String(cause),
    });
  }
  if (!res.ok) throw await publicFailure(res);
  return (await res.json()) as T;
}

/**
 * Split a server-sent-event body into its `data:` payloads. Exported for the tests: frames
 * arrive cut at arbitrary byte boundaries, and a JSON object split across two network chunks
 * must still be read whole.
 */
export function sseFrames(buffer: string): { frames: string[]; rest: string } {
  const frames: string[] = [];
  let rest = buffer.replace(/\r\n/g, "\n");
  let cut = rest.indexOf("\n\n");
  while (cut !== -1) {
    const block = rest.slice(0, cut);
    rest = rest.slice(cut + 2);
    const data = block
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).replace(/^ /, ""))
      .join("\n");
    if (data) frames.push(data);
    cut = rest.indexOf("\n\n");
  }
  return { frames, rest };
}

async function streamAdvisor(
  path: string,
  body: AdvisorChatRequest,
  onEvent: (event: AdvisorEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${V1}${path}`, {
      method: "POST",
      credentials: "omit",
      cache: "no-store",
      headers: { "Content-Type": "text/plain;charset=UTF-8" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (cause) {
    if (signal?.aborted) throw cause;
    throw new BioSphereError(0, {
      code: "network",
      message: "The assistant could not be reached.",
      message_tr: "Asistana ulaşılamadı. Bağlantınızı kontrol edip tekrar deneyin.",
      hint: String(cause),
    });
  }
  if (!res.ok || !res.body) throw await publicFailure(res);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { frames, rest } = sseFrames(buffer);
    buffer = rest;
    for (const frame of frames) {
      const parsed = safeJson(frame);
      if (parsed && typeof parsed === "object" && "type" in parsed) onEvent(parsed as AdvisorEvent);
    }
  }
}

/** Server-component friendly fetch that tolerates a cold backend. */
export async function safeGetProperty(id: string): Promise<PropertyDetail | null> {
  try {
    return await api.properties.get(id);
  } catch {
    return null;
  }
}
