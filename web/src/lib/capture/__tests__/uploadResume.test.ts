import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PART_RETRY_DELAYS_MS, UploadResume, filesFingerprint, planParts } from "@/lib/capture/uploadResume";

/**
 * A real-sized capture: two VP9 segments and the sidecar, as the
 * phone handed them to the uploader. 179 MB -> 23 parts of 8 MB.
 */
const PART = 8 * 1024 * 1024;
const REFERENCE = [
  { name: "capture-1.webm", size: 107_988_105 },
  { name: "capture-2.webm", size: 71_125_857 },
  { name: "motion.json", size: 49_241 },
];

describe("planParts", () => {
  it("cuts the reference capture into 23 contiguous parts covering every byte once", () => {
    const parts = planParts(REFERENCE, PART);
    expect(parts).toHaveLength(13 + 9 + 1);
    expect(parts.map((p) => p.index)).toEqual(parts.map((_, i) => i));
    for (let f = 0; f < REFERENCE.length; f += 1) {
      const mine = parts.filter((p) => p.fileIndex === f);
      expect(mine[0].start).toBe(0);
      expect(mine[mine.length - 1].end).toBe(REFERENCE[f].size);
      for (let k = 1; k < mine.length; k += 1) expect(mine[k].start).toBe(mine[k - 1].end);
      for (const p of mine.slice(0, -1)) expect(p.end - p.start).toBe(PART);
    }
  });

  it("gives an empty file its one part, because assembly needs every file to name one", () => {
    expect(planParts([{ name: "a", size: 0 }], PART)).toEqual([{ index: 0, fileIndex: 0, start: 0, end: 0 }]);
  });
});

describe("UploadResume", () => {
  let n = 0;
  const newId = () => `upload-${(n += 1)}`;

  it("continues the same upload, skipping what the server acknowledged", () => {
    const r = new UploadResume();
    const first = r.begin(REFERENCE, newId);
    expect(first.skip.size).toBe(0);
    for (let i = 0; i < 19; i += 1) r.ack(i); // part 19 dropped
    const second = r.begin(REFERENCE, newId);
    expect(second.uploadId).toBe(first.uploadId);
    expect([...second.skip].sort((a, b) => a - b)).toEqual([...Array(19).keys()]);
    // 4 of 23 parts left to send: 179 MB became 21 MB.
    const left = planParts(REFERENCE, PART).filter((p) => !second.skip.has(p.index));
    expect(left.map((p) => p.index)).toEqual([19, 20, 21, 22]);
    const bytesLeft = left.reduce((s, p) => s + p.end - p.start, 0);
    expect(bytesLeft).toBe(71_125_857 - 6 * PART + 49_241);
    expect(bytesLeft / REFERENCE.reduce((s, f) => s + f.size, 0)).toBeLessThan(0.12);
  });

  it("starts over for different files: a re-shoot must never inherit stale parts", () => {
    const r = new UploadResume();
    const first = r.begin(REFERENCE, newId);
    r.ack(0);
    const reshoot = [{ name: "capture-1.webm", size: 99_000_000 }, REFERENCE[2]];
    const second = r.begin(reshoot, newId);
    expect(second.uploadId).not.toBe(first.uploadId);
    expect(second.skip.size).toBe(0);
    expect(filesFingerprint(reshoot)).not.toBe(filesFingerprint(REFERENCE));
  });

  it("forgets everything once the server no longer holds the parts", () => {
    const r = new UploadResume();
    const first = r.begin(REFERENCE, newId);
    r.ack(3);
    r.reset();
    expect(r.ackedCount).toBe(0);
    expect(r.begin(REFERENCE, newId).uploadId).not.toBe(first.uploadId);
  });

  it("backs off between part retries instead of burning all three in one blip", () => {
    expect(PART_RETRY_DELAYS_MS[0]).toBe(0);
    expect(PART_RETRY_DELAYS_MS.slice(1).every((ms, i) => ms > PART_RETRY_DELAYS_MS[i])).toBe(true);
    expect(PART_RETRY_DELAYS_MS.reduce((a, b) => a + b, 0)).toBeLessThanOrEqual(5_000);
  });
});

// -------------------------------------------------------------------------------------
// The real uploader, against a fake network: XHR for the parts, fetch for the assembly.
// -------------------------------------------------------------------------------------

interface Put {
  uploadId: string;
  index: number;
  bytes: number;
}

function installFakeNetwork(failIndex: { index: number; times: number } | null, finalise: () => Response) {
  const puts: Put[] = [];
  let failuresLeft = failIndex?.times ?? 0;
  class FakeXHR {
    status = 0;
    upload: { onprogress: ((e: { loaded: number }) => void) | null; onerror: (() => void) | null } = {
      onprogress: null,
      onerror: null,
    };
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    onabort: (() => void) | null = null;
    withCredentials = false;
    private url = "";
    open(_method: string, url: string) {
      this.url = url;
    }
    setRequestHeader() {}
    abort() {
      this.onabort?.();
    }
    send(body: Blob) {
      const m = /uploads\/([^/]+)\/parts\/(\d+)$/.exec(this.url);
      const uploadId = m ? m[1] : "?";
      const index = m ? Number(m[2]) : -1;
      queueMicrotask(() => {
        if (failIndex && index === failIndex.index && failuresLeft > 0) {
          failuresLeft -= 1;
          this.onerror?.();
          return;
        }
        puts.push({ uploadId, index, bytes: body.size });
        this.status = 200;
        this.upload.onprogress?.({ loaded: body.size });
        this.onload?.();
      });
    }
  }
  vi.stubGlobal("XMLHttpRequest", FakeXHR);
  const finaliseBodies: { upload_id: string; files: { filename: string; parts: number[] }[] }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_url: string, init: RequestInit) => {
      finaliseBodies.push(JSON.parse(String(init.body)));
      return finalise();
    }),
  );
  return { puts, finaliseBodies };
}

const created = () =>
  new Response(JSON.stringify({ scene: { id: "s1" }, job: { id: "j1" } }), {
    status: 201,
    headers: { "Content-Type": "application/json" },
  });

describe("api.scenes.create with a resume", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  async function settle<T>(p: Promise<T>): Promise<T> {
    let done = false;
    const out = p.finally(() => {
      done = true;
    });
    // Handled here so the rejection is not "unhandled" while the clock is being driven; the
    // caller still receives `out` itself and sees it reject.
    out.catch(() => undefined);
    while (!done) await vi.advanceTimersByTimeAsync(1_000);
    return out;
  }

  it("retries a dropped part with backoff, then resumes the same upload on the user's retry", async () => {
    // Above the 32 MB single-POST ceiling, so this takes the parts path: 40 MB -> 5 parts,
    // the sidecar -> 1.
    const big = [new File([new Uint8Array(40 * 1024 * 1024)], "capture-1.webm"), new File([new Uint8Array(49_241)], "motion.json")];
    const net = installFakeNetwork({ index: 2, times: 3 }, created);
    const { api } = await import("@/lib/api");
    const resume = new UploadResume();
    const meta = { name: "Salon", quality_mode: "balanced", capture_kind: "video" } as const;

    // Attempt 1: part 2 fails all three tries, and the upload rejects.
    await expect(settle(api.scenes.create("p1", big, meta, undefined, undefined, resume))).rejects.toBeTruthy();
    expect(net.puts.map((p) => p.index)).toEqual([0, 1]);
    expect(resume.ackedCount).toBe(2);
    const firstId = net.puts[0].uploadId;

    // Attempt 2 (the retry button): same upload id, parts 0-1 are NOT sent again.
    const result = await settle(api.scenes.create("p1", big, meta, undefined, undefined, resume));
    expect(result.scene.id).toBe("s1");
    const second = net.puts.slice(2);
    expect(second.every((p) => p.uploadId === firstId)).toBe(true);
    expect(second.map((p) => p.index)).toEqual([2, 3, 4, 5]);
    // The assembly still names every part, in order.
    expect(net.finaliseBodies[0].files.map((f) => f.parts)).toEqual([[0, 1, 2, 3, 4], [5]]);
    // Assembled: nothing left to resume.
    expect(resume.ackedCount).toBe(0);
    expect(resume.uploadId).toBeNull();
  });

  it("starts over by itself when the server has lost the staged parts", async () => {
    const big = [new File([new Uint8Array(33 * 1024 * 1024)], "capture-1.webm")];
    let calls = 0;
    const net = installFakeNetwork(null, () => {
      calls += 1;
      return calls === 1
        ? new Response(JSON.stringify({ code: "upload_not_found", message: "gone", message_tr: "yok" }), { status: 404 })
        : created();
    });
    const { api } = await import("@/lib/api");
    const resume = new UploadResume();
    const meta = { name: "Salon", quality_mode: "balanced", capture_kind: "video" } as const;
    // Pretend an earlier attempt sent everything and the parts then expired on the server.
    const stale = resume.begin(big, () => "stale-upload-id");
    for (let i = 0; i < 5; i += 1) resume.ack(i);
    expect(stale.uploadId).toBe("stale-upload-id");

    const result = await settle(api.scenes.create("p1", big, meta, undefined, undefined, resume));
    expect(result.scene.id).toBe("s1");
    // First assembly named the stale id and failed; the automatic restart re-sent all five
    // parts under a fresh id and assembled those.
    expect(net.finaliseBodies.map((b) => b.upload_id)[0]).toBe("stale-upload-id");
    expect(net.puts.map((p) => p.index)).toEqual([0, 1, 2, 3, 4]);
    expect(net.puts.every((p) => p.uploadId !== "stale-upload-id")).toBe(true);
    expect(net.finaliseBodies[1].upload_id).toBe(net.puts[0].uploadId);
  });

  it("does not resume across a different capture", async () => {
    installFakeNetwork(null, created);
    const r = new UploadResume();
    const a = r.begin([{ name: "capture-1.webm", size: 40 * 1024 * 1024 }], () => "a");
    r.ack(0);
    const b = r.begin([{ name: "capture-1.webm", size: 41 * 1024 * 1024 }], () => "b");
    expect(a.uploadId).toBe("a");
    expect(b.uploadId).toBe("b");
    expect(b.skip.size).toBe(0);
  });
});
