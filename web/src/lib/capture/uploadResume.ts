/**
 * Resuming a chunked capture upload after it fails, without re-sending what the server has.
 * =====================================================================================
 *
 * THE SERVER ALREADY PROMISED THIS; THE CLIENT NEVER TOOK IT UP. A large capture goes up as
 * numbered 8 MB parts (`api.ts`, `uploadInParts`) under an upload id, and the engine keeps
 * every part it received for 24 hours and leaves them in place when the final assembly call
 * fails (`routes/properties.py`: "a user who lost the link mid-transfer can finish from the
 * same page after a break"). But the client minted a NEW upload id on every attempt, so the
 * retry button re-sent the whole capture: on the reference capture that is 179 MB and 23
 * parts, over a phone's uplink, because part 19 dropped once.
 *
 * Parts are addressed by index and written idempotently, so resuming needs only two facts
 * kept across attempts: the upload id, and which parts the server answered 2xx for. This
 * module keeps them, and refuses to reuse them for a DIFFERENT set of files (a re-shoot is a
 * new capture, and a stale part with a matching index would otherwise be assembled into it).
 *
 * Pure, no DOM, so it is tested in plain Node with the reference capture's real sizes.
 */

export interface UploadFileShape {
  readonly name: string;
  readonly size: number;
}

/** One part: which file, which byte range, and its global index in the upload. */
export interface PartSlice {
  index: number;
  fileIndex: number;
  start: number;
  end: number;
}

/**
 * The part layout `uploadInParts` uses: each file cut into `partBytes` slices, numbered
 * consecutively across the files in order. An empty file still gets its one (empty) part,
 * because the assembly step expects every file to name at least one.
 */
export function planParts(files: readonly UploadFileShape[], partBytes: number): PartSlice[] {
  if (!(partBytes > 0)) throw new Error("partBytes must be positive");
  const out: PartSlice[] = [];
  let index = 0;
  files.forEach((file, fileIndex) => {
    for (let offset = 0; offset < Math.max(file.size, 1); offset += partBytes) {
      out.push({ index, fileIndex, start: offset, end: Math.min(offset + partBytes, file.size) });
      index += 1;
    }
  });
  return out;
}

/** Identity of a set of files for resume purposes: names, sizes and order. */
export function filesFingerprint(files: readonly UploadFileShape[]): string {
  return files.map((f) => `${f.name}:${f.size}`).join("|");
}

/**
 * Pause before retrying a part that failed: nothing, then 1 s, then 3 s.
 *
 * Three back-to-back attempts on a flaky mobile link mostly fail together -- the link that
 * dropped the first one has not come back 20 ms later -- and that spent the whole retry
 * budget of a part in one blip. Four seconds of patience in total is invisible next to a
 * 179 MB upload and rides out a cell handover.
 */
export const PART_RETRY_DELAYS_MS: readonly number[] = [0, 1_000, 3_000];

/** Server error codes that mean the staged parts are gone and a resume cannot work. */
export const STAGING_LOST_CODES: readonly string[] = ["upload_not_found", "missing_part"];

export interface ResumeAttempt {
  uploadId: string;
  /** Part indices the server already acknowledged; the attempt must not send them again. */
  skip: ReadonlySet<number>;
}

export class UploadResume {
  private id: string | null = null;
  private fingerprint: string | null = null;
  private readonly acked = new Set<number>();

  /**
   * Start (or continue) an attempt for `files`. The same files continue the same upload;
   * different files -- a re-shoot, another room -- start a new one with a fresh id.
   */
  begin(files: readonly UploadFileShape[], newId: () => string): ResumeAttempt {
    const fp = filesFingerprint(files);
    if (this.id === null || this.fingerprint !== fp) {
      this.id = newId();
      this.fingerprint = fp;
      this.acked.clear();
    }
    return { uploadId: this.id, skip: new Set(this.acked) };
  }

  /** The server answered 2xx for this part. */
  ack(index: number): void {
    this.acked.add(index);
  }

  /** The staged parts no longer exist (assembled, expired or swept): start over next time. */
  reset(): void {
    this.id = null;
    this.fingerprint = null;
    this.acked.clear();
  }

  get uploadId(): string | null {
    return this.id;
  }

  get ackedCount(): number {
    return this.acked.size;
  }
}
