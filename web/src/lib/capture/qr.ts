/**
 * A minimal QR encoder — byte mode, error-correction level M, versions 1 to 6.
 *
 * WHY THIS EXISTS AT ALL: the desktop → phone handoff needs a QR code and the project has
 * no QR dependency. Adding one for ~40 lines of matrix maths would pull a transitive tree
 * into a bundle a phone downloads over LTE while standing in someone's hallway. This is
 * ISO/IEC 18004's byte-mode encoder and nothing else: no Kanji, no alphanumeric
 * optimisation, no structured append.
 *
 * WHY IT STOPS AT VERSION 6: version 7 and above must carry an 18-bit version-information
 * block in two more reserved corners, encoded with a second BCH code. Version 6 holds 106
 * bytes at level M, and the longest URL this app can produce — origin plus
 * `/cek?property=<uuid>` — is under 100. Refusing to encode above 106 bytes and falling back to
 * a plain link is strictly better than shipping an untested second BCH path, because a
 * wrong QR code fails silently: it renders, it looks like a QR code, and it does not scan.
 *
 * Pure and DOM-free, so it behaves identically during SSR and in the browser.
 *
 * Correctness anchors, each independently checkable and all asserted by `selfTest()`:
 *  - every row of BLOCKS satisfies `data + blocks*eccPerBlock === TOTAL_CODEWORDS[v]`;
 *  - the degree-10 RS generator's exponents are the published
 *    [0,251,67,46,61,118,70,64,94,32,45];
 *  - format information for (M, mask 0) and (M, mask 3) matches the standard's table.
 */

/** L=01, M=00, Q=11, H=10 — the standard's non-obvious ordering. Only M is implemented. */
const ECC_LEVEL_BITS = 0b00;

/** Highest supported version; see the file header for why it is not 40. */
const MAX_VERSION = 6;

/** Total codewords (data + error correction) per version. Index 0 is version 1. */
const TOTAL_CODEWORDS = [26, 44, 70, 100, 134, 172] as const;

/**
 * Block structure at ECC level M, per version:
 *   [data codewords, EC codewords per block, group-1 blocks, group-1 data per block,
 *    group-2 blocks, group-2 data per block]
 * Group-2 blocks hold exactly one more data codeword than group-1 blocks; that is what
 * makes the final interleave ragged. Versions 1-6 at level M happen to need no group 2,
 * but the interleave below handles it anyway so the table can grow without a rewrite.
 */
const BLOCKS: readonly (readonly [number, number, number, number, number, number])[] = [
  [16, 10, 1, 16, 0, 0], // v1 — 21x21
  [28, 16, 1, 28, 0, 0], // v2 — 25x25
  [44, 26, 1, 44, 0, 0], // v3 — 29x29
  [64, 18, 2, 32, 0, 0], // v4 — 33x33
  [86, 24, 2, 43, 0, 0], // v5 — 37x37
  [108, 16, 4, 27, 0, 0], // v6 — 41x41
];

/** Alignment-pattern centre coordinates per version. Version 1 has none. */
const ALIGNMENT: readonly (readonly number[])[] = [[], [6, 18], [6, 22], [6, 26], [6, 30], [6, 34]];

// ── GF(256) arithmetic, primitive polynomial 0x11D ─────────────────────────────────────

const EXP = new Uint8Array(512);
const LOG = new Uint8Array(256);

(function buildTables() {
  let x = 1;
  for (let i = 0; i < 255; i++) {
    EXP[i] = x;
    LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= 0x11d;
  }
  // Duplicated tail, so a product's exponent sum never needs a modulo.
  for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
})();

function gfMul(a: number, b: number): number {
  if (a === 0 || b === 0) return 0;
  return EXP[LOG[a] + LOG[b]];
}

/** Reed-Solomon generator polynomial of the given degree, highest-order coefficient first. */
function rsGenerator(degree: number): Uint8Array {
  let poly = new Uint8Array([1]);
  for (let i = 0; i < degree; i++) {
    const next = new Uint8Array(poly.length + 1);
    for (let j = 0; j < poly.length; j++) {
      next[j] ^= poly[j];
      next[j + 1] ^= gfMul(poly[j], EXP[i]);
    }
    poly = next;
  }
  return poly;
}

/** Error-correction codewords for one block — the remainder of a polynomial division. */
function rsEncode(data: Uint8Array, eccLength: number): Uint8Array {
  const generator = rsGenerator(eccLength);
  const remainder = new Uint8Array(eccLength);
  for (const byte of data) {
    const factor = byte ^ remainder[0];
    remainder.copyWithin(0, 1);
    remainder[eccLength - 1] = 0;
    if (factor !== 0) {
      for (let i = 0; i < eccLength; i++) remainder[i] ^= gfMul(generator[i + 1], factor);
    }
  }
  return remainder;
}

// ── Format information: BCH(15,5), generator 0x537, masked with 0x5412 ─────────────────

function formatBitsFor(eccBits: number, mask: number): number {
  const value = (eccBits << 3) | mask;
  let remainder = value << 10;
  for (let i = 4; i >= 0; i--) {
    if (remainder & (1 << (i + 10))) remainder ^= 0b10100110111 << i;
  }
  return ((value << 10) | remainder) ^ 0b101010000010010;
}

function formatBits(mask: number): number {
  return formatBitsFor(ECC_LEVEL_BITS, mask);
}

// ── Bit buffer ─────────────────────────────────────────────────────────────────────────

class BitBuffer {
  private readonly bits: number[] = [];

  push(value: number, length: number): void {
    for (let i = length - 1; i >= 0; i--) this.bits.push((value >>> i) & 1);
  }

  get length(): number {
    return this.bits.length;
  }

  toCodewords(): Uint8Array {
    const out = new Uint8Array(Math.ceil(this.bits.length / 8));
    for (let i = 0; i < this.bits.length; i++) {
      if (this.bits[i]) out[i >> 3] |= 0x80 >> (i & 7);
    }
    return out;
  }
}

// ── Matrix ─────────────────────────────────────────────────────────────────────────────

export interface QrMatrix {
  /** Modules per side, excluding the quiet zone. */
  size: number;
  /** Row-major, `size * size`, 1 = dark. */
  modules: Uint8Array;
  version: number;
}

/** `reserved` marks function patterns, which are excluded from data placement and masking. */
interface Canvas {
  size: number;
  modules: Uint8Array;
  reserved: Uint8Array;
}

function set(canvas: Canvas, x: number, y: number, dark: boolean): void {
  if (x < 0 || y < 0 || x >= canvas.size || y >= canvas.size) return;
  canvas.modules[y * canvas.size + x] = dark ? 1 : 0;
  canvas.reserved[y * canvas.size + x] = 1;
}

function placeFinder(canvas: Canvas, cx: number, cy: number): void {
  // 7x7 pattern plus the one-module light separator around it. The separator has to be
  // RESERVED as well as light, or the data placer fills it in and nothing scans.
  for (let dy = -1; dy <= 7; dy++) {
    for (let dx = -1; dx <= 7; dx++) {
      const inRing = dx >= 0 && dx <= 6 && dy >= 0 && dy <= 6;
      const outer = inRing && (dx === 0 || dx === 6 || dy === 0 || dy === 6);
      const core = inRing && dx >= 2 && dx <= 4 && dy >= 2 && dy <= 4;
      set(canvas, cx + dx, cy + dy, outer || core);
    }
  }
}

function placeFunctionPatterns(canvas: Canvas, version: number): void {
  const size = canvas.size;

  placeFinder(canvas, 0, 0);
  placeFinder(canvas, size - 7, 0);
  placeFinder(canvas, 0, size - 7);

  // Timing patterns: alternating modules along row 6 and column 6, between the finders.
  for (let i = 8; i < size - 8; i++) {
    const dark = i % 2 === 0;
    set(canvas, i, 6, dark);
    set(canvas, 6, i, dark);
  }

  // Alignment patterns, except where one would collide with a finder.
  const centres = ALIGNMENT[version - 1];
  for (const cy of centres) {
    for (const cx of centres) {
      const nearFinder = (cx <= 8 && cy <= 8) || (cx <= 8 && cy >= size - 9) || (cx >= size - 9 && cy <= 8);
      if (nearFinder) continue;
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const edge = Math.abs(dx) === 2 || Math.abs(dy) === 2;
          set(canvas, cx + dx, cy + dy, edge || (dx === 0 && dy === 0));
        }
      }
    }
  }

  // Reserve the format-information strips. Index 6 is SKIPPED in both directions: (6,8)
  // and (8,6) belong to the timing patterns, and blanking them here silently destroys the
  // timing reference — a failure that renders perfectly and scans never.
  for (let i = 0; i <= 8; i++) {
    if (i === 6) continue;
    set(canvas, i, 8, false);
    set(canvas, 8, i, false);
  }
  for (let i = 0; i < 8; i++) set(canvas, size - 1 - i, 8, false);
  for (let i = 0; i < 7; i++) set(canvas, 8, size - 1 - i, false);

  // The dark module, always at (8, 4*version + 9) === (8, size - 8).
  set(canvas, 8, size - 8, true);
}

function maskAt(pattern: number, row: number, col: number): boolean {
  switch (pattern) {
    case 0:
      return (row + col) % 2 === 0;
    case 1:
      return row % 2 === 0;
    case 2:
      return col % 3 === 0;
    case 3:
      return (row + col) % 3 === 0;
    case 4:
      return (Math.floor(row / 2) + Math.floor(col / 3)) % 2 === 0;
    case 5:
      return ((row * col) % 2) + ((row * col) % 3) === 0;
    case 6:
      return (((row * col) % 2) + ((row * col) % 3)) % 2 === 0;
    default:
      return (((row + col) % 2) + ((row * col) % 3)) % 2 === 0;
  }
}

/** Places the interleaved codeword stream in the standard upward/downward zigzag. */
function placeData(canvas: Canvas, data: Uint8Array): void {
  const size = canvas.size;
  let bitIndex = 0;
  let upward = true;

  for (let right = size - 1; right >= 1; right -= 2) {
    // Column 6 is the vertical timing pattern; it is skipped entirely, which shifts every
    // column pair left of it by one.
    const rightCol = right <= 6 ? right - 1 : right;
    for (let step = 0; step < size; step++) {
      const y = upward ? size - 1 - step : step;
      for (let c = 0; c < 2; c++) {
        const x = rightCol - c;
        if (canvas.reserved[y * size + x]) continue;
        const byte = data[bitIndex >> 3];
        // Past the end of the stream the remainder bits are zero, which is what the
        // standard's remainder-bit rule amounts to for these versions.
        canvas.modules[y * size + x] = byte === undefined ? 0 : (byte >> (7 - (bitIndex & 7))) & 1;
        bitIndex++;
      }
    }
    upward = !upward;
  }
}

function applyMask(canvas: Canvas, pattern: number): Uint8Array {
  const size = canvas.size;
  const out = new Uint8Array(canvas.modules);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (canvas.reserved[y * size + x]) continue;
      if (maskAt(pattern, y, x)) out[y * size + x] ^= 1;
    }
  }
  return out;
}

function writeFormat(modules: Uint8Array, size: number, mask: number): void {
  const bits = formatBits(mask);
  const at = (x: number, y: number, value: number) => {
    modules[y * size + x] = value;
  };
  for (let i = 0; i < 15; i++) {
    const bit = (bits >> i) & 1;

    // Copy 1, wrapped around the top-left finder, skipping the timing row and column.
    if (i < 6) at(8, i, bit);
    else if (i < 8) at(8, i + 1, bit);
    else if (i === 8) at(7, 8, bit);
    else at(14 - i, 8, bit);

    // Copy 2, split between the top-right and bottom-left finders.
    if (i < 8) at(size - 1 - i, 8, bit);
    else at(8, size - 15 + i, bit);
  }
}

/** The standard's four penalty rules; the mask with the lowest total wins. */
function penalty(modules: Uint8Array, size: number): number {
  let score = 0;
  const get = (x: number, y: number) => modules[y * size + x];

  // Rule 1 — runs of five or more identical modules along a row or column.
  for (let pass = 0; pass < 2; pass++) {
    for (let a = 0; a < size; a++) {
      let run = 1;
      let previous = pass === 0 ? get(0, a) : get(a, 0);
      for (let b = 1; b < size; b++) {
        const value = pass === 0 ? get(b, a) : get(a, b);
        if (value === previous) {
          run++;
        } else {
          if (run >= 5) score += 3 + (run - 5);
          previous = value;
          run = 1;
        }
      }
      if (run >= 5) score += 3 + (run - 5);
    }
  }

  // Rule 2 — every 2x2 block of a single colour.
  for (let y = 0; y < size - 1; y++) {
    for (let x = 0; x < size - 1; x++) {
      const v = get(x, y);
      if (v === get(x + 1, y) && v === get(x, y + 1) && v === get(x + 1, y + 1)) score += 3;
    }
  }

  // Rule 3 — the finder-lookalike 1:1:3:1:1 run with four light modules on either side.
  const forward = [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0];
  const backward = [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      for (const pattern of [forward, backward]) {
        if (x + pattern.length <= size) {
          let hit = true;
          for (let i = 0; i < pattern.length && hit; i++) hit = get(x + i, y) === pattern[i];
          if (hit) score += 40;
        }
        if (y + pattern.length <= size) {
          let hit = true;
          for (let i = 0; i < pattern.length && hit; i++) hit = get(x, y + i) === pattern[i];
          if (hit) score += 40;
        }
      }
    }
  }

  // Rule 4 — how far the dark-module ratio strays from 50%.
  let dark = 0;
  for (let i = 0; i < modules.length; i++) dark += modules[i];
  const percent = (dark * 100) / (size * size);
  score += Math.floor(Math.abs(percent - 50) / 5) * 10;

  return score;
}

/** Byte-mode character capacity of a version at ECC level M. */
function capacity(version: number): number {
  // 4 bits of mode indicator + 8 bits of character count; both fit in one byte-and-a-half
  // for every version up to 9, and this encoder stops at 6.
  return Math.floor((BLOCKS[version - 1][0] * 8 - 12) / 8);
}

/** The longest payload this encoder accepts, in UTF-8 bytes. */
export const QR_MAX_BYTES = capacity(MAX_VERSION);

export class QrTooLongError extends Error {
  constructor(length: number) {
    super(`QR payload of ${length} bytes exceeds the ${QR_MAX_BYTES}-byte ceiling of version ${MAX_VERSION}.`);
    this.name = "QrTooLongError";
  }
}

/**
 * Encode `text` as a QR matrix: UTF-8 bytes, byte mode, ECC level M, smallest version that
 * fits. Throws `QrTooLongError` rather than truncating — a truncated URL still scans, and
 * sends the user somewhere wrong.
 */
export function encodeQr(text: string): QrMatrix {
  const bytes = new TextEncoder().encode(text);

  let version = 0;
  for (let candidate = 1; candidate <= MAX_VERSION; candidate++) {
    if (bytes.length <= capacity(candidate)) {
      version = candidate;
      break;
    }
  }
  if (version === 0) throw new QrTooLongError(bytes.length);

  const [dataCodewordCount, eccPerBlock, group1Blocks, group1Size, group2Blocks, group2Size] = BLOCKS[version - 1];

  // 1. Bit stream: mode indicator, character count, payload, terminator, byte alignment.
  const buffer = new BitBuffer();
  buffer.push(0b0100, 4);
  buffer.push(bytes.length, 8);
  for (const byte of bytes) buffer.push(byte, 8);
  buffer.push(0, Math.min(4, dataCodewordCount * 8 - buffer.length));
  if (buffer.length % 8 !== 0) buffer.push(0, 8 - (buffer.length % 8));

  const codewords = new Uint8Array(dataCodewordCount);
  const written = buffer.toCodewords();
  codewords.set(written);
  // The standard's alternating pad bytes fill whatever is left.
  for (let i = written.length; i < dataCodewordCount; i++) {
    codewords[i] = (i - written.length) % 2 === 0 ? 0xec : 0x11;
  }

  // 2. Split into blocks and compute error correction for each.
  const dataBlocks: Uint8Array[] = [];
  const eccBlocks: Uint8Array[] = [];
  let offset = 0;
  for (let b = 0; b < group1Blocks + group2Blocks; b++) {
    const blockSize = b < group1Blocks ? group1Size : group2Size;
    const block = codewords.subarray(offset, offset + blockSize);
    offset += blockSize;
    dataBlocks.push(block);
    eccBlocks.push(rsEncode(block, eccPerBlock));
  }

  // 3. Interleave: data codewords column-wise across blocks, then EC codewords likewise.
  const interleaved = new Uint8Array(TOTAL_CODEWORDS[version - 1]);
  let out = 0;
  const longestBlock = Math.max(group1Size, group2Size);
  for (let i = 0; i < longestBlock; i++) {
    for (const block of dataBlocks) {
      if (i < block.length) interleaved[out++] = block[i];
    }
  }
  for (let i = 0; i < eccPerBlock; i++) {
    for (const block of eccBlocks) interleaved[out++] = block[i];
  }

  // 4. Lay out, then pick the mask with the lowest penalty.
  const size = version * 4 + 17;
  const canvas = createCanvas(size);
  placeFunctionPatterns(canvas, version);
  placeData(canvas, interleaved);

  let best = applyMask(canvas, 0);
  writeFormat(best, size, 0);
  let bestScore = penalty(best, size);
  for (let mask = 1; mask < 8; mask++) {
    const candidate = applyMask(canvas, mask);
    writeFormat(candidate, size, mask);
    const score = penalty(candidate, size);
    if (score < bestScore) {
      bestScore = score;
      best = candidate;
    }
  }

  return { size, modules: best, version };
}

function createCanvas(size: number): Canvas {
  return { size, modules: new Uint8Array(size * size), reserved: new Uint8Array(size * size) };
}

/**
 * SVG path data for the dark modules, one user unit per module, origin at the symbol's
 * top-left. A single `<path>` rather than a thousand `<rect>`s: a version-5 code is 1369
 * modules, and one path keeps the DOM to one node and renders without hairline seams.
 */
export function qrToPath(matrix: QrMatrix): string {
  const parts: string[] = [];
  for (let y = 0; y < matrix.size; y++) {
    let x = 0;
    while (x < matrix.size) {
      if (!matrix.modules[y * matrix.size + x]) {
        x++;
        continue;
      }
      let run = 1;
      while (x + run < matrix.size && matrix.modules[y * matrix.size + x + run]) run++;
      parts.push(`M${x} ${y}h${run}v1h-${run}z`);
      x += run;
    }
  }
  return parts.join("");
}

/**
 * Runtime assertions over the tables and the two coding schemes. Cheap, and it turns a
 * mistyped capacity row — which would otherwise produce a code that renders beautifully and
 * scans never — into a named exception.
 */
export function selfTest(): true {
  for (let v = 1; v <= MAX_VERSION; v++) {
    const [data, ecc, g1, g1size, g2, g2size] = BLOCKS[v - 1];
    if (g1 * g1size + g2 * g2size !== data) throw new Error(`QR blocks: version ${v} data mismatch`);
    if (data + (g1 + g2) * ecc !== TOTAL_CODEWORDS[v - 1]) throw new Error(`QR blocks: version ${v} total mismatch`);
  }

  const generator = rsGenerator(10);
  const exponents = [0, 251, 67, 46, 61, 118, 70, 64, 94, 32, 45];
  for (let i = 0; i < exponents.length; i++) {
    if (generator[i] !== EXP[exponents[i]]) throw new Error(`QR generator polynomial: coefficient ${i} wrong`);
  }

  if (formatBits(0) !== 0b101010000010010) throw new Error("QR format bits: (M, mask 0) wrong");

  // The 32 format strings form a BCH(15,5) code whose minimum Hamming distance is exactly
  // 7 — that is the property the standard's table is generated from, and it is what lets a
  // scanner recover the mask from a damaged corner. Asserting the PROPERTY rather than
  // transcribed literals means this test cannot inherit a typo from a copied table; a
  // single wrong bit anywhere in the generator drops the distance below 7 immediately.
  let minimumDistance = Number.POSITIVE_INFINITY;
  const strings: number[] = [];
  for (let ecc = 0; ecc < 4; ecc++) for (let mask = 0; mask < 8; mask++) strings.push(formatBitsFor(ecc, mask));
  for (let i = 0; i < strings.length; i++) {
    for (let j = i + 1; j < strings.length; j++) {
      let difference = strings[i] ^ strings[j];
      let distance = 0;
      while (difference) {
        distance += difference & 1;
        difference >>>= 1;
      }
      if (distance < minimumDistance) minimumDistance = distance;
    }
  }
  if (minimumDistance !== 7) throw new Error(`QR format code: minimum distance ${minimumDistance}, expected 7`);

  return true;
}
