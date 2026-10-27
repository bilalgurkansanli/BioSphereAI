/**
 * Generates equirectangular calibration panoramas for eyeballing the 360 player.
 *
 * Why this exists: a photographic panorama hides orientation bugs. A synthetic one with a
 * signed yaw ruler, a level horizon, readable text and hard-edged pole caps makes four
 * specific failure modes obvious in a single screenshot:
 *
 *   mirroring    — the words render backwards
 *   yaw offset   — the label under the crosshair is not the yaw the player reports
 *   pole swap    — the zenith cap is not the colour the legend says it is
 *   seam         — the +/-180 column shows a visible discontinuity or a grey smear
 *
 * No image dependencies: PNG is written by hand (signature + IHDR + IDAT + IEND) with
 * node:zlib doing the deflate, so this runs on a bare `node` with nothing installed.
 *
 * Usage:  node scripts/make-test-pano.mjs
 * Output: web/public/dev/*.png
 */

import { deflateSync } from "node:zlib";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "dev");

// ---------------------------------------------------------------------------------
// PNG encoder (colour type 2 — 8-bit RGB, no alpha, filter 0 on every scanline)
// ---------------------------------------------------------------------------------

const CRC_TABLE = (() => {
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  return table;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const body = Buffer.concat([Buffer.from(type, "latin1"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([len, body, crc]);
}

/** @param {Uint8Array} rgb tightly packed width*height*3 */
function encodePng(rgb, width, height) {
  const stride = width * 3;
  // One extra byte per scanline carries the filter type; 0 = None.
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0;
    Buffer.from(rgb.buffer, rgb.byteOffset + y * stride, stride).copy(raw, y * (stride + 1) + 1);
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type: truecolour
  ihdr[10] = 0; // deflate
  ihdr[11] = 0; // adaptive filtering
  ihdr[12] = 0; // no interlace

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 6 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

// ---------------------------------------------------------------------------------
// 5x7 bitmap font — enough to spell the labels and to reveal mirroring
// ---------------------------------------------------------------------------------

const FONT = {
  "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
  "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
  "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
  "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
  "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
  "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
  "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
  "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
  "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
  "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
  A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
  B: ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
  C: ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
  D: ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
  E: ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
  F: ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
  G: ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
  H: ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
  I: ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
  J: ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
  K: ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
  L: ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
  M: ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
  N: ["10001", "10001", "11001", "10101", "10011", "10001", "10001"],
  O: ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
  P: ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
  Q: ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
  R: ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
  S: ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
  T: ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
  U: ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
  V: ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
  W: ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
  X: ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
  Y: ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
  Z: ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
  "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
  "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
  ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
  "°": ["01100", "10010", "10010", "01100", "00000", "00000", "00000"],
  " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
};

const GLYPH_W = 5;
const GLYPH_H = 7;

// ---------------------------------------------------------------------------------
// Canvas helpers
// ---------------------------------------------------------------------------------

function createCanvas(width, height) {
  const data = new Uint8Array(width * height * 3);
  const px = (x, y, r, g, b) => {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    const i = (y * width + x) * 3;
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
  };
  return { width, height, data, px };
}

/** Wraps horizontally so text drawn across the +/-180 seam stays continuous. */
function wrapX(x, width) {
  return ((x % width) + width) % width;
}

function drawText(cv, text, cx, cy, scale, rgb, align = "center") {
  const chars = [...text.toUpperCase()];
  const advance = (GLYPH_W + 1) * scale;
  const totalW = chars.length * advance - scale;
  let startX = cx;
  if (align === "center") startX = Math.round(cx - totalW / 2);
  else if (align === "right") startX = Math.round(cx - totalW);
  const startY = Math.round(cy - (GLYPH_H * scale) / 2);

  chars.forEach((ch, ci) => {
    const glyph = FONT[ch] ?? FONT[" "];
    for (let gy = 0; gy < GLYPH_H; gy++) {
      for (let gx = 0; gx < GLYPH_W; gx++) {
        if (glyph[gy][gx] !== "1") continue;
        for (let sy = 0; sy < scale; sy++) {
          for (let sx = 0; sx < scale; sx++) {
            const x = wrapX(startX + ci * advance + gx * scale + sx, cv.width);
            cv.px(x, startY + gy * scale + sy, rgb[0], rgb[1], rgb[2]);
          }
        }
      }
    }
  });
}

function vLine(cv, x, y0, y1, thickness, rgb) {
  for (let y = Math.max(0, y0); y < Math.min(cv.height, y1); y++) {
    for (let t = 0; t < thickness; t++) cv.px(wrapX(x + t, cv.width), y, rgb[0], rgb[1], rgb[2]);
  }
}

function hLine(cv, y, thickness, rgb) {
  for (let t = 0; t < thickness; t++) {
    for (let x = 0; x < cv.width; x++) cv.px(x, y + t, rgb[0], rgb[1], rgb[2]);
  }
}

/** HSV -> RGB, h in [0,360). Keeps the yaw ramp perceptually continuous round the seam. */
function hsv(h, s, v) {
  const c = v * s;
  const hp = (((h % 360) + 360) % 360) / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  const [r, g, b] =
    hp < 1 ? [c, x, 0] : hp < 2 ? [x, c, 0] : hp < 3 ? [0, c, x] : hp < 4 ? [0, x, c] : hp < 5 ? [x, 0, c] : [c, 0, x];
  const m = v - c;
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

// ---------------------------------------------------------------------------------
// The panorama itself
// ---------------------------------------------------------------------------------

/**
 * Pixel column u maps to yaw via  u = 0.5 + yaw / 2pi  (0 = centre column), and row v
 * maps to pitch via  v = 0.5 - pitch / pi  (row 0 = zenith). That is exactly the mapping
 * the player's shader assumes, so any disagreement shows up as a misplaced label.
 */
function renderPano({ width, height, hueBase, label, accent }) {
  const cv = createCanvas(width, height);
  const { data } = cv;

  const ZENITH = [232, 62, 140]; // magenta ceiling
  const NADIR = [46, 96, 232]; // blue floor
  const INK = [8, 10, 14];
  const PAPER = [242, 246, 250];

  for (let y = 0; y < height; y++) {
    const pitchDeg = 90 - (y / height) * 180;
    // Hard caps above 75 and below -75 so a pole swap is unmistakable.
    const cap = pitchDeg > 75 ? ZENITH : pitchDeg < -75 ? NADIR : null;

    for (let x = 0; x < width; x++) {
      const yawDeg = (x / width) * 360 - 180;
      let r, g, b;
      if (cap) {
        // Slight radial shading so the pole pinch is visible without hiding the colour.
        const k = 1 - (Math.abs(pitchDeg) - 75) / 30;
        r = Math.round(cap[0] * (0.55 + 0.45 * k));
        g = Math.round(cap[1] * (0.55 + 0.45 * k));
        b = Math.round(cap[2] * (0.55 + 0.45 * k));
      } else {
        // Hue sweeps a full turn with yaw; value darkens toward the floor so up/down reads.
        const v = 0.34 + 0.3 * (1 - Math.abs(pitchDeg) / 75) + (pitchDeg > 0 ? 0.16 : 0);
        [r, g, b] = hsv(hueBase + yawDeg, 0.5, v);
      }
      const i = (y * width + x) * 3;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
    }
  }

  const S = Math.max(1, Math.round(width / 512)); // text scale, 4 at 2048 wide
  const rowFor = (pitchDeg) => Math.round((0.5 - pitchDeg / 180) * height);
  const colFor = (yawDeg) => Math.round((0.5 + yawDeg / 360) * width);

  // Pitch grid: a bright level horizon plus dimmer lines every 30 deg.
  for (const p of [-60, -30, 30, 60]) hLine(cv, rowFor(p), Math.max(1, S / 2), [255, 255, 255]);
  hLine(cv, rowFor(0) - Math.floor(S / 2), S, PAPER);

  for (const p of [-60, -30, 30, 60]) {
    drawText(cv, `${p > 0 ? "+" : ""}${p}°`, colFor(-172), rowFor(p) - GLYPH_H * S, S, PAPER, "left");
  }

  // Yaw ruler every 30 deg, labelled with signed degrees. The label sits just under the
  // horizon so a screenshot of the centre of the frame always contains the ground truth.
  for (let yawDeg = -180; yawDeg < 180; yawDeg += 30) {
    const x = colFor(yawDeg);
    const major = yawDeg === 0;
    vLine(cv, x, rowFor(75), rowFor(-75), major ? S : Math.max(1, S / 2), major ? [23, 226, 197] : PAPER);
    drawText(cv, `${yawDeg > 0 ? "+" : ""}${yawDeg}`, x, rowFor(0) + GLYPH_H * S, S, major ? [23, 226, 197] : PAPER);
  }
  // Minor ticks every 10 deg give a finer read on alignment.
  for (let yawDeg = -180; yawDeg < 180; yawDeg += 10) {
    if (yawDeg % 30 === 0) continue;
    vLine(cv, colFor(yawDeg), rowFor(4), rowFor(-4), 1, PAPER);
  }

  // Words. If any of these read backwards the sphere UVs are mirrored.
  drawText(cv, label, colFor(0), rowFor(22), S * 2, accent);
  drawText(cv, "FRONT", colFor(0), rowFor(13), S, PAPER);
  drawText(cv, "RIGHT", colFor(90), rowFor(13), S, PAPER);
  drawText(cv, "BACK", colFor(180), rowFor(13), S, PAPER);
  drawText(cv, "LEFT", colFor(-90), rowFor(13), S, PAPER);
  drawText(cv, "ZENITH UP", colFor(0), rowFor(84), S, INK);
  drawText(cv, "NADIR DOWN", colFor(0), rowFor(-84), S, PAPER);

  // A hard black/white checker straddling the seam: any bleed or wrap error smears it.
  const cell = Math.max(2, Math.round(width / 128));
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 4; j++) {
      const on = (i + j) % 2 === 0;
      const c = on ? PAPER : INK;
      for (let dy = 0; dy < cell; dy++) {
        for (let dx = 0; dx < cell; dx++) {
          cv.px(wrapX(colFor(180) - 4 * cell + i * cell + dx, width), rowFor(-38) + j * cell + dy, c[0], c[1], c[2]);
        }
      }
    }
  }

  return cv;
}

// ---------------------------------------------------------------------------------

const ROOMS = [
  { key: "a", hueBase: 190, label: "ROOM A", accent: [23, 226, 197] },
  { key: "b", hueBase: 30, label: "ROOM B", accent: [255, 180, 76] },
];

// preview intentionally tiny — it is the instant-on rung and should look like one.
const LEVELS = [
  { name: "preview", width: 512 },
  { name: "sd", width: 2048 },
  { name: "hd", width: 4096 },
];

mkdirSync(OUT_DIR, { recursive: true });

for (const room of ROOMS) {
  for (const level of LEVELS) {
    const started = Date.now();
    const cv = renderPano({ ...room, width: level.width, height: level.width / 2 });
    const png = encodePng(cv.data, cv.width, cv.height);
    const file = join(OUT_DIR, `${room.key}-${level.name}.png`);
    writeFileSync(file, png);
    console.log(
      `${room.key}-${level.name}.png  ${cv.width}x${cv.height}  ` +
        `${(png.length / 1024).toFixed(0)} KB  ${Date.now() - started} ms`,
    );
  }
}

console.log(`\nWrote ${ROOMS.length * LEVELS.length} files to ${OUT_DIR}`);
