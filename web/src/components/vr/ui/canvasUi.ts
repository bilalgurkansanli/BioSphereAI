/**
 * The interface inside the headset, drawn with the 2D canvas.
 *
 * A headset shows no DOM — `immersive-vr` has no overlay — so the menu, the labels and the
 * info cards are pictures on quads. They are drawn here, once per change, with the same tokens
 * as the 2D chrome (`globals.css`, dark theme): the same glass, the same ink, the same accent,
 * and Geist through the body's own font stack. Both engines texture from a canvas, so the 360
 * (three.js) and the walkthrough (PlayCanvas) share every pixel of it.
 *
 * Drawn at roughly twice the size the eye resolves at its viewing distance (a 1024-pixel menu
 * subtends about 26 degrees at 1.35 m, ~39 px per degree, against the ~20-25 a Quest panel
 * resolves) and mipmapped by the caller, so it stays crisp without shimmering as the head moves.
 */

export const VR_COLOURS = {
  glass: "rgba(11, 15, 21, 0.84)",
  glassHover: "rgba(23, 226, 197, 0.16)",
  border: "rgba(255, 255, 255, 0.14)",
  borderHover: "rgba(23, 226, 197, 0.55)",
  ink: "#e9eef4",
  muted: "#94a2b3",
  faint: "#5d6a7a",
  bio: "#17e2c5",
  amber: "#ffb44c",
  danger: "#ff6b6b",
} as const;

let cachedFont: string | null = null;

/** The page's own sans-serif stack — Geist, through `next/font`'s generated family name. */
export function uiFontFamily(): string {
  if (cachedFont) return cachedFont;
  let family = "";
  try {
    family = getComputedStyle(document.body).fontFamily;
  } catch {
    family = "";
  }
  cachedFont = family && family.trim() ? family : "ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif";
  return cachedFont;
}

function font(sizePx: number, weight: number): string {
  return `${weight} ${sizePx}px ${uiFontFamily()}`;
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

function ellipsize(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ctx.measureText(`${text.slice(0, mid)}…`).width <= maxWidth) lo = mid;
    else hi = mid - 1;
  }
  return `${text.slice(0, lo).trimEnd()}…`;
}

/** Greedy word wrap, at most `maxLines`, the last one ellipsised if the text runs on. */
export function wrapLines(ctx: CanvasRenderingContext2D, text: string, maxWidth: number, maxLines: number): string[] {
  const lines: string[] = [];
  for (const paragraph of text.split(/\n+/)) {
    const words = paragraph.split(/\s+/).filter(Boolean);
    let line = "";
    for (const word of words) {
      const next = line ? `${line} ${word}` : word;
      if (ctx.measureText(next).width <= maxWidth || !line) {
        line = next;
      } else {
        lines.push(line);
        line = word;
      }
      if (lines.length >= maxLines) break;
    }
    if (line && lines.length < maxLines) lines.push(line);
    if (lines.length >= maxLines) break;
  }
  const consumed = lines.join(" ").replace(/\s+/g, " ").length;
  if (consumed < text.replace(/\s+/g, " ").trim().length && lines.length) {
    lines[lines.length - 1] = ellipsize(ctx, `${lines[lines.length - 1]}…`, maxWidth);
  }
  return lines.map((l) => ellipsize(ctx, l, maxWidth));
}

// ---------------------------------------------------------------------------------
// Icons, as paths: no icon font to wait for, no image to fetch.
// ---------------------------------------------------------------------------------

export type VrIcon = "prev" | "next" | "close" | "walk" | "info" | "link" | "media";

function drawIcon(ctx: CanvasRenderingContext2D, icon: VrIcon, cx: number, cy: number, size: number, colour: string): void {
  const s = size / 2;
  ctx.save();
  ctx.strokeStyle = colour;
  ctx.fillStyle = colour;
  ctx.lineWidth = Math.max(3, size * 0.11);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  switch (icon) {
    case "prev":
      ctx.moveTo(cx + s * 0.3, cy - s * 0.6);
      ctx.lineTo(cx - s * 0.3, cy);
      ctx.lineTo(cx + s * 0.3, cy + s * 0.6);
      ctx.stroke();
      break;
    case "next":
      ctx.moveTo(cx - s * 0.3, cy - s * 0.6);
      ctx.lineTo(cx + s * 0.3, cy);
      ctx.lineTo(cx - s * 0.3, cy + s * 0.6);
      ctx.stroke();
      break;
    case "close":
      ctx.moveTo(cx - s * 0.5, cy - s * 0.5);
      ctx.lineTo(cx + s * 0.5, cy + s * 0.5);
      ctx.moveTo(cx + s * 0.5, cy - s * 0.5);
      ctx.lineTo(cx - s * 0.5, cy + s * 0.5);
      ctx.stroke();
      break;
    case "walk":
      // Two footprints, offset: the same idea as the 2D chrome's "Footprints" glyph.
      for (const [dx, dy] of [
        [-0.32, 0.18],
        [0.32, -0.22],
      ]) {
        ctx.beginPath();
        ctx.ellipse(cx + dx * s, cy + dy * s, s * 0.2, s * 0.36, 0, 0, Math.PI * 2);
        ctx.fill();
      }
      break;
    case "info":
      ctx.arc(cx, cy - s * 0.48, s * 0.12, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(cx, cy - s * 0.12);
      ctx.lineTo(cx, cy + s * 0.55);
      ctx.stroke();
      break;
    case "link":
      ctx.moveTo(cx - s * 0.45, cy + s * 0.45);
      ctx.lineTo(cx + s * 0.45, cy - s * 0.45);
      ctx.moveTo(cx - s * 0.05, cy - s * 0.45);
      ctx.lineTo(cx + s * 0.45, cy - s * 0.45);
      ctx.lineTo(cx + s * 0.45, cy + s * 0.05);
      ctx.stroke();
      break;
    case "media":
      ctx.moveTo(cx - s * 0.32, cy - s * 0.5);
      ctx.lineTo(cx + s * 0.52, cy);
      ctx.lineTo(cx - s * 0.32, cy + s * 0.5);
      ctx.closePath();
      ctx.fill();
      break;
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------------
// The menu panel
// ---------------------------------------------------------------------------------

export interface PanelButtonSpec {
  type: "button";
  id: string;
  icon?: VrIcon;
  /** Text after the icon. An icon-only button is a circle. */
  label?: string;
  tone?: "default" | "danger";
  disabled?: boolean;
  /** A switch that is on: drawn in the accent, like an active dock segment. */
  pressed?: boolean;
}

export interface PanelTextSpec {
  type: "text";
  title: string;
  subtitle?: string;
}

export type PanelItem = PanelButtonSpec | PanelTextSpec;

export interface PanelRegionUv {
  id: string;
  u0: number;
  v0: number;
  u1: number;
  v1: number;
}

export interface PanelState {
  hovered: string | null;
  /** 0..1 gaze dwell on the hovered button, drawn as a filling underline. */
  progress: number;
}

export const PANEL_WIDTH = 1024;
export const PANEL_HEIGHT = 176;
/** The narrowest the text block may get before the buttons' words shrink to make room. */
const MIN_TEXT_WIDTH = 220;
const BUTTON_FONT = 44;

/**
 * One row, left to right, the 2D dock's grammar: round icon buttons, pill buttons with words,
 * and one text block that takes whatever width is left. Returns the clickable regions in the
 * panel's own u/v (v down), which is what `gaze.rayPanel` reports.
 *
 * A row whose words do not fit is not allowed to run off the edge — the first walkthrough menu
 * measured 1116 px of content on a 1024 px canvas, which would have cut the exit button in
 * half. The buttons' type shrinks (to no less than 70 %) until the text block keeps
 * `MIN_TEXT_WIDTH`; past that the room name is ellipsised, never a control.
 */
export function drawPanel(canvas: HTMLCanvasElement, items: readonly PanelItem[], state: PanelState, width = PANEL_WIDTH): PanelRegionUv[] {
  const W = width;
  const H = PANEL_HEIGHT;
  if (canvas.width !== W) canvas.width = W;
  if (canvas.height !== H) canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];
  ctx.clearRect(0, 0, W, H);

  roundRect(ctx, 2, 2, W - 4, H - 4, H / 2);
  ctx.fillStyle = VR_COLOURS.glass;
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = VR_COLOURS.border;
  ctx.stroke();

  const pad = 24;
  const gap = 16;
  const bh = H - pad * 2;
  const hasText = items.some((item) => item.type === "text");

  // Measure buttons; the text block flexes. Shrink the buttons' words if the row overflows.
  let buttonFont = BUTTON_FONT;
  let widths: number[] = [];
  let fixed = 0;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    ctx.font = font(buttonFont, 600);
    widths = items.map((item) => {
      if (item.type === "text") return 0;
      if (!item.label) return bh;
      return Math.ceil(ctx.measureText(item.label).width) + (item.icon ? bh * 0.62 : 0) + 64;
    });
    fixed = widths.reduce((a, b) => a + b, 0) + gap * (items.length - 1);
    if (W - pad * 2 - fixed >= (hasText ? MIN_TEXT_WIDTH : 0) || buttonFont <= BUTTON_FONT * 0.7) break;
    buttonFont = Math.max(BUTTON_FONT * 0.7, buttonFont - 3);
  }
  const flex = Math.max(120, W - pad * 2 - fixed);

  const regions: PanelRegionUv[] = [];
  let x = pad;
  items.forEach((item, index) => {
    const w = item.type === "text" ? flex : widths[index];
    if (item.type === "text") {
      ctx.textBaseline = "middle";
      ctx.textAlign = "center";
      ctx.fillStyle = VR_COLOURS.ink;
      ctx.font = font(48, 600);
      const cx = x + w / 2;
      if (item.subtitle) {
        ctx.fillText(ellipsize(ctx, item.title, w - 16), cx, H / 2 - 20);
        ctx.font = font(34, 500);
        ctx.fillStyle = VR_COLOURS.muted;
        ctx.fillText(ellipsize(ctx, item.subtitle, w - 16), cx, H / 2 + 30);
      } else {
        ctx.fillText(ellipsize(ctx, item.title, w - 16), cx, H / 2 + 2);
      }
    } else {
      const hovered = state.hovered === item.id && !item.disabled;
      const danger = item.tone === "danger";
      const lit = hovered || item.pressed;
      roundRect(ctx, x, pad, w, bh, bh / 2);
      ctx.fillStyle = lit ? (danger ? "rgba(255, 107, 107, 0.18)" : VR_COLOURS.glassHover) : "rgba(255, 255, 255, 0.04)";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = lit ? (danger ? "rgba(255, 107, 107, 0.7)" : VR_COLOURS.borderHover) : VR_COLOURS.border;
      ctx.stroke();
      const colour = item.disabled ? VR_COLOURS.faint : lit ? (danger ? VR_COLOURS.danger : VR_COLOURS.bio) : VR_COLOURS.ink;
      if (item.label) {
        ctx.font = font(buttonFont, 600);
        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        let tx = x + 32;
        if (item.icon) {
          drawIcon(ctx, item.icon, tx + bh * 0.24, H / 2, bh * 0.42, colour);
          tx += bh * 0.62;
        }
        ctx.fillStyle = colour;
        ctx.fillText(item.label, tx, H / 2 + 2);
      } else if (item.icon) {
        drawIcon(ctx, item.icon, x + w / 2, H / 2, bh * 0.46, colour);
      }
      if (hovered && state.progress > 0) {
        // The dwell, as a bar filling along the bottom of the button.
        ctx.save();
        roundRect(ctx, x, pad, w, bh, bh / 2);
        ctx.clip();
        ctx.fillStyle = danger ? VR_COLOURS.danger : VR_COLOURS.bio;
        ctx.globalAlpha = 0.9;
        ctx.fillRect(x, pad + bh - 10, w * Math.min(1, state.progress), 10);
        ctx.restore();
      }
      if (!item.disabled) regions.push({ id: item.id, u0: x / W, v0: pad / H, u1: (x + w) / W, v1: (pad + bh) / H });
    }
    x += w + gap;
  });
  return regions;
}

// ---------------------------------------------------------------------------------
// Labels, cards, markers
// ---------------------------------------------------------------------------------

export interface LabelSpec {
  text: string;
  /** A second, quieter line — "not shot yet" under a room name. */
  note?: string;
  noteTone?: "muted" | "amber";
}

/** A pill with one or two lines. Returns the aspect (width / height) to size the quad by. */
export function drawLabel(canvas: HTMLCanvasElement, spec: LabelSpec): number {
  const ctx = canvas.getContext("2d");
  if (!ctx) return 1;
  const titleFont = font(46, 600);
  const noteFont = font(34, 500);
  ctx.font = titleFont;
  const titleWidth = ctx.measureText(spec.text).width;
  ctx.font = noteFont;
  const noteWidth = spec.note ? ctx.measureText(spec.note).width : 0;
  const W = Math.min(1024, Math.ceil(Math.max(titleWidth, noteWidth) + 72));
  const H = spec.note ? 132 : 84;
  canvas.width = W;
  canvas.height = H;
  ctx.clearRect(0, 0, W, H);
  roundRect(ctx, 2, 2, W - 4, H - 4, spec.note ? 34 : H / 2);
  ctx.fillStyle = VR_COLOURS.glass;
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = VR_COLOURS.border;
  ctx.stroke();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = titleFont;
  ctx.fillStyle = VR_COLOURS.ink;
  ctx.fillText(ellipsize(ctx, spec.text, W - 48), W / 2, spec.note ? 46 : H / 2 + 2);
  if (spec.note) {
    ctx.font = noteFont;
    ctx.fillStyle = spec.noteTone === "amber" ? VR_COLOURS.amber : VR_COLOURS.muted;
    ctx.fillText(ellipsize(ctx, spec.note, W - 48), W / 2, 94);
  }
  return W / H;
}

/** An info card: a title, the body wrapped, and a close mark. Returns the aspect. */
export function drawCard(canvas: HTMLCanvasElement, title: string, body: string | null, closeHint: string): number {
  const ctx = canvas.getContext("2d");
  if (!ctx) return 1;
  const W = 900;
  const pad = 52;
  ctx.font = font(36, 400);
  const lines = body ? wrapLines(ctx, body, W - pad * 2, 8) : [];
  const H = pad + 60 + (lines.length ? 20 + lines.length * 52 : 0) + 40 + 40 + pad * 0.6;
  canvas.width = W;
  canvas.height = Math.ceil(H);
  ctx.clearRect(0, 0, W, H);
  roundRect(ctx, 2, 2, W - 4, H - 4, 40);
  ctx.fillStyle = "rgba(11, 15, 21, 0.92)";
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = VR_COLOURS.border;
  ctx.stroke();

  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.font = font(48, 650);
  ctx.fillStyle = VR_COLOURS.ink;
  ctx.fillText(ellipsize(ctx, title, W - pad * 2 - 70), pad, pad + 44);
  drawIcon(ctx, "close", W - pad - 14, pad + 28, 30, VR_COLOURS.muted);

  ctx.font = font(36, 400);
  ctx.fillStyle = VR_COLOURS.muted;
  let y = pad + 60 + 20 + 36;
  for (const line of lines) {
    ctx.fillText(line, pad, y);
    y += 52;
  }
  ctx.font = font(28, 500);
  ctx.fillStyle = VR_COLOURS.faint;
  ctx.fillText(closeHint, pad, H - pad * 0.6);
  return W / H;
}

/** The round badge a tag shows on the wall. */
export function drawMarker(canvas: HTMLCanvasElement, icon: VrIcon, accent: string): void {
  const S = 160;
  canvas.width = S;
  canvas.height = S;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, S, S);
  ctx.beginPath();
  ctx.arc(S / 2, S / 2, S / 2 - 8, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(11, 15, 21, 0.78)";
  ctx.fill();
  ctx.lineWidth = 6;
  ctx.strokeStyle = accent;
  ctx.stroke();
  drawIcon(ctx, icon, S / 2, S / 2, S * 0.42, VR_COLOURS.ink);
}
