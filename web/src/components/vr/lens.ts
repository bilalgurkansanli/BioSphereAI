/**
 * The phone-in-a-box headset, as geometry: where the lenses sit over the screen, what each eye
 * can see through them, and the barrel pre-distortion that cancels what the lens does.
 *
 * Pure arithmetic — no React, no WebGL — so the whole Cardboard path can be checked at a desk.
 * Both renderers consume it: three.js for the 360 player and PlayCanvas for the walkthrough,
 * which is the reason the mesh is emitted as plain typed arrays rather than as either engine's
 * geometry.
 *
 * ── THE MODEL ──────────────────────────────────────────────────────────────────────────
 * This is Google's Cardboard model (the one the Cardboard SDK and the old WebVR polyfill
 * shipped), reduced to what a browser can know. Distances are tan-angles: a point on the
 * screen `x` metres from the lens axis, `D` metres behind the lens, sits at screen tan-angle
 * `s = x / D`. The lens makes it APPEAR further out — at visible tan-angle
 *
 *     t = distort(s) = s · (1 + k1·s² + k2·s⁴)
 *
 * (pincushion). So to show the eye a direction `t`, the pixel has to be drawn at
 * `s = distortInverse(t)`: pulled in towards the lens centre, i.e. barrel-distorted. Each eye
 * is rendered undistorted into a texture covering exactly the visible frustum, and a mesh
 * gridded regularly in visible tan-space carries that texture to the pre-distorted screen
 * positions. The Cardboard SDK uses the same 40×40 grid; the error of linear interpolation
 * across one cell at that density is far below a pixel.
 *
 * ── WHY A MESH AND NOT A FRAGMENT SHADER ───────────────────────────────────────────────
 * Per-pixel distortion is the more exact of the two and was the first plan. It does not
 * survive PlayCanvas: the walkthrough prefers WebGPU, where a custom shader means writing and
 * maintaining it twice (GLSL and WGSL). A textured mesh needs no custom shader at all — an
 * unlit material samples it on either API — so both engines share one tested piece of maths.
 *
 * ── WHAT A BROWSER CANNOT KNOW ─────────────────────────────────────────────────────────
 * The physical size of the screen. CSS pixels are not a physical unit, so it is estimated from
 * a typical density (`CSS_PX_PER_INCH`, measured across the phones below). The estimate only
 * moves the lens centres, and only by its own error times half the lens spacing: a 7 % miss
 * shifts each lens centre by about 0.2 mm, which nobody can see. What the lens shows (the
 * frustum) depends on the lens-to-screen distance and the distortion, not on the screen size.
 */

export type LensId = "cardboard" | "vrbox";

export interface LensProfile {
  id: LensId;
  /** Distance between the two lens centres, metres. */
  interLensDistance: number;
  /** Distance from the screen to the lens, metres. */
  screenLensDistance: number;
  /**
   * Height of the lens centre above the lower edge of the phone tray, metres, or `null` when
   * the headset centres the phone vertically between the lenses (every clamp-style VR box).
   */
  trayLensDistance: number | null;
  /** Radial distortion of the lens: t = s·(1 + k1·s² + k2·s⁴), s and t in tan-angle units. */
  k1: number;
  k2: number;
  /** The widest half-angle the lens shows, degrees. Nothing beyond it is rendered. */
  fovDeg: number;
}

/**
 * The two profiles a visitor can pick from.
 *
 * CARDBOARD is Google's own published "Cardboard I/O 2015" viewer (the v2 fold-out), verbatim:
 * 64 mm between lenses, 39 mm from the screen, lens centre 35 mm above the tray, k = (0.34,
 * 0.55), 60 degrees each way. Every licensed Cardboard clone uses the same optics.
 *
 * VR BOX stands for the plastic clamp headsets sold everywhere in Turkey under a dozen names.
 * They have no published optics, so these numbers are an ESTIMATE for the class, and are
 * labelled as one: lenses of about 42 mm focal length set a little further from the screen,
 * centred vertically (the clamp holds the phone mid-height), a 62 mm default spacing (most of
 * them have a slider around it), and milder distortion than Cardboard's very strong lens. An
 * over-correction reads as straight lines bowing inward, an under-correction as bowing
 * outward; the milder guess fails the gentler way. `?vrlens=` overrides any of it in the field
 * (see `parseLensOverride`).
 */
export const LENS_PROFILES: Record<LensId, LensProfile> = {
  cardboard: {
    id: "cardboard",
    interLensDistance: 0.064,
    screenLensDistance: 0.039,
    trayLensDistance: 0.035,
    k1: 0.34,
    k2: 0.55,
    fovDeg: 60,
  },
  vrbox: {
    id: "vrbox",
    interLensDistance: 0.062,
    screenLensDistance: 0.042,
    trayLensDistance: null,
    k1: 0.28,
    k2: 0.24,
    fovDeg: 52,
  },
};

export const DEFAULT_LENS: LensId = "vrbox";

/**
 * CSS pixels per physical inch on a typical phone.
 *
 * Worked from the published screen sizes: iPhone 11 161, iPhone 13/14/15 153, iPhone 15 Pro
 * Max 153, Galaxy A54 157, Redmi Note 12 143, Galaxy S23 141. 152 sits in the middle of that
 * spread; see the header for why the remaining error does not matter.
 */
export const CSS_PX_PER_INCH = 152;

const METRES_PER_INCH = 0.0254;

/** The phone's screen in landscape, metres. */
export interface ScreenSize {
  widthM: number;
  heightM: number;
  /** Bezel between the lower screen edge and the tray. Only matters for tray-aligned lenses. */
  bevelM: number;
}

export function screenFromCss(cssWidth: number, cssHeight: number, cssPxPerInch = CSS_PX_PER_INCH): ScreenSize {
  const long = Math.max(cssWidth, cssHeight);
  const short = Math.min(cssWidth, cssHeight);
  const metresPerCss = METRES_PER_INCH / cssPxPerInch;
  return { widthM: long * metresPerCss, heightM: short * metresPerCss, bevelM: 0.003 };
}

/**
 * The density that makes any window look like a 6.1-inch phone.
 *
 * Only for the desktop debug mode (`?vr=cardboard`): a 1920-pixel window at the phone density
 * would be a 32 cm "phone" whose lens centres sit near the middle of an enormous screen, and
 * the preview would be two small discs. Pretending the window is a phone of ordinary size
 * shows what the phone will show, scaled up.
 */
export function phoneEquivalentDensity(cssWidth: number, cssHeight: number): number {
  const long = Math.max(cssWidth, cssHeight);
  // A 6.1-inch 19.5:9 panel is 5.53 inches along its long side.
  return long / 5.53;
}

// ---------------------------------------------------------------------------------
// The lens function
// ---------------------------------------------------------------------------------

/** Screen tan-angle -> the tan-angle the eye sees it at. Odd, so it keeps the sign. */
export function distort(s: number, k1: number, k2: number): number {
  const s2 = s * s;
  return s * (1 + k1 * s2 + k2 * s2 * s2);
}

/**
 * Visible tan-angle -> where on the screen it has to be drawn. Inverse of {@link distort}.
 *
 * Newton's method. For k1, k2 >= 0 the lens function is strictly increasing with a derivative
 * of at least 1, so the iteration is monotone from the starting guess and converges in a
 * handful of steps; the bisection guard only exists for a hand-entered profile with a negative
 * coefficient, whose curve can fold back on itself far out.
 */
export function distortInverse(t: number, k1: number, k2: number): number {
  if (t === 0) return 0;
  const target = Math.abs(t);
  let r = target / (1 + k1 * target * target + k2 * target ** 4) ** (1 / 3);
  let lo = 0;
  let hi = target * 2 + 2;
  for (let i = 0; i < 40; i += 1) {
    const r2 = r * r;
    const f = r * (1 + k1 * r2 + k2 * r2 * r2) - target;
    if (Math.abs(f) < 1e-12) break;
    if (f > 0) hi = Math.min(hi, r);
    else lo = Math.max(lo, r);
    const df = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2;
    let next = r - f / df;
    if (!(next > lo && next < hi) || df <= 0) next = (lo + hi) / 2;
    if (Math.abs(next - r) < 1e-13) {
      r = next;
      break;
    }
    r = next;
  }
  return Math.sign(t) * r;
}

// ---------------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------------

/** Tan-angles of a view frustum: `left` and `bottom` are negative for a centred view. */
export interface Frustum {
  left: number;
  right: number;
  bottom: number;
  top: number;
}

export type EyeSide = "left" | "right";

export interface EyeLayout {
  side: EyeSide;
  /** What this eye sees through its lens — the frustum to render. */
  visible: Frustum;
  /** Lens centre, metres from the screen's left / lower edge (landscape). */
  lensCentreM: [number, number];
  /** Lateral offset of this eye from the middle of the head, metres. For stereo content. */
  eyeOffsetM: number;
  /** The slice of the screen this eye owns, as fractions of the canvas [x, y, width, height]. */
  viewport: [number, number, number, number];
}

export interface StereoLayout {
  lens: LensProfile;
  screen: ScreenSize;
  eyes: [EyeLayout, EyeLayout];
}

/**
 * Where each eye looks and what it sees, for one lens on one screen.
 *
 * The left eye owns the left half of the landscape screen and looks through a lens whose
 * centre is half the lens spacing left of the middle. Each edge of that half, seen through the
 * lens, sits at `distort(edge)` — clipped to the lens's own field. The right eye is the mirror
 * image; the vertical extent is shared.
 */
export function stereoLayout(lens: LensProfile, screen: ScreenSize): StereoLayout {
  const { widthM: W, heightM: H } = screen;
  const D = lens.screenLensDistance;
  const half = lens.interLensDistance / 2;
  const tanFov = Math.tan((lens.fovDeg * Math.PI) / 180);

  const cy = lens.trayLensDistance === null ? H / 2 : lens.trayLensDistance - screen.bevelM;
  const cxLeft = W / 2 - half;

  const clip = (screenTan: number) =>
    Math.max(-tanFov, Math.min(tanFov, distort(screenTan, lens.k1, lens.k2)));

  const left: Frustum = {
    left: clip((0 - cxLeft) / D),
    right: clip((W / 2 - cxLeft) / D),
    bottom: clip((0 - cy) / D),
    top: clip((H - cy) / D),
  };
  const right: Frustum = {
    left: -left.right,
    right: -left.left,
    bottom: left.bottom,
    top: left.top,
  };

  return {
    lens,
    screen,
    eyes: [
      { side: "left", visible: left, lensCentreM: [cxLeft, cy], eyeOffsetM: -half, viewport: [0, 0, 0.5, 1] },
      { side: "right", visible: right, lensCentreM: [W / 2 + half, cy], eyeOffsetM: half, viewport: [0.5, 0, 0.5, 1] },
    ],
  };
}

// ---------------------------------------------------------------------------------
// The distortion mesh
// ---------------------------------------------------------------------------------

export interface DistortionMesh {
  /** x, y per vertex in normalised device coordinates of the WHOLE canvas (-1..1, y up). */
  positions: Float32Array;
  /** u, v per vertex into this eye's texture. */
  uvs: Float32Array;
  /** 0..1 per vertex: 1 in the middle, falling to 0 at the rim — the soft lens edge. */
  shade: Float32Array;
  indices: Uint16Array;
}

export interface DistortionMeshOptions {
  /** Grid resolution. 40 is the Cardboard SDK's figure. */
  cols?: number;
  rows?: number;
  /**
   * Which texture row is v = 0. three.js render targets are sampled with v = 0 at the
   * BOTTOM; PlayCanvas render targets created with `RENDERTARGET_ORIGIN_TOP` are sampled like
   * images, v = 0 at the TOP. Getting this wrong draws the world upside down, not subtly off.
   */
  vOrigin: "bottom" | "top";
  /** How much of the grid, as a fraction of its width, the rim fades over. */
  fade?: number;
}

/**
 * The mesh that carries one eye's undistorted render to its barrel-distorted place on screen.
 *
 * Gridded regularly in VISIBLE tan-space, so texture coordinates are a plain lerp and every
 * vertex's screen position is the one exact solve `distortInverse` — which is the property the
 * tests pin: at every vertex, the screen point and the direction it shows agree through the
 * lens function.
 */
export function distortionMesh(layout: StereoLayout, eye: EyeLayout, options: DistortionMeshOptions): DistortionMesh {
  const cols = Math.max(2, options.cols ?? 40);
  const rows = Math.max(2, options.rows ?? 40);
  const fade = options.fade ?? 0.045;
  const { k1, k2, screenLensDistance: D } = layout.lens;
  const { widthM: W, heightM: H } = layout.screen;
  const { visible: f, lensCentreM } = eye;

  const count = cols * rows;
  const positions = new Float32Array(count * 2);
  const uvs = new Float32Array(count * 2);
  const shade = new Float32Array(count);

  for (let j = 0; j < rows; j += 1) {
    const v = j / (rows - 1);
    for (let i = 0; i < cols; i += 1) {
      const u = i / (cols - 1);
      const tx = f.left + (f.right - f.left) * u;
      const ty = f.bottom + (f.top - f.bottom) * v;
      const d = Math.hypot(tx, ty);
      // The lens is radial: solve along the radius, keep the direction.
      const ratio = d > 1e-12 ? distortInverse(d, k1, k2) / d : 1;
      const sx = tx * ratio;
      const sy = ty * ratio;
      const xM = lensCentreM[0] + sx * D;
      const yM = lensCentreM[1] + sy * D;

      const k = j * cols + i;
      positions[k * 2] = (xM / W) * 2 - 1;
      positions[k * 2 + 1] = (yM / H) * 2 - 1;
      uvs[k * 2] = u;
      uvs[k * 2 + 1] = options.vOrigin === "bottom" ? v : 1 - v;
      const edge = Math.min(u, 1 - u, v, 1 - v);
      shade[k] = fade > 0 ? smoothstep(0, fade, edge) : 1;
    }
  }

  const indices = new Uint16Array((cols - 1) * (rows - 1) * 6);
  let n = 0;
  for (let j = 0; j < rows - 1; j += 1) {
    for (let i = 0; i < cols - 1; i += 1) {
      const a = j * cols + i;
      const b = a + 1;
      const c = a + cols;
      const e = c + 1;
      // Counter-clockwise in a y-up NDC frame, so front faces are the ones we look at.
      indices[n++] = a;
      indices[n++] = b;
      indices[n++] = e;
      indices[n++] = a;
      indices[n++] = e;
      indices[n++] = c;
    }
  }

  return { positions, uvs, shade, indices };
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

// ---------------------------------------------------------------------------------
// Render target size
// ---------------------------------------------------------------------------------

export interface EyeTextureSizeOptions {
  /** Canvas size in device pixels. */
  canvasWidthPx: number;
  /** 1 matches the screen's own pixel density at the lens centre. */
  quality: number;
  maxTextureSize: number;
  /** Ceiling on pixels per eye, so a dense phone cannot ask for a texture it cannot fill. */
  maxPixels: number;
}

/**
 * How big each eye's render target needs to be.
 *
 * The distortion is a magnification of exactly 1 at the lens centre (the derivative of the
 * lens function there is 1) and a compression everywhere else, so matching the screen's own
 * pixel density AT THE CENTRE is the most any part of the image can use. That density is the
 * screen's pixels per metre times the lens distance, per unit of tan-angle.
 *
 * Measured on an iPhone 14 in landscape with the canvas at the player's 2x pixel-ratio cap
 * (1688 px across 141 mm): the full-quality eye is 1374 × 1176 (1.6 MP) behind Cardboard v2
 * and 1098 × 980 (1.1 MP) behind the VR box profile, against the 844 × 780 (0.66 MP) of screen
 * each eye is squeezed into — the lens compresses the periphery, so the texture holds 1.6-2.5x
 * the pixels it lands on. Hence `quality` and the pixel ceiling: the outer ring of the texture
 * is seen through the most compressed part of the lens, and paying for it at centre density
 * buys the walkthrough nothing it can afford. (The 360 sphere can afford it; it asks for 1.)
 */
export function eyeTextureSize(layout: StereoLayout, eye: EyeLayout, options: EyeTextureSizeOptions): { width: number; height: number } {
  const pxPerMetre = options.canvasWidthPx / layout.screen.widthM;
  const pxPerTan = layout.lens.screenLensDistance * pxPerMetre * options.quality;
  let width = (eye.visible.right - eye.visible.left) * pxPerTan;
  let height = (eye.visible.top - eye.visible.bottom) * pxPerTan;
  const pixels = width * height;
  if (pixels > options.maxPixels) {
    const k = Math.sqrt(options.maxPixels / pixels);
    width *= k;
    height *= k;
  }
  const cap = (x: number) => Math.max(64, Math.min(options.maxTextureSize, Math.round(x / 2) * 2));
  return { width: cap(width), height: cap(height) };
}

// ---------------------------------------------------------------------------------
// Field overrides
// ---------------------------------------------------------------------------------

/**
 * `?vrlens=k1,k2,ipdMm,screenMm,fovDeg` — tune a headset in the field without a build.
 *
 * Every field is optional and positional (`?vrlens=0.3,0.2` changes the distortion only).
 * Anything out of a physically plausible range is ignored rather than clamped: a typo should
 * fall back to the profile, not produce a subtly wrong one that somebody then wears.
 */
export function parseLensOverride(raw: string | null | undefined, base: LensProfile): LensProfile {
  if (!raw) return base;
  // An empty field means "keep the profile's", and `Number("")` is 0 — a valid k1 — so empty
  // strings have to become NaN explicitly or `,,66` would zero the distortion.
  const parts = raw.split(",").map((part) => (part.trim() === "" ? Number.NaN : Number(part.trim())));
  const pick = (index: number, min: number, max: number): number | null => {
    const value = parts[index];
    return parts.length > index && Number.isFinite(value) && value >= min && value <= max ? value : null;
  };
  const k1 = pick(0, -0.2, 1.5);
  const k2 = pick(1, -0.2, 1.5);
  const ipd = pick(2, 50, 80);
  const screen = pick(3, 25, 70);
  const fov = pick(4, 30, 75);
  return {
    ...base,
    k1: k1 ?? base.k1,
    k2: k2 ?? base.k2,
    interLensDistance: ipd !== null ? ipd / 1000 : base.interLensDistance,
    screenLensDistance: screen !== null ? screen / 1000 : base.screenLensDistance,
    fovDeg: fov ?? base.fovDeg,
  };
}
