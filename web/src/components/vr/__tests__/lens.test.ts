/**
 * The Cardboard lens model, checked against the property that makes it correct: a pixel drawn
 * at screen tan-angle `s` is seen through the lens at `distort(s)`, so the direction the mesh
 * samples at every vertex must be exactly what the lens shows at that vertex's screen position.
 * Get that wrong and the room swims as the head turns — the classic Cardboard nausea — while
 * every individual number still looks plausible.
 */

import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  LENS_PROFILES,
  distort,
  distortInverse,
  distortionMesh,
  eyeTextureSize,
  parseLensOverride,
  phoneEquivalentDensity,
  screenFromCss,
  stereoLayout,
  type LensProfile,
} from "../lens";

// iPhone 14 in landscape: 844 × 390 CSS px.
const IPHONE = screenFromCss(844, 390);
const PROFILES: LensProfile[] = Object.values(LENS_PROFILES);

describe("lens function", () => {
  it("round-trips through its inverse across the whole field", () => {
    for (const lens of PROFILES) {
      for (let s = -1.6; s <= 1.6; s += 0.01) {
        const t = distort(s, lens.k1, lens.k2);
        expect(distortInverse(t, lens.k1, lens.k2), `${lens.id} s=${s.toFixed(2)}`).toBeCloseTo(s, 9);
      }
    }
  });

  it("magnifies outward and never folds back for the shipped profiles", () => {
    for (const lens of PROFILES) {
      let previous = -Infinity;
      for (let s = 0; s <= 2; s += 0.01) {
        const t = distort(s, lens.k1, lens.k2);
        expect(t).toBeGreaterThanOrEqual(s);
        expect(t).toBeGreaterThan(previous);
        previous = t;
      }
    }
  });

  it("is the identity at the lens centre", () => {
    expect(distort(0, 0.34, 0.55)).toBe(0);
    expect(distortInverse(0, 0.34, 0.55)).toBe(0);
    // The derivative at the centre is 1: a small angle is not magnified.
    expect(distort(1e-4, 0.34, 0.55) / 1e-4).toBeCloseTo(1, 6);
  });

  it("survives a hand-entered profile with a negative coefficient", () => {
    for (let t = 0; t <= 1.2; t += 0.05) {
      const s = distortInverse(t, -0.08, 0.3);
      expect(distort(s, -0.08, 0.3)).toBeCloseTo(t, 9);
    }
  });
});

describe("stereo layout", () => {
  it("puts each lens half the lens spacing either side of the middle", () => {
    for (const lens of PROFILES) {
      const layout = stereoLayout(lens, IPHONE);
      const [left, right] = layout.eyes;
      expect(right.lensCentreM[0] - left.lensCentreM[0]).toBeCloseTo(lens.interLensDistance, 12);
      expect((left.lensCentreM[0] + right.lensCentreM[0]) / 2).toBeCloseTo(IPHONE.widthM / 2, 12);
      expect(right.eyeOffsetM).toBeCloseTo(-left.eyeOffsetM, 12);
    }
  });

  it("mirrors the right eye's frustum from the left's", () => {
    for (const lens of PROFILES) {
      const [left, right] = stereoLayout(lens, IPHONE).eyes;
      expect(right.visible.left).toBeCloseTo(-left.visible.right, 12);
      expect(right.visible.right).toBeCloseTo(-left.visible.left, 12);
      expect(right.visible.top).toBe(left.visible.top);
      expect(right.visible.bottom).toBe(left.visible.bottom);
    }
  });

  it("sees further outward than inward, because the lens sits inboard of the half-screen's middle", () => {
    // A 140 mm screen gives each eye 70 mm; a 62-64 mm lens spacing puts each lens centre
    // 3-4 mm towards the nose from the middle of its half. The temple side is therefore the
    // longer one — the off-axis frustum a symmetric projection would get wrong.
    for (const lens of PROFILES) {
      const [left] = stereoLayout(lens, IPHONE).eyes;
      expect(-left.visible.left).toBeGreaterThan(left.visible.right);
    }
  });

  it("never shows more than the lens's own field", () => {
    for (const lens of PROFILES) {
      const tanFov = Math.tan((lens.fovDeg * Math.PI) / 180);
      const [left] = stereoLayout(lens, IPHONE).eyes;
      for (const edge of Object.values(left.visible)) expect(Math.abs(edge)).toBeLessThanOrEqual(tanFov + 1e-12);
    }
  });

  it("gives a Cardboard v2 on an iPhone 14 the field the hardware is known for", () => {
    // About 100 degrees each way is what the v2 is sold as; this pins the arithmetic to it.
    const [left] = stereoLayout(LENS_PROFILES.cardboard, IPHONE).eyes;
    const horizontal = (Math.atan(-left.visible.left) + Math.atan(left.visible.right)) * (180 / Math.PI);
    const vertical = (Math.atan(-left.visible.bottom) + Math.atan(left.visible.top)) * (180 / Math.PI);
    expect(horizontal).toBeGreaterThan(95);
    expect(horizontal).toBeLessThan(115);
    expect(vertical).toBeGreaterThan(85);
    expect(vertical).toBeLessThan(110);
  });
});

describe("distortion mesh", () => {
  const COLS = 40;

  it("shows, at every vertex, exactly the direction the lens shows at that screen point", () => {
    for (const lens of PROFILES) {
      const layout = stereoLayout(lens, IPHONE);
      for (const eye of layout.eyes) {
        const mesh = distortionMesh(layout, eye, { vOrigin: "bottom", cols: COLS, rows: COLS });
        const f = eye.visible;
        for (let k = 0; k < mesh.uvs.length / 2; k += 1) {
          // Where on the screen the vertex is, as a screen tan-angle from the lens centre...
          const xM = ((mesh.positions[k * 2] + 1) / 2) * IPHONE.widthM;
          const yM = ((mesh.positions[k * 2 + 1] + 1) / 2) * IPHONE.heightM;
          const sx = (xM - eye.lensCentreM[0]) / lens.screenLensDistance;
          const sy = (yM - eye.lensCentreM[1]) / lens.screenLensDistance;
          const s = Math.hypot(sx, sy);
          const seen = s > 1e-12 ? distort(s, lens.k1, lens.k2) / s : 1;
          // ...and what the texture holds there.
          const tx = f.left + (f.right - f.left) * mesh.uvs[k * 2];
          const ty = f.bottom + (f.top - f.bottom) * mesh.uvs[k * 2 + 1];
          // The mesh is Float32 (it goes straight into a vertex buffer), so this is as tight
          // as storage allows: 5e-6 of a tan-angle is 0.0003 degrees.
          expect(sx * seen).toBeCloseTo(tx, 5);
          expect(sy * seen).toBeCloseTo(ty, 5);
        }
      }
    }
  });

  it("keeps each eye inside its own half of the screen", () => {
    for (const lens of PROFILES) {
      const layout = stereoLayout(lens, IPHONE);
      const [left, right] = layout.eyes;
      const a = distortionMesh(layout, left, { vOrigin: "bottom" });
      const b = distortionMesh(layout, right, { vOrigin: "bottom" });
      for (let k = 0; k < a.positions.length; k += 2) {
        expect(a.positions[k]).toBeLessThanOrEqual(1e-9);
        expect(a.positions[k]).toBeGreaterThanOrEqual(-1 - 1e-9);
        expect(b.positions[k]).toBeGreaterThanOrEqual(-1e-9);
        expect(b.positions[k]).toBeLessThanOrEqual(1 + 1e-9);
        expect(Math.abs(a.positions[k + 1])).toBeLessThanOrEqual(1 + 1e-9);
      }
    }
  });

  it("is the mirror image between the eyes", () => {
    const layout = stereoLayout(LENS_PROFILES.vrbox, IPHONE);
    const a = distortionMesh(layout, layout.eyes[0], { vOrigin: "bottom", cols: 9, rows: 7 });
    const b = distortionMesh(layout, layout.eyes[1], { vOrigin: "bottom", cols: 9, rows: 7 });
    for (let j = 0; j < 7; j += 1) {
      for (let i = 0; i < 9; i += 1) {
        const ka = j * 9 + i;
        const kb = j * 9 + (8 - i);
        expect(b.positions[kb * 2]).toBeCloseTo(-a.positions[ka * 2], 9);
        expect(b.positions[kb * 2 + 1]).toBeCloseTo(a.positions[ka * 2 + 1], 9);
      }
    }
  });

  it("flips v and nothing else for an image-origin texture", () => {
    const layout = stereoLayout(LENS_PROFILES.cardboard, IPHONE);
    const up = distortionMesh(layout, layout.eyes[0], { vOrigin: "bottom", cols: 5, rows: 5 });
    const down = distortionMesh(layout, layout.eyes[0], { vOrigin: "top", cols: 5, rows: 5 });
    for (let k = 0; k < up.uvs.length; k += 2) {
      expect(down.uvs[k]).toBe(up.uvs[k]);
      expect(down.uvs[k + 1]).toBeCloseTo(1 - up.uvs[k + 1], 12);
      expect(down.positions[k]).toBe(up.positions[k]);
    }
  });

  it("fades only at the rim", () => {
    const layout = stereoLayout(LENS_PROFILES.vrbox, IPHONE);
    const mesh = distortionMesh(layout, layout.eyes[0], { vOrigin: "bottom", cols: 41, rows: 41 });
    expect(mesh.shade[20 * 41 + 20]).toBe(1);
    expect(mesh.shade[0]).toBe(0);
    expect(mesh.shade[20 * 41]).toBe(0);
  });

  it("winds every triangle the same way, facing the viewer", () => {
    const layout = stereoLayout(LENS_PROFILES.cardboard, IPHONE);
    const mesh = distortionMesh(layout, layout.eyes[0], { vOrigin: "bottom" });
    for (let n = 0; n < mesh.indices.length; n += 3) {
      const [a, b, c] = [mesh.indices[n], mesh.indices[n + 1], mesh.indices[n + 2]];
      const ax = mesh.positions[a * 2];
      const ay = mesh.positions[a * 2 + 1];
      const cross =
        (mesh.positions[b * 2] - ax) * (mesh.positions[c * 2 + 1] - ay) -
        (mesh.positions[b * 2 + 1] - ay) * (mesh.positions[c * 2] - ax);
      expect(cross).toBeGreaterThan(0);
    }
  });
});

describe("projection", () => {
  it("puts the frustum's edges exactly on the edges of the eye's texture", () => {
    const [left] = stereoLayout(LENS_PROFILES.cardboard, IPHONE).eyes;
    const f = left.visible;
    const near = 0.05;
    const m = new THREE.Matrix4().makePerspective(f.left * near, f.right * near, f.top * near, f.bottom * near, near, 500);
    const at = (tx: number, ty: number) => new THREE.Vector3(tx, ty, -1).applyMatrix4(m);
    expect(at(f.left, 0).x).toBeCloseTo(-1, 9);
    expect(at(f.right, 0).x).toBeCloseTo(1, 9);
    expect(at(0, f.top).y).toBeCloseTo(1, 9);
    expect(at(0, f.bottom).y).toBeCloseTo(-1, 9);
  });
});

describe("eye texture size", () => {
  it("scales with quality and stays inside the pixel budget", () => {
    const layout = stereoLayout(LENS_PROFILES.vrbox, IPHONE);
    const eye = layout.eyes[0];
    const full = eyeTextureSize(layout, eye, { canvasWidthPx: 2532, quality: 1, maxTextureSize: 16384, maxPixels: 1e9 });
    const half = eyeTextureSize(layout, eye, { canvasWidthPx: 2532, quality: 0.5, maxTextureSize: 16384, maxPixels: 1e9 });
    expect(half.width).toBeCloseTo(full.width / 2, -1);
    const capped = eyeTextureSize(layout, eye, { canvasWidthPx: 2532, quality: 1, maxTextureSize: 16384, maxPixels: 1.5e6 });
    expect(capped.width * capped.height).toBeLessThanOrEqual(1.5e6 * 1.01);
    // The aspect survives the cap.
    expect(capped.width / capped.height).toBeCloseTo(full.width / full.height, 1);
  });

  it("never exceeds the GPU's texture limit", () => {
    const layout = stereoLayout(LENS_PROFILES.cardboard, IPHONE);
    const size = eyeTextureSize(layout, layout.eyes[0], { canvasWidthPx: 9000, quality: 1, maxTextureSize: 2048, maxPixels: 1e9 });
    expect(size.width).toBeLessThanOrEqual(2048);
    expect(size.height).toBeLessThanOrEqual(2048);
  });
});

describe("field overrides", () => {
  it("replaces only what was given, positionally", () => {
    const base = LENS_PROFILES.vrbox;
    const tuned = parseLensOverride("0.3,0.2", base);
    expect(tuned.k1).toBe(0.3);
    expect(tuned.k2).toBe(0.2);
    expect(tuned.interLensDistance).toBe(base.interLensDistance);
    const ipd = parseLensOverride(",,66,40,55", base);
    expect(ipd.k1).toBe(base.k1);
    expect(ipd.interLensDistance).toBeCloseTo(0.066, 12);
    expect(ipd.screenLensDistance).toBeCloseTo(0.04, 12);
    expect(ipd.fovDeg).toBe(55);
  });

  it("ignores nonsense instead of clamping it into something plausible", () => {
    const base = LENS_PROFILES.cardboard;
    expect(parseLensOverride("abc,9,500", base)).toEqual(base);
    expect(parseLensOverride("", base)).toEqual(base);
    expect(parseLensOverride(null, base)).toEqual(base);
  });
});

describe("screen size", () => {
  it("lands an iPhone 14 within a few millimetres of its real 141 x 65 mm panel", () => {
    expect(IPHONE.widthM * 1000).toBeGreaterThan(136);
    expect(IPHONE.widthM * 1000).toBeLessThan(146);
    expect(IPHONE.heightM * 1000).toBeGreaterThan(62);
    expect(IPHONE.heightM * 1000).toBeLessThan(68);
  });

  it("is orientation-proof", () => {
    expect(screenFromCss(390, 844)).toEqual(screenFromCss(844, 390));
  });

  it("makes a desktop window behave like a phone in the debug mode", () => {
    const desk = screenFromCss(1920, 1080, phoneEquivalentDensity(1920, 1080));
    expect(desk.widthM * 1000).toBeCloseTo(140.5, 0);
  });
});
