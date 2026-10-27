/**
 * PlayCanvas building blocks for the walkthrough's headset modes: canvas-textured quads, the
 * menu panel, the floor marker, the comfort vignette and the fade.
 *
 * `pc` is always passed in — the engine's own module instance — and never imported, so this
 * chunk stays out of every bundle until VR starts and can never end up with a second copy of
 * PlayCanvas whose classes the engine's scene does not recognise.
 *
 * ── COLOUR ────────────────────────────────────────────────────────────────────────────────
 * Canvas textures are `PIXELFORMAT_SRGBA8`: the sampler decodes them, the camera's gamma pass
 * encodes once on the way out, and the interface lands on screen in exactly the colours
 * `canvasUi` drew. Plain `RGBA8` would be read as linear and come out washed pale.
 *
 * ── ORIENTATION ───────────────────────────────────────────────────────────────────────────
 * PlayCanvas uploads canvases unflipped, so row 0 — the canvas's top — is v = 0. The quads
 * here are built by hand with that in mind rather than borrowed from the `plane` primitive,
 * whose UV layout is one more thing to get upside down. Every quad faces +Z, drawn in the UI
 * layer (last, over the splats) without depth testing, ordered by `drawOrder`.
 */

import type * as PC from "playcanvas";

export type Pc = typeof PC;

export function canvasTexture(pc: Pc, device: PC.GraphicsDevice, canvas: HTMLCanvasElement, name: string): PC.Texture {
  const texture = new pc.Texture(device, {
    name,
    width: canvas.width,
    height: canvas.height,
    format: pc.PIXELFORMAT_SRGBA8,
    mipmaps: true,
    minFilter: pc.FILTER_LINEAR_MIPMAP_LINEAR,
    magFilter: pc.FILTER_LINEAR,
    addressU: pc.ADDRESS_CLAMP_TO_EDGE,
    addressV: pc.ADDRESS_CLAMP_TO_EDGE,
    anisotropy: 4,
  });
  texture.setSource(canvas);
  return texture;
}

/**
 * Re-uploads a canvas texture after the canvas was redrawn.
 *
 * Not `setSource` again: PlayCanvas only marks a level dirty when the source OBJECT changes, so
 * handing it the same canvas uploads nothing and the menu would keep showing its first frame.
 * `dirtyAll` is the public way to say the pixels changed (and to regenerate the mipmaps).
 */
export function refreshTexture(texture: PC.Texture): void {
  texture.dirtyAll();
}

/** A width × height quad facing +Z, v = 0 along its top edge. */
export function quadMesh(pc: Pc, device: PC.GraphicsDevice, width: number, height: number): PC.Mesh {
  const hw = width / 2;
  const hh = height / 2;
  const mesh = new pc.Mesh(device);
  mesh.setPositions([-hw, -hh, 0, hw, -hh, 0, hw, hh, 0, -hw, hh, 0]);
  mesh.setNormals([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
  mesh.setUvs(0, [0, 1, 1, 1, 1, 0, 0, 0]);
  mesh.setIndices([0, 1, 2, 0, 2, 3]);
  mesh.update(pc.PRIMITIVE_TRIANGLES);
  return mesh;
}

/** Unlit, alpha-blended, never depth-tested: interface, not scenery. */
export function overlayMaterial(pc: Pc, texture: PC.Texture | null, colour?: PC.Color): PC.StandardMaterial {
  const material = new pc.StandardMaterial();
  material.useLighting = false;
  material.useSkybox = false;
  material.useFog = false;
  material.useTonemap = false;
  material.diffuse = new pc.Color(0, 0, 0);
  material.emissive = colour ?? new pc.Color(1, 1, 1);
  if (texture) {
    material.emissiveMap = texture;
    material.opacityMap = texture;
    material.opacityMapChannel = "a";
  }
  material.blendType = pc.BLEND_NORMAL;
  material.depthWrite = false;
  material.depthTest = false;
  material.cull = pc.CULLFACE_NONE;
  material.update();
  return material;
}

export interface Overlay {
  entity: PC.Entity;
  material: PC.StandardMaterial;
  instance: PC.MeshInstance;
  texture: PC.Texture | null;
  setOpacity(opacity: number): void;
  destroy(): void;
}

/** A quad entity in the UI layer. `drawOrder` stacks overlays: higher draws later. */
export function makeOverlay(
  pc: Pc,
  app: PC.AppBase,
  options: { name: string; width: number; height: number; texture: PC.Texture | null; colour?: PC.Color; drawOrder: number },
): Overlay {
  const device = app.graphicsDevice;
  const mesh = quadMesh(pc, device, options.width, options.height);
  const material = overlayMaterial(pc, options.texture, options.colour);
  const instance = new pc.MeshInstance(mesh, material);
  instance.drawOrder = options.drawOrder;
  instance.cull = false;
  const entity = new pc.Entity(options.name);
  entity.addComponent("render", { meshInstances: [instance], layers: [pc.LAYERID_UI] });
  let current = 1;
  return {
    entity,
    material,
    instance,
    texture: options.texture,
    setOpacity(opacity: number) {
      if (Math.abs(opacity - current) < 1e-3) return;
      current = opacity;
      material.opacity = opacity;
      material.update();
      entity.enabled = opacity > 0.002;
    },
    destroy() {
      entity.destroy();
      material.destroy();
      mesh.destroy();
      options.texture?.destroy();
    },
  };
}

// ---------------------------------------------------------------------------------
// Canvases for the floor marker and the vignette
// ---------------------------------------------------------------------------------

/**
 * The landing ring: where a teleport will put the visitor, drawn on the floor — teal where it
 * may land, amber where the corridor refuses it — with the gaze dwell as an arc around it.
 */
export function drawMarker(canvas: HTMLCanvasElement, valid: boolean, progress: number): void {
  const S = 256;
  if (canvas.width !== S) canvas.width = S;
  if (canvas.height !== S) canvas.height = S;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, S, S);
  const colour = valid ? "#17e2c5" : "#ffb44c";
  ctx.lineWidth = 14;
  ctx.strokeStyle = colour;
  ctx.globalAlpha = valid ? 0.95 : 0.7;
  ctx.beginPath();
  ctx.arc(S / 2, S / 2, S * 0.36, 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalAlpha = valid ? 0.22 : 0.12;
  ctx.fillStyle = colour;
  ctx.beginPath();
  ctx.arc(S / 2, S / 2, S * 0.33, 0, Math.PI * 2);
  ctx.fill();
  if (!valid) {
    // A refusal says why without words: a bar across the ring.
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 12;
    ctx.beginPath();
    ctx.moveTo(S * 0.3, S * 0.3);
    ctx.lineTo(S * 0.7, S * 0.7);
    ctx.stroke();
  }
  if (progress > 0) {
    ctx.globalAlpha = 1;
    ctx.lineWidth = 16;
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(S / 2, S / 2, S * 0.45, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, progress));
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

/** The gaze cursor, with its dwell arc. */
export function drawReticle(canvas: HTMLCanvasElement, active: boolean, progress: number): void {
  const S = 128;
  if (canvas.width !== S) canvas.width = S;
  if (canvas.height !== S) canvas.height = S;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, S, S);
  const r = active ? S * 0.3 : S * 0.22;
  ctx.lineWidth = 12;
  ctx.strokeStyle = "rgba(0, 0, 0, 0.45)";
  ctx.beginPath();
  ctx.arc(S / 2, S / 2, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 7;
  ctx.strokeStyle = active ? "#17e2c5" : "#ffffff";
  ctx.beginPath();
  ctx.arc(S / 2, S / 2, r, 0, Math.PI * 2);
  ctx.stroke();
  if (progress > 0) {
    ctx.lineWidth = 9;
    ctx.strokeStyle = "#17e2c5";
    ctx.beginPath();
    ctx.arc(S / 2, S / 2, S * 0.42, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, progress));
    ctx.stroke();
  }
}

/**
 * The comfort vignette for smooth movement: clear in the middle, dark at the edges.
 *
 * Moving through a scene with the stick is the one thing here that is real vection — visual
 * motion the inner ear does not feel — and peripheral vision is where vection is sensed. Dimming
 * the periphery only while moving is the standard mitigation (Meta's comfort guidelines, every
 * major title's "tunnelling" option); the clear centre keeps the destination in view.
 */
export function drawVignette(canvas: HTMLCanvasElement): void {
  const S = 512;
  canvas.width = S;
  canvas.height = S;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const gradient = ctx.createRadialGradient(S / 2, S / 2, S * 0.18, S / 2, S / 2, S * 0.5);
  gradient.addColorStop(0, "rgba(0, 0, 0, 0)");
  gradient.addColorStop(0.55, "rgba(0, 0, 0, 0.55)");
  gradient.addColorStop(1, "rgba(0, 0, 0, 1)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, S, S);
}
