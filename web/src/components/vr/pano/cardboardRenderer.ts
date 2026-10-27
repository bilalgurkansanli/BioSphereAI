/**
 * Split-screen stereo for a phone in a VR box, in three.js: each eye rendered undistorted into
 * its own texture, then carried to the screen by the barrel-distortion mesh from `lens.ts`.
 *
 * ── COLOUR ────────────────────────────────────────────────────────────────────────────────
 * The eye targets are `SRGB8_ALPHA8` (three picks it for an 8-bit target whose texture is
 * tagged sRGB). three writes LINEAR values into any render target that is not the screen, and
 * 8-bit linear storage bands visibly in the dark half of a photograph — the shadows under a
 * sofa step. An sRGB target stores the same values perceptually, with the hardware encoding on
 * write and decoding on read, so the composite sees linear light and the final pass encodes
 * once for the screen: no banding, no double gamma, no tone mapping anywhere (`toneMapped`
 * false on the composite, as the player itself never tone-maps a photograph).
 *
 * ── MONO OR STEREO ────────────────────────────────────────────────────────────────────────
 * The eyes are rendered from the same point for the panorama — a single photograph has no
 * parallax to show, and an eye offset would only add a disparity every other depth cue in the
 * picture contradicts. The frusta still differ (each lens sees its own off-axis slice), so the
 * two renders are not interchangeable. `eyeSeparation` exists for content with real depth.
 */

import * as THREE from "three";
import {
  CSS_PX_PER_INCH,
  distortionMesh,
  eyeTextureSize,
  phoneEquivalentDensity,
  screenFromCss,
  stereoLayout,
  type LensProfile,
  type StereoLayout,
} from "../lens";

/** Near and far planes for the eyes; the panorama sphere sits at 100. */
const NEAR = 0.05;
const FAR = 1000;
/**
 * Pixel budget per eye. The sphere is cheap, so the 360 renders at the screen's own density
 * at the lens centre (quality 1) — 1.6 MP per eye on an iPhone 14 behind a Cardboard v2 — and
 * this ceiling only bites on unusually dense or large phones.
 */
const MAX_EYE_PIXELS = 2.4e6;

export class CardboardRenderer {
  private readonly gl: THREE.WebGLRenderer;
  private readonly composite = new THREE.Scene();
  private readonly ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly eyes: THREE.PerspectiveCamera[] = [new THREE.PerspectiveCamera(), new THREE.PerspectiveCamera()];
  private targets: THREE.WebGLRenderTarget[] = [];
  private meshes: THREE.Mesh[] = [];
  private layout: StereoLayout | null = null;
  private readonly offset = new THREE.Vector3();

  constructor(gl: THREE.WebGLRenderer) {
    this.gl = gl;
    for (const eye of this.eyes) eye.matrixAutoUpdate = true;
  }

  /**
   * Builds the layout, the targets and the meshes for this canvas and lens. Cheap enough to
   * call on every resize or lens change; nothing is reused across calls.
   */
  configure(lens: LensProfile, cssWidth: number, cssHeight: number, pixelRatio: number, debug: boolean, quality = 1): void {
    this.releaseTargets();
    const density = debug ? phoneEquivalentDensity(cssWidth, cssHeight) : CSS_PX_PER_INCH;
    const layout = stereoLayout(lens, screenFromCss(cssWidth, cssHeight, density));
    this.layout = layout;
    const maxTextureSize = this.gl.capabilities.maxTextureSize;

    layout.eyes.forEach((eye, index) => {
      const size = eyeTextureSize(layout, eye, {
        canvasWidthPx: Math.max(cssWidth, cssHeight) * pixelRatio,
        quality,
        maxTextureSize,
        maxPixels: MAX_EYE_PIXELS,
      });
      const target = new THREE.WebGLRenderTarget(size.width, size.height, {
        colorSpace: THREE.SRGBColorSpace,
        type: THREE.UnsignedByteType,
        depthBuffer: true,
        stencilBuffer: false,
        generateMipmaps: false,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
      });
      this.targets.push(target);

      const camera = this.eyes[index];
      const f = eye.visible;
      camera.projectionMatrix.makePerspective(f.left * NEAR, f.right * NEAR, f.top * NEAR, f.bottom * NEAR, NEAR, FAR);
      camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
      camera.near = NEAR;
      camera.far = FAR;

      const mesh = distortionMesh(layout, eye, { vOrigin: "bottom" });
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array((mesh.positions.length / 2) * 3);
      const colours = new Float32Array((mesh.positions.length / 2) * 3);
      for (let k = 0; k < mesh.positions.length / 2; k += 1) {
        positions[k * 3] = mesh.positions[k * 2];
        positions[k * 3 + 1] = mesh.positions[k * 2 + 1];
        colours[k * 3] = colours[k * 3 + 1] = colours[k * 3 + 2] = mesh.shade[k];
      }
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("uv", new THREE.BufferAttribute(mesh.uvs, 2));
      geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
      geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
      const material = new THREE.MeshBasicMaterial({
        map: target.texture,
        vertexColors: true,
        toneMapped: false,
        depthTest: false,
        depthWrite: false,
      });
      const quad = new THREE.Mesh(geometry, material);
      quad.frustumCulled = false;
      this.meshes.push(quad);
      this.composite.add(quad);
    });
  }

  /** Clears the screen to black, for the frames before the first head pose. */
  blank(): void {
    this.gl.setRenderTarget(null);
    this.gl.clear();
  }

  private readonly perf = { ms: 0, since: 0 };

  /**
   * A smoothed frame time, published as `window.__vrPerf` twice a second — the same habit as
   * the walkthrough's `__walkPerf`, so a phone in a box can be measured from a remote console.
   */
  track(delta: number, mode: string): void {
    const p = this.perf;
    const ms = Math.min(delta * 1000, 200);
    p.ms = p.ms === 0 ? ms : p.ms * 0.9 + ms * 0.1;
    p.since += delta;
    if (p.since < 0.5) return;
    p.since = 0;
    (window as unknown as { __vrPerf?: unknown }).__vrPerf = {
      mode,
      fps: Math.round(1000 / Math.max(p.ms, 1)),
      frameMs: +p.ms.toFixed(1),
      eyes: this.textureSizes(),
    };
  }

  /** The eye textures' sizes, for the record: `__vrPerf` reports them. */
  textureSizes(): Array<{ width: number; height: number }> {
    return this.targets.map((t) => ({ width: t.width, height: t.height }));
  }

  get configured(): boolean {
    return this.layout !== null;
  }

  /** Renders both eyes and composites them to the canvas. Call in place of the normal render. */
  render(scene: THREE.Scene, headPosition: THREE.Vector3, headQuaternion: THREE.Quaternion, eyeSeparation = 0): void {
    const layout = this.layout;
    if (!layout) return;
    const gl = this.gl;
    const previous = gl.getRenderTarget();
    layout.eyes.forEach((eye, index) => {
      const camera = this.eyes[index];
      camera.quaternion.copy(headQuaternion);
      this.offset.set(eye.eyeOffsetM * eyeSeparation, 0, 0).applyQuaternion(headQuaternion);
      camera.position.copy(headPosition).add(this.offset);
      camera.updateMatrixWorld();
      // r3f leaves `autoClear` on, so each render clears its own target.
      gl.setRenderTarget(this.targets[index]);
      gl.render(scene, camera);
    });
    gl.setRenderTarget(previous);
    gl.render(this.composite, this.ortho);
  }

  private releaseTargets(): void {
    for (const mesh of this.meshes) {
      this.composite.remove(mesh);
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    for (const target of this.targets) target.dispose();
    this.meshes = [];
    this.targets = [];
    this.layout = null;
  }

  dispose(): void {
    this.releaseTargets();
  }
}
