/**
 * The 3D walkthrough on a phone in a VR box (PlayCanvas): two eyes with a real interpupillary
 * offset — the model has depth, unlike a panorama — lens-corrected by the same distortion mesh
 * as the 360 (`lens.ts`), head-tracked by `deviceorientation`, moving by gaze.
 *
 * ── TWO EYES, ONE SORT POSITION ───────────────────────────────────────────────────────────
 * Each eye is its own PlayCanvas camera rendering into its own target, and PlayCanvas keeps a
 * splat sort PER CAMERA. With radial sorting (on for the session) a sort is only redone when a
 * camera's node moves; so the eye cameras' nodes stay at the middle of the head and the
 * interpupillary offset is applied in `calculateTransform` instead. Turning the head then moves
 * neither node and costs no sort at all — and the two eyes, 6 cm apart, would order the splats
 * identically anyway. Only walking (a teleport) re-sorts.
 *
 * ── THE TARGETS ───────────────────────────────────────────────────────────────────────────
 * `PIXELFORMAT_SRGBA8`, so the splats are stored perceptually (PlayCanvas skips its shader gamma
 * for an sRGB target and the hardware encodes), and `RENDERTARGET_ORIGIN_TOP`, so the texture
 * reads like an image on WebGL2 and WebGPU alike — the walkthrough prefers WebGPU on phones,
 * where the native row order is the opposite of WebGL's. The distortion mesh is built with
 * `vOrigin: "top"` to match. The composite camera has an identity projection, so the mesh's
 * normalised device coordinates go straight to the screen.
 *
 * ── COST ──────────────────────────────────────────────────────────────────────────────────
 * Two eyes are two splat renders at roughly half the screen each, plus a second sort's worth
 * of memory. The eye targets start at 0.7 of the lens-centre density and the quality follows
 * the frame time (`walkMath.nextRenderScale`'s thresholds: under 15 ms gives pixels back, over
 * 22 ms takes them), rebuilt at most every three seconds.
 */

import type * as PC from "playcanvas";
import { EYE_HEIGHT, FRAME_MS_FAST, FRAME_MS_SLOW, yawOf } from "@/components/walk/walkMath";
import { DWELL_DEFAULTS, createDwell, resetDwell, stepDwell } from "../dwell";
import { panelFacing, rayPanel, regionAt, type Panel, type Vec3 } from "../gaze";
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
import { MIN_TELEPORT, floorAt, rayToFloor, teleportSpace, validateTeleport, type TeleportSpace, type TeleportVerdict } from "../teleport";
import { PANEL_HEIGHT, PANEL_WIDTH, drawPanel, type PanelItem, type PanelRegionUv } from "../ui/canvasUi";
import type { OrientationFeed } from "../useCardboard";
import type { WalkVrHost } from "./host";
import { canvasTexture, drawMarker, drawReticle, makeOverlay, refreshTexture, type Overlay, type Pc } from "./pcKit";
import type { WalkVrCopy } from "./walkXr";

const DEG = Math.PI / 180;
const NEAR = 0.04;
const FAR = 400;
const PANEL_DISTANCE = 1.2;
const PANEL_PITCH = -40 * DEG;
const PANEL_WIDTH_M = 0.5;
const PANEL_FOLLOW = 32 * DEG;
const MARKER_SIZE = 0.8;
const RETICLE_RADIUS = 0.9 * DEG;
const GAZE_HOLD_M = 0.45;
/** The engine's DOM fade runs 380 ms; the move happens just after it is fully black. */
const FADE_MOVE_S = 0.4;
const FADE_BACK_S = 0.46;
const QUALITY_START = 0.7;
const QUALITY_MIN = 0.42;
const QUALITY_MAX = 0.95;
const MAX_EYE_PIXELS = 1.6e6;

export interface WalkCardboardOptions {
  feed: { readonly current: OrientationFeed };
  lens: LensProfile;
  debug: boolean;
  copy: WalkVrCopy;
  onExit: () => void;
}

interface Eye {
  entity: PC.Entity;
  texture: PC.Texture;
  target: PC.RenderTarget;
  offset: PC.Mat4;
  quad: PC.Entity;
  mesh: PC.Mesh;
  material: PC.StandardMaterial;
}

export class WalkCardboard {
  private readonly host: WalkVrHost;
  private readonly pc: Pc;
  private readonly app: PC.AppBase;
  private readonly options: WalkCardboardOptions;
  private lens: LensProfile;

  private head!: PC.Entity;
  private composite!: PC.Entity;
  private layer!: PC.Layer;
  private eyes: Eye[] = [];
  private layout: StereoLayout | null = null;
  private quality = QUALITY_START;
  private mainWasEnabled = true;
  private radialWas = false;

  private align: PC.Quat | null = null;
  /** The sensor's rotation this frame, before the yaw alignment. */
  private readonly sensor: PC.Quat;
  private space: TeleportSpace;
  private spaceKey: string | null;

  private readonly dwell = createDwell();
  private gazeAnchor: { id: number; verdict: TeleportVerdict } | null = null;
  private gazeAnchorId = 0;
  private gazeTarget: string | null = null;
  private moving: { t: number; verdict: TeleportVerdict; moved: boolean } | null = null;

  private readonly panelCanvas = document.createElement("canvas");
  private readonly markerCanvas = document.createElement("canvas");
  private readonly reticleCanvas = document.createElement("canvas");
  private panel!: Overlay;
  private marker!: Overlay;
  private reticle!: Overlay;
  private panelGeometry: Panel | null = null;
  private panelRegions: PanelRegionUv[] = [];
  private panelYaw = 0;
  private panelTargetYaw = 0;
  private panelPlaced = false;
  private panelDrawn = "";
  private markerDrawn = "";
  private reticleDrawn = "";

  private readonly perf = { ms: 0, since: 0, rebuiltAt: 0 };
  private readonly scratch: { q: PC.Quat; m: PC.Mat4 };
  private readonly cleanups: Array<() => void> = [];

  constructor(host: WalkVrHost, options: WalkCardboardOptions) {
    this.host = host;
    this.pc = host.pc;
    this.app = host.app;
    this.options = options;
    this.lens = options.lens;
    this.spaceKey = host.activeKey();
    this.space = teleportSpace(host.manifest, this.spaceKey);
    this.scratch = { q: new this.pc.Quat(), m: new this.pc.Mat4() };
    this.sensor = new this.pc.Quat();
  }

  attach(): void {
    const { pc, app, host } = this;
    host.stopTour();
    // From here the engine leaves the camera to us (`WalkVrHost.driven`).
    host.driven = true;
    this.radialWas = app.scene.gsplat.radialSorting;
    app.scene.gsplat.radialSorting = true;

    const start = host.pose.position;
    this.head = new pc.Entity("vr-head");
    this.head.setPosition(start[0], floorAt(this.space, start[0], start[2]) + EYE_HEIGHT, start[2]);
    app.root.addChild(this.head);

    this.layer = new pc.Layer({ name: "vr-composite" });
    app.scene.layers.push(this.layer);

    this.composite = new pc.Entity("vr-composite-camera");
    this.composite.addComponent("camera", {
      clearColor: new pc.Color(0, 0, 0, 1),
      layers: [this.layer.id],
      priority: 50,
      nearClip: 0.01,
      farClip: 10,
      toneMapping: pc.TONEMAP_NONE,
    });
    // The mesh is already in normalised device coordinates for the whole canvas.
    this.composite.camera!.calculateProjection = (matrix: PC.Mat4) => {
      matrix.setIdentity();
    };
    app.root.addChild(this.composite);

    const main = host.camera.camera;
    this.mainWasEnabled = main?.enabled ?? true;
    if (main) main.enabled = false;

    this.buildEyes();
    this.buildUi();

    const onUpdate = (dt: number) => this.update(dt);
    app.on("update", onUpdate);
    this.cleanups.push(() => app.off("update", onUpdate));
    const onResize = () => this.buildEyes();
    window.addEventListener("resize", onResize);
    this.cleanups.push(() => window.removeEventListener("resize", onResize));
  }

  detach(): void {
    for (const fn of this.cleanups.splice(0)) fn();
    const { app, host } = this;
    this.destroyEyes();
    for (const overlay of [this.panel, this.marker, this.reticle]) overlay?.destroy();
    this.composite?.destroy();
    if (this.layer) app.scene.layers.remove(this.layer);
    const main = host.camera.camera;
    if (main) main.enabled = this.mainWasEnabled;
    app.scene.gsplat.radialSorting = this.radialWas;
    if (this.head) {
      // Hand the walk back where the head stood, facing where it looked, eyes level.
      const p = this.head.getPosition();
      host.pose.position = [p.x, EYE_HEIGHT, p.z];
      const f = this.head.forward;
      host.pose.yaw = yawOf([f.x, f.y, f.z]);
      host.pose.pitch = 0;
      this.head.destroy();
    }
    host.fade(0);
    host.driven = false;
  }

  setLens(lens: LensProfile): void {
    this.lens = lens;
    if (this.head) this.buildEyes();
  }

  /** The Cardboard's button, a screen tap: whatever the gaze rests on. */
  select(): boolean {
    if (!this.gazeTarget) return false;
    this.activate(this.gazeTarget);
    return true;
  }

  // ---------------------------------------------------------------------------------
  // Eyes
  // ---------------------------------------------------------------------------------

  private buildEyes(): void {
    const { pc, app } = this;
    this.destroyEyes();
    const canvas = app.graphicsDevice.canvas as HTMLCanvasElement;
    const cssWidth = canvas.clientWidth || window.innerWidth;
    const cssHeight = canvas.clientHeight || window.innerHeight;
    const density = this.options.debug ? phoneEquivalentDensity(cssWidth, cssHeight) : CSS_PX_PER_INCH;
    const layout = stereoLayout(this.lens, screenFromCss(cssWidth, cssHeight, density));
    this.layout = layout;
    const device = app.graphicsDevice;
    const main = this.host.camera.camera;
    const layers = (main?.layers ?? [pc.LAYERID_WORLD, pc.LAYERID_DEPTH, pc.LAYERID_SKYBOX, pc.LAYERID_UI, pc.LAYERID_IMMEDIATE]).filter(
      (id) => id !== this.layer.id,
    );
    const clear = main?.clearColor ?? new pc.Color(0.02, 0.027, 0.039, 1);

    layout.eyes.forEach((eye, index) => {
      const size = eyeTextureSize(layout, eye, {
        canvasWidthPx: Math.max(device.width, device.height),
        quality: this.quality,
        maxTextureSize: device.maxTextureSize,
        maxPixels: MAX_EYE_PIXELS,
      });
      const texture = new pc.Texture(device, {
        name: `vr-eye-${index}`,
        width: size.width,
        height: size.height,
        format: pc.PIXELFORMAT_SRGBA8,
        mipmaps: false,
        minFilter: pc.FILTER_LINEAR,
        magFilter: pc.FILTER_LINEAR,
        addressU: pc.ADDRESS_CLAMP_TO_EDGE,
        addressV: pc.ADDRESS_CLAMP_TO_EDGE,
      });
      const target = new pc.RenderTarget({ colorBuffer: texture, depth: true, origin: pc.RENDERTARGET_ORIGIN_TOP });
      const offset = new pc.Mat4().setTranslate(eye.eyeOffsetM, 0, 0);

      const entity = new pc.Entity(`vr-eye-${index}`);
      entity.addComponent("camera", {
        clearColor: clear,
        layers,
        priority: -20 + index,
        nearClip: NEAR,
        farClip: FAR,
        toneMapping: pc.TONEMAP_NONE,
        renderTarget: target,
      });
      const f = eye.visible;
      const camera = entity.camera!;
      camera.calculateProjection = (matrix: PC.Mat4) => {
        matrix.setFrustum(f.left * NEAR, f.right * NEAR, f.bottom * NEAR, f.top * NEAR, NEAR, FAR);
      };
      // The node stays at the head's centre (see the header); the eye's offset lives here.
      camera.calculateTransform = (matrix: PC.Mat4) => {
        matrix.mul2(this.head.getWorldTransform(), offset);
      };
      this.head.addChild(entity);

      // The distortion mesh for this eye, sampling its target.
      const built = distortionMesh(layout, eye, { vOrigin: "top" });
      const count = built.positions.length / 2;
      const positions = new Float32Array(count * 3);
      const colours = new Float32Array(count * 4);
      for (let k = 0; k < count; k += 1) {
        positions[k * 3] = built.positions[k * 2];
        positions[k * 3 + 1] = built.positions[k * 2 + 1];
        positions[k * 3 + 2] = 0;
        colours[k * 4] = colours[k * 4 + 1] = colours[k * 4 + 2] = built.shade[k];
        colours[k * 4 + 3] = 1;
      }
      const mesh = new pc.Mesh(device);
      mesh.setPositions(positions);
      mesh.setUvs(0, built.uvs);
      mesh.setColors(colours);
      mesh.setIndices(built.indices);
      mesh.update(pc.PRIMITIVE_TRIANGLES);
      const material = new pc.StandardMaterial();
      material.useLighting = false;
      material.useSkybox = false;
      material.useFog = false;
      material.useTonemap = false;
      material.diffuse = new pc.Color(0, 0, 0);
      material.emissive = new pc.Color(1, 1, 1);
      material.emissiveMap = texture;
      material.emissiveVertexColor = true;
      material.emissiveVertexColorChannel = "rgb";
      material.depthTest = false;
      material.depthWrite = false;
      material.cull = pc.CULLFACE_NONE;
      material.update();
      const instance = new pc.MeshInstance(mesh, material);
      instance.cull = false;
      const quad = new pc.Entity(`vr-eye-quad-${index}`);
      quad.addComponent("render", { meshInstances: [instance], layers: [this.layer.id] });
      // In front of the composite camera; with an identity projection only x and y matter.
      quad.setLocalPosition(0, 0, -0.5);
      this.composite.addChild(quad);

      this.eyes.push({ entity, texture, target, offset, quad, mesh, material });
    });
    this.perf.rebuiltAt = performance.now();
  }

  private destroyEyes(): void {
    for (const eye of this.eyes) {
      eye.entity.destroy();
      eye.quad.destroy();
      eye.material.destroy();
      eye.mesh.destroy();
      eye.target.destroy();
      eye.texture.destroy();
    }
    this.eyes = [];
  }

  // ---------------------------------------------------------------------------------
  // Interface
  // ---------------------------------------------------------------------------------

  private buildUi(): void {
    const { pc, app } = this;
    const device = app.graphicsDevice;
    drawPanel(this.panelCanvas, this.panelItems(), { hovered: null, progress: 0 });
    this.panel = makeOverlay(pc, app, {
      name: "vr-panel",
      width: PANEL_WIDTH_M,
      height: PANEL_WIDTH_M * (PANEL_HEIGHT / PANEL_WIDTH),
      texture: canvasTexture(pc, device, this.panelCanvas, "vr-panel"),
      drawOrder: 10,
    });
    app.root.addChild(this.panel.entity);
    drawMarker(this.markerCanvas, true, 0);
    this.marker = makeOverlay(pc, app, {
      name: "vr-marker",
      width: MARKER_SIZE,
      height: MARKER_SIZE,
      texture: canvasTexture(pc, device, this.markerCanvas, "vr-marker"),
      drawOrder: 5,
    });
    this.marker.entity.enabled = false;
    app.root.addChild(this.marker.entity);
    drawReticle(this.reticleCanvas, false, 0);
    this.reticle = makeOverlay(pc, app, {
      name: "vr-reticle",
      width: 1,
      height: 1,
      texture: canvasTexture(pc, device, this.reticleCanvas, "vr-reticle"),
      drawOrder: 20,
    });
    app.root.addChild(this.reticle.entity);
  }

  private panelItems(): PanelItem[] {
    return [
      { type: "text", title: this.options.copy.areaName(this.host.activeKey()) },
      { type: "button", id: "panel:exit", icon: "close", label: this.options.copy.exit, tone: "danger" },
    ];
  }

  // ---------------------------------------------------------------------------------
  // The frame
  // ---------------------------------------------------------------------------------

  private update(dt: number): void {
    const f = this.options.feed.current;
    if (!f.valid) return;
    const { pc, host } = this;

    // Head orientation, with the visitor's 2D heading put straight ahead on the first sample.
    const q = this.sensor.set(f.q[0], f.q[1], f.q[2], f.q[3]);
    if (!this.align) {
      const fwd = q.transformVector(new pc.Vec3(0, 0, -1));
      this.align = new pc.Quat().setFromEulerAngles(0, (host.pose.yaw - yawOf([fwd.x, fwd.y, fwd.z])) / DEG, 0);
    }
    this.head.setRotation(this.scratch.q.mul2(this.align, q));

    const head = this.head.getPosition();
    const forward = this.head.forward;
    const headV: Vec3 = [head.x, head.y, head.z];
    const forwardV: Vec3 = [forward.x, forward.y, forward.z];
    host.pose.position = [head.x, head.y, head.z];
    host.pose.yaw = yawOf(forwardV);
    host.env.setPosition(head.x, head.y, head.z);

    const key = host.activeKey();
    if (key !== this.spaceKey) {
      this.spaceKey = key;
      this.space = teleportSpace(host.manifest, key);
      this.panelDrawn = "";
    }

    this.stepMove(dt);
    this.placePanel(dt, headV, forwardV);

    // Gaze: the menu first, then the floor.
    let target: string | null = null;
    let reticleAt: Vec3 | null = null;
    let active = false;
    let markerVerdict: TeleportVerdict | null = null;
    let markerLanding: Vec3 | null = null;
    let panelHover: string | null = null;
    if (!this.moving) {
      const panelHit = this.panelGeometry ? rayPanel(headV, forwardV, this.panelGeometry) : null;
      if (panelHit) {
        panelHover = regionAt(panelHit, this.panelRegions);
        target = panelHover;
        active = panelHover !== null;
        reticleAt = [head.x + forward.x * panelHit.distance, head.y + forward.y * panelHit.distance, head.z + forward.z * panelHit.distance];
      } else {
        const floorY = floorAt(this.space, head.x, head.z);
        const landing = rayToFloor(headV, forwardV, floorY, 10);
        if (landing) {
          const verdict = validateTeleport([landing[0], landing[2]], this.space);
          markerVerdict = verdict;
          markerLanding = landing;
          reticleAt = landing;
          if (verdict.valid && Math.hypot(verdict.point[0] - head.x, verdict.point[1] - head.z) >= MIN_TELEPORT) {
            const anchor = this.gazeAnchor;
            if (!anchor || Math.hypot(anchor.verdict.point[0] - verdict.point[0], anchor.verdict.point[1] - verdict.point[1]) > GAZE_HOLD_M) {
              this.gazeAnchorId += 1;
              this.gazeAnchor = { id: this.gazeAnchorId, verdict };
            }
            target = `floor:${this.gazeAnchor!.id}`;
            active = true;
          }
        }
      }
    }
    this.gazeTarget = target;
    const config = { ...DWELL_DEFAULTS, dwellMs: target === "panel:exit" ? 2000 : DWELL_DEFAULTS.dwellMs };
    const step = stepDwell(this.dwell, target, performance.now(), config);
    if (step.fired) this.activate(step.fired);

    const onFloor = target?.startsWith("floor:") ?? false;
    this.placeMarker(markerVerdict, markerLanding, onFloor ? step.progress : 0);
    this.placeReticle(reticleAt, headV, active, onFloor ? 0 : step.progress);
    this.redrawPanel(panelHover, onFloor ? 0 : step.progress);
    this.adapt(dt);
  }

  private activate(target: string): void {
    resetDwell(this.dwell, performance.now(), DWELL_DEFAULTS.cooldownMs);
    if (target === "panel:exit") {
      this.options.onExit();
      return;
    }
    if (target.startsWith("floor:") && this.gazeAnchor && target === `floor:${this.gazeAnchor.id}`) {
      this.moving = { t: 0, verdict: this.gazeAnchor.verdict, moved: false };
      this.gazeAnchor = null;
      this.host.fade(1);
    }
  }

  /** The page's fade covers the move; the head jumps while the screen is black. */
  private stepMove(dt: number): void {
    const moving = this.moving;
    if (!moving) return;
    moving.t += dt;
    if (!moving.moved && moving.t >= FADE_MOVE_S) {
      const { point, floorY } = moving.verdict;
      this.head.setPosition(point[0], floorY + EYE_HEIGHT, point[1]);
      this.host.pose.position = [point[0], floorY + EYE_HEIGHT, point[1]];
      this.host.relocated();
      moving.moved = true;
      this.host.fade(0);
    }
    if (moving.t >= FADE_MOVE_S + FADE_BACK_S) this.moving = null;
  }

  private placePanel(dt: number, head: Vec3, forward: Vec3): void {
    const yaw = yawOf(forward);
    if (!this.panelPlaced) {
      this.panelYaw = this.panelTargetYaw = yaw;
      this.panelPlaced = true;
    }
    if (Math.abs(wrap(yaw - this.panelTargetYaw)) > PANEL_FOLLOW) this.panelTargetYaw = yaw;
    this.panelYaw = wrap(this.panelYaw + wrap(this.panelTargetYaw - this.panelYaw) * Math.min(1, dt * 5));
    const c = Math.cos(PANEL_PITCH);
    const dir: Vec3 = [-Math.sin(this.panelYaw) * c, Math.sin(PANEL_PITCH), -Math.cos(this.panelYaw) * c];
    const centre: Vec3 = [head[0] + dir[0] * PANEL_DISTANCE, head[1] + dir[1] * PANEL_DISTANCE, head[2] + dir[2] * PANEL_DISTANCE];
    const half = PANEL_WIDTH_M / 2;
    const geometry = panelFacing(centre, head, half, half * (PANEL_HEIGHT / PANEL_WIDTH));
    this.panelGeometry = geometry;
    const n: Vec3 = [
      geometry.right[1] * geometry.up[2] - geometry.right[2] * geometry.up[1],
      geometry.right[2] * geometry.up[0] - geometry.right[0] * geometry.up[2],
      geometry.right[0] * geometry.up[1] - geometry.right[1] * geometry.up[0],
    ];
    const m = this.scratch.m;
    m.set([geometry.right[0], geometry.right[1], geometry.right[2], 0, geometry.up[0], geometry.up[1], geometry.up[2], 0, n[0], n[1], n[2], 0, 0, 0, 0, 1]);
    this.panel.entity.setPosition(centre[0], centre[1], centre[2]);
    this.panel.entity.setRotation(this.scratch.q.setFromMat4(m));
  }

  private redrawPanel(hovered: string | null, progress: number): void {
    const key = `${this.host.activeKey()}|${hovered}|${Math.round(progress * 24)}`;
    if (key === this.panelDrawn) return;
    this.panelRegions = drawPanel(this.panelCanvas, this.panelItems(), { hovered, progress: Math.round(progress * 24) / 24 });
    if (this.panel.texture) refreshTexture(this.panel.texture);
    this.panelDrawn = key;
  }

  private placeMarker(verdict: TeleportVerdict | null, landing: Vec3 | null, progress: number): void {
    const show = verdict !== null && landing !== null && !this.moving;
    this.marker.entity.enabled = show;
    if (!show || !verdict || !landing) return;
    const point = verdict.valid ? verdict.point : ([landing[0], landing[2]] as [number, number]);
    this.marker.entity.setPosition(point[0], (verdict.valid ? verdict.floorY : landing[1]) + 0.01, point[1]);
    this.marker.entity.setEulerAngles(-90, 0, 0);
    const key = `${verdict.valid}|${Math.round(progress * 24)}`;
    if (key !== this.markerDrawn) {
      drawMarker(this.markerCanvas, verdict.valid, Math.round(progress * 24) / 24);
      if (this.marker.texture) refreshTexture(this.marker.texture);
      this.markerDrawn = key;
    }
  }

  private placeReticle(at: Vec3 | null, head: Vec3, active: boolean, progress: number): void {
    // Without a hit the cursor still shows, two metres out, so the visitor can find the centre.
    const forward = this.head.forward;
    const point: Vec3 = at ?? [head[0] + forward.x * 2, head[1] + forward.y * 2, head[2] + forward.z * 2];
    const distance = Math.hypot(point[0] - head[0], point[1] - head[1], point[2] - head[2]);
    this.reticle.entity.setPosition(point[0], point[1], point[2]);
    this.reticle.entity.lookAt(head[0], head[1], head[2]);
    this.reticle.entity.rotateLocal(0, 180, 0);
    const size = 2 * distance * Math.tan(RETICLE_RADIUS);
    this.reticle.entity.setLocalScale(size, size, size);
    const key = `${active}|${Math.round(progress * 24)}`;
    if (key !== this.reticleDrawn) {
      drawReticle(this.reticleCanvas, active, Math.round(progress * 24) / 24);
      if (this.reticle.texture) refreshTexture(this.reticle.texture);
      this.reticleDrawn = key;
    }
  }

  /** Eye resolution follows the frame time, as the 2D viewer's render scale does. */
  private adapt(dt: number): void {
    const p = this.perf;
    const ms = Math.min(dt * 1000, 200);
    p.ms = p.ms === 0 ? ms : p.ms * 0.9 + ms * 0.1;
    p.since += dt;
    if (p.since < 1.5) return;
    p.since = 0;
    // Same fields as the 2D viewer's readout (`?perf=1`); `scale` is the eye quality here.
    (window as unknown as { __walkPerf?: unknown }).__walkPerf = {
      fps: Math.round(1000 / Math.max(p.ms, 1)),
      frameMs: +p.ms.toFixed(1),
      scale: +this.quality.toFixed(2),
      dpr: window.devicePixelRatio || 1,
      mode: "cardboard",
      eyes: this.eyes.map((e) => `${e.texture.width}x${e.texture.height}`),
    };
    if (performance.now() - p.rebuiltAt < 3000) return;
    let next = this.quality;
    if (p.ms > FRAME_MS_SLOW) next = Math.max(QUALITY_MIN, this.quality * 0.85);
    else if (p.ms < FRAME_MS_FAST) next = Math.min(QUALITY_MAX, this.quality * 1.06);
    if (Math.abs(next - this.quality) > 0.01) {
      this.quality = next;
      this.buildEyes();
    }
  }
}

function wrap(angle: number): number {
  const m = ((((angle + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
  return m === -Math.PI ? Math.PI : m;
}
