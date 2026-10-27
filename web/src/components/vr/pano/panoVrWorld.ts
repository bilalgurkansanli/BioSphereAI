/**
 * What the 360 player puts in a headset besides the photograph: the floor rings and wall tags,
 * a menu at the visitor's feet, a gaze cursor, the info cards and the doorway fade — and the
 * per-frame logic that decides what a head or a controller is pointing at.
 *
 * Imperative on purpose, like the rest of the player's frame loop: one object that owns its
 * meshes and is stepped once a frame by whichever driver is presenting (`VrPanoStage` — WebXR
 * or Cardboard). Nothing here calls React; a `setState` per frame per target is the jitter the
 * 2D player was built to avoid.
 *
 * ── EVERYTHING ON THE PANORAMA IS AT INFINITY ─────────────────────────────────────────────
 * The photograph is one image, identical for both eyes: as far as stereo vision is concerned it
 * is infinitely far away. A ring drawn at its real 3.1 m in front of that would be the only
 * thing in the scene with depth — a hoop hovering in front of a floor the eyes place at the
 * horizon, and focusing on one doubles the other. So rings and tags are drawn at `FAR` (50 m),
 * scaled up by exactly the factor that keeps their angular size and shape: from the eye they
 * look identical to the 2D player's metric rings, and they sit ON the photograph. This is the
 * convention krpano and Marzipano use for VR hotspots, for the same reason.
 *
 * The whole content group follows the head's POSITION every frame (the panorama sphere with
 * it, via the player's content group), so leaning or stepping in a headset never parallaxes a
 * photograph that has no parallax to give. It is turned by `contentYaw`; see `orientation.ts`.
 *
 * ── THE MENU IS NOT AT INFINITY ───────────────────────────────────────────────────────────
 * It is a real object at 1.35 m, 36 degrees below the line of sight — where a menu on a
 * lap would be — and it follows the body lazily: it stays put while the visitor looks around
 * and swings back in front of them only once they have turned more than 32 degrees away. A
 * head-locked menu would sit in the middle of every view; a world-locked one would be behind
 * them after the first turn.
 *
 * ── NO DOLLY ──────────────────────────────────────────────────────────────────────────────
 * The 2D player walks the camera towards a ring before cutting (`PortalTransit`). In a headset
 * that is the textbook cause of motion sickness: visual motion the inner ear did not feel. Here
 * a ring fades to black and back, with the next room turned so its arrival view is straight
 * ahead — the "blink" teleport every comfortable VR title uses.
 */

import * as THREE from "three";
import type { Hotspot } from "@/lib/types";
import type { LookControlsApi } from "@/components/player/types";
import { EYE_HEIGHT, RING_RADIUS, TAP_TOLERANCE, floorPoint, tagLift } from "@/components/player/portalGeometry";
import { PENDING_COLOUR, READY_COLOUR } from "@/components/player/PortalRings";
import { DWELL_DEFAULTS, createDwell, resetDwell, stepDwell, type DwellConfig } from "../dwell";
import { alignmentYaw, panoramaYawAt, wrapAngle } from "../orientation";
import {
  angleBetween,
  directionOf,
  directionOnSphere,
  panelFacing,
  pickAngular,
  pickRing,
  rayPanel,
  regionAt,
  rotateY,
  type AngularTarget,
  type Panel,
  type RingTarget,
  type Vec3,
} from "../gaze";
import {
  PANEL_HEIGHT,
  PANEL_WIDTH,
  VR_COLOURS,
  drawCard,
  drawLabel,
  drawMarker,
  drawPanel,
  type PanelItem,
  type PanelRegionUv,
  type VrIcon,
} from "../ui/canvasUi";

const DEG = Math.PI / 180;
/** "Infinity": where rings and tags are drawn. Inside the 100-unit sphere, far beyond any eye. */
export const FAR = 50;
const PANEL_DISTANCE = 1.35;
const PANEL_PITCH = -36 * DEG;
const PANEL_WIDTH_M = 0.62;
const PANEL_FOLLOW = 32 * DEG;
const CARD_DISTANCE = 1.8;
const CARD_WIDTH_M = 0.72;
const MARKER_RADIUS = 1.6 * DEG;
const MARKER_HIT_RADIUS = 2.6 * DEG;
const RETICLE_RADIUS = 0.7 * DEG;
/** Looking further than this from an open card for `CARD_AUTOCLOSE_MS` closes it. */
const CARD_AWAY = 45 * DEG;
const CARD_AUTOCLOSE_MS = 1500;

/** Exit takes longer to choose by gaze than anything else: leaving by accident is the worst miss. */
const DWELL_FOR: Record<string, Partial<DwellConfig>> = {
  "panel:exit": { dwellMs: 2000 },
  "panel:prev": { dwellMs: 1200 },
  "panel:next": { dwellMs: 1200 },
  card: { dwellMs: 1200 },
};

export interface PanoVrCopy {
  exit: string;
  /** Under a ring whose room has no panorama yet. */
  pending: string;
  /** Under a card: how to close it. */
  closeCard: string;
  /** "{i} / {n}", already formatted. */
  position: (index: number, count: number) => string;
}

export interface PanoVrActions {
  navigate: (sceneId: string, fromYaw: number) => void;
  step: (delta: 1 | -1) => void;
  exit: () => void;
}

export interface HeadInput {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
}

/** A pointing ray in world space: a controller, a hand, a Vision Pro pinch. */
export interface PointerInput {
  id: string;
  origin: THREE.Vector3;
  direction: THREE.Vector3;
}

export interface PointerResult {
  id: string;
  /** The id of the thing under the ray, or null. */
  target: string | null;
  /** Where to draw the cursor, metres along the ray. */
  distance: number;
}

interface RingEntry {
  portal: Hotspot;
  target: RingTarget;
  ready: boolean;
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  label: THREE.Mesh;
  hover: number;
  labelOpacity: number;
}

interface TagEntry {
  tag: Hotspot;
  target: AngularTarget;
  mesh: THREE.Mesh;
  label: THREE.Mesh | null;
  hover: number;
  labelOpacity: number;
}

interface CardEntry {
  tagId: string;
  mesh: THREE.Mesh;
  panel: Panel;
  dir: Vec3;
  awaySince: number | null;
}

// ---------------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------------

const QUAD_VERTEX = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

/**
 * The 2D ring's annulus (`PortalRings`), plus the dwell: an outer arc that fills as the gaze
 * rests, so the visitor can see the click coming and look away if they did not mean it.
 * Unlike the 2D ring this one ends in the colour-space chunk, because it renders into an XR
 * layer and into sRGB eye targets, where a hand-written shader that skips the encode is drawn
 * at the wrong brightness.
 */
const RING_FRAGMENT = /* glsl */ `
uniform vec3 uColour;
uniform float uOpacity;
uniform float uHover;
uniform float uProgress;
varying vec2 vUv;

void main() {
  vec2 p = vUv * 2.0 - 1.0;
  float r = length(p);
  if (r > 1.0) discard;
  float aa = fwidth(r) * 1.5 + 0.002;
  float outer = 0.80;
  float inner = mix(0.58, 0.46, uHover);
  float band = smoothstep(outer, outer - aa, r) * smoothstep(inner - aa, inner, r);
  float fill = smoothstep(inner, inner - aa, r) * mix(0.10, 0.24, uHover);

  float turn = fract(atan(p.x, p.y) / 6.28318530718 + 1.0);
  float rim = smoothstep(0.985, 0.985 - aa, r) * smoothstep(0.875 - aa, 0.875, r);
  float arc = rim * step(turn, uProgress) * step(0.001, uProgress);

  float alpha = (band * mix(0.72, 1.0, uHover) + fill + arc * 0.95) * uOpacity;
  if (alpha < 0.004) discard;
  gl_FragColor = vec4(mix(uColour, vec3(1.0), arc * 0.4), alpha);
  #include <colorspace_fragment>
}
`;

/** The gaze cursor: a thin white ring with a dark edge, and the same filling arc. */
const RETICLE_FRAGMENT = /* glsl */ `
uniform float uProgress;
uniform float uActive;
uniform vec3 uAccent;
varying vec2 vUv;

void main() {
  vec2 p = vUv * 2.0 - 1.0;
  float r = length(p);
  if (r > 1.0) discard;
  float aa = fwidth(r) * 1.5 + 0.01;
  float outer = mix(0.55, 0.7, uActive);
  float inner = outer - 0.2;
  float ring = smoothstep(outer, outer - aa, r) * smoothstep(inner - aa, inner, r);
  float halo = smoothstep(outer + 0.12, outer, r) * smoothstep(inner - 0.12 - aa, inner - 0.12, r) * (1.0 - ring);
  float dot = smoothstep(0.12, 0.12 - aa, r) * (1.0 - uActive);

  float turn = fract(atan(p.x, p.y) / 6.28318530718 + 1.0);
  float rim = smoothstep(0.99, 0.99 - aa, r) * smoothstep(0.80 - aa, 0.80, r);
  float arc = rim * step(turn, uProgress) * step(0.001, uProgress);

  vec3 colour = mix(vec3(1.0), uAccent, max(uActive * 0.6, arc));
  float alpha = max(max(ring, dot), arc);
  colour = mix(vec3(0.0), colour, alpha / max(alpha + halo * 0.5, 1e-4));
  alpha = max(alpha, halo * 0.45);
  if (alpha < 0.01) discard;
  gl_FragColor = vec4(colour, alpha);
  #include <colorspace_fragment>
}
`;

/** A controller's beam: bright at the hand, gone by its end. */
const BEAM_FRAGMENT = /* glsl */ `
uniform vec3 uColour;
uniform float uOpacity;
varying vec2 vUv;
void main() {
  // CylinderGeometry's v is 1 at its top, which is the hand end once translated and turned.
  float along = 1.0 - vUv.y;
  float alpha = (1.0 - along) * (1.0 - along) * uOpacity;
  if (alpha < 0.004) discard;
  gl_FragColor = vec4(uColour, alpha);
  #include <colorspace_fragment>
}
`;

function canvasTexture(canvas: HTMLCanvasElement): THREE.CanvasTexture {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = true;
  texture.anisotropy = 4;
  return texture;
}

function quadMaterial(texture: THREE.Texture): THREE.MeshBasicMaterial {
  return new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    toneMapped: false,
    side: THREE.DoubleSide,
  });
}

const _m = new THREE.Matrix4();
const _origin = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _v = new THREE.Vector3();
const _w = new THREE.Vector3();
const _n = new THREE.Vector3();
const _euler = new THREE.Euler();

/** Orients a quad at `position` (in its parent's space) to face that space's origin, upright. */
function faceOrigin(mesh: THREE.Object3D, position: THREE.Vector3): void {
  mesh.position.copy(position);
  _m.lookAt(_origin, position, _up);
  mesh.quaternion.setFromRotationMatrix(_m);
}

function toVec3(v: THREE.Vector3): Vec3 {
  return [v.x, v.y, v.z];
}

// ---------------------------------------------------------------------------------
// The world
// ---------------------------------------------------------------------------------

export interface PanoVrWorldOptions {
  copy: PanoVrCopy;
  /** Read when a choice is made: the player's handlers change identity with its renders. */
  actions: { readonly current: PanoVrActions };
  /** The 2D player's look controls: the view to start from, and the one to hand back. */
  controls: { readonly current: LookControlsApi | null };
}

/** One of three's controller slots, and what it is currently connected to. */
interface PointerSlot {
  index: number;
  controller: THREE.Group;
  source: XRInputSource | null;
  visual: { beam: THREE.Mesh; cursor: THREE.Mesh; dispose: () => void };
  origin: THREE.Vector3;
  direction: THREE.Vector3;
}

export class PanoVrWorld {
  /** World-space things: the menu, the cursor, the veil. */
  readonly root = new THREE.Group();
  /** Things on the panorama: follows the head's position, turned by `contentYaw`. */
  readonly content = new THREE.Group();

  private copy: PanoVrCopy;
  private readonly actions: { readonly current: PanoVrActions };
  private readonly controls: { readonly current: LookControlsApi | null };
  /** A head pose has been seen: there is a view worth handing back. */
  private hasPose = false;
  private slots: PointerSlot[] = [];
  private readonly head: HeadInput = { position: new THREE.Vector3(), quaternion: new THREE.Quaternion() };

  private contentYaw = 0;
  private aligned = false;
  private headYaw = 0;
  private headPitch = 0;
  private readonly headPos = new THREE.Vector3();
  private readonly forward = new THREE.Vector3(0, 0, -1);

  private rings: RingEntry[] = [];
  private tags: TagEntry[] = [];
  private card: CardEntry | null = null;
  private readonly dwell = createDwell();
  private gazeTarget: string | null = null;
  private pointerTargets = new Map<string, string | null>();
  private readonly lastPointerTargets = new Map<string, string | null>();
  private dwellProgress = 0;

  private readonly panelCanvas = document.createElement("canvas");
  private readonly panelTexture: THREE.CanvasTexture;
  private readonly panelMesh: THREE.Mesh;
  private panelItems: PanelItem[] = [];
  private panelRegions: PanelRegionUv[] = [];
  private panelYaw = 0;
  private panelTargetYaw = 0;
  private panelPlaced = false;
  private panelGeometry: Panel | null = null;
  private panelDrawn = "";

  private readonly reticle: THREE.Mesh;
  private readonly reticleMaterial: THREE.ShaderMaterial;
  private readonly veil: THREE.Mesh;
  private readonly veilMaterial: THREE.MeshBasicMaterial;
  private veilTarget = 0;
  private veilRate = 1 / 0.18;

  private room = { name: "", index: 0, count: 1 };

  constructor(options: PanoVrWorldOptions) {
    this.copy = options.copy;
    this.actions = options.actions;
    this.controls = options.controls;
    this.root.name = "vr-root";
    this.content.name = "vr-content";

    this.panelTexture = canvasTexture(this.panelCanvas);
    this.panelMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(PANEL_WIDTH_M, PANEL_WIDTH_M * (PANEL_HEIGHT / PANEL_WIDTH)),
      quadMaterial(this.panelTexture),
    );
    this.panelMesh.renderOrder = 20;
    this.panelMesh.frustumCulled = false;
    this.root.add(this.panelMesh);

    this.reticleMaterial = new THREE.ShaderMaterial({
      uniforms: { uProgress: { value: 0 }, uActive: { value: 0 }, uAccent: { value: new THREE.Color(VR_COLOURS.bio) } },
      vertexShader: QUAD_VERTEX,
      fragmentShader: RETICLE_FRAGMENT,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    this.reticle = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.reticleMaterial);
    this.reticle.renderOrder = 40;
    this.reticle.frustumCulled = false;
    this.root.add(this.reticle);

    this.veilMaterial = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0,
      depthTest: false,
      depthWrite: false,
      side: THREE.BackSide,
      toneMapped: false,
    });
    this.veil = new THREE.Mesh(new THREE.SphereGeometry(0.4, 16, 12), this.veilMaterial);
    this.veil.renderOrder = 100;
    this.veil.frustumCulled = false;
    this.veil.visible = false;
    this.root.add(this.veil);
  }

  // ---------------------------------------------------------------------------------
  // Content
  // ---------------------------------------------------------------------------------

  setCopy(copy: PanoVrCopy): void {
    this.copy = copy;
    this.setRoom(this.room.name, this.room.index, this.room.count);
  }

  /**
   * Hands the view back to the 2D player exactly where the head was looking — "exit VR to the
   * same view" — and puts the panorama back at the origin it lives at outside a headset.
   */
  restoreView(sphere: THREE.Object3D | null): void {
    if (this.hasPose) {
      const view = this.currentView();
      this.controls.current?.lookAt({ yaw: view.yaw, pitch: view.pitch }, { animate: false });
    }
    sphere?.position.set(0, 0, 0);
    sphere?.rotation.set(0, 0, 0);
  }

  /**
   * Takes over three's two controller slots for the session: a beam on each tracked pointer,
   * and `select` routed to whatever it is on. Returns the undo.
   *
   * three re-dispatches the session's input events on the controller it assigned the source to,
   * with the `XRInputSource` as `data` — so the slot learns what is in the visitor's hand (a
   * controller, a tracked hand, a Vision Pro pinch, a gaze-only button) from `connected`.
   */
  attachControllers(gl: THREE.WebGLRenderer, scene: THREE.Scene): () => void {
    const cleanups: Array<() => void> = [];
    this.slots = [0, 1].map((index) => {
      const controller = gl.xr.getController(index);
      const visual = this.createPointerVisual();
      visual.beam.visible = false;
      controller.add(visual.beam);
      scene.add(controller);
      const slot: PointerSlot = { index, controller, source: null, visual, origin: new THREE.Vector3(), direction: new THREE.Vector3(0, 0, -1) };

      const onConnected = (event: { data?: XRInputSource }) => {
        slot.source = event.data ?? null;
        visual.beam.visible = slot.source?.targetRayMode === "tracked-pointer";
      };
      const onDisconnected = () => {
        slot.source = null;
        visual.beam.visible = false;
        visual.cursor.visible = false;
      };
      const onSelect = () => {
        const source = slot.source;
        if (!source) return;
        if (source.targetRayMode === "gaze" || source.targetRayMode === "screen") {
          // A gaze-only headset's button: select what the head is on.
          this.selectGaze();
          return;
        }
        // A controller, a hand's pinch, or Vision Pro's look-and-pinch ("transient-pointer",
        // which exists only for the length of the pinch and so is aimed afresh here).
        slot.origin.copy(controller.position);
        slot.direction.set(0, 0, -1).applyQuaternion(controller.quaternion);
        this.selectPointer(`p${index}`, slot.origin, slot.direction);
      };
      controller.addEventListener("connected", onConnected as never);
      controller.addEventListener("disconnected", onDisconnected as never);
      controller.addEventListener("select", onSelect as never);
      cleanups.push(() => {
        controller.removeEventListener("connected", onConnected as never);
        controller.removeEventListener("disconnected", onDisconnected as never);
        controller.removeEventListener("select", onSelect as never);
        controller.remove(visual.beam);
        scene.remove(controller);
        visual.dispose();
      });
      return slot;
    });
    return () => {
      for (const fn of cleanups) fn();
      this.slots = [];
    };
  }

  /**
   * A WebXR frame. The head is read from the eye cameras three has already posed for THIS frame
   * (WebXRManager fills them in before calling the frame loop); the r3f camera only receives the
   * pose during the render, a frame late for a gaze cursor — which then trails a 100 deg/s head
   * turn by a degree and a half.
   */
  frameXr(gl: THREE.WebGLRenderer, delta: number, sphere: THREE.Object3D | null): void {
    if (!gl.xr.isPresenting) return;
    // Typed as a pair, but empty until the session's first pose arrives.
    const eyes: readonly THREE.PerspectiveCamera[] = gl.xr.getCamera().cameras;
    if (eyes.length === 0) return;
    const head = this.head;
    head.position.set(0, 0, 0);
    for (const eye of eyes) head.position.add(eye.position);
    head.position.multiplyScalar(1 / eyes.length);
    head.quaternion.copy(eyes[0].quaternion);

    const inputs: PointerInput[] = [];
    for (const slot of this.slots) {
      if (slot.source?.targetRayMode !== "tracked-pointer") continue;
      slot.origin.copy(slot.controller.position);
      slot.direction.set(0, 0, -1).applyQuaternion(slot.controller.quaternion);
      inputs.push({ id: `p${slot.index}`, origin: slot.origin, direction: slot.direction });
    }
    // Hands-free (no controller, no tracked hand): the head is the pointer and gaze dwells.
    const results = this.frame(performance.now(), Math.min(delta, 0.1), head, inputs, inputs.length === 0, sphere);
    for (const slot of this.slots) {
      const result = results.find((r) => r.id === `p${slot.index}`);
      if (result) this.placePointer(slot.visual, slot.origin, slot.direction, result);
      else slot.visual.cursor.visible = false;
    }
  }

  /** A phone-headset frame: orientation only, the head fixed at the panorama's centre. */
  frameCardboard(q: readonly [number, number, number, number], delta: number, sphere: THREE.Object3D | null): HeadInput {
    const head = this.head;
    head.position.set(0, 0, 0);
    head.quaternion.set(q[0], q[1], q[2], q[3]);
    this.frame(performance.now(), Math.min(delta, 0.1), head, [], true, sphere);
    return head;
  }

  setRoom(name: string, index: number, count: number): void {
    this.room = { name, index, count };
    this.panelItems = [
      ...(count > 1 ? [{ type: "button", id: "panel:prev", icon: "prev" } as const] : []),
      { type: "text", title: name, subtitle: count > 1 ? this.copy.position(index + 1, count) : undefined },
      ...(count > 1 ? [{ type: "button", id: "panel:next", icon: "next" } as const] : []),
      { type: "button", id: "panel:exit", icon: "close", label: this.copy.exit, tone: "danger" },
    ];
    this.panelDrawn = "";
  }

  /** A new room is on screen (under the black): turn it so its arrival view is ahead. */
  arrive(): void {
    this.aligned = false;
    this.closeCard();
    resetDwell(this.dwell, performance.now(), DWELL_DEFAULTS.cooldownMs);
  }

  setRings(portals: readonly Hotspot[], readyScenes: ReadonlySet<string>, sceneNames: ReadonlyMap<string, string>): void {
    for (const entry of this.rings) this.disposeRing(entry);
    this.rings = [];
    for (const portal of portals) {
      const metric = floorPoint(portal.yaw, portal.pitch);
      if (!metric || !portal.target_scene_id) continue;
      const ready = readyScenes.has(portal.target_scene_id);
      const distance = metric.length();
      const k = FAR / distance;

      const material = new THREE.ShaderMaterial({
        uniforms: {
          uColour: { value: new THREE.Color(ready ? portal.accent || READY_COLOUR : PENDING_COLOUR) },
          uOpacity: { value: 0 },
          uHover: { value: 0 },
          uProgress: { value: 0 },
        },
        vertexShader: QUAD_VERTEX,
        fragmentShader: RING_FRAGMENT,
        transparent: true,
        depthWrite: false,
        depthTest: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(RING_RADIUS * 2 * k, RING_RADIUS * 2 * k), material);
      mesh.position.copy(metric).multiplyScalar(k);
      mesh.rotation.set(-Math.PI / 2, 0, 0);
      mesh.renderOrder = 5;
      mesh.frustumCulled = false;
      this.content.add(mesh);

      // The room's name above the ring's far edge — the same clearance the editor's handle uses.
      const name = sceneNames.get(portal.target_scene_id) ?? portal.title;
      const label = this.makeLabel({ text: name, note: ready ? undefined : this.copy.pending, noteTone: "amber" }, ready ? 1.7 : 2.6);
      const lift = tagLift(portal.pitch) + (ready ? 1.4 : 1.9) * DEG;
      faceOrigin(label, new THREE.Vector3().copy(directionVector(portal.yaw, portal.pitch + lift)).multiplyScalar(FAR * 0.98));
      label.visible = false;
      this.content.add(label);

      this.rings.push({
        portal,
        target: { id: `ring:${portal.id}`, centre: [metric.x, metric.y, metric.z], radius: RING_RADIUS },
        ready,
        mesh,
        material,
        label,
        hover: 0,
        labelOpacity: 0,
      });
    }
  }

  setTags(tags: readonly Hotspot[], sceneNames: ReadonlyMap<string, string>): void {
    for (const entry of this.tags) this.disposeTag(entry);
    this.tags = [];
    for (const tag of tags) {
      if (tag.type === "portal") continue;
      const icon: VrIcon = tag.type === "link" ? "link" : tag.type === "media" ? "media" : "info";
      const canvas = document.createElement("canvas");
      drawMarker(canvas, icon, tag.accent || VR_COLOURS.bio);
      const size = 2 * FAR * 0.97 * Math.tan(MARKER_RADIUS);
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), quadMaterial(canvasTexture(canvas)));
      faceOrigin(mesh, new THREE.Vector3().copy(directionVector(tag.yaw, tag.pitch)).multiplyScalar(FAR * 0.97));
      mesh.renderOrder = 6;
      mesh.frustumCulled = false;
      this.content.add(mesh);

      const targetName = tag.target_scene_id ? sceneNames.get(tag.target_scene_id) : undefined;
      const text = tag.title || targetName || "";
      let label: THREE.Mesh | null = null;
      if (text) {
        label = this.makeLabel({ text }, 1.6);
        faceOrigin(label, new THREE.Vector3().copy(directionVector(tag.yaw, tag.pitch - MARKER_RADIUS - 1.4 * DEG)).multiplyScalar(FAR * 0.96));
        label.visible = false;
        this.content.add(label);
      }
      this.tags.push({
        tag,
        target: { id: `tag:${tag.id}`, dir: directionOf(tag.yaw, tag.pitch), radius: MARKER_HIT_RADIUS },
        mesh,
        label,
        hover: 0,
        labelOpacity: 0,
      });
    }
  }

  /** The player's doorway veil, mirrored in the headset (the DOM one cannot be seen there). */
  setDim(dim: number, durationMs: number): void {
    this.veilTarget = Math.min(1, dim / 0.94);
    this.veilRate = 1 / Math.max(0.06, durationMs / 1000);
  }

  // ---------------------------------------------------------------------------------
  // The frame
  // ---------------------------------------------------------------------------------

  /**
   * One frame: place everything around the head, work out what each ray is on, run the gaze
   * dwell. `sphere` is the player's own content group (the panorama), moved in step.
   */
  frame(now: number, dt: number, head: HeadInput, pointers: readonly PointerInput[], gaze: boolean, sphere: THREE.Object3D | null): PointerResult[] {
    this.hasPose = true;
    this.headPos.copy(head.position);
    this.forward.set(0, 0, -1).applyQuaternion(head.quaternion);
    this.headYaw = Math.atan2(this.forward.z, this.forward.x);
    this.headPitch = Math.asin(Math.max(-1, Math.min(1, this.forward.y)));

    if (!this.aligned) {
      this.contentYaw = alignmentYaw(this.controls.current?.getView().yaw ?? 0, this.headYaw);
      this.aligned = true;
    }
    this.content.position.copy(this.headPos);
    this.content.rotation.set(0, this.contentYaw, 0);
    if (sphere) {
      sphere.position.copy(this.headPos);
      sphere.rotation.set(0, this.contentYaw, 0);
    }

    this.placePanel(dt);
    this.stepVeil(dt);

    // Nothing is chosen through the black: a click that lands mid-fade lands in the next room.
    const blocked = this.veilMaterial.opacity > 0.3 || this.veilTarget > 0.3;

    const gazeHit = blocked ? null : this.hitTest(this.headPos, this.forward, this.gazeTarget);
    this.gazeTarget = gazeHit?.target ?? null;

    const results: PointerResult[] = [];
    this.pointerTargets.clear();
    for (const pointer of pointers) {
      const previous = this.lastPointerTargets.get(pointer.id) ?? null;
      const hit = blocked ? null : this.hitTest(pointer.origin, pointer.direction, previous);
      this.pointerTargets.set(pointer.id, hit?.target ?? null);
      this.lastPointerTargets.set(pointer.id, hit?.target ?? null);
      results.push({ id: pointer.id, target: hit?.target ?? null, distance: hit?.distance ?? FAR });
    }

    // The gaze dwell, hands-free only: with a controller in hand a lingering look must not
    // click things the controller is not pointing at.
    let progress = 0;
    if (gaze && !blocked) {
      const dwellTarget = this.actionable(this.gazeTarget) ? this.gazeTarget : null;
      const config = {
        ...DWELL_DEFAULTS,
        ...(dwellTarget ? (DWELL_FOR[dwellTarget] ?? (dwellTarget.startsWith("card:") ? DWELL_FOR.card : {})) : {}),
      };
      const step = stepDwell(this.dwell, dwellTarget, now, config);
      progress = step.progress;
      if (step.fired) this.activate(step.fired, now);
    } else if (!gaze) {
      resetDwell(this.dwell, now);
    }
    this.dwellProgress = progress;

    this.updateCard(now);
    this.updateVisuals(dt, gaze, gazeHit?.distance ?? FAR);
    return results;
  }

  /**
   * Whether choosing `target` would do anything. The menu's background swallows the ray but is
   * not a button, and a ring into a room that has not been shot is labelled, not walkable: a
   * dwell ring filling up on either would promise a click that never comes.
   */
  private actionable(target: string | null): boolean {
    if (!target || target === "panel:none") return false;
    if (target.startsWith("ring:")) return this.rings.some((r) => r.target.id === target && r.ready);
    return true;
  }

  /** A controller's trigger, a hand's pinch, a Vision Pro pinch. */
  selectPointer(id: string, origin?: THREE.Vector3, direction?: THREE.Vector3): boolean {
    let target = this.pointerTargets.get(id) ?? null;
    if (!target && origin && direction) target = this.hitTest(origin, direction, null)?.target ?? null;
    if (!this.actionable(target) || !target) return false;
    this.activate(target, performance.now());
    return true;
  }

  /** The Cardboard's button (a screen tap): "this one", meaning wherever the gaze is. */
  selectGaze(): boolean {
    if (!this.actionable(this.gazeTarget) || !this.gazeTarget) return false;
    this.activate(this.gazeTarget, performance.now());
    return true;
  }

  /** Where the visitor is looking, in the panorama's own angles — to hand back to the 2D player. */
  currentView(): { yaw: number; pitch: number } {
    return { yaw: panoramaYawAt(this.headYaw, this.contentYaw), pitch: this.headPitch };
  }

  // ---------------------------------------------------------------------------------
  // Hit testing
  // ---------------------------------------------------------------------------------

  private hitTest(origin: THREE.Vector3, direction: THREE.Vector3, current: string | null): { target: string; distance: number } | null {
    const o = toVec3(origin);
    const d = toVec3(direction);

    // Real objects first: the menu, then an open card. They are in front of everything.
    if (this.panelGeometry) {
      const hit = rayPanel(o, d, this.panelGeometry);
      if (hit) {
        const region = regionAt(hit, this.panelRegions);
        // The panel itself swallows the ray even between buttons: nothing behind a menu is
        // meant to be clicked through it.
        return { target: region ?? "panel:none", distance: hit.distance };
      }
    }

    // Into the panorama's frame: relative to the head, un-turned.
    const oc = rotateY([o[0] - this.headPos.x, o[1] - this.headPos.y, o[2] - this.headPos.z], -this.contentYaw);
    const dc = rotateY(d, -this.contentYaw);

    if (this.card) {
      const hit = rayPanel(oc, dc, this.card.panel);
      if (hit) return { target: `card:${this.card.tagId}`, distance: hit.distance };
    }

    const seen = directionOnSphere(oc, dc, FAR);
    const ring = pickRing(seen, this.rings.map((r) => r.target), { eyeHeight: EYE_HEIGHT, minAngle: TAP_TOLERANCE, current });
    const tag = pickAngular(seen, this.tags.map((t) => t.target), { current });
    const best = ring && tag ? (ring.score <= tag.score ? ring : tag) : (ring ?? tag);
    return best ? { target: best.id, distance: FAR } : null;
  }

  private activate(target: string, now: number): void {
    resetDwell(this.dwell, now, DWELL_DEFAULTS.cooldownMs);
    if (target === "panel:exit") {
      this.actions.current.exit();
      return;
    }
    if (target === "panel:prev" || target === "panel:next") {
      this.closeCard();
      this.actions.current.step(target === "panel:next" ? 1 : -1);
      return;
    }
    if (target.startsWith("card:")) {
      this.closeCard();
      return;
    }
    if (target.startsWith("ring:")) {
      const entry = this.rings.find((r) => r.target.id === target);
      // A ring into a room not yet shot is drawn and labelled, and does nothing: in a headset
      // there is no "dark room" card to walk into and back out of.
      if (entry?.ready && entry.portal.target_scene_id) {
        this.closeCard();
        this.actions.current.navigate(entry.portal.target_scene_id, entry.portal.yaw);
      }
      return;
    }
    if (target.startsWith("tag:")) {
      const entry = this.tags.find((t) => t.target.id === target);
      if (!entry) return;
      if (entry.tag.type === "link" && entry.tag.target_scene_id) {
        this.closeCard();
        this.actions.current.navigate(entry.tag.target_scene_id, entry.tag.yaw);
        return;
      }
      if (this.card?.tagId === entry.tag.id) this.closeCard();
      else this.openCard(entry.tag);
    }
  }

  // ---------------------------------------------------------------------------------
  // Cards
  // ---------------------------------------------------------------------------------

  private openCard(tag: Hotspot): void {
    this.closeCard();
    const canvas = document.createElement("canvas");
    const aspect = drawCard(canvas, tag.title, tag.body, this.copy.closeCard);
    const width = CARD_WIDTH_M;
    const height = width / aspect;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), quadMaterial(canvasTexture(canvas)));
    mesh.renderOrder = 30;
    mesh.frustumCulled = false;
    // Just under the tag, so the tag stays visible above the card it opened.
    const dir = directionOf(tag.yaw, Math.max(-60 * DEG, Math.min(50 * DEG, tag.pitch - 6 * DEG)));
    const centre = new THREE.Vector3(dir[0], dir[1], dir[2]).multiplyScalar(CARD_DISTANCE);
    faceOrigin(mesh, centre);
    this.content.add(mesh);
    const panel = panelFacing([centre.x, centre.y, centre.z], [0, 0, 0], width / 2, height / 2);
    this.card = { tagId: tag.id, mesh, panel, dir, awaySince: null };
  }

  private closeCard(): void {
    const card = this.card;
    if (!card) return;
    this.content.remove(card.mesh);
    disposeMesh(card.mesh);
    this.card = null;
  }

  private updateCard(now: number): void {
    const card = this.card;
    if (!card) return;
    const looking = rotateY(toVec3(this.forward), -this.contentYaw);
    if (angleBetween(looking, card.dir) > CARD_AWAY) {
      card.awaySince ??= now;
      if (now - card.awaySince > CARD_AUTOCLOSE_MS) this.closeCard();
    } else {
      card.awaySince = null;
    }
  }

  // ---------------------------------------------------------------------------------
  // Visuals
  // ---------------------------------------------------------------------------------

  private placePanel(dt: number): void {
    if (!this.panelPlaced) {
      this.panelYaw = this.panelTargetYaw = this.headYaw;
      this.panelPlaced = true;
    }
    if (Math.abs(wrapAngle(this.headYaw - this.panelTargetYaw)) > PANEL_FOLLOW) this.panelTargetYaw = this.headYaw;
    const delta = wrapAngle(this.panelTargetYaw - this.panelYaw);
    this.panelYaw = wrapAngle(this.panelYaw + delta * Math.min(1, dt * 5));

    const dir = directionOf(this.panelYaw, PANEL_PITCH);
    const centre: Vec3 = [
      this.headPos.x + dir[0] * PANEL_DISTANCE,
      this.headPos.y + dir[1] * PANEL_DISTANCE,
      this.headPos.z + dir[2] * PANEL_DISTANCE,
    ];
    const half = PANEL_WIDTH_M / 2;
    const geometry = panelFacing(centre, toVec3(this.headPos), half, half * (PANEL_HEIGHT / PANEL_WIDTH));
    this.panelGeometry = geometry;
    this.panelMesh.position.set(centre[0], centre[1], centre[2]);
    _v.set(geometry.right[0], geometry.right[1], geometry.right[2]);
    _w.set(geometry.up[0], geometry.up[1], geometry.up[2]);
    _m.makeBasis(_v, _w, _n.crossVectors(_v, _w));
    this.panelMesh.quaternion.setFromRotationMatrix(_m);
  }

  private stepVeil(dt: number): void {
    const current = this.veilMaterial.opacity;
    const step = dt * this.veilRate;
    const next = current < this.veilTarget ? Math.min(this.veilTarget, current + step) : Math.max(this.veilTarget, current - step);
    this.veilMaterial.opacity = next;
    this.veil.visible = next > 0.001;
    this.veil.position.copy(this.headPos);
  }

  private updateVisuals(dt: number, gaze: boolean, gazeDistance: number): void {
    const hovered = new Set<string>();
    if (gaze && this.gazeTarget) hovered.add(this.gazeTarget);
    for (const target of this.pointerTargets.values()) if (target) hovered.add(target);
    const rate = Math.min(1, dt * 12);
    const fade = Math.min(1, dt * 10);

    for (const ring of this.rings) {
      const on = hovered.has(ring.target.id);
      ring.hover += (Number(on) - ring.hover) * rate;
      const u = ring.material.uniforms;
      u.uHover.value = ring.hover;
      u.uOpacity.value += ((ring.ready ? 1 : 0.55) - u.uOpacity.value) * Math.min(1, dt * 4);
      u.uProgress.value = gaze && this.gazeTarget === ring.target.id ? this.dwellProgress : 0;
      ring.labelOpacity += (Number(on) - ring.labelOpacity) * fade;
      setOpacity(ring.label, ring.labelOpacity);
    }
    for (const tag of this.tags) {
      const on = hovered.has(tag.target.id);
      tag.hover += (Number(on) - tag.hover) * rate;
      tag.mesh.scale.setScalar(1 + tag.hover * 0.18);
      if (tag.label) {
        tag.labelOpacity += (Number(on) - tag.labelOpacity) * fade;
        setOpacity(tag.label, tag.labelOpacity);
      }
    }

    // The menu redraws only when what it shows changes — a texture upload is not free.
    const panelHover = [...hovered].find((id) => id.startsWith("panel:")) ?? null;
    const panelProgress = gaze && panelHover && this.gazeTarget === panelHover ? Math.round(this.dwellProgress * 24) / 24 : 0;
    const key = `${this.room.name}|${this.room.index}|${panelHover}|${panelProgress}`;
    if (key !== this.panelDrawn) {
      this.panelRegions = drawPanel(this.panelCanvas, this.panelItems, { hovered: panelHover, progress: panelProgress });
      this.panelTexture.needsUpdate = true;
      this.panelDrawn = key;
    }

    // The gaze cursor: only when the head is the pointer.
    this.reticle.visible = gaze;
    if (gaze) {
      const distance = Math.min(gazeDistance, FAR * 0.9);
      this.reticle.position.copy(this.headPos).addScaledVector(this.forward, distance);
      this.reticle.quaternion.copy(this.headQuaternion());
      this.reticle.scale.setScalar(distance * Math.tan(RETICLE_RADIUS) * (this.gazeTarget ? 1.25 : 1));
      this.reticleMaterial.uniforms.uProgress.value = this.dwellProgress;
      const active = this.actionable(this.gazeTarget) ? 1 : 0;
      this.reticleMaterial.uniforms.uActive.value += (active - this.reticleMaterial.uniforms.uActive.value) * rate;
    }
  }

  private readonly _q = new THREE.Quaternion();
  private headQuaternion(): THREE.Quaternion {
    // Rebuilt from yaw and pitch so the cursor never rolls with the head.
    return this._q.setFromEuler(_euler.set(this.headPitch, -(this.headYaw + Math.PI / 2), 0, "YXZ"));
  }

  // ---------------------------------------------------------------------------------
  // Controller visuals
  // ---------------------------------------------------------------------------------

  /** A beam for a controller's target-ray space, and a cursor for where it lands. */
  createPointerVisual(): { beam: THREE.Mesh; cursor: THREE.Mesh; dispose: () => void } {
    const beamMaterial = new THREE.ShaderMaterial({
      uniforms: { uColour: { value: new THREE.Color(VR_COLOURS.ink) }, uOpacity: { value: 0.65 } },
      vertexShader: QUAD_VERTEX,
      fragmentShader: BEAM_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
    });
    // A thin cylinder along -Z from the hand; uv.y runs 0 at the hand to 1 at the end.
    const geometry = new THREE.CylinderGeometry(0.0022, 0.0012, 1, 8, 1, true);
    geometry.translate(0, -0.5, 0);
    geometry.rotateX(Math.PI / 2);
    const beam = new THREE.Mesh(geometry, beamMaterial);
    beam.renderOrder = 35;
    beam.frustumCulled = false;

    const cursorMaterial = new THREE.ShaderMaterial({
      uniforms: { uProgress: { value: 0 }, uActive: { value: 0 }, uAccent: { value: new THREE.Color(VR_COLOURS.bio) } },
      vertexShader: QUAD_VERTEX,
      fragmentShader: RETICLE_FRAGMENT,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    const cursor = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), cursorMaterial);
    cursor.renderOrder = 41;
    cursor.frustumCulled = false;
    cursor.visible = false;
    this.root.add(cursor);
    return {
      beam,
      cursor,
      dispose: () => {
        this.root.remove(cursor);
        disposeMesh(beam);
        disposeMesh(cursor);
      },
    };
  }

  /** Places a controller's beam and cursor for this frame's result. */
  placePointer(visual: { beam: THREE.Mesh; cursor: THREE.Mesh }, origin: THREE.Vector3, direction: THREE.Vector3, result: PointerResult): void {
    const real = result.target !== null && result.distance < FAR;
    const length = real ? result.distance : 2.2;
    visual.beam.scale.set(1, 1, length);
    const material = visual.cursor.material as THREE.ShaderMaterial;
    const showCursor = result.target !== null;
    visual.cursor.visible = showCursor;
    if (showCursor) {
      const distance = Math.min(result.distance, FAR * 0.9);
      visual.cursor.position.copy(origin).addScaledVector(direction, distance);
      visual.cursor.lookAt(this.headPos);
      // Constant size as seen from the head, wherever along the ray it lands.
      visual.cursor.scale.setScalar(visual.cursor.position.distanceTo(this.headPos) * Math.tan(RETICLE_RADIUS) * 1.2);
      material.uniforms.uActive.value = this.actionable(result.target) ? 1 : 0;
    }
  }

  // ---------------------------------------------------------------------------------
  // Plumbing
  // ---------------------------------------------------------------------------------

  private makeLabel(spec: Parameters<typeof drawLabel>[1], heightDeg: number): THREE.Mesh {
    const canvas = document.createElement("canvas");
    const aspect = drawLabel(canvas, spec);
    const height = 2 * FAR * Math.tan((heightDeg * DEG) / 2);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(height * aspect, height), quadMaterial(canvasTexture(canvas)));
    mesh.renderOrder = 10;
    mesh.frustumCulled = false;
    return mesh;
  }

  private disposeRing(entry: RingEntry): void {
    this.content.remove(entry.mesh, entry.label);
    disposeMesh(entry.mesh);
    disposeMesh(entry.label);
  }

  private disposeTag(entry: TagEntry): void {
    this.content.remove(entry.mesh);
    disposeMesh(entry.mesh);
    if (entry.label) {
      this.content.remove(entry.label);
      disposeMesh(entry.label);
    }
  }

  dispose(): void {
    for (const entry of this.rings) this.disposeRing(entry);
    for (const entry of this.tags) this.disposeTag(entry);
    this.closeCard();
    this.rings = [];
    this.tags = [];
    disposeMesh(this.panelMesh);
    disposeMesh(this.reticle);
    disposeMesh(this.veil);
  }
}

function directionVector(yaw: number, pitch: number): THREE.Vector3 {
  const d = directionOf(yaw, pitch);
  return new THREE.Vector3(d[0], d[1], d[2]);
}

function setOpacity(mesh: THREE.Mesh, opacity: number): void {
  const material = mesh.material as THREE.MeshBasicMaterial;
  material.opacity = opacity;
  mesh.visible = opacity > 0.01;
}

function disposeMesh(mesh: THREE.Mesh): void {
  mesh.geometry.dispose();
  const material = mesh.material as THREE.Material & { map?: THREE.Texture | null };
  material.map?.dispose();
  material.dispose();
}
