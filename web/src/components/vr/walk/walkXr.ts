/**
 * The 3D walkthrough in a headset (WebXR, PlayCanvas): standing in the model at the visitor's
 * own height, moving by teleport, turning by snap, never landing where the model is garbage.
 *
 * ── STANDING HEIGHT ───────────────────────────────────────────────────────────────────────
 * The session runs in `local-floor` space, so the headset reports the visitor's REAL eye height
 * above their real floor; the rig puts that floor on the model's floor under them
 * (`teleport.floorAt` — the area's median camera height minus the 1.45 m the reconstruction was
 * scaled by). Where `local-floor` is refused the session falls back to `local`, whose origin is
 * the head itself, and the rig is lifted by the walkthrough's own 1.6 m eye height instead.
 *
 * ── MOVING ────────────────────────────────────────────────────────────────────────────────
 * Teleport is the default because it is the locomotion nobody gets sick from: aim (thumbstick
 * forward, or hold the trigger, or pinch), see the arc and the landing ring, release, blink.
 * The landing is validated against the walkthrough's corridor (`teleport.validateTeleport`):
 * teal where it may land, amber with a bar where the model was never filmed, and a near miss
 * is pulled back onto the corridor's edge. Snap turns of 30 degrees on the thumbstick's sides,
 * pivoting about the head. Smooth movement is opt-in from the menu ("Akıcı hareket"), on the
 * left stick, with the comfort vignette closing in while it moves.
 *
 * Hands-free (no controller, no tracked hand) the head is the pointer: look at a spot on the
 * floor and hold, and the ring fills — the same dwell as the 360's rings.
 *
 * ── FRAME RATE ────────────────────────────────────────────────────────────────────────────
 * Splats are fill-bound and a headset draws two eyes, so the session starts at a per-device
 * framebuffer scale (`capabilities.xrFramebufferScale`), with fixed foveation at maximum, radial
 * splat sorting (rotation-invariant, so turning the head never forces a re-sort) and the
 * lowest refresh rate the headset offers (72 Hz on Quest). Then it measures: when more than
 * 15 % of frames over a second and a half miss the refresh, the eye buffers are rebuilt 15 %
 * smaller, and they grow back slowly while there is headroom. The WebXR base layer is replaced
 * through `updateRenderState` — PlayCanvas's WebGL bridge reads the layer from the render state
 * every frame, so it follows without knowing. A dropped frame in a headset is not a stutter, it
 * is the world lurching against the inner ear.
 */

import type * as PC from "playcanvas";
import { EYE_HEIGHT, yawOf } from "@/components/walk/walkMath";
import { DWELL_DEFAULTS, createDwell, resetDwell, stepDwell } from "../dwell";
import { panelFacing, rayPanel, regionAt, type Panel, type Vec3 } from "../gaze";
import {
  MIN_TELEPORT,
  arcToFloor,
  constrainMove,
  floorAt,
  rayToFloor,
  rigAfterTurn,
  rigForTeleport,
  teleportSpace,
  validateTeleport,
  type Rig,
  type TeleportSpace,
  type TeleportVerdict,
} from "../teleport";
import { PANEL_HEIGHT, drawPanel, type PanelItem, type PanelRegionUv } from "../ui/canvasUi";
import type { WalkVrHost } from "./host";
import { canvasTexture, drawMarker, drawReticle, drawVignette, makeOverlay, refreshTexture, type Overlay, type Pc } from "./pcKit";

const DEG = Math.PI / 180;
const SNAP_TURN = 30 * DEG;
/** Slower than the 2D walk's 1.7 m/s: the same speed reads faster with the whole view moving. */
const SMOOTH_SPEED = 1.4;
const PANEL_DISTANCE = 1.2;
const PANEL_PITCH = -38 * DEG;
/** The walkthrough's menu is wider than the 360's: a room name AND a switch beside the exit. */
const PANEL_CANVAS = 1280;
const PANEL_WIDTH_M = 0.7;
const PANEL_FOLLOW = 32 * DEG;
const FADE_OUT_S = 0.12;
const FADE_IN_S = 0.22;
const MARKER_SIZE = 0.8;
const RETICLE_RADIUS = 0.9 * DEG;
/** A gaze landing that moves less than this keeps its dwell going. */
const GAZE_HOLD_M = 0.45;
const STICK_DEADZONE = 0.15;

export interface WalkVrCopy {
  exit: string;
  /** The smooth-movement switch; its state is drawn, not spelled out. */
  smooth: string;
  areaName: (key: string | null) => string;
}

export interface WalkXrOptions {
  /** The reference space the session was granted. */
  space: "local-floor" | "local";
  /** The framebuffer scale the session started at, relative to the headset's default. */
  scale: number;
  copy: WalkVrCopy;
  onExit: () => void;
}

interface SourceState {
  aiming: boolean;
  stickAim: boolean;
  turnLatched: boolean;
  panelPress: string | null;
  verdict: TeleportVerdict | null;
  landing: Vec3 | null;
}

export class WalkXr {
  private readonly host: WalkVrHost;
  private readonly pc: Pc;
  private readonly app: PC.AppBase;
  private readonly options: WalkXrOptions;
  private readonly rigEntity: PC.Entity;

  private rig: Rig = { x: 0, z: 0, yaw: 0 };
  private rigY = 0;
  private aligned = false;
  private readonly start: { x: number; z: number; yaw: number };
  private space: TeleportSpace;
  private spaceKey: string | null;
  private smooth = false;

  private readonly sources = new Map<PC.XrInputSource, SourceState>();
  private readonly dwell = createDwell();
  private gazeAnchor: { id: number; verdict: TeleportVerdict } | null = null;
  private gazeAnchorId = 0;
  private gazeTarget: string | null = null;

  private fade: { phase: "out" | "in"; t: number; verdict: TeleportVerdict | null } | null = null;
  private vignetteLevel = 0;

  private readonly panelCanvas = document.createElement("canvas");
  private readonly markerCanvas = document.createElement("canvas");
  private readonly reticleCanvas = document.createElement("canvas");
  private panel!: Overlay;
  private marker!: Overlay;
  private reticle!: Overlay;
  private vignette!: Overlay;
  private veil!: Overlay;
  private panelGeometry: Panel | null = null;
  private panelRegions: PanelRegionUv[] = [];
  private panelYaw = 0;
  private panelTargetYaw = 0;
  private panelPlaced = false;
  private panelDrawn = "";
  private markerDrawn = "";
  private reticleDrawn = "";

  /** The arc's segment ends, drawn with `drawLines`. Pooled: at 72 Hz an allocation per point adds up. */
  private readonly arcPoints: PC.Vec3[] = [];
  private readonly arcPool: PC.Vec3[] = [];
  private arcColours: { valid: PC.Color; refused: PC.Color } | null = null;
  private readonly scratch: { v: PC.Vec3; q: PC.Quat; m: PC.Mat4 };
  private readonly frame = { ms: 0, slow: 0, count: 0, since: 0, calm: 0, scale: 1, changedAt: 0 };
  private readonly cleanups: Array<() => void> = [];

  constructor(host: WalkVrHost, rig: PC.Entity, options: WalkXrOptions) {
    this.host = host;
    this.pc = host.pc;
    this.app = host.app;
    this.options = options;
    this.rigEntity = rig;
    this.start = { x: host.pose.position[0], z: host.pose.position[2], yaw: host.pose.yaw };
    this.spaceKey = host.activeKey();
    this.space = teleportSpace(host.manifest, this.spaceKey);
    this.frame.scale = options.scale;
    this.scratch = { v: new this.pc.Vec3(), q: new this.pc.Quat(), m: new this.pc.Mat4() };
  }

  attach(): void {
    const { pc, app } = this;
    const device = app.graphicsDevice;
    this.host.stopTour();

    drawPanel(this.panelCanvas, this.panelItems(), { hovered: null, progress: 0 }, PANEL_CANVAS);
    const panelHeight = PANEL_WIDTH_M * (PANEL_HEIGHT / PANEL_CANVAS);
    this.panel = makeOverlay(pc, app, {
      name: "vr-panel",
      width: PANEL_WIDTH_M,
      height: panelHeight,
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
    this.reticle.entity.enabled = false;
    app.root.addChild(this.reticle.entity);

    // Head-locked: children of the camera, which the session poses every frame.
    const vignetteCanvas = document.createElement("canvas");
    drawVignette(vignetteCanvas);
    this.vignette = makeOverlay(pc, app, {
      name: "vr-vignette",
      width: 1.3,
      height: 1.3,
      texture: canvasTexture(pc, device, vignetteCanvas, "vr-vignette"),
      drawOrder: 90,
    });
    this.vignette.entity.setLocalPosition(0, 0, -0.3);
    this.vignette.setOpacity(0);
    this.host.camera.addChild(this.vignette.entity);

    this.veil = makeOverlay(pc, app, { name: "vr-veil", width: 1.6, height: 1.6, texture: null, colour: new pc.Color(0, 0, 0), drawOrder: 100 });
    this.veil.entity.setLocalPosition(0, 0, -0.12);
    // The session opens in black and fades in once the rig is placed: the first frames would
    // otherwise show the model from wherever the tracking origin happened to be.
    this.veil.setOpacity(1);
    this.host.camera.addChild(this.veil.entity);

    const xr = app.xr;
    if (xr) {
      // The lowest refresh the headset offers, for the most time per frame.
      const rates = xr.supportedFrameRates;
      if (rates && rates.length) xr.updateTargetFrameRate(Math.min(...rates));
      xr.fixedFoveation = 1;
      const input = xr.input;
      const onStart = (source: PC.XrInputSource) => this.selectStart(source);
      const onEnd = (source: PC.XrInputSource) => this.selectEnd(source);
      const onSelect = (source: PC.XrInputSource) => this.select(source);
      const onRemove = (source: PC.XrInputSource) => this.sources.delete(source);
      input.on("selectstart", onStart);
      input.on("selectend", onEnd);
      input.on("select", onSelect);
      input.on("remove", onRemove);
      this.cleanups.push(() => {
        input.off("selectstart", onStart);
        input.off("selectend", onEnd);
        input.off("select", onSelect);
        input.off("remove", onRemove);
      });
    }

    const onUpdate = (dt: number) => this.update(dt);
    app.on("update", onUpdate);
    this.cleanups.push(() => app.off("update", onUpdate));
  }

  detach(): void {
    for (const fn of this.cleanups.splice(0)) fn();
    for (const overlay of [this.panel, this.marker, this.reticle, this.vignette, this.veil]) overlay?.destroy();
  }

  // ---------------------------------------------------------------------------------
  // The frame
  // ---------------------------------------------------------------------------------

  private update(dt: number): void {
    const xr = this.app.xr;
    if (!xr?.active) return;
    const camera = this.host.camera;

    if (!this.aligned) this.align();

    const head = camera.getPosition();
    const forward = camera.forward;
    const headV: Vec3 = [head.x, head.y, head.z];
    const forwardV: Vec3 = [forward.x, forward.y, forward.z];

    // Streaming, the area swap and the plan's mark follow the head.
    const pose = this.host.pose;
    pose.position = [head.x, head.y, head.z];
    pose.yaw = yawOf(forwardV);
    pose.pitch = Math.asin(Math.max(-1, Math.min(1, forward.y)));
    this.host.env.setPosition(head.x, head.y, head.z);

    const key = this.host.activeKey();
    if (key !== this.spaceKey) {
      this.spaceKey = key;
      this.space = teleportSpace(this.host.manifest, key);
    }

    this.stepFade(dt);
    this.placePanel(dt, headV, forwardV);
    const busy = this.fade !== null;

    // Inputs.
    let pointers = 0;
    let aimingAny = false;
    let markerVerdict: TeleportVerdict | null = null;
    let markerLanding: Vec3 | null = null;
    let panelHover: string | null = null;
    let moved = 0;
    this.arcPoints.length = 0;

    for (const source of xr.input.inputSources) {
      const state = this.stateOf(source);
      if (source.targetRayMode !== "tracked-pointer") continue;
      pointers += 1;
      const o = source.getOrigin();
      const d = source.getDirection();
      const origin: Vec3 = [o.x, o.y, o.z];
      const direction: Vec3 = [d.x, d.y, d.z];

      const panelHit = this.panelGeometry ? rayPanel(origin, direction, this.panelGeometry) : null;
      const region = panelHit ? regionAt(panelHit, this.panelRegions) : null;
      if (region) panelHover = region;

      const stick = this.stick(source);
      if (stick && !busy) moved = Math.max(moved, this.handleStick(source, state, stick, dt, headV, forwardV));

      if ((state.aiming || state.stickAim) && !panelHit && !busy) {
        aimingAny = true;
        const floorY = floorAt(this.space, head.x, head.z);
        const arc = arcToFloor(origin, direction, floorY);
        state.landing = arc.landing;
        state.verdict = arc.landing ? validateTeleport([arc.landing[0], arc.landing[2]], this.space) : null;
        this.pushArc(arc.points);
        markerVerdict = state.verdict;
        markerLanding = arc.landing;
      } else {
        state.landing = null;
        state.verdict = null;
      }
    }

    // Hands-free: the head points, and resting the gaze clicks.
    const gaze = pointers === 0;
    let progress = 0;
    let reticleAt: Vec3 | null = null;
    let reticleActive = false;
    if (gaze && !busy) {
      const panelHit = this.panelGeometry ? rayPanel(headV, forwardV, this.panelGeometry) : null;
      let target: string | null = null;
      if (panelHit) {
        const region = regionAt(panelHit, this.panelRegions);
        panelHover = region;
        target = region;
        reticleAt = [head.x + forward.x * panelHit.distance, head.y + forward.y * panelHit.distance, head.z + forward.z * panelHit.distance];
        reticleActive = region !== null;
      } else {
        const landing = rayToFloor(headV, forwardV, floorAt(this.space, head.x, head.z), 10);
        if (landing) {
          const verdict = validateTeleport([landing[0], landing[2]], this.space);
          markerVerdict = verdict;
          markerLanding = landing;
          reticleAt = landing;
          const far = Math.hypot(verdict.point[0] - head.x, verdict.point[1] - head.z) >= MIN_TELEPORT;
          if (verdict.valid && far) {
            const anchor = this.gazeAnchor;
            if (!anchor || Math.hypot(anchor.verdict.point[0] - verdict.point[0], anchor.verdict.point[1] - verdict.point[1]) > GAZE_HOLD_M) {
              this.gazeAnchorId += 1;
              this.gazeAnchor = { id: this.gazeAnchorId, verdict };
            }
            target = `floor:${this.gazeAnchor!.id}`;
            reticleActive = true;
          }
        }
      }
      this.gazeTarget = target;
      const config = { ...DWELL_DEFAULTS, dwellMs: target === "panel:exit" ? 2000 : target === "panel:smooth" ? 1200 : DWELL_DEFAULTS.dwellMs };
      const step = stepDwell(this.dwell, target, performance.now(), config);
      progress = step.progress;
      if (step.fired) this.activate(step.fired);
    } else if (!gaze) {
      resetDwell(this.dwell, performance.now());
      this.gazeTarget = null;
    }

    this.drawArc(aimingAny, markerVerdict);
    this.placeMarker(markerVerdict, markerLanding, gaze && this.gazeTarget?.startsWith("floor:") ? progress : 0);
    this.placeReticle(gaze ? reticleAt : null, headV, reticleActive, gaze && this.gazeTarget?.startsWith("panel:") ? progress : 0);
    this.redrawPanel(panelHover, gaze && this.gazeTarget?.startsWith("panel:") ? progress : 0);
    this.stepVignette(dt, moved);
    this.adaptResolution(dt, xr);
  }

  /** First frame with a pose: the head goes where the 2D view stood, facing where it faced. */
  private align(): void {
    const camera = this.host.camera;
    const local = camera.getLocalPosition();
    const fwd = camera.getLocalRotation().transformVector(new this.pc.Vec3(0, 0, -1), this.scratch.v);
    const localYaw = yawOf([fwd.x, 0, fwd.z]);
    const placed = rigForTeleport({ x: 0, z: 0, yaw: this.start.yaw - localYaw }, [local.x, local.z], [this.start.x, this.start.z]);
    this.rig = placed;
    this.rigY = this.floorRig(floorAt(this.space, this.start.x, this.start.z));
    this.applyRig();
    this.aligned = true;
    this.fade = { phase: "in", t: 0, verdict: null };
  }

  private floorRig(floorY: number): number {
    return floorY + (this.options.space === "local" ? EYE_HEIGHT : 0);
  }

  private applyRig(): void {
    this.rigEntity.setPosition(this.rig.x, this.rigY, this.rig.z);
    this.rigEntity.setEulerAngles(0, this.rig.yaw / DEG, 0);
  }

  private headLocal(): [number, number] {
    const local = this.host.camera.getLocalPosition();
    return [local.x, local.z];
  }

  // ---------------------------------------------------------------------------------
  // Controllers
  // ---------------------------------------------------------------------------------

  private stateOf(source: PC.XrInputSource): SourceState {
    let state = this.sources.get(source);
    if (!state) {
      state = { aiming: false, stickAim: false, turnLatched: false, panelPress: null, verdict: null, landing: null };
      this.sources.set(source, state);
    }
    return state;
  }

  /** The thumbstick, from the xr-standard mapping (axes 2 and 3; 0 and 1 on a touchpad). */
  private stick(source: PC.XrInputSource): [number, number] | null {
    const axes = source.gamepad?.axes;
    if (!axes || axes.length < 2) return null;
    return axes.length >= 4 ? [axes[2], axes[3]] : [axes[0], axes[1]];
  }

  /** Returns the smooth speed applied this frame, 0..1, for the vignette. */
  private handleStick(source: PC.XrInputSource, state: SourceState, [x, y]: [number, number], dt: number, head: Vec3, forward: Vec3): number {
    // Smooth movement on the left stick, when the visitor asked for it.
    if (this.smooth && source.handedness === "left") {
      const mx = Math.abs(x) > STICK_DEADZONE ? x : 0;
      const my = Math.abs(y) > STICK_DEADZONE ? y : 0;
      const magnitude = Math.min(1, Math.hypot(mx, my));
      if (magnitude === 0) return 0;
      const fl = Math.hypot(forward[0], forward[2]) || 1;
      const fx = forward[0] / fl;
      const fz = forward[2] / fl;
      // Stick up is -y. Right is the forward turned a quarter clockwise seen from above.
      const vx = (fx * -my + -fz * mx) * SMOOTH_SPEED * dt;
      const vz = (fz * -my + fx * mx) * SMOOTH_SPEED * dt;
      const from: [number, number] = [head[0], head[2]];
      const to = constrainMove(from, [from[0] + vx, from[1] + vz], this.space);
      this.rig = { ...this.rig, x: this.rig.x + (to[0] - from[0]), z: this.rig.z + (to[1] - from[1]) };
      this.applyRig();
      return magnitude;
    }

    // Snap turn on the stick's sides, latched until it comes back to the middle.
    if (Math.abs(x) > 0.7 && Math.abs(y) < 0.5) {
      if (!state.turnLatched) {
        state.turnLatched = true;
        // PlayCanvas yaw is counter-clockwise from above: right is negative.
        this.rig = rigAfterTurn(this.rig, this.headLocal(), x > 0 ? -SNAP_TURN : SNAP_TURN);
        this.applyRig();
      }
    } else if (Math.abs(x) < 0.3) {
      state.turnLatched = false;
    }

    // Teleport aim on the stick pushed forward; released, it jumps.
    if (y < -0.65 && Math.abs(x) < 0.6) {
      state.stickAim = true;
    } else if (state.stickAim && Math.hypot(x, y) < 0.3) {
      state.stickAim = false;
      this.teleportTo(state.verdict);
    }
    return 0;
  }

  private selectStart(source: PC.XrInputSource): void {
    if (this.fade) return;
    const state = this.stateOf(source);
    const region = this.panelRegionUnder(source);
    if (region) {
      state.panelPress = region;
      return;
    }
    if (source.targetRayMode === "gaze") return;
    state.aiming = true;
  }

  private selectEnd(source: PC.XrInputSource): void {
    const state = this.stateOf(source);
    if (state.panelPress) {
      const region = this.panelRegionUnder(source);
      if (region === state.panelPress) this.activate(region);
      state.panelPress = null;
      return;
    }
    if (state.aiming) {
      state.aiming = false;
      if (source.targetRayMode === "tracked-pointer") {
        this.teleportTo(state.verdict);
      } else {
        // A Vision Pro pinch or a screen tap: no arc was drawn, so aim the plain ray now.
        const o = source.getOrigin();
        const d = source.getDirection();
        const head = this.host.camera.getPosition();
        const landing = rayToFloor([o.x, o.y, o.z], [d.x, d.y, d.z], floorAt(this.space, head.x, head.z), 12);
        if (landing) this.teleportTo(validateTeleport([landing[0], landing[2]], this.space));
      }
    }
  }

  private select(source: PC.XrInputSource): void {
    // A gaze-only device's button means "this", whatever the head is resting on.
    if (source.targetRayMode === "gaze" && this.gazeTarget) this.activate(this.gazeTarget);
  }

  private panelRegionUnder(source: PC.XrInputSource): string | null {
    if (!this.panelGeometry) return null;
    const o = source.getOrigin();
    const d = source.getDirection();
    const hit = rayPanel([o.x, o.y, o.z], [d.x, d.y, d.z], this.panelGeometry);
    return hit ? regionAt(hit, this.panelRegions) : null;
  }

  private activate(target: string): void {
    resetDwell(this.dwell, performance.now(), DWELL_DEFAULTS.cooldownMs);
    if (target === "panel:exit") {
      this.options.onExit();
    } else if (target === "panel:smooth") {
      this.smooth = !this.smooth;
      this.panelDrawn = "";
    } else if (target.startsWith("floor:")) {
      if (this.gazeAnchor && target === `floor:${this.gazeAnchor.id}`) this.teleportTo(this.gazeAnchor.verdict);
      this.gazeAnchor = null;
    }
  }

  // ---------------------------------------------------------------------------------
  // Teleport
  // ---------------------------------------------------------------------------------

  private teleportTo(verdict: TeleportVerdict | null): void {
    if (!verdict?.valid || this.fade) return;
    const head = this.host.camera.getPosition();
    if (Math.hypot(verdict.point[0] - head.x, verdict.point[1] - head.z) < MIN_TELEPORT) return;
    this.fade = { phase: "out", t: 0, verdict };
  }

  private stepFade(dt: number): void {
    const fade = this.fade;
    if (!fade) return;
    fade.t += dt;
    if (fade.phase === "out") {
      this.veil.setOpacity(Math.min(1, fade.t / FADE_OUT_S));
      if (fade.t >= FADE_OUT_S) {
        const verdict = fade.verdict;
        if (verdict) {
          this.rig = rigForTeleport(this.rig, this.headLocal(), verdict.point);
          this.rigY = this.floorRig(verdict.floorY);
          this.applyRig();
          this.host.pose.position = [verdict.point[0], this.host.pose.position[1], verdict.point[1]];
          this.host.relocated();
        }
        this.fade = { phase: "in", t: 0, verdict: null };
      }
    } else {
      this.veil.setOpacity(Math.max(0, 1 - fade.t / FADE_IN_S));
      if (fade.t >= FADE_IN_S) {
        this.veil.setOpacity(0);
        this.fade = null;
      }
    }
  }

  // ---------------------------------------------------------------------------------
  // Visuals
  // ---------------------------------------------------------------------------------

  private pooled(index: number, p: Vec3): PC.Vec3 {
    let v = this.arcPool[index];
    if (!v) {
      v = new this.pc.Vec3();
      this.arcPool[index] = v;
    }
    return v.set(p[0], p[1], p[2]);
  }

  private pushArc(points: Vec3[]): void {
    for (let i = 0; i + 1 < points.length; i += 1) {
      const n = this.arcPoints.length;
      this.arcPoints.push(this.pooled(n, points[i]), this.pooled(n + 1, points[i + 1]));
    }
  }

  private drawArc(aiming: boolean, verdict: TeleportVerdict | null): void {
    if (!aiming || this.arcPoints.length < 2) return;
    // The same two colours as the landing ring: teal where it may land, amber where it may not.
    this.arcColours ??= { valid: new this.pc.Color(0.09, 0.89, 0.77, 1), refused: new this.pc.Color(1, 0.71, 0.3, 1) };
    this.app.drawLines(this.arcPoints, verdict?.valid ? this.arcColours.valid : this.arcColours.refused, false);
  }

  private placeMarker(verdict: TeleportVerdict | null, landing: Vec3 | null, progress: number): void {
    const show = verdict !== null && landing !== null && this.fade === null;
    this.marker.entity.enabled = show;
    if (!show || !verdict || !landing) return;
    const point = verdict.valid ? verdict.point : ([landing[0], landing[2]] as [number, number]);
    const floorY = verdict.valid ? verdict.floorY : landing[1];
    this.marker.entity.setPosition(point[0], floorY + 0.01, point[1]);
    this.marker.entity.setEulerAngles(-90, 0, 0);
    const key = `${verdict.valid}|${Math.round(progress * 24)}`;
    if (key !== this.markerDrawn) {
      drawMarker(this.markerCanvas, verdict.valid, Math.round(progress * 24) / 24);
      if (this.marker.texture) refreshTexture(this.marker.texture);
      this.markerDrawn = key;
    }
  }

  private placeReticle(at: Vec3 | null, head: Vec3, active: boolean, progress: number): void {
    this.reticle.entity.enabled = at !== null;
    if (!at) return;
    const distance = Math.hypot(at[0] - head[0], at[1] - head[1], at[2] - head[2]);
    this.reticle.entity.setPosition(at[0], at[1], at[2]);
    this.reticle.entity.lookAt(head[0], head[1], head[2]);
    // lookAt points -Z at the head; the quad faces +Z, so turn it round.
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

  private panelItems(): PanelItem[] {
    return [
      { type: "text", title: this.options.copy.areaName(this.host.activeKey()) },
      { type: "button", id: "panel:smooth", icon: "walk", label: this.options.copy.smooth, pressed: this.smooth },
      { type: "button", id: "panel:exit", icon: "close", label: this.options.copy.exit, tone: "danger" },
    ];
  }

  /** The menu: below the line of sight, following the body lazily (see `panoVrWorld`). */
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
    const geometry = panelFacing(centre, head, half, half * (PANEL_HEIGHT / PANEL_CANVAS));
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
    const key = `${this.host.activeKey()}|${this.smooth}|${hovered}|${Math.round(progress * 24)}`;
    if (key === this.panelDrawn) return;
    this.panelRegions = drawPanel(this.panelCanvas, this.panelItems(), { hovered, progress: Math.round(progress * 24) / 24 }, PANEL_CANVAS);
    if (this.panel.texture) refreshTexture(this.panel.texture);
    this.panelDrawn = key;
  }

  private stepVignette(dt: number, moving: number): void {
    const target = this.smooth ? Math.min(1, moving) * 0.85 : 0;
    this.vignetteLevel += (target - this.vignetteLevel) * Math.min(1, dt * (target > this.vignetteLevel ? 10 : 4));
    this.vignette.setOpacity(this.vignetteLevel);
  }

  // ---------------------------------------------------------------------------------
  // Frame rate
  // ---------------------------------------------------------------------------------

  private adaptResolution(dt: number, xr: PC.XrManager): void {
    const f = this.frame;
    const budget = 1 / (xr.frameRate || 72);
    const ms = Math.min(dt * 1000, 200);
    f.ms = f.ms === 0 ? ms : f.ms * 0.9 + ms * 0.1;
    f.count += 1;
    if (dt > budget * 1.4) f.slow += 1;
    f.since += dt;
    if (f.since < 1.5) return;
    const missed = f.slow / Math.max(1, f.count);
    f.since = 0;
    f.slow = 0;
    f.count = 0;
    const now = performance.now();
    // Same fields as the 2D viewer's readout (`?perf=1`), plus what only a headset has.
    (window as unknown as { __walkPerf?: unknown }).__walkPerf = {
      fps: Math.round(1000 / Math.max(f.ms, 1)),
      frameMs: +f.ms.toFixed(1),
      scale: +f.scale.toFixed(2),
      dpr: window.devicePixelRatio || 1,
      mode: "xr",
      missedFrames: +missed.toFixed(3),
      targetHz: Math.round(1 / budget),
    };
    if (now - f.changedAt < 2500) return;
    if (missed > 0.15 && f.scale > 0.45) {
      this.rescale(Math.max(0.45, f.scale * 0.85), xr);
      f.calm = 0;
    } else if (missed < 0.02) {
      f.calm += 1;
      // Four quiet windows (six seconds) before giving pixels back, and never past the start.
      if (f.calm >= 4 && f.scale < this.options.scale) {
        this.rescale(Math.min(this.options.scale, f.scale * 1.08), xr);
        f.calm = 0;
      }
    } else {
      f.calm = 0;
    }
  }

  private rescale(scale: number, xr: PC.XrManager): void {
    const device = this.app.graphicsDevice as PC.GraphicsDevice & { gl?: WebGL2RenderingContext };
    const session = xr.session;
    if (!session || !device.isWebGL2 || !device.gl || typeof XRWebGLLayer === "undefined") return;
    try {
      const layer = new XRWebGLLayer(session, device.gl, {
        alpha: true,
        depth: true,
        stencil: true,
        antialias: false,
        framebufferScaleFactor: scale,
      });
      if ("fixedFoveation" in layer) (layer as XRWebGLLayer & { fixedFoveation: number }).fixedFoveation = 1;
      session.updateRenderState({ baseLayer: layer });
      this.frame.scale = scale;
      this.frame.changedAt = performance.now();
    } catch {
      // Some runtimes refuse a new layer mid-session; the starting scale then simply stays.
    }
  }
}

function wrap(angle: number): number {
  const m = ((((angle + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
  return m === -Math.PI ? Math.PI : m;
}
