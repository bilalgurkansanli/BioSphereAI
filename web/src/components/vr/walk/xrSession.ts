/**
 * Starting and ending a headset session on the walkthrough's engine.
 *
 * Main-bundle code (the click that starts VR must reach `requestSession` synchronously, and
 * PlayCanvas's `xr.start` makes that call synchronously) and deliberately plain: no React, so
 * the engine's objects are only ever changed from here and from the runtime classes.
 *
 * ── THE RIG ───────────────────────────────────────────────────────────────────────────────
 * PlayCanvas writes the headset's pose into the camera entity's LOCAL transform, so moving the
 * visitor means moving the camera's parent. The camera is put under a rig entity before the
 * session starts — the first frame must not show the model from the world origin — and put
 * back, at the head's last position and heading, when the session ends however it ends: the
 * menu, the headset's own button, or the browser.
 */

import type * as PC from "playcanvas";
import { EYE_HEIGHT, yawOf } from "@/components/walk/walkMath";
import { xrFramebufferScale } from "../capabilities";
import type { WalkVrHost } from "./host";

export interface WalkXrSession {
  rig: PC.Entity;
  space: "local-floor" | "local";
  scale: number;
}

export interface StartWalkXrHandlers {
  started: (session: WalkXrSession) => void;
  failed: (error: Error) => void;
  ended: () => void;
}

/** Whether this engine can present VR at all (a WebGPU device without XR binding cannot). */
export function walkXrAvailable(host: WalkVrHost): boolean {
  const manager = host.app.xr;
  return Boolean(manager && host.camera.camera && manager.isAvailable(host.pc.XRTYPE_VR));
}

/** Starts a session; call from inside the click. Returns false if one could not even be asked for. */
export function startWalkXr(host: WalkVrHost, handlers: StartWalkXrHandlers): boolean {
  const { pc, app, camera } = host;
  const manager = app.xr;
  const component = camera.camera;
  if (!manager || !component || manager.active) return false;

  const rig = new pc.Entity("vr-rig");
  app.root.addChild(rig);
  // Provisional, and already nearly right: a `local-floor` origin starts under the head and
  // faces where the head faces, so a rig at the 2D position, turned to the 2D heading, shows
  // the same view from the first frame. The runtime then corrects the remainder on that frame —
  // it is lazy-loaded, and the frames before it arrives must not show the model from nowhere.
  rig.setPosition(host.pose.position[0], 0, host.pose.position[2]);
  rig.setEulerAngles(0, (host.pose.yaw * 180) / Math.PI, 0);
  camera.reparent(rig);
  camera.setLocalPosition(0, 0, 0);
  camera.setLocalEulerAngles(0, 0, 0);
  host.driven = true;
  const radialWas = app.scene.gsplat.radialSorting;
  // Radial order does not change when the head turns, so turning never forces a re-sort.
  app.scene.gsplat.radialSorting = true;

  const restore = () => {
    const head = camera.getPosition().clone();
    const forward = camera.forward.clone();
    camera.reparent(app.root);
    rig.destroy();
    host.pose.position = [head.x, EYE_HEIGHT, head.z];
    host.pose.yaw = yawOf([forward.x, forward.y, forward.z]);
    host.pose.pitch = 0;
    app.scene.gsplat.radialSorting = radialWas;
    host.driven = false;
    host.fade(0);
  };

  // PlayCanvas multiplies the requested scale by maxPixelRatio / devicePixelRatio, and the
  // engine steers maxPixelRatio as its 2D render scale. Divide that back out so the headset gets
  // the scale chosen for it (`xrFramebufferScale`), whatever the 2D view had settled at.
  const scale = xrFramebufferScale(navigator.userAgent, "splats");
  const relative = app.graphicsDevice.maxPixelRatio / (window.devicePixelRatio || 1);

  const attempt = (space: "local-floor" | "local") => {
    manager.start(component, pc.XRTYPE_VR, space === "local-floor" ? pc.XRSPACE_LOCALFLOOR : pc.XRSPACE_LOCAL, {
      framebufferScaleFactor: scale / Math.max(relative, 0.1),
      callback: (error: Error | null) => {
        if (error) {
          // A runtime without a floor (rare: seated-only devices) still has `local`.
          if (space === "local-floor" && error instanceof DOMException && error.name === "NotSupportedError") {
            attempt("local");
            return;
          }
          restore();
          handlers.failed(error);
          return;
        }
        manager.once("end", () => {
          restore();
          handlers.ended();
        });
        handlers.started({ rig, space, scale });
      },
    });
  };
  attempt("local-floor");
  return true;
}

export function endWalkXr(host: WalkVrHost): void {
  const manager = host.app.xr;
  if (manager?.active) manager.end();
}
