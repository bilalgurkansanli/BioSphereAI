/**
 * What the walkthrough's engine (`walk/useWalkEngine.ts`) hands its VR modes, and the one
 * decision it has to take before VR exists: which graphics API to create.
 *
 * The contract is deliberately narrow. While a VR mode runs it sets `driven`, and the engine's
 * frame loop then leaves the camera alone — no mouse look, no keys, no walk-to, no auto tour, no
 * render-scale steering — while it keeps doing the things that follow the VISITOR: streaming
 * areas in, swapping the drawn area as they move (`pose` is written by the VR mode every frame
 * with the head's position), and moving the floor plan's you-are-here mark. Everything
 * headset-specific lives in `components/vr/walk/`, out of the engine.
 *
 * Type-only imports: this file is in the walkthrough's main bundle (the engine imports it) and
 * must not pull PlayCanvas or any VR code in with it.
 */

import type * as PC from "playcanvas";
import type { WalkManifest } from "@/lib/types";
import type { Vec3 } from "@/components/walk/walkMath";

export interface WalkVrHost {
  readonly pc: typeof PC;
  readonly app: PC.AppBase;
  /** The engine's camera. A headset session reparents it under a rig and puts it back. */
  readonly camera: PC.Entity;
  /** The environment sphere, which must stay centred on the eye. */
  readonly env: PC.Entity;
  /** The engine's pose: VR writes the head here every frame so streaming follows the visitor. */
  readonly pose: { position: Vec3; yaw: number; pitch: number; fov: number };
  readonly manifest: WalkManifest;
  /** True while a VR mode owns the camera. */
  driven: boolean;
  /** The area being drawn. */
  activeKey(): string | null;
  /** After a teleport: choose the area for the new position at once, not after the dwell. */
  relocated(): void;
  /** End the auto tour, if one is running: a headset is not driven around. */
  stopTour(): void;
  /** The page's own fade to black — visible on a phone in a box, not inside a headset. */
  fade(opacity: number): void;
}

/**
 * The graphics APIs the walkthrough may use, in order of preference.
 *
 * WebGPU first, as the engine has always asked — EXCEPT on a device that can present VR.
 * PlayCanvas can only start a WebXR session on a WebGPU device where the browser exposes
 * `XRGPUBinding` (`XrManager._backendSupportsXr`), and Quest Browser ships WebGPU without it:
 * the walkthrough would come up on WebGPU, report `immersive-vr` unavailable, and the VR button
 * would work for the 360 and silently not here. So where `immersive-vr` is supported and
 * `XRGPUBinding` is not, the device is WebGL2 from the start. The price is CPU splat sorting on
 * that device instead of GPU sorting — which is what PlayCanvas's own XR splat examples run on.
 *
 * The probe is capped at 800 ms so a slow XR runtime cannot hold the walkthrough's first paint.
 */
export async function walkDeviceTypes(): Promise<string[]> {
  const preferred = ["webgpu", "webgl2"];
  if (typeof navigator === "undefined" || !navigator.xr?.isSessionSupported) return preferred;
  if (typeof (globalThis as { XRGPUBinding?: unknown }).XRGPUBinding !== "undefined") return preferred;
  try {
    const supported = await Promise.race([
      navigator.xr.isSessionSupported("immersive-vr"),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 800)),
    ]);
    return supported ? ["webgl2"] : preferred;
  } catch {
    return preferred;
  }
}
