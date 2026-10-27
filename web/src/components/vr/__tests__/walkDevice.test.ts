/**
 * Which graphics API the walkthrough asks for. Getting this wrong is silent: on a Quest the
 * walkthrough would come up on WebGPU and its VR button would simply never work.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { walkDeviceTypes } from "../walk/host";

function stubXr(isSessionSupported: () => Promise<boolean>) {
  vi.stubGlobal("navigator", { xr: { isSessionSupported } });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("walkthrough graphics API", () => {
  it("keeps WebGPU first where there is no WebXR at all", async () => {
    vi.stubGlobal("navigator", {});
    expect(await walkDeviceTypes()).toEqual(["webgpu", "webgl2"]);
  });

  it("keeps WebGPU first on a desktop without a headset", async () => {
    stubXr(async () => false);
    expect(await walkDeviceTypes()).toEqual(["webgpu", "webgl2"]);
  });

  it("takes WebGL2 on a device that can present VR without WebGPU XR binding", async () => {
    stubXr(async () => true);
    expect(await walkDeviceTypes()).toEqual(["webgl2"]);
  });

  it("keeps WebGPU where the browser can present VR from WebGPU", async () => {
    stubXr(async () => true);
    vi.stubGlobal("XRGPUBinding", class {});
    expect(await walkDeviceTypes()).toEqual(["webgpu", "webgl2"]);
  });

  it("does not hold the first paint hostage to a hanging XR runtime", async () => {
    vi.useFakeTimers();
    stubXr(() => new Promise<boolean>(() => undefined));
    const pending = walkDeviceTypes();
    await vi.advanceTimersByTimeAsync(800);
    expect(await pending).toEqual(["webgpu", "webgl2"]);
  });

  it("falls back to the default order when the probe throws", async () => {
    stubXr(async () => {
      throw new DOMException("blocked", "SecurityError");
    });
    expect(await walkDeviceTypes()).toEqual(["webgpu", "webgl2"]);
  });
});
