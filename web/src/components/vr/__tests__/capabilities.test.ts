/**
 * Which VR button a visitor gets, row by row. Every row is a real device in a real place: a
 * Quest on our own page, a Quest inside a portal that forgot the `allow` attribute, an iPhone
 * inside the same portal, a laptop.
 */

import { describe, expect, it } from "vitest";
import {
  chooseVrRoute,
  handoffUrl,
  isAppleTouch,
  isHeadsetBrowser,
  parseVrQuery,
  xrFramebufferScale,
  type VrEnvironment,
} from "../capabilities";

const UA = {
  quest3:
    "Mozilla/5.0 (X11; Linux x86_64; Quest 3) AppleWebKit/537.36 (KHTML, like Gecko) OculusBrowser/35.2.0.4.53 SamsungBrowser/4.0 Chrome/128.0.6613.137 VR Safari/537.36",
  quest2:
    "Mozilla/5.0 (X11; Linux x86_64; Quest 2) AppleWebKit/537.36 (KHTML, like Gecko) OculusBrowser/34.1.0.3.56 SamsungBrowser/4.0 Chrome/126.0.6478.122 VR Safari/537.36",
  iphone:
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
  android:
    "Mozilla/5.0 (Linux; Android 14; SM-A546B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Mobile Safari/537.36",
  mac: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
  windows:
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
};

function env(overrides: Partial<VrEnvironment>): VrEnvironment {
  return {
    ua: UA.windows,
    embedded: false,
    xr: "absent",
    phone: false,
    orientationEvents: true,
    fullscreen: true,
    sensorsAllowed: true,
    apple: false,
    debugCardboard: false,
    ...overrides,
  };
}

describe("route", () => {
  it("uses WebXR wherever it is supported", () => {
    expect(chooseVrRoute(env({ ua: UA.quest3, xr: "supported" }))).toBe("xr");
    // Even on a phone: a native runtime beats our split screen.
    expect(chooseVrRoute(env({ ua: UA.android, xr: "supported", phone: true }))).toBe("xr");
  });

  it("sends a headset in a portal without `allow` to a new tab", () => {
    expect(chooseVrRoute(env({ ua: UA.quest2, xr: "blocked", embedded: true }))).toBe("handoff");
  });

  it("does not offer a desktop a new tab it cannot use", () => {
    expect(chooseVrRoute(env({ ua: UA.windows, xr: "blocked", embedded: true }))).toBeNull();
    expect(chooseVrRoute(env({ ua: UA.windows, xr: "unsupported" }))).toBeNull();
    expect(chooseVrRoute(env({ ua: UA.mac, xr: "absent" }))).toBeNull();
  });

  it("gives a phone the Cardboard path on our own page", () => {
    expect(chooseVrRoute(env({ ua: UA.iphone, phone: true, apple: true, fullscreen: false }))).toBe("cardboard");
    expect(chooseVrRoute(env({ ua: UA.android, phone: true }))).toBe("cardboard");
  });

  it("hands an iPhone in an iframe to a new tab — no fullscreen, no reliable sensor prompt", () => {
    expect(chooseVrRoute(env({ ua: UA.iphone, phone: true, apple: true, embedded: true, fullscreen: false }))).toBe("handoff");
  });

  it("keeps an Android phone in a well-configured iframe in place", () => {
    expect(chooseVrRoute(env({ ua: UA.android, phone: true, embedded: true, fullscreen: true }))).toBe("cardboard");
  });

  it("hands an Android phone in an iframe without sensor permission to a new tab", () => {
    expect(chooseVrRoute(env({ ua: UA.android, phone: true, embedded: true, sensorsAllowed: false }))).toBe("handoff");
    expect(chooseVrRoute(env({ ua: UA.android, phone: true, embedded: true, fullscreen: false }))).toBe("handoff");
  });

  it("shows nothing to a tablet or a phone without motion events", () => {
    expect(chooseVrRoute(env({ ua: UA.android, phone: false }))).toBeNull();
    expect(chooseVrRoute(env({ ua: UA.android, phone: true, orientationEvents: false }))).toBeNull();
  });

  it("forces Cardboard for the desktop debug switch", () => {
    expect(chooseVrRoute(env({ debugCardboard: true }))).toBe("cardboard");
    // A real headset still wins: the switch is for looking at the phone path, not hiding XR.
    expect(chooseVrRoute(env({ debugCardboard: true, xr: "supported" }))).toBe("xr");
  });
});

describe("browser sniffing", () => {
  it("knows a headset browser", () => {
    expect(isHeadsetBrowser(UA.quest3)).toBe(true);
    expect(isHeadsetBrowser(UA.quest2)).toBe(true);
    expect(isHeadsetBrowser(UA.android)).toBe(false);
    expect(isHeadsetBrowser(UA.windows)).toBe(false);
  });

  it("knows Apple touch devices, including the iPad that says it is a Mac", () => {
    expect(isAppleTouch(UA.iphone, 5)).toBe(true);
    expect(isAppleTouch(UA.mac, 5)).toBe(true);
    expect(isAppleTouch(UA.mac, 0)).toBe(false);
    expect(isAppleTouch(UA.android, 5)).toBe(false);
  });

  it("asks a splat headset for fewer pixels than a panorama, per device", () => {
    expect(xrFramebufferScale(UA.quest3, "panorama")).toBe(1);
    expect(xrFramebufferScale(UA.quest3, "splats")).toBe(0.8);
    expect(xrFramebufferScale(UA.quest2, "splats")).toBe(0.65);
    expect(xrFramebufferScale(UA.mac, "splats")).toBe(0.5);
    expect(xrFramebufferScale(UA.windows, "splats")).toBe(0.7);
  });
});

describe("query", () => {
  it("reads the debug switch, the handoff flag and the lens override", () => {
    expect(parseVrQuery("?vr=cardboard")).toEqual({ debugCardboard: true, handoff: false, lensOverride: null });
    expect(parseVrQuery("?scene=a&vr=1")).toMatchObject({ debugCardboard: false, handoff: true });
    expect(parseVrQuery("?vrlens=0.3,0.2").lensOverride).toBe("0.3,0.2");
    expect(parseVrQuery("")).toEqual({ debugCardboard: false, handoff: false, lensOverride: null });
  });
});

describe("handoff url", () => {
  it("keeps the room and the view, and asks for VR", () => {
    const url = new URL(handoffUrl("https://tur.example/embed/abc?autorotate=1", { scene: "s2", yaw: 1.23456789, pitch: -0.1, mode: null }));
    expect(url.pathname).toBe("/embed/abc");
    expect(url.searchParams.get("scene")).toBe("s2");
    expect(url.searchParams.get("yaw")).toBe("1.2346");
    expect(url.searchParams.get("pitch")).toBe("-0.1");
    expect(url.searchParams.get("vr")).toBe("1");
    // What the portal put on the link survives.
    expect(url.searchParams.get("autorotate")).toBe("1");
    expect(url.searchParams.has("mode")).toBe(false);
  });

  it("replaces a stale debug switch rather than stacking it", () => {
    const url = new URL(handoffUrl("https://tur.example/embed/abc?vr=cardboard", { mode: "walk", scene: "base" }));
    expect(url.searchParams.getAll("vr")).toEqual(["1"]);
    expect(url.searchParams.get("mode")).toBe("walk");
  });
});
