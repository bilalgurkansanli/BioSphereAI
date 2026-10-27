/**
 * The capture-link door, without a browser.
 *
 * Every decision here is one wrong line away from a phone that films a room and then cannot
 * upload it — or from a QR code that quietly signs its holder in as the agent everywhere. The
 * engine enforces the scope (`engine/tests/test_capture_grant.py`); these pin what the client
 * does with the token: where it keeps it, when it is used, and what happens when it is refused.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  CAPTURE_GRANT_EVENT,
  adoptCaptureGrantFromAddress,
  announceSignedOut,
  authHeaders,
  captureGateMode,
  captureLinkCredentials,
  getCaptureGrant,
  setCaptureGrant,
  setToken,
} from "../session";

class MemoryStorage {
  private readonly map = new Map<string, string>();
  getItem(key: string): string | null {
    return this.map.has(key) ? (this.map.get(key) as string) : null;
  }
  setItem(key: string, value: string): void {
    this.map.set(key, String(value));
  }
  removeItem(key: string): void {
    this.map.delete(key);
  }
  clear(): void {
    this.map.clear();
  }
}

interface FakeWindow {
  location: URL;
  localStorage: MemoryStorage;
  sessionStorage: MemoryStorage;
  history: { state: unknown; replaceState(state: unknown, title: string, url: string): void };
  addEventListener: EventTarget["addEventListener"];
  removeEventListener: EventTarget["removeEventListener"];
  dispatchEvent: EventTarget["dispatchEvent"];
}

const TOKEN = "P-cwk-O3Ak3EHRfFKIsNOBdBIsiX7iYTrJWY7H3MdcA";
let win: FakeWindow;
let grantEvents = 0;

function open(href: string): void {
  const target = new EventTarget();
  win = {
    location: new URL(href),
    localStorage: new MemoryStorage(),
    sessionStorage: new MemoryStorage(),
    history: {
      state: null,
      replaceState(_state, _title, url) {
        win.location = new URL(url, win.location.href);
      },
    },
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
    dispatchEvent: target.dispatchEvent.bind(target),
  };
  vi.stubGlobal("window", win);
  grantEvents = 0;
  win.addEventListener(CAPTURE_GRANT_EVENT, () => {
    grantEvents += 1;
  });
}

beforeEach(() => open("https://app.example/cek?property=prop123456&k=" + TOKEN));
afterEach(() => vi.unstubAllGlobals());

describe("adopting a link from a QR code", () => {
  it("waits while the token is still in the address, so the sign-in guard never mounts first", () => {
    expect(captureGateMode()).toBe("pending");
  });

  it("moves the token into this tab's storage and out of the address bar", () => {
    adoptCaptureGrantFromAddress();
    expect(win.location.searchParams.has("k")).toBe(false);
    expect(win.location.searchParams.get("property")).toBe("prop123456");
    expect(getCaptureGrant()).toEqual({ token: TOKEN, propertyId: "prop123456" });
    expect(win.localStorage.getItem("biosphere.session")).toBeNull();
    expect(captureGateMode()).toBe("grant");
    expect(grantEvents).toBe(1);
  });

  it("is idempotent: a second call with no token in the address changes nothing", () => {
    adoptCaptureGrantFromAddress();
    adoptCaptureGrantFromAddress();
    expect(getCaptureGrant()?.token).toBe(TOKEN);
    expect(grantEvents).toBe(1);
  });

  it("sends the grant as the bearer when there is no session", () => {
    adoptCaptureGrantFromAddress();
    expect(authHeaders()).toEqual({ Authorization: `Bearer ${TOKEN}` });
  });

  it("drops a cut-off token instead of adopting it", () => {
    open("https://app.example/cek?property=prop123456&k=abc");
    adoptCaptureGrantFromAddress();
    expect(getCaptureGrant()).toBeNull();
    expect(win.location.searchParams.has("k")).toBe(false);
    expect(captureGateMode()).toBe("session"); // → the ordinary sign-in door
    expect(grantEvents).toBe(1); // the gate still re-reads
  });

  it("drops a token that arrived without its property", () => {
    open("https://app.example/cek?k=" + TOKEN);
    adoptCaptureGrantFromAddress();
    expect(getCaptureGrant()).toBeNull();
  });
});

describe("a session and a grant together", () => {
  it("prefers the session: an agent signed in on their phone keeps the full account", () => {
    adoptCaptureGrantFromAddress();
    setToken("session-token");
    expect(captureGateMode()).toBe("session");
    expect(authHeaders()).toEqual({ Authorization: "Bearer session-token" });
    expect(captureLinkCredentials("prop123456")).toBe("session");
  });
});

describe("when the engine refuses the grant", () => {
  it("drops it, remembers why, and tells the gate", () => {
    adoptCaptureGrantFromAddress();
    grantEvents = 0;
    announceSignedOut();
    expect(getCaptureGrant()).toBeNull();
    expect(captureGateMode()).toBe("expired");
    expect(authHeaders()).toEqual({});
    expect(grantEvents).toBe(1);
  });

  it("a fresh link clears the expired note", () => {
    adoptCaptureGrantFromAddress();
    announceSignedOut();
    setCaptureGrant({ token: TOKEN, propertyId: "prop123456" });
    expect(captureGateMode()).toBe("grant");
  });

  it("a plain sign-out with no grant does not invent an expired link", () => {
    open("https://app.example/evler");
    setToken("session-token");
    announceSignedOut();
    expect(captureGateMode()).toBe("session");
    expect(grantEvents).toBe(0);
  });
});

describe("what the QR card may do with what this tab holds", () => {
  it("passes a held link on instead of minting (minting with it would be refused and kill it)", () => {
    adoptCaptureGrantFromAddress();
    expect(captureLinkCredentials("prop123456")).toBe(`grant:${TOKEN}`);
  });

  it("never passes one property's link off as another's", () => {
    adoptCaptureGrantFromAddress();
    expect(captureLinkCredentials("someOtherProp")).toBe("none");
  });

  it("has nothing to give with neither a session nor a link", () => {
    open("https://app.example/cek?property=prop123456");
    expect(captureLinkCredentials("prop123456")).toBe("none");
  });
});
