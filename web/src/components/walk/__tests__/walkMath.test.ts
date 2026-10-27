import { describe, expect, it } from "vitest";
import type { WalkArea, WalkFloorPlan, WalkManifest } from "@/lib/types";
import {
  AREA_SWITCH_DWELL_S,
  EYE_HEIGHT,
  SCALE_MIN,
  HFOV_OPEN_MAX,
  PATH_RADIUS_M,
  areaAt,
  clampToBox,
  constrainToCorridor,
  corridorOf,
  defaultFov,
  entryPose,
  lookAlong,
  lookOf,
  nearestOnCorridor,
  forwardOf,
  nextRenderScale,
  pathLength,
  planFraction,
  planPoint,
  splinePoint,
  timeOf,
  toEngine,
  tourOrder,
  toViewer,
  updateAreaSwitch,
  viewerBox,
  yawDelta,
  yawOf,
} from "../walkMath";

function area(key: string, min: number[], max: number[], extra: Partial<WalkArea> = {}): WalkArea {
  return {
    key,
    name: key,
    floor_index: 0,
    url: `${key}.sog`,
    bytes: 1,
    splat_count: 1,
    thumbnail_url: null,
    env_pano_url: null,
    cameras: [[0, -1.4, 0], [1, -1.4, 0], [2, -1.4, 1]],
    centroid: [1, -1.4, 0.3],
    entry: { position: [0, -1.4, 0], forward: [0, 0, 1] },
    bbox: { min, max },
    placed: true,
    warnings: [],
    failed: null,
    ...extra,
  };
}

const manifest: WalkManifest = {
  version: 1,
  frame: {},
  base: area("base", [-10, -2.7, -10], [10, 0.3, 10]),
  rooms: [
    area("big", [0, -2.7, 0], [6, 0.3, 6]),
    area("small", [0, -2.7, 0], [3, 0.3, 3]),
    area("ghost", [-5, -2.7, -5], [-1, 0.3, -1], { placed: false }),
  ],
  floor_plan: null,
  metrics: {},
  warnings: [],
};

describe("frames", () => {
  it("flips y and z both ways", () => {
    expect(toViewer([1, 2, 3])).toEqual([1, -2, -3]);
    expect(toEngine(toViewer([1, 2, 3]))).toEqual([1, 2, 3]);
  });
  it("keeps boxes ordered after the flip", () => {
    const b = viewerBox(manifest.rooms[0]);
    expect(b.min).toEqual([0, -0.3, -6]);
    expect(b.max).toEqual([6, 2.7, 0]);
  });
});

describe("areaAt", () => {
  // Engine (x, y, z) shows up at viewer (x, -y, -z), so a camera at engine (1, -1.4, 1)
  // stands at viewer (1, 1.4, -1).
  const paths: WalkManifest = {
    ...manifest,
    base: area("base", [-10, -2.7, -10], [10, 0.3, 10], { cameras: [[-8, -1.4, -8], [0, -1.4, 0], [8, -1.4, 8]] }),
    rooms: [
      area("big", [0, -2.7, 0], [6, 0.3, 6], { cameras: [[5, -1.4, 5]] }),
      area("small", [0, -2.7, 0], [3, 0.3, 3], { cameras: [[1, -1.4, 1]] }),
      area("ghost", [-5, -2.7, -5], [-1, 0.3, -1], { placed: false, cameras: [[-3, -1.4, -3]] }),
    ],
  };

  it("picks the area whose camera path passes closest", () => {
    expect(areaAt(paths, [1, 1.6, -1]).key).toBe("small");
    expect(areaAt(paths, [5, 1.6, -5]).key).toBe("big");
  });

  it("ignores unplaced areas and falls back to the base outside every box", () => {
    expect(areaAt(paths, [-3, 1.6, 3]).key).toBe("base");
    expect(areaAt(paths, [9, 1.6, -9]).key).toBe("base");
  });

  it("does not hand a walk to a fragment whose box happens to be smallest", () => {
    // The parts of one broken walk overlap almost entirely. A tiny box that contains the
    // visitor is not evidence they are in it: the walk that was filmed from where they
    // stand is, and here that is the base.
    const fragment = area("fragment", [0, -2.7, 0], [2, 0.3, 2], { cameras: [[2, -1.4, 2]] });
    const withFragment: WalkManifest = { ...paths, rooms: [...paths.rooms, fragment] };
    expect(areaAt(withFragment, [0.2, 1.6, -0.2]).key).toBe("base");
  });

  it("keeps the area in use when another is only marginally nearer", () => {
    // Dead level between the base path and the small room: without stickiness this point
    // hands the walk to the base, and half a step later hands it back.
    expect(areaAt(paths, [0.5, 1.6, -0.5]).key).toBe("base");
    expect(areaAt(paths, [0.5, 1.6, -0.5], "small").key).toBe("small");
  });

  it("still gives way once the other area is clearly nearer", () => {
    expect(areaAt(paths, [0.2, 1.6, -0.2], "small").key).toBe("base");
  });

  it("never hands the walk to an area the base already covers", () => {
    // Delivery measured that this piece shows no more than the base does at its own
    // viewpoints. Standing right on its camera, the base still wins.
    const covered: WalkManifest = {
      ...paths,
      rooms: paths.rooms.map((r) => (r.key === "small" ? { ...r, redundant: true } : r)),
    };
    expect(areaAt(covered, [1, 1.6, -1]).key).toBe("base");
    expect(areaAt(covered, [1, 1.6, -1], "small").key).toBe("base");
  });

  it("does not stick to an area the visitor has left", () => {
    // Outside the small room's box it is not a candidate at all, held or not.
    expect(areaAt(paths, [9, 1.6, -9], "small").key).toBe("base");
  });
});

describe("area switch dwell", () => {
  it("takes the first area at once", () => {
    const state = { pending: null as string | null, held: 0 };
    expect(updateAreaSwitch(state, "base", null, 0.016)).toBe("base");
  });

  const most = AREA_SWITCH_DWELL_S * 0.6; // two of these pass the threshold, one does not

  it("waits for the candidate to hold its claim", () => {
    const state = { pending: null as string | null, held: 0 };
    expect(updateAreaSwitch(state, "small", "base", most)).toBeNull();
    expect(updateAreaSwitch(state, "small", "base", most)).toBe("small");
  });

  it("forgets a candidate that does not last", () => {
    const state = { pending: null as string | null, held: 0 };
    updateAreaSwitch(state, "small", "base", most);
    expect(updateAreaSwitch(state, "base", "base", most)).toBeNull();
    expect(state.pending).toBeNull();
    // The claim starts again from zero, so a flicker never accumulates into a switch.
    expect(updateAreaSwitch(state, "small", "base", most)).toBeNull();
  });
});

describe("walking", () => {
  it("clamps to the box with a wall margin", () => {
    const box = viewerBox(manifest.rooms[1]);
    expect(clampToBox([10, 1.6, 10], box, 0.25)).toEqual([2.75, 1.6, -0.25]);
    expect(clampToBox([1, 1.6, -1], box, 0.25)).toEqual([1, 1.6, -1]);
  });
  it("yaw and forward are inverses", () => {
    for (const yaw of [0, 0.5, -1.2, 3]) {
      expect(yawDelta(yawOf(forwardOf(yaw)), yaw)).toBeCloseTo(0, 6);
    }
    // engine forward +Z becomes viewer -Z, which is PlayCanvas's yaw 0.
    expect(yawOf(toViewer([0, 0, 1]))).toBeCloseTo(0, 6);
  });
  it("entry pose stands at eye height facing in", () => {
    const pose = entryPose(manifest.rooms[0]);
    expect(pose.position).toEqual([0, EYE_HEIGHT, 0]);
    expect(pose.yaw).toBeCloseTo(0, 6);
  });
});

describe("corridor", () => {
  // Engine (x, y, z) -> viewer (x, -y, -z): a camera at engine z = 1 stands at viewer z = -1.
  const walk = area("walk", [-10, -2.7, -10], [10, 0.3, 10], {
    cameras: [[0, -1.4, 0], [1, -1.4, 0], [2, -1.4, 0], [6, -1.4, 0]],
  });
  const corridor = corridorOf([walk]);

  it("leaves a visitor near the path where they are", () => {
    expect(constrainToCorridor([1.5, 1.6, -0.6], corridor, 1)).toEqual([1.5, 1.6, -0.6]);
  });

  it("pulls a step off the path back to the corridor's edge, keeping the height", () => {
    const p = constrainToCorridor([1, 1.6, 3], corridor, 1);
    expect(p[0]).toBeCloseTo(1, 6);
    expect(p[1]).toBe(1.6);
    expect(p[2]).toBeCloseTo(1, 6);
  });

  it("does not open a corridor across a cut in the walk", () => {
    // 2 -> 6 m is longer than a stride: the middle of that gap is not walkable.
    const near = nearestOnCorridor(corridor, 4, 0);
    expect(near.d).toBeCloseTo(2, 6);
    expect(constrainToCorridor([4, 1.6, 0], corridor, 1)[0]).toBeCloseTo(3, 6);
  });

  it("constrains nothing when there is no path", () => {
    expect(constrainToCorridor([40, 1.6, 40], corridorOf([area("none", [0, 0, 0], [1, 1, 1], { cameras: [] })]))).toEqual([40, 1.6, 40]);
  });

  it("has a default radius of about a step", () => {
    expect(PATH_RADIUS_M).toBeGreaterThanOrEqual(0.75);
    expect(PATH_RADIUS_M).toBeLessThanOrEqual(1.5);
  });
});

describe("filmed look directions", () => {
  it("reads the optional look from a camera entry and ignores plain entries", () => {
    expect(lookOf([0, -1.4, 0])).toBeNull();
    // engine +Z (forward) is viewer -Z: yaw 0.
    const v = lookOf([0, -1.4, 0, 0, 0, 1])!;
    expect(v[0]).toBeCloseTo(0, 9);
    expect(v[1]).toBeCloseTo(0, 9);
    expect(v[2]).toBeCloseTo(-1, 9);
  });

  it("blends the two nearest cameras' directions along the path", () => {
    // engine +Z then engine +X: yaw 0 then yaw -90 deg (viewer +X).
    const a = area("looks", [-5, -2.7, -5], [5, 0.3, 5], {
      cameras: [[0, -1.4, 0, 0, 0, 1], [1, -1.4, 0, 1, 0, 0]],
    });
    expect(lookAlong(a, 0)!.yaw).toBeCloseTo(0, 6);
    expect(lookAlong(a, 1)!.yaw).toBeCloseTo(-Math.PI / 2, 6);
    expect(lookAlong(a, 0.5)!.yaw).toBeCloseTo(-Math.PI / 4, 6);
  });

  it("pitches up when the phone pointed up (engine -Y is up)", () => {
    const s = Math.SQRT1_2;
    const a = area("up", [-5, -2.7, -5], [5, 0.3, 5], { cameras: [[0, -1.4, 0, 0, -s, s]] });
    expect(lookAlong(a, 0)!.pitch).toBeCloseTo(Math.PI / 4, 6);
  });

  it("has nothing to say about a manifest without looks", () => {
    expect(lookAlong(manifest.base, 0.3)).toBeNull();
  });
});

describe("opening field of view", () => {
  it("caps the width on a landscape screen and leaves a portrait one alone", () => {
    const wide = defaultFov(16 / 9);
    const hfov = (2 * Math.atan(Math.tan((wide * Math.PI) / 360) * (16 / 9)) * 180) / Math.PI;
    expect(hfov).toBeCloseTo(HFOV_OPEN_MAX, 6);
    expect(defaultFov(9 / 19.5)).toBe(70);
    expect(defaultFov(0)).toBe(70);
  });
});

describe("tour order", () => {
  const cam = (t: number) => [0, -1.4, 0, 0, 0, 1, t];
  const parts: WalkManifest = {
    ...manifest,
    base: area("base", [-10, -2.7, -10], [10, 0.3, 10], { cameras: [cam(11.2), cam(50.7)] }),
    rooms: [
      area("base~2", [-1, -2.7, -1], [1, 0.3, 1], { cameras: [cam(0), cam(10.7)], placed: false }),
      area("base~3", [-1, -2.7, -1], [1, 0.3, 1], { cameras: [cam(51.2), cam(59.8)] }),
      area("base~4", [-1, -2.7, -1], [1, 0.3, 1], { cameras: [cam(60.5)], redundant: true }),
      area("kitchen", [-1, -2.7, -1], [1, 0.3, 1], { cameras: [[0, -1.4, 0]] }),
      area("attic", [-1, -2.7, -1], [1, 0.3, 1], { cameras: [[0, -1.4, 0]], placed: false }),
    ],
  };

  it("replays a walk's parts in filming order, placed or not, and skips what the base already shows", () => {
    expect(tourOrder(parts)).toEqual(["base~2", "base", "base~3", "kitchen"]);
  });

  it("reads the filming time only when the manifest carries it", () => {
    expect(timeOf([0, 0, 0])).toBeNull();
    expect(timeOf(cam(3.5))).toBe(3.5);
  });

  it("keeps the manifest order for parts without times", () => {
    expect(tourOrder(manifest)).toEqual(["base", "big", "small"]);
  });
});

describe("spline", () => {
  it("passes through the knots and measures length", () => {
    const pts: [number, number, number][] = [[0, 1.6, 0], [1, 1.6, 0], [2, 1.6, 1]];
    expect(splinePoint(pts, 0)).toEqual([0, 1.6, 0]);
    expect(splinePoint(pts, 1)[0]).toBeCloseTo(2, 6);
    expect(pathLength(pts)).toBeCloseTo(1 + Math.SQRT2, 6);
  });
});

describe("plan mapping", () => {
  const plan: WalkFloorPlan = { url: null, width: 400, height: 300, x0: -2, z0: 4, px_per_m: 50, hotspots: [] };
  it("maps a world point to the plan and back", () => {
    const p = toViewer([1.5, 0, 2.0]);
    const f = planFraction(plan, p);
    expect(f.x).toBeCloseTo(((1.5 + 2) * 50) / 400, 6);
    expect(f.y).toBeCloseTo(((4 - 2.0) * 50) / 300, 6);
    const back = planPoint(plan, f.x, f.y);
    expect(back[0]).toBeCloseTo(p[0], 6);
    expect(back[2]).toBeCloseTo(p[2], 6);
    expect(back[1]).toBe(EYE_HEIGHT);
  });
});

describe("render scale", () => {
  it("gives up pixels when frames are slow and takes them back when they are fast", () => {
    // 30 fps: step down, and keep stepping until the floor.
    let s = 1.5;
    for (let i = 0; i < 20; i += 1) s = nextRenderScale(s, 33, 1.5);
    expect(s).toBeCloseTo(SCALE_MIN, 5);
    // 120 fps: climb back, but never past the ceiling.
    for (let i = 0; i < 40; i += 1) s = nextRenderScale(s, 8, 1.5);
    expect(s).toBeCloseTo(1.5, 5);
  });

  it("holds still inside the band, so the resolution does not flicker", () => {
    expect(nextRenderScale(1.0, 17, 1.5)).toBe(1.0);
    expect(nextRenderScale(1.0, 21.9, 1.5)).toBe(1.0);
    expect(nextRenderScale(1.0, 15.1, 1.5)).toBe(1.0);
  });

  it("falls faster than it climbs", () => {
    const down = 1.0 - nextRenderScale(1.0, 40, 1.5);
    const up = nextRenderScale(1.0, 5, 1.5) - 1.0;
    expect(down).toBeGreaterThan(up);
  });

  it("never exceeds a display that asks for less than the ceiling", () => {
    expect(nextRenderScale(1.0, 5, 1)).toBe(1);
    expect(nextRenderScale(1.4, 17, 1)).toBe(1);
  });
});
