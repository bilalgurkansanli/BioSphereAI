"use client";

/**
 * Walking through the doorway.
 *
 * A ring is a PLACE, and the click that takes you there has to be a journey or the place was
 * a lie. The player already had a doorway transition — dim, pinch the lens, swap, bloom — and
 * it is the right thing for a room chosen off a list, where nothing in the picture says where
 * you are going. It is the wrong thing for a ring: you aimed at a spot on the floor, so the
 * camera should set off towards THAT spot, and the room should grow around you on the way.
 *
 * ── HOW A FIXED CAMERA WALKS ──────────────────────────────────────────────────────────
 * The panorama is a sphere of radius `SPHERE_RADIUS` with the camera at its centre, so there
 * is nowhere to walk to — every pixel is equally far away. But the sphere is FINITE, and that
 * is enough: move the camera off-centre towards the doorway and the wall ahead comes closer and
 * grows, while the walls beside you sweep past the edges of the frame. That sweep is the whole
 * effect. It is the same trick a dolly shot uses, and it works here for the same reason: the eye
 * reads changing perspective as its own motion.
 *
 * The thing to tune is not how far it travels but how much of that is SEEN — see `PORTAL_DOLLY`
 * and `TIMING.portalVeilAt`, which are set together and are meaningless apart. The first version
 * showed 418 ms of travel and 1.5x of growth, and read as a snap zoom into a cut. It now shows
 * 828 ms and 2.4x.
 *
 * It is not geometrically true — a real step forward would reveal what is behind the door
 * frame, and this cannot, because the photograph does not contain it. It does not have to be
 * true, only brief: by the time the distortion could be noticed the veil is black.
 *
 * ── THE LENS DOES NOT MOVE ────────────────────────────────────────────────────────────
 * The first version widened the field of view across the walk, on the game-engine instinct that
 * a wider lens sells speed. That instinct is wrong with a bounded dolly, and measurably so:
 * widening 75 deg to 82 deg multiplies on-screen scale by 0.88, so seven degrees of "rush" hand
 * back an eighth of the approach the dolly had just bought. That is the dolly-zoom, whose entire
 * purpose is to hold a subject the same size while the space around it changes — the exact
 * opposite of what a doorway you are walking towards should do.
 *
 * Narrowing instead would compound the magnification, and costs the other half of the effect:
 * the peripheral sweep, the walls sliding out past the edges of frame, which is what the eye
 * actually reads as self-motion. So the lens is left alone, which is also what walking is.
 *
 * ── STRICTLY HORIZONTAL ───────────────────────────────────────────────────────────────
 * The ring is on the floor, so the direction you clicked points DOWN. Travelling along it
 * would drive the camera into the tiles and read as falling over. What the viewer means by
 * "go there" is the bearing of the ring, not its depression, so only the yaw survives.
 *
 * ── ONE WRITER FOR THE CAMERA POSITION ────────────────────────────────────────────────
 * `useLookControls` owns the camera's rotation and never touches its position; this owns the
 * position and never touches the rotation. When no journey is running it parks the camera back
 * at the centre, so a room swapped in mid-flight can never inherit a displaced viewpoint —
 * which would show as a panorama that is subtly, unfixably stretched.
 */

import { useFrame, useThree } from "@react-three/fiber";
import { useRef, type RefObject } from "react";
import * as THREE from "three";
import { PORTAL_DOLLY, SPHERE_RADIUS } from "./types";

export interface Transit {
  /** Bearing of the doorway: horizontal, unit length. */
  readonly dir: THREE.Vector3;
  readonly startedAt: number;
  readonly durationMs: number;
  /** How far to travel, in world units. */
  readonly distance: number;
  /** Run once, on the frame the journey ends. */
  readonly onArrive: () => void;
}

/** The bearing of a portal, as a direction to travel. Depression discarded; see the header. */
export function travelDirection(yaw: number): THREE.Vector3 {
  return new THREE.Vector3(Math.cos(yaw), 0, Math.sin(yaw));
}

export function beginTransit(yaw: number, durationMs: number, onArrive: () => void): Transit {
  return {
    dir: travelDirection(yaw),
    startedAt: performance.now(),
    durationMs,
    distance: SPHERE_RADIUS * PORTAL_DOLLY,
    onArrive,
  };
}

/**
 * Smoothstep, and the veil is timed around it rather than the other way about.
 *
 * The temptation is a cubic ease-in, because accelerating is what setting off feels like. It
 * is wrong here: by half way an ease-in has covered an eighth of the distance, and half way is
 * most of what the viewer actually sees, so the journey would look like a long hesitation
 * followed by a cut to black. Smoothstep has covered 57% of the distance at 55% of the time,
 * which is where the veil starts, so the growth of the doorway is over before the picture goes.
 * The deceleration at the far end is spent behind black and costs nothing.
 */
function eased(p: number): number {
  return p * p * (3 - 2 * p);
}

export interface PortalTransitProps {
  transit: RefObject<Transit | null>;
}

export function PortalTransit({ transit }: PortalTransitProps) {
  const camera = useThree((s) => s.camera);
  /**
   * The journey whose arrival has already been announced.
   *
   * Kept HERE rather than as a flag on the journey itself, which was the first shape and is a
   * worse one: `Transit` is a description of a trip and the caller owns it, so writing to it
   * from the driver makes two owners for one object. Holding the identity locally also does the
   * job it has to do — `onArrive` swaps the room, so firing it twice would swap twice — while
   * leaving `Transit` fully readonly.
   */
  const announced = useRef<Transit | null>(null);

  useFrame(() => {
    const journey = transit.current;
    if (!journey || journey === announced.current) {
      if (camera.position.lengthSq() > 0) camera.position.set(0, 0, 0);
      return;
    }

    const p = Math.min(1, (performance.now() - journey.startedAt) / journey.durationMs);
    const travelled = eased(p);
    camera.position.copy(journey.dir).multiplyScalar(travelled * journey.distance);

    if (p >= 1) {
      announced.current = journey;
      journey.onArrive();
    }
  });

  return null;
}
