"use client";

/**
 * The rings on the floor. Click one, stand there.
 *
 * ── WHY THIS IS 3D AND EVERY OTHER OVERLAY IS HTML ────────────────────────────────────
 * `HotspotLayer` draws tags as DOM nodes and gives good reasons: text stays crisp, it can be
 * focused and read aloud, and a label is the same size wherever it is. Every one of those
 * reasons fails for this control, because a portal is not a label — it is a PLACE.
 *
 * A place has to lie on the floor and obey perspective: an ellipse near the camera, a thin
 * sliver by the far wall, growing as you approach. Drawn as a flat circle in screen space it
 * reads as a button floating in front of the room, and a viewer has to be TOLD they can walk
 * there. Drawn on the floor plane, the picture says it. That is the whole of what makes a property
 * feel walked rather than operated, and it is worth a mesh.
 *
 * ── THE GEOMETRY IS THE ENGINE'S OWN, AND LIVES NEXT DOOR ─────────────────────────────
 * A ring sits where the hotspot's ray meets the floor — the same intersection `ai/measure.py`
 * computes for single-view metrology, in the player's own frame. That arithmetic is in
 * `portalGeometry.ts` with its tests, because a ring in the wrong place can only be caught by
 * standing in the room, while the rule that puts it there can be caught at a desk.
 *
 * ── CLICKS COME FROM THE TAP HANDLER, NOT FROM A RAYCAST ──────────────────────────────
 * r3f would happily raycast these meshes on pointerup, and it would also fire on the pointerup
 * that ends a look-drag — the gesture a viewer spends most of their time making. The look
 * controls already tell a tap from a drag, with thresholds tuned against real phones, so the
 * activation path reuses that decision and matches the tap's direction against each portal's
 * own. `hitTest` is that match, which the player runs before anything else claims the tap.
 *
 * Hover DOES come from r3f, because a hover cannot be confused with a drag.
 */

import { useFrame } from "@react-three/fiber";
import { useMemo, useRef, useState, type RefObject } from "react";
import * as THREE from "three";
import type { Hotspot } from "@/lib/types";
import { RING_RADIUS, floorPoint } from "./portalGeometry";
import type { Transit } from "./PortalTransit";

// ---------------------------------------------------------------------------------
// The ring itself
// ---------------------------------------------------------------------------------

const VERTEX_SHADER = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

/**
 * A soft annulus on a quad rather than a `RingGeometry`.
 *
 * Ring geometry gives hard tessellated edges that alias badly when the ring is nearly
 * edge-on — which is most of the time, because it is lying on the floor. Drawing the shape in
 * the fragment shader instead makes both edges analytic, so the far side of the ellipse stays
 * smooth at any angle, and it costs two triangles.
 */
const FRAGMENT_SHADER = /* glsl */ `
uniform vec3 uColour;
uniform float uOpacity;
uniform float uHover;
varying vec2 vUv;

void main() {
  vec2 p = vUv * 2.0 - 1.0;
  float r = length(p);
  if (r > 1.0) discard;

  // Antialiasing width in UV units, from the screen-space derivative: on a ring seen almost
  // edge-on one pixel spans a large slice of the quad, and a fixed width would either shimmer
  // or smear. fwidth measures the actual footprint.
  float aa = fwidth(r) * 1.5 + 0.002;

  // The band, and a hover state that thickens it inward rather than growing the ring — a ring
  // that grows on hover reads as moving, and this one is supposed to be a fixed place.
  float outer = 0.86;
  float inner = mix(0.62, 0.50, uHover);
  float band = smoothstep(outer, outer - aa, r) * smoothstep(inner - aa, inner, r);

  // A faint disc inside it, so the ring reads as a surface to stand on rather than as a
  // hoop, and brightens under the pointer.
  float fill = smoothstep(inner, inner - aa, r) * mix(0.10, 0.22, uHover);

  float alpha = (band * mix(0.72, 1.0, uHover) + fill) * uOpacity;
  if (alpha < 0.004) discard;
  gl_FragColor = vec4(uColour, alpha);
}
`;

interface RingProps {
  portal: Hotspot;
  position: THREE.Vector3;
  hovered: boolean;
  /** The target room has no panorama yet. Drawn, but visibly waiting. */
  pending: boolean;
  /** A walk is under way; every ring gets out of the way. See `travelling` in the driver. */
  travelling: RefObject<Transit | null>;
  onHover: (id: string | null) => void;
}

/**
 * The colour a ring whose room has not been shot yet.
 *
 * DRAWN RATHER THAN HIDDEN, which is the product decision in this file worth arguing. The
 * workflow the rings exist for is: walk the flat, drop a ring at every doorway, come back and
 * shoot the rooms. Hiding the ones not yet shot would make that workflow invisible — the agent
 * would place a ring, see nothing, and conclude it had not worked. Amber says "placed, waiting"
 * in the same language the rest of the product uses for it.
 */
export const PENDING_COLOUR = "#e8b45c";

/**
 * And the colour of one that leads somewhere you can actually stand.
 *
 * Exported beside its amber twin because the plan view draws the same two states — a room
 * that has been shot, and a doorway to one that has not — and one vocabulary said twice in
 * two files is one vocabulary until somebody changes a hex code in the file they happened to
 * be in.
 */
export const READY_COLOUR = "#7fe8c8";

function Ring({ portal, position, hovered, pending, travelling, onHover }: RingProps) {
  const material = useRef<THREE.ShaderMaterial>(null);

  const uniforms = useMemo(
    () => ({
      uColour: { value: new THREE.Color(pending ? PENDING_COLOUR : portal.accent || READY_COLOUR) },
      uOpacity: { value: 0.0 },
      uHover: { value: 0.0 },
    }),
    [portal.accent, pending],
  );

  useFrame((_, delta) => {
    const uniform = material.current?.uniforms;
    if (!uniform) return;
    // Eased towards the target rather than set: the hover has to feel like a light coming up,
    // and a step change on a shape this thin reads as a flicker.
    const rate = Math.min(1, delta * 12);
    uniform.uHover.value += (Number(hovered) - uniform.uHover.value) * rate;
    // Fades in on arrival, so a room change does not pop six rings into existence at once.
    // A pending ring settles dimmer: present, but not offering to take you anywhere.
    //
    // AND GETS OUT OF THE WAY THE MOMENT A WALK STARTS. The ring is placed in metres — 3.1 m
    // from the eye for a doorway — while the walk is a fraction of the panorama sphere, which
    // has no metric scale at all. The two frames cannot be reconciled (a 3 m dolly moves a
    // 100-unit sphere by 3%, which is invisible), so the camera passes the ring about an
    // eighth of the way through the journey. Left alone it balloons over the whole frame and
    // vanishes, which reads as a glitch rather than as motion. Faded out fast, it reads as
    // stepping onto the spot and moving on.
    const leaving = travelling.current !== null;
    const target = leaving ? 0 : pending ? 0.55 : 1;
    uniform.uOpacity.value +=
      (target - uniform.uOpacity.value) * Math.min(1, delta * (leaving ? 22 : 4));
  });

  return (
    <mesh
      position={position}
      // Flat on the floor. `RingGeometry`/`PlaneGeometry` stand up in XY by default, so this
      // is the one rotation that makes it a floor marking rather than a poster.
      rotation={[-Math.PI / 2, 0, 0]}
      renderOrder={1}
      onPointerOver={(event) => {
        event.stopPropagation();
        onHover(portal.id);
      }}
      onPointerOut={() => onHover(null)}
    >
      <planeGeometry args={[RING_RADIUS * 2, RING_RADIUS * 2]} />
      <shaderMaterial
        ref={material}
        uniforms={uniforms}
        vertexShader={VERTEX_SHADER}
        fragmentShader={FRAGMENT_SHADER}
        transparent
        // DEPTH WRITING OFF, TESTING ON. The rings sit metres from the eye and the panorama
        // sphere a hundred, so the test alone puts them in front of the room. Writing depth
        // as well would make two overlapping rings punch holes in each other's translucency.
        depthWrite={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

export interface PortalRingsProps {
  portals: Hotspot[];
  /** Scene ids that actually have a panorama. Anything else is drawn as pending. */
  readyScenes: ReadonlySet<string>;
  /** The walk in progress, if any. Read per frame; never a re-render. */
  travelling: RefObject<Transit | null>;
  /** Reported so the container can show the target room's name and a pointer cursor. */
  onHoverChange?: (portal: Hotspot | null) => void;
}

export function PortalRings({ portals, readyScenes, travelling, onHoverChange }: PortalRingsProps) {
  const [hovered, setHovered] = useState<string | null>(null);

  const placed = useMemo(
    () =>
      portals
        .map((portal) => ({ portal, position: floorPoint(portal.yaw, portal.pitch) }))
        .filter((entry): entry is { portal: Hotspot; position: THREE.Vector3 } => entry.position !== null),
    [portals],
  );

  const report = (id: string | null) => {
    setHovered(id);
    onHoverChange?.(id ? (placed.find((e) => e.portal.id === id)?.portal ?? null) : null);
  };

  return (
    <group>
      {placed.map(({ portal, position }) => (
        <Ring
          key={portal.id}
          portal={portal}
          position={position}
          hovered={hovered === portal.id}
          pending={!portal.target_scene_id || !readyScenes.has(portal.target_scene_id)}
          travelling={travelling}
          onHover={report}
        />
      ))}
    </group>
  );
}
