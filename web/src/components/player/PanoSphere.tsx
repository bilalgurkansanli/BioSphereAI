"use client";

/**
 * The panorama itself: one inside-out sphere, one unlit shader, two textures.
 *
 * The second texture is the whole point. Swapping a 2K panorama for a 4K one in a single
 * frame is a visible pop right in the middle of the thing the user is looking at, so both
 * rungs are bound at once and mixed by a uniform over ~450 ms. The user sees the picture
 * sharpen; they never see it change.
 */

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { SPHERE_RADIUS, TIMING } from "./types";

const VERTEX_SHADER = /* glsl */ `
varying vec2 vUv;

void main() {
  // SphereGeometry lays its UVs out for an observer on the OUTSIDE. Seen from the centre
  // that is a mirror image — text reads backwards. Flipping u here fixes it without
  // touching the mesh; a negative scale would do the same but invert the winding order,
  // which silently breaks BackSide culling and every raycast against the sphere.
  vUv = vec2(1.0 - uv.x, uv.y);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAGMENT_SHADER = /* glsl */ `
uniform sampler2D uTexA;
uniform sampler2D uTexB;
uniform float uMix;

varying vec2 vUv;

void main() {
  // Both rungs are sampled with the same UV, so the mix is a pure resolution cross-fade
  // with no geometric drift between the two images.
  vec3 a = texture2D(uTexA, vUv).rgb;
  vec3 b = texture2D(uTexB, vUv).rgb;
  gl_FragColor = vec4(mix(a, b, uMix), 1.0);

  // DO NOT DELETE THIS INCLUDE. It is not boilerplate — it is half of a round trip.
  //
  // Setting texture.colorSpace = SRGBColorSpace makes three upload the image with the
  // GL_SRGB8_ALPHA8 internal format (WebGLTextures.js), so the sampler hands this shader
  // LINEAR values, already decoded in hardware. The matching linear -> display encode is
  // not automatic for a hand-written ShaderMaterial: three only injects the
  // linearToOutputTexel() function into the fragment prefix (WebGLProgram.js) and leaves
  // it to the material's own source to call it. This chunk is that call.
  //
  // Remove it and nothing errors, nothing warns, and the panorama simply renders washed
  // out — a bug that ships, because "the photos look a bit flat" never gets filed.
  //
  // No tone mapping chunk here, deliberately: this is a photograph the camera already
  // graded, not a render that needs an exposure curve.
  #include <colorspace_fragment>
}
`;

export interface PanoSphereProps {
  /** Highest rung loaded so far. Each new identity cross-fades in over the previous. */
  texture: THREE.Texture | null;
  /**
   * Identity of the room. When this changes the next texture replaces the current one
   * outright — cross-fading between two different rooms would ghost one over the other.
   */
  cutKey: string;
}

export function PanoSphere({ texture, cutKey }: PanoSphereProps) {
  const geometry = useMemo(
    // 128x96 segments keeps the equirect distortion smooth at the poles without the
    // vertex count mattering; this is one object drawn once per frame.
    () => new THREE.SphereGeometry(SPHERE_RADIUS, 128, 96),
    [],
  );

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uTexA: { value: null as THREE.Texture | null },
          uTexB: { value: null as THREE.Texture | null },
          uMix: { value: 0 },
        },
        vertexShader: VERTEX_SHADER,
        fragmentShader: FRAGMENT_SHADER,
        side: THREE.BackSide,
        // Nothing else is ever drawn inside the sphere, so depth writes are wasted work.
        depthWrite: false,
        depthTest: false,
        fog: false,
      }),
    [],
  );

  useEffect(() => {
    return () => {
      geometry.dispose();
      material.dispose();
    };
  }, [geometry, material]);

  const fade = useRef({ active: false, elapsed: 0 });
  const lastCutKey = useRef(cutKey);

  useEffect(() => {
    if (!texture) return;
    const uniforms = material.uniforms;
    const roomChanged = lastCutKey.current !== cutKey;
    lastCutKey.current = cutKey;

    if (!uniforms.uTexA.value || roomChanged) {
      // First paint, or a different room: hard cut. Player dims the frame to black over
      // the swap, so there is nothing to soften.
      uniforms.uTexA.value = texture;
      uniforms.uTexB.value = null;
      uniforms.uMix.value = 0;
      fade.current = { active: false, elapsed: 0 };
      return;
    }

    if (uniforms.uTexA.value === texture) return;

    uniforms.uTexB.value = texture;
    uniforms.uMix.value = 0;
    fade.current = { active: true, elapsed: 0 };
  }, [texture, cutKey, material]);

  useFrame((_, delta) => {
    if (!fade.current.active) return;
    fade.current.elapsed += Math.min(delta, 1 / 20);
    const t = Math.min(fade.current.elapsed / (TIMING.rungFadeMs / 1000), 1);
    // Smoothstep rather than linear: a linear cross-fade reads as a wipe at the midpoint.
    material.uniforms.uMix.value = t * t * (3 - 2 * t);

    if (t >= 1) {
      material.uniforms.uTexA.value = material.uniforms.uTexB.value;
      material.uniforms.uTexB.value = null;
      material.uniforms.uMix.value = 0;
      fade.current = { active: false, elapsed: 0 };
    }
  });

  return <mesh geometry={geometry} material={material} frustumCulled={false} renderOrder={-1} />;
}
