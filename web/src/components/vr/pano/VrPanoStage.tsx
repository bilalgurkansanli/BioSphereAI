"use client";

/**
 * The 360 player's headset mode, inside the player's own `<Canvas>`. Lazy-loaded: none of this
 * is downloaded until a VR button exists to press (see `usePanoVr`).
 *
 * Two drivers step the same `PanoVrWorld`:
 *
 * - `XrDriver` — WebXR. three's `WebXRManager` owns the camera (both eyes, from the headset's
 *   pose) and r3f has switched its loop to the session's animation frame, so this driver only
 *   reads the pose and the controllers and leaves the render to r3f. Its `useFrame` therefore
 *   runs at priority 0: any priority above zero tells r3f that we render ourselves, and in XR
 *   we do not.
 * - `CardboardDriver` — the phone headset. Priority 1, because here we DO render: two eyes into
 *   targets and the distortion composite to the screen (`CardboardRenderer`).
 *
 * All per-frame state lives in the world and the renderer, which are plain classes; this file
 * only ties their lifetimes to React's.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import type { Hotspot } from "@/lib/types";
import type * as THREE from "three";
import type { LookControlsApi } from "@/components/player/types";
import type { LensProfile } from "../lens";
import type { OrientationFeed } from "../useCardboard";
import { CardboardRenderer } from "./cardboardRenderer";
import { PanoVrWorld, type PanoVrActions, type PanoVrCopy } from "./panoVrWorld";

export interface VrPanoStageProps {
  mode: "xr" | "cardboard";
  /** The player's panorama group (sphere and rings), moved and turned with the headset. */
  content: React.RefObject<THREE.Group | null>;
  controls: React.RefObject<LookControlsApi | null>;
  /** Identity of the room on screen; a change is an arrival. */
  sceneKey: string;
  room: { name: string; index: number; count: number };
  portals: Hotspot[];
  readyScenes: ReadonlySet<string>;
  sceneNames: ReadonlyMap<string, string>;
  tags: Hotspot[];
  /** The player's doorway veil, mirrored in the headset where the DOM one cannot be seen. */
  dim: number;
  dimDurationMs: number;
  copy: PanoVrCopy;
  actions: PanoVrActions;
  cardboard: {
    feed: React.RefObject<OrientationFeed>;
    lens: LensProfile;
    debug: boolean;
    selectRef: React.RefObject<(() => boolean) | null>;
  } | null;
}

export default function VrPanoStage(props: VrPanoStageProps) {
  const { mode, content, controls, sceneKey, room, portals, readyScenes, sceneNames, tags, dim, dimDurationMs, copy } = props;

  // The player's handlers change identity with its renders; the world reads them at call time.
  const actions = useRef(props.actions);
  useEffect(() => {
    actions.current = props.actions;
  });

  const [world] = useState(() => new PanoVrWorld({ copy, actions, controls }));

  useEffect(() => world.setCopy(copy), [world, copy]);
  useEffect(() => world.setRoom(room.name, room.index, room.count), [world, room.name, room.index, room.count]);
  useEffect(() => world.setRings(portals, readyScenes, sceneNames), [world, portals, readyScenes, sceneNames]);
  useEffect(() => world.setTags(tags, sceneNames), [world, tags, sceneNames]);
  // In Cardboard the player's own DOM veil covers the canvas; drawing a second one in the
  // scene would square the fade.
  useEffect(() => world.setDim(mode === "xr" ? dim : 0, dimDurationMs), [world, mode, dim, dimDurationMs]);

  // A new room is on screen (under the black): turn it so its arrival view is ahead. Also runs
  // on mount, where it is the first alignment.
  useEffect(() => world.arrive(), [world, sceneKey]);

  useEffect(() => {
    // The group outlives this component (it is the player's), so it is read at unmount.
    const group = content;
    return () => {
      world.restoreView(group.current);
      world.dispose();
    };
  }, [world, content]);

  return (
    <>
      <primitive object={world.content} />
      <primitive object={world.root} />
      {mode === "xr" ? (
        <XrDriver world={world} content={content} />
      ) : props.cardboard ? (
        <CardboardDriver world={world} content={content} {...props.cardboard} />
      ) : null}
    </>
  );
}

interface DriverProps {
  world: PanoVrWorld;
  content: React.RefObject<THREE.Group | null>;
}

function XrDriver({ world, content }: DriverProps) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  useEffect(() => world.attachControllers(gl, scene), [world, gl, scene]);
  useFrame((_, delta) => world.frameXr(gl, delta, content.current));
  return null;
}

interface CardboardDriverProps extends DriverProps {
  feed: React.RefObject<OrientationFeed>;
  lens: LensProfile;
  debug: boolean;
  selectRef: React.RefObject<(() => boolean) | null>;
}

function CardboardDriver({ world, content, feed, lens, debug, selectRef }: CardboardDriverProps) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const size = useThree((s) => s.size);
  const renderer = useMemo(() => new CardboardRenderer(gl), [gl]);

  useEffect(() => () => renderer.dispose(), [renderer]);
  useEffect(() => {
    renderer.configure(lens, size.width, size.height, gl.getPixelRatio(), debug);
  }, [renderer, lens, size.width, size.height, debug, gl]);

  // The Cardboard's button is a screen tap: it selects what the gaze is on.
  useEffect(() => {
    selectRef.current = () => world.selectGaze();
    return () => {
      selectRef.current = null;
    };
  }, [selectRef, world]);

  useFrame((_, delta) => {
    const f = feed.current;
    if (!f.valid || !renderer.configured) {
      // Black until the first sensor reading: a frame in the wrong orientation is worse.
      renderer.blank();
      return;
    }
    const head = world.frameCardboard(f.q, delta, content.current);
    renderer.render(scene, head.position, head.quaternion, 0);
    renderer.track(delta, "cardboard-360");
  }, 1);

  return null;
}
