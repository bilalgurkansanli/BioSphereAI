"use client";

/**
 * The hero is the player, not a picture of it.
 *
 * A page about a 360 tour that shows a still photograph is asking to be believed. This one
 * hands the visitor the thing itself: a sphere they can drag. The room in it is SYNTHETIC — the
 * procedural room the engine's quality gate renders and scores every stitch against — so the
 * page shows the player without putting anybody's home on the internet.
 *
 * ── WHAT IT COSTS, AND WHAT PAYS FOR IT ───────────────────────────────────────────────
 * three.js is about 150 KB gzipped and has no business being in the critical path. So it is
 * not: the circle paints a real PERSPECTIVE still (`poster.webp`) as an ordinary `<img>` with
 * the markup, and the sphere is imported and built only after mount. The poster is rendered at
 * exactly the camera's opening angle (`engine/scripts/build_demo_room.py`), so the handover is
 * invisible — no jump, no second composition.
 *
 * ── WHAT IT DOES NOT DO ───────────────────────────────────────────────────────────────
 * It does not autoplay a tour, spin fast, or move under a pointer that is trying to read the
 * headline next to it. The drift is 1.2 degrees a second — under the threshold where motion
 * pulls the eye away from text — and it stops for good the moment somebody takes hold.
 * `prefers-reduced-motion` gets no drift at all, and a browser without WebGL keeps the poster
 * and loses nothing but the dragging.
 */

import { MousePointer2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type * as THREE_NS from "three";
import { cn } from "@/lib/utils";
import { ORB } from "./copy";
import { useText } from "./primitives";

/** Written by `engine/scripts/build_demo_room.py`. */
const ROOM_TEXTURE = "/demo/room.webp";
const POSTER = "/demo/poster.webp";

/** Must match `POSTER_YAW_DEG` / `POSTER_PITCH_DEG` / `POSTER_FOV_DEG` in the asset script. */
const START_YAW = (50 * Math.PI) / 180;
const START_PITCH = (-6 * Math.PI) / 180;
const FOV = 76;

const DRIFT_RAD_PER_S = (1.2 * Math.PI) / 180;
const DRAG_SENSITIVITY = 0.0026;
const PITCH_LIMIT = Math.PI / 2 - 0.25;

export function HeroPano({ className }: { className?: string }) {
  const { tx } = useText();
  const hostRef = useRef<HTMLDivElement>(null);
  const [live, setLive] = useState(false);
  const [touched, setTouched] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let dispose: (() => void) | undefined;

    void (async () => {
      const host = hostRef.current;
      if (!host) return;

      const THREE = await import("three");
      if (cancelled) return;

      // NO SEPARATE CAPABILITY PROBE. A throwaway canvas asked for a context leaks a live
      // WebGL context on every mount (dropping the element does not free it), and past the
      // browser's cap the real renderer's constructor throws. Constructing the renderer IS the
      // probe, and the failure is caught below.
      const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.domElement.style.cssText = "width:100%;height:100%;display:block;touch-action:none;cursor:grab";

      let texture: THREE_NS.Texture;
      try {
        texture = await new THREE.TextureLoader().loadAsync(ROOM_TEXTURE);
      } catch (error) {
        renderer.dispose();
        throw error;
      }
      if (cancelled) {
        texture.dispose();
        renderer.dispose();
        return;
      }
      texture.colorSpace = THREE.SRGBColorSpace;
      // The seam is a wrap in the source; clamping would draw a line down the one column the
      // whole stitching pipeline works hardest to hide.
      texture.wrapS = THREE.RepeatWrapping;

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 1100);
      const geometry = new THREE.SphereGeometry(500, 60, 40);
      geometry.scale(-1, 1, 1); // inside out: we stand at the centre
      const material = new THREE.MeshBasicMaterial({ map: texture });
      scene.add(new THREE.Mesh(geometry, material));
      host.appendChild(renderer.domElement);

      let yaw = START_YAW;
      let pitch = START_PITCH;
      let velocity = 0;
      let dragging = false;
      let steered = false;
      let lastX = 0;
      let lastY = 0;
      const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

      const resize = () => {
        const { width, height } = host.getBoundingClientRect();
        if (!width || !height) return;
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      };
      resize();
      const observer = new ResizeObserver(resize);
      observer.observe(host);

      const canvas = renderer.domElement;
      const onDown = (event: PointerEvent) => {
        dragging = true;
        steered = true;
        setTouched(true);
        velocity = 0;
        lastX = event.clientX;
        lastY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
        canvas.style.cursor = "grabbing";
      };
      const onMove = (event: PointerEvent) => {
        if (!dragging) return;
        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        lastX = event.clientX;
        lastY = event.clientY;
        yaw -= dx * DRAG_SENSITIVITY;
        pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, pitch + dy * DRAG_SENSITIVITY));
        velocity = -dx * DRAG_SENSITIVITY;
      };
      const onUp = (event: PointerEvent) => {
        dragging = false;
        try {
          canvas.releasePointerCapture(event.pointerId);
        } catch {
          /* The pointer was already gone; nothing to release. */
        }
        canvas.style.cursor = "grab";
      };
      canvas.addEventListener("pointerdown", onDown);
      canvas.addEventListener("pointermove", onMove);
      canvas.addEventListener("pointerup", onUp);
      canvas.addEventListener("pointercancel", onUp);

      let raf = 0;
      let previous = performance.now();
      const frame = (now: number) => {
        const step = Math.min((now - previous) / 1000, 0.05);
        previous = now;
        if (!dragging) {
          if (Math.abs(velocity) > 1e-5) {
            yaw += velocity;
            velocity *= 0.94; // a hand-thrown spin that settles, not a carousel
          } else if (!steered && !reduced) {
            yaw += DRIFT_RAD_PER_S * step;
          }
        }
        camera.rotation.set(0, 0, 0);
        camera.rotateY(yaw);
        camera.rotateX(pitch);
        renderer.render(scene, camera);
        raf = requestAnimationFrame(frame);
      };
      raf = requestAnimationFrame(frame);

      dispose = () => {
        cancelAnimationFrame(raf);
        observer.disconnect();
        canvas.removeEventListener("pointerdown", onDown);
        canvas.removeEventListener("pointermove", onMove);
        canvas.removeEventListener("pointerup", onUp);
        canvas.removeEventListener("pointercancel", onUp);
        texture.dispose();
        geometry.dispose();
        material.dispose();
        renderer.dispose();
        canvas.remove();
      };
      setLive(true);
    })().catch(() => {
      // No WebGL, no context left to give, a texture that would not decode. The circle keeps
      // the poster and the page keeps its meaning — only the dragging is lost.
      setLive(false);
    });

    return () => {
      cancelled = true;
      dispose?.();
    };
  }, []);

  return (
    <figure className={cn("relative mx-auto w-full max-w-[min(100%,520px)] lg:max-w-none", className)}>
      <div
        className="absolute inset-[6%] rounded-full bg-[radial-gradient(circle_at_50%_50%,var(--bio-dim),transparent_70%)] blur-2xl"
        aria-hidden
      />

      <div className="ring-glow relative aspect-square overflow-hidden rounded-full border border-line-strong bg-canvas-raised">
        {/* Painted with the HTML, replaced by the sphere at the same angle. `eager` because
            this is the page's largest paint and a lazy hero is a blank circle on a phone. */}
        {/* eslint-disable-next-line @next/next/no-img-element -- next/image would add a
            loader round-trip in front of an already-optimised 23 KB still. */}
        <img
          src={POSTER}
          alt={tx(ORB.alt)}
          width={1400}
          height={1400}
          loading="eager"
          fetchPriority="high"
          // Inline, not a utility class: the value is decided at runtime, and a class that
          // only appears in a ternary has to survive Tailwind's static scan to exist at all.
          style={{ opacity: live ? 0 : 1 }}
          className="absolute inset-0 size-full object-cover transition-opacity duration-700"
        />
        <div ref={hostRef} className="absolute inset-0" />

        {/* Inner shadow keeps the disc reading as a ball rather than a hole in the page. */}
        <div
          className="pointer-events-none absolute inset-0 rounded-full shadow-[inset_0_0_90px_18px_rgb(0_0_0/0.55)]"
          aria-hidden
        />

        <div className="pointer-events-none absolute inset-x-0 top-5 flex justify-center">
          <span className="flex items-center gap-1.5 rounded-full border border-white/10 bg-black/55 px-3 py-1 text-[11px] font-medium text-white/85 backdrop-blur">
            <span className="size-1.5 rounded-full bg-[#17e2c5]" aria-hidden />
            {tx(ORB.badge)}
          </span>
        </div>

        {/* The one instruction on the page, gone as soon as it has been obeyed. */}
        <div
          style={{ opacity: live && !touched ? 1 : 0 }}
          className="pointer-events-none absolute inset-x-0 bottom-7 flex justify-center transition-opacity duration-500"
        >
          <span className="animate-breathe flex items-center gap-1.5 rounded-full border border-white/10 bg-black/60 px-3 py-1.5 text-[12px] font-medium text-white backdrop-blur">
            <MousePointer2 className="size-3.5 text-[#17e2c5]" aria-hidden />
            {tx(ORB.drag)}
          </span>
        </div>
      </div>

      <figcaption className="mx-auto mt-5 max-w-sm text-center text-[12.5px] leading-relaxed text-ink-faint">
        {tx(ORB.caption)}
      </figcaption>
    </figure>
  );
}
