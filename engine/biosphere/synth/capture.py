"""Simulating a person filming a 360 sweep with a phone.

The whole point of this module is to be *unflattering*. A synthetic capture that a
stitcher finds easy proves nothing, so everything here models the specific ways a
hand-held phone sweep is hard:

**Parallax.** The optical centre is not the centre of the sphere. A person holding a
phone at arm's length rotates it about their spine, so the lens traces a circle of
roughly 0.25-0.35 m radius. Every pure-rotation stitcher in existence is wrong by exactly
that amount, and the error grows with proximity — which is why the near edge of a table
ghosts while the far wall lines up. ``arm_radius_m`` is the single knob that turns this
on, and setting it to 0 recovers the impossible ideal of a nodal capture.

**Hand jitter that looks like a hand.** White noise on the angles would be trivially
filtered and is nothing like a human. Real hand tremor is a low-frequency, smooth,
band-limited wobble, so the jitter here is a sum of a few sines at incommensurate
frequencies — smooth, never repeating over the length of a capture, and fully determined
by the seed.

**A speed profile.** People start slowly, speed up once they trust the motion, and stop
for a moment when they notice something. That produces frames whose overlap varies by a
factor of three and a couple of near-duplicate clusters, which is what stresses keyframe
selection.

**Auto-exposure that lags.** Metering is a first-order lag, so panning past a window
darkens the *next* half second of frames rather than that one. The per-frame gain is
recorded in the ground truth, so the harness can score geometry and photometry
separately.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from biosphere.pano import geometry as g
from biosphere.pano.types import CameraPose, Intrinsics

from .render import apply_camera_response, render_equirect, render_view
from .room import Room

__all__ = [
    "Jitter",
    "Trajectory",
    "CaptureEffects",
    "SyntheticCapture",
    "EFFECT_PRESETS",
    "simulate_capture",
]

_S_GRID: Final[int] = 4096
_T_GRID: Final[int] = 2049
_ROW_PITCH_DEG: Final[float] = 38.0


def _smoothstep(x: np.ndarray) -> np.ndarray:
    c = np.clip(x, 0.0, 1.0)
    return c * c * (3.0 - 2.0 * c)


def _piecewise(s: np.ndarray, nodes: Sequence[tuple[float, float]]) -> np.ndarray:
    xs = np.array([n[0] for n in nodes], dtype=np.float64)
    ys = np.array([n[1] for n in nodes], dtype=np.float64)
    return np.interp(s, xs, ys)


def _smooth(values: np.ndarray, sigma_frac: float) -> np.ndarray:
    """Gaussian-smooth a piecewise-linear programme so its corners become C-infinity.

    A real wrist cannot change angular velocity instantaneously; smoothing the corners is
    what turns the idealised "row, transition, row" script into a plausible motion.
    """
    n = values.shape[0]
    sigma = max(1.0, sigma_frac * n)
    radius = int(math.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    padded = np.concatenate([np.full(radius, values[0]), values, np.full(radius, values[-1])])
    return np.convolve(padded, k, mode="valid")


# ======================================================================================
# Hand tremor
# ======================================================================================


@dataclass(slots=True)
class Jitter:
    """Smooth, seeded, band-limited hand tremor on yaw / pitch / roll.

    Frequencies are deliberately mutually irrational multiples of a base rate so the sum
    never repeats within a capture; amplitudes fall off with frequency the way real
    physiological tremor does.
    """

    freqs: np.ndarray  # (3, K) Hz
    amps: np.ndarray  # (3, K) radians
    phases: np.ndarray  # (3, K) radians

    @classmethod
    def make(
        cls,
        seed: int,
        *,
        yaw_deg: float = 0.55,
        pitch_deg: float = 0.50,
        roll_deg: float = 1.15,
    ) -> Jitter:
        rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        ratios = np.array([1.0, 1.6180339887, 2.7182818285, 4.1132503788, 6.5309], dtype=np.float64)
        weights = np.array([1.0, 0.60, 0.34, 0.19, 0.11], dtype=np.float64)
        weights = weights / weights.sum()
        base = np.array([0.31, 0.37, 0.13], dtype=np.float64)  # roll drifts more slowly
        amp_deg = np.array([yaw_deg, pitch_deg, roll_deg], dtype=np.float64)
        freqs = base[:, None] * ratios[None, :] * (1.0 + 0.06 * rng.standard_normal((3, ratios.size)))
        amps = np.radians(amp_deg)[:, None] * weights[None, :]
        phases = rng.uniform(0.0, 2.0 * math.pi, size=(3, ratios.size))
        return cls(freqs=freqs, amps=amps, phases=phases)

    def __call__(self, t: float | np.ndarray) -> np.ndarray:
        tt = np.asarray(t, dtype=np.float64)
        arg = 2.0 * math.pi * self.freqs[..., None] * tt + self.phases[..., None]
        out = (self.amps[..., None] * np.sin(arg)).sum(axis=1)
        return out if tt.ndim else out[:, 0]

    def scaled(self, factor: float) -> Jitter:
        return Jitter(freqs=self.freqs.copy(), amps=self.amps * float(factor), phases=self.phases.copy())


# ======================================================================================
# Trajectory
# ======================================================================================


@dataclass(eq=False)
class Trajectory:
    """A continuous, seeded description of where the phone points and where it *is*.

    The angle programme is a function of *progress* ``s in [0, 1]``, and a separate speed
    profile maps wall-clock time onto progress. Splitting them that way means a pause or
    an acceleration changes the frame spacing without touching the shape of the sweep.
    """

    name: str
    duration_s: float
    fps: float
    jitter: Jitter
    s_grid: np.ndarray
    yaw_grid: np.ndarray
    pitch_grid: np.ndarray
    roll_grid: np.ndarray
    t_grid: np.ndarray
    progress_grid: np.ndarray
    arm_radius_m: float = 0.28
    sway_m: float = 0.013
    centre: np.ndarray | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.arm_radius_m <= 0.0:
            # An ideal nodal capture has to be *exactly* nodal or the self-consistency
            # test loses its meaning: residual body sway would look like a convention bug.
            self.sway_m = 0.0
        if self.centre is not None:
            self.centre = np.asarray(self.centre, dtype=np.float32).reshape(3)

    # -- construction -----------------------------------------------------------------

    @staticmethod
    def _speed_profile(duration_s: float, seed: int, n_pauses: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng((int(seed) & 0x7FFFFFFF) ^ 0x5EED)
        t = np.linspace(0.0, float(duration_s), _T_GRID)
        w = 0.40 + 0.60 * _smoothstep(t / max(0.22 * duration_s, 1e-6))
        w *= 0.50 + 0.50 * _smoothstep((duration_s - t) / max(0.13 * duration_s, 1e-6))
        for _ in range(max(int(n_pauses), 0)):
            tp = float(rng.uniform(0.22, 0.86)) * duration_s
            sw = max(0.04 * duration_s, 1e-3)
            w *= 1.0 - 0.82 * np.exp(-(((t - tp) / sw) ** 2))
        w = np.maximum(w, 0.02)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
        return t, cum / max(float(cum[-1]), 1e-12)

    @classmethod
    def _build(
        cls,
        name: str,
        duration_s: float,
        fps: float,
        yaw_nodes: Sequence[tuple[float, float]],
        pitch_nodes: Sequence[tuple[float, float]],
        *,
        seed: int,
        arm_radius_m: float,
        n_pauses: int,
        smooth_frac: float = 0.012,
        jitter_scale: float = 1.0,
        centre: np.ndarray | None = None,
    ) -> Trajectory:
        if duration_s <= 0.0 or fps <= 0.0:
            raise ValueError("duration_s and fps must be positive")
        s = np.linspace(0.0, 1.0, _S_GRID)
        yaw = _smooth(_piecewise(s, yaw_nodes), smooth_frac)
        pitch = _smooth(_piecewise(s, pitch_nodes), smooth_frac)
        roll = np.zeros_like(s)
        t_grid, progress = cls._speed_profile(duration_s, seed, n_pauses)
        jitter = Jitter.make(seed)
        if jitter_scale != 1.0:
            jitter = jitter.scaled(jitter_scale)
        return cls(
            name=name,
            duration_s=float(duration_s),
            fps=float(fps),
            jitter=jitter,
            s_grid=s,
            yaw_grid=yaw,
            pitch_grid=pitch,
            roll_grid=roll,
            t_grid=t_grid,
            progress_grid=progress,
            arm_radius_m=float(arm_radius_m),
            centre=centre,
            seed=int(seed),
        )

    @classmethod
    def single_row(
        cls,
        duration_s: float = 12.0,
        fps: float = 30.0,
        revolutions: float = 1.0,
        *,
        seed: int = 11,
        arm_radius_m: float = 0.28,
        centre: np.ndarray | None = None,
    ) -> Trajectory:
        """One level sweep. The commonest capture, and the one with no pole coverage."""
        turn = 2.0 * math.pi * float(revolutions)
        return cls._build(
            "single_row",
            duration_s,
            fps,
            [(0.0, 0.0), (1.0, turn)],
            [(0.0, 0.0), (1.0, 0.0)],
            seed=seed,
            arm_radius_m=arm_radius_m,
            n_pauses=1,
            centre=centre,
        )

    @classmethod
    def three_row(
        cls,
        duration_s: float = 24.0,
        fps: float = 30.0,
        *,
        seed: int = 11,
        arm_radius_m: float = 0.28,
        row_pitch_deg: float = _ROW_PITCH_DEG,
        centre: np.ndarray | None = None,
    ) -> Trajectory:
        """Level, then up a row, then down a row — the pattern good capture UIs coach."""
        p = math.radians(row_pitch_deg)
        tau = 2.0 * math.pi
        yaw_nodes = [
            (0.00, 0.0),
            (0.30, tau),
            (0.35, 1.14 * tau),
            (0.65, 2.14 * tau),
            (0.70, 2.28 * tau),
            (1.00, 3.28 * tau),
        ]
        pitch_nodes = [
            (0.00, 0.0),
            (0.30, 0.0),
            (0.35, p),
            (0.65, p),
            (0.70, -p),
            (1.00, -p),
        ]
        return cls._build(
            "three_row",
            duration_s,
            fps,
            yaw_nodes,
            pitch_nodes,
            seed=seed,
            arm_radius_m=arm_radius_m,
            n_pauses=2,
            centre=centre,
        )

    @classmethod
    def three_row_plus_poles(
        cls,
        duration_s: float = 32.0,
        fps: float = 30.0,
        *,
        seed: int = 11,
        arm_radius_m: float = 0.28,
        row_pitch_deg: float = _ROW_PITCH_DEG,
        centre: np.ndarray | None = None,
    ) -> Trajectory:
        """Three rows plus a slow dwell on the zenith and the nadir.

        The dwells rotate slowly rather than stopping dead: a stationary camera at the
        pole gives the solver no baseline at all, and no real person holds still there.
        """
        p = math.radians(row_pitch_deg)
        pole = math.radians(84.0)
        tau = 2.0 * math.pi
        yaw_nodes = [
            (0.00, 0.0),
            (0.26, tau),
            (0.30, 1.12 * tau),
            (0.55, 2.12 * tau),
            (0.59, 2.24 * tau),
            (0.84, 3.24 * tau),
            (0.88, 3.36 * tau),
            (0.94, 3.76 * tau),
            (0.97, 3.88 * tau),
            (1.00, 4.28 * tau),
        ]
        pitch_nodes = [
            (0.00, 0.0),
            (0.26, 0.0),
            (0.30, p),
            (0.55, p),
            (0.59, -p),
            (0.84, -p),
            (0.88, pole),
            (0.94, pole),
            (0.97, -pole),
            (1.00, -pole),
        ]
        return cls._build(
            "three_row_plus_poles",
            duration_s,
            fps,
            yaw_nodes,
            pitch_nodes,
            seed=seed,
            arm_radius_m=arm_radius_m,
            n_pauses=2,
            smooth_frac=0.010,
            centre=centre,
        )

    # -- variants ---------------------------------------------------------------------

    def with_arm_radius(self, arm_radius_m: float) -> Trajectory:
        """A copy orbiting at a different radius; ``0.0`` gives the ideal nodal case."""
        return replace(self, arm_radius_m=float(arm_radius_m), sway_m=self.sway_m)

    def centred_at(self, centre: np.ndarray) -> Trajectory:
        """A copy whose body axis passes through ``centre`` (normally the room's eye)."""
        return replace(self, centre=np.asarray(centre, dtype=np.float32).reshape(3))

    # -- evaluation -------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return max(int(round(self.duration_s * self.fps)), 1)

    def frame_times(self) -> np.ndarray:
        return np.arange(self.n_frames, dtype=np.float64) / float(self.fps)

    def progress_at(self, t: float | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(t, dtype=np.float64), self.t_grid, self.progress_grid)

    def angles_at(self, t: float | np.ndarray) -> np.ndarray:
        """``(yaw, pitch, roll)`` in radians; shape ``(3,)`` or ``(3, N)``."""
        s = self.progress_at(t)
        base = np.stack(
            [
                np.interp(s, self.s_grid, self.yaw_grid),
                np.interp(s, self.s_grid, self.pitch_grid),
                np.interp(s, self.s_grid, self.roll_grid),
            ]
        )
        return base + self.jitter(t)

    def R_at(self, t: float) -> np.ndarray:  # noqa: N802 - matches the CV literature
        yaw, pitch, roll = (float(v) for v in self.angles_at(float(t)))
        return g.from_yaw_pitch_roll(yaw, pitch, roll)

    def origin_at(self, t: float) -> np.ndarray:
        """Where the *lens* is: on a circle of ``arm_radius_m`` about the body axis."""
        centre = np.zeros(3, dtype=np.float32) if self.centre is None else self.centre
        if self.arm_radius_m <= 0.0 and self.sway_m <= 0.0:
            return centre.copy()
        yaw = float(self.angles_at(float(t))[0])
        offset = np.array(
            [math.sin(yaw) * self.arm_radius_m, 0.0, math.cos(yaw) * self.arm_radius_m],
            dtype=np.float32,
        )
        if self.sway_m > 0.0:
            tt = float(t)
            offset += np.float32(self.sway_m) * np.array(
                [
                    math.sin(2.0 * math.pi * 0.23 * tt + 0.7),
                    math.sin(2.0 * math.pi * 0.41 * tt + 2.1),
                    math.sin(2.0 * math.pi * 0.17 * tt + 4.4),
                ],
                dtype=np.float32,
            )
        return centre + offset

    def pose_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return self.R_at(t), self.origin_at(t)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_s": round(self.duration_s, 6),
            "fps": self.fps,
            "n_frames": self.n_frames,
            "arm_radius_m": self.arm_radius_m,
            "sway_m": self.sway_m,
            "seed": self.seed,
            "centre": None if self.centre is None else [float(v) for v in self.centre],
        }


# ======================================================================================
# Camera artefacts
# ======================================================================================


@dataclass(frozen=True, slots=True)
class CaptureEffects:
    """Everything the sensor and the ISP do to a clean render."""

    name: str
    arm_radius_m: float | None = None
    subsamples: int = 1
    exposure_s: float = 0.0
    rolling_shutter_s: float = 0.0
    rs_bands: int = 16
    vignette_strength: float = 0.0
    noise_sigma: float = 0.0
    jpeg_quality: int = 0
    auto_exposure: bool = False
    ae_target: float = 118.0
    ae_tau_s: float = 0.35
    ae_gain_min: float = 0.55
    ae_gain_max: float = 2.2

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arm_radius_m": self.arm_radius_m,
            "subsamples": self.subsamples,
            "exposure_s": self.exposure_s,
            "rolling_shutter_s": self.rolling_shutter_s,
            "vignette_strength": self.vignette_strength,
            "noise_sigma": self.noise_sigma,
            "jpeg_quality": self.jpeg_quality,
            "auto_exposure": self.auto_exposure,
            "ae_target": self.ae_target,
            "ae_tau_s": self.ae_tau_s,
        }


EFFECT_PRESETS: Final[dict[str, CaptureEffects]] = {
    # Nothing but geometry. Used to prove the harness agrees with itself.
    "clean": CaptureEffects(name="clean", arm_radius_m=0.0),
    # A good phone, held by a steady person, in a normally lit room.
    "realistic": CaptureEffects(
        name="realistic",
        arm_radius_m=0.28,
        subsamples=5,
        exposure_s=1.0 / 120.0,
        rolling_shutter_s=0.022,
        vignette_strength=0.22,
        noise_sigma=1.8,
        jpeg_quality=88,
        auto_exposure=True,
        ae_target=120.0,
        ae_tau_s=0.35,
    ),
    # A cheap phone, swung fast, in a dim room. The regression case that hurts.
    "harsh": CaptureEffects(
        name="harsh",
        arm_radius_m=0.35,
        subsamples=9,
        exposure_s=1.0 / 42.0,
        rolling_shutter_s=0.033,
        rs_bands=16,
        vignette_strength=0.36,
        noise_sigma=6.0,
        jpeg_quality=68,
        auto_exposure=True,
        ae_target=126.0,
        ae_tau_s=0.95,
        ae_gain_min=0.45,
        ae_gain_max=3.0,
    ),
}


def resolve_effects(effects: CaptureEffects | str) -> CaptureEffects:
    if isinstance(effects, CaptureEffects):
        return effects
    try:
        return EFFECT_PRESETS[str(effects)]
    except KeyError as exc:
        raise ValueError(
            f"unknown effects preset {effects!r}; expected one of {sorted(EFFECT_PRESETS)}"
        ) from exc


# ======================================================================================
# The capture
# ======================================================================================


@dataclass(eq=False)
class SyntheticCapture:
    """A simulated video plus everything that was true about it while it was made."""

    frame_paths: list[Path]
    poses: list[CameraPose]
    ground_truth: np.ndarray
    intrinsics: Intrinsics
    trajectory: Trajectory
    room: Room
    effects: CaptureEffects
    times: np.ndarray
    origins: np.ndarray
    gains: np.ndarray
    out_dir: Path
    seed: int

    @property
    def n_frames(self) -> int:
        return len(self.frame_paths)

    @property
    def rotations(self) -> list[np.ndarray]:
        return [p.R for p in self.poses]

    def load_frames(self) -> list[np.ndarray]:
        """Read every frame back off disk, in order."""
        out: list[np.ndarray] = []
        for p in self.frame_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"could not read simulated frame {p}")
            out.append(img)
        return out

    def truth_dict(self) -> dict[str, Any]:
        """Everything a scorer needs, JSON-serialisable."""
        intr = self.intrinsics
        return {
            "seed": int(self.seed),
            "room": {
                "style": self.room.style,
                "seed": int(self.room.seed),
                "width_m": round(float(self.room.width_m), 6),
                "depth_m": round(float(self.room.depth_m), 6),
                "height_m": round(float(self.room.height_m), 6),
                "eye": [round(float(v), 6) for v in self.room.eye],
            },
            "trajectory": self.trajectory.as_dict(),
            "effects": self.effects.as_dict(),
            "intrinsics": {
                "focal": round(float(intr.focal), 6),
                "cx": float(intr.cx),
                "cy": float(intr.cy),
                "width": int(intr.width),
                "height": int(intr.height),
                "k1": float(intr.k1),
                "k2": float(intr.k2),
                "hfov_deg": round(float(intr.hfov_deg), 4),
                "vfov_deg": round(float(intr.vfov_deg), 4),
            },
            "frames": [
                {
                    "index": int(pose.frame_index),
                    "file": self.frame_paths[i].name,
                    "t": round(float(self.times[i]), 6),
                    "R": [[round(float(v), 9) for v in row] for row in np.asarray(pose.R)],
                    "yaw_deg": round(math.degrees(pose.yaw), 6),
                    "pitch_deg": round(math.degrees(pose.pitch), 6),
                    "roll_deg": round(math.degrees(g.yaw_pitch_roll(pose.R)[2]), 6),
                    "origin": [round(float(v), 6) for v in self.origins[i]],
                    "gain": round(float(self.gains[i]), 6),
                }
                for i, pose in enumerate(self.poses)
            ],
        }

    def write_truth(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.truth_dict(), indent=2), encoding="utf-8")
        return path


def simulate_capture(
    room: Room,
    trajectory: Trajectory,
    intrinsics: Intrinsics,
    *,
    out_dir: Path | str,
    effects: CaptureEffects | str = "realistic",
    seed: int = 1234,
    arm_radius_m: float | None = None,
    ground_truth_width: int = 2048,
    ground_truth_supersample: int = 1,
    supersample: int = 1,
    write_frames: bool = True,
    frame_suffix: str = ".png",
    progress: Any | None = None,
) -> SyntheticCapture:
    """Render an entire simulated capture, frames and ground truth together.

    ``arm_radius_m`` overrides both the trajectory and the effects preset, which is how
    the harness produces a matched nodal / parallax pair from one trajectory.

    Frames are written losslessly (PNG by default) even when ``effects`` includes a JPEG
    stage — the JPEG is applied *inside* :func:`~biosphere.synth.render.apply_camera_response`
    so that the artefacts are the simulated camera's, not the harness's file format.
    """
    fx = resolve_effects(effects)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = trajectory if trajectory.centre is not None else trajectory.centred_at(room.eye)
    radius = arm_radius_m if arm_radius_m is not None else fx.arm_radius_m
    if radius is not None and float(radius) != traj.arm_radius_m:
        traj = traj.with_arm_radius(float(radius))

    gt_h = int(ground_truth_width) // 2
    ground_truth = render_equirect(
        room, int(ground_truth_width), gt_h, room.eye, supersample=ground_truth_supersample
    )

    times = traj.frame_times()
    n = times.shape[0]
    root = np.random.SeedSequence(int(seed) & 0x7FFFFFFF)
    children = root.spawn(n)

    frame_paths: list[Path] = []
    poses: list[CameraPose] = []
    origins = np.zeros((n, 3), dtype=np.float64)
    gains = np.ones(n, dtype=np.float64)
    gain = 1.0
    dt = 1.0 / float(traj.fps)
    alpha = 1.0 - math.exp(-dt / max(fx.ae_tau_s, 1e-6)) if fx.auto_exposure else 1.0

    for i in range(n):
        t = float(times[i])
        R = traj.R_at(t)
        origin = traj.origin_at(t)
        raw = render_view(
            room,
            R,
            intrinsics,
            origin=origin,
            subsamples=fx.subsamples,
            exposure_s=fx.exposure_s,
            rolling_shutter_s=fx.rolling_shutter_s,
            pose_at=traj.R_at if (fx.subsamples > 1 or fx.rolling_shutter_s > 0.0) else None,
            origin_at=traj.origin_at if (fx.subsamples > 1 or fx.rolling_shutter_s > 0.0) else None,
            t0=t,
            supersample=supersample,
            bands=fx.rs_bands,
        )
        if fx.auto_exposure:
            luma = float(raw.mean())
            desired = float(
                np.clip(fx.ae_target / max(luma, 1.0), fx.ae_gain_min, fx.ae_gain_max)
            )
            if i == 0:
                gain = desired  # metering has already settled before recording starts
            applied = gain
            gain += (desired - gain) * alpha
        else:
            applied = 1.0

        img = apply_camera_response(
            raw,
            gain=applied,
            vignette_strength=fx.vignette_strength,
            noise_sigma=fx.noise_sigma,
            jpeg_quality=fx.jpeg_quality,
            rng=np.random.default_rng(children[i]),
        )

        path = out_dir / f"frame_{i:05d}{frame_suffix}"
        if write_frames:
            if not cv2.imwrite(str(path), img):
                raise RuntimeError(f"failed to write simulated frame {path}")
        frame_paths.append(path)
        poses.append(
            CameraPose(
                frame_index=i,
                R=R,
                intrinsics=intrinsics,
                gain=np.full(3, applied, dtype=np.float64),
                vignette=(-float(fx.vignette_strength), 0.0, 0.0),
            )
        )
        origins[i] = origin
        gains[i] = applied
        if progress is not None:
            progress.stage("simulate", (i + 1) / n, f"frame {i + 1}/{n}")

    return SyntheticCapture(
        frame_paths=frame_paths,
        poses=poses,
        ground_truth=ground_truth,
        intrinsics=intrinsics,
        trajectory=traj,
        room=room,
        effects=fx,
        times=times,
        origins=origins,
        gains=gains,
        out_dir=out_dir,
        seed=int(seed),
    )
