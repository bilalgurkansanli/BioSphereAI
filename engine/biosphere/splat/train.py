"""Brush, run as a subprocess, with its progress read back.

Brush (github.com/ArthurBrussee/brush) trains 3D Gaussian splats on any GPU through wgpu:
no CUDA toolkit, no compiler, one executable. That is the reason it is the trainer here,
on a Windows laptop that has neither nvcc nor MSVC and never will in production either.
Its "MCMC-like" densification with a hard splat cap is also what keeps a room at the
~1M splats the SOG budget wants.

The prebuilt v0.3.0 binary is the one this wrapper speaks to. Its flags were renamed on
the unreleased main branch (``--total-steps`` -> ``--total-train-iters``, and a relative
``--export-path`` moved to the dataset's parent), so the wrapper checks ``--version`` once
and refuses anything it has not been written against rather than guess.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["TrainConfig", "TrainReport", "brush_binary", "brush_version", "train_splat", "BrushError"]

_DEFAULT_TOOLS = Path(os.environ.get("BIOSPHERE_TOOLS_DIR", "C:/biosphere-data/tools"))
_SUPPORTED_VERSIONS = ("0.3.0",)


class BrushError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 30000
    sh_degree: int = 2
    max_splats: int = 1_500_000
    #: Longest image edge Brush loads; frames on disk may be larger.
    max_resolution: int = 1600
    #: Hold out every Nth frame for PSNR; 0 trains on everything.
    eval_split_every: int = 16
    eval_every: int = 2000
    refine_every: int = 200
    seed: int = 42
    extra_args: tuple[str, ...] = ()

    @classmethod
    def for_mode(cls, mode: str) -> TrainConfig:
        if mode == "fast":
            return cls(total_steps=12000, sh_degree=1, max_splats=800_000, max_resolution=1280, eval_every=2000)
        if mode == "studio":
            return cls(total_steps=40000, sh_degree=3, max_splats=2_000_000, max_resolution=1920, eval_every=2000)
        return cls()


@dataclass
class TrainReport:
    ply_path: Path | None = None
    steps: int = 0
    seconds: float = 0.0
    splat_count: int = 0
    psnr: float | None = None
    ssim: float | None = None
    evals: list[tuple[int, float, float]] | None = None
    log_path: Path | None = None
    binary_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps, "seconds": round(self.seconds, 1), "splat_count": self.splat_count,
            "psnr": None if self.psnr is None else round(self.psnr, 3),
            "ssim": None if self.ssim is None else round(self.ssim, 4),
            "evals": [(s, round(p, 3), round(q, 4)) for s, p, q in (self.evals or [])],
            "binary_version": self.binary_version,
        }


def brush_binary() -> Path:
    override = os.environ.get("BIOSPHERE_BRUSH_BIN")
    candidates = [Path(override)] if override else []
    candidates += [_DEFAULT_TOOLS / "brush" / "brush_app.exe", _DEFAULT_TOOLS / "brush" / "brush_app"]
    found = shutil.which("brush_app") or shutil.which("brush")
    if found:
        candidates.append(Path(found))
    for c in candidates:
        if c.exists():
            return c
    raise BrushError(
        "Brush trainer not found. Set BIOSPHERE_BRUSH_BIN or unzip brush-app-x86_64-pc-windows-msvc.zip "
        f"into {_DEFAULT_TOOLS / 'brush'}"
    )


def brush_version(binary: Path | None = None) -> str:
    binary = binary or brush_binary()
    out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60)
    m = re.search(r"(\d+\.\d+\.\d+)", out.stdout + out.stderr)
    return m.group(1) if m else "unknown"


_RE_EVAL = re.compile(r"Eval iter (\d+):\s*PSNR\s*([0-9.]+),\s*ssim\s*([0-9.]+)", re.IGNORECASE)
_RE_REFINE = re.compile(r"Refine iter (\d+),\s*(\d+) splats", re.IGNORECASE)
_RE_STEP = re.compile(r"(\d+)\s*/\s*(\d+)")


def train_splat(
    dataset_dir: Path,
    out_dir: Path,
    *,
    config: TrainConfig | None = None,
    progress=None,
    cancel_check=None,
    binary: Path | None = None,
) -> TrainReport:
    """Train on a COLMAP dataset (``images/`` + ``sparse/0``) and return the final PLY.

    ``progress(fraction, message)`` is called as training advances; ``cancel_check()`` is
    polled between output lines and, when it returns True, the process is killed.
    """
    config = config or TrainConfig()
    binary = binary or brush_binary()
    version = brush_version(binary)
    if version not in _SUPPORTED_VERSIONS:
        raise BrushError(f"Brush {version} is not a version this wrapper knows ({_SUPPORTED_VERSIONS})")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("export_*.ply"):
        stale.unlink()
    export_every = int(config.total_steps)
    args = [
        str(binary), str(dataset_dir),
        "--total-steps", str(config.total_steps),
        "--sh-degree", str(config.sh_degree),
        "--max-splats", str(config.max_splats),
        "--max-resolution", str(config.max_resolution),
        "--refine-every", str(config.refine_every),
        "--export-every", str(export_every),
        "--export-path", str(out_dir),
        "--export-name", "export_{iter}.ply",
        "--seed", str(config.seed),
        "--eval-every", str(config.eval_every),
    ]
    if config.eval_split_every and config.eval_split_every > 0:
        args += ["--eval-split-every", str(config.eval_split_every)]
    args += list(config.extra_args)

    if config.eval_split_every and config.eval_split_every > 0:
        # Brush prints nothing when its output is not a terminal, so the eval renders it
        # writes to disk are the progress signal, and the PSNR is computed from them.
        args += ["--eval-save-to-disk"]
        for stale in out_dir.glob("eval_*"):
            shutil.rmtree(stale, ignore_errors=True)

    report = TrainReport(log_path=out_dir / "brush.log", binary_version=version, evals=[])
    t0 = time.perf_counter()
    env = dict(os.environ)
    with open(report.log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(args) + "\n")
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", env=env, cwd=str(out_dir),
        )
        last_step = 0
        seen_evals: set[int] = set()
        try:
            while True:
                rc = proc.poll()
                # Progress: the newest eval_<iter> directory; between evals, interpolate on
                # wall clock against the pace of the previous interval.
                evals = sorted(int(m.group(1)) for p in out_dir.glob("eval_*") if (m := re.search(r"eval_(\d+)$", p.name)))
                if evals and evals[-1] != last_step:
                    last_step = evals[-1]
                    for step in evals:
                        if step not in seen_evals:
                            seen_evals.add(step)
                            psnr = _psnr_of_eval_dir(out_dir / f"eval_{step}", dataset_dir, config.max_resolution)
                            if psnr is not None:
                                report.evals.append((step, psnr, 0.0))
                                report.psnr = psnr
                if progress is not None:
                    frac = min(last_step / float(config.total_steps), 0.99)
                    msg = f"step {last_step}/{config.total_steps}"
                    if report.psnr is not None:
                        msg += f", PSNR {report.psnr:.2f}"
                    progress(frac, msg)
                if rc is not None:
                    break
                if cancel_check is not None and cancel_check():
                    proc.kill()
                    raise BrushError("training cancelled")
                time.sleep(2.0)
        finally:
            if proc.stdout is not None:
                log.write(proc.stdout.read())
            proc.wait()
    report.seconds = time.perf_counter() - t0
    report.steps = config.total_steps
    plys = sorted(out_dir.glob("export_*.ply"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if proc.returncode != 0 or not plys:
        tail = report.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise BrushError(f"Brush exited with {proc.returncode} and {len(plys)} exports:\n{tail}")
    report.ply_path = plys[-1]
    report.splat_count = _count_vertices(report.ply_path)
    if progress is not None:
        progress(1.0, f"trained {report.splat_count:,} splats")
    return report


def _count_vertices(ply: Path) -> int:
    with open(ply, "rb") as fh:
        for _ in range(64):
            line = fh.readline()
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
            if line.startswith(b"end_header"):
                break
    return 0


def _psnr_of_eval_dir(eval_dir: Path, dataset_dir: Path, max_resolution: int) -> float | None:
    """PSNR of Brush's held-out renders against the dataset images they were held out from."""
    import cv2
    import numpy as np

    image_dir = Path(dataset_dir) / "images"
    if not image_dir.exists():
        return None
    # The undistorter keeps the images' sub-directories (``base/images/f_00012.jpg``); the
    # renders are flat and named by stem.
    by_stem = {p.stem: p for p in image_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    vals = []
    for render in sorted(eval_dir.glob("*.png")):
        gt_path = by_stem.get(render.stem)
        if gt_path is None:
            continue
        a = cv2.imread(str(render), cv2.IMREAD_COLOR)
        b = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
        vals.append(10.0 * np.log10(255.0 ** 2 / max(mse, 1e-6)))
    return float(np.mean(vals)) if vals else None
