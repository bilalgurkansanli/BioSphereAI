"""Liveness plus the three environment facts the dashboard warns about."""

from __future__ import annotations

import shutil
import subprocess
import threading
from functools import lru_cache

import cv2
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from biosphere.api import schemas as sc
from biosphere.api.deps import SettingsDep, StorageDep
from biosphere.core.config import Settings
from biosphere.core.storage import LocalStorage

router = APIRouter(tags=["health"])


@lru_cache(maxsize=4)
def _ffmpeg_present(binary: str) -> bool:
    """Resolve *and* execute: a stale PATH entry pointing at a deleted binary is common."""
    if shutil.which(binary) is None:
        return False
    try:
        proc = subprocess.run([binary, "-version"], capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


@lru_cache(maxsize=1)
def _gpu_name() -> str | None:
    """Name of the CUDA device, if torch is installed and a device is actually usable.

    Cached because importing torch costs a second or two — far too much for a health probe
    that a load balancer may hit every few seconds.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is an optional extra
        return None
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 - broken driver, missing DLL, ...
        return None
    return None


@lru_cache(maxsize=1)
def _version() -> str:
    try:
        from importlib.metadata import version

        return version("biosphere-engine")
    except Exception:  # noqa: BLE001 - running from a source checkout
        return "0.1.0"


#: How long a running job may go without reporting progress before it is called stuck.
#: The longest honest silence in the pipeline is now a single bundle-adjustment evaluation,
#: which reports every twentieth call; ten minutes is far beyond that and far under the
#: half-hour a dense capture legitimately spends in that stage.
_STALL_SECONDS = 600.0


def _stalled() -> list[str]:
    """Job ids that are running and have gone quiet. Empty is the healthy answer."""
    try:
        from biosphere.core.jobs import get_job_manager

        return [job_id for job_id, _ in get_job_manager().stalled_jobs(quiet_seconds=_STALL_SECONDS)]
    except Exception:  # noqa: BLE001 - health must never be the thing that breaks
        return []


def _ai_backend() -> dict[str, object]:
    """Where the room-emptying models run. Must not load one to answer."""
    try:
        from biosphere.ai import backend

        return backend.describe()
    except Exception:  # noqa: BLE001 - health must never be the thing that breaks
        return {"mode": "unknown"}


@lru_cache(maxsize=1)
def _walkthrough_backend() -> dict[str, object]:
    """Whether the 3D walkthrough can be built here: the trainer and the SOG writer on disk.

    Existence checks only, no process is spawned: this answers on every health poll.
    """
    out: dict[str, object] = {"available": False, "trainer": None, "sog_writer": False, "sfm": False}
    try:
        from biosphere.splat.export import splat_transform_command
        from biosphere.splat.train import brush_binary

        try:
            out["trainer"] = str(brush_binary())
        except Exception:  # noqa: BLE001 - reported as absent
            out["trainer"] = None
        try:
            splat_transform_command()
            out["sog_writer"] = True
        except Exception:  # noqa: BLE001
            out["sog_writer"] = False
        try:
            import pycolmap  # noqa: F401

            out["sfm"] = True
        except Exception:  # noqa: BLE001
            out["sfm"] = False
        out["available"] = bool(out["trainer"]) and bool(out["sog_writer"]) and bool(out["sfm"])
    except Exception:  # noqa: BLE001 - the package may be absent on a stitch-only server
        pass
    return out


#: The two probes that import native giants: torch (with CUDA init) for the GPU name, and
#: pycolmap for the walkthrough. Cold, on this laptop, they took over a MINUTE while another
#: process was training on the GPU — and the dashboard's engine pill, whose request has a
#: deadline, called the engine "kapalı" for that whole minute after every restart. So they
#: are warmed on a background thread at startup, and a health request waits for them only
#: briefly: past that it answers with what it knows and says the rest is still being checked.
_warmed = threading.Event()
_warm_started = threading.Lock()
_warm_thread: threading.Thread | None = None

#: How long a health request waits for the warm-up. Long enough that a request arriving a
#: moment after boot on an idle machine gets the full answer; short enough to stay inside
#: the dashboard's request deadline when the machine is anything but idle.
_WARM_WAIT_S = 3.0


def _warm_up(settings: Settings) -> None:
    try:
        _ffmpeg_present(settings.ffmpeg_bin)
        if settings.enable_gpu:
            _gpu_name()
        _walkthrough_backend()
        _version()
    except Exception:  # noqa: BLE001 - health must never be the thing that breaks
        pass
    finally:
        _warmed.set()


def start_warm_up(settings: Settings) -> None:
    """Fill the slow probes' caches off the request path. Idempotent; called at startup."""
    global _warm_thread
    with _warm_started:
        if _warm_thread is None:
            _warm_thread = threading.Thread(target=_warm_up, args=(settings,), name="health-warm-up", daemon=True)
            _warm_thread.start()


def _collect(settings: Settings, storage: LocalStorage) -> sc.HealthResponse:
    start_warm_up(settings)  # a process that skipped the lifespan (a test, a script) still warms
    warm = _warmed.wait(timeout=_WARM_WAIT_S)
    if warm:
        gpu = _gpu_name() if settings.enable_gpu else None
        walkthrough = _walkthrough_backend()
    else:
        # Not "no GPU" and not "no walkthrough": unknown for a few more seconds. The GPU name
        # is left empty (the dashboard's tooltip is the only reader), the walkthrough says so.
        gpu = None
        walkthrough = {"available": False, "trainer": None, "sog_writer": False, "sfm": False, "checking": True}
    return sc.HealthResponse(
        status="ok",
        version=_version(),
        ffmpeg=_ffmpeg_present(settings.ffmpeg_bin),
        opencv=cv2.__version__,
        gpu=gpu,
        media_root_free_gb=round(storage.free_bytes() / (1024**3), 2),
        stalled_jobs=_stalled(),
        ai=_ai_backend(),
        walkthrough=walkthrough,
    )


@router.get("/health", response_model=sc.HealthResponse, summary="Service health")
async def health(settings: SettingsDep, storage: StorageDep) -> sc.HealthResponse:
    """Everything here touches the filesystem or spawns a process, so it runs off-loop."""
    return await run_in_threadpool(_collect, settings, storage)
