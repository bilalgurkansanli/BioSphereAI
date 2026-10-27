"""The stitch job queue: submission, progress, cancellation and output persistence.

Architecture
------------
One in-process ``ThreadPoolExecutor`` runs the pipeline. Everything queue-shaped is behind
:class:`JobManager`, so moving to Celery/RQ later means reimplementing *this one class*
(``submit`` / ``get`` / ``cancel`` / ``subscribe``) against a broker — the routes, the
progress model and the event payloads stay exactly as they are.

The thread/async bridge is the subtle part. The pipeline runs on a worker thread; the SSE
subscribers are ``asyncio.Queue``s living on the event loop. A worker thread must never
``await`` or call ``queue.put`` directly — ``asyncio.Queue`` is not thread-safe and doing so
either corrupts the queue or (worse) appears to work until the first real load. Instead the
loop is captured during lifespan startup and every publish goes through
``loop.call_soon_threadsafe``.

Progress
--------
The pipeline reports ``(stage_name, fraction)``. Stages are wildly uneven — frame extraction
and multi-band blending dominate the wall clock — so a naive "stage index / stage count"
progress bar sits at 20% for two minutes and then jumps. :data:`STAGE_WEIGHTS` maps the
ordered stage list from ``types.ts`` onto measured cost shares, and overall progress is
``cumulative_before(stage) + weight(stage) * stage_fraction``, forced monotonic.
"""

from __future__ import annotations

import asyncio
import math
import shutil
import threading
import time
import traceback
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import orjson
from sqlmodel import Session, select

from biosphere.core.config import Settings, get_settings
from biosphere.core.db import session_scope
from biosphere.core.logging import bind_context, clear_context, get_logger
from biosphere.core.models import ApiException, Job, Property, Scene, Walkthrough, utcnow
from biosphere.core.storage import (
    CUBE_FACES,
    LEVEL_TARGET_WIDTHS,
    LocalStorage,
    get_storage,
    write_jpeg,
    write_png,
    write_webp,
)
from biosphere.pano import geometry as geo
from biosphere.pano.types import (
    CoverageReport,
    PipelineConfig,
    QualityMode,
    StitchError,
    StitchResult,
)

__all__ = [
    "JOB_STAGES",
    "WALK_STAGES",
    "WALK_STAGE_WEIGHTS",
    "WALK_STAGE_MESSAGES",
    "StageTable",
    "STITCH_TABLE",
    "WALK_TABLE",
    "WALK_BACKEND",
    "STAGE_WEIGHTS",
    "STAGE_CUMULATIVE",
    "STAGE_MESSAGES",
    "JobCancelled",
    "JobManager",
    "JobProgress",
    "STITCH_BACKEND",
    "get_job_manager",
    "reset_job_manager",
    "is_terminal_event",
    "TERMINAL_STATUSES",
]

log = get_logger("biosphere.jobs")

#: Mirrors ``JOB_STAGES`` in web/src/lib/types.ts — the UI renders these as a progress rail.
JOB_STAGES: tuple[str, ...] = (
    "upload", "probe", "extract", "select", "features", "match", "bundle",
    "exposure", "warp", "seam", "blend", "refine", "poles", "levels", "done",
)

#: Cost share of each stage, summing to 1.0 over everything except the terminal ``done``.
STAGE_WEIGHTS: dict[str, float] = {
    "upload": 0.02,
    "probe": 0.02,
    "extract": 0.18,
    "select": 0.05,
    "features": 0.10,
    "match": 0.12,
    "bundle": 0.08,
    "exposure": 0.03,
    "warp": 0.10,
    "seam": 0.06,
    "blend": 0.16,
    "refine": 0.04,
    "poles": 0.02,
    "levels": 0.02,
    "done": 0.0,
}

#: Progress already banked when a stage begins, i.e. the sum of every earlier stage's weight.
STAGE_CUMULATIVE: dict[str, float] = {}
_acc = 0.0
for _stage in JOB_STAGES:
    STAGE_CUMULATIVE[_stage] = _acc
    _acc += STAGE_WEIGHTS[_stage]
del _acc, _stage

#: Bilingual copy for every stage. Shown verbatim in the dashboard.
STAGE_MESSAGES: dict[str, tuple[str, str]] = {
    "upload": ("Receiving the capture", "Çekim alınıyor"),
    "probe": ("Inspecting the capture", "Çekim inceleniyor"),
    "extract": ("Extracting frames", "Kareler çıkarılıyor"),
    "select": ("Selecting the sharpest keyframes", "En net anahtar kareler seçiliyor"),
    "features": ("Detecting features", "Öznitelikler bulunuyor"),
    "match": ("Matching frames to each other", "Kareler eşleştiriliyor"),
    "bundle": ("Solving the camera geometry", "Kamera geometrisi çözülüyor"),
    "exposure": ("Balancing exposure", "Pozlama dengeleniyor"),
    "warp": ("Projecting onto the sphere", "Küreye yansıtılıyor"),
    "seam": ("Finding invisible seams", "Görünmez dikişler aranıyor"),
    "blend": ("Blending the panorama", "Panorama harmanlanıyor"),
    "refine": ("Refining fine detail", "İnce detaylar iyileştiriliyor"),
    "poles": ("Filling the poles", "Kutuplar dolduruluyor"),
    "levels": ("Writing image levels", "Görüntü seviyeleri yazılıyor"),
    "done": ("Finished", "Tamamlandı"),
}

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

#: How many events are replayed to a subscriber that connects after the job started.
#: Without this, a client that opens the SSE stream a few milliseconds late would miss the
#: `done` event and hang until its own timeout.
MAX_HISTORY = 400

#: Stages of a WALKTHROUGH job. A different rail from the stitch: a splat is not a panorama
#: and the dashboard should not pretend the two share a life cycle. Mirrors ``WALK_STAGES``
#: in web/src/lib/types.ts.
WALK_STAGES: tuple[str, ...] = (
    "upload", "extract", "features", "sfm", "register", "train", "clean", "export", "done",
)

#: Measured on the Berlin flat (305 + 2 x 160 frames, fast mode, RTX 5050): training is
#: over half of the wall clock and the two solves most of the rest.
WALK_STAGE_WEIGHTS: dict[str, float] = {
    "upload": 0.01,
    "extract": 0.05,
    "features": 0.09,
    "sfm": 0.16,
    "register": 0.03,
    "train": 0.56,
    "clean": 0.02,
    "export": 0.08,
    "done": 0.0,
}

WALK_STAGE_MESSAGES: dict[str, tuple[str, str]] = {
    "upload": ("Receiving the videos", "Videolar alınıyor"),
    "extract": ("Choosing the sharp frames", "Keskin kareler seçiliyor"),
    "features": ("Describing every frame", "Kareler tanımlanıyor"),
    "sfm": ("Solving the camera path", "Kamera yolu çözülüyor"),
    "register": ("Placing the rooms in the flat", "Odalar daireye yerleştiriliyor"),
    "train": ("Training the 3D scene", "3B sahne eğitiliyor"),
    "clean": ("Removing floaters", "Artıklar temizleniyor"),
    "export": ("Compressing and drawing the plan", "Sıkıştırılıyor ve plan çiziliyor"),
    "done": ("3D property ready", "3B tur hazır"),
}


@dataclass(frozen=True)
class StageTable:
    """One job kind's rail: its ordered stages, cost shares, banked progress and copy."""

    stages: tuple[str, ...]
    weights: dict[str, float]
    messages: dict[str, tuple[str, str]]
    cumulative: dict[str, float]

    @classmethod
    def build(
        cls, stages: tuple[str, ...], weights: dict[str, float], messages: dict[str, tuple[str, str]]
    ) -> StageTable:
        cumulative: dict[str, float] = {}
        acc = 0.0
        for stage in stages:
            cumulative[stage] = acc
            acc += weights[stage]
        return cls(stages=stages, weights=dict(weights), messages=dict(messages), cumulative=cumulative)


STITCH_TABLE = StageTable.build(JOB_STAGES, STAGE_WEIGHTS, STAGE_MESSAGES)
WALK_TABLE = StageTable.build(WALK_STAGES, WALK_STAGE_WEIGHTS, WALK_STAGE_MESSAGES)


class JobCancelled(Exception):
    """Raised inside the worker thread when the user cancelled the job."""


# --------------------------------------------------------------------------------------
# Pipeline indirection
# --------------------------------------------------------------------------------------

StitchBackend = Callable[..., StitchResult]


def _default_stitch_backend(
    source: Path | list[Path],
    config: PipelineConfig,
    *,
    work_dir: Path,
    progress: Any | None = None,
) -> StitchResult:
    """Thin wrapper that imports the heavy pipeline **lazily**.

    ``biosphere.pano.pipeline`` pulls in the whole numeric stack and may not even exist yet
    in a given checkout; importing it at module scope would make the API unimportable. The
    import therefore happens inside the worker thread, and a failure becomes a structured
    job error rather than a 500 with a traceback.
    """
    try:
        from biosphere.pano.pipeline import stitch_capture
    except ImportError as exc:
        raise ApiException(
            "pipeline_unavailable",
            "The panorama engine is not available on this server.",
            "Panorama motoru bu sunucuda kullanılamıyor.",
            hint=f"Import of biosphere.pano.pipeline failed: {exc}",
            status_code=503,
        ) from exc
    return stitch_capture(source, config, work_dir=work_dir, progress=progress)


#: Swappable so tests (and a future GPU/remote worker) can stand in for the real pipeline.
STITCH_BACKEND: StitchBackend = _default_stitch_backend

WalkBackend = Callable[..., dict[str, Any]]


def _default_walk_backend(
    spec: dict[str, Any], *, progress: Any | None = None, cancel_check: Any | None = None
) -> dict[str, Any]:
    """Run ``biosphere.splat.worker`` in a child process and return the manifest it wrote.

    A child process on purpose: the SfM core and the GPU trainer both terminate the process
    on an internal check failure, and the process they must not terminate is this API. Its
    stdout is a stream of JSON progress lines; the last line names the manifest or carries
    the error. ``cancel_check`` kills it.

    Two things the plumbing must do or the child dies quietly:

    * **Drain stderr.** COLMAP and the trainer write their own logs there. A pipe nobody
      reads fills after a few tens of kilobytes, the next write blocks, and the child sits
      at 0 % CPU forever - which is exactly what an incremental mapping did on the first
      real upload. stderr goes to a thread and to ``work_dir/worker.stderr.log``.
    * **Notice silence.** Progress lines arrive at least every few minutes from every
      stage; ``BIOSPHERE_WALK_STALL_SECONDS`` (default 30 min) without one means the child
      is stuck, and the job fails with that verdict rather than "running" until a restart.
    """
    import os as _os
    import queue
    import subprocess
    import sys
    import threading
    import time as _time

    import orjson as _orjson

    work_dir = Path(spec["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_path = work_dir / "spec.json"
    spec_path.write_bytes(_orjson.dumps(spec))

    env = dict(_os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("GLOG_minloglevel", "1")
    module = _os.environ.get("BIOSPHERE_WALK_WORKER_MODULE", "biosphere.splat.worker")
    stall_seconds = float(_os.environ.get("BIOSPHERE_WALK_STALL_SECONDS", "1800"))
    proc = subprocess.Popen(
        [sys.executable, "-m", module, str(spec_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", env=env,
    )
    assert proc.stdout is not None and proc.stderr is not None

    lines: queue.Queue[str | None] = queue.Queue()
    stderr_tail: list[str] = []
    stderr_log = work_dir / "worker.stderr.log"

    def pump_stdout() -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                lines.put(line)
        finally:
            lines.put(None)

    def pump_stderr() -> None:
        try:
            with open(stderr_log, "a", encoding="utf-8", errors="replace") as fh:
                for line in proc.stderr:  # type: ignore[union-attr]
                    fh.write(line)
                    stderr_tail.append(line)
                    if len(stderr_tail) > 200:
                        del stderr_tail[:100]
        except Exception:  # noqa: BLE001 - a logging failure must not take the job with it
            pass

    threading.Thread(target=pump_stdout, name="walk-stdout", daemon=True).start()
    threading.Thread(target=pump_stderr, name="walk-stderr", daemon=True).start()

    result_path: str | None = None
    error: dict[str, Any] | None = None
    stalled = False
    last_line_at = _time.monotonic()
    try:
        while True:
            try:
                raw = lines.get(timeout=2.0)
            except queue.Empty:
                if cancel_check is not None and cancel_check():
                    proc.kill()
                    raise JobCancelled("cancelled") from None
                if _time.monotonic() - last_line_at > stall_seconds:
                    stalled = True
                    proc.kill()
                    break
                continue
            if raw is None:
                break
            last_line_at = _time.monotonic()
            line = raw.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = _orjson.loads(line)
            except ValueError:
                continue
            if "stage" in msg and progress is not None:
                progress.stage(str(msg["stage"]), float(msg.get("fraction", 0.0)), str(msg.get("message", "")))
            elif "log" in msg and progress is not None and hasattr(progress, "log"):
                progress.log(str(msg["log"]))
            elif "result" in msg:
                result_path = str(msg["result"])
            elif "error" in msg:
                error = dict(msg["error"])
            if cancel_check is not None and cancel_check():
                proc.kill()
                raise JobCancelled("cancelled")
    finally:
        proc.wait()
    stderr = "".join(stderr_tail)
    if stalled:
        log.error("walk.worker_stalled", stall_seconds=stall_seconds, stderr=stderr[-3000:])
        raise ApiException(
            "walk_worker_stalled",
            f"The 3D engine made no progress for {stall_seconds / 60:.0f} minutes and was stopped.",
            f"3B motor {stall_seconds / 60:.0f} dakika boyunca ilerleme bildirmedi ve durduruldu.",
            hint=stderr[-600:] or "no output from the worker",
            status_code=500,
        )
    if error is not None:
        raise ApiException(
            _snake(error.get("type", "walk_failed")) or "walk_failed",
            error.get("en") or "The 3D property could not be built from these videos.",
            error.get("tr") or "Bu videolardan 3B tur oluşturulamadı.",
            hint=(error.get("message") or "")[:400] or None,
            status_code=422,
        )
    if proc.returncode != 0 or result_path is None:
        log.error("walk.worker_crashed", returncode=proc.returncode, stderr=stderr[-3000:])
        raise ApiException(
            "walk_worker_crashed",
            "The 3D engine stopped unexpectedly while building this property.",
            "3B motor bu turu oluştururken beklenmedik şekilde durdu.",
            hint=stderr[-600:] or f"exit code {proc.returncode}",
            status_code=500,
        )
    return _orjson.loads(Path(result_path).read_bytes())


#: Swappable like STITCH_BACKEND; tests stand in a fake that returns a manifest dict.
WALK_BACKEND: WalkBackend = _default_walk_backend


# --------------------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------------------


class JobProgress:
    """:class:`biosphere.pano.types.ProgressReporter` bound to one job row.

    Writes are throttled: the pipeline may call ``stage()`` hundreds of times per second and
    every call would otherwise be a SQLite transaction plus an SSE frame.
    """

    def __init__(
        self, manager: JobManager, job_id: str, *, started_at: float, table: StageTable = STITCH_TABLE
    ) -> None:
        self._manager = manager
        self._job_id = job_id
        self._started = started_at
        self._table = table
        self._stage = table.stages[0]
        self._progress = 0.0
        self._last_write = 0.0
        self._lock = threading.Lock()

    # -- ProgressReporter protocol ------------------------------------------------------
    def stage(self, name: str, fraction: float, message: str = "") -> None:
        """Record pipeline progress; raises :class:`JobCancelled` if the user hit cancel.

        Cancellation is cooperative — this is the only place the pipeline yields control
        back to us, so it is where we check the flag.

        ``fraction`` is treated as progress *within* ``name``, which is the documented
        ``ProgressReporter`` contract. ``pipeline.StageWeightedProgress`` instead forwards an
        already-globalised fraction, and the mapping stays well-behaved either way: a global
        fraction still lands inside the current stage's band, so the bar remains monotonic
        and never leaves the segment the UI is highlighting.
        """
        if self._manager.is_cancel_requested(self._job_id):
            raise JobCancelled(self._job_id)

        table = self._table
        stage = name if name in table.weights else self._stage
        # Stages only ever move forward. A late report from an earlier stage (or our own
        # bookkeeping calls around the backend) must not drag the rail backwards.
        if table.stages.index(stage) < table.stages.index(self._stage):
            stage = self._stage
        fraction = float(min(max(fraction, 0.0), 1.0))
        overall = table.cumulative.get(stage, 0.0) + table.weights.get(stage, 0.0) * fraction

        with self._lock:
            # Progress must never go backwards; a stage that re-reports an earlier fraction
            # (retries, multi-pass blending) would otherwise make the bar stutter.
            overall = max(overall, self._progress)
            stage_changed = stage != self._stage
            now = time.monotonic()
            significant = overall - self._progress >= 0.005 or now - self._last_write >= 0.25
            self._stage = stage
            self._progress = overall
            if not (stage_changed or significant):
                return
            self._last_write = now

        en, tr = table.messages.get(stage, (stage.title(), stage.title()))
        self._manager.update_progress(
            self._job_id,
            stage=stage,
            progress=overall,
            stage_progress=fraction,
            message=message or en,
            message_tr=tr,
            eta_seconds=self._eta(overall),
        )

    def log(self, message: str, **fields: Any) -> None:
        level = str(fields.pop("level", "info"))
        log.info(message, job_id=self._job_id, **fields)
        self._manager.publish_log(self._job_id, level=level, message=message)

    # -- internals ----------------------------------------------------------------------
    def _eta(self, overall: float) -> float | None:
        """Linear extrapolation from elapsed time.

        Deliberately ``None`` for the first seconds: an ETA computed from 1% of the work is
        noise, and showing "4 hours remaining" for two seconds destroys trust in the number.
        """
        elapsed = time.monotonic() - self._started
        if overall < 0.04 or elapsed < 2.0:
            return None
        remaining = elapsed * (1.0 - overall) / max(overall, 1e-6)
        return round(float(min(remaining, 24 * 3600)), 1)


# --------------------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------------------


def _queued_message(session: Session) -> tuple[str, str]:
    """What a freshly queued job says while it waits: the worker runs one job at a time, so a
    0 % that lasts minutes is not a hang but the job in front of it, and the message says so."""
    ahead = len(session.exec(select(Job).where(Job.status.in_(("queued", "running")))).all())  # type: ignore[attr-defined]
    if ahead == 0:
        return "Queued for processing", "İşlenmek üzere sıraya alındı"
    if ahead == 1:
        return "Queued behind the job that is running now", "Sırada: önce çalışmakta olan iş bitecek"
    return f"Queued behind {ahead} jobs", f"Sırada: önünde {ahead} iş var"


class JobManager:
    """In-process job queue with SSE fan-out.

    Swap-out point for Celery: keep the public methods, replace the executor with a broker
    and :meth:`subscribe` with a Redis pub/sub consumer.
    """

    def __init__(self, settings: Settings | None = None, storage: LocalStorage | None = None) -> None:
        self._settings = settings or get_settings()
        self._storage = storage or get_storage()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(self._settings.worker_concurrency)),
            thread_name_prefix="stitch",
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._history: dict[str, list[dict[str, Any]]] = {}
        #: When each running job last moved. See `stalled_jobs`.
        self._last_tick: dict[str, float] = {}
        self._cancel_requested: set[str] = set()
        self._futures: dict[str, Future[None]] = {}
        self._configs: dict[str, PipelineConfig] = {}

    # -- lifecycle ----------------------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the serving event loop; must be called from lifespan startup."""
        self._loop = loop

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def reap_orphans(self) -> int:
        """Fail jobs left ``queued``/``running`` by a previous process.

        Without this a crash leaves scenes stuck in ``processing`` forever, and the dashboard
        shows a spinner that can never resolve.
        """
        payload = {
            "code": "worker_restarted",
            "message": "The processing service restarted while this job was running.",
            "message_tr": "Bu iş çalışırken işleme servisi yeniden başlatıldı.",
            "hint": "Start the scene again with Reprocess.",
        }
        n = 0
        with session_scope() as session:
            jobs = session.exec(select(Job).where(Job.status.in_(("queued", "running")))).all()  # type: ignore[attr-defined]
            for job in jobs:
                job.status = "failed"
                job.error = payload
                job.finished_at = utcnow()
                session.add(job)
                if job.kind == "walkthrough":
                    self._settle_walkthrough(session, job, status="failed", error=payload)
                else:
                    scene = session.get(Scene, job.scene_id) if job.scene_id else None
                    if scene is not None and scene.status == "processing":
                        scene.status = "failed"
                        scene.error = payload
                        scene.active_job_id = None
                        session.add(scene)
                n += 1
        return n

    def _settle_walkthrough(
        self, session: Session, job: Job, *, status: str, error: dict[str, Any] | None
    ) -> Walkthrough | None:
        """Leave the walkthrough row and its rooms in a terminal state after a failed/cancelled job."""
        walk = session.exec(select(Walkthrough).where(Walkthrough.property_id == job.property_id)).first()
        if walk is None:
            return None
        if walk.active_job_id == job.id or walk.status == "processing":
            walk.status = status if walk.manifest is None else "ready"
            walk.error = error
            walk.active_job_id = None
            walk.updated_at = utcnow()
            session.add(walk)
        for scene_id in list(walk.room_scene_ids or []):
            scene = session.get(Scene, scene_id)
            if scene is None or scene.active_job_id != job.id:
                continue
            scene.status = "ready" if (scene.panorama or (scene.walkthrough or {}).get("status") == "ready") else "failed"
            scene.active_job_id = None
            if error is not None and scene.status == "failed":
                scene.error = error
            session.add(scene)
        return walk

    # -- submission ---------------------------------------------------------------------
    def submit(self, scene_id: str, config: PipelineConfig | None = None) -> Job:
        """Create the job row for ``scene_id`` and hand it to the worker pool."""
        with session_scope() as session:
            scene = session.get(Scene, scene_id)
            if scene is None:
                raise ApiException(
                    "scene_not_found",
                    "That scene no longer exists.",
                    "Bu sahne artık mevcut değil.",
                    status_code=404,
                )
            if not scene.source_paths:
                raise ApiException(
                    "no_source_media",
                    "This scene has no source capture to process.",
                    "Bu sahnede işlenecek kaynak çekim yok.",
                    hint="Upload the video again.",
                    status_code=409,
                )
            message, message_tr = _queued_message(session)
            job = Job(
                property_id=scene.property_id,
                scene_id=scene.id,
                status="queued",
                stage="upload",
                progress=0.0,
                stage_progress=0.0,
                message=message,
                message_tr=message_tr,
                quality_mode=scene.quality_mode,
                capture_kind=scene.capture_kind,
            )
            session.add(job)
            scene.status = "processing"
            scene.error = None
            scene.active_job_id = job.id
            session.add(scene)
            session.flush()
            snapshot = _detach(job)

        if config is not None:
            self._configs[snapshot.id] = config
        self._history[snapshot.id] = []
        self._last_tick[snapshot.id] = time.monotonic()
        future = self._executor.submit(self._run, snapshot.id)
        with self._lock:
            self._futures[snapshot.id] = future
        self._publish(snapshot.id, {"type": "progress", "job": _job_payload(snapshot)})
        return snapshot

    def submit_walkthrough(self, walkthrough_id: str) -> Job:
        """Queue the property-level 3D build for ``walkthrough_id``."""
        with session_scope() as session:
            walk = session.get(Walkthrough, walkthrough_id)
            if walk is None:
                raise ApiException(
                    "walkthrough_not_found",
                    "That walkthrough no longer exists.",
                    "Bu gezinti artık mevcut değil.",
                    status_code=404,
                )
            if not walk.base_source_paths and not walk.room_scene_ids:
                raise ApiException(
                    "no_source_media",
                    "This walkthrough has no videos to process.",
                    "Bu gezintide işlenecek video yok.",
                    status_code=409,
                )
            if walk.active_job_id is not None and walk.status == "processing":
                raise ApiException(
                    "already_processing",
                    "This property's 3D walkthrough is already being built.",
                    "Bu evin 3B gezintisi zaten oluşturuluyor.",
                    hint="Cancel the running job first.",
                    status_code=409,
                )
            message, message_tr = _queued_message(session)
            job = Job(
                property_id=walk.property_id,
                scene_id=None,
                kind="walkthrough",
                status="queued",
                stage="upload",
                progress=0.0,
                stage_progress=0.0,
                message=message,
                message_tr=message_tr,
                quality_mode=walk.quality_mode,
                capture_kind="walk",
            )
            session.add(job)
            walk.status = "processing"
            walk.error = None
            walk.active_job_id = job.id
            walk.updated_at = utcnow()
            session.add(walk)
            for scene_id in list(walk.room_scene_ids or []):
                scene = session.get(Scene, scene_id)
                if scene is None:
                    continue
                scene.status = "processing"
                scene.active_job_id = job.id
                scene.walkthrough = {**(scene.walkthrough or {}), "status": "processing"}
                session.add(scene)
            session.flush()
            snapshot = _detach(job)

        self._history[snapshot.id] = []
        self._last_tick[snapshot.id] = time.monotonic()
        future = self._executor.submit(self._run, snapshot.id)
        with self._lock:
            self._futures[snapshot.id] = future
        self._publish(snapshot.id, {"type": "progress", "job": _job_payload(snapshot)})
        return snapshot

    def get(self, job_id: str) -> Job | None:
        with session_scope() as session:
            job = session.get(Job, job_id)
            return _detach(job) if job is not None else None

    def cancel(self, job_id: str) -> Job:
        """Request cancellation. Queued jobs die immediately; running ones at the next stage."""
        with self._lock:
            self._cancel_requested.add(job_id)
            future = self._futures.get(job_id)
        # If the worker never started, the future can be killed outright.
        started = future is None or not future.cancel()
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise ApiException(
                    "job_not_found",
                    "That job does not exist.",
                    "Bu iş bulunamadı.",
                    status_code=404,
                )
            if job.status in TERMINAL_STATUSES:
                return _detach(job)
            if not started or job.status == "queued":
                job.status = "cancelled"
                job.message = "Cancelled"
                job.message_tr = "İptal edildi"
                job.finished_at = utcnow()
                session.add(job)
                if job.kind == "walkthrough":
                    self._settle_walkthrough(session, job, status="failed", error=None)
                else:
                    scene = session.get(Scene, job.scene_id) if job.scene_id else None
                    if scene is not None:
                        scene.status = "failed" if scene.panorama is None else "ready"
                        scene.active_job_id = None
                        session.add(scene)
            session.flush()
            snapshot = _detach(job)
        if snapshot.status == "cancelled":
            self._publish(job_id, {"type": "progress", "job": _job_payload(snapshot)})
        return snapshot

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancel_requested

    # -- event fan-out ------------------------------------------------------------------
    async def subscribe(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield every :class:`JobEvent` for ``job_id``, replaying what was already emitted."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        with self._lock:
            backlog = list(self._history.get(job_id, ()))
            self._subscribers.setdefault(job_id, set()).add(queue)
        try:
            for event in backlog:
                yield event
                if is_terminal_event(event):
                    return
            while True:
                event = await queue.get()
                yield event
                if is_terminal_event(event):
                    return
        finally:
            with self._lock:
                subs = self._subscribers.get(job_id)
                if subs is not None:
                    subs.discard(queue)
                    if not subs:
                        self._subscribers.pop(job_id, None)

    def has_history(self, job_id: str) -> bool:
        """Whether any event for this job is still buffered in this process."""
        with self._lock:
            return bool(self._history.get(job_id))

    def terminal_event(self, job_id: str) -> dict[str, Any] | None:
        """Rebuild the final event from the database.

        Needed when a client subscribes to a job that finished in a *previous* process (or
        before the in-memory history was created): the row survives, the buffer does not, and
        the stream must still terminate instead of hanging on an open connection.
        """
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None or job.status not in TERMINAL_STATUSES:
                return None
            job_snapshot = _detach(job)
            scene = session.get(Scene, job.scene_id) if job.scene_id else None
            scene_snapshot = _detach(scene) if scene is not None else None
            hotspots = _scene_hotspots(session, job.scene_id) if scene is not None else []
            photos = _scene_photos(session, scene) if scene is not None else []
            walk_snapshot = None
            if job.kind == "walkthrough":
                walk = session.exec(select(Walkthrough).where(Walkthrough.property_id == job.property_id)).first()
                walk_snapshot = _detach(walk) if walk is not None else None

        if job_snapshot.status == "succeeded" and job_snapshot.kind == "walkthrough":
            return {
                "type": "done",
                "job": _job_payload(job_snapshot),
                "scene": None,
                "walkthrough": _walk_row_payload(walk_snapshot) if walk_snapshot is not None else None,
            }
        if job_snapshot.status == "succeeded" and scene_snapshot is not None:
            return {
                "type": "done",
                "job": _job_payload(job_snapshot),
                "scene": _scene_payload(scene_snapshot, hotspots, photos),
            }
        if job_snapshot.status == "failed":
            return {
                "type": "error",
                "job": _job_payload(job_snapshot),
                "error": job_snapshot.error
                or {
                    "code": "unknown_error",
                    "message": "The job failed.",
                    "message_tr": "İşlem başarısız oldu.",
                    "hint": None,
                },
            }
        return {"type": "progress", "job": _job_payload(job_snapshot)}

    def _publish(self, job_id: str, event: dict[str, Any]) -> None:
        """Push an event to every subscriber. Safe to call from the worker thread."""
        with self._lock:
            history = self._history.setdefault(job_id, [])
            history.append(event)
            if len(history) > MAX_HISTORY:
                # Keep the first event (queued) and drop from the middle: the tail carries the
                # terminal event and must survive.
                del history[1 : len(history) - MAX_HISTORY + 1]
            queues = list(self._subscribers.get(job_id, ()))
            loop = self._loop
        if loop is None or not queues:
            return
        for queue in queues:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:  # pragma: no cover - loop already closed during shutdown
                pass

    def publish_log(self, job_id: str, *, level: str, message: str) -> None:
        self._publish(
            job_id,
            {
                "type": "log",
                "level": level if level in {"info", "warn", "error"} else "info",
                "message": message,
                "at": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            },
        )

    def update_progress(
        self,
        job_id: str,
        *,
        stage: str,
        progress: float,
        stage_progress: float,
        message: str,
        message_tr: str,
        eta_seconds: float | None,
    ) -> None:
        """Persist a progress tick and broadcast it."""
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:  # pragma: no cover - job deleted mid-run
                return
            job.stage = stage
            job.progress = round(float(progress), 5)
            job.stage_progress = round(float(stage_progress), 5)
            job.message = message
            job.message_tr = message_tr
            job.eta_seconds = eta_seconds
            if job.status == "queued":
                job.status = "running"
                job.started_at = job.started_at or utcnow()
            session.add(job)
            session.flush()
            snapshot = _detach(job)
        self._last_tick[job_id] = time.monotonic()
        self._publish(job_id, {"type": "progress", "job": _job_payload(snapshot)})

    def stalled_jobs(self, *, quiet_seconds: float) -> list[tuple[str, float]]:
        """Running jobs that have not reported progress for `quiet_seconds`.

        WHAT THIS IS FOR, AND WHY IT ONLY WORKS NOW. `worker_concurrency` is 1, so a job that
        hangs does not cost one capture — it costs every capture after it, queued behind
        something that will never finish, until somebody restarts the engine. A stuck job is
        therefore the most expensive failure this service has, and until now nothing noticed
        one.

        It could not have noticed before: bundle adjustment reported no progress at all and
        ran for thirteen minutes on a dense capture, so any quiet-period test would have
        called healthy work stuck. Now that it ticks, silence means silence.

        Reporting only — a Python thread cannot be killed from outside, so this names the
        problem for the log and for an operator rather than pretending to solve it.
        """
        now = time.monotonic()
        out: list[tuple[str, float]] = []
        with self._lock:
            running = [jid for jid, fut in self._futures.items() if not fut.done()]
        for job_id in running:
            last = self._last_tick.get(job_id)
            if last is not None and now - last >= quiet_seconds:
                out.append((job_id, now - last))
        return out

    # -- the worker ---------------------------------------------------------------------
    def _run(self, job_id: str) -> None:
        """Entry point on the worker thread. Never raises."""
        bind_context(job_id=job_id)
        started = time.monotonic()
        storage = self._storage
        work_dir = storage.work_dir(job_id)
        reporter = JobProgress(self, job_id, started_at=started)

        try:
            with session_scope() as session:
                job = session.get(Job, job_id)
                if job is None:
                    return
                kind = job.kind
                scene = session.get(Scene, job.scene_id) if job.scene_id else None
                if kind != "walkthrough" and scene is None:
                    return
                job.status = "running"
                job.started_at = utcnow()
                table = WALK_TABLE if kind == "walkthrough" else STITCH_TABLE
                job.message, job.message_tr = table.messages[table.stages[1]]
                session.add(job)
                session.flush()
                job_snapshot = _detach(job)
                scene_snapshot = _detach(scene) if scene is not None else None
            self._publish(job_id, {"type": "progress", "job": _job_payload(job_snapshot)})

            if self.is_cancel_requested(job_id):
                raise JobCancelled(job_id)

            if kind == "walkthrough":
                reporter = JobProgress(self, job_id, started_at=started, table=WALK_TABLE)
                self._run_walkthrough(job_id, job_snapshot, reporter, work_dir)
                return

            sources = [Path(p) for p in scene_snapshot.source_paths]
            missing = [p for p in sources if not p.exists()]
            if missing:
                raise ApiException(
                    "source_missing",
                    "The uploaded capture is no longer on disk.",
                    "Yüklenen çekim diskte bulunamıyor.",
                    hint="Upload the video again.",
                    status_code=410,
                )

            config = self._configs.pop(job_id, None) or PipelineConfig.for_mode(
                QualityMode(job_snapshot.quality_mode)
            )
            source: Path | list[Path] = sources[0] if len(sources) == 1 else sources

            reporter.stage("probe", 0.2)
            result = STITCH_BACKEND(source, config, work_dir=work_dir, progress=reporter)
            if self.is_cancel_requested(job_id):
                raise JobCancelled(job_id)

            reporter.stage("levels", 0.1)
            payload = self._persist_outputs(scene_snapshot, result, config)
            self._finish_success(job_id, scene_snapshot.id, payload)

        except JobCancelled:
            self._finish_cancelled(job_id)
        except ApiException as exc:
            log.warning("job.failed", job_id=job_id, code=exc.code, message=exc.message)
            self._finish_failed(job_id, exc.payload())
        except StitchError as exc:
            log.warning("job.stitch_error", job_id=job_id, error=type(exc).__name__)
            self._finish_failed(
                job_id,
                {
                    "code": _snake(type(exc).__name__),
                    "message": getattr(exc, "user_message_en", str(exc)),
                    "message_tr": getattr(exc, "user_message_tr", "Panorama oluşturulamadı."),
                    "hint": str(exc) or None,
                },
            )
        except Exception:  # noqa: BLE001 - last line of defence, must not leak internals
            log.error("job.crashed", job_id=job_id, traceback=traceback.format_exc())
            self._finish_failed(
                job_id,
                {
                    "code": "internal_error",
                    "message": "Something went wrong while processing this capture.",
                    "message_tr": "Bu çekim işlenirken beklenmeyen bir hata oluştu.",
                    "hint": "Try again, or re-shoot the room if the problem persists.",
                },
            )
        finally:
            with self._lock:
                self._futures.pop(job_id, None)
                self._cancel_requested.discard(job_id)
            storage.cleanup_work(job_id)
            clear_context()

    # -- the walkthrough kind ----------------------------------------------------------------
    def _run_walkthrough(self, job_id: str, job: Job, reporter: JobProgress, work_dir: Path) -> None:
        """Build the property's 3D areas; raises like the stitch path so ``_run`` translates errors."""
        from biosphere.splat.types import AreaInput, SplatConfig, SplatQuality

        with session_scope() as session:
            walk = session.exec(select(Walkthrough).where(Walkthrough.property_id == job.property_id)).first()
            if walk is None:
                raise ApiException(
                    "walkthrough_not_found", "The walkthrough was deleted.", "Gezinti silinmiş.", status_code=410
                )
            walk_id = walk.id
            base_paths = [Path(p) for p in (walk.base_source_paths or [])]
            rooms: list[AreaInput] = []
            for scene_id in list(walk.room_scene_ids or []):
                scene = session.get(Scene, scene_id)
                if scene is None or not scene.walk_source_paths:
                    continue
                rooms.append(AreaInput(key=scene.id, name=scene.name, videos=tuple(Path(p) for p in scene.walk_source_paths)))
            quality_mode = walk.quality_mode

        for path in base_paths + [p for r in rooms for p in r.videos]:
            if not path.exists():
                raise ApiException(
                    "source_missing",
                    "One of the uploaded videos is no longer on disk.",
                    "Yüklenen videolardan biri artık diskte değil.",
                    hint="Upload the videos again.",
                    status_code=410,
                )
        out_dir = self._storage.walk_dir(job.property_id)
        # A rebuild replaces the previous delivery wholesale; half of two properties is no property.
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "base": {"key": "base", "name": "Genel gezinti", "videos": [str(p) for p in base_paths]} if base_paths else None,
            "rooms": [
                {"key": r.key, "name": r.name, "videos": [str(v) for v in r.videos], "floor_index": r.floor_index}
                for r in rooms
            ],
            "quality": quality_mode,
            "keep_work": bool(self._settings.keep_work_dirs),
            "work_dir": str(work_dir),
            "out_dir": str(out_dir),
            # The worker reads its model locations from the environment, and settings loaded
            # from a .env file never reach a child's environment on their own.
            "env": {
                k: v
                for k, v in {
                    "BIOSPHERE_AI_MODELS_DIR": str(self._settings.ai_models_dir) if self._settings.ai_models_dir else "",
                    "BIOSPHERE_AI_ALLOW_DOWNLOADS": "true" if self._settings.ai_allow_downloads else "",
                }.items()
                if v
            },
        }
        _ = (AreaInput, SplatConfig, SplatQuality)  # the spec mirrors these; the worker builds them

        reporter.stage("upload", 1.0)
        doc = WALK_BACKEND(spec, progress=reporter, cancel_check=lambda: self.is_cancel_requested(job_id))
        if self.is_cancel_requested(job_id):
            raise JobCancelled(job_id)
        payload = self._walk_payload(doc, out_dir)
        self._finish_walk_success(job_id, walk_id, payload)

    def _walk_payload(self, doc: dict[str, Any], out_dir: Path) -> dict[str, Any]:
        """The manifest with media URLs in place of the on-disk relative paths."""
        storage = self._storage
        doc = dict(doc)

        def url(rel: str | None) -> str | None:
            if not rel:
                return None
            return storage.url_for(out_dir / rel)

        for area in [doc["base"], *doc["rooms"]]:
            for key in ("url", "thumbnail_url", "env_pano_url"):
                area[key] = url(area.get(key))
        if doc.get("floor_plan"):
            doc["floor_plan"]["url"] = url(doc["floor_plan"].get("url"))
        return {"manifest": doc, "metrics": doc.get("metrics") or {}}

    def _finish_walk_success(self, job_id: str, walk_id: str, payload: dict[str, Any]) -> None:
        manifest = payload["manifest"]
        by_key = {a["key"]: a for a in [manifest["base"], *manifest["rooms"]]}
        with session_scope() as session:
            job = session.get(Job, job_id)
            walk = session.get(Walkthrough, walk_id)
            if job is None or walk is None:  # pragma: no cover
                return
            walk.status = "ready"
            walk.manifest = manifest
            walk.metrics = payload["metrics"]
            walk.error = None
            walk.active_job_id = None
            walk.updated_at = utcnow()
            session.add(walk)
            for scene_id in list(walk.room_scene_ids or []):
                scene = session.get(Scene, scene_id)
                if scene is None:
                    continue
                area = by_key.get(scene_id)
                if area is not None and area.get("url") and not area.get("failed"):
                    scene.walkthrough = {
                        "status": "ready",
                        "url": area["url"],
                        "thumbnail_url": area.get("thumbnail_url"),
                        "splat_count": int(area.get("splat_count") or 0),
                        "bytes": int(area.get("bytes") or 0),
                        "placed": bool(area.get("placed")),
                        "error": None,
                    }
                    scene.status = "ready"
                    scene.error = None
                else:
                    failure = (area or {}).get("failed") or "not delivered"
                    scene.walkthrough = {
                        "status": "failed", "url": None, "thumbnail_url": None, "splat_count": 0, "bytes": 0,
                        "placed": False, "error": str(failure)[:300],
                    }
                    scene.status = "ready" if scene.panorama else "failed"
                if scene.active_job_id == job_id:
                    scene.active_job_id = None
                session.add(scene)

            job.status = "succeeded"
            job.stage = "done"
            job.progress = 1.0
            job.stage_progress = 1.0
            job.message, job.message_tr = WALK_STAGE_MESSAGES["done"]
            job.eta_seconds = 0.0
            job.metrics = {k: v for k, v in payload["metrics"].items() if k != "areas"}
            job.error = None
            job.finished_at = utcnow()
            session.add(job)
            property = session.get(Property, job.property_id)
            if property is not None:
                property.updated_at = utcnow()
                session.add(property)
            session.flush()
            job_snapshot = _detach(job)
            walk_snapshot = _detach(walk)

        self._publish(
            job_id,
            {
                "type": "done",
                "job": _job_payload(job_snapshot),
                "scene": None,
                "walkthrough": _walk_row_payload(walk_snapshot),
            },
        )

    # -- output persistence -------------------------------------------------------------
    def _persist_outputs(
        self, scene: Scene, result: StitchResult, config: PipelineConfig
    ) -> dict[str, Any]:
        """Write every delivered artefact into the scene directory and describe it as JSON."""
        storage = self._storage
        out = storage.scene_dir(scene.property_id, scene.id)
        pano = np.ascontiguousarray(result.panorama)
        if pano.ndim == 2:
            pano = cv2.cvtColor(pano, cv2.COLOR_GRAY2BGR)
        height, width = int(pano.shape[0]), int(pano.shape[1])
        quality = int(getattr(config, "jpeg_quality", 92))

        level_urls: dict[str, str | None] = {"uhd_url": None, "hd_url": None, "sd_url": None}
        for name in ("uhd", "hd", "sd"):
            target_w = min(LEVEL_TARGET_WIDTHS[name], width)
            image = _resize_width(pano, target_w)
            path = write_jpeg(out / f"pano_{name}.jpg", image, quality)
            level_urls[f"{name}_url"] = storage.url_for(path)

        preview_w = min(LEVEL_TARGET_WIDTHS["preview"], width)
        preview = _resize_width(pano, preview_w)
        preview = cv2.GaussianBlur(preview, (0, 0), max(preview_w / 256.0, 0.8))
        preview_path = write_jpeg(out / "pano_preview.jpg", preview, 70)
        write_webp(out / "pano_preview.webp", preview, 65)

        cube_urls: dict[str, str] | None = None
        if result.cube_faces:
            cube_dir = out / "cube"
            cube_dir.mkdir(parents=True, exist_ok=True)
            faces: dict[str, str] = {}
            for face in CUBE_FACES:
                src = result.cube_faces.get(face)
                if src is None:
                    continue
                image = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                faces[face] = storage.url_for(write_jpeg(cube_dir / f"{face}.jpg", image, quality))
            cube_urls = faces if len(faces) == len(CUBE_FACES) else None

        thumb = result.thumbnail
        if thumb is None:
            thumb = _rectilinear_view(pano, scene.initial_view or {})
        write_jpeg(out / "thumb.jpg", thumb, 82)

        coverage_payload = _coverage_payload(result.coverage, pano, out, storage)

        metrics_payload = (
            result.metrics.as_dict() if hasattr(result.metrics, "as_dict") else dict(result.metrics or {})
        )
        (out / "metrics.json").write_bytes(
            orjson.dumps(metrics_payload, option=orjson.OPT_INDENT_2, default=str)
        )

        return {
            "panorama": {
                # The preview points at the JPEG rather than the WebP written next to it:
                # every browser can decode it, and the file is a few kilobytes either way.
                "preview_url": storage.url_for(preview_path),
                "sd_url": level_urls["sd_url"],
                "hd_url": level_urls["hd_url"],
                "uhd_url": level_urls["uhd_url"],
                "width": width,
                "height": height,
                "cube_urls": cube_urls,
            },
            "coverage": coverage_payload,
            "metrics": metrics_payload,
        }

    # -- terminal transitions -----------------------------------------------------------
    def _finish_success(self, job_id: str, scene_id: str, payload: dict[str, Any]) -> None:
        with session_scope() as session:
            job = session.get(Job, job_id)
            scene = session.get(Scene, scene_id)
            if job is None or scene is None:  # pragma: no cover
                return
            scene.status = "ready"
            scene.panorama = payload["panorama"]
            scene.coverage = payload["coverage"]
            scene.metrics = payload["metrics"]
            scene.error = None
            scene.active_job_id = None
            session.add(scene)

            job.status = "succeeded"
            job.stage = "done"
            job.progress = 1.0
            job.stage_progress = 1.0
            job.message, job.message_tr = STAGE_MESSAGES["done"]
            job.eta_seconds = 0.0
            job.metrics = payload["metrics"]
            job.error = None
            job.finished_at = utcnow()
            session.add(job)

            property = session.get(Property, job.property_id)
            if property is not None:
                property.updated_at = utcnow()
                if property.cover_scene_id is None:
                    property.cover_scene_id = scene.id
                session.add(property)

            session.flush()
            job_snapshot = _detach(job)
            scene_snapshot = _detach(scene)
            hotspots = _scene_hotspots(session, scene.id)
            photos = _scene_photos(session, scene)

        self._publish(
            job_id,
            {
                "type": "done",
                "job": _job_payload(job_snapshot),
                "scene": _scene_payload(scene_snapshot, hotspots, photos),
            },
        )

    def _finish_failed(self, job_id: str, error: dict[str, Any]) -> None:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:  # pragma: no cover
                return
            job.status = "failed"
            job.error = error
            job.message = error["message"]
            job.message_tr = error["message_tr"]
            job.eta_seconds = None
            job.finished_at = utcnow()
            session.add(job)
            if job.kind == "walkthrough":
                self._settle_walkthrough(session, job, status="failed", error=error)
            else:
                scene = session.get(Scene, job.scene_id) if job.scene_id else None
                if scene is not None:
                    scene.status = "ready" if scene.panorama else "failed"
                    scene.error = error
                    scene.active_job_id = None
                    session.add(scene)
            session.flush()
            snapshot = _detach(job)
        self._publish(
            job_id, {"type": "error", "job": _job_payload(snapshot), "error": error}
        )

    def _finish_cancelled(self, job_id: str) -> None:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:  # pragma: no cover
                return
            job.status = "cancelled"
            job.message = "Cancelled"
            job.message_tr = "İptal edildi"
            job.eta_seconds = None
            job.finished_at = utcnow()
            session.add(job)
            if job.kind == "walkthrough":
                self._settle_walkthrough(session, job, status="failed", error=None)
            else:
                scene = session.get(Scene, job.scene_id) if job.scene_id else None
                if scene is not None:
                    scene.status = "ready" if scene.panorama else "failed"
                    scene.active_job_id = None
                    session.add(scene)
            session.flush()
            snapshot = _detach(job)
        self._publish(job_id, {"type": "progress", "job": _job_payload(snapshot)})


# --------------------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------------------


def _snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i:
            out.append("_")
        out.append(ch.lower())
    return "".join(out).removesuffix("_error")


def _detach(instance: Any) -> Any:
    """Copy a loaded row into a plain, session-free instance.

    The worker and the manager hand rows back to callers after the session closes; without
    a copy every attribute access would raise ``DetachedInstanceError``.
    """
    return type(instance)(**instance.model_dump())


def is_terminal_event(event: dict[str, Any]) -> bool:
    """True for the last event a subscriber should ever receive for a job."""
    if event.get("type") in {"done", "error"}:
        return True
    job = event.get("job")
    return bool(isinstance(job, dict) and job.get("status") in TERMINAL_STATUSES)


def _job_payload(job: Job) -> dict[str, Any]:
    from biosphere.api.schemas import job_to_schema  # local import keeps core import-light

    return job_to_schema(job).model_dump(mode="json")


def _walk_row_payload(walk: Walkthrough) -> dict[str, Any]:
    from biosphere.api.schemas import walkthrough_to_schema

    return walkthrough_to_schema(walk).model_dump(mode="json")


def _scene_payload(scene: Scene, hotspots: list[Any], photos: list[Any]) -> dict[str, Any]:
    from biosphere.api.schemas import scene_to_schema

    return scene_to_schema(scene, hotspots, photos).model_dump(mode="json")


def _scene_photos(session: Session, scene: Scene) -> list[Any]:
    """A room's photographs, as schemas, for the SSE payload.

    Resolved here rather than left out, because ``schemas`` states the invariant this upholds:
    the event payload and the REST response are produced by one converter so they cannot drift.
    A scene that carried photographs over HTTP and dropped them over SSE would make them
    disappear from the UI the moment its stitch finished.
    """
    from biosphere.api.deps import photos_of_scene
    from biosphere.api.schemas import scene_photo_to_schema

    storage = get_storage()
    directory = storage.scene_photos_dir(scene.property_id, scene.id)
    return [
        scene_photo_to_schema(row, storage.url_for(directory / row.filename))
        for row in photos_of_scene(session, scene.id)
    ]


def _scene_hotspots(session: Session, scene_id: str) -> list[Any]:
    from biosphere.core.models import Hotspot

    rows = session.exec(select(Hotspot).where(Hotspot.scene_id == scene_id)).all()
    return [_detach(row) for row in rows]


def _resize_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize preserving the 2:1 equirect aspect, using the right kernel for the direction."""
    if width == image.shape[1]:
        return image
    height = max(1, int(round(width * image.shape[0] / image.shape[1])))
    interp = cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interp)


def _rectilinear_view(pano: np.ndarray, initial_view: dict[str, Any]) -> np.ndarray:
    """Render a real perspective thumbnail from the scene's initial view.

    A centre crop of the equirect would look badly stretched; sampling through the shared
    geometry helpers gives the same image the player shows on load.
    """
    width, height = 640, 360
    yaw = float(initial_view.get("yaw", 0.0) or 0.0)
    pitch = float(initial_view.get("pitch", 0.0) or 0.0)
    fov = float(initial_view.get("fov", 75.0) or 75.0)
    focal = (height / 2.0) / math.tan(math.radians(max(min(fov, 140.0), 20.0)) / 2.0)
    K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    rays_cam = geo.camera_rays(width, height, K)
    R = geo.from_yaw_pitch_roll(yaw, pitch, 0.0)
    rays_world = rays_cam.reshape(-1, 3) @ R.astype(np.float32)
    u, v = geo.equirect_from_rays(rays_world.reshape(height, width, 3), pano.shape[1], pano.shape[0])
    return geo.sample_equirect(pano, u, v, interpolation=cv2.INTER_LINEAR)


def _coverage_payload(
    coverage: CoverageReport | None,
    pano: np.ndarray,
    out_dir: Path,
    storage: LocalStorage,
) -> dict[str, Any] | None:
    """Serialise the coverage report and render the "re-shoot this" heat map."""
    if coverage is None:
        return None

    map_url: str | None = None
    mask = getattr(coverage, "mask", None)
    if isinstance(mask, np.ndarray) and mask.size:
        map_w = 1024
        base = _resize_width(pano, map_w)
        m = cv2.resize(
            mask.astype(np.uint8), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        covered = (m > 127).astype(np.uint8)
        grey = cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        overlay = np.zeros_like(grey)
        overlay[..., 1] = 190  # green where the sweep reached
        overlay[..., 2] = 0
        miss = np.zeros_like(grey)
        miss[..., 2] = 220  # red where the user must re-shoot
        tint = np.where(covered[..., None].astype(bool), overlay, miss)
        heat = cv2.addWeighted(grey, 0.55, tint, 0.45, 0)
        map_url = storage.url_for(write_png(out_dir / "coverage.png", heat))

    gaps = []
    for gap in getattr(coverage, "gaps", []) or []:
        y0, y1, p0, p1 = (float(v) for v in gap)
        gaps.append(
            {
                "yaw_start_deg": y0,
                "yaw_end_deg": y1,
                "pitch_start_deg": p0,
                "pitch_end_deg": p1,
            }
        )

    return {
        "covered_fraction": float(coverage.covered_fraction),
        "horizon_fraction": float(coverage.horizon_fraction),
        "yaw_span_deg": float(coverage.yaw_span_deg),
        "pitch_min_deg": float(coverage.pitch_min_deg),
        "pitch_max_deg": float(coverage.pitch_max_deg),
        "zenith_covered": bool(coverage.zenith_covered),
        "nadir_covered": bool(coverage.nadir_covered),
        "gaps": gaps,
        "map_url": map_url,
    }


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    """Process-wide manager. Created lazily so tests can repoint settings first."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager


def reset_job_manager() -> None:
    """Tear the manager down (tests, and the CLI between runs)."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.shutdown(wait=False)
        _manager = None
