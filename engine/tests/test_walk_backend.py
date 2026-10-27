"""The walkthrough worker plumbing (``jobs._default_walk_backend``) against misbehaving workers.

The worker is a child process talking JSON over stdout. Two failure modes cost real runs:
a child that fills its stderr pipe (COLMAP logs) deadlocks unless something drains it, and a
child that stops talking sits at "running" forever unless something notices. Both are
driven here with stand-in worker modules selected through ``BIOSPHERE_WALK_WORKER_MODULE``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from biosphere.core import jobs as jobs_module
from biosphere.core.jobs import _default_walk_backend

FAKE_WORKERS = Path(__file__).parent / "fake_walk_workers"


@pytest.fixture(autouse=True)
def _fake_workers_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # The child is ``python -m <module>``; it inherits PYTHONPATH, not sys.path.
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(FAKE_WORKERS) + (os.pathsep + existing if existing else ""))


class _Progress:
    def __init__(self) -> None:
        self.stages: list[tuple[str, float, str]] = []

    def stage(self, name: str, fraction: float, message: str = "") -> None:
        self.stages.append((name, fraction, message))


def _spec(tmp_path: Path) -> dict:
    return {"work_dir": str(tmp_path / "work"), "out_dir": str(tmp_path / "out")}


def test_a_worker_that_floods_stderr_still_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOSPHERE_WALK_WORKER_MODULE", "flooding_worker")
    monkeypatch.setenv("BIOSPHERE_WALK_STALL_SECONDS", "120")
    progress = _Progress()
    result = _default_walk_backend(_spec(tmp_path), progress=progress)
    assert result == {"manifest": "ok"}
    assert [s[0] for s in progress.stages] == ["extract", "train"]
    # The flood landed in the log file rather than in a blocked pipe.
    stderr_log = tmp_path / "work" / "worker.stderr.log"
    assert stderr_log.exists() and stderr_log.stat().st_size > 2_000_000


def test_a_silent_worker_is_stopped_with_a_stall_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOSPHERE_WALK_WORKER_MODULE", "stalling_worker")
    monkeypatch.setenv("BIOSPHERE_WALK_STALL_SECONDS", "3")
    with pytest.raises(jobs_module.ApiException) as info:
        _default_walk_backend(_spec(tmp_path), progress=_Progress())
    assert info.value.code == "walk_worker_stalled"


def test_python_on_path_can_import_the_fakes() -> None:
    # Guards the fixture itself: a wrong path would make both tests above fail confusingly.
    assert (FAKE_WORKERS / "flooding_worker.py").exists()
    assert (FAKE_WORKERS / "stalling_worker.py").exists()
    assert sys.executable
    assert json.dumps({"stage": "x"})
