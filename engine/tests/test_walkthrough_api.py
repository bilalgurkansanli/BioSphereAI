"""The walkthrough upload -> job -> manifest life cycle, with the 3D engine faked.

The real engine takes an hour on a GPU; what these tests guard is everything around it:
that the videos land where the worker expects them, that rooms become scenes, that the job
runs under its own stage rail, that the manifest's relative paths become served URLs, that
the public route exposes it, and that a rebuild and a delete leave the rows consistent.
"""

# ruff: noqa: F811 - pytest fixtures are imported by name and then used as parameters
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from biosphere.core.config import Settings

from .test_api import (  # noqa: F401 - fixtures
    API,
    JOB_TIMEOUT_S,
    _signed_in,
    client,
    settings,
    video_factory,
)

FAKE_SOG = b"PK\x03\x04fake-sog"


def fake_walk_backend(spec: dict[str, Any], *, progress=None, cancel_check=None) -> dict[str, Any]:
    """Writes the files a real build writes and returns the manifest with relative paths."""
    out = Path(spec["out_dir"])
    for sub in ("areas", "thumbs", "env"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    keys = (["base"] if spec.get("base") else []) + [r["key"] for r in spec["rooms"]]
    for stage in ("extract", "features", "sfm", "register", "train", "clean", "export"):
        if progress is not None:
            progress.stage(stage, 1.0, f"fake {stage}")
    areas = {}
    for k in keys:
        (out / "areas" / f"{k}.sog").write_bytes(FAKE_SOG)
        (out / "thumbs" / f"{k}.jpg").write_bytes(b"\xff\xd8fake")
        (out / "env" / f"{k}.jpg").write_bytes(b"\xff\xd8fake")
        areas[k] = {
            "key": k, "name": k, "floor_index": 0, "url": f"areas/{k}.sog", "bytes": len(FAKE_SOG), "splat_count": 1000,
            "thumbnail_url": f"thumbs/{k}.jpg", "env_pano_url": f"env/{k}.jpg", "cameras": [[0, -1.4, 0]], "centroid": [0, -1.4, 0],
            "entry": {"position": [0, -1.4, 0], "forward": [0, 0, 1]}, "bbox": {"min": [-2, -2.7, -2], "max": [2, 0.3, 2]},
            "placed": True, "warnings": [], "failed": None,
        }
    (out / "plan.jpg").write_bytes(b"\xff\xd8fake")
    base_key = keys[0]
    doc = {
        "version": 1,
        "frame": {"convention": "colmap", "up": [0, -1, 0], "floor_y": 0.0, "units": "m"},
        "base": areas[base_key],
        "rooms": [areas[k] for k in keys[1:]],
        "floor_plan": {"url": "plan.jpg", "width": 100, "height": 80, "x0": 0.0, "z0": 0.0, "px_per_m": 40.0, "hotspots": [{"key": k, "name": k, "x_pct": 0.5, "y_pct": 0.5, "floor_index": 0} for k in keys[1:]]},
        "metrics": {"seconds_total": 1.0, "areas": {}},
        "warnings": [],
    }
    (out / "tour.json").write_text(json.dumps(doc), encoding="utf-8")
    return doc


@pytest.fixture()
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("biosphere.core.jobs.WALK_BACKEND", fake_walk_backend)


def _wait(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + JOB_TIMEOUT_S
    while time.monotonic() < deadline:
        job = client.get(f"{API}/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _upload(client: TestClient, property_id: str, video: Path, *, rooms: int = 2, with_base: bool = True, quality: str = "fast"):
    files = []
    if with_base:
        files.append(("files", ("general.mp4", video.read_bytes(), "video/mp4")))
    for i in range(rooms):
        files.append(("files", (f"room{i}.mp4", video.read_bytes(), "video/mp4")))
    spec = {
        "quality_mode": quality,
        "base_filenames": ["general.mp4"] if with_base else [],
        "rooms": [{"name": f"Oda {i + 1}", "filenames": [f"room{i}.mp4"]} for i in range(rooms)],
    }
    return client.post(f"{API}/properties/{property_id}/walkthrough", data={"spec": json.dumps(spec)}, files=files)


def test_walkthrough_upload_builds_a_property(client: TestClient, settings: Settings, video_factory, fake_engine) -> None:
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    resp = _upload(client, property["id"], video_factory(6.0))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    walk, job = body["walkthrough"], body["job"]
    assert walk["status"] == "processing" and len(walk["room_scene_ids"]) == 2
    assert job["kind"] == "walkthrough" and job["scene_id"] is None and job["stage"] == "upload"

    # The videos are on disk where the worker looks for them.
    base_dir = settings.uploads_dir / f"walk_{walk['id']}"
    assert sorted(p.name for p in base_dir.iterdir()) == ["source_00.mp4"]
    for scene_id in walk["room_scene_ids"]:
        assert (settings.uploads_dir / scene_id / "walk_00.mp4").exists()

    done = _wait(client, job["id"])
    assert done["status"] == "succeeded" and done["stage"] == "done" and done["progress"] == 1.0

    detail = client.get(f"{API}/properties/{property['id']}").json()
    assert detail["walkthrough"]["status"] == "ready"
    manifest = detail["walkthrough"]["manifest"]
    assert manifest["base"]["url"].startswith("/media/properties/") and manifest["base"]["url"].endswith("base.sog?v=" + manifest["base"]["url"].split("?v=")[-1])
    assert [r["key"] for r in manifest["rooms"]] == walk["room_scene_ids"]
    assert manifest["floor_plan"]["url"].startswith("/media/properties/")
    # Rooms are scenes of kind "walk" carrying their area.
    rooms = [s for s in detail["scenes"] if s["capture_kind"] == "walk"]
    assert len(rooms) == 2
    for s in rooms:
        assert s["status"] == "ready" and s["walkthrough"]["status"] == "ready" and s["walkthrough"]["placed"] is True
        assert s["walkthrough"]["url"].endswith(f"{s['id']}.sog?v=" + s["walkthrough"]["url"].split("?v=")[-1])
    # And the file is actually served, immutable.
    served = client.get(manifest["base"]["url"])
    assert served.status_code == 200 and served.content == FAKE_SOG
    assert "immutable" in served.headers.get("cache-control", "")

    # SSE replay of a finished walkthrough job ends with a done event carrying the walkthrough.
    with client.stream("GET", f"{API}/jobs/{job['id']}/events") as stream:
        frames = [line for line in stream.iter_lines() if line.startswith("data:")]
    last = json.loads(frames[-1][len("data:"):])
    assert last["type"] == "done" and last["scene"] is None and last["walkthrough"]["status"] == "ready"

    # Public read carries it too — once the property is on air, which is what the embed
    # link now requires.
    client.patch(f"{API}/properties/{property['id']}", json={"published": True})
    public = client.get(f"{API}/public/properties/{property['public_token']}").json()
    assert public["walkthrough"]["manifest"]["base"]["splat_count"] == 1000
    only = client.get(f"{API}/public/properties/{property['public_token']}/walkthrough")
    assert only.status_code == 200 and only.json()["id"] == walk["id"]


def test_rebuild_and_delete(client: TestClient, settings: Settings, video_factory, fake_engine) -> None:
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    first = _upload(client, property["id"], video_factory(6.0), rooms=1).json()
    _wait(client, first["job"]["id"])
    again = client.post(f"{API}/properties/{property['id']}/walkthrough/rebuild", json={"quality_mode": "balanced"})
    assert again.status_code == 202, again.text
    job = _wait(client, again.json()["id"])
    assert job["status"] == "succeeded"
    walk = client.get(f"{API}/properties/{property['id']}/walkthrough").json()
    assert walk["quality_mode"] == "balanced" and walk["status"] == "ready"

    gone = client.delete(f"{API}/properties/{property['id']}/walkthrough")
    assert gone.status_code == 204
    assert client.get(f"{API}/properties/{property['id']}/walkthrough").status_code == 404
    assert not (settings.properties_dir / property["id"] / "walk").exists()
    detail = client.get(f"{API}/properties/{property['id']}").json()
    assert detail["walkthrough"] is None
    assert all(s["walkthrough"] is None for s in detail["scenes"])


def test_two_rooms_need_a_general_walkthrough(client: TestClient, video_factory, fake_engine) -> None:
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    resp = _upload(client, property["id"], video_factory(6.0), rooms=2, with_base=False)
    assert resp.status_code == 422 and resp.json()["code"] == "walkthrough_required"
    # A single room without a general video is fine: it becomes the frame itself.
    resp = _upload(client, property["id"], video_factory(6.0), rooms=1, with_base=False)
    assert resp.status_code == 201, resp.text
    job = _wait(client, resp.json()["job"]["id"])
    assert job["status"] == "succeeded"
    manifest = client.get(f"{API}/properties/{property['id']}/walkthrough").json()["manifest"]
    assert manifest["rooms"] == [] and manifest["base"]["key"] != "base"


def test_engine_failure_is_a_failed_job_not_a_dead_service(client: TestClient, video_factory, monkeypatch) -> None:
    def broken(spec, *, progress=None, cancel_check=None):
        from biosphere.core.models import ApiException

        raise ApiException("walk_failed", "Too few frames.", "Çok az kare.", status_code=422)

    monkeypatch.setattr("biosphere.core.jobs.WALK_BACKEND", broken)
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    resp = _upload(client, property["id"], video_factory(6.0), rooms=1)
    job = _wait(client, resp.json()["job"]["id"])
    assert job["status"] == "failed" and job["error"]["code"] == "walk_failed"
    walk = client.get(f"{API}/properties/{property['id']}/walkthrough").json()
    assert walk["status"] == "failed" and walk["error"]["message_tr"] == "Çok az kare."
    detail = client.get(f"{API}/properties/{property['id']}").json()
    room = detail["scenes"][0]
    assert room["status"] == "failed" and room["active_job_id"] is None
    assert client.get(f"{API}/health").status_code == 200
