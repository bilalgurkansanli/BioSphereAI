"""End-to-end tests for the BioSphere API.

Everything runs against a real temporary media root and a real SQLite file — the only thing
faked is the panorama pipeline itself, through the ``STITCH_BACKEND`` indirection. That keeps
the whole job lifecycle (upload -> queue -> worker thread -> progress -> SSE -> outputs on
disk -> URLs served through the media mount) under test without a two-minute stitch.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from biosphere.api import schemas as api_schemas
from biosphere.api.routes.properties import MAX_VIDEO_SEGMENTS
from biosphere.core import storage as storage_module
from biosphere.core.config import Settings, get_settings
from biosphere.core.db import reset_engine
from biosphere.core.jobs import reset_job_manager
from biosphere.pano.types import (
    CoverageReport,
    InsufficientCoverageError,
    PipelineConfig,
    StitchMetrics,
    StitchResult,
    SweepPattern,
)

API = "/api/v1"
JOB_TIMEOUT_S = 60.0

# Where the fake panorama puts a solid block, so the inpainting test has something to erase.
BLOCK = (200, 150, 260, 190)  # x0, y0, x1, y1


# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Point every global at a throwaway media root and database."""
    media_root = tmp_path / "media"
    monkeypatch.setenv("BIOSPHERE_MEDIA_ROOT", str(media_root))
    monkeypatch.setenv("BIOSPHERE_DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("BIOSPHERE_CORS_ALLOW_ORIGINS", '["http://localhost:3000"]')
    monkeypatch.setenv("BIOSPHERE_WORKER_CONCURRENCY", "2")
    # The suite tests ownership and sign-in, which only exist in accounts mode.
    monkeypatch.setenv("BIOSPHERE_AUTH_MODE", "accounts")

    get_settings.cache_clear()
    reset_engine()
    reset_job_manager()
    storage_module._storage = None

    resolved = get_settings()
    try:
        yield resolved
    finally:
        reset_job_manager()
        reset_engine()
        storage_module._storage = None
        get_settings.cache_clear()


@pytest.fixture()
def client(settings: Settings) -> Iterator[TestClient]:
    from biosphere.api.main import create_app

    with TestClient(create_app(settings)) as test_client:
        yield test_client


#: The account every authenticated fixture signs in as.
TEST_EMAIL = "agent@example.com"
TEST_PASSWORD = "correct-horse-battery"


@pytest.fixture()
def anon_client(client: TestClient) -> TestClient:
    """A client with no credentials, for asserting that a route refuses one."""
    return client


@pytest.fixture(autouse=True)
def _signed_in(request: pytest.FixtureRequest) -> Iterator[None]:
    """Register an account and put its token on the shared client.

    Autouse, and applied to the ``client`` fixture in place, because the alternative is
    threading a header through every one of the hundred-odd existing calls. Tests that need
    an anonymous caller ask for ``anon_client``, which clears the header for their duration.
    """
    if "client" not in request.fixturenames:
        yield
        return
    test_client: TestClient = request.getfixturevalue("client")
    response = test_client.post(
        f"{API}/auth/register",
        json={"email": TEST_EMAIL, "name": "Test Agent", "password": TEST_PASSWORD},
    )
    assert response.status_code == 201, response.text
    token = response.json()["token"]
    if "anon_client" in request.fixturenames:
        # The cookie the register call set would authenticate the "anonymous" client.
        test_client.cookies.clear()
        yield
        return
    test_client.headers["Authorization"] = f"Bearer {token}"
    yield
    test_client.headers.pop("Authorization", None)


@pytest.fixture(scope="session")
def video_factory(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Real MP4s produced by ffmpeg — the upload path sniffs them with ffprobe."""
    cache_dir = tmp_path_factory.mktemp("videos")

    def make(seconds: float = 6.0, size: str = "320x240") -> Path:
        path = cache_dir / f"capture_{seconds}_{size}.mp4"
        if not path.exists():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10:duration={seconds}",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path),
                ],
                check=True,
                capture_output=True,
            )
        return path

    return make


@pytest.fixture(scope="session")
def live_webm_factory(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A WebM muxed the way a browser muxes one: no duration written anywhere.

    ``-live 1`` makes the WebM muxer emit clusters of unknown size and leave the Segment
    header's Duration element empty, which is exactly what Chrome's ``MediaRecorder``
    produces — it is writing the file while the camera is still running and can never seek
    back to fill the field in. ffprobe then reports no duration and no frame count at all.
    """
    cache_dir = tmp_path_factory.mktemp("live-webm")

    def make(seconds: float = 12.0, size: str = "320x240") -> Path:
        path = cache_dir / f"live_{seconds}_{size}.webm"
        if not path.exists():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10:duration={seconds}",
                    "-c:v", "libvpx-vp9", "-b:v", "300k", "-f", "webm", "-live", "1", str(path),
                ],
                check=True,
                capture_output=True,
            )
        return path

    return make


# ======================================================================================
# Fake pipeline
# ======================================================================================


def make_panorama(width: int = 512, height: int = 256) -> np.ndarray:
    """A small but *real* equirect: gradients plus hard-edged structure to stitch/inpaint."""
    xs = np.linspace(0, 255, width, dtype=np.float32)
    ys = np.linspace(0, 255, height, dtype=np.float32)
    pano = np.zeros((height, width, 3), np.uint8)
    pano[..., 0] = np.clip(xs[None, :], 0, 255).astype(np.uint8)
    pano[..., 1] = np.clip(ys[:, None], 0, 255).astype(np.uint8)
    pano[..., 2] = np.clip(255.0 - xs[None, :] * 0.5, 0, 255).astype(np.uint8)
    cv2.circle(pano, (int(width * 0.2), int(height * 0.35)), max(6, height // 12), (20, 220, 40), -1)
    cv2.rectangle(pano, BLOCK[:2], BLOCK[2:], (255, 0, 255), -1)
    return pano


#: Where the room fixture's one furnishing sits: straddling the wall-floor junction, which
#: is where furniture actually stands and the hardest place for the emptier to work.
ROOM_BOX = (150, 150, 220, 205)  # x0, y0, x1, y1


def make_room_panorama(width: int = 512, height: int = 256) -> np.ndarray:
    """An equirect shaped like a ROOM, for the tools that need one to mean anything.

    `make_panorama` is a gradient, which is the right fixture for the stitch lifecycle and
    for a polygon inpaint — both only care that pixels exist and change. It is the wrong one
    for `/ai/empty-room`, which works by finding the colours a wall and a floor repeat in
    every direction. A gradient has none: every direction is a different colour, so 48 % of
    it reads as "not the room" and the endpoint correctly refuses to empty it. That refusal
    is itself worth having, so the gradient stays and this is added beside it.
    """
    image = np.empty((height, width, 3), np.uint8)
    image[...] = (150, 148, 145)
    image[: int(height * 0.34), :] = (225, 224, 222)
    image[int(height * 0.63) :, :] = (60, 90, 130)
    shade = 1.0 - 0.12 * (np.arange(height, dtype=np.float32) / height)
    image = np.clip(image.astype(np.float32) * shade[:, None, None], 0, 255).astype(np.uint8)
    x0, y0, x1, y1 = ROOM_BOX
    image[y0:y1, x0:x1] = (185, 60, 45)
    return image


DEFAULT_STAGES: tuple[tuple[str, float], ...] = (
    ("probe", 1.0),
    ("extract", 0.5),
    ("extract", 1.0),
    ("select", 1.0),
    ("features", 1.0),
    ("match", 1.0),
    ("bundle", 1.0),
    ("warp", 1.0),
    ("blend", 0.5),
    ("blend", 1.0),
    ("levels", 1.0),
)


def make_stitch_result() -> StitchResult:
    pano = make_panorama()
    mask = np.full((pano.shape[0], pano.shape[1]), 255, np.uint8)
    mask[: pano.shape[0] // 10, :] = 0  # a missing zenith cap
    coverage = CoverageReport(
        mask=mask,
        covered_fraction=0.93,
        horizon_fraction=1.0,
        yaw_span_deg=360.0,
        pitch_min_deg=-72.0,
        pitch_max_deg=64.0,
        zenith_covered=False,
        nadir_covered=True,
        gaps=[(12.0, 34.0, 55.0, 72.0)],
    )
    metrics = StitchMetrics(
        n_frames_extracted=142,
        n_keyframes=26,
        n_pairs=61,
        n_loop_closures=1,
        mean_reprojection_error_px=0.83,
        max_reprojection_error_px=3.10,
        loop_closure_residual_deg=0.42,
        focal_px=1180.0,
        hfov_deg=61.5,
        seam_gradient_energy=0.0041,
        exposure_spread=0.12,
        sweep_pattern=SweepPattern.SINGLE_ROW,
        stage_seconds={"extract": 1.4, "blend": 2.2},
        warnings=["zenith not covered"],
    )
    return StitchResult(panorama=pano, coverage=coverage, poses=[], metrics=metrics)


def fake_backend(gate: threading.Event | None = None, error: Exception | None = None) -> Any:
    """Build a STITCH_BACKEND stand-in that walks the real stage list."""

    def backend(
        source: Path | list[Path],
        config: PipelineConfig,
        *,
        work_dir: Path,
        progress: Any | None = None,
    ) -> StitchResult:
        assert Path(work_dir).exists(), "the worker must hand the pipeline a real work dir"
        assert isinstance(config, PipelineConfig)
        for index, (stage, fraction) in enumerate(DEFAULT_STAGES):
            if progress is not None:
                progress.stage(stage, fraction, f"{stage} {int(fraction * 100)}%")
            if gate is not None and index == 1:
                # Hold the job open so a subscriber can attach mid-flight.
                gate.wait(timeout=20)
            if error is not None and index == 2:
                raise error
        if progress is not None:
            progress.log("fake pipeline finished")
        return make_stitch_result()

    return backend


def use_backend(monkeypatch: pytest.MonkeyPatch, backend: Any) -> None:
    monkeypatch.setattr("biosphere.core.jobs.STITCH_BACKEND", backend)


# ======================================================================================
# Helpers
# ======================================================================================


def create_property(client: TestClient, title: str = "Bahçeli Villa") -> dict[str, Any]:
    response = client.post(f"{API}/properties", json={"title": title, "address": "İstanbul"})
    assert response.status_code == 201, response.text
    return response.json()


def upload_scene(
    client: TestClient,
    property_id: str,
    video: Path,
    *,
    name: str = "Salon",
    quality_mode: str = "balanced",
    capture_kind: str = "video",
    field: str = "file",
) -> Any:
    with video.open("rb") as handle:
        return client.post(
            f"{API}/properties/{property_id}/scenes",
            data={"name": name, "quality_mode": quality_mode, "capture_kind": capture_kind},
            files={field: (video.name, handle, "video/mp4")},
        )


def wait_for_job(client: TestClient, job_id: str, timeout: float = JOB_TIMEOUT_S) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        response = client.get(f"{API}/jobs/{job_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in {"succeeded", "failed", "cancelled"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s: {last}")


def ready_scene(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Property + fully processed scene, the starting point for the AI and media tests."""
    use_backend(monkeypatch, fake_backend())
    property = create_property(client)
    response = upload_scene(client, property["id"], video_factory())
    assert response.status_code == 201, response.text
    body = response.json()
    job = wait_for_job(client, body["job"]["id"])
    assert job["status"] == "succeeded", job
    scene = client.get(f"{API}/scenes/{body['scene']['id']}").json()
    assert scene["status"] == "ready"
    return property, scene


def assert_api_error(payload: dict[str, Any]) -> None:
    assert set(payload) >= {"code", "message", "message_tr"}
    assert isinstance(payload["code"], str) and payload["code"]
    assert payload["message"] and payload["message_tr"]
    assert payload["message"] != payload["message_tr"], "the Turkish copy must be a translation"


# ======================================================================================
# 1. Health
# ======================================================================================


def test_health_shape(client: TestClient) -> None:
    response = client.get(f"{API}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ffmpeg"] is True
    assert body["opencv"].split(".")[0].isdigit()
    assert isinstance(body["version"], str)
    assert body["gpu"] is None or isinstance(body["gpu"], str)
    assert body["media_root_free_gb"] > 0
    # Empty is the healthy answer, and the only one a fresh service should give. A stuck job
    # matters more than anything else here: with one worker it blocks every capture behind
    # it, so this list is what says the queue is moving.
    assert body["stalled_jobs"] == []
    # `ai` says where the room-emptying models run — a worker, this process, or nowhere — and
    # the dashboard warns off it like the other three environment facts. Its *contents* are
    # deliberately not asserted: they name whichever models this machine has, and a test that
    # pinned them would fail on every machine but one.
    assert isinstance(body["ai"], dict) and body["ai"].get("mode")
    # `walkthrough` says whether the 3D engine (trainer, SOG writer, SfM) is on this machine.
    assert isinstance(body["walkthrough"], dict) and "available" in body["walkthrough"]
    assert set(body) == {
        "status", "version", "ffmpeg", "opencv", "gpu", "media_root_free_gb", "stalled_jobs",
        "ai", "walkthrough",
    }


# ======================================================================================
# 2. Property CRUD
# ======================================================================================


def test_property_crud_roundtrip(client: TestClient) -> None:
    created = create_property(client, "Deniz Manzaralı Daire")
    assert created["title"] == "Deniz Manzaralı Daire"
    assert created["scene_count"] == 0
    assert created["views"] == 0
    assert created["published"] is False
    assert len(created["public_token"]) >= 16
    assert created["created_at"].endswith("Z")

    listed = client.get(f"{API}/properties", params={"limit": 5, "offset": 0}).json()
    assert listed["total"] == 1
    assert listed["limit"] == 5 and listed["offset"] == 0
    assert listed["items"][0]["id"] == created["id"]

    detail = client.get(f"{API}/properties/{created['id']}").json()
    assert detail["scenes"] == []

    patched = client.patch(
        f"{API}/properties/{created['id']}", json={"title": "Yeni Başlık", "published": True}
    )
    assert patched.status_code == 200
    assert patched.json()["title"] == "Yeni Başlık"
    assert patched.json()["published"] is True

    assert client.delete(f"{API}/properties/{created['id']}").status_code == 204
    assert client.get(f"{API}/properties/{created['id']}").status_code == 404


def test_property_404s_are_api_errors(client: TestClient) -> None:
    for method, url in (
        ("get", f"{API}/properties/does-not-exist"),
        ("patch", f"{API}/properties/does-not-exist"),
        ("delete", f"{API}/properties/does-not-exist"),
        ("get", f"{API}/scenes/does-not-exist"),
        ("get", f"{API}/jobs/does-not-exist"),
    ):
        response = getattr(client, method)(url, **({"json": {}} if method == "patch" else {}))
        assert response.status_code == 404, (method, url, response.text)
        assert_api_error(response.json())


# ======================================================================================
# 3. Scene upload
# ======================================================================================


def test_scene_upload_creates_job(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_backend(monkeypatch, fake_backend())
    property = create_property(client)
    response = upload_scene(client, property["id"], video_factory(), name="Mutfak")
    assert response.status_code == 201, response.text
    body = response.json()
    assert set(body) == {"scene", "job"}

    scene, job = body["scene"], body["job"]
    assert scene["name"] == "Mutfak"
    assert scene["property_id"] == property["id"]
    assert scene["capture_kind"] == "video"
    assert scene["quality_mode"] == "balanced"
    assert scene["status"] in {"pending", "processing"}
    assert scene["initial_view"] == {"yaw": 0.0, "pitch": 0.0, "fov": 75.0}
    assert scene["hotspots"] == []

    assert job["scene_id"] == scene["id"]
    assert job["property_id"] == property["id"]
    assert job["status"] in {"queued", "running"}
    assert job["stage"] == "upload"
    assert 0.0 <= job["progress"] <= 1.0
    assert job["created_at"].endswith("Z")
    wait_for_job(client, job["id"])


def test_missing_pipeline_fails_the_job_cleanly(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default backend imports the pipeline lazily.

    A ``None`` entry in ``sys.modules`` makes the import raise exactly as it would on a
    checkout where the engine is not installed. The job must then fail with a structured,
    bilingual error instead of a 500 traceback — and the API must still be importable, which
    is the whole point of deferring that import into the worker.
    """
    monkeypatch.setitem(sys.modules, "biosphere.pano.pipeline", None)
    property = create_property(client)
    response = upload_scene(client, property["id"], video_factory())
    assert response.status_code == 201
    job = wait_for_job(client, response.json()["job"]["id"])
    assert job["status"] == "failed"
    assert job["error"]["code"] == "pipeline_unavailable"
    assert_api_error(job["error"])
    assert client.get(f"{API}/health").status_code == 200


# ======================================================================================
# 4. Full job lifecycle
# ======================================================================================


def test_job_completes_and_publishes_media(
    client: TestClient, settings: Settings, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, scene = ready_scene(client, video_factory, monkeypatch)

    scene_dir = settings.properties_dir / scene["property_id"] / "scenes" / scene["id"]
    for filename in (
        "pano_uhd.jpg", "pano_hd.jpg", "pano_sd.jpg", "pano_preview.jpg",
        "pano_preview.webp", "coverage.png", "thumb.jpg", "metrics.json",
    ):
        assert (scene_dir / filename).is_file(), f"{filename} was not written"

    metrics_on_disk = json.loads((scene_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_on_disk["n_keyframes"] == 26

    panorama = scene["panorama"]
    assert panorama["width"] == 512 and panorama["height"] == 256
    assert panorama["cube_urls"] is None
    for key in ("preview_url", "sd_url", "hd_url", "uhd_url"):
        url = panorama[key]
        assert url and url.startswith("/media/properties/"), key
        served = client.get(url)
        assert served.status_code == 200, f"{key} -> {url} did not resolve"
        assert served.headers["access-control-allow-origin"] == "*"
        assert served.headers["cross-origin-resource-policy"] == "cross-origin"
        assert "immutable" in served.headers["cache-control"]
        assert cv2.imdecode(np.frombuffer(served.content, np.uint8), cv2.IMREAD_COLOR) is not None

    coverage = scene["coverage"]
    assert coverage["covered_fraction"] == pytest.approx(0.93)
    assert coverage["zenith_covered"] is False
    assert coverage["gaps"][0]["yaw_start_deg"] == pytest.approx(12.0)
    assert client.get(coverage["map_url"]).status_code == 200

    assert scene["metrics"]["sweep_pattern"] == "single_row"
    assert scene["metrics"]["warnings"] == ["zenith not covered"]
    assert scene["active_job_id"] is None
    assert scene["error"] is None

    property_detail = client.get(f"{API}/properties/{scene['property_id']}").json()
    assert property_detail["cover_url"], "a finished scene should give the property a cover image"
    assert property_detail["scene_count"] == 1


# ======================================================================================
# 5. Pipeline failure surfaces the Turkish copy
# ======================================================================================


def test_stitch_error_surfaces_bilingual_message(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_backend(monkeypatch, fake_backend(error=InsufficientCoverageError()))
    property = create_property(client)
    response = upload_scene(client, property["id"], video_factory())
    job = wait_for_job(client, response.json()["job"]["id"])

    assert job["status"] == "failed"
    error = job["error"]
    assert_api_error(error)
    assert error["message_tr"] == InsufficientCoverageError.user_message_tr
    assert error["message"] == InsufficientCoverageError.user_message_en
    assert error["code"] == "insufficient_coverage"

    scene = client.get(f"{API}/scenes/{response.json()['scene']['id']}").json()
    assert scene["status"] == "failed"
    assert scene["error"]["message_tr"] == InsufficientCoverageError.user_message_tr
    assert scene["active_job_id"] is None


# ======================================================================================
# 6. SSE
# ======================================================================================


def _parse_sse(chunk_lines: list[str]) -> list[dict[str, Any]]:
    return [json.loads(line[len("data:") :].strip()) for line in chunk_lines if line.startswith("data:")]


def test_job_events_stream_in_order_and_terminate(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = threading.Event()
    use_backend(monkeypatch, fake_backend(gate=gate))
    property = create_property(client)
    response = upload_scene(client, property["id"], video_factory())
    job_id = response.json()["job"]["id"]

    # Released from another thread so the subscriber is attached *before* the job finishes,
    # which is the only way to exercise the worker-thread -> event-loop bridge.
    threading.Timer(0.4, gate.set).start()

    events: list[dict[str, Any]] = []
    saw_retry = False
    with client.stream("GET", f"{API}/jobs/{job_id}/events") as stream:
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        for line in stream.iter_lines():
            if line.startswith("retry:"):
                saw_retry = True
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))
                if events[-1]["type"] in {"done", "error"}:
                    break

    assert saw_retry, "the stream must open with a retry preamble"
    assert events, "no events were received"
    assert events[-1]["type"] == "done", events[-1]

    progress_values = [e["job"]["progress"] for e in events if e["type"] == "progress"]
    assert len(progress_values) >= 3, progress_values
    assert progress_values == sorted(progress_values), "progress must never go backwards"
    assert progress_values[0] < progress_values[-1]

    stages = [e["job"]["stage"] for e in events if e["type"] == "progress"]
    assert "extract" in stages and "blend" in stages

    final = events[-1]
    assert final["job"]["status"] == "succeeded"
    assert final["job"]["progress"] == 1.0
    assert final["job"]["stage"] == "done"
    assert final["scene"]["status"] == "ready"
    assert final["scene"]["panorama"]["hd_url"].startswith("/media/properties/")

    # The stream closes after the terminal event; reconnecting replays it instead of hanging.
    with client.stream("GET", f"{API}/jobs/{job_id}/events") as stream:
        replay = [line for line in stream.iter_lines() if line.startswith("data:")]
    assert _parse_sse(replay)[-1]["type"] == "done"


# ======================================================================================
# 7. Upload rejection
# ======================================================================================


def test_upload_rejects_wrong_extension(client: TestClient, tmp_path: Path) -> None:
    property = create_property(client)
    bogus = tmp_path / "notes.txt"
    bogus.write_text("this is not a video", encoding="utf-8")
    with bogus.open("rb") as handle:
        response = client.post(
            f"{API}/properties/{property['id']}/scenes",
            data={"name": "Salon", "capture_kind": "video"},
            files={"file": (bogus.name, handle, "text/plain")},
        )
    assert response.status_code == 415, response.text
    body = response.json()
    assert_api_error(body)
    assert body["code"] == "unsupported_extension"


def test_upload_rejects_oversized_file(
    client: TestClient, settings: Settings, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "max_upload_bytes", 4096)
    property = create_property(client)
    response = upload_scene(client, property["id"], video_factory())
    assert response.status_code == 413, response.text
    body = response.json()
    assert_api_error(body)
    assert body["code"] == "upload_too_large"
    # The partial file must not survive the rejection.
    assert not any(settings.uploads_dir.glob("*/source_*"))


def test_upload_rejects_short_video(client: TestClient, video_factory: Any) -> None:
    property = create_property(client)
    response = upload_scene(client, property["id"], video_factory(seconds=2.0))
    assert response.status_code == 422, response.text
    body = response.json()
    assert_api_error(body)
    assert body["code"] == "video_too_short"
    assert "saniye" in body["message_tr"]


# ======================================================================================
# 7b. What the guided capture actually posts
# ======================================================================================
#
# The phone does NOT post a single video. It posts, in one `files[]` array:
#   - one video per recorded segment, and
#   - `motion.json`, the sensor track it measured while filming.
#
# Both facts are forced by the client: the sidecar rides along because an upload over a
# phone tether should cost one request and one progress bar, and the segments exist because
# the recorder is sealed and restarted whenever the encoder stalls or a call arrives.
#
# These tests exist because the endpoint used to reject BOTH — `len(uploads) > 1` refused a
# video scene outright — which failed every single guided capture at the very last step,
# after the user had already filmed the whole room.


def post_capture(
    client: TestClient,
    property_id: str,
    videos: list[Path],
    *,
    sidecar: bool = True,
    name: str = "Salon",
) -> Any:
    """Post exactly what `web/src/lib/api.ts` builds: every piece under `files`."""
    handles = [v.open("rb") for v in videos]
    try:
        payload: list[tuple[str, tuple[str, Any, str]]] = [
            ("files", (f"capture-{i + 1}.mp4", h, "video/mp4")) for i, h in enumerate(handles)
        ]
        if sidecar:
            track = io.BytesIO(json.dumps({"schema": 1, "samples": []}).encode())
            payload.append(("files", ("motion.json", track, "application/json")))
        return client.post(
            f"{API}/properties/{property_id}/scenes",
            data={"name": name, "quality_mode": "balanced", "capture_kind": "video"},
            files=payload,
        )
    finally:
        for h in handles:
            h.close()


def test_guided_capture_upload_keeps_the_sidecar_out_of_the_stitch(
    client: TestClient, settings: Settings, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path | list[Path]] = []
    inner = fake_backend()

    def recording(source: Path | list[Path], *args: Any, **kwargs: Any) -> Any:
        seen.append(source)
        return inner(source, *args, **kwargs)

    use_backend(monkeypatch, recording)
    property = create_property(client)

    response = post_capture(client, property["id"], [video_factory()])
    assert response.status_code == 201, response.text
    job = wait_for_job(client, response.json()["job"]["id"])
    assert job["status"] == "succeeded", job

    # One video in, one video to the pipeline: the sidecar must not be mistaken for footage.
    assert len(seen) == 1
    source = seen[0]
    assert isinstance(source, Path), f"a single-segment capture is one path, got {source!r}"
    assert source.suffix == ".mp4"

    # It is kept, though — it is the only record of what the phone believed it filmed.
    scene_id = response.json()["scene"]["id"]
    written = sorted(p.name for p in (settings.uploads_dir / scene_id).iterdir())
    assert "source_00.mp4" in written
    assert any(p.endswith(".json") for p in written), written


def test_guided_capture_upload_accepts_a_split_capture(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path | list[Path]] = []
    inner = fake_backend()

    def recording(source: Path | list[Path], *args: Any, **kwargs: Any) -> Any:
        seen.append(source)
        return inner(source, *args, **kwargs)

    use_backend(monkeypatch, recording)
    property = create_property(client)

    segments = [video_factory(seconds=6.0), video_factory(seconds=7.0)]
    response = post_capture(client, property["id"], segments)
    assert response.status_code == 201, response.text
    job = wait_for_job(client, response.json()["job"]["id"])
    assert job["status"] == "succeeded", job

    # Both pieces reach the pipeline, in the order they were recorded, and the sidecar
    # still does not.
    assert len(seen) == 1
    source = seen[0]
    assert isinstance(source, list) and len(source) == 2, f"expected two segments, got {source!r}"
    assert [p.suffix for p in source] == [".mp4", ".mp4"]
    assert [p.name for p in source] == ["source_00.mp4", "source_01.mp4"]


def test_a_short_opening_segment_does_not_condemn_the_capture(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported failure, exactly.

    The user filmed for about two minutes. The recorder's stall watchdog sealed the first
    few seconds into their own segment, so the upload was a 4-second file followed by a
    two-minute one — and the endpoint refused the whole capture with "the video is only 4.7s
    long", because the length gate was applied per FILE. A capture that legitimately arrives
    in pieces must be measured as a capture.
    """
    use_backend(monkeypatch, fake_backend())
    property = create_property(client)

    # Below `min_video_seconds` on its own; the pair is comfortably over.
    stub = video_factory(seconds=3.0)
    main = video_factory(seconds=8.0)

    response = post_capture(client, property["id"], [stub, main])
    assert response.status_code == 201, response.text
    job = wait_for_job(client, response.json()["job"]["id"])
    assert job["status"] == "succeeded", job


def test_upload_still_refuses_a_capture_that_is_short_in_total(
    client: TestClient, video_factory: Any
) -> None:
    property = create_property(client)
    # 1.0 + 2.0 = 3.0 s, comfortably under the 5 s floor. (2.0 + 3.0 would be exactly the
    # floor, which passes — the gate is `total < min`, not `<=`.)
    response = post_capture(client, property["id"], [video_factory(seconds=1.0), video_factory(seconds=2.0)])
    assert response.status_code == 422, response.text
    body = response.json()
    assert_api_error(body)
    assert body["code"] == "video_too_short"
    # The copy has to name the whole capture, not one of its files.
    assert "parça" in body["message_tr"] or "Çekim" in body["message_tr"]


def test_upload_rejects_more_segments_than_a_capture_can_have(
    client: TestClient, video_factory: Any
) -> None:
    property = create_property(client)
    response = post_capture(client, property["id"], [video_factory()] * (MAX_VIDEO_SEGMENTS + 1))
    assert response.status_code == 422, response.text
    body = response.json()
    assert_api_error(body)
    assert body["code"] == "too_many_files"


def test_upload_accepts_a_browser_recording_with_no_declared_duration(
    client: TestClient, live_webm_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The length gate must measure the footage, not trust the container.

    A browser recording declares no duration, so reading the field gives zero and a user
    who has just filmed a whole room is told the capture is "only 0.0s long" — a rejection
    at the very last step of the flow, for a file that is perfectly good.
    """
    from biosphere.core.storage import probe_media

    recording = live_webm_factory(seconds=12.0)
    # The premise: ffprobe genuinely reports nothing, so this is not a synthetic scenario.
    raw = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
         str(recording)],
        check=True, capture_output=True, text=True,
    )
    meta = json.loads(raw.stdout)
    video = next(s for s in meta["streams"] if s["codec_type"] == "video")
    assert meta["format"].get("duration") is None and video.get("duration") is None
    assert video.get("nb_frames") is None

    assert probe_media(recording).duration == pytest.approx(12.0, abs=0.5)

    use_backend(monkeypatch, fake_backend())
    property = create_property(client)
    with recording.open("rb") as handle:
        response = client.post(
            f"{API}/properties/{property['id']}/scenes",
            data={"name": "Salon", "quality_mode": "balanced", "capture_kind": "video"},
            files=[("files", ("capture-1.webm", handle, "video/webm"))],
        )
    assert response.status_code == 201, response.text
    job = wait_for_job(client, response.json()["job"]["id"])
    assert job["status"] == "succeeded", job


def test_upload_rejects_a_sidecar_with_no_footage(client: TestClient) -> None:
    property = create_property(client)
    track = io.BytesIO(b'{"schema":1}')
    response = client.post(
        f"{API}/properties/{property['id']}/scenes",
        data={"name": "Salon", "capture_kind": "video"},
        files=[("files", ("motion.json", track, "application/json"))],
    )
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "no_file"


# ======================================================================================
# 8. Hotspots
# ======================================================================================


def test_hotspot_crud_and_link_validation(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, scene = ready_scene(client, video_factory, monkeypatch)
    scene_id = scene["id"]

    created = client.post(
        f"{API}/scenes/{scene_id}/hotspots",
        json={"type": "info", "yaw": 0.5, "pitch": -0.2, "title": "Şömine", "body": "Doğalgaz"},
    )
    assert created.status_code == 201, created.text
    hotspot = created.json()
    assert hotspot["scene_id"] == scene_id
    assert hotspot["type"] == "info"
    assert hotspot["target_scene_id"] is None

    patched = client.patch(f"{API}/hotspots/{hotspot['id']}", json={"title": "Şömine (mermer)"})
    assert patched.status_code == 200
    assert patched.json()["title"] == "Şömine (mermer)"

    assert client.get(f"{API}/scenes/{scene_id}").json()["hotspots"][0]["id"] == hotspot["id"]

    dangling = client.post(
        f"{API}/scenes/{scene_id}/hotspots",
        json={"type": "link", "yaw": 1.0, "pitch": 0.0, "title": "Koridor", "target_scene_id": "ghost"},
    )
    assert dangling.status_code == 422, dangling.text
    assert_api_error(dangling.json())
    assert dangling.json()["code"] == "invalid_link_target"

    no_target = client.post(
        f"{API}/scenes/{scene_id}/hotspots",
        json={"type": "link", "yaw": 1.0, "pitch": 0.0, "title": "Koridor"},
    )
    assert no_target.status_code == 422
    assert no_target.json()["code"] == "missing_link_target"

    assert client.delete(f"{API}/hotspots/{hotspot['id']}").status_code == 204
    assert client.get(f"{API}/scenes/{scene_id}").json()["hotspots"] == []
    assert client.delete(f"{API}/hotspots/{hotspot['id']}").status_code == 404


def test_a_floor_ring_leads_to_a_room_even_before_it_is_shot(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A portal is a link that happens to be drawn on the floor, and is validated as one.

    The case that matters is the one the product is built around: an agent walks the flat,
    drops a ring at every doorway, and shoots those rooms afterwards. So a ring must be
    accepted while its destination is still an empty room with no panorama — the player draws
    that one as waiting rather than refusing to store it.
    """
    property, scene = ready_scene(client, video_factory, monkeypatch)
    scene_id = scene["id"]

    unshot = client.post(f"{API}/properties/{property['id']}/rooms", json={"name": "Oda kapısı"})
    assert unshot.status_code == 201, unshot.text
    room = unshot.json()
    assert room["panorama"] is None and room["capture_kind"] == "none"

    ring = client.post(
        f"{API}/scenes/{scene_id}/hotspots",
        json={
            "type": "portal",
            "yaw": 1.344,
            "pitch": -0.471,
            "title": "Oda kapısı",
            "target_scene_id": room["id"],
        },
    )
    assert ring.status_code == 201, ring.text
    assert ring.json()["type"] == "portal"
    assert ring.json()["target_scene_id"] == room["id"]
    assert client.get(f"{API}/scenes/{scene_id}").json()["hotspots"][-1]["type"] == "portal"

    # A ring with nowhere to go would be a control that does nothing, so it is refused at the
    # same gate `link` uses rather than at a second one written for the new type.
    orphan = client.post(
        f"{API}/scenes/{scene_id}/hotspots",
        json={"type": "portal", "yaw": 0.2, "pitch": -0.4, "title": "Boşluk"},
    )
    assert orphan.status_code == 422, orphan.text
    assert orphan.json()["code"] == "missing_link_target"

    ghost = client.post(
        f"{API}/scenes/{scene_id}/hotspots",
        json={
            "type": "portal",
            "yaw": 0.2,
            "pitch": -0.4,
            "title": "Hayalet",
            "target_scene_id": "ghost",
        },
    )
    assert ghost.status_code == 422, ghost.text
    assert ghost.json()["code"] == "invalid_link_target"

    # Retargeting a ring goes through the same check; a PATCH is not a way around it.
    moved = client.patch(f"{API}/hotspots/{ring.json()['id']}", json={"target_scene_id": "ghost"})
    assert moved.status_code == 422, moved.text
    assert moved.json()["code"] == "invalid_link_target"


# ======================================================================================
# 9. Measurement
# ======================================================================================


def test_measure_returns_hand_checked_distances(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hand-derived ground truth.

    With the camera 200 cm above the floor, a point marked 45° below the horizon lies at
    ``r = h / tan(45°) = 200 cm`` from the camera's floor projection. Three such points at
    yaw 0°, 90° and 180° therefore sit at (x, z) = (0, 200), (200, 0) and (0, -200), so each
    consecutive gap is ``sqrt(200² + 200²) = 282.84 cm`` and the triangle they enclose has
    area ``0.5 · 400 · 200 = 40000 cm² = 4 m²``.
    """
    _, scene = ready_scene(client, video_factory, monkeypatch)
    down = -np.pi / 4

    response = client.post(
        f"{API}/ai/measure",
        json={
            "scene_id": scene["id"],
            "points": [
                {"yaw": 0.0, "pitch": down},
                {"yaw": np.pi / 2, "pitch": down},
                {"yaw": np.pi, "pitch": down},
            ],
            "camera_height_cm": 200.0,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["scale_source"] == "camera_height"
    assert body["model_backed"] is False
    assert len(body["segments"]) == 2
    for segment in body["segments"]:
        assert segment["length_cm"] == pytest.approx(282.84, abs=0.2)
        assert segment["length_relative"] == pytest.approx(1.4142, abs=0.001)
    assert body["total_cm"] == pytest.approx(565.69, abs=0.4)
    assert body["area_m2"] == pytest.approx(4.0, abs=0.02)
    assert body["bbox_cm"]["height"] == pytest.approx(200.0)
    assert body["bbox_cm"]["width"] >= body["bbox_cm"]["depth"] > 0
    assert 0.5 < body["confidence"] <= 1.0
    assert body["segments"][0]["from"] == 0 and body["segments"][0]["to"] == 1


def test_measure_returns_null_above_the_horizon(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, scene = ready_scene(client, video_factory, monkeypatch)
    response = client.post(
        f"{API}/ai/measure",
        json={
            "scene_id": scene["id"],
            "points": [
                {"yaw": 0.0, "pitch": -np.pi / 4},
                {"yaw": np.pi / 2, "pitch": -np.pi / 4},
                {"yaw": np.pi, "pitch": float(np.radians(10.0))},
            ],
            "camera_height_cm": 200.0,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["segments"][0]["length_cm"] == pytest.approx(282.84, abs=0.2)
    assert body["segments"][1]["length_cm"] is None, "a point above the horizon has no floor hit"
    assert body["segments"][1]["hint"]
    assert body["total_cm"] is None
    assert body["area_m2"] is None  # only two usable floor points


def test_measure_reference_overrides_camera_height(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A known length rescales everything, whatever camera height was guessed."""
    _, scene = ready_scene(client, video_factory, monkeypatch)
    down = -np.pi / 4
    response = client.post(
        f"{API}/ai/measure",
        json={
            "scene_id": scene["id"],
            "points": [{"yaw": 0.0, "pitch": down}, {"yaw": np.pi / 2, "pitch": down}],
            # The reference is the same geometric segment, declared to be 100 cm long.
            "reference": {
                "yaw_a": 0.0, "pitch_a": down, "yaw_b": np.pi / 2, "pitch_b": down,
                "length_cm": 100.0,
            },
            "camera_height_cm": 200.0,
        },
    )
    body = response.json()
    assert body["scale_source"] == "reference"
    assert body["segments"][0]["length_cm"] == pytest.approx(100.0, abs=0.1)


# ======================================================================================
# 10. Furniture removal
# ======================================================================================


def test_hide_furniture_changes_pixels_inside_the_polygon(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, scene = ready_scene(client, video_factory, monkeypatch)
    x0, y0, x1, y1 = BLOCK
    polygon = [[x0 - 4, y0 - 4], [x1 + 4, y0 - 4], [x1 + 4, y1 + 4], [x0 - 4, y1 + 4]]

    response = client.post(
        f"{API}/ai/hide-furniture",
        json={"scene_id": scene["id"], "polygon": polygon, "strength": 0.4},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_backed"] is False
    assert body["elapsed_ms"] >= 0
    assert body["removed_objects"] and body["removed_objects"][0]["area_px"] > 0
    assert body["removed_objects"][0]["label"] in {"furniture", "rug", "clutter"}

    before = client.get(scene["panorama"]["hd_url"])
    after = client.get(body["image_url"])
    mask_response = client.get(body["mask_url"])
    assert after.status_code == 200 and mask_response.status_code == 200

    original = cv2.imdecode(np.frombuffer(before.content, np.uint8), cv2.IMREAD_COLOR)
    edited = cv2.imdecode(np.frombuffer(after.content, np.uint8), cv2.IMREAD_COLOR)
    assert edited is not None and edited.shape == original.shape

    inside = np.abs(
        edited[y0:y1, x0:x1].astype(int) - original[y0:y1, x0:x1].astype(int)
    ).mean()
    outside = np.abs(edited[:60, :60].astype(int) - original[:60, :60].astype(int)).mean()
    assert inside > 15.0, f"the masked region was not repainted (mean delta {inside:.2f})"
    assert outside < 6.0, "pixels far from the polygon must be left alone"


def test_empty_room_returns_a_ladder_and_caches_it(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staged room arrives as its own resolution ladder, and only computes once.

    A ladder rather than one image because the player streams a scene through one, and a
    single 4K JPEG handed to a phone as if it were a preview is how a tab runs out of VRAM.
    Cached because the control that calls this is a toggle: the first thing anyone does with
    it is press it twice.
    """
    monkeypatch.setattr(sys.modules[__name__], "make_panorama", make_room_panorama)
    _, scene = ready_scene(client, video_factory, monkeypatch)

    first = client.post(f"{API}/ai/empty-room", json={"scene_id": scene["id"]})
    assert first.status_code == 200, first.text
    body = first.json()

    # True on a machine with the scene model installed, False on one without. The contract
    # is that the answer is stated, not what it is.
    assert isinstance(body["model_backed"], bool)
    assert body["cached"] is False
    assert body["regions"], "something in the fixture panorama should separate from the room"
    assert 0.0 < body["removed_fraction"] < 1.0
    levels = body["levels"]
    assert levels["preview_url"] and levels["hd_url"]
    # Always null: the emptied master is capped, because a third of its pixels are invented.
    assert levels["uhd_url"] is None

    second = client.post(f"{API}/ai/empty-room", json={"scene_id": scene["id"]})
    assert second.status_code == 200, second.text
    assert second.json()["cached"] is True, "a second press must not recompute"
    assert second.json()["levels"]["hd_url"] == levels["hd_url"]

    # Every rung has to be reachable through the media mount, or the player shows black.
    for key in ("preview_url", "hd_url"):
        served = client.get(levels[key])
        assert served.status_code == 200
        assert served.headers["cross-origin-resource-policy"] == "cross-origin"

    original = cv2.imdecode(
        np.frombuffer(client.get(scene["panorama"]["hd_url"]).content, np.uint8), cv2.IMREAD_COLOR
    )
    staged = cv2.imdecode(
        np.frombuffer(client.get(levels["hd_url"]).content, np.uint8), cv2.IMREAD_COLOR
    )
    assert staged is not None and staged.shape == original.shape
    x0, y0, x1, y1 = ROOM_BOX
    inside = np.abs(
        staged[y0 + 6 : y1 - 6, x0 + 6 : x1 - 6].astype(int)
        - original[y0 + 6 : y1 - 6, x0 + 6 : x1 - 6].astype(int)
    ).mean()
    assert inside > 15.0, f"the furnishing survived the emptying (mean delta {inside:.2f})"
    # And the room it stood in front of is left exactly as it was.
    untouched = np.abs(staged[:40, :40].astype(int) - original[:40, :40].astype(int)).mean()
    assert untouched < 2.0, "the ceiling was disturbed"


def test_empty_room_rejects_an_unknown_scene(client: TestClient) -> None:
    response = client.post(f"{API}/ai/empty-room", json={"scene_id": "does-not-exist"})
    assert response.status_code == 404
    assert_api_error(response.json())


def test_empty_room_refuses_to_invent_most_of_a_panorama(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The area cap, tested where it lives rather than through a fixture that trips it.

    A per-pixel test says how surprising a pixel is; it cannot say how much of the picture
    it is about to invent. So there is a ceiling, and past it the honest answer is to refuse
    rather than to hand back a panorama that is mostly synthesised.

    The cap is driven to a value this fixture must exceed instead of relying on the fixture
    to exceed the real one: it used to, and then the segmentation improved and stopped —
    which quietly turned this into a test of nothing at all.
    """
    monkeypatch.setattr("biosphere.ai.empty_room._MAX_REMOVED_FRACTION_CLASSICAL", 0.001)
    monkeypatch.setattr("biosphere.ai.empty_room._MAX_REMOVED_FRACTION_MODEL", 0.001)
    _, scene = ready_scene(client, video_factory, monkeypatch)
    response = client.post(f"{API}/ai/empty-room", json={"scene_id": scene["id"]})
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "room_not_recognised"
    assert_api_error(body)


# ======================================================================================
# 11 & 12. Public route and CORS
# ======================================================================================


def test_public_measure_answers_any_origin_and_only_for_its_own_property(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listing-page visitor may measure; the token is still a key to one property only.

    Measuring is the one tool a third-party embed is allowed to run, because it reads no
    pixels and starts no model -- it intersects a marked ray with the floor plane. That makes
    the scope check the whole of its security: `measure_scene` takes a bare scene id, so
    without it an embed token would be a key to every scene in the deployment.
    """
    property, scene = ready_scene(client, video_factory, monkeypatch)
    # The embed endpoints answer only for a PUBLISHED property; that is the whole meaning of
    # the switch, and it is enforced in `public._property_by_token`.
    client.patch(f"{API}/properties/{property['id']}", json={"published": True})
    token = property["public_token"]
    origin = "https://listing-portal.example.com"
    body = {
        "scene_id": scene["id"],
        # Two directions well below the horizon, which is where the floor is.
        "points": [{"yaw": 0.0, "pitch": -0.8}, {"yaw": 0.6, "pitch": -0.7}],
        "camera_height_cm": 160.0,
    }

    ok = client.post(f"{API}/public/properties/{token}/measure", json=body, headers={"Origin": origin})
    assert ok.status_code == 200, ok.text
    assert ok.headers["access-control-allow-origin"] == "*"
    assert ok.headers["content-security-policy"] == "frame-ancestors *"
    answer = ok.json()
    assert answer["scale_source"] == "camera_height"
    assert answer["segments"][0]["length_cm"] is not None
    assert answer["total_cm"] > 0

    # What a portal's frame can actually send. A JSON POST there needs a preflight, and the
    # public policy (GET/HEAD) refuses it — so the browser never sends the request at all:
    preflight = client.options(
        f"{API}/public/properties/{token}/measure",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert "POST" not in preflight.headers.get("access-control-allow-methods", "")
    # ...which is why the player sends a CORS simple request instead: text/plain, no cookies.
    import json as _json

    simple = client.post(
        f"{API}/public/properties/{token}/measure",
        content=_json.dumps(body),
        headers={"Origin": origin, "Content-Type": "text/plain;charset=UTF-8"},
    )
    assert simple.status_code == 200, simple.text
    assert simple.headers["access-control-allow-origin"] == "*"
    assert simple.json()["total_cm"] == answer["total_cm"]
    garbled = client.post(
        f"{API}/public/properties/{token}/measure",
        content="not json",
        headers={"Origin": origin, "Content-Type": "text/plain;charset=UTF-8"},
    )
    assert garbled.status_code == 422

    # Somebody else's scene, through this token, is a 404 -- the same answer an unknown token
    # gets, so the response never confirms that the id exists elsewhere.
    other_property, other_scene = ready_scene(client, video_factory, monkeypatch)
    stolen = dict(body, scene_id=other_scene["id"])
    assert client.post(f"{API}/public/properties/{token}/measure", json=stolen).status_code == 404
    assert client.post(f"{API}/public/properties/not-a-real-token/measure", json=body).status_code == 404
    assert other_property["id"] != property["id"]


def test_public_property_counts_views_and_allows_any_origin(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    property, scene = ready_scene(client, video_factory, monkeypatch)
    # The embed endpoints answer only for a PUBLISHED property; that is the whole meaning of
    # the switch, and it is enforced in `public._property_by_token`.
    client.patch(f"{API}/properties/{property['id']}", json={"published": True})
    token = property["public_token"]
    origin = "https://listing-portal.example.com"

    first = client.get(f"{API}/public/properties/{token}", headers={"Origin": origin})
    assert first.status_code == 200, first.text
    assert first.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in first.headers
    assert first.headers["content-security-policy"] == "frame-ancestors *"
    assert "x-frame-options" not in first.headers

    body = first.json()
    assert body["views"] == 1
    assert body["id"] == property["id"]
    assert len(body["scenes"]) == 1
    assert body["scenes"][0]["id"] == scene["id"]

    second = client.get(f"{API}/public/properties/{token}", headers={"Origin": origin})
    assert second.json()["views"] == 2

    assert client.get(f"{API}/public/properties/not-a-real-token").status_code == 404


def test_cors_write_routes_reject_foreign_origins(client: TestClient) -> None:
    foreign = "https://evil.example.com"

    blocked = client.options(
        f"{API}/properties",
        headers={
            "Origin": foreign,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert blocked.status_code == 400
    assert "access-control-allow-origin" not in blocked.headers

    allowed = client.options(
        f"{API}/properties",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"

    public = client.options(
        f"{API}/public/properties/anything",
        headers={"Origin": foreign, "Access-Control-Request-Method": "GET"},
    )
    assert public.status_code == 200

    # The chunked upload is a PUT; a dashboard on its own origin preflights it too.
    part = client.options(
        f"{API}/properties/uploads/abc/parts/0",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert part.status_code == 200
    assert "PUT" in part.headers["access-control-allow-methods"]
    assert public.headers["access-control-allow-origin"] == "*"


# ======================================================================================
# 13. Deletion removes media
# ======================================================================================


def test_delete_property_removes_media_from_disk(
    client: TestClient, settings: Settings, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    property, scene = ready_scene(client, video_factory, monkeypatch)
    property_dir = settings.properties_dir / property["id"]
    upload_dir = settings.uploads_dir / scene["id"]
    assert property_dir.is_dir() and any(property_dir.rglob("pano_hd.jpg"))
    assert upload_dir.is_dir()

    assert client.delete(f"{API}/properties/{property['id']}").status_code == 204

    assert not property_dir.exists(), "the property media directory must be removed"
    assert not upload_dir.exists(), "the raw upload must be removed too"
    assert client.get(f"{API}/scenes/{scene['id']}").status_code == 404
    assert client.get(f"{API}/public/properties/{property['public_token']}").status_code == 404


# ======================================================================================
# Extras that guard the trickier behaviour
# ======================================================================================


def test_scene_update_reprocess_and_cancel(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, scene = ready_scene(client, video_factory, monkeypatch)

    patched = client.patch(
        f"{API}/scenes/{scene['id']}",
        json={"name": "Oturma Odası", "order": 3, "initial_view": {"yaw": 1.2, "pitch": -0.1, "fov": 90}},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["name"] == "Oturma Odası"
    assert patched.json()["order"] == 3
    assert patched.json()["initial_view"]["fov"] == 90

    gate = threading.Event()
    use_backend(monkeypatch, fake_backend(gate=gate))
    job = client.post(f"{API}/scenes/{scene['id']}/reprocess")
    assert job.status_code == 202, job.text
    job_id = job.json()["id"]

    cancelled = client.post(f"{API}/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    gate.set()
    final = wait_for_job(client, job_id)
    assert final["status"] == "cancelled"
    # The scene keeps the panorama it already had rather than reverting to failed.
    assert client.get(f"{API}/scenes/{scene['id']}").json()["status"] == "ready"


def test_media_mount_does_not_expose_the_database(client: TestClient, settings: Settings) -> None:
    """Only <media_root>/properties is published; the SQLite file and raw uploads are not."""
    assert client.get("/media/db/biosphere.db").status_code == 404
    assert client.get("/media/uploads/anything.mp4").status_code == 404


# ======================================================================================
# Schema drift
# ======================================================================================


def test_stitch_metrics_schema_matches_the_dataclass() -> None:
    """The API model must accept exactly what ``StitchMetrics.as_dict()`` persists.

    ``scene_to_schema`` does ``StitchMetrics(**row.metrics)`` against a model with
    ``extra="forbid"``, so a field added to the dataclass but not mirrored in the schema
    turns every succeeded job into a *failed* one at ``_finish_success`` — and only for
    jobs that actually ran, which is why it survives a green unit suite. Checked both
    ways: a field declared only on the model is dead weight that never receives a value.
    """
    emitted = StitchMetrics().as_dict()
    declared = set(api_schemas.StitchMetrics.model_fields)

    assert set(emitted) == declared, "biosphere.api.schemas.StitchMetrics drifted from the dataclass"

    # Names alone are not enough — validate real values, defaults and populated alike.
    api_schemas.StitchMetrics(**emitted)
    api_schemas.StitchMetrics(**make_stitch_result().metrics.as_dict())


def test_the_job_survives_the_phone_going_away(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing the progress stream must not touch the job.

    The capture takes two minutes and the panorama takes longer; a phone that locks, runs
    out of battery, or walks out of Wi-Fi range in the middle of that must not cost the
    user the property. The work runs on the engine's own worker pool, and the event stream is
    only a window onto it — so this test hangs up the window mid-job and then asserts the
    job still reached the end on its own.
    """
    gate = threading.Event()
    use_backend(monkeypatch, fake_backend(gate=gate))
    property = create_property(client)
    response = upload_scene(client, property["id"], video_factory())
    job_id = response.json()["job"]["id"]

    # Attach, read one frame, then walk away without waiting for a terminal event — exactly
    # what a phone that dies mid-processing does to the connection.
    with client.stream("GET", f"{API}/jobs/{job_id}/events") as stream:
        assert stream.status_code == 200
        for line in stream.iter_lines():
            if line.startswith("data:"):
                break

    # Only now is the pipeline allowed to finish, with nobody listening at all.
    gate.set()
    deadline = time.monotonic() + 30.0
    status = None
    while time.monotonic() < deadline:
        status = client.get(f"{API}/jobs/{job_id}").json()["status"]
        if status in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.05)

    assert status == "succeeded", f"the job must finish without a client attached, got {status}"
    # And it must be findable again: the phone comes back to the property by URL, not by memory.
    scenes = client.get(f"{API}/properties/{property['id']}").json()["scenes"]
    assert any(s["status"] == "ready" for s in scenes), scenes


# ======================================================================================
# A room's photographs
# ======================================================================================


def _photo_bytes(width: int = 800, height: int = 600, shade: int = 200) -> bytes:
    image = np.full((height, width, 3), shade, np.uint8)
    cv2.rectangle(image, (40, 40), (width - 40, height - 40), (20, 90, 160), -1)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def _upload(client: TestClient, scene_id: str, count: int = 2) -> list[dict]:
    files = [
        ("files", (f"oda_{i}.jpg", io.BytesIO(_photo_bytes(shade=180 + i * 10)), "image/jpeg"))
        for i in range(count)
    ]
    response = client.post(f"{API}/scenes/{scene_id}/photos", files=files)
    assert response.status_code == 201, response.text
    return response.json()


def test_a_room_can_exist_without_any_capture(client: TestClient) -> None:
    """The unit is a ROOM: photographs of the kitchen with no 360 of it is a real listing."""
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    response = client.post(f"{API}/properties/{property['id']}/rooms", json={"name": "Çocuk odası"})
    assert response.status_code == 201, response.text
    room = response.json()

    assert room["name"] == "Çocuk odası"
    assert room["capture_kind"] == "none"
    assert room["status"] == "ready", "there is no job to wait for"
    assert room["panorama"] is None
    assert room["photos"] == []
    assert room["active_job_id"] is None


def test_photographs_attach_to_a_room_and_come_back_with_it(client: TestClient) -> None:
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    room = client.post(f"{API}/properties/{property['id']}/rooms", json={"name": "Salon"}).json()

    created = _upload(client, room["id"], count=3)
    assert len(created) == 3
    assert [p["order"] for p in created] == [0, 1, 2], "order is assigned, not left to chance"
    assert all(p["width"] == 800 and p["height"] == 600 for p in created)

    # On the scene...
    scene = client.get(f"{API}/scenes/{room['id']}").json()
    assert [p["id"] for p in scene["photos"]] == [p["id"] for p in created]
    # ...on the property detail...
    detail = client.get(f"{API}/properties/{property['id']}").json()
    assert len(detail["scenes"][0]["photos"]) == 3
    # ...and on the public payload, which is who they were uploaded for.
    client.patch(f"{API}/properties/{property['id']}", json={"published": True})
    public = client.get(f"{API}/public/properties/{property['public_token']}").json()
    assert len(public["scenes"][0]["photos"]) == 3

    # And the bytes are actually served.
    served = client.get(created[0]["url"].split("?")[0])
    assert served.status_code == 200
    assert cv2.imdecode(np.frombuffer(served.content, np.uint8), cv2.IMREAD_COLOR) is not None


def test_a_large_photograph_is_stored_scaled(client: TestClient) -> None:
    from biosphere.api.routes.scenes import _PHOTO_MAX_SIDE

    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    room = client.post(f"{API}/properties/{property['id']}/rooms", json={"name": "Salon"}).json()
    response = client.post(
        f"{API}/scenes/{room['id']}/photos",
        files=[("files", ("big.jpg", io.BytesIO(_photo_bytes(5000, 3000)), "image/jpeg"))],
    )
    assert response.status_code == 201
    assert max(response.json()[0]["width"], response.json()[0]["height"]) == _PHOTO_MAX_SIDE


def test_one_corrupt_file_does_not_cost_the_good_ones(client: TestClient) -> None:
    """Refusing all five to punish one would cost the user four uploads."""
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    room = client.post(f"{API}/properties/{property['id']}/rooms", json={"name": "Salon"}).json()
    response = client.post(
        f"{API}/scenes/{room['id']}/photos",
        files=[
            ("files", ("good.jpg", io.BytesIO(_photo_bytes()), "image/jpeg")),
            ("files", ("bad.jpg", io.BytesIO(b"not a jpeg at all"), "image/jpeg")),
        ],
    )
    assert response.status_code == 201
    assert len(response.json()) == 1, "the readable one is kept"


def test_a_request_with_no_readable_image_is_refused(client: TestClient) -> None:
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    room = client.post(f"{API}/properties/{property['id']}/rooms", json={"name": "Salon"}).json()
    response = client.post(
        f"{API}/scenes/{room['id']}/photos",
        files=[("files", ("bad.jpg", io.BytesIO(b"nope"), "image/jpeg"))],
    )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_media"
    assert response.json()["message_tr"]


def test_a_photograph_can_be_captioned_reordered_and_deleted(
    client: TestClient, settings: Settings
) -> None:
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    room = client.post(f"{API}/properties/{property['id']}/rooms", json={"name": "Salon"}).json()
    created = _upload(client, room["id"], count=2)

    patched = client.patch(
        f"{API}/scene-photos/{created[1]['id']}", json={"caption": "Balkon çıkışı", "order": -1}
    )
    assert patched.status_code == 200
    assert patched.json()["caption"] == "Balkon çıkışı"

    # Reordering is what the list is sorted by, so the second photo now comes first.
    scene = client.get(f"{API}/scenes/{room['id']}").json()
    assert [p["id"] for p in scene["photos"]] == [created[1]["id"], created[0]["id"]]

    on_disk = settings.properties_dir / property["id"] / "scenes" / room["id"] / "photos"
    assert len(list(on_disk.glob("*.jpg"))) == 2

    assert client.delete(f"{API}/scene-photos/{created[0]['id']}").status_code == 204
    assert len(client.get(f"{API}/scenes/{room['id']}").json()["photos"]) == 1
    assert len(list(on_disk.glob("*.jpg"))) == 1, "the file goes with the row"

    assert client.delete(f"{API}/scene-photos/{created[0]['id']}").status_code == 404


def test_deleting_a_room_takes_its_photographs(client: TestClient, settings: Settings) -> None:
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    room = client.post(f"{API}/properties/{property['id']}/rooms", json={"name": "Salon"}).json()
    _upload(client, room["id"], count=2)

    assert client.delete(f"{API}/scenes/{room['id']}").status_code == 204
    assert not (settings.properties_dir / property["id"] / "scenes" / room["id"]).exists()
    assert client.get(f"{API}/properties/{property['id']}").json()["scenes"] == []


def test_photo_variants_start_empty_and_unknown_style_is_refused(client: TestClient) -> None:
    """The two answers that must not need a GPU."""
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    room = client.post(f"{API}/properties/{property['id']}/rooms", json={"name": "Salon"}).json()
    photo = _upload(client, room["id"], count=1)[0]

    assert client.get(f"{API}/scene-photos/{photo['id']}/variants").json() == []

    refused = client.post(f"{API}/scene-photos/{photo['id']}/restyle", json={"style": "art-deco"})
    assert refused.status_code == 422
    assert refused.json()["code"] == "unknown_style"
    assert refused.json()["message_tr"]


def test_the_ai_tools_refuse_an_unknown_photo(client: TestClient) -> None:
    missing = "0" * 12
    for path, body in (("restyle", {"style": "modern"}), ("empty", {})):
        response = client.post(f"{API}/scene-photos/{missing}/{path}", json=body)
        assert response.status_code == 404, path
        assert response.json()["code"] == "photo_not_found"
    assert client.get(f"{API}/scene-photos/{missing}/variants").status_code == 404


def test_deleting_a_photo_takes_its_derived_versions(
    client: TestClient, settings: Settings
) -> None:
    """A derived directory outliving its source is a stale render waiting to be served."""
    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    room = client.post(f"{API}/properties/{property['id']}/rooms", json={"name": "Salon"}).json()
    photo = _upload(client, room["id"], count=1)[0]

    # Stand in for a render: the route only ever reads this directory back.
    derived = (
        settings.properties_dir / property["id"] / "scenes" / room["id"] / "photos" / "derived" / photo["id"]
    )
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "modern.jpg").write_bytes(b"stand-in")
    assert derived.exists()

    assert client.delete(f"{API}/scene-photos/{photo['id']}").status_code == 204
    assert not derived.exists(), "the derived versions go with the photograph"


def test_a_stale_sidecar_is_not_served(client: TestClient, settings: Settings) -> None:
    """Variants are keyed on the original's mtime, so a replaced file invalidates them."""
    import json as _json

    property = client.post(f"{API}/properties", json={"title": "Daire"}).json()
    room = client.post(f"{API}/properties/{property['id']}/rooms", json={"name": "Salon"}).json()
    photo = _upload(client, room["id"], count=1)[0]

    photos_dir = settings.properties_dir / property["id"] / "scenes" / room["id"] / "photos"
    original = next(photos_dir.glob("*.jpg"))
    derived = photos_dir / "derived" / photo["id"]
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "modern.jpg").write_bytes(b"stand-in")

    entry = {"protected_fraction": 0.5, "objects": ["bed"], "elapsed_ms": 1, "model_backed": True}
    written = {"source_mtime": int(original.stat().st_mtime), "variants": {"modern": entry}}
    (derived / "meta.json").write_text(_json.dumps(written), encoding="utf-8")
    assert [v["variant"] for v in client.get(f"{API}/scene-photos/{photo['id']}/variants").json()] == [
        "modern"
    ]

    written["source_mtime"] = written["source_mtime"] - 10_000
    (derived / "meta.json").write_text(_json.dumps(written), encoding="utf-8")
    assert client.get(f"{API}/scene-photos/{photo['id']}/variants").json() == []
