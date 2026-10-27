"""What virtual staging must hold true without a GPU in the room.

None of these tests run a diffusion pass. That is deliberate: a sampler's output is judged by
looking at it (``scripts/bench_restyle.py`` exists for exactly that), and a test that asserted
pixels would be asserting a checkpoint's taste. What IS testable here is everything around the
model, and every one of these guards a fault that was actually hit while building the feature:

* **The prompt budget.** CLIP truncates at 77 tokens and diffusers does not raise — it logs and
  carries on. The first working version shipped 124-token negative prompts, so half of every
  one of them, including all the anti-duplication terms, was discarded on every single pass and
  nothing said so. This is the test that makes that a build failure rather than a mystery.
* **The protection composite.** The promise "your window is still your window" is kept by
  compositing, so it can be checked exactly: protected pixels must come back byte-identical.
* **The edge conditioner is not blank.** An empty Canny map is not a weak conditioner, it is an
  absent one, and it fails silently — the pass simply reverts to depth alone.
* **The path handling on a content-addressed id.** The id is used as a directory name.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from biosphere.ai import restyle
from biosphere.core import storage as storage_module
from biosphere.core.config import Settings, get_settings
from biosphere.core.db import reset_engine
from biosphere.core.jobs import reset_job_manager
from biosphere.core.models import ApiException

API = "/api/v1"


# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    monkeypatch.setenv("BIOSPHERE_MEDIA_ROOT", str(tmp_path / "media"))
    monkeypatch.setenv("BIOSPHERE_DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
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
    """A signed-in client. The restyle routes sit behind ``current_user`` since accounts landed,
    and these tests are about staging, not about sign-in — ``test_auth.py`` owns that."""
    from biosphere.api.main import create_app

    with TestClient(create_app(settings)) as test_client:
        response = test_client.post(
            "/api/v1/auth/register",
            json={"email": "stager@example.com", "name": "Stager", "password": "doğru-at-pil-zımba"},
        )
        assert response.status_code == 201, response.text
        test_client.headers["Authorization"] = f"Bearer {response.json()['token']}"
        yield test_client


def room_photo(width: int = 640, height: int = 480) -> np.ndarray:
    """A crude room: ceiling, wall, floor, one dark doorway. Enough to have architecture."""
    image = np.zeros((height, width, 3), np.uint8)
    horizon = int(height * 0.68)
    ceiling = int(height * 0.16)
    image[:ceiling] = (238, 238, 240)
    image[ceiling:horizon] = (206, 205, 200)
    image[horizon:] = (150, 140, 128)
    cv2.rectangle(image, (int(width * 0.42), ceiling + 8), (int(width * 0.58), horizon), (35, 33, 30), -1)
    return image


def as_jpeg(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


# ======================================================================================
# The style catalogue
# ======================================================================================


def test_five_styles_with_unique_ids_and_both_labels() -> None:
    assert len(restyle.STYLES) == 5
    assert len(set(restyle.STYLE_IDS)) == 5
    for style in restyle.STYLES:
        assert style.label_tr and style.label_en
        assert len(style.palette) > 30, f"{style.id} has no materials to work from"
        assert len(style.staging) > 20, f"{style.id} cannot furnish an empty room"
        assert restyle.style_by_id(style.id) is style
        assert restyle.style_by_id(style.id.upper()) is style


def test_a_palette_names_materials_rather_than_furniture() -> None:
    """The split the redesign turns on: the room owns the nouns, the style owns the finish.

    A palette that names a sofa will put a sofa in a bedroom, because in ``replace`` mode the
    prompt is the only thing telling the model what the hole it is filling was.
    """
    furniture = {
        "sofa", "couch", "bed", "chair", "armchair", "table", "desk", "wardrobe",
        "cabinet", "shelf", "shelving", "lamp", "rug", "sideboard", "console", "stool",
    }
    for style in restyle.STYLES:
        words = {w.strip(" ,") for w in style.palette.lower().split()}
        assert not (words & furniture), f"{style.id}'s palette names furniture: {words & furniture}"


def test_unknown_style_is_a_value_error() -> None:
    assert restyle.style_by_id("bauhaus") is None
    with pytest.raises(ValueError, match="unknown style"):
        restyle.restyle_photo(room_photo(), "bauhaus")


def test_every_prompt_fits_clip_or_it_is_silently_truncated() -> None:
    """The 77-token budget, asserted rather than hoped for. See this module's docstring."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting", subfolder="tokenizer"
        )
    except Exception as exc:  # noqa: BLE001 - no weights on this machine is not a failure
        pytest.skip(f"CLIP tokenizer unavailable: {exc}")

    # The worst case is `replace` mode with a full inventory: `_MAX_OBJECTS` nouns, each of
    # them one of the longer names in the vocabulary.
    inventory = f"{restyle._REPLACE_LEAD} " + ", ".join(
        ["chest of drawers"] * restyle._MAX_OBJECTS
    )
    for style in restyle.STYLES:
        for subject in (inventory, style.staging):
            prompt = f"{subject}, {style.palette}, {restyle._PROMPT_SUFFIX}"
            assert len(tokenizer(prompt).input_ids) <= 77, f"{style.id}: prompt would be truncated"
        negative = (
            f"{restyle._NEGATIVE_BASE}, {style.negative_extra}"
            if style.negative_extra
            else restyle._NEGATIVE_BASE
        )
        assert len(tokenizer(negative).input_ids) <= 77, f"{style.id}: negative would be truncated"


# ======================================================================================
# Geometry of the working size
# ======================================================================================


@pytest.mark.parametrize(
    ("width", "height"),
    [(1024, 1024), (1600, 1200), (1920, 1080), (1080, 1920), (800, 533)],
)
def test_work_size_is_stride_aligned_and_keeps_the_aspect(width: int, height: int) -> None:
    work_w, work_h = restyle._work_size(width, height)
    assert work_w % 8 == 0 and work_h % 8 == 0, "the VAE needs both sides divisible by 8"
    # Within one stride step of the source aspect: squashing a room bends its verticals.
    assert abs(work_w / work_h - width / height) < 0.05
    # And near the token budget the checkpoint was trained at, in either direction.
    assert 0.6 < (work_w * work_h) / restyle.WORK_PIXELS < 1.6


# ======================================================================================
# The conditioners
# ======================================================================================


def test_edges_are_found_on_a_low_contrast_interior() -> None:
    """The failure this replaced: fixed thresholds returned an all-but-empty map."""
    edges = restyle.structure_edges(room_photo())
    assert edges.any(), "an adaptive threshold must still find the doorway"
    assert 0.0005 < float((edges > 0).mean()) < 0.15, "architecture, not every texture"


def test_edges_outside_the_structure_mask_are_dropped() -> None:
    """A furnishing's outline must not condition the pass; see ``structure_edges``."""
    image = room_photo()
    # A high-contrast block standing in the room — a sofa, as far as Canny is concerned.
    cv2.rectangle(image, (40, 300), (200, 420), (20, 20, 220), -1)

    everything = restyle.structure_edges(image, None)
    structure = np.full(image.shape[:2], 255, np.uint8)
    structure[290:430, 30:210] = 0
    architecture_only = restyle.structure_edges(image, structure)

    assert everything[300:420, 40:200].any(), "the block does produce edges"
    assert not architecture_only[300:420, 40:200].any(), "and they must be masked away"


def test_synonym_matching_finds_windowpane_under_either_name() -> None:
    table = {0: "wall", 8: "windowpane, window", 14: "door", 99: "sofa, couch, lounge"}
    found = restyle._ids_named(table, frozenset({"window", "door"}))
    assert found == {8, 14}
    assert restyle._ids_named(table, frozenset({"sofa"})) == {99}


# ======================================================================================
# Determinism
# ======================================================================================


def test_the_seed_is_a_function_of_the_question() -> None:
    image = room_photo()
    mask = np.full(image.shape[:2], 255, np.uint8)

    assert restyle._seed_for(image, mask, "modern") == restyle._seed_for(image, mask, "modern")
    assert restyle._seed_for(image, mask, "modern") != restyle._seed_for(image, mask, "rustik")

    other = mask.copy()
    other[:20, :20] = 0
    assert restyle._seed_for(image, other, "modern") != restyle._seed_for(image, mask, "modern")

    moved = image.copy()
    moved[0, 0] = (1, 2, 3)
    assert restyle._seed_for(moved, mask, "modern") != restyle._seed_for(image, mask, "modern")


# ======================================================================================
# Analysis without a segmenter
# ======================================================================================


def test_analysis_degrades_rather_than_failing_without_a_segmenter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No segmenter is a degraded answer that says so; no depth is a refusal."""
    from biosphere.ai import backend

    image = room_photo()
    monkeypatch.setattr(backend, "segment_photo", lambda _image: None)
    monkeypatch.setattr(restyle, "depth_map", lambda _image: np.full(image.shape[:2], 128, np.uint8))

    room = restyle.analyse_room(image)
    assert room is not None
    assert room.segmented is False
    assert room.protected_fraction == 0.0
    assert (room.editable == 255).all(), "with nothing known to protect, everything is editable"

    monkeypatch.setattr(restyle, "depth_map", lambda _image: None)
    assert restyle.analyse_room(image) is None, "structure conditioning is not optional"


def test_protected_pixels_come_back_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """The composite keeps the product's promise exactly, not approximately.

    The pipeline is replaced with something that paints the whole frame a flat colour — the
    most destructive answer a sampler could give — so anything of the original that survives
    survived because the composite put it back.
    """
    image = room_photo()
    height, width = image.shape[:2]

    protect = np.zeros((height, width), np.uint8)
    protect[60:200, 400:600] = 255  # a "window"
    alpha = restyle.feathered_alpha(protect, 2.0)
    room = restyle.RoomAnalysis(
        depth=np.full((height, width), 128, np.uint8),
        edges=np.zeros((height, width), np.uint8),
        editable=255 - protect,
        protect_alpha=alpha,
        protected_fraction=float((protect > 0).mean()),
        segmented=True,
    )

    class _FlatPipe:
        def set_progress_bar_config(self, **_: Any) -> None: ...

        def __call__(self, **kwargs: Any) -> Any:
            from PIL import Image

            painted = Image.new("RGB", (kwargs["width"], kwargs["height"]), (255, 0, 255))
            return type("Out", (), {"images": [painted]})()

    monkeypatch.setattr(restyle, "_load", lambda: (_FlatPipe(), "cpu"))

    result = restyle.restyle_photo(image, "modern", analysis=room)
    assert result is not None

    solid = protect > 0
    assert solid.any()
    assert (alpha[solid] == 1.0).all(), "the feather must not reach inside the protected region"
    assert np.array_equal(result.image[solid], image[solid]), "a protected pixel must be the original"
    # And the rest genuinely was replaced, so the test is not passing by doing nothing.
    assert not np.array_equal(result.image[alpha < 0.01], image[alpha < 0.01])


# ======================================================================================
# API
# ======================================================================================


def test_style_catalogue_is_served_without_loading_a_model(client: TestClient) -> None:
    response = client.get(f"{API}/ai/restyle/styles")
    assert response.status_code == 200
    body = response.json()
    assert [s["id"] for s in body["styles"]] == list(restyle.STYLE_IDS)
    assert all(s["label_tr"] for s in body["styles"])
    assert isinstance(body["available"], bool)


def test_uploading_the_same_photograph_twice_gives_one_id(client: TestClient) -> None:
    payload = as_jpeg(room_photo())
    first = client.post(
        f"{API}/ai/restyle/photos", files={"file": ("room.jpg", io.BytesIO(payload), "image/jpeg")}
    )
    second = client.post(
        f"{API}/ai/restyle/photos", files={"file": ("again.jpg", io.BytesIO(payload), "image/jpeg")}
    )
    assert first.status_code == 201 and second.status_code == 201
    assert first.json()["photo_id"] == second.json()["photo_id"]
    assert first.json()["ready_styles"] == []
    assert first.json()["width"] == 640


def test_a_photograph_larger_than_the_cap_is_stored_scaled(client: TestClient) -> None:
    from biosphere.api.routes.restyle import _MAX_SIDE

    response = client.post(
        f"{API}/ai/restyle/photos",
        files={"file": ("big.jpg", io.BytesIO(as_jpeg(room_photo(4000, 3000))), "image/jpeg")},
    )
    assert response.status_code == 201
    assert max(response.json()["width"], response.json()["height"]) == _MAX_SIDE


def test_a_file_that_is_not_an_image_is_refused(client: TestClient) -> None:
    response = client.post(
        f"{API}/ai/restyle/photos",
        files={"file": ("room.jpg", io.BytesIO(b"this is not a jpeg"), "image/jpeg")},
    )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_media"
    assert response.json()["message_tr"]


def test_an_unknown_photo_is_a_404_not_a_crash(client: TestClient) -> None:
    response = client.post(
        f"{API}/ai/restyle", json={"photo_id": "0" * 16, "style": "modern"}
    )
    assert response.status_code == 404
    assert response.json()["code"] == "photo_not_found"


def test_an_unknown_style_is_refused_before_any_work(client: TestClient) -> None:
    payload = as_jpeg(room_photo())
    photo = client.post(
        f"{API}/ai/restyle/photos", files={"file": ("room.jpg", io.BytesIO(payload), "image/jpeg")}
    ).json()
    response = client.post(
        f"{API}/ai/restyle", json={"photo_id": photo["photo_id"], "style": "art-deco"}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "unknown_style"


@pytest.mark.parametrize(
    "candidate",
    ["../../etc", "..", "a" * 15, "a" * 17, "ZZZZZZZZZZZZZZZZ", "0123456789abcde/", ""],
)
def test_a_photo_id_that_could_leave_the_tree_is_rejected(candidate: str) -> None:
    from biosphere.api.routes.restyle import _valid_id

    with pytest.raises(ApiException) as raised:
        _valid_id(candidate)
    assert raised.value.code == "invalid_photo_id"


def test_a_well_formed_photo_id_passes() -> None:
    from biosphere.api.routes.restyle import _valid_id

    assert _valid_id("0123456789ABCDEF") == "0123456789abcdef"


# ======================================================================================
# The sphere
# ======================================================================================


def test_the_pole_guard_is_a_blending_weight_not_a_threshold() -> None:
    """Latitudes, not pixel counts, and a continuous weight rather than a mask.

    The distinction is the bug this replaced: as a mask the feather decided WHICH pixels were
    restyled instead of HOW STRONGLY, and the boundary came out dithered. See the module
    docstring for the numbers.
    """
    from biosphere.ai import restyle_pano as rp

    height, width = 256, 512
    guard = rp._pole_guard(height, width)
    assert guard.shape == (height, width)
    assert guard.dtype.kind == "f", "a weight, not a mask"
    assert 0.0 <= guard.min() and guard.max() <= 1.0
    # Every column is the same: the guard is a function of elevation alone.
    assert (guard == guard[:, :1]).all()

    def at(elevation_deg: float) -> float:
        row = int(round((90.0 - elevation_deg) / 180.0 * height))
        return float(guard[min(max(row, 0), height - 1), 0])

    assert at(0.0) == 1.0, "the horizon is always restylable"
    assert at(89.0) == 0.0, "the zenith is held"
    assert at(-89.0) == 0.0, "so is the nadir"

    # The join is a RAMP, and a monotone one: sampled across the feather it must never go back
    # up, or the composite would show a band brighter than both of its neighbours.
    inner = rp.POLE_PROTECT_DEG - rp._POLE_FEATHER_DEG
    for sign in (1.0, -1.0):
        walk = [at(sign * (inner + step)) for step in np.linspace(0.0, rp._POLE_FEATHER_DEG, 9)]
        assert walk[0] > 0.9 and walk[-1] < 0.1, f"no feather at {sign:+.0f} pole"
        pairs = zip(walk[:-1], walk[1:], strict=True)
        assert all(b <= a + 1e-6 for a, b in pairs), "ramp is not monotone"


def test_the_windows_walk_the_ring_rather_than_jumping_about() -> None:
    """Order is the only source of context here; see the module docstring."""
    from biosphere.ai import restyle_pano as rp

    order = rp._ring_order(rp.VIEW_PLAN)
    assert len(order) == sum(n for _, n in rp.VIEW_PLAN)

    for pitch_deg, n_yaw in rp.VIEW_PLAN:
        row = [yaw for yaw, pitch in order if pitch == pitch_deg]
        assert len(row) == n_yaw
        assert row == sorted(row), "a row must be walked in one direction"
        gaps = {round(row[i + 1] - row[i], 3) for i in range(len(row) - 1)}
        assert len(gaps) == 1, "evenly spaced, so every window has one fresh neighbour"
        # Neighbours must overlap: a gap at or beyond the field of view leaves a gap in the
        # sphere and, worse, two windows with nothing in common to agree about.
        assert gaps.pop() < 90.0


def test_solid_angle_weights_sum_to_one_and_favour_the_equator() -> None:
    from biosphere.ai import restyle_pano as rp

    weights = rp._solid_angle_weights(128, 256)
    assert abs(float(weights.sum()) - 1.0) < 1e-4
    equator = float(weights[64, 0])
    near_pole = float(weights[1, 0])
    assert equator > near_pole * 20, "a row by the pole covers almost none of the sphere"


def test_an_unknown_style_is_refused_before_any_window_is_rendered() -> None:
    from biosphere.ai import restyle_pano as rp

    with pytest.raises(ValueError, match="unknown style"):
        rp.restyle_equirect(np.zeros((64, 128, 3), np.uint8), "art-deco")


def test_an_analysis_from_another_image_is_refused_by_name() -> None:
    """`analysis` is a public parameter and reusing one across images is the natural misuse."""
    small = room_photo(320, 240)
    big = room_photo(640, 480)
    room = restyle.RoomAnalysis(
        depth=np.full(small.shape[:2], 128, np.uint8),
        edges=np.zeros(small.shape[:2], np.uint8),
        editable=np.full(small.shape[:2], 255, np.uint8),
        protect_alpha=np.zeros(small.shape[:2], np.float32),
        protected_fraction=0.0,
        segmented=True,
    )
    with pytest.raises(ValueError, match="belongs to one photograph"):
        restyle.restyle_photo(big, "modern", analysis=room)


# ======================================================================================
# The two modes
# ======================================================================================


def _fake_table() -> dict[int, str]:
    """A slice of ADE20K big enough to exercise the three groups."""
    return {
        0: "wall",
        1: "building, edifice",
        2: "sky",
        3: "floor, flooring",
        5: "ceiling",
        8: "windowpane, window",
        14: "door, double door",
        18: "curtain, drape, drapery",
        7: "bed",
        19: "chair",
        23: "sofa, couch, lounge",
        35: "wardrobe, closet, press",
        36: "lamp",
        135: "vase",
    }


def _labelled(height: int, width: int, plan: list[tuple[int, tuple[int, int, int, int]]]) -> np.ndarray:
    labels = np.zeros((height, width), np.int32)  # 0 is wall
    for class_id, (x0, y0, x1, y1) in plan:
        labels[y0:y1, x0:x1] = class_id
    return labels


def _analysis(monkeypatch: pytest.MonkeyPatch, image: np.ndarray, labels: np.ndarray, **kwargs):
    from biosphere.ai import backend

    monkeypatch.setattr(backend, "segment_photo", lambda _i: labels)
    monkeypatch.setattr(restyle, "_label_table", _fake_table)
    monkeypatch.setattr(restyle, "depth_map", lambda i: np.full(i.shape[:2], 128, np.uint8))
    return restyle.analyse_room(image, **kwargs)


def test_a_furnished_room_restyles_its_furniture_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complaint this design answers: a wardrobe was drawn across a curtained window."""
    image = room_photo(640, 480)
    labels = _labelled(
        480,
        640,
        [
            (3, (0, 330, 640, 480)),      # floor
            (5, (0, 0, 640, 70)),         # ceiling
            (8, (430, 90, 600, 300)),     # window
            (18, (410, 80, 620, 320)),    # curtain over it
            (7, (40, 200, 300, 420)),     # a bed
            (35, (320, 120, 400, 380)),   # a wardrobe
        ],
    )
    room = _analysis(monkeypatch, image, labels)
    assert room is not None
    assert room.mode == "replace"

    editable = room.editable > 127
    assert editable[300, 150], "the bed must be restylable"
    assert editable[250, 360], "so must the wardrobe"
    assert not editable[200, 500], "the curtain over the window must NOT be"
    assert not editable[400, 320], "nor the floor"
    assert not editable[30, 320], "nor the ceiling"
    assert not editable[150, 200], "nor the bare wall"

    # And the prompt's nouns are what is actually in the room.
    assert "bed" in room.objects
    assert "wardrobe" in room.objects
    assert "curtain" not in room.objects and "wall" not in room.objects


def test_an_empty_room_is_staged_from_the_floor_upward(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing to replace the question changes, and so does the mask."""
    image = room_photo(640, 480)
    labels = _labelled(
        480,
        640,
        [
            (3, (0, 330, 640, 480)),   # floor
            (5, (0, 0, 640, 70)),      # ceiling
            (8, (430, 90, 600, 300)),  # window, nothing else in the room
        ],
    )
    room = _analysis(monkeypatch, image, labels)
    assert room is not None
    assert room.mode == "stage"
    assert room.objects == (), "there is nothing to inventory"

    editable = room.editable > 127
    assert editable[400, 200], "the floor is where furniture goes"
    assert editable[280, 200], "and the band of wall just above it"
    assert not editable[90, 200], "but not the upper wall — that is where windows get invented"
    assert not editable[200, 500], "and never the window"


def test_the_window_stays_protected_even_when_dressing_is_opted_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`window_dressing=False` frees the curtain. It must never free the glass."""
    image = room_photo(640, 480)
    labels = _labelled(
        480,
        640,
        [
            (3, (0, 330, 640, 480)),
            (8, (430, 90, 600, 300)),
            (18, (150, 80, 300, 320)),   # a curtain on a blank stretch of wall
            (23, (40, 250, 300, 420)),   # and a sofa, so the room reads as furnished
        ],
    )
    freed = _analysis(monkeypatch, image, labels, window_dressing=False)
    held = _analysis(monkeypatch, image, labels, window_dressing=True)
    assert freed is not None and held is not None

    assert (freed.editable > 127)[200, 220], "opting out must free the curtain"
    assert not (held.editable > 127)[200, 220], "the default must hold it"
    for room in (freed, held):
        assert not (room.editable > 127)[200, 500], "the windowpane is never editable"


def test_replace_mode_uses_a_stronger_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old default was cautious because the walls were editable. Now they are not."""
    assert restyle.REPLACE_STRENGTH > restyle.STRENGTH

    captured: dict = {}

    class _Pipe:
        def set_progress_bar_config(self, **_: Any) -> None: ...

        def __call__(self, **kwargs: Any) -> Any:
            from PIL import Image

            captured.update(kwargs)
            return type(
                "Out", (), {"images": [Image.new("RGB", (kwargs["width"], kwargs["height"]))]}
            )()

    monkeypatch.setattr(restyle, "_load", lambda: (_Pipe(), "cpu"))
    image = room_photo(640, 480)
    height, width = image.shape[:2]
    editable = np.zeros((height, width), np.uint8)
    editable[200:400, 100:400] = 255

    for mode, expected in (("replace", restyle.REPLACE_STRENGTH), ("stage", restyle.STRENGTH)):
        room = restyle.RoomAnalysis(
            depth=np.full((height, width), 128, np.uint8),
            edges=np.zeros((height, width), np.uint8),
            editable=editable,
            protect_alpha=restyle.feathered_alpha(255 - editable, 2.0),
            protected_fraction=float((editable == 0).mean()),
            segmented=True,
            mode=mode,  # type: ignore[arg-type]
            objects=("bed",) if mode == "replace" else (),
        )
        result = restyle.restyle_photo(image, "modern", analysis=room)
        assert result is not None
        assert result.strength == expected
        if mode == "replace":
            assert captured["prompt"].startswith(f"{restyle._REPLACE_LEAD} bed"), captured["prompt"]
        else:
            assert restyle.style_by_id("modern").staging.split(",")[0] in captured["prompt"]
