"""Naming the phone, and pricing its lens."""

from __future__ import annotations

import json
import math

import pytest

from biosphere.pano import devices as dv
from biosphere.pano.types import Intrinsics


def _sidecar(**device) -> dict:
    return {"schema": "biosphere.capture.v1", "device": device}


def test_the_equivalence_is_the_diagonal_one():
    """35mm-equivalent is defined on the diagonal, so the aspect ratio must not enter.

    The same lens on a 4:3 frame and on the 3:4 portrait rotation of it is one focal length
    in pixels. Getting this wrong by using the 36 mm frame WIDTH instead is an 8% error,
    which is the size of the bias the whole module exists to remove.
    """
    landscape = dv.focal_px_from_equiv_mm(26.8, 1440, 1080)
    portrait = dv.focal_px_from_equiv_mm(26.8, 1080, 1440)
    assert landscape == pytest.approx(portrait, rel=1e-12)
    # And the horizontal field follows the frame, not the lens.
    assert dv.hfov_deg_from_focal(portrait, 1080) == pytest.approx(51.7, abs=0.2)
    assert dv.hfov_deg_from_focal(landscape, 1440) == pytest.approx(65.7, abs=0.3)


def test_the_reference_phone_agrees_with_its_own_capture():
    """The vivo Y22s's table entry against what the engine measured from its footage.

    A slow, low-parallax capture from that phone self-calibrated to 1115.69 px on a
    1080x1440 frame. The table says 26.8 mm, measured through Camera2 on the device itself.
    If these two ever drift apart, one of them is wrong and this test is the alarm.
    """
    resolved = dv.resolve_device(
        _sidecar(platform="android", modelHints={"uaModel": "V2206", "brand": "vivo"}, trackSettings={"zoom": 1}),
        width=1080,
        height=1440,
    )
    assert resolved.profile.id == "vivo-y22s"
    assert resolved.source == "ua-model"
    assert resolved.focal_px == pytest.approx(1115.69, rel=0.01)
    assert resolved.profile.ois is False


def test_an_iphone_is_identified_as_a_family_never_as_a_phone():
    """One screen, four phones, two different lenses — so the answer carries its own spread."""
    resolved = dv.resolve_device(
        _sidecar(platform="ios", modelHints={"screen": {"cssWidth": 393, "cssHeight": 852, "devicePixelRatio": 3}}),
        width=1080,
        height=1440,
    )
    assert resolved.source == "ios-fingerprint"
    assert len(resolved.candidates) > 1
    # The midpoint of a group holding 24 mm and 26 mm lenses, not either end of it.
    assert 24.0 <= resolved.profile.video_equiv_mm <= 26.0
    assert resolved.sigma_fraction >= 0.06
    # Orientation is the user's business: the same phone on its side is the same phone.
    rotated = dv.resolve_device(
        _sidecar(platform="ios", modelHints={"screen": {"cssWidth": 852, "cssHeight": 393, "devicePixelRatio": 3}}),
        width=1080,
        height=1440,
    )
    assert rotated.profile.id == resolved.profile.id


def test_lidar_is_only_claimed_when_every_member_of_the_family_has_it():
    """A maybe is reported as absent: a measurement offered on a maybe is worse than none."""
    ambiguous = dv.resolve_device(
        _sidecar(platform="ios", modelHints={"screen": {"cssWidth": 390, "cssHeight": 844, "devicePixelRatio": 3}}),
        width=1080,
        height=1440,
    )
    assert ambiguous.profile.lidar is False, "the 12/12 Pro/13/13 Pro/14 group cannot answer this"
    certain = dv.resolve_device(
        _sidecar(platform="ios", modelHints={"screen": {"cssWidth": 440, "cssHeight": 956, "devicePixelRatio": 3}}),
        width=1080,
        height=1440,
    )
    assert certain.profile.lidar is True


def test_an_unknown_phone_still_beats_the_67_degree_assumption():
    resolved = dv.resolve_device(_sidecar(platform="android", trackSettings={}), width=1080, height=1440)
    assert resolved.profile.id == "generic-phone"
    assert resolved.source == "generic"
    # The default the pipeline uses when it knows nothing is 67 degrees, which is out by a
    # factor of 1.5 on a portrait phone frame. Anything in the 45-60 range is an improvement.
    assert 45.0 < resolved.hfov_deg < 60.0
    assert resolved.sigma_fraction >= 0.12, "an unidentified phone must be held loosely"


def test_no_sidecar_means_no_opinion():
    """An upload from somewhere else must behave exactly as it did before this existed."""
    resolved = dv.resolve_device(None, width=1080, height=1440)
    assert resolved.source == "none"


def test_a_virtual_lens_device_widens_the_band():
    tight = dv.resolve_device(
        _sidecar(platform="android", modelHints={"uaModel": "V2206"}, trackSettings={"zoom": 1}),
        width=1080,
        height=1440,
    )
    loose = dv.resolve_device(
        _sidecar(platform="android", modelHints={"uaModel": "V2206"}, trackSettings={"zoom": 1}, virtualLensOnly=True),
        width=1080,
        height=1440,
    )
    assert loose.sigma_fraction > tight.sigma_fraction


def test_zoom_below_one_selects_the_ultrawide():
    """A pinned-to-minimum zoom on a multi-camera device is a different lens, not a crop."""
    resolved = dv.resolve_device(
        _sidecar(platform="android", modelHints={"uaModel": "SM-S911B"}, trackSettings={"zoom": 0.6}),
        width=1080,
        height=1440,
    )
    assert resolved.lens == "ultrawide"
    assert resolved.hfov_deg is not None and resolved.hfov_deg > 80.0


def test_an_ultrawide_named_by_its_label_is_priced_as_the_ultrawide():
    """iOS exposes the 13 mm lens as its own camera, whose zoom reads 1.

    The zoom test cannot see that, and pricing the capture for the main lens would hand the
    bundle a band around HALF the real field of view -- which it then enforces. The label is
    what says which lens recorded.
    """
    main = dv.resolve_device(
        _sidecar(platform="android", modelHints={"uaModel": "SM-S911B"}, trackSettings={"zoom": 1},
                 cameraLabel="camera2 0, facing back"),
        width=1080,
        height=1440,
    )
    ultra = dv.resolve_device(
        _sidecar(platform="ios", modelHints={"uaModel": "SM-S911B"}, trackSettings={"zoom": 1},
                 cameraLabel="Back Ultra Wide Camera"),
        width=1080,
        height=1440,
    )
    assert main.lens == "main" and ultra.lens == "ultrawide"
    assert ultra.hfov_deg is not None and ultra.hfov_deg > 80.0
    assert "camera_label_names_the_ultrawide" in ultra.notes


def test_an_unlisted_ultrawide_still_gets_an_ultrawide_prior_held_loosely():
    unknown = dv.resolve_device(
        _sidecar(platform="android", modelHints={"uaModel": "NOPE-123"}, trackSettings={"zoom": 1},
                 cameraLabel="Ultra Wide camera, facing back"),
        width=1080,
        height=1440,
    )
    assert unknown.lens == "ultrawide"
    assert unknown.hfov_deg is not None and 80.0 < unknown.hfov_deg < 100.0
    assert unknown.sigma_fraction >= 0.12


def test_a_main_lens_called_wide_angle_is_not_mistaken_for_the_ultrawide():
    resolved = dv.resolve_device(
        _sidecar(platform="android", modelHints={"uaModel": "V2206"}, trackSettings={"zoom": 1},
                 cameraLabel="Wide Angle Camera, facing back"),
        width=1080,
        height=1440,
    )
    assert resolved.lens == "main"


def test_the_table_is_well_formed():
    profiles, fingerprints = dv.load_profiles()
    assert len(profiles) > 40
    seen_aliases: dict[str, str] = {}
    for profile in profiles.values():
        assert 10.0 < profile.video_equiv_mm < 60.0, profile.id
        assert 0.0 < profile.sigma_fraction <= 0.3, profile.id
        for alias in profile.aliases:
            assert alias == alias.strip().lower()
            # One SKU cannot be two phones; a duplicate would silently pick the first.
            assert alias not in seen_aliases, f"{alias} claimed by {seen_aliases.get(alias)} and {profile.id}"
            seen_aliases[alias] = profile.id
    for entry in fingerprints:
        assert entry["lidar"] in ("yes", "no", "maybe")
        assert entry["models"]


def test_the_band_matches_what_the_bundle_would_apply():
    """The number the solve is fenced by, spelled out once so a refactor cannot drift it."""
    resolved = dv.resolve_device(
        _sidecar(platform="android", modelHints={"uaModel": "V2206"}, trackSettings={"zoom": 1}),
        width=1080,
        height=1440,
    )
    lo = resolved.focal_px * (1.0 - 2.0 * resolved.sigma_fraction)
    hi = resolved.focal_px * (1.0 + 2.0 * resolved.sigma_fraction)
    # The reference capture's free solve wanted 1286 px; the band is what pulls it back.
    assert lo < 1000.0 and 1200.0 < hi < 1290.0
    intr = Intrinsics(focal=hi, cx=540, cy=720, width=1080, height=1440)
    assert 45.0 < intr.hfov_deg < 50.0


def test_the_capture_apps_copy_of_the_table_is_current():
    """The app's generated table must match the engine's, or the two coach and solve apart.

    The capture app uses the same lens table to pick a preset and to tell the user whether
    their phone has optical stabilisation. If the generated copy went stale, the app would
    coach for one lens while the engine bounded its focal search with another, and nothing
    else in the suite would notice. Regenerate with:

        engine/.venv/Scripts/python.exe engine/scripts/gen_device_profiles.py
    """
    import subprocess
    import sys
    from pathlib import Path

    engine = Path(__file__).resolve().parent.parent
    target = engine.parent / "web" / "src" / "lib" / "capture" / "deviceProfiles.generated.ts"
    if not target.exists():
        pytest.skip("the web app is not checked out beside the engine")
    before = target.read_text(encoding="utf-8")
    subprocess.run(
        [sys.executable, str(engine / "scripts" / "gen_device_profiles.py")],
        check=True,
        capture_output=True,
    )
    assert target.read_text(encoding="utf-8") == before, (
        "web/src/lib/capture/deviceProfiles.generated.ts is out of date; run "
        "engine/scripts/gen_device_profiles.py"
    )
