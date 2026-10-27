"""What the phone that shot the capture tells us about its own lens.

Why this file exists
--------------------
A browser cannot ask a camera what its field of view is. `MediaTrackSettings` carries
width, height, frame rate, zoom, ISO and exposure time — and nothing at all about the
optics; `focalLength` is in a draft of the image-capture extensions that no shipping
browser implements for a video track. So the capture app has always written a guessed
``optics.hfovDeg`` into its sidecar, and the engine has always ignored it and measured the
focal length from the imagery instead.

Measuring it is the right instinct and it is not enough, because the measurement is
BIASED, one-sidedly, by the thing this whole pipeline is built around. A hand-held sweep
turns about the shoulder, not the lens, so every direction is seen from a slightly
different point. The tangential part of that motion adds image flow in the same direction
as the rotation, and a rotation-only model can only close a 360 degree turn by making the
focal length longer than it is — by roughly ``1 + r/d`` for an arm of radius ``r`` in a
room of depth ``d``. It cannot come out short. Measured on the same phone (a vivo Y22s),
two captures of the same bedroom:

    capture                     parallax p90   solved hfov   implied 35mm equivalent
    2026-09-03, wide arm            27.3 px       45.2 deg          31.2 mm
    2026-09-06, tighter arm         19.5 px       51.7 deg          26.8 mm
    the lens itself, measured           --        51.7 deg          26.8 mm

Six and a half degrees apart, from one lens that never changed — and the slower, tighter
capture landed within 0.03 degrees of the lens's own figure, which is the cross-check that
says the bias story is right rather than merely plausible. (The lens figure is not a spec
sheet: the Camera2 API on that phone reports a 65.7 degree horizontal field on its 4:3
still, which is 26.8 mm on the diagonal convention and 51.7 degrees across the 3:4 frame
the app records.) The narrower answer is always the wrong one, and a focal that is too long
bows straight lines outward and makes the room come out smaller than it is — exactly the
"visual distortion" a customer sees.

Bounding it, not fixing it, and the difference is not a detail. Measured on a synthetic
capture with a 0.28 m arm, where the true focal is known to be 579.4 px and the free solve
gives 655.4: pinning the focal AT THE TRUE VALUE made the panorama worse than leaving it
alone (horizon PSNR 18.52 -> 17.18 dB, mean pose error 1.52 -> 2.51 deg). Under real
parallax the geometrically true focal is not the best mosaic focal — the inflated one is a
better approximation of a camera that translated. A two-sigma band at sigma = 0.06 was the
best of the sweep (PSNR 18.77, pose 1.28, loop residual 0.153 -> 0.083 deg); tighter bands
pushed the wrap column past its own gate. So the lens gives the solve a fence, never a
destination.

A lens is a constant of the hardware. If we can name the hardware we can look the number
up, and then the solve has something to be pulled back toward rather than only its own
biased optimum. That is what this module is for: name the phone, look up the lens, and
hand the rest of the pipeline a prior with an honest uncertainty attached.

Naming the phone from a browser
-------------------------------
* **Android** gives it to us directly. `navigator.userAgentData.getHighEntropyValues(
  ["model"])` returns the marketing-ish model string ("vivo 1234", "SM-S911B", "Pixel 8").
  The user agent itself is useless — Chrome froze it at "Android 10; K" in 2022, which is
  exactly what the reference capture's sidecar says.
* **iOS** never gives it. Safari reports no model, and the WebGL renderer string has been
  "Apple GPU" for every device since iOS 12. What is left is the screen: the CSS viewport
  size in points together with the device pixel ratio separates iPhones into small groups,
  and those groups are what the table's ``ios_fingerprints`` map. Several models share a
  fingerprint, so the answer is a group, not a phone, and a group is not always of one mind:
  393x852 at 3x covers the 14 Pro and 15 Pro at 24 mm and the 15 and 16 at 26. The
  fingerprint therefore carries the family's midpoint and a sigma wide enough to span it.
* **Neither** still leaves us better off than before. Every phone main camera sold since
  about 2018 sits between 23 and 27 mm equivalent; :data:`GENERIC_PHONE` says 26 mm with a
  wide uncertainty, which on a 4:3 portrait frame is 53 degrees across — against the 67
  degrees the pipeline assumes when it knows nothing, an error of a factor of 1.5.

From an equivalent focal length to pixels
-----------------------------------------
Phone specifications quote the CIPA 35mm-equivalent focal length, which is defined on the
*diagonal* of a 36x24 mm frame. That definition is what makes the conversion so simple:
the frame's diagonal subtends ``2 atan(21.634 / f35)`` whatever its aspect ratio, so

    focal_px = f35 * diagonal_px / 43.267

with no dependence on how the frame is cropped, as long as ``f35`` is quoted for the frame
we actually received — and that proviso is a trap worth naming. Manufacturers quote the
still frame; 16:9 video is a crop of it and wants a factor of about 1.1. The capture app
records 1080x1440, which is the sensor's own 4:3 aspect, so no crop applies and the still
figure is the right one. It is also the one the reference device's measurement confirms. If
the app ever records 16:9, every number in the table needs that factor.

What this module does NOT do
----------------------------
It does not decide anything on its own. It returns a :class:`ResolvedDevice` carrying a
focal length, a relative uncertainty and where the answer came from; every caller decides
for itself how much to lean on it. A profile is never allowed to override a measurement,
only to bound it — see ``PipelineConfig.focal_prior_px``, the band applied in
``bundle._robust_lm``, and the disagreement warnings in ``solve.solve_capture``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import structlog

__all__ = [
    "DeviceProfile",
    "ResolvedDevice",
    "GENERIC_PHONE",
    "load_profiles",
    "resolve_device",
    "focal_px_from_equiv_mm",
    "FILM_DIAGONAL_MM",
]

log = structlog.get_logger(__name__)

#: Diagonal of a 36 x 24 mm frame, the reference the 35mm-equivalent focal length is
#: defined against (CIPA DCG-001). Every conversion in this module goes through it.
FILM_DIAGONAL_MM = math.hypot(36.0, 24.0)  # 43.267

#: Where the table lives. Data, not code, so that adding a phone is a one-line change and
#: the same file can be generated into the capture app (see scripts/gen_device_profiles.py).
_DATA_PATH = Path(__file__).resolve().parent.parent / "assets" / "device_profiles.json"

Tier = Literal["flagship", "upper-mid", "mid", "budget"]
Source = Literal["ua-model", "ios-fingerprint", "brand", "generic", "none"]


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """One phone's rear-camera geometry, as a constant of the hardware."""

    id: str
    brand: str
    model: str
    tier: Tier
    #: 35mm-equivalent focal length of the lens the capture app selects (the 1x main
    #: camera), for VIDEO. Video is normally a small crop of the still frame.
    video_equiv_mm: float
    #: The ultrawide's equivalent focal, when the phone has one. Only consulted when the
    #: capture reports a zoom factor below 1, which is what a "virtual" multi-camera device
    #: does when the app pins zoom to its minimum.
    ultrawide_equiv_mm: float | None = None
    ois: bool = False
    lidar: bool = False
    sensor: str = ""
    #: Exact `navigator.userAgentData` model strings, lower-cased, that mean this phone.
    aliases: tuple[str, ...] = ()
    #: Relative one-sigma uncertainty of ``video_equiv_mm``. 0.04 is a published, verified
    #: number; 0.10 is "this is the family's usual lens and the variant list is fuzzy".
    sigma_fraction: float = 0.06
    notes: str = ""

    def equiv_mm_for(
        self, zoom: float | None, *, ultrawide_selected: bool = False
    ) -> tuple[float, Literal["main", "ultrawide"]]:
        """The equivalent focal in play, given the zoom factor the capture reported.

        ``ultrawide_selected`` is the other way a capture says it used the 13 mm lens: by
        NAME. iOS exposes the ultra-wide as its own physical camera ("Back Ultra Wide
        Camera"), whose zoom reads 1 because it is that lens at its own 1x -- so the zoom
        test above cannot see it, and the prior would be priced for the main lens: half
        the field of view, which the focal band would then enforce on the solve.
        """
        if self.ultrawide_equiv_mm is not None and (
            ultrawide_selected or (zoom is not None and zoom < 0.95)
        ):
            return float(self.ultrawide_equiv_mm), "ultrawide"
        if ultrawide_selected:
            # A phone whose table entry knows no ultra-wide, filmed through one anyway:
            # the industry's 13-14 mm, held loosely by the caller.
            return GENERIC_ULTRAWIDE_EQUIV_MM, "ultrawide"
        return float(self.video_equiv_mm), "main"


#: Phone ultra-wides have converged too, on 13-14 mm (iPhone 11-16, Galaxy S2x, Pixel 6-9
#: all quote 13 mm; a few mid-range Androids 16 mm). Used only when the capture names an
#: ultra-wide its profile does not list, with ``_ULTRAWIDE_UNLISTED_SIGMA``.
GENERIC_ULTRAWIDE_EQUIV_MM = 13.5
_ULTRAWIDE_UNLISTED_SIGMA = 0.12
#: Camera labels that name the ultra-wide. iOS: "Back Ultra Wide Camera"; Android OEMs that
#: expose it as its own device mostly say "ultra" too. "Wide" alone is NOT one of them: several
#: Android OEMs label the MAIN lens "Wide Angle" (see `useCamera.pickRearCamera`).
_ULTRAWIDE_LABEL = ("ultra wide", "ultrawide", "ultra-wide")


def label_names_ultrawide(label: str | None) -> bool:
    """True when a camera label says the ultra-wide was the lens that recorded."""
    text = str(label or "").strip().lower()
    return any(token in text for token in _ULTRAWIDE_LABEL)


#: The fallback, and a good one. Phone main cameras have converged: 23 mm (a few wide
#: flagships) to 27 mm (older mid-range), overwhelmingly 26. The 12% sigma spans that whole
#: range, so a capture from an unknown phone gets a prior that is loose but never wrong by
#: the factor of 1.5 the 67-degree assumption is wrong by.
GENERIC_PHONE = DeviceProfile(
    id="generic-phone",
    brand="",
    model="phone main camera",
    tier="mid",
    video_equiv_mm=26.0,
    ois=False,
    lidar=False,
    sigma_fraction=0.12,
    notes="No model could be identified; the industry-standard main camera is assumed.",
)


@dataclass(slots=True)
class ResolvedDevice:
    """What we believe about the lens that shot this capture, and how firmly."""

    profile: DeviceProfile
    source: Source
    #: Focal length in the DELIVERED frame's pixels, or None when the frame size is unknown.
    focal_px: float | None = None
    hfov_deg: float | None = None
    #: Relative one-sigma uncertainty of ``focal_px``. Callers turn this into a weight.
    sigma_fraction: float = 0.12
    lens: Literal["main", "ultrawide", "unknown"] = "main"
    #: Model strings that matched the same fingerprint. More than one means the answer is
    #: a family, which is still enough for a field of view.
    candidates: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def identified(self) -> bool:
        """True when a specific phone (or a small family of them) was named."""
        return self.source in ("ua-model", "ios-fingerprint")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.profile.id,
            "brand": self.profile.brand,
            "model": self.profile.model,
            "tier": self.profile.tier,
            "source": self.source,
            "lens": self.lens,
            "focal_px": None if self.focal_px is None else round(self.focal_px, 2),
            "hfov_deg": None if self.hfov_deg is None else round(self.hfov_deg, 2),
            "sigma_fraction": round(self.sigma_fraction, 4),
            "ois": self.profile.ois,
            "lidar": self.profile.lidar,
            "candidates": list(self.candidates),
            "notes": list(self.notes),
        }


def focal_px_from_equiv_mm(equiv_mm: float, width: int, height: int) -> float:
    """Focal length in pixels for a frame of ``width`` x ``height``.

    ``f_px = f35 * diagonal_px / 43.267``. The 35mm equivalent is a statement about the
    diagonal field of view, so the frame's own aspect ratio never enters — which is what
    makes this safe for a portrait video from a 4:3 sensor.
    """
    diagonal_px = math.hypot(float(width), float(height))
    return float(equiv_mm) * diagonal_px / FILM_DIAGONAL_MM


def hfov_deg_from_focal(focal_px: float, width: int) -> float:
    return math.degrees(2.0 * math.atan((float(width) / 2.0) / float(focal_px)))


# ======================================================================================
# The table
# ======================================================================================


@lru_cache(maxsize=1)
def load_profiles() -> tuple[dict[str, DeviceProfile], tuple[dict[str, Any], ...]]:
    """``(profiles by id, iOS fingerprints)``, read once from the JSON beside this package.

    A missing or malformed file is not fatal: the pipeline then behaves exactly as it did
    before this module existed, with :data:`GENERIC_PHONE` as the only profile.
    """
    try:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pragma: no cover - packaging accident
        log.warning("devices.table_unavailable", path=str(_DATA_PATH), error=str(exc))
        return {GENERIC_PHONE.id: GENERIC_PHONE}, ()

    profiles: dict[str, DeviceProfile] = {GENERIC_PHONE.id: GENERIC_PHONE}
    for entry in raw.get("devices", []):
        try:
            profile = DeviceProfile(
                id=str(entry["id"]),
                brand=str(entry.get("brand", "")),
                model=str(entry.get("model", "")),
                tier=entry.get("tier", "mid"),
                video_equiv_mm=float(entry["video_equiv_mm"]),
                ultrawide_equiv_mm=(
                    float(entry["ultrawide_equiv_mm"])
                    if entry.get("ultrawide_equiv_mm") is not None
                    else None
                ),
                ois=bool(entry.get("ois", False)),
                lidar=bool(entry.get("lidar", False)),
                sensor=str(entry.get("sensor", "")),
                aliases=tuple(str(a).strip().lower() for a in entry.get("aliases", []) if str(a).strip()),
                sigma_fraction=float(entry.get("sigma_fraction", 0.06)),
                notes=str(entry.get("notes", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("devices.bad_entry", entry=entry.get("id"), error=str(exc))
            continue
        profiles[profile.id] = profile
    fingerprints = tuple(raw.get("ios_fingerprints", []))
    log.info("devices.table_loaded", n_profiles=len(profiles), n_fingerprints=len(fingerprints))
    return profiles, fingerprints


def _match_ua_model(model: str, profiles: dict[str, DeviceProfile]) -> DeviceProfile | None:
    """Exact alias match first, then a containment match, both case-folded.

    Containment rather than fuzzy matching on purpose: "SM-S911B" and "SM-S918B" are
    different phones with different lenses, and an edit-distance match between them would
    silently hand back the wrong field of view.
    """
    needle = model.strip().lower()
    if not needle:
        return None
    for profile in profiles.values():
        if needle in profile.aliases:
            return profile
    for profile in profiles.values():
        for alias in profile.aliases:
            if len(alias) >= 5 and (alias in needle or needle in alias):
                return profile
    return None


def _match_ios_fingerprint(
    screen: dict[str, Any], fingerprints: tuple[dict[str, Any], ...], profiles: dict[str, DeviceProfile]
) -> tuple[DeviceProfile | None, tuple[str, ...]]:
    """Screen points and pixel ratio -> a FAMILY of iPhones, or nothing.

    A family, never a phone: 393x852 at 3x is an iPhone 14 Pro, 15, 15 Pro or 16, and they
    are not all the same lens (24 mm for the two Pros, 26 for the others) nor all the same
    hardware (only some carry LiDAR). So the profile returned here is synthesised from the
    family -- the midpoint of its lenses, a sigma wide enough to span them, and a LiDAR flag
    only where every member agrees. Pretending the group is its first member would put the
    focal out by the same 8% this whole module exists to remove.
    """
    try:
        w = int(screen.get("cssWidth") or 0)
        h = int(screen.get("cssHeight") or 0)
        dpr = float(screen.get("devicePixelRatio") or 0.0)
    except (TypeError, ValueError):
        return None, ()
    if w <= 0 or h <= 0 or dpr <= 0:
        return None, ()
    # Orientation is the user's business, not the device's.
    short, long_ = min(w, h), max(w, h)
    for entry in fingerprints:
        try:
            ew, eh = int(entry["css_width"]), int(entry["css_height"])
            edpr = float(entry["dpr"])
        except (KeyError, TypeError, ValueError):
            continue
        if (min(ew, eh), max(ew, eh)) != (short, long_) or abs(edpr - dpr) > 0.05:
            continue
        models = tuple(str(m) for m in entry.get("models", []))
        base = profiles.get(str(entry.get("profile_id", "")))
        equiv = entry.get("equiv_mm")
        if equiv is None:
            return base, models
        lidar_flag = str(entry.get("lidar", "maybe"))
        family = DeviceProfile(
            id=f"ios-{ew}x{eh}@{edpr:g}",
            brand="Apple",
            model=models[0] if len(models) == 1 else f"iPhone ({len(models)} models share this screen)",
            tier="flagship",
            video_equiv_mm=float(equiv),
            ultrawide_equiv_mm=13.0,
            ois=True,
            # Only claim the scanner when every model behind this screen has one. "maybe" is
            # reported as absent, because a measurement offered on a maybe is worse than none.
            lidar=lidar_flag == "yes",
            sensor=base.sensor if base else "",
            sigma_fraction=float(entry.get("sigma_fraction", 0.08)),
            notes=str(entry.get("note", "")),
        )
        return family, models
    return None, ()


def resolve_device(
    sidecar: dict[str, Any] | None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> ResolvedDevice:
    """Identify the phone behind a capture sidecar and price its lens.

    ``width`` and ``height`` are the frames the engine actually received; without them the
    answer still names the phone but carries no focal length, because a focal length in
    pixels is meaningless until you know how many pixels there are.
    """
    profiles, fingerprints = load_profiles()
    notes: list[str] = []
    device = (sidecar or {}).get("device") or {}
    hints = device.get("modelHints") or {}
    track = device.get("trackSettings") or {}

    profile: DeviceProfile | None = None
    source: Source = "none"
    candidates: tuple[str, ...] = ()

    ua_model = str(hints.get("uaModel") or "").strip()
    if ua_model:
        profile = _match_ua_model(ua_model, profiles)
        if profile is not None:
            source = "ua-model"
            candidates = (ua_model,)
        else:
            notes.append(f"unknown_model:{ua_model}")

    if profile is None and str(device.get("platform") or "") == "ios":
        profile, candidates = _match_ios_fingerprint(hints.get("screen") or {}, fingerprints, profiles)
        if profile is not None:
            source = "ios-fingerprint"

    if profile is None:
        brand = str(hints.get("brand") or "").strip().lower()
        if brand:
            same_brand = [p for p in profiles.values() if p.brand.lower() == brand and p.id != GENERIC_PHONE.id]
            if same_brand:
                # The brand's median lens, with the spread of its own range as the sigma.
                equivs = sorted(p.video_equiv_mm for p in same_brand)
                median = equivs[len(equivs) // 2]
                spread = (equivs[-1] - equivs[0]) / 2.0 if len(equivs) > 1 else 0.0
                profile = DeviceProfile(
                    id=f"brand-{brand}",
                    brand=same_brand[0].brand,
                    model=f"{same_brand[0].brand} (model not identified)",
                    tier="mid",
                    video_equiv_mm=median,
                    sigma_fraction=max(0.08, spread / max(median, 1e-6)),
                    notes="Brand median; the exact model could not be identified.",
                )
                source = "brand"
                notes.append(f"brand_median_over_{len(same_brand)}_models")

    if profile is None:
        profile = GENERIC_PHONE
        source = "generic" if sidecar else "none"

    zoom = track.get("zoom")
    by_label = label_names_ultrawide(device.get("cameraLabel"))
    equiv_mm, lens = profile.equiv_mm_for(
        float(zoom) if isinstance(zoom, (int, float)) else None, ultrawide_selected=by_label
    )
    if lens == "ultrawide":
        notes.append(
            "camera_label_names_the_ultrawide" if by_label else "zoom_below_1_so_the_ultrawide_was_selected"
        )

    sigma = float(profile.sigma_fraction)
    if lens == "ultrawide" and profile.ultrawide_equiv_mm is None:
        sigma = max(sigma, _ULTRAWIDE_UNLISTED_SIGMA)
        notes.append("ultrawide_not_in_the_table")
    if bool(device.get("virtualLensOnly")):
        # A "virtual" rear device auto-switches lenses; which one was live at any moment is
        # not knowable from here, so the prior has to be loose enough to survive being wrong
        # about it.
        sigma = max(sigma, 0.15)
        notes.append("virtual_lens_device")
    if bool((sidecar or {}).get("optics", {}).get("lensChangedMidCapture")):
        sigma = max(sigma, 0.25)
        notes.append("lens_changed_mid_capture")

    focal_px = hfov = None
    if width and height and width > 0 and height > 0:
        focal_px = focal_px_from_equiv_mm(equiv_mm, width, height)
        hfov = hfov_deg_from_focal(focal_px, width)

    resolved = ResolvedDevice(
        profile=profile,
        source=source,
        focal_px=focal_px,
        hfov_deg=hfov,
        sigma_fraction=sigma,
        lens=lens,
        candidates=candidates,
        notes=notes,
    )
    log.info("devices.resolved", **resolved.as_dict())
    return resolved
