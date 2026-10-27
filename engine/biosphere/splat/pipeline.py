"""The orchestration: videos in, a streamable property out.

    base video ──► frames ──► features ──► SfM ──► level + scale ═══ property frame
                                                                        │
    room video ──► frames ──► features ──► SfM (+ borrowed base frames) ─┴─► Sim3 into the property frame
                                                                        │
                              every area: undistort ──► Brush ──► clean ──► SOG + pictures
                                                                        │
                                                   floor plan of everything ──► manifest

The general walkthrough is the spine. It is solved first, levelled and scaled, and every
room is registered into its frame through frames the two videos share. A room that will
not register is still delivered, levelled on its own, and flagged ``placed: false`` so the
viewer offers it as a jump rather than a walk. A room that fails outright is reported and
skipped; the property is not.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from biosphere.core.logging import get_logger

from .bundle import write_manifest
from .clean import CameraView, clean_table
from .coverage import redundant_areas
from .densify import dense_init
from .export import render_env_pano, render_floor_plan, render_thumbnail, to_sog
from .exposure import balance_exposure
from .features import LocalFeatures, build_database, release_models
from .level import camera_up_prior, estimate_leveling
from .ply import GaussianTable, read_ply, write_ply
from .raster import device_for
from .register import RegistrationReport, align_to_reference, select_anchor_frames
from .sfm import (
    chunk_reconstruction,
    drop_cameras,
    map_images,
    split_at_jumps,
    turn_speeds,
    undistort_for_training,
)
from .train import train_splat
from .types import STAGES, AreaInput, AreaResult, PropertySplatResult, SplatConfig, SplatError

__all__ = ["build_tour"]

log = get_logger("biosphere.splat")

_WEIGHTS = dict(STAGES)
_CUMULATIVE: dict[str, float] = {}
_acc = 0.0
for _name, _w in STAGES:
    _CUMULATIVE[_name] = _acc
    _acc += _w
del _acc, _name, _w

#: The camera list a viewer gets per area.
MAX_CAMERAS_IN_MANIFEST = 64
#: Padding of the walkable box around the camera path, metres.
BBOX_PAD_XZ_M = 1.2
BBOX_CEILING_M = 2.7
BBOX_BELOW_FLOOR_M = 0.3
#: A registered room whose cameras' mean up axis leans further than this from the
#: walkthrough's is mis-registered, whatever its residuals say: one phone, one hand, one
#: way of holding it. A correct registration leans a couple of degrees at most (the
#: walkthrough's own camera prior and floor normal agreed to 0.3 deg on the first real capture).
MAX_PLACED_TILT_DEG = 12.0
#: The capture app's own turning-speed warning, and the share of a walk above it past which
#: the delivery says so: 2.5-8.7 % on the berlin bench, 28 % on the walk that broke apart.
TURN_WARN_DEG_PER_S = 40.0
TURN_WARN_SHARE = 0.15


class _Progress:
    """Stage-local fractions in, one monotone 0..1 out, forwarded to the job reporter."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.high = 0.0

    def stage(self, name: str, fraction: float, message: str = "") -> None:
        frac = float(min(max(fraction, 0.0), 1.0))
        overall = _CUMULATIVE.get(name, self.high) + _WEIGHTS.get(name, 0.0) * frac
        self.high = max(self.high, min(overall, 0.999))
        if self.inner is not None:
            self.inner.stage(name, frac, message)

    def log(self, message: str, **fields: Any) -> None:
        log.info(message, **fields)
        if self.inner is not None and hasattr(self.inner, "log"):
            self.inner.log(message, **fields)


def _area_span(k: int, n: int):
    """Map a per-area fraction onto the stage's [0, 1] when ``n`` areas share the stage."""
    return lambda f: (k + float(f)) / max(n, 1)


def _registered(rec) -> list:
    return sorted((img for img in rec.images.values() if img.has_pose), key=lambda im: im.name)


def _rotations_and_centres(rec, names: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rots, cents, kept = [], [], []
    for img in _registered(rec):
        if names is not None and img.name not in names:
            continue
        pose = img.cam_from_world()
        rots.append(np.asarray(pose.rotation.matrix(), dtype=np.float64))
        cents.append(np.asarray(img.projection_center(), dtype=np.float64))
        kept.append(img.name)
    if not rots:
        return np.zeros((0, 3, 3)), np.zeros((0, 3)), []
    return np.stack(rots), np.stack(cents), kept


def _points(rec) -> np.ndarray:
    pts = [p.xyz for p in rec.points3D.values()]
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def _views_of(rec, names: set[str]) -> list[CameraView]:
    """The solved cameras of ``names``, in the form the cleaner asks a splat about."""
    out: list[CameraView] = []
    for img in rec.images.values():
        if not img.has_pose or img.name not in names:
            continue
        cam = rec.cameras[img.camera_id]
        out.append(CameraView(
            R=np.asarray(img.cam_from_world().rotation.matrix()),
            t=np.asarray(img.cam_from_world().translation),
            K=np.asarray(cam.calibration_matrix()),
            width=int(cam.width), height=int(cam.height),
        ))
    return out


def _tilt_after(rots: np.ndarray, R: np.ndarray, reference_up: np.ndarray) -> float:
    """Degrees between the mean camera up of ``rots`` (cam_from_world) after a world rotation ``R``
    and ``reference_up``."""
    if len(rots) == 0:
        return 0.0
    up = camera_up_prior(np.einsum("nij,kj->nik", rots, R))  # cam_from_new_world = cam_from_world @ R^T
    ref = np.asarray(reference_up, dtype=np.float64)
    ref = ref / max(float(np.linalg.norm(ref)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(float(up @ ref), -1.0, 1.0))))


def _walkable(cams: np.ndarray) -> tuple[np.ndarray, int]:
    """The camera centres a visitor could actually stand at, and how many were dropped.

    The manifest's camera list is not a record of the solve: it is the route the viewer
    picks areas along and walks the automatic property down. A pose the mapper put through the
    floor therefore does real damage - it drags the area's claim towards a place nobody can
    stand, and the property steps to its x and z at eye height, which is a jump across the room.
    Measured on the first real capture: four of six areas span a sane 0.8 m of height, one
    has 21 of 46 cameras up to 8 m BELOW the floor and one has 8 of 47 above the ceiling.

    The band is the walkable box's own, so this drops exactly the poses the visitor is not
    allowed to reach anyway. If most of the walk falls outside it the area is not
    mis-posed but mis-levelled, and throwing the path away would leave the viewer nothing
    to choose along, so it is kept whole.
    """
    keep, n_out = _walkable_mask(cams)
    return cams[keep], n_out


def _walkable_mask(cams: np.ndarray) -> tuple[np.ndarray, int]:
    """:func:`_walkable` as a mask, so the camera orientations can be filtered alongside."""
    if cams.shape[0] == 0:
        return np.zeros(0, dtype=bool), 0
    ok = (cams[:, 1] >= -BBOX_CEILING_M) & (cams[:, 1] <= BBOX_BELOW_FLOOR_M)
    n_out = int((~ok).sum())
    if int(ok.sum()) < max(4, int(0.5 * cams.shape[0])):
        return np.ones(cams.shape[0], dtype=bool), 0
    return ok, n_out


def _thin_index(count: int, n: int) -> np.ndarray:
    if count <= n:
        return np.arange(count)
    return np.linspace(0, count - 1, n).round().astype(int)


def _thin(cams: np.ndarray, n: int) -> list[list[float]]:
    return [[round(float(v), 4) for v in cams[i]] for i in _thin_index(cams.shape[0], n)]


def _entry(rec, own: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """Where the visitor lands, and which way they face: the frame with the deepest view.

    NOT the first frame of the walk. A walkthrough starts at the front door with the phone a
    hand's width from it, and a visitor dropped there sees a wall filling the screen - the
    first thing the first real capture showed was exactly that, and it reads as a broken
    scene rather than as a doorway. So every frame votes with the median depth of the points
    it observes, and the deepest view wins: a hallway or a room seen down its length, which
    is what a visitor should arrive to. Ties and empty models fall back to the first frame.
    """
    frames = [img for img in _registered(rec) if img.name in own]
    if not frames:
        raise SplatError("no registered frame for the entry pose")
    centres = np.stack([np.asarray(img.projection_center(), dtype=np.float64) for img in frames])
    # A camera the mapper flung outside the flat sees everything far away and would win the
    # depth vote every time, so only frames standing where the walk stood may stand for it.
    lo, hi = np.percentile(centres, 2, axis=0), np.percentile(centres, 98, axis=0)
    median_height = float(np.median(centres[:, 1]))
    best = None
    fallback = None
    for img, centre in zip(frames, centres, strict=True):
        plausible = bool(np.all(centre >= lo) and np.all(centre <= hi) and abs(centre[1] - median_height) < 0.6)
        R = np.asarray(img.cam_from_world().rotation.matrix(), dtype=np.float64)
        fwd = R.T @ np.array([0.0, 0.0, 1.0])
        fwd[1] = 0.0
        norm = np.linalg.norm(fwd)
        fwd = fwd / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
        if fallback is None or (plausible and fallback[2] is False):
            fallback = (centre, fwd, plausible)
        if not plausible:
            continue
        depths = [
            float(np.linalg.norm(np.asarray(rec.points3D[p2d.point3D_id].xyz, dtype=np.float64) - centre))
            for p2d in img.points2D
            if p2d.has_point3D() and p2d.point3D_id in rec.points3D
        ]
        if len(depths) < 20:
            continue
        score = float(np.median(depths))
        if best is None or score > best[0]:
            best = (score, centre, fwd)
    if best is not None:
        return best[1], best[2]
    return fallback[0], fallback[1]


def build_tour(
    base: AreaInput | None,
    rooms: list[AreaInput],
    *,
    config: SplatConfig,
    work_dir: Path,
    out_dir: Path,
    progress=None,
    cancel_check=None,
) -> PropertySplatResult:
    """Build the whole property. Raises :class:`SplatError` only when nothing can be delivered."""
    import pycolmap

    q = config.quality
    prog = _Progress(progress)
    t_start = time.perf_counter()
    work_dir = Path(work_dir)
    out_dir = Path(out_dir)
    for sub in ("areas", "thumbs", "env"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    device = device_for()
    warnings: list[str] = []
    metrics: dict[str, Any] = {"quality": q.name, "device": str(device), "areas": {}}

    if base is None:
        if not rooms:
            raise SplatError("no videos", en="No video was given.", tr="Hiç video verilmedi.")
        base, rooms = rooms[0], rooms[1:]
        warnings.append("no_general_walkthrough: the first room is the property frame")
    areas = [base] + list(rooms)
    n_areas = len(areas)

    # ---- 1. frames -----------------------------------------------------------------
    frames_root = work_dir / "frames"
    names: dict[str, list[str]] = {}
    frame_stats: dict[str, dict[str, Any]] = {}
    frame_sets: dict[str, Any] = {}
    for k, area in enumerate(areas):
        span = _area_span(k, n_areas)
        prog.stage("extract", span(0.0), f"{area.name}: extracting frames")
        from .frames import prepare_frames  # local import keeps ffmpeg out of import time

        fs = prepare_frames(
            list(area.videos), frames_root / area.key, config=q.frames,
            progress=lambda s, f, span=span, area=area: prog.stage("extract", span(f), f"{area.name}: {len(area.videos)} video"),
        )
        frame_sets[area.key] = fs
        frame_stats[area.key] = fs.as_dict()
        warnings += [f"{area.key}: {w}" for w in fs.warnings]
        # What the video itself can and cannot give, said up front: a 576-px WhatsApp
        # re-encode of a white flat trains soft whatever the pipeline does with it.
        short_side = min(fs.width, fs.height)
        if short_side and short_side < 720:
            warnings.append(f"{area.key}: low_resolution: {fs.width}x{fs.height} frames; a 1080p original would give far sharper walls")
        total_bytes = sum(Path(v).stat().st_size for v in area.videos if Path(v).exists())
        if fs.source_duration_s > 0 and total_bytes:
            mbps = total_bytes * 8 / fs.source_duration_s / 1e6
            frame_stats[area.key]["bitrate_mbps"] = round(mbps, 2)
            if mbps < 3.0:
                warnings.append(f"{area.key}: heavy_compression: {mbps:.1f} Mbps; the video went through a messenger or a download - use the original file")
        if len(fs.names) < 20:
            raise SplatError(
                f"{area.name}: only {len(fs.names)} usable frames",
                en=f"The video for {area.name} has too few usable frames.",
                tr=f"{area.name} videosunda kullanılabilir kare çok az.",
            )
    prog.stage("extract", 1.0)

    # ---- 2. features, and the chain: probe, bridge, cut ---------------------------------------
    # Every consecutive pair of keyframes is checked before anything is mapped; weak links are
    # bridged with more frames from the video, and links that stay broken split the area into
    # segments. The largest segment keeps the area's name; the others join the property as
    # room-like areas ("Genel gezinti · 2"), registered through anchors when the walk returns
    # somewhere the property frame knows and delivered on their own when it does not.
    from .chain import probe_and_bridge

    feats: dict[str, LocalFeatures] = {}
    extra_areas: list[AreaInput] = []
    #: Image name -> second of the area's own timeline. Bridging mixes 30 fps stretches into a
    #: 3 fps walk, so every "could a walker have moved that far?" test is paced by this.
    stamp_of: dict[str, float] = {}
    for k, area in enumerate(list(areas)):
        span = _area_span(k, n_areas)
        fs = frame_sets[area.key]
        fs2, area_feats, chain = probe_and_bridge(
            fs, list(area.videos), work_dir=frames_root / area.key, matching=q.matching, config=q.chain, device=device,
            progress=lambda f, m, span=span, area=area: prog.stage("features", span(f), f"{area.name}: {m}" if m else area.name),
        )
        frame_stats[area.key] = {**fs2.as_dict(), "chain": chain.as_dict()}
        warnings += [f"{area.key}: {w}" for w in chain.warnings]
        segs = chain.segments or [(0, len(fs2.names))]
        if len(segs) <= 1:
            area_feats.names = [f"{area.key}/images/{n}" for n in fs2.names]
            feats[area.key] = area_feats
            names[area.key] = list(area_feats.names)
            stamp_of.update(dict(zip(area_feats.names, fs2.timestamps, strict=True)))
            continue
        largest = max(range(len(segs)), key=lambda j: segs[j][1] - segs[j][0])
        n_extra = 0
        staging = fs2.image_dir.parent / "images_split"
        staging.mkdir(parents=True, exist_ok=True)
        for j, (a, b) in enumerate(segs):
            idx = list(range(a, b))
            if j == largest:
                key, name, image_dir = area.key, area.name, staging
            else:
                n_extra += 1
                key, name = f"{area.key}~{n_extra + 1}", f"{area.name} · {n_extra + 1}"
                image_dir = frames_root / key / "images"
                image_dir.mkdir(parents=True, exist_ok=True)
            seg_names: list[str] = []
            for m, i in enumerate(idx):
                new_name = f"f_{m:05d}.jpg"
                shutil.move(str(fs2.image_dir / fs2.names[i]), str(image_dir / new_name))
                seg_names.append(new_name)
            sub = area_feats.subset(idx)
            sub.names = [f"{key}/images/{nm}" for nm in seg_names]
            feats[key] = sub
            names[key] = list(sub.names)
            stamp_of.update({nm: fs2.timestamps[i] for nm, i in zip(sub.names, idx, strict=True)})
            seg_stats = {
                **fs2.as_dict(), "n_frames": len(seg_names), "segment_of": area.key,
                "segment_span_s": [round(fs2.timestamps[a], 1), round(fs2.timestamps[b - 1], 1)],
            }
            if j == largest:
                frame_stats[area.key].update(seg_stats)
            else:
                frame_stats[key] = seg_stats
                extra_areas.append(AreaInput(key=key, name=name, videos=area.videos, floor_index=area.floor_index))
        # The largest segment's frames go back where the area's frames live, renumbered.
        for stale in fs2.image_dir.glob("*.jpg"):
            stale.unlink()
        for f in staging.glob("f_*.jpg"):
            shutil.move(str(f), str(fs2.image_dir / f.name))
        shutil.rmtree(staging, ignore_errors=True)
        warnings.append(f"{area.key}: segmented: tracking broke {len(segs) - 1} time(s); delivered as {len(segs)} parts")
    prog.stage("features", 1.0)
    if extra_areas:
        rooms = list(rooms) + extra_areas
        areas = [base] + list(rooms)
        n_areas = len(areas)
    if len(names[base.key]) < 20:
        raise SplatError(
            f"base walkthrough kept only {len(names[base.key])} linked frames",
            en="The general walkthrough could not be tracked: too few frames stay linked to each other.",
            tr="Genel gezinti izlenemedi: birbirine bağlı kalan kare çok az.",
        )

    # ---- 3. base SfM + levelling ----------------------------------------------------------
    sfm_root = work_dir / "sfm"
    recs: dict[str, Any] = {}
    placed: dict[str, bool] = {}
    area_metrics: dict[str, dict[str, Any]] = {a.key: {"frames": frame_stats[a.key]} for a in areas}

    def _map(area: AreaInput, feature_sets: list[LocalFeatures], k: int):
        span = _area_span(k, n_areas)
        adir = sfm_root / area.key
        adir.mkdir(parents=True, exist_ok=True)
        mrep = build_database(
            adir / "db.db", frames_root, feature_sets=feature_sets, camera_model="OPENCV", config=q.matching, device=device,
            progress=lambda s, f: prog.stage("sfm", span(0.45 * f), f"{area.name}: matching"),
        )
        n_imgs = sum(len(fs.names) for fs in feature_sets)
        rec, srep = map_images(
            adir / "db.db", frames_root, adir, n_images=n_imgs, options=q.sfm, times=stamp_of,
            progress=lambda s, f: prog.stage("sfm", span(0.45 + 0.55 * f), f"{area.name}: solving cameras"),
        )
        area_metrics[area.key]["matching"] = mrep.as_dict()
        area_metrics[area.key]["sfm"] = srep.as_dict()
        return rec, srep

    base_rec, base_srep = _map(base, [feats[base.key]], 0)
    own_base = set(names[base.key])
    if base_srep.n_registered < max(20, 0.5 * len(names[base.key])):
        raise SplatError(
            f"base walkthrough registered only {base_srep.n_registered}/{len(names[base.key])} frames",
            en="The general walkthrough could not be tracked well enough. Walk slower and keep the walls in view.",
            tr="Genel gezinti yeterince iyi izlenemedi. Daha yavaş yürüyün ve duvarları görüş alanında tutun.",
        )
    warnings += [f"{base.key}: {w}" for w in (base_srep.warnings or [])]

    # Where the mapper joined two parts of a walk wrongly - a step no walker could take
    # between two consecutive frames - the model is cut. The largest part keeps the area;
    # the others queue up as their own areas ("Genel gezinti · 3"), placed by anchors when
    # they can be and delivered on their own when they cannot. A tangle is never delivered.
    prebuilt: dict[str, Any] = {}

    def _split_area(area: AreaInput, rec) -> tuple[set[str], list[AreaInput]]:
        own_all = set(names[area.key])
        chunks, dropped = split_at_jumps(
            rec, own_all, jump_factor=q.sfm.chunk_jump_factor, min_frames=q.sfm.chunk_min_frames, times=stamp_of
        )
        if dropped:
            drop_cameras(rec, dropped)
            warnings.append(f"{area.key}: short_chunks_dropped: {len(dropped)} frames sat between two seams and were left out")
        if len(chunks) <= 1:
            return (set(chunks[0]) if chunks else own_all), []
        main = chunks[0]
        extras: list[AreaInput] = []
        n_existing = sum(1 for kk in names if kk.startswith(area.key + "~"))
        for ch in chunks[1:]:
            n_existing += 1
            key, name = f"{area.key}~{n_existing + 1}", f"{area.name} · {n_existing + 1}"
            prebuilt[key] = chunk_reconstruction(rec, set(ch), sfm_root / key / "chunk")
            names[key] = list(ch)
            frame_stats[key] = {**frame_stats.get(area.key, {}), "n_frames": len(ch), "chunk_of": area.key}
            area_metrics[key] = {"frames": frame_stats[key], "sfm": {"chunk_of": area.key, "n_registered": len(ch)}}
            extras.append(AreaInput(key=key, name=name, videos=area.videos, floor_index=area.floor_index))
        drop_cameras(rec, [nm for ch in chunks[1:] for nm in ch])
        names[area.key] = list(main)
        warnings.append(
            f"{area.key}: split_at_jumps: the mapper joined {len(chunks)} parts of the walk inconsistently; they are delivered separately"
        )
        return set(main), extras

    # How the walk was turned, before it is cut: the commonest reason a walk breaks into parts
    # (see sfm.turn_speeds for the calibration).
    speeds = turn_speeds(base_rec, set(names[base.key]), stamp_of)
    if speeds.size >= 10:
        fast_share = float((speeds > TURN_WARN_DEG_PER_S).mean())
        metrics["turn_speed"] = {
            "median_deg_s": round(float(np.median(speeds)), 1), "p90_deg_s": round(float(np.percentile(speeds, 90)), 1),
            "share_over_40": round(fast_share, 3), "n_pairs": int(speeds.size),
        }
        if fast_share > TURN_WARN_SHARE:
            warnings.append(
                f"turned_too_fast: {fast_share:.0%} of the walk turned faster than {TURN_WARN_DEG_PER_S:.0f} deg/s "
                f"(p90 {np.percentile(speeds, 90):.0f} deg/s); a quarter turn should take about three seconds"
            )
    own_base, base_extras = _split_area(base, base_rec)
    rots, cents, _ = _rotations_and_centres(base_rec, own_base)
    lev = estimate_leveling(rots, cents, _points(base_rec), assumed_camera_height_m=config.camera_height_m)
    base_rec.transform(pycolmap.Sim3d(lev.scale, pycolmap.Rotation3d(lev.R), lev.t))
    base_rots, cents, _ = _rotations_and_centres(base_rec, own_base)  # now in the property frame, metres
    base_up = camera_up_prior(base_rots) if len(base_rots) else np.array([0.0, -1.0, 0.0])
    (sfm_root / base.key / "property_frame").mkdir(parents=True, exist_ok=True)
    base_rec.write_binary(str(sfm_root / base.key / "property_frame"))
    recs[base.key] = base_rec
    placed[base.key] = True
    area_metrics[base.key]["leveling"] = lev.as_dict()
    metrics["leveling"] = lev.as_dict()
    if lev.floor_inlier_fraction < 0.05:
        warnings.append("weak_floor: the floor plane was found on few points; levelling relies on the phone's own tilt")
    # Walking pace, from the levelled path (metres, on the assumed camera height).
    base_duration = float(frame_stats[base.key].get("source_duration_s") or 0.0)
    if len(cents) > 2 and base_duration > 0:
        path_m = float(np.linalg.norm(np.diff(cents, axis=0), axis=1).sum())
        speed = path_m / base_duration
        metrics["walk_speed_m_per_s"] = round(speed, 2)
        if speed > 1.0:
            warnings.append(f"walked_too_fast: about {speed:.1f} m/s along the path; a step a second (~0.5 m/s) gives the trainer twice the views of every wall")

    # ---- 4. rooms: SfM with borrowed base frames, then Sim3 ------------------------------------
    pending: list[AreaInput] = list(rooms) + base_extras
    rooms = []
    k = 0
    #: Areas whose reconstruction is in the property frame, in the order they were placed. Any of
    #: them can lend anchor frames to what comes next.
    placed_keys: list[str] = [base.key]

    class _PoseReference:
        """The posed images of every placed area, as one thing ``align_to_reference`` can read."""

        def __init__(self, keys: list[str]) -> None:
            images = [img for kk in keys for img in recs[kk].images.values() if img.has_pose]
            self.images = dict(enumerate(images))

    def _anchor_pool(keys: list[str]) -> tuple[LocalFeatures, set[str]]:
        pool = LocalFeatures.concatenate([feats[kk] for kk in keys])
        posed = {img.name for kk in keys for img in recs[kk].images.values() if img.has_pose}
        return pool, posed
    while pending:
        room = pending.pop(0)
        k += 1
        rooms.append(room)
        if all(a.key != room.key for a in areas):
            areas.append(room)
        own = set(names[room.key])
        try:
            if room.key in prebuilt:
                room_rec = prebuilt.pop(room.key)
                sim3, areg = None, RegistrationReport(
                    warnings=["no_shared_frames: a cut-off part of the walk shares no frames with the property frame"]
                )
            else:
                pool, pool_posed = _anchor_pool(placed_keys)
                anchors = select_anchor_frames(feats[room.key], pool, count=24, allowed=pool_posed)
                room_rec, rrep = _map(room, [feats[room.key], pool.subset(anchors)], k)
                warnings += [f"{room.key}: {w}" for w in (rrep.warnings or [])]
                n_own = sum(1 for img in room_rec.images.values() if img.has_pose and img.name in own)
                if n_own < max(20, 0.5 * len(names[room.key])):
                    raise SplatError(f"{room.name}: registered only {n_own}/{len(names[room.key])} own frames")
                own, extras = _split_area(room, room_rec)
                pending += extras
                prog.stage("register", _area_span(k - 1, max(len(rooms) + len(pending), 1))(0.2), f"{room.name}: placing in the property")
                sim3, areg = align_to_reference(room_rec, _PoseReference(placed_keys))
            area_metrics[room.key]["registration"] = areg.as_dict()
            plausible = False
            if sim3 is not None and areg.ok and areg.median_residual_m < 0.6:
                # One more check the transform itself cannot make: after it, the room's
                # cameras must stand at chest height like the walkthrough's did. A room
                # registered through mis-placed anchors lands on the ceiling or under the floor.
                r_rots_raw, r_cents_raw, _ = _rotations_and_centres(room_rec, own)
                mapped_y = np.median(sim3.apply(r_cents_raw)[:, 1])
                base_y = float(np.median(cents[:, 1]))
                plausible = abs(mapped_y - base_y) < 0.6
                if not plausible:
                    areg.warnings = (areg.warnings or []) + [f"implausible_height: room cameras would sit {mapped_y - base_y:+.2f} m from the walkthrough's"]
                    area_metrics[room.key]["registration"] = areg.as_dict()
                # ...and they must be held upright like the walkthrough's were. Height alone
                # missed the first real capture's bathroom: placed at the right height, it
                # was rolled ~30 degrees, through anchors that agreed with each other (9 of 15,
                # 2 cm residual) and with nothing else.
                tilt = _tilt_after(r_rots_raw, sim3.R, base_up)
                areg_tilt = round(tilt, 2)
                area_metrics[room.key]["registration"] = {**areg.as_dict(), "tilt_deg": areg_tilt}
                if plausible and tilt > MAX_PLACED_TILT_DEG:
                    plausible = False
                    areg.warnings = (areg.warnings or []) + [f"implausible_tilt: room cameras would be tilted {tilt:.0f} deg from the walkthrough's"]
                    area_metrics[room.key]["registration"] = {**areg.as_dict(), "tilt_deg": areg_tilt}
            if plausible:
                room_rec.transform(sim3.to_pycolmap())
                placed[room.key] = True
                placed_keys.append(room.key)
            else:
                warnings.append(f"{room.key}: unplaced_room: {'; '.join(areg.warnings or []) or f'residual {areg.median_residual_m:.2f} m'}")
                r_rots, r_cents, _ = _rotations_and_centres(room_rec, own)
                rlev = estimate_leveling(r_rots, r_cents, _points(room_rec), assumed_camera_height_m=config.camera_height_m)
                room_rec.transform(pycolmap.Sim3d(rlev.scale, pycolmap.Rotation3d(rlev.R), rlev.t))
                area_metrics[room.key]["leveling"] = rlev.as_dict()
                placed[room.key] = False
            (sfm_root / room.key / "property_frame").mkdir(parents=True, exist_ok=True)
            room_rec.write_binary(str(sfm_root / room.key / "property_frame"))
            recs[room.key] = room_rec
        except Exception as exc:  # noqa: BLE001 - one room must not sink the property
            log.warning("splat.room_failed", room=room.key, error=str(exc), traceback=traceback.format_exc())
            warnings.append(f"{room.key}: room_failed: {type(exc).__name__}: {str(exc)[:200]}")
            area_metrics[room.key]["failed"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    n_areas = len(areas)
    prog.stage("register", 1.0)

    # ---- 5. train, clean, export every area ------------------------------------------------------
    results: dict[str, AreaResult] = {}
    tables: dict[str, GaussianTable] = {}
    trainable = [a for a in areas if a.key in recs and not (config.base_only and a.key != base.key)]
    for k, area in enumerate(trainable):
        span = _area_span(k, len(trainable))
        res = AreaResult(key=area.key, name=area.name, floor_index=area.floor_index)
        results[area.key] = res
        rec = recs[area.key]
        own = set(names[area.key])
        try:
            # 5a. pinhole dataset in the property frame, own frames only
            tdir = work_dir / "train" / area.key
            dataset = undistort_for_training(
                sfm_root / area.key / "property_frame", frames_root, tdir / "dataset",
                max_image_size=q.train_image_long_side, image_names=sorted(own),
            )
            # The phone metered every frame on its own; the trainer has no exposure to fit,
            # so it fits a haze instead (see exposure.py). Take it out of the pictures first.
            if q.balance_exposure:
                sparse = dataset / "sparse" / "0"
                if not (sparse / "cameras.bin").exists():
                    sparse = dataset / "sparse"
                try:
                    erep = balance_exposure(pycolmap.Reconstruction(str(sparse)), dataset / "images")
                    area_metrics[area.key]["exposure"] = erep.as_dict()
                    warnings += [f"{area.key}: {w}" for w in erep.warnings]
                except Exception as exc:  # noqa: BLE001 - an unbalanced property beats no property
                    log.warning("splat.exposure_failed", area=area.key, error=str(exc))
                    warnings.append(f"{area.key}: exposure_balance_failed: {type(exc).__name__}")
            # A dense start on the bare walls (see densify.py); the SfM points when it cannot run.
            if q.dense_init.enabled:
                prog.stage("train", span(0.0), f"{area.name}: depth")
                release_models()  # the matcher's networks are done; the GPU is the depth model's
                drep = dense_init(dataset, config=q.dense_init, device=device)
                area_metrics[area.key]["dense_init"] = drep.as_dict()
                warnings += [f"{area.key}: {w}" for w in drep.warnings]
            prog.stage("train", span(0.02), f"{area.name}: training")
            trep = train_splat(
                dataset, tdir / "out", config=q.train,
                progress=lambda f, m, span=span, area=area: prog.stage("train", span(0.02 + 0.98 * f), f"{area.name}: {m}"),
                cancel_check=cancel_check,
            )
            area_metrics[area.key]["train"] = trep.as_dict()
            res.metrics = area_metrics[area.key]

            # 5b. clean
            prog.stage("clean", span(0.0), f"{area.name}: removing floaters")
            cam_rots, cams, cam_names = _rotations_and_centres(rec, own)
            table = read_ply(trep.ply_path)
            table, crep = clean_table(table, cams, config=q.clean, views=_views_of(rec, own))
            area_metrics[area.key]["clean"] = crep.as_dict()
            clean_ply = tdir / "clean.ply"
            write_ply(clean_ply, table)
            res.ply_path = clean_ply
            tables[area.key] = table
            prog.stage("clean", span(1.0))

            # 5c. export: SOG + pictures + geometry for the manifest
            prog.stage("export", span(0.0), f"{area.name}: compressing")
            sog = to_sog(clean_ply, out_dir / "areas" / f"{area.key}.sog", sh_degree=q.export_sh_degree, iterations=q.sog_iterations)
            res.sog_path, res.sog_bytes, res.splat_count = sog.path, sog.bytes, sog.count
            area_metrics[area.key]["sog"] = sog.as_dict()
            entry_pos, entry_fwd = _entry(rec, own)
            res.entry_position = [round(float(v), 4) for v in entry_pos]
            res.entry_forward = [round(float(v), 4) for v in entry_fwd]
            walk_mask, n_off_floor = _walkable_mask(cams)
            walk_cams = cams[walk_mask]
            if n_off_floor:
                res.warnings.append(f"off_floor_cameras: {n_off_floor} of {len(cams)} poses sit outside the walkable height and were left out of the route")
                area_metrics[area.key]["off_floor_cameras"] = n_off_floor
            res.cameras = _thin(walk_cams, MAX_CAMERAS_IN_MANIFEST)
            # Where the phone LOOKED at each of those points (unit, property frame): the views
            # the model reproduces best, which the viewer's guided tour follows.
            # ...and when, in seconds of the area's own video: parts of one walk are toured in
            # the order they were filmed, whichever of them could be placed.
            looks = cam_rots[walk_mask][:, 2, :]
            when = np.array([stamp_of.get(n, -1.0) for n in cam_names], dtype=np.float64)[walk_mask]
            res.looks = [
                [*(round(float(v), 4) for v in looks[i]), round(float(when[i]), 2)]
                for i in _thin_index(len(looks), MAX_CAMERAS_IN_MANIFEST)
            ]
            centroid = np.median(walk_cams, axis=0)
            res.centroid = [round(float(v), 4) for v in centroid]
            # Percentiles rather than min/max: the walkable box must not stretch to a frame
            # the mapper placed on the wrong side of the wall.
            lo = np.percentile(walk_cams, 2, axis=0)
            hi = np.percentile(walk_cams, 98, axis=0)
            res.bbox_min = [round(float(lo[0] - BBOX_PAD_XZ_M), 4), round(float(-BBOX_CEILING_M), 4), round(float(lo[2] - BBOX_PAD_XZ_M), 4)]
            res.bbox_max = [round(float(hi[0] + BBOX_PAD_XZ_M), 4), round(float(BBOX_BELOW_FLOOR_M), 4), round(float(hi[2] + BBOX_PAD_XZ_M), 4)]
            prog.stage("export", span(0.4), f"{area.name}: pictures")
            res.thumbnail_path = render_thumbnail(table, entry_pos, entry_fwd, out_dir / "thumbs" / f"{area.key}.jpg", device=device)
            env_centre = np.array([centroid[0], float(np.median(walk_cams[:, 1])), centroid[2]])
            res.env_pano_path = render_env_pano(table, env_centre, out_dir / "env" / f"{area.key}.jpg", device=device)
            if not placed.get(area.key, False):
                res.placed = False
                res.warnings.append("unplaced: this room could not be aligned to the walkthrough; it opens as its own space")
            prog.stage("export", span(0.9))
        except Exception as exc:  # noqa: BLE001
            log.warning("splat.area_failed", area=area.key, error=str(exc), traceback=traceback.format_exc())
            res.failed = f"{type(exc).__name__}: {str(exc)[:300]}"
            warnings.append(f"{area.key}: area_failed: {res.failed}")
            if area.key == base.key:
                raise SplatError(
                    f"base area failed: {exc}",
                    en="Training the general walkthrough failed.",
                    tr="Genel gezintinin eğitimi başarısız oldu.",
                ) from exc

    for area in areas:
        if area.key not in results:
            results[area.key] = AreaResult(
                key=area.key, name=area.name, floor_index=area.floor_index,
                failed=area_metrics[area.key].get("failed", "skipped"), metrics=area_metrics[area.key],
            )

    # ---- 6. which areas may take the picture over ------------------------------------------------
    # Pieces of one broken walk overlap, and the viewer chooses by nearest camera path. An area
    # the base already covers must not be chosen that way: switching to it swaps every splat on
    # screen for another reconstruction of the same wall, or for no reconstruction at all.
    coverage_report = None
    try:
        placed_cameras = {
            a.key: results[a.key].cameras
            for a in areas
            if a.key in tables and placed.get(a.key, False) and results[a.key].failed is None
        }
        if len(placed_cameras) > 1 and base.key in tables:
            prog.stage("export", 0.94, "checking what each room adds")
            coverage_report = redundant_areas(
                {k: tables[k] for k in placed_cameras}, placed_cameras, base.key, device=device
            )
            for key in coverage_report.redundant:
                results[key].redundant = True
                results[key].warnings.append("redundant: the general walkthrough already shows this space")
            metrics["coverage"] = coverage_report.as_dict()
    except Exception as exc:  # a delivery must not fail because a check could not run
        warnings.append(f"coverage_check_failed: {exc}")

    # ---- 7. floor plan + manifest ---------------------------------------------------------------
    plan_path = None
    plan: dict[str, Any] = {"hotspots": []}
    placed_tables = [tables[a.key] for a in areas if a.key in tables and placed.get(a.key, False)]
    if placed_tables:
        prog.stage("export", 0.95, "drawing the floor plan")
        walkable = [results[a.key] for a in areas if placed.get(a.key, False) and results[a.key].bbox_min]
        bounds = None
        if walkable:
            lo = np.min([r.bbox_min for r in walkable], axis=0)
            hi = np.max([r.bbox_max for r in walkable], axis=0)
            bounds = (float(lo[0]), float(hi[0]), float(lo[2]), float(hi[2]))
        fp = render_floor_plan(placed_tables, out_dir / "plan.jpg", bounds=bounds, device=device)
        plan_path = fp.path
        plan.update(fp.as_dict())
        for area in areas:
            r = results[area.key]
            if r.failed or not placed.get(area.key, False) or not r.centroid:
                continue
            x_pct, y_pct = fp.to_fraction(r.centroid[0], r.centroid[2])
            plan["hotspots"].append({"key": area.key, "name": area.name, "x_pct": round(x_pct, 4), "y_pct": round(y_pct, 4), "floor_index": area.floor_index})

    metrics["areas"] = area_metrics
    metrics["seconds_total"] = round(time.perf_counter() - t_start, 1)
    metrics["n_rooms_delivered"] = sum(1 for r in rooms if results[r.key].failed is None and results[r.key].sog_path)
    metrics["bytes_total"] = int(sum(results[a.key].sog_bytes for a in areas))
    result = PropertySplatResult(
        out_dir=out_dir, base=results[base.key], rooms=[results[r.key] for r in rooms], floor_plan_path=plan_path,
        floor_plan=plan, manifest_path=out_dir / "tour.json", metrics=metrics, warnings=warnings,
    )
    write_manifest(result)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    prog.stage("export", 1.0, "done")
    if not config.keep_work:
        shutil.rmtree(work_dir / "train", ignore_errors=True)
        shutil.rmtree(work_dir / "frames", ignore_errors=True)
    return result
