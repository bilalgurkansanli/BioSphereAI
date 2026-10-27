"""Camera poses for one video: mapping, model selection, undistortion.

Two mappers ship inside pycolmap 4: the incremental one (COLMAP proper) and the global one
(GLOMAP, ``global_mapping``). Global is minutes faster on long sequences and cannot drift
the way an incremental chain does, but it needs a well-connected view graph and gives up
entirely on a weak one; incremental is slower and can fragment a sequence into several
models, but registers something from almost anything. So: global first, judged by how many
frames it registered; incremental when that is not good enough; the best model wins, and
"best" is the registered-frame count, because a splat trained from half the frames is a
splat with half the room missing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SfmOptions", "SfmReport", "map_images", "undistort_for_training", "camera_centres", "load_reconstruction",
    "outlier_cameras", "drop_cameras", "strong_pair_violations", "unreliable_cameras", "split_at_jumps",
    "chunk_reconstruction", "turn_speeds",
]


@dataclass(frozen=True)
class SfmOptions:
    try_global: bool = True
    #: A global-mapper result registering at least this fraction of the frames is accepted...
    global_accept_fraction: float = 0.85
    #: ...unless it placed more than this many frames far off the walk (see ``outlier_cameras``),
    #: or more than ``global_max_outlier_fraction`` of them, whichever is larger: a few
    #: far-flung frames are dropped afterwards, and a model that placed 95 % of the walk is
    #: worth more than an incremental run that may not finish...
    global_max_outliers: int = 2
    global_max_outlier_fraction: float = 0.05
    #: ...or contradicts more than this fraction of the strong image pairs (see
    #: ``strong_pair_violations``): two frames that share ``strong_pair_min_inliers`` verified
    #: matches cannot be looking ``strong_pair_max_rotation_deg`` apart or be far across the
    #: flat from each other. The global mapper's failure mode on a weakly-joined sequence is
    #: exactly that - one pass folded onto another - and this is how it is caught. The limits
    #: come from what CORRECT geometry produced on the berlin bench (ground truth): a 100°
    #: lens matches the same wall from directions up to ~95° apart and from ~4 m away, so
    #: 120° and 25 steps flag under 1 % of a correct model's pairs; a fold flags a chunk.
    global_max_violation_fraction: float = 0.03
    strong_pair_min_inliers: int = 80
    strong_pair_max_rotation_deg: float = 120.0
    strong_pair_max_distance_steps: float = 25.0
    #: Below this fraction of frames in the best model the sequence is reported as fragmented.
    warn_fraction: float = 0.7
    #: A registered frame the mapper pinned with fewer 3D points than this, or that sits a
    #: leap away from both its sequence neighbours, is not trusted for training (see
    #: ``unreliable_cameras``): on a white-walled flat these are the doorway frames the
    #: mapper placed on a handful of matches, and a splat trained through a wrong camera is
    #: a smear over the right ones.
    unreliable_min_points: int = 60
    unreliable_jump_factor: float = 8.0
    #: A step between consecutive registered frames longer than this many median steps (per
    #: frame of index gap) is a place the mapper joined two parts of the walk wrongly; the
    #: model is cut there (see ``split_at_jumps``) and each part is delivered on its own.
    chunk_jump_factor: float = 10.0
    chunk_min_frames: int = 12
    num_threads: int = -1
    refine_principal_point: bool = False
    min_num_matches: int = 15
    #: Incremental mapper: prefer robustness over speed on hand-held video.
    init_min_tri_angle: float = 8.0
    abs_pose_min_inlier_ratio: float = 0.25
    ba_global_frames_ratio: float = 1.1


@dataclass
class SfmReport:
    mapper: str = ""
    n_images: int = 0
    n_registered: int = 0
    n_points: int = 0
    mean_reprojection_px: float = 0.0
    mean_track_length: float = 0.0
    focal_px: float = 0.0
    camera_model: str = ""
    camera_params: list[float] | None = None
    seconds_global: float = 0.0
    seconds_incremental: float = 0.0
    n_models: int = 0
    n_outliers_dropped: int = 0
    n_unreliable_dropped: int = 0
    #: Fraction of strong pairs the chosen model contradicts (0 is consistent).
    strong_pair_violation_fraction: float = 0.0
    n_strong_pairs: int = 0
    warnings: list[str] | None = None

    @property
    def registered_fraction(self) -> float:
        return self.n_registered / max(self.n_images, 1)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["registered_fraction"] = round(self.registered_fraction, 4)
        return d


def _describe(rec, report: SfmReport) -> None:
    report.n_registered = int(rec.num_reg_images())
    report.n_points = int(rec.num_points3D())
    report.mean_reprojection_px = float(rec.compute_mean_reprojection_error())
    report.mean_track_length = float(rec.compute_mean_track_length())
    cam = next(iter(rec.cameras.values()))
    report.focal_px = float(cam.mean_focal_length())
    report.camera_model = str(cam.model_name)
    report.camera_params = [float(v) for v in cam.params]


def _largest(recs: dict) -> Any | None:
    best = None
    for rec in recs.values():
        if best is None or rec.num_reg_images() > best.num_reg_images():
            best = rec
    return best


def _sequence_key(name: str) -> tuple[str, int]:
    """``room1/images/f_00012.jpg`` -> (``room1/images``, 12): the video and the frame's place in it."""
    stem = Path(name).stem
    try:
        idx = int(stem.split("_")[-1])
    except ValueError:
        idx = 0
    return str(Path(name).parent).replace("\\", "/"), idx


def _paced_steps(
    frames: list[tuple[int, str, np.ndarray]], times: dict[str, float] | None
) -> tuple[np.ndarray, float]:
    """Distance between consecutive frames per unit of video, and the median of that pace.

    With ``times`` the unit is a second, which is the only unit that compares a 30 fps bridge
    with 3 fps keyframes; without it, a frame index, which is what a uniformly sampled
    sequence gives anyway.
    """
    c = np.stack([f[2] for f in frames])
    dist = np.linalg.norm(np.diff(c, axis=0), axis=1)
    if times is not None:
        dt = np.array([max(times.get(b[1], 0.0) - times.get(a[1], 0.0), 1e-3) for a, b in zip(frames[:-1], frames[1:], strict=True)])
    else:
        dt = np.maximum(np.diff(np.array([f[0] for f in frames], dtype=np.float64)), 1.0)
    pace = dist / dt
    return pace, max(float(np.median(pace)), 1e-9)


def outlier_cameras(
    rec, *, step_factor: float = 25.0, spread_factor: float = 3.0, times: dict[str, float] | None = None
) -> list[str]:
    """Names of registered frames whose centre is nowhere near where the walk says it is.

    A walkthrough is continuous: consecutive frames are a step apart. A frame both of whose
    sequence neighbours are ``step_factor`` median steps away, AND that lies well outside the
    cloud of the other cameras, was not placed by the images around it but by whatever the
    optimiser did with a few bad matches. The global mapper does this to a handful of frames
    in a sparse sequence; left alone they blow up the bounding box, the levelling and every
    room registered through them.
    """
    by_seq: dict[str, list[tuple[int, str, np.ndarray]]] = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        seq, idx = _sequence_key(img.name)
        by_seq.setdefault(seq, []).append((idx, img.name, np.asarray(img.projection_center(), dtype=np.float64)))
    out: list[str] = []
    for frames in by_seq.values():
        if len(frames) < 8:
            continue
        frames.sort(key=lambda t: t[0])
        c = np.stack([f[2] for f in frames])
        steps, s_med = _paced_steps(frames, times)
        med = np.median(c, axis=0)
        dist = np.linalg.norm(c - med, axis=1)
        spread = max(float(np.percentile(dist, 90)), 1e-9)
        n = len(frames)
        for i in range(n):
            prev_far = i == 0 or steps[i - 1] > step_factor * s_med
            next_far = i == n - 1 or steps[i] > step_factor * s_med
            if prev_far and next_far and dist[i] > spread_factor * spread:
                out.append(frames[i][1])
    return out


def strong_pair_violations(
    rec,
    db_path: Path,
    *,
    min_inliers: int = 80,
    max_rotation_deg: float = 120.0,
    max_distance_steps: float = 25.0,
) -> tuple[int, int]:
    """``(violations, strong_pairs)``: how many strongly matched pairs the model contradicts.

    A pair with ``min_inliers`` verified matches sees the same surface; its two cameras are
    therefore within a few steps of each other and turned less than ``max_rotation_deg``
    apart. A model in which such a pair is 150 degrees apart has folded one part of the walk
    onto another, and no reprojection error will say so (the fold is self-consistent). Cheap
    to test, and independent of the mapper that produced the model.
    """
    import sqlite3

    con = sqlite3.connect(str(db_path))
    try:
        names = {int(i): str(n) for i, n in con.execute("SELECT image_id, name FROM images")}
        strong = [
            divmod(int(pid), 2147483647)
            for pid, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries")
            if int(rows or 0) >= min_inliers
        ]
    finally:
        con.close()
    by_name = {img.name: img for img in rec.images.values() if img.has_pose}
    if not by_name:
        return 0, 0
    centres = np.stack([np.asarray(img.projection_center()) for img in by_name.values()])
    # A typical step of the walk, from sequence neighbours (the median is immune to a fold).
    order = sorted(by_name, key=_sequence_key)
    steps = [
        float(np.linalg.norm(np.asarray(by_name[a].projection_center()) - np.asarray(by_name[b].projection_center())))
        for a, b in zip(order[:-1], order[1:], strict=True)
        if _sequence_key(a)[0] == _sequence_key(b)[0]
    ]
    step = max(float(np.median(steps)) if steps else 0.0, 1e-6)
    spread = float(np.percentile(np.linalg.norm(centres - np.median(centres, axis=0), axis=1), 90))
    max_dist = max(max_distance_steps * step, spread)
    cos_limit = np.cos(np.radians(max_rotation_deg))
    n_strong = 0
    bad = 0
    for a, b in strong:
        ia, ib = by_name.get(names.get(a, "")), by_name.get(names.get(b, ""))
        if ia is None or ib is None:
            continue
        n_strong += 1
        ra = np.asarray(ia.cam_from_world().rotation.matrix())
        rb = np.asarray(ib.cam_from_world().rotation.matrix())
        # Viewing directions in the world: the third row of cam_from_world.
        if float(ra[2] @ rb[2]) < cos_limit:
            bad += 1
            continue
        if float(np.linalg.norm(np.asarray(ia.projection_center()) - np.asarray(ib.projection_center()))) > max_dist:
            bad += 1
    return bad, n_strong


def unreliable_cameras(
    rec, *, min_points: int = 60, jump_factor: float = 8.0, times: dict[str, float] | None = None
) -> list[str]:
    """Registered frames the model does not really constrain.

    Two signs, either is enough: the frame sees fewer than ``min_points`` triangulated
    points (a pose from a few matches on a bare wall), or both of its sequence neighbours
    are ``jump_factor`` median steps away (a frame flung out of the walk on its own). The
    checks are per sequence, so a room's frames are judged against the room's own pace.
    """
    by_seq: dict[str, list[tuple[int, str, np.ndarray, int]]] = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        seq, idx = _sequence_key(img.name)
        by_seq.setdefault(seq, []).append(
            (idx, img.name, np.asarray(img.projection_center(), dtype=np.float64), int(img.num_points3D))
        )
    out: list[str] = []
    for frames in by_seq.values():
        frames.sort(key=lambda t: t[0])
        n = len(frames)
        if n < 8:
            continue
        steps, s_med = _paced_steps([(f[0], f[1], f[2]) for f in frames], times)
        for i, (_, name, _, n_points) in enumerate(frames):
            prev_far = i == 0 or steps[i - 1] > jump_factor * s_med
            next_far = i == n - 1 or steps[i] > jump_factor * s_med
            if n_points < min_points or (prev_far and next_far):
                out.append(name)
    return out


def split_at_jumps(
    rec, own_names: set[str], *, jump_factor: float = 10.0, min_frames: int = 12, times: dict[str, float] | None = None
) -> tuple[list[list[str]], list[str]]:
    """Cut the registered frames of one sequence into chunks at impossible jumps.

    Consecutive frames are a fraction of a second apart; a pace of ``jump_factor`` times the
    walk's median pace is not a walk but a seam the mapper stitched wrong. Pace is metres per
    second of video when ``times`` is given - the only measure that compares a 30 fps bridge
    with 3 fps keyframes - and metres per frame index otherwise. Returns ``(chunks, dropped)``:
    the chunks as name lists in sequence order, longest first, and the names of chunks too
    short to deliver.
    """
    frames = [
        (_sequence_key(img.name)[1], img.name, np.asarray(img.projection_center(), dtype=np.float64))
        for img in rec.images.values()
        if img.has_pose and img.name in own_names
    ]
    frames.sort(key=lambda t: t[0])
    if len(frames) < 2 * min_frames:
        return [[f[1] for f in frames]], []
    pace, med = _paced_steps(frames, times)
    cuts = [i for i in range(len(pace)) if pace[i] > jump_factor * med]
    edges = [0, *[i + 1 for i in cuts], len(frames)]
    chunks = [[f[1] for f in frames[a:b]] for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]
    kept = sorted([ch for ch in chunks if len(ch) >= min_frames], key=len, reverse=True)
    dropped = [n for ch in chunks if len(ch) < min_frames for n in ch]
    return kept or [[f[1] for f in frames]], dropped


def turn_speeds(rec, names: set[str] | None, times: dict[str, float], *, min_dt: float = 0.15, max_dt: float = 0.7) -> np.ndarray:
    """How fast the camera turned (deg/s) between consecutive solved frames of one walk.

    Pairs closer than ``min_dt`` (bridge frames 33 ms apart, where a degree of pose noise
    reads as 30 deg/s) or further than ``max_dt`` (a gap) are skipped. The capture warning
    built on this is calibrated on two walks: the berlin bench, which rebuilds cleanly,
    turns faster than 40 deg/s in 2.5-8.7 % of its frame pairs (p90 33 deg/s); the first real
    phone walk, which broke into six parts, in 28 % (p90 57 deg/s, whip pans of 90-220 at the
    breaks).
    """
    imgs = sorted(
        (img for img in rec.images.values() if img.has_pose and img.name in times and (names is None or img.name in names)),
        key=lambda img: times[img.name],
    )
    out: list[float] = []
    for a, b in zip(imgs[:-1], imgs[1:], strict=True):
        dt = times[b.name] - times[a.name]
        if dt < min_dt or dt > max_dt:
            continue
        Ra = np.asarray(a.cam_from_world().rotation.matrix())
        Rb = np.asarray(b.cam_from_world().rotation.matrix())
        out.append(float(np.degrees(np.arccos(np.clip((np.trace(Rb @ Ra.T) - 1.0) / 2.0, -1.0, 1.0)))) / dt)
    return np.asarray(out, dtype=np.float64)


def chunk_reconstruction(rec, keep: set[str], scratch_dir: Path):
    """A copy of ``rec`` holding only the frames in ``keep`` (and the points they observe)."""
    import pycolmap

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    rec.write_binary(str(scratch_dir))
    sub = pycolmap.Reconstruction(str(scratch_dir))
    for img in list(sub.images.values()):
        if img.has_pose and img.name not in keep:
            sub.deregister_frame(img.frame_id)
    for pid in [pid for pid, pt in sub.points3D.items() if pt.track.length() == 0]:
        sub.delete_point3D(pid)
    return sub


def drop_cameras(rec, names: list[str]) -> int:
    """Deregister ``names`` and re-run bundle adjustment so the rest settle without them."""
    import pycolmap

    dropped = 0
    for name in names:
        try:
            img = rec.find_image_with_name(name)
        except Exception:  # noqa: BLE001
            continue
        if img is None or not img.has_pose:
            continue
        rec.deregister_frame(img.frame_id)
        dropped += 1
    if dropped:
        opts = pycolmap.BundleAdjustmentOptions()
        opts.refine_principal_point = False
        try:
            pycolmap.bundle_adjustment(rec, opts)
        except Exception:  # noqa: BLE001 - the model is still usable without the polish
            pass
    return dropped


def map_images(
    db_path: Path,
    image_dir: Path,
    out_dir: Path,
    *,
    n_images: int,
    options: SfmOptions | None = None,
    progress=None,
    times: dict[str, float] | None = None,
):
    """Run the mappers on a database and return ``(reconstruction, SfmReport)``.

    The chosen model is written to ``out_dir / "sparse" / "0"`` in binary, the layout every
    downstream tool (Brush, our own registration, undistortion) reads.
    """
    import pycolmap

    options = options or SfmOptions()
    out_dir = Path(out_dir)
    report = SfmReport(n_images=int(n_images), warnings=[])
    best = None
    # A global model that failed only the fold test is kept aside: the incremental mapper
    # can fold the same weak joint the same way, and then the less-contradicted of the two
    # is the better model, whichever mapper made it.
    folded_global = None
    folded_global_fraction = 1.0

    def violation_fraction(rec) -> tuple[int, int, float]:
        violations, n_strong = strong_pair_violations(
            rec,
            db_path,
            min_inliers=options.strong_pair_min_inliers,
            max_rotation_deg=options.strong_pair_max_rotation_deg,
            max_distance_steps=options.strong_pair_max_distance_steps,
        )
        return violations, n_strong, (violations / n_strong if n_strong else 0.0)

    if options.try_global:
        t0 = time.perf_counter()
        gdir = out_dir / "global"
        gdir.mkdir(parents=True, exist_ok=True)
        try:
            gopts = pycolmap.GlobalPipelineOptions()
            gopts.num_threads = options.num_threads
            gopts.min_num_matches = options.min_num_matches
            recs = pycolmap.global_mapping(db_path, image_dir, gdir, options=gopts)
            cand = _largest(recs)
            report.seconds_global = time.perf_counter() - t0
            bad = outlier_cameras(cand, times=times) if cand is not None else []
            allowed_outliers = max(options.global_max_outliers, int(options.global_max_outlier_fraction * n_images))
            if cand is not None and len(bad) > allowed_outliers:
                report.warnings.append(
                    f"global_mapper_outliers: {len(bad)} frames far off the walk, falling back to incremental"
                )
                cand = None
            if cand is not None:
                violations, n_strong, fraction = violation_fraction(cand)
                if fraction > options.global_max_violation_fraction:
                    report.warnings.append(
                        f"global_mapper_folded: {violations}/{n_strong} strong pairs contradicted "
                        f"({fraction:.1%}), trying incremental"
                    )
                    if cand.num_reg_images() >= options.global_accept_fraction * n_images:
                        folded_global, folded_global_fraction = cand, fraction
                    cand = None
            if cand is not None and cand.num_reg_images() >= options.global_accept_fraction * n_images:
                best = cand
                report.mapper = "global"
            elif cand is not None:
                report.warnings.append(
                    f"global_mapper_partial: registered {cand.num_reg_images()}/{n_images}, falling back to incremental"
                )
            elif not report.warnings:
                report.warnings.append("global_mapper_failed: no model, falling back to incremental")
        except Exception as exc:  # noqa: BLE001 - the incremental mapper is the fallback
            report.seconds_global = time.perf_counter() - t0
            report.warnings.append(f"global_mapper_error: {type(exc).__name__}: {str(exc)[:160]}")
        if progress is not None:
            progress("sfm", 0.35 if best is None else 0.9)

    if best is None:
        t0 = time.perf_counter()
        idir = out_dir / "incremental"
        idir.mkdir(parents=True, exist_ok=True)
        iopts = pycolmap.IncrementalPipelineOptions()
        iopts.num_threads = options.num_threads
        iopts.min_num_matches = options.min_num_matches
        iopts.ba_refine_principal_point = options.refine_principal_point
        iopts.ba_global_frames_ratio = options.ba_global_frames_ratio
        iopts.mapper.init_min_tri_angle = options.init_min_tri_angle
        iopts.mapper.abs_pose_min_inlier_ratio = options.abs_pose_min_inlier_ratio
        recs = pycolmap.incremental_mapping(db_path, image_dir, idir, options=iopts)
        report.seconds_incremental = time.perf_counter() - t0
        report.n_models = len(recs)
        best = _largest(recs)
        report.mapper = "incremental"
        if best is None and folded_global is None:
            raise RuntimeError("structure from motion found no model: too few matched frames")
        if folded_global is not None:
            inc_fraction = violation_fraction(best)[2] if best is not None else 1.0
            inc_reg = best.num_reg_images() if best is not None else 0
            # The incremental model wins only when it is actually less contradicted, or when
            # it registered clearly more of the walk.
            if inc_fraction > folded_global_fraction and inc_reg <= folded_global.num_reg_images() * 1.05:
                report.warnings.append(
                    f"incremental_folded_more: {inc_fraction:.1%} vs global {folded_global_fraction:.1%} "
                    "strong pairs contradicted; keeping the global model"
                )
                best = folded_global
                report.mapper = "global"
        if progress is not None:
            progress("sfm", 0.9)

    # A bundle adjustment that ran away (reprojection error in the millions of pixels, or
    # not a number at all) is not a model; the incremental mapper is the second opinion.
    reproj = float(best.compute_mean_reprojection_error()) if best.num_reg_images() else float("nan")
    if (not np.isfinite(reproj) or reproj > 50.0) and report.mapper == "global":
        report.warnings.append(f"global_mapper_diverged: reprojection error {reproj:.3g} px, trying incremental")
        t0 = time.perf_counter()
        idir = out_dir / "incremental"
        idir.mkdir(parents=True, exist_ok=True)
        iopts = pycolmap.IncrementalPipelineOptions()
        iopts.num_threads = options.num_threads
        iopts.min_num_matches = options.min_num_matches
        iopts.ba_refine_principal_point = options.refine_principal_point
        iopts.ba_global_frames_ratio = options.ba_global_frames_ratio
        recs = pycolmap.incremental_mapping(db_path, image_dir, idir, options=iopts)
        report.seconds_incremental = time.perf_counter() - t0
        cand = _largest(recs)
        if cand is not None and np.isfinite(float(cand.compute_mean_reprojection_error())):
            best = cand
            report.mapper = "incremental"
        else:
            raise RuntimeError("structure from motion diverged: no usable model")
    bad = outlier_cameras(best, times=times)
    if bad:
        n_dropped = drop_cameras(best, bad)
        report.n_outliers_dropped = n_dropped
        report.warnings.append(f"outlier_cameras_dropped: {n_dropped} frames were far off the walk and were removed")
    shaky = unreliable_cameras(
        best, min_points=options.unreliable_min_points, jump_factor=options.unreliable_jump_factor, times=times
    )
    # Never more than a fifth of the model: past that the model itself is the problem, and
    # the frames are better reported than silently thinned.
    if shaky and len(shaky) <= 0.2 * max(best.num_reg_images(), 1):
        n_dropped = drop_cameras(best, shaky)
        report.n_unreliable_dropped = n_dropped
        report.warnings.append(
            f"unreliable_cameras_dropped: {n_dropped} frames pinned by too few points or flung from the walk were left out"
        )
    elif shaky:
        report.warnings.append(f"unreliable_cameras: {len(shaky)} frames are weakly constrained (kept)")
    violations, n_strong = strong_pair_violations(
        best,
        db_path,
        min_inliers=options.strong_pair_min_inliers,
        max_rotation_deg=options.strong_pair_max_rotation_deg,
        max_distance_steps=options.strong_pair_max_distance_steps,
    )
    report.n_strong_pairs = n_strong
    report.strong_pair_violation_fraction = round(violations / n_strong, 4) if n_strong else 0.0
    if n_strong and violations / n_strong > options.global_max_violation_fraction:
        report.warnings.append(
            f"model_inconsistent: {violations}/{n_strong} strong pairs contradicted by the final model"
        )
    _describe(best, report)
    if report.registered_fraction < options.warn_fraction:
        report.warnings.append(
            f"fragmented_sequence: only {report.n_registered}/{n_images} frames registered in the largest model"
        )
    final = out_dir / "sparse" / "0"
    final.mkdir(parents=True, exist_ok=True)
    best.write_binary(str(final))
    if progress is not None:
        progress("sfm", 1.0)
    return best, report


def load_reconstruction(path: Path):
    import pycolmap

    return pycolmap.Reconstruction(str(path))


def camera_centres(rec) -> tuple[list[str], np.ndarray]:
    """Registered image names (sorted) and their projection centres, ``(n, 3)``."""
    items = sorted(((img.name, img.projection_center()) for img in rec.images.values() if img.has_pose), key=lambda t: t[0])
    if not items:
        return [], np.zeros((0, 3), dtype=np.float64)
    names = [n for n, _ in items]
    return names, np.stack([np.asarray(c, dtype=np.float64) for _, c in items])


def undistort_for_training(
    sparse_dir: Path,
    image_dir: Path,
    out_dir: Path,
    *,
    max_image_size: int = -1,
    blank_pixels: float = 0.0,
    image_names: list[str] | None = None,
) -> Path:
    """Write a pinhole dataset (``images/`` + ``sparse/``) the trainer can read.

    The mapper solved an OPENCV camera; the trainer rasterises pinholes. COLMAP's undistorter
    resamples every image onto the ideal pinhole of the same focal length and rewrites the
    model with a PINHOLE camera, which is exactly the pair Brush expects. ``blank_pixels=0``
    crops rather than pads, so the frame never carries black corners into the splat.
    """
    import pycolmap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = pycolmap.UndistortCameraOptions()
    opts.blank_pixels = float(blank_pixels)
    opts.max_image_size = int(max_image_size)
    pycolmap.undistort_images(
        out_dir, sparse_dir, image_dir, image_names=list(image_names or []), output_type="COLMAP", undistort_options=opts
    )
    return out_dir
