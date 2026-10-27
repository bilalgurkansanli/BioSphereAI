"""Feature detection, candidate pair generation and geometric verification.

Everything the solver knows about the world enters through this module, so its job is
less "find lots of points" than "find points that survive verification". Three choices do
most of that work:

CLAHE before detection
    Interior walls, ceilings and skies are the dominant content of a room panorama and
    they are nearly flat. Global contrast stretching cannot help — the useful variation is
    local. CLAHE lifts it without amplifying the already-textured regions into noise.

A low SIFT contrast threshold
    The default (0.04) is tuned for outdoor photographs. Indoors it discards precisely the
    faint wall texture that is the only thing tying two views of a blank corridor
    together. 0.015 roughly triples the keypoint count on interiors, and the ratio test
    plus RANSAC discard the extra rubbish for free.

A low AKAZE detector threshold, for exactly the same reason
    AKAZE's default (0.001) is tuned the same way and fails the same way, but it failed
    *silently* because AKAZE takes no feature budget: ``config.max_features`` was simply
    ignored for it, so the pipeline could not ask for more and had no idea it was not
    getting them. See :data:`AKAZE_THRESHOLD` for the measurement. Every detector now
    honours the budget — generously detected, then capped to the strongest — which is what
    makes the budget, rather than an OpenCV default, the thing that governs yield.

RootSIFT
    L1-normalising a SIFT descriptor and taking the square root makes Euclidean distance
    in the transformed space equal to the Hellinger distance in the original. It is a
    two-line change that measurably improves matching, with no cost at query time.

Pair verification uses a homography rather than an essential matrix because the capture is
(assumed to be) a pure rotation about the optical centre: with no translation there is no
parallax, a homography is the exact model, and it needs no scene depth to be well posed.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import cv2
import numpy as np
import structlog

from . import geometry as geom
from .types import (
    Frame,
    MatchPair,
    PipelineConfig,
    ProgressReporter,
    QualityMode,
    StitchError,
)

__all__ = [
    "FeatureSet",
    "detect_features",
    "candidate_pairs",
    "match_pairs",
    "connected_components",
    "repair_connectivity",
    "is_degenerate_homography",
    "points_are_degenerate",
    "RETRY_FEATURE_MULTIPLE",
    "RETRY_MATCH_RATIO",
]

log = structlog.get_logger(__name__)

#: Long edge the detector runs at. Beyond this SIFT gets slower without finding features
#: that survive the ratio test, because the extra octaves are all sensor noise.
_DETECT_LONG_EDGE = 1600

#: Binary descriptors need Hamming distance and a brute-force matcher.
_BINARY_DETECTORS = {"orb", "brisk", "akaze"}

#: A pure rotation between two overlapping views cannot shear the image this hard; if the
#: fitted homography does, it is fitting noise on a degenerate configuration.
_MAX_AFFINE_CONDITION = 20.0

#: Two frames that overlap cannot be this far apart in orientation.
_MAX_PAIR_ROTATION_DEG = 100.0

#: AKAZE's detector response threshold. OpenCV's default is 0.001, tuned for the same
#: outdoor photography SIFT's contrast threshold is, and it starves interiors in the same
#: way — but far more damagingly, because AKAZE has no ``nfeatures`` parameter, so
#: ``config.max_features`` was ignored for it entirely and nothing downstream could ask for
#: more. FAST mode is the only mode that selects AKAZE, which is why this only ever showed
#: up there.
#:
#: Measured on the reference partial-sweep capture (a 12 s clean nodal sweep at 480x854,
#: 28 keyframes), sweeping the threshold and re-running the whole matching stage:
#:
#: =========  ======  ======  ========  =========  ==============  =========
#: threshold  med kp  min kp  verified  sequential largest compnt  ms/frame
#: =========  ======  ======  ========  =========  ==============  =========
#: 0.001         300      73        38      19/27             36%      41.5
#: 0.0005        436     117        40      20/27             36%      41.1
#: 0.0003        658     198        44      22/27             36%      43.0
#: 0.0001       1742     737        53      26/27             93%      42.5
#: **0.00005**  2594    1232         -          -               -      45.1
#: 0.00003      3119    1515         -          -               -      45.5
#: =========  ======  ======  ========  =========  ==============  =========
#:
#: Two things decide the value. **It is nearly free**: a 33x change in threshold moves
#: detection by 4 ms a frame, because AKAZE's cost is building the nonlinear scale space,
#: not keeping the keypoints — so the stingy default was never buying the speed FAST mode
#: exists for (SIFT costs 48.5 ms/frame on the same frames; AKAZE at 0.00005 costs 45.1).
#: And 0.00005 is the point where the *configured budget* starts binding instead of the
#: threshold: median yield 2594 against FAST's ``max_features`` of 2500, over budget on 17
#: of 28 frames. That is the property worth having — yield governed by what the pipeline
#: asked for, not by a constant nobody chose.
#:
#: The cap that follows keeps the strongest keypoints, so on a well-textured frame the
#: retained set is essentially what the old threshold would have found anyway. The lower
#: threshold only changes anything on the frames that were starved, which are exactly the
#: frames that were breaking the pose graph.
AKAZE_THRESHOLD = 0.00005

#: Fraction of the ratio-tested matches that must survive RANSAC before a pair is trusted.
#:
#: The gate exists for the configuration RANSAC is most dangerous in: a large pile of
#: unrelated correspondences, inside which a small mutually-consistent subset can nearly
#: always be found. It is **not scale-free in Lowe's ratio**, and that coupling is what
#: :func:`repair_connectivity` used to trip over — see :data:`RETRY_MATCH_RATIO`.
MIN_INLIER_RATIO = 0.25

#: Retry settings for :func:`repair_connectivity`, applied in the two stages of
#: :func:`_retry_stages` rather than all at once.
#:
#: Relaxing the inlier floor and Lowe's ratio *together* is self-defeating, and the reason
#: is arithmetic rather than empirical: a looser ratio test admits more raw matches, so the
#: same recovered inliers are divided by a bigger denominator and the pair falls foul of
#: :data:`MIN_INLIER_RATIO` instead. Measured on the reference realistic capture — the five
#: adjacent keyframe pairs whose loss shattered the pose graph, first pass at ratio 0.78
#: against the old single-stage retry at 0.88:
#:
#: =======  =====================  =====================  =======================
#: pair     first pass (0.78)      old retry (0.88)       old retry outcome
#: =======  =====================  =====================  =======================
#: (7, 8)   42 m, 15 inl, r 0.36   131 m, 19 inl, r 0.14  rejected on inlier ratio
#: (10,11)  85 m, 24 inl, r 0.28   275 m, 44 inl, r 0.16  rejected on inlier ratio
#: (20,21)  50 m, 20 inl, r 0.40   176 m, 31 inl, r 0.18  rejected on inlier ratio
#: (30,31)  23 m,  0 inl, r 0.00    84 m, 14 inl, r 0.17  rejected on inlier ratio
#: (31,32)  61 m, 16 inl, r 0.26   159 m, 32 inl, r 0.20  rejected on inlier ratio
#: =======  =====================  =====================  =======================
#:
#: Every one of them is recovered in inlier count and then thrown away on the ratio. The
#: retry rescued nothing on the capture it was written for, silently.
#:
#: Four of the five need no ratio relaxation at all — at the *original* 0.78 they are
#: already ratio-healthy (0.26-0.40) and fail only on the count. So the retry keeps the wide
#: search as its first stage, exactly as shipped, and adds a second, *narrower* one — count
#: lowered, ratio left alone — for the bridges the wide search rejects.
#: :data:`MIN_INLIER_RATIO` keeps its full strength in both. The order is not the intuitive
#: one and it was measured rather than reasoned; see :func:`_retry_stages`.
#:
#: That this ordering is the safe one was measured too, over the 175 keyframe pairs of the
#: same capture whose yaw separation exceeds 1.4x the field of view — pairs that *cannot*
#: overlap, so every match between them is junk by construction:
#:
#: ==============  ==================  =================  ==============================
#: Lowe ratio      junk inlier ratio   junk inliers       junk pairs passing the retry
#: ==============  ==================  =================  ==============================
#: 0.78            med 0.217 max 0.80  med 4 max 14       1 / 175
#: 0.88            med 0.057 max 0.16  med 5 max 22       0 / 175
#: ==============  ==================  =================  ==============================
#:
#: The two gates swap roles as the ratio moves: at 0.78 junk routinely reaches an inlier
#: *ratio* of 0.2-0.8 and is caught by the count instead, while at 0.88 the ratio becomes
#: sharply discriminative and the counts spread. Keeping both gates at full strength in
#: both stages is therefore what makes the escalation safe, and it is why the tempting
#: "relax the ratio gate to match" is exactly the wrong move: at 0.88 a floor of 0.12 would
#: sit *below* the junk maximum of 0.156 and let five of those 175 impossible pairs through.
#:
#: The feature multiple is insurance for a frame that is genuinely keypoint-starved rather
#: than merely marginal, which is why it rides with the second stage. Note that it was a
#: *no-op for AKAZE* until ``config.max_features`` started being enforced for it — AKAZE
#: takes no such parameter, so the retry's insurance did not exist in precisely the mode
#: that had the starved frames. It does now; see :func:`_cap_to_budget`.
RETRY_FEATURE_MULTIPLE = 2
RETRY_MATCH_RATIO = 0.88

#: Adding an edge merges two components, which can expose a further bridge, so the repair
#: iterates. Three rounds is far more than a single sweep has ever needed.
RETRY_MAX_ROUNDS = 3


@dataclass(slots=True)
class FeatureSet:
    """Keypoints and descriptors for one keyframe, in full-frame pixel coordinates."""

    frame_index: int
    keypoints: np.ndarray  # (N, 2) float32, full-frame pixels
    descriptors: np.ndarray | None  # (N, D) float32 or uint8
    size: tuple[int, int]  # (width, height) of the full frame
    scale: float  # detection_size / full_size, <= 1.0

    @property
    def n(self) -> int:
        return int(self.keypoints.shape[0])

    @property
    def is_binary(self) -> bool:
        return self.descriptors is not None and self.descriptors.dtype == np.uint8


# ======================================================================================
# Detection
# ======================================================================================


def _make_detector(config: PipelineConfig):
    """Build the configured detector.

    OpenCV 5 moved AKAZE and BRISK into ``cv2.xfeatures2d``; SIFT and ORB stayed at the
    top level. Resolving that here keeps the rest of the module version-agnostic.

    SIFT and ORB take the feature budget directly. AKAZE and BRISK have no such parameter,
    so they are asked to detect generously and :func:`_cap_to_budget` enforces the budget
    afterwards — otherwise ``config.max_features`` would silently mean nothing for them.
    """
    name = str(config.detector).lower()
    if name == "sift":
        return cv2.SIFT_create(
            nfeatures=int(config.max_features),
            contrastThreshold=0.015,
            edgeThreshold=12,
        )
    if name == "akaze":
        return cv2.xfeatures2d.AKAZE_create(threshold=AKAZE_THRESHOLD)
    if name == "orb":
        return cv2.ORB_create(nfeatures=int(config.max_features))
    if name == "brisk":
        return cv2.xfeatures2d.BRISK_create()
    raise StitchError(f"unknown detector {config.detector!r}; expected sift | akaze | orb")


def _cap_to_budget(kps: list, desc: np.ndarray | None, budget: int) -> tuple[list, np.ndarray | None]:
    """Keep the ``budget`` strongest keypoints, in detection order.

    SIFT and ORB already cap themselves, so this is a no-op for them. It exists for the
    detectors that do not — and it is what turns ``config.max_features`` into a real knob
    for AKAZE, which in turn is what makes :data:`RETRY_FEATURE_MULTIPLE` in
    :func:`repair_connectivity` do the thing its docstring has always claimed: doubling the
    budget for a keypoint-starved frame now genuinely admits weaker keypoints instead of
    silently changing nothing.

    Selection is by response and the survivors are then restored to detection order, so the
    result is deterministic and does not depend on how ties happened to sort.
    """
    if budget <= 0 or len(kps) <= budget:
        return kps, desc
    keep = np.argsort([-kp.response for kp in kps], kind="stable")[:budget]
    keep.sort()
    return [kps[i] for i in keep], (None if desc is None else desc[keep])


def _root_sift(desc: np.ndarray) -> np.ndarray:
    """Hellinger-kernel transform of a float descriptor block."""
    desc = np.asarray(desc, dtype=np.float32)
    norms = np.abs(desc).sum(axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    return np.sqrt(desc / norms, dtype=np.float32)


def _prepare_gray(frame: Frame) -> tuple[np.ndarray, float]:
    """Load a frame as a CLAHE-equalised grayscale image capped at the detection size."""
    img = cv2.imread(str(frame.path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise StitchError(f"cannot read frame image {frame.path}")
    h, w = img.shape[:2]
    long_edge = max(h, w)
    scale = 1.0
    if long_edge > _DETECT_LONG_EDGE:
        scale = _DETECT_LONG_EDGE / float(long_edge)
        img = cv2.resize(
            img,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(img), scale


def detect_features(
    frames: list[Frame],
    config: PipelineConfig,
    progress: ProgressReporter | None = None,
) -> list[FeatureSet]:
    """Detect and describe features for every keyframe.

    Keypoints come back in *full-frame* coordinates even though detection ran on a
    downscaled copy, so every later stage can ignore the detection scale entirely.
    """
    if not frames:
        raise StitchError("no keyframes to detect features on")
    is_binary = str(config.detector).lower() in _BINARY_DETECTORS
    use_root_sift = str(config.detector).lower() == "sift"

    def _one(pos_frame: tuple[int, Frame]) -> FeatureSet:
        pos, frame = pos_frame
        gray, scale = _prepare_gray(frame)
        detector = _make_detector(config)  # detectors are not thread-safe; one per task
        kps, desc = detector.detectAndCompute(gray, None)
        kps, desc = _cap_to_budget(kps, desc, int(config.max_features))
        if not kps:
            return FeatureSet(
                frame_index=pos,
                keypoints=np.zeros((0, 2), dtype=np.float32),
                descriptors=None,
                size=(frame.width, frame.height),
                scale=scale,
            )
        pts = np.array([kp.pt for kp in kps], dtype=np.float32) / max(scale, 1e-9)
        if desc is not None and not is_binary:
            desc = _root_sift(desc) if use_root_sift else np.asarray(desc, dtype=np.float32)
        elif desc is not None:
            desc = np.asarray(desc, dtype=np.uint8)
        h_full, w_full = frame.height, frame.width
        if gray.shape[1] > 0:
            w_full = int(round(gray.shape[1] / max(scale, 1e-9)))
            h_full = int(round(gray.shape[0] / max(scale, 1e-9)))
        return FeatureSet(
            frame_index=pos,
            keypoints=pts,
            descriptors=desc,
            size=(w_full, h_full),
            scale=scale,
        )

    workers = config.n_workers if config.n_workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    out: list[FeatureSet] = [None] * len(frames)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, fs in enumerate(pool.map(_one, list(enumerate(frames)))):
            out[fs.frame_index] = fs
            if progress is not None and (k % 5 == 0 or k == len(frames) - 1):
                progress.stage("features", (k + 1) / len(frames), f"features {k + 1}/{len(frames)}")

    log.info(
        "detect_features",
        detector=config.detector,
        n_frames=len(out),
        median_keypoints=int(np.median([fs.n for fs in out])) if out else 0,
        min_keypoints=min(fs.n for fs in out) if out else 0,
    )
    return out


# ======================================================================================
# Candidate pairs
# ======================================================================================


def candidate_pairs(
    n: int,
    poses_prior_yaw: np.ndarray | None,
    config: PipelineConfig,
    *,
    hfov_deg: float = 67.0,
    poses_prior_pitch: np.ndarray | None = None,
    vfov_deg: float | None = None,
) -> list[tuple[int, int]]:
    """Which frame pairs are worth trying to match.

    Sequential neighbours are always included — they are what makes the pose graph
    connected — and they are taken *cyclically*, so the last few keyframes are paired with
    the first few. That wrap is not an optimisation, it is what turns the pose graph from a
    path into a cycle, and the difference is structural: a path has no redundancy, so one
    keyframe that fails to link (a window, a blurred swing) severs it and everything beyond
    the break is dropped by ``solve._keep_largest_component``. A cycle survives the same
    break connected the long way round. It is also the only edge that measures loop closure
    at all. :func:`repair_connectivity` has always paired cyclically for exactly these
    reasons; it just ran too late, only after the graph had already broken.

    Loop-closure candidates *beyond* the wrap neighbours come from the coarse yaw prior
    produced by :mod:`biosphere.pano.select`: two frames whose predicted yaw separation is
    inside 80 % of the field of view plausibly overlap. ``poses_prior_yaw`` must be in real
    degrees and ``hfov_deg`` must be the capture's own field of view — this test reduces
    modulo a full turn, so a track measured against the wrong focal is not merely noisy but
    wraps in the wrong place. :mod:`biosphere.pano.solve` therefore feeds both from the
    focal the pre-scan measured rather than from the 67-degree phone prior.

    Below 60 frames an exhaustive sweep is only a couple of thousand pairs, which the
    threaded matcher chews through in seconds — and it is strictly more robust than any
    prior, so outside FAST mode we simply take it.
    """
    pairs: set[tuple[int, int]] = set()
    if n < 2:
        return []

    radius = max(1, int(config.neighbour_radius))
    pairs.update(_cyclic_neighbours(n, radius))

    exhaustive = n <= 60 and config.mode is not QualityMode.FAST
    if exhaustive:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.add((i, j))
    elif config.loop_search and poses_prior_yaw is not None:
        yaw = np.asarray(poses_prior_yaw, dtype=np.float64).ravel()
        pitch = (
            np.asarray(poses_prior_pitch, dtype=np.float64).ravel()
            if poses_prior_pitch is not None else None
        )
        # Pitch-aware: two frames whose rows are more than most of a vertical field apart
        # cannot overlap however their yaw lines up. Without this the up and down rows of
        # a three-row sweep were paired with each other on yaw alone.
        pitch_limit = 0.8 * float(vfov_deg) if vfov_deg else None
        if yaw.size == n:
            limit = float(hfov_deg) * 0.8
            for i in range(n):
                for j in range(i + radius + 1, n):
                    # Compare modulo a full turn so the end of the sweep can rejoin its
                    # beginning; that wrap-around pair is the loop closure.
                    delta = abs(float(geom.wrap_pi(math.radians(yaw[j] - yaw[i]))))
                    if math.degrees(delta) >= limit:
                        continue
                    if (
                        pitch is not None and pitch.size == n and pitch_limit is not None
                        and abs(float(pitch[j] - pitch[i])) >= pitch_limit
                    ):
                        continue
                    pairs.add((i, j))

    out = sorted(pairs)
    log.info(
        "candidate_pairs",
        n_frames=n,
        n_pairs=len(out),
        exhaustive=exhaustive,
        loop_search=bool(config.loop_search),
    )
    return out


# ======================================================================================
# Matching
# ======================================================================================


def _raw_matches(
    fs_i: FeatureSet,
    fs_j: FeatureSet,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ratio-tested, mutually-nearest matches between two feature sets.

    The mutual check on top of Lowe's ratio is what keeps repeated architecture (a row of
    identical windows, floor tiles) from generating a consistent-looking but wrong match
    set that RANSAC would then happily fit.
    """
    di, dj = fs_i.descriptors, fs_j.descriptors
    if di is None or dj is None or len(di) < 2 or len(dj) < 2:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    if fs_i.is_binary:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_fwd = matcher.knnMatch(di, dj, k=2)
        knn_bwd = matcher.knnMatch(dj, di, k=2)
    else:
        index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE
        search_params = {"checks": 64}
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        knn_fwd = matcher.knnMatch(di, dj, k=2)
        knn_bwd = matcher.knnMatch(dj, di, k=2)

    def _ratio_pass(knn) -> dict[int, int]:
        keep: dict[int, int] = {}
        for pair in knn:
            if len(pair) < 2:
                continue
            m, s = pair[0], pair[1]
            if m.distance < ratio * s.distance:
                keep[m.queryIdx] = m.trainIdx
        return keep

    fwd = _ratio_pass(knn_fwd)
    bwd = _ratio_pass(knn_bwd)
    pts_i: list[tuple[float, float]] = []
    pts_j: list[tuple[float, float]] = []
    for qi, ti in fwd.items():
        if bwd.get(ti) == qi:  # mutual nearest neighbour
            pts_i.append(tuple(fs_i.keypoints[qi]))
            pts_j.append(tuple(fs_j.keypoints[ti]))
    return (
        np.asarray(pts_i, dtype=np.float32).reshape(-1, 2),
        np.asarray(pts_j, dtype=np.float32).reshape(-1, 2),
    )


def points_are_degenerate(pts: np.ndarray) -> bool:
    """True when a point set is too close to a line to determine a homography.

    A homography needs points in general position. If every inlier lies on one line — a
    picture rail, a skirting board, the edge of a table, the horizon over water — the fit
    is unconstrained perpendicular to that line and RANSAC will happily report a large,
    confident, meaningless inlier set. The check is scale-free: it compares the spread
    across the point cloud's minor axis against its major axis.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 4:
        return True
    centred = pts - pts.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if sv[0] < 1e-9:
        return True
    return bool(sv[1] / sv[0] < 0.02)


def is_degenerate_homography(H: np.ndarray | None) -> bool:
    """True when a fitted homography cannot describe a rotation between two views."""
    if H is None:
        return True
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return True
    if abs(float(np.linalg.det(H))) < 1e-8:
        return True
    a = H[:2, :2]
    sv = np.linalg.svd(a, compute_uv=False)
    if sv[-1] < 1e-9:
        return True
    return bool(sv[0] / sv[-1] > _MAX_AFFINE_CONDITION)


def _verify_pair(
    fs_i: FeatureSet,
    fs_j: FeatureSet,
    i: int,
    j: int,
    K: np.ndarray | None,
    config: PipelineConfig,
) -> MatchPair | None:
    """Match, RANSAC-verify and score one candidate pair. Returns None when rejected.

    ``cv2.setRNGSeed`` is called per pair because OpenCV's default RNG is *thread-local*
    and seeded from a global counter, so which worker thread happens to pick up a pair
    would otherwise decide its RANSAC sample sequence. Re-seeding from the pair's own
    identity makes each verification reproducible independently of scheduling — without
    it, two runs of the same capture drift by ~0.01 degrees.
    """
    cv2.setRNGSeed((int(config.seed) * 1000003 + i * 10007 + j) % (2**31 - 1))
    pts_i, pts_j = _raw_matches(fs_i, fs_j, float(config.match_ratio))
    n_matches = int(pts_i.shape[0])
    if n_matches < max(4, int(config.min_pair_inliers)):
        return None

    H, mask = cv2.findHomography(
        pts_i.reshape(-1, 1, 2),
        pts_j.reshape(-1, 1, 2),
        cv2.USAC_MAGSAC,
        float(config.ransac_thresh_px),
        maxIters=10000,
        confidence=0.9995,
    )
    if is_degenerate_homography(H):
        return None
    mask = np.asarray(mask, dtype=bool).ravel() if mask is not None else np.zeros(n_matches, bool)
    n_inliers = int(mask.sum())
    if n_inliers < int(config.min_pair_inliers):
        return None
    if n_inliers / float(n_matches) < MIN_INLIER_RATIO:
        return None

    inl_i = pts_i[mask]
    inl_j = pts_j[mask]
    if points_are_degenerate(inl_i) or points_are_degenerate(inl_j):
        return None

    if K is not None:
        R = geom.rotation_from_homography(H, K, K)
        if math.degrees(geom.angular_distance(R, np.eye(3))) > _MAX_PAIR_ROTATION_DEG:
            return None

    return MatchPair(
        i=i,
        j=j,
        pts_i=np.ascontiguousarray(inl_i, dtype=np.float32),
        pts_j=np.ascontiguousarray(inl_j, dtype=np.float32),
        H=np.asarray(H, dtype=np.float64),
        confidence=float(n_inliers) / (8.0 + 0.3 * float(n_matches)),
        is_loop_closure=bool(j - i > int(config.neighbour_radius)),
    )


def match_pairs(
    featuresets: list[FeatureSet],
    pairs: list[tuple[int, int]],
    K: np.ndarray | None,
    config: PipelineConfig,
    progress: ProgressReporter | None = None,
) -> list[MatchPair]:
    """Verify every candidate pair, in parallel.

    OpenCV releases the GIL inside ``knnMatch`` and ``findHomography``, so threads (rather
    than processes) give near-linear speedup without paying to pickle descriptor blocks.
    """
    if not pairs:
        return []
    workers = config.n_workers if config.n_workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    results: list[MatchPair] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_verify_pair, featuresets[i], featuresets[j], i, j, K, config)
            for i, j in pairs
        ]
        for fut in futures:
            mp = fut.result()
            done += 1
            if mp is not None:
                results.append(mp)
            if progress is not None and (done % 20 == 0 or done == len(pairs)):
                progress.stage("match", done / len(pairs), f"matched {done}/{len(pairs)} pairs")

    results.sort(key=lambda p: (p.i, p.j))
    log.info(
        "match_pairs",
        n_candidates=len(pairs),
        n_verified=len(results),
        n_loop_closures=sum(1 for p in results if p.is_loop_closure),
        median_inliers=int(np.median([p.n_inliers for p in results])) if results else 0,
    )
    return results


# ======================================================================================
# Connectivity repair
# ======================================================================================


def connected_components(n: int, pairs: Sequence[MatchPair]) -> list[list[int]]:
    """Components of the verified-pair graph, largest first, each sorted ascending."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for pair in pairs:
        if 0 <= pair.i < n and 0 <= pair.j < n:
            ra, rb = find(pair.i), find(pair.j)
            if ra != rb:
                parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values(), key=lambda c: (-len(c), c[0]))


def _cyclic_neighbours(n: int, radius: int) -> list[tuple[int, int]]:
    """Index pairs within ``radius`` steps of each other *around the loop*.

    The wrap matters: on a full turn the last keyframe and the first are neighbours in
    every sense that counts, and that is precisely the pair a sweep past a window tends
    to lose. On a partial sweep the wrap candidates simply fail verification, which costs
    a few matches and nothing else.
    """
    out: set[tuple[int, int]] = set()
    for i in range(n):
        for d in range(1, radius + 1):
            j = (i + d) % n
            if i != j:
                out.add((min(i, j), max(i, j)))
    return sorted(out)


def repair_connectivity(
    frames: list[Frame],
    pairs: list[MatchPair],
    K: np.ndarray | None,
    config: PipelineConfig,
    progress: ProgressReporter | None = None,
) -> list[MatchPair]:
    """Second chance for keyframes the first matching pass failed to link.

    A single keyframe that fails to link costs the user the whole capture, and the frames
    that fail are not random: they are the ones pointing at a window, or caught mid-swing
    and motion-blurred. On the reference failing capture the first pass produced three
    components — the graph broke at a blurred frame and again at a blown-out one — with
    the missing links sitting at 10, 20 and 26 inliers against a floor of 28. They were
    marginal, not wrong.

    So the retry is deliberately narrow: only pairs that *bridge two components*, and only
    between keyframes that are temporal neighbours (cyclically, so the wrap counts). A
    sequential neighbour carries a strong prior that the two views really do overlap,
    which is what justifies a lower inlier floor there and nowhere else. Everything that
    guards against a *wrong* match — the ratio test's mutual check, MAGSAC, the inlier
    ratio, the degeneracy tests — stays at full strength.

    That last sentence used to be untrue in the one way that mattered. The retry loosened
    Lowe's ratio *and* the inlier floor together, and loosening the ratio inflates the
    denominator :data:`MIN_INLIER_RATIO` is measured against, so the retry rejected on the
    ratio every link it had just recovered on the count — all five of them, on the capture
    it was written for, silently. It now escalates instead: see :func:`_retry_stages`.
    """
    n = len(frames)
    if n < 2:
        return pairs

    stages = _retry_stages(config)
    neighbours = _cyclic_neighbours(n, max(1, int(config.neighbour_radius)))
    caches: list[dict[int, FeatureSet]] = [{} for _ in stages]
    recovered: list[MatchPair] = []

    for round_idx in range(RETRY_MAX_ROUNDS):
        components = connected_components(n, pairs)
        if len(components) <= 1:
            break
        owner = {k: ci for ci, comp in enumerate(components) for k in comp}
        bridges = [(i, j) for i, j in neighbours if owner[i] != owner[j]]
        if not bridges:
            break

        log.info(
            "repair_connectivity.round",
            round=round_idx,
            n_components=len(components),
            component_sizes=[len(c) for c in components],
            n_bridge_candidates=len(bridges),
        )
        if progress is not None:
            progress.stage(
                "match",
                1.0,
                f"re-matching {len(components)} disconnected groups",
            )

        existing = {(p.i, p.j) for p in pairs}
        fresh: list[MatchPair] = []
        for stage_idx, (stage_config, label) in enumerate(stages):
            todo = [pair for pair in bridges if pair not in existing]
            if not todo:
                break
            cache = caches[stage_idx]
            needed = sorted({k for pair in todo for k in pair} - set(cache))
            if needed:
                detected = detect_features([frames[k] for k in needed], stage_config)
                for k, fs in zip(needed, detected, strict=True):
                    # detect_features numbers its output 0..len-1; restore the real index.
                    fs.frame_index = k
                    cache[k] = fs
            found = [
                p
                for p in _match_selected(cache, todo, K, stage_config)
                if (p.i, p.j) not in existing
            ]
            log.info(
                "repair_connectivity.stage",
                round=round_idx,
                stage=label,
                inlier_floor=int(stage_config.min_pair_inliers),
                match_ratio=float(stage_config.match_ratio),
                n_tried=len(todo),
                n_recovered=len(found),
            )
            for p in found:
                existing.add((p.i, p.j))
            fresh.extend(found)
            # Stop the moment the graph is whole. The retry's job is connectivity, and a
            # later stage that keeps adding edges after that is no longer repairing
            # anything — it is adding its weakest-supported matches to a solve that did not
            # ask for them. Measured on the nodal gate capture: the wide stage reconnects it
            # on its own, and letting the narrow stage then contribute one extra pair moved
            # the wrap-column ratio from 1.21 to 2.55 and failed the gate.
            if len(connected_components(n, pairs + fresh)) <= 1:
                break

        if not fresh:
            break
        pairs = sorted(pairs + fresh, key=lambda p: (p.i, p.j))
        recovered.extend(fresh)

    if recovered:
        log.info(
            "repair_connectivity.recovered",
            n=len(recovered),
            pairs=[(p.i, p.j, p.n_inliers) for p in recovered],
            components_after=[len(c) for c in connected_components(n, pairs)],
        )
    return pairs


def _retry_stages(config: PipelineConfig) -> list[tuple[PipelineConfig, str]]:
    """The escalation :func:`repair_connectivity` walks, easiest relaxation first.

    Both stages lower the inlier *count* floor, because that is what a bridging sequential
    neighbour's strong overlap prior actually justifies. They differ in nothing else but
    Lowe's ratio, and the order matters: see :data:`RETRY_MATCH_RATIO` for the measurement
    showing that most marginal links are ratio-healthy at the original threshold and only
    fall under :data:`MIN_INLIER_RATIO` once the threshold is loosened.
    """
    floor = max(12, int(config.min_pair_inliers) // 2)
    return [
        # The wide search first, exactly as shipped. Where it works it is the *better*
        # recovery, not merely an equivalent one: the looser ratio test admits more
        # correspondences, so the pair comes back with more inliers supporting it, and the
        # bundle gets a stronger constraint. Measured on the nodal gate capture, where both
        # stages can recover the same nine bridges: this stage returns them with 17-29
        # inliers against 14-27 from the narrow one, and taking the narrow version instead
        # tripled the wrap-column ratio (1.21 -> 3.65) and failed the gate.
        (
            replace(
                config,
                min_pair_inliers=floor,
                match_ratio=RETRY_MATCH_RATIO,
                max_features=int(config.max_features) * RETRY_FEATURE_MULTIPLE,
            ),
            "loose_ratio",
        ),
        # Then, only for bridges the wide search *rejected*, the narrow one: same match set
        # the first pass judged, so :data:`MIN_INLIER_RATIO` keeps its calibration and the
        # pair is judged on the one thing that was marginal. This is the stage that recovers
        # what the shipped retry killed on a ratio it had inflated itself.
        (replace(config, min_pair_inliers=floor), "inlier_floor"),
    ]


def _match_selected(
    cache: dict[int, FeatureSet],
    wanted: list[tuple[int, int]],
    K: np.ndarray | None,
    config: PipelineConfig,
) -> list[MatchPair]:
    """Verify an explicit list of index pairs against a sparse feature-set cache."""
    out: list[MatchPair] = []
    workers = config.n_workers if config.n_workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_verify_pair, cache[i], cache[j], i, j, K, config) for i, j in wanted
        ]
        for fut in futures:
            mp = fut.result()
            if mp is not None:
                out.append(mp)
    return out
