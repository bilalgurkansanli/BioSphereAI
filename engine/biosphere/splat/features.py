"""Learned local features and matches, written into a COLMAP database.

WHY NOT COLMAP'S OWN SIFT. Rooms are the worst case for SIFT: painted walls, repeated
tiles, glass. The official pycolmap wheel for Windows is built without ONNX, so its
ALIKED/LightGlue path aborts the process; the same models run happily in torch through the
``lightglue`` package, on the GPU that will train the splats anyway. So features and
matches are computed here and *imported* into the database the way hloc does it, and
pycolmap is left with what it is best at: geometric verification, mapping, bundle
adjustment.

PAIRS. A walkthrough is a sequence, so every frame is matched to its ``overlap`` successors
(quadratically spaced further out, as COLMAP's sequential matcher does). That alone drifts:
the second lap around the room never meets the first. Loop closure comes from image
retrieval on a cheap global descriptor (pooled local descriptors plus a thumbnail), which
proposes a few far-apart candidates per frame that LightGlue then confirms or rejects.
Multiple sequences (the general walkthrough and each room) are matched within themselves
sequentially and across each other by retrieval only.

CONVENTIONS. LightGlue keypoints are pixel *centre* coordinates with the first pixel at
``(0, 0)``; COLMAP wants the first pixel centre at ``(0.5, 0.5)``, hence the ``+ 0.5`` on
import. Keypoints are extracted on a downscaled copy (``extract_long_side``) and rescaled by
the extractor to full-resolution coordinates, so the database always speaks the pixel
frame of the images on disk.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

__all__ = [
    "MatchingConfig",
    "MatchingReport",
    "LocalFeatures",
    "sequential_pairs",
    "retrieval_pairs",
    "extract_features",
    "match_pairs",
    "build_database",
    "release_models",
]


@dataclass(frozen=True)
class MatchingConfig:
    #: ``aliked`` (Apache-2.0 weights, the default), ``disk`` or ``superpoint`` (research licence).
    extractor: str = "aliked"
    max_keypoints: int = 3072
    #: ALIKED's score threshold. 0.1 is its own default and fine for a sharp 4K frame; a
    #: compressed or blurred frame of a white-walled flat keeps 500 points at that level and
    #: needs it lowered to reach the corners the mapper lives on.
    detection_threshold: float = 0.02
    #: Long side of the image the detector sees. 1024 is LightGlue's own default; the room
    #: frames are 1920 wide, so this halves the work and loses nothing a splat would keep.
    extract_long_side: int = 1024
    #: Sequential window: every frame is paired with the next ``overlap`` frames, then with
    #: frames 2^k apart up to that span when ``quadratic``.
    overlap: int = 12
    quadratic: bool = True
    #: Loop closure: top-k retrieval candidates per frame, at least ``retrieval_min_gap``
    #: frames away in its own sequence (nearer ones are covered by the window).
    retrieval_topk: int = 8
    retrieval_min_gap: int = 24
    #: What the retrieval looks with: ``dinov2`` (ViT-S/14 CLS + mean patch token, fetched
    #: once through torch.hub) or ``pooled`` (pooled local descriptors plus a thumbnail; no
    #: download, much weaker on a flat's repeated white walls). DINOv2 falls back to pooled
    #: when it cannot be loaded.
    retrieval: str = "dinov2"
    #: A loop pair that verified with at least this many inliers pulls its neighbours in:
    #: (i+-k, j+-k) for k up to ``transitive_neighbours`` are matched too. Cheap, and it is
    #: what turns one lucky retrieval hit between two passes into a solid seam.
    transitive_min_inliers: int = 60
    transitive_neighbours: int = 2
    #: Pairs with fewer LightGlue matches than this are not written at all.
    min_matches: int = 30
    #: The focal length the mappers start from, as a fraction of the frame's long side.
    #: COLMAP's own 1.2 is a 26° lens - a phone video is 70-110°, and the incremental mapper's
    #: first pair is initialised (and its early bundle adjustments anchored) at this value.
    #: 0.75 of the long side is a 26 mm phone lens in portrait; the ultra-wide sits near 0.45.
    focal_factor: float = 0.75
    #: LightGlue confidence pruning (its own defaults are 0.95 / 0.99); lower is faster.
    depth_confidence: float = 0.95
    width_confidence: float = 0.99
    filter_threshold: float = 0.1


@dataclass
class MatchingReport:
    n_images: int = 0
    n_pairs_sequential: int = 0
    n_pairs_retrieval: int = 0
    n_pairs_cross: int = 0
    n_pairs_matched: int = 0
    n_pairs_verified: int = 0
    mean_keypoints: float = 0.0
    seconds_extract: float = 0.0
    seconds_match: float = 0.0
    seconds_verify: float = 0.0
    #: Loop pairs that verified strongly enough to seed the second round, and what it added.
    n_loop_seeds: int = 0
    n_pairs_transitive: int = 0
    retrieval: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class LocalFeatures:
    """Per-image keypoints (full-resolution pixel centres) and descriptors, plus a global one."""

    names: list[str]
    keypoints: list[np.ndarray] = field(default_factory=list)  # (N_i, 2) float32
    descriptors: list[torch.Tensor] = field(default_factory=list)  # (N_i, D) float16 on device
    sizes: list[tuple[int, int]] = field(default_factory=list)  # (width, height)
    global_desc: torch.Tensor | None = None  # (n, G) float32, L2-normalised

    def __len__(self) -> int:
        return len(self.names)

    def subset(self, indices: list[int]) -> LocalFeatures:
        """The same features for a subset of the images (order preserved)."""
        gd = None if self.global_desc is None else self.global_desc[torch.as_tensor(list(indices), device=self.global_desc.device)]
        return LocalFeatures(
            names=[self.names[i] for i in indices],
            keypoints=[self.keypoints[i] for i in indices],
            descriptors=[self.descriptors[i] for i in indices],
            sizes=[self.sizes[i] for i in indices],
            global_desc=gd,
        )

    @staticmethod
    def concatenate(parts: list[LocalFeatures]) -> LocalFeatures:
        return LocalFeatures(
            names=[n for p in parts for n in p.names],
            keypoints=[k for p in parts for k in p.keypoints],
            descriptors=[d for p in parts for d in p.descriptors],
            sizes=[s for p in parts for s in p.sizes],
            global_desc=torch.cat([p.global_desc for p in parts])
            if all(p.global_desc is not None for p in parts) and len({p.global_desc.shape[1] for p in parts}) == 1
            else None,
        )


def sequential_pairs(n: int, overlap: int, quadratic: bool = True, offset: int = 0) -> list[tuple[int, int]]:
    """Index pairs of COLMAP's sequential matcher: neighbours, then powers of two."""
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for k in range(1, overlap + 1):
            j = i + k
            if j < n:
                pairs.append((i + offset, j + offset))
        if quadratic:
            k = 1
            while (1 << k) <= overlap and i + (1 << k) < n:
                k += 1
            step = 1 << k
            while i + step < n and step <= overlap * overlap:
                pairs.append((i + offset, i + step + offset))
                step <<= 1
    return sorted(set(pairs))


def retrieval_pairs(
    global_desc: torch.Tensor,
    sequence_of: np.ndarray,
    index_in_sequence: np.ndarray,
    *,
    topk: int,
    min_gap: int,
    min_similarity: float = 0.25,
) -> list[tuple[int, int]]:
    """Far-apart pairs that look alike.

    ``sequence_of[i]`` and ``index_in_sequence[i]`` place image ``i`` in its own video so
    that candidates inside the sequential window are skipped, while frames from *other*
    sequences are always eligible (that is how rooms meet the general walkthrough).
    """
    n = int(global_desc.shape[0])
    if n < 2:
        return []
    sim = global_desc @ global_desc.T
    seq = torch.as_tensor(sequence_of, device=sim.device)
    idx = torch.as_tensor(index_in_sequence, device=sim.device)
    same = seq[:, None] == seq[None, :]
    near = (idx[:, None] - idx[None, :]).abs() < min_gap
    sim = sim.masked_fill(same & near, -1.0)
    sim.fill_diagonal_(-1.0)
    k = min(topk, n - 1)
    vals, cols = torch.topk(sim, k, dim=1)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for v, j in zip(vals[i].tolist(), cols[i].tolist(), strict=True):
            if v < min_similarity:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            pairs.add((a, b))
    return sorted(pairs)


def _load_rgb(path: Path, long_side: int) -> tuple[np.ndarray, float]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable image {path}")
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > long_side:
        scale = long_side / float(max(h, w))
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), scale


#: Networks built once per process. The chain probe matches every consecutive pair on its
#: own (``chain.link_inliers``), and building LightGlue is not free: its nine transformer
#: layers are randomly initialised on the CPU before the weights are loaded over them. Built
#: per pair, a 82-s walk's ~250 links plus its bridges spent 25 minutes of a contended
#: laptop's CPU in ``kaiming_uniform_`` (py-spy) with the GPU idle; built once, the same
#: probe is bounded by the matching itself. ``release_models`` hands the memory back
#: before the trainer needs the GPU.
_MODELS: dict[tuple, Any] = {}


def release_models() -> None:
    """Drop the cached networks (and their GPU memory)."""
    _MODELS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_extractor(config: MatchingConfig, device: torch.device):
    key = ("extractor", config.extractor, config.max_keypoints, config.detection_threshold, str(device))
    if key in _MODELS:
        return _MODELS[key]
    from lightglue import ALIKED, DISK, SuperPoint

    if config.extractor == "aliked":
        model = ALIKED(max_num_keypoints=config.max_keypoints, detection_threshold=config.detection_threshold)
    elif config.extractor == "disk":
        model = DISK(max_num_keypoints=config.max_keypoints)
    elif config.extractor == "superpoint":
        model = SuperPoint(max_num_keypoints=config.max_keypoints)
    else:
        raise ValueError(f"unknown extractor {config.extractor!r}")
    _MODELS[key] = model.eval().to(device)
    return _MODELS[key]


def _make_matcher(config: MatchingConfig, device: torch.device):
    key = ("lightglue", config.extractor, config.depth_confidence, config.width_confidence, config.filter_threshold, str(device))
    if key not in _MODELS:
        from lightglue import LightGlue

        _MODELS[key] = LightGlue(
            features=config.extractor,
            depth_confidence=config.depth_confidence,
            width_confidence=config.width_confidence,
            filter_threshold=config.filter_threshold,
        ).eval().to(device)
    return _MODELS[key]


_DINO_INPUT = (392, 224)  # (h, w) for portrait frames, multiples of the 14-px patch


def _dino_model(device: torch.device):
    """DINOv2 ViT-S/14 through torch.hub, cached under the models dir when one is configured."""
    key = ("dinov2_vits14", str(device))
    if key in _MODELS:
        return _MODELS[key]
    hub = os.environ.get("BIOSPHERE_AI_MODELS_DIR")
    if hub:
        torch.hub.set_dir(str(Path(hub) / "torch_hub"))
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True, verbose=False)
    _MODELS[key] = model.eval().to(device)
    return _MODELS[key]


@torch.no_grad()
def _dino_descriptors(model, thumbs: list[np.ndarray], device: torch.device, batch: int = 32) -> torch.Tensor:
    """L2-normalised [CLS | mean patch] embeddings of ``thumbs`` (RGB uint8 at the DINO input size)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out: list[torch.Tensor] = []
    for i in range(0, len(thumbs), batch):
        x = torch.stack([torch.from_numpy(t).permute(2, 0, 1) for t in thumbs[i : i + batch]]).to(device).float() / 255.0
        x = (x - mean) / std
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            feats = model.forward_features(x)
        g = torch.cat([feats["x_norm_clstoken"].float(), feats["x_norm_patchtokens"].float().mean(1)], dim=1)
        out.append(g / g.norm(dim=1, keepdim=True).clamp_min(1e-6))
    return torch.cat(out) if out else torch.zeros((0, 768), device=device)


def _dino_input(rgb: np.ndarray) -> np.ndarray:
    h, w = _DINO_INPUT
    if rgb.shape[0] < rgb.shape[1]:  # a landscape frame: swap
        h, w = w, h
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def _tiny_descriptor(rgb: np.ndarray) -> np.ndarray:
    """A 24x16 luminance thumbnail, mean-free and unit-length: coarse layout similarity."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    tiny = cv2.resize(gray, (24, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    tiny -= tiny.mean()
    return tiny / max(float(np.linalg.norm(tiny)), 1e-6)


@torch.no_grad()
def extract_features(
    image_dir: Path,
    names: list[str],
    *,
    config: MatchingConfig,
    device: torch.device,
    progress=None,
) -> LocalFeatures:
    """Detect and describe every image; also build the retrieval descriptor."""
    extractor = _make_extractor(config, device)
    feats = LocalFeatures(names=list(names))
    globals_: list[torch.Tensor] = []
    dino_inputs: list[np.ndarray] = []
    t0 = time.perf_counter()
    for i, name in enumerate(names):
        rgb, scale = _load_rgb(Path(image_dir) / name, config.extract_long_side)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(device)
        out = extractor.extract(tensor, resize=None)
        kpts = out["keypoints"][0]
        desc = out["descriptors"][0]
        # Back to full-resolution pixel centres, then COLMAP's half-pixel origin.
        kpts_full = (kpts + 0.5) / scale - 0.5 + 0.5
        feats.keypoints.append(kpts_full.float().cpu().numpy().astype(np.float32))
        # Descriptors rest on the CPU between uses: a property has several videos' worth and the
        # GPU is needed for training; the matcher moves them back per pair.
        feats.descriptors.append(desc.half().cpu())
        h, w = rgb.shape[:2]
        feats.sizes.append((int(round(w / scale)), int(round(h / scale))))
        pooled = torch.cat([desc.mean(0), desc.amax(0)]) if desc.shape[0] else torch.zeros(2 * desc.shape[1], device=device)
        pooled = pooled / pooled.norm().clamp_min(1e-6)
        tiny = torch.from_numpy(_tiny_descriptor(rgb)).to(device)
        g = torch.cat([pooled * 0.8, tiny * 0.6])
        globals_.append(g / g.norm().clamp_min(1e-6))
        if config.retrieval == "dinov2":
            dino_inputs.append(_dino_input(rgb))
        if progress is not None and (i % 10 == 0 or i == len(names) - 1):
            progress((i + 1) / len(names))
    feats.global_desc = torch.stack(globals_).float() if globals_ else torch.zeros((0, 1), device=device)
    feats.retrieval_kind = "pooled"  # type: ignore[attr-defined]
    if config.retrieval == "dinov2" and dino_inputs:
        try:
            model = _dino_model(device)
            feats.global_desc = _dino_descriptors(model, dino_inputs, device)
            feats.retrieval_kind = "dinov2"  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - retrieval is an accelerator, not a requirement
            feats.retrieval_warning = f"dinov2_unavailable: {type(exc).__name__}: {str(exc)[:120]}"  # type: ignore[attr-defined]
    feats.seconds = time.perf_counter() - t0  # type: ignore[attr-defined]
    return feats


@torch.no_grad()
def match_pairs(
    feats: LocalFeatures,
    pairs: list[tuple[int, int]],
    *,
    config: MatchingConfig,
    device: torch.device,
    progress=None,
) -> dict[tuple[int, int], np.ndarray]:
    """LightGlue over ``pairs``; returns ``(K, 2)`` keypoint index pairs per image pair."""
    matcher = _make_matcher(config, device)
    out: dict[tuple[int, int], np.ndarray] = {}
    # Only the images the pairs name go to the device, each once. Moving every image of the
    # set on every call made a one-pair call (the chain probe's) cost a whole walk's features.
    on_device: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def image(i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if i not in on_device:
            w, h = feats.sizes[i]
            on_device[i] = (
                torch.from_numpy(feats.keypoints[i]).to(device) - 0.5,  # matcher wants 0-origin
                feats.descriptors[i].to(device).float(),
                torch.tensor([w, h], dtype=torch.float32, device=device),
            )
        return on_device[i]

    for n, (i, j) in enumerate(pairs):
        if feats.keypoints[i].shape[0] < 8 or feats.keypoints[j].shape[0] < 8:
            continue
        ki, di, si = image(i)
        kj, dj, sj = image(j)
        data = {
            "image0": {"keypoints": ki[None], "descriptors": di[None], "image_size": si[None]},
            "image1": {"keypoints": kj[None], "descriptors": dj[None], "image_size": sj[None]},
        }
        res = matcher(data)
        m = res["matches"][0]
        if m.shape[0] >= config.min_matches:
            out[(i, j)] = m.cpu().numpy().astype(np.uint32)
        if progress is not None and (n % 50 == 0 or n == len(pairs) - 1):
            progress((n + 1) / max(len(pairs), 1))
    return out


def build_database(
    db_path: Path,
    image_dir: Path,
    sequences: list[list[str]] | None = None,
    *,
    feature_sets: list[LocalFeatures] | None = None,
    camera_model: str = "OPENCV",
    config: MatchingConfig | None = None,
    device: torch.device | None = None,
    progress=None,
    shared_camera: bool = True,
) -> MatchingReport:
    """Create a COLMAP database with images, features, matches and verified geometry.

    Give either ``sequences`` (ordered image-name lists, one per video, features computed
    here) or ``feature_sets`` (one :class:`LocalFeatures` per video, already extracted, e.g.
    the general walkthrough's features reused for every room). Names are relative to
    ``image_dir`` and may contain sub-directories. One camera is shared by every image when
    ``shared_camera`` (one phone, one lens), which is what keeps intrinsics well-conditioned.
    """
    import pycolmap

    config = config or MatchingConfig()
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with pycolmap.Database.open(db_path):
        pass  # import_images wants an existing (empty) database

    def sub(stage: str, span: tuple[float, float]):
        if progress is None:
            return None
        a, b = span
        return lambda f: progress(stage, a + (b - a) * float(f))

    t0 = time.perf_counter()
    if feature_sets is None:
        if sequences is None:
            raise ValueError("give sequences or feature_sets")
        feature_sets = [
            extract_features(image_dir, seq, config=config, device=device, progress=sub("features", (k / len(sequences), (k + 1) / len(sequences))))
            for k, seq in enumerate(sequences)
        ]
    sequences = [fs.names for fs in feature_sets]
    feats = LocalFeatures.concatenate(feature_sets)
    names = feats.names
    seq_of = np.concatenate([np.full(len(s), k, dtype=np.int64) for k, s in enumerate(sequences)])
    idx_in = np.concatenate([np.arange(len(s), dtype=np.int64) for s in sequences])
    report = MatchingReport(n_images=len(names))
    report.seconds_extract = time.perf_counter() - t0
    report.mean_keypoints = float(np.mean([k.shape[0] for k in feats.keypoints])) if names else 0.0

    # Images and the camera. import_images infers width/height from the files and sets the
    # 1.2 * long-side focal prior COLMAP always starts from.
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    reader.default_focal_length_factor = float(config.focal_factor)
    pycolmap.import_images(
        db_path, image_dir, camera_mode=pycolmap.CameraMode.SINGLE if shared_camera else pycolmap.CameraMode.AUTO,
        image_names=names, options=reader,
    )

    pairs: list[tuple[int, int]] = []
    offset = 0
    for seq in sequences:
        pairs += sequential_pairs(len(seq), config.overlap, config.quadratic, offset=offset)
        offset += len(seq)
    report.n_pairs_sequential = len(pairs)
    seq_set = set(pairs)
    retrieved = retrieval_pairs(
        feats.global_desc, seq_of, idx_in, topk=config.retrieval_topk, min_gap=config.retrieval_min_gap,
    )
    retrieved = [p for p in retrieved if p not in seq_set]
    report.n_pairs_retrieval = sum(1 for a, b in retrieved if seq_of[a] == seq_of[b])
    report.n_pairs_cross = len(retrieved) - report.n_pairs_retrieval
    pairs = sorted(seq_set | set(retrieved))

    t0 = time.perf_counter()
    matches = match_pairs(feats, pairs, config=config, device=device, progress=sub("match", (0.0, 0.85)))
    report.seconds_match = time.perf_counter() - t0
    report.n_pairs_matched = len(matches)

    with pycolmap.Database.open(db_path) as db:
        ids = {im.name: im.image_id for im in db.read_all_images()}
        with pycolmap.DatabaseTransaction(db):
            for name, kpts in zip(names, feats.keypoints, strict=True):
                db.write_keypoints(ids[name], kpts)
            for (i, j), m in matches.items():
                db.write_matches(ids[names[i]], ids[names[j]], m)
    pairs_txt = db_path.with_suffix(".pairs.txt")
    pairs_txt.write_text("\n".join(f"{names[i]} {names[j]}" for (i, j) in matches) + "\n", encoding="utf-8")

    t0 = time.perf_counter()
    if progress is not None:
        progress("match", 0.9)
    pycolmap.verify_matches(db_path, pairs_txt)
    report.seconds_verify = time.perf_counter() - t0

    # Second round: the neighbours of the loop pairs that held. A retrieval hit joins two
    # frames of different passes; the frames around them see the same wall, and matching
    # those too turns one edge between two chains into a band of them - which is what the
    # global mapper needs to not fold one pass onto the other.
    if config.transitive_neighbours > 0 and retrieved:
        inliers = _verified_inliers(db_path)
        seed = [(i, j) for (i, j) in retrieved if inliers.get((names[i], names[j]), 0) >= config.transitive_min_inliers]
        extra: set[tuple[int, int]] = set()
        k = config.transitive_neighbours
        n = len(names)
        for i, j in seed:
            for di in range(-k, k + 1):
                for dj in range(-k, k + 1):
                    a, b = i + di, j + dj
                    if not (0 <= a < n and 0 <= b < n) or a == b:
                        continue
                    if seq_of[a] != seq_of[i] or seq_of[b] != seq_of[j]:
                        continue
                    pair = (a, b) if a < b else (b, a)
                    if pair not in seq_set and pair not in matches:
                        extra.add(pair)
        if extra:
            t0 = time.perf_counter()
            more = match_pairs(feats, sorted(extra), config=config, device=device)
            report.seconds_match += time.perf_counter() - t0
            if more:
                with pycolmap.Database.open(db_path) as db:
                    ids = {im.name: im.image_id for im in db.read_all_images()}
                    with pycolmap.DatabaseTransaction(db):
                        for (i, j), m in more.items():
                            db.write_matches(ids[names[i]], ids[names[j]], m)
                more_txt = db_path.with_suffix(".pairs2.txt")
                more_txt.write_text("\n".join(f"{names[i]} {names[j]}" for (i, j) in more) + "\n", encoding="utf-8")
                t0 = time.perf_counter()
                pycolmap.verify_matches(db_path, more_txt)
                report.seconds_verify += time.perf_counter() - t0
                report.n_pairs_matched += len(more)
        report.n_pairs_transitive = len(extra)
        report.n_loop_seeds = len(seed)

    with pycolmap.Database.open(db_path) as db:
        report.n_pairs_verified = int(db.num_verified_image_pairs())
    report.retrieval = next((fs.retrieval_kind for fs in feature_sets if hasattr(fs, "retrieval_kind")), "pooled")
    for fs in feature_sets:
        warn = getattr(fs, "retrieval_warning", None)
        if warn and warn not in report.warnings:
            report.warnings.append(warn)
    if progress is not None:
        progress("match", 1.0)
    return report


def _verified_inliers(db_path: Path) -> dict[tuple[str, str], int]:
    """``(name_a, name_b) -> inliers`` of every verified pair in the database (a before b by id)."""
    con = sqlite3.connect(str(db_path))
    try:
        names = {int(i): str(n) for i, n in con.execute("SELECT image_id, name FROM images")}
        out: dict[tuple[str, str], int] = {}
        for pair_id, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries"):
            a, b = divmod(int(pair_id), 2147483647)
            if a in names and b in names:
                out[(names[a], names[b])] = int(rows or 0)
        return out
    finally:
        con.close()
