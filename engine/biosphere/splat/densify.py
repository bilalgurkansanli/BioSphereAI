"""A dense starting cloud for the trainer, from Depth Anything 3 told where the cameras are.

WHY. Structure from motion triangulates what it can match, and a white-walled flat gives it
corners and door frames and almost nothing on the walls themselves - 9,600 points for 79
frames of a living room on the first real capture. The trainer starts from those points and
has to GROW splats out over every bare wall, and growth from a handful of seeds is where the
translucent veils and the needles come from. Depth Anything 3, given the solved poses and
intrinsics, returns a depth map per frame in the same metric frame; every pixel that survives
a confidence cut and agrees with its neighbouring views becomes a point. Brush starts from an
``init.ply`` in the dataset when there is one.

MEASURED on that capture (a 79-frame part: 71 train / 8 held-out frames, fast mode, 12k
steps, the same held-out frames for every row; PSNR raw / after the delivery clean):

    init                                  splats   held-out PSNR        SSIM
    SfM points (9.6k)                     221k     22.66 / 21.89 dB     0.882
    SfM points, growth threshold halved   366k     22.08 / 20.87        0.878
    DA3-BASE, 300k points                 800k     23.16 / 23.12        0.874
    DA3-BASE, 200k points, capped 250k    250k     20.87 / 20.86        0.826   (speckle)
    DA3-BASE, 50k points  <- default      423k     23.09 / 23.04        0.882

So it is the init, not the splat count (more growth from the SfM points is worse), and it
needs room to grow (a cap at the init's own size leaves the small seeds as speckle). 50k
seeds buy the same PSNR as 300k at half the delivered size. The SfM model's "delivered"
loss was mostly the giant-splat cap, since made distance-aware (clean.py: 21.89 -> ~22.6);
against that the dense init is still +0.4 dB, and on the walls it is what the eye sees: solid
plaster where the sparse init leaves translucent veils (exp sheets in the report).

Its depth agrees with the SfM points to 5.0 % median (DA3-SMALL: 6.9 %, so BASE it is).
``engine/scripts/bench_dense_init.py`` reruns the comparison on any solved model.

LICENCES. Only the Apache-2.0 checkpoints are allowed here (DA3-BASE, DA3-SMALL, DA3METRIC-
LARGE, DA3MONO-LARGE); the larger any-view models are CC BY-NC and this product ships closed.
The DA3 code is Apache-2.0 too. It is not a pip dependency: its package pins opencv-python
(which silently replaces the contrib build the panorama engine needs) and numpy<2, so it is
used from a source checkout, like Brush is used from its release binary:

    BIOSPHERE_DA3_SRC       the checkout's ``src`` dir (default <tools>/Depth-Anything-3/src)
    BIOSPHERE_AI_MODELS_DIR weights at <dir>/da3/<NAME>/{config.json,model.safetensors}, or a
                            Hugging Face cache at <dir>/hf (fetched only when
                            BIOSPHERE_AI_ALLOW_DOWNLOADS is true)

Missing either, the trainer starts from the SfM points as before and the report says why.
"""

from __future__ import annotations

import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["DenseInitConfig", "DenseInitReport", "APACHE_CHECKPOINTS", "dense_init", "fuse_depths", "write_point_ply"]

APACHE_CHECKPOINTS = ("DA3-BASE", "DA3-SMALL", "DA3METRIC-LARGE", "DA3MONO-LARGE")
_DEFAULT_TOOLS = Path(os.environ.get("BIOSPHERE_TOOLS_DIR", "C:/biosphere-data/tools"))


@dataclass(frozen=True)
class DenseInitConfig:
    #: Off by default: measured on one 79-frame part only (table above), not yet through a
    #: whole build or the berlin bench. Turn on per quality preset once it has been.
    enabled: bool = False
    #: DA3-BASE over DA3-SMALL: 5.0 % against 6.9 % median depth error on the real capture,
    #: for 0.12B against 0.08B parameters (2.5 GB of VRAM at 24 views either way).
    model: str = "DA3-BASE"
    #: DA3's own default input size (long side); the depth comes back at that resolution.
    process_res: int = 504
    #: Views per forward pass, and how many each pass shares with the previous one.
    chunk: int = 24
    overlap: int = 4
    #: Pixels below this percentile of a frame's own confidence are not used.
    conf_percentile: float = 30.0
    #: A pixel becomes a point only if at least ``min_agree`` of its ``n_check`` nearest views
    #: see the same depth within ``rel_tol``. On the real capture 58-69 % of the confident
    #: pixels pass - what fails is mostly the far side of windows and the frame edges.
    n_check: int = 6
    rel_tol: float = 0.04
    min_agree: int = 2
    #: One point per voxel (metres, the property frame), and a cap on the total: 50k seeds
    #: matched 300k on held-out PSNR (23.09 vs 23.16 dB) and grew to 423k splats, not 800k.
    voxel_m: float = 0.02
    max_points: int = 50_000
    #: Every n-th pixel of the depth map in each direction.
    stride: int = 2


@dataclass
class DenseInitReport:
    used: bool = False
    model: str = ""
    n_views: int = 0
    n_candidates: int = 0
    n_consistent: int = 0
    n_points: int = 0
    seconds: float = 0.0
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------


class _PosePath3D:
    """The one piece of ``evo`` DA3 imports: Umeyama Sim3 alignment of camera positions."""

    def __init__(self, poses_se3) -> None:
        self.poses_se3 = [np.array(p, dtype=np.float64) for p in poses_se3]

    def align(self, ref, correct_scale: bool = True):
        src = np.stack([p[:3, 3] for p in self.poses_se3])
        tgt = np.stack([p[:3, 3] for p in ref.poses_se3])
        s, R, t = _umeyama(src, tgt, with_scale=correct_scale)
        out = []
        for p in self.poses_se3:
            q = np.eye(4)
            q[:3, :3] = R @ p[:3, :3]
            q[:3, 3] = R @ (s * p[:3, 3]) + t
            out.append(q)
        self.poses_se3 = out
        return R, t, s


def _umeyama(src: np.ndarray, tgt: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    a, b = src - mu_s, tgt - mu_t
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max(float((a**2).sum(1).mean()), 1e-12)) if with_scale else 1.0
    return s, R, mu_t - s * (R @ mu_s)


def _stub_export_modules() -> None:
    """DA3 imports exporters (videos, meshes, trajectory plots) at module level; none is used."""

    def stub(name: str, **attrs) -> None:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod

    for name in ("moviepy", "moviepy.editor", "trimesh", "evo", "evo.core", "evo.core.geometry", "evo.core.lie_algebra", "evo.core.sync"):
        stub(name)
    stub("evo.core.trajectory", PosePath3D=_PosePath3D)
    stub("plyfile", PlyData=object, PlyElement=object)


def _da3_src() -> Path:
    return Path(os.environ.get("BIOSPHERE_DA3_SRC", str(_DEFAULT_TOOLS / "Depth-Anything-3" / "src")))


def _weights_ref(name: str) -> str:
    """A local weights folder when one exists, else the Hugging Face repo id."""
    models = os.environ.get("BIOSPHERE_AI_MODELS_DIR")
    if models:
        local = Path(models) / "da3" / name
        if (local / "model.safetensors").exists() and (local / "config.json").exists():
            return str(local)
        os.environ.setdefault("HF_HOME", str(Path(models) / "hf"))
    if os.environ.get("BIOSPHERE_AI_ALLOW_DOWNLOADS", "").lower() not in ("1", "true", "yes"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return f"depth-anything/{name}"


def load_model(name: str, device):
    """DA3 ``name`` on ``device``. Raises when the code or the weights are not available."""
    name = name.upper()
    if name not in APACHE_CHECKPOINTS:
        raise ValueError(f"{name} is not an Apache-2.0 DA3 checkpoint ({', '.join(APACHE_CHECKPOINTS)})")
    src = _da3_src()
    if not (src / "depth_anything_3" / "api.py").exists():
        raise FileNotFoundError(f"Depth Anything 3 source not found at {src} (set BIOSPHERE_DA3_SRC)")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    _stub_export_modules()
    from depth_anything_3.api import DepthAnything3  # type: ignore[import-not-found]

    return DepthAnything3.from_pretrained(_weights_ref(name)).to(device=device).eval()


# ---------------------------------------------------------------------------------------------
# depth and fusion
# ---------------------------------------------------------------------------------------------


def _predict(model, rec, image_root: Path, config: DenseInitConfig) -> dict[str, dict[str, np.ndarray]]:
    import torch

    imgs = sorted((img for img in rec.images.values() if img.has_pose), key=lambda i: i.name)
    out: dict[str, dict[str, np.ndarray]] = {}
    step = max(config.chunk - config.overlap, 1)
    for s in range(0, max(len(imgs) - config.overlap, 1), step):
        batch = imgs[s : s + config.chunk]
        if not batch:
            continue
        paths, exts, ints = [], [], []
        for img in batch:
            cam = rec.cameras[img.camera_id]
            fx, fy, cx, cy = (float(v) for v in cam.params[:4])
            E = np.eye(4, dtype=np.float32)
            E[:3, :4] = np.asarray(img.cam_from_world().matrix(), dtype=np.float32)
            paths.append(str(Path(image_root) / img.name))
            exts.append(E)
            ints.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32))
        with torch.no_grad():
            pred = model.inference(paths, extrinsics=np.stack(exts), intrinsics=np.stack(ints),
                                   align_to_input_ext_scale=True, process_res=config.process_res)
        for k, img in enumerate(batch):
            if img.name in out:
                continue  # the earlier pass already gave it a depth with more context behind it
            out[img.name] = {
                "depth": np.asarray(pred.depth[k], dtype=np.float32),
                "conf": np.asarray(pred.conf[k], dtype=np.float32),
                "K": np.asarray(pred.intrinsics[k], dtype=np.float64),
                "E": exts[k].astype(np.float64),
                "rgb": np.asarray(pred.processed_images[k]),
            }
    return out


def fuse_depths(preds: dict[str, dict[str, np.ndarray]], config: DenseInitConfig, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Back-project every confident pixel, keep what neighbouring views agree on, thin by voxel.

    Each entry of ``preds`` holds ``depth``/``conf`` maps, the pinhole ``K`` at their
    resolution, the world-to-camera ``E`` (4x4) and the ``rgb`` they were computed from.
    Returns ``(xyz, rgb in 0..1, counts)``.
    """
    names = sorted(preds)
    if not names:
        return np.zeros((0, 3)), np.zeros((0, 3)), {"candidates": 0, "consistent": 0}
    centres = np.stack([-(preds[n]["E"][:3, :3].T @ preds[n]["E"][:3, 3]) for n in names])
    st = config.stride
    pts, cols = [], []
    n_cand = 0
    for i, n in enumerate(names):
        p = preds[n]
        d = p["depth"][::st, ::st]
        c = p["conf"][::st, ::st]
        K = p["K"].copy()
        K[:2] /= st
        E = p["E"]
        rgb = p["rgb"][::st, ::st].reshape(-1, 3).astype(np.float32) / 255.0
        h, w = d.shape
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        cam = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], axis=-1).reshape(-1, 3)
        ok = (c.reshape(-1) >= np.percentile(c, config.conf_percentile)) & (d.reshape(-1) > 0.05)
        X = (cam[ok] - E[:3, 3]) @ E[:3, :3]
        rgb = rgb[ok]
        n_cand += int(ok.sum())
        if X.shape[0] == 0:
            continue
        nbrs = [j for j in np.argsort(np.linalg.norm(centres - centres[i], axis=1)) if j != i][: config.n_check]
        agree = np.zeros(X.shape[0], dtype=np.int32)
        for j in nbrs:
            q = preds[names[j]]
            pc = X @ q["E"][:3, :3].T + q["E"][:3, 3]
            z = pc[:, 2]
            Dq = q["depth"]
            hq, wq = Dq.shape
            front = z > 0.05
            zs = np.where(front, z, 1.0)
            uq = q["K"][0, 0] * pc[:, 0] / zs + q["K"][0, 2]
            vq = q["K"][1, 1] * pc[:, 1] / zs + q["K"][1, 2]
            inside = front & (uq >= 0) & (uq < wq) & (vq >= 0) & (vq < hq)
            dq = Dq[np.clip(vq.astype(np.int64), 0, hq - 1), np.clip(uq.astype(np.int64), 0, wq - 1)]
            agree += (inside & (np.abs(dq - z) <= config.rel_tol * z)).astype(np.int32)
        keep = agree >= config.min_agree
        pts.append(X[keep])
        cols.append(rgb[keep])
    xyz = np.concatenate(pts) if pts else np.zeros((0, 3))
    rgb_all = np.concatenate(cols) if cols else np.zeros((0, 3))
    n_cons = int(len(xyz))
    if len(xyz) and config.voxel_m > 0:
        _, first = np.unique(np.floor(xyz / config.voxel_m).astype(np.int64), axis=0, return_index=True)
        xyz, rgb_all = xyz[first], rgb_all[first]
    if len(xyz) > config.max_points:
        sel = np.random.default_rng(seed).choice(len(xyz), config.max_points, replace=False)
        xyz, rgb_all = xyz[sel], rgb_all[sel]
    return xyz, rgb_all, {"candidates": n_cand, "consistent": n_cons}


def write_point_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> Path:
    """A plain coloured point cloud (x y z + uchar red green blue): the init Brush reads."""
    n = len(xyz)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.zeros(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    c = np.clip(np.asarray(rgb) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    arr["red"], arr["green"], arr["blue"] = c[:, 0], c[:, 1], c[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        arr.tofile(fh)
    return path


def dense_init(dataset_dir: Path, *, config: DenseInitConfig | None = None, device=None) -> DenseInitReport:
    """Write ``dataset_dir/init.ply`` from DA3 depth over the dataset's posed images.

    The dataset is the trainer's: ``images/`` + ``sparse/0`` with pinhole cameras in the
    property frame. Never raises: any failure leaves the SfM points as the init and says why.
    """
    import pycolmap
    import torch

    config = config or DenseInitConfig()
    rep = DenseInitReport(model=config.model)
    if not config.enabled:
        rep.reason = "disabled"
        return rep
    t0 = time.perf_counter()
    dataset_dir = Path(dataset_dir)
    sparse = dataset_dir / "sparse" / "0"
    if not (sparse / "cameras.bin").exists():
        sparse = dataset_dir / "sparse"
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    try:
        model = load_model(config.model, device)
        rec = pycolmap.Reconstruction(str(sparse))
        preds = _predict(model, rec, dataset_dir / "images", config)
        rep.n_views = len(preds)
        del model
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        xyz, rgb, counts = fuse_depths(preds, config)
        rep.n_candidates, rep.n_consistent = counts["candidates"], counts["consistent"]
        rep.n_points = int(len(xyz))
        if rep.n_points < 1000:
            rep.reason = f"too_few_points: {rep.n_points}"
            return rep
        write_point_ply(dataset_dir / "init.ply", xyz, rgb)
        rep.used = True
    except Exception as exc:  # noqa: BLE001 - the SfM points are always a valid init
        rep.reason = f"{type(exc).__name__}: {str(exc)[:200]}"
        rep.warnings.append(f"dense_init_unavailable: {rep.reason}")
    finally:
        del model
        rep.seconds = round(time.perf_counter() - t0, 1)
    return rep
