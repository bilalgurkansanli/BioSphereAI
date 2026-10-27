"""The 3D Gaussian Splatting PLY table, as numpy arrays.

The de-facto interchange format between trainers, editors and viewers is the INRIA PLY: one
binary little-endian vertex element whose properties are ``x y z``, optional normals, the
spherical-harmonic colour (``f_dc_0..2`` then ``f_rest_*``), ``opacity`` as a logit,
``scale_0..2`` as log-scales and ``rot_0..3`` as an un-normalised ``wxyz`` quaternion. Every
tool agrees on the names and on the activations, and disagrees on nothing else, so this
module keeps the table in its *stored* parametrisation and offers the activations as
properties rather than converting on load: cleaning and cropping then round-trip bit-exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["GaussianTable", "read_ply", "write_ply", "SH_C0"]

#: The zeroth spherical-harmonic basis constant; ``rgb = 0.5 + SH_C0 * f_dc``.
SH_C0 = 0.28209479177387814

_PLY_TO_NP = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}

_K_BY_DEGREE = {0: 0, 1: 3, 2: 8, 3: 15}
_DEGREE_BY_K = {v: k for k, v in _K_BY_DEGREE.items()}


@dataclass
class GaussianTable:
    """Stored parameters of ``n`` Gaussians. Shapes are documented per field."""

    means: np.ndarray  # (n, 3) float32
    sh_dc: np.ndarray  # (n, 3) float32
    sh_rest: np.ndarray  # (n, 3, k) float32 with k in {0, 3, 8, 15}
    opacity_logit: np.ndarray  # (n,) float32
    log_scales: np.ndarray  # (n, 3) float32
    quats: np.ndarray  # (n, 4) float32, wxyz, not necessarily unit

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_degree(self) -> int:
        return _DEGREE_BY_K[int(self.sh_rest.shape[2])]

    @property
    def opacity(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.opacity_logit.astype(np.float64)))

    @property
    def scales(self) -> np.ndarray:
        return np.exp(self.log_scales)

    @property
    def unit_quats(self) -> np.ndarray:
        n = np.linalg.norm(self.quats, axis=1, keepdims=True)
        return self.quats / np.maximum(n, 1e-12)

    @property
    def base_rgb(self) -> np.ndarray:
        """View-independent colour in [0, 1]: the DC band alone."""
        return np.clip(0.5 + SH_C0 * self.sh_dc, 0.0, 1.0)

    def select(self, mask: np.ndarray) -> GaussianTable:
        """A copy keeping the rows where ``mask`` (bool or index array) says so."""
        return GaussianTable(
            means=self.means[mask], sh_dc=self.sh_dc[mask], sh_rest=self.sh_rest[mask],
            opacity_logit=self.opacity_logit[mask], log_scales=self.log_scales[mask],
            quats=self.quats[mask],
        )

    def with_sh_degree(self, degree: int) -> GaussianTable:
        """Drop (never invent) higher bands."""
        k = min(_K_BY_DEGREE[int(degree)], int(self.sh_rest.shape[2]))
        return GaussianTable(
            means=self.means, sh_dc=self.sh_dc, sh_rest=self.sh_rest[:, :, :k],
            opacity_logit=self.opacity_logit, log_scales=self.log_scales, quats=self.quats,
        )

    def transformed(self, R: np.ndarray, t: np.ndarray, scale: float = 1.0) -> GaussianTable:  # noqa: N803
        """Apply the similarity ``p' = scale * R p + t`` to every Gaussian.

        Rotating a Gaussian rotates its orientation quaternion and, when the world scales,
        its scales scale with it. The SH colour bands would rotate too in principle; the
        direction dependence is small indoors and every editor ignores it, so this does too.
        """
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        means = (float(scale) * (self.means.astype(np.float64) @ R.T) + t).astype(np.float32)
        qr = quat_from_matrix(R)
        quats = quat_multiply(qr[None, :], self.unit_quats.astype(np.float64)).astype(np.float32)
        log_scales = (self.log_scales + np.float32(np.log(float(scale)))).astype(np.float32)
        return GaussianTable(
            means=means, sh_dc=self.sh_dc, sh_rest=self.sh_rest, opacity_logit=self.opacity_logit,
            log_scales=log_scales, quats=quats,
        )

    @staticmethod
    def concatenate(tables: list[GaussianTable]) -> GaussianTable:
        k = min(int(t.sh_rest.shape[2]) for t in tables)
        return GaussianTable(
            means=np.concatenate([t.means for t in tables]),
            sh_dc=np.concatenate([t.sh_dc for t in tables]),
            sh_rest=np.concatenate([t.sh_rest[:, :, :k] for t in tables]),
            opacity_logit=np.concatenate([t.opacity_logit for t in tables]),
            log_scales=np.concatenate([t.log_scales for t in tables]),
            quats=np.concatenate([t.quats for t in tables]),
        )


def quat_from_matrix(R: np.ndarray) -> np.ndarray:  # noqa: N803
    """``wxyz`` unit quaternion of a rotation matrix."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of ``wxyz`` quaternions, broadcasting over leading axes."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _parse_header(fh) -> tuple[list[tuple[str, str]], int, str]:
    magic = fh.readline()
    if not magic.startswith(b"ply"):
        raise ValueError("not a PLY file")
    fmt = ""
    props: list[tuple[str, str]] = []
    count = 0
    in_vertex = False
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("truncated PLY header")
        parts = line.decode("ascii", "replace").strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError("list properties are not supported on the vertex element")
            props.append((parts[2], parts[1]))
        elif parts[0] == "end_header":
            break
    return props, count, fmt


def read_ply(path: Path | str) -> GaussianTable:
    """Read an INRIA-style Gaussian PLY. Unknown properties are ignored."""
    with open(path, "rb") as fh:
        props, count, fmt = _parse_header(fh)
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY format {fmt!r}; only binary_little_endian")
        dtype = np.dtype([(name, _PLY_TO_NP[typ]) for name, typ in props])
        data = np.fromfile(fh, dtype=dtype, count=count)
    names = {n for n, _ in props}
    needed = {
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = needed - names
    if missing:
        raise ValueError(f"PLY is missing Gaussian properties: {sorted(missing)}")
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda s: int(s[7:]))
    k = len(rest_names) // 3
    if k not in _DEGREE_BY_K:
        raise ValueError(f"unexpected number of f_rest_* properties: {len(rest_names)}")

    def col(name: str) -> np.ndarray:
        return data[name].astype(np.float32)

    sh_rest = np.zeros((count, 3, k), dtype=np.float32)
    # INRIA layout is channel-major: the first k entries are the red channel's bands.
    for i, name in enumerate(rest_names):
        sh_rest[:, i // k, i % k] = col(name)
    return GaussianTable(
        means=np.stack([col("x"), col("y"), col("z")], axis=1),
        sh_dc=np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1),
        sh_rest=sh_rest,
        opacity_logit=col("opacity"),
        log_scales=np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=1),
        quats=np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=1),
    )


def write_ply(path: Path | str, table: GaussianTable) -> Path:
    """Write the table in the layout every viewer and splat-transform read."""
    n = table.count
    k = int(table.sh_rest.shape[2])
    fields: list[tuple[str, str]] = [(name, "<f4") for name in ("x", "y", "z", "nx", "ny", "nz")]
    fields += [(f"f_dc_{i}", "<f4") for i in range(3)]
    fields += [(f"f_rest_{i}", "<f4") for i in range(3 * k)]
    fields += [("opacity", "<f4")]
    fields += [(f"scale_{i}", "<f4") for i in range(3)]
    fields += [(f"rot_{i}", "<f4") for i in range(4)]
    out = np.zeros(n, dtype=np.dtype(fields))
    out["x"], out["y"], out["z"] = table.means[:, 0], table.means[:, 1], table.means[:, 2]
    for i in range(3):
        out[f"f_dc_{i}"] = table.sh_dc[:, i]
    for i in range(3 * k):
        out[f"f_rest_{i}"] = table.sh_rest[:, i // k, i % k]
    out["opacity"] = table.opacity_logit
    for i in range(3):
        out[f"scale_{i}"] = table.log_scales[:, i]
    for i in range(4):
        out[f"rot_{i}"] = table.quats[:, i]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {name}" for name, _ in fields]
    header.append("end_header")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        out.tofile(fh)
    return path
