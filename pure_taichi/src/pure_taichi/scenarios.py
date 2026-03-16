from __future__ import annotations

import numpy as np

from .math_utils import barycentric_from_point, clamp_renorm_bary
from .model import MeshData, VortexState


def vortex_positions_from_face_bary(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    bary: np.ndarray,
) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    face_ids = np.asarray(face_ids, dtype=np.int32)
    bary = np.asarray(bary, dtype=np.float64)

    n = int(face_ids.shape[0])
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        fid = int(face_ids[i])
        tri = f[fid]
        b = bary[i]
        out[i] = b[0] * v[tri[0]] + b[1] * v[tri[1]] + b[2] * v[tri[2]]
    return out


def _random_bary(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.random((n, 3))
    x = -np.log(np.maximum(x, 1e-12))
    x /= np.sum(x, axis=1, keepdims=True)
    return x


def _locate_on_mesh(mesh: MeshData, p: np.ndarray) -> tuple[int, np.ndarray]:
    best_fid = -1
    best_bary = None
    best_score = np.inf

    v = mesh.vertices
    f = mesh.faces

    for fid, tri in enumerate(f):
        a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
        bc = barycentric_from_point(p, a, b, c)
        neg = np.maximum(-bc, 0.0).sum()
        if neg < best_score:
            best_score = float(neg)
            best_fid = fid
            best_bary = bc
            if neg <= 1e-8:
                break

    if best_fid < 0 or best_bary is None:
        raise RuntimeError("Failed to locate point on mesh")
    return best_fid, clamp_renorm_bary(best_bary)


def _init_ring(mesh: MeshData, n: int, gamma_scale: float) -> VortexState:
    if n < 2:
        n = 2

    radii = np.linalg.norm(mesh.vertices, axis=1)
    radius = float(np.median(radii))

    face_ids = np.zeros((n,), dtype=np.int32)
    bary = np.zeros((n, 3), dtype=np.float64)
    gamma = np.zeros((n,), dtype=np.float64)

    for i in range(n):
        theta = 2.0 * np.pi * (float(i) / float(n))
        p = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64) * radius
        fid, bc = _locate_on_mesh(mesh, p)
        face_ids[i] = fid
        bary[i] = bc
        gamma[i] = gamma_scale * (1.0 if (i % 2 == 0) else -1.0)

    # Zero-net circulation safeguard for odd n.
    gamma -= gamma.mean()
    return VortexState(face_ids=face_ids, bary=bary, gamma=gamma)


def _init_dipole(mesh: MeshData, gamma_scale: float) -> VortexState:
    radii = np.linalg.norm(mesh.vertices, axis=1)
    radius = float(np.median(radii))

    p0 = np.array([radius, 0.0, 0.0], dtype=np.float64)
    p1 = np.array([-radius, 0.0, 0.0], dtype=np.float64)

    f0, b0 = _locate_on_mesh(mesh, p0)
    f1, b1 = _locate_on_mesh(mesh, p1)

    face_ids = np.array([f0, f1], dtype=np.int32)
    bary = np.vstack([b0, b1]).astype(np.float64)
    gamma = np.array([gamma_scale, -gamma_scale], dtype=np.float64)
    return VortexState(face_ids=face_ids, bary=bary, gamma=gamma)


def _init_random(
    mesh: MeshData,
    n: int,
    gamma_scale: float,
    seed: int,
) -> VortexState:
    rng = np.random.default_rng(int(seed))
    nF = int(mesh.faces.shape[0])

    face_ids = rng.integers(0, nF, size=(n,), endpoint=False, dtype=np.int32)
    bary = _random_bary(rng, n)

    gamma = rng.normal(size=(n,)).astype(np.float64)
    gamma -= gamma.mean()
    norm = float(np.linalg.norm(gamma))
    if norm > 1e-12:
        gamma = gamma / norm
    gamma *= float(gamma_scale) * max(1.0, np.sqrt(float(n)))

    return VortexState(face_ids=face_ids, bary=bary, gamma=gamma)


def init_vortices(
    mesh: MeshData,
    *,
    preset: str,
    n_vortices: int,
    gamma_scale: float,
    seed: int,
) -> VortexState:
    key = str(preset).strip().lower()
    if key == "ring":
        return _init_ring(mesh, int(n_vortices), float(gamma_scale))
    if key == "dipole":
        return _init_dipole(mesh, float(gamma_scale))
    if key == "random":
        return _init_random(mesh, int(n_vortices), float(gamma_scale), int(seed))

    raise ValueError(f"Unknown vortex preset '{preset}'. Supported: ring|dipole|random")
