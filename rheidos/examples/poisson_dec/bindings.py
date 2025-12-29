from __future__ import annotations

import numpy as np

from rheidos.abc.observer import Observer
from rheidos.resources.mesh import Mesh
from rheidos.sim.interaction import Signal


def _compute_vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    tris = indices.reshape(-1, 3)
    v0 = vertices[tris[:, 0]]
    v1 = vertices[tris[:, 1]]
    v2 = vertices[tris[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    normals = np.zeros_like(vertices, dtype=np.float32)
    np.add.at(normals, tris[:, 0], face_normals)
    np.add.at(normals, tris[:, 1], face_normals)
    np.add.at(normals, tris[:, 2], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return normals / lengths


class PoissonMeshDeformer(Observer):
    def __init__(
        self,
        mesh: Mesh,
        u_signal: Signal[np.ndarray],
        *,
        scale: float = 1.0,
        name: str = "PoissonMeshDeformer",
        sort: int = -5,
    ) -> None:
        super().__init__(name=name, sort=sort)
        self._mesh = mesh
        self._u_signal = u_signal
        self._scale = float(scale)
        self._base_vertices = self._load_vertices(mesh)
        if not np.isfinite(self._base_vertices).all():
            raise ValueError("Mesh vertices contain NaN/Inf; cannot deform.")
        bounds_min = self._base_vertices.min(axis=0)
        bounds_max = self._base_vertices.max(axis=0)
        extent = float(np.linalg.norm(bounds_max - bounds_min))
        self._max_displacement = extent * 10.0 if extent > 0.0 else 1.0
        self._normals = self._load_normals(mesh, self._base_vertices)
        if not np.isfinite(self._normals).all():
            raise ValueError("Mesh normals contain NaN/Inf; cannot deform.")
        self._last_version = -1

    def _load_vertices(self, mesh: Mesh) -> np.ndarray:
        verts = mesh.get_vertices()
        if verts is None:
            raise RuntimeError("Mesh has no vertex buffer for deformation.")
        return np.array(verts, dtype=np.float32, copy=True)

    def _load_normals(self, mesh: Mesh, vertices: np.ndarray) -> np.ndarray:
        normals = mesh.get_normals()
        if normals is not None:
            normals_arr = np.array(normals, dtype=np.float32, copy=True)
            if np.isfinite(normals_arr).all():
                return normals_arr
        indices = mesh.get_indices()
        if indices is None:
            raise RuntimeError("Mesh normals missing and indices unavailable.")
        return _compute_vertex_normals(vertices, indices)

    def update(self, dt: float) -> None:
        version = self._u_signal.version()
        if version == 0:
            return
        if version == self._last_version:
            return
        snapshot = self._u_signal.read_snapshot()
        values = np.asarray(snapshot.payload, dtype=np.float32).reshape(-1)
        if not np.isfinite(values).all():
            return
        if values.shape[0] != self._base_vertices.shape[0]:
            raise ValueError(
                "Signal vertex count mismatch: "
                f"{values.shape[0]} != {self._base_vertices.shape[0]}"
            )
        displacements = self._normals * values[:, None] * self._scale
        if displacements.size == 0:
            return
        if float(np.abs(displacements).max()) > self._max_displacement:
            return
        displaced = self._base_vertices + displacements
        if not np.isfinite(displaced).all():
            return
        self._mesh.set_vertices(displaced)
        self._last_version = snapshot.version
