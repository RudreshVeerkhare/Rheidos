from pathlib import Path
from typing import Union

import numpy as np


def load_mesh(path: Union[str, Path], *, center: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import trimesh  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("trimesh is not available. Install 'trimesh'.") from exc

    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from path '{path}'")

    mesh = mesh.copy()
    if mesh.faces is None or mesh.faces.size == 0:
        raise ValueError("Loaded mesh has no faces")
    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()

    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)
    normals = mesh.vertex_normals.astype(np.float32)

    if center:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        offset = (mins + maxs) * 0.5
        vertices = vertices - offset.astype(np.float32)

    return vertices, faces, normals
