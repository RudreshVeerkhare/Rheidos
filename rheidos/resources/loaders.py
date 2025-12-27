from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .mesh import Mesh
from .primitives import Primitive


def load_mesh(
    path: str | Path,
    name: Optional[str] = None,
    center: bool = True,
    *,
    dynamic: bool = False,
) -> Primitive:
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

    colors = None
    if mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == vertices.shape[0]:
            if vc.shape[1] >= 4:
                colors = vc[:, :4]
            elif vc.shape[1] == 3:
                alpha = np.full((vc.shape[0], 1), 255, dtype=vc.dtype)
                colors = np.concatenate([vc, alpha], axis=1)

    if colors is None:
        colors = np.full((vertices.shape[0], 4), [204, 204, 224, 255], dtype=np.uint8)
    else:
        if colors.dtype in (np.float32, np.float64):
            colors = colors.astype(np.float32, copy=False)
            if colors.max() > 1.0:
                colors = (colors / 255.0).astype(np.float32)
        elif colors.dtype != np.uint8:
            colors = colors.astype(np.uint8, copy=False)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)

    if center:
        offset = (mins + maxs) * 0.5
        vertices = vertices - offset.astype(np.float32)
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)

    mesh_obj = Mesh(
        vertices=vertices,
        indices=faces,
        normals=normals,
        colors=colors,
        name=name or Path(path).stem,
        dynamic=dynamic,
    )
    return Primitive(mesh=mesh_obj, bounds=(mins, maxs))
