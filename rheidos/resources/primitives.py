from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .mesh import Mesh


@dataclass
class Primitive:
    mesh: Mesh
    bounds: Tuple[np.ndarray, np.ndarray]


def cube(size: float = 1.0, name: str = "cube") -> Primitive:
    half = size * 0.5
    vertices = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            0, 1, 2, 0, 2, 3,  # bottom
            4, 5, 6, 4, 6, 7,  # top
            0, 4, 5, 0, 5, 1,  # front
            1, 5, 6, 1, 6, 2,  # right
            2, 6, 7, 2, 7, 3,  # back
            3, 7, 4, 3, 4, 0,  # left
        ],
        dtype=np.int32,
    )

    normals = np.array(
        [
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    colors = np.full((8, 4), [0.8, 0.8, 0.9, 1.0], dtype=np.float32)

    mesh = Mesh(vertices=vertices, indices=indices, normals=normals, colors=colors, name=name)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return Primitive(mesh=mesh, bounds=(mins, maxs))

