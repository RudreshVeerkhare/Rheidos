from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray  # (nV, 3)
    faces: np.ndarray  # (nF, 3)
    edges: np.ndarray  # (nE, 2)
    edge_faces: np.ndarray  # (nE, 2)
    face_adjacency: np.ndarray  # (nF, 3)
    face_normals: np.ndarray  # (nF, 3)
    face_areas: np.ndarray  # (nF,)


@dataclass(frozen=True)
class FaceGeometryData:
    J: np.ndarray  # (nF, 3, 2)
    Ginv: np.ndarray  # (nF, 2, 2)
    sqrt_detG: np.ndarray  # (nF,)


@dataclass(frozen=True)
class P2SpaceData:
    edges: np.ndarray  # (nE, 2)
    face_to_edges: np.ndarray  # (nF, 3)
    face_to_dofs: np.ndarray  # (nF, 6)
    ndof: int


@dataclass
class VortexState:
    face_ids: np.ndarray  # (N,)
    bary: np.ndarray  # (N, 3)
    gamma: np.ndarray  # (N,)


@dataclass
class PoissonSystem:
    K: Any
    M: Any
    c: np.ndarray  # (ndof,)
    k_ones_inf: float
    pin_index: int
    free_dofs: np.ndarray
    taichi_matrix: Any = None
    taichi_solver: Any = None
    scipy_factor: Any = None


@dataclass
class Diagnostics:
    residual_l2: float
    rhs_circulation: float
    k_ones_inf: float
    hops_total: int
    hops_max: int
    bary_min: float
    bary_max: float
    bary_sum_min: float
    bary_sum_max: float
    solver_backend: str


@dataclass
class StepResult:
    psi: np.ndarray
    vel_corner: np.ndarray
    vel_face: np.ndarray
    stream_vertex: np.ndarray
    state: VortexState
    diagnostics: Diagnostics
