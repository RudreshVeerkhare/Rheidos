from __future__ import annotations

import numpy as np
import pytest

from pure_taichi.assembly import (
    remove_mean_from_rhs,
    scatter_point_vortex_rhs_numpy,
)
from pure_taichi.mesh import build_face_geometry, build_mesh_topology_geometry
from pure_taichi.solver import build_poisson_system
from pure_taichi.space import build_p2_space_data


@pytest.fixture
def tetra_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def test_matrix_sanity_on_closed_mesh(tetra_mesh) -> None:
    pytest.importorskip("scipy")

    vertices, faces = tetra_mesh
    mesh = build_mesh_topology_geometry(vertices, faces)
    geom = build_face_geometry(mesh.vertices, mesh.faces)
    space = build_p2_space_data(mesh.vertices.shape[0], mesh.faces)

    system = build_poisson_system(
        space.face_to_dofs,
        space.ndof,
        geom.J,
        geom.Ginv,
        geom.sqrt_detG,
        pin_index=0,
        prefer_taichi=False,
    )

    K = system.K
    M = system.M

    K_diff = (K - K.T).toarray()
    M_diff = (M - M.T).toarray()
    assert np.max(np.abs(K_diff)) < 1e-10
    assert np.max(np.abs(M_diff)) < 1e-10

    ones = np.ones((space.ndof,), dtype=np.float64)
    assert np.linalg.norm(K @ ones, ord=np.inf) < 1e-8


def test_rhs_scatter_and_mean_removal(tetra_mesh) -> None:
    vertices, faces = tetra_mesh
    mesh = build_mesh_topology_geometry(vertices, faces)
    geom = build_face_geometry(mesh.vertices, mesh.faces)
    space = build_p2_space_data(mesh.vertices.shape[0], mesh.faces)

    system = build_poisson_system(
        space.face_to_dofs,
        space.ndof,
        geom.J,
        geom.Ginv,
        geom.sqrt_detG,
        pin_index=0,
        prefer_taichi=False,
    )

    face_ids = np.array([0, 1], dtype=np.int32)
    bary = np.array([[1 / 3, 1 / 3, 1 / 3], [0.2, 0.5, 0.3]], dtype=np.float64)
    gamma = np.array([1.0, -0.4], dtype=np.float64)

    rhs = scatter_point_vortex_rhs_numpy(face_ids, bary, gamma, space.face_to_dofs, space.ndof)
    rhs0 = remove_mean_from_rhs(rhs, system.c)

    assert np.isfinite(rhs).all()
    assert np.isfinite(rhs0).all()
    assert abs(float(rhs0.sum())) < abs(float(rhs.sum())) + 1e-12
