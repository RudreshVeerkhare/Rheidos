from __future__ import annotations

import numpy as np
import pytest

from pure_taichi.assembly import scatter_point_vortex_rhs_numpy
from pure_taichi.mesh import build_face_geometry, build_mesh_topology_geometry
from pure_taichi.solver import build_poisson_system, solve_stream_function
from pure_taichi.space import build_p2_space_data


def _tetra() -> tuple[np.ndarray, np.ndarray]:
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


def test_solver_reduced_path_has_low_residual() -> None:
    pytest.importorskip("scipy")

    vertices, faces = _tetra()
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
    gamma = np.array([1.0, -1.0], dtype=np.float64)

    rhs = scatter_point_vortex_rhs_numpy(face_ids, bary, gamma, space.face_to_dofs, space.ndof)
    psi, residual_l2, backend, _ = solve_stream_function(system, rhs, backend="scipy")

    assert backend == "scipy"
    assert np.isfinite(psi).all()
    assert residual_l2 < 1e-6

    c = system.c
    mean_val = float(np.dot(c, psi) / c.sum())
    assert abs(mean_val) < 1e-8


def test_solver_constrained_mode_runs() -> None:
    pytest.importorskip("scipy")

    vertices, faces = _tetra()
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

    face_ids = np.array([0], dtype=np.int32)
    bary = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
    gamma = np.array([1.0], dtype=np.float64)
    rhs = scatter_point_vortex_rhs_numpy(face_ids, bary, gamma, space.face_to_dofs, space.ndof)

    psi, residual_l2, _, _ = solve_stream_function(system, rhs, backend="scipy_constrained")
    assert np.isfinite(psi).all()
    assert np.isfinite(residual_l2)
