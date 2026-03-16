from __future__ import annotations

import numpy as np
import pytest

from pure_taichi.advection import advect_midpoint_batch
from pure_taichi.assembly import scatter_point_vortex_rhs_numpy
from pure_taichi.config import MeshConfig, SimulationConfig, SolverConfig, TimeConfig, VortexConfig
from pure_taichi.mesh import build_face_geometry, build_mesh_topology_geometry
from pure_taichi.sim import run_headless
from pure_taichi.solver import build_poisson_system, solve_stream_function
from pure_taichi.space import build_p2_space_data
from pure_taichi.velocity import build_p2_velocity_fields


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


def test_headless_smoke_runs_finite() -> None:
    pytest.importorskip("scipy")

    cfg = SimulationConfig(
        mesh=MeshConfig(kind="icosphere", subdivisions=1, radius=1.0),
        solver=SolverConfig(backend="scipy"),
        time=TimeConfig(dt=0.01, substeps=1, max_hops=32),
        vortex=VortexConfig(preset="ring", n_vortices=8, gamma_scale=1.0),
        seed=7,
    )

    out = run_headless(cfg, steps=10)
    pos = np.asarray(out["positions"], dtype=np.float64)
    bary = np.asarray(out["bary"], dtype=np.float64)

    assert np.isfinite(pos).all()
    assert np.isfinite(bary).all()
    assert np.allclose(bary.sum(axis=1), 1.0, atol=1e-6)


def test_regression_against_existing_numpy_pipeline_one_step() -> None:
    pytest.importorskip("scipy")

    from rheidos.apps.point_vortex_p2.modules.midpoint_advection import (
        advect_midpoint_batch as ref_advect_midpoint_batch,
    )
    from rheidos.apps.point_vortex_p2.modules.p2_geometry import (
        build_face_geometry as ref_build_face_geometry,
    )
    from rheidos.apps.point_vortex_p2.modules.p2_poisson import (
        assemble_p2_surface_matrices as ref_assemble,
        scatter_point_vortex_rhs as ref_scatter,
        solve_pinned_poisson as ref_solve,
    )
    from rheidos.apps.point_vortex_p2.modules.p2_space import (
        build_p2_space_data as ref_build_space,
    )
    from rheidos.apps.point_vortex_p2.modules.p2_velocity import (
        build_p2_velocity_fields as ref_build_vel,
    )
    from rheidos.apps.point_vortex_p2.modules.surface_mesh import (
        build_mesh_topology_geometry as ref_build_mesh,
    )

    vertices, faces = _tetra()

    mesh = build_mesh_topology_geometry(vertices, faces)
    geom = build_face_geometry(mesh.vertices, mesh.faces)
    space = build_p2_space_data(mesh.vertices.shape[0], mesh.faces)

    face_ids = np.array([0, 1], dtype=np.int32)
    bary = np.array([[1 / 3, 1 / 3, 1 / 3], [0.2, 0.5, 0.3]], dtype=np.float64)
    gamma = np.array([1.0, -1.0], dtype=np.float64)

    system = build_poisson_system(
        space.face_to_dofs,
        space.ndof,
        geom.J,
        geom.Ginv,
        geom.sqrt_detG,
        pin_index=0,
        prefer_taichi=False,
    )
    rhs = scatter_point_vortex_rhs_numpy(face_ids, bary, gamma, space.face_to_dofs, space.ndof)
    psi, _, _, _ = solve_stream_function(system, rhs, backend="scipy")
    vel_corner, vel_face, _ = build_p2_velocity_fields(
        psi,
        space.face_to_dofs,
        geom.J,
        geom.Ginv,
        mesh.face_normals,
        n_vertices=mesh.vertices.shape[0],
    )

    face_out, bary_out, _, _, _ = advect_midpoint_batch(
        mesh.vertices,
        mesh.faces,
        mesh.face_adjacency,
        vel_corner,
        face_ids,
        bary,
        dt=0.01,
    )

    _, _, f_adj_ref, normals_ref, _, _ = ref_build_mesh(vertices, faces)
    _, _, face_to_dofs_ref, ndof_ref = ref_build_space(vertices.shape[0], faces)
    J_ref, Ginv_ref, sqrt_detG_ref = ref_build_face_geometry(vertices, faces)

    K_ref, _ = ref_assemble(face_to_dofs_ref, ndof_ref, J_ref, Ginv_ref, sqrt_detG_ref)
    rhs_ref = ref_scatter(face_ids, bary, gamma, face_to_dofs_ref, ndof_ref)
    psi_ref, _, _ = ref_solve(K_ref, rhs_ref, pin_index=0)
    vel_corner_ref, vel_face_ref, _ = ref_build_vel(
        psi_ref,
        face_to_dofs_ref,
        J_ref,
        Ginv_ref,
        normals_ref,
        n_vertices=vertices.shape[0],
    )

    face_out_ref, bary_out_ref, _, _, _ = ref_advect_midpoint_batch(
        vertices,
        faces,
        f_adj_ref,
        vel_corner_ref,
        face_ids,
        bary,
        dt=0.01,
    )

    assert np.allclose(vel_face, vel_face_ref, atol=1e-6)
    assert np.array_equal(face_out, face_out_ref)
    assert np.allclose(bary_out, bary_out_ref, atol=1e-6)
