from __future__ import annotations

import numpy as np
import pytest

from rheidos.apps.point_vortex_p2.modules.fe_utils import (
    bary_to_ref,
    p2_shape_and_grad_ref,
)
from rheidos.apps.point_vortex_p2.modules.midpoint_advection import (
    advance_const_velocity_event_driven,
    advect_single_field_batch,
    advect_stage_b_from_midpoint_batch,
    advect_midpoint_batch,
)
from rheidos.apps.point_vortex_p2.modules.p2_geometry import build_face_geometry
from rheidos.apps.point_vortex_p2.modules.p2_poisson import (
    assemble_p2_surface_matrices,
    grad_ref_to_surface,
    scatter_point_vortex_rhs,
    solve_pinned_poisson,
)
from rheidos.apps.point_vortex_p2.modules.p2_space import build_p2_space_data
from rheidos.apps.point_vortex_p2.modules.p2_velocity import (
    build_p2_velocity_fields,
    sample_velocity_from_corners,
)
from rheidos.apps.point_vortex_p2.modules.surface_mesh import build_mesh_topology_geometry


@pytest.fixture
def tetra_mesh():
    vertices = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    # Consistent outward orientation for tetra faces.
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


def test_p2_basis_partition_and_gradient_shape():
    rng = np.random.default_rng(123)
    for _ in range(20):
        a = rng.random()
        b = rng.random()
        if a + b > 1.0:
            a, b = 1.0 - a, 1.0 - b
        phi, dphi = p2_shape_and_grad_ref(float(a), float(b))
        assert phi.shape == (6,)
        assert dphi.shape == (6, 2)
        assert np.isclose(phi.sum(), 1.0, atol=1e-12)


def test_p2_space_shared_edge_dof_continuity():
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    _, _, face_to_dofs, ndof = build_p2_space_data(4, faces)

    # Shared edge is (1,2): local edge index 1 in face 0 and index 0 in face 1.
    dof_shared_f0 = int(face_to_dofs[0, 4])
    dof_shared_f1 = int(face_to_dofs[1, 3])

    assert ndof > 0
    assert dof_shared_f0 == dof_shared_f1


def test_matrix_sanity_on_closed_tetra(tetra_mesh):
    scipy = pytest.importorskip("scipy")
    _ = scipy  # silence lint for importorskip

    vertices, faces = tetra_mesh
    _, _, _, _, _, boundary_edges = build_mesh_topology_geometry(vertices, faces)
    assert boundary_edges == 0

    _, _, face_to_dofs, ndof = build_p2_space_data(vertices.shape[0], faces)
    J, Ginv, sqrt_detG = build_face_geometry(vertices, faces)

    K, M = assemble_p2_surface_matrices(face_to_dofs, ndof, J, Ginv, sqrt_detG)

    K_diff = (K - K.T).toarray()
    M_diff = (M - M.T).toarray()
    assert np.max(np.abs(K_diff)) < 1e-10
    assert np.max(np.abs(M_diff)) < 1e-10

    ones = np.ones((ndof,), dtype=np.float64)
    assert np.linalg.norm(K @ ones, ord=np.inf) < 1e-8


def test_rhs_scatter_single_vortex():
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    _, _, face_to_dofs, ndof = build_p2_space_data(3, faces)

    face_ids = np.array([0], dtype=np.int32)
    bary = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
    gamma = np.array([2.0], dtype=np.float64)

    rhs = scatter_point_vortex_rhs(face_ids, bary, gamma, face_to_dofs, ndof)

    xi, eta = bary_to_ref(bary[0])
    phi, _ = p2_shape_and_grad_ref(xi, eta)

    expected = np.zeros((ndof,), dtype=np.float64)
    expected[face_to_dofs[0]] = 2.0 * phi
    assert np.allclose(rhs, expected)


def test_velocity_linear_face_consistency():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.9, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    _, _, face_to_dofs, ndof = build_p2_space_data(vertices.shape[0], faces)
    J, Ginv, _ = build_face_geometry(vertices, faces)

    n0 = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    n_hat = n0 / np.linalg.norm(n0)
    normals = np.array([n_hat], dtype=np.float64)

    rng = np.random.default_rng(7)
    psi = rng.normal(size=(ndof,))

    vel_corner, _, _ = build_p2_velocity_fields(psi, face_to_dofs, J, Ginv, normals, n_vertices=3)

    bary = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    u_interp = sample_velocity_from_corners(vel_corner, 0, bary)

    dofs = face_to_dofs[0]
    local_psi = psi[dofs]
    xi, eta = bary_to_ref(bary)
    _, dphi_ref = p2_shape_and_grad_ref(xi, eta)
    dphi = grad_ref_to_surface(J[0], Ginv[0], dphi_ref)
    grad_psi = np.sum(local_psi[:, None] * dphi, axis=0)
    u_exact = np.cross(normals[0], grad_psi)

    assert np.allclose(u_interp, u_exact, atol=1e-10)


def test_boundary_crossing_raises_error():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    f_adj = np.array([[-1, -1, -1]], dtype=np.int32)

    with pytest.raises(RuntimeError, match="closed surfaces"):
        advance_const_velocity_event_driven(
            vertices,
            faces,
            f_adj,
            face_id=0,
            bary=np.array([0.2, 0.2, 0.6], dtype=np.float64),
            vel_world=np.array([0.8, 0.0, 0.0], dtype=np.float64),
            dt=1.0,
        )


def test_integration_smoke_step(tetra_mesh):
    pytest.importorskip("scipy")

    vertices, faces = tetra_mesh
    _, _, f_adj, normals, _, boundary_edges = build_mesh_topology_geometry(vertices, faces)
    assert boundary_edges == 0

    _, _, face_to_dofs, ndof = build_p2_space_data(vertices.shape[0], faces)
    J, Ginv, sqrt_detG = build_face_geometry(vertices, faces)
    K, _ = assemble_p2_surface_matrices(face_to_dofs, ndof, J, Ginv, sqrt_detG)

    face_ids = np.array([0, 1], dtype=np.int32)
    bary = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [0.2, 0.5, 0.3]], dtype=np.float64)
    gamma = np.array([1.0, -1.0], dtype=np.float64)

    rhs = scatter_point_vortex_rhs(face_ids, bary, gamma, face_to_dofs, ndof)
    psi, residual, _ = solve_pinned_poisson(K, rhs, pin_index=0)

    vel_corner, vel_face, stream_vertex = build_p2_velocity_fields(
        psi,
        face_to_dofs,
        J,
        Ginv,
        normals,
        n_vertices=vertices.shape[0],
    )

    face_out, bary_out, pos_out, hops_total, hops_max = advect_midpoint_batch(
        vertices,
        faces,
        f_adj,
        vel_corner,
        face_ids,
        bary,
        dt=0.01,
    )

    assert np.isfinite(stream_vertex).all()
    assert np.isfinite(vel_face).all()
    assert np.isfinite(residual).all()

    assert face_out.shape == (2,)
    assert bary_out.shape == (2, 3)
    assert pos_out.shape == (2, 3)
    assert np.isfinite(pos_out).all()

    assert hops_total >= 0
    assert hops_max >= 0


def test_coupled_rk2_differs_from_frozen_midpoint(tetra_mesh):
    pytest.importorskip("scipy")

    vertices, faces = tetra_mesh
    _, _, f_adj, normals, _, boundary_edges = build_mesh_topology_geometry(vertices, faces)
    assert boundary_edges == 0

    _, _, face_to_dofs, ndof = build_p2_space_data(vertices.shape[0], faces)
    J, Ginv, sqrt_detG = build_face_geometry(vertices, faces)
    K, _ = assemble_p2_surface_matrices(face_to_dofs, ndof, J, Ginv, sqrt_detG)

    face_ids = np.array([0, 1, 2, 3], dtype=np.int32)
    bary = np.array(
        [
            [0.15, 0.70, 0.15],
            [0.35, 0.10, 0.55],
            [0.25, 0.45, 0.30],
            [0.60, 0.20, 0.20],
        ],
        dtype=np.float64,
    )
    gamma = np.array([1.0, -0.4, 0.7, -1.2], dtype=np.float64)
    dt = 0.15

    rhs0 = scatter_point_vortex_rhs(face_ids, bary, gamma, face_to_dofs, ndof)
    psi0, _, _ = solve_pinned_poisson(K, rhs0, pin_index=0)
    vel_corner0, _, _ = build_p2_velocity_fields(
        psi0,
        face_to_dofs,
        J,
        Ginv,
        normals,
        n_vertices=vertices.shape[0],
    )

    _, _, pos_frozen, _, _ = advect_midpoint_batch(
        vertices,
        faces,
        f_adj,
        vel_corner0,
        face_ids,
        bary,
        dt=dt,
    )

    face_mid, bary_mid, _, _ = advect_single_field_batch(
        vertices,
        faces,
        f_adj,
        vel_corner0,
        face_ids,
        bary,
        0.5 * dt,
    )

    rhs_mid = scatter_point_vortex_rhs(face_mid, bary_mid, gamma, face_to_dofs, ndof)
    psi_mid, _, _ = solve_pinned_poisson(K, rhs_mid, pin_index=0)
    vel_corner_mid, _, _ = build_p2_velocity_fields(
        psi_mid,
        face_to_dofs,
        J,
        Ginv,
        normals,
        n_vertices=vertices.shape[0],
    )
    _, _, pos_coupled, _ = advect_stage_b_from_midpoint_batch(
        vertices,
        faces,
        f_adj,
        vel_corner_mid,
        face_ids,
        bary,
        face_mid,
        bary_mid,
        dt=dt,
    )

    diff = np.linalg.norm(pos_coupled - pos_frozen, axis=1)
    assert float(np.max(diff)) > 1e-8


def test_coupled_rk2_matches_explicit_two_solve_reference(tetra_mesh):
    pytest.importorskip("scipy")

    vertices, faces = tetra_mesh
    _, _, f_adj, normals, _, boundary_edges = build_mesh_topology_geometry(vertices, faces)
    assert boundary_edges == 0

    _, _, face_to_dofs, ndof = build_p2_space_data(vertices.shape[0], faces)
    J, Ginv, sqrt_detG = build_face_geometry(vertices, faces)
    K, _ = assemble_p2_surface_matrices(face_to_dofs, ndof, J, Ginv, sqrt_detG)

    face_ids = np.array([0, 2, 1], dtype=np.int32)
    bary = np.array(
        [
            [0.2, 0.5, 0.3],
            [0.5, 0.25, 0.25],
            [0.1, 0.6, 0.3],
        ],
        dtype=np.float64,
    )
    gamma = np.array([0.8, -0.35, 1.1], dtype=np.float64)
    dt = 0.07

    rhs0 = scatter_point_vortex_rhs(face_ids, bary, gamma, face_to_dofs, ndof)
    psi0, _, _ = solve_pinned_poisson(K, rhs0, pin_index=0)
    vel_corner0, _, _ = build_p2_velocity_fields(
        psi0,
        face_to_dofs,
        J,
        Ginv,
        normals,
        n_vertices=vertices.shape[0],
    )

    face_mid, bary_mid, _, hops_a = advect_single_field_batch(
        vertices,
        faces,
        f_adj,
        vel_corner0,
        face_ids,
        bary,
        0.5 * dt,
    )

    rhs_mid = scatter_point_vortex_rhs(face_mid, bary_mid, gamma, face_to_dofs, ndof)
    psi_mid, _, _ = solve_pinned_poisson(K, rhs_mid, pin_index=0)
    vel_corner_mid, _, _ = build_p2_velocity_fields(
        psi_mid,
        face_to_dofs,
        J,
        Ginv,
        normals,
        n_vertices=vertices.shape[0],
    )

    face_out, bary_out, pos_out, hops_b = advect_stage_b_from_midpoint_batch(
        vertices,
        faces,
        f_adj,
        vel_corner_mid,
        face_ids,
        bary,
        face_mid,
        bary_mid,
        dt=dt,
    )

    face_mid_ref = np.empty_like(face_mid)
    bary_mid_ref = np.empty_like(bary_mid)
    hops_a_ref = np.empty_like(hops_a)
    for i in range(face_ids.shape[0]):
        u0 = sample_velocity_from_corners(vel_corner0, int(face_ids[i]), bary[i])
        fid, bi, _, hop = advance_const_velocity_event_driven(
            vertices,
            faces,
            f_adj,
            int(face_ids[i]),
            bary[i],
            u0,
            0.5 * dt,
        )
        face_mid_ref[i] = fid
        bary_mid_ref[i] = bi
        hops_a_ref[i] = hop

    face_out_ref = np.empty_like(face_out)
    bary_out_ref = np.empty_like(bary_out)
    pos_out_ref = np.empty_like(pos_out)
    hops_b_ref = np.empty_like(hops_b)
    for i in range(face_ids.shape[0]):
        umid = sample_velocity_from_corners(vel_corner_mid, int(face_mid_ref[i]), bary_mid_ref[i])
        fid, bi, pi, hop = advance_const_velocity_event_driven(
            vertices,
            faces,
            f_adj,
            int(face_ids[i]),
            bary[i],
            umid,
            dt,
        )
        face_out_ref[i] = fid
        bary_out_ref[i] = bi
        pos_out_ref[i] = pi
        hops_b_ref[i] = hop

    assert np.array_equal(face_mid, face_mid_ref)
    assert np.allclose(bary_mid, bary_mid_ref)
    assert np.array_equal(hops_a, hops_a_ref)

    assert np.array_equal(face_out, face_out_ref)
    assert np.allclose(bary_out, bary_out_ref)
    assert np.allclose(pos_out, pos_out_ref)
    assert np.array_equal(hops_b, hops_b_ref)
