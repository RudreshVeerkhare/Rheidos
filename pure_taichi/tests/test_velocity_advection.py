from __future__ import annotations

import numpy as np
import pytest

from pure_taichi.advection import advance_const_velocity_event_driven
from pure_taichi.elements import P2Element, bary_to_ref
from pure_taichi.math_utils import grad_ref_to_surface
from pure_taichi.mesh import build_face_geometry
from pure_taichi.space import build_p2_space_data
from pure_taichi.velocity import build_p2_velocity_fields, sample_velocity_from_corners


def test_velocity_linear_face_consistency() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.9, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    space = build_p2_space_data(vertices.shape[0], faces)
    geom = build_face_geometry(vertices, faces)

    n0 = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    normals = np.array([n0 / np.linalg.norm(n0)], dtype=np.float64)

    rng = np.random.default_rng(7)
    psi = rng.normal(size=(space.ndof,))

    vel_corner, _, _ = build_p2_velocity_fields(
        psi,
        space.face_to_dofs,
        geom.J,
        geom.Ginv,
        normals,
        n_vertices=3,
    )

    bary = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    u_interp = sample_velocity_from_corners(vel_corner, 0, bary)

    element = P2Element()
    dofs = space.face_to_dofs[0]
    local_psi = psi[dofs]
    xi, eta = bary_to_ref(bary)
    dphi_ref = element.eval_grad_ref(xi, eta)
    dphi = grad_ref_to_surface(geom.J[0], geom.Ginv[0], dphi_ref)
    grad_psi = np.sum(local_psi[:, None] * dphi, axis=0)
    u_exact = np.cross(normals[0], grad_psi)

    assert np.allclose(u_interp, u_exact, atol=1e-10)


def test_boundary_crossing_raises_error() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    f_adj = np.array([[-1, -1, -1]], dtype=np.int32)

    with pytest.raises(RuntimeError, match="closed mesh"):
        advance_const_velocity_event_driven(
            vertices,
            faces,
            f_adj,
            face_id=0,
            bary=np.array([0.2, 0.2, 0.6], dtype=np.float64),
            vel_world=np.array([0.8, 0.0, 0.0], dtype=np.float64),
            dt=1.0,
        )
