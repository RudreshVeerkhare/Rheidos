from __future__ import annotations

import numpy as np
import pytest

from rheidos.apps.p2.double_obstacle.harmonic_basis import (
    HarmonicBasisFieldModule,
    HarmonicBasisModule,
)
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_velocity import (
    area_weighted_face_vectors_to_vertices,
)
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import World


def _square_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return vertices, faces


def _build_basis_field() -> tuple[
    SurfaceMeshModule,
    HarmonicBasisModule,
    HarmonicBasisFieldModule,
]:
    world = World()
    mesh = world.require(SurfaceMeshModule)
    vertices, faces = _square_mesh()
    mesh.set_mesh(vertices, faces)

    dec = world.require(DEC, mesh=mesh)
    harmonic_basis = world.require(
        HarmonicBasisModule,
        mesh=mesh,
        dec=dec,
        harmonic_dim=2,
    )
    basis_field = world.require(
        HarmonicBasisFieldModule,
        mesh=mesh,
        dec=dec,
        harmonic_basis=harmonic_basis,
    )
    return mesh, harmonic_basis, basis_field


def test_harmonic_basis_field_interpolate_selects_basis_id() -> None:
    mesh, harmonic_basis, basis_field = _build_basis_field()
    basis = np.array(
        [
            [1.0, 2.0, 4.0, 8.0],
            [-1.0, 0.5, 3.0, -2.0],
        ],
        dtype=np.float64,
    )
    harmonic_basis.basis.set(basis)

    coeffs = basis[:, mesh.F_verts.get()]
    j_grad = np.cross(mesh.F_normal.get()[:, None, :], mesh.grad_bary.get())
    expected_face = np.einsum("kfa,fai->kfi", coeffs, j_grad)

    np.testing.assert_allclose(basis_field.vel_per_face.get(), expected_face)

    faceids = np.array([0, 1], dtype=np.int32)
    bary = np.array([[0.2, 0.3, 0.5], [0.25, 0.25, 0.5]], dtype=np.float64)
    np.testing.assert_allclose(
        basis_field.interpolate((faceids, bary), smooth=False, basis_id=1),
        expected_face[1, faceids],
    )

    expected_vertex = area_weighted_face_vectors_to_vertices(
        expected_face[1],
        mesh.F_area.get(),
        mesh.F_verts.get(),
        mesh.V_pos.get().shape[0],
    )
    expected_smoothed = np.einsum(
        "ni,nij->nj",
        bary,
        expected_vertex[mesh.F_verts.get()[faceids]],
    )
    np.testing.assert_allclose(
        basis_field.interpolate((faceids, bary), smooth=True, basis_id=1),
        expected_smoothed,
    )


def test_harmonic_basis_field_rejects_invalid_basis_id() -> None:
    _mesh, harmonic_basis, basis_field = _build_basis_field()
    harmonic_basis.basis.set(np.zeros((2, 4), dtype=np.float64))

    with pytest.raises(RuntimeError, match="basis_id"):
        basis_field.interpolate([], basis_id=2)
