from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rheidos.apps.p2.higher_genus.vortex_dynamics.app import CombinedVelocityModule
from rheidos.apps.p2.modules.higher_genus.harmonic_velocity import (
    HarmonicVelocityFieldModule,
)
from rheidos.apps.p2.modules.p1_space.p1_velocity import (
    area_weighted_face_vectors_to_vertices,
)
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ResourceSpec, World, shape_map


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


def _basis_face_shape(mesh_ref, basis_count: int):
    def shape_fn(reg):
        faces = reg.read(mesh_ref.name, ensure=False)
        if faces is None or not hasattr(faces, "shape"):
            return None
        return (basis_count, int(faces.shape[0]), 3)

    return shape_fn


class _FakeDualHarmonicField(ModuleBase):
    NAME = "FakeDualHarmonicField"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        basis_count: int,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        self.zeta_face = self.resource(
            "zeta_face",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=_basis_face_shape(mesh.F_verts, basis_count),
            ),
            declare=True,
        )


class _FakeVelocity(ModuleBase):
    NAME = "FakeVelocity"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        self.vel_per_face = self.resource(
            "vel_per_face",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(mesh.F_verts, lambda s: (s[0], 3)),
            ),
            declare=True,
        )
        self.vel_per_vertex = self.resource(
            "vel_per_vertex",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(mesh.V_pos, lambda s: (s[0], 3)),
            ),
            declare=True,
        )


def _build_harmonic_velocity(
    basis_count: int,
) -> tuple[SurfaceMeshModule, _FakeDualHarmonicField, HarmonicVelocityFieldModule]:
    world = World()
    mesh = world.require(SurfaceMeshModule)
    vertices, faces = _square_mesh()
    mesh.set_mesh(vertices, faces)
    dual = world.require(
        _FakeDualHarmonicField,
        mesh=mesh,
        basis_count=basis_count,
    )
    harmonic = world.require(
        HarmonicVelocityFieldModule,
        mesh=mesh,
        dual_harmonic_field=dual,
    )
    return mesh, dual, harmonic


def test_area_weighted_face_vectors_to_vertices() -> None:
    face_velocity = np.array(
        [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        dtype=np.float64,
    )
    face_area = np.array([1.0, 3.0], dtype=np.float64)
    face_verts = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    got = area_weighted_face_vectors_to_vertices(
        face_velocity,
        face_area,
        face_verts,
        4,
    )

    expected = np.array(
        [
            [0.5, 3.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, 3.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(got, expected)


def test_harmonic_velocity_facewise_and_smoothed_interpolation() -> None:
    mesh, dual, harmonic = _build_harmonic_velocity(basis_count=2)
    zeta_face = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 3.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    dual.zeta_face.set(zeta_face)
    coefficients = np.array([2.0, -0.5], dtype=np.float64)
    harmonic.set_coefficients(coefficients)

    faceids = np.array([0, 1], dtype=np.int32)
    bary = np.array([[0.2, 0.3, 0.5], [0.25, 0.25, 0.5]], dtype=np.float64)

    expected_face = np.einsum("k,kfi->fi", coefficients, zeta_face)
    np.testing.assert_allclose(harmonic.vel_per_face.get(), expected_face)
    np.testing.assert_allclose(
        harmonic.interpolate((faceids, bary), smooth=False),
        expected_face[faceids],
    )

    expected_vertex = area_weighted_face_vectors_to_vertices(
        expected_face,
        mesh.F_area.get(),
        mesh.F_verts.get(),
        mesh.V_pos.get().shape[0],
    )
    verts = mesh.F_verts.get()[faceids]
    expected_smoothed = np.einsum("ni,nij->nj", bary, expected_vertex[verts])
    np.testing.assert_allclose(
        harmonic.interpolate((faceids, bary), smooth=True),
        expected_smoothed,
    )


def test_harmonic_velocity_empty_basis_returns_zero_velocity() -> None:
    _mesh, dual, harmonic = _build_harmonic_velocity(basis_count=0)
    dual.zeta_face.set(np.empty((0, 2, 3), dtype=np.float64))

    faceids = np.array([0, 1], dtype=np.int32)
    bary = np.array([[0.2, 0.3, 0.5], [0.25, 0.25, 0.5]], dtype=np.float64)

    np.testing.assert_allclose(
        harmonic.interpolate((faceids, bary), smooth=False),
        np.zeros((2, 3), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        harmonic.harmonic_c.get(),
        np.empty((0,), dtype=np.float64),
    )


def test_harmonic_velocity_rejects_wrong_coefficient_shape() -> None:
    _mesh, dual, harmonic = _build_harmonic_velocity(basis_count=2)
    dual.zeta_face.set(np.zeros((2, 2, 3), dtype=np.float64))

    with pytest.raises(ValueError, match="harmonic coefficients"):
        harmonic.set_coefficients(np.array([1.0], dtype=np.float64))


def test_combined_velocity_sums_facewise_before_smoothing() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)
    vertices, faces = _square_mesh()
    mesh.set_mesh(vertices, faces)

    coexact = world.require(_FakeVelocity, mesh=mesh, scope="coexact")
    harmonic = world.require(_FakeVelocity, mesh=mesh, scope="harmonic")
    combined = world.require(
        CombinedVelocityModule,
        mesh=mesh,
        coexact_velocity=coexact,
        harmonic_velocity=harmonic,
    )

    coexact_face = np.array(
        [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        dtype=np.float64,
    )
    harmonic_face = np.array(
        [[0.0, 1.0, 0.0], [6.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    coexact.vel_per_face.set(coexact_face)
    harmonic.vel_per_face.set(harmonic_face)

    expected_face = coexact_face + harmonic_face
    np.testing.assert_allclose(combined.vel_per_face.get(), expected_face)

    faceids = np.array([0, 1], dtype=np.int32)
    bary = np.array([[0.2, 0.3, 0.5], [0.25, 0.25, 0.5]], dtype=np.float64)
    expected_vertex = area_weighted_face_vectors_to_vertices(
        expected_face,
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
        combined.interpolate((faceids, bary), smooth=True),
        expected_smoothed,
    )
