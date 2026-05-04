from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rheidos.apps.p2.higher_genus.harmonic_basis.app import (
    App,
    interpolate_harmonic_basis_velocity,
)
from rheidos.apps.p2.modules.higher_genus.harmonic_basis import (
    HarmonicBasis,
    _orthonormalize_rows_l2,
)
from rheidos.apps.p2.modules.higher_genus.tree_cotree import TreeCotreeModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import World


def _set_mesh(
    mesh: SurfaceMeshModule,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    mesh.set_mesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )


def _tetrahedron_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [0, 2, 3],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _periodic_torus_mesh(
    major_segments: int = 4,
    minor_segments: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    major_radius = 2.0
    minor_radius = 0.6
    for i in range(major_segments):
        theta = 2.0 * np.pi * i / major_segments
        for j in range(minor_segments):
            phi = 2.0 * np.pi * j / minor_segments
            radius = major_radius + minor_radius * np.cos(phi)
            vertices.append(
                [
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    minor_radius * np.sin(phi),
                ]
            )

    def vid(i: int, j: int) -> int:
        return (i % major_segments) * minor_segments + (j % minor_segments)

    faces = []
    for i in range(major_segments):
        for j in range(minor_segments):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )


def _build_harmonic_basis(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[SurfaceMeshModule, DEC, TreeCotreeModule, HarmonicBasis]:
    world = World()
    mesh = world.require(SurfaceMeshModule)
    dec = world.require(DEC, mesh=mesh)
    tree_cotree = world.require(TreeCotreeModule, mesh=mesh)
    harmonic_basis = world.require(
        HarmonicBasis,
        dec=dec,
        tree_cotree=tree_cotree,
    )
    _set_mesh(mesh, vertices, faces)
    return mesh, dec, tree_cotree, harmonic_basis


def test_harmonic_basis_torus_projects_all_generator_forms() -> None:
    vertices, faces = _periodic_torus_mesh()
    mesh, dec, tree_cotree, harmonic_basis = _build_harmonic_basis(vertices, faces)

    gamma = harmonic_basis.gamma.get()
    raw_gamma = harmonic_basis.raw_gamma.get()
    l2_gram = harmonic_basis.l2_gram.get()
    alpha = harmonic_basis.alpha.get()
    beta = harmonic_basis.beta.get()
    star1 = dec.star1.get()
    n_edges = mesh.E_verts.get().shape[0]
    n_vertices = mesh.V_pos.get().shape[0]

    assert tree_cotree.genus.get() == 1
    assert raw_gamma.shape == (2, n_edges)
    assert gamma.shape == (2, n_edges)
    assert l2_gram.shape == (2, 2)
    assert alpha.shape == (2, n_vertices)
    assert beta.shape == (2, n_vertices)

    for basis_form in gamma:
        np.testing.assert_allclose(
            dec.d1(basis_form),
            np.zeros(faces.shape[0], dtype=np.float64),
            atol=1e-10,
        )

    residual = dec.d0_transpose(gamma * star1[None, :])
    np.testing.assert_allclose(
        residual,
        np.zeros((2, n_vertices), dtype=np.float64),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        l2_gram,
        (raw_gamma * star1[None, :]) @ raw_gamma.T,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        (gamma * star1[None, :]) @ gamma.T,
        np.eye(2, dtype=np.float64),
        atol=1e-10,
    )


def test_harmonic_basis_sphere_returns_empty_basis() -> None:
    vertices, faces = _tetrahedron_mesh()
    mesh, _, tree_cotree, harmonic_basis = _build_harmonic_basis(vertices, faces)

    assert tree_cotree.genus.get() == 0
    np.testing.assert_array_equal(
        harmonic_basis.raw_gamma.get(),
        np.empty((0, mesh.E_verts.get().shape[0]), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        harmonic_basis.l2_gram.get(),
        np.empty((0, 0), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        harmonic_basis.gamma.get(),
        np.empty((0, mesh.E_verts.get().shape[0]), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        harmonic_basis.alpha.get(),
        np.empty((0, mesh.V_pos.get().shape[0]), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        harmonic_basis.beta.get(),
        np.empty((0, mesh.V_pos.get().shape[0]), dtype=np.float64),
    )


def test_l2_orthonormalize_rows_matches_dec_inner_product() -> None:
    raw_gamma = np.array(
        [
            [1.0, 2.0, 0.5],
            [0.25, -1.0, 3.0],
        ],
        dtype=np.float64,
    )
    star1 = np.array([2.0, 0.5, 1.5], dtype=np.float64)

    gamma, l2_gram = _orthonormalize_rows_l2(raw_gamma, star1)

    np.testing.assert_allclose(
        l2_gram,
        (raw_gamma * star1[None, :]) @ raw_gamma.T,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        (gamma * star1[None, :]) @ gamma.T,
        np.eye(2, dtype=np.float64),
        atol=1e-12,
    )


class _FakeInputIO:
    def __init__(self, faceids: np.ndarray, bary: np.ndarray) -> None:
        self._faceids = faceids
        self._bary = bary

    def read_point(self, name: str, *, components=None):
        if name == "faceid":
            return self._faceids
        if name == "bary" and components == 3:
            return self._bary
        raise KeyError(name)


class _FakeParm:
    def __init__(self, value) -> None:
        self._value = value

    def eval(self):
        return self._value


class _FakeNode:
    def __init__(self, basis_id) -> None:
        self._parms = {"basis_id": _FakeParm(basis_id)}

    def parm(self, name: str):
        return self._parms.get(name)


class _FakeCookContext:
    def __init__(
        self,
        world: World,
        *,
        faceids: np.ndarray,
        bary: np.ndarray,
        basis_id,
    ) -> None:
        self._world = world
        self.node = _FakeNode(basis_id)
        self.point_writes = {}
        self.detail_writes = {}
        self._input_io = _FakeInputIO(faceids, bary)

    def world(self) -> World:
        return self._world

    def input_io(self, index: int):
        if index != 1:
            raise IndexError(index)
        return self._input_io

    def write_point(self, name: str, values, *, create: bool = True) -> None:
        del create
        self.point_writes[name] = np.asarray(values)

    def write_detail(self, name: str, values, *, create: bool = True) -> None:
        del create
        self.detail_writes[name] = np.asarray(values)


def _build_single_triangle_app() -> tuple[World, App, np.ndarray]:
    world = World()
    mods = world.require(App)
    mods.mesh.set_mesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([[0, 1, 2]], dtype=np.int32),
    )
    gamma = np.array([[2.0, 3.0, 5.0]], dtype=np.float64)
    mods.harmonic_basis.gamma = SimpleNamespace(get=lambda: gamma)
    mods.tree_cotree.genus = SimpleNamespace(get=lambda: 1)
    mods.tree_cotree.generator_count = SimpleNamespace(get=lambda: 2)
    return world, mods, gamma


def test_interpolate_harmonic_basis_velocity_writes_selected_basis() -> None:
    world, mods, gamma = _build_single_triangle_app()
    faceids = np.array([0, 0], dtype=np.int32)
    bary = np.array([[0.2, 0.3, 0.5], [0.7, 0.1, 0.2]], dtype=np.float64)
    ctx = _FakeCookContext(world, faceids=faceids, bary=bary, basis_id=0)

    interpolate_harmonic_basis_velocity(ctx)

    expected = mods.whitney_1form.interpolate(gamma[0], (faceids, bary))
    np.testing.assert_allclose(ctx.point_writes["harmonic_basis_vel"], expected)
    np.testing.assert_array_equal(ctx.detail_writes["harmonic_basis_count"], [1])
    np.testing.assert_array_equal(ctx.detail_writes["harmonic_basis_id"], [0])
    np.testing.assert_array_equal(ctx.detail_writes["genus"], [1])
    np.testing.assert_array_equal(ctx.detail_writes["generator_count"], [2])


def test_interpolate_harmonic_basis_velocity_rejects_out_of_range_basis() -> None:
    world, _, _ = _build_single_triangle_app()
    ctx = _FakeCookContext(
        world,
        faceids=np.array([0], dtype=np.int32),
        bary=np.array([[0.2, 0.3, 0.5]], dtype=np.float64),
        basis_id=1,
    )

    with pytest.raises(ValueError, match="basis_id 1 is out of range"):
        interpolate_harmonic_basis_velocity(ctx)
