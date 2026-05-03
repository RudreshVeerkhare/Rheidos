from __future__ import annotations

import numpy as np

from rheidos.apps.p2.higher_genus.harmonic_basis.app import HarmonicBasis
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.apps.p2.modules.tree_cotree.tree_cotree_module import TreeCotreeModule
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
    alpha = harmonic_basis.alpha.get()
    beta = harmonic_basis.beta.get()
    n_edges = mesh.E_verts.get().shape[0]
    n_vertices = mesh.V_pos.get().shape[0]

    assert tree_cotree.genus.get() == 1
    assert gamma.shape == (2, n_edges)
    assert alpha.shape == (2, n_vertices)
    assert beta.shape == (2, n_vertices)

    for basis_form in gamma:
        np.testing.assert_allclose(
            dec.d1(basis_form),
            np.zeros(faces.shape[0], dtype=np.float64),
            atol=1e-10,
        )

    residual = dec.d0_transpose(gamma * dec.star1.get()[None, :])
    np.testing.assert_allclose(
        residual,
        np.zeros((2, n_vertices), dtype=np.float64),
        atol=1e-7,
    )


def test_harmonic_basis_sphere_returns_empty_basis() -> None:
    vertices, faces = _tetrahedron_mesh()
    mesh, _, tree_cotree, harmonic_basis = _build_harmonic_basis(vertices, faces)

    assert tree_cotree.genus.get() == 0
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
