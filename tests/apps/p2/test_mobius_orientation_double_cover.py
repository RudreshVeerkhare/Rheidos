from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags, eye

from rheidos.apps.p2.mobius.cook import App, DoubleCoverPointVortex
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.mobius.orientation_double_cover import (
    OrientationDoubleCover,
    build_orientation_double_cover,
    connected_component_count,
    infer_coincident_boundary_edge_pairs,
)
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import World


def _load_obj_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError("The test OBJ must contain only triangles")
            faces.append([int(value.split("/", 1)[0]) - 1 for value in fields[1:]])
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )


@pytest.fixture(scope="module")
def mobius_obj_mesh() -> tuple[np.ndarray, np.ndarray]:
    repository_root = Path(__file__).resolve().parents[3]
    return _load_obj_triangles(
        repository_root / "rheidos" / "apps" / "p2" / "mobius" / "mobius.obj"
    )


def _small_cut_mobius() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.array(
        [[x, y, 0.0] for x in range(3) for y in (-1.0, 0.0, 1.0)],
        dtype=np.float64,
    )
    faces: list[list[int]] = []
    for longitudinal in range(2):
        for transverse in range(2):
            a = 3 * longitudinal + transverse
            b = a + 3
            c = b + 1
            d = a + 1
            faces.extend(([a, b, c], [a, c, d]))
    # The right edge is paired in reverse transverse order, producing the twist.
    seams = np.array([[0, 1, 8, 7], [1, 2, 7, 6]], dtype=np.int32)
    return vertices, np.asarray(faces, dtype=np.int32), seams


def _d0_matrix(edges: np.ndarray, vertex_count: int) -> csr_matrix:
    edge_count = edges.shape[0]
    return csr_matrix(
        (
            np.tile(np.array([-1.0, 1.0]), edge_count),
            (
                np.repeat(np.arange(edge_count), 2),
                edges.reshape(-1),
            ),
        ),
        shape=(edge_count, vertex_count),
    )


def test_explicit_twisted_seam_builds_a_cylinder_cover() -> None:
    vertices, faces, seams = _small_cut_mobius()

    data = build_orientation_double_cover(
        vertices,
        faces,
        seam_edge_pairs=seams,
    )

    assert data.cover_faces.shape == (2 * faces.shape[0], 3)
    assert data.cover_vertices.shape[0] == 12
    assert data.cover_edges.shape[0] == 28
    assert data.cover_vertices.shape[0] - data.cover_edges.shape[0] + data.cover_faces.shape[0] == 0
    assert np.count_nonzero(data.cover_edge_faces[:, 1] < 0) == 8
    np.testing.assert_array_equal(data.seam_edge_pairs, seams)
    for projection in (data.pi_vertex, data.pi_edge, data.pi_face):
        _, lift_counts = np.unique(projection, return_counts=True)
        np.testing.assert_array_equal(lift_counts, np.full_like(lift_counts, 2))


def test_explicit_empty_seams_disable_geometric_inference() -> None:
    vertices, faces, _ = _small_cut_mobius()

    data = build_orientation_double_cover(
        vertices,
        faces,
        seam_edge_pairs=np.empty((0, 4), dtype=np.int32),
    )

    # The input is then just an orientable disk, whose orientation cover is two disks.
    assert data.cover_vertices.shape[0] == 2 * vertices.shape[0]
    assert data.cover_vertices.shape[0] - data.cover_edges.shape[0] + data.cover_faces.shape[0] == 2
    assert data.seam_edge_pairs.shape == (0, 4)


def test_seam_pairs_must_reference_two_boundary_edges() -> None:
    vertices, faces, _ = _small_cut_mobius()

    with pytest.raises(ValueError, match="not a boundary edge"):
        build_orientation_double_cover(
            vertices,
            faces,
            seam_edge_pairs=np.array([[0, 4, 8, 7]], dtype=np.int32),
        )


def test_mobius_obj_seam_is_inferred_and_has_two_lifts(
    mobius_obj_mesh: tuple[np.ndarray, np.ndarray],
) -> None:
    vertices, faces = mobius_obj_mesh

    seams = infer_coincident_boundary_edge_pairs(vertices, faces)
    data = build_orientation_double_cover(vertices, faces)

    assert seams.shape == (19, 4)
    np.testing.assert_array_equal(data.seam_edge_pairs, seams)
    assert data.cover_faces.shape == (2774, 3)
    assert data.cover_vertices.shape == (1534, 3)
    assert data.cover_edges.shape == (4308, 2)
    assert data.cover_vertices.shape[0] - data.cover_edges.shape[0] + data.cover_faces.shape[0] == 0
    assert np.count_nonzero(data.cover_edge_faces[:, 1] < 0) == 294

    for permutation in (data.tau_vertex, data.tau_edge, data.tau_face):
        np.testing.assert_array_equal(
            permutation[permutation],
            np.arange(permutation.shape[0]),
        )
        assert not np.any(permutation == np.arange(permutation.shape[0]))
    for operator in (data.P0, data.P1, data.P2):
        identity_error = operator @ operator - eye(operator.shape[0], format="csr")
        identity_error.eliminate_zeros()
        assert identity_error.nnz == 0


def test_rheidos_module_runs_dec_on_connected_mobius_cover(
    mobius_obj_mesh: tuple[np.ndarray, np.ndarray],
) -> None:
    vertices, faces = mobius_obj_mesh
    world = World()
    base_mesh = world.require(SurfaceMeshModule)
    cover = world.require(OrientationDoubleCover, parent_mesh=base_mesh)
    base_mesh.set_mesh(vertices, faces)

    cover_mesh = cover.ensure()
    cover_vertices = cover_mesh.V_pos.get()
    cover_edges = cover_mesh.E_verts.get()
    cover_faces = cover_mesh.F_verts.get()

    assert connected_component_count(cover_mesh) == 1
    assert len(cover_mesh.boundary_edge_components.get()) == 2
    assert cover_mesh.boundary_edge_count.get() == 294
    assert cover.dec.mesh is cover_mesh

    rng = np.random.default_rng(8327)
    zero_cochain = rng.normal(size=cover_vertices.shape[0])
    one_cochain = rng.normal(size=cover_edges.shape[0])
    np.testing.assert_allclose(
        cover.dec.d1(cover.dec.d0(zero_cochain)),
        np.zeros(cover_faces.shape[0]),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        cover.dec.d0(cover.apply_p0(zero_cochain)),
        cover.apply_p1(cover.dec.d0(zero_cochain)),
        atol=0.0,
    )
    np.testing.assert_allclose(
        cover.dec.d1(cover.apply_p1(one_cochain)),
        cover.apply_p2(cover.dec.d1(one_cochain)),
        atol=0.0,
    )

    star1 = cover.dec.star1.get()
    assert np.all(np.isfinite(star1))
    d0 = _d0_matrix(cover_edges, cover_vertices.shape[0])
    cotan_laplacian = d0.T @ diags(star1) @ d0
    np.testing.assert_allclose(
        cotan_laplacian.toarray(),
        cotan_laplacian.T.toarray(),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        cotan_laplacian @ np.ones(cover_vertices.shape[0]),
        np.zeros(cover_vertices.shape[0]),
        atol=1.0e-12,
    )


def test_generated_surface_mesh_rebuilds_when_base_mesh_changes() -> None:
    vertices, faces, seams = _small_cut_mobius()
    world = World()
    base_mesh = world.require(SurfaceMeshModule)
    cover = world.require(OrientationDoubleCover, parent_mesh=base_mesh)
    cover.set_seam_edge_pairs(seams)
    base_mesh.set_mesh(vertices, faces)

    assert cover.cover_mesh.F_verts.get().shape[0] == 16
    assert cover.cover_mesh.E_verts.get().shape[0] == 28

    # Replacing the base input invalidates both generated geometry and the
    # ordinary SurfaceMesh topology derived from it.
    base_mesh.set_mesh(vertices, faces[:2])
    cover.set_seam_edge_pairs(np.empty((0, 4), dtype=np.int32))
    assert cover.cover_mesh.F_verts.get().shape[0] == 4
    assert cover.cover_mesh.E_verts.get().shape[0] == 10


def test_mobius_app_exposes_cover_mesh_and_cover_dec() -> None:
    app = World().require(App)

    assert app.cover.parent_mesh is app.base_mesh
    assert app.cover_mesh is app.cover.cover_mesh
    assert app.dec is app.cover.dec
    assert app.dec.mesh is app.cover_mesh


def test_point_vortices_are_lifted_in_interleaved_deck_pairs() -> None:
    vertices, faces, seams = _small_cut_mobius()
    world = World()
    base_mesh = world.require(SurfaceMeshModule)
    cover = world.require(OrientationDoubleCover, parent_mesh=base_mesh)
    base_vortices = world.require(PointVortexModule)
    lifted_vortices = world.require(
        DoubleCoverPointVortex,
        base_point_vortex=base_vortices,
        base_mesh=base_mesh,
        double_cover=cover,
    )
    cover.set_seam_edge_pairs(seams)
    base_mesh.set_mesh(vertices, faces)
    base_vortices.set_vortex(
        faceids=np.array([1, 5], dtype=np.int32),
        bary=np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]]),
        gamma=np.array([4.0, -2.0]),
        pos=np.zeros((2, 3)),
    )

    np.testing.assert_array_equal(
        lifted_vortices.face_ids.get(),
        np.array([2, 3, 10, 11], dtype=np.int32),
    )
    np.testing.assert_allclose(
        lifted_vortices.bary.get(),
        np.array(
            [
                [0.2, 0.3, 0.5],
                [0.2, 0.5, 0.3],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
            ]
        ),
    )
    np.testing.assert_allclose(
        lifted_vortices.gamma.get(),
        np.array([4.0, -4.0, -2.0, 2.0]),
    )

    # The explicit cover dependency must invalidate the lifted data when the
    # base topology changes, even if the base vortex resources do not.
    cover.set_seam_edge_pairs(np.empty((0, 4), dtype=np.int32))
    base_mesh.set_mesh(vertices, faces[:2])
    with pytest.raises(ValueError, match="face_ids must be in"):
        lifted_vortices.face_ids.get()


def test_point_vortex_lift_rejects_invalid_base_face_ids() -> None:
    vertices, faces, seams = _small_cut_mobius()
    world = World()
    base_mesh = world.require(SurfaceMeshModule)
    cover = world.require(OrientationDoubleCover, parent_mesh=base_mesh)
    base_vortices = world.require(PointVortexModule)
    lifted_vortices = world.require(
        DoubleCoverPointVortex,
        base_point_vortex=base_vortices,
        base_mesh=base_mesh,
        double_cover=cover,
    )
    cover.set_seam_edge_pairs(seams)
    base_mesh.set_mesh(vertices, faces)
    base_vortices.set_vortex(
        faceids=np.array([-1], dtype=np.int32),
        bary=np.array([[0.2, 0.3, 0.5]]),
        gamma=np.array([1.0]),
        pos=np.zeros((1, 3)),
    )

    with pytest.raises(ValueError, match="face_ids must be in"):
        lifted_vortices.face_ids.get()
