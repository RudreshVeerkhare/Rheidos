from __future__ import annotations

import numpy as np
import pytest

from rheidos.apps.p2.modules.higher_genus.tree_cotree import TreeCotreeModule
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


def _square_disk_mesh() -> tuple[np.ndarray, np.ndarray]:
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


def _build_tree_cotree(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[SurfaceMeshModule, TreeCotreeModule]:
    world = World()
    mesh = world.require(SurfaceMeshModule)
    tree_cotree = world.require(TreeCotreeModule, mesh=mesh)
    _set_mesh(mesh, vertices, faces)
    return mesh, tree_cotree


def _assert_cycle_is_closed(
    cycle: np.ndarray,
    e_verts: np.ndarray,
    n_vertices: int,
) -> None:
    incidence = np.zeros(n_vertices, dtype=np.int32)
    for edge_id, sign in cycle:
        u = int(e_verts[edge_id, 0])
        v = int(e_verts[edge_id, 1])
        incidence[u] -= int(sign)
        incidence[v] += int(sign)
    np.testing.assert_array_equal(incidence, np.zeros(n_vertices, dtype=np.int32))


def test_tree_cotree_sphere_has_no_homology_generators() -> None:
    vertices, faces = _tetrahedron_mesh()
    mesh, tree_cotree = _build_tree_cotree(vertices, faces)

    assert tree_cotree.genus.get() == 0
    assert tree_cotree.generator_count.get() == 0
    np.testing.assert_array_equal(
        tree_cotree.generator_edge_ids.get(),
        np.empty((0,), dtype=np.int32),
    )
    assert len(tree_cotree.dual_tree_edge_ids.get()) == faces.shape[0] - 1
    assert len(tree_cotree.primal_tree_edge_ids.get()) == vertices.shape[0] - 1
    assert tree_cotree.primal_generator_cycles.get() == []
    assert tree_cotree.dual_generator_face_loops.get() == []
    assert tree_cotree.generator_chain_matrix.get().shape == (0, mesh.E_verts.get().shape[0])
    assert tree_cotree.closed_dual_generator_1forms.get().shape == (
        0,
        mesh.E_verts.get().shape[0],
    )


def test_tree_cotree_torus_produces_two_closed_generators() -> None:
    vertices, faces = _periodic_torus_mesh()
    mesh, tree_cotree = _build_tree_cotree(vertices, faces)

    dual_tree_edges = set(map(int, tree_cotree.dual_tree_edge_ids.get()))
    primal_tree_edges = set(map(int, tree_cotree.primal_tree_edge_ids.get()))
    generator_edges = set(map(int, tree_cotree.generator_edge_ids.get()))
    all_edges = set(range(mesh.E_verts.get().shape[0]))

    assert tree_cotree.genus.get() == 1
    assert tree_cotree.generator_count.get() == 2
    assert len(generator_edges) == 2
    assert dual_tree_edges.isdisjoint(primal_tree_edges)
    assert dual_tree_edges.isdisjoint(generator_edges)
    assert primal_tree_edges.isdisjoint(generator_edges)
    assert dual_tree_edges | primal_tree_edges | generator_edges == all_edges

    chain_matrix = tree_cotree.generator_chain_matrix.get()
    cycles = tree_cotree.primal_generator_cycles.get()
    assert chain_matrix.shape == (2, mesh.E_verts.get().shape[0])
    assert len(cycles) == 2
    for generator_id, cycle in enumerate(cycles):
        _assert_cycle_is_closed(cycle, mesh.E_verts.get(), vertices.shape[0])
        expected_row = np.zeros(mesh.E_verts.get().shape[0], dtype=np.int32)
        for edge_id, sign in cycle:
            expected_row[int(edge_id)] += int(sign)
        np.testing.assert_array_equal(chain_matrix[generator_id], expected_row)

    face_loops = tree_cotree.dual_generator_face_loops.get()
    crossed_edges = tree_cotree.dual_generator_crossed_edges.get()
    crossing_signs = tree_cotree.dual_generator_crossing_signs.get()
    closed_1forms = tree_cotree.closed_dual_generator_1forms.get()
    assert len(face_loops) == 2
    assert len(crossed_edges) == 2
    assert len(crossing_signs) == 2
    assert closed_1forms.shape == (2, mesh.E_verts.get().shape[0])
    for generator_id, (face_loop, crossed, signs) in enumerate(
        zip(face_loops, crossed_edges, crossing_signs)
    ):
        assert int(face_loop[0]) == int(face_loop[-1])
        assert len(crossed) == len(face_loop) - 1
        assert len(signs) == len(crossed)

        expected_row = np.zeros(mesh.E_verts.get().shape[0], dtype=np.int32)
        for index, (edge_id_raw, sign_raw) in enumerate(zip(crossed, signs)):
            edge_id = int(edge_id_raw)
            sign = int(sign_raw)
            from_face = int(face_loop[index])
            to_face = int(face_loop[index + 1])
            edge_faces = {int(value) for value in mesh.E_faces.get()[edge_id]}

            assert from_face in edge_faces
            assert to_face in edge_faces
            assert from_face != to_face
            local_slots = np.flatnonzero(mesh.F_edges.get()[from_face] == edge_id)
            assert local_slots.shape == (1,)
            assert sign == int(mesh.F_edge_sign.get()[from_face, int(local_slots[0])])
            expected_row[edge_id] += sign

        np.testing.assert_array_equal(closed_1forms[generator_id], expected_row)

    face_coboundary = np.einsum(
        "fl,gfl->gf",
        mesh.F_edge_sign.get(),
        closed_1forms[:, mesh.F_edges.get()],
    )
    np.testing.assert_array_equal(
        face_coboundary,
        np.zeros((2, faces.shape[0]), dtype=np.int32),
    )


def test_tree_cotree_rejects_meshes_with_boundary() -> None:
    vertices, faces = _square_disk_mesh()
    mesh, tree_cotree = _build_tree_cotree(vertices, faces)

    with pytest.raises(ValueError, match="closed surface"):
        tree_cotree.generator_edge_ids.get()

    assert mesh.boundary_edge_count.get() == 4


def test_tree_cotree_recomputes_deterministically() -> None:
    vertices, faces = _periodic_torus_mesh()
    mesh, tree_cotree = _build_tree_cotree(vertices, faces)

    first = {
        "dual": tree_cotree.dual_tree_edge_ids.get().copy(),
        "primal": tree_cotree.primal_tree_edge_ids.get().copy(),
        "generators": tree_cotree.generator_edge_ids.get().copy(),
        "labels": tree_cotree.generator_edge_labels.get().copy(),
        "chains": tree_cotree.generator_chain_matrix.get().copy(),
        "cycles": [cycle.copy() for cycle in tree_cotree.primal_generator_cycles.get()],
        "dual_faces": [loop.copy() for loop in tree_cotree.dual_generator_face_loops.get()],
        "dual_crossed": [
            loop.copy() for loop in tree_cotree.dual_generator_crossed_edges.get()
        ],
        "dual_signs": [
            signs.copy() for signs in tree_cotree.dual_generator_crossing_signs.get()
        ],
        "closed_1forms": tree_cotree.closed_dual_generator_1forms.get().copy(),
    }

    _set_mesh(mesh, vertices, faces)

    np.testing.assert_array_equal(first["dual"], tree_cotree.dual_tree_edge_ids.get())
    np.testing.assert_array_equal(first["primal"], tree_cotree.primal_tree_edge_ids.get())
    np.testing.assert_array_equal(first["generators"], tree_cotree.generator_edge_ids.get())
    np.testing.assert_array_equal(first["labels"], tree_cotree.generator_edge_labels.get())
    np.testing.assert_array_equal(first["chains"], tree_cotree.generator_chain_matrix.get())
    np.testing.assert_array_equal(
        first["closed_1forms"],
        tree_cotree.closed_dual_generator_1forms.get(),
    )
    for expected, actual in zip(first["cycles"], tree_cotree.primal_generator_cycles.get()):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(first["dual_faces"], tree_cotree.dual_generator_face_loops.get()):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(
        first["dual_crossed"],
        tree_cotree.dual_generator_crossed_edges.get(),
    ):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(
        first["dual_signs"],
        tree_cotree.dual_generator_crossing_signs.get(),
    ):
        np.testing.assert_array_equal(expected, actual)
