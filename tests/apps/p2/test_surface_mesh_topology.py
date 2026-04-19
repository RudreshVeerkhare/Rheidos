from __future__ import annotations

import numpy as np
import pytest

from rheidos.apps.p2.modules.surface_mesh.mesh_topology import (
    _build_boundary_components,
)
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


def _annulus_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 4],
            [1, 5, 4],
            [1, 2, 5],
            [2, 6, 5],
            [2, 3, 6],
            [3, 7, 6],
            [3, 0, 7],
            [0, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def test_surface_mesh_boundary_resources_are_empty_for_closed_mesh() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    vertices, faces = _tetrahedron_mesh()
    _set_mesh(mesh, vertices, faces)

    np.testing.assert_array_equal(
        mesh.boundary_edge_ids.get(),
        np.empty((0,), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        mesh.boundary_vertex_ids.get(),
        np.empty((0,), dtype=np.int32),
    )
    assert mesh.boundary_edge_components.get() == []
    assert mesh.boundary_vertex_components.get() == []
    assert mesh.boundary_edge_count.get() == 0


def test_surface_mesh_boundary_resources_expose_single_ordered_component() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    vertices, faces = _square_disk_mesh()
    _set_mesh(mesh, vertices, faces)

    edge_components = mesh.boundary_edge_components.get()
    vertex_components = mesh.boundary_vertex_components.get()

    np.testing.assert_array_equal(
        mesh.boundary_edge_ids.get(),
        np.array([0, 1, 3, 4], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        mesh.boundary_vertex_ids.get(),
        np.array([0, 1, 2, 3], dtype=np.int32),
    )
    assert mesh.boundary_edge_count.get() == 4
    assert len(edge_components) == 1
    assert len(vertex_components) == 1
    np.testing.assert_array_equal(
        edge_components[0],
        np.array([0, 1, 3, 4], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        vertex_components[0],
        np.array([0, 1, 2, 3], dtype=np.int32),
    )


def test_surface_mesh_boundary_resources_expose_multiple_components() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    vertices, faces = _annulus_mesh()
    _set_mesh(mesh, vertices, faces)

    edge_components = mesh.boundary_edge_components.get()
    vertex_components = mesh.boundary_vertex_components.get()

    np.testing.assert_array_equal(
        mesh.boundary_edge_ids.get(),
        np.array([0, 4, 5, 8, 9, 12, 13, 15], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        mesh.boundary_vertex_ids.get(),
        np.arange(8, dtype=np.int32),
    )
    assert mesh.boundary_edge_count.get() == 8
    assert len(edge_components) == 2
    assert len(vertex_components) == 2
    np.testing.assert_array_equal(
        edge_components[0],
        np.array([0, 5, 9, 13], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        edge_components[1],
        np.array([4, 8, 12, 15], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        vertex_components[0],
        np.array([0, 1, 2, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        vertex_components[1],
        np.array([4, 5, 6, 7], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.sort(np.concatenate(edge_components)),
        mesh.boundary_edge_ids.get(),
    )
    np.testing.assert_array_equal(
        np.sort(np.concatenate(vertex_components)),
        mesh.boundary_vertex_ids.get(),
    )


def test_surface_mesh_boundary_component_order_is_deterministic_across_rebuilds() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    vertices, faces = _annulus_mesh()
    _set_mesh(mesh, vertices, faces)
    edge_ids_first = mesh.boundary_edge_ids.get().copy()
    vertex_ids_first = mesh.boundary_vertex_ids.get().copy()
    edge_components_first = [
        component.copy() for component in mesh.boundary_edge_components.get()
    ]
    vertex_components_first = [
        component.copy() for component in mesh.boundary_vertex_components.get()
    ]

    _set_mesh(mesh, vertices, faces)
    edge_ids_second = mesh.boundary_edge_ids.get()
    vertex_ids_second = mesh.boundary_vertex_ids.get()
    edge_components_second = mesh.boundary_edge_components.get()
    vertex_components_second = mesh.boundary_vertex_components.get()

    np.testing.assert_array_equal(edge_ids_first, edge_ids_second)
    np.testing.assert_array_equal(vertex_ids_first, vertex_ids_second)
    assert len(edge_components_first) == len(edge_components_second)
    assert len(vertex_components_first) == len(vertex_components_second)
    for first, second in zip(edge_components_first, edge_components_second):
        np.testing.assert_array_equal(first, second)
    for first, second in zip(vertex_components_first, vertex_components_second):
        np.testing.assert_array_equal(first, second)


def test_build_boundary_components_handles_open_chains() -> None:
    e_verts = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
        ],
        dtype=np.int32,
    )

    edge_ids, edge_components, vertex_ids, vertex_components = _build_boundary_components(
        e_verts,
        np.array([0, 1, 2], dtype=np.int32),
    )

    np.testing.assert_array_equal(edge_ids, np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(vertex_ids, np.array([0, 1, 2, 3], dtype=np.int32))
    assert len(edge_components) == 1
    assert len(vertex_components) == 1
    np.testing.assert_array_equal(
        edge_components[0],
        np.array([0, 1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        vertex_components[0],
        np.array([0, 1, 2, 3], dtype=np.int32),
    )


def test_build_boundary_components_rejects_branched_boundaries() -> None:
    e_verts = np.array(
        [
            [0, 1],
            [1, 2],
            [1, 3],
        ],
        dtype=np.int32,
    )

    with pytest.raises(ValueError, match="Branched boundary detected"):
        _build_boundary_components(
            e_verts,
            np.array([0, 1, 2], dtype=np.int32),
        )
