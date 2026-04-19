from __future__ import annotations

import numpy as np
import pytest

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


def _closest_point_bary_reference(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return np.array([1.0 - v, v, 0.0], dtype=np.float64)

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return np.array([1.0 - w, 0.0, w], dtype=np.float64)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.array([0.0, 1.0 - w, w], dtype=np.float64)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return np.array([1.0 - v - w, v, w], dtype=np.float64)


def _point_from_bary(triangle: np.ndarray, bary: np.ndarray) -> np.ndarray:
    return bary @ triangle


def test_project_on_nearest_face_returns_faceids_and_bary_for_off_plane_points() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    _set_mesh(
        mesh,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
    )

    faceids, bary, projected = mesh.project_on_nearest_face(
        np.array([[0.2, 0.3, 0.7]], dtype=np.float64)
    )

    np.testing.assert_array_equal(faceids, np.array([0], dtype=np.int32))
    np.testing.assert_allclose(bary, np.array([[0.5, 0.2, 0.3]], dtype=np.float64))
    np.testing.assert_allclose(projected, np.array([[0.2, 0.3, 0.0]], dtype=np.float64))


@pytest.mark.parametrize(
    ("point", "expected_bary"),
    [
        ([0.5, -0.2, 0.4], [0.5, 0.5, 0.0]),
        ([-0.4, -0.1, 0.3], [1.0, 0.0, 0.0]),
    ],
)
def test_project_on_nearest_face_clamps_to_edges_and_vertices(
    point: list[float],
    expected_bary: list[float],
) -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    _set_mesh(
        mesh,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
    )

    faceids, bary, projected = mesh.project_on_nearest_face(np.array([point], dtype=np.float64))

    np.testing.assert_array_equal(faceids, np.array([0], dtype=np.int32))
    np.testing.assert_allclose(
        bary,
        np.array([expected_bary], dtype=np.float64),
        atol=1e-12,
    )
    triangle = mesh.V_pos.get()[mesh.F_verts.get()[0]]
    np.testing.assert_allclose(projected[0], _point_from_bary(triangle, bary[0]), atol=1e-12)


def test_project_on_nearest_face_matches_scalar_reference_on_obtuse_triangle() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.3, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    _set_mesh(mesh, vertices, faces)

    point = np.array([[1.8, 0.5, 0.25]], dtype=np.float64)
    faceids, bary, projected = mesh.project_on_nearest_face(point)

    expected_bary = _closest_point_bary_reference(point[0], vertices[faces[0]])
    np.testing.assert_array_equal(faceids, np.array([0], dtype=np.int32))
    np.testing.assert_allclose(bary[0], expected_bary, atol=1e-12)
    np.testing.assert_allclose(
        projected[0],
        expected_bary @ vertices[faces[0]],
        atol=1e-12,
    )


def test_project_on_nearest_face_picks_true_nearest_face_in_3d() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    _set_mesh(
        mesh,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        [[0, 1, 2], [0, 1, 3]],
    )

    faceids, bary, projected = mesh.project_on_nearest_face(
        np.array([[0.2, 0.1, 0.7]], dtype=np.float64)
    )

    np.testing.assert_array_equal(faceids, np.array([1], dtype=np.int32))
    np.testing.assert_allclose(bary, np.array([[0.1, 0.2, 0.7]], dtype=np.float64))
    np.testing.assert_allclose(projected, np.array([[0.2, 0.0, 0.7]], dtype=np.float64))


def test_project_on_nearest_face_ignores_degenerate_faces_and_raises_if_all_degenerate() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    _set_mesh(
        mesh,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        [[0, 1, 2], [0, 1, 3]],
    )

    faceids, bary, projected = mesh.project_on_nearest_face(
        np.array([[0.2, 0.2, 0.4]], dtype=np.float64)
    )

    np.testing.assert_array_equal(faceids, np.array([1], dtype=np.int32))
    np.testing.assert_allclose(bary, np.array([[0.6, 0.2, 0.2]], dtype=np.float64))
    triangle = mesh.V_pos.get()[mesh.F_verts.get()[1]]
    np.testing.assert_allclose(projected[0], _point_from_bary(triangle, bary[0]), atol=1e-12)

    _set_mesh(
        mesh,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0, 1, 2]],
    )

    with pytest.raises(ValueError, match="degenerate faces"):
        mesh.project_on_nearest_face(np.array([[0.2, 0.0, 0.1]], dtype=np.float64))


def test_project_on_nearest_face_handles_empty_inputs_and_invalid_shapes() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    _set_mesh(
        mesh,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
    )

    faceids, bary, projected = mesh.project_on_nearest_face(
        np.empty((0, 3), dtype=np.float64)
    )
    assert faceids.shape == (0,)
    assert bary.shape == (0, 3)
    assert projected.shape == (0, 3)

    with pytest.raises(ValueError, match=r"got \(3,\)"):
        mesh.project_on_nearest_face(np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_project_on_nearest_face_raises_on_empty_mesh() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    _set_mesh(
        mesh,
        [[0.0, 0.0, 0.0]],
        np.empty((0, 3), dtype=np.int32),
    )

    with pytest.raises(ValueError, match="empty mesh"):
        mesh.project_on_nearest_face(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))


def test_project_on_nearest_face_matches_across_chunk_sizes_and_reuses_cached_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)

    _set_mesh(
        mesh,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        [[0, 1, 2], [0, 1, 3]],
    )

    points = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.4, 0.1, 0.6],
            [0.2, 0.7, 0.1],
            [0.8, 0.1, 0.1],
        ],
        dtype=np.float64,
    )

    assert mesh.F_origin.peek() is None
    faceids_ref, bary_ref, projected_ref = mesh.project_on_nearest_face(points)
    origin_buf = mesh.F_origin.peek()
    edge01_buf = mesh.F_edge01.peek()
    edge02_buf = mesh.F_edge02.peek()

    monkeypatch.setattr(SurfaceMeshModule, "PROJECTION_MAX_FACE_POINT_PAIRS", 2)
    faceids_chunked, bary_chunked, projected_chunked = mesh.project_on_nearest_face(points)

    assert mesh.F_origin.peek() is origin_buf
    assert mesh.F_edge01.peek() is edge01_buf
    assert mesh.F_edge02.peek() is edge02_buf
    np.testing.assert_array_equal(faceids_chunked, faceids_ref)
    np.testing.assert_allclose(bary_chunked, bary_ref, atol=1e-12)
    np.testing.assert_allclose(projected_chunked, projected_ref, atol=1e-12)
