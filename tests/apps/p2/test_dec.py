from __future__ import annotations

import numpy as np
import pytest

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import World


def _build_dec(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[SurfaceMeshModule, DEC]:
    world = World()
    mesh = world.require(SurfaceMeshModule)
    dec = world.require(DEC, mesh=mesh)
    mesh.set_mesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )
    return mesh, dec


def test_dec_d0_single_triangle_returns_float_edge_differences() -> None:
    mesh, dec = _build_dec(
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

    zero_form = np.array([2, 5, 11], dtype=np.int32)

    result = dec.d0(zero_form)

    np.testing.assert_array_equal(
        mesh.E_verts.get(),
        np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32),
    )
    np.testing.assert_allclose(result, np.array([3.0, 6.0, 9.0], dtype=np.float64))
    assert result.dtype == np.float64


def test_dec_d0_two_triangle_square_uses_canonical_edge_orientation() -> None:
    mesh, dec = _build_dec(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )

    zero_form = np.array([0.0, 10.0, 15.0, -2.0], dtype=np.float64)

    result = dec.d0(zero_form)

    np.testing.assert_array_equal(
        mesh.E_verts.get(),
        np.array([[0, 1], [1, 2], [0, 2], [2, 3], [0, 3]], dtype=np.int32),
    )
    np.testing.assert_allclose(
        result,
        np.array([10.0, 5.0, 15.0, -17.0, -2.0], dtype=np.float64),
    )


def test_dec_d1_of_d0_is_zero_on_each_face() -> None:
    _, dec = _build_dec(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )

    zero_form = np.array([0.0, 10.0, 15.0, -2.0], dtype=np.float64)

    np.testing.assert_allclose(
        dec.d1(dec.d0(zero_form)),
        np.zeros(2, dtype=np.float64),
    )


def test_dec_d0_rejects_wrong_length() -> None:
    _, dec = _build_dec(
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

    with pytest.raises(ValueError, match="length nV=3"):
        dec.d0(np.array([1.0, 2.0], dtype=np.float64))


def test_dec_d0_rejects_non_1d_input() -> None:
    _, dec = _build_dec(
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

    with pytest.raises(ValueError, match="shape \\(nV,\\)"):
        dec.d0(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))


def test_dec_d1_single_triangle_returns_float_signed_face_sum() -> None:
    mesh, dec = _build_dec(
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

    one_form = np.array([2, 3, 7], dtype=np.int32)

    result = dec.d1(one_form)

    np.testing.assert_array_equal(mesh.F_edges.get(), np.array([[0, 1, 2]]))
    np.testing.assert_array_equal(mesh.F_edge_sign.get(), np.array([[1, 1, -1]]))
    np.testing.assert_allclose(result, np.array([-2.0], dtype=np.float64))
    assert result.dtype == np.float64


def test_dec_d0_transpose_single_triangle_scatters_to_vertices() -> None:
    _, dec = _build_dec(
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

    one_form = np.array([2, 3, 7], dtype=np.int32)

    result = dec.d0_transpose(one_form)

    np.testing.assert_allclose(
        result,
        np.array([-9.0, -1.0, 10.0], dtype=np.float64),
    )
    assert result.dtype == np.float64


def test_dec_d0_transpose_supports_batched_one_forms() -> None:
    _, dec = _build_dec(
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

    one_forms = np.array([[2.0, 3.0, 7.0], [1.0, -2.0, 4.0]], dtype=np.float64)

    np.testing.assert_allclose(
        dec.d0_transpose(one_forms),
        np.array([[-9.0, -1.0, 10.0], [-5.0, 3.0, 2.0]], dtype=np.float64),
    )


def test_dec_d1_two_triangle_square_uses_each_face_orientation() -> None:
    mesh, dec = _build_dec(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )

    one_form = np.array([1.0, 2.0, 10.0, 4.0, 8.0], dtype=np.float64)

    result = dec.d1(one_form)

    np.testing.assert_array_equal(
        mesh.F_edges.get(),
        np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        mesh.F_edge_sign.get(),
        np.array([[1, 1, -1], [1, 1, -1]], dtype=np.int32),
    )
    np.testing.assert_allclose(result, np.array([-7.0, 6.0], dtype=np.float64))

    shared_edges = set(mesh.F_edges.get()[0]).intersection(mesh.F_edges.get()[1])
    assert len(shared_edges) == 1
    shared_edge = shared_edges.pop()
    first_slot = np.flatnonzero(mesh.F_edges.get()[0] == shared_edge)[0]
    second_slot = np.flatnonzero(mesh.F_edges.get()[1] == shared_edge)[0]
    assert (
        mesh.F_edge_sign.get()[0, first_slot]
        == -mesh.F_edge_sign.get()[1, second_slot]
    )


def test_dec_d1_rejects_wrong_length() -> None:
    _, dec = _build_dec(
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

    with pytest.raises(ValueError, match="length nE=3"):
        dec.d1(np.array([1.0, 2.0], dtype=np.float64))


def test_dec_d1_rejects_non_1d_input() -> None:
    _, dec = _build_dec(
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

    with pytest.raises(ValueError, match="shape \\(nE,\\)"):
        dec.d1(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))
