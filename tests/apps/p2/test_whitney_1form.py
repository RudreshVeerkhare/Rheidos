from __future__ import annotations

import numpy as np
import pytest

from rheidos.apps.p2.modules.p1_space.whitney_1form import Whitney1FormInterpolator
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import World


def _build_single_triangle() -> tuple[SurfaceMeshModule, Whitney1FormInterpolator]:
    world = World()
    mesh = world.require(SurfaceMeshModule)
    interpolator = world.require(Whitney1FormInterpolator, mesh=mesh)
    mesh.set_mesh(
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
    return mesh, interpolator


def test_whitney_1form_single_triangle_matches_formula() -> None:
    mesh, interpolator = _build_single_triangle()
    one_form = np.array([2.0, 3.0, 5.0], dtype=np.float64)
    bary = np.array([[0.2, 0.3, 0.5], [0.7, 0.1, 0.2]], dtype=np.float64)

    result = interpolator.interpolate(one_form, (np.array([0, 0]), bary))

    local_values = one_form[mesh.F_edges.get()[0]] * mesh.F_edge_sign.get()[0]
    grad = mesh.grad_bary.get()[0]
    expected = []
    for l0, l1, l2 in bary:
        a01, a12, a20 = local_values
        g0, g1, g2 = grad
        expected.append(
            a01 * (l0 * g1 - l1 * g0)
            + a12 * (l1 * g2 - l2 * g1)
            + a20 * (l2 * g0 - l0 * g2)
        )

    np.testing.assert_allclose(result, np.asarray(expected), atol=1e-12)
    np.testing.assert_allclose(
        np.einsum("ij,j->i", result, mesh.F_normal.get()[0]),
        np.zeros(result.shape[0]),
        atol=1e-12,
    )


def test_whitney_1form_sign_corrects_reversed_local_edge() -> None:
    mesh, interpolator = _build_single_triangle()
    one_form = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    bary = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)

    result = interpolator.interpolate(one_form, (np.array([0]), bary))[0]

    assert mesh.E_verts.get()[2].tolist() == [0, 2]
    assert mesh.F_edge_sign.get()[0, 2] == -1
    grad = mesh.grad_bary.get()[0]
    expected = -5.0 * (bary[0, 2] * grad[0] - bary[0, 0] * grad[2])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_whitney_1form_empty_probes_returns_empty_vectors() -> None:
    _, interpolator = _build_single_triangle()

    result = interpolator.interpolate(np.zeros(3), [])

    assert result.shape == (0, 3)
    assert result.dtype == np.float64


def test_whitney_1form_rejects_invalid_edge_count() -> None:
    _, interpolator = _build_single_triangle()

    with pytest.raises(ValueError, match="length nE=3"):
        interpolator.interpolate(np.zeros(2), [])
