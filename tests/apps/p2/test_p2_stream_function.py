from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rheidos.apps.p2.modules.p2_space.p2_stream_function import P2StreamFunction
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import World


def _set_single_triangle_mesh(mesh: SurfaceMeshModule) -> None:
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


def test_p2_stream_interpolate_uses_regularized_coefficients() -> None:
    world = World()

    mesh = world.require(SurfaceMeshModule)
    stream = world.require(P2StreamFunction)

    _set_single_triangle_mesh(mesh)

    regularized = np.array([1.2, -0.7, 0.9, 0.3, -1.1, 0.8], dtype=np.float64)
    raw = np.array([-0.4, 0.5, 1.1, -0.2, 0.6, -1.3], dtype=np.float64)
    stream.psi = SimpleNamespace(get=lambda: regularized)
    stream.poisson.psi = SimpleNamespace(get=lambda: raw)

    bary = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    result = stream.interpolate([(0, bary)])

    basis = np.array(stream.p2_elements.basis_from_bary(*bary), dtype=np.float64)
    expected = np.dot(basis, regularized)
    raw_value = np.dot(basis, raw)

    np.testing.assert_allclose(result, np.array([expected]))
    assert not np.isclose(result[0], raw_value)


def test_p2_stream_regularization_recomputes_when_eps_changes() -> None:
    world = World()

    mesh = world.require(SurfaceMeshModule)
    vortex = world.require(PointVortexModule)
    stream = world.require(P2StreamFunction)

    _set_single_triangle_mesh(mesh)
    vortex.set_vortex(
        np.array([0], dtype=np.int32),
        np.array([[0.2, 0.3, 0.5]], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.array([[0.3, 0.5, 0.0]], dtype=np.float32),
    )

    stream.constrained_idx.set(np.array([0], dtype=np.int32))
    stream.constrained_values.set(np.array([0], dtype=np.float32))

    stream.eps.set(1e-4)
    psi_small_eps = stream.psi.get().copy()

    stream.eps.set(1e-1)
    psi_large_eps = stream.psi.get().copy()

    assert not np.allclose(psi_small_eps, psi_large_eps)
