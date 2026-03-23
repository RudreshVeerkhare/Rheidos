from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rheidos.apps.p2.modules.p2_space.p2_velocity import P2VelocityField
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import World


def test_p2_velocity_interpolate_uses_row_stored_barycentric_gradients() -> None:
    world = World()

    mesh = world.require(SurfaceMeshModule)
    velocity = world.require(P2VelocityField)

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

    coeffs = np.array([1.2, -0.7, 0.9, 0.3, -1.1, 0.8], dtype=np.float64)
    velocity.stream.psi = SimpleNamespace(get=lambda: coeffs)

    bary = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    result = velocity.interpolate([(0, bary)])[0]

    b1, b2, b3 = bary
    c1, c2, c3, c4, c5, c6 = coeffs

    grad_l1 = np.array([-1.0, -1.0, 0.0], dtype=np.float64)
    grad_l2 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    grad_l3 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    grad_psi = (
        ((4 * b1 - 1) * c1 + 4 * b2 * c4 + 4 * b3 * c6) * grad_l1
        + (4 * b1 * c4 + (4 * b2 - 1) * c2 + 4 * b3 * c5) * grad_l2
        + (4 * b1 * c6 + 4 * b2 * c5 + (4 * b3 - 1) * c3) * grad_l3
    )
    expected = np.cross(normal, grad_psi)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(np.dot(result, normal), 0.0)
