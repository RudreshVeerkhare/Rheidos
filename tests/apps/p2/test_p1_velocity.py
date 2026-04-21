from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p1_space.p1_velocity import P1VelocityFieldModule
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


def _build_velocity(world: World) -> tuple[SurfaceMeshModule, P1VelocityFieldModule]:
    mesh = world.require(SurfaceMeshModule)
    vortex = world.require(PointVortexModule)
    dec = world.require(DEC, mesh=mesh)
    poisson = world.require(
        P1PoissonSolver,
        mesh=mesh,
        dec=dec,
        declare_rhs=False,
    )
    stream = world.require(
        P1StreamFunction,
        mesh=mesh,
        point_vortex=vortex,
        poisson=poisson,
    )
    velocity = world.require(
        P1VelocityFieldModule,
        mesh=mesh,
        dec=dec,
        stream=stream,
    )
    return mesh, velocity


def _prime_stream_resources(velocity: P1VelocityFieldModule) -> None:
    velocity.stream.set_homo_dirichlet_boundary()
    velocity.stream.omega.set(np.zeros(3, dtype=np.float64))
    velocity.stream.solve_cg.get()


def test_p1_velocity_per_face_producer_builds_facewise_velocity() -> None:
    world = World()

    mesh, velocity = _build_velocity(world)

    _set_single_triangle_mesh(mesh)
    _prime_stream_resources(velocity)
    velocity.stream.psi.set(np.array([1.2, -0.7, 0.9], dtype=np.float64))

    result = velocity.vel_per_face.get()[0]

    grad_l1 = np.array([-1.0, -1.0, 0.0], dtype=np.float64)
    grad_l2 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    grad_l3 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    grad_psi = 1.2 * grad_l1 + (-0.7) * grad_l2 + 0.9 * grad_l3
    expected = np.cross(normal, grad_psi)

    np.testing.assert_allclose(result, expected, atol=1e-7)
    np.testing.assert_allclose(np.dot(result, normal), 0.0)


@pytest.mark.parametrize("use_batched_arrays", [False, True])
def test_p1_velocity_interpolate_reuses_cached_facewise_velocity(
    use_batched_arrays: bool,
) -> None:
    world = World()

    mesh, velocity = _build_velocity(world)

    _set_single_triangle_mesh(mesh)
    _prime_stream_resources(velocity)
    velocity.stream.psi.set(np.zeros(3, dtype=np.float64))
    velocity.mesh.F_normal.get()
    velocity.mesh.grad_bary.get()
    velocity.vel_per_face.set(np.array([[2.5, -1.5, 0.0]], dtype=np.float64))
    velocity.stream.psi = SimpleNamespace(
        get=lambda: (_ for _ in ()).throw(AssertionError("psi should not be read"))
    )

    bary = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.7, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    if use_batched_arrays:
        probes = (np.array([0, 0], dtype=np.int32), bary)
    else:
        probes = [(0, bary[0]), (0, bary[1])]

    result = velocity.interpolate(probes, smooth=False)

    np.testing.assert_allclose(
        result,
        np.array(
            [
                [2.5, -1.5, 0.0],
                [2.5, -1.5, 0.0],
            ],
            dtype=np.float64,
        ),
    )
