import numpy as np

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
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


def test_p1_stream_function_owns_rhs_from_point_vortices() -> None:
    world = World()

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

    _set_single_triangle_mesh(mesh)
    vortex.set_vortex(
        np.array([0], dtype=np.int32),
        np.array([[0.2, 0.3, 0.5]], dtype=np.float64),
        np.array([2.0], dtype=np.float64),
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    )

    np.testing.assert_allclose(
        stream.omega.get(),
        np.array([0.4, 0.6, 1.0], dtype=np.float64),
    )


def test_p1_stream_function_exposes_homogeneous_dirichlet_boundary_helper() -> None:
    world = World()

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

    _set_single_triangle_mesh(mesh)

    stream.set_homo_dirichlet_boundary()

    np.testing.assert_array_equal(
        stream.constrained_idx.get(),
        np.array([0, 1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        stream.constrained_values.get(),
        np.zeros(3, dtype=np.float64),
    )
