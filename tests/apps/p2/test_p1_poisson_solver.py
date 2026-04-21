import numpy as np
from types import SimpleNamespace

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
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


def test_p1_poisson_interpolate_accepts_faceid_and_bary_arrays() -> None:
    world = World()

    mesh = world.require(SurfaceMeshModule)
    dec = world.require(DEC, mesh=mesh)
    solver = world.require(P1PoissonSolver, mesh=mesh, dec=dec)

    _set_single_triangle_mesh(mesh)
    solver.psi = SimpleNamespace(get=lambda: np.array([1.0, -0.5, 2.0], dtype=np.float64))

    faceids = np.array([0], dtype=np.int32)
    bary = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)

    result = solver.interpolate((faceids, bary))

    np.testing.assert_allclose(result, np.array([1.05], dtype=np.float64))


def test_p1_poisson_sets_zero_dirichlet_boundary_from_dec_boundary_mask() -> None:
    world = World()

    mesh = world.require(SurfaceMeshModule)
    dec = world.require(DEC, mesh=mesh)
    solver = world.require(P1PoissonSolver, mesh=mesh, dec=dec)

    _set_single_triangle_mesh(mesh)

    solver.set_homo_dirichlet_boundary()

    np.testing.assert_array_equal(
        solver.constrained_idx.get(),
        np.array([0, 1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        solver.constrained_values.get(),
        np.zeros(3, dtype=np.float64),
    )
