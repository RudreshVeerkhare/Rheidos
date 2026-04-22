import sys
import types

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


def test_p1_poisson_cholesky_falls_back_when_ordering_kwarg_is_unsupported(
    monkeypatch,
) -> None:
    world = World()

    mesh = world.require(SurfaceMeshModule)
    dec = world.require(DEC, mesh=mesh)
    solver = world.require(P1PoissonSolver, mesh=mesh, dec=dec, mode="cholesky")

    _set_single_triangle_mesh(mesh)
    solver.set_homo_dirichlet_boundary()

    calls: list[dict[str, str]] = []

    def fake_cholesky(matrix, **kwargs):
        del matrix
        calls.append(dict(kwargs))
        if "ordering_method" in kwargs:
            raise TypeError(
                "cholesky() got an unexpected keyword argument 'ordering_method'"
            )
        return lambda rhs: np.zeros_like(rhs, dtype=np.float64)

    fake_pkg = types.ModuleType("sksparse")
    fake_cholmod = types.ModuleType("sksparse.cholmod")
    fake_cholmod.cholesky = fake_cholesky
    fake_pkg.cholmod = fake_cholmod
    monkeypatch.setitem(sys.modules, "sksparse", fake_pkg)
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", fake_cholmod)

    committed = {}

    class DummyCtx:
        def require_inputs(self) -> None:
            return None

        def commit(self, **buffers) -> None:
            committed.update(buffers)

    solver.build_scalar_laplacian_cholesky(DummyCtx())

    assert calls == [{"ordering_method": "default"}, {}]

    solve = committed["solve_cholesky"]
    np.testing.assert_allclose(
        solve(np.array([3.0, 4.0, 5.0], dtype=np.float64)),
        np.zeros(3, dtype=np.float64),
    )


def test_p1_poisson_cholesky_prefers_cho_factor_solver_when_available(
    monkeypatch,
) -> None:
    world = World()

    mesh = world.require(SurfaceMeshModule)
    dec = world.require(DEC, mesh=mesh)
    solver = world.require(P1PoissonSolver, mesh=mesh, dec=dec, mode="cholesky")

    _set_single_triangle_mesh(mesh)
    solver.set_homo_dirichlet_boundary()

    calls = []

    class FakeFactor:
        def solve(self, rhs):
            return np.full(rhs.shape, 7.0, dtype=np.float64)

    def fake_cho_factor(matrix, **kwargs):
        del matrix
        calls.append(("cho_factor", dict(kwargs)))
        return FakeFactor()

    def fake_cholesky(matrix, **kwargs):
        del matrix
        calls.append(("cholesky", dict(kwargs)))
        return (np.eye(1), np.array([0], dtype=np.int64))

    fake_pkg = types.ModuleType("sksparse")
    fake_cholmod = types.ModuleType("sksparse.cholmod")
    fake_cholmod.cholesky = fake_cholesky
    fake_cholmod.cho_factor = fake_cho_factor
    fake_pkg.cholmod = fake_cholmod
    monkeypatch.setitem(sys.modules, "sksparse", fake_pkg)
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", fake_cholmod)

    committed = {}

    class DummyCtx:
        def require_inputs(self) -> None:
            return None

        def commit(self, **buffers) -> None:
            committed.update(buffers)

    solver.build_scalar_laplacian_cholesky(DummyCtx())

    assert calls == [("cho_factor", {"order": "default"})]

    solve = committed["solve_cholesky"]
    np.testing.assert_allclose(
        solve(np.array([3.0, 4.0, 5.0], dtype=np.float64)),
        np.zeros(3, dtype=np.float64),
    )


def test_p1_poisson_recomputes_psi_when_constraints_change_without_rhs_change() -> None:
    world = World()

    mesh = world.require(SurfaceMeshModule)
    dec = world.require(DEC, mesh=mesh)
    solver = world.require(P1PoissonSolver, mesh=mesh, dec=dec, mode="cg")

    mesh.set_mesh(
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

    solver.constrained_idx.set(np.array([0], dtype=np.int32))
    solver.constrained_values.set(np.array([0.0], dtype=np.float64))
    rhs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    solver.rhs.set(rhs)

    psi_one_pin = solver.psi.get().copy()
    np.testing.assert_allclose(psi_one_pin, np.array([0.0, 5.0, 8.0, 7.0]))

    solver.set_homo_dirichlet_boundary()

    psi_boundary_constrained = solver.psi.get().copy()
    np.testing.assert_allclose(
        psi_boundary_constrained[solver.constrained_idx.get()],
        np.zeros(4, dtype=np.float64),
    )
