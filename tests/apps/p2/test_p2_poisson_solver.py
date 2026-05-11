from types import SimpleNamespace

import numpy as np

from rheidos.apps.p2.modules.p2_space.p2_poisson_solver import P2PoissonSolver


def test_p2_poisson_cg_uses_previous_psi_as_initial_guess() -> None:
    solver = object.__new__(P2PoissonSolver)

    rhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    previous_psi = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    captured = {}
    committed = {}

    class FakePsi:
        value = previous_psi

        def peek(self):
            return self.value

    psi = FakePsi()
    solver.rhs = SimpleNamespace(get=lambda: rhs)
    solver.psi = psi

    def solve(b, x0=None):
        captured["b"] = b
        captured["x0"] = x0
        return np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float64)

    solver.solve_cg = SimpleNamespace(get=lambda: solve)

    class DummyCtx:
        def require_inputs(self) -> None:
            return None

        def ensure_outputs(self) -> None:
            psi.value = np.zeros_like(previous_psi)

        def commit(self, **buffers) -> None:
            committed.update(buffers)

    solver.solve_for_psi(DummyCtx())

    assert captured["b"] is rhs
    np.testing.assert_array_equal(captured["x0"], previous_psi)
    np.testing.assert_array_equal(
        committed["psi"], np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float64)
    )
