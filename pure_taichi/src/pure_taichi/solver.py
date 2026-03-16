from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .assembly import assemble_p2_surface_matrices, remove_mean_from_rhs
from .model import PoissonSystem
from .taichi_compat import ensure_taichi_initialized, taichi_is_available


try:  # pragma: no cover - optional runtime dependency
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:  # pragma: no cover
    sp = None
    spla = None


def _require_scipy() -> tuple[Any, Any]:
    if sp is None or spla is None:
        raise RuntimeError("SciPy is required for sparse solve in pure_taichi")
    return sp, spla


def _build_taichi_reduced_solver(Kff_csr: Any) -> tuple[Any, Any]:
    ti = ensure_taichi_initialized("cpu")

    coo = Kff_csr.tocoo()
    n = int(Kff_csr.shape[0])
    nnz = int(coo.nnz)

    rows_f = ti.field(dtype=ti.i32, shape=nnz)
    cols_f = ti.field(dtype=ti.i32, shape=nnz)
    vals_f = ti.field(dtype=ti.f64, shape=nnz)

    rows_f.from_numpy(coo.row.astype(np.int32, copy=False))
    cols_f.from_numpy(coo.col.astype(np.int32, copy=False))
    vals_f.from_numpy(coo.data.astype(np.float64, copy=False))

    builder = ti.linalg.SparseMatrixBuilder(
        n,
        n,
        max_num_triplets=max(nnz, 1),
        dtype=ti.f64,
    )

    @ti.kernel
    def fill(B: ti.types.sparse_matrix_builder()) -> None:
        for i in range(nnz):
            B[rows_f[i], cols_f[i]] += vals_f[i]

    fill(builder)
    A = builder.build()

    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(A)
    solver.factorize(A)
    return A, solver


def build_poisson_system(
    face_to_dofs: np.ndarray,
    ndof: int,
    J: np.ndarray,
    Ginv: np.ndarray,
    sqrt_detG: np.ndarray,
    *,
    pin_index: int = 0,
    prefer_taichi: bool = True,
) -> PoissonSystem:
    K, M, c = assemble_p2_surface_matrices(
        face_to_dofs,
        ndof,
        J,
        Ginv,
        sqrt_detG,
        prefer_taichi=prefer_taichi,
    )

    if pin_index < 0 or pin_index >= ndof:
        raise ValueError(f"pin_index {pin_index} out of bounds for ndof={ndof}")

    free = np.delete(np.arange(ndof, dtype=np.int32), pin_index)
    ones = np.ones((ndof,), dtype=np.float64)
    k_ones_inf = float(np.max(np.abs(K @ ones))) if ndof > 0 else 0.0

    system = PoissonSystem(
        K=K,
        M=M,
        c=c,
        k_ones_inf=k_ones_inf,
        pin_index=int(pin_index),
        free_dofs=free,
    )

    _, _ = _require_scipy()
    Kff = K[free][:, free]

    try:
        system.scipy_factor = spla.factorized(Kff.tocsc()) if free.size > 0 else None
    except Exception:
        system.scipy_factor = None

    if prefer_taichi and taichi_is_available() and free.size > 0:
        try:
            A, solver = _build_taichi_reduced_solver(Kff)
            system.taichi_matrix = A
            system.taichi_solver = solver
        except Exception:
            system.taichi_matrix = None
            system.taichi_solver = None

    return system


def _solve_reduced_scipy(system: PoissonSystem, rhs0: np.ndarray) -> np.ndarray:
    _, _ = _require_scipy()
    free = system.free_dofs
    pin = int(system.pin_index)
    n = int(rhs0.shape[0])

    psi = np.zeros((n,), dtype=np.float64)
    if free.size == 0:
        return psi

    rhs_f = rhs0[free]

    if system.scipy_factor is not None:
        psi_f = system.scipy_factor(rhs_f)
    else:
        Kff = system.K[free][:, free]
        psi_f = spla.spsolve(Kff.tocsc(), rhs_f)

    psi[free] = np.asarray(psi_f, dtype=np.float64)
    psi[pin] = 0.0
    return psi


def _solve_reduced_taichi(system: PoissonSystem, rhs0: np.ndarray) -> np.ndarray:
    if system.taichi_solver is None:
        raise RuntimeError("Taichi solver not available for this system")

    free = system.free_dofs
    pin = int(system.pin_index)

    rhs_f = np.asarray(rhs0[free], dtype=np.float64)
    psi_f = system.taichi_solver.solve(rhs_f)

    psi = np.zeros((rhs0.shape[0],), dtype=np.float64)
    psi[free] = np.asarray(psi_f, dtype=np.float64)
    psi[pin] = 0.0
    return psi


def _solve_scipy_constrained(system: PoissonSystem, rhs0: np.ndarray) -> np.ndarray:
    sp_mod, spla_mod = _require_scipy()

    c = np.asarray(system.c, dtype=np.float64)
    c_col = sp_mod.csr_matrix(c.reshape(-1, 1))
    zero = sp_mod.csr_matrix((1, 1))

    A = sp_mod.bmat(
        [
            [system.K, c_col],
            [c_col.T, zero],
        ],
        format="csr",
    )

    rhs_aug = np.concatenate([rhs0, [0.0]])
    sol = spla_mod.spsolve(A.tocsc(), rhs_aug)
    return np.asarray(sol[:-1], dtype=np.float64)


def solve_stream_function(
    system: PoissonSystem,
    rhs: np.ndarray,
    *,
    backend: str = "auto",
) -> tuple[np.ndarray, float, str, np.ndarray]:
    rhs = np.asarray(rhs, dtype=np.float64)
    rhs0 = remove_mean_from_rhs(rhs, np.asarray(system.c, dtype=np.float64))

    backend_used = backend

    if backend == "scipy_constrained":
        psi = _solve_scipy_constrained(system, rhs0)
    elif backend == "scipy":
        psi = _solve_reduced_scipy(system, rhs0)
    elif backend == "taichi":
        psi = _solve_reduced_taichi(system, rhs0)
    elif backend == "auto":
        if system.taichi_solver is not None:
            try:
                psi = _solve_reduced_taichi(system, rhs0)
                backend_used = "taichi"
            except Exception:
                psi = _solve_reduced_scipy(system, rhs0)
                backend_used = "scipy"
        else:
            psi = _solve_reduced_scipy(system, rhs0)
            backend_used = "scipy"
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Expected auto|taichi|scipy|scipy_constrained"
        )

    c = np.asarray(system.c, dtype=np.float64)
    total_area = float(c.sum())
    if total_area > 1e-20:
        mean_val = float(np.dot(c, psi) / total_area)
        psi = psi - mean_val

    residual = np.asarray(system.K @ psi - rhs0, dtype=np.float64)
    residual_l2 = float(np.linalg.norm(residual))
    return psi, residual_l2, backend_used, rhs0
