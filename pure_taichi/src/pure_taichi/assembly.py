from __future__ import annotations

from typing import Any

import numpy as np

from .elements import P2Element, REF_Q_PTS, REF_Q_WTS, bary_to_ref
from .math_utils import grad_ref_to_surface
from .taichi_compat import ensure_taichi_initialized, taichi_is_available


try:  # pragma: no cover - optional runtime dependency
    import scipy.sparse as _sp
except Exception:  # pragma: no cover
    _sp = None


def _require_scipy():
    if _sp is None:
        raise RuntimeError("SciPy is required for sparse assembly/solve in pure_taichi")
    return _sp


def compute_local_element_blocks(
    J: np.ndarray,
    Ginv: np.ndarray,
    sqrt_detG: np.ndarray,
    *,
    element: P2Element | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    element = element or P2Element()
    nF = int(J.shape[0])
    Ke = np.zeros((nF, 6, 6), dtype=np.float64)
    Me = np.zeros((nF, 6, 6), dtype=np.float64)
    ce = np.zeros((nF, 6), dtype=np.float64)

    for fid in range(nF):
        Jf = J[fid]
        Ginvf = Ginv[fid]
        jac = float(sqrt_detG[fid])

        for (xi, eta), w in zip(REF_Q_PTS, REF_Q_WTS):
            phi = element.eval_shape(float(xi), float(eta))
            dphi_ref = element.eval_grad_ref(float(xi), float(eta))
            dphi = grad_ref_to_surface(Jf, Ginvf, dphi_ref)

            Ke[fid] += w * jac * (dphi @ dphi.T)
            Me[fid] += w * jac * np.outer(phi, phi)
            ce[fid] += w * jac * phi

    return Ke, Me, ce


def _assemble_from_triplets(
    ndof: int,
    rows: np.ndarray,
    cols: np.ndarray,
    vals_k: np.ndarray,
    vals_m: np.ndarray,
    c: np.ndarray,
):
    sp = _require_scipy()
    K = sp.coo_matrix((vals_k, (rows, cols)), shape=(ndof, ndof)).tocsr()
    M = sp.coo_matrix((vals_m, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K, M, c


def assemble_p2_surface_matrices_numpy(
    face_to_dofs: np.ndarray,
    ndof: int,
    Ke: np.ndarray,
    Me: np.ndarray,
    ce: np.ndarray,
):
    nF = int(face_to_dofs.shape[0])
    rows = np.empty((nF * 36,), dtype=np.int32)
    cols = np.empty((nF * 36,), dtype=np.int32)
    vals_k = np.empty((nF * 36,), dtype=np.float64)
    vals_m = np.empty((nF * 36,), dtype=np.float64)
    c = np.zeros((ndof,), dtype=np.float64)

    k = 0
    for fid in range(nF):
        dofs = face_to_dofs[fid]
        c[dofs] += ce[fid]
        for a in range(6):
            ia = int(dofs[a])
            for b in range(6):
                ib = int(dofs[b])
                rows[k] = ia
                cols[k] = ib
                vals_k[k] = float(Ke[fid, a, b])
                vals_m[k] = float(Me[fid, a, b])
                k += 1

    return _assemble_from_triplets(ndof, rows, cols, vals_k, vals_m, c)


def _scatter_triplets_with_taichi(
    face_to_dofs: np.ndarray,
    ndof: int,
    Ke: np.ndarray,
    Me: np.ndarray,
    ce: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ti = ensure_taichi_initialized("cpu")
    nF = int(face_to_dofs.shape[0])
    ntrip = nF * 36

    face_to_dofs_f = ti.field(dtype=ti.i32, shape=(nF, 6))
    Ke_f = ti.field(dtype=ti.f64, shape=(nF, 6, 6))
    Me_f = ti.field(dtype=ti.f64, shape=(nF, 6, 6))
    ce_f = ti.field(dtype=ti.f64, shape=(nF, 6))

    rows_f = ti.field(dtype=ti.i32, shape=ntrip)
    cols_f = ti.field(dtype=ti.i32, shape=ntrip)
    vals_k_f = ti.field(dtype=ti.f64, shape=ntrip)
    vals_m_f = ti.field(dtype=ti.f64, shape=ntrip)
    c_f = ti.field(dtype=ti.f64, shape=ndof)

    face_to_dofs_f.from_numpy(np.asarray(face_to_dofs, dtype=np.int32))
    Ke_f.from_numpy(np.asarray(Ke, dtype=np.float64))
    Me_f.from_numpy(np.asarray(Me, dtype=np.float64))
    ce_f.from_numpy(np.asarray(ce, dtype=np.float64))

    @ti.kernel
    def scatter() -> None:
        for i in range(ndof):
            c_f[i] = 0.0
        for fid in range(nF):
            for a in ti.static(range(6)):
                ia = face_to_dofs_f[fid, a]
                ti.atomic_add(c_f[ia], ce_f[fid, a])
                for b in ti.static(range(6)):
                    ib = face_to_dofs_f[fid, b]
                    idx = fid * 36 + a * 6 + b
                    rows_f[idx] = ia
                    cols_f[idx] = ib
                    vals_k_f[idx] = Ke_f[fid, a, b]
                    vals_m_f[idx] = Me_f[fid, a, b]

    scatter()

    return (
        rows_f.to_numpy().astype(np.int32, copy=False),
        cols_f.to_numpy().astype(np.int32, copy=False),
        vals_k_f.to_numpy().astype(np.float64, copy=False),
        vals_m_f.to_numpy().astype(np.float64, copy=False),
        c_f.to_numpy().astype(np.float64, copy=False),
    )


def assemble_p2_surface_matrices(
    face_to_dofs: np.ndarray,
    ndof: int,
    J: np.ndarray,
    Ginv: np.ndarray,
    sqrt_detG: np.ndarray,
    *,
    prefer_taichi: bool = True,
) -> tuple[Any, Any, np.ndarray]:
    Ke, Me, ce = compute_local_element_blocks(J, Ginv, sqrt_detG)

    use_taichi = bool(prefer_taichi and taichi_is_available())
    if use_taichi:
        try:
            rows, cols, vals_k, vals_m, c = _scatter_triplets_with_taichi(
                face_to_dofs, ndof, Ke, Me, ce
            )
            return _assemble_from_triplets(ndof, rows, cols, vals_k, vals_m, c)
        except Exception:
            pass

    return assemble_p2_surface_matrices_numpy(face_to_dofs, ndof, Ke, Me, ce)


def scatter_point_vortex_rhs_numpy(
    face_ids: np.ndarray,
    bary: np.ndarray,
    gamma: np.ndarray,
    face_to_dofs: np.ndarray,
    ndof: int,
    *,
    element: P2Element | None = None,
) -> np.ndarray:
    element = element or P2Element()
    b = np.zeros((ndof,), dtype=np.float64)

    for i in range(int(face_ids.shape[0])):
        fid = int(face_ids[i])
        xi, eta = bary_to_ref(bary[i])
        phi = element.eval_shape(xi, eta)
        b[face_to_dofs[fid]] += float(gamma[i]) * phi

    return b


def scatter_point_vortex_rhs_taichi(
    face_ids: np.ndarray,
    bary: np.ndarray,
    gamma: np.ndarray,
    face_to_dofs: np.ndarray,
    ndof: int,
) -> np.ndarray:
    ti = ensure_taichi_initialized("cpu")

    n_v = int(face_ids.shape[0])
    nF = int(face_to_dofs.shape[0])

    face_ids_f = ti.field(dtype=ti.i32, shape=n_v)
    bary_f = ti.Vector.field(3, dtype=ti.f64, shape=n_v)
    gamma_f = ti.field(dtype=ti.f64, shape=n_v)
    face_to_dofs_f = ti.field(dtype=ti.i32, shape=(nF, 6))
    rhs_f = ti.field(dtype=ti.f64, shape=ndof)

    face_ids_f.from_numpy(np.asarray(face_ids, dtype=np.int32))
    bary_f.from_numpy(np.asarray(bary, dtype=np.float64))
    gamma_f.from_numpy(np.asarray(gamma, dtype=np.float64))
    face_to_dofs_f.from_numpy(np.asarray(face_to_dofs, dtype=np.int32))

    @ti.func
    def p2_phi(l0: ti.f64, l1: ti.f64, l2: ti.f64, idx: ti.i32) -> ti.f64:
        out = ti.cast(0.0, ti.f64)
        if idx == 0:
            out = l0 * (2.0 * l0 - 1.0)
        elif idx == 1:
            out = l1 * (2.0 * l1 - 1.0)
        elif idx == 2:
            out = l2 * (2.0 * l2 - 1.0)
        elif idx == 3:
            out = 4.0 * l0 * l1
        elif idx == 4:
            out = 4.0 * l1 * l2
        else:
            out = 4.0 * l2 * l0
        return out

    @ti.kernel
    def scatter() -> None:
        for i in range(ndof):
            rhs_f[i] = 0.0

        for i in range(n_v):
            fid = face_ids_f[i]
            if 0 <= fid < nF:
                bc = bary_f[i]
                l0 = bc[0]
                l1 = bc[1]
                l2 = bc[2]
                g = gamma_f[i]
                for a in ti.static(range(6)):
                    dof = face_to_dofs_f[fid, a]
                    ti.atomic_add(rhs_f[dof], g * p2_phi(l0, l1, l2, a))

    scatter()
    return rhs_f.to_numpy().astype(np.float64, copy=False)


def scatter_point_vortex_rhs(
    face_ids: np.ndarray,
    bary: np.ndarray,
    gamma: np.ndarray,
    face_to_dofs: np.ndarray,
    ndof: int,
    *,
    prefer_taichi: bool = True,
) -> np.ndarray:
    use_taichi = bool(prefer_taichi and taichi_is_available())
    if use_taichi:
        try:
            return scatter_point_vortex_rhs_taichi(face_ids, bary, gamma, face_to_dofs, ndof)
        except Exception:
            pass
    return scatter_point_vortex_rhs_numpy(face_ids, bary, gamma, face_to_dofs, ndof)


def remove_mean_from_rhs(rhs: np.ndarray, c: np.ndarray) -> np.ndarray:
    total_area = float(np.sum(c))
    if total_area <= 1e-20:
        return rhs.copy()
    rho0 = float(np.sum(rhs)) / total_area
    return rhs - rho0 * c
