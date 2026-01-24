from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import taichi as ti

from rheidos.compute import WiredProducer, ResourceRef, out_field
from rheidos.compute.registry import Registry


@dataclass
class SolvePoissonDirichletScipyCGIO:
    # Topology + weights
    E_verts: ResourceRef[Any]  # (nE, vec2i)
    w: ResourceRef[Any]  # (nE,) f32

    # Dirichlet constraints
    mask: ResourceRef[Any]  # (nV,) i32 (1 = constrained)  (may be None)
    value: ResourceRef[Any]  # (nV,) f32                   (may be None)

    # Optional RHS (may be None => 0)
    rhs: ResourceRef[Any]  # (nV,) f32 (may be None)

    # Output
    u: ResourceRef[Any] = out_field()  # (nV,) f32


@ti.data_oriented
class SolvePoissonDirichletScipyCG(WiredProducer[SolvePoissonDirichletScipyCGIO]):
    """
    CPU reference Poisson solve using SciPy sparse + CG, with Dirichlet constraints.

    Builds a symmetric vertex Laplacian from (E_verts, w):
      L_ii += w, L_jj += w, L_ij -= w, L_ji -= w

    Enforces Dirichlet by elimination:
      L_ff u_f = rhs_f - L_fc u_c, with u_c = value_c

    Notes:
    - On a closed surface, if mask has no constrained vertex, system is singular.
    - CG assumes SPD; if weights yield an indefinite operator (e.g., negative cotans),
      CG may fail. In that case swap to MINRES/direct solve for debugging.
    """

    max_iter = 800
    tol = 1e-6
    use_jacobi = True
    always_rebuild_topology = True

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_iter = int(self.max_iter)
        self.tol = float(self.tol)
        self.use_jacobi = bool(self.use_jacobi)
        self.always_rebuild_topology = bool(self.always_rebuild_topology)

        # Cache Laplacian between calls (by shape). Good for stepping sims.
        self._cached_key: Optional[tuple[int, int]] = None
        self._cached_L = None  # scipy.sparse.csr_matrix

    @staticmethod
    def _to_numpy(x: Any) -> Optional[np.ndarray]:
        if x is None:
            return None
        if hasattr(x, "to_numpy"):
            return x.to_numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

    @staticmethod
    def _build_laplacian(nV: int, E: np.ndarray, w: np.ndarray):
        import scipy.sparse as sp

        E = np.asarray(E)
        if E.ndim != 2 or E.shape[1] != 2:
            # If E_verts is a vec2 field, to_numpy() should already be (nE,2).
            raise RuntimeError(f"E_verts expected shape (nE,2), got {E.shape}")

        i = E[:, 0].astype(np.int64, copy=False)
        j = E[:, 1].astype(np.int64, copy=False)
        ww = np.asarray(w, dtype=np.float64).reshape(-1)

        rows = np.concatenate([i, j, i, j])
        cols = np.concatenate([i, j, j, i])
        data = np.concatenate([ww, ww, -ww, -ww])

        L = sp.coo_matrix((data, (rows, cols)), shape=(nV, nV)).tocsr()
        return L

    def compute(self, reg: Registry) -> None:
        # --- inputs ---
        inputs = self.require_inputs(allow_none=("mask", "value", "rhs"))
        E_f = inputs["E_verts"].get()
        w_f = inputs["w"].get()
        mask_f = inputs["mask"].peek()
        value_f = inputs["value"].peek()
        rhs_f = inputs["rhs"].peek()

        E = self._to_numpy(E_f)
        w = self._to_numpy(w_f)

        mask = self._to_numpy(mask_f) if mask_f is not None else None
        value = self._to_numpy(value_f) if value_f is not None else None
        rhs = self._to_numpy(rhs_f) if rhs_f is not None else None

        if E is None or w is None:
            raise RuntimeError(
                "SolvePoissonDirichletScipyCG: E_verts / w could not be converted"
            )

        # Infer nV from value/mask/rhs; prefer value, then mask, then rhs.
        nV = None
        for arr in (value, mask, rhs):
            if arr is not None:
                nV = int(np.asarray(arr).shape[0])
                break
        if nV is None:
            raise RuntimeError(
                "SolvePoissonDirichletScipyCG: cannot infer nV (value/mask/rhs are all None)."
            )

        if mask is None:
            mask = np.zeros((nV,), dtype=np.int32)
        else:
            mask = np.asarray(mask, dtype=np.int32).reshape(-1)

        if rhs is None:
            rhs = np.zeros((nV,), dtype=np.float64)
        else:
            rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)

        if value is None:
            # If there are constraints but no values, that’s ambiguous — fail loudly.
            if np.any(mask != 0):
                raise RuntimeError(
                    "SolvePoissonDirichletScipyCG: mask has constraints but value is None."
                )
            value = np.zeros((nV,), dtype=np.float64)
        else:
            value = np.asarray(value, dtype=np.float64).reshape(-1)

        constrained = np.nonzero(mask != 0)[0]
        free = np.nonzero(mask == 0)[0]

        if constrained.size == 0:
            raise RuntimeError(
                "SolvePoissonDirichletScipyCG: no Dirichlet constraints found (mask all zeros). "
                "Closed-surface Laplacian is singular; pin at least one vertex."
            )

        # --- build/cached Laplacian ---
        key = (nV, int(np.asarray(E).shape[0]))
        if (
            self.always_rebuild_topology
            or (self._cached_key != key)
            or (self._cached_L is None)
        ):
            L = self._build_laplacian(nV, E, w)
            self._cached_L = L
            self._cached_key = key
        else:
            L = self._cached_L

        # --- eliminate Dirichlet ---
        uc = value[constrained]
        A = L[free, :][:, free]
        L_fc = L[free, :][:, constrained]
        b = rhs[free] - (L_fc @ uc)

        # --- solve with SciPy CG ---
        import scipy.sparse.linalg as spla
        from scipy.sparse.linalg import LinearOperator

        M = None
        if self.use_jacobi:
            d = A.diagonal().astype(np.float64, copy=False)
            invd = np.zeros_like(d)
            nz = np.abs(d) > 1e-30
            invd[nz] = 1.0 / d[nz]

            def _mv(x):
                return invd * x

            M = LinearOperator(shape=A.shape, matvec=_mv, dtype=np.float64)

        # SciPy changed cg signature in newer versions (tol vs rtol/atol).
        try:
            x, info = spla.cg(A, b, rtol=self.tol, atol=0.0, maxiter=self.max_iter, M=M)
        except TypeError:
            x, info = spla.cg(A, b, tol=self.tol, atol=0.0, maxiter=self.max_iter, M=M)

        if info != 0:
            raise RuntimeError(
                f"SolvePoissonDirichletScipyCG: cg did not converge (info={info}). "
                "Likely non-SPD operator or too strict tol/max_iter."
            )

        u_np = np.zeros((nV,), dtype=np.float64)
        u_np[constrained] = uc
        u_np[free] = x

        # --- output (ensure + fill + commit) ---
        u_out = self.ensure_outputs(reg)["u"].peek()

        u_out.from_numpy(u_np.astype(np.float32))
        self.io.u.commit()
