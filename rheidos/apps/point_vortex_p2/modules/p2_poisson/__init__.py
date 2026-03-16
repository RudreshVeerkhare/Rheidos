from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rheidos.compute import ModuleBase, ProducerBase, ResourceSpec, World

from ..fe_utils import REF_Q_PTS, REF_Q_WTS, bary_to_ref, p2_shape_and_grad_ref
from ..p2_geometry import FaceGeometryModule
from ..p2_space import P2ScalarSpaceModule
from ..point_vortex import PointVortexModule
from ..surface_mesh import SurfaceMeshModule


def _require_scipy():
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except Exception as exc:  # pragma: no cover - exercised only when SciPy absent at runtime
        raise RuntimeError(
            "point_vortex_p2 requires SciPy for P2 sparse assembly/solve. "
            "Install scipy and retry."
        ) from exc
    return sp, spla


def grad_ref_to_surface(J: np.ndarray, Ginv: np.ndarray, dphi_ref: np.ndarray) -> np.ndarray:
    return (J @ Ginv @ dphi_ref.T).T


def assemble_p2_surface_matrices(
    face_to_dofs: np.ndarray,
    ndof: int,
    J: np.ndarray,
    Ginv: np.ndarray,
    sqrt_detG: np.ndarray,
) -> tuple[Any, Any]:
    """Assemble P2 stiffness and mass matrices on embedded triangle mesh."""
    sp, _ = _require_scipy()

    rows: list[int] = []
    cols: list[int] = []
    vals_k: list[float] = []
    vals_m: list[float] = []

    for fid in range(face_to_dofs.shape[0]):
        dofs = face_to_dofs[fid]
        Jf = J[fid]
        Ginvf = Ginv[fid]
        jac = float(sqrt_detG[fid])

        Ke = np.zeros((6, 6), dtype=np.float64)
        Me = np.zeros((6, 6), dtype=np.float64)

        for (xi, eta), w in zip(REF_Q_PTS, REF_Q_WTS):
            phi, dphi_ref = p2_shape_and_grad_ref(float(xi), float(eta))
            dphi = grad_ref_to_surface(Jf, Ginvf, dphi_ref)
            Ke += w * jac * (dphi @ dphi.T)
            Me += w * jac * np.outer(phi, phi)

        for a in range(6):
            ia = int(dofs[a])
            for b in range(6):
                ib = int(dofs[b])
                rows.append(ia)
                cols.append(ib)
                vals_k.append(float(Ke[a, b]))
                vals_m.append(float(Me[a, b]))

    K = sp.coo_matrix((vals_k, (rows, cols)), shape=(ndof, ndof)).tocsr()
    M = sp.coo_matrix((vals_m, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K, M


def scatter_point_vortex_rhs(
    face_ids: np.ndarray,
    bary: np.ndarray,
    gamma: np.ndarray,
    face_to_dofs: np.ndarray,
    ndof: int,
) -> np.ndarray:
    b = np.zeros((ndof,), dtype=np.float64)
    n = int(face_ids.shape[0])

    for i in range(n):
        fid = int(face_ids[i])
        if fid < 0 or fid >= face_to_dofs.shape[0]:
            raise ValueError(f"Invalid face id {fid} for vortex index {i}")

        bc = bary[i]
        xi, eta = bary_to_ref(bc)
        phi, _ = p2_shape_and_grad_ref(xi, eta)
        b[face_to_dofs[fid]] += float(gamma[i]) * phi

    return b


def solve_pinned_poisson(
    K: Any,
    rhs: np.ndarray,
    pin_index: int,
    *,
    factor: Any = None,
    free: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Solve K psi = rhs with pinned DOF psi[pin_index]=0."""
    _, spla = _require_scipy()

    n = int(rhs.shape[0])
    if pin_index < 0 or pin_index >= n:
        raise ValueError(f"pin_index {pin_index} out of range for n={n}")

    if free is None:
        free = np.delete(np.arange(n, dtype=np.int32), pin_index)

    Kff = K[free][:, free]
    if factor is None:
        factor = spla.factorized(Kff.tocsc())

    psi = np.zeros((n,), dtype=np.float64)
    if free.size > 0:
        rhs_f = rhs[free]
        psi[free] = factor(rhs_f)

    residual = K @ psi - rhs
    return psi, residual, factor


@dataclass
class AssembleP2PoissonProducer(ProducerBase):
    face_to_dofs: str
    ndof: str
    J: str
    Ginv: str
    sqrt_detG: str

    K: str
    M: str
    pin_index: str
    free_dofs: str
    solver_factor: str
    k_ones_inf: str

    @property
    def outputs(self):
        return (
            self.K,
            self.M,
            self.pin_index,
            self.free_dofs,
            self.solver_factor,
            self.k_ones_inf,
        )

    def compute(self, reg) -> None:
        face_to_dofs = np.asarray(reg.read(self.face_to_dofs), dtype=np.int32)
        ndof = int(reg.read(self.ndof))
        J = np.asarray(reg.read(self.J), dtype=np.float64)
        Ginv = np.asarray(reg.read(self.Ginv), dtype=np.float64)
        sqrt_detG = np.asarray(reg.read(self.sqrt_detG), dtype=np.float64)

        K, M = assemble_p2_surface_matrices(face_to_dofs, ndof, J, Ginv, sqrt_detG)

        pin = 0
        free = np.delete(np.arange(ndof, dtype=np.int32), pin)

        _, spla = _require_scipy()
        solver_factor = spla.factorized(K[free][:, free].tocsc()) if free.size > 0 else None

        ones = np.ones((ndof,), dtype=np.float64)
        k_ones_inf = float(np.max(np.abs(K @ ones))) if ndof > 0 else 0.0

        reg.commit(self.K, buffer=K)
        reg.commit(self.M, buffer=M)
        reg.commit(self.pin_index, buffer=int(pin))
        reg.commit(self.free_dofs, buffer=free)
        reg.commit(self.solver_factor, buffer=solver_factor)
        reg.commit(self.k_ones_inf, buffer=float(k_ones_inf))


@dataclass
class SolveP2StreamFunctionProducer(ProducerBase):
    face_ids: str
    bary: str
    gamma: str

    face_to_dofs: str
    ndof: str
    K: str
    pin_index: str
    free_dofs: str
    solver_factor: str

    rhs: str
    psi: str
    residual_l2: str
    rhs_circulation: str

    @property
    def outputs(self):
        return (self.rhs, self.psi, self.residual_l2, self.rhs_circulation)

    def compute(self, reg) -> None:
        face_ids = np.asarray(reg.read(self.face_ids), dtype=np.int32)
        bary = np.asarray(reg.read(self.bary), dtype=np.float64)
        gamma = np.asarray(reg.read(self.gamma), dtype=np.float64)

        face_to_dofs = np.asarray(reg.read(self.face_to_dofs), dtype=np.int32)
        ndof = int(reg.read(self.ndof))

        K = reg.read(self.K)
        pin = int(reg.read(self.pin_index))
        free = np.asarray(reg.read(self.free_dofs), dtype=np.int32)
        factor = reg.read(self.solver_factor)

        rhs = scatter_point_vortex_rhs(face_ids, bary, gamma, face_to_dofs, ndof)
        rhs_circulation = float(rhs.sum())

        psi, residual, _ = solve_pinned_poisson(
            K,
            rhs,
            pin,
            factor=factor,
            free=free,
        )

        reg.commit(self.rhs, buffer=rhs)
        reg.commit(self.psi, buffer=psi)
        reg.commit(self.residual_l2, buffer=float(np.linalg.norm(residual)))
        reg.commit(self.rhs_circulation, buffer=rhs_circulation)


class P2PoissonModule(ModuleBase):
    NAME = "P2Poisson"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule, scope=scope)
        self.space = world.require(P2ScalarSpaceModule, scope=scope)
        self.geom = world.require(FaceGeometryModule, scope=scope)
        self.vort = world.require(PointVortexModule, scope=scope)

        self.K = self.resource("K", spec=ResourceSpec(kind="python", allow_none=True))
        self.M = self.resource("M", spec=ResourceSpec(kind="python", allow_none=True))
        self.pin_index = self.resource(
            "pin_index", spec=ResourceSpec(kind="python", dtype=int, allow_none=True)
        )
        self.free_dofs = self.resource(
            "free_dofs", spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True)
        )
        self.solver_factor = self.resource(
            "solver_factor", spec=ResourceSpec(kind="python", allow_none=True)
        )
        self.k_ones_inf = self.resource(
            "k_ones_inf", spec=ResourceSpec(kind="python", dtype=float, allow_none=True)
        )

        self.rhs = self.resource(
            "rhs", spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True)
        )
        self.psi = self.resource(
            "psi", spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True)
        )
        self.residual_l2 = self.resource(
            "residual_l2", spec=ResourceSpec(kind="python", dtype=float, allow_none=True)
        )
        self.rhs_circulation = self.resource(
            "rhs_circulation", spec=ResourceSpec(kind="python", dtype=float, allow_none=True)
        )

        assemble_prod = AssembleP2PoissonProducer(
            face_to_dofs=self.space.face_to_dofs.name,
            ndof=self.space.ndof.name,
            J=self.geom.J.name,
            Ginv=self.geom.Ginv.name,
            sqrt_detG=self.geom.sqrt_detG.name,
            K=self.K.name,
            M=self.M.name,
            pin_index=self.pin_index.name,
            free_dofs=self.free_dofs.name,
            solver_factor=self.solver_factor.name,
            k_ones_inf=self.k_ones_inf.name,
        )
        assemble_deps = (
            self.space.face_to_dofs,
            self.space.ndof,
            self.geom.J,
            self.geom.Ginv,
            self.geom.sqrt_detG,
        )
        self.declare_resource(self.K, deps=assemble_deps, producer=assemble_prod)
        self.declare_resource(self.M, deps=assemble_deps, producer=assemble_prod)
        self.declare_resource(self.pin_index, deps=assemble_deps, producer=assemble_prod)
        self.declare_resource(self.free_dofs, deps=assemble_deps, producer=assemble_prod)
        self.declare_resource(self.solver_factor, deps=assemble_deps, producer=assemble_prod)
        self.declare_resource(self.k_ones_inf, deps=assemble_deps, producer=assemble_prod)

        solve_prod = SolveP2StreamFunctionProducer(
            face_ids=self.vort.face_ids.name,
            bary=self.vort.bary.name,
            gamma=self.vort.gamma.name,
            face_to_dofs=self.space.face_to_dofs.name,
            ndof=self.space.ndof.name,
            K=self.K.name,
            pin_index=self.pin_index.name,
            free_dofs=self.free_dofs.name,
            solver_factor=self.solver_factor.name,
            rhs=self.rhs.name,
            psi=self.psi.name,
            residual_l2=self.residual_l2.name,
            rhs_circulation=self.rhs_circulation.name,
        )
        solve_deps = (
            self.vort.face_ids,
            self.vort.bary,
            self.vort.gamma,
            self.space.face_to_dofs,
            self.space.ndof,
            self.K,
            self.pin_index,
            self.free_dofs,
            self.solver_factor,
        )
        self.declare_resource(self.rhs, deps=solve_deps, producer=solve_prod)
        self.declare_resource(self.psi, deps=solve_deps, producer=solve_prod)
        self.declare_resource(self.residual_l2, deps=solve_deps, producer=solve_prod)
        self.declare_resource(self.rhs_circulation, deps=solve_deps, producer=solve_prod)
