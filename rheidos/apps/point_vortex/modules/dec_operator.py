from rheidos.compute import ModuleBase, World, ResourceSpec

from .surface_mesh import SurfaceMeshModule
from ..producers.dec_metric import DECMetricProducer


import taichi as ti


@ti.kernel
def _apply_d0(E_verts: ti.template(), xV: ti.template(), yE: ti.template()) -> None:
    # yE[e] = xV[j] - xV[i] with canonical E_verts[e]=(i<j)
    for e in range(E_verts.shape[0]):
        i = E_verts[e][0]
        j = E_verts[e][1]
        yE[e] = xV[j] - xV[i]


@ti.kernel
def _apply_d0T(E_verts: ti.template(), xE: ti.template(), yV: ti.template()) -> None:
    for v in yV:
        yV[v] = 0.0
    for e in range(E_verts.shape[0]):
        i = E_verts[e][0]
        j = E_verts[e][1]
        val = xE[e]
        ti.atomic_add(yV[i], -val)
        ti.atomic_add(yV[j], val)


@ti.kernel
def _apply_d1(
    F_edges: ti.template(),
    F_sign: ti.template(),
    xE: ti.template(),
    yF: ti.template(),
) -> None:
    # yF[f] = sum_k sign[f,k] * xE[F_edges[f,k]]
    for f in range(F_edges.shape[0]):
        s = 0.0
        for k in ti.static(range(3)):
            e = F_edges[f][k]
            sign = F_sign[f][k]
            s += ti.cast(sign, ti.f32) * xE[e]
        yF[f] = s


@ti.kernel
def _apply_d1T(
    F_edges: ti.template(),
    F_sign: ti.template(),
    xF: ti.template(),
    yE: ti.template(),
) -> None:
    for e in yE:
        yE[e] = 0.0
    for f in range(F_edges.shape[0]):
        val = xF[f]
        for k in ti.static(range(3)):
            e = F_edges[f][k]
            sign = F_sign[f][k]
            ti.atomic_add(yE[e], ti.cast(sign, ti.f32) * val)


@ti.kernel
def _apply_diag(d: ti.template(), x: ti.template(), y: ti.template()) -> None:
    for i in range(x.shape[0]):
        y[i] = d[i] * x[i]


@ti.kernel
def _apply_laplacian0_fused(
    E_verts: ti.template(), star1: ti.template(), xV: ti.template(), yV: ti.template()
):
    for v in yV:
        yV[v] = 0.0
    for e in range(E_verts.shape[0]):
        i = E_verts[e][0]
        j = E_verts[e][1]
        w = star1[e]
        contrib = w * (xV[j] - xV[i])
        ti.atomic_add(yV[i], -contrib)
        ti.atomic_add(yV[j], contrib)


class SurfaceDECModule(ModuleBase):
    NAME = "DECModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)

        # Metric Dependant Hodge stars (diagonal matrices)
        self.star0 = self.resource(
            "star0",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.f32, allow_none=True),
            doc="Hodge star on 0-forms (barycentric dual area per vertex). Shape: (nV, )",
        )

        self.star1 = self.resource(
            "star1",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.f32, allow_none=True),
            doc="Hodge star on 1-forms (cotan weights per edge). Shape: (nE, )",
        )

        metric_prod = DECMetricProducer(
            V_pos=self.mesh.V_pos,
            F_verts=self.mesh.F_verts,
            F_area=self.mesh.F_area,
            E_verts=self.mesh.E_verts,
            E_opp=self.mesh.E_opp,
            star0=self.star0,
            star1=self.star1,
        )

        # star0 depends on F_area (which depends on V_pos + F_verts) and F_verts
        self.declare_resource(
            self.star0,
            deps=(
                self.mesh.F_area,
                self.mesh.F_verts,
            ),
            producer=metric_prod,
        )

        # star1 depends on V_pos, E_verts, E_opp
        self.declare_resource(
            self.star1,
            deps=(self.mesh.V_pos, self.mesh.E_verts, self.mesh.E_opp),
            producer=metric_prod,
        )

    def apply_laplacian0(
        self,
        xV: ti.Field,  # (nV, )
        yV: ti.Field,  # (nV, )
    ) -> None:
        """Compute yV = d0^T @ star1 @ d0 @ xV (SPD on closed surfaces up to nullspace)."""

        _apply_laplacian0_fused(self.mesh.E_verts.get(), self.star1.get(), xV, yV)
