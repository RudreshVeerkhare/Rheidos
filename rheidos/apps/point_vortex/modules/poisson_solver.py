from rheidos.compute import (
    ModuleBase,
    World,
    ResourceSpec,
    ResourceRef,
    ShapeFn,
    Registry,
    Shape,
    shape_of,
)
from .surface_mesh import SurfaceMeshModule
from .dec_operator import SurfaceDECModule

from ..producers.poisson_dirichlet import SolvePoissonDirichlet
from ..producers.scipy_cg import SolvePoissonDirichletScipyCG  # NEW

from typing import Optional, Any
import taichi as ti


class PoissonSolverModule(ModuleBase):
    NAME = "poisson"

    def __init__(
        self, world: World, *, scope: str = "", use_scipy_cg: bool = False
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(SurfaceDECModule)

        self.constraint_mask = self.resource(
            "constraint_mask",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                shape_fn=shape_of(self.mesh.V_pos),
                allow_none=True,
            ),
            doc="i32 mask (1=Dirichlet)",
            declare=True,
            buffer=None,
            description="Dirichlet mask",
        )
        self.constraint_value = self.resource(
            "constraint_value",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.mesh.V_pos),
                allow_none=True,
            ),
            doc="f32 Dirichlet values",
            declare=True,
            buffer=None,
            description="Dirichlet values",
        )

        self.rhs = self.resource(
            "rhs",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.mesh.V_pos),
                allow_none=True,
            ),
            doc="f32 rhs on vertices; if None => rhs=0",
            declare=True,
            buffer=None,
            description="Poisson RHS",
        )

        self.u = self.resource(
            "u",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.mesh.V_pos),
                allow_none=True,
            ),
            doc="solution scalar u",
            declare=False,
        )

        if use_scipy_cg:
            solver = SolvePoissonDirichletScipyCG(
                E_verts=self.mesh.E_verts,
                w=self.dec.star1,
                mask=self.constraint_mask,
                value=self.constraint_value,
                rhs=self.rhs,
                u=self.u,
                max_iter=800,
                tol=1e-6,
                use_jacobi=True,
                always_rebuild_topology=True,  # set False if mesh static and you want caching
            )
        else:
            solver = SolvePoissonDirichlet(
                E_verts=self.mesh.E_verts,
                w=self.dec.star1,
                mask=self.constraint_mask,
                value=self.constraint_value,
                rhs=self.rhs,
                u=self.u,
                max_iter=800,
                tol=1e-6,
                poll_block=25,
                use_jacobi=True,
                always_rebuild_topology=False,
                block_dim=256,
            )

        deps = (
            self.mesh.E_verts.name,
            self.dec.star1.name,
            self.constraint_mask.name,
            self.constraint_value.name,
            self.rhs.name,
        )

        self.declare_resource(
            self.u,
            buffer=None,
            deps=deps,
            producer=solver,
            description="Poisson solution",
        )
