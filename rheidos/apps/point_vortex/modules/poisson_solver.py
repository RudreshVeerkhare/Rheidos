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

from typing import Optional, Any
import taichi as ti


class PoissonSolverModule(ModuleBase):
    """
    Resources
    ----------
    E_verts : ResourceRef[Any]
        (nE, vec2i) or similar; indexable as E[e][0], E[e][1]. Edge vertex indices.
    w : ResourceRef[Any]
        (nE,) edge weights (cotan / star1 etc.).
    mask : ResourceRef[Any]
        (nV,) i32, 1=constrained. Dirichlet constraint mask for vertices.
    value : ResourceRef[Any]
        (nV,) f32, value on constrained verts. Dirichlet constraint values.
    rhs : Optional[ResourceRef[Any]], optional
        (nV,) f32 or None. Optional right-hand side: solves K u = rhs on free verts (with Dirichlet). If None, performs harmonic interpolation.
    u : ResourceRef[Any]
        (nV,) f32. Output solution field.
    """

    NAME = "poisson"

    def __init__(self, world: World, *, scope: str = "") -> None:
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

        # Optional RHS: if None -> harmonic interpolation (rhs=0)
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
