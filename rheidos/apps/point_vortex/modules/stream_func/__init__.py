from rheidos.compute import (
    ModuleBase,
    World,
    ResourceSpec,
    module_resource_deps,
    shape_of,
)

from ..point_vortex import PointVortexModule
from ..surface_mesh import SurfaceMeshModule
from ..dec_operator import SurfaceDECModule
from ..poisson_solver import PoissonSolverModule

from .splat_pt_vortex import SplatPtVortexProducer
from .stream_func_producer import StreamFuncProducer

import taichi as ti


class StreamFunctionModule(ModuleBase):
    NAME = "StreamFunctionModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule)
        self.dec = world.require(SurfaceDECModule)
        self.pt_vortex = world.require(PointVortexModule)
        self.poisson = world.require(PoissonSolverModule)

        ## Dual vorticity 2-form
        self.omega = self.resource(
            "omega",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.f32, shape_fn=shape_of(self.mesh.V_pos)
            ),
            doc="Per vertex dual vorticity 2-form. Shape: (nV, )",
            declare=False,
        )

        pt_vortex_splat_producer = SplatPtVortexProducer(
            n_vortices=self.pt_vortex.n_vortices,
            gammas=self.pt_vortex.gammas,
            face_ids=self.pt_vortex.face_ids,
            V_pos=self.mesh.V_pos,
            F_verts=self.mesh.F_verts,
            bary=self.pt_vortex.bary,
            omega=self.omega,
        )

        self.declare_resource(
            self.omega,
            deps=(
                self.mesh.V_pos,
                self.mesh.F_verts,
                *module_resource_deps(self.pt_vortex, exclude=r"^(frame|pos_world)$"),
            ),
            producer=pt_vortex_splat_producer,
        )

        ## Stream function
        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.mesh.V_pos),
                allow_none=True,
            ),
            doc="0-form stream function for fluids. Shape: (nV, f32)",
            declare=False,
        )

        stream_func_producer = StreamFuncProducer(
            omega=self.omega,
            n_vortices=self.pt_vortex.n_vortices,
            vortices_face_ids=self.pt_vortex.face_ids,
            F_verts=self.mesh.F_verts,
            u=self.poisson.u,
            constraint_mask=self.poisson.constraint_mask,
            constraint_values=self.poisson.constraint_value,
            rhs=self.poisson.rhs,
            psi=self.psi,
        )

        self.declare_resource(
            self.psi, deps=(self.omega,), producer=stream_func_producer
        )
