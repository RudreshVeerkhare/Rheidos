from rheidos.compute import (
    ModuleBase,
    World,
    ResourceSpec,
    shape_of,
)

from .point_vortex import PointVortexModule
from .surface_mesh import SurfaceMeshModule
from .dec_operator import SurfaceDECModule
from .poisson_solver import PoissonSolverModule

from ..producers.splat_pt_vortex import SplatPtVortexProducer
from ..producers.stream_func import StreamFuncProducer

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
            self.pt_vortex.n_vortices,
            self.pt_vortex.gammas,
            self.pt_vortex.face_ids,
            self.mesh.V_pos,
            self.mesh.F_verts,
            self.pt_vortex.bary,
            self.omega,
        )

        self.declare_resource(
            self.omega,
            deps=(
                self.mesh.V_pos,
                self.mesh.F_verts,
                self.pt_vortex.n_vortices,  # TODO: Create helper to get all fields as dependency for this. Use regex, and negative mask selection
                self.pt_vortex.gammas,
                self.pt_vortex.face_ids,
                self.pt_vortex.bary,
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
            self.omega,
            self.pt_vortex.n_vortices,
            self.pt_vortex.face_ids,
            self.mesh.F_verts,
            self.poisson.constraint_mask,
            self.poisson.constraint_value,
            self.poisson.u,
            self.psi,
            pin_vertex_id=0,
        )

        self.declare_resource(
            self.psi, deps=(self.omega,), producer=stream_func_producer
        )
