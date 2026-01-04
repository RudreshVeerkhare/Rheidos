from rheidos.compute import (
    ModuleBase,
    World,
    ResourceSpec,
    shape_of,
)


from .surface_mesh import SurfaceMeshModule
from .stream_func import StreamFunctionModule

from ..producers.stream_func_velocity import FaceVelocityFromStreamProducer

import taichi as ti


class VelocityFieldModule(ModuleBase):
    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule)
        self.stream_func = world.require(StreamFunctionModule)

        # piecewise constant velocity per face
        self.F_velocity = self.resource(
            "F_velocity",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.mesh.F_verts),
                allow_none=True,
            ),
            doc="Facewise constant velocity field as $J \nabla \psi$. Shape: (nF, vec3f)",
            declare=False,
        )

        vel_producer = FaceVelocityFromStreamProducer(
            self.mesh.V_pos, self.mesh.F_verts, self.stream_func.psi, self.F_velocity
        )

        self.declare_resource(
            self.F_velocity, deps=(self.stream_func.psi,), producer=vel_producer
        )
