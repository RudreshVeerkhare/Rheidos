from rheidos.compute import (
    ModuleBase,
    World,
    ResourceSpec,
    shape_of,
    shape_with_tail,
)


from ..surface_mesh import SurfaceMeshModule
from ..stream_func import StreamFunctionModule

from .stream_func_velocity import FaceVelocityFromStreamProducer
from .rt0_velocity import FaceCornerVelocityRT0FromStreamProducer
from .per_vertex_velocity import PerVertexVelProducer

import taichi as ti


class VelocityFieldModule(ModuleBase):
    NAME = "VelocityFieldModule"

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
                lanes=3,
            ),
            doc="Facewise constant velocity field as $J \nabla \psi$. Shape: (nF, vec3f)",
            declare=False,
        )

        vel_producer = FaceVelocityFromStreamProducer(
            V_pos=self.mesh.V_pos,
            F_verts=self.mesh.F_verts,
            psi=self.stream_func.psi,
            vel_F=self.F_velocity,
        )

        # piecewise constant velocity per face
        self.FV_velocity = self.resource(
            "FV_velocity",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_with_tail(self.mesh.F_verts, tail=(3,)),
                lanes=3,
                allow_none=True,
            ),
            doc="RT0 interpolation. Shape: (nF, vec3f)",
            declare=False,
        )

        rt0_vel_producer = FaceCornerVelocityRT0FromStreamProducer(
            V_pos=self.mesh.V_pos,
            F_verts=self.mesh.F_verts,
            psi=self.stream_func.psi,
            vel_FV=self.FV_velocity,
        )

        self.declare_resource(
            self.FV_velocity, deps=(self.stream_func.psi,), producer=rt0_vel_producer
        )

        self.declare_resource(
            self.F_velocity, deps=(self.stream_func.psi,), producer=vel_producer
        )

        # per vertex velocity by averaging across all faces a vertex is incident on
        self.V_velocity = self.resource(
            "V_velocity",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.mesh.V_pos),
                lanes=3,
                allow_none=True,
            ),
            doc="per-vertex velocity by averaging across all faces a vertex is incident on. Shape: (nV, vec3f)",
            declare=False,
        )

        per_vert_vel_producer = PerVertexVelProducer(
            V_incident_count=self.mesh.V_incident_count,
            F_verts=self.mesh.F_verts,
            F_velocity=self.F_velocity,
            V_velocity=self.V_velocity,
        )
        self.declare_resource(
            self.V_velocity, deps=(self.F_velocity,), producer=per_vert_vel_producer
        )
