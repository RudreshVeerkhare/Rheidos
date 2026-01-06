from rheidos.compute import (
    ModuleBase,
    World,
    ResourceSpec,
)


from .surface_mesh import SurfaceMeshModule
from .velocity_field import VelocityFieldModule
from .point_vortex import PointVortexModule

from ..producers.const_vel_advection import AdvectVorticesEventDrivenProducer

import taichi as ti
import numpy as np


class EdgeHopPtVortexAdvectionModule(ModuleBase):
    NAME = "EdgeHopPtVortexAdvectionModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule)
        self.pt_vortex = world.require(PointVortexModule)
        self.velocity = world.require(VelocityFieldModule)

        self.dt = self.resource(
            "dt",
            spec=ResourceSpec(kind="python", dtype=float),
            doc="Timestep size in seconds.",
            declare=True,
        )

        self.new_face_ids = self.resource(
            "new_face_ids",
            spec=ResourceSpec(
                kind="numpy", dtype=np.int32, allow_none=True
            ),  # TODO: Add lazy shape evaluator for composing shapes from runtime resource values
            doc="Numpy array of new face_ids after advection step. Shape: (nVortices, i32)",
            declare=False,
        )

        self.new_bary = self.resource(
            "new_bary",
            spec=ResourceSpec(kind="numpy", dtype=np.float32, allow_none=True),
            doc="Numpy array of new barycentric coordinates for each point. Shape: (nVortices, vec3f)",
            declare=False,
        )

        self.new_pos = self.resource(
            "new_pos",
            spec=ResourceSpec(kind="numpy", dtype=np.float32, allow_none=True),
            doc="Numpy array of new position in world coordinates for each point. Shape: (nVortices, vec3f)",
            declare=False,
        )

        advection_producer = AdvectVorticesEventDrivenProducer(
            self.mesh.V_pos,
            self.mesh.F_verts,
            self.mesh.F_adj,
            self.velocity.F_velocity,
            self.pt_vortex.n_vortices,
            self.pt_vortex.face_ids,
            self.pt_vortex.bary,
            self.dt,
            self.new_face_ids,
            self.new_bary,
            self.new_pos,
        )

        self.declare_resource(
            self.new_face_ids,
            deps=(
                self.pt_vortex.n_vortices,
                self.pt_vortex.face_ids,
                self.pt_vortex.bary,
                self.dt,
            ),
            producer=advection_producer,
        )
        self.declare_resource(
            self.new_bary,
            deps=(
                self.pt_vortex.n_vortices,
                self.pt_vortex.face_ids,
                self.pt_vortex.bary,
                self.dt,
            ),
            producer=advection_producer,
        )
        self.declare_resource(
            self.new_pos,
            deps=(
                self.pt_vortex.n_vortices,
                self.pt_vortex.face_ids,
                self.pt_vortex.bary,
                self.dt,
            ),
            producer=advection_producer,
        )
