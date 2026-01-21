from rheidos.compute import ModuleBase, World, ResourceSpec, shape_of

from .surface_mesh import SurfaceMeshModule
from .point_vortex import PointVortexModule
from .velocity_field import VelocityFieldModule
from ..producers.rk4_advection import AdvectVorticesRK4Producer

import taichi as ti


class RK4AdvectionModule(ModuleBase):
    NAME = "RK4AdvectionModule"

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

        self.face_ids_out = self.resource(
            "face_ids_out",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                shape_fn=shape_of(self.pt_vortex.face_ids),
                allow_none=True,
            ),
            doc="Updated face ids after RK4 advection. Shape: (maxV,) i32",
            declare=False,
        )

        self.bary_out = self.resource(
            "bary_out",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.pt_vortex.bary),
                lanes=3,
                allow_none=True,
            ),
            doc="Updated barycentric coordinates after RK4 advection. Shape: (maxV, vec3f)",
            declare=False,
        )

        self.pos_out = self.resource(
            "pos_out",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.pt_vortex.face_ids),
                lanes=3,
                allow_none=True,
            ),
            doc="Updated world-space positions after RK4 advection. Shape: (maxV, vec3f)",
            declare=False,
        )

        rk4_producer = AdvectVorticesRK4Producer(
            self.mesh.V_pos,
            self.mesh.F_verts,
            self.mesh.F_adj,
            self.velocity.V_velocity,
            self.pt_vortex.n_vortices,
            self.pt_vortex.face_ids,
            self.pt_vortex.bary,
            self.dt,
            self.face_ids_out,
            self.bary_out,
            self.pos_out,
        )

        deps = (
            self.mesh.V_pos,
            self.mesh.F_verts,
            self.mesh.F_adj,
            self.velocity.V_velocity,
            self.pt_vortex.n_vortices,
            self.pt_vortex.face_ids,
            self.pt_vortex.bary,
            self.dt,
        )

        self.declare_resource(self.face_ids_out, deps=deps, producer=rk4_producer)
        self.declare_resource(self.bary_out, deps=deps, producer=rk4_producer)
        self.declare_resource(self.pos_out, deps=deps, producer=rk4_producer)

    def advect(self, dt: float) -> None:
        self.dt.set(float(dt))
        self.face_ids_out.get()
