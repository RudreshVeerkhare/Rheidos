import numpy as np
from typing import Callable

from rheidos.apps.p2._io import (
    load_mesh_input,
    load_point_vortex_input,
    read_probe_input,
)
from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p1_space.p1_velocity import P1VelocityFieldModule
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute.world import ModuleBase, World
from rheidos.houdini.runtime.cook_context import CookContext


class App(ModuleBase):
    NAME = "P1SphereApp"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(DEC, mesh=self.mesh)

        # Point Vortices
        self.point_vortex = self.require(PointVortexModule)
        self.coexact_stream_function = self.require(
            P1StreamFunction,
            mesh=self.mesh,
            point_vortex=self.point_vortex,
            dec=self.dec,
        )
        self.coexact_vel = self.require(
            P1VelocityFieldModule,
            mesh=self.mesh,
            stream=self.coexact_stream_function,
            dec=self.dec,
        )

        # Advection
        self.rk4 = self.require(RK4IntegratorModule)

    # y_dot
    @staticmethod
    def rk4_step(ctx: CookContext) -> Callable[[np.ndarray, float], np.ndarray]:
        mods = ctx.world().require(App)

        def _fn(y: np.ndarray, t: float) -> np.ndarray:
            faceids, barys, pos = mods.mesh.project_on_nearest_face(y)
            gammas = mods.point_vortex.gamma.get()
            mods.point_vortex.set_vortex(
                faceids,
                barys,
                gammas,
                pos,
            )
            return mods.coexact_vel.interpolate((faceids, barys))

        return _fn


def setup_coexact_stream_function(ctx: CookContext):
    mods = ctx.world().require(App)
    load_mesh_input(
        ctx, mods.mesh, missing_message="Input 0 has to be mesh input geometry"
    )
    load_point_vortex_input(ctx, mods.point_vortex, index=1)
    mods.coexact_stream_function.set_homo_dirichlet_boundary()


def interpolate_coexact_stream_function(ctx: CookContext) -> None:
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    stream_func = mods.coexact_stream_function.interpolate((faceids, bary))
    ctx.write_point("coexact_stream_func", stream_func)


def interpolate_coexact_velocity(ctx: CookContext) -> None:
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    vel = mods.coexact_vel.interpolate((faceids, bary))
    ctx.write_point("coexact_vel", vel)


def rk4_advect(ctx: CookContext) -> None:
    mods = ctx.world().require(App)
    y_dot = mods.rk4_step(ctx)
    mods.rk4.configure(y_dot=y_dot, timestep=0.001)
    load_point_vortex_input(ctx, mods.point_vortex, index=0)
    y0 = mods.point_vortex.pos_world.get()
    y = mods.rk4.step(y0)
    faceids, barys, pos = mods.mesh.project_on_nearest_face(y)
    gammas = mods.point_vortex.gamma.get()
    mods.point_vortex.set_vortex(
        faceids,
        barys,
        gammas,
        pos,
    )
    ctx.write_point("P", pos)
    ctx.write_point("bary", barys)
    ctx.write_point("faceid", faceids)
