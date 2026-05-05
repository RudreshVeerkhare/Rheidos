import numpy as np

from rheidos.apps.p2._io import load_mesh_input, load_point_vortex_input
from rheidos.apps.p2.double_obstacle.ray_sop_module import RaySopModule
from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p1_space.p1_velocity import P1VelocityFieldModule
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute.world import ModuleBase, World
from rheidos.houdini.runtime.cook_context import CookContext

RAY_SOP_NODE_PATH = "/obj/geo1/solver1/d/s/ray1"


class App(ModuleBase):

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(DEC, mesh=self.mesh)

        # Point Vortex
        self.point_vortex = self.require(PointVortexModule)

        # Coexact stream function part
        self.stream_function = self.require(
            P1StreamFunction,
            mesh=self.mesh,
            point_vortex=self.point_vortex,
            dec=self.dec,
        )
        self.coexact_velocity = self.require(
            P1VelocityFieldModule,
            child=True,
            child_name="coexact_velocity",
            mesh=self.mesh,
            dec=self.dec,
            stream=self.stream_function,
        )

        # Advection
        self.rk4 = self.require(RK4IntegratorModule)
        self.surface_projector = self.require(
            RaySopModule,
            child=True,
            child_name="ray_sop_surface_points_projector",
            node_path=RAY_SOP_NODE_PATH,
        )

    @staticmethod
    def rk4_step(ctx: CookContext):
        mods = ctx.world().require(App)
        mods.surface_projector.configure(node_path=RAY_SOP_NODE_PATH)
        mods.surface_projector.setup(ctx)

        def y_dot(y: np.ndarray, t: float):
            projected = mods.surface_projector.project_points(y)
            faceids, barys, pos = (
                projected.faceids,
                projected.bary,
                projected.pos,
            )
            gammas = mods.point_vortex.gamma.get()
            mods.point_vortex.set_vortex(
                faceids,
                barys,
                gammas,
                pos,
            )
            return mods.coexact_velocity.interpolate((faceids, barys))

        return y_dot


def setup_mesh_and_point_vortices(ctx: CookContext):
    """Loads mesh and vortices from the geometry passed on by the houdini"""
    mods = ctx.world().require(App)

    # Load mesh
    load_mesh_input(
        ctx,
        mods.mesh,
        missing_message="Input 0 has to be a mesh input geometry",
    )

    # Load Point Vortex
    load_point_vortex_input(ctx, mods.point_vortex, index=1)
    mods.stream_function.set_homo_dirichlet_boundary()


def rk4_advect(ctx: CookContext, dt=0.001):
    mods = ctx.world().require(App)
    mods.surface_projector.configure(node_path=RAY_SOP_NODE_PATH)
    mods.surface_projector.setup(ctx)

    y_dot = mods.rk4_step(ctx)
    mods.rk4.configure(y_dot=y_dot, timestep=dt)

    load_point_vortex_input(ctx, mods.point_vortex, index=0)

    y0 = mods.point_vortex.pos_world.get()
    y = mods.rk4.step(y0)

    projected = mods.surface_projector.project_points(y)
    faceids, barys, pos = projected.faceids, projected.bary, projected.pos

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
