import numpy as np

from rheidos.apps.p2._io import (
    load_mesh_input,
    load_point_vortex_input,
    read_probe_input,
)
from rheidos.apps.p2.double_obstacle.combined_stream_function import (
    CombinedStreamFunction,
)
from rheidos.apps.p2.double_obstacle.ray_sop_module import RaySopModule
from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p1_space.p1_velocity import P1VelocityFieldModule
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute.world import ModuleBase, World
from rheidos.houdini.runtime.cook_context import CookContext

from .harmonic_basis import HarmonicBasisFieldModule, HarmonicBasisModule

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

        # Harmonic Part
        self.harmonic_basis_potential = self.require(
            HarmonicBasisModule,
            mesh=self.mesh,
            dec=self.dec,
            harmonic_dim=2,
        )
        self.harmonic_basis_field = self.require(
            HarmonicBasisFieldModule,
            mesh=self.mesh,
            dec=self.dec,
            harmonic_basis=self.harmonic_basis_potential,
        )

        # Combined Stream function
        self.combined_stream_function = self.require(
            CombinedStreamFunction,
            point_vortex=self.point_vortex,
            stream=self.stream_function,
            harmonic_basis_potential=self.harmonic_basis_potential,
        )
        self.velocity = self.require(
            P1VelocityFieldModule,
            child=True,
            child_name="combined_velocity",
            mesh=self.mesh,
            dec=self.dec,
            stream=self.combined_stream_function,
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
    def rk4_step(ctx: CookContext, no_harmonic):
        mods = ctx.world().require(App)
        mods.surface_projector.configure(node_path=RAY_SOP_NODE_PATH)
        mods.surface_projector.setup(ctx)

        velocity_mod = mods.coexact_velocity if no_harmonic else mods.velocity

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

            return velocity_mod.interpolate((faceids, barys))

        return y_dot


# Node callers
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

    # Setup harmonic potential
    bo, bi1, bi2 = mods.mesh.boundary_vertex_components.get()
    mods.harmonic_basis_potential.set_boudaries([1, 2], 0)

    # Compute initial harmonic coefficient
    mods.combined_stream_function.initialize_harmonic_coeffs()


def interpolate_harmonic_potential(ctx: CookContext, basis_id=0):
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    potential_value = mods.harmonic_basis_potential.interpolate(
        (faceids, bary), basis_id=basis_id
    )
    ctx.write_point("harmonic_potential", potential_value)


def interpolate_combined_stream_function(ctx: CookContext):
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    combined_stream_functions = mods.combined_stream_function.interpolate(
        (faceids, bary)
    )
    ctx.write_point("combined_stream_functions", combined_stream_functions)


def interpolate_harmonic_basis_field(ctx: CookContext, basis_id=0):
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    field_value = mods.harmonic_basis_field.interpolate(
        (faceids, bary), basis_id=basis_id
    )
    ctx.write_point("harmonic_field", field_value)


def interpolate_combined_velocity_field(ctx: CookContext, smooth=True):
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    velocity = mods.velocity.interpolate((faceids, bary), smooth=smooth)
    ctx.write_point("velocity", velocity)


def rk4_advect(ctx: CookContext, dt=0.001, no_harmonic=False):
    mods = ctx.world().require(App)
    mods.surface_projector.configure(node_path=RAY_SOP_NODE_PATH)
    mods.surface_projector.setup(ctx)

    y_dot = mods.rk4_step(ctx, no_harmonic=no_harmonic)
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
    harmonic_coeff = mods.combined_stream_function.harmonic_coefficient.get()
    ctx.write_point("P", pos)
    ctx.write_point("bary", barys)
    ctx.write_point("faceid", faceids)
    ctx.write_detail("harmonic_coeff", harmonic_coeff)
