from typing import Callable

from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_annulus_harmoic_stream_function import (
    P1AnnulusHarmonicStreamFunction,
)
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p1_space.p1_velocity import P1VelocityFieldModule
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute.resource import ResourceSpec
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import World
from rheidos.houdini import CookContext
from rheidos.compute import ModuleBase, shape_map

from ._io import load_mesh_input, read_probe_input, load_point_vortex_input

import numpy as np


class CombinedStreamFunction(ModuleBase):

    def __init__(
        self,
        world: World,
        *,
        scope: str = "",
        stream: P1StreamFunction,
        harmonic: P1AnnulusHarmonicStreamFunction,
    ) -> None:
        super().__init__(world, scope=scope)

        self.stream = stream
        self.harmonic = harmonic

        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.stream, lambda s: (s[0],)),
            ),
            doc="Combined stream function. Shape: (nV, 1)",
        )

        self.bind_producers()

    @producer(inputs=("stream.psi", "harmonic.psi"), outputs=("psi",))
    def calculate_combined_psi(self, ctx: ProducerContext):
        ctx.require_inputs()
        psi_s = self.stream.psi.get()
        psi_h = self.harmonic.psi.get()

        ctx.commit(psi=np.array(psi_s + psi_h, dtype=np.float64))


class P1AnnulusHarmonicModule(ModuleBase):

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(DEC, mesh=self.mesh)

        # Point Vortex
        self.point_vortex = self.require(PointVortexModule)
        self.stream_function = self.require(
            P1StreamFunction,
            mesh=self.mesh,
            point_vortex=self.point_vortex,
            dec=self.dec,
        )

        # Harmonic Component
        self.harmonic_stream = self.require(
            P1AnnulusHarmonicStreamFunction, mesh=self.mesh, dec=self.dec
        )

        # Combined Stream function
        self.combined_stream_function = self.require(
            CombinedStreamFunction,
            stream=self.stream_function,
            harmonic=self.harmonic_stream,
        )

        self.harmonic_vel = self.require(
            P1VelocityFieldModule,
            child=True,
            child_name="harmonic_basis",
            mesh=self.mesh,
            dec=self.dec,
            stream=self.harmonic_stream,
        )

        self.vel = self.require(
            P1VelocityFieldModule,
            child=True,
            child_name="p1_velocity",
            mesh=self.mesh,
            dec=self.dec,
            stream=self.combined_stream_function,
        )

        self.rk4 = self.require(RK4IntegratorModule)


def setup_p1_harmonic_stream_function(ctx: CookContext) -> None:
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    load_mesh_input(
        ctx,
        mods.mesh,
        missing_message="Input 0 has to be a mesh input geometry",
    )
    load_point_vortex_input(ctx, mods.point_vortex, index=1)
    mods.stream_function.set_homo_dirichlet_boundary()

    mods.harmonic_stream.set_annulus_dirichlet_boundary()


def interpolate_p1_harmonic_stream_function(ctx: CookContext) -> None:
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    faceids, bary = read_probe_input(ctx, index=0)
    stream_func = mods.harmonic_stream.interpolate((faceids, bary))
    ctx.write_point("harmonic_stream_func", stream_func)


def interpolate_p1_harmonic_velocity(ctx: CookContext) -> None:
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    faceids, bary = read_probe_input(ctx, index=0)
    vel = mods.harmonic_vel.interpolate((faceids, bary))
    ctx.write_point("harmonic_vel", vel)


def interpolate_p1_stream_function(ctx: CookContext) -> None:
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    faceids, bary = read_probe_input(ctx, index=0)
    stream_func = mods.stream_function.interpolate((faceids, bary))
    ctx.write_point("stream_func", stream_func)


def interpolate_p1_velocity(ctx: CookContext) -> None:
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    faceids, bary = read_probe_input(ctx, index=0)
    vel = mods.vel.interpolate((faceids, bary))
    ctx.write_point("vel", vel)


# Advection
def rk4_step(ctx: CookContext) -> Callable[[np.ndarray, float], np.ndarray]:
    mods = ctx.world().require(P1AnnulusHarmonicModule)

    def _fn(y: np.ndarray, t: float) -> np.ndarray:
        faceids, barys, pos = mods.mesh.project_on_nearest_face(y)
        gammas = mods.point_vortex.gamma.get()
        mods.point_vortex.set_vortex(
            faceids.astype(np.int32),
            barys.astype(np.float32),
            gammas.astype(np.float32),
            pos.astype(np.float32),
        )
        return mods.vel.interpolate((faceids, barys))

    return _fn


def rk4_advect(ctx: CookContext) -> None:
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    y_dot = rk4_step(ctx)
    mods.rk4.configure(y_dot=y_dot, timestep=0.01)
    load_point_vortex_input(ctx, mods.point_vortex, index=0)
    y0 = mods.point_vortex.pos_world.get()
    y = mods.rk4.step(y0)
    faceids, barys, pos = mods.mesh.project_on_nearest_face(y)
    gammas = mods.point_vortex.gamma.get()
    mods.point_vortex.set_vortex(
        faceids.astype(np.int32),
        barys.astype(np.float32),
        gammas.astype(np.float32),
        pos.astype(np.float32),
    )
    ctx.write_point("P", pos)
    ctx.write_point("bary", barys)
    ctx.write_point("faceid", faceids)
