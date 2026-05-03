from typing import Callable

from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_annulus_harmoic_stream_function import (
    P1AnnulusHarmonicStreamFunction,
)
from rheidos.apps.p2.modules.p1_space.probe_utils import probe_arrays
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p1_space.p1_velocity import P1VelocityFieldModule
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute.resource import ResourceSpec
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import World
from rheidos.compute import ModuleBase, shape_map
from rheidos.houdini import CookContext
from rheidos import logger

from ._io import load_mesh_input, read_probe_input, load_point_vortex_input

import numpy as np


class CombinedStreamFunction(ModuleBase):
    NAME = "CombinedStreamFunction"

    def __init__(
        self,
        world: World,
        *,
        point_vortex: PointVortexModule,
        stream: P1StreamFunction,
        harmonic: P1AnnulusHarmonicStreamFunction,
        initial_coeff=0,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.point_vortex = point_vortex
        self.stream = stream
        self.harmonic = harmonic
        self._init_coeff = initial_coeff

        self.harmonic_coefficient = self.resource(
            "harmonic_coefficient",
            spec=ResourceSpec(kind="python", dtype=float),
            doc="Evolving harmonic coefficient of the harmonic component.",
        )

        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.stream.psi, lambda s: (s[0],)),
            ),
            doc="Combined stream function. Shape: (nV, 1)",
        )

        self.bind_producers()

    @producer(
        inputs=(
            "harmonic.psi",
            "point_vortex.gamma",
            "point_vortex.face_ids",
            "point_vortex.bary",
        ),
        outputs=("harmonic_coefficient",),
    )
    def compute_harmonic_coefficient(self, ctx: ProducerContext):
        ctx.require_inputs()
        gammas = self.point_vortex.gamma.get()
        faceids = self.point_vortex.face_ids.get()
        barys = self.point_vortex.bary.get()
        hpsi = self.harmonic.interpolate(list(zip(faceids, barys)))

        c = self._init_coeff

        for vid, gamma in enumerate(gammas):
            c -= gamma * hpsi[vid]

        ctx.commit(harmonic_coefficient=c)

    @producer(
        inputs=("stream.psi", "harmonic.psi", "harmonic_coefficient"), outputs=("psi",)
    )
    def calculate_combined_psi(self, ctx: ProducerContext):
        ctx.require_inputs()
        psi_s = self.stream.psi.get()
        psi_h = self.harmonic.psi.get()
        c = self.harmonic_coefficient.get()

        ctx.commit(psi=np.array(psi_s + c * psi_h, dtype=np.float64))

    def interpolate(self, probes) -> np.ndarray:
        psi = self.psi.get()
        faceids, bary = probe_arrays(probes)
        verts = self.stream.mesh.F_verts.get()[faceids]
        return np.einsum("ij,ij->i", psi[verts], bary)


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

        self.coexact_stream_vel = self.require(
            P1VelocityFieldModule,
            child=True,
            child_name="coexact_stream_vel",
            mesh=self.mesh,
            dec=self.dec,
            stream=self.stream_function,
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
            point_vortex=self.point_vortex,
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

    # Advection
    @staticmethod
    def rk4_step(ctx: CookContext) -> Callable[[np.ndarray, float], np.ndarray]:
        mods = ctx.world().require(P1AnnulusHarmonicModule)

        def _fn(y: np.ndarray, t: float) -> np.ndarray:
            faceids, barys, pos = mods.mesh.project_on_nearest_face(y)
            gammas = mods.point_vortex.gamma.get()
            mods.point_vortex.set_vortex(
                faceids,
                barys,
                gammas,
                pos,
            )
            return mods.vel.interpolate((faceids, barys))

        return _fn


def setup_annulus_stream_functions(ctx: CookContext) -> None:
    """Loads mesh and vortices from the geometry passed on by the houdini"""
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    load_mesh_input(
        ctx,
        mods.mesh,
        missing_message="Input 0 has to be a mesh input geometry",
    )
    load_point_vortex_input(ctx, mods.point_vortex, index=1)
    mods.stream_function.set_homo_dirichlet_boundary()
    mods.harmonic_stream.set_annulus_dirichlet_boundary()


def interpolate_p1_coexact_stream_function(ctx: CookContext) -> None:
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    faceids, bary = read_probe_input(ctx, index=0)
    stream_func = mods.stream_function.interpolate((faceids, bary))
    ctx.write_point("coexact_stream_func", stream_func)


def interpolate_p1_coexact_velocity(ctx: CookContext) -> None:
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    faceids, bary = read_probe_input(ctx, index=0)
    vel = mods.coexact_stream_vel.interpolate((faceids, bary))
    ctx.write_point("coexact_vel", vel)


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


def interpolate_combined_stream_function(ctx: CookContext) -> None:
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    faceids, bary = read_probe_input(ctx, index=0)
    stream_func = mods.combined_stream_function.interpolate((faceids, bary))
    ctx.write_point("stream_func", stream_func)


def interpolate_p1_velocity(ctx: CookContext) -> None:
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    faceids, bary = read_probe_input(ctx, index=0)
    vel = mods.vel.interpolate((faceids, bary))
    ctx.write_point("vel", vel)


def _log_harmonic_coefficient(ctx: CookContext, mods: P1AnnulusHarmonicModule) -> None:
    logger.log(
        "harmonic_coefficient",
        float(mods.combined_stream_function.harmonic_coefficient.get()),
        category="p1_annulus",
        flush=True,
    )


def rk4_advect(ctx: CookContext) -> None:
    mods = ctx.world().require(P1AnnulusHarmonicModule)
    y_dot = mods.rk4_step(ctx)
    mods.rk4.configure(y_dot=y_dot, timestep=0.01)
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

    # Log harmonic coeffs as well
    ch = mods.combined_stream_function.harmonic_coefficient.get()
    ctx.write_point("harmonic_coeff", [ch])

    # Harmonic velocity
    harmonic_vel = ch * mods.harmonic_vel.interpolate((faceids, barys))
    ctx.write_point("harmonic_vel", harmonic_vel)

    # Combined velocity
    velocity = mods.vel.interpolate((faceids, barys))
    ctx.write_point("velocity", velocity)

    # Coexact velocity
    coexact_vel = mods.coexact_stream_vel.interpolate((faceids, barys))
    ctx.write_point("coexact_vel", coexact_vel)


def load_points_with_gamma_faceid_bary(ctx: CookContext, index: int):
    # TODO: Convert to a generic geomety loader.
    points_io = ctx.input_io(index)
    if points_io is None:
        raise RuntimeError("Input missing!")

    pos = np.array(points_io.read_point("P", components=3), dtype=np.float64)
    barys = np.array(points_io.read_point("bary", components=3), dtype=np.float64)
    gammas = np.array(points_io.read_point("gamma"), dtype=np.float64)
    faceids = np.array(points_io.read_point("faceid"), dtype=np.int32)

    return pos, barys, gammas, faceids


def set_evaluate_vortex_and_core_velocity(ctx: CookContext):
    """
    This function expects a list of points for which it then sequentially:
        1. Set vortex location
        2. Solve poisson problem
        3. Calculate velocity at core
        4. Consolidate all velocity values and write them to houdini geometry attribute

    Caveat: doesn't reset the vortex location to original
    """
    mods = ctx.world().require(P1AnnulusHarmonicModule)

    # Read input probe locations
    ## Expects points to have `faceid` and `bary` as attribute
    pos, barys, gammas, faceids = load_points_with_gamma_faceid_bary(ctx, index=0)

    velocities = []

    for index in range(len(gammas)):
        # 1. set vortex location
        mods.point_vortex.set_vortex(
            [faceids[index]], [barys[index]], [gammas[index]], [pos[index]]
        )

        # 2. Read velocity at the vortex core
        vel = mods.coexact_stream_vel.interpolate(([faceids[index]], [barys[index]]))

        velocities.append(vel)

    # 3. Write all velocities back to houdini
    ctx.write_point("vortex_core_vel", np.array(velocities).reshape(-1, 3))
