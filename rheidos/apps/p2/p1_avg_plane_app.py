from typing import Callable

import numpy as np
from rheidos.houdini.runtime.cook_context import CookContext

from ._graphs import P1PlaneGraph
from ._io import load_mesh_input, load_point_vortex_input, read_probe_input


class P1PlaneModule:
    def __init__(self, ctx: CookContext) -> None:
        graph = ctx.world().require(P1PlaneGraph)
        self._graph = graph
        self.mesh = graph.mesh
        self.point_vortex = graph.point_vortex
        self.dec = graph.dec
        self.p1_poisson = graph.p1_poisson
        self.p1_stream_func = graph.p1_stream_func
        self.p1_vel = graph.p1_vel
        self.rk4 = graph.rk4


def setup_p1_stream_function(ctx: CookContext) -> None:
    mods = P1PlaneModule(ctx)
    load_mesh_input(ctx, mods.mesh)
    load_point_vortex_input(ctx, mods.point_vortex, index=1)
    mods.p1_stream_func.set_homo_dirichlet_boundary()

    is_closed_surface = mods.mesh.boundary_edge_count.get() == 0
    if is_closed_surface:
        mods.p1_stream_func.distribute_excess_vorticity = True


def interpolate_p1_stream_func(ctx: CookContext) -> None:
    mods = P1PlaneModule(ctx)
    faceids, bary = read_probe_input(ctx, index=1)
    stream_func = mods.p1_stream_func.interpolate((faceids, bary))
    ctx.write_point("stream_func", stream_func)


def interpolate_p1_velocity(ctx: CookContext) -> None:
    mods = P1PlaneModule(ctx)
    faceids, bary = read_probe_input(ctx, index=1)
    vel = mods.p1_vel.interpolate((faceids, bary))
    ctx.write_point("vel", vel)


def rk4_step(ctx: CookContext) -> Callable[[np.ndarray, float], np.ndarray]:
    mods = P1PlaneModule(ctx)

    def _fn(y: np.ndarray, t: float) -> np.ndarray:
        faceids, barys, pos = mods.mesh.project_on_nearest_face(y)
        gammas = mods.point_vortex.gamma.get()
        mods.point_vortex.set_vortex(
            faceids.astype(np.int32),
            barys.astype(np.float32),
            gammas.astype(np.float32),
            pos.astype(np.float32),
        )
        return mods.p1_vel.interpolate((faceids, barys))

    return _fn


def rk4_advect(ctx: CookContext) -> None:
    mods = P1PlaneModule(ctx)
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
