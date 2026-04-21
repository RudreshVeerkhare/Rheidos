from typing import Callable

import numpy as np
from rheidos.houdini.runtime.cook_context import CookContext

from ._graphs import P2PlaneGraph
from ._io import load_mesh_input, load_point_vortex_input, read_probe_input


class P2PlaneModule:
    def __init__(self, ctx: CookContext) -> None:
        graph = ctx.world().require(P2PlaneGraph)
        self._graph = graph
        self.mesh = graph.mesh
        self.point_vortex = graph.point_vortex
        self.p2_space = graph.p2_space
        self.p2_poisson = graph.p2_poisson
        self.p2_stream_func = graph.p2_stream_func
        self.p2_vel = graph.p2_vel
        self.rk4 = graph.rk4


def rk4_step(ctx: CookContext) -> Callable[[np.ndarray, float], np.ndarray]:
    mods = P2PlaneModule(ctx)

    def _fn(y: np.ndarray, t: float) -> np.ndarray:
        faceids, barys, pos = mods.mesh.project_on_nearest_face(y)
        gammas = mods.point_vortex.gamma.get()
        mods.point_vortex.set_vortex(
            faceids.astype(np.int32),
            barys,
            gammas,
            pos,
        )
        return mods.p2_vel.interpolate((faceids, barys))

    return _fn


def rk4_advect(ctx: CookContext) -> None:
    mods = P2PlaneModule(ctx)
    y_dot = rk4_step(ctx)
    mods.rk4.configure(y_dot=y_dot, timestep=0.001)
    load_point_vortex_input(ctx, mods.point_vortex, index=0)
    y0 = mods.point_vortex.pos_world.get()
    y = mods.rk4.step(y0)
    faceids, barys, pos = mods.mesh.project_on_nearest_face(y)
    gammas = mods.point_vortex.gamma.get()
    mods.point_vortex.set_vortex(
        faceids.astype(np.int32),
        barys,
        gammas,
        pos,
    )
    ctx.write_point("P", pos)
    ctx.write_point("bary", barys)
    ctx.write_point("faceid", faceids)


def interpolate_p2_velocity(ctx: CookContext) -> None:
    mods = P2PlaneModule(ctx)
    faceids, bary = read_probe_input(ctx, index=1)
    vel = mods.p2_vel.interpolate((faceids, bary))
    ctx.write_point("vel", vel)


def interpolate_p2_stream_func(ctx: CookContext) -> None:
    mods = P2PlaneModule(ctx)
    faceids, bary = read_probe_input(ctx, index=1)
    stream_func = mods.p2_stream_func.interpolate((faceids, bary))
    ctx.write_point("stream_func", stream_func)


def setup_p2_stream_function(ctx: CookContext) -> None:
    mods = P2PlaneModule(ctx)
    load_mesh_input(ctx, mods.mesh)
    load_point_vortex_input(ctx, mods.point_vortex, index=1)
    mods.p2_stream_func.set_homo_dirichlet_boundary()

    param = ctx.node.parm("eps")
    if param is not None:
        eps = float(param.eval())
        mods.p2_stream_func.eps.set(eps)
