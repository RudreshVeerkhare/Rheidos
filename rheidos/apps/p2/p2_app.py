from rheidos.houdini.runtime.cook_context import CookContext

import numpy as np

from ._graphs import P2StreamGraph
from ._io import load_mesh_input, load_point_vortex_input, read_probe_input


class P2Module:
    def __init__(self, ctx: CookContext) -> None:
        graph = ctx.world().require(P2StreamGraph)
        self._graph = graph
        self.mesh = graph.mesh
        self.point_vortex = graph.point_vortex
        self.p2_space = graph.p2_space
        self.p2_poisson = graph.p2_poisson
        self.p2_stream = graph.p2_stream
        self.p2_vel = graph.p2_vel


def solve_p2_stream_function(ctx: CookContext, eps: float = 0.01) -> None:
    mods = P2Module(ctx)
    load_mesh_input(ctx, mods.mesh)
    load_point_vortex_input(ctx, mods.point_vortex, index=1)
    mods.p2_stream.constrained_idx.set(np.array([0], dtype=np.int32))
    mods.p2_stream.constrained_values.set(np.array([0], dtype=np.float32))
    mods.p2_stream.eps.set(eps)
    mods.p2_stream.psi.get()


def sample_p2_stream_function(ctx: CookContext) -> None:
    mods = P2Module(ctx)
    faceids, bary = read_probe_input(ctx, index=1)
    stream_func = mods.p2_stream.interpolate((faceids, bary))
    ctx.write_point("stream_func", stream_func)


def sample_p2_velocity(ctx: CookContext) -> None:
    mods = P2Module(ctx)
    faceids, bary = read_probe_input(ctx, index=1)
    vel = mods.p2_vel.interpolate((faceids, bary))
    ctx.write_point("vel", vel)


# Backward-compatible aliases for existing import sites.
p2_cook = solve_p2_stream_function
p2_cook2 = sample_p2_stream_function
p2_interpolate_velocity = sample_p2_velocity
