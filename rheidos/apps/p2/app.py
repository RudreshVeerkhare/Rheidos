from rheidos.houdini.runtime.cook_context import CookContext

import numpy as np

from ._graphs import P1StreamGraph
from ._io import load_mesh_input, load_point_vortex_input, read_probe_input


class P1Module:
    def __init__(self, ctx: CookContext) -> None:
        graph = ctx.world().require(P1StreamGraph)
        self._graph = graph
        self.mesh = graph.mesh
        self.point_vortex = graph.point_vortex
        self.dec = graph.dec
        self.p1_poisson = graph.p1_poisson
        self.p1_stream = graph.p1_stream


def sample_p1_stream_function(ctx: CookContext) -> None:
    mods = P1Module(ctx)
    faceids, bary = read_probe_input(ctx, index=1)
    stream_func = mods.p1_stream.interpolate((faceids, bary))
    ctx.write_point("stream_func", stream_func)


def solve_p1_stream_function(ctx: CookContext) -> None:
    mods = P1Module(ctx)
    load_mesh_input(ctx, mods.mesh)
    load_point_vortex_input(ctx, mods.point_vortex, index=1)
    mods.p1_stream.omega.get()
    mods.p1_stream.constrained_idx.set(np.array([0], dtype=np.int32))
    mods.p1_stream.constrained_values.set(np.array([0], dtype=np.float64))
    psi = mods.p1_stream.psi.get()
    ctx.write_point("stream_func", psi)


# Backward-compatible aliases for existing import sites.
cook = solve_p1_stream_function
cook2 = sample_p1_stream_function
