from rheidos.houdini.runtime.cook_context import CookContext

import numpy as np

from ._graphs import P1PoissonTestGraph, P2PoissonTestGraph
from ._io import load_mesh_input, read_probe_input


class P2Module:
    def __init__(self, ctx: CookContext) -> None:
        graph = ctx.world().require(P2PoissonTestGraph)
        self._graph = graph
        self.mesh = graph.mesh
        self.p2_space = graph.p2_space
        self.p2_poisson = graph.p2_poisson


class P1Module:
    def __init__(self, ctx: CookContext) -> None:
        graph = ctx.world().require(P1PoissonTestGraph)
        self._graph = graph
        self.mesh = graph.mesh
        self.dec = graph.dec
        self.p1_poisson = graph.p1_poisson


def p1_cook_test(ctx: CookContext) -> None:
    mods = P1Module(ctx)
    load_mesh_input(ctx, mods.mesh)
    boundary_mask = mods.dec.boundary_mask.get()
    boundary_idx = np.where(boundary_mask)[0]
    dof_pos = mods.mesh.V_pos.get()
    boundary_val = np.array(list(map(lambda x: x[0] * x[2], dof_pos[boundary_idx])))
    mods.p1_poisson.constrained_idx.set(boundary_idx.astype(np.int32))
    mods.p1_poisson.constrained_values.set(boundary_val.astype(np.float32))
    mods.p1_poisson.rhs.set(np.zeros(boundary_mask.shape, dtype=np.float64))
    mods.p1_poisson.psi.get()


def p1_cook2_test(ctx: CookContext) -> None:
    mods = P1Module(ctx)
    faceids, bary = read_probe_input(ctx, index=1)
    stream_func = mods.p1_poisson.interpolate((faceids, bary))
    ctx.write_point("stream_func", stream_func)


def p2_cook_test(ctx: CookContext) -> None:
    mods = P2Module(ctx)
    load_mesh_input(ctx, mods.mesh)
    boundary_mask = mods.p2_space.boundary_mask.get()
    boundary_idx = np.where(boundary_mask)[0]
    dof_pos = mods.p2_space.dof_pos.get()
    boundary_val = np.array(list(map(lambda x: x[0] * x[2], dof_pos[boundary_idx])))
    mods.p2_poisson.constrained_idx.set(boundary_idx.astype(np.int32))
    mods.p2_poisson.constrained_values.set(boundary_val.astype(np.float32))
    mods.p2_poisson.rhs.set(np.zeros(boundary_mask.shape, dtype=np.float64))
    mods.p2_poisson.psi.get()


def p2_cook2_test(ctx: CookContext) -> None:
    mods = P2Module(ctx)
    faceids, bary = read_probe_input(ctx, index=1)
    stream_func = mods.p2_poisson.interpolate((faceids, bary))
    ctx.write_point("stream_func", stream_func)
