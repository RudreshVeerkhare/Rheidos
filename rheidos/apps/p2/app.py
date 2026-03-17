from rheidos.apps.p2.modules.p1_space import P1StreamFunction
from rheidos.houdini.runtime.cook_context import CookContext

from .modules import SurfaceMeshModule, PointVortexModule
import numpy as np


class P2Module:
    def __init__(self, ctx: CookContext) -> None:
        world = ctx.world()
        self.mesh = world.require(SurfaceMeshModule)
        self.point_vortex = world.require(PointVortexModule)
        self.p1_stream = world.require(P1StreamFunction)


def cook2(ctx: CookContext) -> None:
    mods = P2Module(ctx)

    probe_io = ctx.input_io(1)
    if probe_io is None:
        raise RuntimeError("Input 1 is not set")

    faceids = np.array(probe_io.read_point("faceid"), dtype=np.int32)
    bary = np.array(probe_io.read_point("bary", components=3), dtype=np.float32)

    stream_func = mods.p1_stream.interpolate(list(zip(faceids, bary)))

    ctx.write_point("stream_func", stream_func)


def cook(ctx: CookContext) -> None:
    mods = P2Module(ctx)

    # Read and load mesh from Houdini
    mesh_io = ctx.input_io(0)
    if not mesh_io:
        raise RuntimeError(f"Input 0 is not set")

    points = np.array(mesh_io.read_point("P", components=3), dtype=np.float32)
    triangles = np.array(mesh_io.read_prims(arity=3), dtype=np.int32)

    mods.mesh.set_mesh(points, triangles)

    # Read and load point vortices data
    vort_io = ctx.input_io(1)
    if not vort_io:
        raise RuntimeError("Input 1 is not set")

    vortex_pos = np.array(vort_io.read_point("P", components=3), dtype=np.float32)
    vortex_bary = np.array(vort_io.read_point("bary", components=3), dtype=np.float32)
    vortex_gammma = np.array(vort_io.read_point("gamma"), dtype=np.float32)
    vortex_faceid = np.array(vort_io.read_point("faceid"), dtype=np.int32)
    mods.point_vortex.set_vortex(vortex_faceid, vortex_bary, vortex_gammma, vortex_pos)

    # Splat vortices
    omega = mods.p1_stream.omega.get()

    # Set Dirichlet Pin
    mods.p1_stream.constrained_idx.set(np.array([0], dtype=np.int32))
    mods.p1_stream.constrained_values.set(np.array([0], dtype=np.float32))

    # Solve for Stream function
    psi = mods.p1_stream.psi.get()

    # Export Stream function
    ctx.write_point("stream_func", psi)

    print("Herer")
