from rheidos.apps.p2.modules.p2_space.p2_elements import P2Elements
from rheidos.apps.p2.modules.p2_space.p2_stream_function import P2StreamFunction
from rheidos.apps.p2.modules.p2_space.p2_velocity import P2VelocityField
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.houdini.runtime.cook_context import CookContext

import numpy as np


class P2Module:
    def __init__(self, ctx: CookContext) -> None:
        world = ctx.world()
        self.mesh = world.require(SurfaceMeshModule)
        self.point_vortex = world.require(PointVortexModule)
        self.p2_space = world.require(P2Elements)
        self.p2_stream = world.require(P2StreamFunction)
        self.p2_vel = world.require(P2VelocityField)


def p2_cook(ctx: CookContext) -> None:
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
    vortex_gamma = np.array(vort_io.read_point("gamma"), dtype=np.float32)
    vortex_faceid = np.array(vort_io.read_point("faceid"), dtype=np.int32)
    mods.point_vortex.set_vortex(vortex_faceid, vortex_bary, vortex_gamma, vortex_pos)

    # Set Dirichlet Pin
    mods.p2_stream.constrained_idx.set(np.array([0], dtype=np.int32))
    mods.p2_stream.constrained_values.set(np.array([0], dtype=np.float32))

    # Solve for stream function
    mods.p2_stream.psi.get()


def p2_cook2(ctx: CookContext) -> None:
    mods = P2Module(ctx)

    probe_io = ctx.input_io(1)
    if probe_io is None:
        raise RuntimeError("Input 1 is not set")

    faceids = np.array(probe_io.read_point("faceid"), dtype=np.int32)
    bary = np.array(probe_io.read_point("bary", components=3), dtype=np.float32)

    stream_func = mods.p2_stream.interpolate((faceids, bary))

    ctx.write_point("stream_func", stream_func)


def p2_interpolate_velocity(ctx: CookContext):
    mods = P2Module(ctx)

    probe_io = ctx.input_io(1)
    if probe_io is None:
        raise RuntimeError("Input 1 is not set")

    faceids = np.array(probe_io.read_point("faceid"), dtype=np.int32)
    bary = np.array(probe_io.read_point("bary", components=3), dtype=np.float32)

    vel = mods.p2_vel.interpolate((faceids, bary))

    ctx.write_point("vel", vel)
