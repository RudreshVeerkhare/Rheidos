from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.p2_space.p2_elements import P2Elements
from rheidos.apps.p2.modules.p2_space.p2_poisson_solver import P2PoissonSolver
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.houdini.runtime.cook_context import CookContext

import numpy as np


class P2Module:
    def __init__(self, ctx: CookContext) -> None:
        world = ctx.world()
        self.mesh = world.require(SurfaceMeshModule)
        self.p2_space = world.require(P2Elements)
        self.p2_poisson = world.require(P2PoissonSolver)


class P1Module:
    def __init__(self, ctx: CookContext) -> None:
        world = ctx.world()
        self.mesh = world.require(SurfaceMeshModule)
        self.dec = world.require(DEC)
        self.p1_poisson = world.require(P1PoissonSolver)


def p1_cook_test(ctx: CookContext) -> None:
    mods = P1Module(ctx)

    # Read and load mesh from Houdini
    mesh_io = ctx.input_io(0)
    if not mesh_io:
        raise RuntimeError(f"Input 0 is not set")

    points = np.array(mesh_io.read_point("P", components=3), dtype=np.float32)
    triangles = np.array(mesh_io.read_prims(arity=3), dtype=np.int32)

    mods.mesh.set_mesh(points, triangles)

    # Get Boundary Dofs
    boundary_mask = mods.dec.boundary_mask.get()
    boundary_idx = np.where(boundary_mask == True)[0]

    # Calculate function value on boundary $y = xz$
    dof_pos = mods.mesh.V_pos.get()
    boundary_val = np.array(list(map(lambda x: x[0] * x[2], dof_pos[boundary_idx])))

    # Set Dirichlet boundary condition
    mods.p1_poisson.constrained_idx.set(boundary_idx.astype(np.int32))
    mods.p1_poisson.constrained_values.set(boundary_val.astype(np.float32))
    mods.p1_poisson.rhs.set(np.zeros(boundary_mask.shape, dtype=np.float64))

    # Solve for stream function
    mods.p1_poisson.psi.get()


def p1_cook2_test(ctx: CookContext) -> None:
    mods = P1Module(ctx)

    probe_io = ctx.input_io(1)
    if probe_io is None:
        raise RuntimeError("Input 1 is not set")

    faceids = np.array(probe_io.read_point("faceid"), dtype=np.int32)
    bary = np.array(probe_io.read_point("bary", components=3), dtype=np.float32)

    stream_func = mods.p1_poisson.interpolate(list(zip(faceids, bary)))

    ctx.write_point("stream_func", stream_func)


def p2_cook_test(ctx: CookContext) -> None:
    mods = P2Module(ctx)

    # Read and load mesh from Houdini
    mesh_io = ctx.input_io(0)
    if not mesh_io:
        raise RuntimeError(f"Input 0 is not set")

    points = np.array(mesh_io.read_point("P", components=3), dtype=np.float32)
    triangles = np.array(mesh_io.read_prims(arity=3), dtype=np.int32)

    mods.mesh.set_mesh(points, triangles)

    # Get Boundary Dofs
    boundary_mask = mods.p2_space.boundary_mask.get()
    boundary_idx = np.where(boundary_mask == True)[0]

    # Calculate function value on boundary $y = xz$
    dof_pos = mods.p2_space.dof_pos.get()
    boundary_val = np.array(list(map(lambda x: x[0] * x[2], dof_pos[boundary_idx])))

    # Set Dirichlet boundary condition
    mods.p2_poisson.constrained_idx.set(boundary_idx.astype(np.int32))
    mods.p2_poisson.constrained_values.set(boundary_val.astype(np.float32))
    mods.p2_poisson.rhs.set(np.zeros(boundary_mask.shape, dtype=np.float64))

    # Solve for stream function
    mods.p2_poisson.psi.get()


def p2_cook2_test(ctx: CookContext) -> None:
    mods = P2Module(ctx)

    probe_io = ctx.input_io(1)
    if probe_io is None:
        raise RuntimeError("Input 1 is not set")

    faceids = np.array(probe_io.read_point("faceid"), dtype=np.int32)
    bary = np.array(probe_io.read_point("bary", components=3), dtype=np.float32)

    stream_func = mods.p2_poisson.interpolate(list(zip(faceids, bary)))

    ctx.write_point("stream_func", stream_func)
