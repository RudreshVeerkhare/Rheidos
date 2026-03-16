from rheidos.houdini.runtime.cook_context import CookContext

from .modules import SurfaceMeshModule
import numpy as np


class P2Module:
    def __init__(self, ctx: CookContext) -> None:
        world = ctx.world()
        self.mesh = world.require(SurfaceMeshModule)


def cook(ctx: CookContext) -> None:
    mods = P2Module(ctx)

    # Read and load mesh from Houdini
    mesh_io = ctx.input_io(0)
    if not mesh_io:
        raise RuntimeError(f"Input 0 is not set")

    points = np.array(mesh_io.read_point("P", components=3), dtype=np.float32)
    triangles = np.array(mesh_io.read_prims(arity=3), dtype=np.int32)
    
    mods.mesh.set_mesh(points, triangles)

    print("Herer")
