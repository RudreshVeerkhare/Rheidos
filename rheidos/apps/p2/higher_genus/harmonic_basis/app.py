from rheidos.apps.p2._io import load_mesh_input
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.apps.p2.modules.tree_cotree.tree_cotree_module import TreeCotreeModule
from rheidos.compute.world import ModuleBase, World
from rheidos.houdini.runtime.cook_context import CookContext


class App(ModuleBase):
    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(DEC, mesh=self.mesh)
        self.tree_cotree = self.require(TreeCotreeModule, mesh=self.mesh)


def setup_mesh(ctx: CookContext):
    mods = ctx.world().require(App)
    load_mesh_input(
        ctx, mods.mesh, missing_message="Input 0 has to be mesh input geometry"
    )


def tree_cotree(ctx: CookContext):
    mods = ctx.world().require(App)

    print("Hello")
