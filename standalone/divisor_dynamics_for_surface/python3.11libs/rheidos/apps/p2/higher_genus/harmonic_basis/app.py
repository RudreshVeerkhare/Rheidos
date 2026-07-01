import numpy as np

from rheidos.apps.p2._io import load_mesh_input, read_probe_input
from rheidos.apps.p2.modules.higher_genus.harmonic_basis import HarmonicBasis
from rheidos.apps.p2.modules.higher_genus.tree_cotree import TreeCotreeModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.whitney_1form import Whitney1FormInterpolator
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute.world import ModuleBase, World
from rheidos.houdini.runtime.cook_context import CookContext


class App(ModuleBase):
    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(DEC, mesh=self.mesh)
        self.tree_cotree = self.require(TreeCotreeModule, mesh=self.mesh)
        self.harmonic_basis = self.require(
            HarmonicBasis,
            dec=self.dec,
            tree_cotree=self.tree_cotree,
        )
        self.whitney_1form = self.require(
            Whitney1FormInterpolator,
            mesh=self.mesh,
        )


def setup_mesh(ctx: CookContext):
    mods = ctx.world().require(App)
    load_mesh_input(
        ctx, mods.mesh, missing_message="Input 0 has to be mesh input geometry"
    )


def tree_cotree(ctx: CookContext):
    mods = ctx.world().require(App)
    load_mesh_input(
        ctx, mods.mesh, missing_message="Input 0 has to be mesh input geometry"
    )

    gamma = mods.harmonic_basis.gamma.get()
    ctx.write_detail(
        "genus",
        np.array([mods.tree_cotree.genus.get()], dtype=np.int32),
        create=True,
    )
    ctx.write_detail(
        "generator_count",
        np.array([mods.tree_cotree.generator_count.get()], dtype=np.int32),
        create=True,
    )
    ctx.write_detail(
        "harmonic_basis_count",
        np.array([gamma.shape[0]], dtype=np.int32),
        create=True,
    )


def _eval_parm_int(node, name: str, default: int) -> int:
    if node is None:
        return default
    parm = node.parm(name)
    if parm is None:
        return default
    try:
        return int(parm.eval())
    except Exception:
        return default


def _write_harmonic_basis_detail(
    ctx: CookContext,
    mods: App,
    *,
    basis_id: int,
    basis_count: int,
) -> None:
    ctx.write_detail(
        "genus",
        np.array([mods.tree_cotree.genus.get()], dtype=np.int32),
        create=True,
    )
    ctx.write_detail(
        "generator_count",
        np.array([mods.tree_cotree.generator_count.get()], dtype=np.int32),
        create=True,
    )
    ctx.write_detail(
        "harmonic_basis_count",
        np.array([basis_count], dtype=np.int32),
        create=True,
    )
    ctx.write_detail(
        "harmonic_basis_id",
        np.array([basis_id], dtype=np.int32),
        create=True,
    )


def interpolate_harmonic_basis_velocity(
    ctx: CookContext,
    basis_id=None,
) -> None:
    mods = ctx.world().require(App)
    if basis_id is None:
        basis_id = _eval_parm_int(getattr(ctx, "node", None), "basis_id", 0)
    basis_id = int(basis_id)

    faceids, bary = read_probe_input(
        ctx,
        index=1,
        missing_message="Input 1 has to be probe point geometry",
    )

    gamma = mods.harmonic_basis.gamma.get()
    basis_count = int(gamma.shape[0])
    if basis_id < 0 or basis_id >= basis_count:
        raise ValueError(
            f"basis_id {basis_id} is out of range for {basis_count} "
            "harmonic basis forms"
        )

    velocity = mods.whitney_1form.interpolate(gamma[basis_id], (faceids, bary))
    ctx.write_point("harmonic_basis_vel", velocity)
    _write_harmonic_basis_detail(
        ctx,
        mods,
        basis_id=basis_id,
        basis_count=basis_count,
    )
