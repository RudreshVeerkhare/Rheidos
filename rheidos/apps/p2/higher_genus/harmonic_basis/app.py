import numpy as np

from rheidos.apps.p2._io import load_mesh_input
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.apps.p2.modules.tree_cotree.tree_cotree_module import TreeCotreeModule
from rheidos.compute import shape_map
from rheidos.compute.resource import ResourceSpec
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import ModuleBase, World
from rheidos.houdini.runtime.cook_context import CookContext


def _basis_vertex_shape(forms_ref, vertices_ref):
    def shape_fn(reg):
        forms = reg.read(forms_ref.name, ensure=False)
        vertices = reg.read(vertices_ref.name, ensure=False)
        if forms is None or vertices is None:
            return None
        if not hasattr(forms, "shape") or not hasattr(vertices, "shape"):
            return None
        return (int(forms.shape[0]), int(vertices.shape[0]))

    return shape_fn


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


class HarmonicBasis(ModuleBase):
    def __init__(
        self,
        world: World,
        *,
        dec: DEC,
        tree_cotree: TreeCotreeModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.tree_cotree = tree_cotree
        self.mesh = tree_cotree.mesh
        self.dec = dec

        self.omegas = self.resource(
            "omegas",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(
                    self.tree_cotree.closed_dual_generator_1forms,
                    lambda s: (s[0], s[1]),
                ),
            ),
            doc=(
                "Closed generator 1-forms used to extract harmonic representatives. "
                "Shape: (2g,nE)"
            ),
        )

        self.poisson = self.require(
            P1PoissonSolver,
            child=True,
            child_name="poisson",
            mesh=self.tree_cotree.mesh,
            dec=self.dec,
            declare_rhs=False,
        )
        # Closed surfaces have a one-dimensional scalar Laplacian nullspace.
        # Pinning one vertex fixes only that gauge; it does not change d0(alpha).
        self.poisson.constrained_idx.set(np.array([0], dtype=np.int32))
        self.poisson.constrained_values.set(np.array([0.0], dtype=np.float64))

        self.beta = self.resource(
            "beta",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=_basis_vertex_shape(
                    self.tree_cotree.closed_dual_generator_1forms,
                    self.mesh.V_pos,
                ),
            ),
            doc=(
                "Poisson RHS d0^T star1 omega for each generator 1-form. "
                "Shape: (2g,nV)"
            ),
        )

        self.alpha = self.resource(
            "alpha",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.beta, lambda s: (s[0], s[1])),
            ),
            doc=(
                "Scalar correction potentials solving "
                "d0^T star1 d0 alpha = d0^T star1 omega. Shape: (2g,nV)"
            ),
        )

        self.gamma = self.resource(
            "gamma",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.omegas, lambda s: (s[0], s[1])),
            ),
            doc=(
                "Harmonic 1-form representatives for the tree-cotree generator "
                "forms. Shape: (2g,nE)"
            ),
        )

        self.bind_producers()

    @producer(inputs=("tree_cotree.closed_dual_generator_1forms",), outputs=("omegas",))
    def build_omegas(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.commit(
            omegas=np.asarray(
                self.tree_cotree.closed_dual_generator_1forms.get(),
                dtype=np.float64,
            )
        )

    @producer(
        inputs=("omegas", "dec.star1", "mesh.E_verts", "mesh.V_pos"),
        outputs=("beta",),
    )
    def build_beta(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        star1 = self.dec.star1.get()
        omegas = self.omegas.get()

        # The scalar solve uses L = d0^T star1 d0, so the projection RHS must
        # live on vertices: d0^T star1 omega.
        beta = self.dec.d0_transpose(omegas * star1[None, :])
        ctx.commit(beta=beta)

    @producer(inputs=("beta", "poisson.solve_cg"), outputs=("alpha",))
    def solve_alpha(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        beta = self.beta.get()
        solve = self.poisson.solve_cg.get()

        alpha = np.empty_like(beta)
        for basis_id, rhs in enumerate(beta):
            alpha[basis_id] = solve(rhs)

        ctx.commit(alpha=alpha)

    @producer(inputs=("omegas", "alpha", "mesh.E_verts"), outputs=("gamma",))
    def assemble_gamma(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        omegas = self.omegas.get()
        alpha = self.alpha.get()
        e_verts = self.mesh.E_verts.get()

        # Subtracting an exact form does not change the cohomology class of the
        # closed tree-cotree form. The solved correction removes its coexact
        # part, leaving the harmonic representative of that generator class.
        exact = alpha[:, e_verts[:, 1]] - alpha[:, e_verts[:, 0]]
        gamma = omegas - exact
        ctx.commit(gamma=gamma)


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
