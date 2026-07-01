import numpy as np

from rheidos.apps.p2.modules.higher_genus.tree_cotree import TreeCotreeModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.compute import shape_map
from rheidos.compute.resource import ResourceSpec
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import ModuleBase, World


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


def _basis_gram_shape(forms_ref):
    def shape_fn(reg):
        forms = reg.read(forms_ref.name, ensure=False)
        if forms is None or not hasattr(forms, "shape"):
            return None
        return (int(forms.shape[0]), int(forms.shape[0]))

    return shape_fn


def _orthonormalize_rows_l2(
    raw_gamma: np.ndarray,
    star1: np.ndarray,
    *,
    rtol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormalize row-stacked 1-forms with the diagonal star1 inner product."""
    raw_gamma = np.asarray(raw_gamma, dtype=np.float64)
    star1 = np.asarray(star1, dtype=np.float64)

    gram = (raw_gamma * star1[None, :]) @ raw_gamma.T
    gram = 0.5 * (gram + gram.T)
    if raw_gamma.shape[0] == 0:
        return raw_gamma.copy(), gram

    eigvals = np.linalg.eigvalsh(gram)
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    tol = rtol * scale
    min_eig = float(np.min(eigvals))
    if min_eig <= tol:
        raise ValueError(
            "Harmonic basis Gram matrix is not positive definite under the "
            f"DEC 1-form inner product; min eigenvalue {min_eig:.6e}, "
            f"tolerance {tol:.6e}."
        )

    # For row-stacked forms, solve(L, raw_gamma) is Gamma @ L^{-T} in the
    # usual column-stacked notation, with G = L L^T.
    chol = np.linalg.cholesky(gram)
    gamma = np.linalg.solve(chol, raw_gamma)
    return gamma, gram


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
            doc="Closed generator 1-forms. Shape: (2g,nE)",
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
        # Pinning one vertex fixes only that gauge; d0(alpha) is unchanged.
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
            doc="Poisson RHS d0^T star1 omega. Shape: (2g,nV)",
        )

        self.alpha = self.resource(
            "alpha",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.beta, lambda s: (s[0], s[1])),
            ),
            doc="Scalar correction potentials. Shape: (2g,nV)",
        )

        self.raw_gamma = self.resource(
            "raw_gamma",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.omegas, lambda s: (s[0], s[1])),
            ),
            doc="Harmonic representatives before L2 orthonormalization. Shape: (2g,nE)",
        )

        self.l2_gram = self.resource(
            "l2_gram",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=_basis_gram_shape(self.raw_gamma),
            ),
            doc="DEC 1-form Gram matrix of raw_gamma. Shape: (2g,2g)",
        )

        self.gamma = self.resource(
            "gamma",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.raw_gamma, lambda s: (s[0], s[1])),
            ),
            doc="L2-orthonormal harmonic 1-form basis. Shape: (2g,nE)",
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

        # The scalar solve uses L = d0^T star1 d0, so the projection RHS is
        # d0^T star1 omega on vertices.
        ctx.commit(beta=self.dec.d0_transpose(omegas * star1[None, :]))

    @producer(inputs=("beta", "poisson.solve_cg"), outputs=("alpha",))
    def solve_alpha(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        beta = self.beta.get()
        solve = self.poisson.solve_cg.get()

        alpha = np.empty_like(beta)
        for basis_id, rhs in enumerate(beta):
            alpha[basis_id] = solve(rhs)

        ctx.commit(alpha=alpha)

    @producer(inputs=("omegas", "alpha", "mesh.E_verts"), outputs=("raw_gamma",))
    def assemble_raw_gamma(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        omegas = self.omegas.get()
        alpha = self.alpha.get()
        e_verts = self.mesh.E_verts.get()

        # Subtracting an exact form keeps the cohomology class fixed while
        # removing the coexact component selected by the Poisson solve.
        exact = alpha[:, e_verts[:, 1]] - alpha[:, e_verts[:, 0]]
        ctx.commit(raw_gamma=omegas - exact)

    @producer(inputs=("raw_gamma", "dec.star1"), outputs=("gamma", "l2_gram"))
    def orthonormalize_gamma(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        gamma, l2_gram = _orthonormalize_rows_l2(
            self.raw_gamma.get(),
            self.dec.star1.get(),
        )
        ctx.commit(gamma=gamma, l2_gram=l2_gram)
