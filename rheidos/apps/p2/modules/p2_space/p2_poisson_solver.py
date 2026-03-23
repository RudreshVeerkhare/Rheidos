import numpy as np

from rheidos.apps.p2.modules.p2_space.p2_elements import P2Elements
from rheidos.apps.p2.modules.p2_space.probe_utils import probe_arrays
from rheidos.compute import ModuleBase, ResourceSpec, shape_from_scalar
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import World


class P2PoissonSolver(ModuleBase):
    NAME = "P2PoissonSolver"

    def __init__(
        self,
        world: World,
        *,
        scope: str = "",
        declare_rhs: bool = True,
    ) -> None:
        super().__init__(world, scope=scope)

        # These plain requires intentionally use the module's lookup scope.
        # That lets the solver work both as:
        # - a standalone module, and
        # - a child module nested under another module while still sharing
        #   the parent's P2 element space.
        self.p2_space = self.require(P2Elements)

        self.constrained_idx = self.resource(
            "constrained_idx",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
            ),
            declare=True,
            doc="Indices of constrained vertices.",
        )
        self.constrained_values = self.resource(
            "constrained_values",
            spec=ResourceSpec(kind="numpy", dtype=np.float32),
            declare=True,
            doc="Values for the constrained vertices 1-1 mapped to the index in constrained_idx.",
        )
        self.rhs = self.resource(
            "rhs",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_scalar(self.p2_space.n_dof),
            ),
            # Parents that embed this solver as a child can set
            # `declare_rhs=False` and produce directly into the same RHS
            # resource under an alias such as `omega`.
            declare=declare_rhs,
            doc="RHS coefficient to be paired with basis function. Shape: (nDof, )",
        )

        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_scalar(self.p2_space.n_dof),
            ),
            doc="Stream function coefficients for chosen p2 basis.",
        )

        self.solve_cg = self.resource(
            "solve_cg",
            spec=ResourceSpec(kind="python"),
            doc="A pre-factorized poisson solver callable, just need the RHS",
        )

        self.bind_producers()

    @producer(inputs=("solve_cg", "rhs"), outputs=("psi",))
    def solve_for_psi(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()
        solve_cg = self.solve_cg.get()
        rhs = self.rhs.get()

        ctx.commit(psi=solve_cg(rhs))

    @producer(
        inputs=(
            "p2_space.L_stiffness",
            "p2_space.n_dof",
            "constrained_idx",
            "constrained_values",
        ),
        outputs=("solve_cg",),
    )
    def build_cg_solver(self, ctx: ProducerContext) -> None:
        from scipy.sparse.linalg import LinearOperator, cg

        n_dof = self.p2_space.n_dof.get()
        L = self.p2_space.L_stiffness.get()

        constrained_idx = self.constrained_idx.get()
        constrained_values = self.constrained_values.get()
        is_constrained_mask = np.zeros(n_dof, dtype=bool)
        is_constrained_mask[constrained_idx] = True
        free_idx = np.nonzero(~is_constrained_mask)[0]

        L_II = L[free_idx][:, free_idx].tocsr()
        L_IB = L[free_idx][:, constrained_idx].tocsr()

        diag = L_II.diagonal().astype(np.float64)
        inv_diag = np.zeros_like(diag)
        nz = np.abs(diag) > 1e-14
        inv_diag[nz] = 1.0 / diag[nz]
        M = LinearOperator(
            shape=L_II.shape, matvec=lambda x: inv_diag * x, dtype=np.float64
        )

        def solve(b, x0=None, rtol=1e-8, atol=0.0, maxiter=None):
            b = np.asarray(b, dtype=np.float64)
            rhs = b[free_idx] - L_IB @ constrained_values

            x0_free = None
            if x0 is not None:
                x0 = np.array(x0, dtype=np.float64)
                x0_free = x0[free_idx]

            u_free, info = cg(
                L_II, rhs, x0=x0_free, rtol=rtol, atol=atol, maxiter=maxiter, M=M
            )

            if info != 0:
                raise RuntimeError(f"CG did not converge, info={info}")

            u = np.zeros(n_dof, dtype=np.float64)
            u[constrained_idx] = constrained_values
            u[free_idx] = u_free

            return u

        ctx.commit(solve_cg=solve)

    def interpolate(self, probles):
        """Interpolates the value of `psi` using P2 lagrange basis

        Args:
           probes (np.ndarray): [[faceid, [b1, b2, b3]], ...]
        """
        psi = self.psi.get()
        face_dof = self.p2_space.face_dof.get()
        faceids, bary = probe_arrays(probles)
        basis = np.stack(
            self.p2_space.basis_from_bary(bary[:, 0], bary[:, 1], bary[:, 2]), axis=1
        )
        return np.einsum("ij,ij->i", psi[face_dof[faceids]], basis)
