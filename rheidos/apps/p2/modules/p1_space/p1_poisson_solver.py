import numpy as np

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.probe_utils import probe_arrays
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ProducerContext, ResourceSpec, World, producer
from rheidos.compute import shape_map


class P1PoissonSolver(ModuleBase):
    NAME = "P1PoissonSolver"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        dec: DEC,
        scope: str = "",
        declare_rhs: bool = True,
        mode: str = "cg",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        self.dec = dec
        self.mode = mode

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
            spec=ResourceSpec(kind="numpy", dtype=np.float64),
            declare=True,
            doc="Values for the constrained vertices 1-1 mapped to the index in constrained_idx.",
        )

        self.rhs = self.resource(
            "rhs",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                allow_none=True,
                shape_fn=shape_map(self.mesh.V_pos, lambda shape: (shape[0],)),
            ),
            # Parents that embed this solver as a child can set
            # `declare_rhs=False` and produce directly into the same RHS
            # resource under an alias such as `omega`.
            declare=declare_rhs,
            doc="Vorticity field coefficient to be paired with basis function. Shape: (nV, )",
        )

        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                allow_none=True,
                shape_fn=shape_map(self.mesh.V_pos, lambda shape: (shape[0],)),
            ),
            doc="Stream function coefficient to be paired with respective hat basis function. Shape (nV, )",
        )

        self.L_cached = self.resource(
            "L_cached",
            spec=ResourceSpec(kind="python", allow_none=True),
            doc="Sparse SciPy scalar Laplacian matrix based on DEC * d *",
        )

        self.solve_cg = self.resource(
            "solve_cg",
            spec=ResourceSpec(kind="python"),
            doc="A Jacobi-conditioned poisson solver callable based on Conjugate Gradient, just need the RHS",
        )

        self.solve_cholesky = self.resource(
            "solve_cholesky",
            spec=ResourceSpec(kind="python"),
            doc="A pre-factorized poisson solver callable based on sparse cholesky, just need the RHS",
        )

        self.bind_producers()

    @producer(
        inputs=(
            "rhs",
            "mesh.V_pos",
            "mesh.E_verts",
            "dec.star1",
            "constrained_idx",
            "constrained_values",
        ),
        outputs=("psi",),
    )
    def solve_for_stream_func(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()

        if self.mode == "cg":
            rhs = self.rhs.get()
            ctx.commit(psi=np.asarray(self.solve_cg.get()(rhs), dtype=np.float64))
            return

        if self.mode == "cholesky":
            rhs = self.rhs.get()
            ctx.commit(psi=np.asarray(self.solve_cholesky.get()(rhs), dtype=np.float64))
            return

        raise ValueError(f"{self.mode} is not valid mode for the Poisson Solver")

    @producer(
        inputs=(
            "mesh.V_pos",
            "mesh.E_verts",
            "dec.star1",
            "constrained_idx",
            "constrained_values",
        ),
        outputs=("solve_cg",),
    )
    def build_scalar_laplacian_cg(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()

        import scipy.sparse as sp
        from scipy.sparse.linalg import LinearOperator, cg

        star1 = self.dec.star1.get()
        E = self.mesh.E_verts.get()
        V_pos = self.mesh.V_pos.get()

        i = E[:, 0].astype(np.int64, copy=False)
        j = E[:, 1].astype(np.int64, copy=False)
        ww = np.asarray(star1, dtype=np.float64).reshape(-1)

        rows = np.concatenate([i, j, i, j])
        cols = np.concatenate([i, j, j, i])
        data = np.concatenate([ww, ww, -ww, -ww])
        nV = V_pos.shape[0]

        L = sp.coo_matrix((data, (rows, cols)), shape=(nV, nV)).tocsr()

        constrained_idx = self.constrained_idx.get()
        constrained_values = self.constrained_values.get()
        is_constrained_mask = np.zeros(nV, dtype=bool)
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

            u = np.zeros(nV, dtype=np.float64)
            u[constrained_idx] = constrained_values
            u[free_idx] = u_free

            return u

        ctx.commit(solve_cg=solve)

    @producer(
        inputs=(
            "mesh.V_pos",
            "mesh.E_verts",
            "dec.star1",
            "constrained_idx",
            "constrained_values",
        ),
        outputs=("solve_cholesky",),
    )
    def build_scalar_laplacian_cholesky(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()

        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.linalg import spsolve_triangular
        from sksparse.cholmod import cholesky

        try:
            from sksparse.cholmod import cho_factor
        except ImportError:
            cho_factor = None

        # alternatively:
        # from sksparse.cholmod import cho_factor

        star1 = self.dec.star1.get()
        E = self.mesh.E_verts.get()
        V_pos = self.mesh.V_pos.get()

        i = E[:, 0].astype(np.int64, copy=False)
        j = E[:, 1].astype(np.int64, copy=False)
        ww = np.asarray(star1, dtype=np.float64).reshape(-1)

        rows = np.concatenate([i, j, i, j])
        cols = np.concatenate([i, j, j, i])
        data = np.concatenate([ww, ww, -ww, -ww])
        nV = V_pos.shape[0]

        # global scalar Laplacian
        L = sp.coo_matrix((data, (rows, cols)), shape=(nV, nV)).tocsr()

        constrained_idx = np.asarray(self.constrained_idx.get(), dtype=np.int64)
        constrained_values = np.asarray(self.constrained_values.get(), dtype=np.float64)

        is_constrained_mask = np.zeros(nV, dtype=bool)
        is_constrained_mask[constrained_idx] = True
        free_idx = np.nonzero(~is_constrained_mask)[0]

        # reduced SPD system on free DOFs
        L_II = L[free_idx][:, free_idx].tocsc()
        L_IB = L[free_idx][:, constrained_idx].tocsc()

        # Optional symmetry cleanup for numerical safety
        # CHOLMOD uses only one triangle of the matrix, so it is good to make
        # the reduced block explicitly symmetric before factorization.
        L_II = 0.5 * (L_II + L_II.T)

        # scikit-sparse 0.4.x exposes a callable Factor via `cholesky(...)`,
        # while 0.5.x moved the solver object behind `cho_factor(...)` and made
        # `cholesky(...)` return the raw triangular factor (and permutation).
        if cho_factor is not None:
            try:
                factor = cho_factor(L_II, order="default")
            except TypeError as exc:
                if "order" not in str(exc):
                    raise
                factor = cho_factor(L_II)
        else:
            try:
                factor = cholesky(L_II, ordering_method="default")
            except TypeError as exc:
                if "ordering_method" not in str(exc):
                    raise
                factor = cholesky(L_II)
        # or factor = cho_factor(L_II)

        def solve_factor(rhs: np.ndarray) -> np.ndarray:
            if callable(factor):
                return np.asarray(factor(rhs), dtype=np.float64)

            solve = getattr(factor, "solve", None)
            if callable(solve):
                return np.asarray(solve(rhs), dtype=np.float64)

            solve_A = getattr(factor, "solve_A", None)
            if callable(solve_A):
                return np.asarray(solve_A(rhs), dtype=np.float64)

            # Some newer APIs return the triangular factor and permutation
            # directly instead of a solver object.
            if isinstance(factor, tuple) and factor:
                triangular = factor[0]
                perm = (
                    None if len(factor) < 2 else np.asarray(factor[1], dtype=np.int64)
                )

                if perm is None:
                    perm_rhs = rhs
                else:
                    perm_rhs = rhs[perm]

                y = spsolve_triangular(triangular.T, perm_rhs, lower=True)
                z = spsolve_triangular(triangular, y, lower=False)
                z = np.asarray(z, dtype=np.float64).reshape(-1)

                if perm is None:
                    return z

                out = np.empty_like(z)
                out[perm] = z
                return out

            raise TypeError(
                "Unsupported CHOLMOD factor result; expected a callable factor, "
                "a factor object with solve/solve_A, or a (triangular, perm) tuple."
            )

        def solve(b):
            b = np.asarray(b, dtype=np.float64)
            rhs = b[free_idx] - L_IB @ constrained_values

            # solve L_II u_free = rhs
            u_free = solve_factor(rhs)

            u = np.zeros(nV, dtype=np.float64)
            u[constrained_idx] = constrained_values
            u[free_idx] = np.asarray(u_free, dtype=np.float64).reshape(-1)
            return u

        ctx.commit(
            solve_cholesky=solve,
        )

    def interpolate(self, probes) -> np.ndarray:
        """Interpolates the value of `psi` based on the P1 basis.

        Args:
            probes:
                Either an iterable of ``(faceid, bary)`` pairs or a
                ``(faceids, bary)`` tuple of arrays.

        Returns:
            np.ndarray: Values at the probe locations
        """
        psi = self.psi.get()
        F_verts = self.mesh.F_verts.get()

        if isinstance(probes, tuple) and len(probes) == 2:
            faceids = np.asarray(probes[0], dtype=np.int64)
            bary = np.asarray(probes[1], dtype=np.float64)
        else:
            faceids, bary = probe_arrays(probes)

        return np.einsum("ij,ij->i", psi[F_verts[faceids]], bary)

    def set_homo_dirichlet_boundary(self) -> None:
        boundary_mask = self.dec.boundary_mask.get()
        boundary_dofs = np.where(boundary_mask)[0]
        if boundary_dofs.size == 0:
            boundary_dofs = np.array([0], dtype=np.int32)

        self.constrained_idx.set(boundary_dofs.astype(np.int32))
        self.constrained_values.set(np.zeros(boundary_dofs.shape, dtype=np.float64))
