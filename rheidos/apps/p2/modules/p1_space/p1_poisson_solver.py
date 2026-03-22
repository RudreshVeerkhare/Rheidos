import numpy as np

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.probe_utils import probe_arrays
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ProducerContext, ResourceSpec, World, producer
from rheidos.compute import shape_map


class P1PoissonSolver(ModuleBase):
    NAME = "P1PoissonSolver"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(DEC)

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
                allow_none=True,
                shape_fn=shape_map(self.mesh.V_pos, lambda shape: (shape[0],)),
            ),
            declare=True,
            doc="Vorticity field coefficient to be paired with basis function. Shape: (nV, )",
        )

        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float32,
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
            doc="A pre-factorized poisson solver callable, just need the RHS",
        )

        self.bind_producers()

    @producer(inputs=("solve_cg", "rhs"), outputs=("psi",))
    def solve_for_stream_func(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()
        solve_cg = self.solve_cg.get()
        rhs = self.rhs.get()

        ctx.commit(psi=solve_cg(rhs).astype(np.float32))

    @producer(
        inputs=(
            "mesh.V_pos",
            "mesh.E_verts",
            "dec.star1",
            "constrained_idx",
            "constrained_values",
        ),
        outputs=("L_cached", "solve_cg"),
    )
    def build_scalar_laplacian(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()

        import scipy.sparse as sp
        from scipy.sparse.linalg import LinearOperator, cg

        star1 = self.dec.star1.get()
        E = self.mesh.E_verts.get()
        V_pos = self.mesh.V_pos.get()

        i = E[:, 0].astype(np.int64, copy=False)
        j = E[:, 1].astype(np.int64, copy=False)
        ww = np.asarray(star1, dtype=np.float32).reshape(-1)

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

        ctx.commit(L_cached=L, solve_cg=solve)

    def interpolate(self, probes) -> np.ndarray:
        """Interpolates the value of `psi` based on P1 lagrange basis

        Args:
            probes (np.ndarray): [[faceid, [b1, b2, b3]], ...]

        Returns:
            np.ndarray: Values at the probe locations
        """
        psi = self.psi.get()
        F_verts = self.mesh.F_verts.get()
        faceids, bary = probe_arrays(probes)
        return np.einsum("ij,ij->i", psi[F_verts[faceids]], bary)
