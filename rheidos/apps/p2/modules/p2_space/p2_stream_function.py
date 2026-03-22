import numpy as np

from rheidos.apps.p2.modules.p2_space.p2_elements import P2Elements
from rheidos.apps.p2.modules.p2_space.probe_utils import probe_arrays
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ResourceSpec, shape_from_scalar
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import World


class P2StreamFunction(ModuleBase):
    NAME = "P2StreamFunction"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.point_vortex = self.require(PointVortexModule)
        self.p2_elements = self.require(P2Elements)

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

        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_scalar(self.p2_elements.n_dof),
            ),
            doc="Stream function coefficients for chosen p2 basis.",
        )

        self.omega = self.resource(
            "omega",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_scalar(self.p2_elements.n_dof),
            ),
            doc="Vorticity field coefficient to be paired with basis function. Shape: (nDof, )",
        )

        self.solve_cg = self.resource(
            "solve_cg",
            spec=ResourceSpec(kind="python"),
            doc="A pre-factorized poisson solver callable, just need the RHS",
        )

        self.bind_producers()

    @producer(inputs=("solve_cg", "omega"), outputs=("psi",))
    def solve_for_stream_func(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()
        solve_cg = self.solve_cg.get()
        omega = self.omega.get()

        ctx.commit(psi=solve_cg(omega))

    @producer(
        inputs=(
            "point_vortex.face_ids",
            "point_vortex.bary",
            "point_vortex.gamma",
            "p2_elements.face_dof",
        ),
        outputs=("omega",),
    )
    def splat_vortices(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()

        face_to_dof = self.p2_elements.face_dof.get()
        faceids = self.point_vortex.face_ids.get()
        bary = self.point_vortex.bary.get()
        gamma = self.point_vortex.gamma.get()
        omega = np.zeros_like(self.omega.peek())

        for idx, G in enumerate(gamma):
            fid = faceids[idx]
            v1, v2, v3, e1, e2, e3 = face_to_dof[fid]
            b1, b2, b3 = bary[idx]

            p1, p2, p3, p4, p5, p6 = self.p2_elements.basis_from_bary(b1, b2, b3)
            omega[v1] += p1 * G
            omega[v2] += p2 * G
            omega[v3] += p3 * G
            omega[e1] += p4 * G
            omega[e2] += p5 * G
            omega[e3] += p6 * G

        ctx.commit(omega=omega)

    @producer(
        inputs=(
            "p2_elements.L_stiffness",
            "p2_elements.n_dof",
            "constrained_idx",
            "constrained_values",
        ),
        outputs=("solve_cg",),
    )
    def build_cg_solver(self, ctx: ProducerContext) -> None:
        from scipy.sparse.linalg import LinearOperator, cg

        n_dof = self.p2_elements.n_dof.get()
        L = self.p2_elements.L_stiffness.get()

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
        face_dof = self.p2_elements.face_dof.get()
        faceids, bary = probe_arrays(probles)
        basis = np.stack(
            self.p2_elements.basis_from_bary(bary[:, 0], bary[:, 1], bary[:, 2]),
            axis=1,
        )
        return np.einsum("ij,ij->i", psi[face_dof[faceids]], basis)
