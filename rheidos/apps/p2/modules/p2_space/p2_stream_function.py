import numpy as np

from rheidos.apps.p2.modules.p2_space.p2_elements import P2Elements
from rheidos.apps.p2.modules.p2_space.p2_poisson_solver import P2PoissonSolver
from rheidos.apps.p2.modules.p2_space.probe_utils import probe_arrays
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import World
from rheidos.compute import ModuleBase, ResourceSpec, shape_from_scalar


import scipy.sparse.linalg as spla


class P2StreamFunction(ModuleBase):
    NAME = "P2StreamFunction"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        point_vortex: PointVortexModule,
        p2_elements: P2Elements,
        poisson: P2PoissonSolver,
        scope: str = "",
        regularize: bool = True,
    ) -> None:
        super().__init__(world, scope=scope)

        self.regularize = regularize

        self.mesh = mesh
        self.point_vortex = point_vortex
        self.p2_elements = p2_elements

        self.eps = self.resource(
            "eps",
            spec=ResourceSpec(kind="python"),
            doc="eps for the heat based diffusion of the stream function",
            declare=True,
            buffer=1e-2,
        )

        self.poisson = poisson

        # Re-export the solver's public resources so the wrapper stays the
        # composition-facing facade.
        self.p2_elements = self.poisson.p2_space
        self.constrained_idx = self.poisson.constrained_idx
        self.constrained_values = self.poisson.constrained_values
        self.omega = self.poisson.rhs
        self.psi_raw = self.poisson.psi
        self.solve_cg = self.poisson.solve_cg

        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_scalar(self.p2_elements.n_dof),
            ),
            doc="Stream function coefficients for chosen P2 basis.",
        )
        self.bind_producers()

    @producer(
        inputs=(
            "psi_raw",
            "p2_elements.M_mass",
            "p2_elements.L_stiffness",
            "constrained_idx",
            "constrained_values",
            "eps",
        ),
        outputs=("psi",),
    )
    def regularize_psi(self, ctx: ProducerContext):
        ctx.require_inputs()

        if not self.regularize:
            psi = self.psi_raw.get()
            ctx.commit(psi=psi)
            return

        # Regularize
        K = self.p2_elements.L_stiffness.get()
        M = self.p2_elements.M_mass.get()
        psi_non_reg = self.psi_raw.get()
        constrained_idx = self.constrained_idx.get()
        constrained_values = np.asarray(self.constrained_values.get(), dtype=np.float64)
        eps = self.eps.get()

        A = (M + 0.5 * eps * K).tocsc()
        rhs = M @ psi_non_reg

        n_dof = self.p2_elements.n_dof.get()
        is_constrained_mask = np.zeros(n_dof, dtype=bool)
        is_constrained_mask[constrained_idx] = True
        free_idx = np.nonzero(~is_constrained_mask)[0]

        psi_reg = np.zeros(n_dof, dtype=np.float64)
        psi_reg[constrained_idx] = constrained_values

        if free_idx.size > 0:
            A_II = A[free_idx][:, free_idx]
            rhs_free = rhs[free_idx]
            if constrained_idx.size > 0:
                A_IB = A[free_idx][:, constrained_idx]
                rhs_free = rhs_free - A_IB @ constrained_values
            psi_reg[free_idx] = spla.spsolve(A_II, rhs_free)

        ctx.commit(psi=psi_reg)

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

    def interpolate_reg(self, probes):
        """Interpolates the regularized stream function using the P2 basis."""
        psi = self.psi.get()
        face_dof = self.p2_elements.face_dof.get()
        faceids, bary = probe_arrays(probes)
        basis = np.stack(
            self.p2_elements.basis_from_bary(bary[:, 0], bary[:, 1], bary[:, 2]), axis=1
        )
        return np.einsum("ij,ij->i", psi[face_dof[faceids]], basis)

    def interpolate(self, probes):
        """
        Interpolates the stream function using P2 basis
        """
        if not self.regularize:
            return self.poisson.interpolate(probes)

        return self.interpolate_reg(probes)

    def set_homo_dirichlet_boundary(self):
        boundary_mask = self.p2_elements.boundary_mask.get()
        boundary_dofs = np.where(boundary_mask == True)[0]
        self.constrained_idx.set(boundary_dofs.astype(np.int32))
        self.constrained_values.set(np.zeros_like(boundary_dofs, dtype=np.float64))
