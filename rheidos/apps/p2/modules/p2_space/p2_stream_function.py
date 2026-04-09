import numpy as np

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

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.point_vortex = self.require(PointVortexModule)

        self.eps = self.resource(
            "eps",
            spec=ResourceSpec(kind="python"),
            doc="eps for the heat based diffusion of the stream function",
            declare=True,
            buffer=1e-2,
        )

        # Usage guide:
        # - `child=True, child_name="poisson"` gives the solver its own nested
        #   resource namespace under this module
        # - plain requires inside the child solver still resolve through this
        #   module's lookup scope, so the P2 element space is shared
        #   automatically
        # - `declare_rhs=False` lets this wrapper own the vorticity production
        #   while the child solver still owns the CG solve path
        self.poisson = self.require(
            P2PoissonSolver,
            child=True,
            child_name="poisson",
            declare_rhs=False,
        )

        # Re-export the child solver's public resources so existing callers can
        # keep using the stream-function wrapper as the facade module.
        self.p2_elements = self.poisson.p2_space
        self.constrained_idx = self.poisson.constrained_idx
        self.constrained_values = self.poisson.constrained_values
        self.omega = self.poisson.rhs
        self.psi_non_reg = self.poisson.psi
        self.solve_cg = self.poisson.solve_cg

        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_scalar(self.p2_elements.n_dof),
            ),
            doc="Regularized stream function coefficients for chosen p2 basis.",
        )
        self.bind_producers()

    @producer(
        inputs=("psi_non_reg", "p2_elements.M_mass", "p2_elements.L_stiffness", "eps"),
        outputs=("psi",),
    )
    def regularize_psi(self, ctx: ProducerContext):
        ctx.require_inputs()
        K = self.p2_elements.L_stiffness.get()
        M = self.p2_elements.M_mass.get()
        psi_non_reg = self.psi_non_reg.get()
        eps = self.eps.get()

        A = (M + 0.5 * eps * K).tocsc()
        rhs = (M - 0.5 * eps * K) @ psi_non_reg

        psi = spla.spsolve(A, rhs)

        ctx.commit(psi=psi)

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

    def interpolate(self, probes):
        """Interpolates the regularized stream function using the P2 basis."""
        psi = self.psi.get()
        face_dof = self.p2_elements.face_dof.get()
        faceids, bary = probe_arrays(probes)
        basis = np.stack(
            self.p2_elements.basis_from_bary(bary[:, 0], bary[:, 1], bary[:, 2]), axis=1
        )
        return np.einsum("ij,ij->i", psi[face_dof[faceids]], basis)
