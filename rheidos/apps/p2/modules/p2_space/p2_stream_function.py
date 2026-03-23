import numpy as np

from rheidos.apps.p2.modules.p2_space.p2_poisson_solver import P2PoissonSolver
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import World


class P2StreamFunction(ModuleBase):
    NAME = "P2StreamFunction"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.point_vortex = self.require(PointVortexModule)
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
        # keep using the stream-function wrapper as the façade module.
        self.p2_elements = self.poisson.p2_space
        self.constrained_idx = self.poisson.constrained_idx
        self.constrained_values = self.poisson.constrained_values
        self.omega = self.poisson.rhs
        self.psi = self.poisson.psi
        self.solve_cg = self.poisson.solve_cg

        self.bind_producers()

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

    def interpolate(self, probles):
        return self.poisson.interpolate(probles)
