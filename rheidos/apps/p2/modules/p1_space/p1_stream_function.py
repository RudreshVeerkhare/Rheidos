import numpy as np

from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ProducerContext, World, producer


class P1StreamFunction(ModuleBase):
    NAME = "P1StreamFunction"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.point_vortex = self.require(PointVortexModule)
        # Usage guide:
        # - `child=True, child_name="poisson"` gives the solver its own nested
        #   resource namespace under this module
        # - plain requires inside the child solver still resolve through this
        #   module's lookup scope, so mesh/DEC are shared automatically
        # - `declare_rhs=False` lets this wrapper own the vorticity production
        #   while the child solver still owns the CG/Laplacian machinery
        self.poisson = self.require(
            P1PoissonSolver,
            child=True,
            child_name="poisson",
            declare_rhs=False,
        )

        # Re-export the child solver's public resources so existing callers can
        # keep using the stream-function wrapper as the façade module.
        self.dec = self.poisson.dec
        self.constrained_idx = self.poisson.constrained_idx
        self.constrained_values = self.poisson.constrained_values
        self.omega = self.poisson.rhs
        self.psi = self.poisson.psi
        self.L_cached = self.poisson.L_cached
        self.solve_cg = self.poisson.solve_cg

        self.bind_producers()

    @producer(
        inputs=(
            "point_vortex.face_ids",
            "point_vortex.bary",
            "point_vortex.gamma",
            "mesh.F_verts",
        ),
        outputs=("omega",),
    )
    def splat_vortices(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()

        F_verts = self.mesh.F_verts.get()
        faceids = self.point_vortex.face_ids.get()
        bary = self.point_vortex.bary.get()
        gamma = self.point_vortex.gamma.get()
        omega = np.zeros_like(self.omega.peek())

        for idx, G in enumerate(gamma):
            fid = faceids[idx]
            v1, v2, v3 = F_verts[fid]
            b1, b2, b3 = bary[idx]
            omega[v1] += b1 * G
            omega[v2] += b2 * G
            omega[v3] += b3 * G

        ctx.commit(omega=omega)

    def interpolate(self, probes) -> np.ndarray:
        return self.poisson.interpolate(probes)
