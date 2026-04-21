import numpy as np

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ProducerContext, World, producer


class P1StreamFunction(ModuleBase):
    NAME = "P1StreamFunction"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        point_vortex: PointVortexModule,
        dec: DEC | None = None,
        poisson: P1PoissonSolver | None = None,
        scope: str = "",
        distribute_excess_vorticity: bool = False,
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        self.point_vortex = point_vortex
        if poisson is None and dec is None:
            raise ValueError("P1StreamFunction requires either dec or poisson")

        # Usage guide:
        # - `child=True, child_name="poisson"` gives the solver its own nested
        #   resource namespace under this module
        # - plain requires inside the child solver still resolve through this
        #   module's lookup scope, so mesh/DEC are shared automatically
        # - `declare_rhs=False` lets this wrapper own the vorticity production
        #   while the child solver still owns the CG/Laplacian machinery
        if poisson is None:
            self.dec = dec
            self.poisson = self.require(
                P1PoissonSolver,
                child=True,
                child_name="poisson",
                mesh=mesh,
                dec=dec,
                declare_rhs=False,
            )
        else:
            self.poisson = poisson
            self.dec = poisson.dec

        # Re-export the solver's public resources so the wrapper stays the
        # composition-facing facade.

        self.distribute_excess_vorticity = distribute_excess_vorticity
        # Re-export the solver's public resources so the wrapper stays the
        # composition-facing facade.
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
            "mesh.V_pos",
        ),
        outputs=("omega",),
    )
    def splat_vortices(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()

        F_verts = self.mesh.F_verts.get()
        V_pos = self.mesh.V_pos.get()
        faceids = self.point_vortex.face_ids.get()
        bary = self.point_vortex.bary.get()
        gamma = self.point_vortex.gamma.get()

        ctx.ensure_outputs()
        omega = self.omega.peek()
        omega.fill(0.0)

        if self.distribute_excess_vorticity:
            nV = V_pos.shape[0]
            offset = np.sum(gamma) / nV
            omega.fill(-offset)

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

    def set_homo_dirichlet_boundary(self) -> None:
        boundary_mask = self.dec.boundary_mask.get()
        boundary_dofs = np.where(boundary_mask == True)[0]

        # If boundary less closed surfaces then set dirichlet pin
        if len(boundary_dofs) == 0:
            boundary_dofs = np.array([0])

        self.constrained_idx.set(boundary_dofs.astype(np.int32))
        self.constrained_values.set(np.zeros_like(boundary_dofs, dtype=np.float64))
