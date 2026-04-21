import numpy as np

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ProducerContext, World, producer


class P1AnnulusHarmonicStreamFunction(ModuleBase):
    NAME = "P1AnnulusHarmonicStreamFunction"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        dec: DEC | None = None,
        poisson: P1PoissonSolver | None = None,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        if poisson is None and dec is None:
            raise ValueError(
                "P1AnnulusHarmonicStreamFunction requires either dec or poisson"
            )
        # Usage guide:
        # - `child=True, child_name="poisson"` gives the solver its own nested
        #   resource namespace under this module
        # - plain requires inside the child solver still resolve through this
        #   module's lookup scope, so mesh/DEC are shared automatically
        # - `declare_rhs=False` lets this wrapper own the vorticity production
        #   while the child solver still owns the CG/Laplacian machinery
        if poisson is None:
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

        # Re-export the solver's public resources so the wrapper stays the
        # composition-facing facade.
        self.dec = self.poisson.dec
        self.constrained_idx = self.poisson.constrained_idx
        self.constrained_values = self.poisson.constrained_values
        self.rhs = self.poisson.rhs
        self.psi = self.poisson.psi
        self.L_cached = self.poisson.L_cached
        self.solve_cg = self.poisson.solve_cg

        self.bind_producers()

    @producer(inputs=(), outputs=("rhs",))
    def fill_rhs(self, ctx: ProducerContext):
        ctx.ensure_outputs()
        rhs = self.rhs.peek()

        # Set the RHS to 0 to make it into a laplace problem with boundary value
        rhs.fill(0.0)
        ctx.commit(rhs=rhs)

    def interpolate(self, probes) -> np.ndarray:
        return self.poisson.interpolate(probes)

    def set_annulus_dirichlet_boundary(self):
        # Set of boundaries.
        # - There will be 2 sets
        boundary_components = self.mesh.boundary_vertex_components.get()
        if len(boundary_components) != 2:
            raise RuntimeError("Annulus is expected to have 2 boundary condtions.")

        # set dirichlet boundary condition to solve poisson
        vb1, vb2 = boundary_components
        self.constrained_idx.set(np.concatenate([vb1, vb2]))
        self.constrained_values.set(
            np.concatenate(
                [np.full(vb1.shape, 1.0), np.zeros_like(vb2)], dtype=np.float64
            )
        )
