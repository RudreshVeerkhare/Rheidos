from __future__ import annotations

from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_annulus_harmoic_stream_function import (
    P1AnnulusHarmonicStreamFunction,
)
from rheidos.apps.p2.modules.p1_space.p1_annulus_harmonic_velocity import (
    P1AnnulusHarmonicVelocityFieldModule,
)
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p1_space.p1_velocity import P1VelocityFieldModule
from rheidos.apps.p2.modules.p2_space.p2_elements import P2Elements
from rheidos.apps.p2.modules.p2_space.p2_poisson_solver import P2PoissonSolver
from rheidos.apps.p2.modules.p2_space.p2_stream_function import P2StreamFunction
from rheidos.apps.p2.modules.p2_space.p2_velocity import P2VelocityField
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, World


class P1StreamGraph(ModuleBase):
    NAME = "P1StreamGraph"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.mesh = self.require(
            SurfaceMeshModule,
            child=True,
            child_name="mesh",
        )
        self.point_vortex = self.require(
            PointVortexModule,
            child=True,
            child_name="point_vortex",
        )
        self.dec = self.require(
            DEC,
            child=True,
            child_name="dec",
            mesh=self.mesh,
        )
        self.p1_poisson = self.require(
            P1PoissonSolver,
            child=True,
            child_name="poisson",
            mesh=self.mesh,
            dec=self.dec,
            declare_rhs=False,
        )
        self.p1_stream = self.require(
            P1StreamFunction,
            child=True,
            child_name="stream",
            mesh=self.mesh,
            point_vortex=self.point_vortex,
            poisson=self.p1_poisson,
        )


class P1PlaneGraph(ModuleBase):
    NAME = "P1PlaneGraph"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.p1_graph = self.require(
            P1StreamGraph,
            child=True,
            child_name="stack",
        )
        self.mesh = self.p1_graph.mesh
        self.point_vortex = self.p1_graph.point_vortex
        self.dec = self.p1_graph.dec
        self.p1_poisson = self.p1_graph.p1_poisson
        self.p1_stream = self.p1_graph.p1_stream
        self.p1_stream_func = self.p1_stream
        self.p1_vel = self.require(
            P1VelocityFieldModule,
            child=True,
            child_name="velocity",
            mesh=self.mesh,
            dec=self.dec,
            stream=self.p1_stream_func,
        )
        self.rk4 = self.require(
            RK4IntegratorModule,
            child=True,
            child_name="rk4",
        )


class P1AnnulusGraph(ModuleBase):
    NAME = "P1AnnulusGraph"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.mesh = self.require(
            SurfaceMeshModule,
            child=True,
            child_name="mesh",
        )
        self.dec = self.require(
            DEC,
            child=True,
            child_name="dec",
            mesh=self.mesh,
        )
        self.poisson = self.require(
            P1PoissonSolver,
            child=True,
            child_name="poisson",
            mesh=self.mesh,
            dec=self.dec,
            declare_rhs=False,
        )
        self.harmonic_stream = self.require(
            P1AnnulusHarmonicStreamFunction,
            child=True,
            child_name="stream",
            mesh=self.mesh,
            poisson=self.poisson,
        )
        self.harmonic_vel = self.require(
            P1AnnulusHarmonicVelocityFieldModule,
            child=True,
            child_name="velocity",
            mesh=self.mesh,
            dec=self.dec,
            stream=self.harmonic_stream,
        )
        self.p1_vel = self.require(
            P1VelocityFieldModule,
            child=True,
            child_name="og_velocity",
            mesh=self.mesh,
            dec=self.dec,
            stream=self.harmonic_stream,
        )


class P2StreamGraph(ModuleBase):
    NAME = "P2StreamGraph"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.mesh = self.require(
            SurfaceMeshModule,
            child=True,
            child_name="mesh",
        )
        self.point_vortex = self.require(
            PointVortexModule,
            child=True,
            child_name="point_vortex",
        )
        self.p2_space = self.require(
            P2Elements,
            child=True,
            child_name="space",
            mesh=self.mesh,
        )
        self.p2_poisson = self.require(
            P2PoissonSolver,
            child=True,
            child_name="poisson",
            p2_space=self.p2_space,
            declare_rhs=False,
        )
        self.p2_stream = self.require(
            P2StreamFunction,
            child=True,
            child_name="stream",
            mesh=self.mesh,
            point_vortex=self.point_vortex,
            p2_elements=self.p2_space,
            poisson=self.p2_poisson,
        )
        self.p2_stream_func = self.p2_stream
        self.p2_vel = self.require(
            P2VelocityField,
            child=True,
            child_name="velocity",
            mesh=self.mesh,
            p2_space=self.p2_space,
            stream=self.p2_stream,
        )


class P2PlaneGraph(ModuleBase):
    NAME = "P2PlaneGraph"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.p2_graph = self.require(
            P2StreamGraph,
            child=True,
            child_name="stack",
        )
        self.mesh = self.p2_graph.mesh
        self.point_vortex = self.p2_graph.point_vortex
        self.p2_space = self.p2_graph.p2_space
        self.p2_poisson = self.p2_graph.p2_poisson
        self.p2_stream = self.p2_graph.p2_stream
        self.p2_stream_func = self.p2_graph.p2_stream_func
        self.p2_vel = self.p2_graph.p2_vel
        self.rk4 = self.require(
            RK4IntegratorModule,
            child=True,
            child_name="rk4",
        )


class P1PoissonTestGraph(ModuleBase):
    NAME = "P1PoissonTestGraph"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.mesh = self.require(
            SurfaceMeshModule,
            child=True,
            child_name="mesh",
        )
        self.dec = self.require(
            DEC,
            child=True,
            child_name="dec",
            mesh=self.mesh,
        )
        self.p1_poisson = self.require(
            P1PoissonSolver,
            child=True,
            child_name="poisson",
            mesh=self.mesh,
            dec=self.dec,
        )


class P2PoissonTestGraph(ModuleBase):
    NAME = "P2PoissonTestGraph"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.mesh = self.require(
            SurfaceMeshModule,
            child=True,
            child_name="mesh",
        )
        self.p2_space = self.require(
            P2Elements,
            child=True,
            child_name="space",
            mesh=self.mesh,
        )
        self.p2_poisson = self.require(
            P2PoissonSolver,
            child=True,
            child_name="poisson",
            p2_space=self.p2_space,
        )
