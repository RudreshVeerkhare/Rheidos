from typing import Optional

import numpy as np

from rheidos.apps.p2._io import read_probe_input
from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.apps.p2.modules.higher_genus.dual_harmonic_field import (
    DualHarmonicFieldModule,
)
from rheidos.apps.p2.modules.higher_genus.harmonic_basis import HarmonicBasis
from rheidos.apps.p2.modules.higher_genus.tree_cotree import TreeCotreeModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.probe_utils import probe_arrays
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p1_space.p1_velocity import P1VelocityFieldModule
from rheidos.apps.p2.modules.p1_space.whitney_1form import Whitney1FormInterpolator
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute.world import ModuleBase, World
from rheidos.houdini.runtime.cook_context import CookContext

from ..io import load_mesh_input, load_point_vortex_input
from .ray_sop_module import RaySopModule

RAY_SOP_NODE_PATH = "/obj/geo1/solver1/d/s/ray1"


class AbelJacobiModule(ModuleBase):
    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        point_vortex: PointVortexModule,
        dual_harmonic_field: DualHarmonicFieldModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        self.point_vortex = point_vortex
        self.dual_harmonic_field = dual_harmonic_field

    def _positions_from_probes(self, probes, *, name: str) -> np.ndarray:
        faceids, bary = probe_arrays(probes)
        if (
            bary.ndim != 2
            or bary.shape[1] != 3
            or faceids.shape != (bary.shape[0],)
        ):
            raise ValueError(
                f"{name} barycentric coordinates must have shape (N,3), "
                f"got {bary.shape}"
            )

        face_vertices = self.mesh.F_verts.get()[faceids]
        vertex_positions = self.mesh.V_pos.get()[face_vertices]
        return np.einsum("ni,nij->nj", bary, vertex_positions)

    @staticmethod
    def _validate_positions(
        positions: Optional[np.ndarray],
        *,
        count: int,
        name: str,
    ) -> Optional[np.ndarray]:
        if positions is None:
            return None

        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (count, 3):
            raise ValueError(
                f"{name} must have shape ({count},3), got {positions.shape}"
            )
        return positions

    def delta_aj(
        self,
        start_probes,
        end_probes,
        *,
        pos0: Optional[np.ndarray] = None,
        pos1: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return per-vortex Abel-Jacobi increments between consecutive samples."""
        start_faceids, start_bary = probe_arrays(start_probes)
        end_faceids, end_bary = probe_arrays(end_probes)

        if (
            start_bary.ndim != 2
            or start_bary.shape[1] != 3
            or start_faceids.shape != (start_bary.shape[0],)
        ):
            raise ValueError(
                "start_probes barycentric coordinates must have shape (N,3), "
                f"got {start_bary.shape}"
            )
        if (
            end_bary.ndim != 2
            or end_bary.shape[1] != 3
            or end_faceids.shape != (end_bary.shape[0],)
        ):
            raise ValueError(
                "end_probes barycentric coordinates must have shape (N,3), "
                f"got {end_bary.shape}"
            )
        if start_faceids.shape != end_faceids.shape:
            raise ValueError(
                "start_probes and end_probes must contain the same number of "
                f"points, got {start_faceids.shape[0]} and {end_faceids.shape[0]}"
            )

        count = int(start_faceids.shape[0])
        p0 = self._validate_positions(pos0, count=count, name="pos0")
        p1 = self._validate_positions(pos1, count=count, name="pos1")
        if p0 is None:
            p0 = self._positions_from_probes((start_faceids, start_bary), name="pos0")
        if p1 is None:
            p1 = self._positions_from_probes((end_faceids, end_bary), name="pos1")

        xi0 = self.dual_harmonic_field.interpolate(
            (start_faceids, start_bary), field="xi"
        )
        xi1 = self.dual_harmonic_field.interpolate((end_faceids, end_bary), field="xi")

        # Harmonic fields are basis-major (K,N,3), while DeltaAJ is returned
        # vortex-major (N,K) so rows align with point-vortex arrays.
        return 0.5 * np.einsum("kni,ni->nk", xi0 + xi1, p1 - p0)


class App(ModuleBase):
    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        # Mesh
        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(DEC, mesh=self.mesh)

        # Point Vortices
        self.point_vortex = self.require(PointVortexModule)
        self.stream_function = self.require(
            P1StreamFunction,
            mesh=self.mesh,
            point_vortex=self.point_vortex,
            dec=self.dec,
        )
        self.stream_velocity = self.require(
            P1VelocityFieldModule,
            mesh=self.mesh,
            dec=self.dec,
            stream=self.stream_function,
        )

        # Harmonic Basis
        self.tree_cotree = self.require(TreeCotreeModule, mesh=self.mesh)
        self.harmonic_basis = self.require(
            HarmonicBasis,
            dec=self.dec,
            tree_cotree=self.tree_cotree,
        )
        self.whitney_1form = self.require(
            Whitney1FormInterpolator,
            mesh=self.mesh,
        )
        self.dual_harmonic_field = self.require(
            DualHarmonicFieldModule,
            mesh=self.mesh,
            harmonic_basis=self.harmonic_basis,
        )

        # Abel-Jacobi Map
        self.abel_jacobi = self.require(
            AbelJacobiModule,
            mesh=self.mesh,
            point_vortex=self.point_vortex,
            dual_harmonic_field=self.dual_harmonic_field,
        )

        # Advection
        self.rk4 = self.require(RK4IntegratorModule)
        self.surface_projector = self.require(
            RaySopModule,
            child=True,
            child_name="ray_sop_surface_points_projector",
            node_path=RAY_SOP_NODE_PATH,
        )

    @staticmethod
    def rk4_step(ctx: CookContext):
        mods = ctx.world().require(App)
        mods.surface_projector.configure(node_path=RAY_SOP_NODE_PATH)
        mods.surface_projector.setup(ctx)

        def y_dot(y: np.ndarray, t: float) -> np.ndarray:
            projected = mods.surface_projector.project_points(y)
            faceids, barys, pos = (
                projected.faceids,
                projected.bary,
                projected.pos,
            )
            gammas = mods.point_vortex.gamma.get()
            mods.point_vortex.set_vortex(
                faceids,
                barys,
                gammas,
                pos,
            )

            return mods.stream_velocity.interpolate((faceids, barys))

        return y_dot


def setup_mesh_and_point_vortices(ctx: CookContext):
    mods = ctx.world().require(App)
    load_mesh_input(
        ctx, mods.mesh, missing_message="Input 0 has to be mesh input geometry"
    )
    load_point_vortex_input(ctx, mods.point_vortex, index=1)
    mods.stream_function.set_homo_dirichlet_boundary()


def rk4_advect(ctx: CookContext, dt=0.001) -> None:
    mods = ctx.world().require(App)
    mods.surface_projector.configure(node_path=RAY_SOP_NODE_PATH)
    mods.surface_projector.setup(ctx)

    y_dot = mods.rk4_step(ctx)
    mods.rk4.configure(y_dot=y_dot, timestep=dt)
    load_point_vortex_input(ctx, mods.point_vortex, index=0)

    y0 = mods.point_vortex.pos_world.get()
    y = mods.rk4.step(y0)

    projected = mods.surface_projector.project_points(y)
    faceids, barys, pos = projected.faceids, projected.bary, projected.pos

    gammas = mods.point_vortex.gamma.get()
    mods.point_vortex.set_vortex(
        faceids,
        barys,
        gammas,
        pos,
    )
    ctx.write_point("P", pos)
    ctx.write_point("bary", barys)
    ctx.write_point("faceid", faceids)


# Interpolate


def interpolate_xi_dual_harmonic_field(ctx: CookContext, basis_id=0):
    mods = ctx.world().require(App)

    generator_count = mods.tree_cotree.generator_count.get()
    if basis_id < 0 or basis_id >= generator_count:
        raise RuntimeError(
            f"Allowed basis_id range: [0, {generator_count-1}], received: f{basis_id}"
        )

    faceids, bary = read_probe_input(ctx, index=0)
    xi = mods.dual_harmonic_field.interpolate((faceids, bary), field="xi")
    ctx.write_point("xi_dual_harmonic_field", xi[basis_id])


def interpolate_zeta_harmonic_field(ctx: CookContext, basis_id=0):
    mods = ctx.world().require(App)

    generator_count = mods.tree_cotree.generator_count.get()
    if basis_id < 0 or basis_id >= generator_count:
        raise RuntimeError(
            f"Allowed basis_id range: [0, {generator_count-1}], received: f{basis_id}"
        )

    faceids, bary = read_probe_input(ctx, index=0)
    zeta = mods.dual_harmonic_field.interpolate((faceids, bary), field="zeta")
    ctx.write_point("zeta_harmonic_field", zeta[basis_id])
