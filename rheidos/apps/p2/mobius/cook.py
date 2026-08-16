"""Houdini entry points and module composition for the Mobius simulation."""

from __future__ import annotations

import numpy as np

from rheidos.apps.p2._io import read_probe_input
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, World
from rheidos.houdini import CookContext, session

from .cover_vortices import DoubleCoverPointVortex, OrientationCoverVortexState
from .intrinsic_advection import (
    IntrinsicVortexState,
    RK4AdvectorModule,
    ReduceTimestepError,
)
from .io import load_mesh_input, load_point_vortex_input
from .orientation_double_cover import OrientationDoubleCover
from .parity import DeckEvenP1VelocityField, DeckOddP1StreamFunction

SESSION_NAME = "mobius_strip_vortex_dynamics"


def _paired_positions(app: "App") -> np.ndarray:
    """Reconstruct debug/output positions for the derived 2N vortex view."""
    face_ids = app.point_vortex.face_ids.get()
    bary = app.point_vortex.bary.get()
    vertices = app.cover_mesh.V_pos.get()[app.cover_mesh.F_verts.get()[face_ids]]
    return np.einsum("ni,nij->nj", bary, vertices)


def _restore_lifted_state_from_input(ctx: CookContext, app: "App") -> None:
    """Restore cover-sheet state when a solver feeds the previous output back.

    Base face coordinates alone cannot distinguish the two cover sheets.  The
    explicit cover attributes make solver state robust to session rebuilds and
    timeline recooks instead of relying only on Python object persistence.
    """
    input_io = ctx.input_io(0)
    if input_io is None:
        return
    try:
        face_ids = np.asarray(input_io.read_point("cover_faceid"), dtype=np.int32)
        bary = np.asarray(
            input_io.read_point("cover_bary", components=3),
            dtype=np.float64,
        )
        gamma = np.asarray(input_io.read_point("gamma"), dtype=np.float64)
    except KeyError:
        return

    app.vortex_state.set_state(IntrinsicVortexState(face_ids, bary), gamma)
    app.vortex_state.sync_base()


@session(SESSION_NAME, debugger=True)
def write_cover_point_vortices(ctx: CookContext) -> None:
    """Write the derived 2N deck-paired vortices for inspection."""
    app = ctx.world().require(App)
    vortices = app.point_vortex
    ctx.clear_output()
    ctx.create_points(_paired_positions(app))
    ctx.write_point("faceid", vortices.face_ids.get(), create=True)
    ctx.write_point("bary", vortices.bary.get(), create=True)
    ctx.write_point("gamma", vortices.gamma.get(), create=True)


@session(SESSION_NAME, debugger=True)
def setup_mesh_node(ctx: CookContext) -> None:
    app = ctx.world().require(App)
    load_mesh_input(
        ctx,
        app.base_mesh,
        missing_message="Input 0 has to be mesh input geometry",
    )
    load_point_vortex_input(ctx, app.base_point_vortex, index=1)
    app.initialize_lifted_point_vortices()
    app.stream_function.set_homo_dirichlet_boundary()

    # Export the cover and its maps so topology/parity can be inspected in
    # Houdini without exposing implementation-only Python resources.
    output = ctx.output_io()
    output.clear_output()
    output.create_points(app.cover.cover_vertices.get())
    output.create_polygons(app.cover.cover_faces.get())
    output.write_point("base_vertex", app.cover.pi_vertex.get(), create=True)
    output.write_point("deck_vertex", app.cover.tau_vertex.get(), create=True)
    output.write_prim("base_face", app.cover.pi_face.get(), create=True)
    output.write_prim("deck_face", app.cover.tau_face.get(), create=True)
    output.write_point("psi", app.stream_function.psi.get(), create=True)


@session(SESSION_NAME, debugger=True)
def rk4_advection_node(ctx: CookContext, dt: float = 0.01) -> None:
    """Advance N lifted vortices and write both cover and base coordinates."""
    app = ctx.world().require(App)
    _restore_lifted_state_from_input(ctx, app)
    app.rk4_step(dt)

    base = app.base_point_vortex
    lifted = app.lifted_point_vortex
    ctx.clear_output()
    ctx.create_points(base.pos_world.get())
    ctx.write_point("faceid", base.face_ids.get(), create=True)
    ctx.write_point("bary", base.bary.get(), create=True)
    ctx.write_point("cover_faceid", lifted.face_ids.get(), create=True)
    ctx.write_point("cover_bary", lifted.bary.get(), create=True)
    ctx.write_point("gamma", lifted.gamma.get(), create=True)


@session(SESSION_NAME, debugger=True)
def interpolate_stream_velocity_node(ctx: CookContext) -> None:
    app = ctx.world().require(App)
    face_ids, bary = read_probe_input(ctx, index=0)
    ctx.write_point(
        "stream_velocity_field",
        app.stream_velocity.interpolate((face_ids, bary)),
    )


class App(ModuleBase):
    """Compose cover topology, vortex state, parity-aware fields, and RK4."""

    NAME = "MobiusOrientationCoverApp"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.base_mesh = self.require(
            SurfaceMeshModule,
            child=True,
            child_name="base_mesh",
        )
        self.cover = self.require(
            OrientationDoubleCover,
            child=True,
            child_name="orientation_cover",
            parent_mesh=self.base_mesh,
        )
        self.cover_mesh: SurfaceMeshModule = self.cover.cover_mesh
        self.dec: DEC = self.cover.dec

        self.base_point_vortex = self.require(PointVortexModule)
        self.vortex_state = self.require(
            OrientationCoverVortexState,
            child=True,
            child_name="vortices",
            base_mesh=self.base_mesh,
            base_point_vortex=self.base_point_vortex,
            double_cover=self.cover,
        )
        # Short aliases preserve the established app-facing names while the
        # state manager owns the synchronization policy.
        self.lifted_point_vortex = self.vortex_state.lifted_point_vortex
        self.point_vortex = self.vortex_state.paired_point_vortex

        self.stream_function = self.require(
            DeckOddP1StreamFunction,
            mesh=self.cover_mesh,
            point_vortex=self.point_vortex,
            dec=self.dec,
            double_cover=self.cover,
        )
        self.stream_velocity = self.require(
            DeckEvenP1VelocityField,
            mesh=self.cover_mesh,
            stream=self.stream_function,
            double_cover=self.cover,
        )
        self.rk4 = self.require(RK4AdvectorModule, mesh=self.cover_mesh)

    def initialize_lifted_point_vortices(self) -> IntrinsicVortexState:
        return self.vortex_state.initialize_from_base()

    def sync_base_point_vortices(self) -> IntrinsicVortexState:
        return self.vortex_state.sync_base()

    def rk4_step(self, dt: float) -> IntrinsicVortexState:
        """Advance the authoritative N-state and commit only an accepted step."""
        gamma = np.array(
            self.lifted_point_vortex.gamma.get(),
            dtype=np.float64,
            copy=True,
        )
        reference = self.vortex_state.current_state()

        def velocity(state: IntrinsicVortexState) -> np.ndarray:
            # Every trial replaces only the N originals.  Reading velocity
            # lazily rebuilds the 2N odd pair, odd psi, and even velocity for
            # precisely that RK4 stage.
            self.vortex_state.set_state(state, gamma)
            return self.stream_velocity.interpolate(state.probes)

        try:
            accepted = self.rk4.step(reference, velocity, dt)
        except Exception:
            # Producer resources are mutated during stage evaluation.  Restore
            # the accepted state so a caller can safely retry with smaller dt.
            self.vortex_state.set_state(reference, gamma)
            raise

        self.vortex_state.set_state(accepted, gamma)
        self.vortex_state.sync_base()
        return accepted


__all__ = [
    "App",
    "DoubleCoverPointVortex",
    "IntrinsicVortexState",
    "RK4AdvectorModule",
    "ReduceTimestepError",
]
