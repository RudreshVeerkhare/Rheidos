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
from .harmonic_component import (
    DEFAULT_HARMONIC_COEFFICIENT,
    MobiusHarmonicBasis,
    MobiusHarmonicComponent,
)
from .intrinsic_advection import (
    IntrinsicVortexState,
    RK4AdvectorModule,
    ReduceTimestepError,
)
from .io import load_mesh_input, load_point_vortex_input
from .orientation_double_cover import OrientationDoubleCover
from .parity import DeckEvenP1VelocityField, DeckOddP1StreamFunction

SESSION_NAME = "mobius_strip_vortex_dynamics"
HARMONIC_COEFFICIENT_ATTR = "harmonic_coefficient"
ABEL_JACOBI_COORDINATE_ATTR = "abel_jacobi_coordinate"
ABEL_JACOBI_STEP_DELTA_ATTR = "abel_jacobi_step_delta"
ABEL_JACOBI_INVARIANT_ATTR = "abel_jacobi_invariant"


def _paired_positions(app: "App") -> np.ndarray:
    """Reconstruct debug/output positions for the derived 2N vortex view."""
    face_ids = app.point_vortex.face_ids.get()
    bary = app.point_vortex.bary.get()
    vertices = app.cover_mesh.V_pos.get()[app.cover_mesh.F_verts.get()[face_ids]]
    return np.einsum("ni,nij->nj", bary, vertices)


def _read_optional_scalar_detail(input_io, name: str) -> float | None:
    try:
        values = np.asarray(input_io.read_detail(name, dtype=np.float64)).reshape(-1)
    except KeyError:
        return None
    if values.shape != (1,):
        raise ValueError(f"Detail attribute {name!r} must contain one scalar")
    return float(values[0])


def _restore_solver_state_from_input(ctx: CookContext, app: "App") -> None:
    """Restore cover-sheet and harmonic state from solver feedback.

    Base face coordinates alone cannot distinguish the two cover sheets.  The
    explicit cover attributes make solver state robust to session rebuilds and
    timeline recooks.  The accepted harmonic coefficient is persisted for the
    same reason; the Abel--Jacobi coordinate itself is derived from particles.
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
        pass
    else:
        app.vortex_state.set_state(IntrinsicVortexState(face_ids, bary), gamma)
        app.vortex_state.sync_base()
        app.harmonic_component.refresh_accepted_coordinate()

    coefficient = _read_optional_scalar_detail(input_io, HARMONIC_COEFFICIENT_ATTR)
    if coefficient is not None:
        app.harmonic_coefficient.set(coefficient)

    step_delta = _read_optional_scalar_detail(input_io, ABEL_JACOBI_STEP_DELTA_ATTR)
    if step_delta is not None:
        app.abel_jacobi_step_delta.set(step_delta)


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
    app.initialize_harmonic_component()

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
    _restore_solver_state_from_input(ctx, app)
    app.rk4_step(dt)

    base = app.base_point_vortex
    lifted = app.lifted_point_vortex
    deck = app.vortex_state.deck_state()
    ctx.clear_output()
    ctx.create_points(base.pos_world.get())
    ctx.write_point("faceid", base.face_ids.get(), create=True)
    ctx.write_point("bary", base.bary.get(), create=True)
    ctx.write_point("cover_faceid", lifted.face_ids.get(), create=True)
    ctx.write_point("cover_bary", lifted.bary.get(), create=True)
    ctx.write_point("deck_faceid", deck.face_ids, create=True)
    ctx.write_point("deck_bary", deck.bary, create=True)
    ctx.write_point("gamma", lifted.gamma.get(), create=True)
    ctx.write_detail(
        HARMONIC_COEFFICIENT_ATTR,
        np.array([app.harmonic_coefficient.get()], dtype=np.float64),
        create=True,
    )
    ctx.write_detail(
        ABEL_JACOBI_COORDINATE_ATTR,
        np.array([app.abel_jacobi_coordinate.get()], dtype=np.float64),
        create=True,
    )
    ctx.write_detail(
        ABEL_JACOBI_STEP_DELTA_ATTR,
        np.array([app.abel_jacobi_step_delta.get()], dtype=np.float64),
        create=True,
    )
    ctx.write_detail(
        ABEL_JACOBI_INVARIANT_ATTR,
        np.array([app.abel_jacobi_invariant.get()], dtype=np.float64),
        create=True,
    )


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
        self.harmonic_basis = self.require(
            MobiusHarmonicBasis,
            child=True,
            child_name="harmonic_basis",
            mesh=self.cover_mesh,
            dec=self.dec,
            double_cover=self.cover,
        )
        self.harmonic_component = self.require(
            MobiusHarmonicComponent,
            child=True,
            child_name="harmonic_component",
            mesh=self.cover_mesh,
            point_vortex=self.point_vortex,
            basis=self.harmonic_basis,
        )
        # Public aliases make accepted harmonic state easy to query from the
        # running App without exposing its internal module layout.
        self.harmonic_coefficient = self.harmonic_component.harmonic_coefficient
        self.abel_jacobi_coordinate = self.harmonic_component.abel_jacobi_coordinate
        self.abel_jacobi_step_delta = self.harmonic_component.abel_jacobi_step_delta
        self.abel_jacobi_invariant = self.harmonic_component.abel_jacobi_invariant
        self.rk4 = self.require(RK4AdvectorModule, mesh=self.cover_mesh)

    def initialize_lifted_point_vortices(self) -> IntrinsicVortexState:
        return self.vortex_state.initialize_from_base()

    def sync_base_point_vortices(self) -> IntrinsicVortexState:
        return self.vortex_state.sync_base()

    def initialize_harmonic_component(
        self,
        coefficient: float = DEFAULT_HARMONIC_COEFFICIENT,
    ) -> None:
        """Precompute the basis and initialize accepted scalar state."""
        self.harmonic_basis.configure_boundary_pair()
        self.harmonic_component.initialize(coefficient)
        self.harmonic_basis.zeta_face.get()
        self.abel_jacobi_coordinate.get()

    def rk4_step(self, dt: float) -> IntrinsicVortexState:
        """Advance the authoritative N-state and commit only an accepted step."""
        gamma = np.array(
            self.lifted_point_vortex.gamma.get(),
            dtype=np.float64,
            copy=True,
        )
        reference = self.vortex_state.current_state()
        reference_coefficient = float(self.harmonic_coefficient.get())

        # The reference coordinate and coefficient stay frozen for all four
        # RK stages.  Only the installed particle configuration changes.
        self.vortex_state.set_state(reference, gamma)
        reference_coordinate = float(self.abel_jacobi_coordinate.get())

        def velocity(state: IntrinsicVortexState) -> np.ndarray:
            # Every trial replaces only the N originals.  Reading velocity
            # lazily rebuilds the 2N odd pair, odd psi, and even velocity for
            # precisely that RK4 stage.
            self.vortex_state.set_state(state, gamma)
            stage_coefficient = self.harmonic_component.stage_coefficient(
                reference_coordinate,
                reference_coefficient,
            )
            stream_velocity = self.stream_velocity.interpolate(state.probes)
            harmonic_velocity = self.harmonic_component.interpolate(
                state.probes,
                coefficient=stage_coefficient,
            )
            return stream_velocity + harmonic_velocity

        try:
            accepted = self.rk4.step(reference, velocity, dt)
        except Exception:
            # Producer resources are mutated during stage evaluation.  Restore
            # the accepted state so a caller can safely retry with smaller dt.
            self.vortex_state.set_state(reference, gamma)
            raise

        self.vortex_state.set_state(accepted, gamma)
        self.harmonic_component.commit_accepted_step(
            reference_coordinate,
            reference_coefficient,
        )
        self.vortex_state.sync_base()
        return accepted


__all__ = [
    "App",
    "ABEL_JACOBI_COORDINATE_ATTR",
    "ABEL_JACOBI_INVARIANT_ATTR",
    "ABEL_JACOBI_STEP_DELTA_ATTR",
    "DoubleCoverPointVortex",
    "HARMONIC_COEFFICIENT_ATTR",
    "IntrinsicVortexState",
    "RK4AdvectorModule",
    "ReduceTimestepError",
]
