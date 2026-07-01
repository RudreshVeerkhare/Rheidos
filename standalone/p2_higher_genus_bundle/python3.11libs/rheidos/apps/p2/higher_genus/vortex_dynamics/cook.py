from rheidos.houdini import CookContext, session
from ..io import copy_input_to_output
from .app import (
    setup_mesh_and_point_vortices,
    rk4_advect,
    interpolate_xi_dual_harmonic_field,
    interpolate_zeta_harmonic_field,
    interpolate_harmonic_velocity_field,
    interpolate_stream_velocity_field,
    interpolate_velocity_field,
    App,
)

SESSION_NAME = "vortex_dynamics_higer_genus"


@session(SESSION_NAME, debugger=True)
def setup_mesh_and_point_vortices_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    setup_mesh_and_point_vortices(ctx)


@session(SESSION_NAME, debugger=True)
def rk4_advection_node(ctx: CookContext, dt=0.01, no_harmonic=False):
    copy_input_to_output(ctx, 0)
    rk4_advect(ctx, dt=0.01, no_harmonic=no_harmonic)


# Interpolate
@session(SESSION_NAME, debugger=True)
def interpolate_xi_dual_harmonic_node(ctx: CookContext, basis_id=0) -> None:
    copy_input_to_output(ctx, 0)
    interpolate_xi_dual_harmonic_field(ctx, basis_id=basis_id)


@session(SESSION_NAME, debugger=True)
def interpolate_zeta_harmonic_node(ctx: CookContext, basis_id=0) -> None:
    copy_input_to_output(ctx, 0)
    interpolate_zeta_harmonic_field(ctx, basis_id=basis_id)


@session(SESSION_NAME, debugger=True)
def interpolate_harmonic_velocity_field_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    interpolate_harmonic_velocity_field(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_stream_velocity_field_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    interpolate_stream_velocity_field(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_velocity_field_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    interpolate_velocity_field(ctx)


@session(SESSION_NAME, debugger=True)
def export_coexact_stream_function_per_vertex(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    app = ctx.world().require(App)
    coexact_stream_function = app.stream_function.psi.get()
    ctx.write_point("coexact_stream_function", coexact_stream_function)


@session(SESSION_NAME, debugger=True)
def export_velocity_per_face(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    app = ctx.world().require(App)
    vel_per_face = app.combined_velocity.vel_per_face.get()
    ctx.write_prim("vel_per_face", vel_per_face)
