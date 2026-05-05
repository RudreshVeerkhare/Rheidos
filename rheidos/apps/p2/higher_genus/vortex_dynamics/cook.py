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
)

SESSION_NAME = "vortex_dynamics_higer_genus"


@session(SESSION_NAME, debugger=True)
def setup_mesh_and_point_vortices_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    setup_mesh_and_point_vortices(ctx)


@session(SESSION_NAME, debugger=True)
def rk4_advection_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    rk4_advect(ctx, dt=0.01)


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
