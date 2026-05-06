from rheidos.houdini import CookContext, session
from .._io import copy_input_to_output

from .app import (
    setup_mesh_and_point_vortices,
    rk4_advect,
    interpolate_harmonic_potential,
    interpolate_harmonic_basis_field,
    interpolate_combined_stream_function,
    interpolate_combined_velocity_field,
)

SESSION_NAME = "point_vortex_obstacle"


@session(SESSION_NAME, debugger=True)
def setup_mesh_and_point_vortices_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    setup_mesh_and_point_vortices(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_harmonic_potential_node(ctx: CookContext, basis_id=0):
    copy_input_to_output(ctx, 0)
    interpolate_harmonic_potential(ctx, basis_id=basis_id)


@session(SESSION_NAME, debugger=True)
def interpolate_harmonic_field_node(ctx: CookContext, basis_id=0):
    copy_input_to_output(ctx, 0)
    interpolate_harmonic_basis_field(ctx, basis_id=basis_id)


@session(SESSION_NAME, debugger=True)
def interpolate_combined_velocity_field_node(ctx: CookContext, smooth=True):
    copy_input_to_output(ctx, 0)
    interpolate_combined_velocity_field(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_combined_stream_function_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    interpolate_combined_stream_function(ctx)


@session(SESSION_NAME, debugger=True)
def rk4_advection_node(ctx: CookContext, dt=0.01, no_harmonic=False):
    copy_input_to_output(ctx, 0)
    rk4_advect(ctx, dt=dt, no_harmonic=no_harmonic)

    """
    
    1. Harmonic Basis
        - HarmonicBasis - u_stream + Jgrad (psi_h[vid])
        - Combined Stream function
    """
