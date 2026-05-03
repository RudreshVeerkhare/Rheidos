from rheidos.houdini import CookContext, session
from .._io import copy_input_to_output

import rheidos.apps.p2.p1_sphere_app.app as app

SESSION_NAME = "p1_sphere_geodesic_dipole"


@session(SESSION_NAME, debugger=True)
def setup_coexact_stream_function_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    app.setup_coexact_stream_function(ctx)


# Coexact Stream Part
@session(SESSION_NAME, debugger=True)
def interpolate_coexact_stream_function_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    app.interpolate_coexact_stream_function(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_coexact_velocity_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    app.interpolate_coexact_velocity(ctx)


# Advection
@session(SESSION_NAME, debugger=True)
def advection_rk4(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    app.rk4_advect(ctx, dt=0.01)


@session(SESSION_NAME, debugger=True)
def read_coexact_stream_function_per_vertex_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    app.read_coexact_stream_function_per_vertex(ctx)


@session(SESSION_NAME, debugger=True)
def read_facewise_coexact_velocity(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    app.read_facewise_velocity_field(ctx)


@session(SESSION_NAME, debugger=True)
def read_per_vertex_coexact_velocity(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    app.read_per_vertex_velocity_field(ctx)
