from rheidos.houdini import CookContext, session

from ._io import copy_input_to_output
from .p1_avg_plane_app import (
    interpolate_p1_stream_func,
    interpolate_p1_velocity,
    rk4_advect,
    setup_p1_stream_function,
)


SESSION_NAME = "p1_plane"


@session(SESSION_NAME, debugger=True)
def p1_stream_setup(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    setup_p1_stream_function(ctx)


@session(SESSION_NAME, debugger=True)
def p1_stream_interpolate(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 1)
    interpolate_p1_stream_func(ctx)


@session(SESSION_NAME, debugger=True)
def p1_velocity_interpolate(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 1)
    interpolate_p1_velocity(ctx)


@session(SESSION_NAME, debugger=True)
def p1_velocity_rk4_advection(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    rk4_advect(ctx)
