from rheidos.houdini import CookContext, session

from ._io import copy_input_to_output
from .p2_plane_app import (
    interpolate_p2_stream_func,
    interpolate_p2_velocity,
    rk4_advect,
    setup_p2_stream_function,
)

SESSION_NAME = "p2_plane"


@session(SESSION_NAME, debugger=True)
def p2_stream_setup(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    setup_p2_stream_function(ctx)


@session(SESSION_NAME, debugger=True)
def p2_poisson_interpolate(ctx: CookContext) -> None:
    """Input 1: p2_setup SOP node, Input 2: subdivides mesh or scatter points"""
    copy_input_to_output(ctx, 1)
    interpolate_p2_stream_func(ctx)


@session(SESSION_NAME, debugger=True)
def p2_vel_interpolate(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 1)
    interpolate_p2_velocity(ctx)


@session(SESSION_NAME, debugger=True)
def p2_rk4_step(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    rk4_advect(ctx)
