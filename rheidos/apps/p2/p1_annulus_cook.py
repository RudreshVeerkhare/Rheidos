from rheidos.houdini import CookContext, session

from ._io import copy_input_to_output
from .p1_annulus_app import (
    interpolate_p1_stream_function,
    interpolate_p1_velocity,
    interpolate_p1_harmonic_stream_function,
    interpolate_p1_harmonic_velocity,
    setup_p1_harmonic_stream_function,
    rk4_advect,
)


SESSION_NAME = "p1_annulus"


@session(SESSION_NAME, debugger=True)
def p1_harmonic_stream_setup_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    setup_p1_harmonic_stream_function(ctx)


# Stream Part
@session(SESSION_NAME, debugger=True)
def interpolate_p1_stream_function_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    interpolate_p1_stream_function(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_p1_velocity_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    interpolate_p1_velocity(ctx)


@session(SESSION_NAME, debugger=True)
def p1_velocity_rk4_advection(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    rk4_advect(ctx)


# Harmonic Part
@session(SESSION_NAME, debugger=True)
def interpolate_p1_harmonic_stream_function_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    interpolate_p1_harmonic_stream_function(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_p1_harmonic_velocity_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    interpolate_p1_harmonic_velocity(ctx)
