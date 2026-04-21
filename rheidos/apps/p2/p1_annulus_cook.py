import importlib

from rheidos.houdini import CookContext, session

from ._io import copy_input_to_output


SESSION_NAME = "p1_annulus"


def _app():
    return importlib.import_module("rheidos.apps.p2.p1_annulus_app")


@session(SESSION_NAME, debugger=True)
def p1_harmonic_stream_setup_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    _app().setup_p1_harmonic_stream_function(ctx)


# Stream Part
@session(SESSION_NAME, debugger=True)
def interpolate_p1_stream_function_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    _app().interpolate_p1_stream_function(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_p1_velocity_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    _app().interpolate_p1_velocity(ctx)


@session(SESSION_NAME, debugger=True)
def p1_velocity_rk4_advection(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    _app().rk4_advect(ctx)


# Harmonic Part
@session(SESSION_NAME, debugger=True)
def interpolate_p1_harmonic_stream_function_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    _app().interpolate_p1_harmonic_stream_function(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_p1_harmonic_velocity_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    _app().interpolate_p1_harmonic_velocity(ctx)
