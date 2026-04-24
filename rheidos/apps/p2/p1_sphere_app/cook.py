from rheidos.houdini import CookContext, session
from .._io import copy_input_to_output
import importlib

SESSION_NAME = "p1_sphere"


def _app():
    return importlib.import_module("rheidos.apps.p2.p1_sphere_app.app")


@session(SESSION_NAME, debugger=True)
def setup_coexact_stream_function_node(ctx: CookContext):
    copy_input_to_output(ctx, 0)
    _app().setup_coexact_stream_function(ctx)


# Coexact Stream Part
@session(SESSION_NAME, debugger=True)
def interpolate_coexact_stream_function_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    _app().interpolate_coexact_stream_function(ctx)


@session(SESSION_NAME, debugger=True)
def interpolate_coexact_velocity_node(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    _app().interpolate_coexact_velocity(ctx)


# Advection
@session(SESSION_NAME, debugger=True)
def advection_rk4(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    _app().rk4_advect(ctx)
