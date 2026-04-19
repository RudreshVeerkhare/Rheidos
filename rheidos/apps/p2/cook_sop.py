from rheidos.houdini.runtime import session
from rheidos.houdini.runtime.cook_context import CookContext

from ._io import copy_input_to_output
from .app import sample_p1_stream_function, solve_p1_stream_function
from .p2_app import sample_p2_stream_function, sample_p2_velocity, solve_p2_stream_function
from .p2_test_app import p1_cook2_test, p1_cook_test, p2_cook2_test, p2_cook_test

P1_SESSION_KEY = "p1"
P2_SESSION_KEY = "p2"
P2_TEST_SESSION_KEY = "p2_test"
P1_TEST_SESSION_KEY = "p1_test"


# P1 test solve node: copy mesh input 0 to output, then run the P1 test solve.
@session(P1_TEST_SESSION_KEY, debugger=True)
def node7(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    p1_cook_test(ctx)


# P1 test sample node: copy probe input 1 to output, then sample the P1 test solve.
@session(P1_TEST_SESSION_KEY, debugger=True)
def node8(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 1)
    p1_cook2_test(ctx)


# P2 test solve node: copy mesh input 0 to output, then run the P2 test solve.
@session(P2_TEST_SESSION_KEY, debugger=True)
def node5(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    p2_cook_test(ctx)


# P2 test sample node: copy probe input 1 to output, then sample the P2 test solve.
@session(P2_TEST_SESSION_KEY, debugger=True)
def node6(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 1)
    p2_cook2_test(ctx)


def _eval_parm_float(node, name: str, default: float) -> float:
    parm = node.parm(name)
    if parm is None:
        return default
    try:
        return float(parm.eval())
    except Exception:
        return default


# P2 solve node: copy mesh input 0 to output, then solve the regularized stream function.
@session(P2_SESSION_KEY, debugger=True)
def node3(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    eps = _eval_parm_float(ctx.node, "eps", 0.01)
    solve_p2_stream_function(ctx, eps)


# P2 sample node: copy probe input 1 to output, then sample the stream function into stream_func.
@session(P2_SESSION_KEY, debugger=True)
def node4(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 1)
    sample_p2_stream_function(ctx)


# P1 solve node: copy mesh input 0 to output, then solve and export the stream function on mesh points.
@session(P1_SESSION_KEY, debugger=True)
def node1(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 0)
    solve_p1_stream_function(ctx)


# P1 sample node: copy probe input 1 to output, then sample the stream function into stream_func.
@session(P1_SESSION_KEY, debugger=True)
def node2(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 1)
    sample_p1_stream_function(ctx)


# P2 velocity sample node: copy probe input 1 to output, then sample the velocity field into vel.
@session(P2_SESSION_KEY, debugger=True)
def interpolate_vel(ctx: CookContext) -> None:
    copy_input_to_output(ctx, 1)
    sample_p2_velocity(ctx)
