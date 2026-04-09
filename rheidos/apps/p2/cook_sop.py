from rheidos.houdini.runtime import session
from rheidos.houdini.runtime.cook_context import CookContext

from .app import cook, cook2
from .p2_app import p2_cook, p2_cook2, p2_interpolate_velocity
from .p2_test_app import p1_cook2_test, p1_cook_test, p2_cook2_test, p2_cook_test

P1_SESSION_KEY = "p1"
P2_SESSION_KEY = "p2"
P2_TEST_SESSION_KEY = "p2_test"
P1_TEST_SESSION_KEY = "p1_test"


def _copy_input_to_output(ctx: CookContext, index: int) -> None:
    src_io = ctx.input_io(index)
    if src_io is None:
        raise RuntimeError(f"Input geometry {index} is not connected.")

    out_io = ctx.output_io()
    if out_io.geo_out is None:
        raise RuntimeError("CookContext output IO is missing output geometry.")

    out_io.geo_out.clear()
    out_io.geo_out.merge(src_io.geo_in)


@session(P1_TEST_SESSION_KEY, debugger=True)
def node7(ctx: CookContext) -> None:
    _copy_input_to_output(ctx, 0)
    p1_cook_test(ctx)


@session(P1_TEST_SESSION_KEY, debugger=True)
def node8(ctx: CookContext) -> None:
    _copy_input_to_output(ctx, 1)
    p1_cook2_test(ctx)


@session(P2_TEST_SESSION_KEY, debugger=True)
def node5(ctx: CookContext) -> None:
    _copy_input_to_output(ctx, 0)
    p2_cook_test(ctx)


@session(P2_TEST_SESSION_KEY, debugger=True)
def node6(ctx: CookContext) -> None:
    _copy_input_to_output(ctx, 1)
    p2_cook2_test(ctx)


def _eval_parm_float(node, name: str, default: float) -> float:
    parm = node.parm(name)
    if parm is None:
        return default
    try:
        return float(parm.eval())
    except Exception:
        return default


@session(P2_SESSION_KEY, debugger=True)
def node3(ctx: CookContext) -> None:
    _copy_input_to_output(ctx, 0)

    eps = _eval_parm_float(ctx.node, "eps", 0.01)

    p2_cook(ctx, eps)


@session(P2_SESSION_KEY, debugger=True)
def node4(ctx: CookContext) -> None:
    _copy_input_to_output(ctx, 1)
    p2_cook2(ctx)


@session(P1_SESSION_KEY, debugger=True)
def node1(ctx: CookContext) -> None:
    _copy_input_to_output(ctx, 0)
    cook(ctx)


@session(P1_SESSION_KEY, debugger=True)
def node2(ctx: CookContext) -> None:
    _copy_input_to_output(ctx, 1)
    cook2(ctx)


@session(P2_SESSION_KEY, debugger=True)
def interpolate_vel(ctx: CookContext) -> None:
    index = 1
    src_io = ctx.input_io(index)
    if src_io is None:
        raise RuntimeError(f"Input geometry {index} is not connected.")

    out_io = ctx.output_io()
    if out_io.geo_out is None:
        raise RuntimeError("CookContext output IO is missing output geometry.")

    out_io.geo_out.clear()
    out_io.geo_out.merge(src_io.geo_in)

    p2_interpolate_velocity(ctx)
