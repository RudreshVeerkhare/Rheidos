"""Houdini Solver SOP entrypoint for point_vortex_p2."""

import hou

from rheidos.houdini.runtime import build_cook_context, get_runtime, publish_geometry_minimal
from rheidos.houdini.runtime.resource_keys import SIM_DT, SIM_FRAME, SIM_SUBSTEP, SIM_TIME

from rheidos.apps.point_vortex_p2.app import setup, step


def _geo_from_input(node: hou.Node, index: int):
    inputs = node.inputs()
    if index >= len(inputs):
        return None
    src = inputs[index]
    if src is None:
        return None
    try:
        return src.geometry()
    except Exception:
        return None


def _seed_output(geo_out: hou.Geometry, src: hou.Geometry) -> None:
    geo_out.clear()
    geo_out.merge(src)


def _ensure_state(session) -> None:
    if not hasattr(session, "_p2_solver_did_setup"):
        session._p2_solver_did_setup = False
    if not hasattr(session, "_p2_solver_last_step_key"):
        session._p2_solver_last_step_key = None


def _publish_sim_keys(ctx) -> None:
    ctx.publish(SIM_FRAME, ctx.frame)
    ctx.publish(SIM_TIME, ctx.time)
    ctx.publish(SIM_DT, ctx.dt)
    ctx.publish(SIM_SUBSTEP, ctx.substep)


def run_solver() -> None:
    node = hou.pwd()
    geo_out = node.geometry()

    # Input 0: point-vortex state (solver feedback); input 1: static triangle mesh.
    geo_prev = _geo_from_input(node, 0)
    geo_mesh = _geo_from_input(node, 1)

    if geo_prev is None and geo_mesh is None:
        raise RuntimeError("Solver requires at least one connected input geometry")

    source = geo_prev if geo_prev is not None else geo_mesh
    if source is None:
        raise RuntimeError("No source geometry available for solver output seeding")
    _seed_output(geo_out, source)

    session = get_runtime().get_or_create_session(node)
    _ensure_state(session)

    substep = int(node.evalParm("substep")) if node.parm("substep") else 0

    input_geos = [geo_prev, geo_mesh]
    ctx = build_cook_context(
        node,
        source,
        geo_out,
        session,
        geo_inputs=input_geos,
        substep=substep,
        is_solver=True,
    )

    publish_geometry_minimal(ctx)
    _publish_sim_keys(ctx)

    if not session._p2_solver_did_setup:
        setup(ctx)
        session._p2_solver_did_setup = True

    step_key = (ctx.frame, ctx.substep)
    if step_key == session._p2_solver_last_step_key:
        return

    step(ctx)
    session._p2_solver_last_step_key = step_key
