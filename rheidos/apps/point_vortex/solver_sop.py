# Houdini Solver SOP template for Rheidos apps
#
# Wire it like a normal Solver:
#   input 0: prev frame feedback (from the solver itself)
#   input 1: external/current geometry (optional)
#
# This template:
# - Seeds output from prev-frame geometry (preferred) or current input
# - Builds a solver CookContext (is_solver=True, substep support)
# - Publishes geometry + sim timing keys
# - Calls your app's setup(ctx) once (optional) and step(ctx) every cook

import hou

from rheidos.houdini.runtime import (
    build_cook_context,
    get_runtime,
    publish_geometry_minimal,
)
from rheidos.houdini.runtime.resource_keys import (
    SIM_DT,
    SIM_FRAME,
    SIM_SUBSTEP,
    SIM_TIME,
)

# === IMPORT: change this to your app ===
# You can have only `step(ctx)` if you want; `setup(ctx)` is optional.
from rheidos.apps.point_vortex.app import setup, step  # <-- your app entrypoints


def _geo_from_input(node: hou.Node, idx: int) -> hou.Geometry | None:
    ins = node.inputs()
    if len(ins) <= idx or ins[idx] is None:
        return None
    g = ins[idx].geometry()
    return g


def _seed_output(geo_out: hou.Geometry, source: hou.Geometry) -> None:
    geo_out.clear()
    geo_out.merge(source)


def _ensure_state(session) -> None:
    # Store per-node persistent flags/caches on the session object if possible.
    # If your WorldSession already has fields like did_setup/last_step_key, use them.
    if not hasattr(session, "_solver_did_setup"):
        session._solver_did_setup = False
    if not hasattr(session, "_solver_last_step_key"):
        session._solver_last_step_key = None


def _publish_sim_keys(ctx) -> None:
    # Your CookContext likely already has these; publish so producers can depend on them.
    ctx.publish(SIM_FRAME, ctx.frame)
    ctx.publish(SIM_TIME, ctx.time)
    ctx.publish(SIM_DT, ctx.dt)
    ctx.publish(SIM_SUBSTEP, ctx.substep)


def main() -> None:
    node = hou.pwd()
    geo_out = node.geometry()

    geo_prev = _geo_from_input(node, 0)  # solver feedback
    geo_in = _geo_from_input(node, 1)  # optional external input this frame

    source = geo_prev or geo_in
    if source is None:
        raise RuntimeError(
            "Solver SOP needs at least input 0 (prev) or input 1 (current) geometry."
        )

    # 1) Pass-through seed: start from previous frame (preferred), else current input
    _seed_output(geo_out, source)

    # 2) Persistent session for this node
    session = get_runtime().get_or_create_session(node)
    _ensure_state(session)

    # 3) Optional: substep parameter (create an int parm 'substep' if you want)
    substep = int(node.evalParm("substep")) if node.parm("substep") else 0

    # 4) Build a solver context
    ctx = build_cook_context(
        node,
        source,  # input snapshot for reading
        geo_out,  # output geometry to write
        session,
        substep=substep,
        is_solver=True,
    )

    # 5) Publish geometry + sim timing keys
    publish_geometry_minimal(ctx)
    _publish_sim_keys(ctx)

    # 6) Run setup once, then step every time
    if not session._solver_did_setup:
        if callable(setup):
            setup(ctx)
        session._solver_did_setup = True

    # Guard against Houdini double-cooks of same frame/substep (common in Solvers)
    step_key = (ctx.frame, ctx.substep)
    if step_key == session._solver_last_step_key:
        return

    step(ctx)
    session._solver_last_step_key = step_key

    # === OPTIONAL: apply outputs back to Houdini if your app writes resources ===
    # if ctx.exists("out.P"):
    #     ctx.set_P(ctx.fetch("out.P"))
