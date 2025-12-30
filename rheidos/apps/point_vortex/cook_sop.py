# Houdini Python SOP template for Rheidos apps
# - Passes input geometry through
# - Builds a Rheidos CookContext
# - Publishes geometry resources
# - Calls your app's cook(ctx)
#
# Edit the IMPORT section + optional hooks below.

import hou

from rheidos.houdini.debug import (
    consume_break_next_button,
    debug_config_from_node,
    ensure_debug_server,
    maybe_break_now,
    request_break_next,
)
from rheidos.houdini.runtime import (
    build_cook_context,
    get_runtime,
    publish_geometry_minimal,
)

# === IMPORT: change this to your app ===
from rheidos.apps.point_vortex.app import cook  # <-- your app's cook(ctx)


def _get_input_geo(node: hou.Node) -> hou.Geometry:
    inputs = node.inputs()
    if not inputs:
        raise RuntimeError("Connect input geometry to the Python SOP.")
    geo_in = inputs[0].geometry()
    if geo_in is None:
        raise RuntimeError("Input geometry is None.")
    return geo_in


def _seed_output(geo_out: hou.Geometry, geo_in: hou.Geometry) -> None:
    geo_out.clear()
    geo_out.merge(geo_in)


def run_cook() -> None:
    node = hou.pwd()
    geo_out = node.geometry()

    cfg = debug_config_from_node(node)
    ensure_debug_server(cfg, node=node)
    if consume_break_next_button(node):
        request_break_next(node=node)
    maybe_break_now(node=node)

    # 1) Read input geometry and pass-through to output
    geo_in = _get_input_geo(node)
    _seed_output(geo_out, geo_in)

    # 2) Get/create a persistent session for this node (cached across cooks)
    session = get_runtime().get_or_create_session(node)

    # 3) Build CookContext (input geo, output geo, session state)
    ctx = build_cook_context(node, geo_in, geo_out, session)

    # 4) Publish minimal geometry resources into the compute registry/world
    publish_geometry_minimal(ctx)

    # === OPTIONAL: publish extra stuff your app might want ===
    # ctx.publish("ui.some_float", float(node.evalParm("some_float")))
    # ctx.publish("ui.some_int", int(node.evalParm("some_int")))
    # ctx.publish("ui.some_toggle", bool(node.evalParm("some_toggle")))

    # 5) Run your app logic
    cook(ctx)

    # === OPTIONAL: apply outputs back to Houdini if your app writes resources ===
    # Example: if your app produces a resource "out.P" (point positions)
    # if ctx.exists("out.P"):
    #     ctx.set_P(ctx.fetch("out.P"))

    # Example: if your app produces primitive color "out.Cd_prim"
    # if ctx.exists("out.Cd_prim"):
    #     ctx.set_prim_Cd(ctx.fetch("out.Cd_prim"))  # adjust to your API
