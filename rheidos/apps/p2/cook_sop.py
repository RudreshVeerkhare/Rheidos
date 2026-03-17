import hou
from rheidos.houdini.runtime import build_cook_context, session
from rheidos.houdini.debug import (
    consume_break_next_button,
    debug_config_from_node,
    ensure_debug_server,
    maybe_break_now,
    request_break_next,
)

from .app import cook, cook2

P1_SESSION_KEY = "p1"


def _get_input_geo(node: hou.Node, index: int) -> hou.Geometry:
    inputs = node.inputs()
    if index >= len(inputs) or inputs[index] is None:
        raise RuntimeError(f"Connect required input {index} to the Python SOP")
    geo = inputs[index].geometry()
    if geo is None:
        raise RuntimeError(f"Input geometry {index} is None")
    return geo


@session(P1_SESSION_KEY)
def node1(session) -> None:
    node = hou.pwd()
    geo_out = node.geometry()

    cfg = debug_config_from_node(node)
    ensure_debug_server(cfg, node=node)
    if consume_break_next_button(node):
        request_break_next(node=node)
    maybe_break_now(node=node)

    # Input 0: mesh, Input 1: Point-vortex state points
    mesh_geo = _get_input_geo(node, 0)
    vort_geo = _get_input_geo(node, 1)

    geo_out.clear()
    geo_out.merge(mesh_geo)

    ctx = build_cook_context(
        node, mesh_geo, geo_out, session, geo_inputs=[mesh_geo, vort_geo]
    )

    cook(ctx)

    print("Here from node 1!")


@session(P1_SESSION_KEY)
def node2(session) -> None:
    node = hou.pwd()
    geo_out = node.geometry()

    cfg = debug_config_from_node(node)
    ensure_debug_server(cfg, node=node)
    if consume_break_next_button(node):
        request_break_next(node=node)
    maybe_break_now(node=node)

    # Input 0: mesh, Input 1: Point-vortex state points
    mesh_geo = _get_input_geo(node, 0)
    probe_geo = _get_input_geo(node, 1)

    geo_out.clear()
    geo_out.merge(probe_geo)

    ctx = build_cook_context(
        node, mesh_geo, geo_out, session, geo_inputs=[mesh_geo, probe_geo]
    )

    cook2(ctx)

    print("Here from node 2!")
