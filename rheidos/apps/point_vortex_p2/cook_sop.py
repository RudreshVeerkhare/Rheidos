"""Houdini Python SOP entrypoint for point_vortex_p2 single-cook solve."""

import hou

from rheidos.houdini.runtime import build_cook_context, session

from rheidos.apps.point_vortex_p2.app import cook


def _get_input_geo(node: hou.Node, index: int) -> hou.Geometry:
    inputs = node.inputs()
    if index >= len(inputs) or inputs[index] is None:
        raise RuntimeError(f"Connect required input {index} to the Python SOP")
    geo = inputs[index].geometry()
    if geo is None:
        raise RuntimeError(f"Input geometry {index} is None")
    return geo


def _seed_output(geo_out: hou.Geometry, src: hou.Geometry) -> None:
    geo_out.clear()
    geo_out.merge(src)


@session
def run_cook(session) -> None:
    node = hou.pwd()
    geo_out = node.geometry()

    # Input 0: mesh, Input 1: point-vortex state points.
    mesh_geo = _get_input_geo(node, 0)
    vort_geo = _get_input_geo(node, 1)

    _seed_output(geo_out, mesh_geo)

    ctx = build_cook_context(
        node,
        mesh_geo,
        geo_out,
        session,
        geo_inputs=[mesh_geo, vort_geo],
    )

    cook(ctx)
