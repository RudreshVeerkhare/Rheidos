from __future__ import annotations

import numpy as np

from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.houdini.runtime.cook_context import CookContext


def copy_input_to_output(ctx: CookContext, index: int) -> None:
    src_io = ctx.input_io(index)
    if src_io is None:
        raise RuntimeError(f"Input geometry {index} is not connected.")

    out_io = ctx.output_io()
    if out_io.geo_out is None:
        raise RuntimeError("CookContext output IO is missing output geometry.")

    out_io.geo_out.clear()
    out_io.geo_out.merge(src_io.geo_in)


def load_mesh_input(
    ctx: CookContext,
    mesh: SurfaceMeshModule,
    *,
    index: int = 0,
    missing_message: str | None = None,
) -> None:
    mesh_io = ctx.input_io(index)
    if mesh_io is None:
        raise RuntimeError(missing_message or f"Input {index} is not set")

    points = np.array(mesh_io.read_point("P", components=3), dtype=np.float64)
    triangles = np.array(mesh_io.read_prims(arity=3), dtype=np.int32)
    mesh.set_mesh(points, triangles)


def load_point_vortex_input(
    ctx: CookContext,
    point_vortex: PointVortexModule,
    *,
    index: int,
    missing_message: str | None = None,
) -> None:
    vort_io = ctx.input_io(index)
    if vort_io is None:
        raise RuntimeError(missing_message or f"Input {index} is not set")

    vortex_pos = np.array(vort_io.read_point("P", components=3), dtype=np.float64)
    vortex_bary = np.array(vort_io.read_point("bary", components=3), dtype=np.float64)
    vortex_gamma = np.array(vort_io.read_point("gamma"), dtype=np.float64)
    vortex_faceid = np.array(vort_io.read_point("faceid"), dtype=np.int32)
    point_vortex.set_vortex(vortex_faceid, vortex_bary, vortex_gamma, vortex_pos)


def read_probe_input(
    ctx: CookContext,
    *,
    index: int,
    missing_message: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    probe_io = ctx.input_io(index)
    if probe_io is None:
        raise RuntimeError(missing_message or f"Input {index} is not set")

    faceids = np.array(probe_io.read_point("faceid"), dtype=np.int32)
    bary = np.array(probe_io.read_point("bary", components=3), dtype=np.float64)
    return faceids, bary
