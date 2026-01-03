# Houdini Python SOP template for Rheidos apps
# - Passes input geometry through
# - Builds a Rheidos CookContext
# - Publishes geometry resources
# - Calls your app's cook(ctx)
#
# Edit the IMPORT section + optional hooks below.

import hou

from rheidos.houdini.geo import OWNER_DETAIL, OWNER_POINT
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

from rheidos.apps.point_vortex.modules.surface_mesh import SurfaceMeshModule
from rheidos.apps.point_vortex.modules.point_vortex import PointVortexModule
from rheidos.apps.point_vortex.modules.pt_vortex_sim import PtVortexSimModule

# === IMPORT: change this to your app ===
from rheidos.apps.point_vortex.app import cook  # <-- your app's cook(ctx)

import taichi as ti


def _taichi_initialized() -> bool:
    checker = getattr(ti, "is_initialized", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    core = getattr(ti, "core", None)
    if core is not None and hasattr(core, "is_initialized"):
        try:
            return bool(core.is_initialized())
        except Exception:
            return False
    return False


def _ensure_taichi_init(session) -> None:
    if session.stats.get("taichi_initialized"):
        return
    if _taichi_initialized():
        session.stats["taichi_initialized"] = True
        return
    ti.init()
    session.stats["taichi_initialized"] = True


def _ensure_vector_field(ref, count: int, *, lanes: int, dtype) -> "ti.Field":

    shape = (count,)
    if count == 0:
        shape = ()

    if ref is None:
        return ti.Vector.field(lanes, dtype=dtype, shape=shape)

    field = ref.peek()
    if (
        field is None
        or tuple(field.shape) != shape
        or getattr(field, "n", lanes) != lanes
    ):
        field = ti.Vector.field(lanes, dtype=dtype, shape=shape)
        ref.set_buffer(field, bump=False)
    return field


def _ensure_scalar_field(ref, count: int, *, dtype) -> "ti.Field":

    shape = (count,)
    if count == 0:
        shape = ()

    if ref is None:
        return ti.field(dtype=dtype, shape=shape)

    field = ref.peek()
    if field is None or tuple(field.shape) != shape:
        field = ti.field(dtype=dtype, shape=shape)
        ref.set_buffer(field, bump=False)

    return field


def _get_input_geo(node: hou.Node, index: int = 0) -> hou.Geometry:
    inputs = node.inputs()
    if not inputs:
        raise RuntimeError("Connect input geometry to the Python SOP.")
    geo_in = inputs[index].geometry()
    if geo_in is None:
        raise RuntimeError("Input geometry is None.")
    return geo_in


def _seed_output(geo_out: hou.Geometry, geo_in: hou.Geometry) -> None:
    geo_out.clear()
    geo_out.merge(geo_in)


def _get_input_geos(node: hou.Node) -> list[hou.Geometry | None]:
    inputs = node.inputs()
    if not inputs:
        return []
    geos: list[hou.Geometry | None] = []
    for input_node in inputs:
        if input_node is None:
            geos.append(None)
            continue
        try:
            geo = input_node.geometry()
        except Exception:
            geo = None
        geos.append(geo)
    return geos


def node1() -> None:
    node = hou.pwd()
    geo_out = node.geometry()

    # Interactive debugging setup
    cfg = debug_config_from_node(node)
    ensure_debug_server(cfg, node=node)
    if consume_break_next_button(node):
        request_break_next(node=node)
    maybe_break_now(node=node)

    # Input mapping
    # 0 -> Triangle Mesh
    # 1 -> point vortices
    mesh_geo = _get_input_geo(node, index=0)
    points_geo = _get_input_geo(node, index=1)

    # Pass mesh through to output so downstream SOPs see the surface.
    _seed_output(geo_out, mesh_geo)

    session = get_runtime().get_or_create_session(node)

    ctx = build_cook_context(
        node, mesh_geo, geo_out, session, geo_inputs=[mesh_geo, points_geo]
    )
    world = ctx.world()

    _ensure_taichi_init(session)
    ## Read mesh input (index 0) explicitly via the primary IO.
    mesh_io = ctx.io
    mesh_points = mesh_io.read(OWNER_POINT, "P", components=3)
    mesh_triangles = mesh_io.read_prims(arity=3)
    nV = int(mesh_points.shape[0])
    nF = int(mesh_triangles.shape[0])

    mesh = world.require(SurfaceMeshModule)
    V = _ensure_vector_field(mesh.V_pos, nV, lanes=3, dtype=ti.f32)
    V.from_numpy(mesh_points)
    mesh.V_pos.commit()

    F = _ensure_vector_field(mesh.F_verts, nF, lanes=3, dtype=ti.i32)
    F.from_numpy(mesh_triangles)
    mesh.F_verts.commit()

    ## Read scatter points input (index 1) via the input IO.
    points_io = ctx.input_io(1)
    scatter_points = points_io.read(OWNER_POINT, "P", components=3)

    point_vortices = world.require(PointVortexModule)
    point_vortices.set_n_vortices(len(scatter_points))
    point_vortices.set_bary(points_io.read(OWNER_POINT, "bary", components=3))
    point_vortices.set_face_ids(points_io.read(OWNER_POINT, "faceid"))
    point_vortices.set_gammas(points_io.read(OWNER_POINT, "gamma"))

    ## Read stream function and set the output
    pt_vortex_sim = world.require(PtVortexSimModule)
    psi = pt_vortex_sim.psi.get().to_numpy()
    ctx.write(OWNER_POINT, "stream_func", psi, create=True)


def node2() -> None:
    # fetch session from the old node1 and output the updated points geometry
    # TODO: Make sure this runs after node1
    pass


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
    input_geos = _get_input_geos(node)
    if input_geos:
        input_geos[0] = geo_in
    else:
        input_geos = [geo_in]

    # 2) Get/create a persistent session for this node (cached across cooks)
    session = get_runtime().get_or_create_session(node)

    # 3) Build CookContext (input geo, output geo, session state)
    ctx = build_cook_context(node, geo_in, geo_out, session, geo_inputs=input_geos)

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
