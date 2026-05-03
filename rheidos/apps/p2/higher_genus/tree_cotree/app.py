import numpy as np

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.apps.p2.modules.tree_cotree.tree_cotree_module import TreeCotreeModule
from rheidos.compute.world import ModuleBase, World
from rheidos.houdini.runtime.cook_context import CookContext

from ..io import copy_input_to_output, load_mesh_input


class TreeCotreeApp(ModuleBase):
    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(DEC, mesh=self.mesh)
        self.tree_cotree = self.require(TreeCotreeModule, mesh=self.mesh)


def setup_mesh(ctx: CookContext):
    mods = ctx.world().require(TreeCotreeApp)
    load_mesh_input(
        ctx,
        mods.mesh,
        missing_message="Input 0 has to be mesh input geometry",
    )


def _load_tree_cotree(ctx: CookContext) -> TreeCotreeApp:
    mods = ctx.world().require(TreeCotreeApp)
    load_mesh_input(
        ctx,
        mods.mesh,
        missing_message="Input 0 has to be mesh input geometry",
    )
    return mods


def _face_centers_and_normals(mods: TreeCotreeApp) -> tuple[np.ndarray, np.ndarray]:
    vertices = mods.mesh.V_pos.get()
    faces = mods.mesh.F_verts.get()
    centers = vertices[faces].mean(axis=1)
    normals = mods.mesh.F_normal.get()
    return centers, normals


def _write_tree_cotree_detail(ctx: CookContext, mods: TreeCotreeApp) -> None:
    tree_cotree = mods.tree_cotree
    ctx.write_detail(
        "genus",
        np.array([tree_cotree.genus.get()], dtype=np.int32),
        create=True,
    )
    ctx.write_detail(
        "generator_count",
        np.array([tree_cotree.generator_count.get()], dtype=np.int32),
        create=True,
    )


def export_generator_dual_loops(ctx: CookContext, *, offset: float = 0.01) -> None:
    """Create loop curve primitives through generator dual-loop face centers."""

    mods = _load_tree_cotree(ctx)
    centers, normals = _face_centers_and_normals(mods)
    face_loops = mods.tree_cotree.dual_generator_face_loops.get()

    point_positions = []
    point_src_face = []
    point_loop_id = []
    point_loop_order = []
    polygons = []
    prim_loop_id = []
    prim_kind = []

    for loop_id, face_loop in enumerate(face_loops):
        ordered_faces = [int(face_id) for face_id in face_loop.tolist()]
        if len(ordered_faces) > 1 and ordered_faces[0] != ordered_faces[-1]:
            ordered_faces.append(ordered_faces[0])

        polygon = []
        for loop_order, face_id in enumerate(ordered_faces):
            polygon.append(len(point_positions))
            point_positions.append(centers[face_id] + offset * normals[face_id])
            point_src_face.append(face_id)
            point_loop_id.append(loop_id)
            point_loop_order.append(loop_order)

        if polygon:
            polygons.append(polygon)
            prim_loop_id.append(loop_id)
            prim_kind.append("generator_dual_loop")

    ctx.clear_output()
    if point_positions:
        ctx.create_points(np.asarray(point_positions, dtype=np.float64))
        # In Houdini, a closed polygon primitive is a filled surface. To draw a
        # visual loop as a curve, keep the polygon open and duplicate the first
        # point at the end so the endpoints meet in space.
        ctx.create_polygons(polygons, closed=False)

    ctx.write_point("src_face", np.asarray(point_src_face, dtype=np.int32), create=True)
    ctx.write_point("loop_id", np.asarray(point_loop_id, dtype=np.int32), create=True)
    ctx.write_point(
        "loop_order",
        np.asarray(point_loop_order, dtype=np.int32),
        create=True,
    )
    ctx.write_prim("loop_id", np.asarray(prim_loop_id, dtype=np.int32), create=True)
    ctx.write_prim("kind", np.asarray(prim_kind, dtype=str), create=True)
    _write_tree_cotree_detail(ctx, mods)


def export_dual_tree(ctx: CookContext, *, offset: float = 0.01) -> None:
    """Create open curve primitives between adjacent face centers in T*."""

    mods = _load_tree_cotree(ctx)
    centers, normals = _face_centers_and_normals(mods)
    e_faces = mods.mesh.E_faces.get()
    dual_tree_edges = mods.tree_cotree.dual_tree_edge_ids.get()

    point_positions = []
    point_src_face = []
    point_src_edge = []
    point_tree_id = []
    point_endpoint_order = []
    polygons = []
    prim_src_edge = []
    prim_tree_id = []
    prim_kind = []

    for tree_id, edge_id_raw in enumerate(dual_tree_edges):
        edge_id = int(edge_id_raw)
        f0 = int(e_faces[edge_id, 0])
        f1 = int(e_faces[edge_id, 1])
        polygon = []
        for endpoint_order, face_id in enumerate((f0, f1)):
            polygon.append(len(point_positions))
            point_positions.append(centers[face_id] + offset * normals[face_id])
            point_src_face.append(face_id)
            point_src_edge.append(edge_id)
            point_tree_id.append(tree_id)
            point_endpoint_order.append(endpoint_order)
        polygons.append(polygon)
        prim_src_edge.append(edge_id)
        prim_tree_id.append(tree_id)
        prim_kind.append("dual_tree")

    ctx.clear_output()
    if point_positions:
        ctx.create_points(np.asarray(point_positions, dtype=np.float64))
        ctx.create_polygons(polygons, closed=False)

    ctx.write_point("src_face", np.asarray(point_src_face, dtype=np.int32), create=True)
    ctx.write_point("src_edge", np.asarray(point_src_edge, dtype=np.int32), create=True)
    ctx.write_point("tree_id", np.asarray(point_tree_id, dtype=np.int32), create=True)
    ctx.write_point(
        "endpoint_order",
        np.asarray(point_endpoint_order, dtype=np.int32),
        create=True,
    )
    ctx.write_prim("src_edge", np.asarray(prim_src_edge, dtype=np.int32), create=True)
    ctx.write_prim("tree_id", np.asarray(prim_tree_id, dtype=np.int32), create=True)
    ctx.write_prim("kind", np.asarray(prim_kind, dtype=str), create=True)
    _write_tree_cotree_detail(ctx, mods)


def export_primal_tree(ctx: CookContext) -> None:
    """Create open curve primitives between primal mesh vertices in T."""

    mods = _load_tree_cotree(ctx)
    vertices = mods.mesh.V_pos.get()
    e_verts = mods.mesh.E_verts.get()
    primal_tree_edges = mods.tree_cotree.primal_tree_edge_ids.get()

    point_positions = []
    point_src_vertex = []
    point_src_edge = []
    point_tree_id = []
    point_endpoint_order = []
    polygons = []
    prim_src_edge = []
    prim_tree_id = []
    prim_kind = []

    for tree_id, edge_id_raw in enumerate(primal_tree_edges):
        edge_id = int(edge_id_raw)
        v0 = int(e_verts[edge_id, 0])
        v1 = int(e_verts[edge_id, 1])
        polygon = []
        for endpoint_order, vertex_id in enumerate((v0, v1)):
            polygon.append(len(point_positions))
            point_positions.append(vertices[vertex_id])
            point_src_vertex.append(vertex_id)
            point_src_edge.append(edge_id)
            point_tree_id.append(tree_id)
            point_endpoint_order.append(endpoint_order)
        polygons.append(polygon)
        prim_src_edge.append(edge_id)
        prim_tree_id.append(tree_id)
        prim_kind.append("primal_tree")

    ctx.clear_output()
    if point_positions:
        ctx.create_points(np.asarray(point_positions, dtype=np.float64))
        ctx.create_polygons(polygons, closed=False)

    ctx.write_point(
        "src_vertex",
        np.asarray(point_src_vertex, dtype=np.int32),
        create=True,
    )
    ctx.write_point("src_edge", np.asarray(point_src_edge, dtype=np.int32), create=True)
    ctx.write_point("tree_id", np.asarray(point_tree_id, dtype=np.int32), create=True)
    ctx.write_point(
        "endpoint_order",
        np.asarray(point_endpoint_order, dtype=np.int32),
        create=True,
    )
    ctx.write_prim("src_edge", np.asarray(prim_src_edge, dtype=np.int32), create=True)
    ctx.write_prim("tree_id", np.asarray(prim_tree_id, dtype=np.int32), create=True)
    ctx.write_prim("kind", np.asarray(prim_kind, dtype=str), create=True)
    _write_tree_cotree_detail(ctx, mods)


def mark_generator_faces(ctx: CookContext) -> None:
    """Copy the input mesh and mark faces used by any generator dual loop."""

    mods = _load_tree_cotree(ctx)
    copy_input_to_output(ctx, 0)
    n_faces = int(mods.mesh.F_verts.get().shape[0])
    loop_id = np.full(n_faces, -1, dtype=np.int32)
    in_generator_loop = np.zeros(n_faces, dtype=np.int32)

    face_loops = mods.tree_cotree.dual_generator_face_loops.get()
    for generator_id, face_loop in enumerate(face_loops):
        for face_id_raw in face_loop:
            face_id = int(face_id_raw)
            loop_id[face_id] = generator_id
            in_generator_loop[face_id] = 1

    ctx.write_prim("loop_id", loop_id, create=True)
    ctx.write_prim("in_generator_loop", in_generator_loop, create=True)
    _write_tree_cotree_detail(ctx, mods)
