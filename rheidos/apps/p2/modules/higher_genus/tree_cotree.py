from __future__ import annotations

from collections import deque
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute.resource import ResourceSpec
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import ModuleBase, World

_I32 = np.int32


def _as_i32(values: Iterable[int]) -> np.ndarray:
    return np.asarray(list(values), dtype=_I32)


def _validate_closed_mesh(
    *,
    n_vertices: int,
    n_faces: int,
    n_edges: int,
    boundary_edge_count: int,
) -> None:
    if n_vertices == 0 or n_faces == 0 or n_edges == 0:
        raise ValueError("Tree-cotree requires a non-empty triangular surface mesh.")
    if boundary_edge_count != 0:
        raise ValueError(
            "Tree-cotree homology generators currently require a closed surface; "
            f"found {boundary_edge_count} boundary edges."
        )


def _dual_edge_id_for_face_adjacency_slot(
    f_edges: np.ndarray, face_id: int, slot: int
) -> int:
    """Return the primal edge crossed by ``F_adj[face_id, slot]``.

    ``SurfaceMeshModule`` stores ``F_edges`` as directed triangle edges
    ``(a,b), (b,c), (c,a)``. Its ``F_adj`` slots are opposite vertices, so they
    cross ``(b,c), (c,a), (a,b)`` respectively. Keeping this mapping explicit
    prevents the tree-cotree code from depending on an accidental equal index.
    """

    return int(f_edges[face_id, (slot + 1) % 3])


def _build_dual_adjacency(
    *,
    f_adj: np.ndarray,
    f_edges: np.ndarray,
    e_faces: np.ndarray,
) -> List[List[Tuple[int, int]]]:
    n_faces = int(f_adj.shape[0])
    dual_adj: List[List[Tuple[int, int]]] = [[] for _ in range(n_faces)]

    for face_id in range(n_faces):
        for slot in range(3):
            neighbor_face = int(f_adj[face_id, slot])
            if neighbor_face < 0:
                continue
            if neighbor_face >= n_faces:
                raise ValueError(
                    f"Invalid dual adjacency from face {face_id} to {neighbor_face}."
                )

            edge_id = _dual_edge_id_for_face_adjacency_slot(f_edges, face_id, slot)
            adjacent_faces = {int(e_faces[edge_id, 0]), int(e_faces[edge_id, 1])}
            if face_id not in adjacent_faces or neighbor_face not in adjacent_faces:
                raise ValueError(
                    "Inconsistent mesh topology: face adjacency does not match "
                    f"edge {edge_id} adjacent faces."
                )

            dual_adj[face_id].append((neighbor_face, edge_id))

    for entries in dual_adj:
        entries.sort(key=lambda item: (item[0], item[1]))
    return dual_adj


def _build_primal_adjacency(
    *,
    e_verts: np.ndarray,
    blocked_edge_ids: set[int],
    n_vertices: int,
) -> List[List[Tuple[int, int]]]:
    primal_adj: List[List[Tuple[int, int]]] = [[] for _ in range(n_vertices)]

    for edge_id, (u_raw, v_raw) in enumerate(e_verts):
        if edge_id in blocked_edge_ids:
            continue

        u = int(u_raw)
        v = int(v_raw)
        primal_adj[u].append((v, edge_id))
        primal_adj[v].append((u, edge_id))

    for entries in primal_adj:
        entries.sort(key=lambda item: (item[0], item[1]))
    return primal_adj


def _build_tree(
    *,
    adjacency: Sequence[Sequence[Tuple[int, int]]],
    root: int,
    disconnected_message: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic BFS spanning tree over ``adjacency``.

    This is the "X-first search" from the reference: the bag is a queue here,
    and neighbors are sorted before traversal so the same mesh always produces
    the same tree resources.
    """

    n_nodes = len(adjacency)
    parent = np.full(n_nodes, -1, dtype=_I32)
    parent_edge = np.full(n_nodes, -1, dtype=_I32)
    depth = np.full(n_nodes, -1, dtype=_I32)
    tree_edge_ids: List[int] = []

    depth[root] = 0
    bag: deque[int] = deque([root])
    while bag:
        node = bag.popleft()
        for neighbor, edge_id in adjacency[node]:
            if depth[neighbor] >= 0:
                continue

            parent[neighbor] = node
            parent_edge[neighbor] = edge_id
            depth[neighbor] = depth[node] + 1
            tree_edge_ids.append(edge_id)
            bag.append(neighbor)

    if np.any(depth < 0):
        raise ValueError(disconnected_message)

    return _as_i32(tree_edge_ids), parent, parent_edge, depth


def _path_to_root(
    node: int,
    *,
    parent: np.ndarray,
    parent_edge: np.ndarray,
) -> tuple[List[int], List[int]]:
    nodes: List[int] = []
    edges: List[int] = []
    seen: set[int] = set()
    current = int(node)

    while current >= 0:
        if current in seen:
            raise ValueError("Invalid tree parent cycle detected.")
        seen.add(current)
        nodes.append(current)

        parent_node = int(parent[current])
        edge_id = int(parent_edge[current])
        if parent_node < 0:
            break

        edges.append(edge_id)
        current = parent_node

    return nodes, edges


def _tree_path(
    start: int,
    end: int,
    *,
    parent: np.ndarray,
    parent_edge: np.ndarray,
) -> tuple[List[int], List[int]]:
    """Return the unique tree path from ``start`` to ``end``."""

    start_nodes, start_edges = _path_to_root(
        start,
        parent=parent,
        parent_edge=parent_edge,
    )
    end_nodes, end_edges = _path_to_root(
        end,
        parent=parent,
        parent_edge=parent_edge,
    )
    end_index = {node: idx for idx, node in enumerate(end_nodes)}

    lca_start_index = -1
    lca_end_index = -1
    for idx, node in enumerate(start_nodes):
        if node in end_index:
            lca_start_index = idx
            lca_end_index = end_index[node]
            break

    if lca_start_index < 0:
        raise ValueError("Tree path endpoints are in different connected components.")

    path_nodes = start_nodes[: lca_start_index + 1] + list(
        reversed(end_nodes[:lca_end_index])
    )
    path_edges = start_edges[:lca_start_index] + list(
        reversed(end_edges[:lca_end_index])
    )
    return path_nodes, path_edges


def _edge_sign_for_direction(
    e_verts: np.ndarray,
    edge_id: int,
    start_vertex: int,
    end_vertex: int,
) -> int:
    u = int(e_verts[edge_id, 0])
    v = int(e_verts[edge_id, 1])
    if start_vertex == u and end_vertex == v:
        return 1
    if start_vertex == v and end_vertex == u:
        return -1
    raise ValueError(
        f"Edge {edge_id} does not connect vertices {start_vertex} and {end_vertex}."
    )


def _primal_generator_cycle(
    *,
    generator_edge_id: int,
    e_verts: np.ndarray,
    primal_parent_vertex: np.ndarray,
    primal_parent_edge: np.ndarray,
) -> np.ndarray:
    """Close a residual primal edge with the unique path through the primal tree."""

    u = int(e_verts[generator_edge_id, 0])
    v = int(e_verts[generator_edge_id, 1])
    path_vertices, path_edges = _tree_path(
        v,
        u,
        parent=primal_parent_vertex,
        parent_edge=primal_parent_edge,
    )

    # Orient the generator as the residual edge u->v followed by the tree path
    # v->u. Signs are relative to the canonical orientation in mesh.E_verts.
    cycle_rows: List[Tuple[int, int]] = [(generator_edge_id, 1)]
    for edge_id, start_vertex, end_vertex in zip(
        path_edges,
        path_vertices[:-1],
        path_vertices[1:],
    ):
        cycle_rows.append(
            (
                int(edge_id),
                _edge_sign_for_direction(
                    e_verts,
                    int(edge_id),
                    int(start_vertex),
                    int(end_vertex),
                ),
            )
        )

    return np.asarray(cycle_rows, dtype=_I32)


def _dual_generator_loop(
    *,
    generator_edge_id: int,
    e_faces: np.ndarray,
    f_edges: np.ndarray,
    f_edge_sign: np.ndarray,
    dual_parent_face: np.ndarray,
    dual_parent_edge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close a residual dual edge with the unique path through the dual tree."""

    f0 = int(e_faces[generator_edge_id, 0])
    f1 = int(e_faces[generator_edge_id, 1])
    tree_faces, tree_edges = _tree_path(
        f0,
        f1,
        parent=dual_parent_face,
        parent_edge=dual_parent_edge,
    )

    # The face loop walks through T* from f0 to f1, then crosses the residual
    # dual edge back to f0. The crossed-edge list aligns with consecutive face
    # pairs in the loop.
    face_loop = np.asarray([*tree_faces, f0], dtype=_I32)
    crossed_edges = np.asarray([*tree_edges, generator_edge_id], dtype=_I32)
    crossing_signs = _dual_crossing_signs(
        face_loop=face_loop,
        crossed_edges=crossed_edges,
        f_edges=f_edges,
        f_edge_sign=f_edge_sign,
    )
    return face_loop, crossed_edges, crossing_signs


def _dual_crossing_signs(
    *,
    face_loop: np.ndarray,
    crossed_edges: np.ndarray,
    f_edges: np.ndarray,
    f_edge_sign: np.ndarray,
) -> np.ndarray:
    """Orient dual crossings as primal-edge 1-form coefficients.

    When the generator crosses an edge from the face on the left of the
    canonical primal edge to the face on the right, the coefficient is +1; the
    reverse crossing is -1. In this mesh representation the source face is on
    the left exactly when its oriented boundary uses the canonical edge
    direction, so the coefficient is the source face's ``F_edge_sign`` for the
    crossed edge. With this convention incoming and outgoing crossings cancel
    under the primal face coboundary, making the matrix resource a closed
    discrete 1-form.
    """

    signs = np.empty(crossed_edges.shape, dtype=_I32)
    for index, edge_id_raw in enumerate(crossed_edges):
        edge_id = int(edge_id_raw)
        from_face = int(face_loop[index])
        local_slots = np.flatnonzero(f_edges[from_face] == edge_id)
        if local_slots.size != 1:
            raise ValueError(
                f"Dual generator loop crosses edge {edge_id}, but face "
                f"{from_face} does not contain that edge exactly once."
            )
        signs[index] = int(f_edge_sign[from_face, int(local_slots[0])])

    return signs


def _cycle_chain_matrix(cycles: Sequence[np.ndarray], n_edges: int) -> np.ndarray:
    chain = np.zeros((len(cycles), n_edges), dtype=_I32)
    for generator_id, cycle in enumerate(cycles):
        for edge_id, sign in cycle:
            chain[generator_id, int(edge_id)] += int(sign)
    return chain


def _closed_1form_matrix(
    crossed_edges: Sequence[np.ndarray],
    crossing_signs: Sequence[np.ndarray],
    n_edges: int,
    f_edges: np.ndarray,
    f_edge_sign: np.ndarray,
) -> np.ndarray:
    chain = np.zeros((len(crossed_edges), n_edges), dtype=_I32)
    for generator_id, (edges, signs) in enumerate(zip(crossed_edges, crossing_signs)):
        for edge_id, sign in zip(edges, signs):
            chain[generator_id, int(edge_id)] += int(sign)

    coboundary = np.einsum(
        "fl,gfl->gf",
        f_edge_sign,
        chain[:, f_edges],
    )
    if np.any(coboundary != 0):
        raise ValueError(
            "Computed dual generator 1-forms are not closed. Check that the "
            "input mesh is consistently oriented."
        )
    return chain


class TreeCotreeModule(ModuleBase):
    NAME = "TreeCotreeModule"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh

        self.dual_tree_edge_ids = self.resource(
            "dual_tree_edge_ids",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Primal edge ids crossed by the dual spanning tree T*. Shape: (nF-1,)",
        )
        self.dual_tree_parent_face = self.resource(
            "dual_tree_parent_face",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Parent face in the dual spanning tree T*. Root has parent -1. Shape: (nF,)",
        )
        self.dual_tree_parent_edge = self.resource(
            "dual_tree_parent_edge",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Primal edge id crossed from each face to its T* parent. Shape: (nF,)",
        )
        self.dual_tree_depth = self.resource(
            "dual_tree_depth",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Depth of each face in the dual spanning tree T*. Shape: (nF,)",
        )

        self.primal_tree_edge_ids = self.resource(
            "primal_tree_edge_ids",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Primal edge ids in the spanning tree T. Shape: (nV-1,)",
        )
        self.primal_tree_parent_vertex = self.resource(
            "primal_tree_parent_vertex",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Parent vertex in the primal spanning tree T. Root has parent -1. Shape: (nV,)",
        )
        self.primal_tree_parent_edge = self.resource(
            "primal_tree_parent_edge",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Primal edge id from each vertex to its T parent. Shape: (nV,)",
        )
        self.primal_tree_depth = self.resource(
            "primal_tree_depth",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Depth of each vertex in the primal spanning tree T. Shape: (nV,)",
        )

        self.generator_edge_ids = self.resource(
            "generator_edge_ids",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Residual primal edge ids, one seed per homology generator. Shape: (2g,)",
        )
        self.generator_count = self.resource(
            "generator_count",
            spec=ResourceSpec(kind="python", dtype=int),
            doc="Number of homology generator loops.",
        )
        self.genus = self.resource(
            "genus",
            spec=ResourceSpec(kind="python", dtype=int),
            doc="Orientable genus inferred from Euler characteristic.",
        )
        self.generator_edge_labels = self.resource(
            "generator_edge_labels",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Per-edge generator label, or -1 for non-generator edges. Shape: (nE,)",
        )
        self.primal_generator_cycles = self.resource(
            "primal_generator_cycles",
            spec=ResourceSpec(kind="python", dtype=list),
            doc="List of int32 arrays with rows (edge_id, sign), one closed primal cycle per generator.",
        )
        self.dual_generator_face_loops = self.resource(
            "dual_generator_face_loops",
            spec=ResourceSpec(kind="python", dtype=list),
            doc="List of int32 face-id loops in the dual mesh, closed by repeating the first face.",
        )
        self.dual_generator_crossed_edges = self.resource(
            "dual_generator_crossed_edges",
            spec=ResourceSpec(kind="python", dtype=list),
            doc="List of int32 primal edge ids crossed by each dual generator loop.",
        )
        self.dual_generator_crossing_signs = self.resource(
            "dual_generator_crossing_signs",
            spec=ResourceSpec(kind="python", dtype=list),
            doc="List of int32 DEC-oriented crossing signs aligned with dual_generator_crossed_edges.",
        )
        self.closed_dual_generator_1forms = self.resource(
            "closed_dual_generator_1forms",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Closed primal-edge 1-forms dual to the generator loops. Shape: (2g,nE)",
        )
        self.generator_chain_matrix = self.resource(
            "generator_chain_matrix",
            spec=ResourceSpec(kind="numpy", dtype=_I32),
            doc="Signed primal edge chains for all generators. Shape: (2g,nE)",
        )

        self.bind_producers()

    @producer(
        inputs=(
            "mesh.E_verts",
            "mesh.E_faces",
            "mesh.F_edges",
            "mesh.F_edge_sign",
            "mesh.F_adj",
            "mesh.V_pos",
            "mesh.F_verts",
            "mesh.boundary_edge_count",
        ),
        outputs=(
            "dual_tree_edge_ids",
            "dual_tree_parent_face",
            "dual_tree_parent_edge",
            "dual_tree_depth",
            "primal_tree_edge_ids",
            "primal_tree_parent_vertex",
            "primal_tree_parent_edge",
            "primal_tree_depth",
            "generator_edge_ids",
            "generator_count",
            "genus",
            "generator_edge_labels",
            "primal_generator_cycles",
            "dual_generator_face_loops",
            "dual_generator_crossed_edges",
            "dual_generator_crossing_signs",
            "closed_dual_generator_1forms",
            "generator_chain_matrix",
        ),
    )
    def build_tree_cotree_generators(self, ctx: ProducerContext) -> None:
        e_verts = self.mesh.E_verts.get()
        e_faces = self.mesh.E_faces.get()
        f_edges = self.mesh.F_edges.get()
        f_edge_sign = self.mesh.F_edge_sign.get()
        f_adj = self.mesh.F_adj.get()
        v_pos = self.mesh.V_pos.get()
        f_verts = self.mesh.F_verts.get()
        boundary_edge_count = int(self.mesh.boundary_edge_count.get())

        n_vertices = int(v_pos.shape[0])
        n_faces = int(f_verts.shape[0])
        n_edges = int(e_verts.shape[0])
        _validate_closed_mesh(
            n_vertices=n_vertices,
            n_faces=n_faces,
            n_edges=n_edges,
            boundary_edge_count=boundary_edge_count,
        )

        dual_adj = _build_dual_adjacency(
            f_adj=f_adj,
            f_edges=f_edges,
            e_faces=e_faces,
        )
        (
            dual_tree_edge_ids,
            dual_tree_parent_face,
            dual_tree_parent_edge,
            dual_tree_depth,
        ) = _build_tree(
            adjacency=dual_adj,
            root=0,
            disconnected_message="Tree-cotree requires a connected dual face graph.",
        )

        # Every dual tree edge crosses one primal edge. Tree-cotree removes
        # those crossed primal edges before building T, so T and T* are
        # disjoint and the leftovers are exactly the homology generators.
        blocked_by_dual_tree = {int(edge_id) for edge_id in dual_tree_edge_ids}
        primal_adj = _build_primal_adjacency(
            e_verts=e_verts,
            blocked_edge_ids=blocked_by_dual_tree,
            n_vertices=n_vertices,
        )
        (
            primal_tree_edge_ids,
            primal_tree_parent_vertex,
            primal_tree_parent_edge,
            primal_tree_depth,
        ) = _build_tree(
            adjacency=primal_adj,
            root=0,
            disconnected_message=(
                "Tree-cotree requires a connected primal graph after removing "
                "edges crossed by the dual tree."
            ),
        )

        primal_tree_edges = {int(edge_id) for edge_id in primal_tree_edge_ids}
        generator_edge_ids = _as_i32(
            edge_id
            for edge_id in range(n_edges)
            if edge_id not in blocked_by_dual_tree and edge_id not in primal_tree_edges
        )

        chi = n_vertices - n_edges + n_faces
        if (2 - chi) % 2 != 0:
            raise ValueError(
                "Tree-cotree expected an orientable closed surface with integral genus; "
                f"Euler characteristic is {chi}."
            )

        genus = (2 - chi) // 2
        generator_count = int(generator_edge_ids.shape[0])
        if genus < 0 or generator_count != 2 * genus:
            raise ValueError(
                "Tree-cotree decomposition did not match the closed orientable "
                f"surface invariant: residual edges={generator_count}, genus={genus}."
            )

        generator_edge_labels = np.full(n_edges, -1, dtype=_I32)
        primal_generator_cycles: List[np.ndarray] = []
        dual_generator_face_loops: List[np.ndarray] = []
        dual_generator_crossed_edges: List[np.ndarray] = []
        dual_generator_crossing_signs: List[np.ndarray] = []

        for generator_id, edge_id_raw in enumerate(generator_edge_ids):
            edge_id = int(edge_id_raw)
            generator_edge_labels[edge_id] = generator_id

            primal_generator_cycles.append(
                _primal_generator_cycle(
                    generator_edge_id=edge_id,
                    e_verts=e_verts,
                    primal_parent_vertex=primal_tree_parent_vertex,
                    primal_parent_edge=primal_tree_parent_edge,
                )
            )
            face_loop, crossed_edges, crossing_signs = _dual_generator_loop(
                generator_edge_id=edge_id,
                e_faces=e_faces,
                f_edges=f_edges,
                f_edge_sign=f_edge_sign,
                dual_parent_face=dual_tree_parent_face,
                dual_parent_edge=dual_tree_parent_edge,
            )
            dual_generator_face_loops.append(face_loop)
            dual_generator_crossed_edges.append(crossed_edges)
            dual_generator_crossing_signs.append(crossing_signs)

        generator_chain_matrix = _cycle_chain_matrix(
            primal_generator_cycles,
            n_edges,
        )
        closed_dual_generator_1forms = _closed_1form_matrix(
            dual_generator_crossed_edges,
            dual_generator_crossing_signs,
            n_edges,
            f_edges,
            f_edge_sign,
        )

        ctx.commit(
            dual_tree_edge_ids=dual_tree_edge_ids,
            dual_tree_parent_face=dual_tree_parent_face,
            dual_tree_parent_edge=dual_tree_parent_edge,
            dual_tree_depth=dual_tree_depth,
            primal_tree_edge_ids=primal_tree_edge_ids,
            primal_tree_parent_vertex=primal_tree_parent_vertex,
            primal_tree_parent_edge=primal_tree_parent_edge,
            primal_tree_depth=primal_tree_depth,
            generator_edge_ids=generator_edge_ids,
            generator_count=generator_count,
            genus=int(genus),
            generator_edge_labels=generator_edge_labels,
            primal_generator_cycles=primal_generator_cycles,
            dual_generator_face_loops=dual_generator_face_loops,
            dual_generator_crossed_edges=dual_generator_crossed_edges,
            dual_generator_crossing_signs=dual_generator_crossing_signs,
            closed_dual_generator_1forms=closed_dual_generator_1forms,
            generator_chain_matrix=generator_chain_matrix,
        )
