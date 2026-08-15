from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.surface_mesh.mesh_topology import build_mesh_topology
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ProducerContext, ResourceSpec, World, producer


@dataclass(frozen=True)
class _Halfedge:
    face: int
    start: int
    end: int
    start_corner: int
    end_corner: int


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        parent = self.parent
        root = int(item)
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[item]) != item:
            next_item = int(parent[item])
            parent[item] = root
            item = next_item
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class OrientationDoubleCoverData:
    """A quotient-built orientable double cover and its deck operators."""

    cover_vertices: np.ndarray
    cover_faces: np.ndarray
    cover_edges: np.ndarray
    cover_edge_faces: np.ndarray
    cover_face_edges: np.ndarray
    cover_face_edge_sign: np.ndarray
    seam_edge_pairs: np.ndarray
    base_vertex_representative: np.ndarray
    base_edge_representative: np.ndarray
    pi_vertex: np.ndarray
    pi_edge: np.ndarray
    pi_face: np.ndarray
    tau_vertex: np.ndarray
    tau_edge: np.ndarray
    tau_face: np.ndarray
    edge_sign: np.ndarray
    face_sign: np.ndarray
    P0: csr_matrix
    P1: csr_matrix
    P2: csr_matrix


def _validate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices_array = np.ascontiguousarray(vertices, dtype=np.float64)
    faces_array = np.ascontiguousarray(faces, dtype=np.int32)

    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise ValueError(
            f"base_vertices must have shape (nV,3), got {vertices_array.shape}"
        )
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise ValueError(f"base_faces must have shape (nF,3), got {faces_array.shape}")
    if faces_array.shape[0] == 0:
        raise ValueError("The orientation cover requires at least one triangle")
    if not np.all(np.isfinite(vertices_array)):
        raise ValueError("base_vertices contains non-finite coordinates")
    if int(faces_array.min()) < 0 or int(faces_array.max()) >= vertices_array.shape[0]:
        raise ValueError("base_faces contains an out-of-range vertex index")
    if np.any(
        (faces_array[:, 0] == faces_array[:, 1])
        | (faces_array[:, 1] == faces_array[:, 2])
        | (faces_array[:, 2] == faces_array[:, 0])
    ):
        raise ValueError("base_faces contains a triangle with repeated vertices")

    return vertices_array, faces_array


def _edge_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _build_edge_incidence(
    faces: np.ndarray,
) -> tuple[
    list[tuple[int, int]],
    dict[tuple[int, int], int],
    dict[tuple[int, int], list[_Halfedge]],
]:
    edge_ids: dict[tuple[int, int], int] = {}
    edge_keys: list[tuple[int, int]] = []
    incidence: dict[tuple[int, int], list[_Halfedge]] = defaultdict(list)

    for face_id, triangle in enumerate(faces):
        for start_corner, end_corner in ((0, 1), (1, 2), (2, 0)):
            start = int(triangle[start_corner])
            end = int(triangle[end_corner])
            key = _edge_key(start, end)
            if key not in edge_ids:
                edge_ids[key] = len(edge_keys)
                edge_keys.append(key)
            incidence[key].append(
                _Halfedge(
                    face=face_id,
                    start=start,
                    end=end,
                    start_corner=start_corner,
                    end_corner=end_corner,
                )
            )
            if len(incidence[key]) > 2:
                raise ValueError(f"Non-manifold base edge detected at {key}")

    return edge_keys, edge_ids, incidence


def _normalize_seam_edge_pairs(
    seam_edge_pairs: np.ndarray | Sequence[Sequence[int]],
    *,
    vertex_count: int,
    incidence: dict[tuple[int, int], list[_Halfedge]],
) -> np.ndarray:
    raw = np.asarray(seam_edge_pairs)
    if raw.size == 0:
        return np.empty((0, 4), dtype=np.int32)
    if raw.ndim == 3 and raw.shape[1:] == (2, 2):
        raw = raw.reshape((-1, 4))
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(
            "seam_edge_pairs must have shape (n,4) or (n,2,2); each row "
            "is (u, v, u_prime, v_prime) with endpoint correspondence"
        )
    if not np.issubdtype(raw.dtype, np.integer):
        try:
            integral = np.equal(raw, np.floor(raw))
        except (TypeError, ValueError):
            integral = np.zeros(raw.shape, dtype=bool)
        if not np.all(integral):
            raise ValueError("seam_edge_pairs must contain integer vertex ids")

    pairs = np.ascontiguousarray(raw, dtype=np.int32)
    if int(pairs.min()) < 0 or int(pairs.max()) >= vertex_count:
        raise ValueError("seam_edge_pairs contains an out-of-range vertex index")

    used_edges: set[tuple[int, int]] = set()
    for row in pairs:
        u, v, u_prime, v_prime = (int(value) for value in row)
        if u == v or u_prime == v_prime:
            raise ValueError("A seam edge cannot have identical endpoints")
        first = _edge_key(u, v)
        second = _edge_key(u_prime, v_prime)
        if first == second:
            raise ValueError(f"A seam edge cannot be paired with itself: {first}")
        for edge in (first, second):
            incident = incidence.get(edge)
            if incident is None:
                raise ValueError(f"Seam edge {edge} is not present in base_faces")
            if len(incident) != 1:
                raise ValueError(f"Seam edge {edge} is not a boundary edge")
            if edge in used_edges:
                raise ValueError(f"Seam edge {edge} occurs in more than one seam pair")
            used_edges.add(edge)

    return pairs


def infer_coincident_boundary_edge_pairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Infer cut-seam edge pairs whose corresponding endpoints coincide.

    ``tolerance`` is relative to the mesh bounding-box diagonal (with a minimum
    scale of one). An explicit empty seam array can be supplied to
    :func:`build_orientation_double_cover` to disable inference.
    """

    vertices_array, faces_array = _validate_mesh(vertices, faces)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite number")

    _, _, incidence = _build_edge_incidence(faces_array)
    boundary_edges = sorted(edge for edge, sides in incidence.items() if len(sides) == 1)
    if not boundary_edges:
        return np.empty((0, 4), dtype=np.int32)

    boundary_vertices = np.unique(np.asarray(boundary_edges, dtype=np.int32).reshape(-1))
    scale = max(float(np.linalg.norm(np.ptp(vertices_array, axis=0))), 1.0)
    absolute_tolerance = float(tolerance) * scale
    tree = cKDTree(vertices_array[boundary_vertices])
    nearby: dict[int, tuple[int, ...]] = {}
    for vertex_id in boundary_vertices:
        matches = tree.query_ball_point(vertices_array[int(vertex_id)], absolute_tolerance)
        nearby[int(vertex_id)] = tuple(
            sorted(int(boundary_vertices[match]) for match in matches)
        )

    boundary_lookup = set(boundary_edges)
    candidate_by_edge: dict[
        tuple[int, int], tuple[tuple[int, int], int, int]
    ] = {}
    for edge in boundary_edges:
        u, v = edge
        candidates: set[tuple[tuple[int, int], int, int]] = set()
        for u_prime in nearby[u]:
            for v_prime in nearby[v]:
                other = _edge_key(u_prime, v_prime)
                if other != edge and other in boundary_lookup:
                    candidates.add((other, u_prime, v_prime))
        if len(candidates) > 1:
            others = sorted(candidate[0] for candidate in candidates)
            raise ValueError(
                f"Ambiguous coincident seam match for boundary edge {edge}: {others}"
            )
        if candidates:
            candidate_by_edge[edge] = next(iter(candidates))

    rows: list[tuple[int, int, int, int]] = []
    used: set[tuple[int, int]] = set()
    for edge in boundary_edges:
        if edge in used or edge not in candidate_by_edge:
            continue
        other, u_prime, v_prime = candidate_by_edge[edge]
        reverse_candidate = candidate_by_edge.get(other)
        if reverse_candidate is None or reverse_candidate[0] != edge:
            raise ValueError(
                f"Coincident seam matching is not mutual for edges {edge} and {other}"
            )
        rows.append((edge[0], edge[1], u_prime, v_prime))
        used.add(edge)
        used.add(other)

    if not rows:
        return np.empty((0, 4), dtype=np.int32)
    return np.ascontiguousarray(rows, dtype=np.int32)


def _canonical_representatives(union_find: _UnionFind) -> np.ndarray:
    groups: dict[int, list[int]] = defaultdict(list)
    for item in range(union_find.parent.shape[0]):
        groups[union_find.find(item)].append(item)
    representatives = np.empty(union_find.parent.shape[0], dtype=np.int32)
    for members in groups.values():
        representative = min(members)
        representatives[members] = representative
    return representatives


def _corner_id(face: int, copy: int, local_corner: int) -> int:
    return (2 * face + copy) * 3 + local_corner


def _halfedge_direction(halfedge: _Halfedge, start: int, end: int) -> int:
    if halfedge.start == start and halfedge.end == end:
        return 1
    if halfedge.start == end and halfedge.end == start:
        return -1
    raise RuntimeError(
        f"Halfedge {(halfedge.start, halfedge.end)} does not match {(start, end)}"
    )


def _corner_for_vertex(halfedge: _Halfedge, vertex: int) -> int:
    if halfedge.start == vertex:
        return halfedge.start_corner
    if halfedge.end == vertex:
        return halfedge.end_corner
    raise RuntimeError(
        f"Vertex {vertex} is not on halfedge {(halfedge.start, halfedge.end)}"
    )


def _glue_halfedges(
    corners: _UnionFind,
    first: _Halfedge,
    second: _Halfedge,
    *,
    first_endpoints: tuple[int, int],
    second_endpoints: tuple[int, int],
) -> None:
    first_u, first_v = first_endpoints
    second_u, second_v = second_endpoints
    first_direction = _halfedge_direction(first, first_u, first_v)
    second_direction = _halfedge_direction(second, second_u, second_v)
    cross = first_direction == second_direction

    first_u_corner = _corner_for_vertex(first, first_u)
    first_v_corner = _corner_for_vertex(first, first_v)
    second_u_corner = _corner_for_vertex(second, second_u)
    second_v_corner = _corner_for_vertex(second, second_v)

    for first_copy in (0, 1):
        second_copy = first_copy ^ int(cross)
        corners.union(
            _corner_id(first.face, first_copy, first_u_corner),
            _corner_id(second.face, second_copy, second_u_corner),
        )
        corners.union(
            _corner_id(first.face, first_copy, first_v_corner),
            _corner_id(second.face, second_copy, second_v_corner),
        )


def _signed_permutation(
    permutation: np.ndarray,
    signs: np.ndarray,
) -> csr_matrix:
    size = int(permutation.shape[0])
    return csr_matrix(
        (
            np.asarray(signs, dtype=np.int8),
            (np.arange(size, dtype=np.int32), permutation),
        ),
        shape=(size, size),
        dtype=np.int8,
    )


def _assert_sparse_zero(matrix: csr_matrix, message: str) -> None:
    matrix = matrix.copy()
    matrix.eliminate_zeros()
    if matrix.nnz:
        raise RuntimeError(message)


def validate_orientation_double_cover(data: OrientationDoubleCoverData) -> None:
    """Validate lift counts, deck involutions, orientation, and DEC chain maps."""

    vertices = data.cover_vertices
    faces = data.cover_faces
    edges = data.cover_edges
    n_vertices = vertices.shape[0]
    n_edges = edges.shape[0]
    n_faces = faces.shape[0]

    deck_data = (
        ("vertex", data.tau_vertex, np.ones(n_vertices, dtype=np.int8)),
        ("edge", data.tau_edge, data.edge_sign),
        ("face", data.tau_face, data.face_sign),
    )
    for name, permutation, signs in deck_data:
        expected_size = {"vertex": n_vertices, "edge": n_edges, "face": n_faces}[name]
        if permutation.shape != (expected_size,) or signs.shape != (expected_size,):
            raise RuntimeError(f"Invalid {name} deck-map shape")
        if np.any(permutation < 0) or np.any(permutation >= expected_size):
            raise RuntimeError(f"The {name} deck map contains an invalid index")
        if not np.array_equal(permutation[permutation], np.arange(expected_size)):
            raise RuntimeError(f"The {name} deck map is not an involution")
        if not np.array_equal(signs * signs[permutation], np.ones(expected_size)):
            raise RuntimeError(f"The signed {name} deck operator is not an involution")
        if np.any(permutation == np.arange(expected_size)):
            raise RuntimeError(f"The {name} deck map has a fixed simplex")

    if not np.array_equal(data.pi_vertex[data.tau_vertex], data.pi_vertex):
        raise RuntimeError("The vertex deck map does not preserve projection")
    if not np.array_equal(data.pi_edge[data.tau_edge], data.pi_edge):
        raise RuntimeError("The edge deck map does not preserve projection")
    if not np.array_equal(data.pi_face[data.tau_face], data.pi_face):
        raise RuntimeError("The face deck map does not preserve projection")
    if not np.allclose(vertices[data.tau_vertex], vertices, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Deck-paired cover vertices do not have the same projection")

    for projection_name, projection in (
        ("vertex", data.pi_vertex),
        ("edge", data.pi_edge),
        ("face", data.pi_face),
    ):
        counts = Counter(int(value) for value in projection)
        if any(count != 2 for count in counts.values()):
            raise RuntimeError(f"Every projected base {projection_name} must have two lifts")

    for edge_id, adjacent_faces in enumerate(data.cover_edge_faces):
        if int(adjacent_faces[1]) < 0:
            continue
        signs = []
        for face_id in adjacent_faces:
            slots = np.flatnonzero(data.cover_face_edges[int(face_id)] == edge_id)
            if slots.shape[0] != 1:
                raise RuntimeError("Invalid cover face-edge incidence")
            signs.append(data.cover_face_edge_sign[int(face_id), int(slots[0])])
        if int(signs[0]) != -int(signs[1]):
            raise RuntimeError("The constructed cover has inconsistent face winding")

    row_edges = np.repeat(np.arange(n_edges, dtype=np.int32), 2)
    column_vertices = edges.reshape(-1)
    d0 = csr_matrix(
        (
            np.tile(np.array([-1, 1], dtype=np.int8), n_edges),
            (row_edges, column_vertices),
        ),
        shape=(n_edges, n_vertices),
        dtype=np.int8,
    )

    row_faces = np.repeat(np.arange(n_faces, dtype=np.int32), 3)
    d1 = csr_matrix(
        (
            data.cover_face_edge_sign.reshape(-1).astype(np.int8, copy=False),
            (row_faces, data.cover_face_edges.reshape(-1)),
        ),
        shape=(n_faces, n_edges),
        dtype=np.int8,
    )

    _assert_sparse_zero(d1 @ d0, "The cover incidence does not satisfy d1 d0 = 0")
    _assert_sparse_zero(
        d0 @ data.P0 - data.P1 @ d0,
        "The deck operators do not satisfy d0 P0 = P1 d0",
    )
    _assert_sparse_zero(
        d1 @ data.P1 - data.P2 @ d1,
        "The deck operators do not satisfy d1 P1 = P2 d1",
    )


def build_orientation_double_cover(
    base_vertices: np.ndarray,
    base_faces: np.ndarray,
    *,
    seam_edge_pairs: np.ndarray | Sequence[Sequence[int]] | None = None,
    seam_tolerance: float = 1.0e-6,
) -> OrientationDoubleCoverData:
    """Construct the genuine orientable double-cover triangle mesh.

    ``seam_edge_pairs`` rows are ``(u, v, u_prime, v_prime)``. The endpoint
    correspondence is ``u <-> u_prime`` and ``v <-> v_prime``. Passing
    ``None`` infers coincident cut-seam boundary edges; passing an empty array
    explicitly disables seam gluing.
    """

    vertices, faces = _validate_mesh(base_vertices, base_faces)
    edge_keys, edge_ids, incidence = _build_edge_incidence(faces)

    if seam_edge_pairs is None:
        resolved_seams = infer_coincident_boundary_edge_pairs(
            vertices,
            faces,
            tolerance=seam_tolerance,
        )
    else:
        resolved_seams = _normalize_seam_edge_pairs(
            seam_edge_pairs,
            vertex_count=vertices.shape[0],
            incidence=incidence,
        )
    # Inferred pairs go through the same topological checks as explicit pairs.
    resolved_seams = _normalize_seam_edge_pairs(
        resolved_seams,
        vertex_count=vertices.shape[0],
        incidence=incidence,
    )

    corner_union = _UnionFind(2 * faces.shape[0] * 3)
    base_vertex_union = _UnionFind(vertices.shape[0])
    base_edge_union = _UnionFind(len(edge_keys))

    for edge, halfedges in incidence.items():
        if len(halfedges) == 2:
            _glue_halfedges(
                corner_union,
                halfedges[0],
                halfedges[1],
                first_endpoints=edge,
                second_endpoints=edge,
            )

    for row in resolved_seams:
        u, v, u_prime, v_prime = (int(value) for value in row)
        first_key = _edge_key(u, v)
        second_key = _edge_key(u_prime, v_prime)
        _glue_halfedges(
            corner_union,
            incidence[first_key][0],
            incidence[second_key][0],
            first_endpoints=(u, v),
            second_endpoints=(u_prime, v_prime),
        )
        base_vertex_union.union(u, u_prime)
        base_vertex_union.union(v, v_prime)
        base_edge_union.union(edge_ids[first_key], edge_ids[second_key])

    base_vertex_representative = _canonical_representatives(base_vertex_union)
    base_edge_representative = _canonical_representatives(base_edge_union)

    root_to_cover_vertex: dict[int, int] = {}
    corner_to_cover_vertex = np.empty(2 * faces.shape[0] * 3, dtype=np.int32)
    cover_corner_members: list[list[int]] = []
    for corner in range(corner_to_cover_vertex.shape[0]):
        root = corner_union.find(corner)
        cover_vertex = root_to_cover_vertex.get(root)
        if cover_vertex is None:
            cover_vertex = len(root_to_cover_vertex)
            root_to_cover_vertex[root] = cover_vertex
            cover_corner_members.append([])
        corner_to_cover_vertex[corner] = cover_vertex
        cover_corner_members[cover_vertex].append(corner)

    cover_vertices = np.empty((len(cover_corner_members), 3), dtype=np.float64)
    pi_vertex = np.empty(len(cover_corner_members), dtype=np.int32)
    representative_corner = np.empty(len(cover_corner_members), dtype=np.int64)
    for cover_vertex, corners in enumerate(cover_corner_members):
        representative_corner[cover_vertex] = corners[0]
        base_vertex_ids: set[int] = set()
        projected_ids: set[int] = set()
        for corner in corners:
            lifted_face, local_corner = divmod(corner, 3)
            base_face = lifted_face // 2
            base_vertex = int(faces[base_face, local_corner])
            base_vertex_ids.add(base_vertex)
            projected_ids.add(int(base_vertex_representative[base_vertex]))
        if len(projected_ids) != 1:
            raise RuntimeError(
                "A cover-vertex union class projects to multiple base vertices"
            )
        pi_vertex[cover_vertex] = next(iter(projected_ids))
        cover_vertices[cover_vertex] = np.mean(
            vertices[sorted(base_vertex_ids)],
            axis=0,
        )

    cover_faces = np.empty((2 * faces.shape[0], 3), dtype=np.int32)
    for face_id in range(faces.shape[0]):
        cover_faces[2 * face_id] = corner_to_cover_vertex[
            [_corner_id(face_id, 0, corner) for corner in (0, 1, 2)]
        ]
        cover_faces[2 * face_id + 1] = corner_to_cover_vertex[
            [_corner_id(face_id, 1, corner) for corner in (0, 2, 1)]
        ]
    if np.any(
        (cover_faces[:, 0] == cover_faces[:, 1])
        | (cover_faces[:, 1] == cover_faces[:, 2])
        | (cover_faces[:, 2] == cover_faces[:, 0])
    ):
        raise RuntimeError("Cover gluing collapsed a lifted triangle")

    tau_vertex = np.empty(cover_vertices.shape[0], dtype=np.int32)
    for cover_vertex, corner in enumerate(representative_corner):
        lifted_face, local_corner = divmod(int(corner), 3)
        base_face, copy = divmod(lifted_face, 2)
        deck_corner = _corner_id(base_face, 1 - copy, local_corner)
        tau_vertex[cover_vertex] = corner_to_cover_vertex[
            corner_union.find(deck_corner)
        ]
    # The result must not depend on which corner represents a quotient vertex.
    for cover_vertex, corners in enumerate(cover_corner_members):
        for corner in corners:
            lifted_face, local_corner = divmod(corner, 3)
            base_face, copy = divmod(lifted_face, 2)
            deck_corner = _corner_id(base_face, 1 - copy, local_corner)
            if (
                int(corner_to_cover_vertex[corner_union.find(deck_corner)])
                != int(tau_vertex[cover_vertex])
            ):
                raise RuntimeError("The vertex deck map depends on corner representative")

    topology = build_mesh_topology(cover_vertices, cover_faces)
    (
        _,
        cover_edges,
        cover_edge_faces,
        _,
        cover_face_edges,
        cover_face_edge_sign,
        *_,
    ) = topology

    pi_edge = np.full(cover_edges.shape[0], -1, dtype=np.int32)
    lifted_corner_orders = ((0, 1, 2), (0, 2, 1))
    for base_face, triangle in enumerate(faces):
        for copy, logical_order in enumerate(lifted_corner_orders):
            lifted_face = 2 * base_face + copy
            for local_edge, (start_corner, end_corner) in enumerate(
                zip(logical_order, logical_order[1:] + logical_order[:1])
            ):
                base_edge = edge_ids[
                    _edge_key(
                        int(triangle[start_corner]),
                        int(triangle[end_corner]),
                    )
                ]
                projection = int(base_edge_representative[base_edge])
                cover_edge = int(cover_face_edges[lifted_face, local_edge])
                existing = int(pi_edge[cover_edge])
                if existing >= 0 and existing != projection:
                    raise RuntimeError("A cover edge projects to multiple base edges")
                pi_edge[cover_edge] = projection
    if np.any(pi_edge < 0):
        raise RuntimeError("Some cover edges have no base-edge projection")

    pi_face = np.repeat(np.arange(faces.shape[0], dtype=np.int32), 2)
    tau_face = np.arange(2 * faces.shape[0], dtype=np.int32) ^ 1
    face_sign = -np.ones(2 * faces.shape[0], dtype=np.int8)

    cover_edge_lookup = {
        (int(edge[0]), int(edge[1])): edge_id
        for edge_id, edge in enumerate(cover_edges)
    }
    tau_edge = np.empty(cover_edges.shape[0], dtype=np.int32)
    edge_sign = np.empty(cover_edges.shape[0], dtype=np.int8)
    for edge_id, edge in enumerate(cover_edges):
        deck_start = int(tau_vertex[int(edge[0])])
        deck_end = int(tau_vertex[int(edge[1])])
        target_key = _edge_key(deck_start, deck_end)
        try:
            target_edge = cover_edge_lookup[target_key]
        except KeyError as exc:
            raise RuntimeError(f"Deck image of cover edge {edge_id} is missing") from exc
        tau_edge[edge_id] = target_edge
        edge_sign[edge_id] = 1 if (deck_start, deck_end) == target_key else -1

    P0 = _signed_permutation(tau_vertex, np.ones(tau_vertex.shape[0], dtype=np.int8))
    P1 = _signed_permutation(tau_edge, edge_sign)
    P2 = _signed_permutation(tau_face, face_sign)

    data = OrientationDoubleCoverData(
        cover_vertices=np.ascontiguousarray(cover_vertices, dtype=np.float64),
        cover_faces=np.ascontiguousarray(cover_faces, dtype=np.int32),
        cover_edges=np.ascontiguousarray(cover_edges, dtype=np.int32),
        cover_edge_faces=np.ascontiguousarray(cover_edge_faces, dtype=np.int32),
        cover_face_edges=np.ascontiguousarray(cover_face_edges, dtype=np.int32),
        cover_face_edge_sign=np.ascontiguousarray(
            cover_face_edge_sign,
            dtype=np.int32,
        ),
        seam_edge_pairs=np.ascontiguousarray(resolved_seams, dtype=np.int32),
        base_vertex_representative=base_vertex_representative,
        base_edge_representative=base_edge_representative,
        pi_vertex=pi_vertex,
        pi_edge=pi_edge,
        pi_face=pi_face,
        tau_vertex=tau_vertex,
        tau_edge=tau_edge,
        tau_face=tau_face,
        edge_sign=edge_sign,
        face_sign=face_sign,
        P0=P0,
        P1=P1,
        P2=P2,
    )
    validate_orientation_double_cover(data)
    return data


class OrientationDoubleCover(ModuleBase):
    """Rheidos module exposing a generated cover mesh, DEC, and deck maps."""

    NAME = "OrientationDoubleCover"

    def __init__(
        self,
        world: World,
        *,
        parent_mesh: SurfaceMeshModule,
        seam_tolerance: float = 1.0e-6,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        if not np.isfinite(seam_tolerance) or seam_tolerance <= 0.0:
            raise ValueError("seam_tolerance must be a positive finite number")

        self.parent_mesh = parent_mesh
        self.p_mesh = parent_mesh  # compatibility with the original app scaffold
        self.seam_tolerance = float(seam_tolerance)

        self.seam_edge_pairs = self.resource(
            "seam_edge_pairs",
            declare=True,
            spec=ResourceSpec(kind="python", allow_none=True),
            doc=(
                "Optional (n,4) explicit seam pairs. None infers coincident cut-seam "
                "edges; an empty array disables seam gluing."
            ),
        )

        array_outputs = {
            "cover_vertices": np.float64,
            "cover_faces": np.int32,
            "resolved_seam_edge_pairs": np.int32,
            "base_vertex_representative": np.int32,
            "base_edge_representative": np.int32,
            "pi_vertex": np.int32,
            "pi_edge": np.int32,
            "pi_face": np.int32,
            "tau_vertex": np.int32,
            "tau_edge": np.int32,
            "tau_face": np.int32,
            "edge_sign": np.int8,
            "face_sign": np.int8,
        }
        for name, dtype in array_outputs.items():
            setattr(
                self,
                name,
                self.resource(
                    name,
                    spec=ResourceSpec(kind="numpy", dtype=dtype),
                ),
            )
        self.P0 = self.resource("P0", spec=ResourceSpec(kind="python"))
        self.P1 = self.resource("P1", spec=ResourceSpec(kind="python"))
        self.P2 = self.resource("P2", spec=ResourceSpec(kind="python"))

        self.cover_mesh = self.require(
            SurfaceMeshModule,
            child=True,
            child_name="mesh",
            vertices=self.cover_vertices,
            faces=self.cover_faces,
        )
        self.dec = self.require(
            DEC,
            child=True,
            child_name="dec",
            mesh=self.cover_mesh,
        )
        self.bind_producers()

    def set_seam_edge_pairs(
        self,
        seam_edge_pairs: np.ndarray | Sequence[Sequence[int]] | None,
    ) -> None:
        if seam_edge_pairs is None:
            self.seam_edge_pairs.set(None)
            return
        self.seam_edge_pairs.set(np.asarray(seam_edge_pairs))

    @producer(
        inputs=("parent_mesh.V_pos", "parent_mesh.F_verts", "seam_edge_pairs"),
        outputs=(
            "cover_vertices",
            "cover_faces",
            "resolved_seam_edge_pairs",
            "base_vertex_representative",
            "base_edge_representative",
            "pi_vertex",
            "pi_edge",
            "pi_face",
            "tau_vertex",
            "tau_edge",
            "tau_face",
            "edge_sign",
            "face_sign",
            "P0",
            "P1",
            "P2",
        ),
        allow_none=("seam_edge_pairs",),
    )
    def build_cover(self, ctx: ProducerContext) -> None:
        data = build_orientation_double_cover(
            ctx.inputs.parent_mesh.V_pos.get(),
            ctx.inputs.parent_mesh.F_verts.get(),
            seam_edge_pairs=ctx.inputs.seam_edge_pairs.get(),
            seam_tolerance=self.seam_tolerance,
        )
        ctx.commit(
            cover_vertices=data.cover_vertices,
            cover_faces=data.cover_faces,
            resolved_seam_edge_pairs=data.seam_edge_pairs,
            base_vertex_representative=data.base_vertex_representative,
            base_edge_representative=data.base_edge_representative,
            pi_vertex=data.pi_vertex,
            pi_edge=data.pi_edge,
            pi_face=data.pi_face,
            tau_vertex=data.tau_vertex,
            tau_edge=data.tau_edge,
            tau_face=data.tau_face,
            edge_sign=data.edge_sign,
            face_sign=data.face_sign,
            P0=data.P0,
            P1=data.P1,
            P2=data.P2,
        )

    def ensure(self) -> SurfaceMeshModule:
        """Build the cover and return the ordinary generated surface mesh."""

        self.cover_faces.get()
        return self.cover_mesh

    @staticmethod
    def _validate_cochain(values: np.ndarray, size: int, operator: str) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 1 or array.shape[0] != size:
            raise ValueError(f"{operator} expects a cochain with shape ({size},)")
        return array

    def apply_p0(self, zero_cochain: np.ndarray) -> np.ndarray:
        tau = self.tau_vertex.get()
        values = self._validate_cochain(zero_cochain, tau.shape[0], "P0")
        return values[tau]

    def apply_p1(self, one_cochain: np.ndarray) -> np.ndarray:
        tau = self.tau_edge.get()
        values = self._validate_cochain(one_cochain, tau.shape[0], "P1")
        return self.edge_sign.get() * values[tau]

    def apply_p2(self, two_cochain: np.ndarray) -> np.ndarray:
        tau = self.tau_face.get()
        values = self._validate_cochain(two_cochain, tau.shape[0], "P2")
        return self.face_sign.get() * values[tau]


def connected_component_count(mesh: SurfaceMeshModule) -> int:
    """Return the number of face-connected components of a built cover mesh."""

    face_adjacency = mesh.F_adj.get()
    visited = np.zeros(face_adjacency.shape[0], dtype=bool)
    components = 0
    for seed in range(face_adjacency.shape[0]):
        if visited[seed]:
            continue
        components += 1
        queue = deque([seed])
        visited[seed] = True
        while queue:
            face = queue.popleft()
            for neighbor in face_adjacency[face]:
                neighbor_id = int(neighbor)
                if neighbor_id >= 0 and not visited[neighbor_id]:
                    visited[neighbor_id] = True
                    queue.append(neighbor_id)
    return components


__all__ = [
    "OrientationDoubleCover",
    "OrientationDoubleCoverData",
    "build_orientation_double_cover",
    "connected_component_count",
    "infer_coincident_boundary_edge_pairs",
    "validate_orientation_double_cover",
]
