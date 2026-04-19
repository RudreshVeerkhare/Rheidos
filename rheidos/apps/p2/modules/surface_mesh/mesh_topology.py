from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


def _empty_i32(shape: tuple[int, ...]) -> np.ndarray:
    return np.empty(shape, dtype=np.int32)


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _walk_boundary_chain(
    start_vertex: int,
    vertex_neighbors: Dict[int, Set[int]],
    expected_edge_count: int,
) -> List[int]:
    ordered_vertices = [start_vertex]
    prev_vertex: int | None = None
    current_vertex = start_vertex

    while True:
        next_vertices = sorted(
            nbr for nbr in vertex_neighbors[current_vertex] if nbr != prev_vertex
        )
        if len(next_vertices) > 1:
            raise ValueError(
                f"Branched boundary detected at vertex {current_vertex}: "
                f"{next_vertices}"
            )
        if not next_vertices:
            break

        next_vertex = next_vertices[0]
        ordered_vertices.append(next_vertex)
        prev_vertex, current_vertex = current_vertex, next_vertex

        if len(ordered_vertices) > expected_edge_count + 1:
            raise ValueError("Invalid open boundary component traversal")

    return ordered_vertices


def _walk_boundary_loop(
    start_vertex: int,
    next_vertex: int,
    vertex_neighbors: Dict[int, Set[int]],
    expected_vertex_count: int,
) -> List[int]:
    ordered_vertices = [start_vertex]
    prev_vertex: int | None = None
    current_vertex = start_vertex

    while True:
        if prev_vertex is None:
            candidate_vertices = sorted(vertex_neighbors[current_vertex])
            if next_vertex not in candidate_vertices:
                raise ValueError(
                    f"Boundary loop traversal from {start_vertex} to {next_vertex} "
                    "is not valid"
                )
            candidate_vertex = next_vertex
        else:
            candidate_vertices = sorted(
                nbr for nbr in vertex_neighbors[current_vertex] if nbr != prev_vertex
            )
            if len(candidate_vertices) != 1:
                raise ValueError(
                    f"Invalid closed boundary component at vertex {current_vertex}: "
                    f"{candidate_vertices}"
                )
            candidate_vertex = candidate_vertices[0]

        if candidate_vertex == start_vertex:
            break

        ordered_vertices.append(candidate_vertex)
        prev_vertex, current_vertex = current_vertex, candidate_vertex

        if len(ordered_vertices) > expected_vertex_count:
            raise ValueError("Invalid closed boundary component traversal")

    return ordered_vertices


def _boundary_edges_from_vertices(
    ordered_vertices: List[int],
    *,
    closed: bool,
    edge_lookup: Dict[Tuple[int, int], int],
) -> List[int]:
    if not ordered_vertices:
        return []

    edge_pairs = list(zip(ordered_vertices[:-1], ordered_vertices[1:]))
    if closed:
        edge_pairs.append((ordered_vertices[-1], ordered_vertices[0]))

    return [edge_lookup[_edge_key(u, v)] for u, v in edge_pairs]


def _build_boundary_components(
    e_verts: np.ndarray,
    boundary_edge_ids: np.ndarray,
) -> tuple[np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray]]:
    boundary_edge_ids = np.sort(np.asarray(boundary_edge_ids, dtype=np.int32))
    if boundary_edge_ids.size == 0:
        return boundary_edge_ids, [], _empty_i32((0,)), []

    boundary_edges = e_verts[boundary_edge_ids]
    boundary_vertex_ids = np.unique(boundary_edges.reshape(-1)).astype(np.int32, copy=False)

    vertex_neighbors: Dict[int, Set[int]] = {int(vid): set() for vid in boundary_vertex_ids}
    incident_edges: Dict[int, List[int]] = {int(vid): [] for vid in boundary_vertex_ids}
    edge_lookup: Dict[Tuple[int, int], int] = {}

    for eid in boundary_edge_ids:
        edge_id = int(eid)
        u, v = (int(e_verts[edge_id, 0]), int(e_verts[edge_id, 1]))
        edge_lookup[_edge_key(u, v)] = edge_id

        vertex_neighbors[u].add(v)
        vertex_neighbors[v].add(u)
        incident_edges[u].append(edge_id)
        incident_edges[v].append(edge_id)

    for vid in boundary_vertex_ids:
        vertex_id = int(vid)
        degree = len(incident_edges[vertex_id])
        if degree > 2:
            raise ValueError(f"Branched boundary detected at vertex {vertex_id}")

    unvisited_vertices = {int(vid) for vid in boundary_vertex_ids.tolist()}
    boundary_edge_components: List[np.ndarray] = []
    boundary_vertex_components: List[np.ndarray] = []

    while unvisited_vertices:
        seed_vertex = min(unvisited_vertices)
        stack = [seed_vertex]
        component_vertices: List[int] = []
        component_edge_ids: Set[int] = set()

        while stack:
            vertex_id = stack.pop()
            if vertex_id not in unvisited_vertices:
                continue

            unvisited_vertices.remove(vertex_id)
            component_vertices.append(vertex_id)
            component_edge_ids.update(incident_edges[vertex_id])

            for neighbor in sorted(vertex_neighbors[vertex_id], reverse=True):
                if neighbor in unvisited_vertices:
                    stack.append(neighbor)

        if not component_vertices:
            continue

        component_vertices.sort()
        component_edge_count = len(component_edge_ids)
        degree_one_vertices = sorted(
            vid for vid in component_vertices if len(incident_edges[vid]) == 1
        )
        degree_two_vertices = [
            vid for vid in component_vertices if len(incident_edges[vid]) == 2
        ]

        if degree_one_vertices:
            if len(degree_one_vertices) != 2:
                raise ValueError(
                    "Invalid open boundary component: expected exactly two endpoints"
                )
            ordered_vertex_component = _walk_boundary_chain(
                degree_one_vertices[0],
                vertex_neighbors,
                component_edge_count,
            )
            is_closed = False
        else:
            if len(degree_two_vertices) != len(component_vertices):
                raise ValueError(
                    "Invalid closed boundary component: vertices must have degree two"
                )

            if (
                len(component_vertices) == 1
                and component_edge_count == 1
                and int(e_verts[next(iter(component_edge_ids)), 0]) == component_vertices[0]
                and int(e_verts[next(iter(component_edge_ids)), 1]) == component_vertices[0]
            ):
                ordered_vertex_component = [component_vertices[0]]
            else:
                start_vertex = component_vertices[0]
                neighbors = sorted(vertex_neighbors[start_vertex])
                if len(neighbors) != 2:
                    raise ValueError(
                        f"Invalid closed boundary component at vertex {start_vertex}"
                    )

                forward = _walk_boundary_loop(
                    start_vertex,
                    neighbors[0],
                    vertex_neighbors,
                    len(component_vertices),
                )
                reverse = _walk_boundary_loop(
                    start_vertex,
                    neighbors[1],
                    vertex_neighbors,
                    len(component_vertices),
                )
                ordered_vertex_component = list(min(tuple(forward), tuple(reverse)))
            is_closed = True

        if set(ordered_vertex_component) != set(component_vertices):
            raise ValueError("Boundary traversal does not cover the full component")

        ordered_edge_component = _boundary_edges_from_vertices(
            ordered_vertex_component,
            closed=is_closed,
            edge_lookup=edge_lookup,
        )
        if set(ordered_edge_component) != component_edge_ids:
            raise ValueError("Boundary edge traversal does not cover the full component")

        boundary_edge_components.append(
            np.asarray(ordered_edge_component, dtype=np.int32)
        )
        boundary_vertex_components.append(
            np.asarray(ordered_vertex_component, dtype=np.int32)
        )

    component_order = sorted(
        range(len(boundary_vertex_components)),
        key=lambda idx: tuple(boundary_vertex_components[idx].tolist()),
    )
    boundary_edge_components = [boundary_edge_components[idx] for idx in component_order]
    boundary_vertex_components = [
        boundary_vertex_components[idx] for idx in component_order
    ]

    return (
        boundary_edge_ids,
        boundary_edge_components,
        boundary_vertex_ids,
        boundary_vertex_components,
    )


def build_mesh_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[int, List[int]],
    np.ndarray,
    List[np.ndarray],
    np.ndarray,
    List[np.ndarray],
    int,
]:
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"V_pos must have shape (nV,3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"F_verts must have shape (nF,3), got {f.shape}")

    nV = int(v.shape[0])
    nF = int(f.shape[0])

    if f.size > 0:
        mn = int(f.min())
        mx = int(f.max())
        if mn < 0:
            raise ValueError("F_verts contains negative vertex indices")
        if mx >= nV:
            raise ValueError(f"F_verts references vertex id {mx} but V_pos has nV={nV}")

    edge_map: Dict[Tuple[int, int], int] = {}
    e_verts_list: List[Tuple[int, int]] = []
    e_faces_list: List[List[int]] = []
    e_opp_list: List[List[int]] = []

    def add_halfedge(a: int, b: int, fid: int, opp_vert_id: int) -> None:
        i, j = (a, b) if a < b else (b, a)
        key = (i, j)
        if key not in edge_map:
            eid = len(e_verts_list)
            edge_map[key] = eid
            e_verts_list.append((i, j))
            e_faces_list.append([fid, -1])
            e_opp_list.append([opp_vert_id, -1])
            return

        eid = edge_map[key]
        if e_faces_list[eid][1] != -1:
            raise ValueError(f"Non-manifold edge detected at {key}")
        e_faces_list[eid][1] = fid
        e_opp_list[eid][1] = opp_vert_id

    for fid in range(nF):
        a, b, c = (int(f[fid, 0]), int(f[fid, 1]), int(f[fid, 2]))
        add_halfedge(a, b, fid, c)
        add_halfedge(b, c, fid, a)
        add_halfedge(c, a, fid, b)

    n_edges = len(e_verts_list)
    e_verts = (
        np.asarray(e_verts_list, dtype=np.int32)
        if n_edges > 0
        else _empty_i32((0, 2))
    )
    e_faces = (
        np.asarray(e_faces_list, dtype=np.int32)
        if n_edges > 0
        else _empty_i32((0, 2))
    )
    e_opp = (
        np.asarray(e_opp_list, dtype=np.int32)
        if n_edges > 0
        else _empty_i32((0, 2))
    )

    f_edges = _empty_i32((nF, 3))
    f_edge_sign = _empty_i32((nF, 3))

    def eid_and_sign(u: int, w: int) -> Tuple[int, int]:
        i, j = (u, w) if u < w else (w, u)
        eid = edge_map[(i, j)]
        sign = 1 if (u == i and w == j) else -1
        return eid, sign

    for fid in range(nF):
        a, b, c = (int(f[fid, 0]), int(f[fid, 1]), int(f[fid, 2]))
        e0, s0 = eid_and_sign(a, b)
        e1, s1 = eid_and_sign(b, c)
        e2, s2 = eid_and_sign(c, a)
        f_edges[fid, 0], f_edge_sign[fid, 0] = e0, s0
        f_edges[fid, 1], f_edge_sign[fid, 1] = e1, s1
        f_edges[fid, 2], f_edge_sign[fid, 2] = e2, s2

    f_adj = np.full((nF, 3), -1, dtype=np.int32)

    def other_face(eid: int, fid: int) -> int:
        f0, f1 = int(e_faces[eid, 0]), int(e_faces[eid, 1])
        if f0 == fid:
            return f1
        if f1 == fid:
            return f0
        raise RuntimeError(
            f"Internal adjacency error for face {fid}, edge id {eid}: {(f0, f1)}"
        )

    for fid in range(nF):
        a, b, c = (int(f[fid, 0]), int(f[fid, 1]), int(f[fid, 2]))

        i, j = (b, c) if b < c else (c, b)
        f_adj[fid, 0] = other_face(edge_map[(i, j)], fid)

        i, j = (c, a) if c < a else (a, c)
        f_adj[fid, 1] = other_face(edge_map[(i, j)], fid)

        i, j = (a, b) if a < b else (b, a)
        f_adj[fid, 2] = other_face(edge_map[(i, j)], fid)

    flat_faces = f.reshape(-1) if nF > 0 else _empty_i32((0,))
    v_incident_count = np.bincount(flat_faces, minlength=nV).astype(np.int32, copy=False)

    v_incident: Dict[int, List[int]] = {vid: [] for vid in range(nV)}
    seen: Dict[int, Set[int]] = {}
    for fid in range(nF):
        a, b, c = (int(f[fid, 0]), int(f[fid, 1]), int(f[fid, 2]))
        for vid in (a, b, c):
            faces_seen = seen.get(vid)
            if faces_seen is None:
                seen[vid] = {fid}
                v_incident[vid].append(fid)
            elif fid not in faces_seen:
                faces_seen.add(fid)
                v_incident[vid].append(fid)

    boundary_edge_ids, boundary_edge_components, boundary_vertex_ids, boundary_vertex_components = (
        _build_boundary_components(
            e_verts,
            np.flatnonzero(e_faces[:, 1] < 0).astype(np.int32, copy=False),
        )
    )
    boundary_edge_count = int(boundary_edge_ids.shape[0])

    return (
        n_edges,
        e_verts,
        e_faces,
        e_opp,
        f_edges,
        f_edge_sign,
        f_adj,
        v_incident_count,
        v_incident,
        boundary_edge_ids,
        boundary_edge_components,
        boundary_vertex_ids,
        boundary_vertex_components,
        boundary_edge_count,
    )
