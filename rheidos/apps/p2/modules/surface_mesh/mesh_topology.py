from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


def _empty_i32(shape: tuple[int, ...]) -> np.ndarray:
    return np.empty(shape, dtype=np.int32)


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

    boundary_edge_count = int(np.count_nonzero(e_faces[:, 1] < 0))

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
        boundary_edge_count,
    )
