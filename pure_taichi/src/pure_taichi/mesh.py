from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .model import FaceGeometryData, MeshData


def _normalize_rows(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    n = np.linalg.norm(out, axis=1)
    n = np.maximum(n, 1e-20)
    out /= n[:, None]
    out *= float(radius)
    return out


def generate_icosphere(subdivisions: int = 2, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    verts = _normalize_rows(verts, radius)

    for _ in range(int(subdivisions)):
        midpoint_cache: dict[tuple[int, int], int] = {}
        verts_list = verts.tolist()

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            idx = midpoint_cache.get(key)
            if idx is not None:
                return idx
            m = 0.5 * (verts[i] + verts[j])
            verts_list.append(m.tolist())
            idx = len(verts_list) - 1
            midpoint_cache[key] = idx
            return idx

        new_faces: list[list[int]] = []
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend(
                [
                    [a, ab, ca],
                    [b, bc, ab],
                    [c, ca, bc],
                    [ab, bc, ca],
                ]
            )

        verts = np.asarray(verts_list, dtype=np.float64)
        verts = _normalize_rows(verts, radius)
        faces = np.asarray(new_faces, dtype=np.int32)

    return verts, faces


def build_mesh_topology_geometry(vertices: np.ndarray, faces: np.ndarray) -> MeshData:
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (nV,3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (nF,3), got {f.shape}")

    if f.size > 0:
        lo = int(f.min())
        hi = int(f.max())
        if lo < 0 or hi >= v.shape[0]:
            raise ValueError(f"faces reference range [{lo},{hi}] but nV={v.shape[0]}")

    edge_map: dict[tuple[int, int], int] = {}
    edge_vertices: list[tuple[int, int]] = []
    edge_faces: list[list[int]] = []

    face_adjacency = np.full((f.shape[0], 3), -1, dtype=np.int32)

    def add_edge(a: int, b: int, fid: int) -> int:
        key = (a, b) if a < b else (b, a)
        eid = edge_map.get(key)
        if eid is None:
            eid = len(edge_vertices)
            edge_map[key] = eid
            edge_vertices.append(key)
            edge_faces.append([fid, -1])
        else:
            if edge_faces[eid][1] != -1:
                raise ValueError(f"Non-manifold edge detected at {key}")
            edge_faces[eid][1] = fid
        return eid

    for fid, (a, b, c) in enumerate(f):
        add_edge(int(a), int(b), fid)
        add_edge(int(b), int(c), fid)
        add_edge(int(c), int(a), fid)

    edges = np.asarray(edge_vertices, dtype=np.int32)
    edge_faces_np = np.asarray(edge_faces, dtype=np.int32)

    for fid, (a, b, c) in enumerate(f):
        opp_edges = ((int(b), int(c)), (int(c), int(a)), (int(a), int(b)))
        for m, (u, w) in enumerate(opp_edges):
            key = (u, w) if u < w else (w, u)
            eid = edge_map[key]
            f0, f1 = int(edge_faces_np[eid, 0]), int(edge_faces_np[eid, 1])
            face_adjacency[fid, m] = f1 if f0 == fid else f0

    boundary_edges = int(np.count_nonzero(edge_faces_np[:, 1] < 0))
    if boundary_edges > 0:
        raise ValueError(
            f"Mesh must be closed; found {boundary_edges} boundary edges"
        )

    face_normals = np.zeros((f.shape[0], 3), dtype=np.float64)
    face_areas = np.zeros((f.shape[0],), dtype=np.float64)

    for fid, (i0, i1, i2) in enumerate(f):
        x0 = v[int(i0)]
        x1 = v[int(i1)]
        x2 = v[int(i2)]
        cr = np.cross(x1 - x0, x2 - x0)
        nrm = float(np.linalg.norm(cr))
        area = 0.5 * nrm
        if area <= 1e-20:
            raise ValueError(f"Degenerate face at index {fid}")
        face_areas[fid] = area
        face_normals[fid] = cr / nrm

    return MeshData(
        vertices=v,
        faces=f,
        edges=edges,
        edge_faces=edge_faces_np,
        face_adjacency=face_adjacency,
        face_normals=face_normals,
        face_areas=face_areas,
    )


def build_face_geometry(vertices: np.ndarray, faces: np.ndarray) -> FaceGeometryData:
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    nF = int(f.shape[0])
    J = np.zeros((nF, 3, 2), dtype=np.float64)
    Ginv = np.zeros((nF, 2, 2), dtype=np.float64)
    sqrt_detG = np.zeros((nF,), dtype=np.float64)

    for fid, (i0, i1, i2) in enumerate(f):
        x0 = v[int(i0)]
        x1 = v[int(i1)]
        x2 = v[int(i2)]

        e1 = x1 - x0
        e2 = x2 - x0
        Jf = np.column_stack((e1, e2))
        G = Jf.T @ Jf
        detG = float(np.linalg.det(G))
        if detG <= 1e-24:
            raise ValueError(f"Degenerate face geometry at face {fid}")

        J[fid] = Jf
        Ginv[fid] = np.linalg.inv(G)
        sqrt_detG[fid] = np.sqrt(detG)

    return FaceGeometryData(J=J, Ginv=Ginv, sqrt_detG=sqrt_detG)


def build_default_mesh(kind: str = "icosphere", subdivisions: int = 2, radius: float = 1.0) -> MeshData:
    if kind != "icosphere":
        raise ValueError(f"Unsupported mesh kind '{kind}'. Supported: icosphere")

    vertices, faces = generate_icosphere(subdivisions=int(subdivisions), radius=float(radius))
    return build_mesh_topology_geometry(vertices, faces)
