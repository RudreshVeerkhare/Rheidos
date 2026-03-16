from __future__ import annotations

import numpy as np

from .model import P2SpaceData


def build_p2_space_data(n_vertices: int, faces: np.ndarray) -> P2SpaceData:
    f = np.ascontiguousarray(faces, dtype=np.int32)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (nF,3), got {f.shape}")

    edge_map: dict[tuple[int, int], int] = {}
    face_to_edges = np.empty((f.shape[0], 3), dtype=np.int32)

    for fid, (a, b, c) in enumerate(f):
        local = ((int(a), int(b)), (int(b), int(c)), (int(c), int(a)))
        for le, (u, v) in enumerate(local):
            key = (u, v) if u < v else (v, u)
            eid = edge_map.get(key)
            if eid is None:
                eid = len(edge_map)
                edge_map[key] = eid
            face_to_edges[fid, le] = eid

    edges = np.array(list(edge_map.keys()), dtype=np.int32)
    n_edges = int(edges.shape[0])

    face_to_dofs = np.empty((f.shape[0], 6), dtype=np.int32)
    face_to_dofs[:, :3] = f
    face_to_dofs[:, 3:] = n_vertices + face_to_edges

    return P2SpaceData(
        edges=edges,
        face_to_edges=face_to_edges,
        face_to_dofs=face_to_dofs,
        ndof=int(n_vertices + n_edges),
    )
