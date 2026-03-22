from __future__ import annotations

import numpy as np


def build_face_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"V_pos must have shape (nV,3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"F_verts must have shape (nF,3), got {f.shape}")

    nF = int(f.shape[0])
    f_area = np.zeros((nF,), dtype=np.float64)
    f_normal = np.zeros((nF, 3), dtype=np.float64)

    for fid, (i0, i1, i2) in enumerate(f):
        x0 = v[int(i0)]
        x1 = v[int(i1)]
        x2 = v[int(i2)]
        area_n = np.cross(x1 - x0, x2 - x0)
        twice_area = float(np.linalg.norm(area_n))
        f_area[fid] = 0.5 * twice_area

        if twice_area > 1e-20:
            f_normal[fid] = area_n / twice_area
        else:
            f_normal[fid] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return f_area, f_normal
