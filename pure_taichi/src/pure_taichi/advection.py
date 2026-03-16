from __future__ import annotations

import numpy as np

from .math_utils import (
    barycentric_from_point,
    barycentric_gradients,
    clamp_renorm_bary,
    project_tangent,
    renorm_bary,
)
from .velocity import sample_velocity_from_corners


def _interiorize_bary(bary: np.ndarray, *, floor: float = 1e-9) -> np.ndarray:
    """Keep barycentric coords strictly interior to avoid zero-time edge bounces."""
    b = np.asarray(bary, dtype=np.float64)
    b = np.maximum(b, float(floor))
    s = float(np.sum(b))
    if s <= 1e-20:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return b / s


def advance_const_velocity_event_driven(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_adjacency: np.ndarray,
    face_id: int,
    bary: np.ndarray,
    vel_world: np.ndarray,
    dt: float,
    *,
    eps: float = 1e-10,
    max_hops: int = 32,
) -> tuple[int, np.ndarray, np.ndarray, int]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    adj = np.asarray(face_adjacency, dtype=np.int32)

    fid = int(face_id)
    if fid < 0 or fid >= f.shape[0]:
        raise RuntimeError(f"Invalid starting face id {fid}")

    b = _interiorize_bary(renorm_bary(np.asarray(bary, dtype=np.float64)))
    u = np.asarray(vel_world, dtype=np.float64)

    remaining = float(dt)
    hops = 0
    min_cross = max(float(eps) * 100.0, 1e-6)

    while remaining > eps:
        tri = f[fid]
        a, bb, c = v[tri[0]], v[tri[1]], v[tri[2]]

        g0, g1, g2, n_hat, area = barycentric_gradients(a, bb, c)
        if area <= 1e-20:
            raise RuntimeError(f"Degenerate face encountered during advection (face={fid})")

        u_tan = project_tangent(u, n_hat)
        db = np.array(
            [np.dot(g0, u_tan), np.dot(g1, u_tan), np.dot(g2, u_tan)],
            dtype=np.float64,
        )

        t_hit = remaining
        hit_idx = -1
        for i in range(3):
            if db[i] < -eps:
                cand = -b[i] / db[i]
                if cand > min_cross and cand < t_hit:
                    t_hit = float(cand)
                    hit_idx = i

        if hit_idx < 0:
            b = b + remaining * db
            remaining = 0.0
            break

        b = b + t_hit * db
        p = b[0] * a + b[1] * bb + b[2] * c

        remaining -= t_hit
        if remaining <= eps:
            break

        nbr = int(adj[fid, hit_idx])
        if nbr < 0:
            raise RuntimeError(
                "Boundary edge crossing detected in midpoint advection; closed mesh required"
            )

        fid = nbr
        tri_n = f[fid]
        a_n, b_n, c_n = v[tri_n[0]], v[tri_n[1]], v[tri_n[2]]
        b = _interiorize_bary(barycentric_from_point(p, a_n, b_n, c_n))

        hops += 1
        if hops > max_hops:
            # Degrade gracefully instead of hard-failing on near-vertex chatter.
            b = clamp_renorm_bary(b)
            tri = f[fid]
            a, bb, c = v[tri[0]], v[tri[1]], v[tri[2]]
            p_out = b[0] * a + b[1] * bb + b[2] * c
            return fid, b, p_out, hops

    b = clamp_renorm_bary(b)
    tri = f[fid]
    a, bb, c = v[tri[0]], v[tri[1]], v[tri[2]]
    p_out = b[0] * a + b[1] * bb + b[2] * c
    return fid, b, p_out, hops


def advect_midpoint_batch(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_adjacency: np.ndarray,
    vel_corner: np.ndarray,
    face_ids: np.ndarray,
    bary: np.ndarray,
    dt: float,
    *,
    max_hops: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    face_ids = np.asarray(face_ids, dtype=np.int32)
    bary = np.asarray(bary, dtype=np.float64)
    n = int(face_ids.shape[0])

    face_out = np.empty((n,), dtype=np.int32)
    bary_out = np.empty((n, 3), dtype=np.float64)
    pos_out = np.empty((n, 3), dtype=np.float64)

    hops_total = 0
    hops_max = 0

    for i in range(n):
        fid0 = int(face_ids[i])
        b0 = bary[i]

        u0 = sample_velocity_from_corners(vel_corner, fid0, b0)
        fid_mid, b_mid, _, hops_a = advance_const_velocity_event_driven(
            vertices,
            faces,
            face_adjacency,
            fid0,
            b0,
            u0,
            0.5 * dt,
            max_hops=max_hops,
        )

        umid = sample_velocity_from_corners(vel_corner, fid_mid, b_mid)
        fid1, b1, p1, hops_b = advance_const_velocity_event_driven(
            vertices,
            faces,
            face_adjacency,
            fid0,
            b0,
            umid,
            dt,
            max_hops=max_hops,
        )

        face_out[i] = fid1
        bary_out[i] = b1
        pos_out[i] = p1

        hops = int(hops_a + hops_b)
        hops_total += hops
        hops_max = max(hops_max, hops)

    return face_out, bary_out, pos_out, int(hops_total), int(hops_max)
