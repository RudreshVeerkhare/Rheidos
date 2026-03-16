"""Shared finite-element and geometry utilities for point_vortex_p2."""

from __future__ import annotations

import numpy as np

REF_Q_PTS = np.array(
    [
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=np.float64,
)

# Sum(weights) = area(reference triangle) = 1/2
REF_Q_WTS = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)


CORNER_BARY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

CENTROID_BARY = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)


def p2_shape_and_grad_ref(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return P2 scalar basis values and ref gradients on reference triangle."""
    l0 = 1.0 - xi - eta
    l1 = xi
    l2 = eta

    dl0 = np.array([-1.0, -1.0], dtype=np.float64)
    dl1 = np.array([1.0, 0.0], dtype=np.float64)
    dl2 = np.array([0.0, 1.0], dtype=np.float64)

    phi = np.array(
        [
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
        ],
        dtype=np.float64,
    )

    dphi_ref = np.vstack(
        [
            (4.0 * l0 - 1.0) * dl0,
            (4.0 * l1 - 1.0) * dl1,
            (4.0 * l2 - 1.0) * dl2,
            4.0 * (l0 * dl1 + l1 * dl0),
            4.0 * (l1 * dl2 + l2 * dl1),
            4.0 * (l2 * dl0 + l0 * dl2),
        ]
    )
    return phi, dphi_ref


def bary_to_ref(bary: np.ndarray) -> tuple[float, float]:
    """Convert triangle barycentric coordinates (l0,l1,l2) to ref (xi,eta)."""
    return float(bary[1]), float(bary[2])


def barycentric_from_point(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    eps: float = 1e-20,
) -> np.ndarray:
    """Compute barycentric coordinates of point p wrt triangle (a,b,c)."""
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))

    denom = d00 * d11 - d01 * d01
    if abs(denom) < eps:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float64)


def clamp_renorm_bary(bary: np.ndarray, *, eps: float = 1e-20) -> np.ndarray:
    out = np.maximum(bary, 0.0)
    s = float(out.sum())
    if s <= eps:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return out / s


def renorm_bary(bary: np.ndarray, *, eps: float = 1e-20) -> np.ndarray:
    s = float(bary.sum())
    if abs(s) <= eps:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return bary / s


def face_normal_and_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float]:
    cr = np.cross(b - a, c - a)
    nrm = float(np.linalg.norm(cr))
    if nrm < 1e-20:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0
    return cr / nrm, 0.5 * nrm


def barycentric_gradients(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    eps: float = 1e-20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return gradients of barycentric coords and face normal/area."""
    n0 = np.cross(b - a, c - a)
    nn = float(np.dot(n0, n0))
    if nn < eps:
        z = np.zeros(3, dtype=np.float64)
        return z, z, z, np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0

    n_hat = n0 / np.sqrt(nn)
    area = 0.5 * np.sqrt(nn)
    g0 = np.cross(n0, c - b) / nn
    g1 = np.cross(n0, a - c) / nn
    g2 = np.cross(n0, b - a) / nn
    return g0, g1, g2, n_hat, area


def project_tangent(v: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    return v - n_hat * float(np.dot(v, n_hat))
