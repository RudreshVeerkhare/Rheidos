from __future__ import annotations

import numpy as np


def barycentric_from_point(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    eps: float = 1e-20,
) -> np.ndarray:
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


def renorm_bary(bary: np.ndarray, *, eps: float = 1e-20) -> np.ndarray:
    s = float(np.sum(bary))
    if abs(s) <= eps:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(bary, dtype=np.float64) / s


def clamp_renorm_bary(bary: np.ndarray, *, eps: float = 1e-20) -> np.ndarray:
    out = np.maximum(np.asarray(bary, dtype=np.float64), 0.0)
    s = float(np.sum(out))
    if s <= eps:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return out / s


def project_tangent(v: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) - np.asarray(n_hat, dtype=np.float64) * float(
        np.dot(v, n_hat)
    )


def barycentric_gradients(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    eps: float = 1e-20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
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


def grad_ref_to_surface(J: np.ndarray, Ginv: np.ndarray, dphi_ref: np.ndarray) -> np.ndarray:
    return (J @ Ginv @ dphi_ref.T).T
