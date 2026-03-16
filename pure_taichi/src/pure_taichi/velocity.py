from __future__ import annotations

import numpy as np

from .elements import CENTROID_BARY, CORNER_BARY, P2Element, bary_to_ref
from .math_utils import grad_ref_to_surface


def sample_velocity_from_corners(
    vel_corner: np.ndarray,
    face_id: int,
    bary: np.ndarray,
) -> np.ndarray:
    b = np.asarray(bary, dtype=np.float64)
    return (
        b[0] * vel_corner[face_id, 0]
        + b[1] * vel_corner[face_id, 1]
        + b[2] * vel_corner[face_id, 2]
    )


def build_p2_velocity_fields(
    psi: np.ndarray,
    face_to_dofs: np.ndarray,
    J: np.ndarray,
    Ginv: np.ndarray,
    normals: np.ndarray,
    n_vertices: int,
    *,
    element: P2Element | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    element = element or P2Element()

    psi = np.asarray(psi, dtype=np.float64)
    face_to_dofs = np.asarray(face_to_dofs, dtype=np.int32)
    J = np.asarray(J, dtype=np.float64)
    Ginv = np.asarray(Ginv, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)

    nF = int(face_to_dofs.shape[0])
    vel_corner = np.zeros((nF, 3, 3), dtype=np.float64)
    vel_face = np.zeros((nF, 3), dtype=np.float64)

    for fid in range(nF):
        dofs = face_to_dofs[fid]
        local_psi = psi[dofs]

        n_hat = normals[fid]
        nn = float(np.linalg.norm(n_hat))
        if nn <= 1e-20:
            n_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            n_hat = n_hat / nn

        Jf = J[fid]
        Ginvf = Ginv[fid]

        for corner, bc in enumerate(CORNER_BARY):
            xi, eta = bary_to_ref(bc)
            dphi_ref = element.eval_grad_ref(xi, eta)
            dphi = grad_ref_to_surface(Jf, Ginvf, dphi_ref)
            grad_psi = np.sum(local_psi[:, None] * dphi, axis=0)
            vel_corner[fid, corner] = np.cross(n_hat, grad_psi)

        xi, eta = bary_to_ref(CENTROID_BARY)
        dphi_ref = element.eval_grad_ref(xi, eta)
        dphi = grad_ref_to_surface(Jf, Ginvf, dphi_ref)
        grad_psi = np.sum(local_psi[:, None] * dphi, axis=0)
        vel_face[fid] = np.cross(n_hat, grad_psi)

    stream_vertex = psi[:n_vertices].copy()
    return vel_corner, vel_face, stream_vertex
