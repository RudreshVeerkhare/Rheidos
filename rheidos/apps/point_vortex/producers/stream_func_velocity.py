from dataclasses import dataclass
from typing import Any

import taichi as ti
from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry


@dataclass
class FaceVelocityFromStreamIO:
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    psi: ResourceRef[ti.Field]  # (nV, f32)
    vel_F: ResourceRef[ti.Field] = (
        out_field()
    )  # (nF, vec3f) piecewise-constant per face


@ti.data_oriented
class FaceVelocityFromStreamProducer(WiredProducer[FaceVelocityFromStreamIO]):
    """
    Build a per-face constant tangent velocity field from a vertex stream function psi.

    Assumes psi is piecewise-linear on each triangle.
    For each face f, we compute grad(psi) (constant on the face),
    then rotate in the face plane: v = n_hat x grad(psi).

    Note: If your flow spins the "wrong way", flip the sign once globally
    (either v = n x gradpsi or v = -n x gradpsi depending on orientation conventions).
    """

    def __init__(
        self,
        V_pos: ResourceRef[ti.Field],
        F_verts: ResourceRef[ti.Field],
        psi: ResourceRef[ti.Field],
        vel_F: ResourceRef[ti.Field],
    ) -> None:
        super().__init__(FaceVelocityFromStreamIO(V_pos, F_verts, psi, vel_F))

    @ti.kernel
    def _compute_face_vel(
        self,
        V: ti.template(),  # (nV, vec3f)
        F: ti.template(),  # (nF, vec3i)
        psi: ti.template(),  # (nV, f32)
        vel: ti.template(),  # (nF, vec3f)
    ):
        eps = 1e-20

        for f in range(F.shape[0]):
            i = F[f][0]
            j = F[f][1]
            k = F[f][2]

            xi = V[i]
            xj = V[j]
            xk = V[k]

            # Oriented (non-unit) normal
            n0 = ti.math.cross(xj - xi, xk - xi)
            nn = n0.dot(n0)

            if nn < eps:
                vel[f] = ti.math.vec3(0.0, 0.0, 0.0)
            else:
                # Gradients of barycentric coords using n0 (no sqrt needed):
                # grad λ_i = (n0 × (x_k - x_j)) / |n0|^2, etc.
                gi = ti.math.cross(n0, xk - xj) / nn
                gj = ti.math.cross(n0, xi - xk) / nn
                gk = ti.math.cross(n0, xj - xi) / nn

                # grad psi is constant on the face
                grad_psi = psi[i] * gi + psi[j] * gj + psi[k] * gk

                # Rotate within the face plane: v = n_hat x grad_psi
                n_hat = n0 / ti.sqrt(nn)
                vel[f] = ti.math.cross(n_hat, grad_psi)

    def compute(self, reg: Registry) -> None:
        io = self.io
        V = io.V_pos.get()
        F = io.F_verts.get()
        psi = io.psi.get()

        if V is None or F is None or psi is None:
            raise RuntimeError(
                "FaceVelocityFromStreamProducer is missing one or more of V_pos/F_verts/psi"
            )

        nF = F.shape[0]

        vel_F = io.vel_F.peek()
        # For a vector field in Taichi, vel_F should be ti.Vector.field(3, ...)
        if vel_F is None or vel_F.shape != (nF,):
            vel_F = ti.Vector.field(3, ti.f32, shape=(nF,))
            io.vel_F.set_buffer(vel_F, bump=False)

        self._compute_face_vel(V, F, psi, vel_F)
        io.vel_F.commit()
