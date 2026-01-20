from dataclasses import dataclass
from typing import Any

import taichi as ti
from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry


@dataclass
class FaceCornerVelocityRT0FromStreamIO:
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i), assumed consistently oriented
    psi: ResourceRef[ti.Field]  # (nV, f32)

    # RT0 affine field per face, represented as velocity at the 3 face vertices.
    # Evaluate inside face with barycentric interpolation.
    vel_FV: ResourceRef[ti.Field] = out_field()  # (nF, 3) vec3f


@ti.data_oriented
class FaceCornerVelocityRT0FromStreamProducer(
    WiredProducer[FaceCornerVelocityRT0FromStreamIO]
):
    """
    Reconstruct an RT0 (H(div)) velocity field from a stream function psi (0-form on vertices).

    Output: vel_FV[f,0..2] are the velocities at the triangle's vertices (i,j,k).
    Inside the face, evaluate u(x) by barycentric interpolation of these 3 vectors.
    This exactly evaluates the affine RT0 field on that triangle.

    Notes:
    - Requires F_verts to be consistently oriented (CCW w.r.t. outward normal).
    - This yields flux-continuous fields (normal component continuous across edges),
      but tangential component may jump (RT0 is H(div), not C0).
    """

    def __init__(
        self,
        V_pos: ResourceRef[ti.Field],
        F_verts: ResourceRef[ti.Field],
        psi: ResourceRef[ti.Field],
        vel_FV: ResourceRef[ti.Field],
    ) -> None:
        super().__init__(FaceCornerVelocityRT0FromStreamIO(V_pos, F_verts, psi, vel_FV))

    @ti.func
    def _cross2(self, a: ti.math.vec2, b: ti.math.vec2) -> ti.f32:
        return a.x * b.y - a.y * b.x

    @ti.kernel
    def _compute_face_corner_vel_rt0(
        self,
        V: ti.template(),  # (nV, vec3f)
        F: ti.template(),  # (nF, vec3i)
        psi: ti.template(),  # (nV, f32)
        vel_FV: ti.template(),  # (nF, 3) vec3f
    ):
        eps = 1e-20

        for f in range(F.shape[0]):
            i = F[f][0]
            j = F[f][1]
            k = F[f][2]

            xi = V[i]
            xj = V[j]
            xk = V[k]

            e01 = xj - xi
            e02 = xk - xi
            n0 = ti.math.cross(e01, e02)
            nn = n0.dot(n0)

            if nn < eps:
                vel_FV[f, 0] = ti.math.vec3(0.0)
                vel_FV[f, 1] = ti.math.vec3(0.0)
                vel_FV[f, 2] = ti.math.vec3(0.0)
                continue

            # Face basis (t1,t2) spanning the triangle plane
            n_hat = n0 / ti.sqrt(nn)

            # t1 along e01
            l01 = ti.sqrt(e01.dot(e01)) + eps
            t1 = e01 / l01

            # t2 = n_hat x t1 (in-plane, orthonormal if t1 is unit)
            t2 = ti.math.cross(n_hat, t1)

            # Local 2D coordinates of vertices in this face basis
            v0 = ti.math.vec2(0.0, 0.0)
            v1 = ti.math.vec2(e01.dot(t1), e01.dot(t2))  # ~ (|e01|, 0)
            d2 = xk - xi
            v2 = ti.math.vec2(d2.dot(t1), d2.dot(t2))

            # Area in 2D (positive scalar)
            twiceA = ti.abs(self._cross2(v1 - v0, v2 - v0))
            if twiceA < eps:
                vel_FV[f, 0] = ti.math.vec3(0.0)
                vel_FV[f, 1] = ti.math.vec3(0.0)
                vel_FV[f, 2] = ti.math.vec3(0.0)
                continue

            inv_2A = 1.0 / twiceA  # since twiceA = 2A

            # RT0 DOFs: edge-normal flux integrals (outward) on the three edges,
            # in the face's CCW boundary direction:
            #
            # Edge i->j (opposite k): f2 = -(psi[j]-psi[i]) = psi[i]-psi[j]
            # Edge j->k (opposite i): f0 = -(psi[k]-psi[j]) = psi[j]-psi[k]
            # Edge k->i (opposite j): f1 = -(psi[i]-psi[k]) = psi[k]-psi[i]
            #
            # f0 corresponds to edge opposite v0 (between v1,v2), etc.
            f0 = psi[j] - psi[k]
            f1 = psi[k] - psi[i]
            f2 = psi[i] - psi[j]

            # RT0 affine field on triangle:
            # u(x) = sum_r f_r * (x - v_r) / (2A)
            # We'll compute u at the triangle's vertices (v0,v1,v2) in 2D.
            u0_2 = (f1 * (v0 - v1) + f2 * (v0 - v2)) * inv_2A
            u1_2 = (f0 * (v1 - v0) + f2 * (v1 - v2)) * inv_2A
            u2_2 = (f0 * (v2 - v0) + f1 * (v2 - v1)) * inv_2A

            # Map 2D (t1,t2) vector back to 3D
            u0 = u0_2.x * t1 + u0_2.y * t2
            u1 = u1_2.x * t1 + u1_2.y * t2
            u2 = u2_2.x * t1 + u2_2.y * t2

            vel_FV[f, 0] = u0
            vel_FV[f, 1] = u1
            vel_FV[f, 2] = u2

    def compute(self, reg: Registry) -> None:
        io = self.io
        V = io.V_pos.get()
        F = io.F_verts.get()
        psi = io.psi.get()

        if V is None or F is None or psi is None:
            raise RuntimeError(
                "FaceCornerVelocityRT0FromStreamProducer missing V_pos/F_verts/psi"
            )

        nF = F.shape[0]

        vel_FV = io.vel_FV.peek()
        # store as vec3 field with shape (nF, 3)
        if vel_FV is None or vel_FV.shape != (nF, 3):
            vel_FV = ti.Vector.field(3, ti.f32, shape=(nF, 3))
            io.vel_FV.set_buffer(vel_FV, bump=False)

        self._compute_face_corner_vel_rt0(V, F, psi, vel_FV)
        io.vel_FV.commit()
