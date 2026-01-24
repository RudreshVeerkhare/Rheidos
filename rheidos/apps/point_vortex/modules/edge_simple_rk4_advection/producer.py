from dataclasses import dataclass
from typing import Any

import taichi as ti

from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry


@dataclass
class AdvectVorticesBarycentricRK4IO:
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    F_adj: ResourceRef[ti.Field]  # (nF, vec3i)
    vel_F: ResourceRef[ti.Field]  # (nF, vec3f) piecewise-constant per face

    n_vortices: ResourceRef[ti.Field]  # scalar i32
    face_ids: ResourceRef[ti.Field]  # (maxV, i32)
    bary: ResourceRef[ti.Field]  # (maxV, vec3f)

    dt: ResourceRef[Any]  # scalar f32

    face_ids_out: ResourceRef[ti.Field] = out_field()
    bary_out: ResourceRef[ti.Field] = out_field()
    pos_out: ResourceRef[ti.Field] = out_field()


@ti.data_oriented
class AdvectVorticesBarycentricRK4Producer(
    WiredProducer[AdvectVorticesBarycentricRK4IO]
):
    """
    RK4 advection of barycentric coordinates using per-face constant velocity.

    - Each RK4 stage recomputes the velocity from the current vortex state.
    - Face IDs stay fixed during the step; after the RK4 update, a mesh-walk
      relocates endpoints to valid faces.
    """

    max_hops = 10
    eps = 1e-10

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_hops = int(self.max_hops)
        self.eps = float(self.eps)

        self._face0 = None
        self._bary0 = None
        self._k1 = None
        self._k2 = None
        self._k3 = None
        self._k4 = None

    # ----------------- helpers -----------------

    @ti.func
    def _bary_grads(self, a, b, c):
        n0 = ti.math.cross(b - a, c - a)
        nn = n0.dot(n0)

        g0 = ti.math.vec3(0.0, 0.0, 0.0)
        g1 = ti.math.vec3(0.0, 0.0, 0.0)
        g2 = ti.math.vec3(0.0, 0.0, 0.0)

        if nn >= 1e-20:
            g0 = ti.math.cross(n0, c - b) / nn
            g1 = ti.math.cross(n0, a - c) / nn
            g2 = ti.math.cross(n0, b - a) / nn

        return g0, g1, g2, nn

    @ti.func
    def _barycentric(self, p, a, b, c):
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        d20 = v2.dot(v0)
        d21 = v2.dot(v1)
        denom = d00 * d11 - d01 * d01

        u = ti.cast(1.0, ti.f32)
        v = ti.cast(0.0, ti.f32)
        w = ti.cast(0.0, ti.f32)

        if ti.abs(denom) >= 1e-20:
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = ti.cast(1.0, ti.f32) - v - w

        return ti.math.vec3(u, v, w)

    @ti.func
    def _rotate_about_axis(self, p, origin, axis, angle):
        v = p - origin
        ca = ti.cos(angle)
        sa = ti.sin(angle)
        return (
            origin
            + v * ca
            + ti.math.cross(axis, v) * sa
            + axis * (axis.dot(v)) * (1.0 - ca)
        )

    @ti.func
    def _clamp_renorm_bary(self, b):
        bc = ti.math.vec3(
            ti.max(0.0, b[0]),
            ti.max(0.0, b[1]),
            ti.max(0.0, b[2]),
        )
        s = bc[0] + bc[1] + bc[2]
        out = ti.math.vec3(1.0, 0.0, 0.0)
        if s > 1e-20:
            out = bc / s
        return out

    @ti.func
    def _renorm_bary(self, b):
        s = b[0] + b[1] + b[2]
        if ti.abs(s) > 1e-20:
            b = b / s
        return b

    # ----------------- kernels -----------------

    @ti.kernel
    def _backup_bary(
        self,
        bary: ti.template(),
        bary0: ti.template(),
        n: ti.i32,
    ):
        for i in range(n):
            bary0[i] = bary[i]

    @ti.kernel
    def _backup_face_ids(
        self,
        face_ids: ti.template(),
        face0: ti.template(),
        n: ti.i32,
    ):
        for i in range(n):
            face0[i] = face_ids[i]

    @ti.kernel
    def _restore_face_ids(
        self,
        face_ids: ti.template(),
        face0: ti.template(),
        n: ti.i32,
    ):
        for i in range(n):
            face_ids[i] = face0[i]

    @ti.kernel
    def _set_bary_from_base(
        self,
        bary: ti.template(),
        bary0: ti.template(),
        k: ti.template(),
        scale: ti.f32,
        n: ti.i32,
    ):
        for i in range(n):
            bary[i] = bary0[i] + k[i] * scale

    @ti.kernel
    def _apply_rk4_update(
        self,
        bary: ti.template(),
        bary0: ti.template(),
        k1: ti.template(),
        k2: ti.template(),
        k3: ti.template(),
        k4: ti.template(),
        dt: ti.f32,
        n: ti.i32,
    ):
        for i in range(n):
            b = bary0[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * (dt / 6.0)
            bary[i] = self._renorm_bary(b)

    @ti.kernel
    def _compute_dlambda(
        self,
        V_pos: ti.template(),
        F_verts: ti.template(),
        vel_F: ti.template(),
        face_ids: ti.template(),
        k_out: ti.template(),
        n: ti.i32,
        nF: ti.i32,
    ):
        for pid in range(n):
            f = face_ids[pid]
            if f < 0 or f >= nF:
                k_out[pid] = ti.math.vec3(0.0, 0.0, 0.0)
                continue

            fv = F_verts[f]
            a = V_pos[fv[0]]
            b = V_pos[fv[1]]
            c = V_pos[fv[2]]

            g0, g1, g2, nn = self._bary_grads(a, b, c)
            if nn < 1e-20:
                k_out[pid] = ti.math.vec3(0.0, 0.0, 0.0)
                continue

            u = vel_F[f]
            k_out[pid] = ti.math.vec3(g0.dot(u), g1.dot(u), g2.dot(u))

    @ti.kernel
    def _relocate_mesh_walk(
        self,
        V_pos: ti.template(),
        F_verts: ti.template(),
        F_adj: ti.template(),
        face_ids: ti.template(),
        bary: ti.template(),
        n: ti.i32,
        nF: ti.i32,
    ):
        eps = ti.cast(self.eps, ti.f32)

        for pid in range(n):
            f = face_ids[pid]

            if f < 0 or f >= nF:
                face_ids[pid] = -1
                bary[pid] = ti.math.vec3(1.0, 0.0, 0.0)
                continue

            fv = F_verts[f]
            a = V_pos[fv[0]]
            b = V_pos[fv[1]]
            c = V_pos[fv[2]]

            b0 = self._renorm_bary(bary[pid])
            p = b0[0] * a + b0[1] * b + b0[2] * c

            f_cur = f
            b_cur = b0
            done = 0

            for _ in range(self.max_hops):
                if done == 0:
                    if f_cur < 0 or f_cur >= nF:
                        f_cur = -1
                        b_cur = ti.math.vec3(1.0, 0.0, 0.0)
                        done = 1
                    else:
                        fv = F_verts[f_cur]
                        a = V_pos[fv[0]]
                        b = V_pos[fv[1]]
                        c = V_pos[fv[2]]

                        bc = self._barycentric(p, a, b, c)
                        min_b = ti.min(bc[0], ti.min(bc[1], bc[2]))

                        if min_b >= -eps:
                            b_cur = bc
                            done = 1
                        else:
                            idx = 0
                            if bc[1] < bc[idx]:
                                idx = 1
                            if bc[2] < bc[idx]:
                                idx = 2

                            nf = F_adj[f_cur][idx]
                            if nf < 0:
                                b_cur = self._clamp_renorm_bary(bc)
                                done = 1
                            else:
                                fv2 = F_verts[nf]
                                a2 = V_pos[fv2[0]]
                                b2 = V_pos[fv2[1]]
                                c2 = V_pos[fv2[2]]

                                n0 = ti.math.cross(b - a, c - a)
                                n1 = ti.math.cross(b2 - a2, c2 - a2)
                                nn0 = n0.dot(n0)
                                nn1 = n1.dot(n1)

                                if nn0 >= 1e-20 and nn1 >= 1e-20:
                                    n_old = n0 / ti.sqrt(nn0)
                                    n_new = n1 / ti.sqrt(nn1)

                                    e0 = a
                                    e1 = b
                                    if idx == 0:
                                        e0 = b
                                        e1 = c
                                    elif idx == 1:
                                        e0 = c
                                        e1 = a

                                    edge = e1 - e0
                                    edge_len2 = edge.dot(edge)
                                    if edge_len2 >= 1e-20:
                                        axis = edge / ti.sqrt(edge_len2)
                                        cosang = ti.max(
                                            -1.0, ti.min(1.0, n_old.dot(n_new))
                                        )
                                        angle = ti.acos(cosang)
                                        if axis.dot(ti.math.cross(n_old, n_new)) < 0.0:
                                            angle = -angle
                                        p = self._rotate_about_axis(p, e0, axis, angle)

                                f_cur = nf

            if done == 0 and f_cur >= 0 and f_cur < nF:
                fv = F_verts[f_cur]
                a = V_pos[fv[0]]
                b = V_pos[fv[1]]
                c = V_pos[fv[2]]
                b_cur = self._clamp_renorm_bary(self._barycentric(p, a, b, c))

            if f_cur < 0:
                face_ids[pid] = -1
                bary[pid] = ti.math.vec3(1.0, 0.0, 0.0)
            else:
                face_ids[pid] = f_cur
                bary[pid] = self._clamp_renorm_bary(b_cur)

    @ti.kernel
    def _copy_state(
        self,
        face_ids: ti.template(),
        bary: ti.template(),
        face_ids_out: ti.template(),
        bary_out: ti.template(),
        n: ti.i32,
    ):
        for i in range(n):
            face_ids_out[i] = face_ids[i]
            bary_out[i] = bary[i]

    @ti.kernel
    def _write_pos_out(
        self,
        V_pos: ti.template(),
        F_verts: ti.template(),
        face_ids: ti.template(),
        bary: ti.template(),
        pos_out: ti.template(),
        n: ti.i32,
        nF: ti.i32,
    ):
        for pid in range(n):
            f = face_ids[pid]
            if f < 0 or f >= nF:
                pos_out[pid] = ti.math.vec3(0.0, 0.0, 0.0)
                continue

            fv = F_verts[f]
            a = V_pos[fv[0]]
            b = V_pos[fv[1]]
            c = V_pos[fv[2]]
            bc = bary[pid]
            pos_out[pid] = bc[0] * a + bc[1] * b + bc[2] * c

    # ----------------- orchestration -----------------

    def _ensure_tmp_fields(self, shape):
        if self._face0 is None or self._face0.shape != shape:
            self._face0 = ti.field(dtype=ti.i32, shape=shape)
        if self._bary0 is None or self._bary0.shape != shape:
            self._bary0 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        if self._k1 is None or self._k1.shape != shape:
            self._k1 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        if self._k2 is None or self._k2.shape != shape:
            self._k2 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        if self._k3 is None or self._k3.shape != shape:
            self._k3 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        if self._k4 is None or self._k4.shape != shape:
            self._k4 = ti.Vector.field(3, dtype=ti.f32, shape=shape)

    def _ensure_output_fields(self, face_ids, bary):
        io = self.io
        face_ids_out = io.face_ids_out.peek()
        if face_ids_out is None or face_ids_out.shape != face_ids.shape:
            face_ids_out = ti.field(dtype=ti.i32, shape=face_ids.shape)
            io.face_ids_out.set_buffer(face_ids_out, bump=False)

        bary_out = io.bary_out.peek()
        if bary_out is None or bary_out.shape != bary.shape:
            bary_out = ti.Vector.field(3, dtype=ti.f32, shape=bary.shape)
            io.bary_out.set_buffer(bary_out, bump=False)

        pos_out = io.pos_out.peek()
        if pos_out is None or pos_out.shape != face_ids.shape:
            pos_out = ti.Vector.field(3, dtype=ti.f32, shape=face_ids.shape)
            io.pos_out.set_buffer(pos_out, bump=False)

        return face_ids_out, bary_out, pos_out

    def _touch_vortex_state(self) -> None:
        self.io.face_ids.bump()
        self.io.bary.bump()

    def compute(self, reg: Registry) -> None:
        io = self.io
        inputs = self.require_inputs()
        V_pos = inputs["V_pos"].get()
        F_verts = inputs["F_verts"].get()
        F_adj = inputs["F_adj"].get()
        vel_F = inputs["vel_F"].get()
        n_vortices = inputs["n_vortices"].get()
        face_ids = inputs["face_ids"].get()
        bary = inputs["bary"].get()
        dt_value = inputs["dt"].get()

        if hasattr(n_vortices, "__getitem__"):
            n = int(n_vortices[None])
        else:
            n = int(n_vortices)

        if hasattr(dt_value, "__getitem__"):
            dt = float(dt_value[None])
        else:
            dt = float(dt_value)

        outputs = self.ensure_outputs(reg)
        face_ids_out = outputs["face_ids_out"].peek()
        bary_out = outputs["bary_out"].peek()
        pos_out = outputs["pos_out"].peek()

        if n <= 0:
            io.face_ids_out.commit()
            io.bary_out.commit()
            io.pos_out.commit()
            return

        capacity = int(face_ids.shape[0])
        if n > capacity:
            n = capacity

        nF = int(F_verts.shape[0])

        self._ensure_tmp_fields(face_ids.shape)

        self._backup_bary(bary, self._bary0, n)
        self._backup_face_ids(face_ids, self._face0, n)

        self._compute_dlambda(V_pos, F_verts, vel_F, face_ids, self._k1, n, nF)

        self._set_bary_from_base(bary, self._bary0, self._k1, 0.5 * dt, n)
        self._restore_face_ids(face_ids, self._face0, n)
        self._relocate_mesh_walk(V_pos, F_verts, F_adj, face_ids, bary, n, nF)
        self._touch_vortex_state()
        vel_F = io.vel_F.get()
        self._compute_dlambda(V_pos, F_verts, vel_F, face_ids, self._k2, n, nF)

        self._set_bary_from_base(bary, self._bary0, self._k2, 0.5 * dt, n)
        self._restore_face_ids(face_ids, self._face0, n)
        self._relocate_mesh_walk(V_pos, F_verts, F_adj, face_ids, bary, n, nF)
        self._touch_vortex_state()
        vel_F = io.vel_F.get()
        self._compute_dlambda(V_pos, F_verts, vel_F, face_ids, self._k3, n, nF)

        self._set_bary_from_base(bary, self._bary0, self._k3, dt, n)
        self._restore_face_ids(face_ids, self._face0, n)
        self._relocate_mesh_walk(V_pos, F_verts, F_adj, face_ids, bary, n, nF)
        self._touch_vortex_state()
        vel_F = io.vel_F.get()
        self._compute_dlambda(V_pos, F_verts, vel_F, face_ids, self._k4, n, nF)

        self._restore_face_ids(face_ids, self._face0, n)
        self._apply_rk4_update(
            bary, self._bary0, self._k1, self._k2, self._k3, self._k4, dt, n
        )
        self._relocate_mesh_walk(V_pos, F_verts, F_adj, face_ids, bary, n, nF)
        self._touch_vortex_state()

        self._copy_state(face_ids, bary, face_ids_out, bary_out, n)
        self._write_pos_out(V_pos, F_verts, face_ids, bary, pos_out, n, nF)

        io.face_ids_out.commit()
        io.bary_out.commit()
        io.pos_out.commit()
