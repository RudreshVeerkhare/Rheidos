from dataclasses import dataclass
from typing import Any

import taichi as ti

from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry


@ti.data_oriented
class SampleVortexVelProducer:
    """
    Samples a per-vertex velocity field at point-vortex positions stored as (face_id, bary).
    Output: pt_vel[pid] = sum_j bary[j] * V_vel[ F_verts[face_id][j] ].
    """

    def __init__(self) -> None:
        pass

    @ti.kernel
    def _sample_from_vertex_vel(
        self,
        F_verts: ti.template(),  # (nF,) vec3i  or (nF, 3) i32
        V_vel: ti.template(),  # (nV,) vec3f
        pt_bary: ti.template(),  # (maxP,) vec3f
        pt_face: ti.template(),  # (maxP,) i32
        pt_vel: ti.template(),  # (maxP,) vec3f  [out]
        n_pts: ti.i32,
    ):
        for pid in range(n_pts):
            f = pt_face[pid]
            b = pt_bary[pid]
            fv = F_verts[f]
            v0 = V_vel[fv[0]]
            v1 = V_vel[fv[1]]
            v2 = V_vel[fv[2]]
            pt_vel[pid] = b[0] * v0 + b[1] * v1 + b[2] * v2

    def run(
        self,
        *,
        F_verts,
        V_vel,
        pt_bary,
        pt_face,
        pt_vel_out,
        n_pts: int,
    ) -> None:
        self._sample_from_vertex_vel(
            F_verts, V_vel, pt_bary, pt_face, pt_vel_out, n_pts
        )


@ti.data_oriented
class AdvectConstVelEventDrivenProducer:
    """
    Event-driven constant-velocity advection of barycentric coords across a triangle mesh.
    Also contains utility kernels for RK4:
      - backup_state / restore_state
      - rk4_combine_vel
    """

    EPS = 1e-10
    MAX_HOPS = 10

    def __init__(self) -> None:
        pass

    @ti.func
    def _bary_from_pos(self, a, b, c, p):
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        d20 = v2.dot(v0)
        d21 = v2.dot(v1)
        denom = d00 * d11 - d01 * d01
        inv = 1.0 / ti.max(denom, 1e-20)

        v = (d11 * d20 - d01 * d21) * inv
        w = (d00 * d21 - d01 * d20) * inv
        u = 1.0 - v - w
        return ti.Vector([u, v, w], dt=ti.f32)

    @ti.func
    def _clamp_renorm_bary(self, b):
        for i in ti.static(range(3)):
            b[i] = ti.min(1.0, ti.max(0.0, b[i]))
        s = b[0] + b[1] + b[2]
        if s > 1e-20:
            b = b / s
        else:
            b = ti.Vector([1.0, 0.0, 0.0], dt=ti.f32)
        return b

    @ti.func
    def _bary_grads_and_nhat(self, a, b, c):
        ab = b - a
        ac = c - a
        n = ti.math.cross(ab, ac)
        nn = n.dot(n)
        inv_nn = 1.0 / ti.max(nn, 1e-20)

        g0 = ti.math.cross(n, b - c) * inv_nn
        g1 = ti.math.cross(n, c - a) * inv_nn
        g2 = ti.math.cross(n, a - b) * inv_nn

        n_hat = n * ti.rsqrt(ti.max(nn, 1e-20))
        return g0, g1, g2, n_hat

    @ti.func
    def _advance_one_constvel(
        self,
        V_pos: ti.template(),  # (nV,) vec3f
        F_verts: ti.template(),  # (nF,) vec3i
        F_adj: ti.template(),  # (nF,) vec3i
        f0: ti.i32,
        b0: ti.types.vector(3, ti.f32),
        vel_world: ti.types.vector(3, ti.f32),
        dt: ti.f32,
    ):
        f = f0
        b = b0
        t_rem = dt
        active = 1

        for _ in range(AdvectConstVelEventDrivenProducer.MAX_HOPS):
            if active == 1 and t_rem > AdvectConstVelEventDrivenProducer.EPS:
                fv = F_verts[f]
                a = V_pos[fv[0]]
                bb = V_pos[fv[1]]
                c = V_pos[fv[2]]

                g0, g1, g2, n_hat = self._bary_grads_and_nhat(a, bb, c)

                v = vel_world - n_hat * vel_world.dot(n_hat)
                db = ti.Vector([g0.dot(v), g1.dot(v), g2.dot(v)], dt=ti.f32)

                t_hit = t_rem
                hit_idx = -1
                for i in ti.static(range(3)):
                    if db[i] < -AdvectConstVelEventDrivenProducer.EPS:
                        cand = -b[i] / db[i]
                        if cand < t_hit:
                            t_hit = cand
                            hit_idx = i

                b = b + t_hit * db
                t_rem = t_rem - t_hit

                if hit_idx == -1 or t_rem <= AdvectConstVelEventDrivenProducer.EPS:
                    active = 0
                else:
                    fN = F_adj[f][hit_idx]
                    if fN < 0:
                        active = 0
                    else:
                        p = b[0] * a + b[1] * bb + b[2] * c
                        f = fN
                        fvN = F_verts[f]
                        aN = V_pos[fvN[0]]
                        bN = V_pos[fvN[1]]
                        cN = V_pos[fvN[2]]
                        b = self._bary_from_pos(aN, bN, cN, p)
                        b = self._clamp_renorm_bary(b)

        b = self._clamp_renorm_bary(b)
        return f, b

    @ti.kernel
    def backup_state(
        self,
        face_ids: ti.template(),  # (maxP,) i32
        bary: ti.template(),  # (maxP,) vec3f
        face0: ti.template(),  # (maxP,) i32 [out]
        bary0: ti.template(),  # (maxP,) vec3f [out]
        n: ti.i32,
    ):
        for i in range(n):
            face0[i] = face_ids[i]
            bary0[i] = bary[i]

    @ti.kernel
    def restore_state(
        self,
        face_ids: ti.template(),  # (maxP,) i32 [out]
        bary: ti.template(),  # (maxP,) vec3f [out]
        face0: ti.template(),  # (maxP,) i32
        bary0: ti.template(),  # (maxP,) vec3f
        n: ti.i32,
    ):
        for i in range(n):
            face_ids[i] = face0[i]
            bary[i] = bary0[i]

    @ti.kernel
    def rk4_combine_vel(
        self,
        k1: ti.template(),
        k2: ti.template(),
        k3: ti.template(),
        k4: ti.template(),
        kout: ti.template(),
        n: ti.i32,
    ):
        for i in range(n):
            kout[i] = (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * (1.0 / 6.0)

    @ti.kernel
    def advect_inplace(
        self,
        V_pos: ti.template(),
        F_verts: ti.template(),
        F_adj: ti.template(),
        face_ids: ti.template(),  # (maxP,) i32 [in/out]
        bary: ti.template(),  # (maxP,) vec3f [in/out]
        vel_pts: ti.template(),  # (maxP,) vec3f (constant over this substep)
        dt: ti.f32,
        n: ti.i32,
    ):
        for pid in range(n):
            f1, b1 = self._advance_one_constvel(
                V_pos, F_verts, F_adj, face_ids[pid], bary[pid], vel_pts[pid], dt
            )
            face_ids[pid] = f1
            bary[pid] = b1


@dataclass
class AdvectVorticesRK4IO:
    V_pos: ResourceRef[ti.Field]
    F_verts: ResourceRef[ti.Field]
    F_adj: ResourceRef[ti.Field]
    V_velocity: ResourceRef[ti.Field]

    n_vortices: ResourceRef[ti.Field]
    face_ids: ResourceRef[ti.Field]
    bary: ResourceRef[ti.Field]

    dt: ResourceRef[Any]

    face_ids_out: ResourceRef[ti.Field] = out_field()
    bary_out: ResourceRef[ti.Field] = out_field()
    pos_out: ResourceRef[ti.Field] = out_field()


@ti.data_oriented
class AdvectVorticesRK4Producer(WiredProducer[AdvectVorticesRK4IO]):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sampler = SampleVortexVelProducer()
        self.adv = AdvectConstVelEventDrivenProducer()

        self._p0_face = None
        self._p0_bary = None
        self._k1 = None
        self._k2 = None
        self._k3 = None
        self._k4 = None
        self._k = None

    def _touch_vortex_state(self) -> None:
        self.io.face_ids.bump()
        self.io.bary.bump()

    def _ensure_tmp_fields(self, shape):
        if self._p0_face is None or self._p0_face.shape != shape:
            self._p0_face = ti.field(dtype=ti.i32, shape=shape)
        if self._p0_bary is None or self._p0_bary.shape != shape:
            self._p0_bary = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        if self._k1 is None or self._k1.shape != shape:
            self._k1 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        if self._k2 is None or self._k2.shape != shape:
            self._k2 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        if self._k3 is None or self._k3.shape != shape:
            self._k3 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        if self._k4 is None or self._k4.shape != shape:
            self._k4 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        if self._k is None or self._k.shape != shape:
            self._k = ti.Vector.field(3, dtype=ti.f32, shape=shape)

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

    @ti.kernel
    def _write_pos_out(
        self,
        V_pos: ti.template(),
        F_verts: ti.template(),
        face_ids: ti.template(),
        bary: ti.template(),
        pos_out: ti.template(),
        n: ti.i32,
    ):
        for pid in range(n):
            fid = face_ids[pid]
            fv = F_verts[fid]
            x1 = V_pos[fv[0]]
            x2 = V_pos[fv[1]]
            x3 = V_pos[fv[2]]
            pos_out[pid] = x1 * bary[pid][0] + x2 * bary[pid][1] + x3 * bary[pid][2]

    def compute(self, reg: Registry) -> None:
        io = self.io
        inputs = self.require_inputs()
        V_pos = inputs["V_pos"].get()
        F_verts = inputs["F_verts"].get()
        F_adj = inputs["F_adj"].get()
        V_vel = inputs["V_velocity"].get()
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

        shape = face_ids.shape
        self._ensure_tmp_fields(shape)

        self.adv.backup_state(face_ids, bary, self._p0_face, self._p0_bary, n)

        self.sampler.run(
            F_verts=F_verts,
            V_vel=V_vel,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self._k1,
            n_pts=n,
        )

        self.adv.advect_inplace(
            V_pos, F_verts, F_adj, face_ids, bary, self._k1, 0.5 * dt, n
        )
        self._touch_vortex_state()
        V_vel = io.V_velocity.get()
        self.sampler.run(
            F_verts=F_verts,
            V_vel=V_vel,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self._k2,
            n_pts=n,
        )
        self.adv.restore_state(face_ids, bary, self._p0_face, self._p0_bary, n)

        self.adv.advect_inplace(
            V_pos, F_verts, F_adj, face_ids, bary, self._k2, 0.5 * dt, n
        )
        self._touch_vortex_state()
        V_vel = io.V_velocity.get()
        self.sampler.run(
            F_verts=F_verts,
            V_vel=V_vel,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self._k3,
            n_pts=n,
        )
        self.adv.restore_state(face_ids, bary, self._p0_face, self._p0_bary, n)

        self.adv.advect_inplace(V_pos, F_verts, F_adj, face_ids, bary, self._k3, dt, n)
        self._touch_vortex_state()
        V_vel = io.V_velocity.get()
        self.sampler.run(
            F_verts=F_verts,
            V_vel=V_vel,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self._k4,
            n_pts=n,
        )
        self.adv.restore_state(face_ids, bary, self._p0_face, self._p0_bary, n)

        self.adv.rk4_combine_vel(self._k1, self._k2, self._k3, self._k4, self._k, n)
        self.adv.advect_inplace(V_pos, F_verts, F_adj, face_ids, bary, self._k, dt, n)
        self._touch_vortex_state()

        self.adv.backup_state(face_ids, bary, face_ids_out, bary_out, n)
        self._write_pos_out(V_pos, F_verts, face_ids, bary, pos_out, n)

        io.face_ids_out.commit()
        io.bary_out.commit()
        io.pos_out.commit()
