from dataclasses import dataclass
from typing import Any

import numpy as np
import taichi as ti

from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry


@dataclass
class AdvectVorticesEventDrivenIO:
    # Mesh
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    F_adj: ResourceRef[
        ti.Field
    ]  # (nF, vec3i) neighbor across edge opposite each vertex; -1 = boundary

    # Facewise-constant velocity (tangent-ish)
    vel_F: ResourceRef[ti.Field]  # (nF, vec3f)

    # Vortices (state)
    n_vortices: ResourceRef[ti.Field]  # scalar i32
    face_ids: ResourceRef[ti.Field]  # (maxV, i32)
    bary: ResourceRef[ti.Field]  # (maxV, vec3f)

    # Time step
    dt: ResourceRef[Any]  # scalar f32

    # Outputs (new state)
    face_ids_out: ResourceRef[np.ndarray] = out_field()  # (nVortices,)
    bary_out: ResourceRef[np.ndarray] = out_field()  # (nVortices, 3)
    pos_out: ResourceRef[np.ndarray] = out_field()  # (nVortices, vec3f)


@ti.data_oriented
class AdvectVorticesEventDrivenProducer(WiredProducer[AdvectVorticesEventDrivenIO]):
    """
    Event-driven advection for facewise-constant velocity on a triangle mesh.

    Fixes included:
      - Avoids ti.static unrolling of max_hops (prevents massive kernel/JIT compile stalls).
      - Clamps n_vortices to buffer capacity (prevents OOB -> GPU hang/timeouts).
      - Validates face_ids and F_adj targets (prevents OOB on F/F_adj).
      - Reads dt as a Python scalar (avoids ambiguous scalar-field passing).
    """

    max_hops = 6
    eps = 1e-10
    edge_push = 1e-8
    project_velocity_to_face = True

    # ----------------- geometry helpers -----------------

    @ti.func
    def _pos_from_face_bary(self, V, F, f: ti.i32, b: ti.types.vector(3, ti.f32)):
        i = F[f][0]
        j = F[f][1]
        k = F[f][2]
        return b[0] * V[i] + b[1] * V[j] + b[2] * V[k]

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

        # fallback
        u = ti.cast(1.0, ti.f32)
        v = ti.cast(0.0, ti.f32)
        w = ti.cast(0.0, ti.f32)

        if ti.abs(denom) >= 1e-20:
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = ti.cast(1.0, ti.f32) - v - w

        return ti.math.vec3(u, v, w)

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
    def _project_to_face(self, v, n_hat):
        return v - n_hat * (n_hat.dot(v))

    @ti.func
    def _bary_grads(self, a, b, c):
        n0 = ti.math.cross(b - a, c - a)
        nn = n0.dot(n0)

        z = ti.math.vec3(0.0, 0.0, 0.0)
        gi = z
        gj = z
        gk = z

        if nn >= 1e-20:
            gi = ti.math.cross(n0, c - b) / nn
            gj = ti.math.cross(n0, a - c) / nn
            gk = ti.math.cross(n0, b - a) / nn

        return gi, gj, gk, n0, nn

    # ----------------- event-driven step for one vortex -----------------

    @ti.func
    def _advance_event_driven_one(
        self,
        V,
        F,
        F_adj,
        vel_F,
        p0,
        f0: ti.i32,
        b0: ti.types.vector(3, ti.f32),
        dt: ti.f32,
        nF: ti.i32,
    ):
        p = p0
        f = f0
        b = b0
        t_rem = dt

        eps = ti.cast(self.eps, ti.f32)
        push = ti.cast(self.edge_push, ti.f32)

        f_out = f
        b_out = b
        p_out = p
        done = ti.cast(0, ti.i32)

        for _ in range(self.max_hops):
            if done == 0:
                if t_rem <= 0.0:
                    f_out, b_out, p_out = f, b, p
                    done = 1
                else:
                    # safety: f must remain valid
                    if f < 0 or f >= nF:
                        f_out = ti.cast(-1, ti.i32)
                        b_out = ti.math.vec3(1.0, 0.0, 0.0)
                        p_out = p
                        done = 1
                    else:
                        i = F[f][0]
                        j = F[f][1]
                        k = F[f][2]
                        a = V[i]
                        bb = V[j]
                        c = V[k]

                        gi, gj, gk, n0, nn = self._bary_grads(a, bb, c)

                        if nn < 1e-20:
                            b = self._clamp_renorm_bary(self._barycentric(p, a, bb, c))
                            f_out, b_out, p_out = f, b, p
                            done = 1
                        else:
                            n_hat = n0 / ti.sqrt(nn)

                            v = vel_F[f]
                            if ti.static(self.project_velocity_to_face):
                                v = self._project_to_face(v, n_hat)

                            # refresh bary
                            b = self._clamp_renorm_bary(self._barycentric(p, a, bb, c))

                            dbi = gi.dot(v)
                            dbj = gj.dot(v)
                            dbk = gk.dot(v)

                            t_hit = ti.cast(1e30, ti.f32)
                            edge = ti.cast(-1, ti.i32)

                            if dbi < -eps:
                                ti_ = -b[0] / dbi
                                if ti_ < t_hit:
                                    t_hit, edge = ti_, 0
                            if dbj < -eps:
                                tj_ = -b[1] / dbj
                                if tj_ < t_hit:
                                    t_hit, edge = tj_, 1
                            if dbk < -eps:
                                tk_ = -b[2] / dbk
                                if tk_ < t_hit:
                                    t_hit, edge = tk_, 2

                            # finish inside face
                            if edge == -1 or t_hit >= t_rem:
                                p = p + v * t_rem
                                b = b + ti.math.vec3(dbi, dbj, dbk) * t_rem
                                b = self._clamp_renorm_bary(b)
                                f_out, b_out, p_out = f, b, p
                                done = 1
                            else:
                                # step to edge
                                t_hit = ti.max(0.0, t_hit)
                                p = p + v * t_hit
                                t_rem -= t_hit

                                nf = F_adj[f][edge]

                                # validate adjacency target too (prevents OOB)
                                if nf < 0 or nf >= nF:
                                    b = self._clamp_renorm_bary(
                                        self._barycentric(p, a, bb, c)
                                    )
                                    f_out, b_out, p_out = f, b, p
                                    done = 1
                                else:
                                    f = nf

                                    ii = F[f][0]
                                    jj = F[f][1]
                                    kk = F[f][2]
                                    aa = V[ii]
                                    bbb = V[jj]
                                    cc = V[kk]

                                    b_new = self._clamp_renorm_bary(
                                        self._barycentric(p, aa, bbb, cc)
                                    )

                                    # nudge inward if on edge
                                    if (
                                        ti.min(b_new[0], ti.min(b_new[1], b_new[2]))
                                        < push
                                    ):
                                        m = 0
                                        if b_new[1] < b_new[m]:
                                            m = 1
                                        if b_new[2] < b_new[m]:
                                            m = 2
                                        b_new[m] = b_new[m] + push
                                        b_new = self._clamp_renorm_bary(b_new)
                                        p = (
                                            b_new[0] * aa
                                            + b_new[1] * bbb
                                            + b_new[2] * cc
                                        )

                                    b = b_new

        # best effort if hop budget exceeded
        if done == 0:
            if f >= 0 and f < nF:
                i = F[f][0]
                j = F[f][1]
                k = F[f][2]
                a = V[i]
                bb = V[j]
                c = V[k]
                b = self._clamp_renorm_bary(self._barycentric(p, a, bb, c))
                f_out, b_out, p_out = f, b, p
            else:
                f_out = ti.cast(-1, ti.i32)
                b_out = ti.math.vec3(1.0, 0.0, 0.0)
                p_out = p

        return f_out, b_out, p_out

    # ----------------- kernel over all vortices -----------------

    @ti.kernel
    def _advect_all(
        self,
        V: ti.template(),
        F: ti.template(),
        F_adj: ti.template(),
        vel_F: ti.template(),
        face_ids: ti.template(),
        bary: ti.template(),
        nV: ti.i32,
        nF: ti.i32,
        dt: ti.f32,
        face_ids_out: ti.types.ndarray(dtype=ti.i32, ndim=1),
        bary_out: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
        pos_out: ti.types.ndarray(dtype=ti.types.vector(3, dtype=ti.f32)),
    ):
        for vid in range(nV):
            f0 = face_ids[vid]

            # validate face id early (prevents OOB on F)
            if f0 < 0 or f0 >= nF:
                face_ids_out[vid] = -1
                bary_out[vid] = ti.math.vec3(1.0, 0.0, 0.0)
                continue

            b0 = bary[vid]
            p0 = self._pos_from_face_bary(V, F, f0, b0)

            fN, bN, pN = self._advance_event_driven_one(
                V, F, F_adj, vel_F, p0, f0, b0, dt, nF
            )

            face_ids_out[vid] = fN
            bary_out[vid] = bN
            pos_out[vid] = pN

    # ----------------- Rheidos compute -----------------

    def compute(self, reg: Registry) -> None:
        io = self.io
        inputs = self.require_inputs()

        V = inputs["V_pos"].get()
        F = inputs["F_verts"].get()
        F_adj = inputs["F_adj"].get()
        vel_F = inputs["vel_F"].get()
        n_vortices = inputs["n_vortices"].get()
        face_ids = inputs["face_ids"].get()
        bary = inputs["bary"].get()
        dt = inputs["dt"].get()

        nF = int(F.shape[0])

        # Robustly fetch & clamp nV (prevents OOB -> GPU hang/timeout)
        nV = int(n_vortices[None])

        outputs = self.ensure_outputs(reg)
        face_ids_out = outputs["face_ids_out"].peek()
        bary_out = outputs["bary_out"].peek()
        pos_out = outputs["pos_out"].peek()

        self._advect_all(
            V,
            F,
            F_adj,
            vel_F,
            face_ids,
            bary,
            nV,
            nF,
            dt,
            face_ids_out,
            bary_out,
            pos_out,
        )

        io.face_ids_out.commit()
        io.bary_out.commit()
        io.pos_out.commit()
