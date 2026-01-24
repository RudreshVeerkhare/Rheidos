import taichi as ti


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

    # ---------- utilities: barycentric math ----------

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

        # grad(bary_i) are constant over a triangle
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

                # project to face plane to avoid off-surface drift
                v = vel_world - n_hat * vel_world.dot(n_hat)
                db = ti.Vector([g0.dot(v), g1.dot(v), g2.dot(v)], dt=ti.f32)

                # find first time a bary component hits 0 (edge crossing)
                t_hit = t_rem
                hit_idx = -1
                for i in ti.static(range(3)):
                    if db[i] < -AdvectConstVelEventDrivenProducer.EPS:
                        cand = -b[i] / db[i]
                        if cand < t_hit:
                            t_hit = cand
                            hit_idx = i

                # advance within face
                b = b + t_hit * db
                t_rem = t_rem - t_hit

                # no crossing => done
                if hit_idx == -1 or t_rem <= AdvectConstVelEventDrivenProducer.EPS:
                    active = 0
                else:
                    fN = F_adj[f][hit_idx]
                    if fN < 0:
                        active = 0  # boundary: stop
                    else:
                        # hop to neighbor face and recompute bary from world position
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

    # ---------- kernels: copy / combine / advect ----------

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
