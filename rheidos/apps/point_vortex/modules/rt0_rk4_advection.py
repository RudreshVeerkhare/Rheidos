# rk4_advection.py
#
# Updated to use Option 1 (RT0 reconstruction sampled per-face via vel_FV):
#
#   velocity.vel_FV.get() -> ti.Field shape (nF, 3) vec3f
#     where vel_FV[f,loc] is velocity at the face corner "loc" (0,1,2),
#     and "loc" matches F_verts[f][loc].
#
# Sampling at a vortex (face_id, bary):
#   v = sum_loc bary[loc] * vel_FV[face_id, loc]
#
# This evaluates the affine RT0 field exactly (within the face).

from dataclasses import dataclass
from typing import Optional

import taichi as ti

from rheidos.compute import ModuleBase, World  # keep your style
from .surface_mesh import SurfaceMeshModule
from .point_vortex import PointVortexModule
from .velocity_field import VelocityFieldModule


# -----------------------------------------------------------------------------
# Producer: sample velocity at current vortex locations (Option 1: RT0 vel_FV)
# -----------------------------------------------------------------------------


@ti.data_oriented
class SampleVortexVelProducer:
    """
    Samples an RT0 affine velocity field represented per face by corner velocities:

        vel_FV[f,loc]  loc=0..2 corresponds to vertex F_verts[f][loc]

    At vortex position stored as (face_id, bary):
        v = b0*vel_FV[f,0] + b1*vel_FV[f,1] + b2*vel_FV[f,2]
    """

    def __init__(self) -> None:
        pass

    @ti.kernel
    def _sample_from_face_corner_vel(
        self,
        vel_FV: ti.template(),  # (nF, 3) vec3f
        pt_bary: ti.template(),  # (maxP,) vec3f
        pt_face: ti.template(),  # (maxP,) i32
        pt_vel: ti.template(),  # (maxP,) vec3f [out]
        n_pts: ti.i32,
    ):
        for pid in range(n_pts):
            f = pt_face[pid]
            b = pt_bary[pid]
            v0 = vel_FV[f, 0]
            v1 = vel_FV[f, 1]
            v2 = vel_FV[f, 2]
            pt_vel[pid] = b[0] * v0 + b[1] * v1 + b[2] * v2

    def run(
        self,
        *,
        vel_FV,
        pt_bary,
        pt_face,
        pt_vel_out,
        n_pts: int,
    ) -> None:
        self._sample_from_face_corner_vel(vel_FV, pt_bary, pt_face, pt_vel_out, n_pts)


# -----------------------------------------------------------------------------
# Producer: event-driven advection with constant velocity (plus backup/restore/comb)
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# RK4 Module (imperative orchestration, minimal state)
# -----------------------------------------------------------------------------


class RK4AdvectionModule(ModuleBase):
    NAME = "RK4AdvectionModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule)
        self.pt_vortex = world.require(PointVortexModule)

        # IMPORTANT: VelocityFieldModule must expose vel_FV (nF,3) vec3f,
        # computed from current vortex placement.
        self.velocity = world.require(VelocityFieldModule)

        self.sampler = SampleVortexVelProducer()
        self.adv = AdvectConstVelEventDrivenProducer()

        face_ids_field = self.pt_vortex.face_ids.get()
        maxV = int(face_ids_field.shape[0])

        self.p0_face = ti.field(dtype=ti.i32, shape=maxV)
        self.p0_bary = ti.Vector.field(3, dtype=ti.f32, shape=maxV)

        self.k1 = ti.Vector.field(3, dtype=ti.f32, shape=maxV)
        self.k2 = ti.Vector.field(3, dtype=ti.f32, shape=maxV)
        self.k3 = ti.Vector.field(3, dtype=ti.f32, shape=maxV)
        self.k4 = ti.Vector.field(3, dtype=ti.f32, shape=maxV)
        self.k = ti.Vector.field(3, dtype=ti.f32, shape=maxV)

    def _touch_vortex_state(self) -> None:
        # Ensure upstream velocity graph recomputes when vortex state changes
        self.pt_vortex.face_ids.bump()
        self.pt_vortex.bary.bump()

    def advect(self, dt: float) -> None:
        n = int(self.pt_vortex.n_vortices.get()[None])
        if n <= 0:
            return

        face_ids = self.pt_vortex.face_ids.get()
        bary = self.pt_vortex.bary.get()

        V_pos = self.mesh.V_pos.get()
        F_verts = self.mesh.F_verts.get()
        F_adj = self.mesh.F_adj.get()

        # Backup p0
        self.adv.backup_state(face_ids, bary, self.p0_face, self.p0_bary, n)

        # --- k1 @ p0 ---
        vel_FV = (
            self.velocity.FV_velocity.get()
        )  # (nF,3) vec3f, depends on current vortex placement
        self.sampler.run(
            vel_FV=vel_FV,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self.k1,
            n_pts=n,
        )

        # --- k2 @ p0 + 0.5dt*k1 ---
        self.adv.advect_inplace(
            V_pos, F_verts, F_adj, face_ids, bary, self.k1, 0.5 * dt, n
        )
        self._touch_vortex_state()
        vel_FV = self.velocity.FV_velocity.get()
        self.sampler.run(
            vel_FV=vel_FV,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self.k2,
            n_pts=n,
        )
        self.adv.restore_state(face_ids, bary, self.p0_face, self.p0_bary, n)

        # --- k3 @ p0 + 0.5dt*k2 ---
        self.adv.advect_inplace(
            V_pos, F_verts, F_adj, face_ids, bary, self.k2, 0.5 * dt, n
        )
        self._touch_vortex_state()
        vel_FV = self.velocity.FV_velocity.get()
        self.sampler.run(
            vel_FV=vel_FV,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self.k3,
            n_pts=n,
        )
        self.adv.restore_state(face_ids, bary, self.p0_face, self.p0_bary, n)

        # --- k4 @ p0 + dt*k3 ---
        self.adv.advect_inplace(V_pos, F_verts, F_adj, face_ids, bary, self.k3, dt, n)
        self._touch_vortex_state()
        vel_FV = self.velocity.FV_velocity.get()
        self.sampler.run(
            vel_FV=vel_FV,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self.k4,
            n_pts=n,
        )
        self.adv.restore_state(face_ids, bary, self.p0_face, self.p0_bary, n)

        # combine + final step (commit)
        self.adv.rk4_combine_vel(self.k1, self.k2, self.k3, self.k4, self.k, n)
        self.adv.advect_inplace(V_pos, F_verts, F_adj, face_ids, bary, self.k, dt, n)
        self._touch_vortex_state()
