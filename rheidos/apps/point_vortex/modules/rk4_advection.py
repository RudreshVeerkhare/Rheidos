# rk4_advection.py
#
# Minimal, pasteable implementation that matches your plan:
#   - Velocity depends on current vortex placement (so we recompute it each RK stage)
#   - RK4 only needs: one backup of (face_id,bary) at p0 + k1..k4 buffers
#   - Advection for each RK stage assumes sampled velocity is constant over that substep
#   - All kernels live inside the @ti.data_oriented producers:
#       * SampleVortexVelProducer
#       * AdvectConstVelEventDrivenProducer  (includes backup/restore + RK4 combine + event-driven walk)
#
# Assumptions about your modules:
#   mesh.F_verts.get() -> ti.Field shape (nF,) with vec3i or (nF, 3) i32
#   mesh.F_adj.get()   -> ti.Field shape (nF,) with vec3i (neighbor across edge opposite each vertex; -1 boundary)
#   mesh.V_pos.get()   -> ti.Field shape (nV,) with vec3f
#   pt_vortex.face_ids.get() -> ti.Field shape (maxV,) i32
#   pt_vortex.bary.get()     -> ti.Field shape (maxV,) vec3f
#   pt_vortex.n_vortices.get() -> ti.Field scalar i32
#   velocity.V_velocity.get() -> ti.Field shape (nV,) vec3f (per-vertex velocity in world coords)
#
# If your velocity is facewise constant (vel_F), I added a comment where to swap sampling.

from dataclasses import dataclass
from typing import Optional

import taichi as ti

from rheidos.compute import ModuleBase, World  # keep your style
from rheidos.compute.profiler.runtime import get_current_profiler
from .surface_mesh import SurfaceMeshModule
from .point_vortex import PointVortexModule
from .velocity_field import VelocityFieldModule


# -----------------------------------------------------------------------------
# Producer: sample velocity at current vortex locations
# -----------------------------------------------------------------------------


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

    # If you have facewise constant velocity vel_F, use this instead:
    # @ti.kernel
    # def _sample_from_face_vel(self, vel_F: ti.template(), pt_face: ti.template(), pt_vel: ti.template(), n_pts: ti.i32):
    #     for pid in range(n_pts):
    #         pt_vel[pid] = vel_F[pt_face[pid]]

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

        self.mesh = world.require(SurfaceMeshModule)  # or SurfaceMeshModule
        self.pt_vortex = world.require(PointVortexModule)  # or PointVortexModule
        self.velocity = world.require(VelocityFieldModule)  # or VelocityFieldModule

        # Instantiate producers
        self.sampler = SampleVortexVelProducer()
        self.adv = AdvectConstVelEventDrivenProducer()

        # Allocate buffers once (device-side)
        # Determine max vortex capacity from the actual field shape
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
        self.pt_vortex.face_ids.bump()
        self.pt_vortex.bary.bump()

    def advect(self, dt: float) -> None:
        prof = get_current_profiler()

        def span(name: str):
            return prof.span(name, cat="rk4")

        with span("fetch_state"):
            n = int(self.pt_vortex.n_vortices.get()[None])
            if n <= 0:
                return

            face_ids = self.pt_vortex.face_ids.get()
            bary = self.pt_vortex.bary.get()

            V_pos = self.mesh.V_pos.get()
            F_verts = self.mesh.F_verts.get()
            F_adj = self.mesh.F_adj.get()

        def sample_vel(out_buf, label: str) -> None:
            with span(f"{label}_sample"):
                V_vel = self.velocity.V_velocity.get()
                self.sampler.run(
                    F_verts=F_verts,
                    V_vel=V_vel,
                    pt_bary=bary,
                    pt_face=face_ids,
                    pt_vel_out=out_buf,
                    n_pts=n,
                )

        def advect_stage(vel_buf, step_dt: float, label: str) -> None:
            with span(f"{label}_advect"):
                self.adv.advect_inplace(
                    V_pos, F_verts, F_adj, face_ids, bary, vel_buf, step_dt, n
                )
                self._touch_vortex_state()

        with span("backup_state"):
            self.adv.backup_state(face_ids, bary, self.p0_face, self.p0_bary, n)

        # --- k1 @ p0 ---
        sample_vel(self.k1, "k1")

        # --- k2 @ p0 + 0.5dt*k1 ---
        advect_stage(self.k1, 0.5 * dt, "k2")
        sample_vel(self.k2, "k2")
        with span("k2_restore"):
            self.adv.restore_state(face_ids, bary, self.p0_face, self.p0_bary, n)

        # --- k3 @ p0 + 0.5dt*k2 ---
        advect_stage(self.k2, 0.5 * dt, "k3")
        sample_vel(self.k3, "k3")
        with span("k3_restore"):
            self.adv.restore_state(face_ids, bary, self.p0_face, self.p0_bary, n)

        # --- k4 @ p0 + dt*k3 ---
        advect_stage(self.k3, dt, "k4")
        sample_vel(self.k4, "k4")
        with span("k4_restore"):
            self.adv.restore_state(face_ids, bary, self.p0_face, self.p0_bary, n)

        with span("combine_k"):
            self.adv.rk4_combine_vel(self.k1, self.k2, self.k3, self.k4, self.k, n)

        with span("final_advect"):
            self.adv.advect_inplace(
                V_pos, F_verts, F_adj, face_ids, bary, self.k, dt, n
            )
            self._touch_vortex_state()
