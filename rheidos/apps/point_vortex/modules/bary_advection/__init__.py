from rheidos.compute import (
    ModuleBase,
    World,
    ResourceSpec,
    ResourceRef,
    shape_of,
    WiredProducer,
    out_field,
)
from rheidos.compute.registry import Registry

from ..point_vortex import PointVortexModule
from ..stream_func import StreamFunctionModule
from ..surface_mesh import SurfaceMeshModule
from ..self_vel_basis import SelfVelBasisModule

import taichi as ti
from dataclasses import dataclass


@dataclass
class BaryAdvectionRK4ProducerIO:
    dt: ResourceRef[float]
    bary: ResourceRef[ti.Field]
    vortex_gammas: ResourceRef[ti.Field]
    psi: ResourceRef[ti.Field]
    faceids: ResourceRef[ti.Field]
    F_verts: ResourceRef[ti.Field]
    F_area: ResourceRef[ti.Field]
    F_normals: ResourceRef[ti.Field]
    F_adj: ResourceRef[ti.Field]
    V_pos: ResourceRef[ti.Field]
    self_vel_basis: ResourceRef[ti.Field]
    query_face_id: ResourceRef[int]

    # Outputs
    bary_out: ResourceRef[ti.Field] = out_field()
    k1: ResourceRef[ti.Field] = out_field()
    k2: ResourceRef[ti.Field] = out_field()
    k3: ResourceRef[ti.Field] = out_field()
    k4: ResourceRef[ti.Field] = out_field()


@ti.data_oriented
class BaryAdvectionRK4Producer(WiredProducer[BaryAdvectionRK4ProducerIO]):

    IO_TYPE = BaryAdvectionRK4ProducerIO

    # ---------------------------
    # Small helpers
    # ---------------------------

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
    def _safe_normalize(
        self, v: ti.types.vector(3, ti.f32)
    ) -> ti.types.vector(3, ti.f32):
        n2 = v.dot(v)
        out = ti.math.vec3(0.0, 0.0, 1.0)
        if n2 > 1e-20:
            out = v / ti.sqrt(n2)
        return out

    @ti.func
    def _project_tangent(self, v, n_hat):
        return v - (v.dot(n_hat)) * n_hat

    @ti.func
    def _bary_grads(
        self,
        a: ti.types.vector(3, ti.f32),
        b: ti.types.vector(3, ti.f32),
        c: ti.types.vector(3, ti.f32),
        n_hat: ti.types.vector(3, ti.f32),
        A: ti.f32,
    ):
        # Gradients of barycentric coordinates on a 3D triangle (tangent vectors)
        inv2A = 1.0 / (2.0 * A + 1e-20)

        # edge opposite vertex a is (b,c), etc.
        e0 = c - b  # opposite a
        e1 = a - c  # opposite b
        e2 = b - a  # opposite c

        g0 = ti.math.cross(n_hat, e0) * inv2A
        g1 = ti.math.cross(n_hat, e1) * inv2A
        g2 = ti.math.cross(n_hat, e2) * inv2A
        return g0, g1, g2

    # ---------------------------
    # Compute d(bary)/dt (with self-velocity subtraction)
    # ---------------------------

    @ti.kernel
    def _bary_dot(
        self,
        bary: ti.template(),
        vortex_gammas: ti.template(),
        faceids: ti.template(),
        face_areas: ti.template(),
        f_verts: ti.template(),
        v_pos: ti.template(),
        f_normals: ti.template(),
        psi: ti.template(),
        self_vel_basis: ti.template(),  # (nF, 3) vec3 entries
        bary_dot: ti.template(),
    ):
        for vid in faceids:
            faceid = faceids[vid]
            if faceid < 0:
                continue

            i0 = f_verts[faceid][0]
            i1 = f_verts[faceid][1]
            i2 = f_verts[faceid][2]

            p0 = psi[i0]
            p1 = psi[i1]
            p2 = psi[i2]

            A = face_areas[faceid]
            inv2A = 1.0 / (2.0 * A + 1e-20)

            # Base barycentric rates from psi (your original formula)
            lamdot0 = (p2 - p1) * inv2A
            lamdot1 = (p0 - p2) * inv2A
            lamdot2 = (p1 - p0) * inv2A

            # --- self velocity subtraction ---
            # Interpolate self-velocity using *clamped* bary weights (safer at RK stages)
            bw = self._clamp_renorm_bary(bary[vid])

            u_self = (
                bw[0] * self_vel_basis[faceid, 0]
                + bw[1] * self_vel_basis[faceid, 1]
                + bw[2] * self_vel_basis[faceid, 2]
            )

            # If your basis is "per unit gamma", keep this.
            # If your basis ALREADY includes gamma, delete the next line.
            u_self *= vortex_gammas[vid]

            n_hat = self._safe_normalize(f_normals[faceid])
            u_self = self._project_tangent(u_self, n_hat)

            x0 = v_pos[i0]
            x1 = v_pos[i1]
            x2 = v_pos[i2]
            g0, g1, g2 = self._bary_grads(x0, x1, x2, n_hat, A)

            lamdot0 -= u_self.dot(g0)
            lamdot1 -= u_self.dot(g1)
            lamdot2 -= u_self.dot(g2)

            # Optional: kill tiny drift in sum-to-zero
            m = (lamdot0 + lamdot1 + lamdot2) / 3.0
            lamdot0 -= m
            lamdot1 -= m
            lamdot2 -= m

            bary_dot[vid] = ti.math.vec3(lamdot0, lamdot1, lamdot2)

    # ---------------------------
    # RK4 plumbing
    # ---------------------------

    @ti.kernel
    def _step_y(
        self,
        bary_in: ti.template(),  # (nV, vec3f)
        bary_out: ti.template(),  # (nV, vec3f)
        scale: ti.f32,
        bary_dot: ti.template(),  # (nV, vec3f)
    ):
        for vid in bary_in:
            bary_out[vid] = bary_in[vid] + scale * bary_dot[vid]

    @ti.kernel
    def _final_step(
        self,
        bary: ti.template(),  # (nV, vec3f)
        dt: ti.f32,
        k1: ti.template(),
        k2: ti.template(),
        k3: ti.template(),
        k4: ti.template(),
    ):
        for vid in bary:
            bary[vid] = (
                bary[vid] + dt * (k1[vid] + 2 * k2[vid] + 2 * k3[vid] + k4[vid]) / 6.0
            )

    # ---------------------------
    # Edge-walk face id update (your existing code)
    # ---------------------------

    @ti.kernel
    def _update_face_ids(
        self,
        barys: ti.template(),  # (maxV, vec3f)
        faceids: ti.template(),  # (maxV,)
        F_verts: ti.template(),  # (nF, vec3i)
        F_normals: ti.template(),  # (nF, vec3f)
        F_adj: ti.template(),  # (nF, vec3i)
        V_pos: ti.template(),  # (nV, vec3f)
    ):
        for vid in barys:
            faceid = faceids[vid]
            if faceid < 0:
                continue

            bary = barys[vid]

            # Find most-negative bary component
            min_idx = 0
            min_val = bary[0]
            if bary[1] < min_val:
                min_val = bary[1]
                min_idx = 1
            if bary[2] < min_val:
                min_val = bary[2]
                min_idx = 2

            if min_val >= 0.0:
                continue

            opp_face_id = F_adj[faceid][min_idx]
            assert opp_face_id >= 0, f"Opposite face doesn't exists: {opp_face_id}"

            p_opp = F_verts[faceid][min_idx]

            v1 = 1
            v2 = 2
            if min_idx == 1:
                v1 = 0
                v2 = 2
            elif min_idx == 2:
                v1 = 0
                v2 = 1

            p1 = F_verts[faceid][v1]
            p2 = F_verts[faceid][v2]

            # World-space point from current bary
            x = (
                V_pos[p1] * bary[v1]
                + V_pos[p2] * bary[v2]
                + V_pos[p_opp] * bary[min_idx]
            )

            # Rotate around shared edge to neighbor face plane
            axis = V_pos[p2] - V_pos[p1]
            axis_len2 = axis.dot(axis)
            if axis_len2 > 1e-20:
                axis = axis / ti.sqrt(axis_len2)
                n0 = F_normals[faceid]
                n1 = F_normals[opp_face_id]
                n0 = self._safe_normalize(n0)
                n1 = self._safe_normalize(n1)

                s = axis.dot(ti.math.cross(n0, n1))
                c = n0.dot(n1)
                theta = ti.atan2(s, c)

                v = x - V_pos[p1]
                half = 0.5 * theta
                qv = axis * ti.sin(half)
                qw = ti.cos(half)

                t = 2.0 * ti.math.cross(qv, v)
                v_rot = v + qw * t + ti.math.cross(qv, t)
                x = V_pos[p1] + v_rot

            # Recompute bary on neighbor
            a = V_pos[F_verts[opp_face_id][0]]
            b = V_pos[F_verts[opp_face_id][1]]
            c = V_pos[F_verts[opp_face_id][2]]
            nbary = self._barycentric(x, a, b, c)

            barys[vid] = nbary
            faceids[vid] = opp_face_id

    # ---------------------------
    # Main compute
    # ---------------------------

    def compute(self, reg: Registry) -> None:
        # ----------------------------------------------------
        # IMPORTANT: set query_face_id BEFORE self_vel_basis.get()
        # ----------------------------------------------------
        faceids = self.io.faceids.get()
        for fid in faceids.to_numpy():
            self.io.query_face_id.set(fid)

            # trigger vel basis calculation
            self.io.self_vel_basis.get()

        inputs = self.require_inputs()
        outputs = self.ensure_outputs(reg)

        bary = inputs["bary"].get()
        vortex_gammas = inputs["vortex_gammas"].get()
        dt = inputs["dt"].get()
        faceids = inputs["faceids"].get()
        F_verts = inputs["F_verts"].get()
        F_area = inputs["F_area"].get()
        F_normals = inputs["F_normals"].get()
        F_adj = inputs["F_adj"].get()
        V_pos = inputs["V_pos"].get()

        bary_out = outputs["bary_out"].peek()
        k1 = outputs["k1"].peek()
        k2 = outputs["k2"].peek()
        k3 = outputs["k3"].peek()
        k4 = outputs["k4"].peek()

        # Cache start state
        bary_out.copy_from(bary)

        # ---------------- RK4 ----------------
        self_vel_basis = inputs["self_vel_basis"].get()
        # step 1
        psi = inputs["psi"].get()
        self._bary_dot(
            bary,
            vortex_gammas,
            faceids,
            F_area,
            F_verts,
            V_pos,
            F_normals,
            psi,
            self_vel_basis,
            k1,
        )
        # Update face ids if crossed edges
        self._update_face_ids(bary_out, faceids, F_verts, F_normals, F_adj, V_pos)

        # step 2
        self._step_y(bary_out, bary, dt / 2.0, k1)
        inputs["bary"].bump()
        psi = inputs["psi"].get()
        self._bary_dot(
            bary,
            vortex_gammas,
            faceids,
            F_area,
            F_verts,
            V_pos,
            F_normals,
            psi,
            self_vel_basis,
            k2,
        )  # Update face ids if crossed edges
        self._update_face_ids(bary_out, faceids, F_verts, F_normals, F_adj, V_pos)

        # step 3
        self._step_y(bary_out, bary, dt / 2.0, k2)
        inputs["bary"].bump()
        psi = inputs["psi"].get()
        self._bary_dot(
            bary,
            vortex_gammas,
            faceids,
            F_area,
            F_verts,
            V_pos,
            F_normals,
            psi,
            self_vel_basis,
            k3,
        )
        # Update face ids if crossed edges
        self._update_face_ids(bary_out, faceids, F_verts, F_normals, F_adj, V_pos)

        # step 4
        self._step_y(bary_out, bary, dt, k3)
        inputs["bary"].bump()
        psi = inputs["psi"].get()
        self._bary_dot(
            bary,
            vortex_gammas,
            faceids,
            F_area,
            F_verts,
            V_pos,
            F_normals,
            psi,
            self_vel_basis,
            k4,
        )
        # Update face ids if crossed edges
        self._update_face_ids(bary_out, faceids, F_verts, F_normals, F_adj, V_pos)

        # final combine into bary_out
        self._final_step(bary_out, dt, k1, k2, k3, k4)

        # Update face ids if crossed edges
        self._update_face_ids(bary_out, faceids, F_verts, F_normals, F_adj, V_pos)

        self.io.bary_out.commit()


class BaryAdvectionModule(ModuleBase):
    NAME = "BaryAdvectionModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule)
        self.pt_vortex = world.require(PointVortexModule)
        self.stream_func = world.require(StreamFunctionModule)
        self.self_vel_basis = world.require(SelfVelBasisModule)

        self.dt = self.resource(
            "dt",
            spec=ResourceSpec(kind="python", dtype=float),
            doc="Timestep size in seconds.",
            declare=True,
        )

        self.bary_out = self.resource(
            "bary_out",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.pt_vortex.bary),
                lanes=3,
                allow_none=True,
            ),
            doc="Updated barycentric coordinates after RK4 advection. Shape: (maxV, vec3f)",
            declare=False,
        )

        self.k1 = self.resource(
            "k1",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.pt_vortex.bary),
                lanes=3,
                allow_none=True,
            ),
            doc="k1 buffer for step 1",
            declare=True,
        )
        self.k2 = self.resource(
            "k2",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.pt_vortex.bary),
                lanes=3,
                allow_none=True,
            ),
            doc="k2 buffer for step 2",
            declare=True,
        )
        self.k3 = self.resource(
            "k3",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.pt_vortex.bary),
                lanes=3,
                allow_none=True,
            ),
            doc="k3 buffer for step 3",
            declare=True,
        )
        self.k4 = self.resource(
            "k4",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.pt_vortex.bary),
                lanes=3,
                allow_none=True,
            ),
            doc="k4 buffer for step 4",
            declare=True,
        )

        producer = BaryAdvectionRK4Producer(
            dt=self.dt,
            bary=self.pt_vortex.bary,
            vortex_gammas=self.pt_vortex.gammas,
            faceids=self.pt_vortex.face_ids,
            F_verts=self.mesh.F_verts,
            F_area=self.mesh.F_area,
            psi=self.stream_func.psi,
            F_normals=self.mesh.F_normal,
            F_adj=self.mesh.F_adj,
            V_pos=self.mesh.V_pos,
            self_vel_basis=self.self_vel_basis.vel_basis,
            query_face_id=self.self_vel_basis.query_face_id,
            # outputs
            bary_out=self.bary_out,
            k1=self.k1,
            k2=self.k2,
            k3=self.k3,
            k4=self.k4,
        )

        self.declare_resource(
            self.bary_out,
            deps=(self.stream_func.psi, self.pt_vortex.gammas),
            producer=producer,
        )
