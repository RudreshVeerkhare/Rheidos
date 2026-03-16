from dataclasses import dataclass
from typing import Any, Callable

import taichi as ti

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


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------


@dataclass
class VelAdvectionRK4ProducerIO:
    # inputs
    dt: ResourceRef[float]
    bary: ResourceRef[ti.Field]  # (maxV, vec3f)
    vortex_gammas: ResourceRef[
        ti.Field
    ]  # (maxV,)  (kept for deps; not used directly here)
    psi: ResourceRef[ti.Field]  # (nV,) vertex scalar potential/stream function phi
    faceids: ResourceRef[ti.Field]  # (maxV,)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    F_area: ResourceRef[ti.Field]  # (nF,)
    F_normals: ResourceRef[ti.Field]  # (nF, vec3f) (not necessarily unit)
    F_adj: ResourceRef[ti.Field]  # (nF, vec3i)
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)

    # optional/unused in this variant (kept to match your wiring)
    self_vel_basis: ResourceRef[ti.Field]
    query_face_id: ResourceRef[ti.Field]

    # outputs
    bary_out: ResourceRef[ti.Field] = out_field()  # (maxV, vec3f)
    pos_out: ResourceRef[ti.Field] = out_field()  # (maxV, vec3f) world positions

    # RK4 buffers (WORLD velocities; vec3f)
    k1: ResourceRef[ti.Field] = out_field()
    k2: ResourceRef[ti.Field] = out_field()
    k3: ResourceRef[ti.Field] = out_field()
    k4: ResourceRef[ti.Field] = out_field()


# -----------------------------------------------------------------------------
# Producer
# -----------------------------------------------------------------------------


@ti.data_oriented
class VelAdvectionRK4Producer(WiredProducer[VelAdvectionRK4ProducerIO]):

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

    # ---------------------------------------------------------
    # Facewise constant velocity from u = n x grad(phi)
    # where phi is linear over the triangle.
    # ---------------------------------------------------------
    @ti.kernel
    def _vortex_vel_from_phi(
        self,
        faceids: ti.template(),  # (maxV,)
        F_verts: ti.template(),  # (nF, vec3i)
        F_area: ti.template(),  # (nF,)
        F_normals: ti.template(),  # (nF, vec3f)
        V_pos: ti.template(),  # (nV, vec3f)
        phi: ti.template(),  # (nV,)
        vel_out: ti.template(),  # (maxV, vec3f)
    ):
        for vid in faceids:
            fid = faceids[vid]
            i0 = F_verts[fid][0]
            i1 = F_verts[fid][1]
            i2 = F_verts[fid][2]

            a = V_pos[i0]
            b = V_pos[i1]
            c = V_pos[i2]

            p0 = phi[i0]
            p1 = phi[i1]
            p2 = phi[i2]

            A = F_area[fid]
            n = F_normals[fid]
            nl = ti.sqrt(n.dot(n))

            if nl <= 1e-20 or A <= 1e-20:
                vel_out[vid] = ti.math.vec3(0.0, 0.0, 0.0)
                continue

            n = n / nl
            inv2A = 1.0 / (2.0 * A)

            # grad barycentric (constant on face)
            g0 = ti.math.cross(n, (c - b)) * inv2A
            g1 = ti.math.cross(n, (a - c)) * inv2A
            g2 = ti.math.cross(n, (b - a)) * inv2A

            grad_phi = p0 * g0 + p1 * g1 + p2 * g2

            # u = n x grad(phi)
            vel_out[vid] = ti.math.cross(n, grad_phi)

            # If your orientation makes flow go the wrong way, flip:
            # vel_out[vid] = -vel_out[vid]

    # ---------------------------------------------------------
    # Convert bary -> world position (for output / caching x0)
    # ---------------------------------------------------------
    @ti.kernel
    def _bary_to_pos(
        self,
        bary: ti.template(),  # (maxV, vec3f)
        faceids: ti.template(),  # (maxV,)
        F_verts: ti.template(),  # (nF, vec3i)
        V_pos: ti.template(),  # (nV, vec3f)
        pos_out: ti.template(),  # (maxV, vec3f)
    ):
        for vid in bary:
            fid = faceids[vid]
            i0 = F_verts[fid][0]
            i1 = F_verts[fid][1]
            i2 = F_verts[fid][2]
            a = V_pos[i0]
            b = V_pos[i1]
            c = V_pos[i2]
            bc = bary[vid]
            pos_out[vid] = a * bc[0] + b * bc[1] + c * bc[2]

    # ---------------------------------------------------------
    # Stage step:
    #  - world x from bary_in (current face)
    #  - x += scale * vel
    #  - bary_out = barycentric(x) in the SAME face (may be outside)
    # Face crossing is handled by _update_face_ids afterwards.
    # ---------------------------------------------------------
    @ti.kernel
    def _step_bary_via_world_vel(
        self,
        bary_in: ti.template(),  # (maxV, vec3f)
        bary_out: ti.template(),  # (maxV, vec3f)
        faceids: ti.template(),  # (maxV,)
        F_verts: ti.template(),  # (nF, vec3i)
        V_pos: ti.template(),  # (nV, vec3f)
        scale: ti.f32,
        vel: ti.template(),  # (maxV, vec3f)
    ):
        for vid in bary_in:
            fid = faceids[vid]
            i0 = F_verts[fid][0]
            i1 = F_verts[fid][1]
            i2 = F_verts[fid][2]
            a = V_pos[i0]
            b = V_pos[i1]
            c = V_pos[i2]

            bc = bary_in[vid]
            x = a * bc[0] + b * bc[1] + c * bc[2]
            x = x + scale * vel[vid]
            bary_out[vid] = self._barycentric(x, a, b, c)

    # ---------------------------------------------------------
    # Final RK4 combination in WORLD space:
    # pos = pos0 + dt * (k1 + 2k2 + 2k3 + k4)/6
    # (in-place update of pos field)
    # ---------------------------------------------------------
    @ti.kernel
    def _rk4_pos_inplace(
        self,
        pos: ti.template(),  # (maxV, vec3f) contains pos0 on entry
        dt: ti.f32,
        k1: ti.template(),
        k2: ti.template(),
        k3: ti.template(),
        k4: ti.template(),
    ):
        for vid in pos:
            v = (k1[vid] + 2.0 * k2[vid] + 2.0 * k3[vid] + k4[vid]) / 6.0
            pos[vid] = pos[vid] + dt * v

    # ---------------------------------------------------------
    # Bary from current pos using current faceids (single-face)
    # Then _update_face_ids will walk across edges if needed.
    # ---------------------------------------------------------
    @ti.kernel
    def _bary_from_pos_in_face(
        self,
        pos: ti.template(),  # (maxV, vec3f)
        faceids: ti.template(),  # (maxV,)
        F_verts: ti.template(),
        V_pos: ti.template(),
        bary_out: ti.template(),  # (maxV, vec3f)
    ):
        for vid in pos:
            fid = faceids[vid]
            i0 = F_verts[fid][0]
            i1 = F_verts[fid][1]
            i2 = F_verts[fid][2]
            a = V_pos[i0]
            b = V_pos[i1]
            c = V_pos[i2]
            bary_out[vid] = self._barycentric(pos[vid], a, b, c)

    # ---------------------------------------------------------
    # Your existing face-walk (single hop):
    # If bary has a negative component, hop across the corresponding edge,
    # hinge-rotate the world point, and recompute bary on neighbor face.
    # ---------------------------------------------------------
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
            assert faceid >= 0, f"face doesn't exists: {faceid}"

            bary = barys[vid]

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

            # world-space point from (possibly invalid) bary
            x = (
                V_pos[p1] * bary[v1]
                + V_pos[p2] * bary[v2]
                + V_pos[p_opp] * bary[min_idx]
            )

            # hinge-rotate around shared edge
            axis = V_pos[p2] - V_pos[p1]
            axis_len2 = axis.dot(axis)
            if axis_len2 > 1e-20:
                axis = axis / ti.sqrt(axis_len2)
                n0 = F_normals[faceid]
                n1 = F_normals[opp_face_id]
                n0_len = ti.sqrt(n0.dot(n0))
                n1_len = ti.sqrt(n1.dot(n1))
                if n0_len > 1e-20 and n1_len > 1e-20:
                    n0 = n0 / n0_len
                    n1 = n1 / n1_len
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

            # barycentric on neighbor face
            a = V_pos[F_verts[opp_face_id][0]]
            b = V_pos[F_verts[opp_face_id][1]]
            c = V_pos[F_verts[opp_face_id][2]]

            nbary = self._barycentric(x, a, b, c)

            barys[vid] = nbary
            faceids[vid] = opp_face_id

    # ---------------------------------------------------------
    # Compute
    # ---------------------------------------------------------
    def compute(self, reg: Registry) -> None:
        inputs = self.require_inputs()
        outputs = self.ensure_outputs(reg)

        bary = inputs["bary"].get()
        dt = inputs["dt"].get()
        faceids = inputs["faceids"].get()

        F_verts = inputs["F_verts"].get()
        F_area = inputs["F_area"].get()
        F_normals = inputs["F_normals"].get()
        F_adj = inputs["F_adj"].get()
        V_pos = inputs["V_pos"].get()

        # outputs / scratch
        bary_out = outputs["bary_out"].peek()  # used as bary0 cache
        pos_out = outputs["pos_out"].peek()  # used as pos0 cache, then final output

        k1 = outputs["k1"].peek()
        k2 = outputs["k2"].peek()
        k3 = outputs["k3"].peek()
        k4 = outputs["k4"].peek()

        # Cache initial bary (bary0) and initial world position (pos0)
        bary_out.copy_from(bary)
        self._bary_to_pos(bary_out, faceids, F_verts, V_pos, pos_out)  # pos_out = pos0

        # ---------------------------
        # Stage 1: k1 = u(phi) on current faceids
        # ---------------------------
        phi = inputs["psi"].get()
        self._vortex_vel_from_phi(faceids, F_verts, F_area, F_normals, V_pos, phi, k1)

        # ---------------------------
        # Stage 2: bary = Advect(bary0, dt/2, k1) + face-walk
        # ---------------------------
        self._step_bary_via_world_vel(
            bary_out, bary, faceids, F_verts, V_pos, dt / 2.0, k1
        )
        # walk (do a couple hops to be safer)
        self._update_face_ids(bary, faceids, F_verts, F_normals, F_adj, V_pos)
        self._update_face_ids(bary, faceids, F_verts, F_normals, F_adj, V_pos)
        inputs["bary"].bump()
        inputs["faceids"].bump()

        phi = inputs["psi"].get()
        self._vortex_vel_from_phi(faceids, F_verts, F_area, F_normals, V_pos, phi, k2)

        # ---------------------------
        # Stage 3
        # ---------------------------
        self._step_bary_via_world_vel(
            bary_out, bary, faceids, F_verts, V_pos, dt / 2.0, k2
        )
        self._update_face_ids(bary, faceids, F_verts, F_normals, F_adj, V_pos)
        self._update_face_ids(bary, faceids, F_verts, F_normals, F_adj, V_pos)
        inputs["bary"].bump()
        inputs["faceids"].bump()

        phi = inputs["psi"].get()
        self._vortex_vel_from_phi(faceids, F_verts, F_area, F_normals, V_pos, phi, k3)

        # ---------------------------
        # Stage 4
        # ---------------------------
        self._step_bary_via_world_vel(bary_out, bary, faceids, F_verts, V_pos, dt, k3)
        self._update_face_ids(bary, faceids, F_verts, F_normals, F_adj, V_pos)
        self._update_face_ids(bary, faceids, F_verts, F_normals, F_adj, V_pos)
        inputs["bary"].bump()
        inputs["faceids"].bump()

        phi = inputs["psi"].get()
        self._vortex_vel_from_phi(faceids, F_verts, F_area, F_normals, V_pos, phi, k4)

        # ---------------------------
        # Final: RK4 combine in WORLD, then bary from pos, then face-walk,
        # then recompute pos from final bary+face (so hinge-rotation is reflected)
        # ---------------------------
        self._rk4_pos_inplace(pos_out, dt, k1, k2, k3, k4)  # pos_out now holds pos1

        # Convert pos1 to bary in current face guess, then walk across edges
        self._bary_from_pos_in_face(pos_out, faceids, F_verts, V_pos, bary_out)
        self._update_face_ids(bary_out, faceids, F_verts, F_normals, F_adj, V_pos)
        self._update_face_ids(bary_out, faceids, F_verts, F_normals, F_adj, V_pos)
        self._update_face_ids(bary_out, faceids, F_verts, F_normals, F_adj, V_pos)

        # Final pos output consistent with hinge-walked bary/face
        self._bary_to_pos(bary_out, faceids, F_verts, V_pos, pos_out)

        # Write back state (so next frame starts from the real final state)
        bary.copy_from(bary_out)
        inputs["bary"].bump()
        inputs["faceids"].bump()

        # Commit outputs
        self.io.bary_out.commit()
        self.io.pos_out.commit()


# -----------------------------------------------------------------------------
# Module
# -----------------------------------------------------------------------------


class VelAdvectionModule(ModuleBase):
    NAME = "VelAdvectionModule"

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

        self.pos_out = self.resource(
            "pos_out",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.pt_vortex.face_ids),
                lanes=3,
                allow_none=True,
            ),
            doc="Updated world-space positions after RK4 advection. Shape: (maxV, vec3f)",
            declare=False,
        )

        # RK4 buffers (WORLD velocities)
        self.k1 = self.resource(
            "k1",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_of(self.pt_vortex.bary),
                lanes=3,
                allow_none=True,
            ),
            doc="k1 buffer (world velocity) for step 1",
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
            doc="k2 buffer (world velocity) for step 2",
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
            doc="k3 buffer (world velocity) for step 3",
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
            doc="k4 buffer (world velocity) for step 4",
            declare=True,
        )

        producer = VelAdvectionRK4Producer(
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
            # outputs:
            bary_out=self.bary_out,
            pos_out=self.pos_out,
            k1=self.k1,
            k2=self.k2,
            k3=self.k3,
            k4=self.k4,
        )

        # bary_out depends on psi (which depends on bary/faceids + gammas), but keep explicit deps you had.
        self.declare_resource(
            self.bary_out,
            deps=(self.stream_func.psi, self.pt_vortex.gammas),
            producer=producer,
        )
        self.declare_resource(
            self.pos_out,
            deps=(self.stream_func.psi, self.pt_vortex.gammas),
            producer=producer,
        )
