from rheidos.compute import (
    ModuleBase,
    World,
    ResourceSpec,
    ResourceRef,
    shape_of,
    shape_from_scalar,
    WiredProducer,
    out_field,
)
from rheidos.compute.registry import Registry


from ..point_vortex import PointVortexModule
from ..stream_func import StreamFunctionModule
from ..surface_mesh import SurfaceMeshModule

import taichi as ti
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class BaryAdvectionRK4ProducerIO:
    dt: ResourceRef[float]
    bary: ResourceRef[ti.Field]
    psi: ResourceRef[ti.Field]
    faceids: ResourceRef[ti.Field]
    F_verts: ResourceRef[ti.Field]
    F_area: ResourceRef[ti.Field]
    F_normals: ResourceRef[ti.Field]
    F_adj: ResourceRef[ti.Field]
    V_pos: ResourceRef[ti.Field]

    bary_out: ResourceRef[ti.Field] = out_field()
    # face_ids_out: ResourceRef[ti.Field] = out_field()
    k1: ResourceRef[ti.Field] = out_field()
    k2: ResourceRef[ti.Field] = out_field()
    k3: ResourceRef[ti.Field] = out_field()
    k4: ResourceRef[ti.Field] = out_field()


@ti.data_oriented
class BaryAdvectionRK4Producer(WiredProducer[BaryAdvectionRK4ProducerIO]):

    @ti.func
    def _barycentric(
        self,
        p,
        a,
        b,
        c,
    ):
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

    @ti.kernel
    def _bary_dot(
        self,
        faceids: ti.template(),
        face_areas: ti.template(),
        f_verts: ti.template(),
        psi: ti.template(),
        bary_dot: ti.template(),
    ):
        for vid in faceids:
            faceid = faceids[vid]
            v1 = f_verts[faceid][0]
            v2 = f_verts[faceid][1]
            v3 = f_verts[faceid][2]

            p1 = psi[v1]
            p2 = psi[v2]
            p3 = psi[v3]

            A = face_areas[faceid]

            bary_dot[vid][0] = (p3 - p2) / (2 * A)
            bary_dot[vid][1] = (p1 - p3) / (2 * A)
            bary_dot[vid][2] = (p2 - p1) / (2 * A)

    # RK4 wants a function of type => dy/dt = F(x, t)

    # Buffers for k1, k2, k3, k4 and step function callback

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
    def _project_sum1(self, bary: ti.template()):
        for i in bary:
            s = bary[i][0] + bary[i][1] + bary[i][2]
            bary[i] = bary[i] / s

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
                bary[vid] + dt * (k1[vid] + 2 * k2[vid] + 2 * k3[vid] + k4[vid]) / 6
            )

    @ti.func
    def bary_cords(
        self,
        x1: ti.types.vector(3, dtype=ti.f32),
        x2: ti.types.vector(3, dtype=ti.f32),
        A: ti.f32,
    ) -> ti.f32:
        return ti.math.length(ti.math.cross(x1, x2)) / A

    @ti.kernel
    def _update_face_ids(
        self,
        barys: ti.template(),  # (maxV, vec3f)
        faceids: ti.template(),  # (maxV,)
        F_verts: ti.template(),  # (nF, vec3i)
        F_normals: ti.template(),  # (nF, vec3f)
        F_adj: ti.template(),  # (nF, vec3i) adjacency opposite each vertex index
        V_pos: ti.template(),  # (nV, vec3f)
    ):
        for vid in barys:
            faceid = faceids[vid]
            assert faceid >= 0, f"face doesn't exists: {faceid}"

            bary = barys[vid]

            # Find most-negative bary component (edge crossed when min < 0)
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

            # Neighbor face across edge opposite min_idx
            opp_face_id = F_adj[faceid][min_idx]
            assert opp_face_id >= 0, f"Opposite face doesn't exists: {opp_face_id}"

            # Vertices of current face
            p_opp = F_verts[faceid][min_idx]  # vertex opposite the crossed edge

            # Indices of the other two vertices (edge we crossed is (p1,p2))
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

            # Current world-space point from (possibly slightly invalid) bary
            x = (
                V_pos[p1] * bary[v1]
                + V_pos[p2] * bary[v2]
                + V_pos[p_opp] * bary[min_idx]
            )

            # Rotate point around shared edge to bring it into the neighbor face plane
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

            # Recompute barycentric coords on neighbor face using rotated world point
            a = V_pos[F_verts[opp_face_id][0]]
            b = V_pos[F_verts[opp_face_id][1]]
            c = V_pos[F_verts[opp_face_id][2]]

            nbary = self._barycentric(x, a, b, c)

            # Commit updates
            barys[vid] = nbary
            faceids[vid] = opp_face_id

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

        bary_out = outputs["bary_out"].peek()
        k1 = outputs["k1"].peek()
        k2 = outputs["k2"].peek()
        k3 = outputs["k3"].peek()
        k4 = outputs["k4"].peek()

        bary_out.copy_from(
            bary
        )  # for first 4 steps use bary_out as cache for bary_start
        # cause we need to update the bary_in to retrigger the poisson solve

        # step 1
        psi = inputs["psi"].get()  # refresh psi
        self._bary_dot(faceids, F_area, F_verts, psi, k1)

        # step 2
        self._step_y(bary_out, bary, dt / 2, k1)
        # self._project_sum1(bary_out)
        inputs["bary"].bump()
        psi = inputs["psi"].get()
        self._bary_dot(faceids, F_area, F_verts, psi, k2)

        # step 3
        self._step_y(bary_out, bary, dt / 2, k2)
        # self._project_sum1(bary_out)
        inputs["bary"].bump()
        psi = inputs["psi"].get()
        self._bary_dot(faceids, F_area, F_verts, psi, k3)

        # step 4
        self._step_y(bary_out, bary, dt, k3)
        # self._project_sum1(bary_out)
        inputs["bary"].bump()
        psi = inputs["psi"].get()
        self._bary_dot(faceids, F_area, F_verts, psi, k4)

        # final step
        self._final_step(bary_out, dt, k1, k2, k3, k4)
        # self._project_sum1(bary_out)

        # update faceid if vortex crosses an edge

        # 1. Add overflow check
        self._update_face_ids(bary_out, faceids, F_verts, F_normals, F_adj, V_pos)

        self.io.bary_out.commit()


class BaryAdvectionModule(ModuleBase):
    NAME = "BaryAdvectionModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule)
        self.pt_vortex = world.require(PointVortexModule)
        self.stream_func = world.require(StreamFunctionModule)

        self.dt = self.resource(
            "dt",
            spec=ResourceSpec(kind="python", dtype=float),
            doc="Timestep size in seconds.",
            declare=True,
        )

        self.face_ids_out = self.resource(
            "face_ids_out",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                shape_fn=shape_of(self.pt_vortex.face_ids),
                allow_none=True,
            ),
            doc="Updated face ids after RK4 advection. Shape: (maxV,) i32",
            declare=False,
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

        # Scratch buffers for RK4

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
            faceids=self.pt_vortex.face_ids,
            F_verts=self.mesh.F_verts,
            F_area=self.mesh.F_area,
            psi=self.stream_func.psi,
            F_normals=self.mesh.F_normal,
            F_adj=self.mesh.F_adj,
            V_pos=self.mesh.V_pos,
            bary_out=self.bary_out,
            # face_ids_out=self.face_ids_out,
            k1=self.k1,
            k2=self.k2,
            k3=self.k3,
            k4=self.k4,
        )

        self.declare_resource(
            self.bary_out, deps=(self.stream_func.psi,), producer=producer
        )
