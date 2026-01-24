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
from typing import Any

import taichi as ti

from rheidos.compute import (
    ModuleBase,
    World,
    ResourceRef,
    ResourceSpec,
    WiredProducer,
    Registry,
    out_field,
    shape_of,
)
from ..surface_mesh import SurfaceMeshModule
from ..point_vortex import PointVortexModule
from ..velocity_field import VelocityFieldModule
from .sample_vel import SampleVortexVelProducer
from .advect_const_vel import AdvectConstVelEventDrivenProducer


@dataclass
class AdvectVorticesRT0RK4IO:
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    F_adj: ResourceRef[ti.Field]  # (nF, vec3i)
    vel_FV: ResourceRef[ti.Field]  # (nF, 3) vec3f

    n_vortices: ResourceRef[ti.Field]  # scalar i32
    face_ids: ResourceRef[ti.Field]  # (maxV, i32)
    bary: ResourceRef[ti.Field]  # (maxV, vec3f)

    dt: ResourceRef[Any]  # scalar f32

    face_ids_out: ResourceRef[ti.Field] = out_field()
    bary_out: ResourceRef[ti.Field] = out_field()
    pos_out: ResourceRef[ti.Field] = out_field()


@ti.data_oriented
class AdvectVorticesRT0RK4Producer(WiredProducer[AdvectVorticesRT0RK4IO]):
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
        vel_FV = inputs["vel_FV"].get()
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

        self._ensure_tmp_fields(face_ids.shape)

        # Backup p0
        self.adv.backup_state(face_ids, bary, self._p0_face, self._p0_bary, n)

        # --- k1 @ p0 ---
        self.sampler.run(
            vel_FV=vel_FV,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self._k1,
            n_pts=n,
        )

        # --- k2 @ p0 + 0.5dt*k1 ---
        self.adv.advect_inplace(
            V_pos, F_verts, F_adj, face_ids, bary, self._k1, 0.5 * dt, n
        )
        self._touch_vortex_state()
        vel_FV = io.vel_FV.get()
        self.sampler.run(
            vel_FV=vel_FV,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self._k2,
            n_pts=n,
        )
        self.adv.restore_state(face_ids, bary, self._p0_face, self._p0_bary, n)

        # --- k3 @ p0 + 0.5dt*k2 ---
        self.adv.advect_inplace(
            V_pos, F_verts, F_adj, face_ids, bary, self._k2, 0.5 * dt, n
        )
        self._touch_vortex_state()
        vel_FV = io.vel_FV.get()
        self.sampler.run(
            vel_FV=vel_FV,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self._k3,
            n_pts=n,
        )
        self.adv.restore_state(face_ids, bary, self._p0_face, self._p0_bary, n)

        # --- k4 @ p0 + dt*k3 ---
        self.adv.advect_inplace(V_pos, F_verts, F_adj, face_ids, bary, self._k3, dt, n)
        self._touch_vortex_state()
        vel_FV = io.vel_FV.get()
        self.sampler.run(
            vel_FV=vel_FV,
            pt_bary=bary,
            pt_face=face_ids,
            pt_vel_out=self._k4,
            n_pts=n,
        )
        self.adv.restore_state(face_ids, bary, self._p0_face, self._p0_bary, n)

        # combine + final step (commit)
        self.adv.rk4_combine_vel(self._k1, self._k2, self._k3, self._k4, self._k, n)
        self.adv.advect_inplace(V_pos, F_verts, F_adj, face_ids, bary, self._k, dt, n)
        self._touch_vortex_state()

        self.adv.backup_state(face_ids, bary, face_ids_out, bary_out, n)
        self._write_pos_out(V_pos, F_verts, face_ids, bary, pos_out, n)

        io.face_ids_out.commit()
        io.bary_out.commit()
        io.pos_out.commit()


# -----------------------------------------------------------------------------
# RK4 Module (WiredProducer-based)
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

        rk4_producer = AdvectVorticesRT0RK4Producer(
            V_pos=self.mesh.V_pos,
            F_verts=self.mesh.F_verts,
            F_adj=self.mesh.F_adj,
            vel_FV=self.velocity.FV_velocity,
            n_vortices=self.pt_vortex.n_vortices,
            face_ids=self.pt_vortex.face_ids,
            bary=self.pt_vortex.bary,
            dt=self.dt,
            face_ids_out=self.face_ids_out,
            bary_out=self.bary_out,
            pos_out=self.pos_out,
        )

        deps = (
            self.mesh.V_pos,
            self.mesh.F_verts,
            self.mesh.F_adj,
            self.velocity.FV_velocity,
            self.pt_vortex.n_vortices,
            self.pt_vortex.face_ids,
            self.pt_vortex.bary,
            self.dt,
        )

        self.declare_resource(self.face_ids_out, deps=deps, producer=rk4_producer)
        self.declare_resource(self.bary_out, deps=deps, producer=rk4_producer)
        self.declare_resource(self.pos_out, deps=deps, producer=rk4_producer)

    def advect(self, dt: float) -> None:
        self.dt.set(float(dt))
        self.face_ids_out.get()
