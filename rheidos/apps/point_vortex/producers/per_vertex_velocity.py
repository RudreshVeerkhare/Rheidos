from dataclasses import dataclass

import numpy as np
import taichi as ti

from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry


@dataclass
class PerVertexVelProducerIO:
    V_incident: ResourceRef[ti.Field]  # (nV, i32)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    F_velocity: ResourceRef[ti.Field]  # (nF, vec3f)

    V_velocity: ResourceRef[ti.Field] = out_field()  # (nV, vec3f)


@ti.data_oriented
class PerVertexVelProducer(WiredProducer[PerVertexVelProducerIO]):

    def __init__(
        self,
        V_incident: ResourceRef[ti.Field],
        F_verts: ResourceRef[ti.Field],
        F_velocity: ResourceRef[ti.Field],
        V_velocity: ResourceRef[ti.Field],
    ) -> None:
        super().__init__(
            PerVertexVelProducerIO(V_incident, F_verts, F_velocity, V_velocity)
        )

    @ti.kernel
    def _split_velocity_per_vertex(
        self,
        V_incident: ti.template(),
        F_verts: ti.template(),
        F_velocity: ti.template(),
        V_velocity: ti.template(),
    ):
        for vid in V_velocity:
            V_velocity[vid] = ti.Vector([0.0, 0.0, 0.0])

        # Accumulate velocity per vertex
        for fid in F_verts:
            face = F_verts[fid]
            face_vel = F_velocity[fid]

            v1 = face[0]
            v2 = face[1]
            v3 = face[2]

            ti.atomic_add(V_velocity[v1], face_vel)
            ti.atomic_add(V_velocity[v2], face_vel)
            ti.atomic_add(V_velocity[v3], face_vel)

        # Calculate average
        for vid in V_incident:
            v_count = V_incident[vid]
            V_velocity[vid] = V_velocity[vid] / (ti.f32(v_count))

    def compute(self, reg: Registry) -> None:
        io = self.io
        V_incident = io.V_incident.get()
        F_verts = io.F_verts.get()
        F_velocity = io.F_velocity.get()

        if V_incident is None or F_verts is None or F_velocity is None:
            raise RuntimeError(
                "PerVertexVelProducer missing either of V_incident/F_verts/F_velocity"
            )

        V_vel = io.V_velocity.peek()
        if V_vel is None or V_vel.shape != V_incident.shape:
            V_vel = ti.Vector.field(3, dtype=ti.f32, shape=V_incident.shape)
            io.V_velocity.set_buffer(V_vel, bump=False)

        self._split_velocity_per_vertex(V_incident, F_verts, F_velocity, V_vel)

        io.V_velocity.commit()
