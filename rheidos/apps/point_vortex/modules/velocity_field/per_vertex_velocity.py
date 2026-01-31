from dataclasses import dataclass

import numpy as np
import taichi as ti

from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry


@dataclass
class PerVertexVelProducerIO:
    V_incident_count: ResourceRef[ti.Field]  # (nV, i32)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    F_velocity: ResourceRef[ti.Field]  # (nF, vec3f)

    V_velocity: ResourceRef[ti.Field] = out_field()  # (nV, vec3f)


@ti.data_oriented
class PerVertexVelProducer(WiredProducer[PerVertexVelProducerIO]):
    @ti.kernel
    def _split_velocity_per_vertex(
        self,
        V_incident_count: ti.template(),
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
        for vid in V_incident_count:
            v_count = V_incident_count[vid]
            V_velocity[vid] = V_velocity[vid] / (ti.f32(v_count))

    def compute(self, reg: Registry) -> None:
        io = self.io
        inputs = self.require_inputs()
        V_incident_count = inputs["V_incident_count"].get()
        F_verts = inputs["F_verts"].get()
        F_velocity = inputs["F_velocity"].get()

        V_vel = self.ensure_outputs(reg)["V_velocity"].peek()

        self._split_velocity_per_vertex(V_incident_count, F_verts, F_velocity, V_vel)

        io.V_velocity.commit()
