from dataclasses import dataclass

import taichi as ti
from rheidos.compute import WiredProducer, ResourceRef, out_field
from rheidos.compute.registry import Registry


@dataclass
class VortexWorldPositionProducerIO:
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    n_vortices: ResourceRef[ti.Field]  # ()
    face_ids: ResourceRef[ti.Field]  # (100, )
    bary: ResourceRef[ti.Field]  # (100, vec3f)
    pos_world: ResourceRef[ti.Field] = out_field()  # (100, vec3f)


@ti.data_oriented
class VortexWorldPositionProducer(WiredProducer[VortexWorldPositionProducerIO]):
    @ti.kernel
    def _calculate_world_position(
        self,
        V: ti.template(),
        F: ti.template(),
        N: ti.template(),
        face_ids: ti.template(),
        bary: ti.template(),
        pos_world: ti.template(),
    ):
        for vortex_id in range(N[None]):
            fid = face_ids[vortex_id]
            x1 = V[F[fid][0]]
            x2 = V[F[fid][1]]
            x3 = V[F[fid][2]]

            pos_world[vortex_id] = (
                x1 * bary[vortex_id][0]
                + x2 * bary[vortex_id][1]
                + x3 * bary[vortex_id][2]
            )

    def compute(self, reg: Registry) -> None:
        io = self.io
        inputs = self.require_inputs()

        V = inputs["V_pos"].get()
        F = inputs["F_verts"].get()
        n = inputs["n_vortices"].get()
        face_ids = inputs["face_ids"].get()
        bary = inputs["bary"].get()

        pos_world = self.ensure_outputs(reg)["pos_world"].peek()
        self._calculate_world_position(V, F, n, face_ids, bary, pos_world)

        io.pos_world.commit()
