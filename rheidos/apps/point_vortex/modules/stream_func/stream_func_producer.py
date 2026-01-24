from dataclasses import dataclass
from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry

import taichi as ti


@dataclass
class StreamFuncProducerIO:
    omega: ResourceRef[ti.Field]  # (nV, f32)
    n_vortices: ResourceRef[ti.Field]  # (i32)
    vortices_face_ids: ResourceRef[ti.Field]  # (100, i32)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    u: ResourceRef[ti.Field]  # (nV, f32)
    constraint_mask: ResourceRef[ti.Field] = out_field()  # (nV, i32)
    constraint_values: ResourceRef[ti.Field] = out_field()  # (nV, f32)
    rhs: ResourceRef[ti.Field] = out_field()  # (nV, f32)
    psi: ResourceRef[ti.Field] = out_field()  # (nV, f32)


@ti.data_oriented
class StreamFuncProducer(WiredProducer[StreamFuncProducerIO]):
    pin_vertex_id = 0

    @ti.kernel
    def _set_mask(
        self,
        constraint_mask: ti.template(),
        n_vortices: ti.template(),
        vortices_face_ids: ti.template(),
        F: ti.template(),
    ):
        for vid in range(n_vortices[None]):
            face_id = vortices_face_ids[vid]
            constraint_mask[F[face_id][0]] = 1
            constraint_mask[F[face_id][1]] = 1
            constraint_mask[F[face_id][2]] = 1

    def compute(
        self, reg: Registry
    ) -> None:  # TODO: Remove `reg` and out io as parameter
        io = self.io
        inputs = self.require_inputs(ignore=("u",))
        omega = inputs["omega"].get()
        n_vortices = inputs["n_vortices"].get()
        vortices_face_ids = inputs["vortices_face_ids"].get()
        F = inputs["F_verts"].get()

        outputs = self.ensure_outputs(reg)
        constraint_mask = outputs["constraint_mask"].peek()
        constraint_values = outputs["constraint_values"].peek()
        rhs = outputs["rhs"].peek()
        psi = outputs["psi"].peek()

        constraint_mask.fill(0)
        constraint_mask[self.pin_vertex_id] = (
            1  # Dirichlet pin one vertex to remove null space
        )
        constraint_values[self.pin_vertex_id] = (
            0  # Dirichlet pin one vertex to remove null space
        )
        io.constraint_mask.commit()
        io.constraint_values.commit()

        # trigger poisson solve
        rhs.copy_from(omega)
        io.rhs.commit()
        u = io.u.get()  # triggers poisson solve

        psi.copy_from(u)
        io.psi.commit()
