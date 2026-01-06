from dataclasses import dataclass
from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry

import taichi as ti


@dataclass
class StreamFuncProducerIO:
    omega: ResourceRef[ti.Field]  # (nV, f32)
    n_vortices: ResourceRef[ti.Field]  # (i32)
    vortices_face_ids: ResourceRef[ti.Field]  # (100, i32)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    constraint_mask: ResourceRef[ti.Field]  # (nV, i32)
    constraint_values: ResourceRef[ti.Field]  # (nV, f32)
    rhs: ResourceRef[ti.Field]  # (nV, f32)
    u: ResourceRef[ti.Field]  # (nV, f32)
    psi: ResourceRef[ti.Field] = out_field()  # (nV, f32)


@ti.data_oriented
class StreamFuncProducer(WiredProducer[StreamFuncProducerIO]):
    def __init__(
        self,
        omega: ResourceRef[ti.Field],
        n_vortices: ResourceRef[ti.Field],
        vortices_face_ids: ResourceRef[ti.Field],
        F_verts: ResourceRef[ti.Field],
        constraint_mask: ResourceRef[ti.Field],
        constraint_values: ResourceRef[ti.Field],
        rhs: ResourceRef[ti.Field],
        u: ResourceRef[ti.Field],
        psi: ResourceRef[ti.Field],
        pin_vertex_id: int = 0,
    ) -> None:
        super().__init__(
            StreamFuncProducerIO(
                omega,
                n_vortices,
                vortices_face_ids,
                F_verts,
                constraint_mask,
                constraint_values,
                rhs,
                u,
                psi,
            )
        )
        self.pin_vertex_id = pin_vertex_id

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
        omega = io.omega.get()
        n_vortices = io.n_vortices.get()
        vortices_face_ids = io.vortices_face_ids.get()
        F = io.F_verts.get()

        if (
            omega is None
            or n_vortices is None
            or vortices_face_ids is None
            or F is None
        ):
            raise RuntimeError(
                "StreamFuncProducer is missing one or more of omega/n_vortices/vortices_face_ids/F_verts"
            )

        psi = io.psi.peek()
        nV = omega.shape[0]
        if psi is None or psi.shape != (nV,):
            psi = ti.field(ti.f32, shape=(nV,))
            io.psi.set_buffer(psi, bump=False)

        constraint_mask = io.constraint_mask.peek()
        if constraint_mask is None or constraint_mask.shape != (nV,):
            constraint_mask = ti.field(ti.i32, shape=(nV,))
            io.constraint_mask.set_buffer(constraint_mask, bump=False)

        constraint_values = io.constraint_values.peek()
        if constraint_values is None or constraint_values.shape != (nV,):
            constraint_values = ti.field(ti.f32, shape=(nV,))
            io.constraint_values.set_buffer(constraint_values, bump=False)

        rhs = io.rhs.peek()
        if rhs is None or rhs.shape != (nV,):
            rhs = ti.field(ti.f32, shape=(nV,))
            io.rhs.set_buffer(rhs, bump=False)

        constraint_mask.fill(0)
        constraint_mask[self.pin_vertex_id] = (
            1  # Dirichlet pin one vertex to remove null space
        )
        constraint_values[self.pin_vertex_id] = (
            0  # Dirichlet pin one vertex to remove null space
        )

        # trigger poisson solve
        rhs.copy_from(omega)
        u = io.u.get()  # triggers poisson solve

        psi.copy_from(u)
        io.psi.commit()
