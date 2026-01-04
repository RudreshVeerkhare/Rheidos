from dataclasses import dataclass
from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry

import taichi as ti


@dataclass
class StreamFuncProducerIO:
    omega: ResourceRef[ti.Field]  # (nV, f32)
    n_vortices: ResourceRef[ti.Field]  # (i32)
    vortices_face_ids: ResourceRef[ti.Field]  # (100, i32)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    int_mask: ResourceRef[ti.Field]  # (nV, i32)
    values: ResourceRef[ti.Field]  # (nV, f32)
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
        int_mask: ResourceRef[ti.Field],
        values: ResourceRef[ti.Field],
        u: ResourceRef[ti.Field],
        psi: ResourceRef[ti.Field],
        pin_vertex_id: int = 0,
    ) -> None:
        super().__init__(
            StreamFuncProducerIO(
                omega, n_vortices, vortices_face_ids, F_verts, int_mask, values, u, psi
            )
        )
        self.pin_vertex_id = pin_vertex_id

    @ti.kernel
    def _set_mask(
        self,
        int_mask: ti.template(),
        n_vortices: ti.template(),
        vortices_face_ids: ti.template(),
        F: ti.template(),
    ):
        for vid in range(n_vortices[None]):
            face_id = vortices_face_ids[vid]
            int_mask[F[face_id][0]] = 1
            int_mask[F[face_id][1]] = 1
            int_mask[F[face_id][2]] = 1

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

        int_mask = io.int_mask.peek()
        if int_mask is None or int_mask.shape != (nV,):
            int_mask = ti.field(ti.i32, shape=(nV,))
            io.int_mask.set_buffer(int_mask, bump=False)

        values = io.values.peek()
        if values is None or values.shape != (nV,):
            values = ti.field(ti.f32, shape=(nV,))
            io.values.set_buffer(values, bump=False)

        int_mask.fill(0)
        self._set_mask(int_mask, n_vortices, vortices_face_ids, F)
        int_mask[self.pin_vertex_id] = (
            1  # Dirichlet pin one vertex to remove null space
        )

        # trigger poisson solve
        values.copy_from(omega)
        u = io.u.get()  # triggers poisson solve

        psi.copy_from(u)
        io.psi.commit()
