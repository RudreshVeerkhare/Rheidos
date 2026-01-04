from dataclasses import dataclass
from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry

import taichi as ti


@dataclass
class SplatPtVortexProducerIO:
    n_vortices: ResourceRef[ti.Field]  # i32
    gammas: ResourceRef[ti.Field]  # (100, f32)
    face_ids: ResourceRef[ti.Field]  # (100, i32)
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    bary: ResourceRef[ti.Field]  # (100, vec3f)
    omega: ResourceRef[ti.Field] = out_field()  # (nV, f32)


@ti.data_oriented
class SplatPtVortexProducer(WiredProducer[SplatPtVortexProducerIO]):

    def __init__(
        self,
        n_vortices: ResourceRef[ti.Field],
        gammas: ResourceRef[ti.Field],
        face_ids: ResourceRef[ti.Field],
        V_pos: ResourceRef[ti.Field],
        F_verts: ResourceRef[ti.Field],
        bary: ResourceRef[ti.Field],
        omega: ResourceRef[ti.Field],
    ) -> None:
        super().__init__(
            SplatPtVortexProducerIO(
                n_vortices, gammas, face_ids, V_pos, F_verts, bary, omega
            )
        )

    @ti.kernel
    def _splat_gamma_over_vertices_using_bary(
        self,
        n_vortices: ti.template(),  # scalar field, use [None]
        gammas: ti.template(),
        face_ids: ti.template(),
        F_verts: ti.template(),
        bary: ti.template(),
        omega: ti.template(),
    ):
        # Clear omega
        for v in omega:
            omega[v] = 0.0

        for k in range(n_vortices[None]):
            fid = face_ids[k]
            gamma = gammas[k]
            bc = bary[k]  # vec3f
            fv = F_verts[fid]  # vec3i

            ti.atomic_add(omega[fv[0]], gamma * bc[0])
            ti.atomic_add(omega[fv[1]], gamma * bc[1])
            ti.atomic_add(omega[fv[2]], gamma * bc[2])

    # TODO: Hide this repeated broiler plate for check and ensuring buffers existence and shape
    # We already have `out_field` so anyway we only create and update the out_field which is easy to
    # encapsulate
    def compute(self, reg: Registry) -> None:
        io = self.io
        V_pos = io.V_pos.get()
        F_verts = io.F_verts.get()
        N = io.n_vortices.get()
        gammas = io.gammas.get()
        face_ids = io.face_ids.get()
        bary = io.bary.get()

        if (
            (V_pos is None)
            or (F_verts is None)
            or (N is None)
            or (gammas is None)
            or (face_ids is None)
            or (bary is None)
        ):
            raise RuntimeError(
                "SplatPtVortexProducer is missing one or more of V_pos/F_verts/n_vortices/gammas/face_ids/bary"
            )
        nV = V_pos.shape[0]
        omega = io.omega.peek()
        if omega is None or omega.shape != (nV,):
            omega = ti.field(dtype=ti.f32, shape=(nV,))
            io.omega.set_buffer(omega, bump=False)

        self._splat_gamma_over_vertices_using_bary(
            N, gammas, face_ids, F_verts, bary, omega
        )

        io.omega.commit()
