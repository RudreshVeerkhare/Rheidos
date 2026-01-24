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
        inputs = self.require_inputs()
        V_pos = inputs["V_pos"].get()
        F_verts = inputs["F_verts"].get()
        N = inputs["n_vortices"].get()
        gammas = inputs["gammas"].get()
        face_ids = inputs["face_ids"].get()
        bary = inputs["bary"].get()

        omega = self.ensure_outputs(reg)["omega"].peek()

        self._splat_gamma_over_vertices_using_bary(
            N, gammas, face_ids, F_verts, bary, omega
        )

        io.omega.commit()
