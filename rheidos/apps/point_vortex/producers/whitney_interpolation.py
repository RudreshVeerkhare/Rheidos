from rheidos.compute import WiredProducer, out_field, ResourceRef, Registry

from dataclasses import dataclass

import taichi as ti
import numpy as np


@dataclass
class VelocityProducerIO:

    face_ids: ResourceRef[np.ndarray]  # (N, i32)
    barys: ResourceRef[np.ndarray]  # (N, vec3f)
    edge_1form: ResourceRef[ti.Field]  # (nE, f32)
    F_edges: ResourceRef[ti.Field]  # (nF,vec3i)
    F_edge_sign: ResourceRef[ti.Field]  # (nF, vec3i)
    grad_l0: ResourceRef[ti.Field]  # (nF, vec3f)
    grad_l1: ResourceRef[ti.Field]  # (nF, vec3f)
    grad_l2: ResourceRef[ti.Field]  # (nF, vec3f)

    vel: ResourceRef[np.ndarray] = out_field()  # (N, vec3f)


class VelocityProducer(WiredProducer[VelocityProducerIO]):

    def __init__(
        self,
        face_ids: ResourceRef[np.ndarray],
        barys: ResourceRef[np.ndarray],
        edge_1form: ResourceRef[ti.Field],
        F_edges: ResourceRef[ti.Field],
        F_edge_sign: ResourceRef[ti.Field],
        grad_l0: ResourceRef[ti.Field],
        grad_l1: ResourceRef[ti.Field],
        grad_l2: ResourceRef[ti.Field],
        vel: ResourceRef[np.ndarray],
    ) -> None:
        super().__init__(
            VelocityProducerIO(
                face_ids,
                barys,
                edge_1form,
                F_edges,
                F_edge_sign,
                grad_l0,
                grad_l1,
                grad_l2,
                vel,
            )
        )

    @ti.kernel
    def whitney_vec_at(
        self,
        face_ids: ti.types.ndarray(dtype=ti.i32, ndim=1),
        barys: ti.types.ndarray(dtype=ti.types.vector(3, ti.f32), ndim=1),
        edge_1form: ti.template(),
        F_edges: ti.template(),
        F_edge_sign: ti.template(),
        grad_l0: ti.template(),
        grad_l1: ti.template(),
        grad_l2: ti.template(),
        values: ti.types.ndarray(
            dtype=ti.types.vector(3, ti.f32), ndim=1
        ),  # this array is inplace written by kernel
    ):
        for i in face_ids:
            fid = face_ids[i]

            li = barys[0]
            lj = barys[1]
            lk = barys[2]

            gi = grad_l0[fid]
            gj = grad_l1[fid]
            gk = grad_l2[fid]

            # Whitney basis vectors
            w_ij = li * gj - lj * gi
            w_jk = lj * gk - lk * gj
            w_ki = lk * gi - li * gk

            e0 = F_edges[fid][0]  # (i->j)
            e1 = F_edges[fid][1]  # (j->k)
            e2 = F_edges[fid][2]  # (k->i)

            s0 = ti.cast(F_edge_sign[fid][0], ti.f32)
            s1 = ti.cast(F_edge_sign[fid][1], ti.f32)
            s2 = ti.cast(F_edge_sign[fid][2], ti.f32)

            u0 = s0 * edge_1form[e0]
            u1 = s1 * edge_1form[e1]
            u2 = s2 * edge_1form[e2]

            values[i] = u0 * w_ij + u1 * w_jk + u2 * w_ki

    def compute(self, reg: Registry) -> None:
        io = self.io
        face_ids = io.face_ids.get()
        barys = io.barys.get()
        edge_1form = io.edge_1form.get()
        F_edges = io.F_edges.get()
        F_edge_sign = io.F_edge_sign.get()
        grad_l0 = io.grad_l0.get()
        grad_l1 = io.grad_l1.get()
        grad_l2 = io.grad_l2.get()

        if any(
            map(
                lambda x: x is None,
                (
                    face_ids,
                    barys,
                    edge_1form,
                    F_edges,
                    F_edge_sign,
                    grad_l0,
                    grad_l1,
                    grad_l2,
                ),
            )
        ):
            raise RuntimeError(
                "VelocityProducer missing one or more of face_ids/barys/edge_1form/F_edges/F_edge_sign/grad_l0/grad_l1/grad_l2"
            )

        values = io.vel.get()
        if values is None or values.shape != face_ids.shape:
            values = np.zeros(face_ids.shape)
            io.vel.set_buffer(values, bump=False)

        self.whitney_vec_at(
            face_ids,
            barys,
            edge_1form,
            F_edges,
            F_edge_sign,
            grad_l0,
            grad_l1,
            grad_l2,
            values,
        )

        io.vel.commit()
