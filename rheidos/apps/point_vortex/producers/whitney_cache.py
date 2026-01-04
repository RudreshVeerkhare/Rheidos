from dataclasses import dataclass
import taichi as ti

from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry


@dataclass
class FaceWhitneyCacheProducerIO:
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    F_edge_sign: ResourceRef[ti.Field]  # (nF, vec3i)
    F_normal: ResourceRef[ti.Field]  # (nF, vec3f)
    F_area: ResourceRef[ti.Field]  # (nF, f32)

    grad_l0: ResourceRef[ti.Field] = out_field()  # (nF, vec3f) $ \nabla \lambda_i $
    grad_l1: ResourceRef[ti.Field] = out_field()  # (nF, vec3f) $ \nabla \lambda_j $
    grad_l2: ResourceRef[ti.Field] = out_field()  # (nF, vec3f) $ \nabla \lambda_k $


@ti.data_oriented
class FaceWhitneyCacheProducer(WiredProducer[FaceWhitneyCacheProducerIO]):
    """Builds per-face quantities needed for Whitney 1-form reconstruction"""

    def __init__(
        self,
        V_pos: ResourceRef[ti.Field],
        F_verts: ResourceRef[ti.Field],
        F_edge_sign: ResourceRef[ti.Field],
        F_normal: ResourceRef[ti.Field],
        F_area: ResourceRef[ti.Field],
        grad_l0: ResourceRef[ti.Field],
        grad_l1: ResourceRef[ti.Field],
        grad_l2: ResourceRef[ti.Field],
    ) -> None:
        super().__init__(
            FaceWhitneyCacheProducerIO(
                V_pos,
                F_verts,
                F_edge_sign,
                F_normal,
                F_area,
                grad_l0,
                grad_l1,
                grad_l2,
            )
        )

    @ti.kernel
    def _compute_bary_gradient(
        self,
        V: ti.template(),
        F: ti.template(),
        F_normal: ti.template(),
        F_area: ti.template(),
        grad0: ti.template(),
        grad1: ti.template(),
        grad2: ti.template(),
    ):
        esp = 1e-20
        for fid in F:
            i = F[fid][0]
            j = F[fid][1]
            k = F[fid][2]

            xi = V[i]
            xj = V[j]
            xk = V[k]

            n_hat = F_normal[fid]
            twice_area = 2 * F_area[fid]

            grad0[fid] = (
                ti.math.cross(n_hat, xk - xj) / twice_area
            )  # $ \nabla \lambda_i $
            grad1[fid] = (
                ti.math.cross(n_hat, xi - xk) / twice_area
            )  # $ \nabla \lambda_j $
            grad2[fid] = (
                ti.math.cross(n_hat, xj - xi) / twice_area
            )  # $ \nabla \lambda_k $

    def compute(self, reg: Registry) -> None:
        io = self.io
        V = io.V_pos.get()
        F = io.F_verts.get()
        F_normal = io.F_normal.get()
        F_area = io.F_area.get()

        if V is None or F is None or F_normal is None or F_area is None:
            raise RuntimeError(
                "FaceWhitneyCacheProducer missing one or more of V_pos/F_verts/F_normal/F_area"
            )

        nF = F.shape[0]

        grad0 = io.grad_l0.peek()
        if grad0 is None or grad0.shape != (nF,):
            grad0 = ti.Vector.field(3, ti.f32, shape=(nF,))
            io.grad_l0.set_buffer(grad0, bump=False)

        grad1 = io.grad_l1.peek()
        if grad1 is None or grad1.shape != (nF,):
            grad1 = ti.Vector.field(3, ti.f32, shape=(nF,))
            io.grad_l1.set_buffer(grad1, bump=False)

        grad2 = io.grad_l2.peek()
        if grad2 is None or grad2.shape != (nF,):
            grad2 = ti.Vector.field(3, ti.f32, shape=(nF,))
            io.grad_l2.set_buffer(grad2, bump=False)

        io.grad_l0.commit()
        io.grad_l1.commit()
        io.grad_l2.commit()
