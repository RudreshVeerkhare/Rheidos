from rheidos.compute import WiredProducer, ResourceRef, out_field, Registry

from dataclasses import dataclass

import taichi as ti


@dataclass
class DECMetricIO:
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    F_area: ResourceRef[ti.Field]  # (nF,)
    E_verts: ResourceRef[ti.Field]  # (nE, vec2i)
    E_opp: ResourceRef[
        ti.Field
    ]  # (nE, vec2i) opposite vertex per adjacent face side (-1 if boundary)
    star0: ResourceRef[ti.Field] = out_field()  # (nV,)
    star1: ResourceRef[ti.Field] = out_field()  # (nE,)


@ti.data_oriented
class DECMetricProducer(WiredProducer[DECMetricIO]):
    """Build diagonal Hodge stars:
    - star0: barycentric dual area per vertex = sum(face_area/3)
    - star1: cotan weights per edge = 0.5*(cot(alpha)+cot(beta)) (boundary: single cot)
    """

    def __init__(
        self,
        V_pos: ResourceRef[ti.Field],
        F_verts: ResourceRef[ti.Field],
        F_area: ResourceRef[ti.Field],
        E_verts: ResourceRef[ti.Field],
        E_opp: ResourceRef[ti.Field],
        star0: ResourceRef[ti.Field],
        star1: ResourceRef[ti.Field],
    ) -> None:
        super().__init__(
            DECMetricIO(V_pos, F_verts, F_area, E_verts, E_opp, star0, star1)
        )

    @ti.kernel
    def _build_star0(
        self, F_verts: ti.template(), F_area: ti.template(), star0: ti.template()
    ):
        star0.fill(0.0)

        # Accumulate area
        for fid in F_verts:
            a = F_verts[fid][0]
            b = F_verts[fid][1]
            c = F_verts[fid][2]
            w = F_area[fid] / 3.0
            ti.atomic_add(star0[a], w)
            ti.atomic_add(star0[b], w)
            ti.atomic_add(star0[c], w)

    @ti.func
    def _cot_at(
        self,
        xi: ti.types.vector(3, ti.f32),
        xj: ti.types.vector(3, ti.f32),
        xk: ti.types.vector(3, ti.f32),
    ) -> ti.f32:
        u = xi - xk
        v = xj - xk
        cr = ti.math.cross(u, v)
        denom = ti.max(1e-12, ti.sqrt(cr.dot(cr)))
        return u.dot(v) / denom

    @ti.kernel
    def _build_star1(
        self,
        V_pos: ti.template(),
        E_verts: ti.template(),
        E_opp: ti.template(),
        star1: ti.template(),
    ):
        for e in E_verts:
            i = E_verts[e][0]
            j = E_verts[e][1]
            xi = V_pos[i]
            xj = V_pos[j]
            cot_sum = 0.0

            k0 = E_opp[e][0]
            if k0 >= 0:
                cot_sum += self._cot_at(xi, xj, V_pos[k0])

            k1 = E_opp[e][1]
            if k1 >= 0:
                cot_sum += self._cot_at(xi, xj, V_pos[k1])

            star1[e] = 0.5 * cot_sum

    def compute(self, reg: Registry) -> None:
        V = self.io.V_pos.peek()
        F = self.io.F_verts.peek()
        A = self.io.F_area.peek()
        E = self.io.E_verts.peek()
        EO = self.io.E_opp.peek()
        if V is None or F is None or A is None or E is None or EO is None:
            raise RuntimeError(
                "DECMetricProducer: missing one of V_pos/F_verts/F_area/E_verts/E_opp."
            )

        nV = int(V.shape[0])
        nE = int(E.shape[0])

        star0 = self.io.star0.peek()
        star1 = self.io.star1.peek()

        if star0 is None or star0.shape != (nV,):
            star0 = ti.field(dtype=ti.f32, shape=(nV,))
            self.io.star0.set_buffer(star0, bump=False)

        if star1 is None or star1.shape != (nE,):
            star1 = ti.field(dtype=ti.f32, shape=(nE,))
            self.io.star1.set_buffer(star1, bump=False)

        self._build_star0(F, A, star0)
        self._build_star1(V, E, EO, star1)

        self.io.star0.commit()
        self.io.star1.commit()
