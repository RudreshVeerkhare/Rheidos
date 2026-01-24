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
        inputs = self.require_inputs()
        V = inputs["V_pos"].get()
        F = inputs["F_verts"].get()
        A = inputs["F_area"].get()
        E = inputs["E_verts"].get()
        EO = inputs["E_opp"].get()

        outputs = self.ensure_outputs(reg)
        star0 = outputs["star0"].peek()
        star1 = outputs["star1"].peek()

        self._build_star0(F, A, star0)
        self._build_star1(V, E, EO, star1)

        self.io.star0.commit()
        self.io.star1.commit()
