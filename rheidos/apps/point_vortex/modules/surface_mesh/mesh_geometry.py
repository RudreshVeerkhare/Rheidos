from dataclasses import dataclass
from typing import Any

import taichi as ti
from rheidos.compute import WiredProducer, ResourceRef, out_field
from rheidos.compute.registry import Registry


@dataclass
class GeometryProducerIO:
    V_pos: ResourceRef[Any]  # (nV, vec3f)
    F_verts: ResourceRef[Any]  # (nF, vec3i)
    F_area: ResourceRef[Any] = out_field()  # (nF)
    F_normal: ResourceRef[Any] = out_field()  # (nF, vec3f)


@ti.data_oriented
class GeometryProducer(WiredProducer[GeometryProducerIO]):
    @ti.kernel
    def _calculate_face_area_and_normal(
        self,
        V_pos: ti.template(),  # shape (nV,)
        F_verts: ti.template(),  # shape (nF,)
        F_area: ti.template(),  # shape (nF,)
        F_normal: ti.template(),  # shape (nF,)
    ):
        eps = 1e-12
        for f in F_verts:  # iterates over face indices
            v1 = F_verts[f][0]
            v2 = F_verts[f][1]
            v3 = F_verts[f][2]

            x1 = V_pos[v1]
            x2 = V_pos[v2]
            x3 = V_pos[v3]

            area_n = ti.math.cross(x2 - x1, x3 - x1)  # magnitude = 2*area
            twice_area = ti.sqrt(area_n.dot(area_n))

            F_area[f] = 0.5 * twice_area

            inv_len = 1.0 / ti.max(twice_area, eps)
            F_normal[f] = area_n * inv_len  # unit normal (with eps guard)

    def compute(self, reg: Registry) -> None:
        inputs = self.require_inputs()
        V = inputs["V_pos"].get()
        F = inputs["F_verts"].get()

        outputs = self.ensure_outputs(reg)
        F_area = outputs["F_area"].peek()
        F_normal = outputs["F_normal"].peek()

        self._calculate_face_area_and_normal(V, F, F_area, F_normal)

        self.io.F_normal.commit()
        self.io.F_area.commit()
