from rheidos.compute.registry import Registry
from rheidos.compute.resource import ResourceSpec, ResourceRef, ShapeFn, Shape
from rheidos.compute.world import World, ModuleBase
from rheidos.compute.wiring import out_field, WiredProducer
import taichi as ti

from .mesh import MeshModule
from dataclasses import dataclass
from typing import Any, Optional

# TODO: Fix the importing types nightmare 

@dataclass
class BuildDECIO:
    V_pos: ResourceRef[Any]
    F_verts: ResourceRef[Any]
    E_verts: ResourceRef[Any]
    E_opp: ResourceRef[Any]
    star0: ResourceRef[Any] = out_field()
    star1: ResourceRef[Any] = out_field()
    star2: ResourceRef[Any] = out_field()

    @classmethod
    def from_modules(cls, mesh: "MeshModule", dec: "DECModule") -> "BuildDECIO":
        return cls(
            V_pos=mesh.V_pos,
            F_verts=mesh.F_verts,
            E_verts=mesh.E_verts,
            E_opp=mesh.E_opp,
            star0=dec.star0,
            star1=dec.star1,
            star2=dec.star2,
        )

@ti.data_oriented
class BuildDEC(WiredProducer[BuildDECIO]):
    """
    DEC-ish caches for triangle mesh in R^3 (here z=0):
      - star2: face areas
      - star0: vertex barycentric dual areas
      - star1: cotan weights per edge
    """

    def __init__(self, io: BuildDECIO) -> None:
        super().__init__(io)

    @ti.kernel
    def _face_areas(self, V_pos: ti.template(), F_verts: ti.template(), star2: ti.template()):
        for f in F_verts:
            a = F_verts[f][0]
            b = F_verts[f][1]
            c = F_verts[f][2]
            pa = V_pos[a]
            pb = V_pos[b]
            pc = V_pos[c]
            star2[f] = 0.5 * (pb - pa).cross(pc - pa).norm()

    @ti.kernel
    def _vertex_areas_bary(self, F_verts: ti.template(), star2: ti.template(), star0: ti.template()):
        for i in star0:
            star0[i] = 0.0
        for f in F_verts:
            a = F_verts[f][0]
            b = F_verts[f][1]
            c = F_verts[f][2]
            w = star2[f] / 3.0
            ti.atomic_add(star0[a], w)
            ti.atomic_add(star0[b], w)
            ti.atomic_add(star0[c], w)

    @ti.func
    def _cot_angle(
        self,
        pi: ti.types.vector(3, ti.f32),
        pj: ti.types.vector(3, ti.f32),
        pk: ti.types.vector(3, ti.f32),
    ) -> ti.f32:
        u = pi - pk
        v = pj - pk
        denom = u.cross(v).norm() + 1e-12
        return u.dot(v) / denom

    @ti.kernel
    def _cotan_weights(self, V_pos: ti.template(), E_verts: ti.template(), E_opp: ti.template(), star1: ti.template()):
        for e in E_verts:
            i = E_verts[e][0]
            j = E_verts[e][1]
            k0 = E_opp[e][0]
            k1 = E_opp[e][1]

            pi = V_pos[i]
            pj = V_pos[j]

            cot0 = 0.0
            cot1 = 0.0
            if k0 >= 0:
                pk0 = V_pos[k0]
                cot0 = self._cot_angle(pi, pj, pk0)
            if k1 >= 0:
                pk1 = V_pos[k1]
                cot1 = self._cot_angle(pi, pj, pk1)

            star1[e] = 0.5 * (cot0 + cot1)

    def compute(self, reg: Registry) -> None:
        io = self.io
        V = io.V_pos.peek()
        F = io.F_verts.peek()
        E = io.E_verts.peek()
        EO = io.E_opp.peek()
        if V is None or F is None or E is None or EO is None:
            raise RuntimeError("Missing mesh buffers for DEC build.")

        nV = V.shape[0]
        nF = F.shape[0]
        nE = E.shape[0]

        s0 = io.star0.peek()
        s1 = io.star1.peek()
        s2 = io.star2.peek()

        needs_alloc = (
            s0 is None or s1 is None or s2 is None
            or s0.shape != (nV,)
            or s1.shape != (nE,)
            or s2.shape != (nF,)
        )
        if needs_alloc:
            s0 = ti.field(dtype=ti.f32, shape=(nV,))
            s1 = ti.field(dtype=ti.f32, shape=(nE,))
            s2 = ti.field(dtype=ti.f32, shape=(nF,))
            io.star0.set_buffer(s0, bump=False)
            io.star1.set_buffer(s1, bump=False)
            io.star2.set_buffer(s2, bump=False)

        self._face_areas(V, F, s2)
        self._vertex_areas_bary(F, s2, s0)
        self._cotan_weights(V, E, EO, s1)

        io.star0.commit()
        io.star1.commit()
        io.star2.commit()

def _shape_of(ref: ResourceRef[Any]) -> ShapeFn:
    def fn(reg: Registry) -> Optional[Shape]:
        buf = reg.read(ref.name, ensure=False)
        if buf is None or not hasattr(buf, "shape"):
            return None
        return tuple(buf.shape)
    return fn

class DECModule(ModuleBase):
    NAME = "dec"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        mesh = self.require(MeshModule)

        self.star0 = self.resource(
            "star0",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.f32, shape_fn=_shape_of(mesh.V_pos), allow_none=True),
            doc="*0 (vertex dual areas)",
            declare=False,
        )
        self.star1 = self.resource(
            "star1",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.f32, shape_fn=_shape_of(mesh.E_verts), allow_none=True),
            doc="*1 (cotan edge weights)",
            declare=False,
        )
        self.star2 = self.resource(
            "star2",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.f32, shape_fn=_shape_of(mesh.F_verts), allow_none=True),
            doc="*2 (face areas)",
            declare=False,
        )

        dec_builder = BuildDEC(BuildDECIO.from_modules(mesh=mesh, dec=self))
        deps = (mesh.V_pos.name, mesh.F_verts.name, mesh.E_verts.name, mesh.E_opp.name)

        self.declare_resource(self.star0, buffer=None, deps=deps, producer=dec_builder, description="*0")
        self.declare_resource(self.star1, buffer=None, deps=deps, producer=dec_builder, description="*1")
        self.declare_resource(self.star2, buffer=None, deps=deps, producer=dec_builder, description="*2")
