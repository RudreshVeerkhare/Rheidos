from rheidos.compute import World, ModelBase, Resource, Producer, ResourceRef, Registry
from dataclasses import dataclass
from typing import Any
import taichi as ti

@dataclass
class TiMeshAPI:
    V_pos: ResourceRef[Any]
    F_verts: ResourceRef[Any]
    E_verts: ResourceRef[Any]
    F_edges: ResourceRef[Any]


@ti.data_oriented
class BuildTopology(ProducerBase):
    """
    Demo topology builder: creates 3 directed edges per face (not unique edges).
    Outputs:
      - E_verts: shape (3*nF,), vec2i edge endpoints
      - F_edges: shape (nF,), vec3i indices into E_verts
    """
    def __init__(self, mesh_prefix: str) -> None:
        self.mesh_prefix = mesh_prefix
        self.outputs = (
            f"{mesh_prefix}.E_verts",
            f"{mesh_prefix}.F_edges",
        )

    @ti.kernel
    def _build(self, F_verts: ti.template(), E_verts: ti.template(), F_edges: ti.template()):
        for f in F_verts:
            a = F_verts[f][0]
            b = F_verts[f][1]
            c = F_verts[f][2]

            e0 = 3 * f + 0
            e1 = 3 * f + 1
            e2 = 3 * f + 2

            E_verts[e0] = ti.Vector([a, b])
            E_verts[e1] = ti.Vector([b, c])
            E_verts[e2] = ti.Vector([c, a])

            F_edges[f] = ti.Vector([e0, e1, e2])

    def compute(self, reg: Registry) -> None:
        F = reg.read(f"{self.mesh_prefix}.F_verts", ensure=False)
        if F is None:
            raise RuntimeError("F_verts not set.")

        nF = F.shape[0]
        nE = 3 * nF

        E = reg.read(f"{self.mesh_prefix}.E_verts", ensure=False)
        FE = reg.read(f"{self.mesh_prefix}.F_edges", ensure=False)

        # allocate/reallocate if needed
        needs_alloc = (
            E is None or FE is None
            or E.shape != (nE,)
            or FE.shape != (nF,)
        )
        if needs_alloc:
            E = ti.Vector.field(2, dtype=ti.i32, shape=(nE,))
            FE = ti.Vector.field(3, dtype=ti.i32, shape=(nF,))
            reg.commit(f"{self.mesh_prefix}.E_verts", buffer=E)
            reg.commit(f"{self.mesh_prefix}.F_edges", buffer=FE)

        # fill
        self._build(F, E, FE)

        # mark fresh
        reg.bump(f"{self.mesh_prefix}.E_verts")
        reg.bump(f"{self.mesh_prefix}.F_edges")




class TiMeshModule(ModuleBase):
    NAME = "TiMesh"

    def __init__(self, reg: Registry, *, scope: str = "") -> None:
        super().__init__(reg, scope=scope)

