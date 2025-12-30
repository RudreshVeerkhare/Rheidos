from rheidos.compute.registry import Registry
from rheidos.compute.resource import ResourceSpec
from rheidos.compute.world import World, ModuleBase

import taichi as ti

class MeshModule(ModuleBase):
    NAME = "mesh"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.V_pos = self.resource(
            "V_pos",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.f32, lanes=3, allow_none=True),
            doc="Vertex positions (nV, vec3f)",
            declare=True,
            buffer=None,
            description="Vertex positions",
        )
        self.F_verts = self.resource(
            "F_verts",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.i32, lanes=3, allow_none=True),
            doc="Triangle indices (nF, vec3i)",
            declare=True,
            buffer=None,
            description="Face indices",
        )

        # Derived topology resources (declare after wiring producer)
        self.E_verts = self.resource(
            "E_verts",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.i32, lanes=2, allow_none=True),
            doc="Unique edges (nE, vec2i)",
            declare=False,
        )
        self.E_faces = self.resource(
            "E_faces",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.i32, lanes=2, allow_none=True),
            doc="Adjacent faces per edge (nE, vec2i)",
            declare=False,
        )
        self.E_opp = self.resource(
            "E_opp",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.i32, lanes=2, allow_none=True),
            doc="Opposite vertex per edge-side (nE, vec2i)",
            declare=False,
        )

        topo = BuildUniqueEdges(BuildUniqueEdgesIO.from_mesh(self))
        deps = (self.F_verts.name,)

        self.declare_resource(self.E_verts, buffer=None, deps=deps, producer=topo, description="Unique edges")
        self.declare_resource(self.E_faces, buffer=None, deps=deps, producer=topo, description="Edge adjacent faces")
        self.declare_resource(self.E_opp, buffer=None, deps=deps, producer=topo, description="Edge opposite vertices")


#################################
# PRODUCERS
#################################


from dataclasses import dataclass
from rheidos.compute.resource import ResourceRef
from rheidos.compute.wiring import out_field

@dataclass
class BuildUniqueEdgesIO:
    F_verts: ResourceRef[ti.Field]
    E_verts: ResourceRef[ti.Field] = out_field()
    E_faces: ResourceRef[ti.Field] = out_field()
    E_opp: ResourceRef[ti.Field] = out_field()


    @classmethod
    def from_mesh(cls, mesh: "MeshModule") -> "BuildUniqueEdgesIO":
        return cls(
            F_verts=mesh.F_verts,
            E_verts=mesh.E_verts,
            E_faces=mesh.E_faces,
            E_opp=mesh.E_opp,
        )
    
from rheidos.compute.wiring import WiredProducer
import numpy as np
from typing import Dict, Tuple, List, Any

@ti.data_oriented
class BuildUniqueEdges(WiredProducer[BuildUniqueEdgesIO]):
    """
    Build a UNIQUE undirected edge list from triangle faces.
    """

    def __init__(self, io: BuildUniqueEdgesIO) -> None:
        super().__init__(io)

    def compute(self, reg: Registry) -> None:
        io = self.io
        F = io.F_verts.peek()
        if F is None:
            raise RuntimeError("F_verts not set.")

        F_np = F.to_numpy().astype(np.int32)  # (nF,3)

        edge_map: Dict[Tuple[int, int], int] = {}
        E_verts: List[Tuple[int, int]] = []
        E_faces: List[List[int]] = []
        E_opp: List[List[int]] = []

        def add_halfedge(a: int, b: int, f: int, k: int) -> None:
            i, j = (a, b) if a < b else (b, a)
            key = (i, j)
            if key not in edge_map:
                eid = len(E_verts)
                edge_map[key] = eid
                E_verts.append((i, j))
                E_faces.append([f, -1])
                E_opp.append([k, -1])
            else:
                eid = edge_map[key]
                if E_faces[eid][1] != -1:
                    raise RuntimeError(f"Non-manifold edge detected at {key} (more than 2 faces).")
                E_faces[eid][1] = f
                E_opp[eid][1] = k

        for f in range(F_np.shape[0]):
            a, b, c = int(F_np[f, 0]), int(F_np[f, 1]), int(F_np[f, 2])
            add_halfedge(a, b, f, c)
            add_halfedge(b, c, f, a)
            add_halfedge(c, a, f, b)

        E_verts_np = np.array(E_verts, dtype=np.int32)  # (nE,2)
        E_faces_np = np.array(E_faces, dtype=np.int32)  # (nE,2)
        E_opp_np = np.array(E_opp, dtype=np.int32)      # (nE,2)
        nE = E_verts_np.shape[0]

        E = io.E_verts.peek()
        EF = io.E_faces.peek()
        EO = io.E_opp.peek()

        needs_alloc = (
            E is None or EF is None or EO is None
            or E.shape != (nE,)
            or EF.shape != (nE,)
            or EO.shape != (nE,)
        )
        if needs_alloc:
            E = ti.Vector.field(2, dtype=ti.i32, shape=(nE,))
            EF = ti.Vector.field(2, dtype=ti.i32, shape=(nE,))
            EO = ti.Vector.field(2, dtype=ti.i32, shape=(nE,))
            # Allocation-before-fill: store buffer without bump
            io.E_verts.set_buffer(E, bump=False)
            io.E_faces.set_buffer(EF, bump=False)
            io.E_opp.set_buffer(EO, bump=False)

        E.from_numpy(E_verts_np)
        EF.from_numpy(E_faces_np)
        EO.from_numpy(E_opp_np)

        io.E_verts.commit()
        io.E_faces.commit()
        io.E_opp.commit()


### 
### DEC Producer

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