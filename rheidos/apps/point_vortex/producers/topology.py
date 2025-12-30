from rheidos.compute import WiredProducer, ResourceRef, Registry, out_field
from typing import Dict, List, Tuple
from dataclasses import dataclass
import taichi as ti
import numpy as np


@dataclass
class TopologyProducerIO:
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    E_verts: ResourceRef[ti.Field] = out_field()  # (nE, vec2i)
    E_faces: ResourceRef[ti.Field] = out_field()  # (nE, vec2i)
    E_opp: ResourceRef[ti.Field] = out_field()  # (nE, vec2i)


@ti.data_oriented
class TopologyProducer(WiredProducer[TopologyProducerIO]):

    def __init__(
        self,
        F_verts: ResourceRef[ti.Field],
        E_verts: ResourceRef[ti.Field],
        E_faces: ResourceRef[ti.Field],
        E_opp: ResourceRef[ti.Field],
    ) -> None:
        io = TopologyProducerIO(F_verts, E_verts, E_faces, E_opp)
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
                    raise RuntimeError(
                        f"Non-manifold edge detected at {key} (more than 2 faces)."
                    )
                E_faces[eid][1] = f
                E_opp[eid][1] = k

        for f in range(F_np.shape[0]):
            a, b, c = int(F_np[f, 0]), int(F_np[f, 1]), int(F_np[f, 2])
            add_halfedge(a, b, f, c)
            add_halfedge(b, c, f, a)
            add_halfedge(c, a, f, b)

        E_verts_np = np.array(E_verts, dtype=np.int32)  # (nE,2)
        E_faces_np = np.array(E_faces, dtype=np.int32)  # (nE,2)
        E_opp_np = np.array(E_opp, dtype=np.int32)  # (nE,2)
        nE = E_verts_np.shape[0]

        E = io.E_verts.peek()
        EF = io.E_faces.peek()
        EO = io.E_opp.peek()

        needs_alloc = (
            E is None
            or EF is None
            or EO is None
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
