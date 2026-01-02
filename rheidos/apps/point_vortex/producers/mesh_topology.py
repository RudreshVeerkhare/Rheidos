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
    F_edges: ResourceRef[ti.Field] = out_field()  # (nF, vec3i)
    F_edge_sign: ResourceRef[ti.Field] = out_field()  # (nF, vec3i)


# NOTE: F_edge_sign is defined relative to your canonical global edge orientation (E_verts stores (min,max)).
# That’s exactly what you want for consistent d1 signs. This does not fix inconsistent face winding. If your
# mesh has flipped triangles, d1 will reflect that (which is correct mathematically, but often not what you
# intend visually/physically).


@ti.data_oriented
class TopologyProducer(WiredProducer[TopologyProducerIO]):
    def __init__(
        self,
        F_verts: ResourceRef[ti.Field],
        E_verts: ResourceRef[ti.Field],
        E_faces: ResourceRef[ti.Field],
        E_opp: ResourceRef[ti.Field],
        F_edges: ResourceRef[ti.Field],
        F_edge_sign: ResourceRef[ti.Field],
    ) -> None:
        io = TopologyProducerIO(F_verts, E_verts, E_faces, E_opp, F_edges, F_edge_sign)
        super().__init__(io)

    def compute(self, reg: Registry) -> None:
        io = self.io
        F = io.F_verts.peek()
        if F is None:
            raise RuntimeError("F_verts not set.")

        F_np = F.to_numpy().astype(np.int32)  # (nF,3)
        nF = int(F_np.shape[0])

        edge_map: Dict[Tuple[int, int], int] = {}
        E_verts_list: List[Tuple[int, int]] = []
        E_faces_list: List[List[int]] = []
        E_opp_list: List[List[int]] = []

        def add_halfedge(a: int, b: int, f: int, k: int) -> None:
            i, j = (a, b) if a < b else (b, a)
            key = (i, j)
            if key not in edge_map:
                eid = len(E_verts_list)
                edge_map[key] = eid
                E_verts_list.append((i, j))
                E_faces_list.append([f, -1])
                E_opp_list.append([k, -1])
            else:
                eid = edge_map[key]
                if E_faces_list[eid][1] != -1:
                    raise RuntimeError(
                        f"Non-manifold edge detected at {key} (more than 2 faces)."
                    )
                E_faces_list[eid][1] = f
                E_opp_list[eid][1] = k

        # Pass 1: build unique edges + adjacency
        for f in range(nF):
            a, b, c = int(F_np[f, 0]), int(F_np[f, 1]), int(F_np[f, 2])
            add_halfedge(a, b, f, c)
            add_halfedge(b, c, f, a)
            add_halfedge(c, a, f, b)

        # Convert to numpy
        E_verts_np = np.array(E_verts_list, dtype=np.int32)  # (nE,2)
        E_faces_np = np.array(E_faces_list, dtype=np.int32)  # (nE,2)
        E_opp_np = np.array(E_opp_list, dtype=np.int32)  # (nE,2)
        nE = int(E_verts_np.shape[0])

        # Pass 2: build face->edge incidence + sign
        # For each face (a,b,c), boundary edges are (a->b),(b->c),(c->a).
        F_edges_np = np.empty((nF, 3), dtype=np.int32)
        F_sign_np = np.empty((nF, 3), dtype=np.int32)

        def eid_and_sign(u: int, v: int) -> Tuple[int, int]:
            # canonical edge key uses sorted endpoints
            i, j = (u, v) if u < v else (v, u)
            eid = edge_map[(i, j)]
            # sign is +1 if directed (u->v) matches canonical (i->j)
            sign = +1 if (u == i and v == j) else -1
            return eid, sign

        for f in range(nF):
            a, b, c = int(F_np[f, 0]), int(F_np[f, 1]), int(F_np[f, 2])
            e0, s0 = eid_and_sign(a, b)
            e1, s1 = eid_and_sign(b, c)
            e2, s2 = eid_and_sign(c, a)
            F_edges_np[f, 0], F_sign_np[f, 0] = e0, s0
            F_edges_np[f, 1], F_sign_np[f, 1] = e1, s1
            F_edges_np[f, 2], F_sign_np[f, 2] = e2, s2

        # Grab existing buffers (if any)
        E = io.E_verts.peek()
        EF = io.E_faces.peek()
        EO = io.E_opp.peek()
        FE = io.F_edges.peek()
        FS = io.F_edge_sign.peek()

        needs_alloc = (
            E is None
            or EF is None
            or EO is None
            or FE is None
            or FS is None
            or E.shape != (nE,)
            or EF.shape != (nE,)
            or EO.shape != (nE,)
            or FE.shape != (nF,)
            or FS.shape != (nF,)
        )
        if needs_alloc:
            E = ti.Vector.field(2, dtype=ti.i32, shape=(nE,))
            EF = ti.Vector.field(2, dtype=ti.i32, shape=(nE,))
            EO = ti.Vector.field(2, dtype=ti.i32, shape=(nE,))
            FE = ti.Vector.field(3, dtype=ti.i32, shape=(nF,))
            FS = ti.Vector.field(3, dtype=ti.i32, shape=(nF,))

            io.E_verts.set_buffer(E, bump=False)
            io.E_faces.set_buffer(EF, bump=False)
            io.E_opp.set_buffer(EO, bump=False)
            io.F_edges.set_buffer(FE, bump=False)
            io.F_edge_sign.set_buffer(FS, bump=False)

        # Fill
        E.from_numpy(E_verts_np)
        EF.from_numpy(E_faces_np)
        EO.from_numpy(E_opp_np)
        FE.from_numpy(F_edges_np)
        FS.from_numpy(F_sign_np)

        # Commit (mark fresh)
        io.E_verts.commit()
        io.E_faces.commit()
        io.E_opp.commit()
        io.F_edges.commit()
        io.F_edge_sign.commit()
