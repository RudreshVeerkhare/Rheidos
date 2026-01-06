from rheidos.compute import WiredProducer, ResourceRef, Registry, out_field
from typing import Dict, List, Tuple
from dataclasses import dataclass
import taichi as ti
import numpy as np


@dataclass
class TopologyProducerIO:
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    V_pos: ResourceRef[
        ti.Field
    ]  # (nV, vec3f) # Doesn't depent on this but just need this to get nV

    E_verts: ResourceRef[ti.Field] = out_field()  # (nE, vec2i) canonical (min,max)
    E_faces: ResourceRef[ti.Field] = (
        out_field()
    )  # (nE, vec2i) incident faces (f0,f1) (-1 if boundary)
    E_opp: ResourceRef[ti.Field] = (
        out_field()
    )  # (nE, vec2i) "opposite vertex id" per incident face
    F_edges: ResourceRef[ti.Field] = (
        out_field()
    )  # (nF, vec3i) edges for (a->b),(b->c),(c->a)
    F_edge_sign: ResourceRef[ti.Field] = (
        out_field()
    )  # (nF, vec3i) +1 if face edge matches (min->max) else -1
    F_adj: ResourceRef[ti.Field] = (
        out_field()
    )  # (nF, vec3i) per-face adjacency across the edge opposite each vertex
    V_incident: ResourceRef[ti.Field] = (
        out_field()
    )  # (nV, i32) Count of faces a vertex is incident on.


@ti.data_oriented
class TopologyProducer(WiredProducer[TopologyProducerIO]):
    def __init__(
        self,
        F_verts: ResourceRef[ti.Field],
        V_pos: ResourceRef[ti.Field],
        E_verts: ResourceRef[ti.Field],
        E_faces: ResourceRef[ti.Field],
        E_opp: ResourceRef[ti.Field],
        F_edges: ResourceRef[ti.Field],
        F_edge_sign: ResourceRef[ti.Field],
        F_adj: ResourceRef[ti.Field],
        V_incident: ResourceRef[ti.Field],
    ) -> None:
        io = TopologyProducerIO(
            F_verts,
            V_pos,
            E_verts,
            E_faces,
            E_opp,
            F_edges,
            F_edge_sign,
            F_adj,
            V_incident,
        )
        super().__init__(io)

    def compute(self, reg: Registry) -> None:
        io = self.io

        F = io.F_verts.peek()
        if F is None:
            raise RuntimeError("F_verts not set.")

        Vpos = io.V_pos.peek()
        if Vpos is None:
            raise RuntimeError("V_pos not set (needed to determine nV).")

        # Taichi vector field has shape (nV,)
        if len(Vpos.shape) != 1:
            raise RuntimeError(f"V_pos must have shape (nV,); got {Vpos.shape}")
        nV = int(Vpos.shape[0])

        F_np = F.to_numpy().astype(np.int32)  # (nF,3)
        if F_np.ndim != 2 or F_np.shape[1] != 3:
            raise RuntimeError(f"F_verts must have shape (nF,3); got {F_np.shape}")
        nF = int(F_np.shape[0])

        if nF > 0:
            mn = int(F_np.min())
            mx = int(F_np.max())
            if mn < 0:
                raise RuntimeError("F_verts contains negative vertex indices.")
            if mx >= nV:
                raise RuntimeError(
                    f"F_verts references vertex id {mx} but V_pos has nV={nV}."
                )

        edge_map: Dict[Tuple[int, int], int] = {}
        E_verts_list: List[Tuple[int, int]] = []
        E_faces_list: List[List[int]] = []
        E_opp_list: List[List[int]] = []

        def add_halfedge(a: int, b: int, f: int, opp_vert_id: int) -> None:
            i, j = (a, b) if a < b else (b, a)
            key = (i, j)
            if key not in edge_map:
                eid = len(E_verts_list)
                edge_map[key] = eid
                E_verts_list.append((i, j))
                E_faces_list.append([f, -1])
                E_opp_list.append([opp_vert_id, -1])
            else:
                eid = edge_map[key]
                if E_faces_list[eid][1] != -1:
                    raise RuntimeError(
                        f"Non-manifold edge detected at {key} (more than 2 faces)."
                    )
                E_faces_list[eid][1] = f
                E_opp_list[eid][1] = opp_vert_id

        # --------------------
        # Pass 1: unique edges + incident faces (+ opposite vertex per face)
        # --------------------
        for f in range(nF):
            a, b, c = int(F_np[f, 0]), int(F_np[f, 1]), int(F_np[f, 2])
            add_halfedge(a, b, f, c)
            add_halfedge(b, c, f, a)
            add_halfedge(c, a, f, b)

        E_verts_np = np.array(E_verts_list, dtype=np.int32)  # (nE,2)
        E_faces_np = np.array(E_faces_list, dtype=np.int32)  # (nE,2)
        E_opp_np = np.array(E_opp_list, dtype=np.int32)  # (nE,2)
        nE = int(E_verts_np.shape[0])

        # --------------------
        # Pass 2: face -> edge incidence + sign
        # --------------------
        F_edges_np = np.empty((nF, 3), dtype=np.int32)
        F_sign_np = np.empty((nF, 3), dtype=np.int32)

        def eid_and_sign(u: int, v: int) -> Tuple[int, int]:
            i, j = (u, v) if u < v else (v, u)
            eid = edge_map[(i, j)]
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

        # --------------------
        # Pass 3: per-face adjacency across opposite edges
        # --------------------
        F_adj_np = np.full((nF, 3), -1, dtype=np.int32)

        def other_face(eid: int, f: int) -> int:
            f0, f1 = int(E_faces_np[eid, 0]), int(E_faces_np[eid, 1])
            if f0 == f:
                return f1
            if f1 == f:
                return f0
            raise RuntimeError(
                f"Internal error: face {f} not found in E_faces for eid {eid}: {(f0, f1)}"
            )

        for f in range(nF):
            v0, v1, v2 = int(F_np[f, 0]), int(F_np[f, 1]), int(F_np[f, 2])

            # edge opposite v0 is (v1,v2)
            i, j = (v1, v2) if v1 < v2 else (v2, v1)
            nf0 = other_face(edge_map[(i, j)], f)
            F_adj_np[f, 0] = nf0 if nf0 != -1 else -1

            # edge opposite v1 is (v2,v0)
            i, j = (v2, v0) if v2 < v0 else (v0, v2)
            nf1 = other_face(edge_map[(i, j)], f)
            F_adj_np[f, 1] = nf1 if nf1 != -1 else -1

            # edge opposite v2 is (v0,v1)
            i, j = (v0, v1) if v0 < v1 else (v1, v0)
            nf2 = other_face(edge_map[(i, j)], f)
            F_adj_np[f, 2] = nf2 if nf2 != -1 else -1

        # --------------------
        # Pass 4 (NEW): V_incident from faces, sized by V_pos
        # --------------------
        if nV == 0:
            V_incident_np = np.zeros((0,), dtype=np.int32)
        else:
            flat = F_np.reshape(-1) if nF > 0 else np.zeros((0,), dtype=np.int32)
            V_incident_np = np.bincount(flat, minlength=nV).astype(np.int32)  # (nV,)

        # --------------------
        # Allocate / reuse buffers
        # --------------------
        E = io.E_verts.peek()
        EF = io.E_faces.peek()
        EO = io.E_opp.peek()
        FE = io.F_edges.peek()
        FS = io.F_edge_sign.peek()
        FA = io.F_adj.peek()
        VI = io.V_incident.peek()

        needs_alloc_topo = (
            E is None
            or EF is None
            or EO is None
            or FE is None
            or FS is None
            or FA is None
            or E.shape != (nE,)
            or EF.shape != (nE,)
            or EO.shape != (nE,)
            or FE.shape != (nF,)
            or FS.shape != (nF,)
            or FA.shape != (nF,)
        )
        if needs_alloc_topo:
            E = ti.Vector.field(2, dtype=ti.i32, shape=(nE,))
            EF = ti.Vector.field(2, dtype=ti.i32, shape=(nE,))
            EO = ti.Vector.field(2, dtype=ti.i32, shape=(nE,))
            FE = ti.Vector.field(3, dtype=ti.i32, shape=(nF,))
            FS = ti.Vector.field(3, dtype=ti.i32, shape=(nF,))
            FA = ti.Vector.field(3, dtype=ti.i32, shape=(nF,))

            io.E_verts.set_buffer(E, bump=False)
            io.E_faces.set_buffer(EF, bump=False)
            io.E_opp.set_buffer(EO, bump=False)
            io.F_edges.set_buffer(FE, bump=False)
            io.F_edge_sign.set_buffer(FS, bump=False)
            io.F_adj.set_buffer(FA, bump=False)

        needs_alloc_vi = (VI is None) or (VI.shape != (nV,))
        if needs_alloc_vi:
            VI = ti.field(dtype=ti.i32, shape=(nV,))
            io.V_incident.set_buffer(VI, bump=False)

        # --------------------
        # Fill
        # --------------------
        E.from_numpy(E_verts_np)
        EF.from_numpy(E_faces_np)
        EO.from_numpy(E_opp_np)
        FE.from_numpy(F_edges_np)
        FS.from_numpy(F_sign_np)
        FA.from_numpy(F_adj_np)
        VI.from_numpy(V_incident_np)

        # --------------------
        # Commit
        # --------------------
        io.E_verts.commit()
        io.E_faces.commit()
        io.E_opp.commit()
        io.F_edges.commit()
        io.F_edge_sign.commit()
        io.F_adj.commit()
        io.V_incident.commit()
