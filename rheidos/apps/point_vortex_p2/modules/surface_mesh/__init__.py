from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rheidos.compute import (
    ModuleBase,
    ProducerBase,
    ResourceSpec,
    World,
    shape_from_axis,
    shape_from_scalar,
)


def build_mesh_topology_geometry(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build E_verts/E_faces/F_adj/F normals+areas and boundary edge count."""
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"V_pos must have shape (nV,3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"F_verts must have shape (nF,3), got {f.shape}")

    nV = int(v.shape[0])
    if f.size > 0:
        lo = int(f.min())
        hi = int(f.max())
        if lo < 0 or hi >= nV:
            raise ValueError(f"F_verts references [{lo},{hi}] but nV={nV}")

    edge_map: dict[tuple[int, int], int] = {}
    e_verts: list[tuple[int, int]] = []
    e_faces: list[list[int]] = []

    f_adj = np.full((f.shape[0], 3), -1, dtype=np.int32)

    def add_edge(a: int, b: int, fid: int) -> int:
        key = (a, b) if a < b else (b, a)
        eid = edge_map.get(key)
        if eid is None:
            eid = len(e_verts)
            edge_map[key] = eid
            e_verts.append(key)
            e_faces.append([fid, -1])
        else:
            if e_faces[eid][1] != -1:
                raise ValueError(f"Non-manifold edge detected at {key}")
            e_faces[eid][1] = fid
        return eid

    # Build unique edge table with up to 2 incident faces.
    for fid, (a, b, c) in enumerate(f):
        add_edge(int(a), int(b), fid)
        add_edge(int(b), int(c), fid)
        add_edge(int(c), int(a), fid)

    e_verts_np = (
        np.asarray(e_verts, dtype=np.int32)
        if e_verts
        else np.empty((0, 2), dtype=np.int32)
    )
    e_faces_np = (
        np.asarray(e_faces, dtype=np.int32)
        if e_faces
        else np.empty((0, 2), dtype=np.int32)
    )

    # Fill per-face adjacency in barycentric convention: index m -> opposite vertex m edge.
    for fid, (a, b, c) in enumerate(f):
        opp_edges = ((int(b), int(c)), (int(c), int(a)), (int(a), int(b)))
        for m, (u, w) in enumerate(opp_edges):
            key = (u, w) if u < w else (w, u)
            eid = edge_map[key]
            f0, f1 = int(e_faces_np[eid, 0]), int(e_faces_np[eid, 1])
            if f0 == fid:
                f_adj[fid, m] = f1
            elif f1 == fid:
                f_adj[fid, m] = f0
            else:
                raise RuntimeError(f"Internal adjacency error for face {fid}, edge {key}")

    boundary_edge_count = int(np.count_nonzero(e_faces_np[:, 1] < 0))

    f_area = np.zeros((f.shape[0],), dtype=np.float64)
    f_normal = np.zeros((f.shape[0], 3), dtype=np.float64)
    for fid, (i0, i1, i2) in enumerate(f):
        x0 = v[int(i0)]
        x1 = v[int(i1)]
        x2 = v[int(i2)]
        cr = np.cross(x1 - x0, x2 - x0)
        nrm = float(np.linalg.norm(cr))
        f_area[fid] = 0.5 * nrm
        if nrm > 1e-20:
            f_normal[fid] = cr / nrm
        else:
            f_normal[fid] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return (
        int(e_verts_np.shape[0]),
        e_verts_np,
        e_faces_np,
        f_adj,
        f_normal,
        f_area,
        boundary_edge_count,
    )


@dataclass(eq=False)
class MeshTopologyGeometryProducer(ProducerBase):
    V_pos: str
    F_verts: str

    n_edges: str
    E_verts: str
    E_faces: str
    F_adj: str
    F_normal: str
    F_area: str
    boundary_edge_count: str

    @property
    def outputs(self):
        return (
            self.n_edges,
            self.E_verts,
            self.E_faces,
            self.F_adj,
            self.F_normal,
            self.F_area,
            self.boundary_edge_count,
        )

    def compute(self, reg) -> None:
        v = reg.read(self.V_pos)
        f = reg.read(self.F_verts)
        n_edges, e_verts, e_faces, f_adj, f_normal, f_area, bnd = (
            build_mesh_topology_geometry(v, f)
        )

        reg.commit(self.n_edges, buffer=int(n_edges))
        reg.commit(self.E_verts, buffer=e_verts)
        reg.commit(self.E_faces, buffer=e_faces)
        reg.commit(self.F_adj, buffer=f_adj)
        reg.commit(self.F_normal, buffer=f_normal)
        reg.commit(self.F_area, buffer=f_area)
        reg.commit(self.boundary_edge_count, buffer=int(bnd))


class SurfaceMeshModule(ModuleBase):
    NAME = "P2SurfaceMesh"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.V_pos = self.resource(
            "V_pos",
            declare=True,
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
            doc="Mesh vertices (nV,3)",
        )
        self.F_verts = self.resource(
            "F_verts",
            declare=True,
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
            doc="Triangle indices (nF,3)",
        )

        self.n_edges = self.resource(
            "n_edges",
            spec=ResourceSpec(kind="python", dtype=int, allow_none=True),
            doc="Scalar count of unique edges in the mesh.",
        )
        self.E_verts = self.resource(
            "E_verts",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_from_scalar(self.n_edges, tail=(2,)),
                allow_none=True,
            ),
        )
        self.E_faces = self.resource(
            "E_faces",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_from_scalar(self.n_edges, tail=(2,)),
                allow_none=True,
            ),
        )
        self.F_adj = self.resource(
            "F_adj",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_from_axis(self.F_verts, tail=(3,)),
                allow_none=True,
            ),
        )
        self.F_normal = self.resource(
            "F_normal",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_axis(self.F_verts, tail=(3,)),
                allow_none=True,
            ),
        )
        self.F_area = self.resource(
            "F_area",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_axis(self.F_verts),
                allow_none=True,
            ),
        )
        self.boundary_edge_count = self.resource(
            "boundary_edge_count",
            spec=ResourceSpec(kind="python", dtype=int, allow_none=True),
        )

        prod = MeshTopologyGeometryProducer(
            V_pos=self.V_pos.name,
            F_verts=self.F_verts.name,
            n_edges=self.n_edges.name,
            E_verts=self.E_verts.name,
            E_faces=self.E_faces.name,
            F_adj=self.F_adj.name,
            F_normal=self.F_normal.name,
            F_area=self.F_area.name,
            boundary_edge_count=self.boundary_edge_count.name,
        )

        deps = (self.V_pos, self.F_verts)
        self.declare_resource(self.n_edges, deps=deps, producer=prod)
        self.declare_resource(self.E_verts, deps=deps, producer=prod)
        self.declare_resource(self.E_faces, deps=deps, producer=prod)
        self.declare_resource(self.F_adj, deps=deps, producer=prod)
        self.declare_resource(self.F_normal, deps=deps, producer=prod)
        self.declare_resource(self.F_area, deps=deps, producer=prod)
        self.declare_resource(self.boundary_edge_count, deps=deps, producer=prod)

    def set_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.V_pos.set(np.ascontiguousarray(vertices, dtype=np.float64))
        self.F_verts.set(np.ascontiguousarray(faces, dtype=np.int32))
