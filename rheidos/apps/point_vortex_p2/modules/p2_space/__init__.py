from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rheidos.compute import ModuleBase, ProducerBase, ResourceSpec, World

from ..surface_mesh import SurfaceMeshModule


def build_p2_space_data(
    n_vertices: int,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build global P2 scalar DOF layout for triangle mesh."""
    f = np.ascontiguousarray(faces, dtype=np.int32)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (nF,3), got {f.shape}")

    edge_map: dict[tuple[int, int], int] = {}
    face_to_edges = np.empty((f.shape[0], 3), dtype=np.int32)

    for fid, (a, b, c) in enumerate(f):
        local = ((int(a), int(b)), (int(b), int(c)), (int(c), int(a)))
        for le, (u, v) in enumerate(local):
            key = (u, v) if u < v else (v, u)
            eid = edge_map.get(key)
            if eid is None:
                eid = len(edge_map)
                edge_map[key] = eid
            face_to_edges[fid, le] = eid

    edges = np.array(list(edge_map.keys()), dtype=np.int32)
    n_edges = int(edges.shape[0])

    face_to_dofs = np.empty((f.shape[0], 6), dtype=np.int32)
    face_to_dofs[:, :3] = f
    face_to_dofs[:, 3:] = n_vertices + face_to_edges

    ndof = int(n_vertices + n_edges)
    return edges, face_to_edges, face_to_dofs, ndof


@dataclass
class BuildP2SpaceProducer(ProducerBase):
    V_pos: str
    F_verts: str

    edges: str
    face_to_edges: str
    face_to_dofs: str
    ndof: str

    @property
    def outputs(self):
        return (self.edges, self.face_to_edges, self.face_to_dofs, self.ndof)

    def compute(self, reg) -> None:
        v = reg.read(self.V_pos)
        f = reg.read(self.F_verts)

        n_vertices = int(np.asarray(v).shape[0])
        edges, face_to_edges, face_to_dofs, ndof = build_p2_space_data(n_vertices, f)

        reg.commit(self.edges, buffer=edges)
        reg.commit(self.face_to_edges, buffer=face_to_edges)
        reg.commit(self.face_to_dofs, buffer=face_to_dofs)
        reg.commit(self.ndof, buffer=int(ndof))


class P2ScalarSpaceModule(ModuleBase):
    NAME = "P2ScalarSpace"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule, scope=scope)

        self.edges = self.resource(
            "edges",
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
        )
        self.face_to_edges = self.resource(
            "face_to_edges",
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
        )
        self.face_to_dofs = self.resource(
            "face_to_dofs",
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
        )
        self.ndof = self.resource(
            "ndof",
            spec=ResourceSpec(kind="python", dtype=int, allow_none=True),
        )

        prod = BuildP2SpaceProducer(
            V_pos=self.mesh.V_pos.name,
            F_verts=self.mesh.F_verts.name,
            edges=self.edges.name,
            face_to_edges=self.face_to_edges.name,
            face_to_dofs=self.face_to_dofs.name,
            ndof=self.ndof.name,
        )

        deps = (self.mesh.V_pos, self.mesh.F_verts)
        self.declare_resource(self.edges, deps=deps, producer=prod)
        self.declare_resource(self.face_to_edges, deps=deps, producer=prod)
        self.declare_resource(self.face_to_dofs, deps=deps, producer=prod)
        self.declare_resource(self.ndof, deps=deps, producer=prod)
