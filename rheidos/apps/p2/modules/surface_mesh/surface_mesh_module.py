from __future__ import annotations

import numpy as np

from rheidos.compute import (
    ModuleBase,
    ProducerContext,
    ResourceSpec,
    World,
    producer,
    shape_from_scalar,
    shape_map,
)

from .mesh_geometry import build_face_geometry
from .mesh_topology import build_mesh_topology


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
            doc="Set of unique edges between vertices. Shape: (nE,2)",
        )
        self.E_faces = self.resource(
            "E_faces",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_from_scalar(self.n_edges, tail=(2,)),
                allow_none=True,
            ),
            doc="Adjacent faces per edge. Shape: (nE,2)",
        )
        self.E_opp = self.resource(
            "E_opp",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_from_scalar(self.n_edges, tail=(2,)),
                allow_none=True,
            ),
            doc="Opposite vertex per edge side of adjacent triangles. Shape: (nE,2)",
        )
        self.F_edges = self.resource(
            "F_edges",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_map(self.F_verts, lambda shape: (shape[0], 3)),
                allow_none=True,
            ),
            doc="Edge id per face directed edge (a->b, b->c, c->a). Shape: (nF,3)",
        )
        self.F_edge_sign = self.resource(
            "F_edge_sign",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_map(self.F_verts, lambda shape: (shape[0], 3)),
                allow_none=True,
            ),
            doc="Edge orientation sign per face directed edge. Shape: (nF,3)",
        )
        self.F_adj = self.resource(
            "F_adj",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_map(self.F_verts, lambda shape: (shape[0], 3)),
                allow_none=True,
            ),
            doc="Per-face adjacency across opposite edges. Shape: (nF,3)",
        )
        self.V_incident_count = self.resource(
            "V_incident_count",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_map(self.V_pos, lambda shape: (shape[0],)),
                allow_none=True,
            ),
            doc="Count of faces incident on each vertex. Shape: (nV,)",
        )
        self.V_incident = self.resource(
            "V_incident",
            spec=ResourceSpec(kind="python", dtype=dict, allow_none=True),
            doc="Python dict mapping vertex id to incident face ids.",
        )
        self.boundary_edge_count = self.resource(
            "boundary_edge_count",
            spec=ResourceSpec(kind="python", dtype=int, allow_none=True),
            doc="Count of boundary edges in the mesh.",
        )

        self.F_area = self.resource(
            "F_area",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.F_verts, lambda shape: (shape[0],)),
                allow_none=True,
            ),
            doc="Scalar area per triangle face. Shape: (nF,)",
        )
        self.F_normal = self.resource(
            "F_normal",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.F_verts, lambda shape: (shape[0], 3)),
                allow_none=True,
            ),
            doc="Unit normal per triangle face. Shape: (nF,3)",
        )
        self.bind_producers()

    def set_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.V_pos.set(np.ascontiguousarray(vertices, dtype=np.float64))
        self.F_verts.set(np.ascontiguousarray(faces, dtype=np.int32))

    @producer(
        inputs=("V_pos", "F_verts"),
        outputs=(
            "n_edges",
            "E_verts",
            "E_faces",
            "E_opp",
            "F_edges",
            "F_edge_sign",
            "F_adj",
            "V_incident_count",
            "V_incident",
            "boundary_edge_count",
        ),
    )
    def build_topology(self, ctx: ProducerContext) -> None:
        outputs = build_mesh_topology(
            ctx.inputs.V_pos.get(),
            ctx.inputs.F_verts.get(),
        )
        (
            n_edges,
            e_verts,
            e_faces,
            e_opp,
            f_edges,
            f_edge_sign,
            f_adj,
            v_incident_count,
            v_incident,
            boundary_edge_count,
        ) = outputs
        ctx.commit(
            n_edges=int(n_edges),
            E_verts=e_verts,
            E_faces=e_faces,
            E_opp=e_opp,
            F_edges=f_edges,
            F_edge_sign=f_edge_sign,
            F_adj=f_adj,
            V_incident_count=v_incident_count,
            V_incident=v_incident,
            boundary_edge_count=int(boundary_edge_count),
        )

    @producer(
        inputs=("V_pos", "F_verts"),
        outputs=("F_area", "F_normal"),
    )
    def build_geometry(self, ctx: ProducerContext) -> None:
        f_area, f_normal = build_face_geometry(
            ctx.inputs.V_pos.get(),
            ctx.inputs.F_verts.get(),
        )
        ctx.commit(F_area=f_area, F_normal=f_normal)
