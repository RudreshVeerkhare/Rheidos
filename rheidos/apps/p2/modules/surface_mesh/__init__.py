from __future__ import annotations

import numpy as np

from rheidos.compute import (
    ModuleBase,
    ResourceSpec,
    World,
    shape_from_axis,
    shape_from_scalar,
)

from .mesh_geometry import GeometryProducer
from .mesh_topology import TopologyProducer


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
                shape_fn=shape_from_axis(self.F_verts, tail=(3,)),
                allow_none=True,
            ),
            doc="Edge id per face directed edge (a->b, b->c, c->a). Shape: (nF,3)",
        )
        self.F_edge_sign = self.resource(
            "F_edge_sign",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_from_axis(self.F_verts, tail=(3,)),
                allow_none=True,
            ),
            doc="Edge orientation sign per face directed edge. Shape: (nF,3)",
        )
        self.F_adj = self.resource(
            "F_adj",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_from_axis(self.F_verts, tail=(3,)),
                allow_none=True,
            ),
            doc="Per-face adjacency across opposite edges. Shape: (nF,3)",
        )
        self.V_incident_count = self.resource(
            "V_incident_count",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
                shape_fn=shape_from_axis(self.V_pos),
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

        topology_producer = TopologyProducer(
            V_pos=self.V_pos,
            F_verts=self.F_verts,
            n_edges=self.n_edges,
            E_verts=self.E_verts,
            E_faces=self.E_faces,
            E_opp=self.E_opp,
            F_edges=self.F_edges,
            F_edge_sign=self.F_edge_sign,
            F_adj=self.F_adj,
            V_incident_count=self.V_incident_count,
            V_incident=self.V_incident,
            boundary_edge_count=self.boundary_edge_count,
        )
        deps = (self.F_verts, self.V_pos)

        self.declare_resource(self.n_edges, deps=deps, producer=topology_producer)
        self.declare_resource(self.E_verts, deps=deps, producer=topology_producer)
        self.declare_resource(self.E_faces, deps=deps, producer=topology_producer)
        self.declare_resource(self.E_opp, deps=deps, producer=topology_producer)
        self.declare_resource(self.F_edges, deps=deps, producer=topology_producer)
        self.declare_resource(self.F_edge_sign, deps=deps, producer=topology_producer)
        self.declare_resource(self.F_adj, deps=deps, producer=topology_producer)
        self.declare_resource(
            self.V_incident_count, deps=deps, producer=topology_producer
        )
        self.declare_resource(self.V_incident, deps=deps, producer=topology_producer)
        self.declare_resource(
            self.boundary_edge_count, deps=deps, producer=topology_producer
        )

        self.F_area = self.resource(
            "F_area",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_axis(self.F_verts),
                allow_none=True,
            ),
            doc="Scalar area per triangle face. Shape: (nF,)",
        )
        self.F_normal = self.resource(
            "F_normal",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_axis(self.F_verts, tail=(3,)),
                allow_none=True,
            ),
            doc="Unit normal per triangle face. Shape: (nF,3)",
        )

        geometry_producer = GeometryProducer(
            V_pos=self.V_pos,
            F_verts=self.F_verts,
            F_normal=self.F_normal,
            F_area=self.F_area,
        )
        self.declare_resource(
            self.F_normal, deps=(self.F_verts, self.V_pos), producer=geometry_producer
        )
        self.declare_resource(
            self.F_area, deps=(self.F_verts, self.V_pos), producer=geometry_producer
        )

    def set_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.V_pos.set(np.ascontiguousarray(vertices, dtype=np.float64))
        self.F_verts.set(np.ascontiguousarray(faces, dtype=np.int32))
