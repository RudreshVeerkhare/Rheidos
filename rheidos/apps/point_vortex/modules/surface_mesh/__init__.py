from rheidos.compute import ModuleBase, World, ResourceSpec, shape_of, shape_from_scalar
import taichi as ti

from .mesh_topology import TopologyProducer
from .mesh_geometry import GeometryProducer


class SurfaceMeshModule(ModuleBase):
    NAME = "SurfaceTriangleMesh"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.V_pos = self.resource(
            "V_pos",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.f32, lanes=3, allow_none=True
            ),
            doc="Triangle mesh vertex positions, expected shape: (nV, 3)",
            declare=True,
        )

        self.F_verts = self.resource(
            "F_verts",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.i32, lanes=3, allow_none=True
            ),
            doc="Group of indices of vertex forming faces/triangles. Expected shape: (nF, 3)",
            declare=True,
        )

        # Derived resources to keep track of topology which are lazy initialized based on need

        self.n_edges = self.resource(
            "n_edges",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.i32, shape=(), allow_none=True
            ),
            doc="Scalar count of unique edges in the mesh.",
            declare=False,
        )

        # Topology dependant resources
        self.E_verts = self.resource(
            "E_verts",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                lanes=2,
                shape_fn=shape_from_scalar(self.n_edges),
                allow_none=True,
            ),
            doc="Set of unique edges between vertices. Expected Shape: (nE, 2)",
        )

        self.E_faces = self.resource(
            "E_faces",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                lanes=2,
                shape_fn=shape_from_scalar(self.n_edges),
                allow_none=True,
            ),
            doc="Adjacent faces per edge. Shape: (nE, vec2i)",
        )

        self.E_opp = self.resource(
            "E_opp",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                lanes=2,
                shape_fn=shape_from_scalar(self.n_edges),
                allow_none=True,
            ),
            doc="Opposite vertex per edge side of adjacent triangles. Shape: (nE, vec2i)",
        )

        self.F_edges = self.resource(
            "F_edges",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                lanes=3,
                shape_fn=shape_of(self.F_verts),
                allow_none=True,
            ),
            doc="Edge id per face directed edge (a->b, b->c, c->a). Shape: (nF, vec3i)",
        )

        self.F_edge_sign = self.resource(
            "F_edge_sign",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                lanes=3,
                shape_fn=shape_of(self.F_verts),
                allow_none=True,
            ),
            doc="Sign +1/-1 per face directed edge relative to canonical E_verts orientation. Shape: (nF, vec3i)",
        )

        self.F_adj = self.resource(
            "F_adj",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                lanes=3,
                shape_fn=shape_of(self.F_verts),
                allow_none=True,
            ),
            doc=(
                "Per-face neighbor across opposite edges (barycentric convention). "
                "F_adj[f][m] is the adjacent face across the edge where barycentric coord m becomes 0, "
                "i.e. the edge opposite vertex F_verts[f][m]. -1 indicates a boundary edge. "
                "Shape: (nF, vec3i)"
            ),
        )

        self.V_incident = self.resource(
            "V_incident",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                allow_none=True,
                shape_fn=shape_of(self.V_pos),
            ),
            doc="Count of faces a vertex is incident on. Shape: (nV, i32)",
        )

        topology_producer = TopologyProducer(
            F_verts=self.F_verts,
            V_pos=self.V_pos,
            n_edges=self.n_edges,
            E_verts=self.E_verts,
            E_faces=self.E_faces,
            E_opp=self.E_opp,
            F_edges=self.F_edges,
            F_edge_sign=self.F_edge_sign,
            F_adj=self.F_adj,
            V_incident=self.V_incident,
        )
        deps = (self.F_verts, self.V_pos)

        self.declare_resource(self.n_edges, deps=deps, producer=topology_producer)
        self.declare_resource(self.E_verts, deps=deps, producer=topology_producer)
        self.declare_resource(self.E_faces, deps=deps, producer=topology_producer)
        self.declare_resource(self.E_opp, deps=deps, producer=topology_producer)
        self.declare_resource(self.F_edges, deps=deps, producer=topology_producer)
        self.declare_resource(self.F_edge_sign, deps=deps, producer=topology_producer)
        self.declare_resource(self.F_adj, deps=deps, producer=topology_producer)
        self.declare_resource(self.V_incident, deps=deps, producer=topology_producer)

        # Geometry/Metric dependant resources
        self.F_area = self.resource(
            "F_area",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                lanes=None,
                shape_fn=shape_of(self.F_verts),
                allow_none=True,
            ),
            doc="Scalar 1D field of area per triangle face. Shape: (nF, )",
        )

        self.F_normal = self.resource(
            "F_normal",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                lanes=3,
                shape_fn=shape_of(self.F_verts),
                allow_none=True,
            ),
            doc="Vector per triangle face facing. Shape: (nF, vec3f)",
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
