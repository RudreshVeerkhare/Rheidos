from rheidos.compute import ModuleBase, World, ResourceSpec
import taichi as ti

from ..producers.topology import TopologyProducer
from ..producers.geometry import GeometryProducer


class SurfaceMeshModule(ModuleBase):
    NAME = "SurfaceTriangleMesh"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.V_pos = self.resource(
            "V_pos",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.f32, lanes=3, allow_none=True
            ),
            doc="Triangle mesh vertex postions, expected shape: (nV, 3)",
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

        # Topology dependant resources
        self.E_verts = self.resource(
            "E_verts",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.i32, lanes=2, allow_none=True
            ),
            doc="Set of unique edges between vertices. Expected Shape: (nE, 2)",
        )

        self.E_faces = self.resource(
            "E_faces",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.i32, lanes=2, allow_none=True
            ),
            doc="Adjacent faces per edge. Shape: (nE, vec2i)",
        )

        self.E_opp = self.resource(
            "E_opp",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.i32, lanes=2, allow_none=True
            ),
            doc="Opposite vertex per edge side of adjacent triangles. Shape: (nE, vec2i)",
        )

        topology_producer = TopologyProducer(
            F_verts=self.F_verts,
            E_verts=self.E_verts,
            E_faces=self.E_faces,
            E_opp=self.E_opp,
        )
        deps = (self.F_verts,)

        self.declare_resource(self.E_verts, deps=deps, producer=topology_producer)
        self.declare_resource(self.E_faces, deps=deps, producer=topology_producer)
        self.declare_resource(self.E_opp, deps=deps, producer=topology_producer)

        # Geometry/Metric dependant resources
        self.F_area = self.resource(
            "F_area",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.f32, lanes=None, allow_none=True
            ),
            doc="Scalar 1D field of area per triangle face. Shape: (nF, )",
        )

        self.F_normal = self.resource(
            "F_normal",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.f32, lanes=3, allow_none=True
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
