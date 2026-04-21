import numpy as np

from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ProducerContext, ResourceSpec, World, producer
from rheidos.compute import shape_map


def _cot_at(x1, x2, o) -> float:
    e1 = o - x1
    e2 = o - x2
    return np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))


class DEC(ModuleBase):
    NAME = "DEC"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh

        self.star1 = self.resource(
            "star1",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.E_verts, lambda s: (s[0],)),
            ),
            doc="Hodge star on 1-forms (cotan weights per edge). Shape: (nE, )",
        )

        self.boundary_mask = self.resource(
            "boundary_mask",
            spec=ResourceSpec(
                kind="numpy",
                dtype=bool,
                shape_fn=shape_map(self.mesh.V_pos, lambda s: (s[0],)),
            ),
            doc="Boolean mask to identify boundary DOFs/Vertices. Shape: (nV, )",
        )

        self.bind_producers()

    @producer(
        inputs=("mesh.E_faces", "mesh.E_verts", "mesh.V_pos"),
        outputs=("boundary_mask",),
    )
    def build_boundary_mask(self, ctx: ProducerContext):
        ctx.require_inputs()
        e_faces = self.mesh.E_faces.get()
        e_verts = self.mesh.E_verts.get()
        v_pos = self.mesh.V_pos.get()
        nV = len(v_pos)

        mask = np.zeros(nV, dtype=bool)
        for edge_id, faces in enumerate(e_faces):
            if -1 in faces:
                v1, v2 = e_verts[edge_id]
                mask[v1] = True
                mask[v2] = True

        ctx.commit(boundary_mask=mask)

    @producer(inputs=("mesh.V_pos", "mesh.E_verts", "mesh.E_opp"), outputs=("star1",))
    def build_cotan_star1(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()

        e_opp = self.mesh.E_opp.get()
        e_verts = self.mesh.E_verts.get()
        v_pos = self.mesh.V_pos.get()
        star1 = np.zeros_like(self.star1.peek())
        for edge_id, (o1, o2) in enumerate(e_opp):
            v1, v2 = e_verts[edge_id]
            x1 = v_pos[v1]
            x2 = v_pos[v2]
            if o1 >= 0:
                star1[edge_id] += 0.5 * _cot_at(x1, x2, v_pos[o1])
            if o2 >= 0:
                star1[edge_id] += 0.5 * _cot_at(x1, x2, v_pos[o2])

        ctx.commit(star1=star1)
