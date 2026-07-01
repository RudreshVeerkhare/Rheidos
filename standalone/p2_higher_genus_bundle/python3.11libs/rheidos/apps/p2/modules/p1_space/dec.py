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

    def d0(self, zero_form: np.ndarray) -> np.ndarray:
        """Apply the exterior derivative from primal 0-forms to primal 1-forms."""
        # In this DEC module, a primal 0-form is stored as one scalar per mesh
        # vertex. The derivative d0 moves that vertex cochain onto the canonical
        # mesh edges, producing one scalar for every row of mesh.E_verts.
        vertex_values = np.asarray(zero_form, dtype=np.float64)

        # Force the mesh resources now. V_pos gives the number of vertices the
        # input 0-form must cover, and E_verts gives each canonical edge as
        # (tail_vertex, head_vertex). The canonical orientation is chosen by the
        # mesh topology builder and is the same orientation used by all 1-forms
        # in this module.
        v_pos = self.mesh.V_pos.get()
        e_verts = self.mesh.E_verts.get()

        if vertex_values.ndim != 1:
            raise ValueError(
                f"d0 expects a 0-form with shape (nV,), got {vertex_values.shape}"
            )

        n_vertices = v_pos.shape[0]
        if vertex_values.shape[0] != n_vertices:
            raise ValueError(
                f"d0 expects a 0-form with length nV={n_vertices}, "
                f"got {vertex_values.shape[0]}"
            )

        # The discrete exterior derivative d0 is the oriented coboundary from
        # vertices to edges. For each canonical edge (u, v), the edge value is
        # phi(v) - phi(u), matching the orientation stored in E_verts. The input
        # was cast to float64 above so callers get a stable floating 1-form even
        # when they pass integer or boolean vertex data.
        return (vertex_values[e_verts[:, 1]] - vertex_values[e_verts[:, 0]]).astype(
            np.float64, copy=False
        )

    def d0_transpose(self, one_form: np.ndarray) -> np.ndarray:
        """Apply the transpose of d0 from primal 1-forms to primal 0-forms."""
        edge_values = np.asarray(one_form, dtype=np.float64)
        e_verts = self.mesh.E_verts.get()
        v_pos = self.mesh.V_pos.get()

        if edge_values.ndim not in (1, 2):
            raise ValueError(
                "d0_transpose expects a 1-form with shape (nE,) or a batch "
                f"with shape (k,nE), got {edge_values.shape}"
            )

        n_edges = e_verts.shape[0]
        edge_axis = 0 if edge_values.ndim == 1 else 1
        if edge_values.shape[edge_axis] != n_edges:
            raise ValueError(
                f"d0_transpose expects edge axis length nE={n_edges}, "
                f"got {edge_values.shape[edge_axis]}"
            )

        n_vertices = v_pos.shape[0]
        if edge_values.ndim == 1:
            result = np.zeros(n_vertices, dtype=np.float64)
            # d0 has entries -1 at the edge tail and +1 at the edge head.
            # Its transpose therefore scatters edge coefficients back to the
            # incident vertices with those same signs.
            np.add.at(result, e_verts[:, 0], -edge_values)
            np.add.at(result, e_verts[:, 1], edge_values)
            return result

        result = np.zeros((edge_values.shape[0], n_vertices), dtype=np.float64)
        np.add.at(result, (slice(None), e_verts[:, 0]), -edge_values)
        np.add.at(result, (slice(None), e_verts[:, 1]), edge_values)
        return result

    def d1(self, one_form: np.ndarray) -> np.ndarray:
        """Apply the exterior derivative from primal 1-forms to primal 2-forms."""
        # In this DEC module, a primal 1-form is stored as one scalar per
        # canonical mesh edge. The canonical edge orientation is the orientation
        # used by mesh.E_verts, which stores each undirected edge once with a
        # deterministic vertex ordering.
        edge_values = np.asarray(one_form, dtype=np.float64)

        # Force the mesh topology resources now. F_edges gives, for each face,
        # the three global edge ids on that face boundary; F_edge_sign gives the
        # sign needed to read each canonical edge value in the orientation of
        # the local face boundary.
        f_edges = self.mesh.F_edges.get()
        f_edge_sign = self.mesh.F_edge_sign.get()
        e_verts = self.mesh.E_verts.get()

        if edge_values.ndim != 1:
            raise ValueError(
                f"d1 expects a 1-form with shape (nE,), got {edge_values.shape}"
            )

        n_edges = e_verts.shape[0]
        if edge_values.shape[0] != n_edges:
            raise ValueError(
                f"d1 expects a 1-form with length nE={n_edges}, "
                f"got {edge_values.shape[0]}"
            )

        # The discrete exterior derivative d1 is the oriented coboundary of the
        # edge cochain: for each face, sum the three edge values around the
        # oriented face boundary. Multiplying by F_edge_sign converts from the
        # canonical global edge orientation into the local face orientation, and
        # indexing by F_edges gathers the input 1-form values for each face.
        #
        # The input was cast to float64 above so callers get a stable floating
        # 2-form even when they pass integer or boolean edge data.
        return np.einsum("fk,fk->f", f_edge_sign, edge_values[f_edges]).astype(
            np.float64,
            copy=False,
        )

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
