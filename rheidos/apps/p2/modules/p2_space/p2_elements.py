import numpy as np

from rheidos.apps.p2.modules.p2_space.stiffness import (
    cotan_triangle_weights,
    p2_local_stiffness_from_cotan,
    p2_local_lumped_mass_matrix,
)
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ResourceSpec, shape_from_scalar, shape_map
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import World


class P2Elements(ModuleBase):
    NAME = "P2Elements"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.mesh = self.require(SurfaceMeshModule)

        self.n_dof = self.resource(
            "n_dof",
            spec=ResourceSpec(kind="python", dtype=int),
            doc="Total DOFs of the P2 element space",
        )

        self.face_dof = self.resource(
            "face_dof",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int64,
                shape_fn=shape_map(self.mesh.F_verts, lambda s: (s[0], 6)),
            ),
            doc="Face to DOF index mapping from global index. Shape: (nF, 6)",
        )

        self.boundary_mask = self.resource(
            "boundary_mask",
            spec=ResourceSpec(
                kind="numpy", dtype=bool, shape_fn=shape_from_scalar(self.n_dof)
            ),
            doc="Boolean mask to identify boundary dofs. Shape: (nDof, )",
        )

        self.dof_pos = self.resource(
            "dof_pos",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_scalar(self.n_dof, tail=(3,)),
            ),
        )

        self.L_stiffness = self.resource(
            "L_stiffness",
            spec=ResourceSpec(kind="python"),
            doc="SciPy sparse linear matrix representing P2 scalar laplacian",
        )

        self.M_mass = self.resource(
            "M_mass",
            spec=ResourceSpec(kind="python"),
            doc="SciPy sparse array representing the P2 lumped mass matrix.",
        )

        self.bind_producers()

    @producer(
        inputs=("mesh.V_pos", "mesh.E_verts", "mesh.E_faces", "n_dof"),
        outputs=("boundary_mask",),
    )
    def create_boundary_mask(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        n_dof = self.n_dof.get()
        e_verts = self.mesh.E_verts.get()
        e_faces = self.mesh.E_faces.get()
        nV = self.mesh.V_pos.get().shape[0]

        mask = np.zeros(n_dof, dtype=bool)

        for edgeid, faces in enumerate(e_faces):
            if -1 in faces:
                v1, v2 = e_verts[edgeid]
                mask[v1] = True
                mask[v2] = True
                mask[nV + edgeid] = True

        ctx.commit(boundary_mask=mask)

    @producer(inputs=("mesh.V_pos", "n_dof", "mesh.E_verts"), outputs=("dof_pos",))
    def build_dof_pos(self, ctx: ProducerContext):
        ctx.require_inputs()
        n_dof = self.n_dof.get()
        V_pos = self.mesh.V_pos.get()
        E_verts = self.mesh.E_verts.get()

        nV = len(V_pos)
        dof_pos = np.zeros((n_dof, 3), dtype=np.float64)

        for vertex_id in range(nV):
            dof_pos[vertex_id] = V_pos[vertex_id]

        for edge_id, (v1, v2) in enumerate(E_verts):
            mid_point = (V_pos[v1] + V_pos[v2]) / 2
            dof_pos[nV + edge_id] = mid_point

        ctx.commit(dof_pos=dof_pos)

    @producer(
        inputs=("mesh.V_pos", "mesh.F_verts", "mesh.F_edges", "mesh.E_verts"),
        outputs=("face_dof", "n_dof"),
    )
    def create_global_index_for_dofs(self, ctx: ProducerContext):
        """Concatenates vertex_idx and edge_idx to create a global index.
        Mapping: vert_idx -> vert_idx, edge_idx -> nV + edge_idx"""

        ctx.require_inputs()
        V = self.mesh.V_pos.get()
        F = self.mesh.F_verts.get()
        E = self.mesh.E_verts.get()
        F_edges = self.mesh.F_edges.get()

        nV = V.shape[0]
        nE = E.shape[0]
        face_dofs = []
        for faceid in range(len(F)):
            v1, v2, v3 = F[faceid]
            e1, e2, e3 = F_edges[faceid]
            face_dofs.append([v1, v2, v3, nV + e1, nV + e2, nV + e3])

        ctx.commit(face_dof=np.array(face_dofs, dtype=np.int64), n_dof=nV + nE)

    @producer(inputs=("face_dof", "mesh.V_pos", "n_dof"), outputs=("L_stiffness",))
    def build_L_stiffness(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()

        face_dof = self.face_dof.get()
        n_dof = self.n_dof.get()
        V = self.mesh.V_pos.get()

        from scipy.sparse import coo_matrix

        rows = []
        cols = []
        vals = []
        for faceid in range(len(face_dof)):
            lg = face_dof[faceid]

            cot1, cot2, cot3 = cotan_triangle_weights(V[lg[0]], V[lg[1]], V[lg[2]])
            Kl = p2_local_stiffness_from_cotan(cot1, cot2, cot3)

            for a in range(6):
                for b in range(6):
                    rows.append(lg[a])
                    cols.append(lg[b])
                    vals.append(Kl[a, b])

        K_global = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

        ctx.commit(L_stiffness=K_global)

    @producer(inputs=("face_dof", "n_dof", "mesh.F_area"), outputs=("M_mass",))
    def build_M_mass(self, ctx: ProducerContext):
        ctx.require_inputs()

        face_dof = self.face_dof.get()
        n_dof = self.n_dof.get()
        F_areas = self.mesh.F_area.get()

        from scipy.sparse import coo_matrix

        rows = []
        cols = []
        vals = []
        for faceid in range(len(face_dof)):
            lg = face_dof[faceid]
            Ml = p2_local_lumped_mass_matrix(F_areas[faceid])

            for a in range(6):
                for b in range(6):
                    rows.append(lg[a])
                    cols.append(lg[b])
                    vals.append(Ml[a, b])

        M_global = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
        M_global.sum_duplicates()
        ctx.commit(M_mass=M_global)

    def basis_from_bary(self, b1: float, b2: float, b3: float):
        return (
            b1 * (2 * b1 - 1),
            b2 * (2 * b2 - 1),
            b3 * (2 * b3 - 1),
            4 * b1 * b2,
            4 * b2 * b3,
            4 * b3 * b1,
        )
