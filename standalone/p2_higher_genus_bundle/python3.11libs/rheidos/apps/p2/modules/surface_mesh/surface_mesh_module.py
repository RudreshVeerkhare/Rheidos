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


def _clamp_and_normalize_barycentric_rows(bary: np.ndarray) -> np.ndarray:
    bary = np.clip(bary, 0.0, 1.0)
    row_sum = bary.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("Projected barycentric coordinates became invalid")
    bary /= row_sum
    return bary


def barycentric_gradients(x0, x1, x2):
    e01 = x1 - x0
    e02 = x2 - x0
    N = np.cross(e01, e02)  # unnormalized normal
    NN = np.dot(N, N)

    if NN <= 1e-30:
        raise ValueError("Degenerate triangle")

    grad_l0 = np.cross(N, x2 - x1) / NN
    grad_l1 = np.cross(N, x0 - x2) / NN
    grad_l2 = np.cross(N, x1 - x0) / NN

    return grad_l0, grad_l1, grad_l2


class SurfaceMeshModule(ModuleBase):
    NAME = "P2SurfaceMesh"
    PROJECTION_MAX_FACE_POINT_PAIRS = 500_000
    DEGENERATE_FACE_AREA_EPS = 0.5e-20
    BARY_REGION_EPS = 1e-12

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
        self.boundary_edge_ids = self.resource(
            "boundary_edge_ids",
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
            doc="Sorted boundary edge ids into E_verts. Shape: (nBoundaryEdges,)",
        )
        self.boundary_edge_components = self.resource(
            "boundary_edge_components",
            spec=ResourceSpec(kind="python", dtype=list, allow_none=True),
            doc="Ordered boundary edge traversals, one per connected boundary component.",
        )
        self.boundary_vertex_ids = self.resource(
            "boundary_vertex_ids",
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
            doc="Sorted unique vertex ids on the boundary. Shape: (nBoundaryVertices,)",
        )
        self.boundary_vertex_components = self.resource(
            "boundary_vertex_components",
            spec=ResourceSpec(kind="python", dtype=list, allow_none=True),
            doc="Ordered boundary vertex traversals aligned with boundary_edge_components.",
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
        self.F_origin = self.resource(
            "F_origin",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.F_verts, lambda shape: (shape[0], 3)),
                allow_none=True,
            ),
            doc="First vertex of each face. Shape: (nF,3)",
        )
        self.F_edge01 = self.resource(
            "F_edge01",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.F_verts, lambda shape: (shape[0], 3)),
                allow_none=True,
            ),
            doc="Per-face edge from vertex 0 to vertex 1. Shape: (nF,3)",
        )
        self.F_edge02 = self.resource(
            "F_edge02",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.F_verts, lambda shape: (shape[0], 3)),
                allow_none=True,
            ),
            doc="Per-face edge from vertex 0 to vertex 2. Shape: (nF,3)",
        )

        self.grad_bary = self.resource(
            "grad_bary",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.F_verts, lambda s: (s[0], 3, 3)),
            ),
            doc="A per face 3x3 matrix of [∇l1 ∇l2 ∇l3], where ∇li is a 3x1 gradient vector associated with lambda_i of vertex F_verts[i]. Shape: (nF, 3, 3)",
        )

        self.bind_producers()

    @producer(inputs=("V_pos", "F_verts"), outputs=("grad_bary",))
    def build_grad_bary(self, ctx: ProducerContext):
        v = ctx.inputs.V_pos.get()
        f = ctx.inputs.F_verts.get()

        x0 = v[f[:, 0]]
        x1 = v[f[:, 1]]
        x2 = v[f[:, 2]]
        e01 = x1 - x0
        e02 = x2 - x0
        n = np.cross(e01, e02)
        nn = np.einsum("fi,fi->f", n, n)
        twice_area = np.sqrt(nn)
        valid = twice_area > 1e-20

        grad = np.zeros((f.shape[0], 3, 3), dtype=np.float64)
        if np.any(valid):
            denom = nn[valid, None]
            grad[valid, 0] = np.cross(n[valid], x2[valid] - x1[valid]) / denom
            grad[valid, 1] = np.cross(n[valid], x0[valid] - x2[valid]) / denom
            grad[valid, 2] = np.cross(n[valid], x1[valid] - x0[valid]) / denom
        ctx.commit(grad_bary=grad)

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
            "boundary_edge_ids",
            "boundary_edge_components",
            "boundary_vertex_ids",
            "boundary_vertex_components",
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
            boundary_edge_ids,
            boundary_edge_components,
            boundary_vertex_ids,
            boundary_vertex_components,
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
            boundary_edge_ids=boundary_edge_ids,
            boundary_edge_components=boundary_edge_components,
            boundary_vertex_ids=boundary_vertex_ids,
            boundary_vertex_components=boundary_vertex_components,
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

    @producer(
        inputs=("V_pos", "F_verts"),
        outputs=("F_origin", "F_edge01", "F_edge02"),
    )
    def build_projection_geometry(self, ctx: ProducerContext) -> None:
        v = ctx.inputs.V_pos.get()
        f = ctx.inputs.F_verts.get()

        x0 = v[f[:, 0]]
        x1 = v[f[:, 1]]
        x2 = v[f[:, 2]]
        ctx.commit(
            F_origin=x0,
            F_edge01=x1 - x0,
            F_edge02=x2 - x0,
        )

    def project_on_nearest_face(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Projects 3D points onto the nearest triangle on the current surface mesh.

        Args:
            points (np.ndarray): Point positions in 3D space. Shape: (N, 3)

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - face ids of shape (N,)
                - barycentric coordinates on the winning face of shape (N, 3)
                - projected 3D positions on the winning face of shape (N, 3)
        """
        p = np.ascontiguousarray(points, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"points must have shape (N,3), got {p.shape}")
        if p.shape[0] == 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        face_origin = self.F_origin.get()
        face_edge01 = self.F_edge01.get()
        face_edge02 = self.F_edge02.get()
        face_normal = self.F_normal.get()
        face_area = self.F_area.get()
        grad_bary = self.grad_bary.get()

        n_faces = int(face_origin.shape[0])
        if n_faces == 0:
            raise ValueError("Cannot project onto an empty mesh")

        valid_faces = face_area > self.DEGENERATE_FACE_AREA_EPS
        if not np.any(valid_faces):
            raise ValueError("Cannot project onto a mesh with only degenerate faces")

        face_origin = np.ascontiguousarray(face_origin, dtype=np.float64)
        face_edge01 = np.ascontiguousarray(face_edge01, dtype=np.float64)
        face_edge02 = np.ascontiguousarray(face_edge02, dtype=np.float64)
        face_normal = np.ascontiguousarray(face_normal, dtype=np.float64)
        grad_bary = np.ascontiguousarray(grad_bary, dtype=np.float64)

        edge01_len2 = np.einsum("fi,fi->f", face_edge01, face_edge01)
        edge02_len2 = np.einsum("fi,fi->f", face_edge02, face_edge02)
        face_edge12 = face_edge02 - face_edge01
        edge12_len2 = np.einsum("fi,fi->f", face_edge12, face_edge12)
        safe_edge01_len2 = np.where(edge01_len2 > 0.0, edge01_len2, 1.0)
        safe_edge02_len2 = np.where(edge02_len2 > 0.0, edge02_len2, 1.0)
        safe_edge12_len2 = np.where(edge12_len2 > 0.0, edge12_len2, 1.0)

        max_pairs = max(1, int(self.PROJECTION_MAX_FACE_POINT_PAIRS))
        chunk_size = max(1, max_pairs // max(1, n_faces))

        faceids = np.empty((p.shape[0],), dtype=np.int32)
        bary = np.empty((p.shape[0], 3), dtype=np.float64)
        projected = np.empty((p.shape[0], 3), dtype=np.float64)
        valid_faces_row = valid_faces[None, :]

        for start in range(0, p.shape[0], chunk_size):
            stop = min(start + chunk_size, p.shape[0])
            chunk = p[start:stop]

            rel = chunk[:, None, :] - face_origin[None, :, :]
            signed_plane_distance = np.einsum("mfi,fi->mf", rel, face_normal)
            plane_bary = np.einsum("mfi,fji->mfj", rel, grad_bary)
            plane_bary[:, :, 0] += 1.0

            negative = plane_bary < -self.BARY_REGION_EPS
            negative_count = negative.sum(axis=2)
            valid_pairs = np.broadcast_to(valid_faces_row, negative_count.shape)

            candidate_bary = np.zeros_like(plane_bary)

            interior_mask = valid_pairs & (negative_count == 0)
            candidate_bary[interior_mask] = plane_bary[interior_mask]

            edge_ab_mask = valid_pairs & (negative_count == 1) & negative[:, :, 2]
            edge_ac_mask = valid_pairs & (negative_count == 1) & negative[:, :, 1]
            edge_bc_mask = valid_pairs & (negative_count == 1) & negative[:, :, 0]

            t_ab = np.clip(
                np.einsum("mfi,fi->mf", rel, face_edge01) / safe_edge01_len2[None, :],
                0.0,
                1.0,
            )
            t_ac = np.clip(
                np.einsum("mfi,fi->mf", rel, face_edge02) / safe_edge02_len2[None, :],
                0.0,
                1.0,
            )
            rel_from_b = rel - face_edge01[None, :, :]
            t_bc = np.clip(
                np.einsum("mfi,fi->mf", rel_from_b, face_edge12)
                / safe_edge12_len2[None, :],
                0.0,
                1.0,
            )

            candidate_bary[:, :, 0][edge_ab_mask] = 1.0 - t_ab[edge_ab_mask]
            candidate_bary[:, :, 1][edge_ab_mask] = t_ab[edge_ab_mask]
            candidate_bary[:, :, 0][edge_ac_mask] = 1.0 - t_ac[edge_ac_mask]
            candidate_bary[:, :, 2][edge_ac_mask] = t_ac[edge_ac_mask]
            candidate_bary[:, :, 1][edge_bc_mask] = 1.0 - t_bc[edge_bc_mask]
            candidate_bary[:, :, 2][edge_bc_mask] = t_bc[edge_bc_mask]

            vertex_mask = valid_pairs & (negative_count >= 2)
            v0_mask = vertex_mask & ~negative[:, :, 0]
            v1_mask = vertex_mask & ~negative[:, :, 1]
            v2_mask = vertex_mask & ~negative[:, :, 2]
            candidate_bary[:, :, 0][v0_mask] = 1.0
            candidate_bary[:, :, 1][v1_mask] = 1.0
            candidate_bary[:, :, 2][v2_mask] = 1.0

            diff = (
                rel
                - candidate_bary[:, :, 1, None] * face_edge01[None, :, :]
                - candidate_bary[:, :, 2, None] * face_edge02[None, :, :]
            )
            dist2 = np.einsum("mfi,mfi->mf", diff, diff)
            dist2[interior_mask] = signed_plane_distance[interior_mask] ** 2
            dist2[:, ~valid_faces] = np.inf

            best_face = np.argmin(dist2, axis=1).astype(np.int32, copy=False)
            best_bary = candidate_bary[np.arange(stop - start), best_face]
            best_bary = _clamp_and_normalize_barycentric_rows(best_bary)
            best_origin = face_origin[best_face]
            best_edge01 = face_edge01[best_face]
            best_edge02 = face_edge02[best_face]

            faceids[start:stop] = best_face
            bary[start:stop] = best_bary
            projected[start:stop] = (
                best_origin
                + best_bary[:, 1, None] * best_edge01
                + best_bary[:, 2, None] * best_edge02
            )

        return faceids, bary, projected
