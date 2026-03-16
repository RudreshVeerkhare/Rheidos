from rheidos.compute import (
    ModuleBase,
    World,
    ResourceSpec,
    shape_of,
    shape_with_tail,
    WiredProducer,
    ResourceRef,
    out_field,
)
from rheidos.compute.registry import Registry

from ..point_vortex import PointVortexModule
from ..surface_mesh import SurfaceMeshModule

import taichi as ti
import numpy as np

from collections import deque
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


def init_is_comp(reg: Registry, io: "SelfVelBasisProducerIO"):
    f_verts = io.F_verts.peek()
    if f_verts is None:
        return None
    # F_verts is (nF,) with lanes=3, so shape is (nF,)
    is_comp = ti.field(dtype=ti.int32, shape=f_verts.shape)
    is_comp.fill(-1)
    return is_comp


@dataclass
class SelfVelBasisProducerIO:
    query_face_id: ResourceRef[int]
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    V_incident: ResourceRef[Dict[int, List[int]]]

    # basis[fid, local_corner] = U_corner (vec3f), where local_corner in {0,1,2}
    # So basis has shape (nF, 3) with lanes=3
    basis: ResourceRef[ti.Field] = out_field()
    is_comp: ResourceRef[ti.Field] = out_field(alloc=init_is_comp)  # (nF, int)


@ti.data_oriented
class SelfVelBasisProducer(WiredProducer[SelfVelBasisProducerIO]):

    # -----------------------------
    # Patch extraction (faces as vertex triplets)
    # -----------------------------
    def bfs_faces(self, root_face_id: int, max_depth: int = 2) -> np.ndarray:
        faces = self.io.F_verts.get().to_numpy()  # (nF, 3)
        v_inc = self.io.V_incident.get()

        queue = deque([(int(root_face_id), 0)])
        visited_faces: Set[int] = set()
        F_patch: List[np.ndarray] = []

        while queue:
            face_id, depth = queue.popleft()
            if depth > max_depth or face_id in visited_faces:
                continue

            F_patch.append(faces[face_id])

            # expand by vertex-to-incident-faces adjacency
            for vid in faces[face_id]:
                for next_fid in v_inc[int(vid)]:
                    if next_fid in visited_faces:
                        continue
                    queue.append((int(next_fid), depth + 1))

            visited_faces.add(face_id)

        return np.array(F_patch, dtype=np.int32)

    # -----------------------------
    # DEC cotan weight helper
    # -----------------------------
    @staticmethod
    def _cot_angle(
        p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, eps: float = 1e-30
    ) -> float:
        """
        cot(angle at p0) in triangle (p0,p1,p2), embedded in R^3.
        cot(theta) = dot(u,v) / ||u x v||, u=p1-p0, v=p2-p0
        """
        u = p1 - p0
        v = p2 - p0
        cross = np.cross(u, v)
        denom = float(np.linalg.norm(cross))
        return float(np.dot(u, v) / max(denom, eps))

    # -----------------------------
    # Patch boundary via edge counts
    # -----------------------------
    @staticmethod
    def _patch_boundary_vertices(F_patch: np.ndarray) -> Set[int]:
        edge_count: Dict[Tuple[int, int], int] = {}
        for a, b, c in F_patch:
            edges = [(a, b), (b, c), (c, a)]
            for u, v in edges:
                uu, vv = (int(u), int(v))
                if uu > vv:
                    uu, vv = vv, uu
                edge_count[(uu, vv)] = edge_count.get((uu, vv), 0) + 1

        boundary: Set[int] = set()
        for (u, v), cnt in edge_count.items():
            if cnt == 1:
                boundary.add(u)
                boundary.add(v)
        return boundary

    # -----------------------------
    # Assemble DEC scalar Laplacian stiffness K_II (Dirichlet boundary eliminated)
    # -----------------------------
    def create_patch_laplacian(self, F_patch: np.ndarray):
        """
        Returns:
          KII_dense : (m,m) dense numpy array for interior unknowns
          interior_list : list of global vertex ids in interior (unknowns), length m
          boundary_set : set of global vertex ids on patch boundary (Dirichlet, psi=0)
          localI : dict global_vid -> [0..m-1]
        """
        V_pos = self.io.V_pos.get().to_numpy().astype(np.float64)  # (nV,3)
        F_patch = F_patch.astype(np.int32)

        # Collect patch vertices
        V_patch: Set[int] = set(map(int, np.unique(F_patch).tolist()))

        boundary = self._patch_boundary_vertices(F_patch)
        interior = V_patch - boundary
        interior_list = sorted(interior)
        localI = {vid: idx for idx, vid in enumerate(interior_list)}
        m = len(interior_list)

        # Dense KII (patch sizes are small-ish; simplifies everything)
        KII = np.zeros((m, m), dtype=np.float64)

        # Accumulate undirected edge weights w_uv = 0.5 * sum(cot(opposite angles)) over patch triangles
        edge_w: Dict[Tuple[int, int], float] = {}

        for a, b, c in F_patch:
            a = int(a)
            b = int(b)
            c = int(c)
            pa, pb, pc = V_pos[a], V_pos[b], V_pos[c]

            cot_a = self._cot_angle(pa, pb, pc)  # opposite edge (b,c)
            cot_b = self._cot_angle(pb, pc, pa)  # opposite edge (c,a)
            cot_c = self._cot_angle(pc, pa, pb)  # opposite edge (a,b)

            # edge (b,c) += 0.5*cot_a
            u, v = (b, c) if b < c else (c, b)
            edge_w[(u, v)] = edge_w.get((u, v), 0.0) + 0.5 * cot_a

            # edge (c,a) += 0.5*cot_b
            u, v = (c, a) if c < a else (a, c)
            edge_w[(u, v)] = edge_w.get((u, v), 0.0) + 0.5 * cot_b

            # edge (a,b) += 0.5*cot_c
            u, v = (a, b) if a < b else (b, a)
            edge_w[(u, v)] = edge_w.get((u, v), 0.0) + 0.5 * cot_c

        # Assemble reduced stiffness KII from edge weights with Dirichlet boundary elimination
        for (u, v), w in edge_w.items():
            if w == 0.0:
                continue

            u_int = u in interior
            v_int = v in interior

            # Diagonal: if endpoint is interior unknown, add +w
            if u_int:
                iu = localI[u]
                KII[iu, iu] += w
            if v_int:
                iv = localI[v]
                KII[iv, iv] += w

            # Off-diagonal only if both endpoints are interior unknowns
            if u_int and v_int:
                iu = localI[u]
                iv = localI[v]
                KII[iu, iv] -= w
                KII[iv, iu] -= w

        return KII, interior_list, boundary, localI

    # -----------------------------
    # Geometry: barycentric gradients and face velocity operator
    # -----------------------------
    @staticmethod
    def _face_geom_ops(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
        """
        Returns (n_hat, c0, c1, c2) where:
          g_a = grad(lambda_a)
          c_a = n_hat x g_a
          u = psi0*c0 + psi1*c1 + psi2*c2  (up to sign convention)
        """
        e1 = p1 - p0
        e2 = p2 - p0
        n = np.cross(e1, e2)
        nn = float(np.dot(n, n))
        if nn < 1e-30:
            n_hat = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            z = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            return n_hat, z, z, z

        n_hat = n / np.sqrt(nn)

        # g0 = n x (p2 - p1) / ||n||^2
        g0 = np.cross(n, (p2 - p1)) / nn
        g1 = np.cross(n, (p0 - p2)) / nn
        g2 = np.cross(n, (p1 - p0)) / nn

        c0 = np.cross(n_hat, g0)
        c1 = np.cross(n_hat, g1)
        c2 = np.cross(n_hat, g2)

        return n_hat, c0, c1, c2

    @ti.kernel
    def _write_basis_and_mark(
        self,
        basis: ti.template(),
        is_comp: ti.template(),
        fid: ti.i32,
        U0: ti.types.vector(3, ti.f32),
        U1: ti.types.vector(3, ti.f32),
        U2: ti.types.vector(3, ti.f32),
    ):
        basis[fid, 0] = U0
        basis[fid, 1] = U1
        basis[fid, 2] = U2
        is_comp[fid] = 1

    def compute(self, reg: Registry) -> None:
        inputs = self.require_inputs()
        outputs = self.ensure_outputs(reg)

        basis = outputs["basis"].peek()
        is_comp = outputs["is_comp"].peek()

        query_face_id = int(inputs["query_face_id"].get())
        if query_face_id < 0:
            self.io.basis.commit()
            self.io.is_comp.commit()
            return

        if is_comp[query_face_id] != -1:
            # already computed
            self.io.basis.commit()
            self.io.is_comp.commit()
            return

        # Geometry arrays
        V_pos = inputs["V_pos"].get().to_numpy().astype(np.float64)
        F_verts = inputs["F_verts"].get().to_numpy().astype(np.int32)

        # 1) patch faces (as vertex triplets)
        F_patch = self.bfs_faces(query_face_id, max_depth=2)

        # 2) Assemble patch Dirichlet-reduced stiffness KII
        KII, interior_list, boundary, localI = self.create_patch_laplacian(F_patch)
        m = KII.shape[0]

        # If no interior unknowns, basis is zero
        if m == 0:
            self._write_basis_and_mark(
                basis,
                is_comp,
                query_face_id,
                ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
                ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
                ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
            )
            self.io.basis.commit()
            self.io.is_comp.commit()
            return

        # Pre-factorization (dense). Patch is small; this is fine.
        # If you later want sparse CG, swap this block.
        try:
            # Cholesky is fastest if SPD
            L = np.linalg.cholesky(KII)

            def solve_KII(rhs: np.ndarray) -> np.ndarray:
                # solve L y = rhs, then L^T x = y
                y = np.linalg.solve(L, rhs)
                x = np.linalg.solve(L.T, y)
                return x

        except np.linalg.LinAlgError:
            # Fallback: generic solve (handles some non-SPD cases, slower)
            def solve_KII(rhs: np.ndarray) -> np.ndarray:
                return np.linalg.solve(KII, rhs)

        # 3) Face geometry operator (L)
        i, j, k = map(int, F_verts[query_face_id])
        p0, p1, p2 = V_pos[i], V_pos[j], V_pos[k]
        _, c0, c1, c2 = self._face_geom_ops(p0, p1, p2)

        def face_psi_value(vid: int, psiI: np.ndarray) -> float:
            if vid in boundary:
                return 0.0
            idx = localI.get(vid, None)
            if idx is None:
                # not in patch (should not happen for i,j,k), treat as boundary
                return 0.0
            return float(psiI[idx])

        # 4) Solve 3 unit-load problems and convert to velocity basis vectors
        # U_corner means: "velocity on this face produced by RHS = e_corner (unit at that corner vertex)"
        U_vecs = []
        for corner_vid in (i, j, k):
            rhs = np.zeros(m, dtype=np.float64)
            if corner_vid not in boundary:
                rhs[localI[corner_vid]] = 1.0

            psiI = solve_KII(rhs)

            psi_i = face_psi_value(i, psiI)
            psi_j = face_psi_value(j, psiI)
            psi_k = face_psi_value(k, psiI)

            # u = psi_i*c0 + psi_j*c1 + psi_k*c2
            # If your global convention uses the opposite sign, flip here consistently.
            u = psi_i * c0 + psi_j * c1 + psi_k * c2
            U_vecs.append(u.astype(np.float32))

        U0 = ti.Vector(U_vecs[0].tolist(), dt=ti.f32)
        U1 = ti.Vector(U_vecs[1].tolist(), dt=ti.f32)
        U2 = ti.Vector(U_vecs[2].tolist(), dt=ti.f32)

        # 5) write basis for this face and mark computed
        self._write_basis_and_mark(basis, is_comp, query_face_id, U0, U1, U2)
        self.io.basis.commit()
        self.io.is_comp.commit()


class SelfVelBasisModule(ModuleBase):
    NAME = "SelfVelBasisModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.surface_mesh = world.require(SurfaceMeshModule)
        self.pt_vortex = world.require(PointVortexModule)

        self.query_face_id = self.resource(
            "query_face_id",
            spec=ResourceSpec(kind="python", dtype=int, allow_none=True),
            doc="The query face for which the producer will run computation.",
            declare=True,
            buffer=-1,
        )

        # IMPORTANT: basis is per-face-per-corner, not per-vertex.
        # shape_of(F_verts) returns (nF,) since F_verts is vec3; add a tail dim for corners.
        self.vel_basis = self.resource(
            "vel_basis",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.f32,
                shape_fn=shape_with_tail(self.surface_mesh.F_verts, tail=(3,)),
                lanes=3,
                allow_none=True,
            ),
            doc="Self-velocity basis U per face-corner. Shape: (nF, 3, vec3f)",
        )

        self.is_computed = self.resource(
            "is_computed",
            spec=ResourceSpec(
                kind="taichi_field",
                dtype=ti.i32,
                shape_fn=shape_of(self.surface_mesh.F_verts),
                allow_none=True,
            ),
            doc="Track faces for which the basis is computed. Shape: (nF, int32)",
        )

        producer = SelfVelBasisProducer(
            query_face_id=self.query_face_id,
            V_pos=self.surface_mesh.V_pos,
            F_verts=self.surface_mesh.F_verts,
            V_incident=self.surface_mesh.V_incident,
            basis=self.vel_basis,
            is_comp=self.is_computed,
        )

        self.declare_resource(
            self.vel_basis,
            deps=(self.query_face_id, self.surface_mesh.F_verts),
            producer=producer,
        )

        self.declare_resource(
            self.is_computed,
            deps=(self.query_face_id, self.surface_mesh.F_verts),
            producer=producer,
        )
