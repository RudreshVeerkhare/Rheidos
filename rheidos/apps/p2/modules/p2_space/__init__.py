from posixpath import sep

from rheidos.apps.p2.modules.point_vortex import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh import SurfaceMeshModule
from rheidos.compute import ModuleBase, ResourceSpec, shape_from_scalar, shape_map
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import World

import numpy as np


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

        self.bind_producers()

    @producer(
        inputs=("mesh.V_pos", "mesh.E_verts", "mesh.E_faces", "n_dof"),
        outputs=("boundary_mask",),
    )
    def create_boundary_mask(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        n_dof = self.n_dof.get()
        E_verts = self.mesh.E_verts.get()
        E_faces = self.mesh.E_faces.get()
        nV = self.mesh.V_pos.get().shape[0]

        mask = np.zeros(n_dof, dtype=bool)

        for edgeid, faces in enumerate(E_faces):
            if -1 in faces:  # Edge is a boundary one
                v1, v2 = E_verts[edgeid]
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
            # local-to-global P2 map
            lg = face_dof[faceid]  # (v1, v2, v3, e1, e2, e3)

            # compute cotan weights
            cot1, cot2, cot3 = _cotan_triangle_weights(V[lg[0]], V[lg[1]], V[lg[2]])

            # compute local stiffness matrix from cotan
            Kl = _p2_local_stiffness_from_cotan(cot1, cot2, cot3)

            # assemble global stiffness matrix
            for a in range(6):
                for b in range(6):
                    rows.append(lg[a])
                    cols.append(lg[b])
                    vals.append(Kl[a, b])

        K_global = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

        ctx.commit(L_stiffness=K_global)

    def basis_from_bary(self, b1: float, b2: float, b3: float):
        return (
            b1 * (2 * b1 - 1),  # phi1
            b2 * (2 * b2 - 1),  # phi2
            b3 * (2 * b3 - 1),  # phi3
            4 * b1 * b2,  # phi4
            4 * b2 * b3,  # phi5
            4 * b3 * b1,  # phi6
        )


def _p2_local_stiffness_from_cotan(cot1, cot2, cot3):
    # local ordering: [v1, v2, v3, e12, e23, e31]
    return np.array(
        [
            [(cot2 + cot3) / 2, cot3 / 6, cot2 / 6, -2 * cot3 / 3, 0.0, -2 * cot2 / 3],
            [cot3 / 6, (cot1 + cot3) / 2, cot1 / 6, -2 * cot3 / 3, -2 * cot1 / 3, 0.0],
            [cot2 / 6, cot1 / 6, (cot1 + cot2) / 2, 0.0, -2 * cot1 / 3, -2 * cot2 / 3],
            [
                -2 * cot3 / 3,
                -2 * cot3 / 3,
                0.0,
                4 * (cot1 + cot2 + cot3) / 3,
                -4 * cot2 / 3,
                -4 * cot1 / 3,
            ],
            [
                0.0,
                -2 * cot1 / 3,
                -2 * cot1 / 3,
                -4 * cot2 / 3,
                4 * (cot1 + cot2 + cot3) / 3,
                -4 * cot3 / 3,
            ],
            [
                -2 * cot2 / 3,
                0.0,
                -2 * cot2 / 3,
                -4 * cot1 / 3,
                -4 * cot3 / 3,
                4 * (cot1 + cot2 + cot3) / 3,
            ],
        ],
        dtype=float,
    )


def _cotan_triangle_weights(x1, x2, x3):
    cot_at = lambda o, a, b: np.dot((a - o), (b - o)) / np.linalg.norm(
        np.cross((a - o), (b - o))
    )
    return cot_at(x1, x2, x3), cot_at(x2, x1, x3), cot_at(x3, x2, x1)


class P2PoissonSolver(ModuleBase):
    NAME = "P2PoissonSolver"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.p2_space = self.require(P2Elements)

        # Inputs
        self.constrained_idx = self.resource(
            "constrained_idx",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
            ),
            declare=True,
            doc="Indices of constrained vertices.",
        )
        self.constrained_values = self.resource(
            "constrained_values",
            spec=ResourceSpec(kind="numpy", dtype=np.float32),
            declare=True,
            doc="Values for the constrained vertices 1-1 mapped to the index in constrained_idx.",
        )
        self.rhs = self.resource(
            "rhs",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_scalar(self.p2_space.n_dof),
            ),
            declare=True,
            doc="RHS coefficient to be paired with basis function. Shape: (nDof, )",
        )

        # Derived
        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_scalar(self.p2_space.n_dof),
            ),
            doc="Stream function coefficients for chosen p2 basis.",
        )

        self.solve_cg = self.resource(
            "solve_cg",
            spec=ResourceSpec(kind="python"),
            doc="A pre-factorized poisson solver callable, just need the RHS",
        )

        self.bind_producers()

    @producer(inputs=("solve_cg", "rhs"), outputs=("psi",))
    def solve_for_psi(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()
        solve_cg = self.solve_cg.get()
        rhs = self.rhs.get()

        ctx.commit(psi=solve_cg(rhs))

    @producer(
        inputs=(
            "p2_space.L_stiffness",
            "p2_space.n_dof",
            "constrained_idx",
            "constrained_values",
        ),
        outputs=("solve_cg",),
    )
    def build_cg_solver(self, ctx: ProducerContext) -> None:
        from scipy.sparse.linalg import cg, LinearOperator

        n_dof = self.p2_space.n_dof.get()
        L = self.p2_space.L_stiffness.get()

        # Pre-factor and setup the CG solver
        constrained_idx = self.constrained_idx.get()
        constrained_values = self.constrained_values.get()
        is_constrained_mask = np.zeros(n_dof, dtype=bool)
        is_constrained_mask[constrained_idx] = True
        free_idx = np.nonzero(~is_constrained_mask)[0]

        # Extract blocks
        L_II = L[free_idx][:, free_idx].tocsr()
        L_IB = L[free_idx][:, constrained_idx].tocsr()

        # Jacobi pre-conditioning
        diag = L_II.diagonal().astype(np.float64)
        inv_diag = np.zeros_like(diag)
        nz = np.abs(diag) > 1e-14
        inv_diag[nz] = 1.0 / diag[nz]
        M = LinearOperator(
            shape=L_II.shape, matvec=lambda x: inv_diag * x, dtype=np.float64
        )

        def solve(b, x0=None, rtol=1e-8, atol=0.0, maxiter=None):
            b = np.asarray(b, dtype=np.float64)
            rhs = b[free_idx] - L_IB @ constrained_values

            x0_free = None
            if x0 is not None:
                x0 = np.array(x0, dtype=np.float64)
                x0_free = x0[free_idx]

            u_free, info = cg(
                L_II, rhs, x0=x0_free, rtol=rtol, atol=atol, maxiter=maxiter, M=M
            )

            if info != 0:
                raise RuntimeError(f"CG did not converge, info={info}")

            u = np.zeros(n_dof, dtype=np.float64)
            u[constrained_idx] = constrained_values
            u[free_idx] = u_free

            return u

        ctx.commit(solve_cg=solve)

    def interpolate(self, probles):
        """Interpolates the value of `psi` using P2 lagrange basis

        Args:
           probes (np.ndarray): [[faceid, [b1, b2, b3]], ...]
        """
        psi = self.psi.get()
        face_dof = self.p2_space.face_dof.get()

        values = np.zeros((len(probles),))

        for idx, (faceid, (b1, b2, b3)) in enumerate(probles):
            v1, v2, v3, e1, e2, e3 = face_dof[faceid]
            p1, p2, p3, p4, p5, p6 = self.p2_space.basis_from_bary(b1, b2, b3)

            values[idx] += (
                p1 * psi[v1]
                + p2 * psi[v2]
                + p3 * psi[v3]
                + p4 * psi[e1]
                + p5 * psi[e2]
                + p6 * psi[e3]
            )

        return values


class P2StreamFunction(ModuleBase):
    NAME = "P2StreamFunction"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.point_vortex = self.require(PointVortexModule)
        self.p2_elements = self.require(P2Elements)

        # Inputs
        self.constrained_idx = self.resource(
            "constrained_idx",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.int32,
            ),
            declare=True,
            doc="Indices of constrained vertices.",
        )
        self.constrained_values = self.resource(
            "constrained_values",
            spec=ResourceSpec(kind="numpy", dtype=np.float32),
            declare=True,
            doc="Values for the constrained vertices 1-1 mapped to the index in constrained_idx.",
        )

        # Derived
        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_scalar(self.p2_elements.n_dof),
            ),
            doc="Stream function coefficients for chosen p2 basis.",
        )

        self.omega = self.resource(
            "omega",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_from_scalar(self.p2_elements.n_dof),
            ),
            doc="Vorticity field coefficient to be paired with basis function. Shape: (nDof, )",
        )

        self.solve_cg = self.resource(
            "solve_cg",
            spec=ResourceSpec(kind="python"),
            doc="A pre-factorized poisson solver callable, just need the RHS",
        )

        self.bind_producers()

    @producer(inputs=("solve_cg", "omega"), outputs=("psi",))
    def solve_for_stream_func(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()
        solve_cg = self.solve_cg.get()
        omega = self.omega.get()

        ctx.commit(psi=solve_cg(omega))

    @producer(
        inputs=(
            "point_vortex.face_ids",
            "point_vortex.bary",
            "point_vortex.gamma",
            "p2_elements.face_dof",
        ),
        outputs=("omega",),
    )
    def splat_vortices(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()

        face_to_dof = self.p2_elements.face_dof.get()
        faceids = self.point_vortex.face_ids.get()
        bary = self.point_vortex.bary.get()
        gamma = self.point_vortex.gamma.get()
        omega = np.zeros_like(self.omega.peek())

        for idx, G in enumerate(gamma):
            fid = faceids[idx]
            # get dof nodes of a face
            v1, v2, v3, e1, e2, e3 = face_to_dof[fid]
            b1, b2, b3 = bary[idx]

            # P2 basis functions
            p1, p2, p3, p4, p5, p6 = self.p2_elements.basis_from_bary(b1, b2, b3)
            omega[v1] += p1 * G
            omega[v2] += p2 * G
            omega[v3] += p3 * G
            omega[e1] += p4 * G
            omega[e2] += p5 * G
            omega[e3] += p6 * G

        ctx.commit(omega=omega)

    @producer(
        inputs=(
            "p2_elements.L_stiffness",
            "p2_elements.n_dof",
            "constrained_idx",
            "constrained_values",
        ),
        outputs=("solve_cg",),
    )
    def build_cg_solver(self, ctx: ProducerContext) -> None:
        from scipy.sparse.linalg import cg, LinearOperator

        n_dof = self.p2_elements.n_dof.get()
        L = self.p2_elements.L_stiffness.get()

        # Pre-factor and setup the CG solver
        constrained_idx = self.constrained_idx.get()
        constrained_values = self.constrained_values.get()
        is_constrained_mask = np.zeros(n_dof, dtype=bool)
        is_constrained_mask[constrained_idx] = True
        free_idx = np.nonzero(~is_constrained_mask)[0]

        # Extract blocks
        L_II = L[free_idx][:, free_idx].tocsr()
        L_IB = L[free_idx][:, constrained_idx].tocsr()

        # Jacobi pre-conditioning
        diag = L_II.diagonal().astype(np.float64)
        inv_diag = np.zeros_like(diag)
        nz = np.abs(diag) > 1e-14
        inv_diag[nz] = 1.0 / diag[nz]
        M = LinearOperator(
            shape=L_II.shape, matvec=lambda x: inv_diag * x, dtype=np.float64
        )

        def solve(b, x0=None, rtol=1e-8, atol=0.0, maxiter=None):
            b = np.asarray(b, dtype=np.float64)
            rhs = b[free_idx] - L_IB @ constrained_values

            x0_free = None
            if x0 is not None:
                x0 = np.array(x0, dtype=np.float64)
                x0_free = x0[free_idx]

            u_free, info = cg(
                L_II, rhs, x0=x0_free, rtol=rtol, atol=atol, maxiter=maxiter, M=M
            )

            if info != 0:
                raise RuntimeError(f"CG did not converge, info={info}")

            u = np.zeros(n_dof, dtype=np.float64)
            u[constrained_idx] = constrained_values
            u[free_idx] = u_free

            return u

        ctx.commit(solve_cg=solve)

    def interpolate(self, probles):
        """Interpolates the value of `psi` using P2 lagrange basis

        Args:
           probes (np.ndarray): [[faceid, [b1, b2, b3]], ...]
        """
        psi = self.psi.get()
        face_dof = self.p2_elements.face_dof.get()

        values = np.zeros((len(probles),))

        for idx, (faceid, (b1, b2, b3)) in enumerate(probles):
            v1, v2, v3, e1, e2, e3 = face_dof[faceid]
            p1, p2, p3, p4, p5, p6 = self.p2_elements.basis_from_bary(b1, b2, b3)

            values[idx] += (
                p1 * psi[v1]
                + p2 * psi[v2]
                + p3 * psi[v3]
                + p4 * psi[e1]
                + p5 * psi[e2]
                + p6 * psi[e3]
            )

        return values
