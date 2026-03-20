from typing import List, Tuple

import numpy as np

from rheidos.apps.p2.modules.point_vortex import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh import SurfaceMeshModule
from rheidos.compute import ModuleBase, ResourceSpec, World, shape_map
from rheidos.compute import producer, ProducerContext


def _cot_at(x1, x2, o) -> float:
    e1 = o - x1
    e2 = o - x2
    return np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))


class DEC(ModuleBase):
    NAME = "DEC"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)

        self.star1 = self.resource(
            "star1",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float32,
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
        E_faces = self.mesh.E_faces.get()
        E_verts = self.mesh.E_verts.get()
        V_pos = self.mesh.V_pos.get()
        nV = len(V_pos)

        mask = np.zeros(nV, dtype=bool)
        for edge_id, faces in enumerate(E_faces):
            if -1 in faces:  # boundary edge
                v1, v2 = E_verts[edge_id]
                mask[v1] = True
                mask[v2] = True

        ctx.commit(boundary_mask=mask)

    @producer(inputs=("mesh.V_pos", "mesh.E_verts", "mesh.E_opp"), outputs=("star1",))
    def build_cotan_star1(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()

        E_opp = self.mesh.E_opp.get()
        E_verts = self.mesh.E_verts.get()
        V_pos = self.mesh.V_pos.get()
        star1 = np.zeros_like(self.star1.peek())
        for edge_id, (o1, o2) in enumerate(E_opp):
            v1, v2 = E_verts[edge_id]
            x1 = V_pos[v1]
            x2 = V_pos[v2]
            if o1 >= 0:
                star1[edge_id] += 0.5 * _cot_at(x1, x2, V_pos[o1])
            if o2 >= 0:
                star1[edge_id] += 0.5 * _cot_at(x1, x2, V_pos[o2])

        ctx.commit(star1=star1)


class P1PoissonSolver(ModuleBase):
    NAME = "P1PoissonSolver"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(DEC)

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
                allow_none=True,
                shape_fn=shape_map(self.mesh.V_pos, lambda shape: (shape[0],)),
            ),
            declare=True,
            doc="Vorticity field coefficient to be paired with basis function. Shape: (nV, )",
        )

        # Derived

        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float32,
                allow_none=True,
                shape_fn=shape_map(self.mesh.V_pos, lambda shape: (shape[0],)),
            ),
            doc="Stream function coefficient to be paired with respective hat basis function. Shape (nV, )",
        )

        self.L_cached = self.resource(
            "L_cached",
            spec=ResourceSpec(kind="python", allow_none=True),
            doc="Sparse SciPy scalar Laplacian matrix based on DEC * d *",
        )

        self.solve_cg = self.resource(
            "solve_cg",
            spec=ResourceSpec(kind="python"),
            doc="A pre-factorized poisson solver callable, just need the RHS",
        )

        self.bind_producers()

    @producer(inputs=("solve_cg", "rhs"), outputs=("psi",))
    def solve_for_stream_func(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()
        solve_cg = self.solve_cg.get()
        rhs = self.rhs.get()

        ctx.commit(psi=solve_cg(rhs).astype(np.float32))

    @producer(
        inputs=(
            "mesh.V_pos",
            "mesh.E_verts",
            "dec.star1",
            "constrained_idx",
            "constrained_values",
        ),
        outputs=("L_cached", "solve_cg"),
    )
    def build_scalar_laplacian(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()

        import scipy.sparse as sp
        from scipy.sparse.linalg import cg, LinearOperator

        # Build Laplacian
        star1 = self.dec.star1.get()
        E = self.mesh.E_verts.get()
        V_pos = self.mesh.V_pos.get()

        i = E[:, 0].astype(np.int64, copy=False)
        j = E[:, 1].astype(np.int64, copy=False)
        ww = np.asarray(star1, dtype=np.float32).reshape(-1)

        rows = np.concatenate([i, j, i, j])
        cols = np.concatenate([i, j, j, i])
        data = np.concatenate([ww, ww, -ww, -ww])
        nV = V_pos.shape[0]

        L = sp.coo_matrix((data, (rows, cols)), shape=(nV, nV)).tocsr()

        # Pre-factor and setup the CG solver
        constrained_idx = self.constrained_idx.get()
        constrained_values = self.constrained_values.get()
        is_constrained_mask = np.zeros(nV, dtype=bool)
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

            u = np.zeros(nV, dtype=np.float64)
            u[constrained_idx] = constrained_values
            u[free_idx] = u_free

            return u

        ctx.commit(L_cached=L, solve_cg=solve)

    def interpolate(self, probes) -> np.ndarray:
        """Interpolates the value of `psi` based on P1 lagrange basis

        Args:
            probes (np.ndarray): [[faceid, [b1, b2, b3]], ...]

        Returns:
            np.ndarray: Values at the probe locations
        """
        psi = self.psi.get()
        F_verts = self.mesh.F_verts.get()

        values = np.zeros((len(probes),))

        # TODO: Vectorize the for loop
        for idx, (faceid, (b1, b2, b3)) in enumerate(probes):
            v1, v2, v3 = F_verts[faceid]
            values[idx] += b1 * psi[v1] + b2 * psi[v2] + b3 * psi[v3]

        return values


class P1StreamFunction(ModuleBase):
    NAME = "P1StreamFunction"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = self.require(SurfaceMeshModule)
        self.point_vortex = self.require(PointVortexModule)
        self.dec = self.require(DEC)

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
                dtype=np.float32,
                allow_none=True,
                shape_fn=shape_map(self.mesh.V_pos, lambda shape: (shape[0],)),
            ),
            doc="Stream function coefficient to be paired with respective hat basis function. Shape (nV, )",
        )

        self.omega = self.resource(
            "omega",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float32,
                allow_none=True,
                shape_fn=shape_map(self.mesh.V_pos, lambda shape: (shape[0],)),
            ),
            doc="Vorticity field coefficient to be paired with basis function. Shape: (nV, )",
        )

        self.L_cached = self.resource(
            "L_cached",
            spec=ResourceSpec(kind="python", allow_none=True),
            doc="Sparse SciPy scalar Laplacian matrix based on DEC * d *",
        )

        self.solve_cg = self.resource(
            "solve_cg",
            spec=ResourceSpec(kind="python"),
            doc="A pre-factorized poisson solver callable, just need the RHS",
        )

        self.bind_producers()

    @producer(inputs=("solve_cg",), outputs=("psi",))
    def solve_for_stream_func(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()
        solve_cg = self.solve_cg.get()
        omega = self.omega.get()

        ctx.commit(psi=solve_cg(omega).astype(np.float32))

    @producer(
        inputs=(
            "mesh.V_pos",
            "mesh.E_verts",
            "dec.star1",
            "constrained_idx",
            "constrained_values",
        ),
        outputs=("L_cached", "solve_cg"),
    )
    def build_scalar_laplacian(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()

        import scipy.sparse as sp
        from scipy.sparse.linalg import cg, LinearOperator

        # Build Laplacian
        star1 = self.dec.star1.get()
        E = self.mesh.E_verts.get()
        V_pos = self.mesh.V_pos.get()

        i = E[:, 0].astype(np.int64, copy=False)
        j = E[:, 1].astype(np.int64, copy=False)
        ww = np.asarray(star1, dtype=np.float32).reshape(-1)

        rows = np.concatenate([i, j, i, j])
        cols = np.concatenate([i, j, j, i])
        data = np.concatenate([ww, ww, -ww, -ww])
        nV = V_pos.shape[0]

        L = sp.coo_matrix((data, (rows, cols)), shape=(nV, nV)).tocsr()

        # Pre-factor and setup the CG solver
        constrained_idx = self.constrained_idx.get()
        constrained_values = self.constrained_values.get()
        is_constrained_mask = np.zeros(nV, dtype=bool)
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

            u = np.zeros(nV, dtype=np.float64)
            u[constrained_idx] = constrained_values
            u[free_idx] = u_free

            return u

        ctx.commit(L_cached=L, solve_cg=solve)

    @producer(
        inputs=(
            "point_vortex.face_ids",
            "point_vortex.bary",
            "point_vortex.gamma",
            "mesh.F_verts",
        ),
        outputs=("omega",),
    )
    def splat_vortices(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.ensure_outputs()

        F_verts = self.mesh.F_verts.get()
        faceids = self.point_vortex.face_ids.get()
        bary = self.point_vortex.bary.get()
        gamma = self.point_vortex.gamma.get()
        omega = np.zeros_like(self.omega.peek())

        for idx, G in enumerate(gamma):
            fid = faceids[idx]
            v1, v2, v3 = F_verts[fid]
            b1, b2, b3 = bary[idx]
            omega[v1] += b1 * G
            omega[v2] += b2 * G
            omega[v3] += b3 * G

        ctx.commit(omega=omega)

    def interpolate(self, probes) -> np.ndarray:
        """Interpolates the value of `psi` based on P1 lagrange basis

        Args:
            probes (np.ndarray): [[faceid, [b1, b2, b3]], ...]

        Returns:
            np.ndarray: Values at the probe locations
        """
        psi = self.psi.get()
        F_verts = self.mesh.F_verts.get()

        values = np.zeros((len(probes),))

        # TODO: Vectorize the for loop
        for idx, (faceid, (b1, b2, b3)) in enumerate(probes):
            v1, v2, v3 = F_verts[faceid]
            values[idx] += b1 * psi[v1] + b2 * psi[v2] + b3 * psi[v3]

        return values
