from typing import List

import numpy as np

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.p1_space.p1_velocity import (
    area_weighted_face_vectors_to_vertices,
)
from rheidos.apps.p2.modules.p1_space.probe_utils import probe_arrays
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import shape_map
from rheidos.compute.resource import ResourceSpec
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import ModuleBase, World


class HarmonicBasisModule(ModuleBase):
    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        dec: DEC,
        harmonic_dim: int,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        self.dec = dec
        self.dim = harmonic_dim

        self.poisson = self.require(
            P1PoissonSolver,
            child=True,
            child_name="poisson_solver",
            mesh=self.mesh,
            dec=self.dec,
            declare_rhs=False,
        )

        # we just need stream functions as basis field is Jgrad(psi_h)
        self.basis = self.resource(
            "basis",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.V_pos, lambda s: (self.dim, s[0])),
            ),
            doc="Basis stream function we need to get harmonic vector fields. Shape: (N, nV) where N is dimension of the harmonic subspace",
        )

        self.bind_producers()

    @producer(inputs=("mesh.V_pos",), outputs=("basis",))
    def solve_for_harmonic_stream_function(self, ctx: ProducerContext):

        if self.inner_boundaries is None or self.outer_boundary is None:
            raise RuntimeError(
                f"Boundaries components are not set, call `set_boundaries` with appropriate boundaries before"
            )

        ctx.require_inputs()

        psi_h = list()

        # The RHS of the poisson solve is all 0 for getting the harmonic field
        nV = self.mesh.V_pos.get().shape[0]
        rhs = np.zeros((nV,), dtype=np.float64)

        for i in range(len(self.inner_boundaries)):
            constrained_idx = []
            constrained_values = []

            # Set one inner boundary component 1 and other all 0 at
            # a time
            for j in range(len(self.inner_boundaries)):
                inner_b = self.inner_boundaries[j]
                is_current = int(j == i)

                constrained_idx.extend(inner_b)
                constrained_values.extend([is_current] * len(inner_b))

            constrained_idx.extend(self.outer_boundary)
            constrained_values.extend([0] * len(self.outer_boundary))

            # Set constraints and solve for psi_h
            self.poisson.constrained_idx.set(np.array(constrained_idx, dtype=np.int32))
            self.poisson.constrained_values.set(
                np.array(constrained_values, dtype=np.float64)
            )

            # Solve for psi_h
            solve = self.poisson.solve_cg.get()
            psi_h.append(solve(rhs))

        ctx.commit(basis=np.array(psi_h))

    def set_boudaries(self, inner: List[int], outer: int):
        """This function expects the interger ids of sets in the
        `mesh.boundary_vertex_components` which are classied as inner
        and outer.

        Args:
            inner (List[int]): List of indexes belonging to the `mesh.boundary_vertex_components`
            outer (int): Index considered as outer belonging to `mesh.boundary_vertex_components`
        """

        boundary_components = self.mesh.boundary_vertex_components.get()

        if len(inner) != self.dim:
            raise RuntimeError(
                f"Count of internal boundaries and dimension of harmonic subspace don't match"
            )

        # segregate boundaries
        self.inner_boundaries = list(map(lambda idx: boundary_components[idx], inner))
        self.outer_boundary = boundary_components[outer]

    def interpolate(self, probes, basis_id=0):
        """Interpolates the value of `psi` based on the P1 basis.

        Args:
            probes:
                Either an iterable of ``(faceid, bary)`` pairs or a
                ``(faceids, bary)`` tuple of arrays.

        Returns:
            np.ndarray: Values at the probe locations
        """

        if not (0 <= basis_id < self.dim):
            raise RuntimeError(f"`basis_id` has to be in range: [0, {self.dim-1}]")

        psi = self.basis.get()[basis_id]
        F_verts = self.mesh.F_verts.get()

        if isinstance(probes, tuple) and len(probes) == 2:
            faceids = np.asarray(probes[0], dtype=np.int64)
            bary = np.asarray(probes[1], dtype=np.float64)
        else:
            faceids, bary = probe_arrays(probes)

        return np.einsum("ij,ij->i", psi[F_verts[faceids]], bary)


class HarmonicBasisFieldModule(ModuleBase):
    NAME = "HarmonicBasisFieldModule"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        dec: DEC,
        harmonic_basis: HarmonicBasisModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        self.dec = dec
        self.harmonic_basis = harmonic_basis

        self.vel_per_face = self.resource(
            "vel_per_face",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(
                    self.mesh.F_verts,
                    lambda s: (self.harmonic_basis.dim, s[0], 3),
                ),
            ),
            doc=(
                "Facewise constant velocity Jgrad(psi_h) for each harmonic "
                "basis. Shape: (N, nF, 3)"
            ),
        )

        self.vel_per_vertex = self.resource(
            "vel_per_vertex",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(
                    self.mesh.V_pos,
                    lambda s: (self.harmonic_basis.dim, s[0], 3),
                ),
            ),
            doc=(
                "Per-vertex area-weighted harmonic basis velocity in R^3. "
                "Shape: (N, nV, 3)"
            ),
        )

        self.bind_producers()

    def _check_basis_id(self, basis_id: int) -> int:
        basis_id = int(basis_id)
        if not (0 <= basis_id < self.harmonic_basis.dim):
            raise RuntimeError(
                f"`basis_id` has to be in range: [0, {self.harmonic_basis.dim - 1}]"
            )
        return basis_id

    @producer(
        inputs=("vel_per_face", "mesh.F_area", "mesh.F_verts", "mesh.V_pos"),
        outputs=("vel_per_vertex",),
    )
    def per_vertex_vel_calculate(self, ctx: ProducerContext):
        ctx.require_inputs()

        f_vel = self.vel_per_face.get()
        f_area = self.mesh.F_area.get()
        f_verts = self.mesh.F_verts.get()
        n_vertices = self.mesh.V_pos.get().shape[0]

        if self.harmonic_basis.dim == 0:
            ctx.commit(vel_per_vertex=np.empty((0, n_vertices, 3), dtype=np.float64))
            return

        ctx.commit(
            vel_per_vertex=np.stack(
                [
                    area_weighted_face_vectors_to_vertices(
                        f_vel[basis_id],
                        f_area,
                        f_verts,
                        n_vertices,
                    )
                    for basis_id in range(self.harmonic_basis.dim)
                ],
                axis=0,
            )
        )

    @producer(
        inputs=(
            "mesh.F_verts",
            "mesh.F_normal",
            "mesh.grad_bary",
            "harmonic_basis.basis",
        ),
        outputs=("vel_per_face",),
    )
    def per_face_vel_calculate(self, ctx: ProducerContext):
        ctx.require_inputs()

        coeffs = self.harmonic_basis.basis.get()[:, self.mesh.F_verts.get()]
        # grad_bary stores [grad lambda1, grad lambda2, grad lambda3] as rows.
        j_grad = np.cross(
            self.mesh.F_normal.get()[:, None, :],
            self.mesh.grad_bary.get(),
        )

        ctx.commit(vel_per_face=np.einsum("kfa,fai->kfi", coeffs, j_grad))

    def interpolate(self, probes, smooth=True, basis_id=0):
        """Calculates and returns velocity for one harmonic basis field.

        Args:
           probes (np.ndarray): [[faceid, [b1, b2, b3]], ...]
           smooth (bool): If true, interpolate area-weighted per-vertex
               velocities. Otherwise, return facewise constant velocities.
           basis_id (int): Harmonic basis index to sample.
        """
        basis_id = self._check_basis_id(basis_id)
        faceids, bary = probe_arrays(probes)
        if faceids.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        if not smooth:
            return self.vel_per_face.get()[basis_id, faceids]

        verts = self.mesh.F_verts.get()[faceids]
        vel_verts = self.vel_per_vertex.get()[basis_id, verts]

        b1, b2, b3 = map(lambda x: x.reshape(-1, 1), bary.T)
        v1, v2, v3 = vel_verts[:, 0, :], vel_verts[:, 1, :], vel_verts[:, 2, :]

        return b1 * v1 + b2 * v2 + b3 * v3
