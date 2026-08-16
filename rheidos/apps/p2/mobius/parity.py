"""Deck-parity-aware P1 fields on the orientation double cover."""

from __future__ import annotations

import numpy as np

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.p1_space.probe_utils import probe_arrays
from rheidos.apps.p2.modules.p1_space.p1_velocity import (
    area_weighted_face_vectors_to_vertices,
)
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import (
    ModuleBase,
    ProducerContext,
    ResourceSpec,
    World,
    producer,
    shape_map,
)

from .cover_vortices import DoubleCoverPointVortex
from .orientation_double_cover import OrientationDoubleCover


class DeckParityError(RuntimeError):
    """A lifted field failed its required deck-transformation parity."""


def validate_deck_parity(
    values: np.ndarray,
    tau: np.ndarray,
    *,
    parity: int,
    name: str,
    rtol: float = 1.0e-7,
    atol: float = 1.0e-10,
) -> None:
    """Require ``values[tau] == parity * values`` along the first axis."""
    values = np.asarray(values, dtype=np.float64)
    residual = values[tau] - parity * values
    error = float(np.max(np.abs(residual), initial=0.0))
    scale = float(np.max(np.abs(values), initial=0.0))
    if error > atol + rtol * scale:
        parity_name = "even" if parity == 1 else "odd"
        raise DeckParityError(
            f"{name} is not deck-{parity_name}: maximum residual {error:.3e}"
        )


def project_deck_parity(
    values: np.ndarray,
    tau: np.ndarray,
    *,
    parity: int,
) -> np.ndarray:
    """Return the exact even or odd component of a lifted field."""
    values = np.asarray(values, dtype=np.float64)
    return 0.5 * (values + parity * values[tau])


class DeckOddP1StreamFunction(ModuleBase):
    """P1 Poisson solve with odd deck parity enforced on RHS and solution.

    The Mobius stream function is a pseudoscalar.  On its orientation cover it
    therefore satisfies ``psi[tau_vertex] = -psi``.  Enforcing that invariant
    here prevents small solver drift from entering velocity interpolation and
    catches asymmetric constraints or incorrectly paired vortices early.
    """

    NAME = "DeckOddP1StreamFunction"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        dec: DEC,
        point_vortex: DoubleCoverPointVortex,
        double_cover: OrientationDoubleCover,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        self.mesh = mesh
        self.dec = dec
        self.point_vortex = point_vortex
        self.cover = double_cover
        self.poisson = self.require(
            P1PoissonSolver,
            child=True,
            child_name="poisson",
            mesh=mesh,
            dec=dec,
            declare_rhs=False,
        )

        # The custom splat producer writes the Poisson RHS directly.  The
        # public psi is separate from poisson.psi so parity can be checked and
        # projected before any velocity module observes it.
        self.omega = self.poisson.rhs
        self.raw_psi = self.poisson.psi
        self.psi = self.resource(
            "psi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.V_pos, lambda shape: (shape[0],)),
            ),
            doc="Exactly deck-odd P1 stream function. Shape: (nV,)",
        )

        self.constrained_idx = self.poisson.constrained_idx
        self.constrained_values = self.poisson.constrained_values
        self.L_cached = self.poisson.L_cached
        self.solve_cg = self.poisson.solve_cg
        self.bind_producers()

    @producer(
        inputs=(
            "point_vortex.face_ids",
            "point_vortex.bary",
            "point_vortex.gamma",
            "mesh.F_verts",
            "mesh.V_pos",
            "cover.tau_vertex",
        ),
        outputs=("omega",),
    )
    def splat_odd_vorticity(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        face_ids = ctx.inputs.point_vortex.face_ids.get()
        bary = ctx.inputs.point_vortex.bary.get()
        gamma = ctx.inputs.point_vortex.gamma.get()
        face_vertices = ctx.inputs.mesh.F_verts.get()[face_ids]
        tau_vertex = ctx.inputs.cover.tau_vertex.get()

        omega = np.zeros(ctx.inputs.mesh.V_pos.get().shape[0], dtype=np.float64)
        np.add.at(
            omega,
            face_vertices.reshape(-1),
            (bary * gamma[:, None]).reshape(-1),
        )

        # The 2N vortex pairing should already make this odd.  Validate before
        # projection so a pairing or corner-map regression cannot be hidden.
        validate_deck_parity(
            omega,
            tau_vertex,
            parity=-1,
            name="P1 vorticity RHS",
            rtol=0.0,
            atol=1.0e-12,
        )
        omega = project_deck_parity(omega, tau_vertex, parity=-1)
        validate_deck_parity(
            omega,
            tau_vertex,
            parity=-1,
            name="projected P1 vorticity RHS",
            rtol=0.0,
            atol=0.0,
        )
        ctx.commit(omega=omega)

    @producer(
        inputs=("poisson.psi", "cover.tau_vertex"),
        outputs=("psi",),
    )
    def enforce_odd_stream_function(self, ctx: ProducerContext) -> None:
        raw_psi = ctx.inputs.poisson.psi.get()
        tau_vertex = ctx.inputs.cover.tau_vertex.get()

        # A deck-equivariant Laplacian with symmetric constraints preserves
        # oddness.  A large pre-projection residual signals an architectural
        # error; the final projection removes only iterative solver noise.
        validate_deck_parity(
            raw_psi,
            tau_vertex,
            parity=-1,
            name="raw P1 stream function",
        )
        psi = project_deck_parity(raw_psi, tau_vertex, parity=-1)
        validate_deck_parity(
            psi,
            tau_vertex,
            parity=-1,
            name="P1 stream function",
            rtol=0.0,
            atol=0.0,
        )
        ctx.commit(psi=psi)

    def set_homo_dirichlet_boundary(self) -> None:
        boundary = self.mesh.boundary_vertex_ids.get()
        tau_vertex = self.cover.tau_vertex.get()

        if boundary.size:
            # A physical cover boundary must be closed under the deck map.  If
            # it is not, zero Dirichlet constraints would break parity.
            if not np.array_equal(np.sort(tau_vertex[boundary]), np.sort(boundary)):
                raise DeckParityError("Cover boundary is not invariant under tau_vertex")
            constrained = boundary
        else:
            # On a closed cover, pin a complete deck pair to preserve the odd
            # subspace while removing the scalar Laplacian's constant mode.
            constrained = np.unique(np.array([0, tau_vertex[0]], dtype=np.int32))

        self.constrained_idx.set(np.asarray(constrained, dtype=np.int32))
        self.constrained_values.set(np.zeros(constrained.shape[0], dtype=np.float64))

    def interpolate(self, probes) -> np.ndarray:
        face_ids, bary = probe_arrays(probes)
        coefficients = self.psi.get()[self.mesh.F_verts.get()[face_ids]]
        return np.einsum("ni,ni->n", bary, coefficients)


class DeckEvenP1VelocityField(ModuleBase):
    """Velocity derived from odd psi, enforced even under the deck map.

    Deck-related faces have opposite normals while the odd stream function has
    opposite gradients.  The two signs cancel in ``n x grad(psi)``, so the R3
    velocity vectors must agree at deck-paired points.
    """

    NAME = "DeckEvenP1VelocityField"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        stream: DeckOddP1StreamFunction,
        double_cover: OrientationDoubleCover,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        self.mesh = mesh
        self.stream = stream
        self.cover = double_cover
        self.vel_per_face = self.resource(
            "vel_per_face",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.F_verts, lambda shape: (shape[0], 3)),
            ),
            doc="Exactly deck-even facewise tangent velocity. Shape: (nF,3)",
        )
        self.vel_per_vertex = self.resource(
            "vel_per_vertex",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.V_pos, lambda shape: (shape[0], 3)),
            ),
            doc="Exactly deck-even vertex velocity. Shape: (nV,3)",
        )
        self.bind_producers()

    @producer(
        inputs=(
            "mesh.F_verts",
            "mesh.F_normal",
            "mesh.grad_bary",
            "stream.psi",
            "cover.tau_face",
        ),
        outputs=("vel_per_face",),
    )
    def calculate_face_velocity(self, ctx: ProducerContext) -> None:
        faces = ctx.inputs.mesh.F_verts.get()
        coefficients = ctx.inputs.stream.psi.get()[faces]
        rotated_gradients = np.cross(
            ctx.inputs.mesh.F_normal.get()[:, None, :],
            ctx.inputs.mesh.grad_bary.get(),
        )
        velocity = np.einsum("fi,fij->fj", coefficients, rotated_gradients)
        tau_face = ctx.inputs.cover.tau_face.get()

        validate_deck_parity(
            velocity,
            tau_face,
            parity=1,
            name="raw P1 face velocity",
        )
        velocity = project_deck_parity(velocity, tau_face, parity=1)
        validate_deck_parity(
            velocity,
            tau_face,
            parity=1,
            name="P1 face velocity",
            rtol=0.0,
            atol=0.0,
        )
        ctx.commit(vel_per_face=velocity)

    @producer(
        inputs=(
            "vel_per_face",
            "mesh.F_area",
            "mesh.F_verts",
            "mesh.V_pos",
            "cover.tau_vertex",
        ),
        outputs=("vel_per_vertex",),
    )
    def calculate_vertex_velocity(self, ctx: ProducerContext) -> None:
        velocity = area_weighted_face_vectors_to_vertices(
            ctx.inputs.vel_per_face.get(),
            ctx.inputs.mesh.F_area.get(),
            ctx.inputs.mesh.F_verts.get(),
            ctx.inputs.mesh.V_pos.get().shape[0],
        )
        tau_vertex = ctx.inputs.cover.tau_vertex.get()
        validate_deck_parity(
            velocity,
            tau_vertex,
            parity=1,
            name="raw P1 vertex velocity",
        )
        velocity = project_deck_parity(velocity, tau_vertex, parity=1)
        validate_deck_parity(
            velocity,
            tau_vertex,
            parity=1,
            name="P1 vertex velocity",
            rtol=0.0,
            atol=0.0,
        )
        ctx.commit(vel_per_vertex=velocity)

    def interpolate(self, probes, smooth: bool = True) -> np.ndarray:
        face_ids, bary = probe_arrays(probes)
        if face_ids.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if not smooth:
            return self.vel_per_face.get()[face_ids]

        vertex_ids = self.mesh.F_verts.get()[face_ids]
        vertex_velocity = self.vel_per_vertex.get()[vertex_ids]
        return np.einsum("ni,nij->nj", bary, vertex_velocity)


__all__ = [
    "DeckEvenP1VelocityField",
    "DeckOddP1StreamFunction",
    "DeckParityError",
    "project_deck_parity",
    "validate_deck_parity",
]
