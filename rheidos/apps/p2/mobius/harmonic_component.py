"""Harmonic velocity and Abel--Jacobi state on the Mobius double cover."""

from __future__ import annotations

import numpy as np

from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_poisson_solver import P1PoissonSolver
from rheidos.apps.p2.modules.p1_space.probe_utils import probe_arrays
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
from .parity import project_deck_parity, validate_deck_parity


DEFAULT_HARMONIC_COEFFICIENT = 0.0


class MobiusHarmonicBasis(ModuleBase):
    """Precompute the one-dimensional harmonic basis on the cylinder cover.

    The scalar potential ``U`` is fixed to +/- 1/2 on the two deck-paired
    boundary components.  Its exact one-form ``xi = d0 U`` determines the
    normalized physical velocity basis ``zeta = -J grad(U) / <xi, xi>``.
    """

    NAME = "MobiusHarmonicBasis"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        dec: DEC,
        double_cover: OrientationDoubleCover,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        self.mesh = mesh
        self.dec = dec
        self.cover = double_cover

        self.poisson = self.require(
            P1PoissonSolver,
            child=True,
            child_name="poisson",
            mesh=self.mesh,
            dec=self.dec,
            declare_rhs=False,
        )
        self.rhs = self.poisson.rhs
        self.raw_potential = self.poisson.psi

        self.potential = self.resource(
            "potential",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.V_pos, lambda shape: (shape[0],)),
            ),
            doc="Exactly deck-odd harmonic potential U. Shape: (nV,)",
        )
        self.xi = self.resource(
            "xi",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.E_verts, lambda shape: (shape[0],)),
            ),
            doc="Exact harmonic one-form xi=d0 U. Shape: (nE,)",
        )
        self.energy = self.resource(
            "energy",
            spec=ResourceSpec(kind="python", dtype=float),
            doc="Harmonic one-form energy xi^T star1 xi.",
        )
        self.zeta_face = self.resource(
            "zeta_face",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.F_verts, lambda shape: (shape[0], 3)),
            ),
            doc="Deck-even facewise harmonic velocity basis. Shape: (nF,3)",
        )
        self.bind_producers()

    def configure_boundary_pair(self) -> None:
        """Assign a deterministic +/- 1/2 value to the paired boundaries."""
        components = self.mesh.boundary_vertex_components.get()
        if len(components) != 2:
            raise RuntimeError(
                "The Mobius cylinder cover must have exactly two boundary components"
            )

        # The sign of a one-dimensional basis is arbitrary.  Choosing the
        # component with the smallest vertex id makes that sign reproducible.
        positive_index = min(
            range(2),
            key=lambda index: int(np.min(components[index])),
        )
        positive = components[positive_index]
        negative = components[1 - positive_index]
        tau_vertex = self.cover.tau_vertex.get()
        if not np.array_equal(np.sort(tau_vertex[positive]), np.sort(negative)):
            raise RuntimeError(
                "The two cover boundaries are not exchanged by the deck map"
            )

        constrained_idx = np.concatenate((positive, negative)).astype(
            np.int32,
            copy=False,
        )
        constrained_values = np.concatenate(
            (
                np.full(positive.shape, 0.5, dtype=np.float64),
                np.full(negative.shape, -0.5, dtype=np.float64),
            )
        )
        self.poisson.constrained_idx.set(np.ascontiguousarray(constrained_idx))
        self.poisson.constrained_values.set(
            np.ascontiguousarray(constrained_values)
        )

    @producer(inputs=("mesh.V_pos",), outputs=("rhs",))
    def fill_zero_rhs(self, ctx: ProducerContext) -> None:
        vertex_count = ctx.inputs.mesh.V_pos.get().shape[0]
        ctx.commit(rhs=np.zeros(vertex_count, dtype=np.float64))

    @producer(
        inputs=("poisson.psi", "cover.tau_vertex"),
        outputs=("potential",),
    )
    def enforce_odd_potential(self, ctx: ProducerContext) -> None:
        raw_potential = ctx.inputs.poisson.psi.get()
        tau_vertex = ctx.inputs.cover.tau_vertex.get()
        validate_deck_parity(
            raw_potential,
            tau_vertex,
            parity=-1,
            name="raw Mobius harmonic potential",
        )
        potential = project_deck_parity(
            raw_potential,
            tau_vertex,
            parity=-1,
        )
        ctx.commit(potential=potential)

    @producer(
        inputs=(
            "potential",
            "dec.star1",
            "mesh.F_verts",
            "mesh.F_normal",
            "mesh.grad_bary",
            "cover.tau_face",
        ),
        outputs=("xi", "energy", "zeta_face"),
    )
    def build_velocity_basis(self, ctx: ProducerContext) -> None:
        potential = ctx.inputs.potential.get()
        xi = self.dec.d0(potential)
        energy = float(np.dot(xi, ctx.inputs.dec.star1.get() * xi))
        if not np.isfinite(energy) or energy <= 0.0:
            raise RuntimeError("The Mobius harmonic basis has non-positive energy")

        face_coefficients = potential[ctx.inputs.mesh.F_verts.get()]
        gradient = np.einsum(
            "fi,fij->fj",
            face_coefficients,
            ctx.inputs.mesh.grad_bary.get(),
        )
        zeta_face = -np.cross(ctx.inputs.mesh.F_normal.get(), gradient) / energy
        tau_face = ctx.inputs.cover.tau_face.get()
        validate_deck_parity(
            zeta_face,
            tau_face,
            parity=1,
            name="raw Mobius harmonic velocity basis",
        )
        zeta_face = project_deck_parity(zeta_face, tau_face, parity=1)
        ctx.commit(xi=xi, energy=energy, zeta_face=zeta_face)

    def interpolate_potential(self, probes) -> np.ndarray:
        """Evaluate the P1 potential at barycentric particle locations."""
        face_ids, bary = probe_arrays(probes)
        coefficients = self.potential.get()[self.mesh.F_verts.get()[face_ids]]
        return np.einsum("ni,ni->n", bary, coefficients)

    def interpolate_velocity(self, probes) -> np.ndarray:
        """Sample the unsmoothed, facewise-constant harmonic velocity basis."""
        face_ids, _ = probe_arrays(probes)
        return self.zeta_face.get()[face_ids]


class MobiusHarmonicComponent(ModuleBase):
    """Own accepted harmonic state and derive the current Abel--Jacobi data."""

    NAME = "MobiusHarmonicComponent"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        point_vortex: DoubleCoverPointVortex,
        basis: MobiusHarmonicBasis,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        self.mesh = mesh
        self.point_vortex = point_vortex
        self.basis = basis

        self.harmonic_coefficient = self.resource(
            "harmonic_coefficient",
            spec=ResourceSpec(kind="python", dtype=float),
            buffer=DEFAULT_HARMONIC_COEFFICIENT,
            declare=True,
            doc="Harmonic coefficient of the last accepted simulation state.",
        )
        self.abel_jacobi_coordinate = self.resource(
            "abel_jacobi_coordinate",
            spec=ResourceSpec(kind="python", dtype=float),
            buffer=0.0,
            declare=True,
            doc="Abel--Jacobi coordinate A(p) of the last accepted state.",
        )
        self.abel_jacobi_step_delta = self.resource(
            "abel_jacobi_step_delta",
            spec=ResourceSpec(kind="python", dtype=float),
            buffer=0.0,
            declare=True,
            doc="A(p_next)-A(p0) for the last accepted RK step.",
        )
        self.abel_jacobi_invariant = self.resource(
            "abel_jacobi_invariant",
            spec=ResourceSpec(kind="python", dtype=float),
            doc="Conserved quantity A(p)-c for the accepted state.",
        )
        self.bind_producers()

    def evaluate_coordinate(self) -> float:
        """Evaluate A(p) for the currently installed, possibly trial, state."""
        face_ids = self.point_vortex.face_ids.get()
        bary = self.point_vortex.bary.get()
        gamma = self.point_vortex.gamma.get()
        values_at_vortices = self.basis.interpolate_potential((face_ids, bary))
        # point_vortex is exactly the interleaved 2N set used by the stream
        # Poisson RHS.  Both members of a deck pair are included here, so no
        # additional factor of two belongs in Equation (36).
        return float(np.dot(gamma, values_at_vortices))

    def refresh_accepted_coordinate(self) -> float:
        """Publish A(p) after installing an accepted particle configuration."""
        coordinate = self.evaluate_coordinate()
        self.abel_jacobi_coordinate.set(coordinate)
        return coordinate

    @producer(
        inputs=("abel_jacobi_coordinate", "harmonic_coefficient"),
        outputs=("abel_jacobi_invariant",),
    )
    def evaluate_invariant(self, ctx: ProducerContext) -> None:
        coordinate = float(ctx.inputs.abel_jacobi_coordinate.get())
        coefficient = float(ctx.inputs.harmonic_coefficient.get())
        ctx.commit(abel_jacobi_invariant=coordinate - coefficient)

    def initialize(
        self,
        coefficient: float = DEFAULT_HARMONIC_COEFFICIENT,
    ) -> None:
        self.harmonic_coefficient.set(float(coefficient))
        self.abel_jacobi_step_delta.set(0.0)
        self.refresh_accepted_coordinate()

    def stage_coefficient(
        self,
        reference_coordinate: float,
        reference_coefficient: float,
    ) -> float:
        """Apply Equation (36) to the currently installed RK-stage state."""
        current_coordinate = self.evaluate_coordinate()
        return float(reference_coefficient + current_coordinate - reference_coordinate)

    def interpolate(self, probes, coefficient: float | None = None) -> np.ndarray:
        if coefficient is None:
            coefficient = float(self.harmonic_coefficient.get())
        return float(coefficient) * self.basis.interpolate_velocity(probes)

    def commit_accepted_step(
        self,
        reference_coordinate: float,
        reference_coefficient: float,
    ) -> tuple[float, float]:
        """Commit Equation (36) after the accepted particles are installed."""
        current_coordinate = self.refresh_accepted_coordinate()
        delta = current_coordinate - float(reference_coordinate)
        coefficient = float(reference_coefficient) + delta
        self.harmonic_coefficient.set(coefficient)
        self.abel_jacobi_step_delta.set(delta)
        return coefficient, delta


__all__ = [
    "DEFAULT_HARMONIC_COEFFICIENT",
    "MobiusHarmonicBasis",
    "MobiusHarmonicComponent",
]
