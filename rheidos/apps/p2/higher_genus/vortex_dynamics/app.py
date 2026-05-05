from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from rheidos.apps.p2._io import read_probe_input
from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.apps.p2.modules.higher_genus.dual_harmonic_field import (
    DualHarmonicFieldModule,
)
from rheidos.apps.p2.modules.higher_genus.harmonic_basis import HarmonicBasis
from rheidos.apps.p2.modules.higher_genus.harmonic_velocity import (
    HarmonicVelocityFieldModule,
)
from rheidos.apps.p2.modules.higher_genus.tree_cotree import TreeCotreeModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.p1_space.p1_velocity import (
    area_weighted_face_vectors_to_vertices,
)
from rheidos.apps.p2.modules.p1_space.probe_utils import probe_arrays
from rheidos.apps.p2.modules.p1_space.p1_stream_function import P1StreamFunction
from rheidos.apps.p2.modules.p1_space.p1_velocity import P1VelocityFieldModule
from rheidos.apps.p2.modules.p1_space.whitney_1form import Whitney1FormInterpolator
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ResourceSpec, shape_map
from rheidos.compute.wiring import ProducerContext, producer
from rheidos.compute.world import ModuleBase, World
from rheidos.houdini.runtime.cook_context import CookContext

from ..io import load_mesh_input, load_point_vortex_input
from .ray_sop_module import RaySopModule

RAY_SOP_NODE_PATH = "/obj/geo1/solver1/d/s/ray1"
HARMONIC_C_ATTR = "harmonic_c"
AJ_TOTAL_ATTR = "aj_total"
AJ_DELTA_ATTR = "aj_delta"


@dataclass(frozen=True)
class VortexProjection:
    faceids: np.ndarray
    bary: np.ndarray
    pos: np.ndarray

    @property
    def probes(self) -> tuple[np.ndarray, np.ndarray]:
        return self.faceids, self.bary


def _as_vortex_projection(projected) -> VortexProjection:
    faceids = np.asarray(projected.faceids, dtype=np.int32)
    bary = np.asarray(projected.bary, dtype=np.float64)
    pos = np.asarray(projected.pos, dtype=np.float64)
    if faceids.ndim != 1 or bary.shape != (faceids.shape[0], 3):
        raise ValueError(
            "Projected vortex probes must have faceids (N,) and bary (N,3); "
            f"got {faceids.shape} and {bary.shape}"
        )
    if pos.shape != (faceids.shape[0], 3):
        raise ValueError(
            f"Projected vortex positions must have shape ({faceids.shape[0]},3), "
            f"got {pos.shape}"
        )
    return VortexProjection(
        np.ascontiguousarray(faceids),
        np.ascontiguousarray(bary),
        np.ascontiguousarray(pos),
    )


def _project_vectors_to_normals(
    vectors: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    return vectors - np.einsum("ni,ni->n", vectors, normals)[:, None] * normals


def _divisor_aj_delta(delta_a_per_vortex: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    delta_a_per_vortex = np.asarray(delta_a_per_vortex, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    if delta_a_per_vortex.ndim != 2:
        raise ValueError(
            "delta_a_per_vortex must have shape (N,K), "
            f"got {delta_a_per_vortex.shape}"
        )
    if gamma.shape != (delta_a_per_vortex.shape[0],):
        raise ValueError(
            "gamma must have shape "
            f"({delta_a_per_vortex.shape[0]},), got {gamma.shape}"
        )
    return np.sum(gamma[:, None] * delta_a_per_vortex, axis=0)


def _read_detail_vector_or_zero(ctx: CookContext, name: str, size: int) -> np.ndarray:
    try:
        value = ctx.input_io(0).read_detail(name, dtype=np.float64)
    except KeyError:
        return np.zeros((size,), dtype=np.float64)

    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.shape != (size,):
        raise ValueError(f"Detail attribute {name!r} must have shape ({size},)")
    return value


def _write_detail_vector(ctx: CookContext, name: str, value: np.ndarray) -> None:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.size == 0:
        # Houdini detail attributes cannot represent a zero-tuple cleanly.
        # Missing state is equivalent to the empty harmonic state on K=0 meshes.
        return
    ctx.write_detail(name, value, create=True)


class AbelJacobiModule(ModuleBase):
    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        point_vortex: PointVortexModule,
        dual_harmonic_field: DualHarmonicFieldModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        self.point_vortex = point_vortex
        self.dual_harmonic_field = dual_harmonic_field

    def _positions_from_probes(self, probes, *, name: str) -> np.ndarray:
        faceids, bary = probe_arrays(probes)
        if bary.ndim != 2 or bary.shape[1] != 3 or faceids.shape != (bary.shape[0],):
            raise ValueError(
                f"{name} barycentric coordinates must have shape (N,3), "
                f"got {bary.shape}"
            )

        face_vertices = self.mesh.F_verts.get()[faceids]
        vertex_positions = self.mesh.V_pos.get()[face_vertices]
        return np.einsum("ni,nij->nj", bary, vertex_positions)

    @staticmethod
    def _validate_positions(
        positions: Optional[np.ndarray],
        *,
        count: int,
        name: str,
    ) -> Optional[np.ndarray]:
        if positions is None:
            return None

        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (count, 3):
            raise ValueError(
                f"{name} must have shape ({count},3), got {positions.shape}"
            )
        return positions

    def delta_aj(
        self,
        start_probes,
        end_probes,
        *,
        pos0: Optional[np.ndarray] = None,
        pos1: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return per-vortex Abel-Jacobi increments between consecutive samples."""
        start_faceids, start_bary = probe_arrays(start_probes)
        end_faceids, end_bary = probe_arrays(end_probes)

        if (
            start_bary.ndim != 2
            or start_bary.shape[1] != 3
            or start_faceids.shape != (start_bary.shape[0],)
        ):
            raise ValueError(
                "start_probes barycentric coordinates must have shape (N,3), "
                f"got {start_bary.shape}"
            )
        if (
            end_bary.ndim != 2
            or end_bary.shape[1] != 3
            or end_faceids.shape != (end_bary.shape[0],)
        ):
            raise ValueError(
                "end_probes barycentric coordinates must have shape (N,3), "
                f"got {end_bary.shape}"
            )
        if start_faceids.shape != end_faceids.shape:
            raise ValueError(
                "start_probes and end_probes must contain the same number of "
                f"points, got {start_faceids.shape[0]} and {end_faceids.shape[0]}"
            )

        count = int(start_faceids.shape[0])
        p0 = self._validate_positions(pos0, count=count, name="pos0")
        p1 = self._validate_positions(pos1, count=count, name="pos1")
        if p0 is None:
            p0 = self._positions_from_probes((start_faceids, start_bary), name="pos0")
        if p1 is None:
            p1 = self._positions_from_probes((end_faceids, end_bary), name="pos1")

        xi0 = self.dual_harmonic_field.interpolate(
            (start_faceids, start_bary), field="xi"
        )
        xi1 = self.dual_harmonic_field.interpolate((end_faceids, end_bary), field="xi")

        # Harmonic fields are basis-major (K,N,3), while DeltaAJ is returned
        # vortex-major (N,K) so rows align with point-vortex arrays.
        return 0.5 * np.einsum("kni,ni->nk", xi0 + xi1, p1 - p0)


class CombinedVelocityModule(ModuleBase):
    NAME = "CombinedVelocityModule"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        coexact_velocity: P1VelocityFieldModule,
        harmonic_velocity: HarmonicVelocityFieldModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)

        self.mesh = mesh
        self.coexact_velocity = coexact_velocity
        self.harmonic_velocity = harmonic_velocity

        self.vel_per_face = self.resource(
            "vel_per_face",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.F_verts, lambda s: (s[0], 3)),
            ),
            doc="Facewise sum of coexact and harmonic velocity. Shape: (nF,3)",
        )

        self.vel_per_vertex = self.resource(
            "vel_per_vertex",
            spec=ResourceSpec(
                kind="numpy",
                dtype=np.float64,
                shape_fn=shape_map(self.mesh.V_pos, lambda s: (s[0], 3)),
            ),
            doc="Area-weighted smoothed combined velocity at vertices. Shape: (nV,3)",
        )

        self.bind_producers()

    @producer(
        inputs=("coexact_velocity.vel_per_face", "harmonic_velocity.vel_per_face"),
        outputs=("vel_per_face",),
    )
    def per_face_vel_calculate(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.commit(
            vel_per_face=(
                self.coexact_velocity.vel_per_face.get()
                + self.harmonic_velocity.vel_per_face.get()
            )
        )

    @producer(
        inputs=("vel_per_face", "mesh.F_area", "mesh.F_verts", "mesh.V_pos"),
        outputs=("vel_per_vertex",),
    )
    def per_vertex_vel_calculate(self, ctx: ProducerContext) -> None:
        ctx.require_inputs()
        ctx.commit(
            vel_per_vertex=area_weighted_face_vectors_to_vertices(
                self.vel_per_face.get(),
                self.mesh.F_area.get(),
                self.mesh.F_verts.get(),
                self.mesh.V_pos.get().shape[0],
            )
        )

    def interpolate(self, probes, smooth=True) -> np.ndarray:
        faceids, bary = probe_arrays(probes)
        if faceids.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        if not smooth:
            return self.vel_per_face.get()[faceids]

        verts = self.mesh.F_verts.get()[faceids]
        vel_verts = self.vel_per_vertex.get()[verts]

        b1, b2, b3 = map(lambda x: x.reshape(-1, 1), bary.T)
        v1, v2, v3 = vel_verts[:, 0, :], vel_verts[:, 1, :], vel_verts[:, 2, :]

        return b1 * v1 + b2 * v2 + b3 * v3


def _evaluate_total_velocity(
    mods: "App",
    state: VortexProjection,
    c_trial: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    mods.point_vortex.set_vortex(state.faceids, state.bary, gamma, state.pos)
    mods.harmonic_velocity.set_coefficients(c_trial)

    # The stream solve depends on the trial vortex positions, while the
    # harmonic part depends on the AJ-implied coefficients for the same trial.
    velocity = mods.combined_velocity.interpolate(state.probes)
    normals = mods.mesh.F_normal.get()[state.faceids]
    return _project_vectors_to_normals(velocity, normals)


def _trial_c_from_projection(
    mods: "App",
    ref: VortexProjection,
    trial: VortexProjection,
    c_ref: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta_a = mods.abel_jacobi.delta_aj(
        ref.probes,
        trial.probes,
        pos0=ref.pos,
        pos1=trial.pos,
    )
    delta_A = _divisor_aj_delta(delta_a, gamma)
    return c_ref + delta_A, delta_A


def _rk4_step_with_abel_jacobi(
    mods: "App",
    ref: VortexProjection,
    c_ref: np.ndarray,
    dt: float,
    projector: Callable[[np.ndarray], VortexProjection],
) -> tuple[VortexProjection, np.ndarray, np.ndarray]:
    gamma = np.asarray(mods.point_vortex.gamma.get(), dtype=np.float64)
    c_ref = np.asarray(c_ref, dtype=np.float64)
    if gamma.shape != (ref.pos.shape[0],):
        raise ValueError(f"gamma must have shape ({ref.pos.shape[0]},)")

    # Every trial configuration is measured from the same frozen reference.
    v1 = _evaluate_total_velocity(mods, ref, c_ref, gamma)

    p2 = projector(ref.pos + 0.5 * dt * v1)
    c2, _ = _trial_c_from_projection(mods, ref, p2, c_ref, gamma)
    v2 = _evaluate_total_velocity(mods, p2, c2, gamma)

    p3 = projector(ref.pos + 0.5 * dt * v2)
    c3, _ = _trial_c_from_projection(mods, ref, p3, c_ref, gamma)
    v3 = _evaluate_total_velocity(mods, p3, c3, gamma)

    p4 = projector(ref.pos + dt * v3)
    c4, _ = _trial_c_from_projection(mods, ref, p4, c_ref, gamma)
    v4 = _evaluate_total_velocity(mods, p4, c4, gamma)

    ref_normals = mods.mesh.F_normal.get()[ref.faceids]
    v_rk4 = (
        _project_vectors_to_normals(v1, ref_normals)
        + 2.0 * _project_vectors_to_normals(v2, ref_normals)
        + 2.0 * _project_vectors_to_normals(v3, ref_normals)
        + _project_vectors_to_normals(v4, ref_normals)
    ) / 6.0

    accepted = projector(ref.pos + dt * v_rk4)
    c_next, dA_final = _trial_c_from_projection(mods, ref, accepted, c_ref, gamma)

    mods.point_vortex.set_vortex(
        accepted.faceids,
        accepted.bary,
        gamma,
        accepted.pos,
    )
    mods.harmonic_velocity.set_coefficients(c_next)
    return accepted, c_next, dA_final


class App(ModuleBase):
    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        # Mesh
        self.mesh = self.require(SurfaceMeshModule)
        self.dec = self.require(DEC, mesh=self.mesh)

        # Point Vortices
        self.point_vortex = self.require(PointVortexModule)
        self.stream_function = self.require(
            P1StreamFunction,
            mesh=self.mesh,
            point_vortex=self.point_vortex,
            dec=self.dec,
        )
        self.stream_velocity = self.require(
            P1VelocityFieldModule,
            mesh=self.mesh,
            dec=self.dec,
            stream=self.stream_function,
        )

        # Harmonic Basis
        self.tree_cotree = self.require(TreeCotreeModule, mesh=self.mesh)
        self.harmonic_basis = self.require(
            HarmonicBasis,
            dec=self.dec,
            tree_cotree=self.tree_cotree,
        )
        self.whitney_1form = self.require(
            Whitney1FormInterpolator,
            mesh=self.mesh,
        )
        self.dual_harmonic_field = self.require(
            DualHarmonicFieldModule,
            mesh=self.mesh,
            harmonic_basis=self.harmonic_basis,
        )
        self.harmonic_velocity = self.require(
            HarmonicVelocityFieldModule,
            mesh=self.mesh,
            dual_harmonic_field=self.dual_harmonic_field,
        )
        self.combined_velocity = self.require(
            CombinedVelocityModule,
            mesh=self.mesh,
            coexact_velocity=self.stream_velocity,
            harmonic_velocity=self.harmonic_velocity,
        )

        # Abel-Jacobi Map
        self.abel_jacobi = self.require(
            AbelJacobiModule,
            mesh=self.mesh,
            point_vortex=self.point_vortex,
            dual_harmonic_field=self.dual_harmonic_field,
        )

        # Advection
        self.rk4 = self.require(RK4IntegratorModule)
        self.surface_projector = self.require(
            RaySopModule,
            child=True,
            child_name="ray_sop_surface_points_projector",
            node_path=RAY_SOP_NODE_PATH,
        )

    @staticmethod
    def rk4_step(ctx: CookContext, dt: float):
        mods = ctx.world().require(App)
        mods.surface_projector.configure(node_path=RAY_SOP_NODE_PATH)
        mods.surface_projector.setup(ctx)

        k_basis = int(mods.dual_harmonic_field.zeta_face.get().shape[0])
        c_ref = _read_detail_vector_or_zero(ctx, HARMONIC_C_ATTR, k_basis)
        aj_total = _read_detail_vector_or_zero(ctx, AJ_TOTAL_ATTR, k_basis)
        ref = VortexProjection(
            np.asarray(mods.point_vortex.face_ids.get(), dtype=np.int32),
            np.asarray(mods.point_vortex.bary.get(), dtype=np.float64),
            np.asarray(mods.point_vortex.pos_world.get(), dtype=np.float64),
        )

        projected, c_next, dA_final = _rk4_step_with_abel_jacobi(
            mods,
            ref,
            c_ref,
            dt,
            lambda points: _as_vortex_projection(
                mods.surface_projector.project_points(points)
            ),
        )
        return projected, c_next, dA_final, aj_total + dA_final


def setup_mesh_and_point_vortices(ctx: CookContext):
    mods = ctx.world().require(App)
    load_mesh_input(
        ctx, mods.mesh, missing_message="Input 0 has to be mesh input geometry"
    )
    load_point_vortex_input(ctx, mods.point_vortex, index=1)
    mods.stream_function.set_homo_dirichlet_boundary()


def rk4_advect(ctx: CookContext, dt=0.001) -> None:
    mods = ctx.world().require(App)
    mods.surface_projector.configure(node_path=RAY_SOP_NODE_PATH)
    mods.surface_projector.setup(ctx)

    load_point_vortex_input(ctx, mods.point_vortex, index=0)
    mods.rk4.configure(timestep=dt)

    projected, c_next, dA_final, aj_total_next = mods.rk4_step(ctx, dt)
    mods.rk4.time.set(mods.rk4.time.get() + dt)

    ctx.write_point("P", projected.pos)
    ctx.write_point("bary", projected.bary)
    ctx.write_point("faceid", projected.faceids)
    _write_detail_vector(ctx, HARMONIC_C_ATTR, c_next)
    _write_detail_vector(ctx, AJ_DELTA_ATTR, dA_final)
    _write_detail_vector(ctx, AJ_TOTAL_ATTR, aj_total_next)


# Interpolate


def interpolate_harmonic_velocity_field(ctx: CookContext):
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    harmonic_velocity_field = mods.harmonic_velocity.interpolate((faceids, bary))
    ctx.write_point("harmonic_velocity_field", harmonic_velocity_field)


def interpolate_stream_velocity_field(ctx: CookContext):
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    stream_velocity_field = mods.stream_velocity.interpolate((faceids, bary))
    ctx.write_point("stream_velocity_field", stream_velocity_field)


def interpolate_velocity_field(ctx: CookContext):
    mods = ctx.world().require(App)
    faceids, bary = read_probe_input(ctx, index=0)
    velocity_field = mods.combined_velocity.interpolate((faceids, bary))
    ctx.write_point("velocity_field", velocity_field)


def interpolate_xi_dual_harmonic_field(ctx: CookContext, basis_id=0):
    mods = ctx.world().require(App)

    generator_count = mods.tree_cotree.generator_count.get()
    if basis_id < 0 or basis_id >= generator_count:
        raise RuntimeError(
            f"Allowed basis_id range: [0, {generator_count-1}], received: f{basis_id}"
        )

    faceids, bary = read_probe_input(ctx, index=0)
    xi = mods.dual_harmonic_field.interpolate((faceids, bary), field="xi")
    ctx.write_point("xi_dual_harmonic_field", xi[basis_id])


def interpolate_zeta_harmonic_field(ctx: CookContext, basis_id=0):
    mods = ctx.world().require(App)

    generator_count = mods.tree_cotree.generator_count.get()
    if basis_id < 0 or basis_id >= generator_count:
        raise RuntimeError(
            f"Allowed basis_id range: [0, {generator_count-1}], received: f{basis_id}"
        )

    faceids, bary = read_probe_input(ctx, index=0)
    zeta = mods.dual_harmonic_field.interpolate((faceids, bary), field="zeta")
    ctx.write_point("zeta_harmonic_field", zeta[basis_id])
