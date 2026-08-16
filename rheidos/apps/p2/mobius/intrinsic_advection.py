"""Local intrinsic advection and RK4 integration on triangle meshes.

The routines in this module deliberately know nothing about the Mobius cover
or point-vortex pairing.  They advance arrays of face ids and barycentric
coordinates using only per-face geometry and immediate face adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, ProducerContext, ResourceSpec, World, producer


class ReduceTimestepError(RuntimeError):
    """A local trajectory cannot be continued safely for the requested step."""


@dataclass(frozen=True)
class IntrinsicVortexState:
    """The coordinates needed to locate N vortices on a triangle mesh."""

    face_ids: np.ndarray
    bary: np.ndarray

    @property
    def probes(self) -> tuple[np.ndarray, np.ndarray]:
        return self.face_ids, self.bary


@dataclass(frozen=True)
class IntrinsicAdvectionResult:
    """A retracted state and the tangent transport along each local path."""

    state: IntrinsicVortexState
    transport_from_start: np.ndarray
    crossing_counts: np.ndarray


def positions_from_state(
    mesh: SurfaceMeshModule,
    state: IntrinsicVortexState,
) -> np.ndarray:
    """Reconstruct R3 positions without projecting or searching for faces."""
    face_vertices = mesh.F_verts.get()[state.face_ids]
    return np.einsum("ni,nij->nj", state.bary, mesh.V_pos.get()[face_vertices])


class RK4AdvectorModule(ModuleBase):
    """RK4 integration using local barycentric retractions.

    Euler retractions walk through immediate neighbors until their remaining
    time is exhausted.  The walk stays entirely local: it never searches the
    mesh or projects an endpoint onto a globally selected face.
    """

    NAME = "RK4AdvectorModule"
    EPS = 1.0e-10

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        self.mesh = mesh
        self.transport = self.resource(
            "transport",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
            doc="Intrinsic transport per face and opposite edge. Shape: (nF,3,3,3)",
        )
        self.corner_map = self.resource(
            "corner_map",
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
            doc=(
                "Neighbor local corner for each source corner and opposite edge. "
                "Shape: (nF,3,3)"
            ),
        )
        self.time = self.resource(
            "time",
            spec=ResourceSpec(kind="python", dtype=float),
            buffer=0.0,
            declare=True,
        )
        self.bind_producers()

    @producer(
        inputs=("mesh.V_pos", "mesh.F_verts", "mesh.F_adj", "mesh.F_normal"),
        outputs=("transport", "corner_map"),
    )
    def build_transport(self, ctx: ProducerContext) -> None:
        """Precompute the unfolded tangent-plane map for every adjacency."""
        vertices = ctx.inputs.mesh.V_pos.get()
        faces = ctx.inputs.mesh.F_verts.get()
        neighbors = ctx.inputs.mesh.F_adj.get()
        normals = ctx.inputs.mesh.F_normal.get()
        transport = np.zeros((faces.shape[0], 3, 3, 3), dtype=np.float64)
        corner_map = np.full((faces.shape[0], 3, 3), -1, dtype=np.int32)

        for face_id in range(faces.shape[0]):
            for opposite_corner in range(3):
                neighbor = int(neighbors[face_id, opposite_corner])
                if neighbor < 0:
                    continue

                # Using sorted global ids gives both directed adjacencies the
                # same edge direction.  The two cross products then encode
                # only the change of tangent plane, not a sign convention.
                edge_vertices = np.delete(faces[face_id], opposite_corner)
                vertex_a, vertex_b = sorted(map(int, edge_vertices))
                edge = vertices[vertex_b] - vertices[vertex_a]
                edge /= np.linalg.norm(edge)

                edge_normal_here = np.cross(normals[face_id], edge)
                edge_normal_there = np.cross(normals[neighbor], edge)
                transport[face_id, opposite_corner] = (
                    np.outer(edge, edge)
                    + np.outer(edge_normal_there, edge_normal_here)
                )

                for source_corner, vertex_id in enumerate(faces[face_id]):
                    if source_corner == opposite_corner:
                        continue
                    corner_map[face_id, opposite_corner, source_corner] = int(
                        np.flatnonzero(faces[neighbor] == vertex_id)[0]
                    )

        ctx.commit(transport=transport, corner_map=corner_map)

    @staticmethod
    def _clean_and_normalize(bary: np.ndarray, eps: float) -> np.ndarray:
        # Only numerical zeros are clamped.  Clipping every value would hide a
        # genuine second-edge crossing, which must instead reject the step.
        bary[np.abs(bary) < eps] = 0.0
        bary /= bary.sum(axis=-1, keepdims=True)
        return bary

    def _tangent_vectors(
        self,
        vectors: np.ndarray,
        face_ids: np.ndarray,
    ) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        normals = self.mesh.F_normal.get()[face_ids]
        return vectors - np.einsum("ni,ni->n", vectors, normals)[:, None] * normals

    def _first_exit(
        self,
        bary: np.ndarray,
        bary_rate: np.ndarray,
    ) -> tuple[float, int]:
        """Return the first nonnegative zero-coordinate time and its corner."""
        exit_time = np.inf
        exit_corner = -1
        for corner in range(3):
            if bary_rate[corner] >= -self.EPS:
                continue
            candidate = -bary[corner] / bary_rate[corner]
            if candidate >= -self.EPS and candidate < exit_time:
                exit_time = max(float(candidate), 0.0)
                exit_corner = corner
        return exit_time, exit_corner

    def _walk_vortex(
        self,
        vortex_id: int,
        face_id: int,
        bary: np.ndarray,
        velocity: np.ndarray,
        dt: float,
    ) -> tuple[int, np.ndarray, np.ndarray, int]:
        """Walk one constant, parallel-transported direction across the mesh."""
        gradients = self.mesh.grad_bary.get()
        normals = self.mesh.F_normal.get()
        neighbors = self.mesh.F_adj.get()
        transport = self.transport.get()
        corner_map = self.corner_map.get()

        current_face = int(face_id)
        current_bary = np.array(bary, dtype=np.float64, copy=True)
        current_velocity = np.array(velocity, dtype=np.float64, copy=True)
        remaining_dt = float(dt)
        path_transport = np.eye(3, dtype=np.float64)
        crossing_count = 0

        # At a vertex, two or more transitions can have zero travel time.  A
        # repeated directed edge without consuming time would be a numerical
        # cycle, so remember only those zero-time transitions.  Ordinary
        # positive-time paths may revisit faces on a closed surface.
        zero_time_edges: set[tuple[int, int]] = set()

        while remaining_dt > self.EPS:
            normal = normals[current_face]
            current_velocity -= np.dot(current_velocity, normal) * normal
            bary_rate = gradients[current_face] @ current_velocity
            exit_time, exit_corner = self._first_exit(current_bary, bary_rate)

            if exit_corner < 0 or exit_time >= remaining_dt:
                current_bary += remaining_dt * bary_rate
                if np.any(current_bary < -self.EPS):
                    raise ReduceTimestepError(
                        f"Vortex {vortex_id} left face {current_face} without a "
                        "detectable edge crossing."
                    )
                self._clean_and_normalize(current_bary, self.EPS)
                remaining_dt = 0.0
                break

            current_bary += exit_time * bary_rate
            current_bary[exit_corner] = 0.0
            self._clean_and_normalize(current_bary, self.EPS)

            neighbor = int(neighbors[current_face, exit_corner])
            if neighbor < 0:
                raise ReduceTimestepError(
                    f"Vortex {vortex_id} attempted to cross a physical boundary "
                    f"from face {current_face}."
                )

            if exit_time <= self.EPS:
                edge_key = (current_face, exit_corner)
                if edge_key in zero_time_edges:
                    raise ReduceTimestepError(
                        f"Vortex {vortex_id} entered a zero-time cycle while "
                        "traversing a mesh vertex; reduce the timestep."
                    )
                zero_time_edges.add(edge_key)
            else:
                zero_time_edges.clear()

            neighbor_bary = np.zeros(3, dtype=np.float64)
            source_corners = np.flatnonzero(
                corner_map[current_face, exit_corner] >= 0
            )
            target_corners = corner_map[
                current_face,
                exit_corner,
                source_corners,
            ]
            neighbor_bary[target_corners] = current_bary[source_corners]

            edge_transport = transport[current_face, exit_corner]
            current_velocity = edge_transport @ current_velocity
            path_transport = edge_transport @ path_transport
            remaining_dt = max(remaining_dt - exit_time, 0.0)
            current_face = neighbor
            current_bary = neighbor_bary
            crossing_count += 1

        # A sub-EPS remainder is below the barycentric cleanup tolerance.  It
        # is intentionally discarded rather than triggering another unstable
        # zero-time transition at a corner.
        self._clean_and_normalize(current_bary, self.EPS)
        return current_face, current_bary, path_transport, crossing_count

    def advect_euler_with_transport(
        self,
        state: IntrinsicVortexState,
        velocity: np.ndarray,
        dt: float,
    ) -> IntrinsicAdvectionResult:
        """Retract tangent velocities through any number of adjacent faces."""
        face_ids = np.asarray(state.face_ids, dtype=np.int32)
        bary = np.asarray(state.bary, dtype=np.float64)
        velocity = self._tangent_vectors(velocity, face_ids)

        gradients = self.mesh.grad_bary.get()
        bary_rate = np.einsum("nij,nj->ni", gradients[face_ids], velocity)

        # Find the first barycentric coordinate that reaches zero.  This path
        # is vectorized because most vortices normally remain in their face.
        exit_times = np.full_like(bary_rate, np.inf)
        decreasing_rows, decreasing_corners = np.nonzero(bary_rate < -self.EPS)
        candidate_times = -bary[decreasing_rows, decreasing_corners] / bary_rate[
            decreasing_rows, decreasing_corners
        ]
        valid = candidate_times >= -self.EPS
        exit_times[
            decreasing_rows[valid], decreasing_corners[valid]
        ] = np.maximum(candidate_times[valid], 0.0)

        rows = np.arange(face_ids.shape[0])
        exit_corner = np.argmin(exit_times, axis=1)
        exit_time = exit_times[rows, exit_corner]
        crosses_edge = exit_time < dt

        next_face_ids = face_ids.copy()
        next_bary = bary.copy()
        path_transport = np.repeat(
            np.eye(3, dtype=np.float64)[None, :, :],
            face_ids.shape[0],
            axis=0,
        )
        crossing_counts = np.zeros(face_ids.shape[0], dtype=np.int32)
        stays = ~crosses_edge
        next_bary[stays] += dt * bary_rate[stays]
        if np.any(next_bary[stays] < -self.EPS):
            raise ReduceTimestepError(
                "Intrinsic advection missed a first edge crossing; reduce the timestep."
            )
        next_bary[stays] = self._clean_and_normalize(next_bary[stays], self.EPS)

        # Crossers require different local paths.  Continue each one through
        # neighbors until its complete displacement is consumed.
        for vortex_id in np.flatnonzero(crosses_edge):
            (
                next_face_ids[vortex_id],
                next_bary[vortex_id],
                path_transport[vortex_id],
                crossing_counts[vortex_id],
            ) = self._walk_vortex(
                int(vortex_id),
                int(face_ids[vortex_id]),
                bary[vortex_id],
                velocity[vortex_id],
                dt,
            )

        return IntrinsicAdvectionResult(
            state=IntrinsicVortexState(
                np.ascontiguousarray(next_face_ids),
                np.ascontiguousarray(next_bary),
            ),
            transport_from_start=np.ascontiguousarray(path_transport),
            crossing_counts=crossing_counts,
        )

    def advect_euler(
        self,
        state: IntrinsicVortexState,
        velocity: np.ndarray,
        dt: float,
    ) -> IntrinsicVortexState:
        """Return only the retracted state for callers that do not need transport."""
        return self.advect_euler_with_transport(state, velocity, dt).state

    def _transport_stage_to_start(
        self,
        vectors: np.ndarray,
        stage: IntrinsicAdvectionResult,
        start_face_ids: np.ndarray,
    ) -> np.ndarray:
        """Invert each stage's accumulated path transport on tangent vectors."""
        vectors = self._tangent_vectors(vectors, stage.state.face_ids)
        transported = np.einsum(
            "nji,nj->ni",
            stage.transport_from_start,
            vectors,
        )
        return self._tangent_vectors(transported, start_face_ids)

    def step(
        self,
        initial: IntrinsicVortexState,
        velocity: Callable[[IntrinsicVortexState], np.ndarray],
        dt: float,
    ) -> IntrinsicVortexState:
        """Advance one intrinsic RK4 step from a frozen starting state."""
        base_faces = initial.face_ids
        k1 = self._tangent_vectors(velocity(initial), base_faces)

        stage2 = self.advect_euler_with_transport(initial, k1, 0.5 * dt)
        k2 = self._transport_stage_to_start(
            velocity(stage2.state),
            stage2,
            base_faces,
        )

        stage3 = self.advect_euler_with_transport(initial, k2, 0.5 * dt)
        k3 = self._transport_stage_to_start(
            velocity(stage3.state),
            stage3,
            base_faces,
        )

        stage4 = self.advect_euler_with_transport(initial, k3, dt)
        k4 = self._transport_stage_to_start(
            velocity(stage4.state),
            stage4,
            base_faces,
        )

        # The four directions now share the starting tangent plane.  Treat the
        # weighted sum as one displacement and retract it with dt=1.
        displacement = dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        accepted = self.advect_euler(initial, displacement, 1.0)
        self.time.set(self.time.get() + dt)
        return accepted


__all__ = [
    "IntrinsicAdvectionResult",
    "IntrinsicVortexState",
    "RK4AdvectorModule",
    "ReduceTimestepError",
    "positions_from_state",
]
