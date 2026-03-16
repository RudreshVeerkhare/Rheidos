from __future__ import annotations

from dataclasses import replace

import numpy as np

from .advection import advect_midpoint_batch
from .assembly import scatter_point_vortex_rhs
from .config import SimulationConfig
from .mesh import build_default_mesh, build_face_geometry, build_mesh_topology_geometry
from .model import Diagnostics, MeshData, StepResult, VortexState
from .scenarios import init_vortices, vortex_positions_from_face_bary
from .solver import build_poisson_system, solve_stream_function
from .space import build_p2_space_data
from .velocity import build_p2_velocity_fields


class P2PointVortexSim:
    def __init__(
        self,
        config: SimulationConfig,
        *,
        mesh: MeshData | None = None,
    ) -> None:
        self.config = config

        if mesh is None:
            self.mesh = build_default_mesh(
                kind=config.mesh.kind,
                subdivisions=config.mesh.subdivisions,
                radius=config.mesh.radius,
            )
        else:
            self.mesh = build_mesh_topology_geometry(mesh.vertices, mesh.faces)

        self.geometry = build_face_geometry(self.mesh.vertices, self.mesh.faces)
        self.space = build_p2_space_data(self.mesh.vertices.shape[0], self.mesh.faces)

        prefer_taichi = self.config.solver.backend in {"auto", "taichi"}
        self.system = build_poisson_system(
            self.space.face_to_dofs,
            self.space.ndof,
            self.geometry.J,
            self.geometry.Ginv,
            self.geometry.sqrt_detG,
            pin_index=self.config.solver.pin_index,
            prefer_taichi=prefer_taichi,
        )

        self.state = init_vortices(
            self.mesh,
            preset=self.config.vortex.preset,
            n_vortices=self.config.vortex.n_vortices,
            gamma_scale=self.config.vortex.gamma_scale,
            seed=self.config.seed,
        )

        self.psi = np.zeros((self.space.ndof,), dtype=np.float64)
        self.vel_corner = np.zeros((self.mesh.faces.shape[0], 3, 3), dtype=np.float64)
        self.vel_face = np.zeros((self.mesh.faces.shape[0], 3), dtype=np.float64)
        self.stream_vertex = np.zeros((self.mesh.vertices.shape[0],), dtype=np.float64)
        self.last_diag = Diagnostics(
            residual_l2=0.0,
            rhs_circulation=0.0,
            k_ones_inf=float(self.system.k_ones_inf),
            hops_total=0,
            hops_max=0,
            bary_min=0.0,
            bary_max=0.0,
            bary_sum_min=0.0,
            bary_sum_max=0.0,
            solver_backend="none",
        )

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = self.config.seed
        self.state = init_vortices(
            self.mesh,
            preset=self.config.vortex.preset,
            n_vortices=self.config.vortex.n_vortices,
            gamma_scale=self.config.vortex.gamma_scale,
            seed=int(seed),
        )

    def current_positions(self) -> np.ndarray:
        return vortex_positions_from_face_bary(
            self.mesh.vertices,
            self.mesh.faces,
            self.state.face_ids,
            self.state.bary,
        )

    def _solve_fields(self) -> tuple[float, float, str]:
        prefer_taichi = self.config.solver.backend in {"auto", "taichi"}
        rhs = scatter_point_vortex_rhs(
            self.state.face_ids,
            self.state.bary,
            self.state.gamma,
            self.space.face_to_dofs,
            self.space.ndof,
            prefer_taichi=prefer_taichi,
        )

        psi, residual_l2, backend_used, _ = solve_stream_function(
            self.system,
            rhs,
            backend=self.config.solver.backend,
        )
        self.psi = psi

        self.vel_corner, self.vel_face, self.stream_vertex = build_p2_velocity_fields(
            self.psi,
            self.space.face_to_dofs,
            self.geometry.J,
            self.geometry.Ginv,
            self.mesh.face_normals,
            n_vertices=self.mesh.vertices.shape[0],
        )

        rhs_circ = float(np.sum(rhs))
        return rhs_circ, float(residual_l2), backend_used

    def solve_fields(self) -> StepResult:
        rhs_circ, residual_l2, backend_used = self._solve_fields()

        bary = np.asarray(self.state.bary, dtype=np.float64)
        bary_sum = bary.sum(axis=1) if bary.size > 0 else np.array([0.0], dtype=np.float64)

        self.last_diag = Diagnostics(
            residual_l2=float(residual_l2),
            rhs_circulation=rhs_circ,
            k_ones_inf=float(self.system.k_ones_inf),
            hops_total=0,
            hops_max=0,
            bary_min=float(bary.min()) if bary.size > 0 else 0.0,
            bary_max=float(bary.max()) if bary.size > 0 else 0.0,
            bary_sum_min=float(bary_sum.min()),
            bary_sum_max=float(bary_sum.max()),
            solver_backend=backend_used,
        )

        return StepResult(
            psi=self.psi.copy(),
            vel_corner=self.vel_corner.copy(),
            vel_face=self.vel_face.copy(),
            stream_vertex=self.stream_vertex.copy(),
            state=VortexState(
                face_ids=self.state.face_ids.copy(),
                bary=self.state.bary.copy(),
                gamma=self.state.gamma.copy(),
            ),
            diagnostics=self.last_diag,
        )

    def step(self, dt: float | None = None, *, substeps: int | None = None) -> StepResult:
        if dt is None:
            dt = float(self.config.time.dt)
        if substeps is None:
            substeps = int(self.config.time.substeps)
        if substeps <= 0:
            raise ValueError("substeps must be positive")

        dt_sub = float(dt) / float(substeps)
        total_hops = 0
        max_hops = 0
        rhs_circ = 0.0
        backend_used = "none"
        residual_l2 = 0.0

        for _ in range(substeps):
            rhs_circ, residual_l2, backend_used = self._solve_fields()

            face_out, bary_out, _, hops_total, hops_max = advect_midpoint_batch(
                self.mesh.vertices,
                self.mesh.faces,
                self.mesh.face_adjacency,
                self.vel_corner,
                self.state.face_ids,
                self.state.bary,
                dt_sub,
                max_hops=self.config.time.max_hops,
            )

            self.state = VortexState(
                face_ids=face_out,
                bary=bary_out,
                gamma=self.state.gamma.copy(),
            )

            total_hops += int(hops_total)
            max_hops = max(max_hops, int(hops_max))
        bary = np.asarray(self.state.bary, dtype=np.float64)
        bary_sum = bary.sum(axis=1) if bary.size > 0 else np.array([0.0], dtype=np.float64)

        self.last_diag = Diagnostics(
            residual_l2=residual_l2,
            rhs_circulation=float(rhs_circ),
            k_ones_inf=float(self.system.k_ones_inf),
            hops_total=int(total_hops),
            hops_max=int(max_hops),
            bary_min=float(bary.min()) if bary.size > 0 else 0.0,
            bary_max=float(bary.max()) if bary.size > 0 else 0.0,
            bary_sum_min=float(bary_sum.min()),
            bary_sum_max=float(bary_sum.max()),
            solver_backend=backend_used,
        )

        return StepResult(
            psi=self.psi.copy(),
            vel_corner=self.vel_corner.copy(),
            vel_face=self.vel_face.copy(),
            stream_vertex=self.stream_vertex.copy(),
            state=VortexState(
                face_ids=self.state.face_ids.copy(),
                bary=self.state.bary.copy(),
                gamma=self.state.gamma.copy(),
            ),
            diagnostics=self.last_diag,
        )


def run_headless(
    config: SimulationConfig,
    *,
    steps: int = 100,
    seed: int | None = None,
) -> dict[str, object]:
    if seed is not None:
        config = replace(config, seed=int(seed))

    sim = P2PointVortexSim(config)
    result = sim.solve_fields()

    for _ in range(int(steps)):
        result = sim.step()

    pos = sim.current_positions()
    return {
        "positions": pos,
        "face_ids": result.state.face_ids,
        "bary": result.state.bary,
        "gamma": result.state.gamma,
        "diagnostics": result.diagnostics,
        "stream_vertex": result.stream_vertex,
        "vel_face": result.vel_face,
    }
