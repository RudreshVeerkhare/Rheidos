from __future__ import annotations

import numpy as np
import pytest

from rheidos.apps.p2.mobius.cook import App
from rheidos.apps.p2.mobius.parity import (
    DeckParityError,
    validate_deck_parity,
)
from rheidos.compute import World


def _grid_with_interior_vertex() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[x, y, 0.0] for y in range(3) for x in range(3)],
        dtype=np.float64,
    )
    faces: list[list[int]] = []
    for y in range(2):
        for x in range(2):
            lower_left = 3 * y + x
            lower_right = lower_left + 1
            upper_left = lower_left + 3
            upper_right = upper_left + 1
            faces.extend(
                (
                    [lower_left, lower_right, upper_right],
                    [lower_left, upper_right, upper_left],
                )
            )
    return vertices, np.asarray(faces, dtype=np.int32)


def _use_zero_harmonic_component(app: App) -> None:
    """This test isolates parity rebuilding in the coexact pipeline."""
    class ScalarState:
        def __init__(self) -> None:
            self.value = 0.0

        def get(self) -> float:
            return self.value

        def set(self, value: float) -> None:
            self.value = float(value)

    coefficient = ScalarState()
    coordinate = ScalarState()

    class ZeroHarmonicComponent:
        @staticmethod
        def stage_coefficient(reference_coordinate, reference_coefficient) -> float:
            return float(reference_coefficient)

        @staticmethod
        def interpolate(probes, coefficient=None) -> np.ndarray:
            face_ids, _ = probes
            return np.zeros((face_ids.shape[0], 3), dtype=np.float64)

        @staticmethod
        def commit_accepted_step(reference_coordinate, reference_coefficient):
            return float(reference_coefficient), 0.0

    app.harmonic_component = ZeroHarmonicComponent()
    app.harmonic_coefficient = coefficient
    app.abel_jacobi_coordinate = coordinate


def test_parity_validation_rejects_a_mismatched_deck_pair() -> None:
    tau = np.array([1, 0, 3, 2], dtype=np.int32)
    values = np.array([2.0, -2.0, 1.0, 1.0], dtype=np.float64)

    with pytest.raises(DeckParityError, match="not deck-odd"):
        validate_deck_parity(values, tau, parity=-1, name="test field")


def test_lifted_p1_pipeline_enforces_required_deck_parities() -> None:
    vertices, faces = _grid_with_interior_vertex()
    app = World().require(App)

    # Empty seam pairing creates two copies of the disk.  This is a compact
    # cover with a free interior P1 degree of freedom, so psi is nonzero and
    # its parity is tested through an actual Poisson solve.
    app.cover.set_seam_edge_pairs(np.empty((0, 4), dtype=np.int32))
    app.base_mesh.set_mesh(vertices, faces)
    app.base_point_vortex.set_vortex(
        faceids=np.array([0], dtype=np.int32),
        bary=np.array([[0.1, 0.1, 0.8]], dtype=np.float64),
        gamma=np.array([1.0], dtype=np.float64),
        pos=np.zeros((1, 3), dtype=np.float64),
    )
    app.initialize_lifted_point_vortices()
    app.stream_function.set_homo_dirichlet_boundary()

    omega = app.stream_function.omega.get()
    psi = app.stream_function.psi.get()
    face_velocity = app.stream_velocity.vel_per_face.get()
    vertex_velocity = app.stream_velocity.vel_per_vertex.get()
    tau_vertex = app.cover.tau_vertex.get()
    tau_face = app.cover.tau_face.get()

    assert np.max(np.abs(psi)) > 0.0
    np.testing.assert_allclose(omega[tau_vertex], -omega, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(psi[tau_vertex], -psi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        face_velocity[tau_face],
        face_velocity,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        vertex_velocity[tau_vertex],
        vertex_velocity,
        rtol=0.0,
        atol=0.0,
    )


def test_actual_rk4_stages_rebuild_parity_matched_lifted_solves() -> None:
    vertices, faces = _grid_with_interior_vertex()
    app = World().require(App)
    app.cover.set_seam_edge_pairs(np.empty((0, 4), dtype=np.int32))
    app.base_mesh.set_mesh(vertices, faces)
    app.base_point_vortex.set_vortex(
        faceids=np.array([0], dtype=np.int32),
        bary=np.array([[0.1, 0.1, 0.8]], dtype=np.float64),
        gamma=np.array([1.0], dtype=np.float64),
        pos=np.zeros((1, 3), dtype=np.float64),
    )
    app.initialize_lifted_point_vortices()
    app.stream_function.set_homo_dirichlet_boundary()
    _use_zero_harmonic_component(app)

    accepted = app.rk4_step(0.01)

    assert accepted.face_ids.shape == (1,)
    assert app.point_vortex.face_ids.get().shape == (2,)
    tau_vertex = app.cover.tau_vertex.get()
    tau_face = app.cover.tau_face.get()
    psi = app.stream_function.psi.get()
    velocity = app.stream_velocity.vel_per_face.get()
    np.testing.assert_allclose(psi[tau_vertex], -psi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        velocity[tau_face],
        velocity,
        rtol=0.0,
        atol=0.0,
    )
