from __future__ import annotations

import numpy as np
import pytest

from rheidos.apps.p2.mobius.cook import (
    ABEL_JACOBI_STEP_DELTA_ATTR,
    HARMONIC_COEFFICIENT_ATTR,
    App,
    _restore_solver_state_from_input,
)
from rheidos.apps.p2.mobius.harmonic_component import (
    DEFAULT_HARMONIC_COEFFICIENT,
)
from rheidos.apps.p2.mobius.intrinsic_advection import ReduceTimestepError
from rheidos.compute import World


def _coarse_mobius() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a small nondegenerate cut strip with a half-twisted seam."""
    theta_values = np.linspace(0.0, 2.0 * np.pi, 5)
    transverse_values = np.linspace(-0.4, 0.4, 3)
    vertices: list[list[float]] = []
    for theta in theta_values:
        for transverse in transverse_values:
            radius = 2.0 + transverse * np.cos(0.5 * theta)
            vertices.append(
                [
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    transverse * np.sin(0.5 * theta),
                ]
            )

    faces: list[list[int]] = []
    for longitudinal in range(theta_values.shape[0] - 1):
        for transverse in range(transverse_values.shape[0] - 1):
            a = 3 * longitudinal + transverse
            b = a + 3
            c = b + 1
            d = a + 1
            faces.extend(([a, b, c], [a, c, d]))

    final_column = 3 * (theta_values.shape[0] - 1)
    seams = np.array(
        [
            [0, 1, final_column + 2, final_column + 1],
            [1, 2, final_column + 1, final_column],
        ],
        dtype=np.int32,
    )
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
        seams,
    )


def _initialized_app(*, coefficient: float = 0.0) -> App:
    vertices, faces, seams = _coarse_mobius()
    app = World().require(App)
    app.cover.set_seam_edge_pairs(seams)
    app.base_mesh.set_mesh(vertices, faces)
    app.base_point_vortex.set_vortex(
        faceids=np.array([4, 11], dtype=np.int32),
        bary=np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], dtype=np.float64),
        gamma=np.array([2.0, -0.75], dtype=np.float64),
        pos=np.zeros((2, 3), dtype=np.float64),
    )
    app.initialize_lifted_point_vortices()
    app.stream_function.set_homo_dirichlet_boundary()
    app.initialize_harmonic_component(coefficient)
    return app


def test_harmonic_basis_obeys_boundary_equation_energy_and_deck_parity() -> None:
    app = _initialized_app()
    basis = app.harmonic_basis
    mesh = app.cover_mesh
    potential = basis.potential.get()
    xi = basis.xi.get()
    energy = basis.energy.get()
    zeta = basis.zeta_face.get()

    boundaries = mesh.boundary_vertex_components.get()
    positive = min(boundaries, key=lambda component: int(np.min(component)))
    negative = boundaries[0] if positive is boundaries[1] else boundaries[1]
    np.testing.assert_allclose(potential[positive], 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(potential[negative], -0.5, rtol=0.0, atol=0.0)

    tau_vertex = app.cover.tau_vertex.get()
    tau_face = app.cover.tau_face.get()
    np.testing.assert_allclose(potential[tau_vertex], -potential, atol=1.0e-14)
    np.testing.assert_allclose(app.cover.apply_p1(xi), -xi, atol=1.0e-14)
    np.testing.assert_allclose(zeta[tau_face], zeta, rtol=0.0, atol=0.0)

    gradient = np.einsum(
        "fi,fij->fj",
        potential[mesh.F_verts.get()],
        mesh.grad_bary.get(),
    )
    face_energy = np.einsum("fi,fi,f->", gradient, gradient, mesh.F_area.get())
    assert energy == pytest.approx(face_energy, rel=1.0e-12)
    np.testing.assert_allclose(
        zeta,
        -np.cross(mesh.F_normal.get(), gradient) / energy,
        atol=1.0e-14,
    )

    # star(zeta) dot xi is the vector-proxy form of zeta wedge xi.
    pairing = np.einsum(
        "fi,fi,f->",
        np.cross(mesh.F_normal.get(), zeta),
        gradient,
        mesh.F_area.get(),
    )
    assert pairing == pytest.approx(1.0, rel=1.0e-12)

    laplacian_u = app.dec.d0_transpose(app.dec.star1.get() * xi)
    interior = np.ones(potential.shape[0], dtype=bool)
    interior[np.concatenate(boundaries)] = False
    np.testing.assert_allclose(laplacian_u[interior], 0.0, atol=1.0e-8)


def test_abel_jacobi_resource_sums_exactly_the_derived_lifted_particles() -> None:
    app = _initialized_app(coefficient=0.7)
    paired = app.point_vortex
    paired_values = app.harmonic_basis.interpolate_potential(
        (paired.face_ids.get(), paired.bary.get())
    )
    expected = float(np.dot(paired.gamma.get(), paired_values))
    assert app.abel_jacobi_coordinate.get() == pytest.approx(expected)

    originals = app.lifted_point_vortex
    original_values = app.harmonic_basis.interpolate_potential(
        (originals.face_ids.get(), originals.bary.get())
    )
    expected_from_originals = 2.0 * float(
        np.dot(originals.gamma.get(), original_values)
    )
    assert expected == pytest.approx(expected_from_originals)
    assert app.abel_jacobi_invariant.get() == pytest.approx(expected - 0.7)


def test_harmonic_velocity_sampling_is_facewise_constant_and_unsmoothed() -> None:
    app = _initialized_app(coefficient=1.25)
    face_id = int(app.lifted_point_vortex.face_ids.get()[0])
    probes = (
        np.array([face_id, face_id], dtype=np.int32),
        np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], dtype=np.float64),
    )

    sampled = app.harmonic_component.interpolate(probes)
    expected = 1.25 * app.harmonic_basis.zeta_face.get()[face_id]
    np.testing.assert_allclose(sampled, np.repeat(expected[None, :], 2, axis=0))


def test_rk4_uses_one_frozen_abel_jacobi_reference_and_commits_resources() -> None:
    app = _initialized_app(coefficient=0.25)

    class ZeroStreamVelocity:
        @staticmethod
        def interpolate(probes) -> np.ndarray:
            face_ids, _ = probes
            return np.zeros((face_ids.shape[0], 3), dtype=np.float64)

    app.stream_velocity = ZeroStreamVelocity()
    reference_coordinate = float(app.abel_jacobi_coordinate.get())
    reference_coefficient = float(app.harmonic_coefficient.get())
    invariant = reference_coordinate - reference_coefficient

    stage_samples: list[tuple[float, float]] = []
    original_interpolate = app.harmonic_component.interpolate

    def recording_interpolate(probes, coefficient=None) -> np.ndarray:
        stage_coordinate = app.harmonic_component.evaluate_coordinate()
        # Public monitoring resources remain on the accepted state throughout
        # all transient RK configurations.
        assert app.abel_jacobi_coordinate.get() == reference_coordinate
        stage_samples.append(
            (stage_coordinate, float(coefficient))
        )
        return original_interpolate(probes, coefficient=coefficient)

    app.harmonic_component.interpolate = recording_interpolate
    app.rk4_step(0.02)

    assert len(stage_samples) == 4
    for stage_coordinate, stage_coefficient in stage_samples:
        assert stage_coefficient == pytest.approx(
            reference_coefficient + stage_coordinate - reference_coordinate
        )

    accepted_coordinate = float(app.abel_jacobi_coordinate.get())
    expected_delta = accepted_coordinate - reference_coordinate
    assert app.abel_jacobi_step_delta.get() == pytest.approx(expected_delta)
    assert app.harmonic_coefficient.get() == pytest.approx(
        reference_coefficient + expected_delta
    )
    assert app.abel_jacobi_invariant.get() == pytest.approx(invariant)


def test_rejected_rk4_stage_restores_particles_and_keeps_accepted_resources() -> None:
    app = _initialized_app(coefficient=0.4)
    initial = app.vortex_state.current_state()
    initial_coordinate = float(app.abel_jacobi_coordinate.get())
    app.abel_jacobi_step_delta.set(0.125)

    original_rk4 = app.rk4

    def reject_after_trial(reference, velocity, dt):
        trial_velocity = velocity(reference)
        trial = original_rk4.advect_euler(reference, trial_velocity, 0.01)
        velocity(trial)
        raise ReduceTimestepError("synthetic rejected stage")

    app.rk4.step = reject_after_trial
    with pytest.raises(ReduceTimestepError, match="synthetic rejected stage"):
        app.rk4_step(0.02)

    np.testing.assert_array_equal(
        app.lifted_point_vortex.face_ids.get(),
        initial.face_ids,
    )
    np.testing.assert_allclose(app.lifted_point_vortex.bary.get(), initial.bary)
    assert app.harmonic_coefficient.get() == pytest.approx(0.4)
    assert app.abel_jacobi_step_delta.get() == pytest.approx(0.125)
    assert app.abel_jacobi_coordinate.get() == pytest.approx(initial_coordinate)


def test_solver_feedback_restores_accepted_harmonic_monitoring_state() -> None:
    app = _initialized_app()
    lifted = app.lifted_point_vortex

    class InputIO:
        point_values = {
            "cover_faceid": lifted.face_ids.get().copy(),
            "cover_bary": lifted.bary.get().copy(),
            "gamma": lifted.gamma.get().copy(),
        }
        detail_values = {
            HARMONIC_COEFFICIENT_ATTR: np.array([0.85]),
            ABEL_JACOBI_STEP_DELTA_ATTR: np.array([-0.125]),
        }

        def read_point(self, name, *, components=None):
            return self.point_values[name]

        def read_detail(self, name, *, dtype=None):
            try:
                value = self.detail_values[name]
            except KeyError as exc:
                raise KeyError(name) from exc
            return np.asarray(value, dtype=dtype)

    class Context:
        input = InputIO()

        def input_io(self, index):
            return self.input if index == 0 else None

    _restore_solver_state_from_input(Context(), app)

    assert app.harmonic_coefficient.get() == pytest.approx(0.85)
    assert app.abel_jacobi_step_delta.get() == pytest.approx(-0.125)
    assert app.abel_jacobi_invariant.get() == pytest.approx(
        app.abel_jacobi_coordinate.get() - 0.85
    )


def test_harmonic_coefficient_resource_has_a_central_zero_default() -> None:
    app = World().require(App)
    assert DEFAULT_HARMONIC_COEFFICIENT == 0.0
    assert app.harmonic_coefficient.get() == 0.0
    assert app.abel_jacobi_step_delta.get() == 0.0
