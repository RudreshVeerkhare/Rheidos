from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rheidos.apps.p2.higher_genus.vortex_dynamics.app import (
    VortexProjection,
    _rk4_step_with_abel_jacobi,
)


class _ArrayRef:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def get(self) -> np.ndarray:
        return self._value


class _PointVortex:
    def __init__(self, gamma: np.ndarray) -> None:
        self.gamma = _ArrayRef(gamma)
        self.states = []

    def set_vortex(self, faceids, bary, gamma, pos) -> None:
        self.states.append(
            (
                np.asarray(faceids).copy(),
                np.asarray(bary).copy(),
                np.asarray(gamma).copy(),
                np.asarray(pos).copy(),
            )
        )


class _HarmonicVelocity:
    def __init__(self) -> None:
        self.current = None
        self.history = []

    def set_coefficients(self, coefficients: np.ndarray) -> None:
        self.current = np.asarray(coefficients, dtype=np.float64).copy()
        self.history.append(self.current.copy())


class _CombinedVelocity:
    def __init__(self, harmonic_velocity: _HarmonicVelocity) -> None:
        self.harmonic_velocity = harmonic_velocity

    def interpolate(self, probes) -> np.ndarray:
        faceids, _bary = probes
        coeff = float(self.harmonic_velocity.current[0])
        return np.repeat([[coeff, 0.0, 0.0]], len(faceids), axis=0)


class _AbelJacobi:
    def delta_aj(self, start_probes, end_probes, *, pos0, pos1) -> np.ndarray:
        del start_probes, end_probes
        return (np.asarray(pos1)[:, 0] - np.asarray(pos0)[:, 0])[:, None]


def test_rk4_reconstructs_trial_coefficients_from_aj_delta() -> None:
    gamma = np.array([2.0], dtype=np.float64)
    harmonic_velocity = _HarmonicVelocity()
    mods = SimpleNamespace(
        point_vortex=_PointVortex(gamma),
        harmonic_velocity=harmonic_velocity,
        combined_velocity=_CombinedVelocity(harmonic_velocity),
        abel_jacobi=_AbelJacobi(),
        mesh=SimpleNamespace(
            F_normal=_ArrayRef(np.array([[0.0, 0.0, 1.0]], dtype=np.float64))
        ),
    )
    ref = VortexProjection(
        np.array([0], dtype=np.int32),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    )

    def projector(points: np.ndarray) -> VortexProjection:
        return VortexProjection(
            np.array([0], dtype=np.int32),
            np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            np.asarray(points, dtype=np.float64).copy(),
        )

    accepted, c_next, dA_final = _rk4_step_with_abel_jacobi(
        mods,
        ref,
        np.array([1.0], dtype=np.float64),
        1.0,
        projector,
    )

    # The stage coefficients are 1, 2, 3, and 7 for this toy ODE. If RK4 used
    # stale c_ref at trial stages, every recorded stage would stay at 1.
    np.testing.assert_allclose(
        np.asarray(harmonic_velocity.history).reshape(-1),
        np.array([1.0, 2.0, 3.0, 7.0, 7.0], dtype=np.float64),
    )
    np.testing.assert_allclose(accepted.pos, np.array([[3.0, 0.0, 0.0]]))
    np.testing.assert_allclose(dA_final, np.array([6.0], dtype=np.float64))
    np.testing.assert_allclose(c_next, np.array([7.0], dtype=np.float64))
