from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rheidos.apps.p2.double_obstacle.app import (
    App,
    VortexProjection,
    _read_harmonic_c,
    _rk4_step_with_abel_jacobi,
    _write_harmonic_state,
)
from rheidos.apps.p2.double_obstacle.harmonic_basis import HarmonicBasisModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import World


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
    def evaluate(self, probes, gamma: np.ndarray) -> np.ndarray:
        _faceids, bary = probes
        return np.array([float(np.sum(np.asarray(gamma) * bary[:, 0]))])


class _InputIO:
    def __init__(self, attrs: dict[str, np.ndarray]) -> None:
        self.attrs = attrs

    def read_detail(self, name: str, dtype):
        if name not in self.attrs:
            raise KeyError(name)
        return np.asarray(self.attrs[name], dtype=dtype)


class _Ctx:
    def __init__(self, attrs: dict[str, np.ndarray]) -> None:
        self.attrs = attrs
        self.writes = {}

    def input_io(self, index: int) -> _InputIO:
        assert index == 0
        return _InputIO(self.attrs)

    def write_detail(self, name: str, value, *, create: bool = True) -> None:
        self.writes[name] = (np.asarray(value, dtype=np.float64).copy(), create)


def test_double_obstacle_app_wires_velocity_and_aj_modules() -> None:
    mods = World().require(App)

    assert mods.harmonic_velocity.mesh is mods.mesh
    assert mods.harmonic_velocity.dual_harmonic_field is mods.harmonic_basis_field
    assert mods.combined_velocity.mesh is mods.mesh
    assert mods.combined_velocity.coexact_velocity is mods.coexact_velocity
    assert mods.combined_velocity.harmonic_velocity is mods.harmonic_velocity
    assert mods.velocity is mods.combined_velocity
    assert mods.abel_jacobi.mesh is mods.mesh
    assert mods.abel_jacobi.point_vortex is mods.point_vortex
    assert mods.abel_jacobi.harmonic_basis is mods.harmonic_basis_potential


def test_planar_double_obstacle_boundary_classification_uses_largest_loop_as_outer() -> None:
    world = World()
    mesh = world.require(SurfaceMeshModule)
    dec = world.require(DEC, mesh=mesh)
    harmonic_basis = world.require(
        HarmonicBasisModule,
        mesh=mesh,
        dec=dec,
        harmonic_dim=2,
    )
    mesh.V_pos.set(
        np.array(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [-0.5, -0.5, 0.0],
                [0.0, -0.5, 0.0],
                [0.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.9, 0.5, 0.0],
                [0.9, 0.9, 0.0],
                [0.5, 0.9, 0.0],
            ],
            dtype=np.float64,
        )
    )
    components = [
        np.array([4, 5, 6, 7], dtype=np.int32),
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([8, 9, 10, 11], dtype=np.int32),
    ]
    mesh.boundary_vertex_components.set(components)

    harmonic_basis.set_planar_double_obstacle_boundaries()

    np.testing.assert_array_equal(harmonic_basis.outer_boundary, components[1])
    assert len(harmonic_basis.inner_boundaries) == 2
    np.testing.assert_array_equal(harmonic_basis.inner_boundaries[0], components[0])
    np.testing.assert_array_equal(harmonic_basis.inner_boundaries[1], components[2])


def test_rk4_recomputes_double_obstacle_coefficients_from_trial_aj() -> None:
    gamma = np.array([1.0], dtype=np.float64)
    harmonic_velocity = _HarmonicVelocity()
    mods = SimpleNamespace(
        point_vortex=_PointVortex(gamma),
        harmonic_velocity=harmonic_velocity,
        combined_velocity=_CombinedVelocity(harmonic_velocity),
        coexact_velocity=None,
        abel_jacobi=_AbelJacobi(),
        harmonic_basis_potential=SimpleNamespace(dim=1),
        mesh=SimpleNamespace(
            F_normal=_ArrayRef(np.array([[0.0, 0.0, 1.0]], dtype=np.float64))
        ),
    )
    ref = VortexProjection(
        np.array([0], dtype=np.int32),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
    )

    def projector(points: np.ndarray) -> VortexProjection:
        points = np.asarray(points, dtype=np.float64)
        return VortexProjection(
            np.array([0], dtype=np.int32),
            np.array([[points[0, 0], 0.0, 0.0]], dtype=np.float64),
            points.copy(),
        )

    accepted, c_next, aj_next = _rk4_step_with_abel_jacobi(
        mods,
        ref,
        np.array([0.0], dtype=np.float64),
        1.0,
        projector,
    )

    np.testing.assert_allclose(
        np.asarray(harmonic_velocity.history).reshape(-1),
        np.array([1.0, 1.5, 1.75, 2.75, 2.708333333333333], dtype=np.float64),
    )
    np.testing.assert_allclose(accepted.pos, np.array([[2.708333333333333, 0.0, 0.0]]))
    np.testing.assert_allclose(aj_next, np.array([2.708333333333333]))
    np.testing.assert_allclose(c_next, np.array([2.708333333333333]))


def test_harmonic_detail_state_prefers_primary_attr_and_writes_legacy_alias() -> None:
    ctx = _Ctx(
        {
            "harmonic_c": np.array([1.0, 2.0], dtype=np.float64),
            "harmonic_coeff": np.array([3.0, 4.0], dtype=np.float64),
        }
    )

    np.testing.assert_allclose(_read_harmonic_c(ctx, 2), np.array([1.0, 2.0]))
    np.testing.assert_allclose(
        _read_harmonic_c(
            _Ctx({"harmonic_coeff": np.array([3.0, 4.0], dtype=np.float64)}),
            2,
        ),
        np.array([3.0, 4.0]),
    )

    _write_harmonic_state(
        ctx,
        aj=np.array([0.25, 0.5], dtype=np.float64),
        harmonic_c=np.array([1.0, 2.0], dtype=np.float64),
        invariant_c=np.array([-0.75, -1.5], dtype=np.float64),
    )

    np.testing.assert_allclose(ctx.writes["AJ"][0], np.array([0.25, 0.5]))
    np.testing.assert_allclose(ctx.writes["harmonic_c"][0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(ctx.writes["harmonic_coeff"][0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(ctx.writes["invariant_C"][0], np.array([-0.75, -1.5]))
