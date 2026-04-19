from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rheidos.apps.p2 import p2_plane_app
from rheidos.apps.p2.modules.intergrator.rk4 import RK4IntegratorModule
from rheidos.compute import World


class _FakeResource:
    def __init__(self, value) -> None:
        self._value = np.array(value, copy=True)

    def get(self):
        return np.array(self._value, copy=True)

    def set(self, value) -> None:
        self._value = np.array(value, copy=True)


class _FakeMesh:
    def project_on_nearest_face(self, points: np.ndarray):
        points = np.asarray(points, dtype=np.float64)
        count = points.shape[0]
        return (
            np.zeros(count, dtype=np.int32),
            np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (count, 1)),
            points.copy(),
        )


class _FakePointVortex:
    def __init__(self) -> None:
        self.gamma = _FakeResource(np.array([1.0], dtype=np.float32))
        self.pos_world = _FakeResource(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))

    def set_vortex(self, faceids, bary, gamma, pos) -> None:
        self.gamma.set(gamma)
        self.pos_world.set(pos)


class _FakeVelocity:
    def interpolate(self, probes) -> np.ndarray:
        faceids, _ = probes
        return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (len(faceids), 1))


class _FakeInputIO:
    def __init__(self, point_vortex: _FakePointVortex) -> None:
        self._point_vortex = point_vortex

    def read_point(self, name: str, components=None):
        if name == "P":
            return self._point_vortex.pos_world.get()
        if name == "bary":
            return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        if name == "gamma":
            return self._point_vortex.gamma.get()
        if name == "faceid":
            return np.array([0], dtype=np.int32)
        raise KeyError(name)


class _FakeCtx:
    def __init__(
        self,
        world: World,
        *,
        dt: float,
        time: float,
        point_vortex: _FakePointVortex,
    ) -> None:
        self._world = world
        self.dt = dt
        self.time = time
        self.writes = {}
        self._vort_io = _FakeInputIO(point_vortex)

    def world(self) -> World:
        return self._world

    def input_io(self, index: int):
        if index == 0:
            return self._vort_io
        return None

    def write_point(self, name: str, value) -> None:
        self.writes[name] = np.array(value, copy=True)


def test_rk4_advect_reuses_integrator_and_ignores_houdini_timing(
    monkeypatch,
) -> None:
    world = World()
    fake_mods = SimpleNamespace(
        mesh=_FakeMesh(),
        point_vortex=_FakePointVortex(),
        p2_space=object(),
        p2_stream_func=object(),
        p2_vel=_FakeVelocity(),
        rk4=world.require(RK4IntegratorModule),
    )

    monkeypatch.setattr(p2_plane_app, "P2PlaneModule", lambda ctx: fake_mods)

    ctx_first = _FakeCtx(world, dt=0.25, time=1.0, point_vortex=fake_mods.point_vortex)
    p2_plane_app.rk4_advect(ctx_first)
    np.testing.assert_allclose(
        fake_mods.point_vortex.pos_world.get(),
        np.array([[0.001, 0.0, 0.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        ctx_first.writes["P"],
        np.array([[0.001, 0.0, 0.0]], dtype=np.float64),
    )

    ctx_second = _FakeCtx(
        world,
        dt=10.0,
        time=200.0,
        point_vortex=fake_mods.point_vortex,
    )
    p2_plane_app.rk4_advect(ctx_second)
    np.testing.assert_allclose(
        fake_mods.point_vortex.pos_world.get(),
        np.array([[0.002, 0.0, 0.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        ctx_second.writes["P"],
        np.array([[0.002, 0.0, 0.0]], dtype=np.float64),
    )
