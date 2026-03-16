from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rheidos.apps.point_vortex_p2 import app as p2_app


class _FakeGeo:
    def __init__(self, change_id: int | None, point_count: int, prim_count: int) -> None:
        self._change_id = change_id
        self._point_count = int(point_count)
        self._prim_count = int(prim_count)

    def dataId(self) -> int:
        if self._change_id is None:
            raise RuntimeError("no data id")
        return int(self._change_id)

    def intrinsicValue(self, name: str) -> int:
        if name == "pointcount":
            return self._point_count
        if name == "primitivecount":
            return self._prim_count
        raise KeyError(name)

    def points(self):
        return [None] * self._point_count

    def prims(self):
        return [None] * self._prim_count


class _FakeMeshIO:
    def __init__(self, geo: _FakeGeo, points: np.ndarray, triangles: np.ndarray) -> None:
        self.geo_in = geo
        self._points = points
        self._triangles = triangles
        self.read_point_calls = 0
        self.read_prim_calls = 0

    def read(self, owner: str, name: str, *, components: int | None = None):
        _ = owner, name, components
        self.read_point_calls += 1
        return self._points

    def read_point(self, name: str, *, components: int | None = None):
        _ = name, components
        self.read_point_calls += 1
        return self._points

    def read_prims(self, arity: int = 3):
        _ = arity
        self.read_prim_calls += 1
        return self._triangles


def test_resolve_step_dt_first_step_falls_back_to_ctx_dt():
    session = SimpleNamespace()
    ctx = SimpleNamespace(session=session, time=10.0, dt=0.25)

    dt = p2_app._resolve_step_dt(ctx)

    assert dt == 0.25
    assert session._p2_last_time == 10.0


def test_resolve_step_dt_uses_positive_time_delta():
    session = SimpleNamespace(_p2_last_time=2.0)
    ctx = SimpleNamespace(session=session, time=2.6, dt=0.1)

    dt = p2_app._resolve_step_dt(ctx)

    assert np.isclose(dt, 0.6)
    assert session._p2_last_time == 2.6


def test_resolve_step_dt_non_positive_delta_falls_back_then_default():
    session = SimpleNamespace(_p2_last_time=5.0)
    ctx = SimpleNamespace(session=session, time=5.0, dt=0.2)
    assert p2_app._resolve_step_dt(ctx) == 0.2

    ctx = SimpleNamespace(session=session, time=4.0, dt=0.0)
    assert p2_app._resolve_step_dt(ctx) == 0.01


def test_mesh_guard_unchanged_key_skips_reload(monkeypatch):
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    triangles = np.array([[0, 0, 0]], dtype=np.int32)
    mesh_io = _FakeMeshIO(_FakeGeo(change_id=42, point_count=1, prim_count=1), points, triangles)
    session = SimpleNamespace(_p2_mesh_input_key=(1, 1, 42))
    ctx = SimpleNamespace(session=session)

    calls: list[tuple[np.ndarray, np.ndarray]] = []
    monkeypatch.setattr(
        p2_app,
        "_load_mesh",
        lambda mods, pts, tris: calls.append((np.asarray(pts), np.asarray(tris))),
    )

    changed = p2_app._ensure_mesh_current(ctx, object(), mesh_io)

    assert changed is False
    assert calls == []
    assert mesh_io.read_point_calls == 0
    assert mesh_io.read_prim_calls == 0


def test_mesh_guard_changed_key_reloads(monkeypatch):
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    triangles = np.array([[0, 0, 0]], dtype=np.int32)
    mesh_io = _FakeMeshIO(_FakeGeo(change_id=99, point_count=1, prim_count=1), points, triangles)
    session = SimpleNamespace(_p2_mesh_input_key=(1, 1, 42))
    ctx = SimpleNamespace(session=session)

    calls: list[tuple[np.ndarray, np.ndarray]] = []
    monkeypatch.setattr(
        p2_app,
        "_load_mesh",
        lambda mods, pts, tris: calls.append((np.asarray(pts), np.asarray(tris))),
    )

    changed = p2_app._ensure_mesh_current(ctx, object(), mesh_io)

    assert changed is True
    assert len(calls) == 1
    assert mesh_io.read_point_calls == 1
    assert mesh_io.read_prim_calls == 1
    assert session._p2_mesh_input_key == (1, 1, 99)


def test_mesh_guard_missing_key_always_reloads(monkeypatch):
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    triangles = np.array([[0, 0, 0]], dtype=np.int32)
    mesh_io = _FakeMeshIO(_FakeGeo(change_id=None, point_count=1, prim_count=1), points, triangles)
    session = SimpleNamespace()
    ctx = SimpleNamespace(session=session)

    calls: list[tuple[np.ndarray, np.ndarray]] = []
    monkeypatch.setattr(
        p2_app,
        "_load_mesh",
        lambda mods, pts, tris: calls.append((np.asarray(pts), np.asarray(tris))),
    )

    changed_0 = p2_app._ensure_mesh_current(ctx, object(), mesh_io)
    changed_1 = p2_app._ensure_mesh_current(ctx, object(), mesh_io)

    assert changed_0 is True
    assert changed_1 is True
    assert len(calls) == 2
    assert mesh_io.read_point_calls == 2
    assert mesh_io.read_prim_calls == 2
