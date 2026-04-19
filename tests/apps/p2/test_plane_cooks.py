from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
import sys

import pytest

debug_mod = importlib.import_module("rheidos.houdini.debug")
session_mod = importlib.import_module("rheidos.houdini.runtime.session")


class _FakeGeometry:
    def __init__(self, name: str) -> None:
        self.name = name
        self.clear_calls = 0
        self.merged: list[object] = []

    def clear(self) -> None:
        self.clear_calls += 1

    def merge(self, other: object) -> None:
        self.merged.append(other)


class _FakeInputNode:
    def __init__(self, geo: _FakeGeometry) -> None:
        self._geo = geo

    def geometry(self) -> _FakeGeometry:
        return self._geo


class _FakeParm:
    def __init__(self, value) -> None:
        self._value = value

    def eval(self):
        return self._value


class _FakeNode:
    def __init__(
        self,
        node_path: str,
        *,
        inputs: list[_FakeInputNode | None],
        geo_out: _FakeGeometry,
    ) -> None:
        self._node_path = node_path
        self._inputs = list(inputs)
        self._geo_out = geo_out
        self._parms = {
            "reset_node": _FakeParm(0),
            "nuke_all": _FakeParm(0),
            "debug_enable": _FakeParm(0),
            "debug_break_next": _FakeParm(0),
        }

    def path(self) -> str:
        return self._node_path

    def inputs(self):
        return list(self._inputs)

    def geometry(self) -> _FakeGeometry:
        return self._geo_out

    def parm(self, name: str):
        return self._parms.get(name)


class _FakeHipFile:
    def __init__(self, hip_path: str) -> None:
        self._hip_path = hip_path

    def path(self) -> str:
        return self._hip_path


@pytest.fixture
def fake_hou(monkeypatch):
    hou = ModuleType("hou")
    hou.session = SimpleNamespace()
    hou.hipFile = _FakeHipFile("/tmp/test.hip")
    hou._pwd_node = None
    hou.pwd = lambda: hou._pwd_node
    hou.frame = lambda: 24.0
    hou.time = lambda: 1.0
    hou.fps = lambda: 24.0
    monkeypatch.setitem(sys.modules, "hou", hou)
    yield hou
    session_mod.set_sim_context(None)
    monkeypatch.delitem(sys.modules, "hou", raising=False)


@pytest.fixture(autouse=True)
def _disable_debug(monkeypatch):
    monkeypatch.setattr(
        debug_mod,
        "ensure_debug_server",
        lambda cfg, *, node=None: SimpleNamespace(),
    )
    monkeypatch.setattr(
        debug_mod, "consume_break_next_button", lambda node=None: False
    )
    monkeypatch.setattr(debug_mod, "request_break_next", lambda *, node=None: None)
    monkeypatch.setattr(debug_mod, "maybe_break_now", lambda *, node=None: None)


@pytest.fixture
def p1_plane_cook_module(monkeypatch):
    captured = {}

    def _capture(name):
        def _fn(ctx, *args, **kwargs):
            captured[name] = {"ctx": ctx, "args": args, "kwargs": kwargs}

        return _fn

    app_mod = ModuleType("rheidos.apps.p2.p1_avg_plane_app")
    app_mod.setup_p1_stream_function = _capture("setup_p1_stream_function")
    app_mod.interpolate_p1_stream_func = _capture("interpolate_p1_stream_func")
    app_mod.interpolate_p1_velocity = _capture("interpolate_p1_velocity")
    app_mod.rk4_advect = _capture("rk4_advect")

    monkeypatch.setitem(sys.modules, "rheidos.apps.p2.p1_avg_plane_app", app_mod)
    monkeypatch.delitem(sys.modules, "rheidos.apps.p2.p1_avg_plane_cook", raising=False)

    module = importlib.import_module("rheidos.apps.p2.p1_avg_plane_cook")
    return module, captured


@pytest.fixture
def p2_plane_cook_module(monkeypatch):
    captured = {}

    def _capture(name):
        def _fn(ctx, *args, **kwargs):
            captured[name] = {"ctx": ctx, "args": args, "kwargs": kwargs}

        return _fn

    app_mod = ModuleType("rheidos.apps.p2.p2_plane_app")
    app_mod.setup_p2_stream_function = _capture("setup_p2_stream_function")
    app_mod.interpolate_p2_stream_func = _capture("interpolate_p2_stream_func")
    app_mod.interpolate_p2_velocity = _capture("interpolate_p2_velocity")
    app_mod.rk4_advect = _capture("rk4_advect")

    monkeypatch.setitem(sys.modules, "rheidos.apps.p2.p2_plane_app", app_mod)
    monkeypatch.delitem(sys.modules, "rheidos.apps.p2.p2_plane_cook", raising=False)

    module = importlib.import_module("rheidos.apps.p2.p2_plane_cook")
    return module, captured


def test_p1_stream_setup_copies_mesh_input(fake_hou, p1_plane_cook_module) -> None:
    plane_cook, captured = p1_plane_cook_module
    mesh_geo = _FakeGeometry("mesh")
    vort_geo = _FakeGeometry("vort")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python1",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(vort_geo)],
        geo_out=geo_out,
    )

    plane_cook.p1_stream_setup()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [mesh_geo]
    assert captured["setup_p1_stream_function"]["ctx"].input_io(0).geo_in is mesh_geo


def test_p1_stream_interpolate_copies_probe_input(
    fake_hou,
    p1_plane_cook_module,
) -> None:
    plane_cook, captured = p1_plane_cook_module
    mesh_geo = _FakeGeometry("mesh")
    probe_geo = _FakeGeometry("probe")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python2",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(probe_geo)],
        geo_out=geo_out,
    )

    plane_cook.p1_stream_interpolate()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [probe_geo]
    assert captured["interpolate_p1_stream_func"]["ctx"].input_io(1).geo_in is probe_geo


def test_p1_velocity_interpolate_copies_probe_input(
    fake_hou,
    p1_plane_cook_module,
) -> None:
    plane_cook, captured = p1_plane_cook_module
    mesh_geo = _FakeGeometry("mesh")
    probe_geo = _FakeGeometry("probe")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python3",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(probe_geo)],
        geo_out=geo_out,
    )

    plane_cook.p1_velocity_interpolate()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [probe_geo]
    assert captured["interpolate_p1_velocity"]["ctx"].input_io(1).geo_in is probe_geo


def test_p1_velocity_rk4_advection_copies_vortex_input(
    fake_hou,
    p1_plane_cook_module,
) -> None:
    plane_cook, captured = p1_plane_cook_module
    mesh_geo = _FakeGeometry("mesh")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python4",
        inputs=[_FakeInputNode(mesh_geo)],
        geo_out=geo_out,
    )

    plane_cook.p1_velocity_rk4_advection()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [mesh_geo]
    assert captured["rk4_advect"]["ctx"].input_io(0).geo_in is mesh_geo


def test_p2_stream_setup_copies_mesh_input(fake_hou, p2_plane_cook_module) -> None:
    plane_cook, captured = p2_plane_cook_module
    mesh_geo = _FakeGeometry("mesh")
    vort_geo = _FakeGeometry("vort")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python5",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(vort_geo)],
        geo_out=geo_out,
    )

    plane_cook.p2_stream_setup()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [mesh_geo]
    assert captured["setup_p2_stream_function"]["ctx"].input_io(0).geo_in is mesh_geo


def test_p2_poisson_interpolate_copies_probe_input(
    fake_hou,
    p2_plane_cook_module,
) -> None:
    plane_cook, captured = p2_plane_cook_module
    mesh_geo = _FakeGeometry("mesh")
    probe_geo = _FakeGeometry("probe")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python6",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(probe_geo)],
        geo_out=geo_out,
    )

    plane_cook.p2_poisson_interpolate()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [probe_geo]
    assert captured["interpolate_p2_stream_func"]["ctx"].input_io(1).geo_in is probe_geo


def test_p2_velocity_interpolate_copies_probe_input(
    fake_hou,
    p2_plane_cook_module,
) -> None:
    plane_cook, captured = p2_plane_cook_module
    mesh_geo = _FakeGeometry("mesh")
    probe_geo = _FakeGeometry("probe")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python7",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(probe_geo)],
        geo_out=geo_out,
    )

    plane_cook.p2_vel_interpolate()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [probe_geo]
    assert captured["interpolate_p2_velocity"]["ctx"].input_io(1).geo_in is probe_geo


def test_p2_rk4_step_copies_vortex_input(fake_hou, p2_plane_cook_module) -> None:
    plane_cook, captured = p2_plane_cook_module
    mesh_geo = _FakeGeometry("mesh")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python8",
        inputs=[_FakeInputNode(mesh_geo)],
        geo_out=geo_out,
    )

    plane_cook.p2_rk4_step()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [mesh_geo]
    assert captured["rk4_advect"]["ctx"].input_io(0).geo_in is mesh_geo
