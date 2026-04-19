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

    def evalAsString(self) -> str:
        return str(self._value)

    def set(self, value) -> None:
        self._value = value


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


@pytest.fixture
def cook_sop_module(monkeypatch):
    captured = {}

    def _capture(name):
        def _fn(ctx, *args, **kwargs):
            captured[name] = {"ctx": ctx, "args": args, "kwargs": kwargs}

        return _fn

    app_mod = ModuleType("rheidos.apps.p2.app")
    app_mod.solve_p1_stream_function = _capture("solve_p1_stream_function")
    app_mod.sample_p1_stream_function = _capture("sample_p1_stream_function")
    p2_app_mod = ModuleType("rheidos.apps.p2.p2_app")
    p2_app_mod.solve_p2_stream_function = _capture("solve_p2_stream_function")
    p2_app_mod.sample_p2_stream_function = _capture("sample_p2_stream_function")
    p2_app_mod.sample_p2_velocity = _capture("sample_p2_velocity")
    p2_test_app_mod = ModuleType("rheidos.apps.p2.p2_test_app")
    p2_test_app_mod.p1_cook_test = _capture("p1_cook_test")
    p2_test_app_mod.p1_cook2_test = _capture("p1_cook2_test")
    p2_test_app_mod.p2_cook_test = _capture("p2_cook_test")
    p2_test_app_mod.p2_cook2_test = _capture("p2_cook2_test")

    monkeypatch.setitem(sys.modules, "rheidos.apps.p2.app", app_mod)
    monkeypatch.setitem(sys.modules, "rheidos.apps.p2.p2_app", p2_app_mod)
    monkeypatch.setitem(sys.modules, "rheidos.apps.p2.p2_test_app", p2_test_app_mod)
    monkeypatch.delitem(sys.modules, "rheidos.apps.p2.cook_sop", raising=False)

    module = importlib.import_module("rheidos.apps.p2.cook_sop")
    return module, captured


@pytest.fixture(autouse=True)
def _disable_debug(monkeypatch):
    monkeypatch.setattr(debug_mod, "ensure_debug_server", lambda cfg, *, node=None: SimpleNamespace())
    monkeypatch.setattr(debug_mod, "consume_break_next_button", lambda node=None: False)
    monkeypatch.setattr(debug_mod, "request_break_next", lambda *, node=None: None)
    monkeypatch.setattr(debug_mod, "maybe_break_now", lambda *, node=None: None)


def test_node1_copies_mesh_input_to_output(fake_hou, cook_sop_module) -> None:
    cook_sop, captured = cook_sop_module
    mesh_geo = _FakeGeometry("mesh")
    vort_geo = _FakeGeometry("vort")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python1",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(vort_geo)],
        geo_out=geo_out,
    )

    cook_sop.node1()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [mesh_geo]
    assert captured["solve_p1_stream_function"]["ctx"].input_io(0).geo_in is mesh_geo
    assert captured["solve_p1_stream_function"]["ctx"].output_io().geo_out is geo_out


def test_node2_copies_probe_input_to_output(fake_hou, cook_sop_module) -> None:
    cook_sop, captured = cook_sop_module
    mesh_geo = _FakeGeometry("mesh")
    probe_geo = _FakeGeometry("probe")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python2",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(probe_geo)],
        geo_out=geo_out,
    )

    cook_sop.node2()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [probe_geo]
    assert captured["sample_p1_stream_function"]["ctx"].input_io(1).geo_in is probe_geo
    assert captured["sample_p1_stream_function"]["ctx"].output_io().geo_out is geo_out


def test_node3_copies_mesh_input_to_output_and_passes_eps(fake_hou, cook_sop_module) -> None:
    cook_sop, captured = cook_sop_module
    mesh_geo = _FakeGeometry("mesh")
    vort_geo = _FakeGeometry("vort")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python3",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(vort_geo)],
        geo_out=geo_out,
    )
    fake_hou._pwd_node._parms["eps"] = _FakeParm(0.25)

    cook_sop.node3()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [mesh_geo]
    assert captured["solve_p2_stream_function"]["ctx"].input_io(0).geo_in is mesh_geo
    assert captured["solve_p2_stream_function"]["ctx"].output_io().geo_out is geo_out
    assert captured["solve_p2_stream_function"]["args"] == (0.25,)


def test_node4_copies_probe_input_to_output(fake_hou, cook_sop_module) -> None:
    cook_sop, captured = cook_sop_module
    mesh_geo = _FakeGeometry("mesh")
    probe_geo = _FakeGeometry("probe")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python4",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(probe_geo)],
        geo_out=geo_out,
    )

    cook_sop.node4()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [probe_geo]
    assert captured["sample_p2_stream_function"]["ctx"].input_io(1).geo_in is probe_geo
    assert captured["sample_p2_stream_function"]["ctx"].output_io().geo_out is geo_out


def test_interpolate_vel_copies_probe_input_to_output(fake_hou, cook_sop_module) -> None:
    cook_sop, captured = cook_sop_module
    mesh_geo = _FakeGeometry("mesh")
    probe_geo = _FakeGeometry("probe")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python5",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(probe_geo)],
        geo_out=geo_out,
    )

    cook_sop.interpolate_vel()

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [probe_geo]
    assert captured["sample_p2_velocity"]["ctx"].input_io(1).geo_in is probe_geo
    assert captured["sample_p2_velocity"]["ctx"].output_io().geo_out is geo_out


def test_legacy_aliases_point_to_descriptive_helpers() -> None:
    app_mod = importlib.import_module("rheidos.apps.p2.app")
    p2_app_mod = importlib.import_module("rheidos.apps.p2.p2_app")

    assert app_mod.cook is app_mod.solve_p1_stream_function
    assert app_mod.cook2 is app_mod.sample_p1_stream_function
    assert p2_app_mod.p2_cook is p2_app_mod.solve_p2_stream_function
    assert p2_app_mod.p2_cook2 is p2_app_mod.sample_p2_stream_function
    assert p2_app_mod.p2_interpolate_velocity is p2_app_mod.sample_p2_velocity
