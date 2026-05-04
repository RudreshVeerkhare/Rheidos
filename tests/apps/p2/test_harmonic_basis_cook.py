from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

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
        basis_id: int = 0,
    ) -> None:
        self._node_path = node_path
        self._inputs = list(inputs)
        self._geo_out = geo_out
        self._parms = {
            "reset_node": _FakeParm(0),
            "nuke_all": _FakeParm(0),
            "debug_enable": _FakeParm(0),
            "debug_break_next": _FakeParm(0),
            "basis_id": _FakeParm(basis_id),
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
def harmonic_basis_cook_module(monkeypatch):
    captured = {}

    def _capture(name):
        def _fn(ctx, *args, **kwargs):
            captured[name] = {"ctx": ctx, "args": args, "kwargs": kwargs}

        return _fn

    app_mod = ModuleType("rheidos.apps.p2.higher_genus.harmonic_basis.app")
    app_mod.setup_mesh = _capture("setup_mesh")
    app_mod.tree_cotree = _capture("tree_cotree")
    app_mod.interpolate_harmonic_basis_velocity = _capture(
        "interpolate_harmonic_basis_velocity"
    )

    monkeypatch.setitem(
        sys.modules,
        "rheidos.apps.p2.higher_genus.harmonic_basis.app",
        app_mod,
    )
    monkeypatch.delitem(
        sys.modules,
        "rheidos.apps.p2.higher_genus.harmonic_basis.cook",
        raising=False,
    )

    module = importlib.import_module("rheidos.apps.p2.higher_genus.harmonic_basis.cook")
    return module, captured


def test_harmonic_basis_velocity_node_copies_probe_input_and_forwards_basis_id(
    fake_hou,
    harmonic_basis_cook_module,
) -> None:
    cook, captured = harmonic_basis_cook_module
    mesh_geo = _FakeGeometry("mesh")
    probe_geo = _FakeGeometry("probe")
    geo_out = _FakeGeometry("out")
    fake_hou._pwd_node = _FakeNode(
        "/obj/geo1/python_harmonic_velocity",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(probe_geo)],
        geo_out=geo_out,
    )

    cook.interpolate_harmonic_basis_velocity_node(basis_id=3)

    assert geo_out.clear_calls == 1
    assert geo_out.merged == [probe_geo]
    call = captured["interpolate_harmonic_basis_velocity"]
    ctx = call["ctx"]
    assert ctx.input_io(1).geo_in is probe_geo
    assert call["kwargs"] == {"basis_id": 3}
