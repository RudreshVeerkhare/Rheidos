from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
import sys
import warnings

import pytest

debug_mod = importlib.import_module("rheidos.houdini.debug")
driver_mod = importlib.import_module("rheidos.houdini.runtime.driver")
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
        inputs: list[_FakeInputNode | None] | None = None,
        geo_out: _FakeGeometry | None = None,
        parms: dict[str, object] | None = None,
    ) -> None:
        self._node_path = node_path
        self._inputs = list(inputs or [])
        self._geo_out = geo_out or _FakeGeometry("out")
        parm_values = {
            "reset_node": 0,
            "nuke_all": 0,
            "debug_enable": 0,
            "debug_break_next": 0,
        }
        if parms:
            parm_values.update(parms)
        self._parms = {name: _FakeParm(value) for name, value in parm_values.items()}

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
    mesh_geo = _FakeGeometry("mesh")
    probe_geo = _FakeGeometry("probe")

    hou = ModuleType("hou")
    hou.session = SimpleNamespace()
    hou.hipFile = _FakeHipFile("/tmp/test.hip")
    hou._pwd_node = _FakeNode(
        "/obj/geo1/python1",
        inputs=[_FakeInputNode(mesh_geo), _FakeInputNode(probe_geo)],
        geo_out=_FakeGeometry("out"),
    )
    hou.pwd = lambda: hou._pwd_node
    hou.frame = lambda: 24.0
    hou.time = lambda: 1.0
    hou.fps = lambda: 24.0

    monkeypatch.setitem(sys.modules, "hou", hou)
    yield hou
    session_mod.set_sim_context(None)
    monkeypatch.delitem(sys.modules, "hou", raising=False)


def test_session_decorator_injects_ctx_with_explicit_input_and_output_io(fake_hou) -> None:
    @session_mod.session
    def entry(ctx):
        return ctx

    ctx = entry()

    assert ctx.session is session_mod.get_runtime().get_or_create_session(fake_hou._pwd_node)
    assert ctx.input_io(0).geo_in is fake_hou._pwd_node.inputs()[0].geometry()
    assert ctx.input_io(0).geo_out is None
    assert ctx.input_io(1).geo_in is fake_hou._pwd_node.inputs()[1].geometry()
    assert ctx.output_geo is fake_hou._pwd_node.geometry()
    assert ctx.output_io().geo_in is fake_hou._pwd_node.geometry()
    assert ctx.output_io().geo_out is fake_hou._pwd_node.geometry()


def test_session_decorator_reuses_node_local_session(fake_hou) -> None:
    @session_mod.session
    def entry(ctx):
        return ctx.session

    first = entry()
    second = entry()

    assert first is second


def test_session_decorator_separates_node_local_sessions_by_node(fake_hou) -> None:
    @session_mod.session
    def entry(ctx):
        return ctx.session

    fake_hou._pwd_node = _FakeNode("/obj/geo1/python1")
    first = entry()

    fake_hou._pwd_node = _FakeNode("/obj/geo1/python2")
    second = entry()

    assert first is not second


def test_named_session_shares_across_nodes_in_same_hip(fake_hou) -> None:
    @session_mod.session("p1")
    def node1(ctx):
        return ctx.session

    @session_mod.session(key="p1")
    def node2(ctx):
        return ctx.session

    fake_hou._pwd_node = _FakeNode("/obj/geo1/python1")
    first = node1()

    fake_hou._pwd_node = _FakeNode("/obj/geo1/python2")
    second = node2()

    assert first is second


def test_named_session_is_scoped_by_hip_path(fake_hou) -> None:
    @session_mod.session("p1")
    def entry(ctx):
        return ctx.session

    fake_hou.hipFile._hip_path = "/tmp/a.hip"
    first = entry()

    fake_hou.hipFile._hip_path = "/tmp/b.hip"
    second = entry()

    assert first is not second


def test_session_decorator_validates_key_and_signature(fake_hou) -> None:
    del fake_hou

    with pytest.raises(TypeError, match="session key must be a string"):
        session_mod.session(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="session key must be a non-empty string"):
        session_mod.session("   ")

    with pytest.raises(TypeError, match="must be defined as"):
        session_mod.session(lambda: None)

    with pytest.raises(TypeError, match="must be defined as"):
        session_mod.session(lambda session: None)

    def _extra(ctx, other):
        return ctx, other

    with pytest.raises(TypeError, match="must be defined as"):
        session_mod.session(_extra)


def test_session_decorator_rejects_explicit_ctx_argument(fake_hou) -> None:
    @session_mod.session
    def entry(ctx):
        return ctx

    with pytest.raises(TypeError, match="do not pass arguments explicitly"):
        entry(object())


def test_named_session_warns_once_for_mixed_owners(fake_hou) -> None:
    def owner_one(ctx):
        return ctx.session

    owner_one.__module__ = "app.owner_one"
    owner_one = session_mod.session("p1")(owner_one)

    def owner_two(ctx):
        return ctx.session

    owner_two.__module__ = "app.owner_two"
    owner_two = session_mod.session("p1")(owner_two)

    fake_hou._pwd_node = _FakeNode("/obj/geo1/python1")
    first = owner_one()

    fake_hou._pwd_node = _FakeNode("/obj/geo1/python2")
    with pytest.warns(RuntimeWarning, match="Named session 'p1'"):
        second = owner_two()

    assert first is second
    assert first.log_entries[-1]["message"] == "session.named_owner_mismatch"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        owner_two()
    assert caught == []


def test_named_session_reset_button_targets_shared_key(fake_hou) -> None:
    runtime = session_mod.get_runtime()
    node1 = _FakeNode("/obj/geo1/python1")
    shared = runtime.get_or_create_session(node1, key="p1")
    shared.stats["marker"] = 1

    @session_mod.session("p1")
    def entry(ctx):
        return ctx.session

    fake_hou._pwd_node = _FakeNode("/obj/geo1/python2", parms={"reset_node": 1})
    session_obj = entry()

    assert session_obj is shared
    assert session_obj.stats["last_reset_reason"] == "reset_node button"
    assert "marker" not in session_obj.stats
    assert fake_hou._pwd_node.parm("reset_node").eval() == 0


def test_debugger_false_skips_debug_setup(fake_hou, monkeypatch) -> None:
    def _fail(*args, **kwargs):
        raise AssertionError("debug setup should be skipped")

    monkeypatch.setattr(debug_mod, "ensure_debug_server", _fail)

    @session_mod.session
    def entry(ctx):
        return ctx

    entry()


def test_debugger_true_forces_enabled_flag(fake_hou, monkeypatch) -> None:
    captured = {}
    monkeypatch.delenv("RHEDIOS_DEBUG", raising=False)
    monkeypatch.delenv("RHEIDOS_DEBUG", raising=False)
    fake_hou._pwd_node = _FakeNode("/obj/geo1/python1", parms={"debug_enable": 0})

    def _capture(cfg, *, node=None):
        captured["cfg"] = cfg
        captured["node"] = node
        return SimpleNamespace()

    monkeypatch.setattr(debug_mod, "ensure_debug_server", _capture)
    monkeypatch.setattr(debug_mod, "consume_break_next_button", lambda node=None: False)
    monkeypatch.setattr(debug_mod, "request_break_next", lambda *, node=None: None)
    monkeypatch.setattr(debug_mod, "maybe_break_now", lambda *, node=None: None)

    @session_mod.session(debugger=True)
    def entry(ctx):
        return ctx

    entry()

    assert captured["cfg"].enabled is True
    assert captured["node"] is fake_hou._pwd_node


def test_profiler_false_skips_profiler_setup(fake_hou, monkeypatch) -> None:
    def _fail(*args, **kwargs):
        raise AssertionError("profiler setup should be skipped")

    monkeypatch.setattr(driver_mod, "_configure_profiler", _fail)

    @session_mod.session
    def entry(ctx):
        return ctx

    entry()


def test_profiler_true_forces_default_profiler_config(fake_hou, monkeypatch) -> None:
    captured = {}

    def _capture(session_obj, config, node):
        captured["session"] = session_obj
        captured["config"] = config
        captured["node"] = node

    monkeypatch.setattr(driver_mod, "_configure_profiler", _capture)

    @session_mod.session(profiler=True)
    def entry(ctx):
        return ctx.session

    session_obj = entry()
    config = captured["config"]

    assert captured["session"] is session_obj
    assert captured["node"] is fake_hou._pwd_node
    assert config.profile is True
    assert config.profile_logdir is None
    assert config.profile_export_hz == 5.0
    assert config.profile_mode == "coarse"
    assert config.profile_trace_cooks == 64
    assert config.profile_trace_edges == 20000
    assert config.profile_overhead is False
    assert config.profile_taichi is True
    assert config.profile_taichi_every == 30
    assert config.profile_taichi_sync is True
    assert config.profile_taichi_scoped_once is False


def test_named_session_reset_and_nuke_all_cover_shared_keys(fake_hou) -> None:
    runtime = session_mod.ComputeRuntime()
    node1 = _FakeNode("/obj/geo1/python1")
    node2 = _FakeNode("/obj/geo1/python2")

    shared = runtime.get_or_create_session(node1, key="p1")
    shared.stats["marker"] = 1

    runtime.reset_session(node2, "reset shared", key="p1")
    assert shared.stats["last_reset_reason"] == "reset shared"
    assert "marker" not in shared.stats

    runtime.nuke_all("wipe", reset_taichi=False)
    assert runtime.sessions == {}


def test_get_or_create_session_without_key_reuses_node_local_session(fake_hou) -> None:
    runtime = session_mod.ComputeRuntime()
    node = _FakeNode("/obj/geo1/python1")

    first = runtime.get_or_create_session(node)
    second = runtime.get_or_create_session(node)

    assert first is second


def test_session_decorator_raises_when_hou_is_unavailable(monkeypatch) -> None:
    @session_mod.session
    def entry(ctx):
        return ctx

    monkeypatch.delitem(sys.modules, "hou", raising=False)

    with pytest.raises(RuntimeError, match="Houdini 'hou' module not available"):
        entry()
