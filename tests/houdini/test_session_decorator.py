from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
import sys
import warnings

import pytest

session_mod = importlib.import_module("rheidos.houdini.runtime.session")


class _FakeNode:
    def __init__(self, node_path: str) -> None:
        self._node_path = node_path

    def path(self) -> str:
        return self._node_path


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
    hou._pwd_node = _FakeNode("/obj/geo1/python1")
    hou.pwd = lambda: hou._pwd_node
    monkeypatch.setitem(sys.modules, "hou", hou)
    yield hou
    session_mod.set_sim_context(None)
    monkeypatch.delitem(sys.modules, "hou", raising=False)


def test_session_decorator_reuses_node_local_session(fake_hou) -> None:
    @session_mod.session
    def entry(session):
        return session

    first = entry()
    second = entry()

    assert first is second


def test_session_decorator_separates_node_local_sessions_by_node(fake_hou) -> None:
    @session_mod.session
    def entry(session):
        return session

    fake_hou._pwd_node = _FakeNode("/obj/geo1/python1")
    first = entry()

    fake_hou._pwd_node = _FakeNode("/obj/geo1/python2")
    second = entry()

    assert first is not second


def test_named_session_shares_across_nodes_in_same_hip(fake_hou) -> None:
    @session_mod.session("p1")
    def node1(session):
        return session

    @session_mod.session(key="p1")
    def node2(*, session):
        return session

    fake_hou._pwd_node = _FakeNode("/obj/geo1/python1")
    first = node1()

    fake_hou._pwd_node = _FakeNode("/obj/geo1/python2")
    second = node2()

    assert first is second


def test_named_session_is_scoped_by_hip_path(fake_hou) -> None:
    @session_mod.session("p1")
    def entry(session):
        return session

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

    with pytest.raises(
        TypeError, match="must accept a 'session' parameter"
    ):
        session_mod.session(lambda: None)


def test_session_decorator_rejects_explicit_session_argument(fake_hou) -> None:
    @session_mod.session
    def entry(session):
        return session

    with pytest.raises(TypeError, match="do not pass it explicitly"):
        entry(object())


def test_named_session_warns_once_for_mixed_owners(fake_hou) -> None:
    def owner_one(session):
        return session

    owner_one.__module__ = "app.owner_one"
    owner_one = session_mod.session("p1")(owner_one)

    def owner_two(session):
        return session

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


def test_get_or_create_session_without_key_is_backward_compatible(fake_hou) -> None:
    runtime = session_mod.ComputeRuntime()
    node = _FakeNode("/obj/geo1/python1")

    first = runtime.get_or_create_session(node)
    second = runtime.get_or_create_session(node)

    assert first is second


def test_session_decorator_raises_when_hou_is_unavailable(monkeypatch) -> None:
    @session_mod.session
    def entry(session):
        return session

    monkeypatch.delitem(sys.modules, "hou", raising=False)

    with pytest.raises(RuntimeError, match="Houdini 'hou' module not available"):
        entry()
