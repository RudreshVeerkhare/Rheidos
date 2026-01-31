import importlib
import os
import sys
import types

import pytest

from rheidos.houdini.runtime import reset_pipeline
from rheidos.houdini.runtime.dev_state import get_dev_state, pop_busy, push_busy
from rheidos.houdini.runtime.module_reloader import _purge_package, reload_project


def _reset_dev_state() -> None:
    state = get_dev_state()
    state.busy_count = 0
    state.busy_reasons.clear()
    state.reloading = False
    state.last_reload_error = None
    state.last_reload_at = None


def test_dev_state_busy_stack() -> None:
    _reset_dev_state()

    push_busy("cook")
    assert get_dev_state().busy_count == 1
    assert get_dev_state().busy_reasons[-1] == "cook"

    push_busy("solver")
    assert get_dev_state().busy_count == 2
    assert get_dev_state().busy_reasons[-1] == "solver"

    pop_busy()
    assert get_dev_state().busy_count == 1
    assert get_dev_state().busy_reasons[-1] == "cook"

    pop_busy()
    assert get_dev_state().busy_count == 0
    assert get_dev_state().busy_reasons == []


def test_purge_package_removes_modules() -> None:
    pkg = types.ModuleType("fakepkg")
    sub = types.ModuleType("fakepkg.sub")
    sys.modules["fakepkg"] = pkg
    sys.modules["fakepkg.sub"] = sub

    removed = _purge_package("fakepkg")
    assert "fakepkg" in removed
    assert "fakepkg.sub" in removed
    assert "fakepkg" not in sys.modules
    assert "fakepkg.sub" not in sys.modules


def test_reload_project_pulls_latest(tmp_path, monkeypatch) -> None:
    pkg_dir = tmp_path / "mypkg"
    pkg_dir.mkdir()
    init_file = pkg_dir / "__init__.py"
    init_file.write_text("value = 1\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("mypkg")
    assert mod.value == 1

    init_file.write_text("value = 2\n", encoding="utf-8")
    stat = init_file.stat()
    os.utime(init_file, (stat.st_atime, stat.st_mtime + 2))
    reloaded = reload_project("mypkg")
    assert reloaded.value == 2


def test_reset_pipeline_blocks_when_busy() -> None:
    _reset_dev_state()
    push_busy("cook")
    with pytest.raises(RuntimeError, match="Cannot reload"):
        reset_pipeline.reset_and_reload(pkg="rheidos")
    pop_busy()


def test_reset_pipeline_sequence(monkeypatch) -> None:
    _reset_dev_state()
    calls = []

    class DummySim:
        def close(self, reason: str) -> None:
            calls.append(("close", reason))

    dummy = DummySim()

    def fake_sync() -> bool:
        calls.append("sync")
        return True

    def fake_reset() -> None:
        calls.append("reset")

    def fake_init(cfg) -> None:
        calls.append(("init", cfg))

    def fake_reload(pkg: str):
        calls.append(("reload", pkg))
        return types.ModuleType(pkg)

    monkeypatch.setattr(reset_pipeline, "taichi_sync", fake_sync)
    monkeypatch.setattr(reset_pipeline, "taichi_reset", fake_reset)
    monkeypatch.setattr(reset_pipeline, "taichi_init", fake_init)
    monkeypatch.setattr(reset_pipeline, "reload_project", fake_reload)
    monkeypatch.setattr(reset_pipeline, "_rehydrate_context", lambda pkg: calls.append(("rehydrate", pkg)))
    monkeypatch.setattr(reset_pipeline, "get_sim_context", lambda create=False: dummy)
    monkeypatch.setattr(reset_pipeline, "set_sim_context", lambda sim: calls.append(("set_sim", sim)))
    monkeypatch.setattr(reset_pipeline, "get_runtime", lambda create=False: None)

    reset_pipeline.reset_and_reload(
        pkg="rheidos",
        taichi_cfg={"arch": "cpu"},
        rebuild_fn=lambda: calls.append("rebuild"),
    )

    assert calls == [
        ("close", "reset_and_reload"),
        ("set_sim", None),
        "sync",
        "reset",
        ("init", {"arch": "cpu"}),
        ("reload", "rheidos"),
        ("rehydrate", "rheidos"),
        "rebuild",
    ]


def test_reset_pipeline_uses_runtime_when_no_sim(monkeypatch) -> None:
    _reset_dev_state()
    calls = []

    class DummyRuntime:
        def nuke_all(self, reason: str, *, reset_taichi: bool = True) -> None:
            calls.append(("nuke_all", reason, reset_taichi))

    monkeypatch.setattr(reset_pipeline, "taichi_sync", lambda: True)
    monkeypatch.setattr(reset_pipeline, "taichi_reset", lambda: None)
    monkeypatch.setattr(reset_pipeline, "taichi_init", lambda cfg=None: None)
    monkeypatch.setattr(reset_pipeline, "reload_project", lambda pkg: types.ModuleType(pkg))
    monkeypatch.setattr(reset_pipeline, "_rehydrate_context", lambda pkg: None)
    monkeypatch.setattr(reset_pipeline, "get_sim_context", lambda create=False: None)
    monkeypatch.setattr(reset_pipeline, "set_sim_context", lambda sim: None)
    monkeypatch.setattr(reset_pipeline, "get_runtime", lambda create=False: DummyRuntime())

    reset_pipeline.reset_and_reload(pkg="rheidos")

    assert calls == [("nuke_all", "reset_and_reload", False)]
