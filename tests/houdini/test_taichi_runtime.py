import os
import sys
import types

from rheidos.houdini.runtime import taichi_runtime


def test_taichi_init_sets_offline_cache(monkeypatch) -> None:
    fake = types.ModuleType("taichi")
    calls = {}
    fake.cpu = object()

    def is_initialized():
        return False

    def init(**cfg):
        calls["cfg"] = cfg

    fake.is_initialized = is_initialized
    fake.init = init

    monkeypatch.delenv("TI_OFFLINE_CACHE", raising=False)
    monkeypatch.setitem(sys.modules, "taichi", fake)

    taichi_runtime.taichi_init({"arch": "cpu"})

    cfg = calls["cfg"]
    assert cfg["arch"] is fake.cpu
    assert cfg["offline_cache"] is False
    assert os.environ["TI_OFFLINE_CACHE"] == "0"


def test_taichi_reset_skips_when_not_initialized(monkeypatch) -> None:
    fake = types.ModuleType("taichi")
    calls = {"reset": 0}

    def is_initialized():
        return False

    def reset():
        calls["reset"] += 1

    fake.is_initialized = is_initialized
    fake.reset = reset
    monkeypatch.setitem(sys.modules, "taichi", fake)

    taichi_runtime.taichi_reset()
    assert calls["reset"] == 0


def test_taichi_sync_handles_exception(monkeypatch) -> None:
    fake = types.ModuleType("taichi")

    def sync():
        raise RuntimeError("boom")

    fake.sync = sync
    monkeypatch.setitem(sys.modules, "taichi", fake)

    assert taichi_runtime.taichi_sync() is False
