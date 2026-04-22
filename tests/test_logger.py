from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

logger_mod = importlib.import_module("rheidos.logger")
from rheidos import logger as top_logger
from rheidos.compute import ModuleBase, ProducerContext, ResourceSpec, World, producer
from rheidos.logger import logger as submodule_logger


class _FakeTBLogger:
    instances: list["_FakeTBLogger"] = []

    def __init__(self, cfg=None, *, enabled: bool = True) -> None:
        self.cfg = cfg
        self.enabled = enabled
        self.scalar_calls: list[tuple[str, float, int]] = []
        self.flush_calls = 0
        self.close_calls = 0
        _FakeTBLogger.instances.append(self)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalar_calls.append((tag, value, step))

    def flush(self) -> None:
        self.flush_calls += 1

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture(autouse=True)
def _reset_logger(monkeypatch):
    logger_mod._reset_for_tests()
    _FakeTBLogger.instances.clear()
    monkeypatch.setattr(logger_mod, "TBLogger", _FakeTBLogger)
    yield
    logger_mod._reset_for_tests()


def test_logger_exported_from_top_level() -> None:
    assert top_logger is submodule_logger


def test_logger_configure_and_log_creates_standalone_run(tmp_path: Path) -> None:
    top_logger.configure(logdir=str(tmp_path), run_name="My Run")
    top_logger.log("energy", np.float64(1.25), flush=True)

    writer = _FakeTBLogger.instances[-1]
    assert writer.scalar_calls == [("simulation/energy", 1.25, 1)]
    assert writer.flush_calls == 1

    run_dir = Path(writer.cfg.logdir)
    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("run-my-run-0001__")

    latest = json.loads((tmp_path / "latest-run.json").read_text(encoding="utf-8"))
    assert latest["custom_name"] == "My Run"
    assert latest["run_dir"] == str(run_dir)
    assert latest["run_id"] == 1

    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_meta["custom_name"] == "My Run"
    assert run_meta["run_dir"] == str(run_dir)
    assert run_meta["run_id"] == 1


def test_logger_rejects_non_scalars_and_late_config_changes(tmp_path: Path) -> None:
    top_logger.configure(logdir=str(tmp_path), run_name="stable")

    with pytest.raises(TypeError, match="scalar values"):
        top_logger.log("bad", np.array([1.0], dtype=np.float64))

    assert _FakeTBLogger.instances == []

    top_logger.log("good", 2.0)

    with pytest.raises(RuntimeError, match="run_name"):
        top_logger.configure(run_name="other")

    with pytest.raises(RuntimeError, match="logdir"):
        top_logger.configure(logdir=str(tmp_path / "other"))


def test_logger_uses_ambient_step_hints_and_monotonic_fallback(tmp_path: Path) -> None:
    scope = logger_mod._make_scope(default_logdir=str(tmp_path))

    with logger_mod._activate_scope(scope, step_hint=17):
        top_logger.configure(run_name="ambient")
        top_logger.log("solver/residual", 0.5)
        top_logger.log("energy", 1.0, step=23)

    with logger_mod._activate_scope(scope):
        top_logger.log("state", 2.0)

    writer = _FakeTBLogger.instances[-1]
    assert writer.scalar_calls == [
        ("solver/residual", 0.5, 17),
        ("simulation/energy", 1.0, 23),
        ("simulation/state", 2.0, 24),
    ]


def test_logger_run_ids_ignore_legacy_session_dirs(tmp_path: Path) -> None:
    (tmp_path / "session-20260420-120000-1234").mkdir()
    (tmp_path / "run-0002__2026-04-20_10-00-00").mkdir()
    (tmp_path / "run-previous-0007__2026-04-20_11-00-00").mkdir()

    top_logger.configure(logdir=str(tmp_path), run_name="Fresh Run")
    top_logger.log("value", 1.0)

    run_dir = Path(_FakeTBLogger.instances[-1].cfg.logdir)
    assert run_dir.name.startswith("run-fresh-run-0008__")


def test_logger_works_inside_compute_producer(tmp_path: Path) -> None:
    top_logger.configure(logdir=str(tmp_path), run_name="compute")
    world = World()

    class DemoModule(ModuleBase):
        NAME = "demo"

        def __init__(self, world: World, *, scope: str = "") -> None:
            super().__init__(world, scope=scope)
            self.value = self.resource(
                "value",
                declare=True,
                spec=ResourceSpec(kind="numpy", dtype=np.float64, shape=(1,)),
                buffer=np.array([4.0], dtype=np.float64),
            )
            self.out = self.resource(
                "out",
                spec=ResourceSpec(kind="numpy", dtype=np.float64, shape=(1,)),
            )
            self.bind_producers()

        @producer(inputs=("value",), outputs=("out",))
        def square(self, ctx: ProducerContext) -> None:
            value = float(ctx.inputs.value.get()[0])
            top_logger.log("producer_value", value, category="compute")
            ctx.commit(out=np.array([value * value], dtype=np.float64))

    module = world.require(DemoModule)
    world.reg.ensure(module.out.name)

    writer = _FakeTBLogger.instances[-1]
    assert writer.scalar_calls == [("compute/producer_value", 4.0, 1)]
    np.testing.assert_allclose(module.out.peek(), np.array([16.0], dtype=np.float64))
