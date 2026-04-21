from types import SimpleNamespace
from typing import Optional

import numpy as np

import rheidos.apps.p2.p1_annulus_app as annulus_app
import rheidos.houdini.runtime.driver as driver_mod
from rheidos.apps.p2.p1_annulus_app import P1AnnulusHarmonicModule
from rheidos.compute import World


class _FakeCtx:
    def __init__(self, world: World) -> None:
        self._world = world

    def world(self) -> World:
        return self._world


def test_p1_annulus_module_reuses_cached_graph_and_exposes_legacy_attrs() -> None:
    world = World()
    ctx = _FakeCtx(world)

    first = P1AnnulusHarmonicModule(ctx)
    second = P1AnnulusHarmonicModule(ctx)

    assert first._graph is second._graph
    assert first.mesh is first._graph.mesh
    assert first.dec is first._graph.dec
    assert first.poisson is first._graph.poisson
    assert first.harmonic_stream is first._graph.harmonic_stream
    assert first.harmonic_vel is first._graph.harmonic_vel
    assert first.mesh.prefix.startswith(f"{first._graph.prefix}.")
    assert first.harmonic_stream.prefix.startswith(f"{first._graph.prefix}.")


def test_p1_annulus_module_child_modules_are_owned_by_graph() -> None:
    world = World()
    ctx = _FakeCtx(world)
    mods = P1AnnulusHarmonicModule(ctx)

    deps = world.module_dependencies()
    graph_key = mods._graph._module_key

    assert graph_key is not None
    assert mods.mesh._module_key in deps[graph_key]
    assert mods.dec._module_key in deps[graph_key]
    assert mods.poisson._module_key in deps[graph_key]
    assert mods.harmonic_stream._module_key in deps[graph_key]
    assert mods.harmonic_vel._module_key in deps[graph_key]


def test_p1_annulus_combined_stream_psi_uses_stream_resource_shape() -> None:
    world = World()
    mods = world.require(P1AnnulusHarmonicModule)

    mods.mesh.set_mesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([[0, 1, 2]], dtype=np.int32),
    )
    mods.stream_function.psi.set(np.array([1.0, -0.5, 2.0], dtype=np.float64))

    mods.combined_stream_function.psi.set(
        np.array([1.1, -0.3, 1.7], dtype=np.float64)
    )
    np.testing.assert_allclose(
        mods.combined_stream_function.psi.peek(),
        np.array([1.1, -0.3, 1.7], dtype=np.float64),
    )


class _FakeTB:
    def __init__(self, *, enabled: bool = True, cfg: Optional[object] = object()) -> None:
        self.enabled = enabled
        self.cfg = cfg
        self.scalar_calls: list[tuple[str, float, int]] = []
        self.flush_calls = 0

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalar_calls.append((tag, value, step))

    def flush(self) -> None:
        self.flush_calls += 1


class _FakeParm:
    def __init__(self, value: object) -> None:
        self._value = value

    def evalAsString(self) -> str:
        return str(self._value)


def test_log_harmonic_coefficient_writes_tensorboard_scalar() -> None:
    tb = _FakeTB()
    ctx = SimpleNamespace(frame=17.9, session=SimpleNamespace(tb=tb))
    mods = SimpleNamespace(
        combined_stream_function=SimpleNamespace(
            harmonic_coefficient=SimpleNamespace(get=lambda: np.float64(2.5))
        )
    )

    annulus_app._log_harmonic_coefficient(ctx, mods)

    assert tb.scalar_calls == [("p1_annulus/harmonic_coefficient", 2.5, 17)]
    assert tb.flush_calls == 1


def test_log_harmonic_coefficient_noops_without_configured_tensorboard() -> None:
    tb = _FakeTB(cfg=None)
    ctx = SimpleNamespace(frame=5.0, session=SimpleNamespace(tb=tb))
    mods = SimpleNamespace(
        combined_stream_function=SimpleNamespace(
            harmonic_coefficient=SimpleNamespace(get=lambda: np.float64(1.25))
        )
    )

    annulus_app._log_harmonic_coefficient(ctx, mods)

    assert tb.scalar_calls == []
    assert tb.flush_calls == 0


def test_ensure_harmonic_coefficient_tb_logger_configures_session_tb(monkeypatch) -> None:
    initial_tb = SimpleNamespace(enabled=False, cfg=None)
    session = SimpleNamespace(tb=initial_tb)
    node = SimpleNamespace(
        parm=lambda name: _FakeParm("/tmp/custom_tb") if name == "profile_logdir" else None
    )
    ctx = SimpleNamespace(session=session, node=node)
    configured_tb = _FakeTB()
    calls = {}

    monkeypatch.setattr(
        driver_mod,
        "_resolve_profile_logdir",
        lambda node_obj, config: calls.setdefault("logdir", "/tmp/resolved_tb"),
    )

    def _configure_tb_logger(session_obj, logdir: str) -> None:
        calls["configured_logdir"] = logdir
        session_obj.tb = configured_tb

    monkeypatch.setattr(driver_mod, "_configure_tb_logger", _configure_tb_logger)

    annulus_app._ensure_harmonic_coefficient_tb_logger(ctx)

    assert calls["logdir"] == "/tmp/resolved_tb"
    assert calls["configured_logdir"] == "/tmp/resolved_tb"
    assert session.tb is configured_tb
