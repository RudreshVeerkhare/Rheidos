from types import SimpleNamespace

import numpy as np

import rheidos.apps.p2.p1_annulus_app as annulus_app
from rheidos.apps.p2.p1_annulus_app import P1AnnulusHarmonicModule
from rheidos.compute import World


def test_p1_annulus_module_reuses_cached_graph_and_exposes_legacy_attrs() -> None:
    world = World()
    first = world.require(P1AnnulusHarmonicModule)
    second = world.require(P1AnnulusHarmonicModule)

    assert first is second
    assert first._graph is first
    assert first.mesh is first._graph.mesh
    assert first.dec is first._graph.dec
    assert first.poisson is first._graph.poisson
    assert first.harmonic_stream is first._graph.harmonic_stream
    assert first.harmonic_vel is first._graph.harmonic_vel
    assert first.poisson is first.stream_function.poisson


def test_p1_annulus_module_child_modules_are_owned_by_graph() -> None:
    world = World()
    mods = world.require(P1AnnulusHarmonicModule)

    deps = world.module_dependencies()
    graph_key = mods._graph._module_key

    assert graph_key is not None
    assert mods.mesh._module_key in deps[graph_key]
    assert mods.dec._module_key in deps[graph_key]
    assert mods.stream_function._module_key in deps[graph_key]
    assert mods.harmonic_stream._module_key in deps[graph_key]
    assert mods.combined_stream_function._module_key in deps[graph_key]
    assert mods.harmonic_vel._module_key in deps[graph_key]
    assert mods.vel._module_key in deps[graph_key]
    assert mods.rk4._module_key in deps[graph_key]


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


def test_p1_annulus_combined_stream_interpolate_uses_combined_psi(
    monkeypatch,
) -> None:
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
    monkeypatch.setattr(
        mods.combined_stream_function,
        "psi",
        SimpleNamespace(get=lambda: np.array([1.0, 2.0, 4.0], dtype=np.float64)),
    )

    def _unexpected_component_interpolate(_probes):
        raise AssertionError("combined interpolation should use combined psi")

    monkeypatch.setattr(
        mods.stream_function, "interpolate", _unexpected_component_interpolate
    )
    monkeypatch.setattr(
        mods.harmonic_stream, "interpolate", _unexpected_component_interpolate
    )

    result = mods.combined_stream_function.interpolate(
        (
            np.array([0], dtype=np.int32),
            np.array([[0.2, 0.3, 0.5]], dtype=np.float64),
        )
    )

    np.testing.assert_allclose(result, np.array([2.8], dtype=np.float64))


def test_log_harmonic_coefficient_uses_rheidos_logger(monkeypatch) -> None:
    calls = []
    ctx = SimpleNamespace(frame=17.9)
    mods = SimpleNamespace(
        combined_stream_function=SimpleNamespace(
            harmonic_coefficient=SimpleNamespace(get=lambda: np.float64(2.5))
        )
    )
    monkeypatch.setattr(
        annulus_app.logger,
        "log",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    annulus_app._log_harmonic_coefficient(ctx, mods)

    assert calls == [
        (
            ("harmonic_coefficient", 2.5),
            {"category": "p1_annulus", "flush": True},
        )
    ]
