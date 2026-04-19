from types import SimpleNamespace

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
