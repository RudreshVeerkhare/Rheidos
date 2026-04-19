from __future__ import annotations

from rheidos.apps.p2.app import P1Module as P1StreamFacade
from rheidos.apps.p2.p1_avg_plane_app import P1PlaneModule
from rheidos.apps.p2.p2_app import P2Module as P2StreamFacade
from rheidos.apps.p2.p2_plane_app import P2PlaneModule
from rheidos.apps.p2.p2_test_app import P1Module as P1TestFacade
from rheidos.apps.p2.p2_test_app import P2Module as P2TestFacade
from rheidos.compute import World


class _FakeCtx:
    def __init__(self, world: World) -> None:
        self._world = world

    def world(self) -> World:
        return self._world


def test_p1_stream_facade_reuses_cached_graph_and_preserves_attrs() -> None:
    world = World()
    ctx = _FakeCtx(world)

    first = P1StreamFacade(ctx)
    second = P1StreamFacade(ctx)

    assert first._graph is second._graph
    assert first.mesh is first._graph.mesh
    assert first.point_vortex is first._graph.point_vortex
    assert first.dec is first._graph.dec
    assert first.p1_poisson is first._graph.p1_poisson
    assert first.p1_stream is first._graph.p1_stream
    assert first.mesh.prefix.startswith(f"{first._graph.prefix}.")
    assert first.p1_stream.prefix.startswith(f"{first._graph.prefix}.")


def test_p2_stream_facade_reuses_cached_graph_and_preserves_attrs() -> None:
    world = World()
    ctx = _FakeCtx(world)

    first = P2StreamFacade(ctx)
    second = P2StreamFacade(ctx)

    assert first._graph is second._graph
    assert first.mesh is first._graph.mesh
    assert first.point_vortex is first._graph.point_vortex
    assert first.p2_space is first._graph.p2_space
    assert first.p2_poisson is first._graph.p2_poisson
    assert first.p2_stream is first._graph.p2_stream
    assert first.p2_vel is first._graph.p2_vel
    assert first.mesh.prefix.startswith(f"{first._graph.prefix}.")
    assert first.p2_stream.prefix.startswith(f"{first._graph.prefix}.")


def test_p1_plane_facade_reuses_cached_graph_and_preserves_attrs() -> None:
    world = World()
    ctx = _FakeCtx(world)

    first = P1PlaneModule(ctx)
    second = P1PlaneModule(ctx)
    deps = world.module_dependencies()

    assert first._graph is second._graph
    assert first.mesh is first._graph.mesh
    assert first.point_vortex is first._graph.point_vortex
    assert first.dec is first._graph.dec
    assert first.p1_poisson is first._graph.p1_poisson
    assert first.p1_stream_func is first._graph.p1_stream_func
    assert first.p1_vel is first._graph.p1_vel
    assert first.rk4 is first._graph.rk4
    assert first._graph.p1_graph._module_key in deps[first._graph._module_key]
    assert first.p1_vel._module_key in deps[first._graph._module_key]
    assert first.rk4._module_key in deps[first._graph._module_key]


def test_p2_plane_facade_reuses_cached_graph_and_preserves_attrs() -> None:
    world = World()
    ctx = _FakeCtx(world)

    first = P2PlaneModule(ctx)
    second = P2PlaneModule(ctx)
    deps = world.module_dependencies()

    assert first._graph is second._graph
    assert first.mesh is first._graph.mesh
    assert first.point_vortex is first._graph.point_vortex
    assert first.p2_space is first._graph.p2_space
    assert first.p2_poisson is first._graph.p2_poisson
    assert first.p2_stream_func is first._graph.p2_stream_func
    assert first.p2_vel is first._graph.p2_vel
    assert first.rk4 is first._graph.rk4
    assert first._graph.p2_graph._module_key in deps[first._graph._module_key]
    assert first.rk4._module_key in deps[first._graph._module_key]


def test_poisson_test_facades_reuse_cached_graphs_and_preserve_attrs() -> None:
    world = World()
    ctx = _FakeCtx(world)

    p1_first = P1TestFacade(ctx)
    p1_second = P1TestFacade(ctx)
    p2_first = P2TestFacade(ctx)
    p2_second = P2TestFacade(ctx)

    assert p1_first._graph is p1_second._graph
    assert p1_first.mesh is p1_first._graph.mesh
    assert p1_first.dec is p1_first._graph.dec
    assert p1_first.p1_poisson is p1_first._graph.p1_poisson
    assert p2_first._graph is p2_second._graph
    assert p2_first.mesh is p2_first._graph.mesh
    assert p2_first.p2_space is p2_first._graph.p2_space
    assert p2_first.p2_poisson is p2_first._graph.p2_poisson
