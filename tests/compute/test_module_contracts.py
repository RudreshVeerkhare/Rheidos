from __future__ import annotations

import numpy as np
import pytest

from rheidos.compute import (
    ModuleBase,
    ModuleInputContractError,
    ResourceSpec,
    World,
    producer,
    resource_view,
)


class AltMeshModule(ModuleBase):
    NAME = "alt_mesh"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.vertices = self.resource(
            "vertices",
            declare=True,
            buffer=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        )


class PositionCountModule(ModuleBase):
    NAME = "position_count"

    def __init__(self, world: World, *, mesh, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.mesh = mesh
        self.count = self.resource(
            "count",
            spec=ResourceSpec(kind="numpy", dtype=np.int64, shape=(1,)),
        )
        self.bind_producers()

    @producer(inputs=("mesh.V_pos",), outputs=("count",))
    def build_count(self, ctx) -> None:
        verts = ctx.inputs.mesh.V_pos.get()
        ctx.commit(count=np.array([verts.shape[0]], dtype=np.int64))


class BadMeshProvider:
    def __init__(self) -> None:
        self.V_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)


def test_bind_producers_reports_missing_injected_provider_resources() -> None:
    world = World()
    mesh = world.require(AltMeshModule)

    with pytest.raises(ModuleInputContractError) as excinfo:
        world.require(PositionCountModule, mesh=mesh)

    message = str(excinfo.value)
    assert "PositionCountModule input contract validation failed" in message
    assert "provider 'mesh' (AltMeshModule)" in message
    assert "build_count -> mesh.V_pos" in message
    assert "Did you mean 'mesh.vertices'?" in message
    assert "Available names: mesh.vertices." in message


def test_bind_producers_rejects_injected_non_resource_refs() -> None:
    world = World()
    provider = BadMeshProvider()

    with pytest.raises(ModuleInputContractError) as excinfo:
        world.require(PositionCountModule, mesh=provider)

    message = str(excinfo.value)
    assert "provider 'mesh' (BadMeshProvider)" in message
    assert "expected ResourceRef" in message


def test_resource_view_adapts_structurally_compatible_provider() -> None:
    world = World()
    mesh = world.require(AltMeshModule)
    adapted_mesh = resource_view(mesh, V_pos="vertices")

    module = world.require(PositionCountModule, mesh=adapted_mesh)

    world.reg.ensure(module.count.name)
    np.testing.assert_array_equal(module.count.get(), np.array([1], dtype=np.int64))


def test_resource_view_aliases_are_hash_stable_for_world_cache_keys() -> None:
    world = World()
    mesh = world.require(AltMeshModule)

    first = world.require(PositionCountModule, mesh=resource_view(mesh, V_pos="vertices"))
    second = world.require(
        PositionCountModule,
        mesh=resource_view(mesh, V_pos="vertices"),
    )

    assert first is second


def test_resource_view_rejects_unknown_alias_targets_with_help() -> None:
    world = World()
    mesh = world.require(AltMeshModule)

    with pytest.raises(AttributeError, match="Did you mean 'vertices'\\?"):
        resource_view(mesh, V_pos="verticse")


def test_world_records_explicit_module_dependencies_from_resource_views() -> None:
    world = World()
    mesh = world.require(AltMeshModule)
    adapted_mesh = resource_view(mesh, V_pos="vertices")

    module = world.require(PositionCountModule, mesh=adapted_mesh)
    deps = world.module_dependencies()

    assert mesh._module_key in deps[module._module_key]
