"""Decorator producer runtime coverage."""

from __future__ import annotations

import numpy as np
import pytest

from rheidos.compute import (
    ModuleBase,
    ProducerContext,
    ResourceSpec,
    World,
    producer,
    producer_output,
)


class TestProducerDecorator:
    def test_producer_output_records_alloc(self) -> None:
        def alloc_fn(_reg, _ctx):
            return np.zeros((2,), dtype=np.float32)

        output = producer_output("result", alloc=alloc_fn)
        assert output.name == "result"
        assert output.alloc is alloc_fn

    def test_decorated_producer_executes_and_commits(self) -> None:
        world = World()

        class DemoModule(ModuleBase):
            NAME = "demo"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.value = self.resource(
                    "value",
                    declare=True,
                    buffer=np.array([4.0], dtype=np.float32),
                )
                self.result = self.resource(
                    "result",
                    spec=ResourceSpec(kind="numpy", dtype=np.float32, shape=(1,)),
                )
                self.bind_producers()

            @producer(inputs=("value",), outputs=("result",))
            def square(self, ctx: ProducerContext) -> None:
                value = ctx.inputs.value.get()
                ctx.commit(result=value * value)

        module = world.require(DemoModule)
        world.reg.ensure(module.result.name)
        assert np.array_equal(module.result.peek(), np.array([16.0], dtype=np.float32))

    def test_context_ensure_outputs_uses_custom_alloc(self) -> None:
        world = World()

        def alloc_out(_reg, ctx: ProducerContext):
            return np.zeros_like(ctx.inputs.a.peek())

        class AllocModule(ModuleBase):
            NAME = "alloc"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.a = self.resource(
                    "a",
                    declare=True,
                    buffer=np.array([2.0], dtype=np.float32),
                )
                self.out = self.resource("out")
                self.bind_producers()

            @producer(
                inputs=("a",),
                outputs=(producer_output("out", alloc=alloc_out),),
            )
            def fill(self, ctx: ProducerContext) -> None:
                out = ctx.ensure_outputs(require_shape=False)["out"].peek()
                out[:] = ctx.inputs.a.get() + 3.0
                ctx.outputs.out.commit()

        module = world.require(AllocModule)
        world.reg.ensure(module.out.name)
        assert np.array_equal(module.out.peek(), np.array([5.0], dtype=np.float32))

    def test_context_commit_rejects_unknown_output(self) -> None:
        world = World()

        class BadCommitModule(ModuleBase):
            NAME = "bad_commit"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.a = self.resource("a", declare=True, buffer=np.array([1.0]))
                self.out = self.resource(
                    "out",
                    spec=ResourceSpec(kind="numpy", dtype=np.float64, shape=(1,)),
                )
                self.bind_producers()

            @producer(inputs=("a",), outputs=("out",))
            def run(self, ctx: ProducerContext) -> None:
                ctx.commit(missing=np.array([1.0]))

        module = world.require(BadCommitModule)
        with pytest.raises(KeyError, match="Did you mean 'out'\\?"):
            world.reg.ensure(module.out.name)

    def test_context_inputs_attribute_typo_is_informative(self) -> None:
        world = World()

        class TypoInputModule(ModuleBase):
            NAME = "typo_input"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.mesh = self.resource(
                    "mesh",
                    declare=True,
                    buffer=np.array([1.0]),
                )
                self.out = self.resource(
                    "out",
                    spec=ResourceSpec(kind="numpy", dtype=np.float64, shape=(1,)),
                )
                self.bind_producers()

            @producer(inputs=("mesh",), outputs=("out",))
            def run(self, ctx: ProducerContext) -> None:
                ctx.inputs.mseh.get()

        module = world.require(TypoInputModule)
        with pytest.raises(AttributeError, match="Did you mean 'mesh'\\?"):
            world.reg.ensure(module.out.name)
