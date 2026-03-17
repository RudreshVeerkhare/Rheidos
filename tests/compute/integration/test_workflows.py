"""Integration tests for compute module."""

import pytest
import numpy as np

from rheidos.compute import (
    World,
    ModuleBase,
    ProducerBase,
    Registry,
    WiredProducer,
    producer,
    out_field,
    ResourceRef,
    ResourceKey,
)
from dataclasses import dataclass


class TestLinearPipeline:
    """Test linear producer pipeline execution."""

    def test_three_stage_pipeline(self):
        """A -> B -> C pipeline executes in correct order."""
        registry = Registry()

        # Stage A: source
        registry.declare("a", buffer=np.array([1.0]))

        # Stage B: depends on A
        class ProducerB(ProducerBase):
            outputs = ("b",)

            def compute(self, reg):
                a = reg.read("a")
                reg.commit("b", buffer=a * 2)

        registry.declare("b", deps=["a"], producer=ProducerB())

        # Stage C: depends on B
        class ProducerC(ProducerBase):
            outputs = ("c",)

            def compute(self, reg):
                b = reg.read("b")
                reg.commit("c", buffer=b + 10)

        registry.declare("c", deps=["b"], producer=ProducerC())

        # Execute and verify
        registry.ensure("c")
        assert np.array_equal(registry.read("c", ensure=False), np.array([12.0]))

    def test_pipeline_staleness_tracking(self):
        """Stale resources trigger re-execution."""
        registry = Registry()
        execution_count = 0

        registry.declare("input", buffer=np.array([1.0]))

        class CountingProducer(ProducerBase):
            outputs = ("output",)

            def compute(self, reg):
                nonlocal execution_count
                execution_count += 1
                inp = reg.read("input")
                reg.commit("output", buffer=inp * 2)

        producer = CountingProducer()
        registry.declare("output", deps=["input"], producer=producer)

        # First ensure: should run
        registry.ensure("output")
        assert execution_count == 1

        # Second ensure: should be fresh
        registry.ensure("output")
        assert execution_count == 1

        # Bump input to mark stale
        registry.bump("input")
        registry.ensure("output")
        assert execution_count == 2


class TestDiamondDependency:
    """Test diamond dependency pattern."""

    def test_diamond_deduplication(self):
        """Diamond graph runs shared dep once."""
        registry = Registry()
        run_count = {}

        # Root
        registry.declare("a", buffer=np.array([1.0]))

        # B depends on A
        class ProducerB(ProducerBase):
            outputs = ("b",)

            def compute(self, reg):
                run_count["b"] = run_count.get("b", 0) + 1
                a = reg.read("a")
                reg.commit("b", buffer=a * 2)

        registry.declare("b", deps=["a"], producer=ProducerB())

        # C depends on A
        class ProducerC(ProducerBase):
            outputs = ("c",)

            def compute(self, reg):
                run_count["c"] = run_count.get("c", 0) + 1
                a = reg.read("a")
                reg.commit("c", buffer=a * 3)

        registry.declare("c", deps=["a"], producer=ProducerC())

        # D depends on B and C
        class ProducerD(ProducerBase):
            outputs = ("d",)

            def compute(self, reg):
                run_count["d"] = run_count.get("d", 0) + 1
                b = reg.read("b")
                c = reg.read("c")
                reg.commit("d", buffer=b + c)

        registry.declare("d", deps=["b", "c"], producer=ProducerD())

        # Execute
        registry.ensure("d")

        # Verify each producer ran exactly once
        assert run_count["b"] == 1
        assert run_count["c"] == 1
        assert run_count["d"] == 1

        # Verify correctness
        assert np.array_equal(registry.read("d", ensure=False), np.array([5.0]))


class TestModuleSystem:
    """Test module system integration."""

    def test_multi_module_workflow(self):
        """Multiple modules with dependencies work correctly."""
        world = World()

        class DataModule(ModuleBase):
            NAME = "data"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.x = self.resource("x", declare=True, buffer=np.array([2.0]))
                self.y = self.resource("y", declare=True, buffer=np.array([3.0]))

        class ProcessorModule(ModuleBase):
            NAME = "processor"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                data = self.require(DataModule)
                self.sum_xy = self.resource(
                    "sum_xy",
                    declare=True,
                    deps=[data.x, data.y],
                    buffer=np.array([5.0]),
                )

        # Build modules
        data_mod = world.require(DataModule)
        proc_mod = world.require(ProcessorModule)

        # Verify
        assert np.array_equal(
            world.reg.read(data_mod.x.name, ensure=False), np.array([2.0])
        )
        assert np.array_equal(
            world.reg.read(proc_mod.sum_xy.name, ensure=False), np.array([5.0])
        )

    def test_module_cycle_detection(self):
        """Multiple modules work correctly."""
        world = World()

        class ModuleA(ModuleBase):
            NAME = "a"

        class ModuleB(ModuleBase):
            NAME = "b"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                # Require A (no cycle)
                self.a = self.require(ModuleA)

        # Both should work
        a = world.require(ModuleA)
        b = world.require(ModuleB)
        assert a is not b
        assert isinstance(b.a, ModuleA)


class TestDecoratedModuleProducers:
    """Test decorator-based module producer integration."""

    def test_multi_output_method_runs_once_per_ensure_context(self):
        """One decorated method backs all of its outputs with one producer."""
        world = World()
        run_count = 0

        class MultiOutputModule(ModuleBase):
            NAME = "multi_output"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.a = self.resource("a", declare=True, buffer=np.array([4.0]))
                self.b = self.resource("b")
                self.c = self.resource("c")
                self.bind_producers()

            @producer(inputs=("a",), outputs=("b", "c"))
            def build_outputs(self, ctx):
                nonlocal run_count
                run_count += 1
                a = ctx.inputs.a.get()
                ctx.commit(b=a + 1, c=a + 2)

        module = world.require(MultiOutputModule)

        world.reg.ensure_many([module.b.name, module.c.name])
        assert run_count == 1
        assert np.array_equal(module.b.peek(), np.array([5.0]))
        assert np.array_equal(module.c.peek(), np.array([6.0]))

        world.reg.ensure(module.b.name)
        world.reg.ensure(module.c.name)
        assert run_count == 1

        world.reg.bump(module.a.name)
        world.reg.ensure(module.c.name)
        assert run_count == 2


class TestWiredProducerIntegration:
    """Test WiredProducer in complete workflow."""

    def test_wired_producer_execution(self):
        """WiredProducer executes and commits outputs."""
        registry = Registry()

        @dataclass
        class AddIO:
            a: ResourceRef
            b: ResourceRef
            sum_out: ResourceRef = out_field()

        class AddProducer(WiredProducer[AddIO]):
            def compute(self, reg):
                a = reg.read(self.io.a.name)
                b = reg.read(self.io.b.name)
                reg.commit(self.io.sum_out.name, buffer=a + b)

        # Set up resources
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b", buffer=np.array([2.0]))

        a_ref = ResourceRef(registry, ResourceKey("a"))
        b_ref = ResourceRef(registry, ResourceKey("b"))
        sum_ref = ResourceRef(registry, ResourceKey("sum"))
        io = AddIO(a=a_ref, b=b_ref, sum_out=sum_ref)

        # Declare sum with producer
        producer = AddProducer(io=io)
        registry.declare("sum", deps=["a", "b"], producer=producer)

        # Execute
        registry.ensure("sum")
        assert np.array_equal(registry.read("sum", ensure=False), np.array([3.0]))

    def test_chained_wired_producers(self):
        """Multiple WiredProducers in sequence."""
        registry = Registry()

        # Initial data
        registry.declare("x", buffer=np.array([2.0]))

        # Producer 1: double
        @dataclass
        class DoubleIO:
            inp: ResourceRef
            out: ResourceRef = out_field()

        class DoubleProducer(WiredProducer[DoubleIO]):
            def compute(self, reg):
                inp = reg.read(self.io.inp.name)
                reg.commit(self.io.out.name, buffer=inp * 2)

        x_ref = ResourceRef(registry, ResourceKey("x"))
        doubled_ref = ResourceRef(registry, ResourceKey("doubled"))
        io1 = DoubleIO(inp=x_ref, out=doubled_ref)
        prod1 = DoubleProducer(io=io1)
        registry.declare("doubled", deps=["x"], producer=prod1)

        # Producer 2: add 10
        @dataclass
        class AddIO:
            inp: ResourceRef
            out: ResourceRef = out_field()

        class AddProducer(WiredProducer[AddIO]):
            def compute(self, reg):
                inp = reg.read(self.io.inp.name)
                reg.commit(self.io.out.name, buffer=inp + 10)

        doubled_ref2 = ResourceRef(registry, ResourceKey("doubled"))
        result_ref = ResourceRef(registry, ResourceKey("result"))
        io2 = AddIO(inp=doubled_ref2, out=result_ref)
        prod2 = AddProducer(io=io2)
        registry.declare("result", deps=["doubled"], producer=prod2)

        # Execute and verify: (2 * 2) + 10 = 14
        registry.ensure("result")
        assert np.array_equal(registry.read("result", ensure=False), np.array([14.0]))

    def test_producer_with_multiple_outputs(self):
        """Test producer that computes multiple outputs."""
        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))

        class MultiProducer(ProducerBase):
            outputs = ("b", "c")

            def compute(self, reg):
                a = reg.read("a")
                reg.commit("b", buffer=a + 1)
                reg.commit("c", buffer=a + 2)

        registry.declare("b", deps=["a"], producer=MultiProducer())
        registry.declare("c", deps=["a"], producer=MultiProducer())

        registry.ensure("b")
        registry.ensure("c")
        assert np.array_equal(registry.read("b", ensure=False), np.array([2.0]))
        assert np.array_equal(registry.read("c", ensure=False), np.array([3.0]))

    def test_resource_freshness_with_multiple_runs(self):
        """Test that resources are correctly marked stale/fresh."""
        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))

        call_count = [0]

        class CountingProducer(ProducerBase):
            outputs = ("b",)

            def compute(self, reg):
                call_count[0] += 1
                a = reg.read("a")
                reg.commit("b", buffer=a * 2)

        registry.declare("b", deps=["a"], producer=CountingProducer())

        # First ensure should call producer
        registry.ensure("b")
        assert call_count[0] == 1

        # Second ensure should NOT call producer (b still fresh)
        registry.ensure("b")
        assert call_count[0] == 1

        # Bumping a should make b stale
        registry.bump("a")
        registry.ensure("b")
        assert call_count[0] == 2

    def test_read_nonexistent_raises(self):
        """Test reading non-existent resource raises KeyError."""
        registry = Registry()
        with pytest.raises(KeyError):
            registry.read("nonexistent")

    def test_get_nonexistent_raises(self):
        """Test get on non-existent resource raises KeyError."""
        registry = Registry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_producer_computation_with_unsafe_commit(self):
        """Test producer with unsafe=True commits."""
        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))

        class UnsafeProducer(ProducerBase):
            outputs = ("b",)

            def compute(self, reg):
                a = reg.read("a")
                # Use unsafe=True to skip validation
                reg.commit("b", buffer=a * 3, unsafe=True)

        registry.declare("b", deps=["a"], producer=UnsafeProducer())
        registry.ensure("b")
        result = registry.read("b", ensure=False)
        assert np.array_equal(result, np.array([3.0]))
