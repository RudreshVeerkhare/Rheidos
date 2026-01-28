"""Shared fixtures for compute module tests."""

import pytest
import numpy as np
from dataclasses import dataclass

from rheidos.compute import (
    Registry,
    ProducerBase,
    ResourceSpec,
    ResourceRef,
    ResourceKey,
    World,
    ModuleBase,
    WiredProducer,
    out_field,
)


@pytest.fixture
def registry():
    """Fresh Registry for each test."""
    return Registry()


@pytest.fixture
def world():
    """Fresh World with embedded Registry."""
    return World()


@pytest.fixture
def numpy_spec():
    """Standard numpy ResourceSpec."""
    return ResourceSpec(kind="numpy", dtype=np.float32, shape=(10,))


@pytest.fixture
def numpy_spec_no_shape():
    """Numpy spec without shape constraint."""
    return ResourceSpec(kind="numpy", dtype=np.float32)


class SimpleProducer(ProducerBase):
    """Minimal producer that outputs a constant array."""

    outputs = ("result",)

    def __init__(self, value=42.0):
        self.value = value

    def compute(self, reg):
        reg.commit("result", buffer=np.array([self.value]))


@pytest.fixture
def simple_producer():
    """Minimal producer for basic tests."""
    return SimpleProducer(value=10.0)


class AddProducer(ProducerBase):
    """Producer that adds two inputs."""

    outputs = ("sum",)

    def __init__(self, a_name="a", b_name="b"):
        self.a_name = a_name
        self.b_name = b_name

    def compute(self, reg):
        a = reg.read(self.a_name)
        b = reg.read(self.b_name)
        reg.commit("sum", buffer=a + b)


@pytest.fixture
def linear_chain(registry):
    """Registry with linear chain: A -> B -> C."""
    # A (producer)
    reg = registry
    reg.declare("a", buffer=np.array([1.0]))

    # B depends on A
    class ProducerB(ProducerBase):
        outputs = ("b",)

        def compute(self, reg):
            a = reg.read("a")
            reg.commit("b", buffer=a * 2)

    reg.declare("b", deps=["a"], producer=ProducerB())

    # C depends on B
    class ProducerC(ProducerBase):
        outputs = ("c",)

        def compute(self, reg):
            b = reg.read("b")
            reg.commit("c", buffer=b + 10)

    reg.declare("c", deps=["b"], producer=ProducerC())

    return reg


@pytest.fixture
def diamond_chain(registry):
    """Registry with diamond: A -> B, A -> C -> D."""
    reg = registry

    # Root: A
    reg.declare("a", buffer=np.array([1.0]))

    # B depends on A
    class ProducerB(ProducerBase):
        outputs = ("b",)

        def compute(self, reg):
            a = reg.read("a")
            reg.commit("b", buffer=a * 2)

    reg.declare("b", deps=["a"], producer=ProducerB())

    # C depends on A (diamond point)
    class ProducerC(ProducerBase):
        outputs = ("c",)

        def compute(self, reg):
            a = reg.read("a")
            reg.commit("c", buffer=a * 3)

    reg.declare("c", deps=["a"], producer=ProducerC())

    # D depends on B and C
    class ProducerD(ProducerBase):
        outputs = ("d",)

        def compute(self, reg):
            b = reg.read("b")
            c = reg.read("c")
            reg.commit("d", buffer=b + c)

    reg.declare("d", deps=["b", "c"], producer=ProducerD())

    return reg


# ============================================================================
# WiredProducer fixtures
# ============================================================================


@dataclass
class SimpleIO:
    """Simple IO for WiredProducer tests."""

    out_val: ResourceRef = out_field()


class SimpleWiredProducer(WiredProducer[SimpleIO]):
    """Minimal WiredProducer."""

    def compute(self, reg):
        reg.commit(self.io.out_val.name, buffer=np.array([42.0]))


@pytest.fixture
def simple_wired_producer(registry):
    """WiredProducer with simple output."""
    out_ref = ResourceRef(registry, ResourceKey("test_out"))
    io = SimpleIO(out_val=out_ref)
    return SimpleWiredProducer(io=io)


@dataclass
class AddIO:
    """IO for addition WiredProducer."""

    a: ResourceRef  # input
    b: ResourceRef  # input
    sum_val: ResourceRef = out_field()  # output


class AddWiredProducer(WiredProducer[AddIO]):
    """WiredProducer that adds two inputs."""

    def compute(self, reg):
        a = reg.read(self.io.a.name)
        b = reg.read(self.io.b.name)
        reg.commit(self.io.sum_val.name, buffer=a + b)


@pytest.fixture
def add_wired_producer(registry):
    """WiredProducer with inputs and output."""
    a_ref = ResourceRef(registry, ResourceKey("a"))
    b_ref = ResourceRef(registry, ResourceKey("b"))
    out_ref = ResourceRef(registry, ResourceKey("sum_val"))
    io = AddIO(a=a_ref, b=b_ref, sum_val=out_ref)
    return AddWiredProducer(io=io)


# ============================================================================
# Module fixtures
# ============================================================================


class DataModule(ModuleBase):
    """Simple data module."""

    NAME = "data"

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)
        self.x = self.resource("x", declare=True, buffer=np.array([1.0]))
        self.y = self.resource("y", declare=True, buffer=np.array([2.0]))


class ProcessorModule(ModuleBase):
    """Processor module that depends on DataModule."""

    NAME = "processor"

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)
        data = self.require(DataModule)
        self.z = self.resource(
            "z",
            declare=True,
            deps=[data.x, data.y],
            buffer=np.array([3.0]),
        )


@pytest.fixture
def simple_world():
    """World with pre-configured modules."""
    w = World()
    w.require(DataModule)
    return w
