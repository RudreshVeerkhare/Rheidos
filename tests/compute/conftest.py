"""Shared fixtures for compute module tests."""

import pytest
import numpy as np

from rheidos.compute import (
    Registry,
    ResourceSpec,
    World,
)
from rheidos.compute.registry import ProducerBase


@pytest.fixture
def registry():
    """Fresh Registry for each test."""
    return Registry()


@pytest.fixture
def numpy_spec():
    """Standard numpy ResourceSpec."""
    return ResourceSpec(kind="numpy", dtype=np.float32, shape=(10,))


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
