"""Tests for rheidos.compute.resource module."""

import pytest
import numpy as np
from rheidos.compute import (
    ResourceSpec,
    Resource,
    ResourceKey,
    ResourceRef,
    Registry,
)


class TestResourceSpec:
    """Test ResourceSpec dataclass."""

    def test_spec_defaults(self):
        """ResourceSpec with defaults."""
        spec = ResourceSpec(kind="numpy")
        assert spec.kind == "numpy"
        assert spec.dtype is None
        assert spec.lanes is None
        assert spec.shape is None
        assert spec.shape_fn is None
        assert spec.allow_none is True

    def test_spec_with_all_fields(self):
        """ResourceSpec with all fields."""
        shape_fn = lambda reg: (10,)
        spec = ResourceSpec(
            kind="taichi",
            dtype=np.float32,
            lanes=3,
            shape=(10,),
            shape_fn=shape_fn,
            allow_none=False,
        )
        assert spec.kind == "taichi"
        assert spec.dtype == np.float32
        assert spec.lanes == 3
        assert spec.shape == (10,)
        assert spec.shape_fn is shape_fn
        assert spec.allow_none is False

    def test_spec_frozen_immutable(self):
        """ResourceSpec is frozen (immutable)."""
        spec = ResourceSpec(kind="numpy")
        with pytest.raises(AttributeError):
            spec.kind = "taichi"


class TestResource:
    """Test Resource dataclass."""

    def test_resource_minimal(self):
        """Resource with minimal fields."""
        r = Resource(name="test")
        assert r.name == "test"
        assert r.buffer is None
        assert r.deps == ()
        assert r.producer is None
        assert r.version == 0
        assert r.dep_sig == ()

    def test_resource_with_buffer(self):
        """Resource with buffer."""
        buf = np.array([1, 2, 3])
        r = Resource(name="data", buffer=buf)
        assert np.array_equal(r.buffer, buf)

    def test_resource_with_deps(self):
        """Resource with dependencies."""
        r = Resource(name="result", deps=("a", "b"))
        assert r.deps == ("a", "b")

    def test_resource_version_tracking(self):
        """Resource version changes."""
        r = Resource(name="data", buffer=np.array([1.0]))
        initial_version = r.version
        r.version += 1
        assert r.version == initial_version + 1

    def test_resource_dep_sig_tracking(self):
        """Resource dep_sig tracking."""
        r = Resource(name="data", deps=("a", "b"))
        r.dep_sig = (("a", 1), ("b", 2))
        assert r.dep_sig == (("a", 1), ("b", 2))


class TestResourceKey:
    """Test ResourceKey generic frozen dataclass."""

    def test_resource_key_creation(self):
        """Create ResourceKey."""
        key = ResourceKey("full_name")
        assert key.full_name == "full_name"
        assert key.spec is None

    def test_resource_key_with_spec(self):
        """ResourceKey with spec."""
        spec = ResourceSpec(kind="numpy")
        key = ResourceKey("full_name", spec=spec)
        assert key.spec is spec

    def test_resource_key_frozen(self):
        """ResourceKey is frozen (immutable)."""
        key = ResourceKey("name")
        with pytest.raises(AttributeError):
            key.full_name = "new_name"


class TestResourceRef:
    """Test ResourceRef class."""

    def test_resource_ref_creation(self):
        """Create ResourceRef."""
        registry = Registry()
        key = ResourceKey("test_name")
        ref = ResourceRef(registry, key)
        assert ref.name == "test_name"
        assert ref._reg is registry
        assert ref._key is key

    def test_resource_ref_with_spec(self):
        """ResourceRef with spec."""
        registry = Registry()
        spec = ResourceSpec(kind="numpy")
        key = ResourceKey("test_name", spec=spec)
        ref = ResourceRef(registry, key)
        assert ref.name == "test_name"

    def test_resource_ref_with_doc(self):
        """ResourceRef with documentation."""
        registry = Registry()
        key = ResourceKey("test_name")
        doc = "Test documentation"
        ref = ResourceRef(registry, key, doc=doc)
        assert ref.doc == doc
        assert ref.__doc__ == doc

    def test_resource_ref_ensure(self):
        """ResourceRef.ensure() calls registry.ensure()."""
        from rheidos.compute.registry import ProducerBase

        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))

        class Producer(ProducerBase):
            outputs = ("b",)

            def compute(self, reg):
                reg.commit("b", buffer=np.array([2.0]))

        registry.declare("b", deps=["a"], producer=Producer())

        key = ResourceKey("b")
        ref = ResourceRef(registry, key)
        ref.ensure()
        # After ensure, b should be computed
        assert ref.peek() is not None

    def test_resource_ref_get(self):
        """ResourceRef.get() reads with ensure=True."""
        registry = Registry()
        registry.declare("data", buffer=np.array([1.0]))
        key = ResourceKey("data")
        ref = ResourceRef(registry, key)
        value = ref.get()
        assert np.array_equal(value, np.array([1.0]))

    def test_resource_ref_peek(self):
        """ResourceRef.peek() reads without ensure."""
        registry = Registry()
        registry.declare("data", buffer=np.array([1.0]))
        key = ResourceKey("data")
        ref = ResourceRef(registry, key)
        value = ref.peek()
        assert np.array_equal(value, np.array([1.0]))

    def test_resource_ref_set(self):
        """ResourceRef.set() commits with new buffer."""
        registry = Registry()
        registry.declare("data", buffer=np.array([1.0]))
        key = ResourceKey("data")
        ref = ResourceRef(registry, key)
        new_val = np.array([5.0])
        ref.set(new_val)
        assert np.array_equal(ref.peek(), np.array([5.0]))

    def test_resource_ref_set_buffer(self):
        """ResourceRef.set_buffer() sets without bumping."""
        registry = Registry()
        registry.declare("data", buffer=np.array([1.0]))
        key = ResourceKey("data")
        ref = ResourceRef(registry, key)
        new_val = np.array([5.0])
        ref.set_buffer(new_val, bump=False)
        assert np.array_equal(ref.peek(), np.array([5.0]))

    def test_resource_ref_commit(self):
        """ResourceRef.commit() marks resource fresh."""
        registry = Registry()
        registry.declare("data", buffer=np.array([1.0]))
        key = ResourceKey("data")
        ref = ResourceRef(registry, key)
        initial_version = registry.get("data").version
        ref.commit()
        # Version should increase
        assert registry.get("data").version > initial_version

    def test_resource_ref_bump(self):
        """ResourceRef.bump() increments version."""
        registry = Registry()
        registry.declare("data", buffer=np.array([1.0]))
        key = ResourceKey("data")
        ref = ResourceRef(registry, key)
        initial_version = registry.get("data").version
        ref.bump()
        assert registry.get("data").version > initial_version

    def test_resource_ref_mark_fresh(self):
        """ResourceRef.mark_fresh() is alias for commit()."""
        registry = Registry()
        registry.declare("data", buffer=np.array([1.0]))
        key = ResourceKey("data")
        ref = ResourceRef(registry, key)
        initial_version = registry.get("data").version
        ref.mark_fresh()
        assert registry.get("data").version > initial_version

    def test_resource_ref_touch(self):
        """ResourceRef.touch() is alias for commit()."""
        registry = Registry()
        registry.declare("data", buffer=np.array([1.0]))
        key = ResourceKey("data")
        ref = ResourceRef(registry, key)
        initial_version = registry.get("data").version
        ref.touch()
        assert registry.get("data").version > initial_version

    def test_resource_ref_spec_property(self):
        """ResourceRef.spec property returns spec from key."""
        registry = Registry()
        spec = ResourceSpec(kind="numpy", dtype=np.float32)
        key = ResourceKey("data", spec=spec)
        ref = ResourceRef(registry, key)
        assert ref.spec == spec
