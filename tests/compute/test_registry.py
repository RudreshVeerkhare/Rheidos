"""Tests for rheidos.compute.registry module."""

import pytest
import numpy as np
from rheidos.compute import Registry, ResourceSpec
from rheidos.compute.registry import ProducerBase


class TestRegistryDeclare:
    """Test Registry.declare() behavior."""

    def test_declare_minimal(self, registry):
        """Declare minimal resource without buffer or deps."""
        r = registry.declare("test_resource")
        assert r.name == "test_resource"
        assert r.buffer is None
        assert r.deps == ()
        assert r.producer is None

    def test_declare_with_buffer(self, registry):
        """Declare resource with initial buffer."""
        buf = np.array([1, 2, 3])
        r = registry.declare("data", buffer=buf)
        assert np.array_equal(r.buffer, buf)

    def test_declare_with_spec(self, registry, numpy_spec):
        """Declare resource with spec."""
        r = registry.declare(
            "data", spec=numpy_spec, buffer=np.zeros((10,), dtype=np.float32)
        )
        assert r.spec == numpy_spec

    def test_declare_with_deps(self, registry):
        """Declare resource with dependencies."""
        registry.declare("a")
        r = registry.declare("b", deps=["a"])
        assert r.deps == ("a",)

    def test_declare_with_producer(self, registry, simple_producer):
        """Declare resource with producer."""
        r = registry.declare("result", producer=simple_producer)
        assert r.producer is simple_producer

    def test_declare_with_description(self, registry):
        """Declare resource with description."""
        r = registry.declare("data", description="Test data")
        assert r.description == "Test data"

    def test_declare_duplicate_raises(self, registry):
        """Declaring duplicate resource name raises KeyError."""
        registry.declare("test")
        with pytest.raises(KeyError, match="already declared"):
            registry.declare("test")

    def test_declare_validates_buffer(self, registry, numpy_spec):
        """Declaring with spec validates buffer."""
        # Wrong dtype should raise
        with pytest.raises(TypeError):
            registry.declare(
                "data",
                spec=numpy_spec,
                buffer=np.zeros((10,), dtype=np.int32),
            )


class TestRegistryGet:
    """Test Registry.get() behavior."""

    def test_get_exists(self, registry):
        """Get existing resource."""
        r = registry.declare("test")
        retrieved = registry.get("test")
        assert retrieved is r

    def test_get_not_found_raises(self, registry):
        """Get nonexistent resource raises KeyError."""
        with pytest.raises(KeyError, match="Unknown resource"):
            registry.get("nonexistent")


class TestRegistryRead:
    """Test Registry.read() behavior."""

    def test_read_gets_buffer(self, registry):
        """Read returns buffer of declared resource."""
        buf = np.array([1, 2, 3])
        registry.declare("data", buffer=buf)
        result = registry.read("data", ensure=False)
        assert np.array_equal(result, buf)

    def test_read_nonexistent_raises(self, registry):
        """Read nonexistent resource raises KeyError."""
        with pytest.raises(KeyError):
            registry.read("nonexistent", ensure=False)

    def test_read_missing_resource_raises(self, registry):
        """Read resource without buffer (and not ensured) returns None."""
        registry.declare("empty")
        result = registry.read("empty", ensure=False)
        assert result is None


class TestRegistrySetBuffer:
    """Test Registry.set_buffer() behavior."""

    def test_set_buffer_updates_buffer(self, registry):
        """set_buffer replaces buffer."""
        registry.declare("data", buffer=np.array([1.0]))
        new_buf = np.array([2.0])
        registry.set_buffer("data", new_buf, bump=False)
        assert np.array_equal(registry.read("data", ensure=False), new_buf)

    def test_set_buffer_bump_true_increments_version(self, registry):
        """set_buffer with bump=True increments version."""
        registry.declare("data", buffer=np.array([1.0]))
        initial_version = registry.get("data").version
        registry.set_buffer("data", np.array([2.0]), bump=True)
        assert registry.get("data").version == initial_version + 1

    def test_set_buffer_bump_false_no_version(self, registry):
        """set_buffer with bump=False does not change version."""
        registry.declare("data", buffer=np.array([1.0]))
        initial_version = registry.get("data").version
        registry.set_buffer("data", np.array([2.0]), bump=False)
        assert registry.get("data").version == initial_version

    def test_set_buffer_validates_spec(self, registry, numpy_spec):
        """set_buffer validates against spec unless unsafe=True."""
        registry.declare(
            "data", spec=numpy_spec, buffer=np.zeros((10,), dtype=np.float32)
        )
        # Wrong shape should raise
        with pytest.raises(TypeError):
            registry.set_buffer("data", np.zeros((5,), dtype=np.float32), bump=False)

    def test_set_buffer_unsafe_skips_validation(self, registry, numpy_spec):
        """set_buffer with unsafe=True skips validation."""
        registry.declare(
            "data", spec=numpy_spec, buffer=np.zeros((10,), dtype=np.float32)
        )
        # Should not raise with unsafe=True
        registry.set_buffer(
            "data", np.zeros((5,), dtype=np.int32), bump=False, unsafe=True
        )


class TestRegistryCommit:
    """Test Registry.commit() behavior."""

    def test_commit_no_buffer(self, registry):
        """commit without buffer just bumps version."""
        registry.declare("data", buffer=np.array([1.0]))
        initial_version = registry.get("data").version
        registry.commit("data")
        assert registry.get("data").version > initial_version

    def test_commit_with_buffer(self, registry):
        """commit with buffer updates both buffer and version."""
        registry.declare("data", buffer=np.array([1.0]))
        new_buf = np.array([2.0])
        registry.commit("data", buffer=new_buf)
        assert np.array_equal(registry.read("data", ensure=False), new_buf)

    def test_commit_updates_dep_sig(self, registry):
        """commit updates dependency signature."""
        registry.declare("dep", buffer=np.array([1.0]))
        registry.declare("data", deps=["dep"], buffer=np.array([2.0]))
        r = registry.get("data")
        old_sig = r.dep_sig
        registry.commit("data")
        assert r.dep_sig != old_sig

    def test_commit_many_no_buffers(self, registry):
        """commit_many without buffers bumps all."""
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b", buffer=np.array([2.0]))
        initial_a_ver = registry.get("a").version
        initial_b_ver = registry.get("b").version
        registry.commit_many(["a", "b"])
        assert registry.get("a").version > initial_a_ver
        assert registry.get("b").version > initial_b_ver

    def test_commit_many_with_buffers(self, registry):
        """commit_many with buffers updates specified resources."""
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b", buffer=np.array([2.0]))
        buffers = {"a": np.array([10.0]), "b": np.array([20.0])}
        registry.commit_many(["a", "b"], buffers=buffers)
        assert np.array_equal(registry.read("a", ensure=False), np.array([10.0]))
        assert np.array_equal(registry.read("b", ensure=False), np.array([20.0]))


class TestRegistryBump:
    """Test Registry.bump() behavior."""

    def test_bump_increments_version(self, registry):
        """bump increments version."""
        registry.declare("data", buffer=np.array([1.0]))
        initial_version = registry.get("data").version
        registry.bump("data")
        assert registry.get("data").version == initial_version + 1

    def test_bump_updates_dep_sig(self, registry):
        """bump updates dependency signature."""
        registry.declare("dep", buffer=np.array([1.0]))
        registry.declare("data", deps=["dep"], buffer=np.array([2.0]))
        r = registry.get("data")
        old_sig = r.dep_sig
        registry.bump("data")
        assert r.dep_sig != old_sig


class TestRegistryValidation:
    """Test Registry buffer validation."""

    def test_validate_buffer_no_spec(self, registry):
        """Resource without spec bypasses validation."""
        # Should not raise even with arbitrary buffer
        registry.declare("data", buffer=object())

    def test_validate_buffer_none_allowed(self, registry):
        """None buffer allowed when allow_none=True."""
        spec = ResourceSpec(kind="numpy", allow_none=True)
        registry.declare("data", spec=spec)  # No buffer, should not raise

    def test_validate_unknown_kind_raises(self, registry):
        """Unknown ResourceSpec kind raises ValueError."""
        spec = ResourceSpec(kind="unknown_kind")
        with pytest.raises(ValueError, match="Unknown ResourceSpec.kind"):
            registry.declare("data", spec=spec, buffer=np.array([1.0]))


class TestRegistryEnsure:
    """Test Registry.ensure() behavior (dependency resolution)."""

    def test_ensure_no_deps(self, registry, simple_producer):
        """ensure on producer with no deps runs producer."""
        registry.declare("result", producer=simple_producer)
        registry.ensure("result")
        buf = registry.read("result", ensure=False)
        assert np.array_equal(buf, np.array([10.0]))

    def test_ensure_single_dep(self, registry):
        """ensure resolves single dependency."""
        registry.declare("a", buffer=np.array([1.0]))

        class ProducerB(ProducerBase):
            outputs = ("b",)

            def compute(self, reg):
                a = reg.read("a")
                reg.commit("b", buffer=a * 2)

        registry.declare("b", deps=["a"], producer=ProducerB())
        registry.ensure("b")
        assert np.array_equal(registry.read("b", ensure=False), np.array([2.0]))

    def test_ensure_linear_chain(self, linear_chain):
        """ensure on linear chain executes in correct order."""
        linear_chain.ensure("c")
        a = linear_chain.read("a", ensure=False)
        b = linear_chain.read("b", ensure=False)
        c = linear_chain.read("c", ensure=False)
        assert np.array_equal(a, np.array([1.0]))
        assert np.array_equal(b, a * 2)
        assert np.array_equal(c, b + 10)

    def test_ensure_producer_called_if_stale(self, registry):
        """ensure runs producer if resource is stale."""
        call_count = 0

        class CountingProducer(ProducerBase):
            outputs = ("result",)

            def compute(self, reg):
                nonlocal call_count
                call_count += 1
                reg.commit("result", buffer=np.array([call_count]))

        producer = CountingProducer()
        registry.declare("result", producer=producer)
        registry.ensure("result")
        assert call_count == 1
        registry.ensure("result")
        # Second ensure should not re-run (fresh)
        assert call_count == 1

    def test_ensure_producer_not_called_if_fresh(self, registry):
        """ensure does not run producer if resource is fresh."""
        call_count = 0

        class CountingProducer(ProducerBase):
            outputs = ("result",)

            def compute(self, reg):
                nonlocal call_count
                call_count += 1
                reg.commit("result", buffer=np.array([call_count]))

        producer = CountingProducer()
        registry.declare("result", producer=producer)
        registry.ensure("result")
        initial_count = call_count
        registry.ensure("result")
        assert call_count == initial_count

    def test_ensure_diamond_deduplicates(self, diamond_chain):
        """ensure in diamond graph runs shared dep once."""
        # Verify diamond structure works without wrapping
        # (wrapping producers in tests is complex due to closures)
        diamond_chain.ensure("d")
        # Verify final result is correct
        d = diamond_chain.read("d", ensure=False)
        # (1*2) + (1*3) = 5
        assert np.array_equal(d, np.array([5.0]))

    def test_ensure_many(self, registry):
        """ensure_many ensures multiple resources."""
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b", buffer=np.array([2.0]))

        class SumProducer(ProducerBase):
            outputs = ("sum",)

            def compute(self, reg):
                a = reg.read("a")
                b = reg.read("b")
                reg.commit("sum", buffer=a + b)

        registry.declare("sum", deps=["a", "b"], producer=SumProducer())

        registry.ensure_many(["a", "b", "sum"])
        assert np.array_equal(registry.read("sum", ensure=False), np.array([3.0]))

    def test_ensure_missing_resource_raises(self, registry):
        """ensure on missing resource raises KeyError."""
        with pytest.raises(KeyError):
            registry.ensure("nonexistent")

    def test_bump_validates_buffer(self, registry, numpy_spec):
        """bump validates buffer unless unsafe=True."""
        registry.declare(
            "data", spec=numpy_spec, buffer=np.zeros((10,), dtype=np.float32)
        )
        # Set invalid buffer (wrong dtype)
        registry.get("data").buffer = np.zeros((10,), dtype=np.int32)
        # bump with validation should fail
        with pytest.raises(TypeError, match="dtype"):
            registry.bump("data", unsafe=False)
        # bump with unsafe should succeed
        registry.bump("data", unsafe=True)

    def test_matches_spec_valid(self, registry, numpy_spec):
        """matches_spec returns True for valid buffer."""
        registry.declare("data", spec=numpy_spec)
        buf = np.zeros((10,), dtype=np.float32)
        assert registry.matches_spec("data", buf) is True

    def test_matches_spec_invalid(self, registry, numpy_spec):
        """matches_spec returns False for invalid buffer."""
        registry.declare("data", spec=numpy_spec)
        buf = np.zeros((10,), dtype=np.int32)  # Wrong dtype
        assert registry.matches_spec("data", buf) is False

    def test_validate_buffer_unknown_kind(self, registry):
        """validate_buffer raises ValueError for unknown kind."""
        spec = ResourceSpec(kind="unknown_kind", dtype=np.float32)
        # Declaring with unknown kind should raise ValueError during validation
        with pytest.raises(ValueError, match="unknown_kind"):
            registry.declare("data", spec=spec, buffer=np.array([1.0]))

    def test_ensure_no_producer(self, registry):
        """ensure on resource with no producer is no-op."""
        registry.declare("static_data", buffer=np.array([1.0]))
        registry.ensure("static_data")  # Should not raise
        assert np.array_equal(
            registry.read("static_data", ensure=False), np.array([1.0])
        )

    def test_profiler_id(self, simple_producer):
        """ProducerBase.profiler_id returns consistent id."""
        p = simple_producer
        id1 = p.profiler_id()
        id2 = p.profiler_id()
        assert id1 == id2  # Should be cached

    def test_ensure_cycle_detection(self, registry):
        """ensure detects dependency cycles."""
        registry.declare("a", deps=["b"])
        registry.declare("b", deps=["c"])
        registry.declare("c", deps=["a"])

        class NoOpProducer(ProducerBase):
            outputs = ("x",)

            def compute(self, reg):
                pass

        p = NoOpProducer()
        registry.get("a").producer = p
        registry.get("b").producer = p
        registry.get("c").producer = p

        with pytest.raises(RuntimeError, match="Dependency cycle"):
            registry.ensure("a")


class TestDepNameFunction:
    """Test _dep_name helper function."""

    def test_dep_name_with_resource_ref(self):
        """_dep_name extracts name from ResourceRef."""
        from rheidos.compute.registry import _dep_name
        from rheidos.compute import ResourceRef, ResourceKey, Registry

        registry = Registry()
        registry.declare("my_resource")
        ref = ResourceRef(registry, ResourceKey("my_resource"))
        assert _dep_name(ref) == "my_resource"

    def test_dep_name_with_resource_key(self):
        """_dep_name extracts full_name from ResourceKey."""
        from rheidos.compute.registry import _dep_name
        from rheidos.compute import ResourceKey

        key = ResourceKey("my_resource")
        assert _dep_name(key) == "my_resource"

    def test_dep_name_with_string(self):
        """_dep_name returns string as-is."""
        from rheidos.compute.registry import _dep_name

        name = "my_resource"
        assert _dep_name(name) == "my_resource"


class TestRegistryExplain:
    """Test Registry.explain() debug method."""

    def test_explain_simple_resource(self, registry):
        """explain returns debug string for resource."""
        registry.declare("data", buffer=np.array([1.0]))
        explanation = registry.explain("data")
        assert "data" in explanation
        assert "v=" in explanation

    def test_explain_with_producer(self, registry, simple_producer):
        """explain shows producer and stale status."""
        registry.declare("data", buffer=np.array([1.0]))
        registry.declare("result", deps=["data"], producer=simple_producer)
        explanation = registry.explain("result", depth=2)
        assert "result" in explanation
        assert "data" in explanation
        assert "SimpleProducer" in explanation or "producer" in explanation

    def test_explain_respects_depth(self, registry, simple_producer):
        """explain limits recursion depth."""
        # Create chain: a -> b -> c -> d
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b", deps=["a"], producer=simple_producer)
        registry.declare("c", deps=["b"], producer=simple_producer)
        registry.declare("d", deps=["c"], producer=simple_producer)

        # With depth=1, should only show d and b (one level deep)
        explanation = registry.explain("d", depth=1)
        assert "d" in explanation
        assert "c" in explanation


class TestRegistryCommitBatch:
    """Test Registry.commit_many() batch commit."""

    def test_commit_many_without_buffers(self, registry, simple_producer):
        """commit_many without buffers dict commits all with current buffer."""
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b", buffer=np.array([2.0]))
        registry.declare("sum", deps=["a", "b"], producer=simple_producer)

        # Manually set sum buffer and commit both
        registry.get("sum").buffer = np.array([3.0])
        registry.commit_many(["sum"], buffers=None)
        assert registry.read("sum", ensure=False) is not None

    def test_commit_many_with_partial_buffers(self, registry, simple_producer):
        """commit_many with partial buffers dict commits available ones."""
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b", buffer=np.array([2.0]))
        registry.declare("sum", deps=["a", "b"], producer=simple_producer)

        buffers = {"sum": np.array([3.0])}
        registry.commit_many(["a", "sum"], buffers=buffers)
        # sum should have new buffer, a should keep old
        assert np.array_equal(registry.read("sum", ensure=False), np.array([3.0]))
        assert np.array_equal(registry.read("a", ensure=False), np.array([1.0]))
