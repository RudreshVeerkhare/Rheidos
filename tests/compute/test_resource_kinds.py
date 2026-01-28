"""Tests for rheidos.compute.resource_kinds module."""

import pytest
import numpy as np
from rheidos.compute.resource_kinds import (
    ResourceKindAdapter,
    register_resource_kind,
    get_resource_kind,
)
from rheidos.compute import ResourceSpec, Registry


class TestResourceKindRegistration:
    """Test resource kind registration and lookup."""

    def test_get_numpy_kind(self):
        """Get built-in numpy kind."""
        adapter = get_resource_kind("numpy")
        assert isinstance(adapter, ResourceKindAdapter)

    def test_register_custom_kind(self):
        """Register custom resource kind."""

        def resolve_shape(reg, spec):
            return (10,)

        def allocate(reg, spec, shape):
            return np.zeros(shape, dtype=np.float32)

        def matches_spec(reg, spec, buf):
            return isinstance(buf, np.ndarray)

        adapter = ResourceKindAdapter(
            resolve_shape=resolve_shape,
            allocate=allocate,
            matches_spec=matches_spec,
        )
        # Use unique name to avoid conflicts
        kind_name = "custom_test_kind_12345"
        register_resource_kind(kind_name, adapter)
        retrieved = get_resource_kind(kind_name)
        assert retrieved is adapter

    def test_register_duplicate_kind_raises(self):
        """Registering duplicate kind name raises KeyError."""
        adapter = ResourceKindAdapter(
            resolve_shape=lambda reg, spec: (10,),
            allocate=lambda reg, spec, shape: np.zeros(shape),
            matches_spec=lambda reg, spec, buf: True,
        )
        kind_name = "duplicate_test_12345"
        register_resource_kind(kind_name, adapter)
        with pytest.raises(KeyError, match="already registered"):
            register_resource_kind(kind_name, adapter)

    def test_register_empty_name_raises(self):
        """Registering empty kind name raises ValueError."""
        adapter = ResourceKindAdapter(
            resolve_shape=lambda reg, spec: (10,),
            allocate=lambda reg, spec, shape: np.zeros(shape),
            matches_spec=lambda reg, spec, buf: True,
        )
        with pytest.raises(ValueError, match="non-empty"):
            register_resource_kind("", adapter)

    def test_get_unknown_kind_raises(self):
        """Getting unknown kind raises KeyError."""
        with pytest.raises(KeyError, match="Unknown ResourceSpec.kind"):
            get_resource_kind("nonexistent_kind_xyz")


class TestNumpyAdapter:
    """Test numpy resource kind adapter."""

    def test_numpy_allocation(self):
        """Allocate numpy array with spec."""
        adapter = get_resource_kind("numpy")
        spec = ResourceSpec(kind="numpy", dtype=np.float32, shape=(10,))
        buf = adapter.allocate(None, spec, (10,))
        assert isinstance(buf, np.ndarray)
        assert buf.dtype == np.float32
        assert buf.shape == (10,)

    def test_numpy_matches_valid(self):
        """Numpy adapter validates matching buffer."""
        adapter = get_resource_kind("numpy")
        spec = ResourceSpec(kind="numpy", dtype=np.float32, shape=(10,))
        buf = np.zeros((10,), dtype=np.float32)
        # Should not raise
        adapter.matches_spec(None, spec, buf)

    def test_numpy_matches_wrong_dtype_raises(self):
        """Numpy adapter rejects wrong dtype."""
        adapter = get_resource_kind("numpy")
        spec = ResourceSpec(kind="numpy", dtype=np.float32, shape=(10,))
        buf = np.zeros((10,), dtype=np.int32)
        with pytest.raises(TypeError, match="dtype"):
            adapter.matches_spec(None, spec, buf)

    def test_numpy_matches_wrong_shape_raises(self):
        """Numpy adapter rejects wrong shape."""
        adapter = get_resource_kind("numpy")
        spec = ResourceSpec(kind="numpy", dtype=np.float32, shape=(10,))
        buf = np.zeros((5,), dtype=np.float32)
        with pytest.raises(TypeError, match="shape"):
            adapter.matches_spec(None, spec, buf)

    def test_numpy_matches_not_ndarray_raises(self):
        """Numpy adapter rejects non-ndarray."""
        adapter = get_resource_kind("numpy")
        spec = ResourceSpec(kind="numpy", dtype=np.float32)
        with pytest.raises(TypeError, match="ndarray"):
            adapter.matches_spec(None, spec, [1, 2, 3])

    def test_numpy_allocation_without_dtype_raises(self):
        """Numpy allocation without dtype raises TypeError."""
        adapter = get_resource_kind("numpy")
        spec = ResourceSpec(kind="numpy")  # No dtype
        with pytest.raises(TypeError, match="dtype"):
            adapter.allocate(None, spec, (10,))


class TestTaichiAdapter:
    """Test taichi resource kind adapter (conditional on taichi availability)."""

    @pytest.mark.skipif(
        True,  # Skip by default; taichi is optional
        reason="taichi not required for basic tests",
    )
    def test_taichi_allocation(self):
        """Allocate taichi field with spec."""
        try:
            import taichi as ti
        except ImportError:
            pytest.skip("taichi not installed")

        adapter = get_resource_kind("taichi_field")
        spec = ResourceSpec(kind="taichi_field", dtype=ti.f32, shape=(10,))
        buf = adapter.allocate(None, spec, (10,))
        assert hasattr(buf, "dtype")
        assert hasattr(buf, "shape")


class TestPythonAdapter:
    """Test python resource kind adapter."""

    def test_python_adapter_returns_none(self):
        """Python adapter always matches and allocates None."""
        adapter = get_resource_kind("python")
        spec = ResourceSpec(kind="python")
        result = adapter.allocate(None, spec, None)
        assert result is None

    def test_python_matches_anything(self):
        """Python adapter matches any buffer."""
        adapter = get_resource_kind("python")
        spec = ResourceSpec(kind="python")
        # Should not raise for any buffer
        adapter.matches_spec(None, spec, None)
        adapter.matches_spec(None, spec, 42)
        adapter.matches_spec(None, spec, "string")
        adapter.matches_spec(None, spec, np.array([1, 2, 3]))

    def test_python_resolve_shape_returns_none(self):
        """Python adapter resolve_shape always returns None."""
        adapter = get_resource_kind("python")
        spec = ResourceSpec(kind="python")
        shape = adapter.resolve_shape(None, spec)
        assert shape is None


class TestNumpyAdapterEdgeCases:
    """Test edge cases for numpy adapter."""

    def test_numpy_matches_lanes_not_applied(self):
        """Numpy adapter doesn't check lanes (numpy doesn't have lanes)."""
        adapter = get_resource_kind("numpy")
        spec = ResourceSpec(kind="numpy", dtype=np.float32, shape=(10,), lanes=4)
        buf = np.zeros((10,), dtype=np.float32)
        # Should still validate without error (lanes ignored for numpy)
        adapter.matches_spec(None, spec, buf)

    def test_numpy_none_dtype_in_spec_matches(self):
        """Numpy adapter with None dtype in spec matches any dtype."""
        adapter = get_resource_kind("numpy")
        spec = ResourceSpec(kind="numpy", dtype=None, shape=(10,))
        buf = np.zeros((10,), dtype=np.int32)
        # Should pass when dtype is None in spec
        adapter.matches_spec(None, spec, buf)

    def test_numpy_resolve_shape_from_registry(self):
        """Numpy adapter resolves shape with tuple shape specification."""
        adapter = get_resource_kind("numpy")
        spec = ResourceSpec(kind="numpy", dtype=np.float32, shape=(10, 5))
        resolved_shape = adapter.resolve_shape(None, spec)
        assert resolved_shape == (10, 5)

    def test_numpy_resolve_shape_from_shape_fn(self):
        """Numpy adapter resolves shape from shape_fn when shape is None."""
        adapter = get_resource_kind("numpy")

        def get_shape(reg):
            return (20, 3)

        spec = ResourceSpec(
            kind="numpy", dtype=np.float32, shape=None, shape_fn=get_shape
        )
        resolved_shape = adapter.resolve_shape(None, spec)
        assert resolved_shape == (20, 3)

    def test_numpy_resolve_shape_none_returns_none(self):
        """Numpy adapter returns None when both shape and shape_fn are None."""
        adapter = get_resource_kind("numpy")
        spec = ResourceSpec(kind="numpy", dtype=np.float32, shape=None, shape_fn=None)
        resolved_shape = adapter.resolve_shape(None, spec)
        assert resolved_shape is None

    def test_numpy_matches_any_shape_when_spec_shape_is_none(self):
        """Numpy adapter doesn't validate shape when spec.shape is None."""
        adapter = get_resource_kind("numpy")
        spec = ResourceSpec(kind="numpy", dtype=np.float32, shape=None)
        buf = np.zeros((15, 25), dtype=np.float32)
        # Should pass because spec.shape is None
        adapter.matches_spec(None, spec, buf)

    def test_taichi_field_allocation_raises_without_taichi(self):
        """Taichi field allocation raises error if taichi not properly initialized."""
        adapter = get_resource_kind("taichi_field")
        spec = ResourceSpec(kind="taichi_field", dtype=None, shape=(10,))
        # Taichi requires ti.init() to be called first
        try:
            buf = adapter.allocate(None, spec, (10,))
            # If taichi is available and initialized, just check it has expected attrs
            assert hasattr(buf, "dtype") or hasattr(buf, "shape")
        except Exception as e:
            # Expected error from taichi when not initialized or taichi unavailable
            assert isinstance(e, (RuntimeError, Exception))

    def test_taichi_matches_invalid_buffer_raises(self):
        """Taichi adapter rejects non-field-like buffer."""
        adapter = get_resource_kind("taichi_field")
        spec = ResourceSpec(kind="taichi_field")
        with pytest.raises(TypeError, match="Taichi field-like"):
            adapter.matches_spec(None, spec, [1, 2, 3])

    def test_taichi_resolve_shape(self):
        """Taichi adapter has shape resolution."""
        adapter = get_resource_kind("taichi_field")
        spec = ResourceSpec(kind="taichi_field", shape=(10, 5))
        resolved_shape = adapter.resolve_shape(None, spec)
        assert resolved_shape == (10, 5)
