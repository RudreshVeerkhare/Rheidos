"""Tests for rheidos.compute.__init__.py helpers."""

import pytest
import numpy as np

from rheidos.compute import Registry, shape_of, shape_from_scalar, shape_with_tail


class TestShapeOf:
    """Test shape_of() helper."""

    def test_shape_of_returns_fn(self):
        """shape_of returns a function."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10, 5)))
        ref = registry.get("data")

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_of(resource_ref)
        assert callable(fn)

    def test_shape_of_resolves_shape(self):
        """shape_of function resolves buffer shape."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10, 5)))

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_of(resource_ref)
        shape = fn(registry)
        assert shape == (10, 5)

    def test_shape_of_missing_resource(self):
        """shape_of with missing resource returns None."""
        registry = Registry()
        # Declare a resource without a buffer
        registry.declare("missing")

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("missing"))
        fn = shape_of(resource_ref)
        shape = fn(registry)
        assert shape is None

    def test_shape_of_no_shape_attr(self):
        """shape_of with buffer lacking shape returns None."""
        registry = Registry()
        registry.declare("data", buffer=42)  # Scalar, no shape attr

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_of(resource_ref)
        shape = fn(registry)
        assert shape is None


class TestShapeFromScalar:
    """Test shape_from_scalar() helper."""

    def test_shape_from_scalar_int(self):
        """shape_from_scalar with integer."""
        registry = Registry()
        registry.declare("data", buffer=5)

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_from_scalar(resource_ref)
        shape = fn(registry)
        assert shape == (5,)

    def test_shape_from_scalar_float(self):
        """shape_from_scalar with float."""
        registry = Registry()
        registry.declare("data", buffer=5.0)

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_from_scalar(resource_ref)
        shape = fn(registry)
        assert shape == (5,)

    def test_shape_from_scalar_with_tail(self):
        """shape_from_scalar with tail tuple."""
        registry = Registry()
        registry.declare("data", buffer=5)

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_from_scalar(resource_ref, tail=(2, 3))
        shape = fn(registry)
        assert shape == (5, 2, 3)

    def test_shape_from_scalar_none_buffer(self):
        """shape_from_scalar with None buffer."""
        registry = Registry()
        registry.declare("data")  # No buffer

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_from_scalar(resource_ref)
        shape = fn(registry)
        assert shape is None


class TestShapeWithTail:
    """Test shape_with_tail() helper."""

    def test_shape_with_tail_basic(self):
        """shape_with_tail appends tail to shape."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10,)))

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_with_tail(resource_ref, tail=(2, 3))
        shape = fn(registry)
        assert shape == (10, 2, 3)

    def test_shape_with_tail_empty_tail(self):
        """shape_with_tail with empty tail."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10,)))

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_with_tail(resource_ref, tail=())
        shape = fn(registry)
        assert shape == (10,)

    def test_shape_with_tail_none_buffer(self):
        """shape_with_tail with None buffer."""
        registry = Registry()
        registry.declare("data")  # No buffer

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_with_tail(resource_ref, tail=(2, 3))
        shape = fn(registry)
        assert shape is None

    def test_shape_from_scalar_non_numeric(self):
        """shape_from_scalar with non-numeric buffer."""
        registry = Registry()
        registry.declare("data", buffer="not_a_number")  # Non-numeric

        from rheidos.compute import ResourceRef, ResourceKey, shape_from_scalar

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_from_scalar(resource_ref)
        shape = fn(registry)
        # Should return None due to exception
        assert shape is None

    def test_shape_of_missing_shape_attr(self):
        """shape_of with buffer that has no shape attribute."""
        registry = Registry()
        registry.declare("data", buffer=42)  # Plain int has no shape

        from rheidos.compute import ResourceRef, ResourceKey, shape_of

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_of(resource_ref)
        shape = fn(registry)
        # Should return None
        assert shape is None

    def test_shape_from_scalar_with_indexing(self):
        """shape_from_scalar with buffer supporting indexing."""
        registry = Registry()
        # A buffer that supports indexing (numpy array)
        registry.declare("data", buffer=np.array([5.0]))

        from rheidos.compute import ResourceRef, ResourceKey, shape_from_scalar

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_from_scalar(resource_ref)
        shape = fn(registry)
        # Should return (5,)
        assert shape == (5,)
