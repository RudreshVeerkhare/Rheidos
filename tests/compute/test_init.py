"""Tests for rheidos.compute.__init__.py helpers."""

import numpy as np
from rheidos.compute import (
    ProducerContext,
    ProducerResourceNamespace,
    Registry,
    producer_output,
    shape_from_axis,
    shape_map,
    shape_of,
    shape_from_scalar,
    shape_with_tail,
)


class TestProducerTypingExports:
    """Test public typing exports for decorator producers."""

    def test_producer_context_exported(self):
        """ProducerContext is publicly exported for method annotations."""
        assert ProducerContext.__name__ == "ProducerContext"

    def test_producer_resource_namespace_exported(self):
        """ProducerResourceNamespace is publicly exported for ctx.inputs/outputs."""
        assert ProducerResourceNamespace.__name__ == "ProducerResourceNamespace"

    def test_producer_output_exported(self):
        """producer_output remains part of the public decorator API."""
        output = producer_output("result")
        assert output.name == "result"

    def test_legacy_class_producer_apis_not_exported(self):
        """Legacy class-producer authoring helpers are no longer public exports."""
        import rheidos.compute as compute

        assert not hasattr(compute, "ProducerBase")
        assert not hasattr(compute, "WiredProducer")
        assert not hasattr(compute, "out_field")


class TestShapeOf:
    """Test shape_of() helper."""

    def test_shape_of_returns_fn(self):
        """shape_of returns a function."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10, 5)))

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


class TestShapeMap:
    """Test shape_map() helper."""

    def test_shape_map_returns_fn(self):
        """shape_map returns a function."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10, 5)))

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_map(resource_ref, lambda shape: (shape[0],))
        assert callable(fn)

    def test_shape_map_resolves_mapped_shape(self):
        """shape_map applies the mapper to the resolved shape."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10, 5)))

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_map(resource_ref, lambda shape: (shape[0], 3))
        shape = fn(registry)
        assert shape == (10, 3)

    def test_shape_map_missing_resource(self):
        """shape_map with missing resource returns None."""
        registry = Registry()
        registry.declare("missing")

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("missing"))
        fn = shape_map(resource_ref, lambda shape: shape)
        shape = fn(registry)
        assert shape is None

    def test_shape_map_no_shape_attr(self):
        """shape_map with buffer lacking shape returns None."""
        registry = Registry()
        registry.declare("data", buffer=42)

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_map(resource_ref, lambda shape: shape)
        shape = fn(registry)
        assert shape is None

    def test_shape_map_mapper_error_returns_none(self):
        """shape_map returns None when the mapper raises."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10, 5)))

        from rheidos.compute import ResourceRef, ResourceKey

        def mapper(shape):
            raise ValueError(f"bad shape: {shape}")

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_map(resource_ref, mapper)
        shape = fn(registry)
        assert shape is None

    def test_shape_map_supports_projection_and_extension(self):
        """shape_map supports shrinking and extending resolved shapes."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10, 5)))
        registry.declare("vector", buffer=np.zeros((10,)))

        from rheidos.compute import ResourceRef, ResourceKey

        data_ref = ResourceRef(registry, ResourceKey("data"))
        vector_ref = ResourceRef(registry, ResourceKey("vector"))
        project = shape_map(data_ref, lambda shape: (shape[0],))
        extend = shape_map(vector_ref, lambda shape: (shape[0], 3))

        assert project(registry) == (10,)
        assert extend(registry) == (10, 3)


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


class TestShapeFromAxis:
    """Test shape_from_axis() helper."""

    def test_shape_from_axis_basic(self):
        """shape_from_axis projects one axis into a scalar shape."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10, 5)))

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_from_axis(resource_ref, axis=0)
        shape = fn(registry)
        assert shape == (10,)

    def test_shape_from_axis_with_tail(self):
        """shape_from_axis appends a tail after projection."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10, 5)))

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_from_axis(resource_ref, axis=0, tail=(3,))
        shape = fn(registry)
        assert shape == (10, 3)

    def test_shape_from_axis_invalid_axis_returns_none(self):
        """shape_from_axis returns None when the axis is invalid."""
        registry = Registry()
        registry.declare("data", buffer=np.zeros((10, 5)))

        from rheidos.compute import ResourceRef, ResourceKey

        resource_ref = ResourceRef(registry, ResourceKey("data"))
        fn = shape_from_axis(resource_ref, axis=2)
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
