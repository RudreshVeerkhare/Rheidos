from .registry import Registry
from .resource import Resource, ResourceKey, ResourceKind, ResourceRef, ResourceSpec
from .resource_kinds import ResourceKindAdapter, register_resource_kind
from .typing import FieldLike, ResourceName, Shape, ShapeFn
from .wiring import (
    ProducerContext,
    ProducerResourceNamespace,
    producer,
    producer_output,
)
from .world import (
    ModuleBase,
    ModuleInputContractError,
    Namespace,
    ResourceView,
    World,
    module_resource_deps,
    resource_view,
)

from typing import Any, Callable, Optional, Tuple


def shape_map(
    ref: ResourceRef[Any], mapper: Callable[[Shape], Shape]
) -> ShapeFn:
    def fn(reg: Registry) -> Optional[Shape]:
        buf = reg.read(ref.name, ensure=False)
        if buf is None or not hasattr(buf, "shape"):
            return None
        try:
            return tuple(mapper(tuple(buf.shape)))
        except Exception:
            return None

    return fn


def shape_of(ref: ResourceRef[Any]) -> ShapeFn:
    return shape_map(ref, lambda shape: shape)


def shape_from_scalar(ref: ResourceRef[Any], *, tail: Tuple[int, ...] = ()) -> ShapeFn:
    def fn(reg: Registry) -> Optional[Shape]:
        buf = reg.read(ref.name, ensure=False)
        if buf is None:
            return None
        try:
            if hasattr(buf, "__getitem__"):
                try:
                    n = int(buf[None])
                except Exception:
                    try:
                        n = int(buf[()])
                    except Exception:
                        n = int(buf[0])
            else:
                n = int(buf)
        except Exception:
            return None
        return (n,) + tuple(tail)

    return fn


def shape_from_axis(
    ref: ResourceRef[Any],
    axis: int = 0,
    *,
    tail: Tuple[int, ...] = (),
) -> ShapeFn:
    return shape_map(ref, lambda shape: (int(shape[axis]),) + tuple(tail))


def shape_with_tail(ref: ResourceRef[Any], *, tail: Tuple[int, ...] = ()) -> ShapeFn:
    return shape_map(ref, lambda shape: shape + tuple(tail))


__all__ = [
    "FieldLike",
    "ModuleBase",
    "ModuleInputContractError",
    "Namespace",
    "module_resource_deps",
    "ProducerContext",
    "ProducerResourceNamespace",
    "Registry",
    "Resource",
    "ResourceKey",
    "ResourceKind",
    "ResourceName",
    "ResourceRef",
    "ResourceSpec",
    "ResourceView",
    "ResourceKindAdapter",
    "Shape",
    "ShapeFn",
    "World",
    "producer",
    "producer_output",
    "resource_view",
    "register_resource_kind",
    "shape_map",
    "shape_of",
    "shape_from_axis",
    "shape_from_scalar",
    "shape_with_tail",
]
