from .registry import ProducerBase, Registry
from .resource import Resource, ResourceKey, ResourceKind, ResourceRef, ResourceSpec
from .typing import FieldLike, ResourceName, Shape, ShapeFn
from .wiring import WiredProducer, out_field
from .world import ModuleBase, Namespace, World

from typing import Any, Optional


def shape_of(ref: ResourceRef[Any]) -> ShapeFn:
    def fn(reg: Registry) -> Optional[Shape]:
        buf = reg.read(ref.name, ensure=False)
        if buf is None or not hasattr(buf, "shape"):
            return None
        return tuple(buf.shape)

    return fn


__all__ = [
    "FieldLike",
    "ModuleBase",
    "Namespace",
    "ProducerBase",
    "Registry",
    "Resource",
    "ResourceKey",
    "ResourceKind",
    "ResourceName",
    "ResourceRef",
    "ResourceSpec",
    "Shape",
    "ShapeFn",
    "WiredProducer",
    "World",
    "out_field",
    "shape_of",
]
