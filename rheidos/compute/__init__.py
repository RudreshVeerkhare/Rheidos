from .graph import (
    export_dependency_graph_dot,
    format_dependency_graph,
    format_dependency_graph_dot,
    print_dependency_graph,
)
from .registry import ProducerBase, Registry
from .resource import Resource, ResourceKey, ResourceKind, ResourceRef, ResourceSpec
from .typing import FieldLike, ResourceName, Shape, ShapeFn
from .wiring import WiredProducer, out_field
from .world import ModuleBase, Namespace, World, module_resource_deps

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
    "export_dependency_graph_dot",
    "format_dependency_graph",
    "format_dependency_graph_dot",
    "ModuleBase",
    "Namespace",
    "module_resource_deps",
    "print_dependency_graph",
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
