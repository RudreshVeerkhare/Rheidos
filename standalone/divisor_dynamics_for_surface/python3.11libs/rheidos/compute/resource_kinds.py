from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .resource import ResourceSpec
from .typing import Shape


@dataclass(frozen=True)
class ResourceKindAdapter:
    resolve_shape: Callable[[Any, ResourceSpec], Optional[Shape]]
    allocate: Callable[[Any, ResourceSpec, Shape], Any]
    matches_spec: Callable[[Any, ResourceSpec, Any], bool]
    requires_shape: bool = True


_ADAPTERS: Dict[str, ResourceKindAdapter] = {}


def register_resource_kind(kind: str, adapter: ResourceKindAdapter) -> None:
    if not kind:
        raise ValueError("Resource kind name must be non-empty")
    if kind in _ADAPTERS:
        raise KeyError(f"Resource kind already registered: {kind}")
    _ADAPTERS[kind] = adapter


def get_resource_kind(kind: str) -> ResourceKindAdapter:
    try:
        return _ADAPTERS[kind]
    except KeyError as e:
        raise KeyError(f"Unknown ResourceSpec.kind: {kind}") from e


def _resolve_shape(reg: Any, spec: ResourceSpec) -> Optional[Shape]:
    shape = spec.shape
    if shape is None and spec.shape_fn is not None:
        shape = spec.shape_fn(reg)
    if shape is None:
        return None
    return tuple(shape)


def _alloc_numpy(reg: Any, spec: ResourceSpec, shape: Shape) -> Any:
    if spec.dtype is None:
        raise TypeError("numpy allocation requires dtype")
    return np.zeros(shape, dtype=np.dtype(spec.dtype))


def _matches_numpy(reg: Any, spec: ResourceSpec, buf: Any) -> bool:
    if not isinstance(buf, np.ndarray):
        raise TypeError(f"expected numpy ndarray, got {type(buf)}")
    if spec.dtype is not None and buf.dtype != np.dtype(spec.dtype):
        raise TypeError(f"expected dtype {spec.dtype}, got {buf.dtype}")
    exp_shape = _resolve_shape(reg, spec)
    if exp_shape is not None and tuple(buf.shape) != tuple(exp_shape):
        raise TypeError(f"expected shape {exp_shape}, got {tuple(buf.shape)}")
    return True


def _alloc_taichi(reg: Any, spec: ResourceSpec, shape: Shape) -> Any:
    try:
        import taichi as ti
    except Exception as e:
        raise RuntimeError("taichi_field allocation requires taichi") from e
    dtype = spec.dtype if spec.dtype is not None else ti.f32
    if spec.lanes is None:
        return ti.field(dtype=dtype, shape=shape)
    return ti.Vector.field(spec.lanes, dtype=dtype, shape=shape)


def _matches_taichi(reg: Any, spec: ResourceSpec, buf: Any) -> bool:
    if not hasattr(buf, "dtype") or not hasattr(buf, "shape"):
        raise TypeError(f"expected Taichi field-like buffer, got {type(buf)}")

    if spec.dtype is not None:
        try:
            if buf.dtype != spec.dtype:
                raise TypeError(f"expected dtype {spec.dtype}, got {buf.dtype}")
        except Exception as e:
            raise TypeError(f"could not validate dtype: {e}") from e

    exp_shape = _resolve_shape(reg, spec)
    if exp_shape is not None:
        try:
            if tuple(buf.shape) != tuple(exp_shape):
                raise TypeError(f"expected shape {exp_shape}, got {tuple(buf.shape)}")
        except Exception as e:
            raise TypeError(f"could not validate shape: {e}") from e

    if spec.lanes is not None:
        lanes_actual = getattr(buf, "n", None)
        if lanes_actual is not None and int(lanes_actual) != int(spec.lanes):
            raise TypeError(f"expected lanes {spec.lanes}, got {lanes_actual}")
    return True


def _resolve_shape_python(reg: Any, spec: ResourceSpec) -> Optional[Shape]:
    return None


def _alloc_python(reg: Any, spec: ResourceSpec, shape: Shape) -> Any:
    return None


def _matches_python(reg: Any, spec: ResourceSpec, buf: Any) -> bool:
    return True


register_resource_kind(
    "numpy",
    ResourceKindAdapter(
        resolve_shape=_resolve_shape,
        allocate=_alloc_numpy,
        matches_spec=_matches_numpy,
        requires_shape=True,
    ),
)
register_resource_kind(
    "taichi_field",
    ResourceKindAdapter(
        resolve_shape=_resolve_shape,
        allocate=_alloc_taichi,
        matches_spec=_matches_taichi,
        requires_shape=True,
    ),
)
register_resource_kind(
    "python",
    ResourceKindAdapter(
        resolve_shape=_resolve_shape_python,
        allocate=_alloc_python,
        matches_spec=_matches_python,
        requires_shape=False,
    ),
)
