from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

ResourceName = str
Shape = Tuple[int, ...]
ShapeFn = Callable[["Registry"], Optional[Shape]]


@runtime_checkable
class FieldLike(Protocol):
    dtype: Any
    shape: Shape
