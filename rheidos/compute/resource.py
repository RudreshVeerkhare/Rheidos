"""
Resource is an abstraction for encapsulating data-containers
"""

from dataclasses import dataclass
from typing import Any, Tuple, Optional, Generic, TypeVar, Literal, cast

from .typing import ResourceName, Shape, ShapeFn


# =============================================================================
# User-facing typed handles (IntelliSense-friendly)
# =============================================================================

T = TypeVar("T")
ResourceKind = Literal["taichi_field", "numpy", "python"]

@dataclass(frozen=True)
class ResourceSpec:
    """
    Lightweight runtime schema for buffers.

    kind:
      - "taichi_field": ti.field / ti.Vector.field / ti.Matrix.field (best-effort checks)
      - "numpy":        np.ndarray
      - "python":       any (no checks unless you extend)

    dtype:
      - For taichi_field: ti.f32, ti.i32, ...
      - For numpy: np.float32, np.int32, ...

    lanes:
      - For ti.Vector.field(n, ...): lanes=n (best effort; if unknown, we skip)

    shape / shape_fn:
      - If provided, enforce exact .shape match.
      - shape_fn lets you compute expected shape from deps at commit-time.

    allow_none:
      - if False, disallow None buffer.

    Notes:
      - This is runtime validation, not static typing.
      - Taichi fields are intentionally treated "field-like" (best effort).
    """

    kind: ResourceKind  # "taichi_field" | "numpy" | "python"
    dtype: Optional[Any] = None
    lanes: Optional[int] = None
    shape: Optional[Shape] = None
    shape_fn: Optional[ShapeFn] = None
    allow_none: bool = True


@dataclass
class Resource:
    """
    Single unified resource type.

    - deps: names of dependencies (can be empty)
    - producer: optional. If present, ensure() may run it when stale.
    - version: increments when you bump this resource.
    - dep_sig: snapshot of deps' versions when this resource was last bumped.
    """

    name: ResourceName
    buffer: Any = None

    deps: Tuple[ResourceName, ...] = ()
    producer: Optional["ProducerBase"] = None

    version: int = 0
    dep_sig: Tuple[Tuple[ResourceName, int], ...] = ()

    description: str = ""
    spec: Optional[ResourceSpec] = None



@dataclass(frozen=True)
class ResourceKey(Generic[T]):
    full_name: str
    # Use forward reference to avoid cyclic import
    spec: Optional["ResourceSpec"] = None


class ResourceRef(Generic[T]):
    def __init__(self, reg: "Registry", key: ResourceKey[T], doc: str = "") -> None:
        self._reg = reg
        self._key = key
        self.doc = doc
        self.__doc__ = doc

    @property
    def name(self) -> str:
        return self._key.full_name

    @property
    def spec(self) -> Optional[ResourceSpec]:
        return self._key.spec

    def ensure(self) -> None:
        self._reg.ensure(self.name)

    def get(self, *, ensure: bool = True) -> T:
        return cast(T, self._reg.read(self.name, ensure=ensure))

    def set(self, value: T, *, unsafe: bool = False) -> None:
        # "Replace buffer + mark fresh"
        self._reg.commit(self.name, buffer=value, unsafe=unsafe)

    def set_buffer(self, value: T, *, bump: bool = False, unsafe: bool = False) -> None:
        # Allocation-before-fill patterns generally want bump=False.
        self._reg.set_buffer(self.name, value, bump=bump, unsafe=unsafe)

    def commit(self, *, unsafe: bool = False) -> None:
        # keyword-only is intentional: makes unsafe explicit.
        self._reg.commit(self.name, unsafe=unsafe)

    # Clearer aliases (non-breaking; you can pick one convention and stick to it)
    def mark_fresh(self, *, unsafe: bool = False) -> None:
        self.commit(unsafe=unsafe)

    def touch(self, *, unsafe: bool = False) -> None:
        self.commit(unsafe=unsafe)

    def bump(self, *, unsafe: bool = False) -> None:
        self._reg.bump(self.name, unsafe=unsafe)
