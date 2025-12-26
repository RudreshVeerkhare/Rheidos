"""
Resource is an abstraction for encapsulating data-containers
"""

from dataclasses import dataclass
from typing import Any, Tuple, Optional, Generic, TypeVar, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ProducerBase, ResourceSpec, Registry

ResourceName = str

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
    # Use forward reference to avoid cyclic import
    producer: Optional["ProducerBase"] = None
    version: int = 0
    dep_sign: Tuple[Tuple[ResourceName, int], ...] = ()
    desc: str = ""


# =============================================================================
# User-facing typed handles (IntelliSense-friendly)
# =============================================================================

T = TypeVar("T")

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
