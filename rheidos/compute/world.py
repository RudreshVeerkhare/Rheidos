from dataclasses import dataclass
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from .resource import ResourceRef, ResourceKey
from .registry import Registry, ResourceSpec, ProducerBase

M = TypeVar("M", bound="ModuleBase")
ArgsKey = Tuple[Tuple[Any, ...], Tuple[Tuple[str, Any], ...]]
ModuleKey = Tuple[str, Type["ModuleBase"], ArgsKey]
DepLike = str | ResourceRef[Any] | ResourceKey[Any]
ModuleDepPattern = str


def _make_args_key(args: Iterable[Any], kwargs: Dict[str, Any]) -> ArgsKey:
    key = (tuple(args), tuple(sorted(kwargs.items())))
    try:
        hash(key)
    except TypeError as exc:
        raise TypeError(
            "Module require arguments must be hashable. "
            "Use only hashable values or provide a hashable surrogate."
        ) from exc
    return key


# =============================================================================
# Implicit naming / scope
# =============================================================================


@dataclass(frozen=True)
class Namespace:
    parts: Tuple[str, ...] = ()

    def child(self, name: str) -> "Namespace":
        return Namespace(self.parts + (name,))

    @property
    def prefix(self) -> str:
        return ".".join(self.parts)

    def qualify(self, attr: str) -> str:
        return ".".join(self.parts + (attr,))


# =============================================================================
# Modules + World (NO REQUIRES; dynamic deps only) + cycle detection
# =============================================================================


class ModuleBase:
    NAME: str = "Module"

    def __init__(self, world: "World", *, scope: str = "") -> None:
        self.world = world
        self.reg = world.reg
        self.scope = scope

        root = Namespace((scope,)) if scope else Namespace(())
        self.ns = root.child(self.NAME)

    @property
    def prefix(self) -> str:
        return self.ns.prefix

    def qualify(self, attr: str) -> str:
        return self.ns.qualify(attr)

    # Backwards-friendly alias (internal usage only)
    def r(self, attr: str) -> str:
        return self.qualify(attr)

    def require(self, module_cls: Type[M], *args: Any, **kwargs: Any) -> M:
        # Dynamic dependency discovery happens here.
        return self.world.require(module_cls, *args, scope=self.scope, **kwargs)

    # ---- ergonomic resource declaration ----

    def resource(
        self,
        attr: str,
        *,
        spec: Optional["ResourceSpec"] = None,
        doc: str = "",
        declare: bool = False,
        buffer: Any = None,
        deps: Sequence[DepLike] = (),
        producer: Optional["ProducerBase"] = None,
        description: str = "",
    ) -> ResourceRef[Any]:
        """
        Create a module-scoped ResourceRef, optionally declaring it in the registry.
        """
        key = ResourceKey[Any](self.qualify(attr), spec=spec)
        ref = ResourceRef[Any](self.reg, key, doc=doc)
        if declare:
            self.reg.declare(
                ref.name,
                buffer=buffer,
                deps=tuple(deps),
                producer=producer,
                description=description or doc,
                spec=ref.spec,
            )
        return ref

    def declare_resource(
        self,
        ref: ResourceRef[Any],
        *,
        buffer: Any = None,
        deps: Sequence[DepLike] = (),
        producer: Optional["ProducerBase"] = None,
        description: str = "",
    ) -> None:
        self.reg.declare(
            ref.name,
            buffer=buffer,
            deps=tuple(deps),
            producer=producer,
            description=description or ref.doc,
            spec=ref.spec,
        )


def module_resource_deps(
    module: ModuleBase,
    *,
    include: ModuleDepPattern = r".*",
    exclude: ModuleDepPattern = r"^$",
) -> Tuple[ResourceRef[Any], ...]:
    include_re = re.compile(include)
    exclude_re = re.compile(exclude) if exclude else None
    deps = []
    for name, value in vars(module).items():
        if not isinstance(value, ResourceRef):
            continue
        if not include_re.search(name):
            continue
        if exclude_re is not None and exclude_re.search(name):
            continue
        deps.append(value)
    return tuple(deps)


class World:
    def __init__(self) -> None:
        self.reg = Registry()
        self._modules: Dict[ModuleKey, ModuleBase] = {}
        self._module_deps: Dict[ModuleKey, Set[ModuleKey]] = {}

        # Cycle detection for module requires
        self._building_stack: List[ModuleKey] = []
        self._building_set: Set[ModuleKey] = set()

    def module_dependencies(self) -> Dict[ModuleKey, Set[ModuleKey]]:
        return {k: set(v) for k, v in self._module_deps.items()}

    def _record_module_dep(self, parent: Optional[ModuleKey], child: ModuleKey) -> None:
        self._module_deps.setdefault(child, set())
        if parent is None or parent == child:
            return
        self._module_deps.setdefault(parent, set()).add(child)

    def require(
        self, module_cls: Type[M], *args: Any, scope: str = "", **kwargs: Any
    ) -> M:
        args_key = _make_args_key(args, kwargs)
        key: ModuleKey = (scope, module_cls, args_key)

        parent = self._building_stack[-1] if self._building_stack else None

        # Already built
        existing = self._modules.get(key)
        if existing is not None:
            self._record_module_dep(parent, key)
            return cast(M, existing)

        # Cycle detection (dynamic deps are discovered by execution; cycles are fatal)
        if key in self._building_set:
            try:
                i = self._building_stack.index(key)
            except ValueError:
                i = 0
            cyc = self._building_stack[i:] + [key]

            def fmt(k: ModuleKey) -> str:
                sc, cls, akey = k
                scs = sc if sc else "<root>"
                args, kw = akey
                parts = [repr(a) for a in args]
                parts.extend(f"{k}={repr(v)}" for k, v in kw)
                suffix = "(" + ", ".join(parts) + ")" if parts else ""
                return f"{scs}:{cls.__name__}{suffix}"

            cycle_str = " -> ".join(fmt(k) for k in cyc)
            raise RuntimeError(f"Module dependency cycle detected: {cycle_str}")

        self._record_module_dep(parent, key)

        # Build
        self._building_stack.append(key)
        self._building_set.add(key)
        try:
            m = module_cls(self, *args, scope=scope, **kwargs)
            self._modules[key] = m
            return cast(M, m)
        finally:
            # Always unwind even if __init__ raises
            popped = self._building_stack.pop()
            self._building_set.remove(popped)
