from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, TypeVar, cast

from .resource import ResourceRef, ResourceKey
from .registry import Registry, ResourceSpec, ProducerBase

M = TypeVar("M", bound="ModuleBase")
ModuleKey = Tuple[str, Type["ModuleBase"]]
DepLike = str | ResourceRef[Any] | ResourceKey[Any]
ModuleDepPattern = str

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

    def require(self, module_cls: Type[M]) -> M:
        # Dynamic dependency discovery happens here.
        return self.world.require(module_cls, scope=self.scope)

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

    def export_dag_dot(
        self,
        *,
        include_resources: bool = True,
        include_producers: bool = True,
        include_modules: bool = False,
        sort: bool = True,
        rankdir: str = "LR",
    ) -> str:
        from .graph import format_dependency_graph_dot

        return format_dependency_graph_dot(
            self,
            include_resources=include_resources,
            include_producers=include_producers,
            include_modules=include_modules,
            sort=sort,
            rankdir=rankdir,
        )

    def _record_module_dep(self, parent: Optional[ModuleKey], child: ModuleKey) -> None:
        self._module_deps.setdefault(child, set())
        if parent is None or parent == child:
            return
        self._module_deps.setdefault(parent, set()).add(child)

    def require(self, module_cls: Type[M], *, scope: str = "") -> M:
        key: ModuleKey = (scope, module_cls)

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
                sc, cls = k
                scs = sc if sc else "<root>"
                return f"{scs}:{cls.__name__}"

            cycle_str = " -> ".join(fmt(k) for k in cyc)
            raise RuntimeError(f"Module dependency cycle detected: {cycle_str}")

        self._record_module_dep(parent, key)

        # Build
        self._building_stack.append(key)
        self._building_set.add(key)
        try:
            m = module_cls(self, scope=scope)
            self._modules[key] = m
            return cast(M, m)
        finally:
            # Always unwind even if __init__ raises
            popped = self._building_stack.pop()
            self._building_set.remove(popped)
