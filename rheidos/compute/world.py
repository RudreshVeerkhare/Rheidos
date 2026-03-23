from dataclasses import dataclass
from difflib import get_close_matches
import re
from typing import (
    Any,
    Callable,
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
from .wiring import _DecoratedMethodProducer, get_producer_spec

M = TypeVar("M", bound="ModuleBase")
ArgsKey = Tuple[Tuple[Any, ...], Tuple[Tuple[str, Any], ...]]
# Module identity now has two independent scope dimensions:
# - identity_scope: where this module's own resources live
# - lookup_scope: where plain require(...) calls inside the module resolve
#   their dependencies.
#
# Child modules use a nested identity_scope but inherit their parent's
# lookup_scope so they naturally share the same surrounding modules.
ModuleKey = Tuple[str, str, Type["ModuleBase"], ArgsKey]
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


def _scope_label(scope: str) -> str:
    return scope if scope else "<root>"


def _validate_child_name(child_name: str) -> None:
    if child_name == "" or "." in child_name:
        raise ValueError(
            "child_name must be a single non-empty namespace segment"
        )


def format_module_key(key: ModuleKey) -> str:
    """Format a module cache key for logs, graphs, and cycle errors.

    When a module's lookup scope differs from its identity scope, the output
    shows both so nested-child behavior is visible in diagnostics.
    """
    identity_scope, lookup_scope, module_cls, args_key = key
    module_name = getattr(module_cls, "NAME", module_cls.__name__)
    prefix = f"{identity_scope}.{module_name}" if identity_scope else module_name
    args, kwargs = args_key
    parts = [repr(a) for a in args]
    parts.extend(f"{k}={repr(v)}" for k, v in kwargs)
    suffix = "(" + ", ".join(parts) + ")" if parts else ""
    label = prefix + suffix
    if lookup_scope != identity_scope:
        label = f"{label} [lookup={_scope_label(lookup_scope)}]"
    if module_name != module_cls.__name__:
        return f"{label} ({module_cls.__name__})"
    return label


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


@dataclass(frozen=True)
class _ModuleBuildContext:
    """Internal module-construction context pushed by World.require().

    Existing ModuleBase subclasses still accept only `scope=...` publicly.
    The active build context is how the world injects the extra lookup-scope
    information needed for child modules without forcing every module class
    to grow new constructor parameters.
    """
    module_cls: Type["ModuleBase"]
    identity_scope: str
    lookup_scope: str
    parent_key: Optional["ModuleKey"] = None


class ModuleBase:
    NAME: str = "Module"

    def __init__(self, world: "World", *, scope: str = "") -> None:
        self.world = world
        self.reg = world.reg
        self.scope = scope
        self._bound_producer_methods: Set[str] = set()
        self._module_key: Optional[ModuleKey] = None

        build_ctx = world._active_build_context()
        if (
            build_ctx is not None
            and build_ctx.module_cls is self.__class__
            and build_ctx.identity_scope == scope
        ):
            # This instance is being built by World.require(). In that case the
            # world may intentionally separate the module's identity namespace
            # from the namespace used for dependency lookup.
            self._lookup_scope = build_ctx.lookup_scope
        else:
            # Direct manual construction keeps legacy behavior: scope controls
            # both identity and dependency lookup.
            self._lookup_scope = scope

        root = Namespace((scope,)) if scope else Namespace(())
        self.ns = root.child(self.NAME)

    @property
    def prefix(self) -> str:
        return self.ns.prefix

    @property
    def lookup_scope(self) -> str:
        """Scope used for plain dependency lookup inside this module.

        Examples:
        - normal module: `lookup_scope == scope`
        - child module: resources live under a nested `scope`, but
          `lookup_scope` is inherited from the parent so `self.require(Foo)`
          reuses the same surrounding dependencies visible to the parent.
        """
        return self._lookup_scope

    def qualify(self, attr: str) -> str:
        return self.ns.qualify(attr)

    # Backwards-friendly alias (internal usage only)
    def r(self, attr: str) -> str:
        return self.qualify(attr)

    def require(
        self,
        module_cls: Type[M],
        *args: Any,
        child: bool = False,
        child_name: Optional[str] = None,
        **kwargs: Any,
    ) -> M:
        """Require another module relative to this module.

        Usage:
        - `self.require(Foo)`:
          resolve `Foo` in this module's lookup scope
        - `self.require(Foo, child=True, child_name="solver")`:
          create a child module whose resources live under
          `<this module>.<child_name>.<Foo.NAME>.*`
          while plain requires inside the child still resolve through this
          module's lookup scope.

        The `child=True` form is the normal ergonomic API for composition.
        Use explicit `world.require(..., parent=...)` only from code that is
        outside a module instance and therefore has no implicit parent.
        """
        if child:
            if "parent" in kwargs:
                raise TypeError(
                    "ModuleBase.require() does not accept 'parent'; "
                    "the parent module is implicit"
                )
            return self.world.require(
                module_cls,
                *args,
                child=True,
                child_name=child_name,
                parent=self,
                **kwargs,
            )

        # Non-child requires intentionally use lookup_scope rather than the
        # module's identity scope so child modules can share their parent's
        # surrounding dependencies without extra code in the child class.
        kwargs.setdefault("scope", self.lookup_scope)
        return self.world.require(
            module_cls,
            *args,
            child=False,
            child_name=child_name,
            **kwargs,
        )

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

    def _binding_candidates(self, value: Any) -> List[str]:
        try:
            items = vars(value).items()
        except TypeError:
            return []
        names = []
        for name, child in items:
            if name.startswith("_"):
                continue
            if isinstance(child, (ResourceRef, ModuleBase)):
                names.append(name)
        return sorted(names)

    def _binding_help(
        self,
        *,
        missing: str,
        container: Any,
        path_prefix: str = "",
    ) -> str:
        candidates = self._binding_candidates(container)
        parts: List[str] = []
        suggestion = get_close_matches(missing, candidates, n=1, cutoff=0.6)
        if suggestion:
            guess = suggestion[0]
            if path_prefix:
                guess = f"{path_prefix}{guess}"
            parts.append(f"Did you mean '{guess}'?")
        if candidates:
            visible = [f"{path_prefix}{name}" if path_prefix else name for name in candidates]
            parts.append(f"Available names: {', '.join(visible)}.")
        return (" " + " ".join(parts)) if parts else ""

    def _resolve_bound_resource(
        self,
        attr: str,
        *,
        method_name: str,
        kind: str,
    ) -> ResourceRef[Any]:
        value: Any = self
        resolved_parts: List[str] = []
        for part in attr.split("."):
            resolved_parts.append(part)
            if not hasattr(value, part):
                resolved_attr = ".".join(resolved_parts)
                prefix = ".".join(resolved_parts[:-1])
                help_text = self._binding_help(
                    missing=part,
                    container=value,
                    path_prefix=f"{prefix}." if prefix else "",
                )
                raise AttributeError(
                    f"{self.__class__.__name__}.{method_name} references unknown "
                    f"{kind} resource '{resolved_attr}'.{help_text}"
                )
            value = getattr(value, part)
        if not isinstance(value, ResourceRef):
            help_text = self._binding_help(
                missing="",
                container=value,
                path_prefix=f"{attr}." if attr else "",
            )
            raise TypeError(
                f"{self.__class__.__name__}.{method_name} expected '{attr}' "
                f"to be a ResourceRef, got {type(value).__name__}.{help_text}"
            )
        return value

    def _iter_decorated_methods(
        self,
    ) -> Iterable[Tuple[str, Callable[..., Any], Any]]:
        seen: Set[str] = set()
        for cls in self.__class__.__mro__:
            for name, value in vars(cls).items():
                if name in seen:
                    continue
                seen.add(name)
                spec = get_producer_spec(value)
                if spec is None:
                    continue
                yield name, getattr(self, name), spec

    def bind_producers(self) -> None:
        for method_name, method, spec in self._iter_decorated_methods():
            if method_name in self._bound_producer_methods:
                raise RuntimeError(
                    f"{self.__class__.__name__}.{method_name} is already bound"
                )

            input_refs = {
                name: self._resolve_bound_resource(
                    name,
                    method_name=method_name,
                    kind="input",
                )
                for name in spec.inputs
            }
            output_refs: Dict[str, ResourceRef[Any]] = {}
            output_allocs: Dict[str, Any] = {}
            for output in spec.outputs:
                ref = self._resolve_bound_resource(
                    output.name,
                    method_name=method_name,
                    kind="output",
                )
                try:
                    self.reg.get(ref.name)
                except KeyError:
                    pass
                else:
                    raise RuntimeError(
                        f"{self.__class__.__name__}.{method_name} output "
                        f"'{output.name}' is already declared"
                    )
                output_refs[output.name] = ref
                output_allocs[output.name] = output.alloc

            producer = _DecoratedMethodProducer(
                module=self,
                method_name=method_name,
                bound_method=method,
                input_refs=input_refs,
                output_refs=output_refs,
                output_allocs=output_allocs,
                allow_none=spec.allow_none,
                ignore=spec.ignore,
            )
            deps = tuple(input_refs.values())
            for ref in output_refs.values():
                self.declare_resource(ref, deps=deps, producer=producer)
            self._bound_producer_methods.add(method_name)


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
        self._build_context_stack: List[_ModuleBuildContext] = []

    def _active_build_context(self) -> Optional[_ModuleBuildContext]:
        if not self._build_context_stack:
            return None
        return self._build_context_stack[-1]

    def module_dependencies(self) -> Dict[ModuleKey, Set[ModuleKey]]:
        return {k: set(v) for k, v in self._module_deps.items()}

    def _record_module_dep(self, parent: Optional[ModuleKey], child: ModuleKey) -> None:
        self._module_deps.setdefault(child, set())
        if parent is None or parent == child:
            return
        self._module_deps.setdefault(parent, set()).add(child)

    def _build_context_for_require(
        self,
        *,
        module_cls: Type[M],
        scope: str,
        child: bool,
        child_name: Optional[str],
        parent: Optional[ModuleBase],
    ) -> _ModuleBuildContext:
        """Normalize public require(...) arguments into identity/lookup scopes.

        Rules:
        - normal module:
          identity_scope = lookup_scope = explicit `scope`
        - child module:
          identity_scope = parent.qualify(child_name)
          lookup_scope = parent.lookup_scope
        """
        if child:
            if child_name is None:
                raise ValueError("child=True requires child_name")
            _validate_child_name(child_name)
            if scope:
                raise ValueError("scope cannot be combined with child=True")
            if parent is None:
                raise ValueError("child=True on World.require() requires parent")

            caller_key = self._building_stack[-1] if self._building_stack else None
            parent_key = caller_key or parent._module_key
            return _ModuleBuildContext(
                module_cls=module_cls,
                identity_scope=parent.qualify(child_name),
                lookup_scope=parent.lookup_scope,
                parent_key=parent_key,
            )

        if child_name is not None:
            raise ValueError("child_name requires child=True")
        if parent is not None:
            raise ValueError("parent requires child=True")
        return _ModuleBuildContext(
            module_cls=module_cls,
            identity_scope=scope,
            lookup_scope=scope,
        )

    def require(
        self,
        module_cls: Type[M],
        *args: Any,
        scope: str = "",
        child: bool = False,
        child_name: Optional[str] = None,
        parent: Optional[ModuleBase] = None,
        **kwargs: Any,
    ) -> M:
        """Build or reuse a module instance.

        Usage:
        - `world.require(Foo, scope="sim")`
          normal explicit scoped module
        - `world.require(Foo, child=True, child_name="solver", parent=parent_mod)`
          advanced child-module creation from code outside a ModuleBase method

        Notes:
        - `child=True` is only meaningful when there is an explicit parent
        - cache identity includes both identity_scope and lookup_scope because
          different lookup scopes can change which dependencies a module reuses
        """
        build_ctx = self._build_context_for_require(
            module_cls=module_cls,
            scope=scope,
            child=child,
            child_name=child_name,
            parent=parent,
        )
        args_key = _make_args_key(args, kwargs)
        key: ModuleKey = (
            build_ctx.identity_scope,
            build_ctx.lookup_scope,
            module_cls,
            args_key,
        )

        dep_parent = self._building_stack[-1] if self._building_stack else build_ctx.parent_key

        # Already built
        existing = self._modules.get(key)
        if existing is not None:
            self._record_module_dep(dep_parent, key)
            return cast(M, existing)

        # Cycle detection (dynamic deps are discovered by execution; cycles are fatal)
        if key in self._building_set:
            try:
                i = self._building_stack.index(key)
            except ValueError:
                i = 0
            cyc = self._building_stack[i:] + [key]

            cycle_str = " -> ".join(format_module_key(k) for k in cyc)
            raise RuntimeError(f"Module dependency cycle detected: {cycle_str}")

        modules_before = dict(self._modules)
        module_deps_before = {
            module_key: set(deps) for module_key, deps in self._module_deps.items()
        }
        resource_names_before = self.reg.declared_names()
        self._record_module_dep(dep_parent, key)

        # Build
        self._building_stack.append(key)
        self._building_set.add(key)
        self._build_context_stack.append(build_ctx)
        try:
            # ModuleBase.__init__ reads the active build context to learn its
            # lookup_scope without changing the public constructor contract.
            m = module_cls(self, *args, scope=build_ctx.identity_scope, **kwargs)
            m._module_key = key
            self._modules[key] = m
            return cast(M, m)
        except Exception:
            # Roll back anything declared during the failed build so retries
            # surface the original exception instead of duplicate declarations.
            self._modules = modules_before
            self._module_deps = module_deps_before
            self.reg.undeclare_many(
                self.reg.declared_names() - resource_names_before
            )
            raise
        finally:
            # Always unwind even if __init__ raises
            popped = self._building_stack.pop()
            self._building_set.remove(popped)
            self._build_context_stack.pop()
