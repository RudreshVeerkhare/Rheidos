from difflib import get_close_matches
from dataclasses import dataclass, field, is_dataclass, fields
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from .resource import ResourceRef
from .resource_kinds import get_resource_kind
from .registry import ProducerBase

# =============================================================================
# Producers: typed IO dataclasses + one-phase wiring
# =============================================================================


def out_field(*, alloc: Optional[Callable[[Any, Any], Any]] = None) -> Any:
    meta = {"io": "out"}
    if alloc is not None:
        meta["alloc"] = alloc
    return field(metadata=meta)


IO = TypeVar("IO")
F = TypeVar("F", bound=Callable[..., Any])
OutputAlloc = Optional[Callable[[Any, Any], Any]]
_PRODUCER_SPEC_ATTR = "__rheidos_producer_spec__"


@dataclass(frozen=True)
class ProducerOutput:
    name: str
    alloc: OutputAlloc = None


def producer_output(name: str, *, alloc: OutputAlloc = None) -> ProducerOutput:
    return ProducerOutput(name=name, alloc=alloc)


@dataclass(frozen=True)
class ProducerSpec:
    inputs: Tuple[str, ...]
    outputs: Tuple[ProducerOutput, ...]
    allow_none: Tuple[str, ...] = ()
    ignore: Tuple[str, ...] = ()


def _normalize_output(output: Union[str, ProducerOutput]) -> ProducerOutput:
    if isinstance(output, str):
        return ProducerOutput(name=output)
    if isinstance(output, ProducerOutput):
        return output
    raise TypeError("producer outputs must be str or ProducerOutput")


def producer(
    *,
    inputs: Sequence[str],
    outputs: Sequence[Union[str, ProducerOutput]],
    allow_none: Iterable[str] = (),
    ignore: Iterable[str] = (),
) -> Callable[[F], F]:
    spec = ProducerSpec(
        inputs=tuple(inputs),
        outputs=tuple(_normalize_output(output) for output in outputs),
        allow_none=tuple(allow_none),
        ignore=tuple(ignore),
    )

    def decorator(fn: F) -> F:
        setattr(fn, _PRODUCER_SPEC_ATTR, spec)
        return fn

    return decorator


def get_producer_spec(value: Any) -> Optional[ProducerSpec]:
    fn = getattr(value, "__func__", value)
    return getattr(fn, _PRODUCER_SPEC_ATTR, None)


def _format_name_help(
    missing: str,
    available: Iterable[str],
    *,
    suggestion_prefix: str = "",
) -> str:
    names = sorted(set(available))
    parts: List[str] = []
    suggestion = get_close_matches(missing, names, n=1, cutoff=0.6)
    if not suggestion and len(names) == 1:
        suggestion = names
    if suggestion:
        guess = suggestion[0]
        if suggestion_prefix:
            guess = f"{suggestion_prefix}{guess}"
        parts.append(f"Did you mean '{guess}'?")
    if names:
        parts.append(f"Available names: {', '.join(names)}.")
    return (" " + " ".join(parts)) if parts else ""


def _require_input_refs(
    owner: str,
    refs: Mapping[str, ResourceRef[Any]],
    *,
    allow_none: Iterable[str] = (),
    ignore: Iterable[str] = (),
) -> Dict[str, ResourceRef[Any]]:
    allow = set(allow_none)
    skip = set(ignore)
    missing: List[str] = []
    resolved: Dict[str, ResourceRef[Any]] = {}
    for name, ref in refs.items():
        if name in skip:
            continue
        buf = ref.peek()
        if buf is None and name not in allow:
            missing.append(name)
        resolved[name] = ref
    if missing:
        miss_str = ", ".join(missing)
        raise RuntimeError(f"{owner} missing required inputs: {miss_str}")
    return resolved


def _require_output_shape(
    owner: str,
    reg: Any,
    name: str,
    ref: ResourceRef[Any],
    alloc: OutputAlloc,
) -> None:
    if alloc is not None:
        return
    spec = ref.spec
    if spec is None:
        raise RuntimeError(
            f"{owner} output '{name}' has no ResourceSpec; "
            "set spec with shape/shape_fn or provide a custom alloc."
        )
    adapter = get_resource_kind(spec.kind)
    if not adapter.requires_shape:
        return
    shape = adapter.resolve_shape(reg, spec)
    if shape is None:
        raise RuntimeError(
            f"{owner} output '{name}' is missing shape; "
            "set ResourceSpec.shape or shape_fn, or provide a custom alloc."
        )


def _allocate_output(
    reg: Any,
    ref: ResourceRef[Any],
    *,
    alloc: OutputAlloc,
    alloc_target: Any,
) -> Any:
    if alloc is not None:
        return alloc(reg, alloc_target)
    spec = ref.spec
    if spec is None:
        return None
    adapter = get_resource_kind(spec.kind)
    shape = adapter.resolve_shape(reg, spec)
    if shape is None:
        return None
    return adapter.allocate(reg, spec, shape)


def _ensure_output_refs(
    owner: str,
    reg: Any,
    refs: Mapping[str, ResourceRef[Any]],
    *,
    allocs: Mapping[str, OutputAlloc],
    alloc_target: Any,
    strict: bool = True,
    realloc: bool = True,
    require_shape: bool = True,
) -> Dict[str, ResourceRef[Any]]:
    missing: List[str] = []
    resolved: Dict[str, ResourceRef[Any]] = {}
    for name, ref in refs.items():
        buf = ref.peek()
        needs_alloc = buf is None
        if not needs_alloc and realloc:
            spec = ref.spec
            if spec is not None and not reg.matches_spec(ref.name, buf):
                needs_alloc = True
        if needs_alloc:
            alloc = allocs.get(name)
            if require_shape:
                _require_output_shape(owner, reg, name, ref, alloc)
            buf = _allocate_output(reg, ref, alloc=alloc, alloc_target=alloc_target)
            if buf is not None:
                ref.set_buffer(buf, bump=False)
        if ref.peek() is None:
            missing.append(name)
        resolved[name] = ref
    if missing and strict:
        miss_str = ", ".join(missing)
        raise RuntimeError(f"{owner} outputs are unset: {miss_str}")
    return resolved


class ProducerResourceNamespace:
    def __init__(
        self,
        refs: Mapping[str, ResourceRef[Any]],
        *,
        label: str = "resources",
    ) -> None:
        self._refs = dict(refs)
        self._label = label
        self._children: Dict[str, "ProducerResourceNamespace"] = {}
        nested: Dict[str, Dict[str, ResourceRef[Any]]] = {}
        for name, ref in self._refs.items():
            if "." not in name:
                continue
            head, tail = name.split(".", 1)
            nested.setdefault(head, {})[tail] = ref
        for head, child_refs in nested.items():
            self._children[head] = ProducerResourceNamespace(
                child_refs,
                label=f"{self._label}.{head}",
            )

    def _available_names(self) -> List[str]:
        return sorted(set(self._refs) | set(self._children))

    def __getattr__(self, name: str) -> Any:
        child = self._children.get(name)
        if child is not None:
            return child
        try:
            return self._refs[name]
        except KeyError as e:
            help_text = _format_name_help(name, self._available_names())
            raise AttributeError(
                f"{self._label} has no resource named '{name}'.{help_text}"
            ) from e

    def __getitem__(self, name: str) -> ResourceRef[Any]:
        if "." in name:
            head, tail = name.split(".", 1)
            child = self._children.get(head)
            if child is None:
                help_text = _format_name_help(head, self._available_names())
                raise KeyError(
                    f"{self._label} has no resource named '{name}'.{help_text}"
                )
            return child[tail]
        try:
            return self._refs[name]
        except KeyError as e:
            help_text = _format_name_help(name, self._available_names())
            raise KeyError(
                f"{self._label} has no resource named '{name}'.{help_text}"
            ) from e

    def items(self) -> Iterable[Tuple[str, ResourceRef[Any]]]:
        return self._refs.items()

    def as_dict(self) -> Dict[str, ResourceRef[Any]]:
        return dict(self._refs)


class ProducerContext:
    reg: Any
    inputs: ProducerResourceNamespace
    outputs: ProducerResourceNamespace

    def __init__(
        self,
        *,
        reg: Any,
        owner: str,
        input_refs: Mapping[str, ResourceRef[Any]],
        output_refs: Mapping[str, ResourceRef[Any]],
        output_allocs: Mapping[str, OutputAlloc],
        allow_none: Iterable[str] = (),
        ignore: Iterable[str] = (),
    ) -> None:
        self.reg = reg
        self._owner = owner
        self._input_refs = dict(input_refs)
        self._output_refs = dict(output_refs)
        self._output_allocs = dict(output_allocs)
        self._default_allow_none = tuple(allow_none)
        self._default_ignore = tuple(ignore)
        self.inputs = ProducerResourceNamespace(
            self._input_refs,
            label=f"{owner} inputs",
        )
        self.outputs = ProducerResourceNamespace(
            self._output_refs,
            label=f"{owner} outputs",
        )

    def require_inputs(
        self,
        *,
        allow_none: Iterable[str] = (),
        ignore: Iterable[str] = (),
    ) -> Dict[str, ResourceRef[Any]]:
        merged_allow = tuple(dict.fromkeys((*self._default_allow_none, *allow_none)))
        merged_ignore = tuple(dict.fromkeys((*self._default_ignore, *ignore)))
        return _require_input_refs(
            self._owner,
            self._input_refs,
            allow_none=merged_allow,
            ignore=merged_ignore,
        )

    def ensure_outputs(
        self,
        *,
        strict: bool = True,
        realloc: bool = True,
        require_shape: bool = True,
    ) -> Dict[str, ResourceRef[Any]]:
        return _ensure_output_refs(
            self._owner,
            self.reg,
            self._output_refs,
            allocs=self._output_allocs,
            alloc_target=self,
            strict=strict,
            realloc=realloc,
            require_shape=require_shape,
        )

    def commit(self, **buffers: Any) -> None:
        for name, buffer in buffers.items():
            ref = self._output_refs.get(name)
            if ref is None:
                help_text = _format_name_help(name, self._output_refs)
                raise KeyError(
                    f"{self._owner} has no output named '{name}'.{help_text}"
                )
            self.reg.commit(ref.name, buffer=buffer)


class _DecoratedMethodProducer(ProducerBase):
    def __init__(
        self,
        *,
        module: Any,
        method_name: str,
        bound_method: Callable[[ProducerContext], Any],
        input_refs: Mapping[str, ResourceRef[Any]],
        output_refs: Mapping[str, ResourceRef[Any]],
        output_allocs: Mapping[str, OutputAlloc],
        allow_none: Iterable[str] = (),
        ignore: Iterable[str] = (),
    ) -> None:
        self._module = module
        self._method_name = method_name
        self._bound_method = bound_method
        self._input_refs = dict(input_refs)
        self._output_refs = dict(output_refs)
        self._output_allocs = dict(output_allocs)
        self._allow_none = tuple(allow_none)
        self._ignore = tuple(ignore)
        self.inputs = tuple(ref.name for ref in self._input_refs.values())
        self.outputs = tuple(ref.name for ref in self._output_refs.values())

    def debug_name(self) -> str:
        cls = self._module.__class__
        return f"{cls.__module__}.{cls.__qualname__}.{self._method_name}"

    def compute(self, reg: Any) -> None:
        ctx = ProducerContext(
            reg=reg,
            owner=self.debug_name(),
            input_refs=self._input_refs,
            output_refs=self._output_refs,
            output_allocs=self._output_allocs,
            allow_none=self._allow_none,
            ignore=self._ignore,
        )
        ctx.require_inputs()
        self._bound_method(ctx)


class WiredProducer(ProducerBase, Generic[IO]):
    """
    A producer that is *wired* to concrete ResourceRef IO at construction time.

    Contract:
      - IO must be a dataclass.
      - Outputs are inferred from IO fields marked with out_field().
    """

    IO_TYPE: Optional[type] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "IO_TYPE", None) is not None:
            return
        io_type = None
        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base)
            if origin is None:
                continue
            if origin is WiredProducer or (
                isinstance(origin, type) and issubclass(origin, WiredProducer)
            ):
                args = get_args(base)
                if args:
                    io_type = args[0]
                    break
        if isinstance(io_type, type):
            cls.IO_TYPE = io_type

    def __init__(self, io: Optional[IO] = None, **kwargs: Any) -> None:
        if io is None:
            io_type = getattr(self, "IO_TYPE", None)
            if io_type is None:
                raise TypeError(
                    f"{self.__class__.__name__} requires io or IO_TYPE for kwargs wiring"
                )
            io = io_type(**kwargs)
        elif kwargs:
            raise TypeError(
                f"{self.__class__.__name__} expects either io or kwargs, not both"
            )

        if not is_dataclass(io):
            raise TypeError(f"{self.__class__.__name__} expects a dataclass IO object")
        self.io: IO = io

        input_refs: Dict[str, ResourceRef[Any]] = {}
        output_refs: Dict[str, ResourceRef[Any]] = {}
        output_allocs: Dict[str, OutputAlloc] = {}
        for f in fields(io):
            ref = getattr(io, f.name)
            if f.metadata.get("io") == "out":
                if not isinstance(ref, ResourceRef):
                    raise TypeError(
                        f"IO field '{f.name}' marked out but is not a ResourceRef"
                    )
                output_refs[f.name] = ref
                output_allocs[f.name] = f.metadata.get("alloc")
                continue
            if isinstance(ref, ResourceRef):
                input_refs[f.name] = ref
        self._input_refs = input_refs
        self._output_refs = output_refs
        self._output_allocs = output_allocs
        self.inputs = tuple(ref.name for ref in input_refs.values())
        self.outputs = tuple(ref.name for ref in output_refs.values())
        self.setup()

    def setup(self) -> None:
        """Hook for post-wiring initialization."""
        pass

    def _iter_io_fields(self, *, kind: str) -> Iterable[Tuple[Any, Any]]:
        for f in fields(self.io):
            is_out = f.metadata.get("io") == "out"
            if kind == "out" and not is_out:
                continue
            if kind == "in" and is_out:
                continue
            yield f, getattr(self.io, f.name)

    def input_refs(self) -> Dict[str, ResourceRef[Any]]:
        return dict(self._input_refs)

    def output_refs(self) -> Dict[str, ResourceRef[Any]]:
        return dict(self._output_refs)

    def require_inputs(
        self,
        *,
        allow_none: Iterable[str] = (),
        ignore: Iterable[str] = (),
    ) -> Dict[str, ResourceRef[Any]]:
        """
        Returns a dict of input ResourceRef objects.

        Validates that required inputs have buffers set (using peek).
        Users call .get() or .peek() on the returned refs to access buffers.

        Args:
            allow_none: Field names that are allowed to be None
            ignore: Field names to skip validation and exclude from result
        """
        return _require_input_refs(
            self.__class__.__name__,
            self._input_refs,
            allow_none=allow_none,
            ignore=ignore,
        )

    def ensure_outputs(
        self,
        reg: Any,
        *,
        strict: bool = True,
        realloc: bool = True,
        require_shape: bool = True,
    ) -> Dict[str, ResourceRef[Any]]:
        """
        Returns a dict of output ResourceRef objects.

        Allocates missing output buffers using ResourceSpec or custom allocators.
        Users call .get() or .peek() on the returned refs to access buffers.

        Args:
            reg: Registry for resource kind adapters and allocation
            strict: Raise error if any outputs remain None after allocation
            realloc: Reallocate if buffer doesn't match spec
            require_shape: Require shape/shape_fn in spec for allocation
        """
        return _ensure_output_refs(
            self.__class__.__name__,
            reg,
            self._output_refs,
            allocs=self._output_allocs,
            alloc_target=self.io,
            strict=strict,
            realloc=realloc,
            require_shape=require_shape,
        )
