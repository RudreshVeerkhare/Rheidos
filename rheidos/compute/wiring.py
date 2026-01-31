from dataclasses import field, is_dataclass, fields
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
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

        ins: List[str] = []
        outs: List[str] = []
        for f in fields(io):
            ref = getattr(io, f.name)
            if f.metadata.get("io") == "out":
                if not isinstance(ref, ResourceRef):
                    raise TypeError(
                        f"IO field '{f.name}' marked out but is not a ResourceRef"
                    )
                outs.append(ref.name)
                continue
            if isinstance(ref, ResourceRef):
                ins.append(ref.name)
        self.inputs = tuple(ins)
        self.outputs = tuple(outs)
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
        refs: Dict[str, ResourceRef[Any]] = {}
        for f, ref in self._iter_io_fields(kind="in"):
            if isinstance(ref, ResourceRef):
                refs[f.name] = ref
        return refs

    def output_refs(self) -> Dict[str, ResourceRef[Any]]:
        refs: Dict[str, ResourceRef[Any]] = {}
        for f, ref in self._iter_io_fields(kind="out"):
            if not isinstance(ref, ResourceRef):
                raise TypeError(
                    f"IO field '{f.name}' marked out but is not a ResourceRef"
                )
            refs[f.name] = ref
        return refs

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
        allow = set(allow_none)
        skip = set(ignore)
        missing: List[str] = []
        refs: Dict[str, ResourceRef[Any]] = {}
        for f, ref in self._iter_io_fields(kind="in"):
            if f.name in skip:
                continue
            if not isinstance(ref, ResourceRef):
                continue
            # Validate buffer exists without affecting dependencies
            buf = ref.peek()
            if buf is None and f.name not in allow:
                missing.append(f.name)
            refs[f.name] = ref
        if missing:
            miss_str = ", ".join(missing)
            raise RuntimeError(
                f"{self.__class__.__name__} missing required inputs: {miss_str}"
            )
        return refs

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
        missing: List[str] = []
        refs: Dict[str, ResourceRef[Any]] = {}
        for f, ref in self._iter_io_fields(kind="out"):
            if not isinstance(ref, ResourceRef):
                raise TypeError(
                    f"IO field '{f.name}' marked out but is not a ResourceRef"
                )
            buf = ref.peek()
            needs_alloc = buf is None
            if not needs_alloc and realloc:
                spec = ref.spec
                if spec is not None and not reg.matches_spec(ref.name, buf):
                    needs_alloc = True
            if needs_alloc:
                if require_shape:
                    self._require_output_shape(reg, f, ref)
                buf = self._allocate_output(reg, f, ref)
                if buf is not None:
                    ref.set_buffer(buf, bump=False)
            buf = ref.peek()
            if buf is None:
                missing.append(f.name)
            refs[f.name] = ref
        if missing and strict:
            miss_str = ", ".join(missing)
            raise RuntimeError(
                f"{self.__class__.__name__} outputs are unset: {miss_str}"
            )
        return refs

    def _require_output_shape(self, reg: Any, f: Any, ref: ResourceRef[Any]) -> None:
        if f.metadata.get("alloc") is not None:
            return
        spec = ref.spec
        if spec is None:
            raise RuntimeError(
                f"{self.__class__.__name__} output '{f.name}' has no ResourceSpec; "
                "set spec with shape/shape_fn or provide out_field(alloc=...)."
            )
        adapter = get_resource_kind(spec.kind)
        if not adapter.requires_shape:
            return
        shape = adapter.resolve_shape(reg, spec)
        if shape is None:
            raise RuntimeError(
                f"{self.__class__.__name__} output '{f.name}' is missing shape; "
                "set ResourceSpec.shape or shape_fn, or provide out_field(alloc=...)."
            )

    def _allocate_output(self, reg: Any, f: Any, ref: ResourceRef[Any]) -> Any:
        alloc = f.metadata.get("alloc")
        if alloc is not None:
            return alloc(reg, self.io)
        spec = ref.spec
        if spec is None:
            return None
        adapter = get_resource_kind(spec.kind)
        shape = adapter.resolve_shape(reg, spec)
        if shape is None:
            return None
        return adapter.allocate(reg, spec, shape)
