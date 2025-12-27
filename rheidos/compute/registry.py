from typing import Tuple, Set, List, Any, Sequence, Optional, Iterable, Callable, TypeVar, Dict
from dataclasses import dataclass, field
import numpy as np
from .resource import Resource, ResourceSpec
from .typing import ResourceName

T = TypeVar("T")



class ProducerBase:
    """
    Producer contract:

      - self.outputs: tuple of resource names this producer updates
      - compute(reg): must reg.commit(...) or reg.bump(...) each output (after filling it)

    The registry decides when to run producers (via ensure()) based on dependency freshness.
    """

    outputs: Tuple[ResourceName, ...]

    def compute(self, reg: "Registry") -> None:
        raise NotImplementedError


@dataclass
class _EnsureCtx:
    ran: Set[ProducerBase] = field(default_factory=set)
    stack: List[ResourceName] = field(default_factory=list)


class Registry:
    def __init__(self) -> None:
        self._res: Dict[ResourceName, Resource] = {}

    # ---- declare ----

    def declare(
        self,
        name: ResourceName,
        *,
        buffer: Any = None,
        deps: Sequence[ResourceName] = (),
        producer: Optional[ProducerBase] = None,
        description: str = "",
        spec: Optional[ResourceSpec] = None,
    ) -> Resource:
        if name in self._res:
            raise KeyError(f"Resource already declared: {name}")
        r = Resource(
            name=name,
            buffer=buffer,
            deps=tuple(deps),
            producer=producer,
            description=description,
            spec=spec,
        )
        self._res[name] = r
        if buffer is not None:
            self._validate_buffer(r, buffer)
        return r

    def get(self, name: ResourceName) -> Resource:
        try:
            return self._res[name]
        except KeyError as e:
            raise KeyError(f"Unknown resource: {name}") from e

    # ---- mutation ----

    def read(self, name: ResourceName, *, ensure: bool = True) -> Any:
        if ensure:
            self.ensure(name)
        return self.get(name).buffer

    def set_buffer(self, name: ResourceName, buffer: Any, *, bump: bool = False, unsafe: bool = False) -> None:
        """
        Replace buffer, optionally bumping version.

        - bump=True           : validates (unless unsafe), sets buffer, bumps
        - bump=False (default): validates (unless unsafe), sets buffer WITHOUT bump (allocation-before-fill)
        """
        r = self.get(name)
        if not unsafe:
            self._validate_buffer(r, buffer)
        r.buffer = buffer
        if bump:
            self.bump(name, unsafe=unsafe)

    def commit(self, name: ResourceName, *, buffer: Any = None, unsafe: bool = False) -> None:
        """
        Replace buffer (optional) and mark resource fresh relative to current deps.

        Freedom mode: allowed for any resource.
        Validation: on by default; pass unsafe=True to bypass.
        """
        if buffer is not None:
            self.set_buffer(name, buffer, bump=False, unsafe=unsafe)
        self.bump(name, unsafe=unsafe)

    def bump(self, name: ResourceName, unsafe: bool = False) -> None:
        """
        Mark resource fresh relative to current deps.

        Freedom mode: allowed for any resource (even produced ones).
        Validation: on by default; pass unsafe=True to bypass.
        """
        r = self.get(name)
        if not unsafe:
            self._validate_buffer(r, r.buffer)
        r.version += 1
        r.dep_sig = self._current_dep_sig(r.deps)

    # ---- validation ----

    def _validate_buffer(self, r: Resource, buf: Any) -> None:
        spec = r.spec
        if spec is None:
            return

        if buf is None:
            if spec.allow_none:
                return
            raise TypeError(f"[{r.name}] buffer is None but allow_none=False")

        if spec.kind == "python":
            return

        if spec.kind == "numpy":
            if not isinstance(buf, np.ndarray):
                raise TypeError(f"[{r.name}] expected numpy ndarray, got {type(buf)}")
            if spec.dtype is not None and buf.dtype != np.dtype(spec.dtype):
                raise TypeError(f"[{r.name}] expected dtype {spec.dtype}, got {buf.dtype}")
            exp_shape = spec.shape
            if exp_shape is None and spec.shape_fn is not None:
                exp_shape = spec.shape_fn(self)
            if exp_shape is not None and tuple(buf.shape) != tuple(exp_shape):
                raise TypeError(f"[{r.name}] expected shape {exp_shape}, got {tuple(buf.shape)}")
            return

        if spec.kind == "taichi_field":
            # Best-effort checks for Taichi fields.
            if not hasattr(buf, "dtype") or not hasattr(buf, "shape"):
                raise TypeError(f"[{r.name}] expected Taichi field-like buffer, got {type(buf)}")

            if spec.dtype is not None:
                try:
                    if buf.dtype != spec.dtype:
                        raise TypeError(f"[{r.name}] expected dtype {spec.dtype}, got {buf.dtype}")
                except Exception as e:
                    raise TypeError(f"[{r.name}] could not validate dtype: {e}") from e

            exp_shape = spec.shape
            if exp_shape is None and spec.shape_fn is not None:
                exp_shape = spec.shape_fn(self)
            if exp_shape is not None:
                try:
                    if tuple(buf.shape) != tuple(exp_shape):
                        raise TypeError(f"[{r.name}] expected shape {exp_shape}, got {tuple(buf.shape)}")
                except Exception as e:
                    raise TypeError(f"[{r.name}] could not validate shape: {e}") from e

            if spec.lanes is not None:
                # Vector.field often exposes .n (lanes). If not available, skip.
                lanes_actual = getattr(buf, "n", None)
                if lanes_actual is not None and int(lanes_actual) != int(spec.lanes):
                    raise TypeError(f"[{r.name}] expected lanes {spec.lanes}, got {lanes_actual}")

            return

        raise ValueError(f"[{r.name}] Unknown spec.kind={spec.kind}")

    # ---- ensure ----

    def ensure(self, name: ResourceName) -> None:
        self._ensure(name, _EnsureCtx())

    def ensure_many(self, names: Iterable[ResourceName]) -> None:
        ctx = _EnsureCtx()
        for n in names:
            self._ensure(n, ctx)

    def _ensure(self, name: ResourceName, ctx: _EnsureCtx) -> None:
        if name in ctx.stack:
            cycle = " -> ".join(ctx.stack + [name])
            raise RuntimeError(f"Dependency cycle detected: {cycle}")

        r = self.get(name)

        # ensure() works for all: no producer => no-op.
        if r.producer is None:
            return

        # ensure deps first
        ctx.stack.append(name)
        for d in r.deps:
            self._ensure(d, ctx)
        ctx.stack.pop()

        if not self._is_stale(r):
            return

        p = r.producer

        if p not in ctx.ran:
            # Ensure deps of all outputs (safe for fused/multi-output producers)
            for out in p.outputs:
                out_r = self.get(out)
                for d in out_r.deps:
                    self._ensure(d, ctx)

            p.compute(self)
            ctx.ran.add(p)

            # Validate producer committed outputs
            for out in p.outputs:
                out_r = self.get(out)
                if out_r.producer is p and self._is_stale(out_r):
                    raise RuntimeError(
                        f"Producer {p.__class__.__name__} ran but '{out}' is still stale. "
                        f"Did you forget reg.commit()/reg.bump() for it?"
                    )

        if self._is_stale(self.get(name)):
            raise RuntimeError(f"Producer {p.__class__.__name__} ran but '{name}' is still stale.")

    def _is_stale(self, r: Resource) -> bool:
        if r.producer is None:
            return False
        if r.version == 0:
            return True
        return r.dep_sig != self._current_dep_sig(r.deps)

    def _current_dep_sig(self, deps: Sequence[ResourceName]) -> Tuple[Tuple[ResourceName, int], ...]:
        return tuple((d, self.get(d).version) for d in deps)

    # ---- debug ----

    def explain(self, name: ResourceName, depth: int = 4) -> str:
        lines: List[str] = []
        seen: Set[ResourceName] = set()

        def rec(n: ResourceName, lvl: int) -> None:
            if lvl > depth:
                return
            r = self.get(n)
            stale = self._is_stale(r)
            prod = r.producer.__class__.__name__ if r.producer else "None"
            spec = ""
            if r.spec is not None:
                spec = f" spec(kind={r.spec.kind}, dtype={r.spec.dtype}, lanes={r.spec.lanes})"
            lines.append(
                f"{'  '*lvl}- {n} v={r.version} producer={prod}{spec}" + (" STALE" if stale else "")
            )
            if n in seen:
                return
            seen.add(n)
            for d in r.deps:
                rec(d, lvl + 1)

        rec(name, 0)
        return "\n".join(lines)
