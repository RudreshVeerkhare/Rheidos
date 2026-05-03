"""Composable SOP verb helpers for Houdini-backed modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, TYPE_CHECKING

import numpy as np

from rheidos.compute import ModuleBase, World

if TYPE_CHECKING:
    import hou


class SopVerbUnavailableError(RuntimeError):
    """Raised when a SOP node cannot be represented as a verb."""


class SopFunctionSetupError(RuntimeError):
    """Raised when a SOP function module is used before it is configured."""


def _get_hou() -> "hou":
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs outside Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc
    return hou


def _resolve_node(node_or_path: Any) -> Any:
    if isinstance(node_or_path, str):
        hou = _get_hou()
        node = hou.node(node_or_path)
        if node is None:
            raise ValueError(f"SOP node does not exist: {node_or_path}")
        return node
    if node_or_path is None:
        raise ValueError("SOP node is required")
    return node_or_path


def _node_path(node: Any) -> str:
    path = getattr(node, "path", None)
    if callable(path):
        try:
            return str(path())
        except Exception:
            pass
    return repr(node)


def _numeric_float_type(hou: Any, dtype: Any) -> Any:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
        return hou.numericData.Float64
    if dtype == np.dtype(np.float32):
        return hou.numericData.Float32
    raise TypeError(f"Unsupported float dtype for Houdini binary IO: {dtype}")


def _numeric_int_type(hou: Any, dtype: Any) -> Any:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.int64):
        return hou.numericData.Int64
    if dtype == np.dtype(np.int32):
        return hou.numericData.Int32
    if dtype == np.dtype(np.int16):
        return hou.numericData.Int16
    if dtype == np.dtype(np.int8):
        return hou.numericData.Int8
    raise TypeError(f"Unsupported int dtype for Houdini binary IO: {dtype}")


def _increment_all_data_ids(geo: Any) -> None:
    increment = getattr(geo, "incrementAllDataIds", None)
    if callable(increment):
        increment()


def points_np_to_geo(points: Any, *, dtype: Any = np.float64) -> "hou.Geometry":
    """Build writable Houdini point geometry from an ``(N, 3)`` NumPy-like array."""

    hou = _get_hou()
    points_np = np.asarray(points, dtype=dtype)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points_np.shape}")
    points_np = np.ascontiguousarray(points_np)

    geo = hou.Geometry()
    zero = (0.0, 0.0, 0.0)
    geo.createPoints((zero,) * int(points_np.shape[0]))
    geo.setPointFloatAttribValuesFromString(
        "P",
        points_np,
        _numeric_float_type(hou, points_np.dtype),
    )
    _increment_all_data_ids(geo)
    return geo


def tri_mesh_np_to_geo(
    vertices: Any,
    faces: Any,
    *,
    dtype: Any = np.float64,
) -> "hou.Geometry":
    """Build Houdini polygon geometry from vertices ``(N, 3)`` and faces ``(M, 3)``."""

    vertices_np = np.asarray(vertices, dtype=dtype)
    faces_np = np.asarray(faces, dtype=np.int64)
    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices_np.shape}")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {faces_np.shape}")

    geo = points_np_to_geo(vertices_np, dtype=dtype)
    geo.createPolygons(tuple(tuple(int(v) for v in row) for row in faces_np))
    _increment_all_data_ids(geo)
    return geo


def point_attrib_to_numpy(
    geo: "hou.Geometry",
    name: str,
    *,
    dtype: Any,
    components: Optional[int] = None,
) -> np.ndarray:
    """Read a numeric point attribute using Houdini's binary bulk APIs."""

    hou = _get_hou()
    dtype_np = np.dtype(dtype)
    attrib = geo.findPointAttrib(name)
    if attrib is None:
        raise KeyError(f"Missing point attribute '{name}'")
    tuple_size = int(components if components is not None else attrib.size())
    if tuple_size < 1:
        raise ValueError("components must be >= 1")

    if dtype_np.kind == "f":
        data = geo.pointFloatAttribValuesAsString(
            name,
            _numeric_float_type(hou, dtype_np),
        )
    elif dtype_np.kind in ("i", "u", "b"):
        data = geo.pointIntAttribValuesAsString(
            name,
            _numeric_int_type(hou, dtype_np),
        )
    else:
        raise TypeError(f"Unsupported numeric dtype: {dtype_np}")

    values = np.frombuffer(data, dtype=dtype_np)
    point_count = int(geo.pointCount())
    expected = point_count * tuple_size
    if values.size != expected:
        raise ValueError(
            f"point attribute '{name}' expected {expected} values, got {values.size}"
        )
    if tuple_size == 1:
        return values.reshape((point_count,))
    return values.reshape((point_count, tuple_size))


class SopInputProvider(Protocol):
    """Protocol for objects that provide a single verb input geometry."""

    def resolve(
        self,
        module: "SopFunctionModule",
        *,
        verb_input: int,
        call: "SopCall",
    ) -> Optional["hou.Geometry"]:
        ...


@dataclass(frozen=True)
class CallGeo:
    """Use geometry passed to ``SopFunctionModule.run(..., slot=geo)``."""

    slot: str

    def resolve(
        self,
        module: "SopFunctionModule",
        *,
        verb_input: int,
        call: "SopCall",
    ) -> Optional["hou.Geometry"]:
        del module, verb_input
        try:
            return call.values[self.slot]
        except KeyError as exc:
            raise KeyError(
                f"Missing call geometry slot '{self.slot}' for SOP input"
            ) from exc


@dataclass(frozen=True)
class StaticGeo:
    """Use an already-owned Houdini geometry object."""

    geo: Optional["hou.Geometry"]

    def resolve(
        self,
        module: "SopFunctionModule",
        *,
        verb_input: int,
        call: "SopCall",
    ) -> Optional["hou.Geometry"]:
        del module, verb_input, call
        return self.geo


@dataclass(frozen=True)
class CtxInputGeo:
    """Use a geometry input from the module's last ``setup(ctx)`` call.

    ``freeze=False`` preserves the live input geometry reference and avoids
    framework-level copies. ``freeze=True`` explicitly detaches the geometry.
    ``cache='cook'`` resolves once per setup/cook, while ``cache='session'``
    keeps the first resolved geometry until the module is reset or reconfigured.
    """

    index: int
    freeze: bool = False
    cache: str = "cook"

    def __post_init__(self) -> None:
        if self.cache not in {"none", "cook", "session"}:
            raise ValueError("cache must be one of: 'none', 'cook', 'session'")

    def resolve(
        self,
        module: "SopFunctionModule",
        *,
        verb_input: int,
        call: "SopCall",
    ) -> Optional["hou.Geometry"]:
        del call
        cache_key = (verb_input, self)
        if self.cache == "cook" and cache_key in module._cook_geo_cache:
            return module._cook_geo_cache[cache_key]
        if self.cache == "session" and cache_key in module._session_geo_cache:
            return module._session_geo_cache[cache_key]

        ctx = module.require_setup_context()
        geo = ctx.input_geo(self.index)
        if geo is not None and self.freeze:
            geo = geo.freeze()

        if self.cache == "cook":
            module._cook_geo_cache[cache_key] = geo
        elif self.cache == "session":
            module._session_geo_cache[cache_key] = geo
        return geo


@dataclass
class SopCall:
    """Mutable call state passed through ``SopFunctionModule`` hooks."""

    args: tuple[Any, ...] = ()
    values: dict[str, Any] = field(default_factory=dict)
    parms: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    time: Optional[float] = None
    add_time_dep: bool = False


class SopVerbRunner:
    """Low-level wrapper around a single ``hou.SopVerb``."""

    def __init__(self, node_or_path: Any) -> None:
        self.node = _resolve_node(node_or_path)
        has_verb = getattr(self.node, "hasVerb", None)
        if callable(has_verb) and not bool(has_verb()):
            raise SopVerbUnavailableError(
                f"Node {_node_path(self.node)} does not expose a SOP verb"
            )
        verb_fn = getattr(self.node, "verb", None)
        if not callable(verb_fn):
            raise SopVerbUnavailableError(
                f"Node {_node_path(self.node)} does not expose a SOP verb"
            )
        self.verb = verb_fn()
        if self.verb is None:
            raise SopVerbUnavailableError(
                f"Node {_node_path(self.node)} does not expose a SOP verb"
            )

    def execute(
        self,
        inputs: Sequence[Optional["hou.Geometry"]],
        *,
        parms: Optional[Mapping[str, Any]] = None,
        time: Optional[float] = None,
        add_time_dep: bool = False,
    ) -> "hou.Geometry":
        hou = _get_hou()
        input_list = list(inputs)
        min_inputs_fn = getattr(self.verb, "minNumInputs", None)
        if callable(min_inputs_fn):
            min_inputs = int(min_inputs_fn())
            if len(input_list) < min_inputs:
                raise ValueError(
                    f"SOP verb for {_node_path(self.node)} requires at least "
                    f"{min_inputs} inputs, got {len(input_list)}"
                )

        if time is None:
            self.verb.loadParmsFromNode(self.node)
        else:
            self.verb.loadParmsFromNodeAtTime(self.node, float(time))
        if parms:
            self.verb.setParms(dict(parms))

        out_geo = hou.Geometry()
        if time is None:
            self.verb.execute(out_geo, input_list)
        else:
            self.verb.executeAtTime(
                out_geo,
                input_list,
                float(time),
                bool(add_time_dep),
            )
        return out_geo


class SopFunctionModule(ModuleBase):
    """Base class for composable ModuleBase-backed SOP verb functions."""

    NAME = "SopFunctionModule"
    SOP_NODE_PATH: Optional[str] = None
    SOP_INPUTS: Mapping[int, SopInputProvider] = {}

    def __init__(
        self,
        world: World,
        *,
        node_path: Optional[str] = None,
        sop_inputs: Optional[Mapping[int, SopInputProvider]] = None,
        default_parms: Optional[Mapping[str, Any]] = None,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        self.node_path = node_path or self.SOP_NODE_PATH
        if not self.node_path:
            raise ValueError(
                f"{self.__class__.__name__} requires node_path or SOP_NODE_PATH"
            )
        self.sop_input_providers = dict(
            self.SOP_INPUTS if sop_inputs is None else sop_inputs
        )
        self.default_parms = dict(default_parms or {})
        self._ctx: Any = None
        self._runner: Optional[SopVerbRunner] = None
        self._cook_geo_cache: dict[Any, Optional["hou.Geometry"]] = {}
        self._session_geo_cache: dict[Any, Optional["hou.Geometry"]] = {}

    def configure(
        self,
        *,
        node_path: Optional[str] = None,
        sop_inputs: Optional[Mapping[int, SopInputProvider]] = None,
        default_parms: Optional[Mapping[str, Any]] = None,
        clear_session_cache: bool = True,
    ) -> None:
        """Reconfigure this module without changing its composition identity."""

        if node_path is not None and node_path != self.node_path:
            self.node_path = node_path
            self._runner = None
        if sop_inputs is not None:
            self.sop_input_providers = dict(sop_inputs)
        if default_parms is not None:
            self.default_parms = dict(default_parms)
        self._cook_geo_cache.clear()
        if clear_session_cache:
            self._session_geo_cache.clear()

    def setup(self, ctx: Any) -> "SopFunctionModule":
        """Attach the current Houdini cook context for provider resolution."""

        self._ctx = ctx
        self._cook_geo_cache.clear()
        return self

    def require_setup_context(self) -> Any:
        if self._ctx is None:
            raise SopFunctionSetupError(
                f"{self.__class__.__name__}.setup(ctx) must be called before "
                "resolving CookContext-backed SOP inputs"
            )
        return self._ctx

    def _get_runner(self) -> SopVerbRunner:
        if self._runner is None:
            self._runner = SopVerbRunner(self.node_path)
        return self._runner

    def preprocess(self, call: SopCall) -> SopCall:
        """Hook for subclasses to normalize call values before input assembly."""

        return call

    def resolve_sop_inputs(self, call: SopCall) -> list[Optional["hou.Geometry"]]:
        if not self.sop_input_providers:
            return []
        max_input = max(self.sop_input_providers)
        inputs: list[Optional["hou.Geometry"]] = [None] * (max_input + 1)
        for verb_input, provider in sorted(self.sop_input_providers.items()):
            if verb_input < 0:
                raise ValueError("SOP input indices must be >= 0")
            inputs[verb_input] = provider.resolve(
                self,
                verb_input=verb_input,
                call=call,
            )
        return inputs

    def sop_inputs(self, call: SopCall) -> Sequence[Optional["hou.Geometry"]]:
        """Hook for subclasses that need custom input assembly."""

        return self.resolve_sop_inputs(call)

    def postprocess(self, out_geo: "hou.Geometry", meta: Mapping[str, Any]) -> Any:
        """Hook for subclasses to return domain objects instead of raw geometry."""

        del meta
        return out_geo

    def run(
        self,
        *args: Any,
        parms: Optional[Mapping[str, Any]] = None,
        time: Optional[float] = None,
        add_time_dep: bool = False,
        **kwargs: Any,
    ) -> Any:
        call = SopCall(
            args=tuple(args),
            values=dict(kwargs),
            parms={**self.default_parms, **dict(parms or {})},
            time=time,
            add_time_dep=bool(add_time_dep),
        )
        call = self.preprocess(call)
        inputs = self.sop_inputs(call)
        out_geo = self._get_runner().execute(
            inputs,
            parms=call.parms,
            time=call.time,
            add_time_dep=call.add_time_dep,
        )
        return self.postprocess(out_geo, call.meta)

    @staticmethod
    def points_to_geo(points: Any, *, dtype: Any = np.float64) -> "hou.Geometry":
        return points_np_to_geo(points, dtype=dtype)

    @staticmethod
    def tri_mesh_to_geo(
        vertices: Any,
        faces: Any,
        *,
        dtype: Any = np.float64,
    ) -> "hou.Geometry":
        return tri_mesh_np_to_geo(vertices, faces, dtype=dtype)


__all__ = [
    "CallGeo",
    "CtxInputGeo",
    "SopCall",
    "SopFunctionModule",
    "SopFunctionSetupError",
    "SopInputProvider",
    "SopVerbRunner",
    "SopVerbUnavailableError",
    "StaticGeo",
    "point_attrib_to_numpy",
    "points_np_to_geo",
    "tri_mesh_np_to_geo",
]
