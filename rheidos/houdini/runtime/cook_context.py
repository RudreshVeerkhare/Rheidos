"""CookContext helpers for Houdini compute integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np

from rheidos.compute.resource import ResourceSpec
from rheidos.compute.world import World

from ..geo.adapter import GeometryIO
from ..geo.schema import GeometrySchema, OWNER_POINT
from .session import WorldSession

if TYPE_CHECKING:
    import hou
    from .session import AccessMode, SessionAccess


def _get_hou() -> "hou":
    try:
        import hou  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs in Houdini
        raise RuntimeError("Houdini 'hou' module not available") from exc
    return hou


def _resource_exists(reg: Any, name: str) -> bool:
    try:
        reg.get(name)
    except KeyError:
        return False
    return True


def _spec_for_value(value: Any) -> ResourceSpec:
    if isinstance(value, np.ndarray):
        return ResourceSpec(kind="numpy", dtype=value.dtype)
    return ResourceSpec(kind="python")


def _make_read_only_io(geo: "hou.Geometry") -> GeometryIO:
    io = GeometryIO(geo)
    # Prevent accidental writes to input geometry.
    io.geo_out = None
    return io


@dataclass
class CookContext:
    node: "hou.Node"
    frame: float
    time: float
    dt: float
    substep: int
    is_solver: bool
    session: WorldSession
    geo_in: "hou.Geometry"
    geo_out: "hou.Geometry"
    io: GeometryIO
    geo_inputs: tuple[Optional["hou.Geometry"], ...] = ()
    io_inputs: tuple[Optional[GeometryIO], ...] = ()
    schema: Optional[GeometrySchema] = None

    def world(self) -> World:
        if self.session.world is None:
            self.session.world = World()
        return self.session.world

    def clear_cache(self) -> None:
        self.io.clear_cache()
        for io in self.io_inputs:
            if io is None or io is self.io:
                continue
            io.clear_cache()

    def describe(self, owner: Optional[str] = None) -> GeometrySchema:
        schema = self.io.describe(owner=owner)
        if owner is None:
            self.schema = schema
        return schema

    def input_geo(self, index: int, *, required: bool = True) -> Optional["hou.Geometry"]:
        inputs = self.geo_inputs or (self.geo_in,)
        if index < 0 or index >= len(inputs):
            raise IndexError(f"Input index {index} out of range (0-{len(inputs) - 1})")
        geo = inputs[index]
        if geo is None and required:
            raise RuntimeError(f"Input geometry {index} is not connected.")
        return geo

    def input_io(self, index: int, *, required: bool = True) -> Optional[GeometryIO]:
        if not self.io_inputs:
            if index == 0:
                return self.io
            raise IndexError("Input IO list is empty; only index 0 is available.")
        if index < 0 or index >= len(self.io_inputs):
            raise IndexError(f"Input index {index} out of range (0-{len(self.io_inputs) - 1})")
        io = self.io_inputs[index]
        if io is None and required:
            raise RuntimeError(f"Input geometry {index} is not connected.")
        return io

    def read(
        self,
        owner: str,
        name: str,
        *,
        dtype: Optional[Any] = None,
        components: Optional[int] = None,
    ) -> np.ndarray:
        return self.io.read(owner, name, dtype=dtype, components=components)

    def write(self, owner: str, name: str, values: Any, *, create: bool = True) -> None:
        self.io.write(owner, name, values, create=create)

    def read_prims(self, arity: int = 3) -> np.ndarray:
        return self.io.read_prims(arity=arity)

    def read_group(self, owner: str, group_name: str, *, as_mask: bool = False) -> np.ndarray:
        return self.io.read_group(owner, group_name, as_mask=as_mask)

    def read_group_default(
        self,
        owner: str,
        group_name: str,
        *,
        as_mask: Optional[bool] = None,
    ) -> np.ndarray:
        if as_mask is None:
            as_mask = bool(self.is_solver)
        return self.read_group(owner, group_name, as_mask=as_mask)

    def P(self) -> np.ndarray:
        return self.read(OWNER_POINT, "P", components=3)

    def set_P(self, values: Any) -> None:
        self.write(OWNER_POINT, "P", values, create=True)

    def triangles(self) -> np.ndarray:
        return self.read_prims(arity=3)

    def publish(self, key: str, value: Any) -> None:
        reg = self.world().reg
        if not _resource_exists(reg, key):
            reg.declare(key, buffer=value, spec=_spec_for_value(value))
            reg.commit(key)
            return
        reg.commit(key, buffer=value)

    def publish_many(self, items: Dict[str, Any]) -> None:
        reg = self.world().reg
        keys = sorted(items.keys())
        pending: Dict[str, Any] = {}
        for key in keys:
            value = items[key]
            if not _resource_exists(reg, key):
                reg.declare(key, buffer=value, spec=_spec_for_value(value))
                reg.commit(key)
            else:
                pending[key] = value
        if not pending:
            return
        commit_many = getattr(reg, "commit_many", None)
        if callable(commit_many):
            commit_many(pending.keys(), buffers=pending)
            return
        for key, value in pending.items():
            reg.commit(key, buffer=value)

    def fetch(self, key: str) -> Any:
        return self.world().reg.read(key, ensure=True)

    def ensure(self, key: str) -> None:
        self.world().reg.ensure(key)

    def session_access(
        self,
        node_path: str,
        *,
        mode: "AccessMode" = "read",
        create: bool = False,
    ) -> "SessionAccess":
        from .session import get_runtime

        access = get_runtime().session_access(node_path, mode=mode, create=create)

        caller_key = self.session.user_module_key
        target_key = access.session.user_module_key
        if caller_key and target_key and caller_key != target_key:
            self.log(
                "session_access.module_mismatch",
                target_node_path=node_path,
                caller_module=caller_key,
                target_module=target_key,
            )

        return access

    def log(self, message: str, **payload: Any) -> None:
        self.session.log_event(
            message,
            node_path=self.node.path(),
            frame=self.frame,
            time=self.time,
            dt=self.dt,
            substep=self.substep,
            **payload,
        )


def build_cook_context(
    node: "hou.Node",
    geo_in: "hou.Geometry",
    geo_out: "hou.Geometry",
    session: WorldSession,
    *,
    geo_inputs: Optional[Sequence[Optional["hou.Geometry"]]] = None,
    substep: int = 0,
    is_solver: bool = False,
) -> CookContext:
    hou = _get_hou()
    fps = hou.fps()
    dt = 1.0 / fps if fps else 0.0
    if geo_inputs is None:
        geo_inputs_tuple = (geo_in,)
    else:
        geo_inputs_tuple = tuple(geo_inputs)
        if not geo_inputs_tuple:
            geo_inputs_tuple = (geo_in,)
    io_inputs = tuple(
        _make_read_only_io(geo) if geo is not None else None for geo in geo_inputs_tuple
    )
    return CookContext(
        node=node,
        frame=hou.frame(),
        time=hou.time(),
        dt=dt,
        substep=substep,
        is_solver=is_solver,
        session=session,
        geo_in=geo_in,
        geo_out=geo_out,
        io=GeometryIO(geo_in, geo_out),
        geo_inputs=geo_inputs_tuple,
        io_inputs=io_inputs,
    )
