"""CookContext helpers for Houdini compute integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from rheidos.compute.resource import ResourceSpec
from rheidos.compute.world import World

from ..geo.adapter import GeometryIO
from ..geo.schema import GeometrySchema, OWNER_POINT
from .session import WorldSession

if TYPE_CHECKING:
    import hou


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


@dataclass
class CookContext:
    node: "hou.Node"
    frame: float
    time: float
    dt: float
    substep: int
    session: WorldSession
    geo_in: "hou.Geometry"
    geo_out: "hou.Geometry"
    io: GeometryIO
    schema: Optional[GeometrySchema] = None

    def world(self) -> World:
        if self.session.world is None:
            self.session.world = World()
        return self.session.world

    def clear_cache(self) -> None:
        self.io.clear_cache()

    def describe(self, owner: Optional[str] = None) -> GeometrySchema:
        schema = self.io.describe(owner=owner)
        if owner is None:
            self.schema = schema
        return schema

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
        for key in sorted(items.keys()):
            self.publish(key, items[key])

    def fetch(self, key: str) -> Any:
        return self.world().reg.read(key, ensure=True)

    def ensure(self, key: str) -> None:
        self.world().reg.ensure(key)


def build_cook_context(
    node: "hou.Node",
    geo_in: "hou.Geometry",
    geo_out: "hou.Geometry",
    session: WorldSession,
    *,
    substep: int = 0,
) -> CookContext:
    hou = _get_hou()
    fps = hou.fps()
    dt = 1.0 / fps if fps else 0.0
    return CookContext(
        node=node,
        frame=hou.frame(),
        time=hou.time(),
        dt=dt,
        substep=substep,
        session=session,
        geo_in=geo_in,
        geo_out=geo_out,
        io=GeometryIO(geo_in, geo_out),
    )
