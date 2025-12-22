from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

try:
    from panda3d.core import Geom, GeomLines, GeomNode, GeomVertexData, GeomVertexFormat, GeomVertexWriter, NodePath, Vec3
except Exception:  # pragma: no cover
    Geom = None  # type: ignore
    GeomLines = None  # type: ignore
    GeomNode = None  # type: ignore
    GeomVertexData = None  # type: ignore
    GeomVertexFormat = None  # type: ignore
    GeomVertexWriter = None  # type: ignore
    NodePath = None  # type: ignore
    Vec3 = None  # type: ignore

from ..abc.view import View
from ..sim.base import VectorFieldSample
from ..store import StoreState
from ..visualization.color_schemes import ColorScheme, create_color_scheme


VectorProvider = Callable[[], Optional[VectorFieldSample]]


class VectorFieldView(View):
    """
    Generic arrow/hedgehog renderer that consumes vector samples from a provider.
    """

    def __init__(
        self,
        vector_provider: VectorProvider,
        *,
        color_scheme: str | ColorScheme = "sequential",
        scale: float = 1.0,
        thickness: float = 2.0,
        visible_store_key: Optional[str] = None,
        store: Optional[StoreState] = None,
        name: Optional[str] = None,
        sort: int = 0,
    ) -> None:
        super().__init__(name=name or "VectorFieldView", sort=sort)
        self._vector_provider = vector_provider
        self._color_scheme = (
            create_color_scheme(color_scheme) if isinstance(color_scheme, str) else color_scheme
        )
        self._scale = float(scale)
        self._thickness = float(thickness)
        self._visible_store_key = visible_store_key
        self._store = store

        self._node: Optional[NodePath] = None

    def setup(self, session: Any) -> None:
        super().setup(session)
        if GeomNode is None:
            return
        self._node = self._session.render.attachNewNode(GeomNode(self.name))
        self._node.setRenderModeThickness(self._thickness)
        self._node.hide()

    def teardown(self) -> None:
        if self._node is not None:
            self._node.removeNode()
            self._node = None

    def on_enable(self) -> None:
        if self._node is not None:
            self._node.show()

    def on_disable(self) -> None:
        if self._node is not None:
            self._node.hide()

    def update(self, dt: float) -> None:
        if self._node is None or Geom is None:
            return

        if self._store is not None and self._visible_store_key:
            visible = bool(self._store.get(self._visible_store_key, True))
            if not visible:
                self._node.hide()
                return

        sample = self._vector_provider()
        if sample is None:
            self._node.hide()
            return

        sample.validate()
        if sample.positions.size == 0:
            self._node.hide()
            return

        geom_node = self._node.node()
        if not isinstance(geom_node, GeomNode):
            return

        magnitudes = (
            sample.magnitudes
            if sample.magnitudes is not None
            else np.linalg.norm(sample.vectors, axis=1).astype(np.float32)
        )
        colors = self._color_scheme.apply(magnitudes)
        colors_flat = colors.reshape(-1, colors.shape[-1])
        if colors_flat.shape[0] != sample.positions.shape[0]:
            raise ValueError(
                f"Color count {colors_flat.shape[0]} does not match vectors {sample.positions.shape[0]}"
            )

        geom_node.removeAllGeoms()
        vformat = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData(self.name + "-vectors", vformat, Geom.UHStatic)
        pos_writer = GeomVertexWriter(vdata, "vertex")
        color_writer = GeomVertexWriter(vdata, "color")

        count = sample.positions.shape[0]
        for i in range(count):
            base = np.asarray(sample.positions[i], dtype=np.float32)
            tip = base + np.asarray(sample.vectors[i], dtype=np.float32) * self._scale
            try:
                start = Vec3(float(base[0]), float(base[1]), float(base[2]))
                end = Vec3(float(tip[0]), float(tip[1]), float(tip[2]))
            except Exception:
                continue
            pos_writer.addData3f(start)
            pos_writer.addData3f(end)
            rgba = colors_flat[i]
            color_writer.addData4f(float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
            color_writer.addData4f(float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))

        prim = GeomLines(Geom.UHStatic)
        prim.reserveNumVertices(count * 2)
        for i in range(count):
            prim.addVertices(2 * i, 2 * i + 1)
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node.addGeom(geom)
        self._node.show()
        sample.dirty = False


__all__ = ["VectorFieldView", "VectorProvider"]
