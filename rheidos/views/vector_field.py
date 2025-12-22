from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

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
from ..sim.base import FieldInfo, VectorFieldSample
from ..store import StoreState
from ..visualization.color_schemes import ColorScheme, create_color_scheme


VectorProvider = Callable[[], Optional[VectorFieldSample]]
VectorFieldSource = Union[VectorProvider, FieldInfo[VectorFieldSample]]


class VectorFieldView(View):
    """
    Generic arrow/hedgehog renderer that consumes vector samples from a provider.
    """

    def __init__(
        self,
        vector_source: VectorFieldSource,
        *,
        color_scheme: str | ColorScheme = "sequential",
        scale: float = 1.0,
        thickness: float = 2.0,
        arrow_heads: bool = False,
        arrow_head_length: float = 0.2,
        arrow_head_angle_deg: float = 20.0,
        auto_scale_max_length: Optional[float] = None,
        auto_color_max: bool = False,
        visible_store_key: Optional[str] = None,
        store: Optional[StoreState] = None,
        name: Optional[str] = None,
        sort: int = 0,
    ) -> None:
        super().__init__(name=name or "VectorFieldView", sort=sort)
        if isinstance(vector_source, FieldInfo):
            self._vector_provider = vector_source.provider
            self._field_meta = vector_source.meta
        else:
            self._vector_provider = vector_source
            self._field_meta = None
        self._color_scheme = (
            create_color_scheme(color_scheme) if isinstance(color_scheme, str) else color_scheme
        )
        self._scale = float(scale)
        self._thickness = float(thickness)
        self._arrow_heads = bool(arrow_heads)
        self._arrow_head_length = float(max(0.0, arrow_head_length))
        self._arrow_head_angle_deg = float(max(1.0, min(89.0, arrow_head_angle_deg)))
        self._auto_scale_max_length = (
            float(auto_scale_max_length) if auto_scale_max_length is not None else None
        )
        self._auto_color_max = bool(auto_color_max)
        self._color_max_cache: Optional[float] = None
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
        self._color_max_cache = None

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
        max_mag = float(np.max(magnitudes)) if magnitudes.size else 0.0
        # Optional dynamic color normalization: track running max so colors use the historical peak
        # (avoids normalizing the single vector to 1.0 every frame). Legend follows the same peak.
        if self._auto_color_max and max_mag > 0.0:
            self._color_max_cache = max(self._color_max_cache or 0.0, max_mag)
            try:
                if hasattr(self._color_scheme, "set_max_value"):
                    getattr(self._color_scheme, "set_max_value")(self._color_max_cache)  # type: ignore[attr-defined]
                elif hasattr(self._color_scheme, "max_value"):
                    setattr(self._color_scheme, "max_value", self._color_max_cache)
            except Exception:
                pass

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

        scale = self._scale
        ref_mag = self._color_max_cache if self._color_max_cache is not None else max_mag
        if self._auto_scale_max_length is not None and ref_mag > 1e-8:
            # Map the historical peak magnitude to auto_scale_max_length; smaller magnitudes shrink linearly.
            scale = self._scale * (self._auto_scale_max_length / ref_mag)

        count = sample.positions.shape[0]
        segment_count = 0
        for i in range(count):
            base = np.asarray(sample.positions[i], dtype=np.float32)
            tip = base + np.asarray(sample.vectors[i], dtype=np.float32) * scale
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
            segment_count += 1

            if self._arrow_heads:
                dir_vec = np.asarray(end - start, dtype=np.float32)
                dir_len = float(np.linalg.norm(dir_vec))
                if dir_len > 1e-8:
                    dir_n = dir_vec / dir_len
                    # pick a stable perpendicular
                    side = np.cross(dir_n, np.array([0.0, 0.0, 1.0], dtype=np.float32))
                    if np.linalg.norm(side) < 1e-6:
                        side = np.cross(dir_n, np.array([0.0, 1.0, 0.0], dtype=np.float32))
                    side /= max(1e-8, np.linalg.norm(side))
                    angle_tan = math.tan(math.radians(self._arrow_head_angle_deg))
                    head_len = min(self._arrow_head_length, dir_len * 0.35)

                    left_dir = (-dir_n + side * angle_tan)
                    left_dir /= max(1e-8, np.linalg.norm(left_dir))
                    right_dir = (-dir_n - side * angle_tan)
                    right_dir /= max(1e-8, np.linalg.norm(right_dir))

                    left_tip = end + Vec3(*(left_dir * head_len))
                    right_tip = end + Vec3(*(right_dir * head_len))

                    pos_writer.addData3f(end)
                    pos_writer.addData3f(left_tip)
                    pos_writer.addData3f(end)
                    pos_writer.addData3f(right_tip)
                    for _ in range(4):
                        color_writer.addData4f(
                            float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])
                        )
                    segment_count += 2

        prim = GeomLines(Geom.UHStatic)
        prim.reserveNumVertices(segment_count * 2)
        vert_idx = 0
        for _ in range(segment_count):
            prim.addVertices(vert_idx, vert_idx + 1)
            vert_idx += 2
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node.addGeom(geom)
        self._node.show()
        sample.dirty = False


__all__ = ["VectorFieldView", "VectorProvider", "VectorFieldSource"]
