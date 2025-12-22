from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

try:
    from panda3d.core import CardMaker, NodePath, TextureStage, TransparencyAttrib
except Exception:  # pragma: no cover
    CardMaker = None  # type: ignore
    NodePath = None  # type: ignore
    TextureStage = None  # type: ignore
    TransparencyAttrib = None  # type: ignore

from ..abc.view import View
from ..resources.texture import Texture2D
from ..sim.base import FieldInfo, ScalarFieldSample
from ..store import StoreState
from ..visualization.color_schemes import ColorScheme, create_color_scheme

ScalarProvider = Callable[[], Optional[ScalarFieldSample]]
ScalarFieldSource = Union[ScalarProvider, FieldInfo[ScalarFieldSample]]


class ScalarFieldView(View):
    """
    Generic scalar grid renderer that colors values into a texture-mapped quad.
    """

    def __init__(
        self,
        scalar_source: ScalarFieldSource,
        *,
        color_scheme: str | ColorScheme = "sequential",
        frame: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
        visible_store_key: Optional[str] = None,
        store: Optional[StoreState] = None,
        name: Optional[str] = None,
        sort: int = 0,
    ) -> None:
        super().__init__(name=name or "ScalarFieldView", sort=sort)
        if isinstance(scalar_source, FieldInfo):
            self._scalar_provider = scalar_source.provider
            self._field_meta = scalar_source.meta
        else:
            self._scalar_provider = scalar_source
            self._field_meta = None
        self._color_scheme = (
            create_color_scheme(color_scheme) if isinstance(color_scheme, str) else color_scheme
        )
        self._frame = frame
        self._visible_store_key = visible_store_key
        self._store = store

        self._node: Optional[NodePath] = None
        self._tex: Optional[Texture2D] = None
        self._ts: Optional[TextureStage] = None

    def setup(self, session: Any) -> None:
        super().setup(session)
        if CardMaker is None or TextureStage is None or TransparencyAttrib is None:
            return
        cm = CardMaker(self.name)
        cm.setFrame(*self._frame)
        self._node = self._session.render.attachNewNode(cm.generate())
        self._node.setTransparency(TransparencyAttrib.MAlpha)

        self._tex = Texture2D(name=f"{self.name}-tex")
        self._ts = TextureStage(f"{self.name}-ts")
        self._node.setTexture(self._ts, self._tex.tex)
        self._node.hide()

    def teardown(self) -> None:
        if self._node is not None:
            self._node.removeNode()
            self._node = None
        self._tex = None
        self._ts = None

    def on_enable(self) -> None:
        if self._node is not None:
            self._node.show()

    def on_disable(self) -> None:
        if self._node is not None:
            self._node.hide()

    def update(self, dt: float) -> None:
        if self._node is None or self._tex is None:
            return

        if self._store is not None and self._visible_store_key:
            visible = bool(self._store.get(self._visible_store_key, True))
            if not visible:
                self._node.hide()
                return

        sample = self._scalar_provider()
        if sample is None:
            self._node.hide()
            return

        sample.validate()
        if sample.values.size == 0:
            self._node.hide()
            return

        colors = self._color_scheme.apply(sample.values)
        if colors.ndim != 3 or colors.shape[2] != 4:
            raise ValueError(
                f"ScalarFieldView expects colors shaped (H, W, 4); got {colors.shape}"
            )

        rgba = np.clip(colors, 0.0, 1.0)
        image = (rgba * 255.0).astype(np.uint8, copy=False)
        self._tex.from_numpy_rgba(image)
        self._node.show()
        sample.dirty = False


__all__ = ["ScalarFieldView", "ScalarProvider", "ScalarFieldSource"]
