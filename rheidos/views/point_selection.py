from __future__ import annotations

from typing import Optional, Sequence

try:
    from panda3d.core import (
        Geom,
        GeomNode,
        GeomPoints,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
        NodePath,
        Vec3,
    )
except Exception:  # pragma: no cover
    Geom = None  # type: ignore
    GeomNode = None  # type: ignore
    GeomPoints = None  # type: ignore
    GeomVertexData = None  # type: ignore
    GeomVertexFormat = None  # type: ignore
    GeomVertexWriter = None  # type: ignore
    NodePath = None  # type: ignore
    Vec3 = None  # type: ignore

from ..abc.view import View


def _to_vec3(p: Vec3 | Sequence[float]) -> Vec3:
    if Vec3 is None:
        raise RuntimeError("Panda3D Vec3 is unavailable")
    if isinstance(p, Vec3):
        return p
    return Vec3(float(p[0]), float(p[1]), float(p[2]))


class PointSelectionView(View):
    """
    Lightweight overlay to visualize picked/selected points as thick GL points.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        sort: int = 0,
        selected_color: tuple[float, float, float, float] = (1.0, 0.35, 0.25, 1.0),
        hover_color: tuple[float, float, float, float] = (1.0, 0.95, 0.35, 1.0),
        thickness: float = 9.0,
    ) -> None:
        super().__init__(name=name or "PointSelectionView", sort=sort)
        self._selected_color = selected_color
        self._hover_color = hover_color
        self._thickness = float(thickness)

        self._root: Optional[NodePath] = None
        self._selected_np: Optional[NodePath] = None
        self._hover_np: Optional[NodePath] = None

    # ---- public API -------------------------------------------------

    def set_selected(self, points: Sequence[Vec3 | Sequence[float]]) -> None:
        self._update_geom(self._selected_np, points)

    def set_hover(self, point: Optional[Vec3 | Sequence[float]]) -> None:
        if point is None:
            self._update_geom(self._hover_np, [])
            return
        self._update_geom(self._hover_np, [point])

    # ---- lifecycle --------------------------------------------------

    def setup(self, session) -> None:
        super().setup(session)
        if (
            Geom is None
            or GeomNode is None
            or GeomPoints is None
            or GeomVertexData is None
            or GeomVertexFormat is None
            or GeomVertexWriter is None
            or NodePath is None
        ):
            return

        parent = getattr(session, "render", None)
        if parent is None:
            return

        self._root = parent.attachNewNode(self.name)

        self._selected_np = self._root.attachNewNode(GeomNode("selected-points"))
        self._selected_np.setColor(*self._selected_color)
        self._selected_np.setRenderModeThickness(self._thickness)
        self._selected_np.hide()

        self._hover_np = self._root.attachNewNode(GeomNode("hover-point"))
        self._hover_np.setColor(*self._hover_color)
        self._hover_np.setRenderModeThickness(self._thickness + 2.0)
        self._hover_np.hide()

    def teardown(self) -> None:
        for np in (self._selected_np, self._hover_np, self._root):
            try:
                if np is not None:
                    np.removeNode()
            except Exception:
                pass
        self._selected_np = None
        self._hover_np = None
        self._root = None

    def on_enable(self) -> None:
        if self._root is not None:
            self._root.show()

    def on_disable(self) -> None:
        if self._root is not None:
            self._root.hide()

    # ---- internals --------------------------------------------------

    def _update_geom(self, nodepath: Optional[NodePath], points: Sequence[Vec3 | Sequence[float]]) -> None:
        if (
            nodepath is None
            or GeomNode is None
            or GeomPoints is None
            or GeomVertexData is None
            or GeomVertexFormat is None
            or GeomVertexWriter is None
        ):
            return

        geom_node = nodepath.node()
        if not isinstance(geom_node, GeomNode):
            return

        geom_node.removeAllGeoms()

        if len(points) == 0:
            nodepath.hide()
            return

        vdata = GeomVertexData(self.name + "-pts", GeomVertexFormat.getV3(), Geom.UHStatic)
        writer = GeomVertexWriter(vdata, "vertex")
        for p in points:
            try:
                v = _to_vec3(p)
            except Exception:
                continue
            writer.addData3f(v)

        prim = GeomPoints(Geom.UHStatic)
        count = vdata.getNumRows()
        prim.reserveNumVertices(count)
        for i in range(count):
            prim.addVertex(i)
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node.addGeom(geom)
        nodepath.show()
