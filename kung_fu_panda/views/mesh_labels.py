from __future__ import annotations

from typing import Optional

import numpy as np

from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionTraverser,
    Geom,
    GeomPoints,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    GeomNode,
    TextNode,
    Vec3,
)

from ..abc.view import View
from ..resources.mesh import Mesh


class MeshPositionLabelsView(View):
    """Show the coordinates of the vertex closest to the mouse pointer."""

    def __init__(
        self,
        mesh: Mesh,
        name: Optional[str] = None,
        sort: int = 0,
        scale_factor: float = 0.015,
        offset_factor: float = 0.02,
        text_color: tuple[float, float, float, float] = (1.0, 0.9, 0.3, 1.0),
        fmt: str = "({x:.4f}, {y:.4f}, {z:.4f})",
        include_index: bool = True,
    ) -> None:
        super().__init__(name=name or "MeshPositionLabelsView", sort=sort)
        self._mesh = mesh
        self._scale_factor = scale_factor
        self._offset_factor = offset_factor
        self._text_color = text_color
        self._fmt = fmt
        self._include_index = include_index

        # Session-bound objects initialised in setup
        self._group = None
        self._text_node = None
        self._text_np = None
        self._highlight_np = None
        self._picker_traverser: Optional[CollisionTraverser] = None
        self._picker_queue: Optional[CollisionHandlerQueue] = None
        self._picker_ray: Optional[CollisionRay] = None
        self._picker_np = None
        self._geom_np = None

        self._mouse_watcher = None
        self._cam_node = None

        # Cached mesh data
        self._positions = None  # np.ndarray [N,3]
        self._normals = None  # Optional[np.ndarray]
        self._text_scale = 0.01
        self._offset_mag = 0.01

    def setup(self, session) -> None:
        super().setup(session)

        self._mouse_watcher = session.base.mouseWatcherNode
        self._cam_node = session.base.camNode

        self._group = session.render.attachNewNode(self.name)

        # Compute size-based scale/offset using tight bounds (Panda doc: NodePath.getTightBounds)
        bounds = self._mesh.node_path.getTightBounds()
        if bounds:
            mins, maxs = bounds
            radius = (maxs - mins).length() * 0.5
        else:
            radius = 1.0
        radius = max(radius, 1e-4)
        self._text_scale = radius * self._scale_factor
        self._offset_mag = radius * self._offset_factor

        # Cache vertex data into numpy for fast nearest searches
        handle = self._mesh.vdata.getArray(0).getHandle()
        raw = memoryview(handle.getData())
        self._positions = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3).copy()

        # Normals (optional)
        try:
            n_handle = self._mesh.vdata.getArray(1).getHandle()
            n_raw = memoryview(n_handle.getData())
            self._normals = np.frombuffer(n_raw, dtype=np.float32).reshape(-1, 3).copy()
        except Exception:
            self._normals = None

        # Text node for the single hovered vertex
        self._text_node = TextNode(f"{self.name}-label")
        self._text_node.setTextColor(*self._text_color)
        self._text_np = self._group.attachNewNode(self._text_node)
        self._text_np.setScale(self._text_scale)
        self._text_np.hide()

        cam = session.base.camera
        if cam is not None:
            self._text_np.setBillboardPointEye(cam, 0.0, False)

        # Highlight marker: a single point rendered thicker
        vdata = GeomVertexData("highlight", GeomVertexFormat.getV3(), Geom.UHStatic)
        writer = GeomVertexWriter(vdata, "vertex")
        writer.addData3f(0, 0, 0)
        prim = GeomPoints(Geom.UHStatic)
        prim.addVertex(0)
        prim.closePrimitive()
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node = GeomNode("highlight")
        geom_node.addGeom(geom)
        self._highlight_np = self._group.attachNewNode(geom_node)
        self._highlight_np.setRenderModeThickness(9)
        self._highlight_np.setColor(1.0, 0.25, 0.25, 1.0)
        self._highlight_np.hide()

        # Hidden copy of the mesh for picking collisions
        self._into_mask = BitMask32.bit(3)
        self._from_mask = BitMask32.bit(3)
        self._geom_np = self._mesh.node_path.copyTo(self._group)
        self._geom_np.hide()
        self._geom_np.setCollideMask(self._into_mask)

        # Collision ray cast from camera through mouse position (Panda doc: CollisionRay.setFromLens)
        self._picker_traverser = CollisionTraverser(f"{self.name}-picker")
        self._picker_queue = CollisionHandlerQueue()
        self._picker_ray = CollisionRay()
        picker_node = CollisionNode(f"{self.name}-ray")
        picker_node.addSolid(self._picker_ray)
        picker_node.setFromCollideMask(self._from_mask)
        picker_node.setIntoCollideMask(BitMask32.allOff())
        self._picker_np = session.base.camera.attachNewNode(picker_node)
        self._picker_traverser.addCollider(self._picker_np, self._picker_queue)

    def teardown(self) -> None:
        if self._picker_traverser and self._picker_np is not None:
            self._picker_traverser.removeCollider(self._picker_np)
        if self._picker_np is not None:
            self._picker_np.removeNode()

        if self._group is not None:
            self._group.removeNode()

        self._group = None
        self._text_node = None
        self._text_np = None
        self._highlight_np = None
        self._picker_traverser = None
        self._picker_queue = None
        self._picker_ray = None
        self._picker_np = None
        self._geom_np = None
        self._positions = None
        self._normals = None
        self._mouse_watcher = None
        self._cam_node = None

    def on_enable(self) -> None:
        if self._group is not None:
            self._group.show()

    def on_disable(self) -> None:
        if self._group is not None:
            self._group.hide()
        self._clear_selection()

    def _clear_selection(self) -> None:
        if self._text_np is not None:
            self._text_np.hide()
        if self._highlight_np is not None:
            self._highlight_np.hide()

    def update(self, dt: float) -> None:
        if (
            self._mouse_watcher is None
            or self._cam_node is None
            or self._picker_traverser is None
            or self._picker_queue is None
            or self._picker_ray is None
            or self._positions is None
            or self._geom_np is None
        ):
            return

        if not self._mouse_watcher.hasMouse():
            self._clear_selection()
            return

        mouse_pos = self._mouse_watcher.getMouse()
        self._picker_ray.setFromLens(self._cam_node, mouse_pos)

        self._picker_queue.clearEntries()
        self._picker_traverser.traverse(self._group)

        if self._picker_queue.getNumEntries() == 0:
            self._clear_selection()
            return

        self._picker_queue.sortEntries()
        entry = self._picker_queue.getEntry(0)
        hit_point = entry.getSurfacePoint(self._geom_np)

        target = np.array([hit_point.x, hit_point.y, hit_point.z], dtype=np.float32)
        diff = self._positions - target
        dists_sq = np.einsum("ij,ij->i", diff, diff)
        index = int(np.argmin(dists_sq))
        vertex = self._positions[index]

        pos_vec = Vec3(float(vertex[0]), float(vertex[1]), float(vertex[2]))

        if self._highlight_np is not None:
            self._highlight_np.setPos(self._geom_np, pos_vec)
            self._highlight_np.show()

        if self._text_np is not None and self._text_node is not None:
            offset_vec = Vec3(0, 0, 1)
            if self._normals is not None and index < len(self._normals):
                n = Vec3(
                    float(self._normals[index][0]),
                    float(self._normals[index][1]),
                    float(self._normals[index][2]),
                )
                if n.lengthSquared() > 1e-6:
                    n.normalize()
                    offset_vec = n
            self._text_np.setPos(self._geom_np, pos_vec + offset_vec * self._offset_mag)
            if self._include_index:
                text = f"{index}: " + self._fmt.format(x=vertex[0], y=vertex[1], z=vertex[2])
            else:
                text = self._fmt.format(x=vertex[0], y=vertex[1], z=vertex[2])
            self._text_node.setText(text)
            self._text_np.show()
