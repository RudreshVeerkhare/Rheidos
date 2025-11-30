from __future__ import annotations

from typing import Optional

from panda3d.core import NodePath

from ..abc.view import View
from ..resources.mesh import Mesh


class MeshSurfaceView(View):
    def __init__(self, mesh: Mesh, name: Optional[str] = None, sort: int = 0, material: Optional[object] = None, two_sided: bool = False) -> None:
        super().__init__(name=name or "MeshSurfaceView", sort=sort)
        self._mesh = mesh
        self._node: Optional[NodePath] = None
        self._material = material
        self._two_sided = two_sided

    def setup(self, session) -> None:
        super().setup(session)
        node = self._mesh.node_path.copyTo(self._session.render)
        node.setName(self.name)
        node.setRenderModeFilled()
        node.setTwoSided(self._two_sided)
        node.setShaderAuto()
        if self._material is not None:
            node.setMaterial(self._material, 1)
        self._node = node

    def teardown(self) -> None:
        if self._node:
            self._node.removeNode()
            self._node = None

    def on_enable(self) -> None:
        if self._node:
            self._node.show()

    def on_disable(self) -> None:
        if self._node:
            self._node.hide()


class MeshWireframeView(View):
    def __init__(self, mesh: Mesh, name: Optional[str] = None, sort: int = 0) -> None:
        super().__init__(name=name or "MeshWireframeView", sort=sort)
        self._mesh = mesh
        self._node: Optional[NodePath] = None

    def setup(self, session) -> None:
        super().setup(session)
        node = self._mesh.node_path.copyTo(self._session.render)
        node.setName(self.name)
        node.setRenderModeWireframe()
        node.setColor(0.0, 0.85, 1.0, 1.0)
        node.setLightOff()
        node.setTwoSided(True)
        self._node = node

    def teardown(self) -> None:
        if self._node:
            self._node.removeNode()
            self._node = None

    def on_enable(self) -> None:
        if self._node:
            self._node.show()

    def on_disable(self) -> None:
        if self._node:
            self._node.hide()
