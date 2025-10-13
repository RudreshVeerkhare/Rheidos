from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np

from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    Material,
    NodePath,
    SamplerState,
    TextureStage,
    Vec3,
    Vec4,
)

from ..abc.view import View
from ..resources.texture import Texture2D


_NodeTarget = Union[NodePath, "Mesh"]


class StudioView(View):
    def __init__(
        self,
        name: Optional[str] = None,
        sort: int = -20,
        ground_size: float = 40.0,
        ground_tiles: int = 40,
        checker_light: tuple[float, float, float, float] = (0.92, 0.93, 0.96, 1.0),
        checker_dark: tuple[float, float, float, float] = (0.86, 0.87, 0.90, 1.0),
        sky_color: tuple[float, float, float, float] = (0.92, 0.95, 1.0, 1.0),
        add_lights: bool = True,
        apply_material_to: Optional[Union[_NodeTarget, Sequence[_NodeTarget]]] = None,
        material: Optional[Material] = None,
        ground_height: Optional[float] = None,  # world Z height of ground plane
        ground_from_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
        ground_margin: float = 0.0,
    ) -> None:
        super().__init__(name=name or "StudioView", sort=sort)
        self.ground_size = float(ground_size)
        self.ground_tiles = int(max(1, ground_tiles))
        self.checker_light = checker_light
        self.checker_dark = checker_dark
        self.sky_color = sky_color
        self.add_lights = add_lights
        self._targets: list[_NodeTarget] = []
        if apply_material_to is not None:
            if isinstance(apply_material_to, (list, tuple)):
                self._targets = list(apply_material_to)  # type: ignore[list-item]
            else:
                self._targets = [apply_material_to]
        self._material = material
        self._ground_height = ground_height
        self._ground_from_bounds = ground_from_bounds
        self._ground_margin = float(ground_margin)

        self._group: Optional[NodePath] = None
        self._ground_np: Optional[NodePath] = None
        self._ambient_np: Optional[NodePath] = None
        self._key_np: Optional[NodePath] = None
        self._fill_np: Optional[NodePath] = None

    def setup(self, session) -> None:
        super().setup(session)
        session.base.setBackgroundColor(*self.sky_color)

        self._group = session.render.attachNewNode(self.name)

        if self.add_lights:
            amb = AmbientLight("ambient")
            amb.setColor(Vec4(0.18, 0.18, 0.22, 1.0))
            self._ambient_np = self._group.attachNewNode(amb)
            session.render.setLight(self._ambient_np)

            key = DirectionalLight("key")
            key.setColor(Vec4(0.85, 0.85, 0.9, 1.0))
            key.setShadowCaster(False)
            self._key_np = self._group.attachNewNode(key)
            self._key_np.setHpr(-35, -45, 0)
            session.render.setLight(self._key_np)

            fill = DirectionalLight("fill")
            fill.setColor(Vec4(0.35, 0.35, 0.45, 1.0))
            self._fill_np = self._group.attachNewNode(fill)
            self._fill_np.setHpr(60, -20, 0)
            session.render.setLight(self._fill_np)

        self._ground_np = self._build_ground(self._group)
        # Elevate/offset ground along the UP axis (Z in Panda3D)
        z = 0.0
        if self._ground_height is not None:
            z = float(self._ground_height)
        elif self._ground_from_bounds is not None:
            mins, _ = self._ground_from_bounds
            z = float(mins[2] - self._ground_margin)
        self._ground_np.setZ(z)

        if self._targets:
            mat = self._material or self._build_default_material()
            for t in self._targets:
                self._apply_material(t, mat)

    def teardown(self) -> None:
        if self._group is not None:
            self._group.removeNode()
        self._group = None
        self._ground_np = None
        self._ambient_np = None
        self._key_np = None
        self._fill_np = None

    def on_enable(self) -> None:
        if self._group is not None:
            self._group.show()

    def on_disable(self) -> None:
        if self._group is not None:
            self._group.hide()

    def _build_default_material(self) -> Material:
        m = Material("StudioGloss")
        m.setDiffuse((0.80, 0.82, 0.90, 1.0))
        m.setSpecular((1.0, 1.0, 1.0, 1.0))
        m.setShininess(48.0)
        return m

    def _apply_material(self, target: _NodeTarget, material: Material) -> None:
        if hasattr(target, "node_path"):
            np_target = getattr(target, "node_path")
        else:
            np_target = target  # type: ignore[assignment]
        try:
            np_target.setShaderAuto()
            np_target.setMaterial(material, 1)
        except Exception:
            pass

    def _build_ground(self, parent: NodePath) -> NodePath:
        s = self.ground_size
        tiles = float(self.ground_tiles)

        vdata = GeomVertexData("ground", GeomVertexFormat.getV3t2(), Geom.UHStatic)
        vw = GeomVertexWriter(vdata, "vertex")
        tw = GeomVertexWriter(vdata, "texcoord")

        vw.addData3f(-s, -s, 0.0)
        tw.addData2f(0.0, 0.0)
        vw.addData3f(s, -s, 0.0)
        tw.addData2f(tiles, 0.0)
        vw.addData3f(s, s, 0.0)
        tw.addData2f(tiles, tiles)
        vw.addData3f(-s, s, 0.0)
        tw.addData2f(0.0, tiles)

        prim = GeomTriangles(Geom.UHStatic)
        prim.addVertices(0, 1, 2)
        prim.addVertices(0, 2, 3)
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode("ground")
        node.addGeom(geom)
        np_ground = parent.attachNewNode(node)
        np_ground.setTwoSided(True)
        np_ground.setShaderAuto()

        tex = self._build_checker_texture()
        t = tex.tex
        try:
            t.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            t.setMagfilter(SamplerState.FT_linear)
            t.setAnisotropicDegree(4)
            t.setWrapU(SamplerState.WM_repeat)
            t.setWrapV(SamplerState.WM_repeat)
        except Exception:
            pass
        np_ground.setTexture(TextureStage.getDefault(), t, 1)

        return np_ground

    def _build_checker_texture(self) -> Texture2D:
        cells = max(2, self.ground_tiles)
        cell_px = 32
        w = h = cells * cell_px
        i = np.indices((h, w)).sum(axis=0) // cell_px
        pattern = (i % 2).astype(np.uint8)
        c0 = np.array(self.checker_light, dtype=np.float32)
        c1 = np.array(self.checker_dark, dtype=np.float32)
        img = (pattern[..., None] * c1 + (1 - pattern[..., None]) * c0)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        tex = Texture2D("ground_checker")
        tex.from_numpy_rgba(img)
        return tex

    # --- public helpers ---
    def set_ground_height(self, z: float) -> None:
        if self._ground_np is not None:
            self._ground_np.setZ(float(z))

    def snap_ground_to_bounds(self, bounds: tuple[np.ndarray, np.ndarray], margin: float = 0.0) -> None:
        mins, _ = bounds
        self.set_ground_height(float(mins[2] - margin))
