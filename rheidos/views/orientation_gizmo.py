from __future__ import annotations

from typing import Optional

try:
    from panda3d.core import (
        Camera,
        DisplayRegion,
        LineSegs,
        NodePath,
        OrthographicLens,
        PerspectiveLens,
        AntialiasAttrib,
    )
except Exception:  # pragma: no cover
    Camera = None  # type: ignore
    DisplayRegion = None  # type: ignore
    LineSegs = None  # type: ignore
    NodePath = None  # type: ignore
    OrthographicLens = None  # type: ignore
    PerspectiveLens = None  # type: ignore

from ..abc.view import View


class OrientationGizmoView(View):
    """
    Small axis widget rendered in the top-left corner that rotates with the
    current camera orientation. X=red, Y=green, Z=blue.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        sort: int = 1000,
        size: float = 0.18,  # fraction of window height (keeps square)
        margin: float = 0.02,  # uniform fraction from edges
        margin_x: Optional[float] = None,  # overrides margin horizontally if set
        margin_y: Optional[float] = None,  # overrides margin vertically if set
        thickness: float = 3.0,
        fov_deg: float = 28.0,
        inner_margin: float = 0.08,  # inside the square, keep a small border
        anchor: str = "top-left",  # one of: top-left, top-right, bottom-left, bottom-right
    ) -> None:
        super().__init__(name=name or "OrientationGizmoView", sort=sort)
        self.size = float(size)
        self.margin = float(margin)
        self.margin_x = float(margin if margin_x is None else margin_x)
        self.margin_y = float(margin if margin_y is None else margin_y)
        self.thickness = float(thickness)
        self.fov_deg = float(fov_deg)
        self.inner_margin = float(inner_margin)
        self.anchor = anchor

        self._render_np: Optional[NodePath] = None
        self._axes_np: Optional[NodePath] = None
        self._cam_np: Optional[NodePath] = None
        self._dr: Optional[DisplayRegion] = None
        self._lens: Optional[PerspectiveLens] = None
        self._last_wh: tuple[int, int] = (0, 0)

    def setup(self, session) -> None:
        super().setup(session)
        if Camera is None or LineSegs is None:
            return

        # Off-to-the-side scene graph for the gizmo
        self._render_np = NodePath(self.name + "-render")
        self._axes_np = self._build_axes(self._render_np)
        self._axes_np.setDepthTest(False)
        self._axes_np.setDepthWrite(False)
        # Try to smooth lines; if MSAA is available, Panda will pick it up.
        try:
            self._axes_np.setAntialias(AntialiasAttrib.MAuto)
        except Exception:
            pass
        # Keep origin centered to avoid chopping when any axis points left.
        try:
            self._axes_np.setPos(0.0, 0.0, 0.0)
        except Exception:
            pass

        # Camera & display region (overlay)
        cam = Camera(self.name + "-cam")
        # Use perspective so all three axes are visible (including +Y forward)
        lens = PerspectiveLens()
        lens.setFov(self.fov_deg)
        lens.setNear(0.01)
        lens.setFar(100.0)
        lens.setAspectRatio(1.0)  # region is square
        cam.setLens(lens)
        self._lens = lens
        self._cam_np = self._render_np.attachNewNode(cam)
        self._cam_np.setPos(0, -3.5, 0)
        self._cam_np.lookAt(0, 0, 0)

        # Create a display region; we'll set exact dimensions after we know window size
        dr = session.win.makeDisplayRegion(0, 0, 0, 0)
        dr.setSort(self.sort)
        dr.setCamera(self._cam_np)
        dr.setClearDepthActive(True)
        dr.setClearDepth(1.0)
        # Don't clear color so we overlay on top of the main scene
        dr.setClearColorActive(False)
        self._dr = dr

        # Initialize region to a square anchored at top-left using current window size
        self._update_display_region_dims()

    def update(self, dt: float) -> None:
        # Rotate the axes opposite to the main camera so the axes appear
        # from the viewer's perspective without moving the gizmo camera.
        if self._axes_np is None:
            return
        try:
            base = self._session.base
            render = self._session.render
            # Keep region square on resize
            self._update_display_region_dims()
            q = base.camera.getQuat(render)
            q.invertInPlace()
            self._axes_np.setQuat(q)
        except Exception:
            pass

    def teardown(self) -> None:
        # Remove display region
        try:
            if self._dr is not None and self._session.win is not None:
                self._session.win.removeDisplayRegion(self._dr)
        except Exception:
            pass
        self._dr = None

        if self._render_np is not None:
            self._render_np.removeNode()
        self._render_np = None
        self._axes_np = None
        self._cam_np = None

    # --- helpers ---
    def _build_axes(self, parent: NodePath) -> NodePath:
        ls = LineSegs()
        ls.setThickness(self.thickness)
        L = 0.9 - self.inner_margin
        # X (red): draw both directions to avoid clipping on rotation
        ls.setColor(1, 0, 0, 1)
        ls.moveTo(-L, 0, 0)
        ls.drawTo(L, 0, 0)
        # Y (green)
        ls.setColor(0, 1, 0, 1)
        ls.moveTo(0, -L, 0)
        ls.drawTo(0, L, 0)
        # Z (blue)
        ls.setColor(0, 0, 1, 1)
        ls.moveTo(0, 0, -L)
        ls.drawTo(0, 0, L)
        node = ls.create(False)
        return parent.attachNewNode(node)

    def _update_display_region_dims(self) -> None:
        if self._dr is None or self._session is None or self._session.win is None:
            return
        win = self._session.win
        w = max(1, int(win.getXSize()))
        h = max(1, int(win.getYSize()))
        if (w, h) == self._last_wh:
            return
        self._last_wh = (w, h)

        aspect = float(w) / float(h)
        # Height based on window height; width chosen to keep square pixels
        h_norm = max(0.0, min(self.size, 1.0))
        w_norm = h_norm / aspect

        # Anchor logic: interpret margins as distance from the respective edges
        a = (self.anchor or "top-left").lower()
        mx = max(0.0, min(1.0, self.margin_x))
        my = max(0.0, min(1.0, self.margin_y))
        if a == "top-right":
            x2 = 1.0 - mx
            x1 = x2 - w_norm
            y2 = 1.0 - my
            y1 = y2 - h_norm
        elif a == "bottom-left":
            x1 = mx
            x2 = x1 + w_norm
            y1 = my
            y2 = y1 + h_norm
        elif a == "bottom-right":
            x2 = 1.0 - mx
            x1 = x2 - w_norm
            y1 = my
            y2 = y1 + h_norm
        else:  # top-left
            x1 = mx
            x2 = x1 + w_norm
            y2 = 1.0 - my
            y1 = y2 - h_norm

        # Clamp to the window [0,1] range
        if x1 < 0.0:
            x2 -= x1
            x1 = 0.0
        if y1 < 0.0:
            y2 -= y1
            y1 = 0.0
        if x2 > 1.0:
            x1 -= (x2 - 1.0)
            x2 = 1.0
        if y2 > 1.0:
            y1 -= (y2 - 1.0)
            y2 = 1.0
        self._dr.setDimensions(x1, x2, y1, y2)
