from __future__ import annotations
from typing import Dict, Optional
from math import copysign
from panda3d.core import Vec3, WindowProperties, NodePath, Quat

from ..abc.controller import Controller


def _safe_normalize(v: Vec3, eps: float = 1e-12) -> Vec3:
    if v.length_squared() <= eps:
        return Vec3(0, 1, 0)  # fallback forward
    v.normalize()
    return v


class FpvCameraController(Controller):
    def __init__(
        self,
        speed: float = 6.0,
        speed_fast: float = 12.0,
        mouse_sensitivity: float = 0.15,  # deg-per-pixel-like feel
        invert_y: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "FpvCameraController")

        # Tunables
        self.speed = speed
        self.speed_fast = speed_fast
        self.mouse_sensitivity = mouse_sensitivity
        self.invert_y = invert_y

        # Input state
        self._keys: Dict[str, bool] = {}
        self._accepted: list[str] = []
        self._task_name: Optional[str] = None

        # Center for mouse warp
        self._center_x = 0
        self._center_y = 0

        # Rig & camera parenting
        self._rig_np: Optional[NodePath] = None
        self._orig_cam_parent: Optional[NodePath] = None

        # World-space forward ("eye") that we maintain
        self._eye_w: Vec3 = Vec3(0, 1, 0)  # set on attach from current camera
        self._GLOBAL_UP = Vec3(0, 0, 1)

        # Elevation safety: avoid eye ~ +/- GLOBAL_UP (right becomes undefined)
        self._elev_cos_max = (
            0.99985  # ~= cos(1 deg). Never let |dot(eye, up)| exceed this.
        )

    def _build_frame_from_eye(self) -> tuple[Vec3, Vec3, Vec3]:
        """Return (right, up_local, forward) from current world-space eye."""
        f = _safe_normalize(self._eye_w)
        # local up = global up with the forward component projected out
        u = self._GLOBAL_UP - f * self._GLOBAL_UP.dot(f)
        u = _safe_normalize(u)
        # RIGHT-HANDED BASIS: right = forward × up_local   (NOT up_local × forward)
        r = f.cross(u)
        r = _safe_normalize(r)
        return r, u, f

    def _apply_yaw_pitch(self, yaw_deg: float, pitch_deg: float) -> None:
        """Rotate world-space eye: yaw about local up, then pitch about right."""
        r, u, f = self._build_frame_from_eye()

        # --- yaw about local up (keep horizon stable wrt world up) ---
        if abs(yaw_deg) > 0.0:
            q_yaw = Quat()
            q_yaw.set_from_axis_angle(yaw_deg, u)  # axis in world
            f = q_yaw.xform(f)

        # Recompute frame after yaw for a correct pitch axis
        # (u changes only slightly due to numerical noise; recompute robustly)
        self._eye_w = _safe_normalize(f)
        r, u, f = self._build_frame_from_eye()

        # --- pitch about right (tilt head up/down) ---
        if abs(pitch_deg) > 0.0:
            q_pitch = Quat()
            q_pitch.set_from_axis_angle(pitch_deg, r)
            f = q_pitch.xform(f)

        # Normalize & clamp elevation to avoid eye ≈ ±up
        f = _safe_normalize(f)
        d = f.dot(self._GLOBAL_UP)
        if abs(d) > self._elev_cos_max:
            # Slide back toward the limit keeping sign
            d = copysign(self._elev_cos_max, d)
            # Rebuild f with same azimuth but clamped elevation:
            # project onto plane, then add limited vertical component
            horiz = self._GLOBAL_UP.cross(f).cross(self._GLOBAL_UP)
            horiz = _safe_normalize(horiz)
            # sin(theta) component on the horizontal ring
            sin_t = (1.0 - d * d) ** 0.5
            f = _safe_normalize(horiz * sin_t + self._GLOBAL_UP * d)

        self._eye_w = f

    def _orient_rig_to_eye(self) -> None:
        """Point rig along self._eye_w using global up as reference (no roll)."""
        # Use lookAt with explicit up; this sets HPR relative to parent (render).
        rig = self._rig_np
        render = self._session.render
        if rig is None:
            return
        pos = rig.getPos(render)
        target = pos + self._eye_w  # a point in front of the rig
        rig.lookAt(render, target)  # first align forward
        # Re-apply with up to enforce no roll (two-step to be safe across Panda versions)
        rig.lookAt(render, target, self._GLOBAL_UP)

    # ---------- lifecycle ----------
    def attach(self, session) -> None:
        super().attach(session)
        base = session.base
        render = session.render
        camera = base.camera

        # Save parent & create rig
        self._orig_cam_parent = camera.getParent()
        self._rig_np = render.attachNewNode("fpv_rig")

        # Initialize world-space eye from current camera orientation
        cam_q_w = camera.getQuat(render)
        self._eye_w = _safe_normalize(cam_q_w.xform(Vec3(0, 1, 0)))

        # Place rig at camera's world position, and orient from eye
        self._rig_np.setPos(render, camera.getPos(render))
        self._orient_rig_to_eye()

        # Parent camera under rig, zero local transform
        camera.reparentTo(self._rig_np)
        camera.setPos(0, 0, 0)
        camera.setHpr(0, 0, 0)

        # Mouse capture
        win = session.win
        props = WindowProperties()
        props.setCursorHidden(True)
        win.requestProperties(props)
        self._center_x = win.getXSize() // 2
        self._center_y = win.getYSize() // 2
        win.movePointer(0, self._center_x, self._center_y)

        # Keys
        for key in ["w", "s", "a", "d", "q", "e", "shift"]:
            session.accept(key, self._on_key, [key, True])
            session.accept(f"{key}-up", self._on_key, [key, False])
            self._accepted.extend([key, f"{key}-up"])

        # Per-frame task
        self._task_name = f"fpv-camera-{id(self)}"
        session.task_mgr.add(self._update_task, self._task_name, sort=-50)

    def detach(self) -> None:
        if self._task_name:
            self._session.task_mgr.remove(self._task_name)
            self._task_name = None

        for evt in self._accepted:
            self._session.ignore(evt)
        self._accepted.clear()
        self._keys.clear()

        # Release mouse
        props = WindowProperties()
        props.setCursorHidden(False)
        self._session.win.requestProperties(props)

        # Restore camera world transform and remove rig
        base = self._session.base
        render = self._session.render
        camera = base.camera
        if self._rig_np is not None:
            cam_pos = camera.getPos(render)
            cam_quat = camera.getQuat(render)
            camera.reparentTo(self._orig_cam_parent or render)
            camera.setPos(render, cam_pos)
            camera.setQuat(render, cam_quat)
            self._rig_np.removeNode()

        self._rig_np = None
        self._orig_cam_parent = None

    # ---------- input & update ----------
    def _on_key(self, key: str, pressed: bool) -> None:
        self._keys[key] = pressed

    def _update_task(self, task) -> int:
        dt = self._session.clock.getDt() if self._session.clock else 0.0
        win = self._session.win
        render = self._session.render

        if self._rig_np is None:
            return task.cont

        # --- Mouse look -> update eye vector ---
        ptr = win.getPointer(0)
        dx = ptr.getX() - self._center_x
        dy = ptr.getY() - self._center_y
        if dx or dy:
            win.movePointer(0, self._center_x, self._center_y)
            yaw_deg = -dx * self.mouse_sensitivity
            pitch_deg = (dy if self.invert_y else -dy) * self.mouse_sensitivity
            self._apply_yaw_pitch(yaw_deg, pitch_deg)
            self._orient_rig_to_eye()

        # --- Movement in the local (r, f, u_local) frame ---
        move = Vec3(0, 0, 0)
        if self._keys.get("w"):
            move.y += 1.0
        if self._keys.get("s"):
            move.y -= 1.0
        if self._keys.get("a"):
            move.x -= 1.0
        if self._keys.get("d"):
            move.x += 1.0
        if self._keys.get("q"):
            move.z -= 1.0
        if self._keys.get("e"):
            move.z += 1.0

        if move.length_squared() > 0.0:
            move.normalize()
            speed = self.speed_fast if self._keys.get("shift") else self.speed

            r, u, f = self._build_frame_from_eye()
            # NOTE: If you want Q/E to be pure world-up (platformer-style),
            # replace `u` with `self._GLOBAL_UP` in the combination below.
            world_dir = r * move.x + f * move.y + u * move.z

            self._rig_np.setPos(self._rig_np.getPos() + world_dir * (speed * dt))

        return task.cont
