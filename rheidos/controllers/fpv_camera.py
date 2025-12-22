from __future__ import annotations

from typing import Dict, Optional
from math import sin, cos, radians, degrees, asin, atan2
from panda3d.core import Vec3, WindowProperties, NodePath, Quat
from direct.showbase.MessengerGlobal import messenger

from ..abc.controller import Controller
from ..abc.action import Action


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
        roll_speed: float = 120.0,  # deg/sec for roll keys
        pitch_limit_deg: float = 89.0,
        lock_mouse: bool = False,
        hide_cursor: Optional[bool] = None,
        invert_y: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "FpvCameraController")

        # Tunables
        self.speed = speed
        self.speed_fast = speed_fast
        self.mouse_sensitivity = mouse_sensitivity
        self.roll_speed = roll_speed
        self.pitch_limit_deg = max(1.0, min(89.9, pitch_limit_deg))
        self.lock_mouse = bool(lock_mouse)
        self._hide_cursor = hide_cursor if hide_cursor is not None else self.lock_mouse
        self.invert_y = invert_y

        # Input state
        self._keys: Dict[str, bool] = {}
        self._accepted: list[str] = []
        self._task_name: Optional[str] = None

        # Center for mouse warp
        self._center_x = 0
        self._center_y = 0

        # Mouse capture state
        self._mouse_active = False
        self._relative_mode = False
        self._relative_supported = True
        self._last_mouse_x = 0
        self._last_mouse_y = 0

        # Rig & camera parenting
        self._rig_np: Optional[NodePath] = None
        self._orig_cam_parent: Optional[NodePath] = None

        # World-space forward ("eye") that we maintain
        self._eye_w: Vec3 = Vec3(0, 1, 0)  # set on attach from current camera
        self._GLOBAL_UP = Vec3(0, 0, 1)

        # Euler-like state (stable because yaw is world-up, pitch clamped)
        self._yaw_deg = 0.0
        self._pitch_deg = 0.0
        self._roll_deg = 0.0

        self._enabled = True

    def _ui_wants_mouse(self) -> bool:
        """Check if an ImGui UI (or similar) wants exclusive mouse capture."""
        try:
            from imgui_bundle import imgui

            io = imgui.get_io()
            return bool(io.want_capture_mouse)
        except Exception:
            return False

    def actions(self) -> tuple[Action, ...]:
        return (
            Action(
                id="fpv-enabled",
                label="FPV Enabled",
                kind="toggle",
                group="Camera",
                order=0,
                get_value=lambda session: self._enabled,
                set_value=lambda session, v: self._set_enabled(bool(v)),
                invoke=lambda session, v=None: self._set_enabled(
                    not self._enabled if v is None else bool(v)
                ),
            ),
        )

    def _right_from_yaw(self, yaw_deg: float) -> Vec3:
        """Return a stable right vector for the given yaw (world up is +Z)."""
        yaw_rad = radians(yaw_deg)
        # Heading yaw=0 => forward +Y, right +X
        r = Vec3(cos(yaw_rad), -sin(yaw_rad), 0.0)
        return _safe_normalize(r)

    def _compute_basis(self) -> tuple[Vec3, Vec3, Vec3]:
        """
        Build a right-handed basis (r, u, f) from yaw/pitch/roll:
        - Yaw is around world up (stable near poles)
        - Pitch is around the yawed right axis (clamped)
        - Roll is around the camera forward (flycam feel)
        """
        yaw_rad = radians(self._yaw_deg)
        pitch_rad = radians(self._pitch_deg)

        # Forward derived from yaw/pitch (no roll yet)
        cp = cos(pitch_rad)
        sp = sin(pitch_rad)
        sy = sin(yaw_rad)
        cy = cos(yaw_rad)
        f = Vec3(sy * cp, cy * cp, sp)
        f = _safe_normalize(f)

        # Right is yawed world-right (pitch doesn't affect this axis)
        r = self._right_from_yaw(self._yaw_deg)

        # Up from right-handed cross
        u = r.cross(f)
        u = _safe_normalize(u)

        # Apply roll about forward
        if abs(self._roll_deg) > 0.0:
            q_roll = Quat()
            q_roll.set_from_axis_angle(self._roll_deg, f)
            r = _safe_normalize(q_roll.xform(r))
            u = _safe_normalize(q_roll.xform(u))

        return r, u, f

    def _set_angles_from_forward(
        self, forward_w: Vec3, up_w: Optional[Vec3] = None
    ) -> None:
        """Initialize yaw/pitch/roll from a world-space forward (and optional up)."""
        f = _safe_normalize(forward_w)
        # Yaw/pitch from forward
        self._pitch_deg = degrees(asin(max(-1.0, min(1.0, f.z))))
        self._pitch_deg = max(-self.pitch_limit_deg, min(self.pitch_limit_deg, self._pitch_deg))
        self._yaw_deg = degrees(atan2(f.x, f.y))
        self._roll_deg = 0.0

        # Roll from provided up vector (if any)
        if up_w is not None:
            up_proj = up_w - f * f.dot(up_w)
            if up_proj.length_squared() > 1e-10:
                up_proj.normalize()
                r = self._right_from_yaw(self._yaw_deg)
                base_up = _safe_normalize(r.cross(f))
                # Signed angle between base_up and projected up around forward
                cross_term = base_up.cross(up_proj)
                s = f.dot(cross_term)
                c = base_up.dot(up_proj)
                self._roll_deg = degrees(atan2(s, c))

    def _apply_yaw_pitch(self, yaw_delta: float, pitch_delta: float) -> None:
        """Adjust yaw/pitch, clamp pitch, and refresh basis."""
        self._yaw_deg += yaw_delta
        self._pitch_deg = max(
            -self.pitch_limit_deg, min(self.pitch_limit_deg, self._pitch_deg + pitch_delta)
        )
        self._update_orientation()

    def _apply_roll(self, roll_delta: float) -> None:
        self._roll_deg += roll_delta
        self._update_orientation()

    def _update_orientation(self) -> None:
        """Recompute basis from yaw/pitch/roll, orient rig, and cache forward."""
        rig = self._rig_np
        render = self._session.render
        if rig is None:
            return

        r, u, f = self._compute_basis()
        self._eye_w = f

        pos = rig.getPos(render)
        target = pos + f
        rig.lookAt(render, target, u)

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
        cam_forward = _safe_normalize(cam_q_w.xform(Vec3(0, 1, 0)))
        cam_up = _safe_normalize(cam_q_w.xform(Vec3(0, 0, 1)))
        self._set_angles_from_forward(cam_forward, cam_up)

        # Place rig at camera's world position, and orient from eye
        self._rig_np.setPos(render, camera.getPos(render))
        self._update_orientation()

        # Parent camera under rig, zero local transform
        camera.reparentTo(self._rig_np)
        camera.setPos(0, 0, 0)
        camera.setHpr(0, 0, 0)

        # Mouse capture
        win = session.win
        self._center_x = win.getXSize() // 2
        self._center_y = win.getYSize() // 2
        props = WindowProperties()
        props.setCursorHidden(False)
        props.setMouseMode(WindowProperties.M_absolute)
        win.requestProperties(props)
        self._relative_mode = False
        self._mouse_active = False

        # Keys
        for key in ["w", "s", "a", "d", "q", "e", "z", "c", "shift"]:
            messenger.accept(key, self, self._on_key, [key, True])
            messenger.accept(f"{key}-up", self, self._on_key, [key, False])
            self._accepted.extend([key, f"{key}-up"])

        messenger.accept("mouse1", self, self._on_mouse_down)
        messenger.accept("mouse1-up", self, self._on_mouse_up)
        self._accepted.extend(["mouse1", "mouse1-up"])

        # Per-frame task
        self._task_name = f"fpv-camera-{id(self)}"
        session.task_mgr.add(self._update_task, self._task_name, sort=-50)

    def detach(self) -> None:
        if self._task_name:
            self._session.task_mgr.remove(self._task_name)
            self._task_name = None

        for evt in self._accepted:
            try:
                messenger.ignore(evt, self)
            except Exception:
                pass
        self._accepted.clear()
        self._keys.clear()

        # Release mouse
        self._release_mouse_capture()

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
        if not self._enabled:
            return
        self._keys[key] = pressed

    def _on_mouse_down(self) -> None:
        if not self._enabled:
            return
        if self._ui_wants_mouse():
            return
        win = self._session.win
        if win is None:
            return
        ptr = win.getPointer(0)
        self._center_x = ptr.getX()
        self._center_y = ptr.getY()
        self._acquire_mouse_capture()

    def _on_mouse_up(self) -> None:
        if not self._enabled:
            return
        self._release_mouse_capture()

    def _acquire_mouse_capture(self) -> None:
        if self._mouse_active:
            return
        win = self._session.win
        if win is None:
            return
        props = WindowProperties()
        props.setCursorHidden(self._hide_cursor)
        if self.lock_mouse and self._relative_supported:
            props.setMouseMode(WindowProperties.M_relative)
        win.requestProperties(props)
        current_props = win.getProperties()
        self._relative_mode = (
            self.lock_mouse
            and current_props.getMouseMode() == WindowProperties.M_relative
        )
        self._relative_supported = self._relative_mode if self.lock_mouse else False
        if self._relative_mode:
            win.movePointer(0, 0, 0)
            self._last_mouse_x = 0
            self._last_mouse_y = 0
        elif self.lock_mouse:
            # Absolute fallback when relative is unavailable: recenter to avoid hitting edges
            self._center_x = win.getXSize() // 2
            self._center_y = win.getYSize() // 2
            win.movePointer(0, self._center_x, self._center_y)
        else:
            # Free cursor: track last seen position without warping
            ptr = win.getPointer(0)
            self._last_mouse_x = ptr.getX()
            self._last_mouse_y = ptr.getY()
        self._mouse_active = True

    def _release_mouse_capture(self) -> None:
        if not self._mouse_active:
            # ensure cursor visible even if never activated
            win = self._session.win
            if win is not None:
                props = WindowProperties()
                props.setCursorHidden(False)
                props.setMouseMode(WindowProperties.M_absolute)
                win.requestProperties(props)
            return

        win = self._session.win
        if win is None:
            self._mouse_active = False
            return

        props = WindowProperties()
        props.setCursorHidden(False)
        props.setMouseMode(WindowProperties.M_absolute)
        win.requestProperties(props)
        self._mouse_active = False
        self._relative_mode = False

    def _update_task(self, task) -> int:
        if not self._enabled:
            return task.cont

        dt = self._session.clock.getDt() if self._session.clock else 0.0
        win = self._session.win
        render = self._session.render

        if self._rig_np is None:
            return task.cont

        # Skip capture and movement when UI wants the mouse; release if held.
        if self._ui_wants_mouse():
            self._release_mouse_capture()
            return task.cont

        # --- Mouse look -> update eye vector ---
        if self._mouse_active:
            ptr = win.getPointer(0)
            if self._relative_mode:
                dx = ptr.getX()
                dy = ptr.getY()
                if dx or dy:
                    win.movePointer(0, 0, 0)
            elif self.lock_mouse:
                dx = ptr.getX() - self._center_x
                dy = ptr.getY() - self._center_y
                if dx or dy:
                    win.movePointer(0, self._center_x, self._center_y)
            else:
                dx = ptr.getX() - self._last_mouse_x
                dy = ptr.getY() - self._last_mouse_y
                self._last_mouse_x = ptr.getX()
                self._last_mouse_y = ptr.getY()
            if dx or dy:
                yaw_deg = dx * self.mouse_sensitivity
                pitch_deg = (dy if self.invert_y else -dy) * self.mouse_sensitivity
                self._apply_yaw_pitch(yaw_deg, pitch_deg)

        # --- Roll (keys) ---
        roll_input = 0.0
        if self._keys.get("z"):
            roll_input -= 1.0
        if self._keys.get("c"):
            roll_input += 1.0
        if roll_input != 0.0:
            self._apply_roll(roll_input * self.roll_speed * dt)

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

            r, u, f = self._compute_basis()
            world_dir = r * move.x + f * move.y + u * move.z

            self._rig_np.setPos(self._rig_np.getPos() + world_dir * (speed * dt))

        return task.cont

    def _set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled:
            self._keys.clear()
            self._release_mouse_capture()
