"""
Small demo factories for YAML-driven views/controllers.
Referenced by rheidos/examples/scene_configs/point_selection.yaml.
"""

from __future__ import annotations

from panda3d.core import Vec4

from rheidos.controllers import (
    PauseController,
    ScreenshotController,
    FpvCameraController,
    ExitController,
)
from rheidos.views import OrientationGizmoView


def make_orientation_gizmo(engine, config):
    """
    Build an OrientationGizmoView using YAML config keys:
      size, margin, margin_x, margin_y, thickness, fov_deg, inner_margin, anchor, sort
    """
    return OrientationGizmoView(
        name=config.get("name"),
        sort=int(config.get("sort", 1000)),
        size=float(config.get("size", 0.16)),
        margin=float(config.get("margin", 0.02)),
        margin_x=config.get("margin_x"),
        margin_y=config.get("margin_y"),
        thickness=float(config.get("thickness", 3.0)),
        fov_deg=float(config.get("fov_deg", 28.0)),
        inner_margin=float(config.get("inner_margin", 0.08)),
        anchor=config.get("anchor", "top-left"),
    )


def make_screenshot_controller(engine, config):
    """
    Build a ScreenshotController using YAML config keys:
      key (default 'o'), filename (default 'config_shot.png'), name (optional)
    """
    return ScreenshotController(
        engine=engine,
        key=config.get("key", "o"),
        filename=config.get("filename", "config_shot.png"),
        name=config.get("name", "ConfigScreenshot"),
    )


def make_pause_controller(engine, config):
    """
    Build a PauseController using YAML config keys:
      key (default 'space'), name (optional)
    """
    return PauseController(
        engine=engine,
        key=config.get("key", "space"),
        name=config.get("name", "ConfigPause"),
    )


def make_fpv_camera_controller(engine, config):
    """
    Build an FpvCameraController using YAML config keys:
      speed, speed_fast, mouse_sensitivity, roll_speed, pitch_limit_deg,
      lock_mouse, hide_cursor, invert_y, name
    """
    return FpvCameraController(
        speed=float(config.get("speed", 6.0)),
        speed_fast=float(config.get("speed_fast", 12.0)),
        mouse_sensitivity=float(config.get("mouse_sensitivity", 0.15)),
        roll_speed=float(config.get("roll_speed", 120.0)),
        pitch_limit_deg=float(config.get("pitch_limit_deg", 89.0)),
        lock_mouse=bool(config.get("lock_mouse", False)),
        hide_cursor=config.get("hide_cursor"),
        invert_y=bool(config.get("invert_y", False)),
        name=config.get("name"),
    )


def make_exit_controller(engine, config):
    """
    Build an ExitController using YAML config keys:
      key (default 'escape'), name (optional)
    """
    return ExitController(
        engine=engine,
        key=config.get("key", "escape"),
        name=config.get("name", "ConfigExit"),
    )
