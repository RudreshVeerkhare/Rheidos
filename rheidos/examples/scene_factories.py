"""
Small demo factories for YAML-driven views/controllers.
Referenced by rheidos/examples/scene_configs/point_selection.yaml.
"""

from __future__ import annotations

from panda3d.core import Vec4

from rheidos.controllers import PauseController, ScreenshotController
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
