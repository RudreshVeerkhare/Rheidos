from __future__ import annotations

"""
Demo: Render a mesh with a GUI+shortcut toggle to show/hide it.
 - Shortcut: 'v' toggles visibility.
 - GUI: panel auto-populates with a toggle button via controller actions.
"""

from panda3d.core import loadPrcFileData
loadPrcFileData("", "want-tk 0")


from rheidos.engine import Engine
from rheidos.views import MeshSurfaceView, AxesView, StudioView
from rheidos.resources import cube
from rheidos.controllers import FpvCameraController
from rheidos.abc.controller import Controller
from rheidos.abc.action import Action


class MeshVisibilityController(Controller):
    def __init__(self, engine: Engine, view_name: str = "mesh", name: str | None = None) -> None:
        super().__init__(name=name or "MeshVisibilityController")
        self.engine = engine
        self.view_name = view_name
        self.ui_order = -10  # keep at top of the panel

    def _current_state(self) -> bool:
        view = self.engine._views.get(self.view_name)  # type: ignore[attr-defined]
        if view is None:
            return False
        return bool(getattr(view, "_enabled", getattr(view, "enabled", False)))

    def _set_state(self, visible: bool) -> None:
        self.engine.enable_view(self.view_name, bool(visible))

    def actions(self) -> tuple[Action, ...]:
        toggle = Action(
            id="toggle-mesh",
            label="Mesh Visible",
            kind="toggle",
            group="Views",
            order=0,
            shortcut="v",
            get_value=lambda session: self._current_state(),
            set_value=lambda session, value: self._set_state(bool(value)),
            invoke=lambda session, value=None: self._set_state(
                not self._current_state() if value is None else bool(value)
            ),
        )
        return (toggle,)


def main() -> None:
    eng = Engine(window_title="Rheidos — GUI Mesh Toggle", interactive=False)

    # Build a simple mesh and views
    primitive = cube(size=2.0, name="cube")
    eng.add_view(StudioView(apply_material_to=primitive.mesh))
    eng.add_view(AxesView(axis_length=1.5, sort=-5))
    eng.add_view(MeshSurfaceView(primitive.mesh, name="mesh", sort=0))

    # Controls: FPV camera and mesh visibility toggle (GUI + shortcut)
    eng.add_controller(FpvCameraController(speed=6.0, speed_fast=12.0))
    eng.add_controller(MeshVisibilityController(eng, view_name="mesh"))

    eng.start()


if __name__ == "__main__":
    main()
