from __future__ import annotations

from kung_fu_panda.engine import Engine
from kung_fu_panda.views.axes import AxesView
from kung_fu_panda.controllers import PauseController, ScreenshotController


def main() -> None:
    eng = Engine(window_title="Kung Fu Panda — Demo", interactive=False)

    # Add a simple axes view
    eng.add_view(AxesView(axis_length=1.5, sort=0))

    # Controls: space to pause, 's' for screenshot
    eng.add_controller(PauseController(eng, key="space"))
    eng.add_controller(ScreenshotController(eng, key="s", filename="demo_shot.png"))

    # Block in Panda3D loop
    eng.start()


if __name__ == "__main__":
    main()
