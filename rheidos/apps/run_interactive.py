from __future__ import annotations

from rheidos.engine import Engine
from rheidos.views.axes import AxesView
from rheidos.controllers import PauseController, ScreenshotController


def main() -> None:
    eng = Engine(window_title="Rheidos — Interactive", interactive=True, auto_start=False)

    eng.add_view(AxesView(axis_length=1.5, sort=0))
    eng.add_controller(PauseController(eng, key="space"))
    eng.add_controller(ScreenshotController(eng, key="s", filename="interactive_shot.png"))

    print("Interactive loop started. Press Ctrl+C to stop.")
    try:
        eng.start()
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()


if __name__ == "__main__":
    main()
