from __future__ import annotations

from rheidos.engine import Engine
from rheidos.views import AxesView, PointVortexStreamFunctionView
from rheidos.controllers import ExitController


def main() -> None:
    positions = [(-1.5, -0.5), (1.2, 0.3), (0.3, 1.4)]
    strengths = [3.0, -2.5, 1.5]

    engine = Engine(window_title="Point Vortex Stream", interactive=False)
    stream_view = PointVortexStreamFunctionView(
        positions=positions,
        strengths=strengths,
        plane_z=0.0,
        resolution=(600, 600),
        margin=0.8,
    )

    engine.add_view(stream_view)
    engine.add_view(AxesView(axis_length=2.0, sort=5))

    cam = engine.session.base.camera
    cam.setPos(0.0, -0.01, 6.0)
    cam.lookAt(0.0, 0.0, 0.0)

    engine.add_controller(ExitController(engine, key="escape"))
    engine.start()


if __name__ == "__main__":
    main()
