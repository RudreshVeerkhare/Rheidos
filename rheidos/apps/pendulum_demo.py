from __future__ import annotations

import math
from typing import Mapping, Optional

import numpy as np

from rheidos.engine import Engine
from rheidos.sim.base import (
    FieldMeta,
    FieldRegistry,
    ScalarFieldSample,
    Simulation,
    SimulationState,
    VectorFieldSample,
)
from rheidos.views import LegendView, VectorFieldView, StudioView, AxesView
from rheidos.controllers import FpvCameraController
from rheidos.resources.primitives import cube
from rheidos.visualization.color_schemes import SequentialColorScheme


class PendulumSimulation(Simulation):
    """
    Lightweight planar pendulum simulation that exposes a vector from the pivot
    to the bob as a VectorFieldSample. This keeps the sim decoupled from Panda3D
    and compatible with generic views.
    """

    def __init__(self, name: str = "pendulum") -> None:
        self.name = name
        self.state = SimulationState()
        self.vector_fields: FieldRegistry[VectorFieldSample] = FieldRegistry()
        self.scalar_fields: FieldRegistry[ScalarFieldSample] = FieldRegistry()
        self.theta = 0.0  # radians
        self.omega = 0.0  # rad/s
        self.length = 1.0
        self.gravity = 9.81
        self.damping = 0.02
        self.pivot_height = 1.0
        self._theta0 = 0.0
        self._omega0 = 0.0
        self._register_fields()

    def configure(self, cfg: Optional[Mapping[str, object]] = None) -> None:
        cfg = dict(cfg or {})
        self.length = float(cfg.get("length", 1.25))
        self.gravity = float(cfg.get("gravity", 9.81))
        self.damping = float(cfg.get("damping", 0.02))
        self.pivot_height = float(cfg.get("pivot_height", self.length + 0.1))
        self._theta0 = float(cfg.get("theta0", math.radians(40.0)))
        self._omega0 = float(cfg.get("omega0", 0.0))
        self.theta = self._theta0
        self.omega = self._omega0

    def reset(self, seed: Optional[int] = None) -> None:
        # seed unused; kept for protocol compatibility
        self.theta = self._theta0
        self.omega = self._omega0

    def step(self, dt: float) -> None:
        if dt <= 0:
            return
        # simple damped pendulum: theta'' + (damping)*theta' + (g/L) sin(theta) = 0
        alpha = -(self.gravity / self.length) * math.sin(self.theta) - self.damping * self.omega
        self.omega += alpha * dt
        self.theta += self.omega * dt

    def get_state(self) -> SimulationState:
        return self.state

    def get_positions_view(self) -> np.ndarray:
        # Bob position in XZ plane (Z-up). Pivot is at (0, 0, pivot_height).
        x = self.length * math.sin(self.theta)
        z = self.pivot_height - self.length * math.cos(self.theta)
        return np.array([[x, 0.0, z]], dtype=np.float32)

    def _velocity_field(self) -> VectorFieldSample:
        # Velocity vector anchored at bob position (XZ plane)
        pos = self.get_positions_view()
        vx = self.length * self.omega * math.cos(self.theta)
        vz = self.length * self.omega * math.sin(self.theta)
        vel = np.array([[vx, 0.0, vz]], dtype=np.float32)
        speed = np.array([math.sqrt(vx * vx + vz * vz)], dtype=np.float32)
        sample = VectorFieldSample(positions=pos, vectors=vel, magnitudes=speed)
        sample.validate()
        return sample

    def _register_fields(self) -> None:
        velocity_meta = FieldMeta(
            field_id="bob_velocity",
            label="Bob Velocity",
            units="m/s",
            description="Velocity vector at the pendulum bob",
        )
        self.vector_fields.register(velocity_meta, self._velocity_field, overwrite=True)

    def get_vector_fields(self):
        return self.vector_fields.items()

    def get_scalar_fields(self):
        return self.scalar_fields.items()

    def get_metadata(self):
        return {"length": self.length, "gravity": self.gravity, "damping": self.damping}


from rheidos.abc.observer import Observer


class PendulumObserver(Observer):
    """Observer that advances the pendulum using store-controlled parameters."""

    def __init__(
        self,
        sim: PendulumSimulation,
        store,
        *,
        shaft_nodes=None,
        ball_nodes=None,
        name: str | None = None,
        sort: int = -5,
    ) -> None:
        super().__init__(name=name or "PendulumObserver", sort=sort)
        self.sim = sim
        self.store = store
        self._accum = 0.0
        self._dt_max = 0.02
        self._shaft_nodes = shaft_nodes or ()
        self._ball_nodes = ball_nodes or ()

    def update(self, dt: float) -> None:
        if dt <= 0:
            return
        pendulum = dict(self.store.get("pendulum", {}))
        if pendulum.get("pause", False):
            return
        if pendulum.get("reset", False):
            pendulum["reset"] = False
            self.store.set("pendulum", pendulum)
            self.sim.configure(
                {
                    "length": pendulum.get("length", self.sim.length),
                    "gravity": pendulum.get("gravity", self.sim.gravity),
                    "damping": pendulum.get("damping", self.sim.damping),
                    "theta0": pendulum.get("theta0", self.sim._theta0),
                    "omega0": pendulum.get("omega0", 0.0),
                    "pivot_height": pendulum.get("pivot_height", self.sim.length + 0.1),
                }
            )
            self._accum = 0.0
            return

        target_dt = min(float(pendulum.get("dt", 0.01)), self._dt_max)
        # allow live parameter edits
        self.sim.length = float(pendulum.get("length", self.sim.length))
        self.sim.gravity = float(pendulum.get("gravity", self.sim.gravity))
        self.sim.damping = float(pendulum.get("damping", self.sim.damping))
        self.sim.pivot_height = float(pendulum.get("pivot_height", self.sim.length + 0.1))

        self._accum += dt
        while self._accum >= target_dt:
            self.sim.step(target_dt)
            self._accum -= target_dt

        # Update geometry transforms
        pos = self.sim.get_positions_view()[0]
        x, z = float(pos[0]), float(pos[2])
        mid_x = 0.5 * x
        mid_z = 0.5 * (self.sim.pivot_height + z)
        for node in self._shaft_nodes:
            try:
                node.setPos(mid_x, 0.0, mid_z)
                node.lookAt(x, 0.0, z)
                node.setScale(0.06, self.sim.length, 0.06)
            except Exception:
                pass
        for node in self._ball_nodes:
            try:
                node.setPos(x, 0.0, z)
                node.setScale(0.2)
            except Exception:
                pass


def main() -> None:
    eng = Engine(window_title="Pendulum Demo", interactive=False)

    # Scene: ground + lighting + orientation helper
    studio = StudioView(ground_size=10.0, ground_tiles=12, sort=-50)
    eng.add_view(studio)
    eng.add_view(AxesView(axis_length=2.0, sort=-45))

    # Geometry for shaft and ball (placeholder cubes; swap for sphere/rod meshes if available)
    shaft_prim = cube(size=1.0)
    shaft_surface = shaft_prim.mesh.node_path.copyTo(eng.session.render)
    shaft_surface.setName("shaft")
    shaft_surface.setColor(0.9, 0.9, 0.95, 1.0)
    shaft_surface.setShaderAuto()
    shaft_wire = shaft_prim.mesh.node_path.copyTo(eng.session.render)
    shaft_wire.setName("shaft-wire")
    shaft_wire.setRenderModeWireframe()
    shaft_wire.setColor(0.1, 0.6, 1.0, 1.0)
    shaft_wire.setTwoSided(True)

    ball_prim = cube(size=1.0)
    ball_surface = ball_prim.mesh.node_path.copyTo(eng.session.render)
    ball_surface.setName("ball")
    ball_surface.setColor(1.0, 0.6, 0.3, 1.0)
    ball_surface.setShaderAuto()
    ball_wire = ball_prim.mesh.node_path.copyTo(eng.session.render)
    ball_wire.setName("ball-wire")
    ball_wire.setRenderModeWireframe()
    ball_wire.setColor(0.9, 0.3, 0.1, 1.0)
    ball_wire.setTwoSided(True)

    sim = PendulumSimulation()
    sim.configure({"length": 1.5, "gravity": 9.81, "theta0": math.radians(50), "pivot_height": 1.6})
    observer = PendulumObserver(
        sim,
        eng.store,
        shaft_nodes=(shaft_surface, shaft_wire),
        ball_nodes=(ball_surface, ball_wire),
    )
    eng.add_observer(observer)

    # Views: velocity vector at bob position; color encodes speed magnitude
    scheme = SequentialColorScheme(max_value=1.0)
    velocity_field = sim.get_vector_fields().get("bob_velocity")
    if velocity_field is None:
        raise RuntimeError("PendulumSimulation missing 'bob_velocity' vector field")
    vec_view = VectorFieldView(
        velocity_field,
        color_scheme=scheme,
        scale=1.0,
        thickness=4.0,
        arrow_heads=True,
        arrow_head_length=0.3,
        arrow_head_angle_deg=24.0,
        auto_scale_max_length=0.6,
        auto_color_max=True,
        visible_store_key=("pendulum", "show_vector"),
        store=eng.store,
    )
    legend_view = LegendView(
        scheme_provider=lambda: scheme,
        store=eng.store,
        visible_store_key=("pendulum", "show_legend"),
    )

    eng.add_view(vec_view)
    eng.add_view(legend_view)

    # Camera controls
    eng.add_controller(FpvCameraController(speed=4.0, speed_fast=8.0))

    eng.store.set(
        "pendulum",
        {
            "dt": 0.01,
            "pause": False,
            "reset": False,
            "show_vector": True,
            "show_legend": True,
            "length": sim.length,
            "gravity": sim.gravity,
            "damping": sim.damping,
            "theta0": sim._theta0,
            "omega0": sim._omega0,
            "pivot_height": sim.pivot_height,
        },
    )

    eng.start()


if __name__ == "__main__":
    main()
