# Point Vortex Simulation Tutorial (Taichi + Rheidos Framework)

End-to-end guide for building a Taichi-backed point-vortex simulation that conforms to the `Simulation` protocol, feeds `VectorFieldView` and `ScalarFieldView` for velocity and stream function overlays, and uses point selection for pre-seeding vortices.

## 1) Define Store Keys
- Namespaced keys keep UI/views decoupled: `vortex/dt`, `vortex/pause`, `vortex/reset`, `vortex/color_scheme`, `vortex/show_vector_field`, `vortex/show_scalar_field`, `vortex/vector_scale`, `vortex/scalar_resolution`, `vortex/seed_points`.
- Document conventions only (no shared constants yet); reserve `*/color_scheme` and `*/show_legend` for legend bindings.

## 2) Implement the Simulation (`app/point_vortex/sim.py`)
```python
import math
from typing import Mapping, Optional
import numpy as np
import taichi as ti

from rheidos.sim.base import Simulation, SimulationState, VectorFieldSample, ScalarFieldSample

class PointVortexSimulation(Simulation):
    def __init__(self, name: str = "point_vortex") -> None:
        self.name = name
        self.cfg = {}
        self.state = SimulationState()
        self._ti_initialized = False
        self._N = 0

    def configure(self, cfg: Optional[Mapping[str, object]] = None) -> None:
        cfg = dict(cfg or {})
        self.cfg = {
            "N": int(cfg.get("N", 16)),
            "bounds": np.array(cfg.get("bounds", [[-2, -2], [2, 2]]), dtype=np.float32),
            "core_radius": float(cfg.get("core_radius", 1e-3)),
            "device": cfg.get("device", "cpu"),
        }
        self._N = self.cfg["N"]
        if not self._ti_initialized:
            ti.init(arch=ti.gpu if self.cfg["device"] == "gpu" else ti.cpu)
            self._ti_initialized = True
        self._build_fields()
        self.reset(seed=cfg.get("seed"))

    def _build_fields(self) -> None:
        N = self._N
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.strength = ti.field(dtype=ti.f32, shape=N)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.stream = ti.field(dtype=ti.f32, shape=(256, 256))

    @ti.kernel
    def _compute_velocity(self) -> None:
        for i in range(self._N):
            v = ti.Vector([0.0, 0.0])
            pi = self.pos[i]
            for j in range(self._N):
                if i == j:
                    continue
                pj = self.pos[j]
                r = pi - pj
                r2 = r.norm_sqr() + self.cfg["core_radius"] ** 2
                v += self.strength[j] * ti.Vector([-r.y, r.x]) / r2
            self.vel[i] = v / (2 * math.pi)

    @ti.kernel
    def _integrate(self, dt: ti.f32) -> None:
        for i in range(self._N):
            self.pos[i] += dt * self.vel[i]

    @ti.kernel
    def _compute_stream(self) -> None:
        for i, j in self.stream:
            mins = ti.Vector(self.cfg["bounds"][0])
            maxs = ti.Vector(self.cfg["bounds"][1])
            xy = mins + (maxs - mins) * ti.Vector([i, j]) / ti.Vector(self.stream.shape)
            psi = 0.0
            for k in range(self._N):
                r = xy - self.pos[k]
                psi += -self.strength[k] * ti.atan2(r.y, r.x) / (2 * math.pi)
            self.stream[i, j] = psi

    def reset(self, seed: Optional[int] = None, positions_override=None) -> None:
        bounds = self.cfg["bounds"]
        rng = np.random.default_rng(seed)
        if positions_override is not None:
            arr = np.asarray(positions_override, dtype=np.float32)
            if arr.shape[0] < self._N or arr.shape[1] < 2:
                raise ValueError("positions_override must be (>=N, 2)")
            pos_np = arr[: self._N, :2]
        else:
            pos_np = rng.uniform(bounds[0], bounds[1], size=(self._N, 2)).astype(np.float32)
        self.pos.from_numpy(pos_np)
        self.strength.from_numpy(rng.uniform(-1.0, 1.0, size=(self._N,)).astype(np.float32))
        self.state.dirty = {"positions": True, "vectors": True, "scalar": True}

    def step(self, dt: float) -> None:
        if dt <= 0:
            return
        self._compute_velocity()
        self._integrate(dt)
        self._compute_stream()
        self.state.dirty = {"positions": True, "vectors": True, "scalar": True}

    def get_state(self) -> SimulationState:
        return self.state

    def get_positions_view(self):
        # padded to 3D for view consumption
        return np.hstack([self.pos.to_numpy(), np.zeros((self._N, 1), dtype=np.float32)])

    def get_vectors_view(self) -> Optional[VectorFieldSample]:
        pos2d = self.pos.to_numpy().astype(np.float32)
        vel2d = self.vel.to_numpy().astype(np.float32)
        zeros = np.zeros((self._N, 1), dtype=np.float32)
        positions = np.hstack([pos2d, zeros])
        vectors = np.hstack([vel2d, zeros])
        sample = VectorFieldSample(positions=positions, vectors=vectors)
        sample.validate()
        return sample

    def get_scalar_view(self) -> Optional[ScalarFieldSample]:
        values = self.stream.to_numpy().astype(np.float32)
        sample = ScalarFieldSample(values=values)
        sample.validate()
        return sample

    def get_metadata(self):
        return {"bounds": self.cfg["bounds"], "core_radius": self.cfg["core_radius"]}
```
Notes: replace the naive stream calculation with your formulation if needed; positions/vectors are padded to 3D for view compatibility.

## 3) Add a Stepper Observer (`app/point_vortex/observer.py`)
```python
from rheidos.abc.observer import Observer

class VortexStepObserver(Observer):
    def __init__(self, sim, store, name=None, sort=-5):
        super().__init__(name=name or "VortexStepObserver", sort=sort)
        self.sim = sim
        self.store = store
        self._accum = 0.0
        self._dt_max = 0.02  # clamp if desired

    def update(self, dt: float) -> None:
        if dt <= 0:
            return
        if self.store.get("vortex/pause", False):
            return
        if self.store.get("vortex/reset", False):
            self.store.set("vortex/reset", False)
            self.sim.reset()
        target_dt = min(float(self.store.get("vortex/dt", 0.01)), self._dt_max)
        self._accum += dt
        while self._accum >= target_dt:
            self.sim.step(target_dt)
            self._accum -= target_dt
```

## 4) Providers for Views (`app/point_vortex/views.py`)
```python
from rheidos.sim.base import VectorFieldSample, ScalarFieldSample

def make_vector_provider(sim):
    def _provider():
        return sim.get_vectors_view()
    return _provider

def make_scalar_provider(sim):
    def _provider():
        return sim.get_scalar_view()
    return _provider
```

## 5) Wire Up Engine, Views, and Legend (`app/point_vortex/main.py`)
```python
from rheidos.engine import Engine
from rheidos.views import VectorFieldView, ScalarFieldView, LegendView
from rheidos.visualization import create_color_scheme
from app.point_vortex.sim import PointVortexSimulation
from app.point_vortex.observer import VortexStepObserver
from app.point_vortex.views import make_vector_provider, make_scalar_provider

def main():
    eng = Engine(window_title="Point Vortex", interactive=False)
    sim = PointVortexSimulation()
    sim.configure({"N": 32, "device": "cpu"})
    observer = VortexStepObserver(sim, eng.store)
    eng.add_observer(observer)

    scheme = create_color_scheme("diverging")
    vec_view = VectorFieldView(
        make_vector_provider(sim),
        color_scheme=scheme,
        scale=0.6,
        visible_store_key="vortex/show_vector_field",
        store=eng.store,
    )
    scal_view = ScalarFieldView(
        make_scalar_provider(sim),
        color_scheme="sequential",
        frame=(-2, 2, -2, 2),
        visible_store_key="vortex/show_scalar_field",
        store=eng.store,
    )
    legend_view = LegendView(
        scheme_provider=lambda: scheme,
        store=eng.store,
        visible_store_key="vortex/show_legend",
    )

    eng.add_view(vec_view)
    eng.add_view(scal_view)
    eng.add_view(legend_view)

    eng.store.update(
        **{
            "vortex/dt": 0.01,
            "vortex/pause": False,
            "vortex/show_vector_field": True,
            "vortex/show_scalar_field": True,
            "vortex/show_legend": True,
        }
    )

    eng.start()

if __name__ == "__main__":
    main()
```

## 6) Point Selection Before Stepping
Use the built-in selector to gather seed points, then pass them into `reset`.
```python
from panda3d.core import BitMask32
from rheidos.views import MeshSurfaceView, PointSelectionView
from rheidos.controllers import SceneSurfacePointSelector
from rheidos.resources.primitives import cube  # or a flat plane mesh

prim = cube(size=4.0)  # replace with a plane if available
surface = MeshSurfaceView(prim.mesh, collide_mask=BitMask32.bit(4))
markers = PointSelectionView()

eng.add_view(surface)
eng.add_view(markers)
eng.add_controller(
    SceneSurfacePointSelector(
        engine=eng,
        pick_mask=BitMask32.bit(4),
        select_button="mouse1",
        clear_shortcut="c",
        markers_view=markers,
        store_key="vortex/seed_points",
    )
)
```
After selection, before stepping:
```python
seed_pts = eng.store.get("vortex/seed_points")
sim.reset(positions_override=seed_pts)
```

## 7) Optional ImGui Controls
Bind imgui widgets to store keys with `StoreBoundControls`:
```python
from rheidos.ui.panels.controls_base import StoreBoundControls

class VortexPanel:
    id = "vortex-panel"
    title = "Vortex Controls"
    order = 10
    separate_window = False

    def __init__(self, store):
        self._ctl = StoreBoundControls(store)

    def draw(self, imgui):
        self._ctl.checkbox(imgui, "Show vectors", "vortex/show_vector_field", True)
        self._ctl.checkbox(imgui, "Show scalar", "vortex/show_scalar_field", True)
        self._ctl.slider_float(imgui, "Vector scale", "vortex/vector_scale", 0.1, 2.0, 1.0)
        self._ctl.slider_float(imgui, "dt", "vortex/dt", 0.001, 0.05, 0.01)
```
Register the panel via `Engine(imgui_panel_factories=(lambda session, store: VortexPanel(store),))`.

## 8) Run and Validate
- Install deps: `pip install -e .` (ensure `taichi`, `panda3d`, `imgui-bundle`, `p3dimgui`).
- Run `python app/point_vortex/main.py`.
- (Optional) Select seed points, then call `sim.reset(positions_override=eng.store.get("vortex/seed_points"))`.
- Toggle overlays and dt via store or the ImGui panel.

## What Changed
- Uses the new `Simulation` protocol (`rheidos/sim/base.py`) and generic views (`VectorFieldView`, `ScalarFieldView`, `LegendView`).
- Stream/heatmap overlays flow through `ScalarFieldView`; the legacy `stream_function.py` helper was removed.
