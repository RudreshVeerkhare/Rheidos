# Point Vortex Simulation Tutorial (Taichi + Rheidos Framework)

Incremental, runnable steps for a point-vortex sim: scene config drives geometry/picking; code handles all simulation work; point selector seeds vortices; velocity and scalar stream overlays are viewable; kernels are swappable (Euler/RK4) and DEC can plug in later.

## Store keys to use throughout
`vortex/dt`, `vortex/pause`, `vortex/reset`, `vortex/device`, `vortex/show_vector_field`, `vortex/show_scalar_field`, `vortex/show_legend`, `vortex/vector_scale`, `vortex/scalar_resolution`, `vortex/seed_points`, `vortex/integrator`, `vortex/enable_dec`.

## Step 0 — Env prep (once)
- `pip install -e .` (Taichi, Panda3D, imgui included). For GPU: set `TI_ARCH=ti.gpu`.

## Step 1 — Scene config (geometry + picking)
Declare pickable surface + camera/lights in `app/point_vortex/scene_configs/vortex.yaml`:
```yaml
meshes:
  - path: ../../../models/plane_2x2.obj
    name: domain
    pickable: true
    surface: true
    material: {diffuse: [0.8, 0.82, 0.9, 1.0]}
camera: {auto_frame: true}
studio: {enabled: true}
app:
  bounds: [[-2, -2], [2, 2]]
  N: 32
```
Smoke test: `python -m rheidos.examples.point_selection --scene-config app/point_vortex/scene_configs/vortex.yaml`.

## Step 2 — Minimal loader (no sim yet)
`app/point_vortex/tutorial.py`:
```python
from pathlib import Path
from rheidos.engine import Engine
from rheidos.scene_config import load_scene_from_config

def main():
    cfg_path = Path("app/point_vortex/scene_configs/vortex.yaml")
    eng = Engine(window_title="Vortex Tutorial", interactive=False)
    cfg = load_scene_from_config(eng, cfg_path)
    eng.start()

if __name__ == "__main__":
    main()
```
Run to confirm the config loads.

## Step 3 — Add point selection (seed capture)
Append to `tutorial.py`:
```python
from panda3d.core import BitMask32
from rheidos.views import PointSelectionView
from rheidos.controllers import SceneSurfacePointSelector

markers = PointSelectionView(name="seed_markers")
eng.add_view(markers)
eng.add_controller(
    SceneSurfacePointSelector(
        engine=eng,
        pick_mask=BitMask32.bit(4),
        markers_view=markers,
        store_key="vortex/seed_points",
    )
)
```
Run, click the surface; `vortex/seed_points` fills with picks.

## Step 4 — Core simulation (`app/point_vortex/sim.py`)
Taichi handles Biot–Savart velocity, Euler integration, stream function; public API stays NumPy for views.
```python
import math, numpy as np, taichi as ti
from typing import Mapping, Optional
from rheidos.sim.base import FieldMeta, FieldRegistry, ScalarFieldSample, Simulation, SimulationState, VectorFieldSample

class PointVortexSimulation(Simulation):
    def __init__(self, integrator="euler"):
        self.name = "point_vortex"
        self.state = SimulationState()
        self.vector_fields = FieldRegistry[VectorFieldSample]()
        self.scalar_fields = FieldRegistry[ScalarFieldSample]()
        self._ti_initialized = False
        self._N = 0
        self._integrator = integrator

    def configure(self, cfg: Optional[Mapping[str, object]] = None):
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
        self._register_fields()
        self.reset(seed=cfg.get("seed"), positions_override=cfg.get("positions"))

    def _build_fields(self):
        N = self._N
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.strength = ti.field(dtype=ti.f32, shape=N)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.stream = ti.field(dtype=ti.f32, shape=(256, 256))

    @ti.kernel
    def _compute_velocity(self):
        for i in range(self._N):
            v = ti.Vector([0.0, 0.0])
            pi = self.pos[i]
            for j in range(self._N):
                if i == j: continue
                r = pi - self.pos[j]
                r2 = r.norm_sqr() + self.cfg["core_radius"] ** 2
                v += self.strength[j] * ti.Vector([-r.y, r.x]) / r2
            self.vel[i] = v / (2 * math.pi)

    @ti.kernel
    def _integrate_euler(self, dt: ti.f32):
        for i in range(self._N):
            self.pos[i] += dt * self.vel[i]

    @ti.kernel
    def _compute_stream(self):
        for i, j in self.stream:
            mins = ti.Vector(self.cfg["bounds"][0]); maxs = ti.Vector(self.cfg["bounds"][1])
            xy = mins + (maxs - mins) * ti.Vector([i, j]) / ti.Vector(self.stream.shape)
            psi = 0.0
            for k in range(self._N):
                r = xy - self.pos[k]
                psi += -self.strength[k] * ti.atan2(r.y, r.x) / (2 * math.pi)
            self.stream[i, j] = psi

    def _integrate(self, dt: float):
        self._compute_velocity()
        self._integrate_euler(dt)  # swap to RK4 in Step 7
        self._compute_stream()

    def reset(self, seed=None, positions_override=None):
        rng = np.random.default_rng(seed)
        bounds = self.cfg["bounds"]
        if positions_override is not None:
            arr = np.asarray(positions_override, dtype=np.float32)
            pos_np = arr[: self._N, :2]
        else:
            pos_np = rng.uniform(bounds[0], bounds[1], size=(self._N, 2)).astype(np.float32)
        self.pos.from_numpy(pos_np)
        self.strength.from_numpy(rng.uniform(-1.0, 1.0, size=(self._N,)).astype(np.float32))
        self.state.dirty = {"positions": True, "vectors": True, "scalar": True}

    def step(self, dt: float):
        if dt <= 0: return
        self._integrate(dt)
        self.state.dirty = {"positions": True, "vectors": True, "scalar": True}

    def get_state(self): return self.state
    def get_positions_view(self):
        return np.hstack([self.pos.to_numpy(), np.zeros((self._N, 1), dtype=np.float32)])

    def _velocity_field(self):
        pos2d, vel2d = self.pos.to_numpy(), self.vel.to_numpy()
        zeros = np.zeros((self._N, 1), dtype=np.float32)
        sample = VectorFieldSample(
            positions=np.hstack([pos2d, zeros]),
            vectors=np.hstack([vel2d, zeros]),
        )
        sample.validate(); return sample

    def _stream_field(self):
        values = self.stream.to_numpy().astype(np.float32)
        sample = ScalarFieldSample(values=values); sample.validate(); return sample

    def _register_fields(self):
        self.vector_fields.register(FieldMeta("velocity", "Velocity", units="m/s"), self._velocity_field, overwrite=True)
        self.scalar_fields.register(FieldMeta("stream", "Stream Function"), self._stream_field, overwrite=True)

    def get_vector_fields(self): return self.vector_fields.items()
    def get_scalar_fields(self): return self.scalar_fields.items()
    def get_metadata(self): return {"bounds": self.cfg["bounds"]}
```
Headless smoke test: `python - <<'PY'\nfrom app.point_vortex.sim import PointVortexSimulation as S\ns=S(); s.configure({\"N\":4}); s.step(0.01); print(s.get_positions_view()[:2])\nPY`

## Step 5 — Observer + seeds from selector
Add to `tutorial.py` (after sim creation):
```python
from rheidos.abc.observer import Observer
from app.point_vortex.sim import PointVortexSimulation

sim = PointVortexSimulation()
sim.configure({"N": 32, "bounds": cfg["app"]["bounds"]})

class VortexStepper(Observer):
    def __init__(self, sim, store):
        super().__init__("VortexStepper", sort=-5)
        self.sim, self.store, self.accum = sim, store, 0.0
    def update(self, dt):
        if dt <= 0: return
        if self.store.get("vortex/pause", False): return
        if self.store.get("vortex/reset", False):
            seeds = self.store.get("vortex/seed_points")
            sim.reset(positions_override=seeds)
            self.store.set("vortex/reset", False)
        target = float(self.store.get("vortex/dt", 0.01))
        self.accum += dt
        while self.accum >= target:
            sim.step(target); self.accum -= target

eng.add_observer(VortexStepper(sim, eng.store))
eng.store.update(**{"vortex/dt": 0.01, "vortex/reset": False})
```
Click to select points, then set `eng.store.set("vortex/reset", True)` (console or panel) to reseed from picks.

## Step 6 — Views for velocity + stream overlays
Add views after the stepper:
```python
from rheidos.visualization import create_color_scheme
from rheidos.views import VectorFieldView, ScalarFieldView, LegendView

vel = sim.get_vector_fields()["velocity"]
psi = sim.get_scalar_fields()["stream"]
scheme = create_color_scheme("diverging")

eng.add_view(VectorFieldView(
    vel, color_scheme=scheme, scale=0.6,
    visible_store_key="vortex/show_vector_field", store=eng.store))
eng.add_view(ScalarFieldView(
    psi, color_scheme="sequential", frame=(-2, 2, -2, 2),
    visible_store_key="vortex/show_scalar_field", store=eng.store))
eng.add_view(LegendView(
    scheme_provider=lambda: scheme,
    visible_store_key="vortex/show_legend", store=eng.store))

eng.store.update(**{
    "vortex/show_vector_field": True,
    "vortex/show_scalar_field": True,
    "vortex/show_legend": True,
})
```
Run `python app/point_vortex/tutorial.py`; you should see arrows plus stream texture.

## Step 7 — Swappable kernels (Euler ↔ RK4)
- Add a registry in `PointVortexSimulation`: `self._integrators = {"euler": self._integrate_euler, "rk4": self._integrate_rk4}` and a `set_integrator(name)` helper.
- Implement `_integrate_rk4` with temporary position buffers; reuse `_compute_velocity` at k1..k4 stages.
- In `VortexStepper.update`, read `vortex/integrator` from the store and call `sim.set_integrator(...)` before stepping.
- Default to Euler; switch via `eng.store.set("vortex/integrator", "rk4")`.

## Step 8 — DEC-friendly extension (modular, fast)
- Keep the point-vortex kernels separate from DEC utilities. Add `app/point_vortex/dec.py` with a `DECGrid` that owns Taichi fields for 0/1/2-forms and precomputed incidence matrices.
- Expose DEC ops (`grad`, `curl`, `apply_hodge`) as Taichi funcs; gate execution behind `vortex/enable_dec`.
- Register DEC outputs as extra fields (e.g., `dec_stream`) in `FieldRegistry`; they become selectable in `ScalarFieldView` without changing the main sim.
- Performance: reuse Taichi fields, avoid host copies unless pushing a `ScalarFieldSample`/`VectorFieldSample`.

## Step 9 — Run and validate
- End-to-end: `python app/point_vortex/tutorial.py` (optionally add argparse to pass `--scene-config`).
- Select seeds → set `vortex/reset` to reseed from picks → watch velocity arrows + stream overlay.
- Headless check already shown in Step 4; add pytest that calls `sim.step` for a few frames and asserts `np.isfinite` on positions/fields.
- UI polish: use `StoreBoundControls` for toggles/sliders (dt, integrator, overlays, DEC enable, vector scale).
