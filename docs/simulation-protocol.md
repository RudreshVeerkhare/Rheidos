# Simulation Protocol and Usage Guide

Reusable contract for simulations that feed Rheidos views/controllers without tight coupling. This is framework-only (no Taichi/Panda3D dependencies inside the protocol).

## Why This Exists
- Keep simulation code portable: sims expose NumPy views and metadata; renderers/controllers stay decoupled from Taichi/Panda3D.
- Make multiple sims swappable: a registry and consistent accessors mean you can plug a new model into existing views/UI.
- Reduce footguns: explicit shape/dtype validation and dirty flags keep buffer handoffs predictable.

## Core Interfaces (rheidos/sim/base.py)
- `Simulation` protocol: `configure(cfg)`, `reset(seed=None)`, `step(dt)`, `get_state()`, `get_positions_view()`, `get_vector_fields()`, `get_scalar_fields()`, `get_metadata()`. Purpose: a tiny lifecycle + data surface so observers/views don’t depend on sim internals.
- `FieldMeta` + `FieldInfo` + `FieldRegistry`: declare field ids/labels/units/descriptions and bind providers to them so controllers/UI can enumerate available vector/scalar fields.
- `SimulationState`: keeps `config`, `buffers`, per-buffer `dirty` flags, and `metadata`. Purpose: one place to store validated buffers and change flags for pull-based rendering. Use `set_buffer(key, value, spec=None)` to validate shape/dtype on write; `mark_dirty(key)` or `mark_all_clean()` as needed.
- `ArraySpec`: runtime shape/dtype contract (e.g., `(None, 3) float32`); `validate(array)` raises `ValueError` on mismatch. Purpose: fail fast when a producer hands off the wrong shape or dtype.
- `VectorFieldSample`: positions `(N,3) float32`, vectors `(N,3) float32`, optional magnitudes `(N,)`; `dirty` flag; `validate()` enforces shape/length. Purpose: uniform handoff to any vector-field view.
- `ScalarFieldSample`: scalar grid `(H,W)` or flattened values; optional `uvs (N,2)`; `dirty` flag; `validate()` enforces consistency. Purpose: standard payload for textures/heatmaps/stream overlays.
- Registry helpers: `register_simulation(name, factory, overwrite=False)`, `create_simulation(name, **kwargs)`, `list_simulations()`. Purpose: late binding so configs or scene files can pick sims by name.

### Validation Rules
- Shapes must match exactly except where dimensions are `None`.
- Dtypes are enforced via `astype(copy=False)` to keep downstream consumers predictable.
- Vector fields require equal lengths for positions/vectors (and magnitudes if provided).
- Scalar fields require matching flattened size if UVs are supplied.

## Color Schemes and Legends (rheidos/visualization/color_schemes.py)
Why: keep colormaps swappable and self-describing so views and HUD legends stay in sync.
- `ColorScheme` protocol: `apply(values_np) -> colors_np (…,4) float32`, `legend() -> ColorLegend`.
- Built-ins: `diverging`, `sequential`, `categorical`. Access via `create_color_scheme(name)`, list via `list_color_schemes()`.
- Metadata: `ColorLegend` (`title`, `ticks`, `stops`, optional `units`/`description`) feeds `LegendView` without hardcoding labels.

## Generic Views
Why: renderers stay generic and consume providers instead of sim internals.
- `VectorFieldView(vector_source, color_scheme="sequential", scale=1.0, thickness=2.0, visible_store_key=None, store=None)`: renders arrows/hedgehogs from a `VectorFieldSample` provider or `FieldInfo`; colors via `ColorScheme`.
- `ScalarFieldView(scalar_source, color_scheme="sequential", frame=(-1,1,-1,1), visible_store_key=None, store=None)`: maps a scalar grid to an RGBA texture on a quad from a provider or `FieldInfo`.
- `LegendView(legend_provider=None, scheme_provider=None, visible_store_key=None, store=None)`: HUD legend from scheme metadata (imgui).
- Views use dirty flags on samples for pull-based updates; visibility can be store-controlled.

## Store Conventions
Why: keep UI/controllers/views decoupled and prevent key collisions.
- Namespaced keys per sim/app (e.g., `vortex/*`). Suggested keys for overlays: `*/show_vector_field`, `*/show_scalar_field`, `*/color_scheme`, `*/show_legend`, `*/vector_scale`, `*/scalar_resolution`, `*/pause`, `*/reset`, `*/dt`.
- Constants live in docs; no shared code constants yet.

## Minimal Example: Dummy Simulation (multi-field)
Purpose: small simulation that exposes multiple vector fields and a scalar field with metadata for UI/legend selection.
```python
import numpy as np
from rheidos.sim.base import (
    FieldMeta,
    FieldRegistry,
    ScalarFieldSample,
    Simulation,
    SimulationState,
    VectorFieldSample,
)

class DummySim(Simulation):
    def __init__(self):
        self.name = "dummy"
        self.state = SimulationState()
        self.vector_fields = FieldRegistry[VectorFieldSample]()
        self.scalar_fields = FieldRegistry[ScalarFieldSample]()
        self._t = 0.0
        self._register_fields()

    def configure(self, cfg=None): pass
    def reset(self, seed=None): self._t = 0.0

    def step(self, dt: float):  # simple rotate
        self._t += dt

    def get_state(self): return self.state

    def get_positions_view(self):
        r = 1.0
        pts = np.array([[r * np.cos(self._t), r * np.sin(self._t), 0.0]], dtype=np.float32)
        return pts

    def _tangent_velocity(self):
        pts = self.get_positions_view()
        vecs = np.array([[-np.sin(self._t), np.cos(self._t), 0.0]], dtype=np.float32)
        sample = VectorFieldSample(positions=pts, vectors=vecs)
        sample.validate()
        return sample

    def _radial_accel(self):
        pts = self.get_positions_view()
        vecs = np.array([[-np.cos(self._t), -np.sin(self._t), 0.0]], dtype=np.float32) * 0.3
        sample = VectorFieldSample(positions=pts, vectors=vecs)
        sample.validate()
        return sample

    def _height_scalar(self):
        grid = np.outer(np.linspace(0, 1, 32, dtype=np.float32), np.ones(32, dtype=np.float32))
        sample = ScalarFieldSample(values=grid)
        sample.validate()
        return sample

    def _register_fields(self):
        self.vector_fields.register(
            FieldMeta(field_id="velocity", label="Velocity", units="m/s", description="Tangent velocity"),
            self._tangent_velocity,
        )
        self.vector_fields.register(
            FieldMeta(field_id="accel", label="Radial Accel", units="m/s^2"),
            self._radial_accel,
        )
        self.scalar_fields.register(
            FieldMeta(field_id="height", label="Height", units="m"),
            self._height_scalar,
        )

    def get_vector_fields(self): return self.vector_fields.items()
    def get_scalar_fields(self): return self.scalar_fields.items()
    def get_metadata(self): return {}
```

### Wiring Into the Engine
```python
from rheidos.engine import Engine
from rheidos.views import VectorFieldView, ScalarFieldView, LegendView
from rheidos.visualization import create_color_scheme

sim = DummySim()
sim.configure()

eng = Engine(window_title="Sim Demo", interactive=False)
scheme = create_color_scheme("diverging")

vel_field = sim.get_vector_fields()["velocity"]
acc_field = sim.get_vector_fields()["accel"]
height_field = sim.get_scalar_fields()["height"]

eng.add_view(VectorFieldView(vel_field,
                             color_scheme=scheme,
                             scale=0.8,
                             visible_store_key="sim/show_velocity",
                             store=eng.store))
eng.add_view(VectorFieldView(acc_field,
                             color_scheme="categorical",
                             scale=1.2,
                             visible_store_key="sim/show_accel",
                             store=eng.store))
eng.add_view(ScalarFieldView(height_field,
                             color_scheme="sequential",
                             frame=(-1,1,-1,1),
                             visible_store_key="sim/show_height",
                             store=eng.store))
eng.add_view(LegendView(scheme_provider=lambda: scheme,
                        store=eng.store,
                        visible_store_key="sim/show_legend"))

eng.store.update(sim_dt=0.02, **{
    "sim/show_velocity": True,
    "sim/show_accel": False,
    "sim/show_height": True,
    "sim/show_legend": True,
})

class Stepper:
    def update(self, dt):
        sim.step(dt)
eng.add_observer(Stepper())  # minimal observer to advance sim
eng.start()
```

## Best Practices
- Enforce shapes/dtypes via `ArraySpec` or `validate()` before exposing buffers to views; fail fast beats silent rendering bugs.
- Keep `Simulation` free of Panda3D/Taichi types in its public accessors; convert to NumPy views so consumers stay lightweight.
- Use store-backed toggles for overlays and dt to keep UI/controllers/views in sync and scene-config-friendly.
- Pad 2D data to 3D (e.g., append zeros for Z) when feeding 3D-oriented views.
- Mark buffers dirty on updates; clear after copies if you add downstream polling so consumers can throttle work.

## Extending
- Register additional sims: `register_simulation("my_sim", lambda **kw: MySim(**kw))`.
- Add custom color schemes: implement `ColorScheme`, then `register_color_scheme("my_scheme", scheme)`.
- Compose new views on top of `VectorFieldSample` / `ScalarFieldSample` providers to stay sim-agnostic.
