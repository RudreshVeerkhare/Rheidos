# Simulation Design

This document captures a modular simulation architecture for Rheidos and a concrete point‑vortex implementation built on Taichi. The point‑vortex piece is an **app**, not core; sim-specific Taichi code and app wiring live under `app/point_vortex/`, while reusable framework pieces stay under `rheidos/`. It is written to be implementation-ready: it spells out interfaces, state flow, store keys, and the step-by-step plan to deliver the feature set (particles, vector fields, color schemes, legends, UI, scene-config integration).

## Intent and Goals
- Add a reusable simulation layer so multiple physics models can plug into views/controllers without tight coupling.
- Ship a point-vortex simulation (Taichi kernels for fast Biot–Savart velocity + integration) with stable stepping and reset.
- Share simulation state across observers, views, controllers, and UI via StoreState with minimal copying.
- Support multiple visualization modes (particles, vector fields on surfaces/planes, scalar/texture overlays) plus color schemes and HUD legends.
- Keep everything togglable from controllers/UI and configurable via scene configs.

## Architectural Approach
- **Simulation interface**: A small `Simulation` protocol/base with `configure(cfg)`, `reset(seed=None)`, `step(dt)`, read-only accessors (`get_positions_view()`, `get_vector_fields()`, `get_scalar_fields()`, metadata for legends/color maps). No Panda3D dependencies here.
- **State container**: `SimulationState` holds immutable config (bounds, core radius, max N) and runtime buffers (Taichi fields, NumPy views, dirty flags).
- **Color schemes**: `ColorScheme` strategy interface with `apply(values_np) -> colors_np` and `legend()` metadata; registry map for selection by name.
- **StoreState contract**: Controllers/UI mutate keys; views/observers subscribe. Keeps UI, input, and rendering in sync with the sim without direct coupling.
- **Stepper location**: An `Observer` runs simulation steps each frame (fixed-step accumulator) so sims progress even when views are disabled. Respects engine pause (`dt <= 0`).
- **Data flow**: Taichi fields → NumPy views (contiguous float32) → mesh/texture setters. Only copy when dirty; reuse buffers to avoid churn.

## App vs Framework Split
- **Framework (reusable, under `rheidos/`)**:
  - Simulation interface/protocols and state container + field registries (no Taichi).
  - ColorScheme interface + registry; HUD Legend view; generic vector/scalar surface views that accept buffers/samplers from any app.
  - StoreState key conventions and helpers; controller/panel patterns that are sim-agnostic.
  - Scene-config plumbing remains generic; apps register their own factories.
- **App (point vortex, under `app/point_vortex/`)**:
  - Taichi fields, kernels, integrator, seeding/reset logic, and sim config.
  - App-specific observer/stepper that owns the point-vortex sim instance.
- Adapters that feed the generic framework views (particles, vector field, scalar/texture overlays) with vortex data.
  - App UI/panel and scene-config factories for point-vortex demos/examples.

## Store Keys (proposed)
- `vortex/dt` (float), `vortex/core_radius` (float), `vortex/viscosity` (float, optional)
- `vortex/pause` (bool), `vortex/reset` (bool trigger), `vortex/device` ("cpu"/"gpu")
- `vortex/color_scheme` (str), `vortex/show_legend` (bool)
- `vortex/show_vector_field` (bool), `vortex/vector_density` (int), `vortex/vector_scale` (float)
- `vortex/show_scalar_field` (bool), `vortex/scalar_resolution` (int|tuple)

Store naming guidance: prefix per sim/app (e.g., `vortex/*`), keep overlay toggles scoped (`*/show_vector_field`, `*/show_scalar_field`), and reserve `*/color_scheme` + `*/show_legend` for legend/color bindings to avoid collisions if multiple sims run together.

## Module Layout (target)
- **Framework (reusable)**
  - `rheidos/sim/base.py`: Simulation protocol, SimulationState, field registries (no Taichi).
  - `rheidos/visualization/color_schemes.py`: ColorScheme interface + built-in schemes + registry.
  - `rheidos/views/legend.py`: HUD legend view driven by ColorScheme metadata.
  - `rheidos/views/vector_field.py`: Generic arrow/hedgehog renderer for vector samples on planes/UV grids.
  - `rheidos/views/scalar_field.py`: Generic texture/colormap surface for scalar fields (can wrap stream/heatmaps).
  - `rheidos/ui/panels/controls_base.py` (optional): helper patterns for store-bound sliders/toggles (sim-agnostic).
- **App (point vortex)**
  - `app/point_vortex/sim.py`: Taichi-backed point-vortex implementation (fields, kernels, config, accessors).
  - `app/point_vortex/observer.py`: `VortexStepObserver` with fixed-step accumulator and store integration.
  - `app/point_vortex/views.py`: Adapters that push sim buffers into framework views (particles via vector field samples, scalar overlays, etc.).
  - `app/point_vortex/panel.py`: ImGui panel binding to vortex store keys (dt, reset, schemes, overlays, density, device).
  - `app/point_vortex/scene_factories.py`: Factories to register app components for scene-config use.
  - `app/point_vortex/main.py`: Scripted demo entry point.
  - `app/point_vortex/scene_configs/point_vortex.yaml`: Declarative config for the app.
  - `docs/simulation-design.md`: This doc (kept up to date with design/plan).

Legacy note: the old `rheidos/views/stream_function.py` helper was removed; scalar overlays now flow through the generic `ScalarFieldView` fed by app-specific adapters.

## Plan

### Requirements
- Reusable simulation interface so multiple physics models (starting with point vortices) share lifecycle, state access, and view/controller hooks.
- Point vortex dynamics implemented in Taichi (fast kernels, GPU/CPU selectable) with stable stepping and reset.
- Shared simulation state exposed to all views/controllers without tight coupling; minimal copies between Taichi fields and Panda3D meshes/textures.
- Multiple render modalities: particles, vector fields on surfaces/planes, scalar/texture overlays, HUD legends.
- Color scheme strategy layer with GUI binding; legends reflect active scheme and are togglable via controllers/UI.
- View/controller toggles and UI checkboxes that stay consistent (store-backed).
- Scene-config compatibility for declarative demos and an example script showing end-to-end usage.

### Scope
- In: new simulation base module, point-vortex Taichi sim, observer stepping, views for particles/vector fields/legends, controllers/UI wiring, scene-config factory entries, docs.
- Out: generalized fluid solvers, persistence, distributed/multi-process runs, advanced PDE solvers beyond point vortices.

### Files and Entry Points
- **Framework**
  - `rheidos/sim/base.py` (new): Simulation interfaces/state containers/registry helpers.
  - `rheidos/visualization/color_schemes.py` (new): ColorScheme interface + built-ins + registry.
  - `rheidos/views/legend.py` (new): HUD legend view driven by ColorScheme metadata.
  - `rheidos/views/vector_field.py` (new): Generic vector field renderer.
  - `rheidos/views/scalar_field.py` (new): Generic scalar/texture view for surfaces.
- **App**
  - `app/point_vortex/sim.py` (new): Taichi fields, kernels, config for point vortices.
  - `app/point_vortex/observer.py` (new): per-frame stepping observer with fixed-step accumulator.
  - `app/point_vortex/views.py` (new): adapters into framework views (particles, vector field, scalar overlay).
  - `app/point_vortex/panel.py` (new): ImGui/Store-driven controls.
  - `app/point_vortex/scene_factories.py` (new): factories for sim, observer, views, legend, panel.
  - `app/point_vortex/main.py` and `app/point_vortex/scene_configs/point_vortex.yaml` (new): runnable demo and declarative config.
  - `docs/simulation-design.md`: this design/plan document.

### Data Model / API Changes
- Define a `Simulation` protocol/base with `configure(cfg)`, `reset(seed=None)`, `step(dt)`, read-only data accessors (`get_positions_view()`, `get_vector_fields()`, `get_scalar_fields()`), and field metadata for legends/color maps and UI selection.
- Introduce `SimulationState` dataclass capturing immutable config (e.g., bounds, core radius, max N) and runtime mutable buffers (NumPy views) with per-buffer dirty flags and runtime shape/dtype validation.
- Document store key conventions (namespaced, app-prefixed) for legend/overlay toggles; no code constants yet.
- Add a `ColorScheme` strategy interface (`apply(values_np) -> colors_np`, `legend()` metadata) plus registry map to switch schemes by name.
- Establish a `VectorFieldSample` helper (positions + vectors + magnitudes) for views to consume uniformly; dirty-flag polling is the baseline observation mechanism.

### Decisions locked in
- Enforce buffer shapes/dtypes at runtime (raise on mismatch rather than warn).
- Store key namespaces are documented only (no shared constants yet).
- Dirty-flag polling is sufficient for now; callbacks/subscriptions can come later if needed.

### Action Items
- **Framework (reusable)**
  - [x] Baseline module layout: create `rheidos/sim/base.py` with `Simulation`, `SimulationState`, and field registries; ensure zero Panda3D/Taichi imports here to keep it lightweight.
  - [x] Color schemes: add `rheidos/visualization/color_schemes.py` with interface, built-in schemes (circulation diverging, speed sequential, categorical), and registry; ensure legend metadata is standardized.
  - [x] Legend view: add `rheidos/views/legend.py` to render a HUD legend driven by ColorScheme metadata; bind visibility to a store key.
  - [x] Generic field views: add `rheidos/views/vector_field.py` (arrow/hedgehog renderer) and `rheidos/views/scalar_field.py` (texture/colormap surface) that accept data providers so any app can feed them.
  - [x] Optional UI helpers: if useful, add `rheidos/ui/panels/controls_base.py` patterns for store-bound sliders/toggles that app panels can reuse.
- **App (point vortex)**
  - [ ] Point-vortex Taichi core: in `app/point_vortex/sim.py`, initialize Taichi lazily (honor user device flags), declare fields (`pos`, `strength`, `vel`, optional `core_radius`), write kernels for pairwise Biot–Savart velocity, optional core regularization, and integration (semi-implicit Euler or RK2). Expose NumPy views (contiguous float32) for render layers and a `mark_dirty()`/`is_dirty()` mechanism.
  - [ ] State/config plumbing: accept config dict/dataclass (N, strengths, bounds, core radius, timestep clamp, seed, periodic vs open boundaries). Add `reset` to reseed positions/strengths deterministically. Provide `set_params` for dt/core radius/viscosity that updates fields safely.
  - [ ] Stepper observer: build `app/point_vortex/observer.py` that (a) accumulates render `dt` into fixed sim steps, (b) respects engine pause (`dt<=0`) and store toggles for per-sim pause, (c) writes back dirty flags and last-step stats (CFL-like metric, max speed) into StoreState.
  - [ ] Rendering data bridges: create a `VortexBuffers` helper in the app that owns reusable NumPy arrays for particles and vector samples; wire Taichi-to-NumPy updates only when dirty. Keep double-buffering if needed to avoid write/read races during rendering.
  - [ ] App views/adapters: in `app/point_vortex/views.py`, feed the framework vector/scalar/legend views (particles via vector samples, scalar overlay adapter). Support size scaling by strength and store-based toggles.
  - [ ] App UI/controllers: add `app/point_vortex/panel.py` (ImGui) that edits store keys for dt, reset, scheme selection, show/hide overlays, sample density, device choice; add lightweight controllers for view group toggles and reset triggers (controllers can stay in app or reuse framework patterns).
  - [ ] Scene-config factories: add `app/point_vortex/scene_factories.py` with factory functions to instantiate the sim, observer, adapters, legend, and panel from YAML keys (`type: vortex_sim`, `type: vortex_vector_field_view`, etc.). Document required/optional keys.
  - [ ] Demo and docs: create `app/point_vortex/main.py` plus `app/point_vortex/scene_configs/point_vortex.yaml` showing toggles, legends, and multiple schemes. Keep `docs/simulation-design.md` updated with any API adjustments.
  - [ ] Validation harness: add a lightweight test script under `app/point_vortex/` to run a short sim headless (no render) asserting energy/impulse conservation within tolerance and verifying color mapping outputs; optionally add a pytest that mocks `ColorScheme.apply` and ensures views honor store toggles.

### Testing and Validation
- Headless Taichi step tests: small N analytic checks (two-vortex rotation period, equal/opp pair translation).
- Render smoke tests: run `python rheidos/examples/point_vortex_sim.py --frames 120` ensuring no exceptions and geometry buffers stay in-bounds.
- Performance sanity: profile CPU vs GPU Taichi backend for N=1k/5k vortices; confirm copy-to-NumPy path stays under budget (log timings via observer).
- UI/controller sync: toggle overlays and color schemes via keyboard and ImGui; verify store keys remain consistent and legends update.
- Scene-config load: run YAML-driven example to confirm factories wire the same setup as the imperative script.

### Risks and Edge Cases
- O(N^2) velocity kernel may be slow for large N; need backend selection or adaptive downsampling for vector fields.
- Singularities if vortices coincide; core radius must regularize and reset should avoid overlapping seeds.
- Copy cost between Taichi and Panda3D buffers; mitigate with reuse/contiguous arrays and update throttling.
- Device init conflicts if Taichi is already initialized elsewhere; lazy init with defensive checks.
- HUD/text rendering may fail if imgui/panel deps missing; guard imports and degrade gracefully.

### Open Questions
- Preferred integrator and stability constraints (Euler vs RK2 vs midpoint) for this use case?
- Should boundaries be periodic, reflective, or free-space by default, and do we need wall-vortex image handling?
- Target backend defaults (CPU vs GPU) and whether to auto-switch based on hardware.
