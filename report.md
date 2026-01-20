# Rheidos Compute + Houdini Technical Report

## Executive Summary
Rheidos combines a small lazy dataflow compute core with a Houdini runtime layer so Python SOP/Solver nodes can drive Taichi kernels. The compute layer manages resources, producers, and modules with versioned staleness; the Houdini layer wraps cooks into sessions with geometry I/O, publishing, timing, and debugging. The point_vortex app shows the intended pattern (DEC + Poisson + vortex advection) and is experimental.

Targets (not yet benchmarked in this repo):
- interactive cook: <=16.7 ms/frame at 60 FPS for small scenes
- solver step: <=8-12 ms per substep for interactive scrubbing

## Scope, Non-scope, Audience
Scope:
- Compute core (`rheidos/compute`) and Houdini runtime (`rheidos/houdini`) behavior.
- Houdini SOP/Solver templates and the point_vortex example app.

Non-scope:
- Not a general FEM suite or a full reactive ECS.
- Not guaranteed deterministic across all Taichi backends or GPU drivers.

Audience:
- Houdini users integrating interactive simulation.
- Framework developers extending modules and producers.
- Researchers evaluating interactive GPU kernels inside a DCC.

## Requirements and Quality Scenarios
- When geometry topology is unchanged, cooks should reuse cached triangle data and avoid full re-reads.
- When Houdini double-cooks the same frame/substep, the solver should reuse cached output instead of recomputing.
- When a resource is stale, `ResourceRef.get()` must run its producer exactly once and enforce commit/bump rules.
- Missing attributes or shape mismatches should fail fast with clear errors.

## System Context (C4-1)

```
[Houdini Node] --geo--> [CookContext + GeometryIO] --publish--> [World + Registry]
        |                                                    |
        |<----- attrs/out.P (optional) ----------------------|
        +-- user module (cook/step) + Taichi kernels
```

External dependencies: Houdini `hou`, NumPy, and Taichi.

## Architecture Overview (C4-2/3)
Compute core:
- `World` builds module graphs with dynamic dependency discovery and cycle checks.
- `Registry` tracks resources, versions, and dependency signatures to resolve staleness.
- `ResourceSpec` validates numpy and Taichi buffers (dtype, shape, lanes).

Houdini runtime:
- `ComputeRuntime` caches `WorldSession` per HIP path + node path.
- `CookContext` provides geometry I/O, publishing, timing, and session access.
- `publish_geometry_minimal()` publishes `geo.P` and `geo.triangles` with topology caching.
- `run_cook()` / `run_solver()` handle errors, timing, and debug hooks.

Example app:
- Point_vortex modules compute stream function, velocity fields, and RK4 advection.

## Execution Model
- Python SOP cooks are stateless unless you use the runtime session; `run_cook()` builds a `CookContext`, publishes geometry, calls `cook(ctx)`, and applies `out.P` if present.
- Solver SOP cooks are stateful; `run_solver()` publishes `sim.*` keys, runs `setup(ctx)` once, then `step(ctx)` per cook.
- Double-cook guard uses `(frame, substep)` keys and cached snapshots to avoid redundant work.
- `CookContext.dt` derives from `hou.fps()`; `frame/time/substep` come from Houdini.

## Data Model and Memory Layout
- Canonical geometry resources: `geo.P` (Nx3 float32) and `geo.triangles` (Mx3 int64).
- Attribute/group resources can be published with `publish_point_attrib`, `publish_prim_attrib`, `publish_group`.
- Taichi fields follow allocate-before-fill: `set_buffer(..., bump=False)` then `commit()`.
- Cross-world cost is dominated by Houdini GEO -> NumPy -> Taichi transfers; `GeometryIO` caches reads per cook.

## Houdini Integration and Operator Notes
- Templates live in `rheidos/apps/point_vortex/cook_sop.py` and `solver_sop.py`.
- Node config uses `script_path` or `module_path`, plus `reset_node`, `nuke_all`, `profile`, and `debug_log`.
- `nuke_all` clears all sessions and triggers a Taichi hard reset.
- Debugging uses debugpy via `rheidos.houdini.debug` with break-next and ownership controls.

## Performance and Latency
Frame time model:
- Frame time ~= Houdini cook + data marshaling + Taichi kernels + copy-back + viewport update

Instrumentation:
- Enable `profile` to record per-stage timings in `session.stats['timings']`.
- Timing labels include `publish_geometry`, `user_cook`, `user_setup`, `user_step`, `apply_output`, `apply_snapshot`.

Measurement method (recommended):
- Warm up once to pay Taichi compilation; measure steady-state after 10+ cooks.
- Record Houdini version, Taichi backend, GPU/CPU, nV/nF, and nVortices.
- Time geometry publishing separately to isolate marshaling cost.

Results:
- No benchmark outputs are stored in this repo. Use the above instrumentation to populate results for your target scenes.

## Correctness and Failure Modes
- `ResourceSpec` enforces dtype/shape checks; invalid buffers fail at commit time.
- `GeometryIO.read_prims()` requires polygon prims with fixed arity; mixed prims hard-fail.
- Changing user scripts without reset is blocked to avoid stale state.

## Extending and Debugging
- Add solvers as `ModuleBase` + `ProducerBase` with `ResourceSpec` and explicit dependencies.
- Use `World.require()` to instantiate modules and `Registry.explain()` to trace staleness.
- Use `session.log_entries` for diagnostics and debugpy for interactive debugging.

## Reproducibility Checklist
- Pin versions in `pyproject.toml` and record the Taichi backend and GPU driver.
- Save a minimal HIP scene wiring the SOP/Solver template with inputs.
- Capture `session.stats` and `session.log_entries` alongside benchmark numbers.

## Glossary
- Cook: A Houdini evaluation of a node.
- Resource: A named buffer in the compute registry.
- Producer: Code that fills resources and commits versions.
- WorldSession: Per-node cache holding the compute world and diagnostics.
