# Explanation

This section describes the surviving architecture after the repo cleanup.

## Compute model: resources, decorator producers, modules

The compute layer is a small lazy dataflow system.

Core ideas:
- `ResourceRef`: named state stored in the registry
- `@producer`: a decorated module method that fills one or more resources
- `ModuleBase`: a namespace for related resources and producers
- `World`: the container that owns module instances and the registry

The intended authoring path is:
1. declare resources in a `ModuleBase`
2. mark compute methods with `@producer(...)`
3. call `bind_producers()` in the module constructor

When you call `ref.get()`, the registry checks freshness, ensures dependencies, and runs the bound producer if needed.

## Registry semantics and validation

Each resource tracks:
- a version
- a dependency signature captured at last commit
- an optional `ResourceSpec`

A resource is stale when:
- it has never been committed, or
- any dependency version no longer matches its stored dependency signature

`ResourceSpec` adds runtime validation for:
- kind
- dtype
- vector lanes
- explicit or lazily-resolved shape

This keeps the system lightweight while still catching common mismatches early.

## Modules and scopes

Modules are instantiated with `World.require(...)`.

Important behavior:
- dependencies between modules are discovered dynamically as constructors call `require(...)`
- module cycles are detected during construction
- scopes create independent copies of the same module graph by prefixing resource names

This is the mechanism the active P2 app uses to assemble mesh, vortex, and scalar-space modules.

## Houdini runtime architecture

The Houdini runtime wraps node execution into three main pieces:
- `ComputeRuntime`: session cache keyed by HIP path and node path
- `WorldSession`: persistent per-node state such as `World`, stats, profiler, and logs
- `CookContext`: per-cook access to geometry I/O, session state, time data, and publishing helpers

The supported hand-written entrypoint pattern is the session decorator used in `rheidos/apps/p2/cook_sop.py`.

Generic drivers still exist:
- `run_cook(...)`
- `run_solver(...)`

They remain useful for dynamic script loading and generic solver entrypoints, but the active app surface in this repo is the P2 entrypoint/module stack.

## GeometryIO and CookContext

`GeometryIO` is intentionally NumPy-first:
- reads return NumPy arrays
- writes normalize shapes for Houdini owners
- group reads can return masks or indices
- primitive reads assume fixed arity, triangles by default

`CookContext` builds on that and adds:
- `P()` / `set_P()`
- input and output geometry helpers
- `publish(...)` into the compute world
- `session_access(...)` for other node sessions
- access to runtime state used by the shared `rheidos.logger`

## Taichi and reset behavior

The compute layer still supports `taichi_field` resource kinds and the Houdini runtime still preserves:
- Taichi init/reset helpers
- hard reset hooks
- profiler sampling hooks

Those utilities remain as shared infrastructure even though the old Taichi-heavy app stacks have been removed.
