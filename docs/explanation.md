# Explanation

This section provides background and rationale for the compute and Houdini layers.

## Compute model: resources, producers, modules

The compute module is a small, explicit dataflow system. It is not a full reactive
framework or an ECS. The goal is to keep data flow visible while still getting
lazy, on-demand computation.

- Resource: a named piece of state stored in the registry.
- Producer: code that fills one or more resources.
- Module: a namespace that groups related resources and producers.

When you call `ResourceRef.get()`, the registry checks whether the resource is stale.
If it is, it ensures the dependencies and runs the owning producer. This gives you a
lazy compute graph without requiring a heavy scheduler.

## Registry semantics and validation

The registry tracks a version number and a dependency signature for each resource.
A resource is stale if:
- It has never been committed (`version == 0`), or
- Any dependency has a different version than the signature stored at last commit.

The key mutation operations are:
- `set_buffer`: replace the buffer, optionally bumping the version.
- `commit`: mark fresh relative to current deps (and optionally replace the buffer).
- `bump`: mark fresh without changing the buffer.

`ResourceSpec` adds runtime validation for buffer type, dtype, lanes, and shape.
Validation is a best-effort check, especially for Taichi fields. You can bypass
checks with `unsafe=True` when prototyping.

## Modules and dynamic dependencies

Modules are instantiated via `World.require`. Dependencies are discovered dynamically
as the module constructors run. The world detects module cycles to avoid hard-to-debug
recursive initialization.

Scopes allow you to create multiple independent copies of the same module graph.
The scope becomes a prefix in the resource names.

## Houdini runtime architecture

The Houdini integration wraps cooking into a small runtime layer:
- `ComputeRuntime` holds a cache of `WorldSession` objects keyed by HIP path and node path.
- `WorldSession` stores the `World` plus cached geometry state and diagnostics.
- `CookContext` is created per cook and exposes geometry I/O, publishing, and timing data.

The template scripts under `rheidos/apps/point_vortex` show the intended integration
pattern: seed output geometry, build a `CookContext`, publish minimal geometry keys,
then call user code.

The `run_cook` and `run_solver` drivers add stricter behavior, including script/module
loading rules and automatic handling of `out.P`. They are available but not the primary
workflow in this repo.

## GeometryIO design and constraints

`GeometryIO` provides a stable, NumPy-first interface to Houdini geometry.
Key design points:
- Attribute reads return NumPy arrays, cached per cook for reuse.
- Writes normalize shapes (1D becomes `(N, 1)`; detail becomes `(1, tuple_size)`).
- Group reads can return indices or boolean masks.
- Primitive reads only support polygon prims of a fixed arity (triangles by default).

These choices keep I/O explicit and predictable, which is important when you are
bridging to compute graphs or Taichi fields.

## Taichi fields in the compute layer

Taichi fields are supported as a ResourceSpec kind (`"taichi_field"`). Validation
checks are best-effort (dtype, shape, and vector lanes when available). For dynamic
allocation, the recommended pattern is allocate-before-fill using `set_buffer` and
`commit` after you write the data.

The Houdini runtime includes `reset_taichi_hard()` for full Taichi resets when you
need to clear global state between sessions.
