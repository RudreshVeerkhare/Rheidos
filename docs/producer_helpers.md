# Producer Helpers (IO-driven producers)

This document describes the helper features that reduce boilerplate when writing
producers. The goal is to let IO dataclasses and ResourceSpec drive:

- input validation
- output allocation
- error messages when required buffers are missing

The helpers are implemented in `WiredProducer` and work for NumPy, Taichi, and
any registered custom resource kinds.

## Quick start

Minimal producer with IO dataclass, fast wiring, and auto allocation:

```python
from dataclasses import dataclass
import numpy as np
from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry


def alloc_out(reg, io):
    a = io.a.peek()
    if a is None:
        return None
    return np.zeros_like(a)


@dataclass
class AddIO:
    a: ResourceRef[np.ndarray]
    b: ResourceRef[np.ndarray]
    out: ResourceRef[np.ndarray] = out_field(alloc=alloc_out)


class AddProducer(WiredProducer[AddIO]):
    def compute(self, reg: Registry) -> None:
        inputs = self.require_inputs()
        outputs = self.ensure_outputs(reg)
        outputs["out"].peek()[:] = inputs["a"].get() + inputs["b"].get()
        self.io.out.commit()
```

## IO dataclass and out_field

`WiredProducer` uses an IO dataclass to discover inputs and outputs.

- Any `ResourceRef` field is treated as an input by default.
- Fields marked with `out_field()` are treated as outputs.

This information is used to:

- populate `self.inputs` and `self.outputs`
- validate that out fields are `ResourceRef` instances
- power helper methods like `require_inputs` and `ensure_outputs`

## Wiring patterns

Auto-inferred IO type is the primary way to use this abstraction.
If you subclass as `class AddProducer(WiredProducer[AddIO])`, `IO_TYPE` is
inferred and you can use kwargs-based wiring without extra boilerplate:

```python
class AddProducer(WiredProducer[AddIO]):
    pass

producer = AddProducer(a=a_ref, b=b_ref, out=out_ref)
```

If inference fails (aliases, forward refs, or complex inheritance), set
`IO_TYPE` explicitly:

```python
class AddProducer(WiredProducer[AddIO]):
    IO_TYPE = AddIO
```

You can also pass an IO instance explicitly:

```python
producer = AddProducer(AddIO(a_ref, b_ref, out_ref))
```

Caveats:

- Auto-inference is best-effort and only works for direct generic subclasses.
- Type aliases or indirection can hide the generic argument from `__orig_bases__`.
- If inference fails, kwargs wiring raises a clear error asking for `IO_TYPE`.

## Post-wiring setup hook

`WiredProducer.setup()` is a no-op hook called at the end of `__init__`, after
`self.io`, `self.inputs`, and `self.outputs` are wired. Override it to perform
lightweight initialization that needs access to wired refs.

Example:

```python
class AddProducer(WiredProducer[AddIO]):
    def setup(self) -> None:
        self.has_a = self.io.a.peek() is not None
```

## Input validation with require_inputs

`require_inputs` performs common input checks and returns a dict of ResourceRef objects:

```python
inputs = self.require_inputs()
V = inputs["V_pos"].get()  # or .peek() if you don't need dependency tracking
```

Behavior:

- Validates that required inputs have buffers set (using peek internally).
- If any required input is `None`, it raises a `RuntimeError` listing missing fields.
- `allow_none` lets you mark optional inputs by field name.
- `ignore` lets you skip fields you will not use.
- Returns ResourceRef objects - you call `.get()` or `.peek()` to access buffers.

Example with optional rhs:

```python
inputs = self.require_inputs(allow_none=("rhs",))
rhs_ref = inputs["rhs"]
rhs = rhs_ref.peek()  # Could be None
```

## Output allocation with ensure_outputs

`ensure_outputs(reg)` allocates missing outputs and returns a dict of ResourceRef objects:

- If an output buffer is `None`, it tries to allocate.
- If a buffer exists but does not match the spec, it reallocates.
- Shape resolution uses `ResourceSpec.shape` or `ResourceSpec.shape_fn`.
- NumPy outputs allocate with `np.zeros` by default.
- Taichi outputs allocate with `ti.field` or `ti.Vector.field` by default.

Example:

```python
outputs = self.ensure_outputs(reg)
out = outputs["out"].peek()  # or .get()
```

Important notes:

- Returns ResourceRef objects - you call `.get()` or `.peek()` to access buffers.
- `ensure_outputs` does not commit outputs. You still call `ref.commit()` after fill.
- If `shape` or `shape_fn` cannot resolve, allocation fails by default.
- In `strict=True` mode (default), missing outputs raise an error.
- `realloc=False` skips reallocation even if spec does not match.
- `require_shape=True` (default) enforces that outputs declare a shape via
  `ResourceSpec.shape` or `shape_fn` (or a custom `alloc`).

## ResourceSpec and dynamic shapes

Dynamic shapes are handled via `shape_fn`. Use the helper `shape_from_scalar`
when output shapes are based on a scalar resource (like `n_vortices`).

Example for an `(n, 3)` NumPy output:

```python
from rheidos.compute import ResourceSpec, shape_from_scalar

spec=ResourceSpec(
    kind="numpy",
    dtype=np.float32,
    shape_fn=shape_from_scalar(n_vortices_ref, tail=(3,)),
    allow_none=True,
)
```

`shape_from_scalar` accepts a `ResourceRef` that holds a scalar (Python, NumPy,
or Taichi scalar field). It returns `(n,)` or `(n, ...)` with the optional tail.

Use `shape_map` for shapes derived from another field/array when you need to
transform the resolved shape lazily:

```python
from rheidos.compute import shape_map

spec=ResourceSpec(
    kind="taichi_field",
    dtype=ti.f32,
    shape_fn=shape_map(mesh.F_verts, lambda shape: (shape[0], 3)),
    lanes=3,
    allow_none=True,
)
```

Convenience wrappers remain available for common cases: `shape_of`,
`shape_from_axis`, and `shape_with_tail`.

## Custom allocators with out_field(alloc=...)

For outputs that cannot be expressed using `ResourceSpec`, provide a custom
allocator via `out_field` metadata:

```python
def alloc_out(reg, io):
    n = int(io.n_vortices.get()[None])
    return np.zeros((n, 3), dtype=np.float32)


@dataclass
class FooIO:
    n_vortices: ResourceRef[Any]
    out: ResourceRef[np.ndarray] = out_field(alloc=alloc_out)
```

`alloc` receives `(reg, io)` and should return a buffer or `None`.
If it returns `None`, `ensure_outputs` falls back to `ResourceSpec`.

## Extending ResourceSpec kinds

To support new buffer types, register a custom resource kind:

```python
from rheidos.compute import ResourceKindAdapter, register_resource_kind

register_resource_kind(
    "torch_tensor",
    ResourceKindAdapter(
        resolve_shape=...,
        allocate=...,
        matches_spec=...,
        requires_shape=True,
    ),
)
```

Your adapter controls how shapes are resolved, how buffers are allocated, and how
they are validated. Set `requires_shape=False` if the kind does not use shapes.

## Lifecycle: allocate, fill, commit

The general pattern is:

1. `inputs = self.require_inputs()`
2. `outputs = self.ensure_outputs(reg)`
3. Fill outputs.
4. `ref.commit()` for each output.

`ensure_outputs` only calls `set_buffer`, so the resource is not fresh until you
commit or bump after writing.

## Troubleshooting

Common errors and fixes:

- "missing required inputs": use `allow_none` for optional fields or ensure
  the upstream resources are set before compute.
- "outputs are unset": add `ResourceSpec.shape` or `shape_fn`, or provide
  `out_field(alloc=...)`.
- "output is missing shape": set a shape/shape_fn or register a custom allocator.
- "expected dtype/shape": update the `ResourceSpec` or reallocate your output
  to match.
- "taichi_field allocation requires taichi": import Taichi or allocate with a
  custom `alloc` function.

## Example: point_vortex advection outputs

The point_vortex advection module declares dynamic shapes from `n_vortices`:

```python
self.new_face_ids = self.resource(
    "new_face_ids",
    spec=ResourceSpec(
        kind="numpy",
        dtype=np.int32,
        shape_fn=shape_from_scalar(self.pt_vortex.n_vortices),
        allow_none=True,
    ),
)
```

With this spec in place, producers can call `ensure_outputs(reg)` and get a
buffer of the correct shape without manual checks and allocation.
