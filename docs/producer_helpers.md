# Decorator Producers

The supported producer authoring path in this repo is the decorator-based API:
- `@producer(...)`
- `ProducerContext`
- `producer_output(...)`

## Quick start

```python
import numpy as np

from rheidos.compute import ModuleBase, ResourceSpec, World, producer


class DemoModule(ModuleBase):
    NAME = "demo"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.a = self.resource(
            "a",
            declare=True,
            spec=ResourceSpec(kind="numpy", dtype=np.float32, shape=(1,)),
            buffer=np.array([2.0], dtype=np.float32),
        )
        self.out = self.resource(
            "out",
            spec=ResourceSpec(kind="numpy", dtype=np.float32, shape=(1,)),
        )
        self.bind_producers()

    @producer(inputs=("a",), outputs=("out",))
    def run(self, ctx) -> None:
        ctx.commit(out=ctx.inputs.a.get() + 3.0)
```

## `@producer`

Use `@producer` on module methods that accept a single `ProducerContext`.

```python
@producer(inputs=("a", "b"), outputs=("out",))
def add(self, ctx) -> None:
    ctx.commit(out=ctx.inputs.a.get() + ctx.inputs.b.get())
```

Input and output names are resolved against `ResourceRef` attributes on the
owning module. Nested paths such as `"mesh.V_pos"` are supported.

## `ProducerContext`

`ProducerContext` exposes bound resource namespaces plus helper methods:

```python
ctx.inputs.a.get()
ctx.outputs.out.peek()
ctx.commit(out=array_value)
```

Key methods:
- `require_inputs(...)` validates required input buffers
- `ensure_outputs(...)` allocates or reallocates outputs from `ResourceSpec`
- `commit(...)` commits named outputs by declared output name

## Custom allocation with `producer_output`

Use `producer_output` when an output buffer should be allocated explicitly:

```python
from rheidos.compute import producer, producer_output


def alloc_out(_reg, ctx):
    return np.zeros_like(ctx.inputs.a.peek())


@producer(
    inputs=("a",),
    outputs=(producer_output("out", alloc=alloc_out),),
)
def fill(self, ctx) -> None:
    out = ctx.ensure_outputs(require_shape=False)["out"].peek()
    out[:] = ctx.inputs.a.get() + 1.0
    ctx.outputs.out.commit()
```

## Dynamic shapes

Dynamic output shapes come from `ResourceSpec.shape_fn`.

Common helpers:
- `shape_map(ref, mapper)`
- `shape_of(ref)`
- `shape_from_axis(ref, axis=0, tail=())`
- `shape_from_scalar(ref, tail=())`
- `shape_with_tail(ref, tail=())`

Example:

```python
from rheidos.compute import ResourceSpec, shape_from_scalar

self.count = self.resource(
    "count",
    declare=True,
    spec=ResourceSpec(kind="python"),
    buffer=8,
)
self.positions = self.resource(
    "positions",
    spec=ResourceSpec(
        kind="numpy",
        dtype=np.float32,
        shape_fn=shape_from_scalar(self.count, tail=(3,)),
        allow_none=True,
    ),
)
```

## Custom resource kinds

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

Your adapter controls shape resolution, allocation, and validation. Set
`requires_shape=False` if the kind does not depend on a resolved shape.

## Runtime pattern

The normal producer flow is:

1. Validate required inputs with `ctx.require_inputs()`.
2. Allocate outputs with `ctx.ensure_outputs()`.
3. Fill output buffers.
4. Commit them with `ctx.commit(...)` or `ctx.outputs.name.commit()`.

`ensure_outputs()` only installs buffers. A resource is not fresh until it is
committed or bumped.

## Troubleshooting

If a producer fails to bind or run:
- call `bind_producers()` after declaring resources
- verify every name in `@producer(...)` resolves to a `ResourceRef`
- use `world.reg.explain(...)` to inspect freshness and dependency state
- use `ctx.require_inputs()` when you want explicit missing-input failures
- add `shape` or `shape_fn` to the output `ResourceSpec` when auto-allocation fails
