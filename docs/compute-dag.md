# Compute Module: A Gentle Tour
*A beginner-friendly guide to the compute DAG: why it exists, how to think about it, and how to use it.*

---

## Why this module exists

When you build simulations, you keep hitting the same pain points:

- You want **derived data** (edges, cotan weights, Poisson solves) to update only when needed.
- You want **clear wiring** between pieces of data and the code that produces them.
- You want **freedom** to override caches or inject values while prototyping.

The compute module is a small framework that hits those goals without being rigid.
It is not a heavy ECS or a full reactive system. It is a pragmatic tool for research code.

---

## The mental model (three nouns)

### 1) Resource = a named piece of state
A resource is a value stored in the registry. It can be a Taichi field, a numpy array, or a plain Python object.

Think of it as a labeled container:

- It has a name (like `mesh.V_pos`).
- It has a buffer (the actual data).
- It may be produced by code (a producer).

Resources can be inputs or derived outputs. The system does not force a strict separation.

### 2) Producer = code that fills resources
A producer is a small computation that knows how to build one or more resources.

It is **wired** to real `ResourceRef`s through an IO dataclass, so it does not invent string names.
That makes it easy to read and easy to reuse.

### 3) Module = a namespace and a home for related resources
A module groups a set of resources and producers under a stable prefix, like `mesh` or `poisson`.
Modules help you keep your graph organized and avoid name collisions.

---

## The promise: lazy computation

The registry tracks freshness automatically. If you ask for `poisson.u`:

- It checks whether `poisson.u` is stale.
- If stale, it runs the producer that owns it.
- It ensures the producer's dependencies are up to date first.

This is how a compute DAG becomes "compute on demand".

---

## A first example (Poisson app)

Below is a tiny example mirroring `apps/poisson_dec`:

```py
from rheidos.compute import World
from apps.poisson_dec.compute.mesh import MeshModule
from apps.poisson_dec.compute.dec import DECModule
from apps.poisson_dec.compute.poisson import PoissonSolverModule

world = World()
mesh = world.require(MeshModule)
dec = world.require(DECModule)
poisson = world.require(PoissonSolverModule)

# Provide inputs
mesh.V_pos.set(V_field)
mesh.F_verts.set(F_field)
poisson.constraint_mask.set(mask)
poisson.constraint_value.set(val)

# Ask for the solution (triggers lazy compute)
u = poisson.u.get()
```

You never call the solver directly. You ask for `u`, and the graph figures it out.

---

## Wiring a producer (IO dataclass)

This is the pattern used by the mesh, DEC, and Poisson builders:

```py
from dataclasses import dataclass
from rheidos.compute import ResourceRef, WiredProducer, out_field

@dataclass
class BuildDECIO:
    V_pos: ResourceRef
    F_verts: ResourceRef
    E_verts: ResourceRef
    E_opp: ResourceRef
    star0: ResourceRef = out_field()
    star1: ResourceRef = out_field()
    star2: ResourceRef = out_field()

class BuildDEC(WiredProducer[BuildDECIO]):
    def compute(self, reg):
        io = self.io
        V = io.V_pos.get(ensure=False)
        F = io.F_verts.get(ensure=False)
        E = io.E_verts.get(ensure=False)
        EO = io.E_opp.get(ensure=False)
        # ... allocate / fill ...
        io.star0.commit()
        io.star1.commit()
        io.star2.commit()
```

Key idea: inputs are already ensured by the registry before `compute()` runs,
so producers should usually use `ensure=False` for their reads.

---

## Allocation-before-fill (common pattern)

When you need to allocate a Taichi field before filling it:

```py
bar = io.bar.get(ensure=False)
if bar is None or bar.shape != (n,):
    bar = ti.field(dtype=ti.f32, shape=(n,))
    io.bar.set_buffer(bar, bump=False)

# fill bar...
io.bar.commit()
```

`set_buffer(..., bump=False)` stores the buffer without marking it fresh.
`commit()` marks it fresh after you actually filled the data.

---

## Debugging the graph

If something feels stale or wrong, ask the registry for an explanation:

```py
print(world.reg.explain(poisson.u.name, depth=6))
```

It will show which resources are stale and who produces them.

---

## Freedom (and responsibility)

A key design choice: **nothing is read-only**.

You are allowed to override a derived resource and commit it manually.
This is a deliberate feature for research workflows.

That freedom means you should adopt a simple discipline:

- If you write to a resource, call `commit()`.
- Use `unsafe=True` only when you really want to skip validation.

---

## ResourceSpec in one paragraph

`ResourceSpec` provides runtime validation for buffers:

- Kind: `taichi_field`, `numpy`, or `python`.
- Optional dtype, shape, and lane checks.

It catches mistakes early without forcing heavy static typing.

---

## Module scope (multiple instances)

You can build multiple copies of the same module graph using scopes:

```py
app_a = world.require(PoissonApp, scope="simA")
app_b = world.require(PoissonApp, scope="simB")
```

Their resources are isolated:

- `simA.mesh.V_pos` vs `simB.mesh.V_pos`

---

## Recommended import style

For a nice developer experience, import from the compute package root:

```py
from rheidos.compute import World, ModuleBase, ResourceRef, ResourceSpec, WiredProducer, out_field
```

Shared typing aliases live in:

```py
from rheidos.compute.typing import Shape, ShapeFn, ResourceName
```

---

## A small checklist for new producers

- Read inputs with `ensure=False`.
- Allocate outputs if needed with `set_buffer(..., bump=False)`.
- Fill outputs.
- `commit()` every output.

If you follow this, the graph will stay correct and easy to reason about.

---

## Summary

The compute module is a small, focused system that gives you:

- **Explicit data flow** via resources and producers.
- **Lazy computation** via `ensure()`.
- **Flexibility** for research code (manual overrides are allowed).

If you want something to update automatically, connect it with deps and a producer.
If you want to hack in a new value, just set and commit it.

That balance is the core philosophy of the module.
