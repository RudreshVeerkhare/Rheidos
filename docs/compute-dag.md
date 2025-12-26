# Compute DAG Architecture
*A resource–producer–module system for Taichi-first simulation tooling (with explicit wiring, lazy evaluation, and minimal string plumbing).*

This doc explains the architecture, the design philosophy behind it, and how to use it effectively.

---

## 1. What this framework is trying to be

You want three things at once:

1. **Lazy computation**: derived things (edges, DEC stars, Poisson solve) should compute only when needed.
2. **Explicit wiring**: no magical “module-to-module string plumbing” inside producers.
3. **Fast prototyping**: you should be able to mutate any resource, even if it’s “derived”, without fighting the framework.

This architecture is a compromise that aims to be:
- **Structured enough** to prevent a messy dependency jungle,
- **Loose enough** to allow research-grade experimentation,
- **Honest enough** to surface mistakes early (stale data, missing commits, cycles).

---

## 2. Core mental model (the three nouns)

### 2.1 Resource
A **Resource** is a named piece of state in the registry. It stores:

- `buffer`: the actual data (Taichi field, numpy array, python object…)
- `deps`: a tuple of other resource names this resource depends on
- `producer`: optional compute node that can produce/update it
- `version`: integer “freshness stamp”
- `dep_sig`: snapshot of dependency versions when it was last marked fresh
- `spec`: optional runtime schema/validation

> A resource can be user-set (“input”) or derived (“produced”),  the system does not enforce a distinction.

### 2.2 Producer
A **Producer** is a computation node.

Contract:
- It has `outputs: tuple[str, ...]`
- It has `compute(reg) -> None`
- It must call `reg.commit(output)` or `reg.bump(output)` for each output it updates

Producers are intended to be:
- **Name-blind** (they don’t invent resource strings),
- **Wired once** to actual `ResourceRef` handles via a typed IO dataclass,
- **Triggered lazily** by `reg.ensure(...)`.

### 2.3 Module
A **Module** groups related resources and producers under a scoped namespace, e.g.:

- `mesh.V_pos`
- `mesh.E_verts`
- `dec.star1`
- `poisson.u`

Modules provide:
- naming conventions + scoping
- ergonomic `self.resource(...)` helpers
- a place to build producers and declare registry nodes

Modules are instantiated by the `World`.

---

## 3. The registry: freshness, staleness, and “ensure”

### 3.1 Freshness rule
A resource is considered **fresh** if:

- it has a producer and has been committed at least once (`version > 0`)
- and its stored `dep_sig` matches current versions of its deps

It is **stale** if:

- `version == 0` (never committed)
- or dependency versions changed since last commit

### 3.2 `ensure(name)`
When you call `reg.ensure("poisson.u")`:

1. It recursively ensures dependencies first.
2. If the resource is stale and has a producer, it runs the producer.
3. It verifies the producer actually committed all its outputs.
4. It errors on dependency cycles (resource-level cycles).

This is the engine that turns your code into a lazy DAG.

### 3.3 Multi-output producers
A producer can output many resources. The registry handles this by:

- ensuring deps for every output of the producer before running
- requiring that each output is committed/bumped after compute
- marking the producer “ran” to avoid repeated work in a single ensure pass

This allows fused computations like:
- topology builder produces `E_verts`, `E_faces`, `E_opp` together

---

## 4. How not to shoot yourself

### 4.1 What does it mean ?
The registry does **not** treat “produced resources” as read-only.

You can:
- replace a derived resource’s buffer manually
- call `commit()` on it even if it has a producer

This is intentional: research workflows often require overriding caches and injecting intermediate results.

### 4.2 Trade-offs
This gives speed and flexibility, but it shifts responsibility:

- You can accidentally mark `None` as fresh (if `allow_none=True`).
- You can bypass producers and forget to recompute something.
- You can hide a bug by manually committing a broken buffer.

### 4.3 Safety knobs already present
- `ResourceSpec` validation (dtype, lanes, shape, shape_fn)
- `unsafe=True` flags to bypass validation (explicit)

### 4.4 Recommended discipline
Use these conventions:

- For normal code paths, always:
  - write buffer data
  - then call `ref.commit()`

- For allocation-before-fill patterns:
  - `ref.set_buffer(field, bump=False)`
  - fill
  - `ref.commit()`

- Avoid `unsafe=True` unless you’re explicitly spiking something.

---

## 5. ResourceSpec validation

`ResourceSpec` is runtime validation, not static typing.

Supported “kinds”:

- `taichi_field`: best-effort check for Taichi-like fields
- `numpy`: np.ndarray check
- `python`: no validation

Validation checks:
- dtype equality
- lanes for vector fields (`buf.n` if available)
- shape exact match
- dynamic shape via `shape_fn(reg)`

### 5.1 Dynamic shape (`shape_fn`)
This is how derived buffers can validate against upstream buffers without hardcoding sizes.

Example:
- `dec.star1` shape should match `mesh.E_verts.shape`

Implementation pattern:
- `_shape_of(mesh.E_verts)` returns a function that reads the buffer and returns its shape

---

## 6. Typed handles: ResourceKey + ResourceRef

### 6.1 Problem being solved
You don’t want user code to do string plumbing like:

```py
reg.read("scope.mesh.V_pos")
````

It kills IDE hints and encourages string bugs.

### 6.2 ResourceRef

`ResourceRef[T]` wraps `(registry, ResourceKey)`:

* `.get()` reads (and ensures by default)
* `.set(value)` replaces buffer and commits
* `.set_buffer(value, bump=False)` sets buffer without freshness bump
* `.commit()` marks fresh relative to deps

This gives:

* strong-ish ergonomics
* no strings in user code
* a single “official place” for committing state

---

## 7. Namespacing and scoping

### 7.1 Namespace

A module has a namespace derived from:

* optional `scope` (instance name)
* module `NAME`

So:

* scope = `"simA"`, module NAME `"mesh"` → prefix `"simA.mesh"`
* scope = `""`, module NAME `"mesh"` → prefix `"mesh"`

### 7.2 Why scope exists

You can instantiate multiple copies of a module graph:

```py
appA = world.require(PoissonApp, scope="simA")
appB = world.require(PoissonApp, scope="simB")
```

Their resources won’t collide:

* `simA.mesh.V_pos` vs `simB.mesh.V_pos`

---

## 8. World and module lifecycle

### 8.1 World.require(module_cls)

The world memoizes modules by `(scope, module_cls)`.

Important properties:

* Only one instance of a module class per scope
* Modules can call `self.require(OtherModule)` to express dependencies dynamically

### 8.2 Cycle detection (module-level)

Since dependencies are discovered by executing `__init__`, cycles can happen.

Example cycle:

* `A.__init__()` calls `require(B)`
* `B.__init__()` calls `require(A)`

The `World` detects this with a build stack and raises a readable error:

```
Module dependency cycle detected: <root>:A -> <root>:B -> <root>:A
```

---

## 9. Producers: WiredProducer + IO dataclasses

### 9.1 Why IO dataclasses exist

We want producers to:

* be reusable
* avoid string plumbing
* be explicit about inputs/outputs

So we define an IO dataclass:

```py
@dataclass
class BuildDECIO:
    V_pos: ResourceRef[Any]
    F_verts: ResourceRef[Any]
    ...
    star1: ResourceRef[Any] = out_field()
```

Fields marked with `out_field()` are considered outputs.

### 9.2 WiredProducer

`WiredProducer(io)`:

* stores `self.io`
* introspects output fields and sets `self.outputs`

This enforces:

* if it’s a producer instance, it’s already wired
* no two-phase “bind IO later”

### 9.3 Producer compute style

Inside `compute` you typically:

* read required inputs using `.get(ensure=False|True)`
* allocate outputs if needed
* fill outputs
* call `.commit()` on each output

---

## 10. Typical usage guide

### 10.1 Basic flow

A typical “app” flow looks like:

1. Create `World`
2. Require some top-level module (app)
3. Set input resources (buffers)
4. Read derived resource → triggers entire chain

Example:

```py
world = World()
app = world.require(PoissonApp)

# set inputs
app.mesh.V_pos.set(V_field)
app.mesh.F_verts.set(F_field)

app.poisson.constraint_mask.set(mask)
app.poisson.constraint_value.set(val)

# get output (lazy triggers)
u = app.poisson.u.get()
```

### 10.2 Debugging the DAG

Use:

```py
print(world.reg.explain(app.poisson.u.name, depth=7))
```

You’ll see:

* each node version
* whether it’s stale
* which producer owns it

---

## 11. How to add your own module

### 11.1 Create module

```py
class MyModule(ModuleBase):
    NAME = "my"

    def __init__(self, world: World, *, scope: str = ""):
        super().__init__(world, scope=scope)

        mesh = self.require(MeshModule)

        self.foo = self.resource("foo", spec=..., declare=True)
        self.bar = self.resource("bar", spec=..., declare=False)

        prod = MyProducer(MyIO.from_modules(mesh, self))
        self.declare_resource(self.bar, deps=(mesh.V_pos.name, self.foo.name), producer=prod)
```

### 11.2 Create IO and producer

```py
@dataclass
class MyIO:
    V_pos: ResourceRef[Any]
    foo: ResourceRef[Any]
    bar: ResourceRef[Any] = out_field()

@ti.data_oriented
class MyProducer(WiredProducer[MyIO]):
    def compute(self, reg: Registry):
        io = self.io
        V = io.V_pos.get(ensure=True)
        foo = io.foo.get(ensure=True)

        # allocate/fill bar
        bar = io.bar.get(ensure=False)
        if bar is None or bar.shape != (V.shape[0],):
            bar = ti.field(dtype=ti.f32, shape=(V.shape[0],))
            io.bar.set_buffer(bar, bump=False)

        # ... fill bar ...
        io.bar.commit()
```

---

## 12. Best practices (so your future self doesn’t hate you)

### 12.1 Always commit outputs

If your producer writes an output but forgets to commit, `ensure()` will throw.
That’s good. Keep it that way.

### 12.2 Prefer “read ensure=True” inside producers for upstream derived inputs

If an input is derived (like `mesh.E_verts`), read it with `ensure=True`:

```py
E = io.E_verts.get(ensure=True)
```

That keeps producers robust even if called independently.

### 12.3 Use `set_buffer(..., bump=False)` for allocation-before-fill

This prevents marking fresh before values exist.

### 12.4 Keep producers deterministic

Producers should ideally:

* be pure functions of their deps (plus module config)
* not depend on hidden global state

If you need time-dependent behavior, make “time” a resource dependency.

### 12.5 Put “parameters” in resources too

Instead of storing parameters inside modules, prefer resources:

* makes dependencies explicit
* makes caching/freshness correct

Example:

* viscosity, timestep, solver iterations → resources

---

## 13. Philosophy summary

This architecture intentionally leans into:

* **Explicit state**: everything is a resource
* **Lazy evaluation**: compute only when asked
* **Explicit wiring**: producers do not do string plumbing
* **Research flexibility**: the `.bump()` on any resources lets you override anything and tap into the DAGs auto-update.
* **Fail-fast correctness**: stale checks + commit enforcement + cycle detection

It’s not trying to be a rigid ECS, or a heavy DI container, or a magical reactive graph.
It’s a pragmatic research tool: structured enough to scale, loose enough to explore.

---

## 14. FAQ

### Q: Why not just call producers directly?

You can, but then dependency order and recomputation become manual.
The registry gives you automatic “compute-on-demand” with correctness checks.

### Q: Why are resources still stored by string in the registry?

Because registries need a universal key.
But you *don’t* expose strings to users; you expose `ResourceRef`.

### Q: Can I swap Taichi out?

Yes. `ResourceSpec.kind="python"` is totally legal, and producers can operate on python objects.
Taichi is just one buffer backend.

### Q: How do I build richer tooling (graph views, debug UI)?

The registry already has:

* resource metadata
* deps
* producer ownership
* freshness state

That’s enough to build a graph visualizer or inspector UI later.

---

## 15. Minimal cheat sheet

* Declare inputs with `declare=True`
* Declare derived resources with `declare=False` then `declare_resource(...)` once wired
* In producers:

  * read inputs
  * allocate outputs if needed (`set_buffer(..., bump=False)`)
  * write outputs
  * `commit()` outputs
* To compute something: `ref.get()` or `reg.ensure(name)`
* To inspect: `reg.explain(name)`

---
