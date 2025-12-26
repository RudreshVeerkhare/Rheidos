# Compute DAG Architecture
*A resource–producer–module system for Taichi-first simulation tooling (with explicit wiring, lazy evaluation, and minimal plumbing).*

This doc explains the architecture, the design philosophy behind it, and how to use it effectively.

---

## 1. What this framework is trying to be

You want three things at once:

1. **Lazy computation**: derived things (edges, DEC stars, Poisson solve) should compute only when needed.
2. **Explicit wiring**: no magical “module-to-module string plumbing” inside producers.
3. **Fast prototyping**: you should be able to mutate any resource, even if it’s “derived”, without fighting the framework.

This architecture is a sweet-spot that aims to be:
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

> A resource can be user-set (“input”) or derived (“produced”), but system does not enforce a distinction.

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

### 4.1 What's the catch?
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
