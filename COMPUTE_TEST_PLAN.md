# Compute Module Test Plan (90% Coverage Target)

## Goals

- **Protect collaboration:** prevent accidental regressions in dependency resolution, spec validation, and module wiring.
- **Fast feedback:** full compute test suite < 10 seconds.
- **Stable coverage gate:** 90% overall on `rheidos.compute` with meaningful assertions.

---

## Strategy (Rethought)

Focus on **behavioral contracts** and **risk‑driven scenarios** rather than exhaustive micro‑tests. Achieve coverage via **parametrized tests** and **shared fixtures**.

### Test Layers

1. **Contract Tests (unit)**
   - Validate each public API behavior in isolation.
2. **Workflow Tests (integration)**
   - Validate producer execution order, dependency freshness, and module composition.
3. **Compatibility Tests (optional)**
   - Taichi availability and field validation; skip cleanly if taichi is missing.

---

## Coverage Targets by Risk

| Area                        | Risk      | Target | Notes                           |
| --------------------------- | --------- | ------ | ------------------------------- |
| Registry ensure/commit/bump | Very High | 95%    | Core execution semantics        |
| Module system (World)       | High      | 90%    | Collaboration safety            |
| Wired producers             | High      | 90%    | IO wiring correctness           |
| Resource kinds              | Medium    | 85%    | Numpy required; Taichi optional |
| typing + **init** helpers   | Low       | 100%   | Small and stable                |

---

## Core Behavior Contracts (Must Not Break)

### 1) Registry Freshness Contract

**Guarantee:** A resource is fresh if its version is newer than its dependency signature.

Tests:

- `ensure()` runs producer **only** if stale.
- `commit()` increments version and updates dep signature.
- `bump()` updates dep signature without mutation.
- `ensure_many()` shares execution context (no duplicate work).

### 2) Dependency Execution Order

**Guarantee:** Dependencies are produced before dependents.

Tests:

- Linear chain A -> B -> C
- Diamond graph with shared dependency
- Multiple producers sharing inputs

### 3) Spec Validation

**Guarantee:** Buffers match their spec, errors are precise.

Tests:

- None buffer allowed vs disallowed
- Shape/dtype mismatches raise `TypeError` with contextual message
- Custom kind registration errors

### 4) WiredProducer IO Wiring

**Guarantee:** Outputs are inferred only from `out_field()`; inputs are inferred from `ResourceRef`s.

Tests:

- IO dataclass inference
- Mixed input/output fields
- Missing outputs or bad IO types raise `TypeError`
- `require_inputs()` rejects missing buffers unless allowed

### 5) Module Dependency Safety

**Guarantee:** World prevents module cycles and deduplicates module instances per scope.

Tests:

- Module cycle detection (A -> B -> A)
- Same module required twice returns same instance
- Cross‑scope module resolution is isolated

---

## Test Matrix (Minimal, High‑Value)

### `registry.py`

- **Freshness:** stale vs fresh; dep_sig updates
- **Ensure ordering:** chain and diamond graphs
- **Producer execution:** only once per ensure context
- **Validation:** allow_none false; unknown kind; spec mismatch
- **Read/commit:** read with ensure vs without

### `world.py`

- **Namespace:** qualify/prefix correctness
- **Resource declaration:** scoped names and spec propagation
- **Module require:** singletons, cycle detection, scope isolation

### `wiring.py`

- **IO inference:** generic IO type, explicit IO_TYPE
- **Input/output wiring:** correct mapping and errors
- **require_inputs:** allow_none, ignore, missing data

### `resource_kinds.py`

- **Numpy adapter:** allocation, dtype, shape
- **Registration:** duplicate and unknown kind
- **Taichi adapter:** only when taichi installed (skipped otherwise)

### `__init__.py`

- `shape_of`, `shape_from_scalar`, `shape_with_tail` using mock resources

---

## Fixtures (Shared)

`tests/compute/conftest.py`

- `registry()` — fresh `Registry`
- `world()` — fresh `World`
- `numpy_spec()` — standard `ResourceSpec(kind="numpy")`
- `simple_producer()` — producer that commits a single output
- `linear_chain()` — A->B->C resource/producers
- `diamond_chain()` — A->(B,C)->D

**Why:** minimizes test code, ensures consistency across collaborators.

---

## Integration Scenarios (Must‑Have)

### Scenario A: Linear Pipeline

- Producer A -> Producer B -> Producer C
- Ensure C triggers A then B then C
- Ensure C again does not re-run any producers

### Scenario B: Diamond Dependencies

- Producer D depends on B and C; both depend on A
- Ensure D runs A once, then B/C, then D

### Scenario C: Module System Cycle

- Module A requires B; Module B requires A
- World raises cycle error with clear message

---

## Test Organization

```
tests/
  compute/
    conftest.py
    test_registry.py
    test_world.py
    test_wiring.py
    test_resource.py
    test_resource_kinds.py
    test_graph.py
    test_init.py
    integration/
      test_pipeline.py
      test_modules.py
```

---

## Coverage Gate

- Run: `pytest tests/compute --cov=rheidos.compute --cov-fail-under=90`
- Exclude profiler modules unless explicitly requested
- Use `@pytest.mark.skipif(taichi_missing)` for taichi‑specific tests

---

## Collaboration Safeguards

- **CI gate:** 90% coverage on compute module
- **PR checklist:** any change to compute must add or update tests
- **Fast suite:** compute tests only, no global test run required for small changes

---

## Next Actions

1. Build fixtures in `tests/compute/conftest.py`.
2. Implement `registry.py` and `world.py` tests first (highest risk).
3. Add wiring and resource kind tests.
4. Add integration scenarios.
5. Enable coverage gate in CI.
