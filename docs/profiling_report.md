# Profiling Architecture and DAG Trace

This document describes the current profiling system, including the
execution-discovered DAG, stable IDs, execution tree capture, and UI stack. It
is intentionally concrete: data structures, call sites, and payload shapes are
documented so you can reason about accuracy and overhead.

## High-Level Architecture

Profiling is split into three layers:

- Collection (sim thread): `Profiler` hooks in the compute runtime capture
  timing, execution nesting, and dependency edges with O(1) operations.
- Aggregation (thread-safe store): `SummaryStore` holds time series and producer
  metadata, `TraceStore` holds per-cook traces and dependency graphs.
- Presentation (non-sim thread): `SummaryServer` serves JSON and static UI; the
  browser renders a DAG explorer, trace flamegraph, and details drawer.

Key files:

- `rheidos/compute/profiler/core.py`: `Profiler`, spans, modes, stack handling.
- `rheidos/compute/profiler/ids.py`: stable ID interning.
- `rheidos/compute/profiler/trace_store.py`: exec tree + edge storage.
- `rheidos/compute/profiler/summary_store.py`: aggregated metrics and details.
- `rheidos/compute/registry.py`, `rheidos/compute/resource.py`: runtime hooks.
- `rheidos/compute/profiler/summary_server.py`: HTTP + WS API.
- `rheidos/compute/profiler/ui/*`: HTML/CSS/JS UI.

## Stable IDs (Producers + Resources)

Stable IDs are used everywhere in the DAG and execution tree so identity is
consistent across cooks and sessions:

- `PRODUCER_IDS.intern(name: str) -> int`
- `RESOURCE_IDS.intern(name: str) -> int`
- Implementation: `ids.py` uses a 128-bit blake2b hash of the name and stores
  bidirectional maps.

Registry integration:

- `ProducerBase.profiler_id()` caches a producer ID on the instance.
- `Registry.declare(...)` assigns:
  - `Resource.resource_id` for the resource name.
  - `Resource.producer_id` for its producing producer (if any).

This makes `resource.get()` dependency capture a pure integer pipeline with no
string work on the hot path.

Producer identity metadata:

- `Profiler.register_producer_metadata(full_name, class_name)` stores a
  class name per producer ID and forwards it to `SummaryStore`.
- `Registry._ensure()` registers `producer.__class__.__name__` before timing.
- UI labels use `class_name` as primary text; `full_name` is always preserved
  for tooltips and copy.

## Cook IDs and Lifecycle

Cook IDs are monotonic integers managed by `Profiler.next_cook_index()`:

- The solver calls `next_cook_index()` per cook.
- `SummaryStore.update_global(cook_id=...)` tracks the current cook.
- `TraceStore.begin_cook(cook_id)` initializes per-cook capture (when enabled).

Cook IDs are used as:

- the update ID for producer timing rows,
- the index for execution tree snapshots, and
- the reference ID in UI tooltips and tables.

## Execution Stack and Truthful Producer Timing

Producer timing is measured at the actual execution boundary in
`Registry._ensure()`:

1. `Profiler.span("compute", cat="producer", producer=producer_name)` wraps
   `ProducerBase.compute()`.
2. `Profiler._span_enter()` pushes a `_ProducerRun` onto a thread-local stack.
3. `Profiler._span_exit()` records inclusive duration and pops the stack.

Key details:

- The stack is thread-local so nested producer calls build a real execution tree
  (parent is `stack[-2]`).
- Re-entrancy/cycles are detected: if a producer ID already exists on the
  stack, tree capture is skipped and `dropped_events` is incremented.
- Non-producer spans inside a producer run are tracked as child spans and
  surfaced in producer details.

## TraceStore: Exec Tree + Edge Capture

`TraceStore` is the capture backend for per-cook execution trees and observed
dependencies. It is intentionally compact and bounded.

Core data structures (`trace_store.py`):

- `ExecNode`: `(producer_id, parent_index, inclusive_ns)`
- `CookTrace`:
  - `nodes`: list of `ExecNode`
  - `edges_pr`: set of `(producer_id, resource_id)` for resource reads
  - `edges_pp`: set of `(producer_id, producer_id)` for producer dependencies
  - `edges_pp_meta`: map of `(producer_id, producer_id) -> set(resource_id)`

Bounded storage:

- The trace store keeps the last N cooks (`TraceConfig.max_cooks`).
- Edges are capped per cook (`TraceConfig.max_edges_per_cook`).
- When a cook exceeds its edge cap, edge capture stops for that cook and a
  dropped counter increments (sim continues).
- `dag_version` increments when new producers/edges appear or old ones are
  evicted, giving the UI a stable topology version.

## Dynamic Dependency Capture (resource.get)

Dependencies are discovered in `Registry.read()` and recorded by
`Profiler.record_resource_read(...)`:

- Called on every `ResourceRef.get()` (even when the resource is fresh).
- Uses `Resource.resource_id` and `Resource.producer_id` to record:
  - producer -> resource edges, and
  - producer -> producer edges (if the resource has a producer).

This avoids the "who ran" trap: a dependency is visible even if it was cached.

## Observed vs Union DAG

Two DAGs are maintained:

- Observed: edges from the current/last cook.
- Union: edges accumulated across the last N cooks, with `seen_count`.

`TraceStore.snapshot_dag(mode=...)` controls which one is returned.

## Execution Tree and Exclusive Time

`TraceStore.snapshot_exec_tree(...)` returns the most recent cook tree with
computed exclusive time:

- Exclusive time is computed post-hoc as:
  `exclusive_ns = inclusive_ns - sum(children_inclusive_ns)`.
- Each node includes `depth` and `parent` to allow UI tree layout.
- Each node includes `full_name` and `class_name` for label + tooltip use.

## Snapshot APIs and Payload Shapes

Profiler snapshot helpers:

- `Profiler.snapshot_dag(mode="union"|"observed")`:
  - `{ cook_id, dag_version, nodes:[{id,name,full_name,class_name}], edges:[{source,target,seen_count,via_resources}] }`
- `Profiler.snapshot_metrics()`:
  - `{ cook_id, frame, substep, rows:[{id,name,full_name,class_name,ema_ms,last_ms,last_update_id,calls,kernel_ms,kernel_frac,overhead_est_ms,executed_this_cook}] }`
- `Profiler.snapshot_exec_tree(cook_id=None)`:
  - `{ cook_id, nodes:[{id,producer_id,name,full_name,class_name,parent,depth,inclusive_ms,exclusive_ms}] }`
- `Profiler.snapshot_node_details(producer_id)`:
  - `{ id,name,full_name,class_name,metrics,inputs,outputs,resources_read,staleness_reason,last_exec_subtree }`
- `SummaryStore.snapshot_producer_details(producer_name)` (served at `/api/producer/<name>`):
  - `{ id,full_name,class_name,last_update,inputs,outputs,staleness_reason,top_child_spans }`

Summary store compact snapshot (used by legacy UI + WS):

- `SummaryStore.snapshot_compact()` includes:
  - `rows`, `categories`, `wall_ms`, `kernel_ms`, `kernel_fraction`,
  - `cook_id`, `frame`, `substep`, `tick`,
  - `dropped_events`, `profiler_overhead_us`, `edges_recorded`,
  - `dag_version`.
- Producer rows include `full_name` and `class_name`.
- WS payloads include this compact snapshot at the top level and may attach
  `metrics`, `dag`, and `exec_tree` when available.

## SummaryStore: Aggregation Rules

`SummaryStore` is the source of truth for aggregated metrics:

- Producer rows are only created for `cat="producer"` and `name="compute"`.
- Producer `class_name` is stored from `register_producer_metadata()` and
  falls back to the last path segment of `full_name`.
- Category tables aggregate all non-producer spans by `(cat, name, producer)`.
- Global stats track cook wall time and global taichi totals.
- Producer details track input/output versions and top child spans.

## SummaryServer Endpoints

`SummaryServer` serves both the legacy summary feed and the new trace APIs:

- `GET /api/summary` (legacy)
- `GET /api/producer/<name>` (legacy)
- `GET /api/dag?mode=union|observed`
- `GET /api/metrics`
- `GET /api/exec_tree?cook_id=<int>`
- `GET /api/node/<producer_id>`
- `GET /` and `GET /ws` for UI and WS snapshots (summary + metrics + optional dag/exec_tree)
  - `GET /ws?mode=observed` streams the observed DAG instead of the union DAG.

All producer-facing endpoints now include `full_name` and `class_name` so the
UI never has to join metrics back into DAG nodes for labeling.

## UI (DAG Explorer + Trace Explorer)

UI files: `rheidos/compute/profiler/ui` (React + Vite build in `dist/`).
- Build: `npm install` then `npm run build` from `rheidos/compute/profiler/ui`.

Pages and layout:

- Hash router switches between `#/dag` and `#/trace`.
- Deep links can capture `mode`, `sel` (producer id), and trace `cook`.
- Two-column layout with a persistent details drawer on the right.
- A live/pause toggle prevents UI thrash while keeping selection stable.
- Health bar surfaces `dropped_events`, `edges_recorded`, and
  `profiler_overhead_us`.

DAG explorer:

- React Flow + ELK layout for stable layered DAGs.
- Nodes use `class_name` labels with short suffixes for duplicates.
- Metrics update in-place; layout only re-runs when the DAG changes.
- Executed nodes/edges are highlighted per cook.
- Search dims non-matches and `Enter` focuses the best match.

Trace explorer:

- Canvas flamegraph built from `/api/exec_tree` snapshots.
- Inclusive width corresponds to total time; exclusive time is rendered as a
  trailing "(self)" block.
- Stable color by producer ID hash; labels appear when rectangles are wide.
- Follow-latest toggle + cook selector with in-memory LRU layout cache.
- Search filters frame labels; hover tooltip merges exec-tree timing with
  kernel/overhead from `/api/metrics`.

Details drawer:

- Header shows `class_name`, `full_name`, and `producer_id` with copy buttons.
- Last cook inclusive/exclusive, kernel, overhead, calls, last update.
- Resources read, inputs/outputs, plus incoming/outgoing via resources.
- Parent/child lists are clickable to reselect a producer.

Polling and scheduling:

- `/api/summary` still uses WS or polling (legacy summary feed).
- A single UI scheduler ticks `/api/metrics` (~4 Hz).
- On cook change, the scheduler fetches `/api/dag` and `/api/exec_tree`.
- Manual cook selection for the trace view uses `/api/exec_tree?cook_id=...`.

Layout is done in the browser to keep the sim thread clean.

## Profiling Modes

`ProfilerConfig.mode` controls capture:

- `off`: no spans, no edges.
- `coarse`: timings + execution tree + observed edges.
- `deps_only`: edges only (no timings).
- `sampled_taichi`: same as `coarse` plus taichi sampling gate.

## Overhead Audit

Optional overhead tracking (`ProfilerConfig.overhead_enabled`) measures:

- time spent in profiling hooks (micro-timed),
- number of edges recorded per cook.

These are exported as:

- `profiler_overhead_us`
- `edges_recorded`

## Bounded Storage + Drop Policy

- `profile_trace_cooks`: how many cook traces are retained.
- `profile_trace_edges`: per-cook edge cap.
- When a cook exceeds the edge cap, edge recording stops for that cook and
  `dropped_events` increments.

## Taichi Sampling and Kernel Attribution

Sampling is controlled by `profile_taichi`, `profile_taichi_every`, and
`profile_taichi_sync`:

- When sampling is active, `Registry._ensure()` uses `TaichiProbe` to clear,
  sync, and read kernel totals around each producer compute.
- Per-producer kernel and overhead metrics are recorded as:
  - `producer_kernel_ms`
  - `producer_overhead_ms`
- `SummaryStore` computes `kernel_frac` as `kernel_ms / wall_ms`.

## Configuration and Enablement

Houdini node parameters (see `solver_sop.py`):

- `profile` (master enable)
- `profile_mode` (`off`, `coarse`, `deps_only`, `sampled_taichi`)
- `profile_export_hz`
- `profile_trace_cooks`
- `profile_trace_edges`
- `profile_overhead`
- `profile_taichi`, `profile_taichi_every`, `profile_taichi_sync`

Environment flags:

- `RHEIDOS_UI=1` to enable the UI server (default on).

UI URL is stored in `session.stats["profile_ui_url"]` after a cook.

## Caveats and Interpretation

- Async kernels: wall spans are dispatch time unless a sync occurs.
- Producer spans wrap `compute()` only; syncs inside the producer affect timing.
- Dependencies are tracked on read, even when cached, to preserve semantics.
- DAG union only reflects the last N cooks (bounded buffer).
- Re-entrancy/cycles skip tree capture for that run and increment drops.
- EMA smoothing hides spikes; check last_ms for transient issues.

## Validation Tests

Key tests:

- `tests/test_profiler_trace.py`: stable IDs, exec tree nesting, edges, class names.
- `tests/test_summary_server.py`: `/api/summary`, `/api/dag`, `/api/metrics`,
  `/api/exec_tree`, `/api/node/<id>`, class name fields.
- `tests/test_summary_store.py`: SummaryStore aggregation rules, class name fields.
