# Houdini Integration Docs (Diataxis)

Pick the section that matches your intent:
- Tutorials: learn by doing
- How-to guides: solve a task
- Reference: API facts
- Explanation: design rationale and tradeoffs

## Tutorials (teach me)

### Scale point positions with GeometryIO in a Python SOP

Audience: Houdini users who want a hands-on intro to the geometry adapter.

Prerequisites:
- Houdini installed with `hython` available.
- `rheidos` importable via `packages/rheidos.json`.
- A Geometry network with a Box SOP feeding a Python SOP.

Steps:
1) Action: Create a Box SOP and connect it to a Python SOP.
   Result: A Python SOP receives input geometry.
2) Action: Paste this into the Python SOP.
   Snippet:
   ```python
   import hou
   from rheidos.houdini.geo import GeometryIO, OWNER_POINT

   node = hou.pwd()
   geo_out = node.geometry()
   geo_out.clear()
   geo_in = node.inputs()[0].geometry()
   geo_out.merge(geo_in)

   io = GeometryIO(geo_in, geo_out)
   P = io.read(OWNER_POINT, "P", components=3)
   io.write(OWNER_POINT, "P", P * 1.1)
   ```
   Result: The box scales up by 10% in the viewport.

### Publish minimal geometry and write color with CookContext

Audience: Houdini users who want to use the compute world bridge.

Prerequisites:
- The same setup as the previous tutorial.
- NumPy available in Houdini's Python.

Steps:
1) Action: Create a Box SOP and connect it to a Python SOP.
   Result: A Python SOP receives input geometry.
2) Action: Paste this into the Python SOP.
   Snippet:
   ```python
   import hou
   import numpy as np
   from rheidos.houdini.geo import OWNER_POINT
   from rheidos.houdini.runtime import (
       GEO_P,
       build_cook_context,
       get_runtime,
       publish_geometry_minimal,
   )

   node = hou.pwd()
   geo_out = node.geometry()
   geo_out.clear()
   geo_in = node.inputs()[0].geometry()
   geo_out.merge(geo_in)

   session = get_runtime().get_or_create_session(node)
   ctx = build_cook_context(node, geo_in, geo_out, session)

   publish_geometry_minimal(ctx)
   P = ctx.fetch(GEO_P)
   colors = (P - P.min(axis=0)) / (np.ptp(P, axis=0) + 1e-6)
   ctx.write(OWNER_POINT, "Cd", colors)
   ```
   Result: The geometry shows a color gradient based on position.

## How-to guides (help me do X)

### Add the repo to Houdini's Python path

1) Put `packages/rheidos.json` in a Houdini packages folder (or set
   `HOUDINI_PACKAGE_DIR` to this repo's `packages` directory).
2) If the repo moves, update the paths inside `packages/rheidos.json`.
3) Restart Houdini.

### Build and install Rheidos SOP HDAs

1) Run the builder from hython:
   ```bash
   hython rheidos/houdini/nodes/build_hda.py
   ```
2) Confirm `rheidos/houdini/otls/rheidos_houdini.otl` exists.
3) Restart Houdini so it scans the new OTL.
4) In a Geometry network, create a node named "Rheidos Cook SOP" or
   "Rheidos Solver SOP".

### Run a stateless cook with the Cook SOP HDA

1) Create a Rheidos Cook SOP and connect input 0 (optional).
2) Set `script_path` to `rheidos/houdini/scripts/cook_demo.py`.
3) Leave `module_path` empty and set `mode` to `cook`.
4) Toggle `reset_node` once if you just changed scripts.

### Run the Poisson DEC cook script

1) Ensure the input geometry is triangulated (Triangulate SOP or Convert with "Triangulate").
2) Add point groups `poisson_pos` and `poisson_neg` (or a float point attribute `poisson_charge`).
3) Create a Rheidos Cook SOP and connect input 0.
4) Set `script_path` to `rheidos/houdini/scripts/poisson_dec.py` and `mode` to `cook`.
5) Read the output point attribute `poisson_u` for visualization or downstream processing.

### Run a solver step with the Solver SOP HDA

1) Create a Rheidos Solver SOP with input 0 = previous frame and input 1 = current
   input (optional).
2) Set `script_path` to `rheidos/houdini/scripts/solver_demo.py`.
3) Set `mode` to `solver`.
4) If you need substeps, set `substep` on the node.

### Run the drivers without HDAs (manual Python SOP)

1) Create a Python SOP and connect input 0.
2) Add spare parameters named: `script_path`, `module_path`, `mode`, `reset_node`,
   `nuke_all`, `profile`, `debug_log`, `last_error` (read-only).
3) Paste this into the Python SOP:
   ```python
   from rheidos.houdini.scripts.cook_sop import main
   main()
   ```
4) Set `script_path` and cook the node.

### Prepare geometry for publish_geometry_minimal

1) If your geometry is not triangles, insert a Triangulate SOP (or Convert with
   "Triangulate") upstream.
2) If you cannot triangulate, skip `publish_geometry_minimal(ctx)` and only read
   what you need (for example `ctx.P()`).

### Switch user scripts safely (no hot reload)

1) Change `script_path` or `module_path`.
2) Toggle `reset_node` on (it auto-clears after the cook) or call
   `get_runtime().reset_session(...)`.

### Record diagnostics from user scripts

1) Toggle `profile` to collect timing spans in `session.stats["last_timings"]`.
2) Call `ctx.log("my message", value=123)` inside your script.
3) Inspect `session.log_entries` in a Python shell.

### Output geometry via out.P

Publish point positions under `out.P` from your user script:

```python
def cook(ctx) -> None:
    P = ctx.P()
    ctx.publish("out.P", P * 2.0)
```

The driver applies `out.P` to point positions after the user function returns.

### Reset a node session from a node script

```python
import hou
from rheidos.houdini.runtime import get_runtime

node = hou.pwd()
get_runtime().reset_session(node, reason="user reset")
```

### Nuke all sessions and reset Taichi

```python
from rheidos.houdini.runtime import get_runtime

get_runtime().nuke_all(reason="global reset")
```

### Read a point attribute with GeometryIO

In a Python SOP with input 0 connected:

```python
import hou
from rheidos.houdini.geo import GeometryIO, OWNER_POINT

node = hou.pwd()
geo_in = node.inputs()[0].geometry()
io = GeometryIO(geo_in, geo_in)
P = io.read(OWNER_POINT, "P", components=3)
```

### Write a new point attribute with GeometryIO

In a Python SOP with input 0 connected:

```python
import hou
import numpy as np
from rheidos.houdini.geo import GeometryIO, OWNER_POINT

node = hou.pwd()
geo_out = node.geometry()
geo_out.clear()
geo_in = node.inputs()[0].geometry()
geo_out.merge(geo_in)

io = GeometryIO(geo_in, geo_out)
values = np.random.rand(len(geo_out.points()), 1)
io.write(OWNER_POINT, "u", values, create=True)
```

### Read triangle indices

In a Python SOP with input 0 connected:

```python
import hou
from rheidos.houdini.geo import GeometryIO

node = hou.pwd()
geo_in = node.inputs()[0].geometry()
io = GeometryIO(geo_in, geo_in)
triangles = io.read_prims(arity=3)
```

### Read a point group as a mask

In a Python SOP with input 0 connected:

```python
import hou
from rheidos.houdini.geo import GeometryIO, OWNER_POINT

node = hou.pwd()
geo_in = node.inputs()[0].geometry()
io = GeometryIO(geo_in, geo_in)
mask = io.read_group(OWNER_POINT, "my_group", as_mask=True)
```

### Read a point group with solver defaults

In a Cook/Solver script:

```python
from rheidos.houdini.geo import OWNER_POINT

mask = ctx.read_group_default(OWNER_POINT, "my_group")
```

### Publish standard geometry keys

In a Python SOP with input 0 connected:

```python
import hou
from rheidos.houdini.runtime import build_cook_context, get_runtime, publish_geometry_minimal

node = hou.pwd()
geo_out = node.geometry()
geo_out.clear()
geo_in = node.inputs()[0].geometry()
geo_out.merge(geo_in)

session = get_runtime().get_or_create_session(node)
ctx = build_cook_context(node, geo_in, geo_out, session)

publish_geometry_minimal(ctx)
```

### Publish a point group mask

In a Python SOP with input 0 connected:

```python
import hou
from rheidos.houdini.runtime import build_cook_context, get_runtime, publish_group

node = hou.pwd()
geo_out = node.geometry()
geo_out.clear()
geo_in = node.inputs()[0].geometry()
geo_out.merge(geo_in)

session = get_runtime().get_or_create_session(node)
ctx = build_cook_context(node, geo_in, geo_out, session)

publish_group(ctx, "my_group", as_mask=True)
```

### Fetch a compute resource and write it to geometry

In a Python SOP with input 0 connected:

```python
import hou
from rheidos.houdini.geo import OWNER_POINT
from rheidos.houdini.runtime import build_cook_context, get_runtime

node = hou.pwd()
geo_out = node.geometry()
geo_out.clear()
geo_in = node.inputs()[0].geometry()
geo_out.merge(geo_in)

session = get_runtime().get_or_create_session(node)
ctx = build_cook_context(node, geo_in, geo_out, session)

P = ctx.fetch("geo.P")
ctx.write(OWNER_POINT, "P", P)
```

### Parse node parameters into a NodeConfig

This assumes the node has the required parms: `script_path`, `module_path`, `mode`,
`reset_node`, `nuke_all`, `profile`, `debug_log`.

```python
import hou
from rheidos.houdini.nodes import read_node_config

node = hou.pwd()
config = read_node_config(node)
print(config)
```

## Reference (tell me the truth)

### Module: `rheidos.houdini`

Exports:
- `CookContext`
- `ComputeRuntime`
- `GEO_P`, `GEO_TRIANGLES`
- `SIM_TIME`, `SIM_DT`, `SIM_FRAME`, `SIM_SUBSTEP`
- `SessionKey`
- `WorldSession`
- `build_cook_context(node, geo_in, geo_out, session, substep=0, is_solver=False) -> CookContext`
- `get_runtime() -> ComputeRuntime`
- `make_session_key(node: hou.Node) -> SessionKey`
- `point_attrib(name: str) -> str`
- `point_group_indices(name: str) -> str`
- `point_group_mask(name: str) -> str`
- `prim_attrib(name: str) -> str`
- `publish_geometry_minimal(ctx: CookContext) -> None`
- `publish_group(ctx: CookContext, group_name: str, as_mask: bool = True) -> None`
- `publish_point_attrib(ctx: CookContext, name: str) -> None`
- `publish_prim_attrib(ctx: CookContext, name: str) -> None`
- `run_cook(node, geo_in, geo_out) -> None`
- `run_solver(node, geo_prev, geo_in, geo_out, substep=0) -> None`

### Module: `rheidos.houdini.geo`

Exports:
- `AttribDesc`
- `GeometryIO`
- `GeometrySchema`
- `OWNER_POINT`, `OWNER_PRIM`, `OWNER_VERTEX`, `OWNER_DETAIL`
- `OWNERS`

### Module: `rheidos.houdini.geo.schema`

`AttribDesc` (dataclass, frozen)
- `name: str`
- `owner: str`
- `storage_type: str`
- `tuple_size: int`

`GeometrySchema` (dataclass, frozen)
- `point: Tuple[AttribDesc, ...]`
- `prim: Tuple[AttribDesc, ...]`
- `vertex: Tuple[AttribDesc, ...]`
- `detail: Tuple[AttribDesc, ...]`
- `by_owner(owner: str) -> Tuple[AttribDesc, ...]`

Constants:
- `OWNER_POINT`, `OWNER_PRIM`, `OWNER_VERTEX`, `OWNER_DETAIL`
- `OWNERS`

### Module: `rheidos.houdini.geo.adapter`

`GeometryIO` (dataclass)
- `geo_in: hou.Geometry`
- `geo_out: Optional[hou.Geometry]`
- `clear_cache() -> None`
- `describe(owner: Optional[str] = None) -> GeometrySchema`
- `read(owner: str, name: str, dtype=None, components: Optional[int] = None) -> np.ndarray`
- `write(owner: str, name: str, values, create: bool = True) -> None`
- `read_prims(arity: int = 3) -> np.ndarray`
- `read_group(owner: str, group_name: str, as_mask: bool = False) -> np.ndarray`

### Module: `rheidos.houdini.runtime.session`

`SessionKey` (dataclass, frozen)
- `hip_path: str`
- `node_path: str`

`WorldSession` (dataclass)
- `world: Optional[World]`
- `user_module: Optional[ModuleType]`
- `user_module_key: Optional[str]`
- `did_setup: bool`
- `last_step_key: Optional[Tuple[Any, ...]]`
- `last_output_cache: Dict[str, np.ndarray]`
- `last_geo_snapshot: Optional[Any]`
- `last_triangles: Optional[np.ndarray]`
- `last_topology_sig: Optional[Tuple[int, int, int]]`
- `last_topology_key: Optional[Tuple[Any, ...]]`
- `last_error: Optional[BaseException]`
- `last_traceback: Optional[str]`
- `log_entries: Deque[Dict[str, Any]]`
- `stats: Dict[str, Any]`
- `created_at: float`
- `last_cook_at: Optional[float]`
- `reset(reason: str) -> None`
- `record_error(exc: BaseException, tb_str: str) -> None`
- `clear_error() -> None`

`ComputeRuntime`
- `sessions: Dict[SessionKey, WorldSession]`
- `get_or_create_session(node: hou.Node) -> WorldSession`
- `reset_session(node: hou.Node, reason: str) -> None`
- `nuke_all(reason: str) -> None`

Module helpers:
- `get_runtime() -> ComputeRuntime`
- `make_session_key(node: hou.Node) -> SessionKey`

### Module: `rheidos.houdini.nodes`

Exports:
- `build_assets(output_path: Optional[str] = None) -> str`
- `NodeConfig`
- `read_node_config(node: hou.Node) -> NodeConfig`

### Module: `rheidos.houdini.runtime.cook_context`

`CookContext` (dataclass)
- `node: hou.Node`
- `frame: float`
- `time: float`
- `dt: float`
- `substep: int`
- `is_solver: bool`
- `session: WorldSession`
- `geo_in: hou.Geometry`
- `geo_out: hou.Geometry`
- `io: GeometryIO`
- `schema: Optional[GeometrySchema]`
- `world() -> World`
- `clear_cache() -> None`
- `describe(owner: Optional[str] = None) -> GeometrySchema`
- `read(owner: str, name: str, dtype=None, components: Optional[int] = None) -> np.ndarray`
- `write(owner: str, name: str, values, create: bool = True) -> None`
- `read_prims(arity: int = 3) -> np.ndarray`
- `read_group(owner: str, group_name: str, as_mask: bool = False) -> np.ndarray`
- `read_group_default(owner: str, group_name: str, as_mask: Optional[bool] = None) -> np.ndarray`
- `P() -> np.ndarray`
- `set_P(values) -> None`
- `triangles() -> np.ndarray`
- `publish(key: str, value) -> None`
- `publish_many(items: Dict[str, Any]) -> None`
- `fetch(key: str) -> Any`
- `ensure(key: str) -> None`
- `log(message: str, **payload) -> None`

Functions:
- `build_cook_context(node, geo_in, geo_out, session, substep=0, is_solver=False) -> CookContext`

### Module: `rheidos.houdini.runtime.driver`

Functions:
- `run_cook(node, geo_in, geo_out) -> None`
- `run_solver(node, geo_prev, geo_in, geo_out, substep=0) -> None`

### Module: `rheidos.houdini.runtime.user_script`

Functions:
- `resolve_user_module(session: WorldSession, config: NodeConfig, node: hou.Node) -> ModuleType`

### Module: `rheidos.houdini.runtime.resource_keys`

Constants:
- `GEO_P`, `GEO_TRIANGLES`
- `SIM_TIME`, `SIM_DT`, `SIM_FRAME`, `SIM_SUBSTEP`

Functions:
- `point_attrib(name: str) -> str`
- `prim_attrib(name: str) -> str`
- `point_group_mask(name: str) -> str`
- `point_group_indices(name: str) -> str`

### Module: `rheidos.houdini.runtime.publish`

Functions:
- `publish_geometry_minimal(ctx: CookContext) -> None`
- `publish_group(ctx: CookContext, group_name: str, as_mask: bool = True) -> None`
- `publish_point_attrib(ctx: CookContext, name: str) -> None`
- `publish_prim_attrib(ctx: CookContext, name: str) -> None`

### Module: `rheidos.houdini.runtime.taichi_reset`

- `reset_taichi_hard() -> None`

### Module: `rheidos.houdini.nodes.config`

`NodeConfig` (dataclass, frozen)
- `script_path: Optional[str]`
- `module_path: Optional[str]`
- `mode: str`
- `reset_node: bool`
- `nuke_all: bool`
- `profile: bool`
- `debug_log: bool`

Functions:
- `read_node_config(node: hou.Node) -> NodeConfig`

### Script: `rheidos.houdini.scripts.smoke`

- `main() -> None`

### Script: `rheidos.houdini.scripts.cook_sop`

- `main() -> None`

### Script: `rheidos.houdini.scripts.solver_sop`

- `main() -> None`

### Script: `rheidos.houdini.scripts.cook_demo`

- `cook(ctx) -> None`

### Script: `rheidos.houdini.scripts.solver_demo`

- `setup(ctx) -> None`
- `step(ctx) -> None`

## Explanation (help me understand why)

### Why HDAs and OTL files exist

HDAs package a node's UI, scripts, and defaults into a reusable Houdini node type.
The `.otl` file is an asset library that Houdini scans to register those node types.
This removes manual Python SOP setup and guarantees the expected parameter names.

### What geometry is required

The HDA itself does not require any fixed attributes. The runtime expects point
positions `P` (always present in Houdini geometry) and, when `publish_geometry_minimal`
is used, triangle topology. Any other attributes are only required if your script
reads or writes them.

### Why triangles are required right now

The compute bridge publishes `geo.triangles` as explicit topology for downstream
compute nodes. That keeps the data contract clear and avoids hidden triangulation.

### Why GeometryIO is bulk and owner-based

Houdini geometry is naturally segmented by owner (point, prim, vertex, detail). Using
owner strings keeps the API predictable and avoids special cases, while bulk IO keeps
attribute access fast and consistent across data types.

### Why CookContext is thin

CookContext is a narrow wrapper: it stores timing and session data, exposes geometry IO,
and forwards to the compute world. This keeps node code small and makes it easy to test
IO, publish, and fetch separately.

### Why standardized resource keys exist

A stable key schema makes scripts interchangeable and reduces string drift across nodes.
Keys like `geo.P` and `geo.triangles` act as a shared contract between Houdini and the
compute modules.

### Why cook-local caching exists

Repeated reads inside a single cook are common. A cook-local cache avoids repeated
attribute pulls without hiding changes across cooks.

### Why a session cache exists

Houdini cooks can run repeatedly, often within a single UI session. The session cache
keeps compute state tied to a specific hip file and node path so that repeated cooks can
re-use a world and maintain solver state when needed. This avoids accidental cross-node
state sharing while preserving deterministic behavior per node.

### Why cook and solver drivers are separate

Cook is stateless and recalculates each time. Solver keeps state across frames and needs
different control flow: setup once, step per frame, and deterministic skip behavior.

### Why solver uses a step key and geometry snapshot

Houdini can re-cook the same frame/substep multiple times. A stable step key prevents
double-stepping, and the snapshot lets the driver re-emit the last result without
re-running user code.

### Why `out.P` exists alongside direct geometry writes

Direct `ctx.write`/`ctx.set_P` keeps Houdini the source of truth. The `out.P` path is an
optional bridge for users who prefer to push outputs into the compute registry.

### Why there is no hot reload

Houdini state can be subtle and hard to reason about if code reloads implicitly. This
integration is designed to be explicit: you either reset a node or "nuke all" to start
from a clean state. This keeps reproducibility and debugging predictable.

### Why Taichi is reset on "nuke all"

Taichi can carry global state across runs. A hard reset clears kernels and global caches
so that a global reset in Houdini truly means "start from scratch."

### Why parameter parsing is strict

Node parameters define the user-facing contract. The parser raises if required parms are
missing, which surfaces configuration errors early and keeps node scripts deterministic.

### Current scope

This package provides runtime session management, parameter parsing, geometry adapters,
CookContext helpers, standardized resource keys, publish utilities, diagnostics hooks,
HDA build tooling, and node drivers for cook/solver modes (plus minimal demo scripts).
