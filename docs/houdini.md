# Houdini Integration Docs (Diataxis)

This document follows the Diataxis model. Pick the section that matches your intent:
learn by doing (Tutorials), solve a task (How-to guides), check facts (Reference), or
understand design choices (Explanation).

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
1) Action: Paste this into the Python SOP.
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

1) Put `packages/rheidos.json` in a Houdini packages folder.
2) Update `RHEIDOS_REPO` to the absolute repo path.
3) Restart Houdini.

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
- `build_cook_context(node, geo_in, geo_out, session, substep=0) -> CookContext`
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
- `did_setup: bool`
- `last_step_key: Optional[Tuple[Any, ...]]`
- `last_output_cache: Dict[str, np.ndarray]`
- `last_error: Optional[BaseException]`
- `last_traceback: Optional[str]`
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

### Module: `rheidos.houdini.runtime.cook_context`

`CookContext` (dataclass)
- `node: hou.Node`
- `frame: float`
- `time: float`
- `dt: float`
- `substep: int`
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
- `P() -> np.ndarray`
- `set_P(values) -> None`
- `triangles() -> np.ndarray`
- `publish(key: str, value) -> None`
- `publish_many(items: Dict[str, Any]) -> None`
- `fetch(key: str) -> Any`
- `ensure(key: str) -> None`

Functions:
- `build_cook_context(node, geo_in, geo_out, session, substep=0) -> CookContext`

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

## Explanation (help me understand why)

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

### Why there is cook-local caching

Repeated reads inside a single cook are common. A cook-local cache avoids repeated
attribute pulls without hiding changes across cooks.

### Why a session cache exists

Houdini cooks can run repeatedly, often within a single UI session. The session cache
keeps compute state tied to a specific hip file and node path so that repeated cooks can
re-use a world and maintain solver state when needed. This avoids accidental cross-node
state sharing while preserving deterministic behavior per node.

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
CookContext helpers, standardized resource keys, and basic publish utilities. Node
drivers for cook/solver modes are not yet included.
