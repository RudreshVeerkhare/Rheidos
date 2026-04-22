# Reference

This reference lists the supported public API surface in the compute and
Houdini packages that back the active `rheidos/apps/p2` application.

## Module: `rheidos.logger`

Exports from `rheidos.logger` and `rheidos`:

- `logger`
- `SimulationLogger`

### `logger`

- `configure(*, logdir=None, run_name=None) -> None`
- `log(name, value, *, category="simulation", step=None, flush=False) -> None`

Use `logger.configure(...)` to set the process-local defaults for standalone
simulations, or to override the active runtime scope before the first scalar is
written for that run.

## Module: `rheidos.compute`

Exports from `rheidos.compute`:

- `FieldLike`, `ResourceName`, `Shape`, `ShapeFn`
- `ModuleBase`, `Namespace`, `World`, `module_resource_deps`
- `Registry`
- `Resource`, `ResourceKey`, `ResourceKind`, `ResourceRef`, `ResourceSpec`
- `ResourceKindAdapter`, `register_resource_kind`
- `ProducerContext`, `ProducerResourceNamespace`, `producer`, `producer_output`
- `shape_map`, `shape_of`, `shape_from_axis`, `shape_from_scalar`, `shape_with_tail`

The supported producer authoring path is the decorator-based API. Internal
class-based producer abstractions still exist inside the runtime, but they are
not part of the public authoring surface.

### ResourceSpec (dataclass, frozen)

- `kind: ResourceKind`
- `dtype: Optional[Any]`
- `lanes: Optional[int]`
- `shape: Optional[Shape]`
- `shape_fn: Optional[ShapeFn]`
- `allow_none: bool`

Use `ResourceSpec` to validate buffers and drive automatic allocation.

### Resource (dataclass)

- `name: ResourceName`
- `buffer: Any`
- `deps: Tuple[ResourceName, ...]`
- `producer: Optional[Any]`
- `version: int`
- `dep_sig: Tuple[Tuple[ResourceName, int], ...]`
- `description: str`
- `spec: Optional[ResourceSpec]`

`Resource` is the registry's stored record for a declared value.

### ResourceKey[T] (dataclass, frozen)

- `full_name: str`
- `spec: Optional[ResourceSpec]`

### ResourceRef[T]

- `name: str`
- `spec: Optional[ResourceSpec]`
- `ensure() -> None`
- `get() -> T`
- `peek() -> T`
- `set(value: T, *, unsafe: bool = False) -> None`
- `set_buffer(value: T, *, bump: bool = False, unsafe: bool = False) -> None`
- `commit(*, unsafe: bool = False) -> None`
- `mark_fresh(*, unsafe: bool = False) -> None`
- `touch(*, unsafe: bool = False) -> None`
- `bump(*, unsafe: bool = False) -> None`

Use `ResourceRef` inside modules and producers instead of hard-coding fully
qualified resource names.

### Registry

- `declare(name, *, buffer=None, deps=(), producer=None, description="", spec=None) -> Resource`
- `get(name) -> Resource`
- `read(name, *, ensure=True) -> Any`
- `set_buffer(name, buffer, *, bump=False, unsafe=False) -> None`
- `commit(name, *, buffer=None, unsafe=False) -> None`
- `commit_many(names, *, buffers=None, unsafe=False) -> None`
- `bump(name, unsafe=False) -> None`
- `ensure(name) -> None`
- `ensure_many(names) -> None`
- `matches_spec(name, buf) -> bool`
- `declared_names() -> Set[ResourceName]`
- `undeclare_many(names) -> None`
- `explain(name, depth=4) -> str`

`Registry.ensure(...)` is the entry point that resolves dependencies and runs
bound producers when resources are stale.

### ResourceKindAdapter

- `resolve_shape(reg, spec) -> Optional[Shape]`
- `allocate(reg, spec, shape) -> Any`
- `matches_spec(reg, spec, buf) -> bool`
- `requires_shape: bool`

### register_resource_kind

- `register_resource_kind(kind: str, adapter: ResourceKindAdapter) -> None`

Register a custom buffer kind when NumPy, Python, or Taichi fields are not
enough for a module.

### ProducerResourceNamespace

- attribute access for bound refs such as `ctx.inputs.velocity`
- item access for nested names such as `ctx.inputs["mesh.V_pos"]`
- `items() -> Iterable[Tuple[str, ResourceRef[Any]]]`
- `as_dict() -> Dict[str, ResourceRef[Any]]`

### ProducerContext

- `reg`
- `inputs: ProducerResourceNamespace`
- `outputs: ProducerResourceNamespace`
- `require_inputs(*, allow_none=(), ignore=()) -> Dict[str, ResourceRef[Any]]`
- `ensure_outputs(*, strict=True, realloc=True, require_shape=True) -> Dict[str, ResourceRef[Any]]`
- `commit(**buffers) -> None`

Each decorated producer method receives a `ProducerContext`.

### producer

- `producer(*, inputs, outputs, allow_none=(), ignore=())`

Use `@producer(...)` on `ModuleBase` methods that accept a single
`ProducerContext`. Input and output names are resolved against module
attributes when `bind_producers()` runs.

### producer_output

- `producer_output(name: str, *, alloc=None)`

Wrap an output name when that output requires a custom allocator.

### Shape helpers

- `shape_map(ref, mapper) -> ShapeFn`
- `shape_of(ref) -> ShapeFn`
- `shape_from_axis(ref, axis=0, *, tail=()) -> ShapeFn`
- `shape_from_scalar(ref, *, tail=()) -> ShapeFn`
- `shape_with_tail(ref, *, tail=()) -> ShapeFn`

These helpers are intended for `ResourceSpec.shape_fn`.

### Namespace

- `parts: Tuple[str, ...]`
- `child(name: str) -> Namespace`
- `prefix: str`
- `qualify(attr: str) -> str`

### ModuleBase

- `NAME: str`
- `prefix: str`
- `qualify(attr: str) -> str`
- `resource(attr, *, spec=None, doc="", declare=False, buffer=None, deps=(), producer=None, description="") -> ResourceRef`
- `declare_resource(ref, *, buffer=None, deps=(), producer=None, description="") -> None`
- `require(module_cls, *args, **kwargs) -> module instance`
- `bind_producers() -> None`

Modules declare scoped resources and bind decorated producer methods.

### module_resource_deps

- `module_resource_deps(module, *, include=r".*", exclude=r"^$") -> Tuple[ResourceRef[Any], ...]`

Collect resource refs from a module by attribute name pattern.

### World

- `reg: Registry`
- `require(module_cls, *args, scope="", **kwargs) -> module instance`
- `module_dependencies() -> Dict[...]`

`World.require(...)` is the supported path for wiring reusable modules together.

## Module: `rheidos.houdini`

Exports:

- `AccessMode`, `CookContext`, `ComputeRuntime`, `SessionAccess`, `WorldSession`, `SessionKey`
- `build_cook_context(node, geo_in, geo_out, session, geo_inputs=None, substep=0, is_solver=False) -> CookContext`
- `get_runtime() -> ComputeRuntime`
- `make_session_key(node) -> SessionKey`
- `make_session_key_for_path(node_path) -> SessionKey`
- `session`, usable as `@session`, `@session("name")`, or `@session(key="name")`
- `run_cook(node, geo_in, geo_out) -> None`
- `run_solver(node, geo_prev, geo_in, geo_out, substep=0) -> None`
- `publish_geometry_minimal(ctx, input_index=None) -> None`
- `publish_group(ctx, group_name, as_mask=True, input_index=None) -> None`
- `publish_point_attrib(ctx, name, input_index=None) -> None`
- `publish_prim_attrib(ctx, name, input_index=None) -> None`
- `GEO_P`, `GEO_TRIANGLES`, `geo_P(index=0) -> str`, `geo_triangles(index=0) -> str`
- `SIM_TIME`, `SIM_DT`, `SIM_FRAME`, `SIM_SUBSTEP`
- `point_attrib(name, index=0) -> str`
- `prim_attrib(name, index=0) -> str`
- `point_group_mask(name, index=0) -> str`
- `point_group_indices(name, index=0) -> str`

## Module: `rheidos.houdini.debug`

### DebugConfig (dataclass, frozen)

- `enabled: bool`
- `host: str`
- `port: int`
- `port_strategy: Literal["fixed", "fallback", "auto"]`
- `allow_remote: bool`
- `take_ownership: bool`
- `owner_hint: Optional[str]`
- `log: bool`

### DebugState (dataclass)

- `started: bool`
- `host: str`
- `port: int`
- `owner_node_path: Optional[str]`
- `pid: int`
- `warned_missing_debugpy: bool`
- `warned_port_bind: bool`
- `warned_remote_host: bool`
- `warned_break_unattached: bool`
- `warned_break_failed: bool`
- `break_next: bool`
- `break_owner: Optional[str]`
- `info_printed: bool`

Functions:

- `debug_config_from_node(node) -> DebugConfig`
- `ensure_debug_server(cfg, node=None) -> DebugState`
- `request_break_next(node=None) -> None`
- `maybe_break_now(node=None) -> None`
- `consume_break_next_button(node) -> bool`

## Module: `rheidos.houdini.geo`

Exports:

- `GeometryIO`
- `GeometrySchema`, `AttribDesc`
- `OWNER_POINT`, `OWNER_PRIM`, `OWNER_VERTEX`, `OWNER_DETAIL`, `OWNERS`

### GeometryIO

- `clear_cache() -> None`
- `describe(owner: Optional[str] = None) -> GeometrySchema`
- `read(owner, name, *, dtype=None, components=None) -> np.ndarray`
- `write(owner, name, values, *, create=True) -> None`
- `read_prims(arity=3) -> np.ndarray`
- `read_group(owner, group_name, *, as_mask=False) -> np.ndarray`

### GeometrySchema

- `point: Tuple[AttribDesc, ...]`
- `prim: Tuple[AttribDesc, ...]`
- `vertex: Tuple[AttribDesc, ...]`
- `detail: Tuple[AttribDesc, ...]`
- `by_owner(owner: str) -> Tuple[AttribDesc, ...]`

### AttribDesc

- `name: str`
- `owner: str`
- `storage_type: str`
- `tuple_size: int`

## Module: `rheidos.houdini.runtime.cook_context`

### CookContext

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
- `geo_inputs: Tuple[Optional[hou.Geometry], ...]`
- `io_inputs: Tuple[Optional[GeometryIO], ...]`
- `schema: Optional[GeometrySchema]`
- `world() -> World`
- `clear_cache() -> None`
- `describe(owner: Optional[str] = None) -> GeometrySchema`
- `input_geo(index, required=True) -> Optional[hou.Geometry]`
- `input_io(index, required=True) -> Optional[GeometryIO]`
- `read(owner, name, *, dtype=None, components=None) -> np.ndarray`
- `write(owner, name, values, *, create=True) -> None`
- `read_prims(arity=3) -> np.ndarray`
- `read_group(owner, group_name, *, as_mask=False) -> np.ndarray`
- `read_group_default(owner, group_name, *, as_mask: Optional[bool] = None) -> np.ndarray`
- `P() -> np.ndarray`
- `set_P(values) -> None`
- `triangles() -> np.ndarray`
- `publish(key, value) -> None`
- `publish_many(items: Dict[str, Any]) -> None`
- `fetch(key) -> Any`
- `ensure(key) -> None`
- `session_access(node_path, *, mode="read", create=False) -> SessionAccess`
- `log(message, **payload) -> None`

### Session access examples

Read-only access (default) to another node session:

```python
with ctx.session_access("/obj/geo1/py_sop2") as other:
    values = other.reg.read("some.resource", ensure=True)
    other.log("read resource", key="some.resource")
```

Write access when you need to update the other session:

```python
with ctx.session_access("/obj/geo1/py_sop2", mode="write") as other:
    other.reg.commit("some.resource", buffer=data)
    other.log("updated resource", key="some.resource")
```

Use `create=True` to create the target session if it does not exist yet.

## Module: rheidos.houdini.runtime.session

### SessionKey (dataclass, frozen)

- `hip_path: str`
- `node_path: str`

### AccessMode (type alias)

- `"read" | "write"`

### WorldSession (dataclass)

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
- `stats: Dict[str, Any]`
- `created_at: float`
- `last_cook_at: Optional[float]`
- `reset(reason: str) -> None`
- `record_error(exc, tb_str) -> None`
- `clear_error() -> None`

### SessionAccess (dataclass)

- `session: WorldSession`
- `node_path: str`
- `mode: AccessMode`
- `reg: RegistryAccess (read/ensure; write ops require mode="write")`

### ComputeRuntime

- `sessions: Dict[SessionKey, WorldSession]`
- `get_or_create_session(node, key=None) -> WorldSession`
- `get_session_by_path(node_path, create=False) -> WorldSession`
- `session_access(node_path, mode="read", create=False) -> SessionAccess`
- `reset_session(node, reason, key=None) -> None`
- `nuke_all(reason) -> None`

Functions:

- `get_runtime() -> ComputeRuntime`
- `make_session_key(node) -> SessionKey`
- `make_session_key_for_path(node_path) -> SessionKey`
- `session`, for manual Houdini entrypoints only

Decorator examples:

```python
from rheidos.houdini.runtime import session

@session
def run_cook(session):
    ...

@session("p1")
def node1(session):
    ...

@session(key="p1")
def node2(*, session):
    ...
```

Notes:

- `@session` injects the node-local session, equivalent to the old `get_or_create_session(node)` pattern.
- `@session("name")` injects a named shared session scoped to the current `.hip` file/runtime.
- This decorator is intended only for hand-written Houdini SOP entrypoints such as `node1()`, `node2()`, `run_cook()`, and `run_solver()`. Driver-managed app callables like `cook(ctx)`, `setup(ctx)`, and `step(ctx)` are unchanged in this pass.

## Module: rheidos.houdini.runtime.user_script

- `resolve_user_module(session, config, node) -> ModuleType`

## Module: rheidos.houdini.runtime.resource_keys

Constants:

- `GEO_P`, `GEO_TRIANGLES`
- `SIM_TIME`, `SIM_DT`, `SIM_FRAME`, `SIM_SUBSTEP`

Functions:

- `geo_P(index=0) -> str`
- `geo_triangles(index=0) -> str`
- `point_attrib(name, index=0) -> str`
- `prim_attrib(name, index=0) -> str`
- `point_group_mask(name, index=0) -> str`
- `point_group_indices(name, index=0) -> str`

## Module: rheidos.houdini.runtime.publish

- `publish_geometry_minimal(ctx, input_index=None) -> None`
- `publish_group(ctx, group_name, as_mask=True, input_index=None) -> None`
- `publish_point_attrib(ctx, name, input_index=None) -> None`
- `publish_prim_attrib(ctx, name, input_index=None) -> None`

## Module: rheidos.houdini.runtime.driver

- `run_cook(node, geo_in, geo_out) -> None`
- `run_solver(node, geo_prev, geo_in, geo_out, substep=0) -> None`

## Module: rheidos.houdini.runtime.taichi_reset

- `reset_taichi_hard() -> None`

## Module: rheidos.houdini.nodes.config

### NodeConfig (dataclass, frozen)

- `script_path: Optional[str]`
- `module_path: Optional[str]`
- `mode: str`
- `reset_node: bool`
- `nuke_all: bool`
- `profile: bool`
- `debug_log: bool`

### read_node_config(node) -> NodeConfig

## Module: rheidos.houdini.nodes

Exports:

- `NodeConfig`
- `read_node_config(node) -> NodeConfig`
