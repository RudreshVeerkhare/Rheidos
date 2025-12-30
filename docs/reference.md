# Reference

This reference lists the public API surface in the compute and Houdini packages.

## Module: rheidos.compute

Exports from `rheidos.compute`:
- `FieldLike`, `ResourceName`, `Shape`, `ShapeFn`
- `ModuleBase`, `Namespace`, `World`
- `ProducerBase`, `Registry`
- `Resource`, `ResourceKey`, `ResourceKind`, `ResourceRef`, `ResourceSpec`
- `WiredProducer`, `out_field`

### ResourceSpec (dataclass, frozen)
- `kind: ResourceKind` (`"taichi_field"`, `"numpy"`, `"python"`)
- `dtype: Optional[Any]`
- `lanes: Optional[int]`
- `shape: Optional[Shape]`
- `shape_fn: Optional[ShapeFn]`
- `allow_none: bool`

### Resource (dataclass)
- `name: ResourceName`
- `buffer: Any`
- `deps: Tuple[ResourceName, ...]`
- `producer: Optional[ProducerBase]`
- `version: int`
- `dep_sig: Tuple[Tuple[ResourceName, int], ...]`
- `description: str`
- `spec: Optional[ResourceSpec]`

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

### ProducerBase
- `outputs: Tuple[ResourceName, ...]`
- `compute(reg: Registry) -> None`

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
- `explain(name, depth=4) -> str`

### WiredProducer[IO]
- Requires IO to be a dataclass.
- Outputs are inferred from fields marked with `out_field()`.

### ModuleBase
- `NAME: str`
- `resource(attr, *, spec=None, doc="", declare=False, buffer=None, deps=(), producer=None, description="") -> ResourceRef`
- `declare_resource(ref, *, buffer=None, deps=(), producer=None, description="") -> None`
- `require(module_cls) -> module instance`
- `prefix: str`
- `qualify(attr: str) -> str`

### World
- `reg: Registry`
- `require(module_cls, *, scope="") -> module instance`

## Module: rheidos.houdini

Exports:
- `CookContext`, `ComputeRuntime`, `WorldSession`, `SessionKey`
- `build_cook_context(node, geo_in, geo_out, session, substep=0, is_solver=False) -> CookContext`
- `get_runtime() -> ComputeRuntime`
- `make_session_key(node) -> SessionKey`
- `run_cook(node, geo_in, geo_out) -> None`
- `run_solver(node, geo_prev, geo_in, geo_out, substep=0) -> None`
- `publish_geometry_minimal(ctx) -> None`
- `publish_group(ctx, group_name, as_mask=True) -> None`
- `publish_point_attrib(ctx, name) -> None`
- `publish_prim_attrib(ctx, name) -> None`
- `GEO_P`, `GEO_TRIANGLES`
- `SIM_TIME`, `SIM_DT`, `SIM_FRAME`, `SIM_SUBSTEP`
- `point_attrib(name) -> str`
- `prim_attrib(name) -> str`
- `point_group_mask(name) -> str`
- `point_group_indices(name) -> str`

## Module: rheidos.houdini.geo

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

## Module: rheidos.houdini.runtime.cook_context

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
- `schema: Optional[GeometrySchema]`
- `world() -> World`
- `clear_cache() -> None`
- `describe(owner: Optional[str] = None) -> GeometrySchema`
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
- `log(message, **payload) -> None`

## Module: rheidos.houdini.runtime.session

### SessionKey (dataclass, frozen)
- `hip_path: str`
- `node_path: str`

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
- `log_entries: Deque[Dict[str, Any]]`
- `stats: Dict[str, Any]`
- `created_at: float`
- `last_cook_at: Optional[float]`
- `reset(reason: str) -> None`
- `record_error(exc, tb_str) -> None`
- `clear_error() -> None`
- `log_event(message, **payload) -> None`
- `clear_log() -> None`

### ComputeRuntime
- `sessions: Dict[SessionKey, WorldSession]`
- `get_or_create_session(node) -> WorldSession`
- `reset_session(node, reason) -> None`
- `nuke_all(reason) -> None`

Functions:
- `get_runtime() -> ComputeRuntime`
- `make_session_key(node) -> SessionKey`

## Module: rheidos.houdini.runtime.user_script

- `resolve_user_module(session, config, node) -> ModuleType`

## Module: rheidos.houdini.runtime.resource_keys

Constants:
- `GEO_P`, `GEO_TRIANGLES`
- `SIM_TIME`, `SIM_DT`, `SIM_FRAME`, `SIM_SUBSTEP`

Functions:
- `point_attrib(name) -> str`
- `prim_attrib(name) -> str`
- `point_group_mask(name) -> str`
- `point_group_indices(name) -> str`

## Module: rheidos.houdini.runtime.publish

- `publish_geometry_minimal(ctx) -> None`
- `publish_group(ctx, group_name, as_mask=True) -> None`
- `publish_point_attrib(ctx, name) -> None`
- `publish_prim_attrib(ctx, name) -> None`

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
