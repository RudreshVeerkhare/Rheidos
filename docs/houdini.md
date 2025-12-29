# Houdini Integration Docs (Diataxis)

This document follows the Diataxis model. Pick the section that matches your intent:
learn by doing (Tutorials), solve a task (How-to guides), check facts (Reference), or
understand design choices (Explanation).

## Tutorials (teach me)

### Run the smoke test in hython

Audience: Houdini users who want a quick end-to-end import check.

Prerequisites:
- Houdini installed with hython available.
- The repository available on disk.
- `packages/rheidos.json` placed in your Houdini packages folder.

Steps:
1) Action: Place the package file in a Houdini packages directory.
   Context: This adds the repo to `PYTHONPATH` when Houdini starts.
   Snippet:
   ```json
   {
     "env": {
       "RHEIDOS_REPO": "/Users/codebox/dev/kung_fu_panda",
       "PYTHONPATH": "$PYTHONPATH:$RHEIDOS_REPO"
     }
   }
   ```
   Result: Houdini can import `rheidos` from this repo.
2) Action: Run the smoke script with hython.
   Context: This validates `hou`, `taichi`, and `rheidos` imports.
   Snippet:
   ```sh
   hython rheidos/houdini/scripts/smoke.py
   ```
   Result: You see lines for Houdini, Python, Taichi, and "rheidos: import ok".

### Create a runtime session in the Houdini Python Shell

Audience: Houdini users who want to verify the session cache behavior.

Prerequisites:
- A running Houdini session.
- Any node in the scene (for example a Geometry node).

Steps:
1) Action: Select a node in the Houdini UI.
   Context: The session key is derived from the hip file path and node path.
   Result: A node is active/selected in the network editor.
2) Action: Create or fetch a session from the Python Shell.
   Snippet:
   ```python
   import hou
   from rheidos.houdini.runtime import get_runtime

   node = hou.selectedNodes()[0]
   session = get_runtime().get_or_create_session(node)
   print(session)
   ```
   Result: A `WorldSession` object prints without errors.
3) Action: Reset the session.
   Snippet:
   ```python
   get_runtime().reset_session(node, reason="tutorial reset")
   ```
   Result: The session is cleared and ready for a clean next cook.

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
- `ComputeRuntime`
- `SessionKey`
- `WorldSession`
- `get_runtime() -> ComputeRuntime`
- `make_session_key(node: hou.Node) -> SessionKey`

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

This package currently provides runtime session management, parameter parsing, and a
smoke test. Geometry adapters, CookContext, and node drivers should be documented in
their own tutorials, how-tos, and references once implemented.
