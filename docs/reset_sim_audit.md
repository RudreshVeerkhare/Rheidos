# Reset/Reload No-Zombie Audit

## Codebase scan notes

- Searched for module-level singletons (`SIM = ...`) and module-level Taichi fields; none found.
- Searched for `lru_cache` usage; none found.
- Existing state is rooted in `WorldSession` / `ComputeRuntime` and cached on Houdini sessions.

## Changes made for reset safety

- Added a `SimContext` rooted at `hou.session.RHEIDOS_SIM` with explicit teardown.
- Added a Houdini-scoped dev state with busy tracking to block reload during active cooks.
- Added a reset pipeline that: tears down sessions, syncs/resets/re-inits Taichi, purges/reloads modules, and rehydrates context.
- Centralized Taichi init via `taichi_runtime` to disable offline cache by default in dev.
