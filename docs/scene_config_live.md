Scene config live editing
=========================

This document explains how the live scene-config editor works (`SceneConfigPanel` + `SceneConfigLiveManager`), how diffs are applied, and what to watch out for when integrating custom views/controllers or running long simulations.

Components
----------
- `rheidos.scene_config_live.SceneConfigLiveManager`: holds the last applied config + `SceneConfigResult`, computes diffs, and applies changes on the render thread.
- `rheidos.ui.panels.scene_config_panel.SceneConfigPanel`: ImGui UI that edits the config text, applies diffs or force reloads, and optionally saves to disk.
- Demo wiring: `rheidos/examples/point_selection.py` registers the panel when the scene config contains `ui.scene_config_panel: true`.

Apply flow
----------
1) Panel buffer holds the current config text (YAML/JSON). It does not auto-save unless you enable “Save to disk on apply” or click “Save Buffer to Disk.”
2) “Apply Diff”:
   - Parses the buffer, computes a diff against the last applied config, and applies only what changed.
   - Runs on the render thread (panel draw happens there).
3) “Force Reload”:
   - Tears down everything created from the config and rebuilds from the current buffer.
4) “Reload From Disk”:
   - Refreshes the editor buffer from the file without applying changes to the scene.

Diff rules (by section)
-----------------------
- Meshes: identity is `name` (or path stem if unnamed).
  - Added: built and inserted.
  - Removed: surface/wire views are removed.
  - Updated: if config for a named mesh changes (path/material/transform/etc.), its views are rebuilt; order follows the config.
  - Renaming acts as remove+add.
- Studio: if the `studio` block changes, the studio view is recreated; otherwise, ground snapping updates if bounds are available.
- Lights: reapplied when `lights` changes or when studio is rebuilt.
- Camera: reapplied when `camera` changes or when auto_frame is on and bounds change.
- Background: reapplied when `background_color` changes.
- Custom views/controllers: if `views` or `controllers` sections change, existing ones from config are removed and rebuilt wholesale.

Threading
---------
All scene mutations must happen on the render thread. The panel runs inside the ImGui draw (render thread), so its calls are safe. If you use the manager outside the panel, queue calls via `engine.dispatch(...)`.

Caveats and limitations
-----------------------
- Name identity: duplicate mesh names will collide; give meshes stable, unique names.
- Side effects: custom view/controller factories may keep global state or spawn threads; when sections change, old ones are removed and new ones are built, so code should tolerate attach/detach cycles.
- Resource reuse: updated meshes are rebuilt from disk; heavy assets may stutter on apply. Use force reload sparingly with large scenes.
- External references: controllers/views outside the config that hold references to config-created NodePaths may break if those nodes are rebuilt. Prefer looking up by name each frame or using weak references.
- Simulation state: apply/force reload does not preserve per-frame state inside config-created views/controllers. Detaching will reset their internal state.
- Schema diffs: only top-level sections listed above are considered. Adding new schema keys requires extending the diff logic in `SceneConfigLiveManager`.
- ImGui availability: if `panda3d-imgui` is missing, the panel factory is skipped; the scene still loads normally.

Integration guidelines for future views/controllers
---------------------------------------------------
- Make constructors idempotent and side-effect-light so they can be rebuilt safely.
- Keep names stable and unique; use `name` in YAML for meshes and custom components.
- For controllers that depend on scene geometry, add guards so they handle missing views gracefully when a reload happens mid-run.
- Avoid caching NodePaths from config-created views; resolve by name when needed or hook into the engine store to share state.
- If you need to preserve state across reloads, store it in `Engine.store` and restore it in your view/controller `setup`.

Debugging tips
--------------
- Panel status line shows the diff summary; errors surface inline.
- Use “Force Reload” to recover from inconsistent state after schema changes.
- When adding new config-driven capabilities, write pure diff helpers and unit-test them before wiring into the manager.
