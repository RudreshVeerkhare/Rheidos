# Actions & UI Panel

Rheidos controllers declare **Actions** (button/toggle metadata) instead of constructing UI widgets. The engine:

- binds shortcuts from actions via the `InputRouter`
- renders the actions panel (ImGui if available via `panda3d-imgui`, DirectGUI fallback)
- keeps toggle state in sync using `get_value`/`set_value`

You never touch DirectGUI from controllers; you just return actions.

## Action API (current kinds)

```python
from rheidos.abc.action import Action

Action(
    id="pause",
    label="Pause",
    kind="toggle",          # or "button"
    group="Sim",            # grouping in UI
    order=0,                # ordering within group
    shortcut="space",       # optional Panda3D key string
    invoke=lambda session, value=None: ...,   # called on click/hotkey
    get_value=lambda session: bool(...),      # toggles: read state
    set_value=lambda session, val: ...,       # toggles: write state
)
```

Controller-level ordering for the panel: set `self.ui_order` (default 0).

## Button example (Screenshot)

```python
from rheidos.abc.controller import Controller
from rheidos.abc.action import Action

class ScreenshotController(Controller):
    def __init__(self, engine, filename="shot.png"):
        super().__init__("Screenshot")
        self.engine = engine
        self.filename = filename
        self.ui_order = 10

    def actions(self):
        return (
            Action(
                id="screenshot",
                label="Take Screenshot",
                kind="button",
                group="Utils",
                order=0,
                shortcut="p",
                invoke=lambda session, value=None: self.engine.screenshot(self.filename),
            ),
        )
```

Result: a button in the “Utils” group and a `p` hotkey; clicking or pressing `p` calls `engine.screenshot`.

## Toggle example (Pause)

```python
class PauseController(Controller):
    def __init__(self, engine):
        super().__init__("Pause")
        self.engine = engine
        self.ui_order = -5

    def actions(self):
        return (
            Action(
                id="pause",
                label="Paused",
                kind="toggle",
                group="Sim",
                order=0,
                shortcut="space",
                get_value=lambda session: self.engine.is_paused(),
                set_value=lambda session, v: self.engine.set_paused(bool(v)),
                invoke=lambda session, v=None: self.engine.set_paused(
                    not self.engine.is_paused() if v is None else bool(v)
                ),
            ),
        )
```

Result: a checkbox + `space` hotkey. The panel reflects external changes because `get_value` is polled (~5 Hz) and `set_value`/`invoke` write back.

## Mesh visibility toggle (from `examples/gui_mesh_toggle.py`)

```python
class MeshVisibilityController(Controller):
    def __init__(self, engine, view_name="mesh"):
        super().__init__("Mesh Visibility")
        self.engine = engine
        self.view_name = view_name
        self.ui_order = -10

    def _is_visible(self):
        view = self.engine._views.get(self.view_name)
        return bool(getattr(view, "_enabled", getattr(view, "enabled", False))) if view else False

    def _set_visible(self, vis: bool):
        self.engine.enable_view(self.view_name, bool(vis))

    def actions(self):
        return (
            Action(
                id="toggle-mesh",
                label="Mesh Visible",
                kind="toggle",
                group="Views",
                order=0,
                shortcut="v",
                get_value=lambda session: self._is_visible(),
                set_value=lambda session, val: self._set_visible(bool(val)),
                invoke=lambda session, val=None: self._set_visible(
                    not self._is_visible() if val is None else bool(val)
                ),
            ),
        )
```

## Notes & behaviors

- **Hotkey capture:** `InputRouter` skips actions when ImGui wants keyboard/mouse (`io.want_capture_*`) to avoid duplicate handling while typing in UI.
- **ImGui vs DirectGUI:** If `panda3d-imgui` is installed, the panel uses ImGui (drawn every frame). Otherwise, the DirectGUI panel is rebuilt on controller changes.
- **Thread safety:** actions run on the render thread (same as controllers). Keep callbacks lightweight; for heavy work, queue via `engine.dispatch`.
- **Grouping/order:** sort is `(controller.ui_order, controller.name)` then `(action.group, action.order, action.label)`, so panels stay stable as you add more controllers.
