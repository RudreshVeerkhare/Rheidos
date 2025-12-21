# ImGui Panels (Plugins)

ImGui panels are lightweight plugins rendered inside the existing “Rheidos Tools” window (same one that lists controller actions). Panels are discovered from explicit factories you pass to the engine, so adding or removing panels never requires touching core UI code.

## Panel contract

Implement an object with:

- `id`: stable string (used for `##` ImGui IDs)
- `title`: display name in the collapsing header
- `order`: integer for ordering among panels
- `draw(imgui)`: render function; gets the `imgui_bundle.imgui` module

Example panel:

```python
from rheidos.ui.imgui_manager import ImGuiPanel

class StatsPanel:
    id = "stats"
    title = "Stats"
    order = 0

    def __init__(self, engine):
        self.engine = engine

    def draw(self, imgui):
        fps = self.engine.session.clock.get_dt() if hasattr(self.engine.session, "clock") else 0.0
        imgui.text(f"dt: {fps:.4f}")
```

## Registering panels

Panels are created from **panel factories**: callables `(session, store) -> panel_or_none`. Register them explicitly:

- At engine construction:

```python
from rheidos.engine import Engine
from rheidos.ui.panels.store_state import StoreStatePanel

eng = Engine(
    window_title="Rheidos",
    imgui_panel_factories=(
        lambda session, store: StoreStatePanel(store=store),
        lambda session, store: StatsPanel(eng),  # capture engine if needed
    ),
)
```

- After engine is running (render thread-safe helpers):

```python
eng.add_imgui_panel_factory(lambda session, store: StatsPanel(eng))
# or replace the whole set:
eng.set_imgui_panel_factories([lambda session, store: StoreStatePanel(store)])
```

If `imgui_panel_factories` is omitted and `panda3d-imgui` is available, the engine installs the default `StoreStatePanel`.

## StoreState panel

`rheidos.ui.panels.store_state.StoreStatePanel` pretty-prints the live `StoreState` with light sanitization:

- Polls the store every `0.5s` (configurable `refresh_interval`)
- Max 50 items per dict/sequence (`max_items`)
- Max nesting depth 3 (`max_depth`)
- Truncates `repr` strings to 140 chars (`max_repr_len`)
- Falls back to `(empty)` if the store is missing or empty

You can override parameters by providing your own factory:

```python
eng = Engine(
    imgui_panel_factories=(
        lambda session, store: StoreStatePanel(
            store=store,
            refresh_interval=1.0,
            max_items=100,
            max_depth=4,
            max_repr_len=200,
        ),
    )
)
```

## Caveats and behaviors

- Requires `panda3d-imgui` + `p3dimgui`; if unavailable, panels are skipped (DirectGUI still handles controller actions).
- Panels share the single “Rheidos Tools” window; headers use `##id` to avoid label collisions.
- Panel `draw` should be cheap; heavy work can hurt frame time. Do throttling inside the panel (as `StoreStatePanel` does).
- Panel exceptions are swallowed to keep the UI alive; log inside the panel if you need diagnostics.
- StoreState panel shows sanitized snapshots; large collections are trimmed and deeply nested data collapses to `"..."` markers.
- Factories run on ImGuiUIManager init or when set via `set_imgui_panel_factories`; if a factory raises or returns `None`, it is skipped.
