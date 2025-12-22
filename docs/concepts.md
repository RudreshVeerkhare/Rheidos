Concepts

High level
- Engine owns the Panda3D window and task loop and orchestrates everything.
- PandaSession is a thin wrapper around Panda3D’s `ShowBase` for convenience.
- Views render things each frame (meshes, overlays, axes, labels, etc.).
- Controllers handle input and can enable/disable views or trigger actions.
- Observers are like views but for non‑render per‑frame work (logging, stepping a simulation, etc.).
- StoreState is a tiny thread‑safe key/value store to share flags or small pieces of state across components.

Why this split?
- Views stay focused on rendering and can be toggled or swapped freely.
- Controllers focus on input/state transitions and don’t render directly.
- Observers do background updates or bookkeeping without cluttering render code.

Engine
- File: `rheidos/engine.py`
- Blocking start: `eng.start()` (good for scripts/demos)
- Async start for notebooks: `await eng.start_async()`; stop with `await eng.stop_async()`
- Add/remove components:
  - `eng.add_view(view)` / `eng.remove_view(name)` / `eng.enable_view(name, enabled)`
  - `eng.add_controller(controller)` / `eng.remove_controller(name)`
  - `eng.add_observer(observer)` / `eng.remove_observer(name)`
- Screenshot: `eng.screenshot("frame.png")`
- Pause flag to freeze updates: `eng.set_paused(True/False)`

PandaSession
- File: `rheidos/session.py`
- Holds `base`, `task_mgr`, `render`, `win`, and `clock` (global clock)
- Exposes `.accept()` / `.ignore()` for binding/unbinding Panda3D events

Views
- Base: `rheidos/abc/view.py`
- Lifecycle methods you can override:
  - `setup(session)`: create nodes, attach geometry
  - `update(dt)`: per‑frame work (animation, material updates, etc.)
  - `teardown()`: remove/detach nodes
  - `on_enable()` / `on_disable()`: react to being toggled
- Sort order: lower `sort` runs earlier in the frame
- Example (a minimal blinking node):

  ```python
  from rheidos.abc.view import View

  class BlinkView(View):
      def setup(self, session):
          self._np = session.render.attachNewNode(self.name)
          self._t = 0.0
      def update(self, dt: float):
          self._t += dt
          visible = int(self._t * 2.0) % 2 == 0
          (self._np.show() if visible else self._np.hide())
      def teardown(self):
          self._np.removeNode()
  ```

Observers
- Base: `rheidos/abc/observer.py`
- Same shape as `View` but intended for non‑rendering per‑frame logic
- Great for stepping a simulation, sampling metrics, writing logs, etc.

Controllers
- Base: `rheidos/abc/controller.py`
- Attach to wire up inputs; detach to tear them down
- Use Panda3D’s event system via `session.accept("key", callback)`
- Built‑ins located in `rheidos/controllers`:
  - `FpvCameraController`: first‑person flycam with mouselook, roll, and WASD/QE
  - `ToggleViewController`: cycles predefined groups of view names on/off
  - `PauseController`: toggles `Engine` pause flag
  - `ScreenshotController`: saves a screenshot to file
  - `ExitController`: exits the app (and stops the engine if interactive)

StoreState
- File: `rheidos/store.py`
- Methods: `.get(key)`, `.set(key, value)`, `.update(**kvs)`, `.subscribe(key, fn)`
- A handy place to expose "global switches" or small shared data
- Example: stop a simulation when a threshold is reached and pause rendering

  ```python
  store = eng.store
  def on_heat(val):
      if val > 1.0:
          eng.set_paused(True)
  unsub = store.subscribe("heat", on_heat)
  store.set("heat", 1.2)
  ```

Data flow and threading
- Everything runs on the main thread (Panda3D render thread == main thread)
- In interactive mode, the engine steps Panda once per async tick
- Use `eng.dispatch(fn)` to schedule small thread‑safe changes to run on the next frame if you ever end up off thread (rare in typical usage)

Task ordering
- Engine inserts two internal tasks at very early sorts for dispatch and FPS title
- Your Views/Observers run according to their `sort` (lower first)
