from __future__ import annotations

import os
import queue
import time
import asyncio
from typing import Any, Callable, Dict, Optional

try:
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
    from panda3d.core import ClockObject, WindowProperties, loadPrcFileData
except Exception as e:  # pragma: no cover
    ShowBase = None  # type: ignore
    Task = None  # type: ignore
    loadPrcFileData = None  # type: ignore

from .session import PandaSession
from .store import StoreState
from .abc.view import View
from .abc.observer import Observer
from .abc.controller import Controller


class Engine:
    """
    Panda3D engine that:
      - In blocking apps: base.run() (same as stock Panda)
      - In notebooks: uses an asyncio task that calls taskMgr.step() at target FPS,
        so cells remain interactive and you can add/remove tasks/views live.
    """

    def __init__(
        self,
        window_title: str = "Kung Fu Panda",
        window_size: tuple[int, int] = (1280, 720),
        fps: int = 60,
        interactive: bool = False,
        auto_start: Optional[bool] = None,
        msaa_samples: Optional[int] = 4,
    ) -> None:
        if ShowBase is None:
            raise RuntimeError("Panda3D is not available. Install 'panda3d'.")

        loadPrcFileData("", f"window-title {window_title}")
        loadPrcFileData("", f"win-size {window_size[0]} {window_size[1]}")
        # Try to enable multisample anti-aliasing for smoother overlays/lines.
        if msaa_samples and msaa_samples > 0:
            try:
                loadPrcFileData("", "framebuffer-multisample 1")
                loadPrcFileData("", f"multisamples {int(msaa_samples)}")
            except Exception:
                pass

        # Important: construct ShowBase on the main (notebook) thread.
        self._base = ShowBase(windowType="onscreen")
        self._base_title = window_title
        self._fps_title_interval = 0.5
        self._last_fps_title = 0.0

        # Friendly default camera/background
        try:
            self._base.disableMouse()
            self._base.camera.setPos(3.0, -6.0, 3.0)
            self._base.camera.lookAt(0.0, 0.0, 0.0)
            self._base.setBackgroundColor(0.05, 0.05, 0.08, 1.0)
        except Exception:
            pass

        self._session = PandaSession(self._base)
        self._store = StoreState()

        self._views: Dict[str, View] = {}
        self._view_tasks: Dict[str, str] = {}
        self._observers: Dict[str, Observer] = {}
        self._observer_tasks: Dict[str, str] = {}
        self._controllers: Dict[str, Controller] = {}

        self._interactive = interactive
        self._auto_start = interactive if auto_start is None else bool(auto_start)
        self._fps = int(max(1, fps))
        self._paused = False

        # Notebook async loop state
        self._running = False
        self._runner_task: Optional[asyncio.Task] = None

        # Dispatch queue (processed every frame in a very-early task)
        self._dispatch_q: "queue.Queue[Callable[[], None]]" = queue.Queue()

        # Service & FPS title tasks
        self._service_task_name = "engine-service"
        self._session.task_mgr.add(
            self._service_task, name=self._service_task_name, sort=-1000
        )

        self._fps_task_name = "engine-fps-title"
        self._session.task_mgr.add(
            self._fps_title_task, name=self._fps_task_name, sort=-995
        )

        if self._interactive and self._auto_start:
            # fire-and-forget: schedule start on the current loop if present
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.start_async(self._fps))
            except RuntimeError:
                # No running event loop => we're not in a notebook; fall back to blocking run
                self.start()

    # ---------------- Public API ----------------

    @property
    def store(self) -> StoreState:
        return self._store

    @property
    def session(self) -> PandaSession:
        return self._session

    def start(self) -> None:
        """
        Blocking start (typical desktop app).
        If interactive=True, prefer start_async() in a notebook instead.
        """
        if self._interactive:
            # If someone explicitly called start() in interactive mode outside a notebook,
            # keep the async machinery and avoid GL/thread issues.
            asyncio.run(self._main_async_run())
        else:
            self._base.run()

    async def start_async(self, fps: Optional[int] = None) -> None:
        """
        Notebook-friendly start: schedules an asyncio task that steps Panda3D.
        Idempotent: safe to call multiple times.
        """
        if self._running:
            return
        if fps is not None:
            self._fps = int(max(1, fps))
        self._running = True
        self._runner_task = asyncio.create_task(self._run_loop(), name="P3DAsyncLoop")

    def is_running(self) -> bool:
        return self._running

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self._store.set("paused", paused)

    def is_paused(self) -> bool:
        return self._paused

    async def stop_async(self) -> None:
        """
        Notebook-friendly stop: awaits the render loop to finish.
        """
        if not self._running:
            return
        self._running = False
        if self._runner_task:
            try:
                await self._runner_task
            finally:
                self._runner_task = None

    def stop(self) -> None:
        """
        Stop for blocking apps. In a notebook, prefer `await stop_async()`.
        """
        if self._running:
            # best-effort cooperative stop on current loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.stop_async())
            except RuntimeError:
                # no running loop; spin until the task exits
                self._running = False
                t0 = time.perf_counter()
                while (
                    self._runner_task
                    and not self._runner_task.done()
                    and (time.perf_counter() - t0) < 2.0
                ):
                    time.sleep(0.01)
                self._runner_task = None

    # ----- Views / Observers / Controllers (run on render thread == main thread) -----

    def add_view(self, view: View) -> None:
        def _impl() -> None:
            name = view.name
            if name in self._views:
                raise ValueError(f"View '{name}' already exists")
            view.setup(self._session)
            task_name = f"view-{name}"
            # Use keyword args so we never trip on uponDeath/appendTask positions.
            self._session.task_mgr.add(
                lambda task: self._view_task_wrapper(view, task),
                name=task_name,
                sort=getattr(view, "sort", 0),
            )
            self._views[name] = view
            self._view_tasks[name] = task_name
            view._enabled = True
            try:
                view.on_enable()
            except Exception:
                pass

        self._run_on_render_thread(_impl)

    def remove_view(self, name: str) -> None:
        def _impl() -> None:
            task_name = self._view_tasks.pop(name, None)
            if task_name:
                self._session.task_mgr.remove(task_name)
            view = self._views.pop(name, None)
            if view:
                try:
                    view.on_disable()
                except Exception:
                    pass
                view.teardown()
                view._enabled = False

        self._run_on_render_thread(_impl)

    def enable_view(self, name: str, enabled: bool = True) -> None:
        def _impl() -> None:
            if name not in self._views:
                return
            existing = self._view_tasks.get(name)
            view = self._views[name]
            if enabled and not existing:
                task_name = f"view-{name}"
                try:
                    view.on_enable()
                except Exception:
                    pass
                self._session.task_mgr.add(
                    lambda task: self._view_task_wrapper(view, task),
                    name=task_name,
                    sort=getattr(view, "sort", 0),
                )
                self._view_tasks[name] = task_name
                view._enabled = True
            elif not enabled and existing:
                self._session.task_mgr.remove(existing)
                self._view_tasks.pop(name, None)
                view._enabled = False
                try:
                    view.on_disable()
                except Exception:
                    pass

        self._run_on_render_thread(_impl)

    def add_observer(self, observer: Observer) -> None:
        def _impl() -> None:
            name = observer.name
            if name in self._observers:
                raise ValueError(f"Observer '{name}' already exists")
            observer.setup(self._session)
            task_name = f"observer-{name}"
            self._session.task_mgr.add(
                lambda task: self._observer_task_wrapper(observer, task),
                name=task_name,
                sort=getattr(observer, "sort", 0),
            )
            self._observers[name] = observer
            self._observer_tasks[name] = task_name

        self._run_on_render_thread(_impl)

    def remove_observer(self, name: str) -> None:
        def _impl() -> None:
            task_name = self._observer_tasks.pop(name, None)
            if task_name:
                self._session.task_mgr.remove(task_name)
            obs = self._observers.pop(name, None)
            if obs:
                obs.teardown()

        self._run_on_render_thread(_impl)

    def add_controller(self, controller: Controller) -> None:
        def _impl() -> None:
            name = controller.name
            if name in self._controllers:
                raise ValueError(f"Controller '{name}' already exists")
            controller.attach(self._session)
            self._controllers[name] = controller

        self._run_on_render_thread(_impl)

    def remove_controller(self, name: str) -> None:
        def _impl() -> None:
            ctrl = self._controllers.pop(name, None)
            if ctrl is not None:
                ctrl.detach()

        self._run_on_render_thread(_impl)

    def screenshot(self, filename: str, use_default: bool = False) -> None:
        def _impl() -> None:
            if self._base.win is None:
                raise RuntimeError("No active window for screenshot")

            if not use_default:
                _, ext = os.path.splitext(filename)
                if ext:
                    self._base.win.saveScreenshot(filename)
                    return
            self._base.screenshot(namePrefix=filename, defaultFilename=use_default)

        self._run_on_render_thread(_impl)

    def dispatch(self, fn: Callable[[], None]) -> None:
        """Queue a callable to run on the next frame on the render thread."""
        self._dispatch_q.put(fn)

    # ---------------- Internals ----------------

    async def _main_async_run(self) -> None:
        """Helper to run async loop in non-notebook environments."""
        await self.start_async(self._fps)
        try:
            while self._running:
                await asyncio.sleep(0.1)
        finally:
            await self.stop_async()

    async def _run_loop(self) -> None:
        """Cooperative render loop for notebooks."""
        target_dt = 1.0 / float(self._fps)
        last = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            # Step Panda3D once
            try:
                self._base.taskMgr.step()
            except Exception:
                # If the window closed or another fatal error occurred, exit.
                break

            # Try to keep roughly target FPS
            elapsed = time.perf_counter() - now
            sleep_for = max(0.0, target_dt - elapsed)
            # Yield to notebook even if sleep_for ~ 0
            await asyncio.sleep(sleep_for)
            last = now

        # Loop exiting: ensure flag is down
        self._running = False

    def _run_on_render_thread(self, fn: Callable[[], None]) -> None:
        """
        In this design, the "render thread" is the main thread, so:
          - if the async loop is running (notebook), we queue fn for the next frame
          - otherwise, we just run fn immediately
        """
        if self._running:
            # Queue for execution during the next service task tick
            self._dispatch_q.put(fn)
        else:
            fn()

    def _service_task(self, task: Any) -> int:
        while True:
            try:
                fn = self._dispatch_q.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                # Swallow to keep the engine ticking; log if you have a logger
                pass
        return Task.cont

    def _fps_title_task(self, task: Any) -> int:
        if self._base.win is None:
            return Task.cont
        now = time.perf_counter()
        if now - self._last_fps_title >= self._fps_title_interval:
            fps = ClockObject.getGlobalClock().getAverageFrameRate()
            props = WindowProperties()
            props.setTitle(f"{self._base_title} [{fps:05.2f} FPS]")
            self._base.win.requestProperties(props)
            self._last_fps_title = now
        return Task.cont

    def _view_task_wrapper(self, view: View, task: Any) -> int:
        dt = 0.0 if self._paused else ClockObject.getGlobalClock().getDt()
        try:
            view.update(dt)
        except Exception:
            pass
        return Task.cont

    def _observer_task_wrapper(self, observer: Observer, task: Any) -> int:
        dt = 0.0 if self._paused else ClockObject.getGlobalClock().getDt()
        try:
            observer.update(dt)
        except Exception:
            pass
        return Task.cont
