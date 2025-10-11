from __future__ import annotations

import os
import queue
import threading
import time
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
    def __init__(
        self,
        window_title: str = "Kung Fu Panda",
        window_size: tuple[int, int] = (1280, 720),
        fps: int = 60,
        interactive: bool = False,
    ) -> None:
        if ShowBase is None:
            raise RuntimeError("Panda3D is not available. Install 'panda3d'.")

        loadPrcFileData("", f"window-title {window_title}")
        loadPrcFileData("", f"win-size {window_size[0]} {window_size[1]}")

        self._base = ShowBase()
        self._base_title = window_title
        self._fps_title_interval = 0.5
        self._last_fps_title = 0.0
        # Set a reasonable default camera pose so origin-centered content is visible.
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
        self._fps = fps
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._run_flag = threading.Event()
        self._dispatch_q: "queue.Queue[Callable[[], None]]" = queue.Queue()

        # Service task to process dispatch queue at very early sort
        self._service_task_name = "engine-service"
        self._session.task_mgr.add(self._service_task, self._service_task_name, sort=-1000)

        self._fps_task_name = "engine-fps-title"
        self._session.task_mgr.add(self._fps_title_task, self._fps_task_name, sort=-995)

    # Public API
    @property
    def store(self) -> StoreState:
        return self._store

    @property
    def session(self) -> PandaSession:
        return self._session

    def start(self) -> None:
        if self._interactive:
            if self._thread and self._thread.is_alive():
                return
            self._run_flag.set()
            self._thread = threading.Thread(target=self._interactive_loop, name="P3DLoop", daemon=True)
            self._thread.start()
        else:
            self._base.run()

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._run_flag.clear()
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self._store.set("paused", paused)

    def is_paused(self) -> bool:
        return self._paused

    def add_view(self, view: View) -> None:
        name = view.name
        if name in self._views:
            raise ValueError(f"View '{name}' already exists")
        view.setup(self._session)
        task_name = f"view-{name}"
        self._session.task_mgr.add(lambda task: self._view_task_wrapper(view, task), task_name, sort=view.sort)
        self._views[name] = view
        self._view_tasks[name] = task_name
        view._enabled = True
        try:
            view.on_enable()
        except Exception:
            pass
        view._enabled = True

    def remove_view(self, name: str) -> None:
        if name in self._view_tasks:
            self._session.task_mgr.remove(self._view_tasks[name])
            view = self._views[name]
            try:
                view.on_disable()
            except Exception:
                pass
            view.teardown()
            view._enabled = False
            del self._view_tasks[name]
            del self._views[name]

    def enable_view(self, name: str, enabled: bool = True) -> None:
        if name not in self._views:
            return
        # Remove and re-add to toggle; simplest approach
        existing = self._view_tasks.get(name)
        if enabled and not existing:
            task_name = f"view-{name}"
            view = self._views[name]
            try:
                view.on_enable()
            except Exception:
                pass
            self._session.task_mgr.add(lambda task: self._view_task_wrapper(view, task), task_name, sort=view.sort)
            self._view_tasks[name] = task_name
            view._enabled = True
        elif not enabled and existing:
            self._session.task_mgr.remove(existing)
            del self._view_tasks[name]
            view = self._views[name]
            view._enabled = False
            try:
                view.on_disable()
            except Exception:
                pass

    def add_observer(self, observer: Observer) -> None:
        name = observer.name
        if name in self._observers:
            raise ValueError(f"Observer '{name}' already exists")
        observer.setup(self._session)
        task_name = f"observer-{name}"
        self._session.task_mgr.add(lambda task: self._observer_task_wrapper(observer, task), task_name, sort=observer.sort)
        self._observers[name] = observer
        self._observer_tasks[name] = task_name

    def remove_observer(self, name: str) -> None:
        if name in self._observer_tasks:
            self._session.task_mgr.remove(self._observer_tasks[name])
            self._observers[name].teardown()
            del self._observer_tasks[name]
            del self._observers[name]

    def add_controller(self, controller: Controller) -> None:
        name = controller.name
        if name in self._controllers:
            raise ValueError(f"Controller '{name}' already exists")
        controller.attach(self._session)
        self._controllers[name] = controller

    def remove_controller(self, name: str) -> None:
        ctrl = self._controllers.pop(name, None)
        if ctrl is not None:
            ctrl.detach()

    def screenshot(self, filename: str, use_default: bool = False) -> None:
        if self._base.win is None:
            raise RuntimeError("No active window for screenshot")

        if not use_default:
            _, ext = os.path.splitext(filename)
            if ext:
                self._base.win.saveScreenshot(filename)
                return

        # Fall back to ShowBase helper (uses Panda3D naming conventions)
        self._base.screenshot(namePrefix=filename, defaultFilename=use_default)

    # Dispatch helpers
    def dispatch(self, fn: Callable[[], None]) -> None:
        self._dispatch_q.put(fn)

    # Internals
    def _interactive_loop(self) -> None:
        target_dt = 1.0 / float(max(1, self._fps))
        while self._run_flag.is_set():
            start = time.perf_counter()
            try:
                self._base.taskMgr.step()
            except Exception:
                break
            elapsed = time.perf_counter() - start
            sleep_for = target_dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _service_task(self, task: Any) -> int:
        while True:
            try:
                fn = self._dispatch_q.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
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
        if self._paused:
            dt = 0.0
        else:
            # Use Panda3D global clock (robust across versions)
            dt = ClockObject.getGlobalClock().getDt()
        try:
            view.update(dt)
        except Exception:
            pass
        return Task.cont

    def _observer_task_wrapper(self, observer: Observer, task: Any) -> int:
        if self._paused:
            dt = 0.0
        else:
            dt = ClockObject.getGlobalClock().getDt()
        try:
            observer.update(dt)
        except Exception:
            pass
        return Task.cont
