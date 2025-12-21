from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionTraverser,
    Vec3,
)

from ..abc.action import Action
from ..abc.controller import Controller
from ..resources.mesh import Mesh
from ..views.point_selection import PointSelectionView


@dataclass(frozen=True)
class SelectedPoint:
    index: Optional[int]
    world: Tuple[float, float, float]
    local: Tuple[float, float, float]
    normal: Optional[Tuple[float, float, float]] = None
    snapped_to_vertex: bool = True


@dataclass
class _PickResult:
    surface_local: Vec3
    surface_world: Vec3
    surface_normal: Optional[Vec3]
    vertex_index: Optional[int]
    vertex_local: Optional[Vec3]
    vertex_world: Optional[Vec3]


class PointSelectorController(Controller):
    """
    Click-to-select points on a mesh surface.

    - Left click toggles the hit point (either the surface point or nearest vertex).
    - Selection is stored (optionally in Engine.store) and mirrored to a PointSelectionView.
    - Hover preview shown when a marker view is provided.
    """

    def __init__(
        self,
        engine,
        mesh: Mesh,
        target_view: Optional[str] = None,
        markers_view: Optional[PointSelectionView] = None,
        store_key: Optional[str] = "selected_points",
        name: Optional[str] = None,
        select_button: str = "mouse1",
        clear_shortcut: Optional[str] = "c",
        snap_to_vertex: bool = True,
        on_change: Optional[Callable[[List[SelectedPoint]], None]] = None,
    ) -> None:
        super().__init__(name=name or "PointSelectorController")
        self.engine = engine
        self.mesh = mesh
        self.target_view = target_view
        self.markers_view = markers_view
        self.store_key = store_key
        self.select_button = select_button
        self.clear_shortcut = clear_shortcut
        self.snap_to_vertex = bool(snap_to_vertex)
        self.on_change = on_change
        self.ui_order = -8

        self._positions: Optional[np.ndarray] = None
        self._selected: Dict[object, SelectedPoint] = {}
        self._hover: Optional[Vec3] = None
        self._active = True

        self._group = None
        self._geom_np = None
        self._picker_traverser: Optional[CollisionTraverser] = None
        self._picker_queue: Optional[CollisionHandlerQueue] = None
        self._picker_ray: Optional[CollisionRay] = None
        self._picker_np = None
        self._mouse_watcher = None
        self._cam_node = None
        self._task_name: Optional[str] = None
        self._accepted: list[str] = []

        self._into_mask = BitMask32.bit(4)
        self._from_mask = BitMask32.bit(4)

    # ---- Controller interface --------------------------------------

    def actions(self) -> tuple[Action, ...]:
        return (
            Action(
                id="point-select-enabled",
                label="Point Select Enabled",
                kind="toggle",
                group="Selection",
                order=0,
                get_value=lambda session: self._active,
                set_value=lambda session, v: self._set_active(bool(v)),
                invoke=lambda session, v=None: self._set_active(
                    not self._active if v is None else bool(v)
                ),
            ),
            Action(
                id="point-select-clear",
                label="Clear Selected Points",
                kind="button",
                group="Selection",
                order=1,
                shortcut=self.clear_shortcut,
                invoke=lambda session, _: self.clear_selection(),
            ),
            Action(
                id="point-select-snap",
                label="Snap To Nearest Vertex",
                kind="toggle",
                group="Selection",
                order=2,
                get_value=lambda session: self.snap_to_vertex,
                set_value=lambda session, v: self._set_snap(bool(v)),
                invoke=lambda session, v=None: self._set_snap(
                    not self.snap_to_vertex if v is None else bool(v)
                ),
            ),
        )

    # ---- Lifecycle -------------------------------------------------

    def attach(self, session) -> None:
        super().attach(session)
        self._mouse_watcher = getattr(session.base, "mouseWatcherNode", None)
        self._cam_node = getattr(session.base, "camNode", None)

        self._positions = self._extract_positions(self.mesh)
        self._build_picker()

        if self.select_button:
            session.accept(self.select_button, self._on_click)
            self._accepted.append(self.select_button)

        self._task_name = f"point-selector-hover-{id(self)}"
        session.task_mgr.add(self._hover_task, self._task_name, sort=-45)

    def detach(self) -> None:
        if self._task_name and self._session is not None:
            try:
                self._session.task_mgr.remove(self._task_name)
            except Exception:
                pass
        self._task_name = None

        if self._session is not None:
            for evt in self._accepted:
                try:
                    self._session.ignore(evt)
                except Exception:
                    pass
        self._accepted.clear()

        try:
            if self._picker_traverser is not None and self._picker_np is not None:
                self._picker_traverser.removeCollider(self._picker_np)
        except Exception:
            pass

        for np in (self._picker_np, self._geom_np, self._group):
            try:
                if np is not None:
                    np.removeNode()
            except Exception:
                pass

        self._group = None
        self._geom_np = None
        self._picker_traverser = None
        self._picker_queue = None
        self._picker_ray = None
        self._picker_np = None
        self._mouse_watcher = None
        self._cam_node = None

    # ---- Selection operations --------------------------------------

    def clear_selection(self) -> None:
        self._selected.clear()
        self._push_updates()

    def selected_points(self) -> list[SelectedPoint]:
        return list(self._selected.values())

    # ---- Internals --------------------------------------------------

    def _set_active(self, active: bool) -> None:
        self._active = bool(active)
        if not self._active:
            self._hover = None
            if self.markers_view is not None:
                try:
                    self.markers_view.set_hover(None)
                except Exception:
                    pass

    def _set_snap(self, snap: bool) -> None:
        self.snap_to_vertex = bool(snap)
        if not self._active:
            return
        hit = self._pick_under_mouse()
        if hit is None:
            if self._hover is not None:
                self._set_hover(None)
            return
        self._set_hover(self._resolve_hit_point(hit))

    def _ui_wants_mouse(self) -> bool:
        try:
            from imgui_bundle import imgui

            io = imgui.get_io()
            return bool(io.want_capture_mouse)
        except Exception:
            return False

    def _extract_positions(self, mesh: Mesh) -> Optional[np.ndarray]:
        try:
            handle = mesh.vdata.getArray(0).getHandle()
            raw = memoryview(handle.getData())
            return np.frombuffer(raw, dtype=np.float32).reshape(-1, 3).copy()
        except Exception:
            return None

    def _find_parent(self):
        if self.engine is not None and self.target_view:
            view = getattr(self.engine, "_views", {}).get(self.target_view)
            np = getattr(view, "_node", None)
            if np is not None:
                return np
        return getattr(self._session, "render", None)

    def _build_picker(self) -> None:
        parent = self._find_parent()
        if parent is None:
            return

        self._group = parent.attachNewNode(self.name + "-picker-group")
        self._geom_np = self.mesh.node_path.copyTo(self._group)
        self._geom_np.hide()
        self._geom_np.setCollideMask(self._into_mask)

        self._picker_traverser = CollisionTraverser(self.name + "-picker")
        self._picker_queue = CollisionHandlerQueue()
        self._picker_ray = CollisionRay()
        picker_node = CollisionNode(self.name + "-ray")
        picker_node.addSolid(self._picker_ray)
        picker_node.setFromCollideMask(self._from_mask)
        picker_node.setIntoCollideMask(BitMask32.allOff())
        self._picker_np = self._session.base.camera.attachNewNode(picker_node)
        self._picker_traverser.addCollider(self._picker_np, self._picker_queue)

    def _hover_task(self, task) -> int:
        if not self._active or self._ui_wants_mouse():
            self._set_hover(None)
            return task.cont

        hit = self._pick_under_mouse()
        if hit is None:
            self._set_hover(None)
        else:
            self._set_hover(self._resolve_hit_point(hit))
        return task.cont

    def _set_hover(self, point: Optional[Vec3]) -> None:
        self._hover = point
        if self.markers_view is None:
            return
        try:
            self.markers_view.set_hover(point)
        except Exception:
            pass

    def _on_click(self) -> None:
        if not self._active or self._ui_wants_mouse():
            return
        hit = self._pick_under_mouse()
        if hit is None:
            return
        self._toggle_selection(hit)

    def _resolve_hit_point(self, hit: _PickResult) -> Optional[Vec3]:
        if hit is None:
            return None
        if self.snap_to_vertex and hit.vertex_world is not None:
            return hit.vertex_world
        return hit.surface_world

    def _pick_under_mouse(self) -> Optional[_PickResult]:
        if (
            self._mouse_watcher is None
            or self._cam_node is None
            or self._picker_traverser is None
            or self._picker_queue is None
            or self._picker_ray is None
            or self._geom_np is None
            or self._group is None
        ):
            return None

        if not self._mouse_watcher.hasMouse():
            return None
        mouse_pos = self._mouse_watcher.getMouse()
        self._picker_ray.setFromLens(self._cam_node, mouse_pos)

        self._picker_queue.clearEntries()
        self._picker_traverser.traverse(self._group)
        if self._picker_queue.getNumEntries() == 0:
            return None

        self._picker_queue.sortEntries()
        entry = self._picker_queue.getEntry(0)
        try:
            local_hit = entry.getSurfacePoint(self._geom_np)
            world_hit = entry.getSurfacePoint(self._session.render)
            normal = entry.getSurfaceNormal(self._geom_np)
        except Exception:
            return None

        idx = self._nearest_vertex_index(local_hit)
        v_local = None
        v_world = None
        if idx is not None and self._positions is not None:
            try:
                v_local = Vec3(
                    float(self._positions[idx][0]),
                    float(self._positions[idx][1]),
                    float(self._positions[idx][2]),
                )
                mat = self._geom_np.getMat(self._session.render)
                v_world = mat.xformPoint(v_local)
            except Exception:
                v_local = None
                v_world = None

        return _PickResult(
            surface_local=local_hit,
            surface_world=world_hit,
            surface_normal=normal,
            vertex_index=idx,
            vertex_local=v_local,
            vertex_world=v_world,
        )

    def _nearest_vertex_index(self, point: Vec3) -> Optional[int]:
        if self._positions is None or self._positions.size == 0:
            return None
        target = np.array([point.x, point.y, point.z], dtype=np.float32)
        diff = self._positions - target
        dists_sq = np.einsum("ij,ij->i", diff, diff)
        idx = int(np.argmin(dists_sq))
        return idx

    def _toggle_selection(self, hit: _PickResult) -> None:
        use_vertex = self.snap_to_vertex and hit.vertex_world is not None

        point_world = hit.vertex_world if use_vertex else hit.surface_world
        point_local = (
            hit.vertex_local if use_vertex and hit.vertex_local is not None else hit.surface_local
        )
        idx = hit.vertex_index if use_vertex else None

        key = idx if idx is not None else (
            round(point_world.x, 5),
            round(point_world.y, 5),
            round(point_world.z, 5),
        )

        if key in self._selected:
            self._selected.pop(key, None)
        else:
            normal = None
            if hit.surface_normal is not None:
                normal = (
                    float(hit.surface_normal.x),
                    float(hit.surface_normal.y),
                    float(hit.surface_normal.z),
                )
            pt = SelectedPoint(
                index=idx,
                world=(float(point_world.x), float(point_world.y), float(point_world.z)),
                local=(float(point_local.x), float(point_local.y), float(point_local.z)),
                normal=normal,
                snapped_to_vertex=use_vertex,
            )
            self._selected[key] = pt

        self._push_updates()

    def _push_updates(self) -> None:
        pts = self.selected_points()
        if self.markers_view is not None:
            try:
                self.markers_view.set_selected([Vec3(*p.world) for p in pts])
            except Exception:
                pass

        if self.store_key and getattr(self.engine, "store", None) is not None:
            try:
                payload = [
                    {
                        "index": p.index,
                        "world": p.world,
                        "local": p.local,
                        "normal": p.normal,
                        "snapped_to_vertex": p.snapped_to_vertex,
                    }
                    for p in pts
                ]
                self.engine.store.set(self.store_key, payload)
            except Exception:
                pass

        if self.on_change is not None:
            try:
                self.on_change(pts)
            except Exception:
                pass
