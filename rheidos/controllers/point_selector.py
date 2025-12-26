from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from direct.showbase.MessengerGlobal import messenger
from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionTraverser,
    Geom,
    GeomNode,
    Vec3,
)

from ..abc.action import Action
from ..abc.controller import Controller
from ..views.point_selection import PointSelectionView


@dataclass(frozen=True)
class SelectedPoint:
    index: Optional[int]
    world: Tuple[float, float, float]
    local: Tuple[float, float, float]
    normal: Optional[Tuple[float, float, float]] = None
    snapped_to_vertex: bool = False
    node_name: Optional[str] = None


@dataclass
class _PickResult:
    surface_world: Vec3
    surface_local: Vec3
    surface_normal: Optional[Vec3]
    geom_index: Optional[int]
    into_node: Optional[object]
    vertex_index: Optional[int]
    vertex_local: Optional[Vec3]
    vertex_world: Optional[Vec3]


class _ScenePointSelectorBase(Controller):
    """
    Shared scene-level picking: casts a ray against the scene using a collide mask.
    Derived classes decide whether to snap to vertices or keep surface hits.
    """

    _name_counts: Dict[str, int] = {}

    @classmethod
    def _sanitize_name_part(cls, text: str) -> str:
        if not text:
            return ""
        out = []
        for ch in text:
            if ch.isalnum() or ch in ("-", "_"):
                out.append(ch)
            else:
                out.append("-")
        return "".join(out).strip("-")

    @classmethod
    def _allocate_name(cls, base: str) -> str:
        count = cls._name_counts.get(base, 0) + 1
        cls._name_counts[base] = count
        return base if count == 1 else f"{base}-{count}"

    @classmethod
    def _normalize_buttons(
        cls, select_button: Union[str, Sequence[str], None]
    ) -> list[str]:
        if select_button is None:
            return []
        if isinstance(select_button, (list, tuple, set)):
            return [str(btn) for btn in select_button if btn]
        return [str(select_button)]

    def __init__(
        self,
        name: Optional[str],
        engine,
        markers_view: Optional[PointSelectionView] = None,
        store_key: Optional[str] = None,
        pick_root: Optional[object] = None,
        pick_mask: BitMask32 = BitMask32.bit(4),
        select_button: Union[str, Sequence[str]] = "mouse1",
        clear_shortcut: str = "c",
        on_change: Optional[Callable[[List[SelectedPoint]], None]] = None,
        snap_to_vertex: bool = False,
        ui_order: int = -8,
    ) -> None:
        resolved_name = name
        if resolved_name is None:
            base = self.__class__.__name__
            if store_key:
                safe_key = self._sanitize_name_part(str(store_key))
                if safe_key:
                    base = f"{base}-{safe_key}"
            resolved_name = self._allocate_name(base)
        super().__init__(name=resolved_name)
        self.engine = engine
        self.markers_view = markers_view
        self.store_key = store_key
        self.pick_root = pick_root
        self.pick_mask = pick_mask
        self.select_button = select_button
        self._select_buttons = self._normalize_buttons(select_button)
        self.clear_shortcut = clear_shortcut
        self.on_change = on_change
        self.snap_to_vertex = bool(snap_to_vertex)
        self.ui_order = ui_order

        self._active = True
        self._hover: Optional[Vec3] = None
        self._selected: Dict[object, SelectedPoint] = {}

        self._picker_traverser: Optional[CollisionTraverser] = None
        self._picker_queue: Optional[CollisionHandlerQueue] = None
        self._picker_ray: Optional[CollisionRay] = None
        self._picker_np = None
        self._mouse_watcher = None
        self._cam_node = None
        self._task_name: Optional[str] = None
        self._accepted: list[str] = []
        self._root_np = None

        self._geom_cache: Dict[int, np.ndarray] = {}

    # ---- Controller interface --------------------------------------

    def actions(self) -> tuple[Action, ...]:
        return (
            Action(
                id=f"{self.name}-enabled",
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
                id=f"{self.name}-clear",
                label="Clear Selected Points",
                kind="button",
                group="Selection",
                order=1,
                shortcut=self.clear_shortcut,
                invoke=lambda session, _: self.clear_selection(),
            ),
        )

    # ---- Lifecycle -------------------------------------------------

    def attach(self, session) -> None:
        super().attach(session)
        self._mouse_watcher = getattr(session.base, "mouseWatcherNode", None)
        self._cam_node = getattr(session.base, "camNode", None)
        self._root_np = self.pick_root or getattr(session, "render", None)
        if self._root_np is None:
            return

        self._build_picker()

        if self._select_buttons:
            # Use messenger with a unique object (self) so other controllers can bind the same event.
            for button in self._select_buttons:
                if not button:
                    continue
                messenger.accept(button, self, self._on_click)
                self._accepted.append(button)

        self._task_name = f"scene-point-hover-{id(self)}"
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
                    messenger.ignore(evt, self)
                except Exception:
                    pass
        self._accepted.clear()

        try:
            if self._picker_traverser is not None and self._picker_np is not None:
                self._picker_traverser.removeCollider(self._picker_np)
        except Exception:
            pass

        if self._picker_np is not None:
            try:
                self._picker_np.removeNode()
            except Exception:
                pass

        self._picker_traverser = None
        self._picker_queue = None
        self._picker_ray = None
        self._picker_np = None
        self._mouse_watcher = None
        self._cam_node = None
        self._root_np = None
        self._geom_cache.clear()

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

    def _ui_wants_mouse(self) -> bool:
        try:
            from imgui_bundle import imgui

            io = imgui.get_io()
            return bool(io.want_capture_mouse)
        except Exception:
            return False

    def _build_picker(self) -> None:
        self._picker_traverser = CollisionTraverser(self.name + "-picker")
        self._picker_queue = CollisionHandlerQueue()
        self._picker_ray = CollisionRay()
        picker_node = CollisionNode(self.name + "-ray")
        picker_node.addSolid(self._picker_ray)
        picker_node.setFromCollideMask(self.pick_mask)
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

    def _pick_under_mouse(self) -> Optional[_PickResult]:
        if (
            self._mouse_watcher is None
            or self._cam_node is None
            or self._picker_traverser is None
            or self._picker_queue is None
            or self._picker_ray is None
            or self._picker_np is None
            or self._root_np is None
        ):
            return None

        if not self._mouse_watcher.hasMouse():
            return None
        mouse_pos = self._mouse_watcher.getMouse()
        self._picker_ray.setFromLens(self._cam_node, mouse_pos)

        self._picker_queue.clearEntries()
        self._picker_traverser.traverse(self._root_np)
        if self._picker_queue.getNumEntries() == 0:
            return None

        self._picker_queue.sortEntries()
        entry = self._picker_queue.getEntry(0)
        into_np = entry.getIntoNodePath()
        try:
            surface_local = entry.getSurfacePoint(into_np)
            surface_world = entry.getSurfacePoint(self._root_np)
            surface_normal = entry.getSurfaceNormal(into_np)
            try:
                geom_index = entry.getGeomIndex()  # type: ignore[attr-defined]
            except Exception:
                geom_index = None
        except Exception:
            return None

        vertex_idx = None
        v_local = None
        v_world = None
        if self.snap_to_vertex:
            vertex_idx, v_local, v_world = self._nearest_vertex(
                into_np, entry, surface_local, geom_index
            )

        return _PickResult(
            surface_world=surface_world,
            surface_local=surface_local,
            surface_normal=surface_normal,
            geom_index=geom_index,
            into_node=into_np,
            vertex_index=vertex_idx,
            vertex_local=v_local,
            vertex_world=v_world,
        )

    def _nearest_vertex(
        self, into_np, entry, surface_local: Vec3, geom_index: Optional[int] = None
    ) -> tuple[Optional[int], Optional[Vec3], Optional[Vec3]]:
        try:
            node = into_np.node()
        except Exception:
            return (None, None, None)
        if not isinstance(node, GeomNode):
            return (None, None, None)

        if geom_index is None:
            try:
                geom_index = entry.getGeomIndex()  # type: ignore[attr-defined]
            except Exception:
                geom_index = 0

        try:
            geom = node.getGeom(geom_index)
        except Exception:
            geom = node.getGeom(0) if node.getNumGeoms() > 0 else None
        if geom is None or not isinstance(geom, Geom):
            return (None, None, None)

        key = id(geom)
        if key not in self._geom_cache:
            try:
                vdata = geom.getVertexData()
                arr = vdata.getArray(0)
                handle = arr.getHandle()
                raw = memoryview(handle.getData())
                self._geom_cache[key] = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3).copy()
            except Exception:
                return (None, None, None)

        positions = self._geom_cache.get(key)
        if positions is None or positions.size == 0:
            return (None, None, None)

        target = np.array([surface_local.x, surface_local.y, surface_local.z], dtype=np.float32)
        diff = positions - target
        dists_sq = np.einsum("ij,ij->i", diff, diff)
        idx = int(np.argmin(dists_sq))

        try:
            v_local = Vec3(float(positions[idx][0]), float(positions[idx][1]), float(positions[idx][2]))
            xf = into_np.getTransform(self._root_np)
            v_world = xf.xformPoint(v_local)
        except Exception:
            v_local = None
            v_world = None
        return (idx, v_local, v_world)

    def _resolve_hit_point(self, hit: _PickResult) -> Optional[Vec3]:
        if self.snap_to_vertex and hit.vertex_world is not None:
            return hit.vertex_world
        return hit.surface_world

    def _toggle_selection(self, hit: _PickResult) -> None:
        point_world = self._resolve_hit_point(hit)
        if point_world is None:
            return
        point_local = hit.vertex_local if (self.snap_to_vertex and hit.vertex_local is not None) else hit.surface_local
        idx = hit.vertex_index if self.snap_to_vertex else None

        # Build a stable key per hit so toggling does not collide across duplicated geoms.
        def _node_key(np):
            if np is None:
                return None
            for attr in ("get_key", "getKey"):
                try:
                    return getattr(np, attr)()
                except Exception:
                    continue
            try:
                return id(np.node())
            except Exception:
                return id(np)

        node_key = _node_key(hit.into_node)
        if idx is not None:
            if node_key is not None and hit.geom_index is not None:
                key = (node_key, hit.geom_index, idx)
            elif node_key is not None:
                key = (node_key, idx)
            elif hit.geom_index is not None:
                key = (hit.geom_index, idx)
            else:
                key = idx
        else:
            coord_key = (
                round(point_world.x, 5),
                round(point_world.y, 5),
                round(point_world.z, 5),
            )
            key = (node_key, coord_key) if node_key is not None else coord_key

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
                snapped_to_vertex=self.snap_to_vertex and hit.vertex_world is not None,
                node_name=hit.into_node.getName() if hit.into_node is not None else None,
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
                        "node_name": p.node_name,
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


class SceneSurfacePointSelector(_ScenePointSelectorBase):
    """Global scene picker that returns exact surface hit points (no vertex snap)."""

    def __init__(
        self,
        engine,
        markers_view: Optional[PointSelectionView] = None,
        store_key: Optional[str] = "surface_points",
        pick_root: Optional[object] = None,
        pick_mask: BitMask32 = BitMask32.bit(4),
        select_button: str = "mouse1",
        clear_shortcut: str = "c",
        on_change: Optional[Callable[[List[SelectedPoint]], None]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name,
            engine=engine,
            markers_view=markers_view,
            store_key=store_key,
            pick_root=pick_root,
            pick_mask=pick_mask,
            select_button=select_button,
            clear_shortcut=clear_shortcut,
            on_change=on_change,
            snap_to_vertex=False,
            ui_order=-9,
        )


class SceneVertexPointSelector(_ScenePointSelectorBase):
    """Global scene picker that snaps to the nearest vertex of the hit geometry."""

    def __init__(
        self,
        engine,
        markers_view: Optional[PointSelectionView] = None,
        store_key: Optional[str] = "vertex_points",
        pick_root: Optional[object] = None,
        pick_mask: BitMask32 = BitMask32.bit(4),
        select_button: str = "mouse1",
        clear_shortcut: str = "c",
        on_change: Optional[Callable[[List[SelectedPoint]], None]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name,
            engine=engine,
            markers_view=markers_view,
            store_key=store_key,
            pick_root=pick_root,
            pick_mask=pick_mask,
            select_button=select_button,
            clear_shortcut=clear_shortcut,
            on_change=on_change,
            snap_to_vertex=True,
            ui_order=-8,
        )
