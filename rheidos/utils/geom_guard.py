from __future__ import annotations

import math
import sys
from typing import Optional

from ..abc.observer import Observer

try:
    from panda3d.core import GeomNode, GeomVertexReader, NodePath
except Exception:  # pragma: no cover
    GeomNode = None  # type: ignore
    GeomVertexReader = None  # type: ignore
    NodePath = None  # type: ignore


def _vec3_finite(v) -> bool:
    return math.isfinite(float(v.x)) and math.isfinite(float(v.y)) and math.isfinite(float(v.z))


class GeomNanGuard(Observer):
    """
    Debug guard that scans geometry for NaN/Inf positions before render.

    If invalid vertices are found, it can optionally hide the offending node to
    avoid a Panda3D bounding volume assertion.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        sort: int = 900,
        check_every: int = 1,
        disable_on_error: bool = True,
        max_reports: int = 5,
    ) -> None:
        super().__init__(name=name or "GeomNanGuard", sort=sort)
        self._roots: list[NodePath] = []
        self._frame = 0
        self._check_every = max(1, int(check_every))
        self._disable_on_error = bool(disable_on_error)
        self._max_reports = max(1, int(max_reports))

    def setup(self, session) -> None:
        super().setup(session)
        if NodePath is None:
            return
        roots = []
        for attr in ("render",):
            root = getattr(session, attr, None)
            if root is not None:
                roots.append(root)
        base = getattr(session, "base", None)
        for attr in ("render2d", "aspect2d"):
            root = getattr(base, attr, None) if base is not None else None
            if root is not None:
                roots.append(root)
        self._roots = roots

    def update(self, dt: float) -> None:
        if not self._roots or GeomNode is None or GeomVertexReader is None:
            return
        self._frame += 1
        if self._frame % self._check_every != 0:
            return
        self._scan()

    def _scan(self) -> None:
        matches = []
        for root in self._roots:
            try:
                matches.extend(root.findAllMatches("**/+GeomNode"))
            except Exception:
                continue
        reports = 0
        for np in matches:
            try:
                node = np.node()
            except Exception:
                continue
            if not isinstance(node, GeomNode):
                continue
            if not self._check_node(np, node):
                reports += 1
                if reports >= self._max_reports:
                    break

    def _check_node(self, np: NodePath, node: GeomNode) -> bool:
        for gi in range(node.getNumGeoms()):
            try:
                geom = node.getGeom(gi)
                vdata = geom.getVertexData()
            except Exception:
                continue
            if vdata is None or vdata.getNumRows() == 0:
                continue
            try:
                reader = GeomVertexReader(vdata, "vertex")
            except Exception:
                continue
            for _ in range(vdata.getNumRows()):
                try:
                    v = reader.getData3f()
                except Exception:
                    break
                if not _vec3_finite(v):
                    self._report(np, gi, vdata.getNumRows(), v)
                    if self._disable_on_error:
                        try:
                            np.hide()
                        except Exception:
                            pass
                    return False
        return True

    def _report(self, np: NodePath, geom_idx: int, rows: int, v) -> None:
        name = ""
        try:
            name = np.getName()
        except Exception:
            name = "<unnamed>"
        msg = (
            f"[GeomNanGuard] Non-finite vertex detected: node='{name}', geom={geom_idx}, "
            f"rows={rows}, v=({float(v.x):.5g},{float(v.y):.5g},{float(v.z):.5g})"
        )
        print(msg, file=sys.stderr)
