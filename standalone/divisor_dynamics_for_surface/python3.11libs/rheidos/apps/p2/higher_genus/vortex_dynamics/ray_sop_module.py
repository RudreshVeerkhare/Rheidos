from rheidos.houdini.sop import (
    CallGeo,
    CtxInputGeo,
    SopFunctionModule,
    point_attrib_to_numpy,
)

import numpy as np
from dataclasses import dataclass

RAY_HIT_PRIM_ATTR = "hitprim"
RAY_HIT_UVW_ATTR = "hitprimuv"


@dataclass(frozen=True)
class RaySopOutput:
    pos: np.ndarray
    faceids: np.ndarray
    bary: np.ndarray


class RaySopModule(SopFunctionModule):
    NAME = "RaySopModule"
    SOP_INPUTS = {
        0: CallGeo("query"),
        1: CtxInputGeo(1, cache="cook"),
    }

    def project_points(self, points: np.ndarray) -> RaySopOutput:
        query_geo = self.points_to_geo(np.asarray(points, dtype=np.float64))
        return self.run(query=query_geo)

    @staticmethod
    def triangle_bary_from_hituvw(
        hituvw: np.ndarray,
    ) -> np.ndarray:
        hituvw = np.asarray(hituvw, dtype=np.float64)
        if hituvw.ndim != 2 or hituvw.shape[1] != 3:
            raise ValueError(f"hituvw must have shape (N, 3), got {hituvw.shape}")
        bary = np.empty((hituvw.shape[0], 3), dtype=np.float64)
        bary[:, 0] = 1.0 - hituvw[:, 0] - hituvw[:, 1]
        bary[:, 1] = hituvw[:, 0]
        bary[:, 2] = hituvw[:, 1]
        return bary

    def postprocess(self, out_geo, meta) -> RaySopOutput:
        del meta
        pos = point_attrib_to_numpy(out_geo, "P", dtype=np.float64, components=3)
        faceids = point_attrib_to_numpy(out_geo, RAY_HIT_PRIM_ATTR, dtype=np.int32)
        hituvw = point_attrib_to_numpy(
            out_geo,
            RAY_HIT_UVW_ATTR,
            dtype=np.float64,
            components=3,
        )
        return RaySopOutput(
            pos,
            faceids,
            self.triangle_bary_from_hituvw(hituvw),
        )
