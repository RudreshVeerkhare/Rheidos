from __future__ import annotations

import numpy as np

from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import ModuleBase, World

from .probe_utils import probe_arrays


class Whitney1FormInterpolator(ModuleBase):
    NAME = "Whitney1FormInterpolator"

    def __init__(
        self,
        world: World,
        *,
        mesh: SurfaceMeshModule,
        scope: str = "",
    ) -> None:
        super().__init__(world, scope=scope)
        self.mesh = mesh

    def interpolate(self, one_form, probes) -> np.ndarray:
        """Interpolate a primal DEC 1-form as a tangent vector field.

        Args:
            one_form: Edge cochain in the canonical orientation of mesh.E_verts.
                Shape: (nE,).
            probes: Either (faceids, bary) arrays or [(faceid, bary), ...].

        Returns:
            Interpolated sharp/vector proxy. Shape: (nProbes, 3).
        """
        edge_values = np.asarray(one_form, dtype=np.float64)
        e_verts = self.mesh.E_verts.get()
        if edge_values.ndim != 1:
            raise ValueError(
                "Whitney1FormInterpolator.interpolate expects a 1-form with "
                f"shape (nE,), got {edge_values.shape}"
            )
        if edge_values.shape[0] != e_verts.shape[0]:
            raise ValueError(
                "Whitney1FormInterpolator.interpolate expects a 1-form with "
                f"length nE={e_verts.shape[0]}, got {edge_values.shape[0]}"
            )

        faceids, bary = probe_arrays(probes)
        if faceids.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        face_edges = self.mesh.F_edges.get()[faceids]
        face_edge_sign = self.mesh.F_edge_sign.get()[faceids]
        local_edge_values = edge_values[face_edges] * face_edge_sign
        grad = self.mesh.grad_bary.get()[faceids]

        l0, l1, l2 = bary.T
        g0, g1, g2 = grad[:, 0, :], grad[:, 1, :], grad[:, 2, :]
        a01, a12, a20 = local_edge_values.T

        return (
            a01[:, None] * (l0[:, None] * g1 - l1[:, None] * g0)
            + a12[:, None] * (l1[:, None] * g2 - l2[:, None] * g1)
            + a20[:, None] * (l2[:, None] * g0 - l0[:, None] * g2)
        )
