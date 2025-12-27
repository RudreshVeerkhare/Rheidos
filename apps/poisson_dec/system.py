from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import taichi as ti

from rheidos.compute import World
from apps.poisson_dec.compute import DECModule, MeshModule, PoissonSolverModule

Charge = Tuple[int, float]


@dataclass
class ChargeBuffer:
    size: int
    values: np.ndarray
    mask: np.ndarray

    @classmethod
    def allocate(cls, size: int) -> "ChargeBuffer":
        values = np.zeros((size,), dtype=np.float32)
        mask = np.zeros((size,), dtype=np.int32)
        return cls(size=size, values=values, mask=mask)

    def update(self, charges: Sequence[Charge]) -> None:
        self.values.fill(0.0)
        self.mask.fill(0)
        for idx, value in charges:
            if idx < 0 or idx >= self.size:
                raise IndexError(f"Charge index {idx} out of range 0..{self.size - 1}")
            self.values[idx] = value
            self.mask[idx] = 1


class PoissonSystem:
    def __init__(self) -> None:
        self.world = World()
        self.mesh = self.world.require(MeshModule)
        self.dec = self.world.require(DECModule)
        self.poisson = self.world.require(PoissonSolverModule)

        self._charge_buffer: Optional[ChargeBuffer] = None
        self._n_vertices: Optional[int] = None

    def set_mesh_from_numpy(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        n_vertices = int(vertices.shape[0])
        n_faces = int(faces.shape[0])

        v_field = ti.Vector.field(3, dtype=ti.f32, shape=(n_vertices,))
        f_field = ti.Vector.field(3, dtype=ti.i32, shape=(n_faces,))
        v_field.from_numpy(vertices.astype(np.float32, copy=False))
        f_field.from_numpy(faces.astype(np.int32, copy=False))

        self.set_mesh_fields(v_field, f_field)

    def set_mesh_fields(self, vertices: Any, faces: Any) -> None:
        self.mesh.V_pos.set(vertices)
        self.mesh.F_verts.set(faces)

        n_vertices = int(vertices.shape[0])
        self._ensure_constraints(n_vertices)

    def apply_charges(self, charges: Sequence[Charge]) -> None:
        if self._charge_buffer is None or self._n_vertices is None:
            raise RuntimeError("Mesh must be set before applying charges.")

        self._charge_buffer.update(charges)

        mask = self.poisson.constraint_mask.get(ensure=False)
        value = self.poisson.constraint_value.get(ensure=False)
        if mask is None or value is None:
            raise RuntimeError("Constraint buffers are not initialized.")

        value.from_numpy(self._charge_buffer.values)
        mask.from_numpy(self._charge_buffer.mask)
        self.poisson.constraint_value.commit()
        self.poisson.constraint_mask.commit()

    def solve(self) -> Any:
        return self.poisson.u.get()

    def _ensure_constraints(self, n_vertices: int) -> None:
        mask = self.poisson.constraint_mask.get(ensure=False)
        value = self.poisson.constraint_value.get(ensure=False)

        needs_alloc = (
            mask is None
            or value is None
            or mask.shape != (n_vertices,)
            or value.shape != (n_vertices,)
        )
        if needs_alloc:
            mask = ti.field(dtype=ti.i32, shape=(n_vertices,))
            value = ti.field(dtype=ti.f32, shape=(n_vertices,))
            mask.fill(0)
            value.fill(0)
            self.poisson.constraint_mask.set(mask)
            self.poisson.constraint_value.set(value)

        if self._charge_buffer is None or self._charge_buffer.size != n_vertices:
            self._charge_buffer = ChargeBuffer.allocate(n_vertices)

        self._n_vertices = n_vertices
