from dataclasses import dataclass
from typing import Any

import numpy as np
import taichi as ti

from rheidos.compute import ResourceRef, WiredProducer, out_field, Registry


@dataclass
class AdvectVorticesRK4IO:
    # Mesh
    V_pos: ResourceRef[ti.Field]  # (nV, vec3f)
    F_verts: ResourceRef[ti.Field]  # (nF, vec3i)
    F_adj: ResourceRef[
        ti.Field
    ]  # (nF, vec3i) neighbor across edge opposite each vertex; -1 = boundary
    F_normal: ResourceRef[ti.Field]  # (nF, vec3f) unit normal per face
    F_area: ResourceRef[ti.Field]  # (nF, f32) area per face

    # Per vertex velocity (use barycentric interpolation to get smooth vel at any point)
    V_velocity: ResourceRef[ti.Field]  # (nV, vec3f)

    # Vortices (state)
    n_vortices: ResourceRef[ti.Field]  # scalar i32
    face_ids: ResourceRef[ti.Field]  # (maxV, i32)
    bary: ResourceRef[ti.Field]  # (maxV, vec3f)

    # Time step
    dt: ResourceRef[Any]  # scalar f32

    # Outputs (new state)
    face_ids_out: ResourceRef[np.ndarray] = out_field()  # (nVortices,)
    bary_out: ResourceRef[np.ndarray] = out_field()  # (nVortices, 3)
    pos_out: ResourceRef[np.ndarray] = out_field()  # (nVortices, vec3f)
