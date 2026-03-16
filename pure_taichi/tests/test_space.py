from __future__ import annotations

import numpy as np

from pure_taichi.space import build_p2_space_data


def test_shared_edge_midpoint_dof_is_global() -> None:
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    space = build_p2_space_data(4, faces)

    # Shared edge is (1,2): local edge index 1 in face0 and index 0 in face1.
    dof_f0 = int(space.face_to_dofs[0, 4])
    dof_f1 = int(space.face_to_dofs[1, 3])

    assert space.ndof > 0
    assert dof_f0 == dof_f1
