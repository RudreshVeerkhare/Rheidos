import numpy as np


def p2_local_stiffness_from_cotan(cot1, cot2, cot3):
    return np.array(
        [
            [(cot2 + cot3) / 2, cot3 / 6, cot2 / 6, -2 * cot3 / 3, 0.0, -2 * cot2 / 3],
            [cot3 / 6, (cot1 + cot3) / 2, cot1 / 6, -2 * cot3 / 3, -2 * cot1 / 3, 0.0],
            [cot2 / 6, cot1 / 6, (cot1 + cot2) / 2, 0.0, -2 * cot1 / 3, -2 * cot2 / 3],
            [
                -2 * cot3 / 3,
                -2 * cot3 / 3,
                0.0,
                4 * (cot1 + cot2 + cot3) / 3,
                -4 * cot2 / 3,
                -4 * cot1 / 3,
            ],
            [
                0.0,
                -2 * cot1 / 3,
                -2 * cot1 / 3,
                -4 * cot2 / 3,
                4 * (cot1 + cot2 + cot3) / 3,
                -4 * cot3 / 3,
            ],
            [
                -2 * cot2 / 3,
                0.0,
                -2 * cot2 / 3,
                -4 * cot1 / 3,
                -4 * cot3 / 3,
                4 * (cot1 + cot2 + cot3) / 3,
            ],
        ],
        dtype=float,
    )


def cotan_triangle_weights(x1, x2, x3):
    cot_at = lambda o, a, b: np.dot((a - o), (b - o)) / np.linalg.norm(
        np.cross((a - o), (b - o))
    )
    return cot_at(x1, x2, x3), cot_at(x2, x1, x3), cot_at(x3, x2, x1)
