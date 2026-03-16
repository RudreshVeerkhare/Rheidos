from __future__ import annotations

import numpy as np

from pure_taichi.elements import P2Element


def test_p2_basis_partition_and_grad_shape() -> None:
    element = P2Element()
    rng = np.random.default_rng(123)

    for _ in range(32):
        a = rng.random()
        b = rng.random()
        if a + b > 1.0:
            a, b = 1.0 - a, 1.0 - b

        phi = element.eval_shape(float(a), float(b))
        dphi = element.eval_grad_ref(float(a), float(b))

        assert phi.shape == (6,)
        assert dphi.shape == (6, 2)
        assert np.isclose(phi.sum(), 1.0, atol=1e-12)
