from __future__ import annotations

import numpy as np

import rheidos.apps.p2.p1_sphere_app.app as app_mod
from rheidos.apps.p2.p1_sphere_app.app import (
    App,
    DEFAULT_REFERENCE_SURFACE_PROJECT_SOP,
    RAY_HIT_PRIM_ATTR,
    RAY_HIT_UVW_ATTR,
    ProjectPointsToReferenceSurface,
    triangle_bary_from_hituvw,
)
from rheidos.compute import World


def test_p1_sphere_app_wires_reference_surface_projector() -> None:
    world = World()

    app = world.require(App)

    assert isinstance(app.reference_surface_projector, ProjectPointsToReferenceSurface)
    assert app.reference_surface_projector.node_path == DEFAULT_REFERENCE_SURFACE_PROJECT_SOP
    assert app.reference_surface_projector.prefix.startswith(app.prefix)
    assert app.reference_surface_projector.sop_input_providers[1].cache == "cook"


def test_triangle_bary_from_hituvw_uses_ray_triangle_contract() -> None:
    hituvw = np.array(
        [[0.25, 0.25, 0.0], [0.75, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )

    bary = triangle_bary_from_hituvw(hituvw)

    np.testing.assert_allclose(
        bary,
        np.array(
            [[0.5, 0.25, 0.25], [0.25, 0.75, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(bary.sum(axis=1), 1.0)


def test_ray_projection_postprocess_reads_minimal_ray_attributes(monkeypatch) -> None:
    projected = np.array([[0.25, 0.25, 0.0]], dtype=np.float64)
    faceids = np.array([0], dtype=np.int32)
    hituvw = np.array([[0.25, 0.25, 0.0]], dtype=np.float64)

    def _read(_geo, name, *, dtype, components=None):
        del dtype, components
        if name == "P":
            return projected
        if name == RAY_HIT_PRIM_ATTR:
            return faceids
        if name == RAY_HIT_UVW_ATTR:
            return hituvw
        raise KeyError(name)

    monkeypatch.setattr(app_mod, "point_attrib_to_numpy", _read)
    module = World().require(
        ProjectPointsToReferenceSurface,
        node_path=DEFAULT_REFERENCE_SURFACE_PROJECT_SOP,
    )

    result = module.postprocess(object(), {})

    np.testing.assert_allclose(result.pos, projected)
    np.testing.assert_array_equal(result.faceids, faceids)
    np.testing.assert_allclose(result.bary, np.array([[0.5, 0.25, 0.25]]))
