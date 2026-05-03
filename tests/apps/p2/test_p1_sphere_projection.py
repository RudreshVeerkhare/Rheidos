from __future__ import annotations

from rheidos.apps.p2.p1_sphere_app.app import (
    App,
    DEFAULT_REFERENCE_SURFACE_PROJECT_SOP,
    PROJECT_TO_REFERENCE_SURFACE_VEX,
    ProjectPointsToReferenceSurface,
)
from rheidos.compute import World


def test_p1_sphere_app_wires_reference_surface_projector() -> None:
    world = World()

    app = world.require(App)

    assert isinstance(app.reference_surface_projector, ProjectPointsToReferenceSurface)
    assert app.reference_surface_projector.node_path == DEFAULT_REFERENCE_SURFACE_PROJECT_SOP
    assert app.reference_surface_projector.prefix.startswith(app.prefix)


def test_reference_surface_projection_vex_uses_second_input_contract() -> None:
    assert "xyzdist(1, @P, prim, uvw)" in PROJECT_TO_REFERENCE_SURFACE_VEX
    assert 'primuv(1, "P", prim, uvw)' in PROJECT_TO_REFERENCE_SURFACE_VEX
    assert "priminteriorweights(1, prim, uvw, verts, weights)" in PROJECT_TO_REFERENCE_SURFACE_VEX
    assert "i@faceid = prim" in PROJECT_TO_REFERENCE_SURFACE_VEX
    assert "v@bary" in PROJECT_TO_REFERENCE_SURFACE_VEX
