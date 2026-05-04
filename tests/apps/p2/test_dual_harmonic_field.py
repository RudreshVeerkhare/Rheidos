from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rheidos.apps.p2.higher_genus.vortex_dynamics.app import AbelJacobiModule, App
from rheidos.apps.p2.modules.higher_genus.dual_harmonic_field import (
    DualHarmonicFieldModule,
)
from rheidos.apps.p2.modules.higher_genus.harmonic_basis import HarmonicBasis
from rheidos.apps.p2.modules.higher_genus.tree_cotree import TreeCotreeModule
from rheidos.apps.p2.modules.p1_space.dec import DEC
from rheidos.apps.p2.modules.point_vortex.point_vortex_module import PointVortexModule
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import World

TORUS_OBJ = (
    Path(__file__).resolve().parents[3]
    / "rheidos/apps/p2/higher_genus/vortex_dynamics/torus.obj"
)


def _load_tri_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    for line in path.read_text().splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            face = [int(token.split("/")[0]) - 1 for token in fields[1:]]
            if len(face) != 3:
                raise ValueError(f"{path}: expected triangular faces, got {line!r}")
            faces.append(face)

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )


def _tetrahedron_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [0, 2, 3],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _square_disk_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return vertices, faces


class _FakeDualHarmonicField:
    def __init__(self, xi_face: np.ndarray) -> None:
        self.xi_face = np.asarray(xi_face, dtype=np.float64)

    def interpolate(self, probes, field: str = "xi") -> np.ndarray:
        assert field == "xi"
        faceids, _bary = probes
        return self.xi_face[:, faceids, :]


def _build_dual_module(vertices: np.ndarray, faces: np.ndarray) -> SimpleNamespace:
    world = World()
    mesh = world.require(SurfaceMeshModule)
    dec = world.require(DEC, mesh=mesh)
    tree_cotree = world.require(TreeCotreeModule, mesh=mesh)
    point_vortex = world.require(PointVortexModule)
    harmonic_basis = world.require(
        HarmonicBasis,
        dec=dec,
        tree_cotree=tree_cotree,
    )
    dual = world.require(
        DualHarmonicFieldModule,
        mesh=mesh,
        harmonic_basis=harmonic_basis,
    )
    abel_jacobi = world.require(
        AbelJacobiModule,
        mesh=mesh,
        point_vortex=point_vortex,
        dual_harmonic_field=dual,
    )
    mesh.set_mesh(vertices, faces)
    return SimpleNamespace(
        world=world,
        mesh=mesh,
        dec=dec,
        tree_cotree=tree_cotree,
        point_vortex=point_vortex,
        harmonic_basis=harmonic_basis,
        dual=dual,
        abel_jacobi=abel_jacobi,
    )


def _max_face_edge_integral_error(
    mesh: SurfaceMeshModule,
    gamma: np.ndarray,
    xi_face: np.ndarray,
) -> np.ndarray:
    vertices = mesh.V_pos.get()
    faces = mesh.F_verts.get()
    edge_vectors = np.stack(
        (
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 1]],
            vertices[faces[:, 0]] - vertices[faces[:, 2]],
        ),
        axis=1,
    )
    target = gamma[:, mesh.F_edges.get()] * mesh.F_edge_sign.get()[None, :, :]
    got = np.einsum("kfi,fsi->kfs", xi_face, edge_vectors)
    return np.max(np.abs(got - target), axis=(1, 2))


def _probe_positions(
    mesh: SurfaceMeshModule,
    faceids: np.ndarray,
    bary: np.ndarray,
) -> np.ndarray:
    face_vertices = mesh.F_verts.get()[faceids]
    vertex_positions = mesh.V_pos.get()[face_vertices]
    return np.einsum("ni,nij->nj", bary, vertex_positions)


def _expected_trapezoid_delta_aj(
    mesh: SurfaceMeshModule,
    dual: DualHarmonicFieldModule,
    start_probes: tuple[np.ndarray, np.ndarray],
    end_probes: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    start_faceids, start_bary = start_probes
    end_faceids, end_bary = end_probes
    p0 = _probe_positions(mesh, start_faceids, start_bary)
    p1 = _probe_positions(mesh, end_faceids, end_bary)
    xi0 = dual.interpolate(start_probes, field="xi")
    xi1 = dual.interpolate(end_probes, field="xi")
    return 0.5 * np.einsum("kni,ni->nk", xi0 + xi1, p1 - p0)


@pytest.fixture(scope="module")
def torus_dual_module() -> SimpleNamespace:
    vertices, faces = _load_tri_obj(TORUS_OBJ)
    assert vertices.shape == (2011, 3)
    assert faces.shape == (4022, 3)
    return _build_dual_module(vertices, faces)


def test_vortex_dynamics_app_wires_dual_harmonic_field() -> None:
    mods = World().require(App)

    assert mods.dual_harmonic_field.mesh is mods.mesh
    assert mods.dual_harmonic_field.harmonic_basis is mods.harmonic_basis
    assert mods.abel_jacobi.mesh is mods.mesh
    assert mods.abel_jacobi.point_vortex is mods.point_vortex
    assert mods.abel_jacobi.dual_harmonic_field is mods.dual_harmonic_field


def test_dual_harmonic_field_torus_obj_builds_dual_basis(
    torus_dual_module: SimpleNamespace,
) -> None:
    mesh = torus_dual_module.mesh
    tree_cotree = torus_dual_module.tree_cotree
    harmonic_basis = torus_dual_module.harmonic_basis
    dual = torus_dual_module.dual

    gamma = harmonic_basis.gamma.get()
    xi_face = dual.xi_face.get()
    zeta_face = dual.zeta_face.get()

    assert mesh.boundary_edge_count.get() == 0
    assert tree_cotree.genus.get() == 1
    assert tree_cotree.generator_count.get() == 2
    assert gamma.shape == (2, mesh.E_verts.get().shape[0])
    assert xi_face.shape == (2, 4022, 3)
    assert zeta_face.shape == (2, 4022, 3)
    assert dual.raw_zeta_face.get().shape == (2, 4022, 3)
    assert dual.lambda_raw.get().shape == (2, 2)
    assert dual.final_pairing.get().shape == (2, 2)

    np.testing.assert_allclose(
        _max_face_edge_integral_error(mesh, gamma, xi_face),
        np.zeros(2, dtype=np.float64),
        atol=1e-9,
    )

    face_normals = mesh.F_normal.get()
    np.testing.assert_allclose(
        np.einsum("kfi,fi->kf", xi_face, face_normals),
        np.zeros((2, face_normals.shape[0]), dtype=np.float64),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.einsum("kfi,fi->kf", zeta_face, face_normals),
        np.zeros((2, face_normals.shape[0]), dtype=np.float64),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        dual.final_pairing.get(),
        np.eye(2, dtype=np.float64),
        atol=1e-10,
    )


def test_dual_harmonic_field_utility_methods(
    torus_dual_module: SimpleNamespace,
) -> None:
    mesh = torus_dual_module.mesh
    dual = torus_dual_module.dual
    faceids = np.array([0, 17], dtype=np.int32)
    bary0 = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=np.float64)
    bary1 = np.array([[0.3, 0.2, 0.5], [0.4, 0.4, 0.2]], dtype=np.float64)

    face_vertices = mesh.F_verts.get()[faceids]
    vertex_positions = mesh.V_pos.get()[face_vertices]
    p0 = np.einsum("ni,nij->nj", bary0, vertex_positions)
    p1 = np.einsum("ni,nij->nj", bary1, vertex_positions)

    xi_face = dual.xi_face.get()
    got_integral = dual.integrate_xi_same_face(p0, p1, faceids)
    expected_integral = np.einsum("kni,ni->nk", xi_face[:, faceids, :], p1 - p0)
    np.testing.assert_allclose(got_integral, expected_integral, atol=1e-12)

    coefficients = np.array([0.75, -0.25], dtype=np.float64)
    zeta_face = dual.zeta_face.get()
    got_velocity = dual.harmonic_velocity_at_faces(coefficients, faceids)
    expected_velocity = np.einsum(
        "k,kni->ni",
        coefficients,
        zeta_face[:, faceids, :],
    )
    np.testing.assert_allclose(got_velocity, expected_velocity, atol=1e-12)


def test_abel_jacobi_delta_aj_same_face_uses_trapezoid_rule(
    torus_dual_module: SimpleNamespace,
) -> None:
    mesh = torus_dual_module.mesh
    dual = torus_dual_module.dual
    abel_jacobi = torus_dual_module.abel_jacobi
    faceids = np.array([0, 17], dtype=np.int32)
    bary0 = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=np.float64)
    bary1 = np.array([[0.3, 0.2, 0.5], [0.4, 0.4, 0.2]], dtype=np.float64)

    got = abel_jacobi.delta_aj((faceids, bary0), (faceids, bary1))
    expected = _expected_trapezoid_delta_aj(
        mesh,
        dual,
        (faceids, bary0),
        (faceids, bary1),
    )

    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_abel_jacobi_delta_aj_allows_different_endpoint_faces(
    torus_dual_module: SimpleNamespace,
) -> None:
    mesh = torus_dual_module.mesh
    dual = torus_dual_module.dual
    abel_jacobi = torus_dual_module.abel_jacobi
    faceids0 = np.array([0, 17], dtype=np.int32)
    faceids1 = np.array([1, 23], dtype=np.int32)
    bary0 = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=np.float64)
    bary1 = np.array([[0.5, 0.2, 0.3], [0.2, 0.5, 0.3]], dtype=np.float64)

    got = abel_jacobi.delta_aj((faceids0, bary0), (faceids1, bary1))
    expected = _expected_trapezoid_delta_aj(
        mesh,
        dual,
        (faceids0, bary0),
        (faceids1, bary1),
    )

    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_abel_jacobi_delta_aj_accepts_explicit_positions(
    torus_dual_module: SimpleNamespace,
) -> None:
    mesh = torus_dual_module.mesh
    dual = torus_dual_module.dual
    abel_jacobi = torus_dual_module.abel_jacobi
    faceids0 = np.array([0, 17], dtype=np.int32)
    faceids1 = np.array([1, 23], dtype=np.int32)
    bary0 = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=np.float64)
    bary1 = np.array([[0.5, 0.2, 0.3], [0.2, 0.5, 0.3]], dtype=np.float64)
    pos0 = _probe_positions(mesh, faceids0, bary0)
    pos1 = _probe_positions(mesh, faceids1, bary1)

    got = abel_jacobi.delta_aj(
        (faceids0, bary0),
        (faceids1, bary1),
        pos0=pos0,
        pos1=pos1,
    )
    expected = 0.5 * np.einsum(
        "kni,ni->nk",
        dual.interpolate((faceids0, bary0), field="xi")
        + dual.interpolate((faceids1, bary1), field="xi"),
        pos1 - pos0,
    )

    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_abel_jacobi_delta_aj_empty_basis_returns_vortex_rows() -> None:
    vertices, faces = _tetrahedron_mesh()
    mods = _build_dual_module(vertices, faces)
    faceids = np.array([0, 1], dtype=np.int32)
    bary0 = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=np.float64)
    bary1 = np.array([[0.3, 0.2, 0.5], [0.4, 0.4, 0.2]], dtype=np.float64)

    got = mods.abel_jacobi.delta_aj((faceids, bary0), (faceids, bary1))

    np.testing.assert_array_equal(got, np.empty((2, 0), dtype=np.float64))


def test_abel_jacobi_delta_aj_uses_dynamic_generator_count() -> None:
    vertices, faces = _square_disk_mesh()
    world = World()
    mesh = world.require(SurfaceMeshModule)
    point_vortex = world.require(PointVortexModule)
    xi_face = np.array(
        [
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.0, 2.0, 0.0], [0.0, 0.5, 0.0]],
            [[1.0, 1.0, 0.0], [0.25, 0.75, 0.0]],
            [[-0.5, 0.0, 0.0], [0.0, -1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    dual = _FakeDualHarmonicField(xi_face)
    abel_jacobi = world.require(
        AbelJacobiModule,
        mesh=mesh,
        point_vortex=point_vortex,
        dual_harmonic_field=dual,
    )
    mesh.set_mesh(vertices, faces)
    faceids0 = np.array([0, 1], dtype=np.int32)
    faceids1 = np.array([1, 0], dtype=np.int32)
    bary0 = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=np.float64)
    bary1 = np.array([[0.4, 0.2, 0.4], [0.2, 0.6, 0.2]], dtype=np.float64)
    pos0 = _probe_positions(mesh, faceids0, bary0)
    pos1 = _probe_positions(mesh, faceids1, bary1)

    got = abel_jacobi.delta_aj((faceids0, bary0), (faceids1, bary1))
    expected = 0.5 * np.einsum(
        "kni,ni->nk",
        xi_face[:, faceids0, :] + xi_face[:, faceids1, :],
        pos1 - pos0,
    )

    assert got.shape == (2, 4)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_abel_jacobi_delta_aj_validates_input_shapes(
    torus_dual_module: SimpleNamespace,
) -> None:
    abel_jacobi = torus_dual_module.abel_jacobi
    faceids = np.array([0, 17], dtype=np.int32)
    bary = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=np.float64)

    with pytest.raises(ValueError, match="same number"):
        abel_jacobi.delta_aj(
            (faceids, bary),
            (np.array([0], dtype=np.int32), bary[:1]),
        )

    with pytest.raises(ValueError, match="pos0"):
        abel_jacobi.delta_aj((faceids, bary), (faceids, bary), pos0=np.zeros((1, 3)))


def test_dual_harmonic_field_interpolate_uses_probe_convention(
    torus_dual_module: SimpleNamespace,
) -> None:
    dual = torus_dual_module.dual
    faceids = np.array([0, 17], dtype=np.int32)
    bary = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=np.float64)

    np.testing.assert_allclose(
        dual.interpolate((faceids, bary)),
        dual.zeta_face.get()[:, faceids, :],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        dual.interpolate((faceids, bary), field="xi"),
        dual.xi_face.get()[:, faceids, :],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        dual.interpolate(list(zip(faceids, bary)), field="zeta_tilde"),
        dual.raw_zeta_face.get()[:, faceids, :],
        atol=1e-12,
    )

    with pytest.raises(ValueError, match="Unknown harmonic field"):
        dual.interpolate((faceids, bary), field="not_a_field")


def test_dual_harmonic_field_sphere_returns_empty_basis() -> None:
    vertices, faces = _tetrahedron_mesh()
    mods = _build_dual_module(vertices, faces)
    dual = mods.dual

    assert mods.tree_cotree.genus.get() == 0
    np.testing.assert_array_equal(
        dual.xi_face.get(),
        np.empty((0, faces.shape[0], 3), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        dual.raw_zeta_face.get(),
        np.empty((0, faces.shape[0], 3), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        dual.zeta_face.get(),
        np.empty((0, faces.shape[0], 3), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        dual.lambda_raw.get(),
        np.empty((0, 0), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        dual.final_pairing.get(),
        np.empty((0, 0), dtype=np.float64),
    )
