from __future__ import annotations

import numpy as np
import pytest

from rheidos.apps.p2.mobius.cook import App
from rheidos.apps.p2.mobius.intrinsic_advection import (
    IntrinsicVortexState,
    RK4AdvectorModule,
    ReduceTimestepError,
    positions_from_state,
)
from rheidos.apps.p2.modules.surface_mesh.surface_mesh_module import SurfaceMeshModule
from rheidos.compute import World


def _planar_square() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return vertices, faces


def _planar_grid(
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[x, y, 0.0] for y in range(height + 1) for x in range(width + 1)],
        dtype=np.float64,
    )
    faces: list[list[int]] = []
    row_size = width + 1
    for y in range(height):
        for x in range(width):
            lower_left = row_size * y + x
            lower_right = lower_left + 1
            upper_left = lower_left + row_size
            upper_right = upper_left + 1
            faces.extend(
                (
                    [lower_left, lower_right, upper_right],
                    [lower_left, upper_right, upper_left],
                )
            )
    return vertices, np.asarray(faces, dtype=np.int32)


def _advector(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[SurfaceMeshModule, RK4AdvectorModule]:
    world = World()
    mesh = world.require(SurfaceMeshModule)
    mesh.set_mesh(vertices, faces)
    return mesh, world.require(RK4AdvectorModule, mesh=mesh)


def _folded_strip() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angles = np.array([0.0, 0.55, -0.35, 0.8], dtype=np.float64)
    directions = np.column_stack(
        (np.cos(angles), np.zeros(angles.shape[0]), np.sin(angles))
    )
    columns = np.zeros((directions.shape[0] + 1, 3), dtype=np.float64)
    columns[1:] = np.cumsum(directions, axis=0)
    width = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    vertices = np.vstack((columns, columns + width))

    faces: list[list[int]] = []
    row_size = columns.shape[0]
    for cell in range(directions.shape[0]):
        lower_left = cell
        lower_right = cell + 1
        upper_left = row_size + cell
        upper_right = upper_left + 1
        faces.extend(
            (
                [lower_left, lower_right, upper_right],
                [lower_left, upper_right, upper_left],
            )
        )
    return vertices, np.asarray(faces, dtype=np.int32), columns, directions


def test_euler_advection_crosses_one_edge_by_local_barycentric_transfer() -> None:
    vertices, faces = _planar_square()
    mesh, advector = _advector(vertices, faces)
    initial = IntrinsicVortexState(
        np.array([0], dtype=np.int32),
        np.array([[0.2, 0.4, 0.4]], dtype=np.float64),
    )

    result = advector.advect_euler(
        initial,
        np.array([[0.4, 0.4, 3.0]], dtype=np.float64),
        1.0,
    )

    np.testing.assert_array_equal(result.face_ids, np.array([1], dtype=np.int32))
    np.testing.assert_allclose(result.bary, np.array([[0.2, 0.6, 0.2]]))
    np.testing.assert_allclose(
        positions_from_state(mesh, result),
        np.array([[0.8, 0.8, 0.0]]),
    )
    np.testing.assert_allclose(result.bary.sum(axis=1), 1.0)


def test_euler_advection_rejects_physical_boundary_crossing() -> None:
    vertices, faces = _planar_square()
    _, advector = _advector(vertices, faces)
    initial = IntrinsicVortexState(
        np.array([0], dtype=np.int32),
        np.array([[0.2, 0.4, 0.4]], dtype=np.float64),
    )

    with pytest.raises(ReduceTimestepError, match="physical boundary"):
        advector.advect_euler(
            initial,
            np.array([[0.0, -1.0, 0.0]], dtype=np.float64),
            0.5,
        )


def test_euler_advection_walks_across_many_edges() -> None:
    vertices, faces = _planar_grid(width=4, height=1)
    mesh, advector = _advector(vertices, faces)
    initial = IntrinsicVortexState(
        np.array([1], dtype=np.int32),
        np.array([[0.7, 0.1, 0.2]], dtype=np.float64),
    )

    result = advector.advect_euler_with_transport(
        initial,
        np.array([[3.5, 0.0, 0.0]], dtype=np.float64),
        1.0,
    )

    assert result.crossing_counts[0] > 2
    np.testing.assert_array_equal(result.state.face_ids, np.array([6]))
    np.testing.assert_allclose(result.state.bary, np.array([[0.4, 0.3, 0.3]]))
    np.testing.assert_allclose(
        positions_from_state(mesh, result.state),
        np.array([[3.6, 0.3, 0.0]]),
    )


def test_euler_advection_walks_through_a_mesh_vertex() -> None:
    vertices, faces = _planar_grid(width=2, height=2)
    mesh, advector = _advector(vertices, faces)
    initial = IntrinsicVortexState(
        np.array([1], dtype=np.int32),
        np.array([[0.7, 0.2, 0.1]], dtype=np.float64),
    )

    result = advector.advect_euler_with_transport(
        initial,
        np.array([[1.6, 1.4, 0.0]], dtype=np.float64),
        1.0,
    )

    # The trajectory passes exactly through (1,1), requiring consecutive
    # zero-time transitions before it enters the top-right cell.
    assert result.crossing_counts[0] >= 2
    np.testing.assert_array_equal(result.state.face_ids, np.array([6]))
    np.testing.assert_allclose(result.state.bary, np.array([[0.2, 0.1, 0.7]]))
    np.testing.assert_allclose(
        positions_from_state(mesh, result.state),
        np.array([[1.8, 1.7, 0.0]]),
    )


def test_intrinsic_rk4_supports_multi_edge_trial_and_final_states() -> None:
    vertices, faces = _planar_grid(width=4, height=1)
    mesh, advector = _advector(vertices, faces)
    initial = IntrinsicVortexState(
        np.array([1], dtype=np.int32),
        np.array([[0.7, 0.1, 0.2]], dtype=np.float64),
    )

    result = advector.step(
        initial,
        lambda state: np.repeat(
            [[3.5, 0.0, 0.0]],
            state.face_ids.shape[0],
            axis=0,
        ),
        1.0,
    )

    np.testing.assert_array_equal(result.face_ids, np.array([6]))
    np.testing.assert_allclose(
        positions_from_state(mesh, result),
        np.array([[3.6, 0.3, 0.0]]),
    )


def test_multi_edge_path_transport_and_rk4_follow_a_folded_strip() -> None:
    vertices, faces, columns, directions = _folded_strip()
    mesh, advector = _advector(vertices, faces)
    initial = IntrinsicVortexState(
        np.array([1], dtype=np.int32),
        np.array([[0.7, 0.1, 0.2]], dtype=np.float64),
    )
    speed = 3.5

    euler = advector.advect_euler_with_transport(
        initial,
        speed * directions[[0]],
        1.0,
    )
    transported_direction = euler.transport_from_start[0] @ directions[0]
    expected_position = columns[3] + 0.6 * directions[3] + np.array([0.0, 0.3, 0.0])

    assert euler.crossing_counts[0] > 2
    np.testing.assert_allclose(transported_direction, directions[3], atol=1.0e-14)
    np.testing.assert_allclose(
        positions_from_state(mesh, euler.state)[0],
        expected_position,
    )

    def parallel_velocity(state: IntrinsicVortexState) -> np.ndarray:
        cells = state.face_ids // 2
        return speed * directions[cells]

    rk4 = advector.step(initial, parallel_velocity, 1.0)
    np.testing.assert_allclose(
        positions_from_state(mesh, rk4)[0],
        expected_position,
    )


def test_edge_transport_preserves_tangent_norm_and_shared_edge_component() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    mesh, advector = _advector(vertices, faces)
    adjacency = mesh.F_adj.get()
    edge_from_zero = int(np.flatnonzero(adjacency[0] == 1)[0])
    edge_from_one = int(np.flatnonzero(adjacency[1] == 0)[0])
    matrices = advector.transport.get()

    edge = vertices[2] - vertices[1]
    edge /= np.linalg.norm(edge)
    vector = np.array([0.8, 0.3, 0.0], dtype=np.float64)
    transported = matrices[0, edge_from_zero] @ vector
    round_trip = matrices[1, edge_from_one] @ transported

    assert np.dot(transported, mesh.F_normal.get()[1]) == pytest.approx(0.0)
    assert np.linalg.norm(transported) == pytest.approx(np.linalg.norm(vector))
    assert np.dot(transported, edge) == pytest.approx(np.dot(vector, edge))
    np.testing.assert_allclose(round_trip, vector)


def test_intrinsic_rk4_matches_exponential_growth_inside_one_face() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh, advector = _advector(vertices, faces)
    initial = IntrinsicVortexState(
        np.array([0], dtype=np.int32),
        np.array([[0.75, 0.125, 0.125]], dtype=np.float64),
    )

    def velocity(state: IntrinsicVortexState) -> np.ndarray:
        positions = positions_from_state(mesh, state)
        # The normal component exercises the local tangent cleanup.
        return np.column_stack(
            (positions[:, 0], np.zeros(positions.shape[0]), np.full(positions.shape[0], 9.0))
        )

    result = advector.step(initial, velocity, 0.2)
    position = positions_from_state(mesh, result)[0]

    assert position[0] == pytest.approx(0.5 * np.exp(0.2), abs=2.0e-6)
    assert position[1] == pytest.approx(0.5)
    assert advector.time.get() == pytest.approx(0.2)


def test_app_integrates_only_n_lifted_vortices_and_syncs_base_state() -> None:
    vertices, faces = _planar_square()
    app = World().require(App)
    app.cover.set_seam_edge_pairs(np.empty((0, 4), dtype=np.int32))
    app.base_mesh.set_mesh(vertices, faces)
    app.base_point_vortex.set_vortex(
        faceids=np.array([0], dtype=np.int32),
        bary=np.array([[0.4, 0.3, 0.3]], dtype=np.float64),
        gamma=np.array([2.0], dtype=np.float64),
        pos=np.array([[0.3, 0.3, 0.0]], dtype=np.float64),
    )
    app.initialize_lifted_point_vortices()

    pair_history: list[np.ndarray] = []

    class ConstantVelocity:
        def interpolate(self, probes) -> np.ndarray:
            face_ids, _ = probes
            pair_history.append(app.point_vortex.face_ids.get().copy())
            return np.repeat([[0.1, 0.0, 0.0]], face_ids.shape[0], axis=0)

    app.stream_velocity = ConstantVelocity()
    accepted = app.rk4_step(0.5)

    assert accepted.face_ids.shape == (1,)
    assert all(pair.shape == (2,) for pair in pair_history)
    np.testing.assert_allclose(
        app.base_point_vortex.pos_world.get(),
        np.array([[0.35, 0.3, 0.0]]),
    )
    np.testing.assert_allclose(
        app.point_vortex.gamma.get(),
        np.array([2.0, -2.0]),
    )


def test_rejected_app_step_restores_the_last_accepted_lifted_state() -> None:
    vertices, faces = _planar_square()
    app = World().require(App)
    app.cover.set_seam_edge_pairs(np.empty((0, 4), dtype=np.int32))
    app.base_mesh.set_mesh(vertices, faces)
    app.base_point_vortex.set_vortex(
        faceids=np.array([0], dtype=np.int32),
        bary=np.array([[0.4, 0.3, 0.3]], dtype=np.float64),
        gamma=np.array([2.0], dtype=np.float64),
        pos=np.array([[0.3, 0.3, 0.0]], dtype=np.float64),
    )
    initial = app.initialize_lifted_point_vortices()

    class BoundaryCrossingVelocity:
        def interpolate(self, probes) -> np.ndarray:
            face_ids, _ = probes
            # The first half-stage exits the disk through a physical boundary.
            return np.repeat([[0.0, -10.0, 0.0]], face_ids.shape[0], axis=0)

    app.stream_velocity = BoundaryCrossingVelocity()
    with pytest.raises(ReduceTimestepError, match="physical boundary"):
        app.rk4_step(0.1)

    np.testing.assert_array_equal(
        app.lifted_point_vortex.face_ids.get(),
        initial.face_ids,
    )
    np.testing.assert_allclose(app.lifted_point_vortex.bary.get(), initial.bary)
    assert app.rk4.time.get() == pytest.approx(0.0)
