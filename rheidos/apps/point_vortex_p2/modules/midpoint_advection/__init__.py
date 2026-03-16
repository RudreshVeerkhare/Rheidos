from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rheidos.compute import ModuleBase, ProducerBase, ResourceSpec, World

from ..fe_utils import barycentric_from_point, barycentric_gradients, clamp_renorm_bary, project_tangent, renorm_bary
from ..p2_velocity import P2VelocityModule, sample_velocity_from_corners
from ..point_vortex import PointVortexModule
from ..surface_mesh import SurfaceMeshModule


def advance_const_velocity_event_driven(
    vertices: np.ndarray,
    faces: np.ndarray,
    f_adj: np.ndarray,
    face_id: int,
    bary: np.ndarray,
    vel_world: np.ndarray,
    dt: float,
    *,
    eps: float = 1e-10,
    max_hops: int = 32,
) -> tuple[int, np.ndarray, np.ndarray, int]:
    """Advance one particle with constant world velocity and edge walking."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    adj = np.asarray(f_adj, dtype=np.int32)

    fid = int(face_id)
    if fid < 0 or fid >= f.shape[0]:
        raise RuntimeError(f"Invalid starting face id {fid}")

    b = renorm_bary(np.asarray(bary, dtype=np.float64))
    u = np.asarray(vel_world, dtype=np.float64)
    remaining = float(dt)
    hops = 0

    while remaining > eps:
        tri = f[fid]
        a, bb, c = v[tri[0]], v[tri[1]], v[tri[2]]

        g0, g1, g2, n_hat, area = barycentric_gradients(a, bb, c)
        if area <= 1e-20:
            raise RuntimeError(f"Degenerate face encountered during advection (face={fid})")

        u_tan = project_tangent(u, n_hat)
        db = np.array([np.dot(g0, u_tan), np.dot(g1, u_tan), np.dot(g2, u_tan)], dtype=np.float64)

        t_hit = remaining
        hit_idx = -1
        for i in range(3):
            if db[i] < -eps:
                cand = -b[i] / db[i]
                if cand < t_hit:
                    t_hit = float(cand)
                    hit_idx = i

        if hit_idx < 0:
            b = b + remaining * db
            remaining = 0.0
            break

        # Advance to crossing.
        b = b + t_hit * db
        p = b[0] * a + b[1] * bb + b[2] * c

        remaining -= t_hit
        if remaining <= eps:
            break

        nbr = int(adj[fid, hit_idx])
        if nbr < 0:
            raise RuntimeError(
                "Boundary edge crossing detected in midpoint advection. "
                "point_vortex_p2 requires closed surfaces."
            )

        fid = nbr
        tri_n = f[fid]
        a_n, b_n, c_n = v[tri_n[0]], v[tri_n[1]], v[tri_n[2]]
        b = barycentric_from_point(p, a_n, b_n, c_n)
        b = renorm_bary(b)

        hops += 1
        if hops > max_hops:
            raise RuntimeError("Exceeded max_hops during edge walking; check mesh/advection settings")

    b = clamp_renorm_bary(b)
    tri = f[fid]
    a, bb, c = v[tri[0]], v[tri[1]], v[tri[2]]
    p_out = b[0] * a + b[1] * bb + b[2] * c
    return fid, b, p_out, hops


def advect_single_field_batch(
    vertices: np.ndarray,
    faces: np.ndarray,
    f_adj: np.ndarray,
    vel_corner: np.ndarray,
    face_ids: np.ndarray,
    bary: np.ndarray,
    dt: float,
    *,
    eps: float = 1e-10,
    max_hops: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Advect with one frozen velocity field.

    Velocity is sampled at each particle's current (face, bary), then held constant
    during edge-walk advection for this substep.
    """
    face_ids = np.asarray(face_ids, dtype=np.int32)
    bary = np.asarray(bary, dtype=np.float64)
    n = int(face_ids.shape[0])

    face_out = np.empty((n,), dtype=np.int32)
    bary_out = np.empty((n, 3), dtype=np.float64)
    pos_out = np.empty((n, 3), dtype=np.float64)
    hops_out = np.empty((n,), dtype=np.int32)

    for i in range(n):
        fid0 = int(face_ids[i])
        b0 = bary[i]
        u0 = sample_velocity_from_corners(vel_corner, fid0, b0)
        fid1, b1, p1, hops = advance_const_velocity_event_driven(
            vertices,
            faces,
            f_adj,
            fid0,
            b0,
            u0,
            dt,
            eps=eps,
            max_hops=max_hops,
        )
        face_out[i] = fid1
        bary_out[i] = b1
        pos_out[i] = p1
        hops_out[i] = int(hops)

    return face_out, bary_out, pos_out, hops_out


def advect_stage_b_from_midpoint_batch(
    vertices: np.ndarray,
    faces: np.ndarray,
    f_adj: np.ndarray,
    vel_corner: np.ndarray,
    face_ids_start: np.ndarray,
    bary_start: np.ndarray,
    face_ids_mid: np.ndarray,
    bary_mid: np.ndarray,
    dt: float,
    *,
    eps: float = 1e-10,
    max_hops: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    RK2 stage B: sample velocity at midpoint state, then push full dt from start.
    """
    face_ids_start = np.asarray(face_ids_start, dtype=np.int32)
    bary_start = np.asarray(bary_start, dtype=np.float64)
    face_ids_mid = np.asarray(face_ids_mid, dtype=np.int32)
    bary_mid = np.asarray(bary_mid, dtype=np.float64)

    n = int(face_ids_start.shape[0])
    if face_ids_mid.shape[0] != n or bary_start.shape[0] != n or bary_mid.shape[0] != n:
        raise ValueError("start and midpoint arrays must have matching first dimension")

    face_out = np.empty((n,), dtype=np.int32)
    bary_out = np.empty((n, 3), dtype=np.float64)
    pos_out = np.empty((n, 3), dtype=np.float64)
    hops_out = np.empty((n,), dtype=np.int32)

    for i in range(n):
        fid0 = int(face_ids_start[i])
        b0 = bary_start[i]
        fid_mid = int(face_ids_mid[i])
        b_mid = bary_mid[i]

        umid = sample_velocity_from_corners(vel_corner, fid_mid, b_mid)
        fid1, b1, p1, hops = advance_const_velocity_event_driven(
            vertices,
            faces,
            f_adj,
            fid0,
            b0,
            umid,
            dt,
            eps=eps,
            max_hops=max_hops,
        )
        face_out[i] = fid1
        bary_out[i] = b1
        pos_out[i] = p1
        hops_out[i] = int(hops)

    return face_out, bary_out, pos_out, hops_out


def advect_midpoint_batch(
    vertices: np.ndarray,
    faces: np.ndarray,
    f_adj: np.ndarray,
    vel_corner: np.ndarray,
    face_ids: np.ndarray,
    bary: np.ndarray,
    dt: float,
    *,
    eps: float = 1e-10,
    max_hops: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Midpoint advection: stage A half-step then stage B full-step from start."""
    face_ids = np.asarray(face_ids, dtype=np.int32)
    bary = np.asarray(bary, dtype=np.float64)

    face_mid, bary_mid, _, hops_a = advect_single_field_batch(
        vertices,
        faces,
        f_adj,
        vel_corner,
        face_ids,
        bary,
        0.5 * dt,
        eps=eps,
        max_hops=max_hops,
    )
    face_out, bary_out, pos_out, hops_b = advect_stage_b_from_midpoint_batch(
        vertices,
        faces,
        f_adj,
        vel_corner,
        face_ids,
        bary,
        face_mid,
        bary_mid,
        dt,
        eps=eps,
        max_hops=max_hops,
    )

    hops_per_particle = hops_a.astype(np.int64) + hops_b.astype(np.int64)
    hops_total = int(np.sum(hops_per_particle, dtype=np.int64))
    hops_max = int(np.max(hops_per_particle)) if hops_per_particle.size > 0 else 0
    return face_out, bary_out, pos_out, hops_total, hops_max


@dataclass
class MidpointAdvectionProducer(ProducerBase):
    dt: str
    V_pos: str
    F_verts: str
    F_adj: str
    vel_corner: str
    face_ids: str
    bary: str

    face_ids_out: str
    bary_out: str
    pos_out: str
    hops_total: str
    hops_max: str

    @property
    def outputs(self):
        return (
            self.face_ids_out,
            self.bary_out,
            self.pos_out,
            self.hops_total,
            self.hops_max,
        )

    def compute(self, reg) -> None:
        dt = float(reg.read(self.dt))
        v = np.asarray(reg.read(self.V_pos), dtype=np.float64)
        f = np.asarray(reg.read(self.F_verts), dtype=np.int32)
        adj = np.asarray(reg.read(self.F_adj), dtype=np.int32)
        vel_corner = np.asarray(reg.read(self.vel_corner), dtype=np.float64)
        face_ids = np.asarray(reg.read(self.face_ids), dtype=np.int32)
        bary = np.asarray(reg.read(self.bary), dtype=np.float64)

        face_out, bary_out, pos_out, hops_total, hops_max = advect_midpoint_batch(
            v,
            f,
            adj,
            vel_corner,
            face_ids,
            bary,
            dt,
        )

        reg.commit(self.face_ids_out, buffer=face_out)
        reg.commit(self.bary_out, buffer=bary_out)
        reg.commit(self.pos_out, buffer=pos_out)
        reg.commit(self.hops_total, buffer=int(hops_total))
        reg.commit(self.hops_max, buffer=int(hops_max))


class MidpointAdvectionModule(ModuleBase):
    NAME = "P2MidpointAdvection"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.mesh = world.require(SurfaceMeshModule, scope=scope)
        self.vort = world.require(PointVortexModule, scope=scope)
        self.velocity = world.require(P2VelocityModule, scope=scope)

        self.dt = self.resource(
            "dt",
            declare=True,
            spec=ResourceSpec(kind="python", dtype=float, allow_none=True),
        )

        self.face_ids_out = self.resource(
            "face_ids_out",
            spec=ResourceSpec(kind="numpy", dtype=np.int32, allow_none=True),
        )
        self.bary_out = self.resource(
            "bary_out",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
        )
        self.pos_out = self.resource(
            "pos_out",
            spec=ResourceSpec(kind="numpy", dtype=np.float64, allow_none=True),
        )
        self.hops_total = self.resource(
            "hops_total",
            spec=ResourceSpec(kind="python", dtype=int, allow_none=True),
        )
        self.hops_max = self.resource(
            "hops_max",
            spec=ResourceSpec(kind="python", dtype=int, allow_none=True),
        )

        prod = MidpointAdvectionProducer(
            dt=self.dt.name,
            V_pos=self.mesh.V_pos.name,
            F_verts=self.mesh.F_verts.name,
            F_adj=self.mesh.F_adj.name,
            vel_corner=self.velocity.vel_corner.name,
            face_ids=self.vort.face_ids.name,
            bary=self.vort.bary.name,
            face_ids_out=self.face_ids_out.name,
            bary_out=self.bary_out.name,
            pos_out=self.pos_out.name,
            hops_total=self.hops_total.name,
            hops_max=self.hops_max.name,
        )

        deps = (
            self.dt,
            self.mesh.V_pos,
            self.mesh.F_verts,
            self.mesh.F_adj,
            self.velocity.vel_corner,
            self.vort.face_ids,
            self.vort.bary,
        )

        self.declare_resource(self.face_ids_out, deps=deps, producer=prod)
        self.declare_resource(self.bary_out, deps=deps, producer=prod)
        self.declare_resource(self.pos_out, deps=deps, producer=prod)
        self.declare_resource(self.hops_total, deps=deps, producer=prod)
        self.declare_resource(self.hops_max, deps=deps, producer=prod)

    def advect(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.dt.set(float(dt))
        return (
            np.asarray(self.face_ids_out.get(), dtype=np.int32),
            np.asarray(self.bary_out.get(), dtype=np.float64),
            np.asarray(self.pos_out.get(), dtype=np.float64),
        )
