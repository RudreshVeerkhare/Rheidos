import taichi as ti


@ti.data_oriented
class SampleVortexVelProducer:
    """
    Samples an RT0 affine velocity field represented per face by corner velocities:

        vel_FV[f,loc]  loc=0..2 corresponds to vertex F_verts[f][loc]

    At vortex position stored as (face_id, bary):
        v = b0*vel_FV[f,0] + b1*vel_FV[f,1] + b2*vel_FV[f,2]
    """

    def __init__(self) -> None:
        pass

    @ti.kernel
    def _sample_from_face_corner_vel(
        self,
        vel_FV: ti.template(),  # (nF, 3) vec3f
        pt_bary: ti.template(),  # (maxP,) vec3f
        pt_face: ti.template(),  # (maxP,) i32
        pt_vel: ti.template(),  # (maxP,) vec3f [out]
        n_pts: ti.i32,
    ):
        for pid in range(n_pts):
            f = pt_face[pid]
            b = pt_bary[pid]
            v0 = vel_FV[f, 0]
            v1 = vel_FV[f, 1]
            v2 = vel_FV[f, 2]
            pt_vel[pid] = b[0] * v0 + b[1] * v1 + b[2] * v2

    def run(
        self,
        *,
        vel_FV,
        pt_bary,
        pt_face,
        pt_vel_out,
        n_pts: int,
    ) -> None:
        self._sample_from_face_corner_vel(vel_FV, pt_bary, pt_face, pt_vel_out, n_pts)
