"""Coordinate and dynamics conventions used by the WBC code.

Frames:
    W: MuJoCo/world frame.
    B: floating base body frame, the body named "base" in the Go2 model.
    F: foot frame/point, currently represented by the MuJoCo foot geom point.

Floating-base state:
    qpos[0:3] = base position p_WB, expressed in W.
    qpos[3:7] = base quaternion [w, x, y, z].
    qpos[7:] = actuated joint positions.

    qvel[0:3] = base linear velocity v_WB, expressed in W.
    qvel[3:6] = base angular velocity omega_WB, expressed in W.
    qvel[6:] = actuated joint velocities.

MuJoCo Jacobians:
    mj_jacGeom/mj_jacSite translational Jacobian jacp maps qvel to world-frame
    point velocity:

        v_WF = J_WF(q) v

    The WBC contact and swing tasks use this world-frame Jacobian directly.

Contact forces:
    Contact force variables are ordered per foot as [fx, fy, fz], expressed in
    W and applied at the foot geom point. The generalized force contribution is:

        tau_contact = J_WF(q).T f_W

Dynamics:
    All dynamics live in generalized velocity coordinates:

        M(q) vdot + h(q, v) = B tau + Jc(q).T f

    vdot has length nv, not nq. For floating bases, qpos and qvel have
    different dimensions because qpos uses a quaternion.
"""

from __future__ import annotations


BASE_BODY_NAME = "base"

QPOS_BASE_POS = slice(0, 3)
QPOS_BASE_QUAT = slice(3, 7)
QPOS_JOINTS = slice(7, None)

QVEL_BASE_LINEAR = slice(0, 3)
QVEL_BASE_ANGULAR = slice(3, 6)
QVEL_JOINTS = slice(6, None)

FOOT_FORCE_DIM = 3
