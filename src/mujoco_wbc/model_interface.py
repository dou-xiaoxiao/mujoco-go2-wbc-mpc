"""Model-level dynamics and kinematics accessors for MuJoCo WBC.

All dynamics quantities live in MuJoCo's generalized velocity space:

    M(q) vdot + h(q, v) = B(q, v) tau + J(q)^T f

For a floating-base quadruped with 12 actuated joints this means nq=19,
nv=18, and nu=12. The base orientation in qpos is a quaternion, while qvel
stores a 3D angular velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .conventions import BASE_BODY_NAME


Array = np.ndarray


@dataclass(frozen=True)
class FrameJacobian:
    """Translational and rotational Jacobians for one MuJoCo frame."""

    jacp: Array
    jacr: Array


class MuJoCoModelInterface:
    """Thin wrapper around MjModel/MjData for WBC math interfaces."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.base_body_name = BASE_BODY_NAME
        self.base_body_id = self._body_id(self.base_body_name)
        mujoco.mj_forward(self.model, self.data)

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def nu(self) -> int:
        return self.model.nu

    @property
    def q(self) -> Array:
        return self.data.qpos.copy()

    @property
    def v(self) -> Array:
        return self.data.qvel.copy()

    def set_keyframe(self, name: str) -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id < 0:
            raise ValueError(f"Unknown keyframe: {name}")
        self.data.qpos[:] = self.model.key_qpos[key_id]
        if self.model.key_ctrl.shape[1] == self.nu:
            self.data.ctrl[:] = self.model.key_ctrl[key_id]
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def mass_matrix(self) -> Array:
        mass = np.zeros((self.nv, self.nv), dtype=float)
        mujoco.mj_fullM(self.model, self.data, mass)
        return mass

    def passive_forces(self) -> Array:
        return self.data.qfrc_passive.copy()

    def bias_forces(self, include_passive: bool = False) -> Array:
        """Return h(q, v) in generalized velocity coordinates.

        MuJoCo stores gravity/Coriolis/centrifugal terms in qfrc_bias and
        passive joint damping/friction in qfrc_passive. For the WBC equation
        M vdot + h = B tau + J^T f, passive generalized forces should be moved
        to the left side, hence h = qfrc_bias - qfrc_passive by default.
        """

        if include_passive:
            return self.data.qfrc_bias.copy()
        return self.data.qfrc_bias.copy() - self.data.qfrc_passive.copy()

    def actuation_matrix(self) -> Array:
        """Return B such that qfrc_actuator = B tau for unit motor controls."""

        original_ctrl = self.data.ctrl.copy()
        original_qacc = self.data.qacc.copy()
        matrix = np.zeros((self.nv, self.nu), dtype=float)

        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        for actuator_id in range(self.nu):
            self.data.ctrl[:] = 0.0
            self.data.ctrl[actuator_id] = 1.0
            mujoco.mj_forward(self.model, self.data)
            matrix[:, actuator_id] = self.data.qfrc_actuator

        self.data.ctrl[:] = original_ctrl
        self.data.qacc[:] = original_qacc
        mujoco.mj_forward(self.model, self.data)
        return matrix

    def actuator_forces(self) -> Array:
        return self.data.qfrc_actuator.copy()

    def total_mass(self) -> float:
        return float(np.sum(self.model.body_mass))

    def center_of_mass(self) -> Array:
        """Return the full robot COM in W.

        For the free-floating Go2 model, the subtree rooted at the floating
        base contains the complete robot.
        """

        return self.data.subtree_com[self.base_body_id].copy()

    def composite_inertia_world_about_com(self) -> Array:
        """Return whole-robot rotational inertia about the COM, expressed in W."""

        com = self.center_of_mass()
        inertia = np.zeros((3, 3), dtype=float)

        for body_id in range(1, self.model.nbody):
            mass = float(self.model.body_mass[body_id])
            if mass <= 0.0:
                continue

            body_inertia_frame = np.diag(self.model.body_inertia[body_id])
            r_world = self.data.xipos[body_id] - com
            rotation_world_from_body = self.data.xmat[body_id].reshape(3, 3)
            rotation_body_from_inertia = quat_to_matrix(self.model.body_iquat[body_id])
            rotation_world_from_inertia = rotation_world_from_body @ rotation_body_from_inertia
            body_inertia_world = rotation_world_from_inertia @ body_inertia_frame @ rotation_world_from_inertia.T

            inertia += body_inertia_world + mass * (
                np.dot(r_world, r_world) * np.eye(3) - np.outer(r_world, r_world)
            )

        return inertia

    def composite_inertia_base_about_com(self) -> Array:
        """Return whole-robot rotational inertia about the COM, expressed in B."""

        rotation_base_from_world = self.base_rotation_base_from_world()
        return rotation_base_from_world @ self.composite_inertia_world_about_com() @ rotation_base_from_world.T

    def base_position(self) -> Array:
        return self.data.xpos[self.base_body_id].copy()

    def base_linear_velocity(self) -> Array:
        return self.data.qvel[0:3].copy()

    def base_angular_velocity(self) -> Array:
        return self.data.qvel[3:6].copy()

    def base_rotation_world_from_base(self) -> Array:
        """Return R_WB, mapping vectors expressed in B to vectors in W."""

        return self.data.xmat[self.base_body_id].reshape(3, 3).copy()

    def base_rotation_base_from_world(self) -> Array:
        """Return R_BW, mapping vectors expressed in W to vectors in B."""

        return self.base_rotation_world_from_base().T

    def world_vector_to_base(self, vector_world: Array) -> Array:
        return self.base_rotation_base_from_world() @ np.asarray(vector_world, dtype=float)

    def base_vector_to_world(self, vector_base: Array) -> Array:
        return self.base_rotation_world_from_base() @ np.asarray(vector_base, dtype=float)

    def world_point_to_base(self, point_world: Array) -> Array:
        return self.world_vector_to_base(np.asarray(point_world, dtype=float) - self.base_position())

    def base_point_to_world(self, point_base: Array) -> Array:
        return self.base_position() + self.base_vector_to_world(point_base)

    def geom_jacobian(self, geom_name: str) -> FrameJacobian:
        geom_id = self._geom_id(geom_name)
        jacp = np.zeros((3, self.nv), dtype=float)
        jacr = np.zeros((3, self.nv), dtype=float)
        mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, geom_id)
        return FrameJacobian(jacp=jacp, jacr=jacr)

    def site_jacobian(self, site_name: str) -> FrameJacobian:
        site_id = self._site_id(site_name)
        jacp = np.zeros((3, self.nv), dtype=float)
        jacr = np.zeros((3, self.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return FrameJacobian(jacp=jacp, jacr=jacr)

    def geom_position(self, geom_name: str) -> Array:
        return self.data.geom_xpos[self._geom_id(geom_name)].copy()

    def geom_has_contact(self, geom_name: str) -> bool:
        """Return True when the named geom is in any active MuJoCo contact."""

        geom_id = self._geom_id(geom_name)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            if contact.geom1 == geom_id or contact.geom2 == geom_id:
                return True
        return False

    def geom_position_in_base(self, geom_name: str) -> Array:
        return self.world_point_to_base(self.geom_position(geom_name))

    def geom_velocity(self, geom_name: str) -> Array:
        return self.geom_jacobian(geom_name).jacp @ self.data.qvel

    def geom_velocity_in_base(self, geom_name: str) -> Array:
        """Return foot point linear velocity expressed in B.

        This is only the world point velocity re-expressed in the base frame:

            R_BW v_WF

        It is not the time derivative of p_BF. The relative derivative would
        also subtract base translation and angular terms.
        """

        return self.world_vector_to_base(self.geom_velocity(geom_name))

    def site_position(self, site_name: str) -> Array:
        return self.data.site_xpos[self._site_id(site_name)].copy()

    def site_position_in_base(self, site_name: str) -> Array:
        return self.world_point_to_base(self.site_position(site_name))

    def site_velocity(self, site_name: str) -> Array:
        return self.site_jacobian(site_name).jacp @ self.data.qvel

    def site_velocity_in_base(self, site_name: str) -> Array:
        return self.world_vector_to_base(self.site_velocity(site_name))

    def geom_jdot_v(self, geom_name: str, eps: float = 1e-6) -> Array:
        return self._jdot_v_fd(lambda iface: iface.geom_jacobian(geom_name).jacp, eps)

    def site_jdot_v(self, site_name: str, eps: float = 1e-6) -> Array:
        return self._jdot_v_fd(lambda iface: iface.site_jacobian(site_name).jacp, eps)

    def stacked_geom_jacobian(self, geom_names: list[str]) -> Array:
        return np.vstack([self.geom_jacobian(name).jacp for name in geom_names])

    def stacked_geom_jdot_v(self, geom_names: list[str]) -> Array:
        return np.concatenate([self.geom_jdot_v(name) for name in geom_names])

    def dynamics_residual(self, vdot: Array, tau: Array, contact_geoms: list[str], contact_forces: Array) -> Array:
        """Compute M vdot + h - B tau - Jc^T f."""

        contact_forces = np.asarray(contact_forces, dtype=float).reshape(-1)
        jc = self.stacked_geom_jacobian(contact_geoms)
        return (
            self.mass_matrix() @ np.asarray(vdot, dtype=float)
            + self.bias_forces()
            - self.actuation_matrix() @ np.asarray(tau, dtype=float)
            - jc.T @ contact_forces
        )

    def _jdot_v_fd(self, jacobian_fn, eps: float) -> Array:
        q0 = self.data.qpos.copy()
        v0 = self.data.qvel.copy()
        j0 = jacobian_fn(self)

        scratch = MuJoCoModelInterface.__new__(MuJoCoModelInterface)
        scratch.model_path = self.model_path
        scratch.model = self.model
        scratch.data = mujoco.MjData(self.model)
        scratch.data.qpos[:] = q0
        scratch.data.qvel[:] = v0
        mujoco.mj_integratePos(self.model, scratch.data.qpos, v0, eps)
        mujoco.mj_forward(self.model, scratch.data)
        j1 = jacobian_fn(scratch)

        return ((j1 - j0) / eps) @ v0

    def _geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"Unknown geom: {name}")
        return geom_id

    def _site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Unknown site: {name}")
        return site_id

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Unknown body: {name}")
        return body_id


def quat_to_matrix(quat: Array) -> Array:
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)
