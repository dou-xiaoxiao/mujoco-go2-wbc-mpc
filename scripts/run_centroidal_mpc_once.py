"""Solve one centroidal contact-force QP for Unitree Go2 standing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
FOOT_GEOMS = ("FL", "FR", "RL", "RR")

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import CentroidalForceQP, CentroidalForceQPConfig, MuJoCoModelInterface  # noqa: E402


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")

    config = CentroidalForceQPConfig(contact_geoms=FOOT_GEOMS)
    solution = CentroidalForceQP(config).solve(robot)

    forces = solution.contact_forces.reshape(len(FOOT_GEOMS), 3)
    gravity = np.asarray(robot.model.opt.gravity, dtype=float)
    weight = robot.total_mass() * abs(float(gravity[2]))

    print("=== Centroidal force QP ===")
    print(f"status              = {solution.status}")
    print(f"objective           = {solution.objective:.6f}")
    print(f"mass                = {robot.total_mass():.6f} kg")
    print(f"gravity             = {np.round(gravity, 6).tolist()}")
    print(f"COM world           = {np.round(solution.center_of_mass, 6).tolist()}")

    print("\n=== Contact positions and forces, world frame ===")
    for name, position, force in zip(FOOT_GEOMS, solution.contact_positions, forces):
        ratio_x = abs(force[0]) / max(config.friction_mu * force[2], 1.0e-9)
        ratio_y = abs(force[1]) / max(config.friction_mu * force[2], 1.0e-9)
        print(
            f"{name}: "
            f"p={np.round(position, 5).tolist()}  "
            f"f=[{force[0]:9.4f}, {force[1]:9.4f}, {force[2]:9.4f}]  "
            f"fric_ratio=({ratio_x:.3f}, {ratio_y:.3f})"
        )

    print("\n=== Wrench check at COM ===")
    print(f"desired force       = {np.round(solution.desired_wrench[0:3], 6).tolist()}")
    print(f"net force           = {np.round(solution.net_force, 6).tolist()}")
    print(f"force error         = {np.round(solution.wrench_error[0:3], 9).tolist()}")
    print(f"desired torque      = {np.round(solution.desired_wrench[3:6], 6).tolist()}")
    print(f"net torque          = {np.round(solution.net_torque, 6).tolist()}")
    print(f"torque error        = {np.round(solution.wrench_error[3:6], 9).tolist()}")
    print(f"sum_fz / weight     = {np.sum(forces[:, 2]) / weight:.9f}")

    print("\nEquation:")
    print("sum_i f_i = m (a_com_des - g)")
    print("sum_i (p_i - p_com) x f_i = tau_com_des")


if __name__ == "__main__":
    main()
