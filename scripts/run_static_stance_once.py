"""Solve the first static four-foot stance WBC QP for Unitree Go2."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
FOOT_GEOMS = ("FL", "FR", "RL", "RR")

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import MuJoCoModelInterface, StanceWBCConfig, StanceWBCQP  # noqa: E402


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")
    qpos_ref = robot.q

    config = StanceWBCConfig(foot_geoms=FOOT_GEOMS)
    solution = StanceWBCQP(config).solve(robot, qpos_ref)

    forces = solution.contact_forces.reshape(len(FOOT_GEOMS), 3)
    total_force = np.sum(forces, axis=0)
    total_mass = float(np.sum(robot.model.body_mass))
    weight = total_mass * abs(float(robot.model.opt.gravity[2]))

    print("=== Static stance WBC QP ===")
    print(f"status              = {solution.status}")
    print(f"objective           = {solution.objective:.6f}")
    print(f"nq, nv, nu          = {robot.nq}, {robot.nv}, {robot.nu}")
    print(f"decision variables  = vdot({robot.nv}) + tau({robot.nu}) + f({solution.contact_forces.size})")

    print("\n=== Residual checks ===")
    print(f"||dynamics residual|| = {np.linalg.norm(solution.dynamics_residual):.3e}")
    print(f"||stance residual||   = {np.linalg.norm(solution.stance_residual):.3e}")
    print(f"max |vdot|            = {np.max(np.abs(solution.vdot)):.3e}")
    print(f"max |tau|             = {np.max(np.abs(solution.tau)):.3e}")

    print("\n=== Contact forces, world frame ===")
    for name, force in zip(FOOT_GEOMS, forces):
        print(f"{name}: fx={force[0]:9.4f}  fy={force[1]:9.4f}  fz={force[2]:9.4f}")
    print(f"sum: fx={total_force[0]:9.4f}  fy={total_force[1]:9.4f}  fz={total_force[2]:9.4f}")
    print(f"robot weight          = {weight:.4f} N")
    print(f"sum_fz / weight       = {total_force[2] / weight:.6f}")

    print("\n=== Joint torques ===")
    for idx, torque in enumerate(solution.tau):
        print(f"tau[{idx:02d}] = {torque:9.4f}")

    print("\nEquation:")
    print("M(q) vdot + h(q, v) = B tau + Jc(q)^T f")
    print("Jc(q) vdot + Jdot_c(q, v) v = 0")


if __name__ == "__main__":
    main()
