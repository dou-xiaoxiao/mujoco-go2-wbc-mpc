"""Solve one three-stance plus FL-swing WBC QP."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
STANCE_FEET = ("FR", "RL", "RR")
SWING_FOOT = "FL"
BODY_SHIFT_XY = np.array([-0.04, -0.04])

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import MuJoCoModelInterface, SingleLegSwingWBCConfig, SingleLegSwingWBCQP  # noqa: E402


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")
    qpos_ref = robot.q
    qpos_ref[0:2] += BODY_SHIFT_XY

    swing_pos_ref = robot.geom_position(SWING_FOOT)
    swing_pos_ref[2] += 0.03

    controller = SingleLegSwingWBCQP(
        SingleLegSwingWBCConfig(stance_foot_geoms=STANCE_FEET, swing_foot_geom=SWING_FOOT)
    )
    solution = controller.solve(robot, qpos_ref, swing_pos_ref)

    forces = solution.contact_forces.reshape(len(STANCE_FEET), 3)
    total_force = np.sum(forces, axis=0)
    total_mass = float(np.sum(robot.model.body_mass))
    weight = total_mass * abs(float(robot.model.opt.gravity[2]))

    print("=== Single-leg swing WBC QP ===")
    print(f"status              = {solution.status}")
    print(f"stance feet         = {STANCE_FEET}")
    print(f"swing foot          = {SWING_FOOT}")
    print(f"body shift xy       = {BODY_SHIFT_XY.tolist()} m")
    print(f"decision variables  = vdot({robot.nv}) + tau({robot.nu}) + f({solution.contact_forces.size})")

    print("\n=== Residual checks ===")
    print(f"||dynamics residual|| = {np.linalg.norm(solution.dynamics_residual):.3e}")
    print(f"||stance residual||   = {np.linalg.norm(solution.stance_residual):.3e}")
    print(f"||swing accel error|| = {np.linalg.norm(solution.swing_accel_error):.3e}")
    print(f"max |vdot|            = {np.max(np.abs(solution.vdot)):.3e}")
    print(f"max |tau|             = {np.max(np.abs(solution.tau)):.3e}")

    print("\n=== Stance contact forces, world frame ===")
    for name, force in zip(STANCE_FEET, forces):
        print(f"{name}: fx={force[0]:9.4f}  fy={force[1]:9.4f}  fz={force[2]:9.4f}")
    print(f"sum: fx={total_force[0]:9.4f}  fy={total_force[1]:9.4f}  fz={total_force[2]:9.4f}")
    print(f"robot weight          = {weight:.4f} N")
    print(f"sum_fz / weight       = {total_force[2] / weight:.6f}")


if __name__ == "__main__":
    main()
