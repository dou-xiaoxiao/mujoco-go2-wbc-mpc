"""Run a short headless MuJoCo simulation with the static stance WBC torque."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
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

    controller = StanceWBCQP(StanceWBCConfig(foot_geoms=FOOT_GEOMS))
    dt = float(robot.model.opt.timestep)
    duration = 2.0
    steps = int(duration / dt)

    max_dyn_residual = 0.0
    max_stance_residual = 0.0
    max_tau = 0.0

    for _ in range(steps):
        solution = controller.solve(robot, qpos_ref)
        if solution.status not in ("solved", "solved inaccurate"):
            raise RuntimeError(f"OSQP failed with status: {solution.status}")

        robot.data.ctrl[:] = solution.tau
        mujoco.mj_step(robot.model, robot.data)

        max_dyn_residual = max(max_dyn_residual, float(np.linalg.norm(solution.dynamics_residual)))
        max_stance_residual = max(max_stance_residual, float(np.linalg.norm(solution.stance_residual)))
        max_tau = max(max_tau, float(np.max(np.abs(solution.tau))))

    base_pos = robot.data.qpos[0:3].copy()
    base_quat = robot.data.qpos[3:7].copy()
    joint_error = robot.data.qpos[7:] - qpos_ref[7:]

    print("=== Static stance closed-loop smoke test ===")
    print(f"duration             = {duration:.3f} s")
    print(f"dt                   = {dt:.4f} s")
    print(f"steps                = {steps}")
    print(f"final base pos       = {np.round(base_pos, 5).tolist()}")
    print(f"final base quat      = {np.round(base_quat, 5).tolist()}")
    print(f"final joint err norm = {np.linalg.norm(joint_error):.3e}")
    print(f"max |tau|            = {max_tau:.3e}")
    print(f"max dyn residual     = {max_dyn_residual:.3e}")
    print(f"max stance residual  = {max_stance_residual:.3e}")


if __name__ == "__main__":
    main()
