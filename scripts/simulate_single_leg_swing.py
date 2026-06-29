"""Run a headless FL swing-in-place test with three stance feet."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
STANCE_FEET = ("FR", "RL", "RR")
SWING_FOOT = "FL"
SWING_START = 1.0
SWING_DURATION = 1.2
SWING_HEIGHT = 0.06
BODY_SHIFT_XY = np.array([-0.04, -0.04])

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    MuJoCoModelInterface,
    SingleLegSwingWBCConfig,
    SingleLegSwingWBCQP,
    StanceWBCConfig,
    StanceWBCQP,
)


def swing_reference(initial_pos: np.ndarray, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = initial_pos.copy()
    vel = np.zeros(3)
    acc = np.zeros(3)
    if SWING_START <= time_s < SWING_START + SWING_DURATION:
        phase = (time_s - SWING_START) / SWING_DURATION
        angle = np.pi * phase
        pos[2] += SWING_HEIGHT * np.sin(angle)
        vel[2] = SWING_HEIGHT * np.pi / SWING_DURATION * np.cos(angle)
        acc[2] = -SWING_HEIGHT * (np.pi / SWING_DURATION) ** 2 * np.sin(angle)
    return pos, vel, acc


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")
    qpos_ref = robot.q
    shifted_qpos_ref = qpos_ref.copy()
    shifted_qpos_ref[0:2] += BODY_SHIFT_XY
    initial_swing_pos = robot.geom_position(SWING_FOOT)

    stance_controller = StanceWBCQP(StanceWBCConfig(foot_geoms=("FL", "FR", "RL", "RR")))
    swing_controller = SingleLegSwingWBCQP(
        SingleLegSwingWBCConfig(stance_foot_geoms=STANCE_FEET, swing_foot_geom=SWING_FOOT)
    )
    dt = float(robot.model.opt.timestep)
    duration = 2.5
    steps = int(duration / dt)

    max_dyn_residual = 0.0
    max_stance_residual = 0.0
    max_swing_pos_error = 0.0
    max_tau = 0.0
    failed_statuses: dict[str, int] = {}

    for _ in range(steps):
        sim_time = float(robot.data.time)
        if sim_time < SWING_START:
            target_pos = initial_swing_pos.copy()
            solution = stance_controller.solve(robot, shifted_qpos_ref)
        else:
            target_pos, target_vel, target_acc = swing_reference(initial_swing_pos, sim_time)
            solution = swing_controller.solve(robot, shifted_qpos_ref, target_pos, target_vel, target_acc)
        if solution.status not in ("solved", "solved inaccurate"):
            failed_statuses[solution.status] = failed_statuses.get(solution.status, 0) + 1
            robot.data.ctrl[:] = 0.0
        else:
            robot.data.ctrl[:] = solution.tau

        mujoco.mj_step(robot.model, robot.data)

        swing_error = robot.geom_position(SWING_FOOT) - target_pos
        max_swing_pos_error = max(max_swing_pos_error, float(np.linalg.norm(swing_error)))
        max_dyn_residual = max(max_dyn_residual, float(np.linalg.norm(solution.dynamics_residual)))
        max_stance_residual = max(max_stance_residual, float(np.linalg.norm(solution.stance_residual)))
        max_tau = max(max_tau, float(np.max(np.abs(solution.tau))))

    final_swing_pos = robot.geom_position(SWING_FOOT)
    final_base_pos = robot.data.qpos[0:3].copy()

    print("=== Single-leg swing closed-loop smoke test ===")
    print(f"duration              = {duration:.3f} s")
    print(f"stance feet           = {STANCE_FEET}")
    print(f"swing foot            = {SWING_FOOT}")
    print(f"body shift xy         = {BODY_SHIFT_XY.tolist()} m")
    print(f"final base pos        = {np.round(final_base_pos, 5).tolist()}")
    print(f"final swing pos       = {np.round(final_swing_pos, 5).tolist()}")
    print(f"initial swing pos     = {np.round(initial_swing_pos, 5).tolist()}")
    print(f"max swing pos error   = {max_swing_pos_error:.3e} m")
    print(f"max |tau|             = {max_tau:.3e} Nm")
    print(f"max dyn residual      = {max_dyn_residual:.3e}")
    print(f"max stance residual   = {max_stance_residual:.3e}")
    print(f"failed QP statuses    = {failed_statuses}")


if __name__ == "__main__":
    main()
