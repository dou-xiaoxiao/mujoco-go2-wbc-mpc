"""Launch a viewer for FL swing-in-place with three stance feet."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
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
    phase_time = (time_s - SWING_START) % SWING_DURATION
    if time_s >= SWING_START:
        phase = phase_time / SWING_DURATION
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
    next_log_time = 0.0

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            sim_time = float(robot.data.time)
            if sim_time < SWING_START:
                target_pos = initial_swing_pos.copy()
                solution = stance_controller.solve(robot, shifted_qpos_ref)
            else:
                target_pos, target_vel, target_acc = swing_reference(initial_swing_pos, sim_time)
                solution = swing_controller.solve(robot, shifted_qpos_ref, target_pos, target_vel, target_acc)
            if solution.status in ("solved", "solved inaccurate"):
                robot.data.ctrl[:] = solution.tau
            else:
                robot.data.ctrl[:] = 0.0

            mujoco.mj_step(robot.model, robot.data)
            viewer.sync()

            if robot.data.time >= next_log_time:
                swing_error = robot.geom_position(SWING_FOOT) - target_pos
                print(
                    "t={:.2f}s swing_err={} max_tau={:.2f} status={}".format(
                        robot.data.time,
                        np.round(swing_error, 4).tolist(),
                        float(np.max(np.abs(solution.tau))),
                        solution.status,
                    )
                )
                next_log_time += 0.5

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
