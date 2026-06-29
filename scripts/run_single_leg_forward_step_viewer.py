"""Launch a viewer for an FL forward step with three stance feet."""

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
STEP_DELTA = np.array([0.05, 0.0, 0.0])
BODY_SHIFT_XY = np.array([-0.04, -0.04])
TOUCHDOWN_Z_TOL = 0.012

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    MuJoCoModelInterface,
    SingleLegSwingWBCConfig,
    SingleLegSwingWBCQP,
    StanceWBCConfig,
    StanceWBCQP,
    swing_foothold_reference,
)


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")
    qpos_ref = robot.q
    shifted_qpos_ref = qpos_ref.copy()
    shifted_qpos_ref[0:2] += BODY_SHIFT_XY

    initial_swing_pos = robot.geom_position(SWING_FOOT)
    foothold = initial_swing_pos + STEP_DELTA

    stance_controller = StanceWBCQP(StanceWBCConfig(foot_geoms=("FL", "FR", "RL", "RR")))
    swing_controller = SingleLegSwingWBCQP(
        SingleLegSwingWBCConfig(
            stance_foot_geoms=STANCE_FEET,
            swing_foot_geom=SWING_FOOT,
            weight_swing_foot=1600.0,
            kp_swing=500.0,
            kd_swing=45.0,
        )
    )

    dt = float(robot.model.opt.timestep)
    next_log_time = 0.0
    touchdown_time: float | None = None

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            sim_time = float(robot.data.time)

            foot_pos = robot.geom_position(SWING_FOOT)
            swing_is_done = sim_time >= SWING_START + SWING_DURATION
            foot_is_near_ground = foot_pos[2] <= foothold[2] + TOUCHDOWN_Z_TOL
            if touchdown_time is None and swing_is_done and foot_is_near_ground:
                touchdown_time = sim_time

            if sim_time < SWING_START or touchdown_time is not None:
                target_pos = initial_swing_pos.copy()
                solution = stance_controller.solve(robot, shifted_qpos_ref)
                phase = "stance"
            else:
                ref = swing_foothold_reference(
                    initial_position=initial_swing_pos,
                    step_delta=STEP_DELTA,
                    swing_height=SWING_HEIGHT,
                    start_time=SWING_START,
                    duration=SWING_DURATION,
                    time_s=sim_time,
                )
                target_pos = ref.position
                solution = swing_controller.solve(robot, shifted_qpos_ref, ref.position, ref.velocity, ref.acceleration)
                phase = "swing"

            if solution.status in ("solved", "solved inaccurate"):
                robot.data.ctrl[:] = solution.tau
            else:
                robot.data.ctrl[:] = 0.0

            mujoco.mj_step(robot.model, robot.data)
            viewer.sync()

            if robot.data.time >= next_log_time:
                swing_error = robot.geom_position(SWING_FOOT) - target_pos
                foothold_error = robot.geom_position(SWING_FOOT) - foothold
                print(
                    "t={:.2f}s phase={} swing_err={} foothold_err={} max_tau={:.2f} status={}".format(
                        robot.data.time,
                        phase,
                        np.round(swing_error, 4).tolist(),
                        np.round(foothold_error, 4).tolist(),
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
