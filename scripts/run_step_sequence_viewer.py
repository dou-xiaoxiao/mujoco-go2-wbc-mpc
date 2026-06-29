"""Launch a viewer for a slow four-leg step sequence with WBC."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
ALL_FEET = ("FL", "FR", "RL", "RR")
LEG_SEQUENCE = ("FL", "RR", "FR", "RL")
PRE_SHIFT_DURATION = 1.0
SWING_DURATION = 1.2
HOLD_DURATION = 0.8
SWING_HEIGHT = 0.05
STEP_DELTA = np.array([0.03, 0.0, 0.0])
TOUCHDOWN_Z_TOL = 0.012
BODY_SHIFT_MAG = 0.04
BODY_SHIFT_BY_LEG = {
    "FL": np.array([-BODY_SHIFT_MAG, -BODY_SHIFT_MAG]),
    "RR": np.array([BODY_SHIFT_MAG, BODY_SHIFT_MAG]),
    "FR": np.array([-BODY_SHIFT_MAG, BODY_SHIFT_MAG]),
    "RL": np.array([BODY_SHIFT_MAG, -BODY_SHIFT_MAG]),
}

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    MuJoCoModelInterface,
    SingleLegSwingWBCConfig,
    SingleLegSwingWBCQP,
    StanceWBCConfig,
    StanceWBCQP,
    swing_foothold_reference,
)


@dataclass
class StepPhase:
    leg: str
    initial_foot_pos: np.ndarray
    foothold: np.ndarray
    start_time: float


def stance_feet_for(swing_leg: str) -> tuple[str, ...]:
    return tuple(foot for foot in ALL_FEET if foot != swing_leg)


def body_shift_for_leg(robot: MuJoCoModelInterface, leg: str) -> np.ndarray:
    del robot
    return BODY_SHIFT_BY_LEG[leg]


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")
    qpos_home = robot.q

    stance_controller = StanceWBCQP(StanceWBCConfig(foot_geoms=ALL_FEET))
    swing_controllers = {
        leg: SingleLegSwingWBCQP(
            SingleLegSwingWBCConfig(
                stance_foot_geoms=stance_feet_for(leg),
                swing_foot_geom=leg,
                weight_swing_foot=1400.0,
                kp_swing=450.0,
                kd_swing=42.0,
            )
        )
        for leg in ALL_FEET
    }

    dt = float(robot.model.opt.timestep)
    active_phase: StepPhase | None = None
    next_log_time = 0.0
    nominal_base_xy = qpos_home[0:2].copy()
    leg_index = 0
    mode = "pre-shift"
    mode_start_time = 0.0

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            sim_time = float(robot.data.time)
            leg = LEG_SEQUENCE[leg_index % len(LEG_SEQUENCE)]
            mode_time = sim_time - mode_start_time

            qpos_ref = qpos_home.copy()
            qpos_ref[0:2] = nominal_base_xy + body_shift_for_leg(robot, leg)

            if mode == "pre-shift":
                active_phase = None
                phase_name = "pre-shift"
                solution = stance_controller.solve(robot, qpos_ref)
                if mode_time >= PRE_SHIFT_DURATION:
                    initial_pos = robot.geom_position(leg)
                    active_phase = StepPhase(
                        leg=leg,
                        initial_foot_pos=initial_pos,
                        foothold=initial_pos + STEP_DELTA,
                        start_time=sim_time,
                    )
                    mode = "swing"
                    mode_start_time = sim_time

            elif mode == "swing":
                if active_phase is None or active_phase.leg != leg:
                    initial_pos = robot.geom_position(leg)
                    active_phase = StepPhase(
                        leg=leg,
                        initial_foot_pos=initial_pos,
                        foothold=initial_pos + STEP_DELTA,
                        start_time=sim_time,
                    )

                ref = swing_foothold_reference(
                    initial_position=active_phase.initial_foot_pos,
                    step_delta=STEP_DELTA,
                    swing_height=SWING_HEIGHT,
                    start_time=active_phase.start_time,
                    duration=SWING_DURATION,
                    time_s=sim_time,
                )
                phase_name = "swing"
                solution = swing_controllers[leg].solve(robot, qpos_ref, ref.position, ref.velocity, ref.acceleration)
                swing_is_done = sim_time >= active_phase.start_time + SWING_DURATION
                foot_is_near_ground = robot.geom_position(leg)[2] <= active_phase.foothold[2] + TOUCHDOWN_Z_TOL
                if swing_is_done and foot_is_near_ground:
                    active_phase = None
                    mode = "hold"
                    mode_start_time = sim_time
            else:
                active_phase = None
                phase_name = "hold"
                solution = stance_controller.solve(robot, qpos_ref)
                if mode_time >= HOLD_DURATION:
                    leg_index += 1
                    nominal_base_xy = robot.data.qpos[0:2].copy()
                    mode = "pre-shift"
                    mode_start_time = sim_time

            if solution.status in ("solved", "solved inaccurate"):
                robot.data.ctrl[:] = solution.tau
            else:
                robot.data.ctrl[:] = 0.0

            mujoco.mj_step(robot.model, robot.data)
            viewer.sync()

            if robot.data.time >= next_log_time:
                foot_pos = robot.geom_position(leg)
                print(
                    "t={:.2f}s leg={} phase={} foot={} base={} max_tau={:.2f} status={}".format(
                        robot.data.time,
                        leg,
                        phase_name,
                        np.round(foot_pos, 4).tolist(),
                        np.round(robot.data.qpos[0:3], 4).tolist(),
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
