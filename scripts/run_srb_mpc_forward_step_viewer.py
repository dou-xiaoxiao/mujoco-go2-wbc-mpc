"""Launch a viewer for FL forward step with SRB-MPC force references."""

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
FOOT_GEOMS = ("FL", "FR", "RL", "RR")
STANCE_FEET = ("FR", "RL", "RR")
SWING_FOOT = "FL"
SWING_START = 1.0
SWING_DURATION = 1.2
SWING_HEIGHT = 0.06
STEP_DELTA = np.array([0.05, 0.0, 0.0])
BODY_SHIFT_XY = np.array([-0.04, -0.04])
TOUCHDOWN_Z_TOL = 0.012
MPC_NORMAL_FORCE_MIN = 5.0
MPC_UPDATE_DT = 0.03

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    CentroidalMPC,
    CentroidalMPCConfig,
    current_single_leg_phase,
    MuJoCoModelInterface,
    SingleLegSwingWBCConfig,
    SingleLegSwingWBCQP,
    StanceWBCConfig,
    StanceWBCQP,
    single_leg_swing_schedule,
    swing_foothold_reference,
)


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")
    qpos_ref = robot.q
    shifted_qpos_ref = qpos_ref.copy()
    shifted_qpos_ref[0:2] += BODY_SHIFT_XY

    initial_com_ref = robot.center_of_mass()
    initial_com_ref[0:2] += BODY_SHIFT_XY
    initial_swing_pos = robot.geom_position(SWING_FOOT)
    foothold = initial_swing_pos + STEP_DELTA

    mpc_config = CentroidalMPCConfig(
        contact_geoms=FOOT_GEOMS,
        horizon_steps=10,
        dt=0.03,
        normal_force_min=MPC_NORMAL_FORCE_MIN,
    )
    mpc = CentroidalMPC(mpc_config)

    stance_controller = StanceWBCQP(
        StanceWBCConfig(
            foot_geoms=FOOT_GEOMS,
            weight_force=1.0,
        )
    )
    swing_controller = SingleLegSwingWBCQP(
        SingleLegSwingWBCConfig(
            stance_foot_geoms=STANCE_FEET,
            swing_foot_geom=SWING_FOOT,
            normal_force_min=MPC_NORMAL_FORCE_MIN,
            weight_swing_foot=1600.0,
            weight_force=1.0,
            kp_swing=500.0,
            kd_swing=45.0,
        )
    )

    dt = float(robot.model.opt.timestep)
    next_log_time = 0.0
    touchdown_time: float | None = None
    next_mpc_update = 0.0
    mpc_force_ref = np.zeros(3 * len(FOOT_GEOMS))
    mpc_stance_force_ref = np.zeros(3 * len(STANCE_FEET))
    mpc_status = "not run"
    fl_swing_knots = 0

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            sim_time = float(robot.data.time)

            foot_pos = robot.geom_position(SWING_FOOT)
            swing_is_done = sim_time >= SWING_START + SWING_DURATION
            foot_is_near_ground = foot_pos[2] <= foothold[2] + TOUCHDOWN_Z_TOL
            if touchdown_time is None and swing_is_done and foot_is_near_ground:
                touchdown_time = sim_time

            phase = current_single_leg_phase(SWING_FOOT, sim_time, SWING_START, SWING_DURATION, touchdown_time)
            in_stance_phase = phase == "stance"
            if sim_time >= next_mpc_update:
                contact_schedule = single_leg_swing_schedule(
                    FOOT_GEOMS,
                    SWING_FOOT,
                    current_time=sim_time,
                    horizon_steps=mpc_config.horizon_steps,
                    dt=mpc_config.dt,
                    swing_start=SWING_START,
                    swing_duration=SWING_DURATION,
                    touchdown_time=touchdown_time,
                )
                mpc_solution = mpc.solve(robot, initial_com_ref, contact_schedule=contact_schedule)
                mpc_force_ref = mpc_solution.first_contact_forces
                mpc_stance_force_ref = mpc_solution.contact_forces[0, 1:, :].reshape(-1)
                mpc_status = mpc_solution.status
                fl_swing_knots = int(np.count_nonzero(~contact_schedule[:, 0]))
                next_mpc_update += MPC_UPDATE_DT

            if in_stance_phase:
                target_pos = initial_swing_pos.copy()
                force_ref = mpc_force_ref
                solution = stance_controller.solve(robot, shifted_qpos_ref, force_ref=force_ref)
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
                force_ref = mpc_stance_force_ref
                solution = swing_controller.solve(
                    robot,
                    shifted_qpos_ref,
                    ref.position,
                    ref.velocity,
                    ref.acceleration,
                    force_ref=force_ref,
                )

            if solution.status in ("solved", "solved inaccurate") and mpc_status in ("solved", "solved inaccurate"):
                robot.data.ctrl[:] = solution.tau
            else:
                robot.data.ctrl[:] = 0.0

            mujoco.mj_step(robot.model, robot.data)
            viewer.sync()

            if robot.data.time >= next_log_time:
                swing_error = robot.geom_position(SWING_FOOT) - target_pos
                force_norm = float(np.linalg.norm(force_ref))
                print(
                    "t={:.2f}s phase={} swing_err={} max_tau={:.2f} |f_ref|={:.2f} mpc={} wbc={}".format(
                        robot.data.time,
                        phase,
                        np.round(swing_error, 4).tolist(),
                        float(np.max(np.abs(solution.tau))),
                        force_norm,
                        mpc_status,
                        solution.status,
                    )
                )
                print(f"  FL swing knots in MPC horizon = {fl_swing_knots}")
                next_log_time += 0.5

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
if __name__ == "__main__":
    main()
