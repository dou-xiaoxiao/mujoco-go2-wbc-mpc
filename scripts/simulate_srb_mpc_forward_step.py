"""Run a headless FL forward-step test with SRB-MPC force references."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
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
    stance_controller = StanceWBCQP(StanceWBCConfig(foot_geoms=FOOT_GEOMS, weight_force=1.0))
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
    duration = 4.0
    steps = int(duration / dt)

    touchdown_time: float | None = None
    max_dyn_residual = 0.0
    max_stance_residual = 0.0
    max_mpc_residual = 0.0
    max_swing_pos_error = 0.0
    max_tau = 0.0
    failed_statuses: dict[str, int] = {}
    next_mpc_update = 0.0
    mpc_force_ref = np.zeros(3 * len(FOOT_GEOMS))
    mpc_stance_force_ref = np.zeros(3 * len(STANCE_FEET))
    mpc_status = "not run"
    mpc_residual = 0.0
    max_future_swing_knots = 0

    for _ in range(steps):
        sim_time = float(robot.data.time)
        foot_pos = robot.geom_position(SWING_FOOT)
        swing_is_done = sim_time >= SWING_START + SWING_DURATION
        foot_is_near_ground = foot_pos[2] <= foothold[2] + TOUCHDOWN_Z_TOL
        if touchdown_time is None and swing_is_done and foot_is_near_ground:
            touchdown_time = sim_time

        phase = current_single_leg_phase(SWING_FOOT, sim_time, SWING_START, SWING_DURATION, touchdown_time)
        in_stance_phase = phase == "stance"
        if sim_time >= next_mpc_update:
            schedule = single_leg_swing_schedule(
                FOOT_GEOMS,
                SWING_FOOT,
                current_time=sim_time,
                horizon_steps=mpc_config.horizon_steps,
                dt=mpc_config.dt,
                swing_start=SWING_START,
                swing_duration=SWING_DURATION,
                touchdown_time=touchdown_time,
            )
            mpc_solution = mpc.solve(robot, initial_com_ref, contact_schedule=schedule)
            mpc_force_ref = mpc_solution.first_contact_forces
            mpc_stance_force_ref = mpc_solution.contact_forces[0, 1:, :].reshape(-1)
            mpc_status = mpc_solution.status
            mpc_residual = float(np.linalg.norm(mpc_solution.dynamics_residual))
            max_future_swing_knots = max(max_future_swing_knots, int(np.count_nonzero(~schedule[:, 0])))
            next_mpc_update += MPC_UPDATE_DT

        if in_stance_phase:
            target_pos = initial_swing_pos.copy()
            solution = stance_controller.solve(robot, shifted_qpos_ref, force_ref=mpc_force_ref)
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
            solution = swing_controller.solve(
                robot,
                shifted_qpos_ref,
                ref.position,
                ref.velocity,
                ref.acceleration,
                force_ref=mpc_stance_force_ref,
            )

        if solution.status not in ("solved", "solved inaccurate") or mpc_status not in ("solved", "solved inaccurate"):
            key = f"mpc={mpc_status}, wbc={solution.status}"
            failed_statuses[key] = failed_statuses.get(key, 0) + 1
            robot.data.ctrl[:] = 0.0
        else:
            robot.data.ctrl[:] = solution.tau

        mujoco.mj_step(robot.model, robot.data)

        if touchdown_time is None and sim_time >= SWING_START:
            swing_error = robot.geom_position(SWING_FOOT) - target_pos
            max_swing_pos_error = max(max_swing_pos_error, float(np.linalg.norm(swing_error)))
        max_dyn_residual = max(max_dyn_residual, float(np.linalg.norm(solution.dynamics_residual)))
        max_stance_residual = max(max_stance_residual, float(np.linalg.norm(solution.stance_residual)))
        max_mpc_residual = max(max_mpc_residual, mpc_residual)
        max_tau = max(max_tau, float(np.max(np.abs(solution.tau))))

    final_swing_pos = robot.geom_position(SWING_FOOT)
    final_base_pos = robot.data.qpos[0:3].copy()

    print("=== SRB-MPC + WBC forward-step smoke test ===")
    print(f"duration              = {duration:.3f} s")
    print(f"MPC update dt         = {MPC_UPDATE_DT:.3f} s")
    print(f"max FL swing knots in horizon = {max_future_swing_knots}")
    print(f"touchdown time        = {touchdown_time}")
    print(f"initial swing pos     = {np.round(initial_swing_pos, 5).tolist()}")
    print(f"target foothold       = {np.round(foothold, 5).tolist()}")
    print(f"final swing pos       = {np.round(final_swing_pos, 5).tolist()}")
    print(f"final foothold error  = {np.round(final_swing_pos - foothold, 5).tolist()}")
    print(f"final base pos        = {np.round(final_base_pos, 5).tolist()}")
    print(f"max swing pos error   = {max_swing_pos_error:.3e} m")
    print(f"max |tau|             = {max_tau:.3e} Nm")
    print(f"max WBC dyn residual  = {max_dyn_residual:.3e}")
    print(f"max stance residual   = {max_stance_residual:.3e}")
    print(f"max MPC residual      = {max_mpc_residual:.3e}")
    print(f"failed statuses       = {failed_statuses}")
if __name__ == "__main__":
    main()
