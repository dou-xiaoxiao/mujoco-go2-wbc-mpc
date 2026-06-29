"""Run multiple commanded forward crawl cycles with SRB-MPC force references."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"

FOOT_GEOMS = ("FL", "FR", "RL", "RR")
CRAWL_SEQUENCE = ("FL", "RR", "FR", "RL")
CYCLES = 2
SWING_START = 1.0
SWING_DURATION = 1.2
SWING_GAP = 0.8
SWING_HEIGHT = 0.04
STEP_DELTA = np.array([0.024, 0.0, 0.0])
TOUCHDOWN_Z_TOL = 0.02
MPC_NORMAL_FORCE_MIN = 5.0
MPC_UPDATE_DT = 0.03
PRE_SHIFT_TIME = 0.6
SUPPORT_CENTROID_RATIO = 0.85

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    CentroidalMPC,
    CentroidalMPCConfig,
    CrawlGaitConfig,
    CrawlGaitPlanner,
    MuJoCoModelInterface,
    SingleLegSwingWBCConfig,
    SingleLegSwingWBCQP,
    StanceWBCConfig,
    StanceWBCQP,
)


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")
    home_qpos_ref = robot.q
    home_com_ref = robot.center_of_mass()
    initial_base_pos = robot.data.qpos[0:3].copy()
    nominal_body_xy = home_qpos_ref[0:2].copy()
    initial_foot_positions = {foot: robot.geom_position(foot) for foot in FOOT_GEOMS}

    repeated_sequence = CRAWL_SEQUENCE * CYCLES
    planner = CrawlGaitPlanner(
        CrawlGaitConfig(
            foot_geoms=FOOT_GEOMS,
            sequence=repeated_sequence,
            first_swing_start=SWING_START,
            swing_duration=SWING_DURATION,
            swing_gap=SWING_GAP,
            swing_height=SWING_HEIGHT,
            step_delta=STEP_DELTA,
            pre_shift_time=PRE_SHIFT_TIME,
            support_centroid_ratio=SUPPORT_CENTROID_RATIO,
        )
    )
    swing_windows = planner.swing_windows()

    mpc_config = CentroidalMPCConfig(
        contact_geoms=FOOT_GEOMS,
        horizon_steps=12,
        dt=0.03,
        normal_force_min=MPC_NORMAL_FORCE_MIN,
    )
    mpc = CentroidalMPC(mpc_config)
    stance_controller = StanceWBCQP(
        StanceWBCConfig(foot_geoms=FOOT_GEOMS, weight_force=1.0, kp_stance=120.0, kd_stance=24.0)
    )
    swing_controllers = {
        foot: SingleLegSwingWBCQP(
            SingleLegSwingWBCConfig(
                stance_foot_geoms=tuple(other for other in FOOT_GEOMS if other != foot),
                swing_foot_geom=foot,
                normal_force_min=MPC_NORMAL_FORCE_MIN,
                weight_swing_foot=1600.0,
                weight_force=1.0,
                kp_swing=500.0,
                kd_swing=45.0,
                kp_stance=120.0,
                kd_stance=24.0,
            )
        )
        for foot in FOOT_GEOMS
    }

    dt = float(robot.model.opt.timestep)
    duration = swing_windows[-1][1] + swing_windows[-1][2] + 1.0
    steps = int(duration / dt)

    completed_windows: set[int] = set()
    active_window_id: int | None = None
    next_window_id = 0
    foothold_planner = planner.rolling_foothold_planner(initial_foot_positions)
    touchdown_times: dict[int, float] = {}
    max_dyn_residual = 0.0
    max_stance_residual = 0.0
    max_mpc_residual = 0.0
    max_swing_pos_error = 0.0
    max_tau = 0.0
    failed_statuses: dict[str, int] = {}
    next_mpc_update = 0.0
    mpc_force_ref = np.zeros(3 * len(FOOT_GEOMS))
    mpc_status = "not run"
    mpc_residual = 0.0

    for _ in range(steps):
        sim_time = float(robot.data.time)
        if active_window_id is None and planner.should_start_window(sim_time, next_window_id):
            active_window_id = next_window_id
            foot = swing_windows[active_window_id][0]
            foothold_planner.start_swing(active_window_id, foot)

        current_window = planner.window_by_id(active_window_id)

        if current_window is not None:
            window_id, foot, start_time, swing_duration = current_window
            foot_pos = robot.geom_position(foot)
            swing_is_done = sim_time >= start_time + swing_duration
            foot_is_near_ground = foot_pos[2] <= foothold_planner.target_for_window(window_id)[2] + TOUCHDOWN_Z_TOL
            if swing_is_done and foot_is_near_ground:
                completed_windows.add(window_id)
                touchdown_times[window_id] = sim_time
                foothold_planner.touchdown(foot, robot.geom_position(foot))
                active_window_id = None
                next_window_id = window_id + 1
                current_window = None

        body_xy_ref = planner.body_xy_reference(
            nominal_body_xy,
            foothold_planner.locked_positions,
            sim_time,
            active_window_id,
            next_window_id,
        )

        if sim_time >= next_mpc_update:
            schedule = planner.contact_schedule(
                current_time=sim_time,
                horizon_steps=mpc_config.horizon_steps,
                dt=mpc_config.dt,
                completed_windows=completed_windows,
                active_window_id=active_window_id,
            )
            com_ref = home_com_ref.copy()
            com_ref[0:2] += body_xy_ref - nominal_body_xy
            mpc_solution = mpc.solve(robot, com_ref, contact_schedule=schedule)
            mpc_force_ref = mpc_solution.first_contact_forces
            mpc_status = mpc_solution.status
            mpc_residual = float(np.linalg.norm(mpc_solution.dynamics_residual))
            next_mpc_update += MPC_UPDATE_DT

        qpos_ref = home_qpos_ref.copy()
        qpos_ref[0:2] = body_xy_ref

        if current_window is None:
            solution = stance_controller.solve(
                robot,
                qpos_ref,
                force_ref=mpc_force_ref,
                stance_pos_refs=foothold_planner.locked_positions,
            )
            target_pos = None
        else:
            window_id, foot, start_time, swing_duration = current_window
            ref = foothold_planner.swing_reference(
                window_id,
                swing_height=SWING_HEIGHT,
                start_time=start_time,
                duration=swing_duration,
                time_s=sim_time,
            )
            target_pos = ref.position
            solution = swing_controllers[foot].solve(
                robot,
                qpos_ref,
                ref.position,
                ref.velocity,
                ref.acceleration,
                force_ref=force_ref_for_stance_feet(mpc_force_ref, foot),
                stance_pos_refs={stance_foot: foothold_planner.locked_positions[stance_foot] for stance_foot in FOOT_GEOMS if stance_foot != foot},
            )

        if solution.status not in ("solved", "solved inaccurate") or mpc_status not in ("solved", "solved inaccurate"):
            key = f"mpc={mpc_status}, wbc={solution.status}"
            failed_statuses[key] = failed_statuses.get(key, 0) + 1
            robot.data.ctrl[:] = 0.0
        else:
            robot.data.ctrl[:] = solution.tau

        mujoco.mj_step(robot.model, robot.data)

        if current_window is not None and target_pos is not None:
            swing_error = robot.geom_position(current_window[1]) - target_pos
            max_swing_pos_error = max(max_swing_pos_error, float(np.linalg.norm(swing_error)))
        max_dyn_residual = max(max_dyn_residual, float(np.linalg.norm(solution.dynamics_residual)))
        max_stance_residual = max(max_stance_residual, float(np.linalg.norm(solution.stance_residual)))
        max_mpc_residual = max(max_mpc_residual, mpc_residual)
        max_tau = max(max_tau, float(np.max(np.abs(solution.tau))))

    final_base_pos = robot.data.qpos[0:3].copy()

    print("=== SRB-MPC + WBC continuous forward crawl smoke test ===")
    print(f"cycles                = {CYCLES}")
    print(f"sequence              = {repeated_sequence}")
    print(f"step delta            = {STEP_DELTA.tolist()} m")
    print(f"duration              = {duration:.3f} s")
    print(f"touchdowns completed  = {len(touchdown_times)} / {len(swing_windows)}")
    print(f"initial base pos      = {np.round(initial_base_pos, 5).tolist()}")
    print(f"final base pos        = {np.round(final_base_pos, 5).tolist()}")
    print(f"base displacement     = {np.round(final_base_pos - initial_base_pos, 5).tolist()}")
    for foot in FOOT_GEOMS:
        print(f"{foot}: final={np.round(robot.geom_position(foot), 5).tolist()}")
    print(f"max swing pos error   = {max_swing_pos_error:.3e} m")
    print(f"max |tau|             = {max_tau:.3e} Nm")
    print(f"max WBC dyn residual  = {max_dyn_residual:.3e}")
    print(f"max stance residual   = {max_stance_residual:.3e}")
    print(f"max MPC residual      = {max_mpc_residual:.3e}")
    print(f"failed statuses       = {failed_statuses}")


def force_ref_for_stance_feet(force_ref_all: np.ndarray, swing_foot: str) -> np.ndarray:
    forces = force_ref_all.reshape(len(FOOT_GEOMS), 3)
    return np.vstack([force for foot, force in zip(FOOT_GEOMS, forces) if foot != swing_foot]).reshape(-1)


if __name__ == "__main__":
    main()
