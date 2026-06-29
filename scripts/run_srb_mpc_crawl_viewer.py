"""Launch a viewer for quasi-static crawl with SRB-MPC force references."""

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
CRAWL_SEQUENCE = ("FL", "RR", "FR", "RL")
SWING_START = 1.0
SWING_DURATION = 1.2
SWING_GAP = 0.8
SWING_HEIGHT = 0.035
STEP_DELTA = np.array([0.0, 0.0, 0.0])
TOUCHDOWN_Z_TOL = 0.018
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
    nominal_body_xy = home_qpos_ref[0:2].copy()
    initial_foot_positions = {foot: robot.geom_position(foot) for foot in FOOT_GEOMS}

    planner = CrawlGaitPlanner(
        CrawlGaitConfig(
            foot_geoms=FOOT_GEOMS,
            sequence=CRAWL_SEQUENCE,
            first_swing_start=SWING_START,
            swing_duration=SWING_DURATION,
            swing_gap=SWING_GAP,
            swing_height=SWING_HEIGHT,
            step_delta=STEP_DELTA,
            pre_shift_time=PRE_SHIFT_TIME,
            support_centroid_ratio=SUPPORT_CENTROID_RATIO,
        )
    )
    target_footholds = planner.target_footholds(initial_foot_positions)

    mpc_config = CentroidalMPCConfig(
        contact_geoms=FOOT_GEOMS,
        horizon_steps=12,
        dt=0.03,
        normal_force_min=MPC_NORMAL_FORCE_MIN,
    )
    mpc = CentroidalMPC(mpc_config)
    stance_controller = StanceWBCQP(
        StanceWBCConfig(
            foot_geoms=FOOT_GEOMS,
            weight_force=1.0,
            kp_stance=120.0,
            kd_stance=24.0,
        )
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
    completed_windows: set[int] = set()
    active_window_id: int | None = None
    next_window_id = 0
    locked_foot_positions = {foot: pos.copy() for foot, pos in initial_foot_positions.items()}
    touchdown_times: dict[str, float] = {}
    next_mpc_update = 0.0
    next_log_time = 0.0
    mpc_force_ref = np.zeros(3 * len(FOOT_GEOMS))
    mpc_status = "not run"
    mpc_residual = 0.0
    swing_knots = {foot: 0 for foot in FOOT_GEOMS}

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            sim_time = float(robot.data.time)

            if active_window_id is None and planner.should_start_window(sim_time, next_window_id):
                active_window_id = next_window_id

            current_window = planner.window_by_id(active_window_id)
            active_foot = current_window[1] if current_window is not None else None

            if current_window is not None:
                window_id, foot, start_time, swing_duration = current_window
                foot_pos = robot.geom_position(foot)
                swing_is_done = sim_time >= start_time + swing_duration
                foot_is_near_ground = foot_pos[2] <= target_footholds[foot][2] + TOUCHDOWN_Z_TOL
                if swing_is_done and foot_is_near_ground:
                    completed_windows.add(window_id)
                    touchdown_times[foot] = sim_time
                    locked_foot_positions[foot] = robot.geom_position(foot)
                    active_window_id = None
                    next_window_id = window_id + 1
                    current_window = None
                    active_foot = None

            if sim_time >= next_mpc_update:
                schedule = planner.contact_schedule(
                    current_time=sim_time,
                    horizon_steps=mpc_config.horizon_steps,
                    dt=mpc_config.dt,
                    completed_windows=completed_windows,
                    active_window_id=active_window_id,
                )
                refs = planner.reference_bundle(
                    home_qpos_ref,
                    home_com_ref,
                    nominal_body_xy,
                    locked_foot_positions,
                    sim_time,
                    active_window_id,
                    next_window_id,
                )
                mpc_solution = mpc.solve(
                    robot,
                    refs.com_position_ref,
                    com_velocity_ref=refs.com_velocity_ref,
                    contact_schedule=schedule,
                )
                mpc_force_ref = mpc_solution.first_contact_forces
                mpc_status = mpc_solution.status
                mpc_residual = float(np.linalg.norm(mpc_solution.dynamics_residual))
                for foot_idx, foot in enumerate(FOOT_GEOMS):
                    swing_knots[foot] = int(np.count_nonzero(~schedule[:, foot_idx]))
                next_mpc_update += MPC_UPDATE_DT

            qpos_ref = home_qpos_ref.copy()
            refs = planner.reference_bundle(
                home_qpos_ref,
                home_com_ref,
                nominal_body_xy,
                locked_foot_positions,
                sim_time,
                active_window_id,
                next_window_id,
            )
            qpos_ref[0:3] = refs.base_position_ref
            qpos_ref[3:7] = refs.base_orientation_ref

            if current_window is None:
                phase_name = "stance"
                force_ref = mpc_force_ref
                solution = stance_controller.solve(
                    robot,
                    qpos_ref,
                    force_ref=force_ref,
                    stance_pos_refs=locked_foot_positions,
                )
            else:
                _, foot, start_time, swing_duration = current_window
                phase_name = f"{foot}-swing"
                ref = planner.swing_reference(
                    foot,
                    initial_foot_positions,
                    start_time=start_time,
                    duration=swing_duration,
                    time_s=sim_time,
                )
                force_ref = force_ref_for_stance_feet(mpc_force_ref, foot)
                stance_pos_refs = {
                    stance_foot: locked_foot_positions[stance_foot]
                    for stance_foot in FOOT_GEOMS
                    if stance_foot != foot
                }
                solution = swing_controllers[foot].solve(
                    robot,
                    qpos_ref,
                    ref.position,
                    ref.velocity,
                    ref.acceleration,
                    force_ref=force_ref,
                    stance_pos_refs=stance_pos_refs,
                )

            if solution.status in ("solved", "solved inaccurate") and mpc_status in ("solved", "solved inaccurate"):
                robot.data.ctrl[:] = solution.tau
            else:
                robot.data.ctrl[:] = 0.0

            mujoco.mj_step(robot.model, robot.data)
            viewer.sync()

            if robot.data.time >= next_log_time:
                active = active_foot if active_foot is not None else "-"
                print(
                    "t={:.2f}s phase={} active={} base={} max_tau={:.2f} mpc={} wbc={} mpc_res={:.1e}".format(
                        robot.data.time,
                        phase_name,
                        active,
                        np.round(robot.data.qpos[0:3], 4).tolist(),
                        float(np.max(np.abs(solution.tau))),
                        mpc_status,
                        solution.status,
                        mpc_residual,
                    )
                )
                print(f"  swing knots={swing_knots} touchdowns={touchdown_times}")
                next_log_time += 0.5

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


def force_ref_for_stance_feet(force_ref_all: np.ndarray, swing_foot: str) -> np.ndarray:
    forces = force_ref_all.reshape(len(FOOT_GEOMS), 3)
    return np.vstack([force for foot, force in zip(FOOT_GEOMS, forces) if foot != swing_foot]).reshape(-1)


if __name__ == "__main__":
    main()
