"""Launch a viewer for continuous commanded forward crawl."""

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
CYCLES = 6
SWING_START = 1.0
SWING_DURATION = 1.2
SWING_GAP = 0.8
SWING_HEIGHT = 0.04
COMMAND_VX = 0.003
COMMAND_VY = 0.0
COMMAND_YAW_RATE = 0.0
MAX_STEP_LENGTH = 0.035
COMMAND_VELOCITY_REF_SCALE = 1.0
TOUCHDOWN_Z_TOL = 0.02
TOUCHDOWN_XY_TOL = 0.025
TOUCHDOWN_MIN_PHASE = 0.80
SOFT_LANDING_START_PHASE = 0.95
SOFT_LANDING_MAX_DESCENT_RATE = 0.25
SOFT_LANDING_STOPPING_ACCEL = 1.0
LANDING_FORCE_RAMP_TIME = 0.15
MPC_NORMAL_FORCE_MIN = 5.0
MPC_UPDATE_DT = 0.06
WBC_UPDATE_DT = 0.01
VIEWER_SYNC_DT = 1.0 / 60.0
PROFILE_LOG_DT = 2.0
PRE_SHIFT_TIME = 0.6
SUPPORT_CENTROID_RATIO = 0.85

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    CentroidalMPC,
    CentroidalMPCConfig,
    CrawlCommand,
    CrawlGaitConfig,
    CrawlGaitPlanner,
    LoopProfiler,
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
            pre_shift_time=PRE_SHIFT_TIME,
            support_centroid_ratio=SUPPORT_CENTROID_RATIO,
            command=CrawlCommand(vx=COMMAND_VX, vy=COMMAND_VY, yaw_rate=COMMAND_YAW_RATE),
            max_step_length=MAX_STEP_LENGTH,
            command_velocity_ref_scale=COMMAND_VELOCITY_REF_SCALE,
        )
    )
    swing_windows = planner.swing_windows()
    commanded_step_delta = planner.step_delta()

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
    completed_windows: set[int] = set()
    active_window_id: int | None = None
    next_window_id = 0
    foothold_planner = planner.rolling_foothold_planner(initial_foot_positions)
    touchdown_times: dict[int, float] = {}
    touchdown_times_by_foot: dict[str, float] = {}
    next_mpc_update = 0.0
    next_wbc_update = 0.0
    next_viewer_sync = 0.0
    next_log_time = 0.0
    next_profile_time = PROFILE_LOG_DT
    mpc_force_ref = np.zeros(3 * len(FOOT_GEOMS))
    mpc_status = "not run"
    mpc_residual = 0.0
    last_tau = np.zeros(robot.nu)
    last_wbc_status = "not run"
    last_max_tau = 0.0
    last_phase_key: int | None = None
    profiler = LoopProfiler()
    profile_wall_start = time.perf_counter()
    profile_sim_start = 0.0

    print(
        "commanded crawl: cycles={}, command=[{:.4f}, {:.4f}, {:.4f}], step_delta={} m, mpc_dt={:.3f}s, wbc_dt={:.3f}s".format(
            CYCLES,
            COMMAND_VX,
            COMMAND_VY,
            COMMAND_YAW_RATE,
            np.round(commanded_step_delta, 5).tolist(),
            MPC_UPDATE_DT,
            WBC_UPDATE_DT,
        )
    )

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            sim_time = float(robot.data.time)

            with profiler.time("schedule"):
                if active_window_id is None and planner.should_start_window(sim_time, next_window_id):
                    active_window_id = next_window_id
                    foot = swing_windows[active_window_id][0]
                    foothold_planner.start_swing(active_window_id, foot)

                current_window = planner.window_by_id(active_window_id)
                active_foot = current_window[1] if current_window is not None else None

                if current_window is not None:
                    window_id, foot, start_time, swing_duration = current_window
                    foot_pos = robot.geom_position(foot)
                    target_pos = foothold_planner.target_for_window(window_id)
                    swing_phase = (sim_time - start_time) / swing_duration
                    swing_is_done = sim_time >= start_time + swing_duration
                    foot_is_near_ground = foot_pos[2] <= target_pos[2] + TOUCHDOWN_Z_TOL
                    foot_is_near_target_xy = np.linalg.norm(foot_pos[0:2] - target_pos[0:2]) <= TOUCHDOWN_XY_TOL
                    foot_has_contact = robot.geom_has_contact(foot)
                    touchdown_allowed = swing_phase >= TOUCHDOWN_MIN_PHASE or swing_is_done
                    touchdown_detected = foot_has_contact or swing_is_done
                    if touchdown_allowed and foot_is_near_ground and foot_is_near_target_xy and touchdown_detected:
                        completed_windows.add(window_id)
                        touchdown_times[window_id] = sim_time
                        touchdown_times_by_foot[foot] = sim_time
                        foothold_planner.touchdown(foot, robot.geom_position(foot))
                        active_window_id = None
                        next_window_id = window_id + 1
                        current_window = None
                        active_foot = None

                refs = planner.reference_bundle(
                    home_qpos_ref,
                    home_com_ref,
                    nominal_body_xy,
                    foothold_planner.locked_positions,
                    sim_time,
                    active_window_id,
                    next_window_id,
                )

            if sim_time >= next_mpc_update:
                with profiler.time("mpc"):
                    schedule = planner.contact_schedule(
                        current_time=sim_time,
                        horizon_steps=mpc_config.horizon_steps,
                        dt=mpc_config.dt,
                        completed_windows=completed_windows,
                        active_window_id=active_window_id,
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
                next_mpc_update += MPC_UPDATE_DT

            qpos_ref = home_qpos_ref.copy()
            qpos_ref[0:3] = refs.base_position_ref
            qpos_ref[3:7] = refs.base_orientation_ref

            phase_key = active_window_id
            if phase_key != last_phase_key:
                next_wbc_update = sim_time
                last_phase_key = phase_key
            phase_name = "stance" if current_window is None else f"{current_window[1]}-swing"

            if sim_time >= next_wbc_update:
                with profiler.time("wbc"):
                    ramped_force_ref = landing_ramped_force_ref(mpc_force_ref, sim_time, touchdown_times_by_foot)
                    if current_window is None:
                        solution = stance_controller.solve(
                            robot,
                            qpos_ref,
                            force_ref=ramped_force_ref,
                            stance_pos_refs=foothold_planner.locked_positions,
                        )
                    else:
                        window_id, foot, start_time, swing_duration = current_window
                        ref = foothold_planner.swing_reference(
                            window_id,
                            swing_height=SWING_HEIGHT,
                            start_time=start_time,
                            duration=swing_duration,
                            time_s=sim_time,
                        )
                        swing_phase = (sim_time - start_time) / swing_duration
                        swing_pos_ref, swing_vel_ref, swing_acc_ref = soft_landing_reference(
                            ref.position,
                            ref.velocity,
                            ref.acceleration,
                            robot.geom_position(foot),
                            foothold_planner.target_for_window(window_id),
                            swing_phase,
                            WBC_UPDATE_DT,
                        )
                        solution = swing_controllers[foot].solve(
                            robot,
                            qpos_ref,
                            swing_pos_ref,
                            swing_vel_ref,
                            swing_acc_ref,
                            force_ref=force_ref_for_stance_feet(ramped_force_ref, foot),
                            stance_pos_refs={stance_foot: foothold_planner.locked_positions[stance_foot] for stance_foot in FOOT_GEOMS if stance_foot != foot},
                        )

                    last_wbc_status = solution.status
                    if solution.status in ("solved", "solved inaccurate") and mpc_status in ("solved", "solved inaccurate"):
                        last_tau = solution.tau.copy()
                    else:
                        last_tau = np.zeros(robot.nu)
                    last_max_tau = float(np.max(np.abs(last_tau)))
                next_wbc_update += WBC_UPDATE_DT

            robot.data.ctrl[:] = last_tau

            with profiler.time("mj_step"):
                mujoco.mj_step(robot.model, robot.data)
            if robot.data.time >= next_viewer_sync:
                with profiler.time("viewer"):
                    viewer.sync()
                next_viewer_sync += VIEWER_SYNC_DT

            if robot.data.time >= next_log_time:
                print(
                    "t={:.2f}s phase={} active={} step={}/{} base={} disp={} body_ref={} touchdowns={} max_tau={:.2f} mpc={} wbc={} mpc_res={:.1e}".format(
                        robot.data.time,
                        phase_name,
                        active_foot if active_foot is not None else "-",
                        min(next_window_id, len(swing_windows)),
                        len(swing_windows),
                        np.round(robot.data.qpos[0:3], 4).tolist(),
                        np.round(robot.data.qpos[0:3] - initial_base_pos, 4).tolist(),
                        np.round(refs.base_position_ref[0:2], 4).tolist(),
                        len(touchdown_times),
                        last_max_tau,
                        mpc_status,
                        last_wbc_status,
                        mpc_residual,
                    )
                )
                next_log_time += 0.5

            if robot.data.time >= next_profile_time:
                summary = " | ".join(profiler.summary_lines())
                wall_now = time.perf_counter()
                sim_elapsed = float(robot.data.time) - profile_sim_start
                wall_elapsed = wall_now - profile_wall_start
                rtf = sim_elapsed / wall_elapsed if wall_elapsed > 0.0 else 0.0
                print(f"profile: sim={sim_elapsed:.2f}s wall={wall_elapsed:.2f}s rtf={rtf:.2f} | {summary}")
                profiler.reset()
                profile_wall_start = wall_now
                profile_sim_start = float(robot.data.time)
                next_profile_time += PROFILE_LOG_DT

            elapsed = time.perf_counter() - step_start
            profiler.add("loop", elapsed)
            if elapsed < dt:
                with profiler.time("sleep"):
                    time.sleep(dt - elapsed)


def force_ref_for_stance_feet(force_ref_all: np.ndarray, swing_foot: str) -> np.ndarray:
    forces = force_ref_all.reshape(len(FOOT_GEOMS), 3)
    return np.vstack([force for foot, force in zip(FOOT_GEOMS, forces) if foot != swing_foot]).reshape(-1)


def landing_ramped_force_ref(
    force_ref_all: np.ndarray,
    time_s: float,
    touchdown_times_by_foot: dict[str, float],
) -> np.ndarray:
    forces = force_ref_all.reshape(len(FOOT_GEOMS), 3).copy()
    for foot_id, foot in enumerate(FOOT_GEOMS):
        touchdown_time = touchdown_times_by_foot.get(foot)
        if touchdown_time is None:
            continue
        elapsed = time_s - touchdown_time
        if elapsed >= LANDING_FORCE_RAMP_TIME:
            continue
        ratio = max(elapsed / LANDING_FORCE_RAMP_TIME, 0.0)
        ramp = ratio * ratio * (3.0 - 2.0 * ratio)
        forces[foot_id] *= ramp
    return forces.reshape(-1)


def soft_landing_reference(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    current_foot_position: np.ndarray,
    target_foot_position: np.ndarray,
    swing_phase: float,
    control_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = position.copy()
    vel = velocity.copy()
    acc = acceleration.copy()
    if swing_phase < SOFT_LANDING_START_PHASE:
        return pos, vel, acc

    distance_to_ground = max(float(current_foot_position[2] - target_foot_position[2]), 0.0)
    stopping_speed = np.sqrt(2.0 * SOFT_LANDING_STOPPING_ACCEL * distance_to_ground)
    descent_rate_limit = min(SOFT_LANDING_MAX_DESCENT_RATE, stopping_speed)
    max_drop = descent_rate_limit * control_dt
    pos[2] = max(pos[2], current_foot_position[2] - max_drop)
    vel[2] = max(vel[2], -descent_rate_limit)
    acc[2] = max(acc[2], 0.0)
    return pos, vel, acc


if __name__ == "__main__":
    main()
