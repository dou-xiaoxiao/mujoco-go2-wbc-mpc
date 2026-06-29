"""Launch a viewer for commanded crawl locomotion."""

from __future__ import annotations

import argparse
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
MAX_STEP_LENGTH = 0.035
COMMAND_VELOCITY_REF_SCALE = 1.0
TOUCHDOWN_Z_TOL = 0.014
TOUCHDOWN_XY_TOL = 0.025
TOUCHDOWN_MIN_PHASE = 0.80
LANDING_FORCE_RAMP_TIME = 0.15
LANDING_FORCE_ZERO_WEIGHT = 40.0
MPC_NORMAL_FORCE_MIN = 5.0
MPC_UPDATE_DT = 0.06
WBC_UPDATE_DT = 0.01
PROFILE_LOG_DT = 2.0
PRE_SHIFT_TIME = 0.6
SUPPORT_CENTROID_RATIO = 0.85

DEMO_PRESETS = {
    "forward": {"vx": 0.003, "vy": 0.0, "yaw_rate": 0.0},
    "stand-step": {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0},
    "lateral-left": {"vx": 0.0, "vy": 0.002, "yaw_rate": 0.0},
    "turn-left": {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.035},
}

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the command-driven crawl MPC/WBC demo in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--demo",
        choices=tuple(DEMO_PRESETS.keys()),
        default="forward",
        help="Named command preset. Explicit --vx/--vy/--yaw-rate values override it.",
    )
    parser.add_argument("--vx", type=float, default=None, help="Desired forward COM velocity in m/s.")
    parser.add_argument("--vy", type=float, default=None, help="Desired lateral COM velocity in m/s.")
    parser.add_argument("--yaw-rate", type=float, default=None, help="Desired yaw rate in rad/s.")
    parser.add_argument("--cycles", type=int, default=CYCLES, help="Number of FL/RR/FR/RL crawl cycles.")
    parser.add_argument("--swing-duration", type=float, default=SWING_DURATION, help="Single swing duration in seconds.")
    parser.add_argument("--swing-gap", type=float, default=SWING_GAP, help="Time between swing windows in seconds.")
    parser.add_argument("--swing-height", type=float, default=SWING_HEIGHT, help="Swing clearance in meters.")
    parser.add_argument("--max-step-length", type=float, default=MAX_STEP_LENGTH, help="Planar foothold delta limit in meters.")
    parser.add_argument("--mpc-dt", type=float, default=MPC_UPDATE_DT, help="Wall simulation period between MPC solves.")
    parser.add_argument("--wbc-dt", type=float, default=WBC_UPDATE_DT, help="Wall simulation period between WBC solves.")
    parser.add_argument("--viewer-hz", type=float, default=60.0, help="Viewer sync rate. Use 0 to sync every MuJoCo step.")
    parser.add_argument("--profile-dt", type=float, default=PROFILE_LOG_DT, help="Profiler print period in simulated seconds.")
    parser.add_argument("--no-sleep", action="store_true", help="Do not sleep to match MuJoCo real-time step.")
    return parser


def resolve_command(args: argparse.Namespace) -> CrawlCommand:
    preset = DEMO_PRESETS[args.demo]
    vx = preset["vx"] if args.vx is None else args.vx
    vy = preset["vy"] if args.vy is None else args.vy
    yaw_rate = preset["yaw_rate"] if args.yaw_rate is None else args.yaw_rate
    return CrawlCommand(vx=vx, vy=vy, yaw_rate=yaw_rate)


def validate_args(args: argparse.Namespace) -> None:
    if args.cycles <= 0:
        raise ValueError("--cycles must be positive")
    if args.swing_duration <= 0.0:
        raise ValueError("--swing-duration must be positive")
    if args.swing_gap < 0.0:
        raise ValueError("--swing-gap must be non-negative")
    if args.swing_height < 0.0:
        raise ValueError("--swing-height must be non-negative")
    if args.max_step_length <= 0.0:
        raise ValueError("--max-step-length must be positive")
    if args.mpc_dt <= 0.0:
        raise ValueError("--mpc-dt must be positive")
    if args.wbc_dt <= 0.0:
        raise ValueError("--wbc-dt must be positive")
    if args.viewer_hz < 0.0:
        raise ValueError("--viewer-hz must be non-negative")
    if args.profile_dt < 0.0:
        raise ValueError("--profile-dt must be non-negative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    command = resolve_command(args)
    viewer_sync_dt = 0.0 if args.viewer_hz <= 0.0 else 1.0 / args.viewer_hz

    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")

    home_qpos_ref = robot.q
    home_com_ref = robot.center_of_mass()
    initial_base_pos = robot.data.qpos[0:3].copy()
    nominal_body_xy = home_qpos_ref[0:2].copy()
    initial_foot_positions = {foot: robot.geom_position(foot) for foot in FOOT_GEOMS}
    repeated_sequence = CRAWL_SEQUENCE * args.cycles

    planner = CrawlGaitPlanner(
        CrawlGaitConfig(
            foot_geoms=FOOT_GEOMS,
            sequence=repeated_sequence,
            first_swing_start=SWING_START,
            swing_duration=args.swing_duration,
            swing_gap=args.swing_gap,
            swing_height=args.swing_height,
            pre_shift_time=PRE_SHIFT_TIME,
            support_centroid_ratio=SUPPORT_CENTROID_RATIO,
            command=command,
            max_step_length=args.max_step_length,
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
    next_profile_time = args.profile_dt
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
            args.cycles,
            command.vx,
            command.vy,
            command.yaw_rate,
            np.round(commanded_step_delta, 5).tolist(),
            args.mpc_dt,
            args.wbc_dt,
        )
    )
    print(
        "demo preset={}, swing_duration={:.2f}s, swing_gap={:.2f}s, swing_height={:.3f}m, viewer_hz={:.1f}".format(
            args.demo,
            args.swing_duration,
            args.swing_gap,
            args.swing_height,
            args.viewer_hz,
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
                next_mpc_update += args.mpc_dt

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
                    landing_zero_weights = landing_force_zero_weights(sim_time, touchdown_times_by_foot)
                    if current_window is None:
                        solution = stance_controller.solve(
                            robot,
                            qpos_ref,
                            force_ref=ramped_force_ref,
                            force_zero_weights=landing_zero_weights,
                            stance_pos_refs=foothold_planner.locked_positions,
                        )
                    else:
                        window_id, foot, start_time, swing_duration = current_window
                        ref = foothold_planner.swing_reference(
                            window_id,
                            swing_height=args.swing_height,
                            start_time=start_time,
                            duration=swing_duration,
                            time_s=sim_time,
                        )
                        solution = swing_controllers[foot].solve(
                            robot,
                            qpos_ref,
                            ref.position,
                            ref.velocity,
                            ref.acceleration,
                            force_ref=force_ref_for_stance_feet(ramped_force_ref, foot),
                            force_zero_weights=force_ref_for_stance_feet(landing_zero_weights, foot),
                            stance_pos_refs={stance_foot: foothold_planner.locked_positions[stance_foot] for stance_foot in FOOT_GEOMS if stance_foot != foot},
                        )

                    last_wbc_status = solution.status
                    if solution.status in ("solved", "solved inaccurate") and mpc_status in ("solved", "solved inaccurate"):
                        last_tau = solution.tau.copy()
                    else:
                        last_tau = np.zeros(robot.nu)
                    last_max_tau = float(np.max(np.abs(last_tau)))
                next_wbc_update += args.wbc_dt

            robot.data.ctrl[:] = last_tau

            with profiler.time("mj_step"):
                mujoco.mj_step(robot.model, robot.data)
            if viewer_sync_dt <= 0.0 or robot.data.time >= next_viewer_sync:
                with profiler.time("viewer"):
                    viewer.sync()
                next_viewer_sync += viewer_sync_dt

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

            if args.profile_dt > 0.0 and robot.data.time >= next_profile_time:
                summary = " | ".join(profiler.summary_lines())
                wall_now = time.perf_counter()
                sim_elapsed = float(robot.data.time) - profile_sim_start
                wall_elapsed = wall_now - profile_wall_start
                rtf = sim_elapsed / wall_elapsed if wall_elapsed > 0.0 else 0.0
                print(f"profile: sim={sim_elapsed:.2f}s wall={wall_elapsed:.2f}s rtf={rtf:.2f} | {summary}")
                profiler.reset()
                profile_wall_start = wall_now
                profile_sim_start = float(robot.data.time)
                next_profile_time += args.profile_dt

            elapsed = time.perf_counter() - step_start
            profiler.add("loop", elapsed)
            if not args.no_sleep and elapsed < dt:
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


def landing_force_zero_weights(
    time_s: float,
    touchdown_times_by_foot: dict[str, float],
) -> np.ndarray:
    weights = np.zeros((len(FOOT_GEOMS), 3), dtype=float)
    for foot_id, foot in enumerate(FOOT_GEOMS):
        touchdown_time = touchdown_times_by_foot.get(foot)
        if touchdown_time is None:
            continue
        elapsed = time_s - touchdown_time
        if elapsed >= LANDING_FORCE_RAMP_TIME:
            continue
        ratio = max(elapsed / LANDING_FORCE_RAMP_TIME, 0.0)
        release = ratio * ratio * (3.0 - 2.0 * ratio)
        weights[foot_id, :] = LANDING_FORCE_ZERO_WEIGHT * (1.0 - release)
    return weights.reshape(-1)


if __name__ == "__main__":
    main()
