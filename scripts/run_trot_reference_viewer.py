"""Run a simple upstream-provided trot reference through MPC + generic WBC."""

from __future__ import annotations

import argparse
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

FOOT_GEOMS = ("FL", "FR", "RL", "RR")
TROT_PAIRS = (("FL", "RR"), ("FR", "RL"))
MPC_UPDATE_DT = 0.08
WBC_UPDATE_DT = 0.02
VIEWER_SYNC_DT = 1.0 / 60.0
PROFILE_LOG_DT = 2.0
MPC_NORMAL_FORCE_MIN = 5.0

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    CentroidalMPC,
    CentroidalMPCConfig,
    GeneralContactWBCConfig,
    GeneralContactWBCQP,
    landing_force_zero_weights,
    landing_ramped_force_ref,
    LoopProfiler,
    MuJoCoModelInterface,
    StanceWBCConfig,
    StanceWBCQP,
    swing_foothold_reference,
    update_touchdown_hysteresis,
)


@dataclass(frozen=True)
class TrotWindow:
    swing_feet: tuple[str, str]
    start_time: float
    duration: float

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass
class SwingPlan:
    foot: str
    start_position: np.ndarray
    target_position: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a diagonal-pair trot reference through MPC + WBC.")
    parser.add_argument("--vx", type=float, default=0.012, help="Forward velocity command in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command in m/s.")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="Yaw rate command in rad/s.")
    parser.add_argument("--cycles", type=int, default=6, help="Number of two-phase trot cycles.")
    parser.add_argument("--swing-duration", type=float, default=0.35, help="Diagonal pair swing duration in seconds.")
    parser.add_argument("--stance-gap", type=float, default=0.30, help="All-stance gap between diagonal swings.")
    parser.add_argument("--swing-height", type=float, default=0.025, help="Swing clearance in meters.")
    parser.add_argument("--max-step-length", type=float, default=0.035, help="Planar foothold delta limit in meters.")
    parser.add_argument("--viewer-hz", type=float, default=60.0, help="Viewer sync rate. Use 0 to sync every step.")
    parser.add_argument("--mpc-dt", type=float, default=MPC_UPDATE_DT, help="MPC update period in seconds.")
    parser.add_argument("--wbc-dt", type=float, default=WBC_UPDATE_DT, help="WBC update period in seconds.")
    parser.add_argument("--profile-dt", type=float, default=PROFILE_LOG_DT, help="Profiler print period in sim seconds.")
    parser.add_argument("--touchdown-z-tol", type=float, default=0.018, help="Foot-point z tolerance for trot touchdown.")
    parser.add_argument("--touchdown-extra-time", type=float, default=0.25, help="Maximum extra swing time while waiting for touchdown.")
    parser.add_argument("--touchdown-hold-time", type=float, default=0.010, help="Continuous touchdown-condition time before switching to stance.")
    parser.add_argument("--landing-force-ramp-time", type=float, default=0.05, help="Force-reference ramp time after touchdown.")
    parser.add_argument("--landing-force-zero-weight", type=float, default=5.0, help="Temporary WBC force-to-zero weight after touchdown.")
    parser.add_argument("--start-roll-tol", type=float, default=0.04, help="Maximum absolute roll before starting the next trot swing.")
    parser.add_argument("--start-y-tol", type=float, default=0.04, help="Maximum absolute lateral base error before starting the next trot swing.")
    parser.add_argument("--max-start-delay", type=float, default=0.50, help="Maximum delay applied to a trot swing while waiting for recovery.")
    parser.add_argument("--no-sleep", action="store_true", help="Do not sleep to match MuJoCo real-time step.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.cycles <= 0:
        raise ValueError("--cycles must be positive")
    if args.swing_duration <= 0.0:
        raise ValueError("--swing-duration must be positive")
    if args.stance_gap < 0.0:
        raise ValueError("--stance-gap must be non-negative")
    if args.swing_height < 0.0:
        raise ValueError("--swing-height must be non-negative")
    if args.max_step_length <= 0.0:
        raise ValueError("--max-step-length must be positive")
    if args.viewer_hz < 0.0:
        raise ValueError("--viewer-hz must be non-negative")
    if args.mpc_dt <= 0.0:
        raise ValueError("--mpc-dt must be positive")
    if args.wbc_dt <= 0.0:
        raise ValueError("--wbc-dt must be positive")
    if args.profile_dt < 0.0:
        raise ValueError("--profile-dt must be non-negative")
    if args.touchdown_z_tol < 0.0:
        raise ValueError("--touchdown-z-tol must be non-negative")
    if args.touchdown_extra_time < 0.0:
        raise ValueError("--touchdown-extra-time must be non-negative")
    if args.touchdown_hold_time < 0.0:
        raise ValueError("--touchdown-hold-time must be non-negative")
    if args.landing_force_ramp_time < 0.0:
        raise ValueError("--landing-force-ramp-time must be non-negative")
    if args.landing_force_zero_weight < 0.0:
        raise ValueError("--landing-force-zero-weight must be non-negative")
    if args.start_roll_tol < 0.0:
        raise ValueError("--start-roll-tol must be non-negative")
    if args.start_y_tol < 0.0:
        raise ValueError("--start-y-tol must be non-negative")
    if args.max_start_delay < 0.0:
        raise ValueError("--max-start-delay must be non-negative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    viewer_sync_dt = 0.0 if args.viewer_hz <= 0.0 else 1.0 / args.viewer_hz

    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")

    home_qpos_ref = robot.q.copy()
    home_com_ref = robot.center_of_mass()
    initial_base_pos = robot.data.qpos[0:3].copy()
    initial_foot_positions = {foot: robot.geom_position(foot) for foot in FOOT_GEOMS}
    locked_positions = {foot: pos.copy() for foot, pos in initial_foot_positions.items()}

    windows = build_trot_windows(args.cycles, args.swing_duration, args.stance_gap)
    period = 2.0 * (args.swing_duration + args.stance_gap)
    nominal_step_delta = limited_planar_delta(
        np.array([args.vx * period, args.vy * period, 0.0], dtype=float),
        args.max_step_length,
    )

    mpc_config = CentroidalMPCConfig(
        contact_geoms=FOOT_GEOMS,
        horizon_steps=12,
        dt=0.03,
        normal_force_min=MPC_NORMAL_FORCE_MIN,
        weight_orientation=500.0,
        weight_angular_velocity=80.0,
        weight_orientation_axes=(2000.0, 2000.0, 600.0),
        weight_angular_velocity_axes=(300.0, 300.0, 90.0),
        weight_force_rate=2.0e-4,
        weight_initial_force_rate=8.0e-4,
    )
    mpc = CentroidalMPC(mpc_config)
    stance_controller = StanceWBCQP(
        StanceWBCConfig(
            foot_geoms=FOOT_GEOMS,
            weight_force=1.0,
            weight_base_ori=320.0,
            weight_joint_posture=14.0,
            weight_tau_rate=5.0e-4,
            kp_base_ori=220.0,
            kd_base_ori=44.0,
            kp_stance=100.0,
            kd_stance=20.0,
            use_jdot_v=False,
        )
    )
    generic_controllers: dict[tuple[tuple[str, ...], tuple[str, ...]], GeneralContactWBCQP] = {}

    dt = float(robot.model.opt.timestep)
    active_window_id: int | None = None
    next_window_id = 0
    active_plans: dict[str, SwingPlan] = {}
    completed_windows: set[int] = set()
    window_delay_used = np.zeros(len(windows), dtype=float)
    touchdown_times_by_foot: dict[str, float] = {}
    touchdown_candidate_times: dict[str, float] = {}
    next_mpc_update = 0.0
    next_wbc_update = 0.0
    next_viewer_sync = 0.0
    next_log_time = 0.0
    next_profile_time = args.profile_dt
    last_tau = np.zeros(robot.nu)
    last_wbc_status = "not run"
    last_max_tau = 0.0
    mpc_force_ref = np.zeros(3 * len(FOOT_GEOMS))
    mpc_status = "not run"
    mpc_residual = 0.0
    solve_failures = 0
    last_dyn_residual = 0.0
    last_stance_residual = 0.0
    last_swing_error = 0.0
    profiler = LoopProfiler()
    profile_wall_start = time.perf_counter()
    profile_sim_start = 0.0

    print(
        "trot reference: cycles={}, command=[{:.4f}, {:.4f}, {:.4f}], step_delta={} m, swing={:.2f}s gap={:.2f}s".format(
            args.cycles,
            args.vx,
            args.vy,
            args.yaw_rate,
            np.round(nominal_step_delta, 5).tolist(),
            args.swing_duration,
            args.stance_gap,
        )
    )
    command_start_time = windows[0].start_time if windows else 0.0

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            sim_time = float(robot.data.time)

            with profiler.time("schedule"):
                if active_window_id is None and next_window_id < len(windows) and sim_time >= windows[next_window_id].start_time:
                    if should_delay_next_trot_window(
                        robot,
                        initial_base_pos,
                        args.start_roll_tol,
                        args.start_y_tol,
                    ) and window_delay_used[next_window_id] < args.max_start_delay:
                        delay = min(dt, args.max_start_delay - window_delay_used[next_window_id])
                        windows = delay_trot_windows(windows, next_window_id, delay)
                        window_delay_used[next_window_id:] += delay
                    else:
                        active_window_id = next_window_id
                        window = windows[active_window_id]
                        active_plans = {
                            foot: SwingPlan(
                                foot=foot,
                                start_position=locked_positions[foot].copy(),
                                target_position=locked_positions[foot] + foothold_delta_for_foot(
                                    foot,
                                    initial_foot_positions,
                                    nominal_step_delta,
                                    args.yaw_rate * period,
                                    args.max_step_length,
                                ),
                            )
                            for foot in window.swing_feet
                        }
                        for foot in window.swing_feet:
                            touchdown_candidate_times.pop(foot, None)
                        next_wbc_update = sim_time
                        next_mpc_update = sim_time

                current_window = windows[active_window_id] if active_window_id is not None else None
                if current_window is not None and should_finish_trot_window(
                    robot,
                    current_window,
                    active_plans,
                    sim_time,
                    args.touchdown_z_tol,
                    args.touchdown_extra_time,
                    args.touchdown_hold_time,
                    touchdown_candidate_times,
                ):
                    for foot in current_window.swing_feet:
                        locked_positions[foot] = active_plans[foot].target_position.copy()
                        touchdown_times_by_foot[foot] = sim_time
                        touchdown_candidate_times.pop(foot, None)
                    completed_windows.add(active_window_id)
                    active_window_id = None
                    next_window_id += 1
                    active_plans = {}
                    current_window = None
                    next_wbc_update = sim_time
                    next_mpc_update = sim_time

                swing_refs = {}
                if current_window is not None:
                    swing_refs = {
                        foot: swing_foothold_reference(
                            initial_position=plan.start_position,
                            step_delta=plan.target_position - plan.start_position,
                            swing_height=args.swing_height,
                            start_time=current_window.start_time,
                            duration=current_window.duration,
                            time_s=sim_time,
                        )
                        for foot, plan in active_plans.items()
                    }

                contact_schedule = trot_contact_schedule(
                    windows,
                    sim_time,
                    mpc_config.horizon_steps,
                    mpc_config.dt,
                    active_window=current_window,
                )
                command_time = max(0.0, sim_time - command_start_time)
                planned_foot_positions = planned_feet_from_refs(locked_positions, swing_refs)
                yaw_ref = args.yaw_rate * command_time
                base_ref = foot_centered_base_reference(
                    home_qpos_ref,
                    initial_base_pos,
                    initial_foot_positions,
                    planned_foot_positions,
                    yaw=yaw_ref,
                )
                base_ref[1] = initial_base_pos[1] + args.vy * command_time
                com_ref = home_com_ref.copy()
                com_ref[0:2] += base_ref[0:2] - initial_base_pos[0:2]
                com_vel_ref = np.array([args.vx, args.vy, 0.0], dtype=float)
                orientation_ref = np.array([0.0, 0.0, yaw_ref], dtype=float)
                angular_velocity_ref = np.array([0.0, 0.0, args.yaw_rate], dtype=float)

            if sim_time >= next_mpc_update:
                with profiler.time("mpc"):
                    mpc_solution = mpc.solve(
                        robot,
                        com_ref,
                        com_velocity_ref=com_vel_ref,
                        orientation_ref=orientation_ref,
                        angular_velocity_ref=angular_velocity_ref,
                        contact_schedule=contact_schedule,
                    )
                    mpc_force_ref = mpc_solution.first_contact_forces
                    mpc_status = mpc_solution.status
                    mpc_residual = float(np.linalg.norm(mpc_solution.dynamics_residual))
                next_mpc_update += args.mpc_dt

            if sim_time >= next_wbc_update:
                with profiler.time("wbc"):
                    ramped_force_ref = landing_ramped_force_ref(
                        mpc_force_ref,
                        FOOT_GEOMS,
                        sim_time,
                        touchdown_times_by_foot,
                        args.landing_force_ramp_time,
                    )
                    landing_zero_weights = landing_force_zero_weights(
                        FOOT_GEOMS,
                        sim_time,
                        touchdown_times_by_foot,
                        args.landing_force_ramp_time,
                        args.landing_force_zero_weight,
                    )
                    if current_window is None:
                        solution = stance_controller.solve(
                            robot,
                            base_ref,
                            force_ref=ramped_force_ref,
                            force_zero_weights=landing_zero_weights,
                            stance_pos_refs=locked_positions,
                        )
                    else:
                        swing_feet = current_window.swing_feet
                        stance_feet = tuple(foot for foot in FOOT_GEOMS if foot not in swing_feet)
                        key = (stance_feet, swing_feet)
                        if key not in generic_controllers:
                            generic_controllers[key] = GeneralContactWBCQP(
                                GeneralContactWBCConfig(
                                    stance_foot_geoms=stance_feet,
                                    swing_foot_geoms=swing_feet,
                                    normal_force_min=MPC_NORMAL_FORCE_MIN,
                                    weight_swing_foot=1400.0,
                                    weight_force=1.0,
                                    weight_base_ori=450.0,
                                    weight_joint_posture=10.0,
                                    weight_tau_rate=5.0e-4,
                                    kp_swing=450.0,
                                    kd_swing=42.0,
                                    kp_base_ori=240.0,
                                    kd_base_ori=46.0,
                                    kp_stance=100.0,
                                    kd_stance=20.0,
                                    use_jdot_v=False,
                                )
                            )
                        solution = generic_controllers[key].solve(
                            robot,
                            base_ref,
                            swing_pos_refs={foot: ref.position for foot, ref in swing_refs.items()},
                            swing_vel_refs={foot: ref.velocity for foot, ref in swing_refs.items()},
                            swing_acc_refs={foot: ref.acceleration for foot, ref in swing_refs.items()},
                            force_ref=force_ref_for_feet(ramped_force_ref, stance_feet),
                            force_zero_weights=force_ref_for_feet(landing_zero_weights, stance_feet),
                            stance_pos_refs={foot: locked_positions[foot] for foot in stance_feet},
                        )

                    last_wbc_status = solution.status
                    if solution.status in ("solved", "solved inaccurate") and mpc_status in ("solved", "solved inaccurate"):
                        last_tau = solution.tau.copy()
                        last_dyn_residual = float(np.linalg.norm(solution.dynamics_residual))
                        last_stance_residual = float(np.linalg.norm(solution.stance_residual))
                        last_swing_error = float(np.linalg.norm(getattr(solution, "swing_accel_error", np.zeros(0))))
                    else:
                        solve_failures += 1
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
                phase = "stance" if current_window is None else "+".join(current_window.swing_feet) + "-swing"
                roll, pitch, yaw = quat_to_rpy(robot.data.qpos[3:7])
                contacts = "".join("1" if robot.geom_has_contact(foot) else "0" for foot in FOOT_GEOMS)
                fz = mpc_force_ref.reshape(len(FOOT_GEOMS), 3)[:, 2]
                print(
                    "t={:.2f}s phase={} step={}/{} base={} rpy={} contacts={} fz={} tau={:.2f} res=[{:.1e},{:.1e},{:.1e}] fails={} mpc={} wbc={} mpc_res={:.1e}".format(
                        robot.data.time,
                        phase,
                        min(next_window_id, len(windows)),
                        len(windows),
                        np.round(robot.data.qpos[0:3], 4).tolist(),
                        np.round([roll, pitch, yaw], 3).tolist(),
                        contacts,
                        np.round(fz, 1).tolist(),
                        last_max_tau,
                        last_dyn_residual,
                        last_stance_residual,
                        last_swing_error,
                        solve_failures,
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


def build_trot_windows(cycles: int, swing_duration: float, stance_gap: float) -> list[TrotWindow]:
    windows: list[TrotWindow] = []
    start = 1.0
    stride = swing_duration + stance_gap
    for idx in range(2 * cycles):
        windows.append(TrotWindow(swing_feet=TROT_PAIRS[idx % 2], start_time=start + idx * stride, duration=swing_duration))
    return windows


def trot_contact_schedule(
    windows: list[TrotWindow],
    current_time: float,
    horizon_steps: int,
    dt: float,
    active_window: TrotWindow | None = None,
) -> np.ndarray:
    schedule = np.ones((horizon_steps, len(FOOT_GEOMS)), dtype=bool)
    foot_to_index = {foot: idx for idx, foot in enumerate(FOOT_GEOMS)}
    for step in range(horizon_steps):
        knot_time = current_time + step * dt
        for window in windows:
            if window.start_time <= knot_time < window.end_time:
                for foot in window.swing_feet:
                    schedule[step, foot_to_index[foot]] = False
        if active_window is not None:
            for foot in active_window.swing_feet:
                schedule[step, foot_to_index[foot]] = False
    return schedule


def should_finish_trot_window(
    robot: MuJoCoModelInterface,
    window: TrotWindow,
    active_plans: dict[str, SwingPlan],
    time_s: float,
    touchdown_z_tol: float,
    touchdown_extra_time: float,
    touchdown_hold_time: float = 0.0,
    touchdown_candidate_times: dict[str, float] | None = None,
) -> bool:
    if time_s < window.end_time:
        return False
    if time_s >= window.end_time + touchdown_extra_time:
        return True

    condition_by_foot = {}
    for foot in window.swing_feet:
        plan = active_plans[foot]
        foot_pos = robot.geom_position(foot)
        target_pos = plan.target_position
        near_ground = foot_pos[2] <= target_pos[2] + touchdown_z_tol
        near_target_xy = np.linalg.norm(foot_pos[0:2] - target_pos[0:2]) <= 0.04
        condition_by_foot[foot] = bool(near_ground and near_target_xy)

    if touchdown_hold_time <= 0.0 or touchdown_candidate_times is None:
        return all(condition_by_foot.values())
    return update_touchdown_hysteresis(
        condition_by_foot,
        time_s,
        touchdown_hold_time,
        touchdown_candidate_times,
    )


def should_delay_next_trot_window(
    robot: MuJoCoModelInterface,
    initial_base_pos: np.ndarray,
    roll_tol: float,
    y_tol: float,
) -> bool:
    roll, _, _ = quat_to_rpy(robot.data.qpos[3:7])
    y_error = float(robot.data.qpos[1] - initial_base_pos[1])
    return abs(roll) > roll_tol or abs(y_error) > y_tol


def delay_trot_windows(windows: list[TrotWindow], start_index: int, delay: float) -> list[TrotWindow]:
    if delay <= 0.0:
        return windows
    shifted = list(windows)
    for idx in range(start_index, len(shifted)):
        window = shifted[idx]
        shifted[idx] = TrotWindow(
            swing_feet=window.swing_feet,
            start_time=window.start_time + delay,
            duration=window.duration,
        )
    return shifted


def base_reference(home_qpos_ref: np.ndarray, initial_base_pos: np.ndarray, time_s: float, vx: float, vy: float, yaw_rate: float) -> np.ndarray:
    qpos_ref = home_qpos_ref.copy()
    qpos_ref[0] = initial_base_pos[0] + vx * time_s
    qpos_ref[1] = initial_base_pos[1] + vy * time_s
    qpos_ref[3:7] = yaw_quat(yaw_rate * time_s)
    return qpos_ref


def planned_feet_from_refs(
    locked_positions: dict[str, np.ndarray],
    swing_refs: dict[str, object],
) -> dict[str, np.ndarray]:
    planned = {foot: pos.copy() for foot, pos in locked_positions.items()}
    for foot, ref in swing_refs.items():
        planned[foot] = ref.position.copy()
    return planned


def foot_centered_base_reference(
    home_qpos_ref: np.ndarray,
    initial_base_pos: np.ndarray,
    initial_foot_positions: dict[str, np.ndarray],
    planned_foot_positions: dict[str, np.ndarray],
    yaw: float,
) -> np.ndarray:
    qpos_ref = home_qpos_ref.copy()
    foot_delta_xy = np.mean(
        np.vstack(
            [
                planned_foot_positions[foot][0:2] - initial_foot_positions[foot][0:2]
                for foot in FOOT_GEOMS
            ]
        ),
        axis=0,
    )
    qpos_ref[0:2] = initial_base_pos[0:2] + foot_delta_xy
    qpos_ref[3:7] = yaw_quat(yaw)
    return qpos_ref


def yaw_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def quat_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = np.asarray(quat, dtype=float)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def foothold_delta_for_foot(
    foot: str,
    initial_foot_positions: dict[str, np.ndarray],
    step_delta: np.ndarray,
    yaw_delta: float,
    max_step_length: float,
) -> np.ndarray:
    delta = step_delta.copy()
    if yaw_delta != 0.0:
        center_xy = np.mean(np.vstack([pos[0:2] for pos in initial_foot_positions.values()]), axis=0)
        foot_xy = initial_foot_positions[foot][0:2]
        offset_xy = foot_xy - center_xy
        delta[0:2] += yaw_delta * np.array([-offset_xy[1], offset_xy[0]], dtype=float)
    return limited_planar_delta(delta, max_step_length)


def limited_planar_delta(delta: np.ndarray, max_step_length: float) -> np.ndarray:
    limited = np.asarray(delta, dtype=float).copy()
    planar_norm = float(np.linalg.norm(limited[0:2]))
    if planar_norm > max_step_length:
        limited[0:2] *= max_step_length / planar_norm
    return limited


def force_ref_for_feet(force_ref_all: np.ndarray, selected_feet: tuple[str, ...]) -> np.ndarray:
    forces_by_foot = {
        foot: force
        for foot, force in zip(FOOT_GEOMS, force_ref_all.reshape(len(FOOT_GEOMS), 3))
    }
    return np.vstack([forces_by_foot[foot] for foot in selected_feet]).reshape(-1)


if __name__ == "__main__":
    main()
