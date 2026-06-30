"""Generate smooth locomotion demos by offline rollout, then 60 Hz replay/render.

The live viewer scripts solve MPC/WBC in the same loop that draws frames. If
the QP stack is slower than real time, the visual result looks choppy even when
the simulated motion is valid. This script separates the two jobs:

1. Roll out the controller headlessly and store qpos/qvel samples.
2. Replay or render those stored states at a fixed visual frame rate.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
DEFAULT_STATES_PATH = PROJECT_ROOT / "outputs" / "trot_straight_turn_demo.npz"
DEFAULT_GIF_PATH = PROJECT_ROOT / "outputs" / "trot_straight_turn_demo.gif"
DEFAULT_SETTLE_TIME = 0.8
DEFAULT_END_PADDING = 0.8
DEFAULT_SWING_DURATION = 0.28
DEFAULT_STANCE_GAP = 0.52
DEFAULT_SWING_HEIGHT = 0.018
DEFAULT_MAX_STEP_LENGTH = 0.026
DEFAULT_LATERAL_FOOT_WIDTH_CORRECTION = 0.0
DEFAULT_LATERAL_FOOT_WIDTH_MARGIN = 0.04

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    CentroidalMPC,
    CentroidalMPCConfig,
    GeneralContactWBCConfig,
    GeneralContactWBCQP,
    MuJoCoModelInterface,
    StanceWBCConfig,
    StanceWBCQP,
    swing_foothold_reference,
)


def _load_trot_module():
    module_path = PROJECT_ROOT / "scripts" / "run_trot_reference_viewer.py"
    spec = importlib.util.spec_from_file_location("run_trot_reference_viewer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trot = _load_trot_module()


@dataclass(frozen=True)
class CommandSegment:
    duration: float
    vx: float
    vy: float
    yaw_rate: float


@dataclass(frozen=True)
class DemoWindow:
    swing_feet: tuple[str, ...]
    start_time: float
    duration: float
    step_delta: np.ndarray
    yaw_delta: float

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass
class SwingPlan:
    foot: str
    start_position: np.ndarray
    target_position: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record smooth MPC/WBC locomotion demos.")
    parser.add_argument(
        "--preset",
        choices=("straight-turn", "straight", "turn-left", "forward-2m", "forward-2m-smooth", "trot-l-route"),
        default="straight-turn",
        help=(
            "Reference preset. straight/turn presets use diagonal trot; "
            "forward-2m presets use single-leg crawl-walk."
        ),
    )
    parser.add_argument("--states-output", type=Path, default=DEFAULT_STATES_PATH)
    parser.add_argument("--gif-output", type=Path, default=DEFAULT_GIF_PATH)
    parser.add_argument("--no-gif", action="store_true", help="Only save the qpos/qvel rollout.")
    parser.add_argument("--viewer-replay", action="store_true", help="Replay stored states in the MuJoCo viewer.")
    parser.add_argument("--fps", type=float, default=60.0, help="Visual sample/replay/render frame rate.")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Replay/render speed multiplier for stored states.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--settle-time", type=float, default=DEFAULT_SETTLE_TIME)
    parser.add_argument("--end-padding", type=float, default=DEFAULT_END_PADDING)
    parser.add_argument("--swing-duration", type=float, default=DEFAULT_SWING_DURATION)
    parser.add_argument("--stance-gap", type=float, default=DEFAULT_STANCE_GAP)
    parser.add_argument("--swing-height", type=float, default=DEFAULT_SWING_HEIGHT)
    parser.add_argument("--max-step-length", type=float, default=DEFAULT_MAX_STEP_LENGTH)
    parser.add_argument(
        "--lateral-foot-width-correction",
        type=float,
        default=DEFAULT_LATERAL_FOOT_WIDTH_CORRECTION,
        help="Blend swing targets back toward the nominal body-frame lateral foot offset.",
    )
    parser.add_argument(
        "--lateral-foot-width-margin",
        type=float,
        default=DEFAULT_LATERAL_FOOT_WIDTH_MARGIN,
        help="Allowed body-frame lateral foot offset error before correction is applied.",
    )
    parser.add_argument("--mpc-dt", type=float, default=trot.MPC_UPDATE_DT)
    parser.add_argument("--wbc-dt", type=float, default=trot.WBC_UPDATE_DT)
    parser.add_argument("--log-dt", type=float, default=0.5)
    parser.add_argument("--start-roll-tol", type=float, default=0.05)
    parser.add_argument("--start-y-tol", type=float, default=0.05)
    parser.add_argument("--max-start-delay", type=float, default=0.8)
    parser.add_argument("--touchdown-z-tol", type=float, default=0.02)
    parser.add_argument("--touchdown-extra-time", type=float, default=0.35)
    parser.add_argument("--stop-on-fall", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.playback_speed <= 0.0:
        raise ValueError("--playback-speed must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.settle_time < 0.0:
        raise ValueError("--settle-time must be non-negative")
    if args.end_padding < 0.0:
        raise ValueError("--end-padding must be non-negative")
    if args.swing_duration <= 0.0:
        raise ValueError("--swing-duration must be positive")
    if args.stance_gap < 0.0:
        raise ValueError("--stance-gap must be non-negative")
    if args.swing_height < 0.0:
        raise ValueError("--swing-height must be non-negative")
    if args.max_step_length <= 0.0:
        raise ValueError("--max-step-length must be positive")
    if not 0.0 <= args.lateral_foot_width_correction <= 1.0:
        raise ValueError("--lateral-foot-width-correction must be in [0, 1]")
    if args.lateral_foot_width_margin < 0.0:
        raise ValueError("--lateral-foot-width-margin must be non-negative")
    if args.mpc_dt <= 0.0:
        raise ValueError("--mpc-dt must be positive")
    if args.wbc_dt <= 0.0:
        raise ValueError("--wbc-dt must be positive")


def main() -> None:
    args = build_parser().parse_args()
    apply_preset_defaults(args)
    validate_args(args)

    segments = preset_segments(args.preset)
    states_path = prepare_output_path(args.states_output)
    gif_path = prepare_output_path(args.gif_output)

    times, qpos, qvel = rollout_demo(args, segments)
    save_rollout(states_path, times, qpos, qvel, args, segments)
    print(f"saved rollout: {states_path} ({len(times)} frames at {args.fps:.1f} Hz)")

    if not args.no_gif:
        render_gif(states_path, gif_path, args.width, args.height, args.fps, args.playback_speed)
        print(f"saved gif: {gif_path}")

    if args.viewer_replay:
        replay_viewer(states_path, args.fps, args.playback_speed)


def preset_segments(name: str) -> list[CommandSegment]:
    if name == "straight":
        return [CommandSegment(duration=4.2, vx=0.010, vy=0.0, yaw_rate=0.0)]
    if name == "turn-left":
        return [CommandSegment(duration=4.8, vx=0.006, vy=0.0, yaw_rate=0.055)]
    if name == "forward-2m":
        return [CommandSegment(duration=36.0, vx=0.070, vy=0.0, yaw_rate=0.0)]
    if name == "forward-2m-smooth":
        return [CommandSegment(duration=36.0, vx=0.070, vy=0.0, yaw_rate=0.0)]
    if name == "trot-l-route":
        return [
            CommandSegment(duration=37.5, vx=0.040, vy=0.0, yaw_rate=0.0),
            CommandSegment(duration=20.0, vx=0.004, vy=0.0, yaw_rate=np.pi / 40.0),
            CommandSegment(duration=3.0, vx=0.0, vy=0.0, yaw_rate=0.0),
            CommandSegment(duration=13.0, vx=0.040, vy=0.0, yaw_rate=0.0),
            CommandSegment(duration=3.0, vx=0.0, vy=0.0, yaw_rate=0.0),
            CommandSegment(duration=13.0, vx=0.040, vy=0.0, yaw_rate=0.0),
            CommandSegment(duration=3.0, vx=0.0, vy=0.0, yaw_rate=0.0),
            CommandSegment(duration=13.0, vx=0.040, vy=0.0, yaw_rate=0.0),
        ]
    if name == "straight-turn":
        return [
            CommandSegment(duration=2.4, vx=0.010, vy=0.0, yaw_rate=0.0),
            CommandSegment(duration=3.0, vx=0.006, vy=0.0, yaw_rate=0.055),
        ]
    raise ValueError(f"Unknown preset: {name}")


def apply_preset_defaults(args: argparse.Namespace) -> None:
    if args.preset not in ("forward-2m", "forward-2m-smooth", "trot-l-route"):
        return

    if args.preset == "forward-2m":
        preset = {
            "swing_duration": 0.18,
            "stance_gap": 0.12,
            "swing_height": 0.022,
            "max_step_length": 0.08,
            "lateral_foot_width_correction": 0.0,
            "lateral_foot_width_margin": DEFAULT_LATERAL_FOOT_WIDTH_MARGIN,
            "end_padding": 1.5,
            "playback_speed": 2.0,
        }
    elif args.preset == "forward-2m-smooth":
        preset = {
            "swing_duration": 0.20,
            "stance_gap": 0.16,
            "swing_height": 0.030,
            "max_step_length": 0.09,
            "lateral_foot_width_correction": 0.0,
            "lateral_foot_width_margin": DEFAULT_LATERAL_FOOT_WIDTH_MARGIN,
            "end_padding": 1.5,
            "playback_speed": 2.0,
        }
    else:
        preset = {
            "swing_duration": 0.20,
            "stance_gap": 0.45,
            "swing_height": 0.035,
            "max_step_length": 0.045,
            "lateral_foot_width_correction": 0.35,
            "lateral_foot_width_margin": 0.025,
            "end_padding": 0.0,
            "playback_speed": 4.0,
        }

    apply_default_if_not_passed(args, "swing_duration", "--swing-duration", DEFAULT_SWING_DURATION, preset["swing_duration"])
    apply_default_if_not_passed(args, "stance_gap", "--stance-gap", DEFAULT_STANCE_GAP, preset["stance_gap"])
    apply_default_if_not_passed(args, "swing_height", "--swing-height", DEFAULT_SWING_HEIGHT, preset["swing_height"])
    apply_default_if_not_passed(args, "max_step_length", "--max-step-length", DEFAULT_MAX_STEP_LENGTH, preset["max_step_length"])
    apply_default_if_not_passed(
        args,
        "lateral_foot_width_correction",
        "--lateral-foot-width-correction",
        DEFAULT_LATERAL_FOOT_WIDTH_CORRECTION,
        preset["lateral_foot_width_correction"],
    )
    apply_default_if_not_passed(
        args,
        "lateral_foot_width_margin",
        "--lateral-foot-width-margin",
        DEFAULT_LATERAL_FOOT_WIDTH_MARGIN,
        preset["lateral_foot_width_margin"],
    )
    apply_default_if_not_passed(args, "end_padding", "--end-padding", DEFAULT_END_PADDING, preset["end_padding"])
    apply_default_if_not_passed(args, "playback_speed", "--playback-speed", 1.0, preset["playback_speed"])


def apply_default_if_not_passed(
    args: argparse.Namespace,
    attr: str,
    option: str,
    parser_default: float,
    preset_default: float,
) -> None:
    if not option_was_passed(option) and getattr(args, attr) == parser_default:
        setattr(args, attr, preset_default)


def option_was_passed(option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in sys.argv[1:])


def rollout_demo(args: argparse.Namespace, segments: list[CommandSegment]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")

    foot_geoms = trot.FOOT_GEOMS
    home_qpos_ref = robot.q.copy()
    home_com_ref = robot.center_of_mass()
    initial_base_pos = robot.data.qpos[0:3].copy()
    initial_foot_positions = {foot: robot.geom_position(foot) for foot in foot_geoms}
    locked_positions = {foot: pos.copy() for foot, pos in initial_foot_positions.items()}

    windows = build_demo_windows(
        args.preset,
        segments,
        args.settle_time,
        args.swing_duration,
        args.stance_gap,
        args.max_step_length,
    )
    if not windows:
        raise RuntimeError("Demo generated no swing windows")

    end_time = args.settle_time + sum(segment.duration for segment in segments) + args.end_padding
    sample_dt = 1.0 / args.fps
    mpc_config = CentroidalMPCConfig(
        contact_geoms=foot_geoms,
        horizon_steps=12,
        dt=0.03,
        normal_force_min=trot.MPC_NORMAL_FORCE_MIN,
        weight_com_position=650.0,
        weight_com_velocity=15.0,
        weight_orientation=1400.0,
        weight_angular_velocity=120.0,
    )
    mpc = CentroidalMPC(mpc_config)
    stance_controller = StanceWBCQP(
        StanceWBCConfig(
            foot_geoms=foot_geoms,
            weight_base_pos=350.0,
            weight_base_ori=240.0,
            weight_force=1.0,
            kp_base_pos=180.0,
            kd_base_pos=42.0,
            kp_base_ori=180.0,
            kd_base_ori=36.0,
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
    window_delay_used = np.zeros(len(windows), dtype=float)
    next_mpc_update = 0.0
    next_wbc_update = 0.0
    next_log_time = 0.0
    next_sample_time = 0.0
    last_tau = np.zeros(robot.nu)
    mpc_force_ref = np.zeros(3 * len(foot_geoms))
    solve_failures = 0
    times: list[float] = []
    qpos_samples: list[np.ndarray] = []
    qvel_samples: list[np.ndarray] = []

    print(
        "rollout preset={} end={:.2f}s frames={} swing={:.2f}s gap={:.2f}s".format(
            args.preset,
            end_time,
            int(np.ceil(end_time * args.fps)),
            args.swing_duration,
            args.stance_gap,
        )
    )

    while robot.data.time < end_time:
        sim_time = float(robot.data.time)

        if active_window_id is None and next_window_id < len(windows) and sim_time >= windows[next_window_id].start_time:
            _, ref_y_for_delay, _ = integrated_command_pose(segments, max(0.0, sim_time - args.settle_time))
            if (
                should_delay_next_demo_window(
                    robot,
                    initial_base_pos[1] + ref_y_for_delay,
                    args.start_roll_tol,
                    args.start_y_tol,
                )
                and window_delay_used[next_window_id] < args.max_start_delay
            ):
                delay = min(dt, args.max_start_delay - window_delay_used[next_window_id])
                windows = delay_demo_windows(windows, next_window_id, delay)
                window_delay_used[next_window_id:] += delay
            else:
                active_window_id = next_window_id
                window = windows[active_window_id]
                base_yaw_for_foothold = trot.quat_to_rpy(robot.data.qpos[3:7])[2]
                active_plans = {
                    foot: SwingPlan(
                        foot=foot,
                        start_position=locked_positions[foot].copy(),
                        target_position=lateral_width_regulated_foothold(
                            foot,
                            locked_positions,
                            robot.data.qpos[0:3],
                            base_yaw_for_foothold,
                            initial_base_pos,
                            initial_foot_positions,
                            window.step_delta,
                            window.yaw_delta,
                            args.max_step_length,
                            args.lateral_foot_width_correction,
                            args.lateral_foot_width_margin,
                        ),
                    )
                    for foot in window.swing_feet
                }
                next_wbc_update = sim_time
                next_mpc_update = sim_time

        current_window = windows[active_window_id] if active_window_id is not None else None
        if current_window is not None and trot.should_finish_trot_window(
            robot,
            current_window,
            active_plans,
            sim_time,
            args.touchdown_z_tol,
            args.touchdown_extra_time,
        ):
            for foot in current_window.swing_feet:
                locked_positions[foot] = active_plans[foot].target_position.copy()
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

        command_time = max(0.0, sim_time - args.settle_time)
        command = command_at_time(segments, command_time)
        ref_x, ref_y, ref_yaw = integrated_command_pose(segments, command_time)
        com_vel_ref = command_velocity_world(segments, command_time)
        contact_schedule = demo_contact_schedule(
            windows,
            sim_time,
            mpc_config.horizon_steps,
            mpc_config.dt,
            active_window=current_window,
        )
        planned_foot_positions = trot.planned_feet_from_refs(locked_positions, swing_refs)
        base_ref = trot.foot_centered_base_reference(
            home_qpos_ref,
            initial_base_pos,
            initial_foot_positions,
            planned_foot_positions,
            yaw=ref_yaw,
        )
        base_ref[0] = 0.85 * base_ref[0] + 0.15 * (initial_base_pos[0] + ref_x)
        base_ref[1] = initial_base_pos[1] + ref_y

        com_ref = home_com_ref.copy()
        com_ref[0:2] += base_ref[0:2] - initial_base_pos[0:2]
        orientation_ref = np.array([0.0, 0.0, ref_yaw], dtype=float)
        angular_velocity_ref = np.array([0.0, 0.0, command.yaw_rate], dtype=float)

        if sim_time >= next_mpc_update:
            mpc_solution = mpc.solve(
                robot,
                com_ref,
                com_velocity_ref=com_vel_ref,
                orientation_ref=orientation_ref,
                angular_velocity_ref=angular_velocity_ref,
                contact_schedule=contact_schedule,
            )
            if mpc_solution.status in ("solved", "solved inaccurate"):
                mpc_force_ref = mpc_solution.first_contact_forces
            else:
                solve_failures += 1
            next_mpc_update += args.mpc_dt

        if sim_time >= next_wbc_update:
            if current_window is None:
                solution = stance_controller.solve(
                    robot,
                    base_ref,
                    force_ref=mpc_force_ref,
                    stance_pos_refs=locked_positions,
                )
            else:
                swing_feet = current_window.swing_feet
                stance_feet = tuple(foot for foot in foot_geoms if foot not in swing_feet)
                key = (stance_feet, swing_feet)
                if key not in generic_controllers:
                    generic_controllers[key] = GeneralContactWBCQP(
                        GeneralContactWBCConfig(
                            stance_foot_geoms=stance_feet,
                            swing_foot_geoms=swing_feet,
                            normal_force_min=trot.MPC_NORMAL_FORCE_MIN,
                            weight_base_pos=420.0,
                            weight_swing_foot=1200.0,
                            weight_force=1.0,
                            weight_base_ori=350.0,
                            kp_base_pos=190.0,
                            kd_base_pos=45.0,
                            kp_swing=360.0,
                            kd_swing=38.0,
                            kp_base_ori=260.0,
                            kd_base_ori=44.0,
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
                    force_ref=trot.force_ref_for_feet(mpc_force_ref, stance_feet),
                    stance_pos_refs={foot: locked_positions[foot] for foot in stance_feet},
                )
            if solution.status in ("solved", "solved inaccurate"):
                last_tau = solution.tau.copy()
            else:
                solve_failures += 1
            next_wbc_update += args.wbc_dt

        robot.data.ctrl[:] = last_tau
        mujoco.mj_step(robot.model, robot.data)

        while robot.data.time >= next_sample_time:
            times.append(float(robot.data.time))
            qpos_samples.append(robot.data.qpos.copy())
            qvel_samples.append(robot.data.qvel.copy())
            next_sample_time += sample_dt

        if robot.data.time >= next_log_time:
            roll, pitch, yaw = trot.quat_to_rpy(robot.data.qpos[3:7])
            phase = "stance" if current_window is None else "+".join(current_window.swing_feet)
            print(
                "t={:.2f}s phase={} base={} rpy={} contacts={} fails={}".format(
                    robot.data.time,
                    phase,
                    np.round(robot.data.qpos[0:3], 3).tolist(),
                    np.round([roll, pitch, yaw], 3).tolist(),
                    contact_string(robot, foot_geoms),
                    solve_failures,
                )
            )
            next_log_time += args.log_dt

        if args.stop_on_fall and has_fallen(robot):
            print(f"stopping early: fall detected at t={robot.data.time:.3f}s")
            break

    return np.asarray(times), np.vstack(qpos_samples), np.vstack(qvel_samples)


def build_demo_windows(
    preset: str,
    segments: list[CommandSegment],
    settle_time: float,
    swing_duration: float,
    stance_gap: float,
    max_step_length: float,
) -> list[DemoWindow]:
    windows: list[DemoWindow] = []
    stride = swing_duration + stance_gap
    swing_sets = preset_swing_sets(preset)
    foot_period = len(swing_sets) * stride
    total_command_time = sum(segment.duration for segment in segments)
    start_time = settle_time
    idx = 0
    while start_time < settle_time + total_command_time:
        command_time = start_time - settle_time
        command = command_at_time(segments, command_time)
        _, _, yaw = integrated_command_pose(segments, command_time)
        if is_zero_command(command):
            start_time += stride
            idx += 1
            continue
        planar_step = rotate_yaw(yaw) @ np.array([command.vx * foot_period, command.vy * foot_period], dtype=float)
        step_delta = trot.limited_planar_delta(
            np.array([planar_step[0], planar_step[1], 0.0], dtype=float),
            max_step_length,
        )
        yaw_delta = command.yaw_rate * foot_period
        windows.append(
            DemoWindow(
                swing_feet=swing_sets[idx % len(swing_sets)],
                start_time=start_time,
                duration=swing_duration,
                step_delta=step_delta,
                yaw_delta=yaw_delta,
            )
        )
        start_time += stride
        idx += 1
    return windows


def is_zero_command(command: CommandSegment) -> bool:
    return abs(command.vx) < 1.0e-12 and abs(command.vy) < 1.0e-12 and abs(command.yaw_rate) < 1.0e-12


def preset_swing_sets(preset: str) -> tuple[tuple[str, ...], ...]:
    if preset in ("forward-2m", "forward-2m-smooth"):
        return (("FL",), ("RR",), ("FR",), ("RL",))
    return trot.TROT_PAIRS


def should_delay_next_demo_window(
    robot: MuJoCoModelInterface,
    y_ref: float,
    roll_tol: float,
    y_tol: float,
) -> bool:
    roll, _, _ = trot.quat_to_rpy(robot.data.qpos[3:7])
    y_error = float(robot.data.qpos[1] - y_ref)
    return abs(roll) > roll_tol or abs(y_error) > y_tol


def demo_contact_schedule(
    windows: list[DemoWindow],
    current_time: float,
    horizon_steps: int,
    dt: float,
    active_window: DemoWindow | None = None,
) -> np.ndarray:
    schedule = np.ones((horizon_steps, len(trot.FOOT_GEOMS)), dtype=bool)
    foot_to_index = {foot: idx for idx, foot in enumerate(trot.FOOT_GEOMS)}
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


def delay_demo_windows(windows: list[DemoWindow], start_index: int, delay: float) -> list[DemoWindow]:
    if delay <= 0.0:
        return windows
    shifted = list(windows)
    for idx in range(start_index, len(shifted)):
        window = shifted[idx]
        shifted[idx] = DemoWindow(
            swing_feet=window.swing_feet,
            start_time=window.start_time + delay,
            duration=window.duration,
            step_delta=window.step_delta,
            yaw_delta=window.yaw_delta,
        )
    return shifted


def command_at_time(segments: list[CommandSegment], time_s: float) -> CommandSegment:
    elapsed = 0.0
    for segment in segments:
        if time_s <= elapsed + segment.duration:
            return segment
        elapsed += segment.duration
    return CommandSegment(duration=0.0, vx=0.0, vy=0.0, yaw_rate=0.0)


def integrated_command_pose(segments: list[CommandSegment], time_s: float) -> tuple[float, float, float]:
    x = 0.0
    y = 0.0
    yaw = 0.0
    remaining = max(0.0, time_s)
    for segment in segments:
        dt = min(remaining, segment.duration)
        if dt <= 0.0:
            break
        velocity_world = rotate_yaw(yaw) @ np.array([segment.vx, segment.vy], dtype=float)
        x += velocity_world[0] * dt
        y += velocity_world[1] * dt
        yaw += segment.yaw_rate * dt
        remaining -= dt
    return x, y, yaw


def command_velocity_world(segments: list[CommandSegment], time_s: float) -> np.ndarray:
    _, _, yaw = integrated_command_pose(segments, time_s)
    command = command_at_time(segments, time_s)
    planar = rotate_yaw(yaw) @ np.array([command.vx, command.vy], dtype=float)
    return np.array([planar[0], planar[1], 0.0], dtype=float)


def rotate_yaw(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([[c, -s], [s, c]], dtype=float)


def lateral_width_regulated_foothold(
    foot: str,
    locked_positions: dict[str, np.ndarray],
    base_position: np.ndarray,
    base_yaw: float,
    initial_base_pos: np.ndarray,
    initial_foot_positions: dict[str, np.ndarray],
    step_delta: np.ndarray,
    yaw_delta: float,
    max_step_length: float,
    correction: float,
    margin: float,
) -> np.ndarray:
    start = locked_positions[foot]
    target = start + foothold_delta_from_layout(
        foot,
        locked_positions,
        step_delta,
        yaw_delta,
        max_step_length,
    )
    if correction <= 0.0:
        return target

    rotation_world_from_base = rotate_yaw(base_yaw)
    nominal_offset_base = initial_foot_positions[foot][0:2] - initial_base_pos[0:2]
    target_offset_base = rotation_world_from_base.T @ (target[0:2] - np.asarray(base_position[0:2], dtype=float))
    lateral_error = target_offset_base[1] - nominal_offset_base[1]
    if abs(lateral_error) <= margin:
        return target

    corrected_offset_base = target_offset_base.copy()
    corrected_offset_base[1] -= float(correction) * (lateral_error - np.sign(lateral_error) * margin)
    corrected_xy = np.asarray(base_position[0:2], dtype=float) + rotation_world_from_base @ corrected_offset_base

    regulated = target.copy()
    regulated[0:2] = clamp_planar_step(start[0:2], corrected_xy, max_step_length)
    return regulated


def clamp_planar_step(start_xy: np.ndarray, target_xy: np.ndarray, max_step_length: float) -> np.ndarray:
    delta = np.asarray(target_xy, dtype=float) - np.asarray(start_xy, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm <= max_step_length:
        return np.asarray(target_xy, dtype=float).copy()
    return np.asarray(start_xy, dtype=float) + delta * (max_step_length / norm)


def foothold_delta_from_layout(
    foot: str,
    foot_positions: dict[str, np.ndarray],
    step_delta: np.ndarray,
    yaw_delta: float,
    max_step_length: float,
) -> np.ndarray:
    delta = np.asarray(step_delta, dtype=float).copy()
    if yaw_delta != 0.0:
        center_xy = np.mean(np.vstack([pos[0:2] for pos in foot_positions.values()]), axis=0)
        offset_xy = foot_positions[foot][0:2] - center_xy
        delta[0:2] += yaw_delta * np.array([-offset_xy[1], offset_xy[0]], dtype=float)
    return trot.limited_planar_delta(delta, max_step_length)


def contact_string(robot: MuJoCoModelInterface, foot_geoms: tuple[str, ...]) -> str:
    return "".join("1" if robot.geom_has_contact(foot) else "0" for foot in foot_geoms)


def has_fallen(robot: MuJoCoModelInterface) -> bool:
    roll, pitch, _ = trot.quat_to_rpy(robot.data.qpos[3:7])
    return bool(robot.data.qpos[2] < 0.18 or abs(roll) > 0.75 or abs(pitch) > 0.75)


def save_rollout(
    path: Path,
    times: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    args: argparse.Namespace,
    segments: list[CommandSegment],
) -> None:
    segment_array = np.asarray([[s.duration, s.vx, s.vy, s.yaw_rate] for s in segments], dtype=float)
    np.savez_compressed(
        path,
        times=times,
        qpos=qpos,
        qvel=qvel,
        fps=float(args.fps),
        preset=args.preset,
        segments=segment_array,
        playback_speed=float(args.playback_speed),
    )


def render_gif(states_path: Path, gif_path: Path, width: int, height: int, fps: float, playback_speed: float) -> None:
    rollout = np.load(states_path, allow_pickle=False)
    qpos = rollout["qpos"]
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = 1.15
    camera.azimuth = 135.0
    camera.elevation = -20.0

    frames: list[Image.Image] = []
    for idx in range(qpos.shape[0]):
        data.qpos[:] = qpos[idx]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        camera.lookat[:] = data.qpos[0:3]
        camera.lookat[2] = 0.20
        renderer.update_scene(data, camera=camera)
        frames.append(Image.fromarray(renderer.render()))

    duration_ms = max(1, int(round(1000.0 / (fps * playback_speed))))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def replay_viewer(states_path: Path, fps: float, playback_speed: float) -> None:
    rollout = np.load(states_path, allow_pickle=False)
    qpos = rollout["qpos"]
    qvel = rollout["qvel"]
    frame_dt = 1.0 / (fps * playback_speed)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for idx in range(qpos.shape[0]):
            if not viewer.is_running():
                break
            frame_start = time.perf_counter()
            data.qpos[:] = qpos[idx]
            data.qvel[:] = qvel[idx]
            mujoco.mj_forward(model, data)
            viewer.sync()
            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)


def absolute_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def prepare_output_path(path: Path) -> Path:
    resolved = absolute_output_path(path)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        probe = resolved.parent / f".write_test_{resolved.name}.tmp"
        probe.write_bytes(b"")
        probe.unlink()
        return resolved
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "mujoco_wbc_project" / resolved.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        print(f"warning: could not create {resolved.parent} ({exc}); using {fallback.parent}")
        return fallback


if __name__ == "__main__":
    main()
