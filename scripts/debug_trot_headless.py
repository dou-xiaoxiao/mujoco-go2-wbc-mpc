"""Headless trot diagnostic for contact/force symmetry issues."""

from __future__ import annotations

import sys
import importlib.util
import argparse
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a headless trot diagnostic.")
    parser.add_argument("--end-time", type=float, default=6.0)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--start-pair", choices=("FL_RR", "FR_RL"), default="FL_RR")
    parser.add_argument("--start-roll-tol", type=float, default=0.04)
    parser.add_argument("--start-y-tol", type=float, default=0.04)
    parser.add_argument("--max-start-delay", type=float, default=0.50)
    args = parser.parse_args()

    robot = MuJoCoModelInterface(trot.MODEL_PATH)
    robot.set_keyframe("home")

    foot_geoms = trot.FOOT_GEOMS
    vx = 0.012
    vy = 0.0
    yaw_rate = 0.0
    swing_duration = 0.55
    stance_gap = 0.20
    swing_height = 0.035
    max_step_length = 0.035

    home_qpos_ref = robot.q.copy()
    home_com_ref = robot.center_of_mass()
    initial_base_pos = robot.data.qpos[0:3].copy()
    initial_foot_positions = {foot: robot.geom_position(foot) for foot in foot_geoms}
    locked_positions = {foot: pos.copy() for foot, pos in initial_foot_positions.items()}

    windows = trot.build_trot_windows(args.cycles, swing_duration, stance_gap)
    if args.start_pair == "FR_RL":
        windows = [
            trot.TrotWindow(
                swing_feet=trot.TROT_PAIRS[(idx + 1) % 2],
                start_time=window.start_time,
                duration=window.duration,
            )
            for idx, window in enumerate(windows)
        ]
    window_delay_used = np.zeros(len(windows), dtype=float)
    period = 2.0 * (swing_duration + stance_gap)
    nominal_step_delta = trot.limited_planar_delta(
        np.array([vx * period, vy * period, 0.0], dtype=float),
        max_step_length,
    )

    mpc_config = CentroidalMPCConfig(
        contact_geoms=foot_geoms,
        horizon_steps=12,
        dt=0.03,
        normal_force_min=trot.MPC_NORMAL_FORCE_MIN,
    )
    mpc = CentroidalMPC(mpc_config)
    stance_controller = StanceWBCQP(
        StanceWBCConfig(foot_geoms=foot_geoms, weight_force=1.0, kp_stance=100.0, kd_stance=20.0)
    )
    generic_controllers: dict[tuple[tuple[str, ...], tuple[str, ...]], GeneralContactWBCQP] = {}

    active_window_id: int | None = None
    next_window_id = 0
    active_plans: dict[str, trot.SwingPlan] = {}
    next_mpc_update = 0.0
    next_wbc_update = 0.0
    next_log_time = 0.0
    last_tau = np.zeros(robot.nu)
    mpc_force_ref = np.zeros(3 * len(foot_geoms))
    last_wbc_force = np.zeros(0)
    solve_failures = 0

    print("initial feet:", {foot: np.round(pos, 4).tolist() for foot, pos in initial_foot_positions.items()})
    print("nominal step:", np.round(nominal_step_delta, 5).tolist())

    while robot.data.time < args.end_time:
        sim_time = float(robot.data.time)

        if active_window_id is None and next_window_id < len(windows) and sim_time >= windows[next_window_id].start_time:
            if trot.should_delay_next_trot_window(
                robot,
                initial_base_pos,
                args.start_roll_tol,
                args.start_y_tol,
            ) and window_delay_used[next_window_id] < args.max_start_delay:
                delay = min(float(robot.model.opt.timestep), args.max_start_delay - window_delay_used[next_window_id])
                windows = trot.delay_trot_windows(windows, next_window_id, delay)
                window_delay_used[next_window_id:] += delay
            else:
                active_window_id = next_window_id
                window = windows[active_window_id]
                active_plans = {
                    foot: trot.SwingPlan(
                        foot=foot,
                        start_position=locked_positions[foot].copy(),
                        target_position=locked_positions[foot]
                        + trot.foothold_delta_for_foot(
                            foot,
                            initial_foot_positions,
                            nominal_step_delta,
                            yaw_rate * period,
                            max_step_length,
                        ),
                    )
                    for foot in window.swing_feet
                }
                next_wbc_update = sim_time
                next_mpc_update = sim_time
                print(
                    f"START t={sim_time:.3f} swing={window.swing_feet} targets="
                    f"{ {foot: np.round(plan.target_position, 4).tolist() for foot, plan in active_plans.items()} }"
                )

        current_window = windows[active_window_id] if active_window_id is not None else None
        if current_window is not None and trot.should_finish_trot_window(
            robot,
            current_window,
            active_plans,
            sim_time,
            touchdown_z_tol=0.018,
            touchdown_extra_time=0.25,
        ):
            print(
                f"FINISH t={sim_time:.3f} swing={current_window.swing_feet} contacts="
                f"{contact_string(robot, foot_geoms)}"
            )
            for foot in current_window.swing_feet:
                locked_positions[foot] = robot.geom_position(foot).copy()
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
                    swing_height=swing_height,
                    start_time=current_window.start_time,
                    duration=current_window.duration,
                    time_s=sim_time,
                )
                for foot, plan in active_plans.items()
            }

        contact_schedule = trot.trot_contact_schedule(
            windows,
            sim_time,
            mpc_config.horizon_steps,
            mpc_config.dt,
            active_window=current_window,
        )
        command_time = max(0.0, sim_time - windows[0].start_time)
        planned_foot_positions = trot.planned_feet_from_refs(locked_positions, swing_refs)
        base_ref = trot.foot_centered_base_reference(
            home_qpos_ref,
            initial_base_pos,
            initial_foot_positions,
            planned_foot_positions,
            yaw=yaw_rate * command_time,
        )
        base_ref[1] = initial_base_pos[1] + vy * command_time
        com_ref = home_com_ref.copy()
        com_ref[0:2] += base_ref[0:2] - initial_base_pos[0:2]
        com_vel_ref = np.array([vx, vy, 0.0], dtype=float)

        if sim_time >= next_mpc_update:
            mpc_solution = mpc.solve(
                robot,
                com_ref,
                com_velocity_ref=com_vel_ref,
                contact_schedule=contact_schedule,
            )
            mpc_force_ref = mpc_solution.first_contact_forces
            next_mpc_update += trot.MPC_UPDATE_DT

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
                            weight_swing_foot=1400.0,
                            weight_force=1.0,
                            kp_swing=450.0,
                            kd_swing=42.0,
                            kp_stance=100.0,
                            kd_stance=20.0,
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
                last_wbc_force = solution.contact_forces.copy()
            else:
                solve_failures += 1
            next_wbc_update += trot.WBC_UPDATE_DT

        robot.data.ctrl[:] = last_tau
        mujoco.mj_step(robot.model, robot.data)

        if robot.data.time >= next_log_time:
            phase = "stance" if current_window is None else "+".join(current_window.swing_feet)
            roll, pitch, yaw = trot.quat_to_rpy(robot.data.qpos[3:7])
            print(
                "t={:.2f} phase={} base={} rpy={} contacts={} mpc_fz={} wbc_fz={} tau={:.1f} fails={}".format(
                    robot.data.time,
                    phase,
                    np.round(robot.data.qpos[0:3], 3).tolist(),
                    np.round([roll, pitch, yaw], 3).tolist(),
                    contact_string(robot, foot_geoms),
                    np.round(mpc_force_ref.reshape(len(foot_geoms), 3)[:, 2], 1).tolist(),
                    np.round(last_wbc_force.reshape(-1, 3)[:, 2], 1).tolist() if last_wbc_force.size else [],
                    float(np.max(np.abs(last_tau))),
                    solve_failures,
                )
            )
            next_log_time += 0.25


def contact_string(robot: MuJoCoModelInterface, foot_geoms: tuple[str, ...]) -> str:
    return "".join("1" if robot.geom_has_contact(foot) else "0" for foot in foot_geoms)


if __name__ == "__main__":
    main()
