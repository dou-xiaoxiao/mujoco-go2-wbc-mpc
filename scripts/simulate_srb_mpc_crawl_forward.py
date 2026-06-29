"""Run one commanded forward crawl cycle with SRB-MPC force references."""

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
COMMAND_VX = 0.004
COMMAND_VY = 0.0
SWING_START = 1.0
SWING_DURATION = 1.2
SWING_GAP = 0.8
SWING_HEIGHT = 0.04
MAX_STEP_LENGTH = 0.035
TOUCHDOWN_Z_TOL = 0.02
MPC_NORMAL_FORCE_MIN = 5.0
MPC_UPDATE_DT = 0.03
PRE_SHIFT_TIME = 0.6
SUPPORT_CENTROID_RATIO = 0.85

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    CentroidalMPC,
    CentroidalMPCConfig,
    CrawlCommand,
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

    planner = CrawlGaitPlanner(
        CrawlGaitConfig(
            foot_geoms=FOOT_GEOMS,
            sequence=CRAWL_SEQUENCE,
            first_swing_start=SWING_START,
            swing_duration=SWING_DURATION,
            swing_gap=SWING_GAP,
            swing_height=SWING_HEIGHT,
            command=CrawlCommand(vx=COMMAND_VX, vy=COMMAND_VY),
            max_step_length=MAX_STEP_LENGTH,
            pre_shift_time=PRE_SHIFT_TIME,
            support_centroid_ratio=SUPPORT_CENTROID_RATIO,
        )
    )
    target_footholds = planner.target_footholds(initial_foot_positions)
    swing_windows = planner.swing_windows()

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
    duration = swing_windows[-1][1] + swing_windows[-1][2] + 1.0
    steps = int(duration / dt)

    completed_windows: set[int] = set()
    active_window_id: int | None = None
    next_window_id = 0
    locked_foot_positions = {foot: pos.copy() for foot, pos in initial_foot_positions.items()}
    touchdown_times: dict[str, float] = {}
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
    max_swing_knots = {foot: 0 for foot in FOOT_GEOMS}
    final_body_ref_xy = nominal_body_xy.copy()

    for _ in range(steps):
        sim_time = float(robot.data.time)
        if active_window_id is None and planner.should_start_window(sim_time, next_window_id):
            active_window_id = next_window_id

        current_window = planner.window_by_id(active_window_id)

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

        refs = planner.reference_bundle(
            home_qpos_ref,
            home_com_ref,
            nominal_body_xy,
            locked_foot_positions,
            sim_time,
            active_window_id,
            next_window_id,
        )
        final_body_ref_xy = refs.base_position_ref[0:2].copy()

        if sim_time >= next_mpc_update:
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
            for foot_idx, foot in enumerate(FOOT_GEOMS):
                max_swing_knots[foot] = max(max_swing_knots[foot], int(np.count_nonzero(~schedule[:, foot_idx])))
            next_mpc_update += MPC_UPDATE_DT

        qpos_ref = home_qpos_ref.copy()
        qpos_ref[0:3] = refs.base_position_ref
        qpos_ref[3:7] = refs.base_orientation_ref

        if current_window is None:
            solution = stance_controller.solve(
                robot,
                qpos_ref,
                force_ref=mpc_force_ref,
                stance_pos_refs=locked_foot_positions,
            )
            target_pos = None
        else:
            _, foot, start_time, swing_duration = current_window
            ref = planner.swing_reference(
                foot,
                initial_foot_positions,
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
                stance_pos_refs={stance_foot: locked_foot_positions[stance_foot] for stance_foot in FOOT_GEOMS if stance_foot != foot},
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

    print("=== SRB-MPC + WBC commanded forward crawl smoke test ===")
    print(f"sequence              = {CRAWL_SEQUENCE}")
    print(f"command vx/vy         = {COMMAND_VX:.4f} / {COMMAND_VY:.4f} m/s")
    print(f"cycle duration        = {planner.cycle_duration():.3f} s")
    print(f"planned step delta    = {np.round(planner.step_delta(), 5).tolist()} m")
    print(f"duration              = {duration:.3f} s")
    print(f"touchdown times       = {touchdown_times}")
    print(f"max swing knots       = {max_swing_knots}")
    for foot in CRAWL_SEQUENCE:
        final_pos = robot.geom_position(foot)
        print(
            f"{foot}: final={np.round(final_pos, 5).tolist()} "
            f"target={np.round(target_footholds[foot], 5).tolist()} "
            f"error={np.round(final_pos - target_footholds[foot], 5).tolist()}"
        )
    print(f"initial base pos      = {np.round(initial_base_pos, 5).tolist()}")
    print(f"final base pos        = {np.round(final_base_pos, 5).tolist()}")
    print(f"base displacement     = {np.round(final_base_pos - initial_base_pos, 5).tolist()}")
    print(f"final body xy ref     = {np.round(final_body_ref_xy, 5).tolist()}")
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
