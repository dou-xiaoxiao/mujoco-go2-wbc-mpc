"""Run a headless slow four-leg step sequence with WBC."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
ALL_FEET = ("FL", "FR", "RL", "RR")
LEG_SEQUENCE = ("FL", "RR", "FR", "RL")
PRE_SHIFT_DURATION = 1.0
SWING_DURATION = 1.2
HOLD_DURATION = 0.8
SWING_HEIGHT = 0.05
STEP_DELTA = np.array([0.03, 0.0, 0.0])
TOUCHDOWN_Z_TOL = 0.012
BODY_SHIFT_MAG = 0.04
BODY_SHIFT_BY_LEG = {
    "FL": np.array([-BODY_SHIFT_MAG, -BODY_SHIFT_MAG]),
    "RR": np.array([BODY_SHIFT_MAG, BODY_SHIFT_MAG]),
    "FR": np.array([-BODY_SHIFT_MAG, BODY_SHIFT_MAG]),
    "RL": np.array([BODY_SHIFT_MAG, -BODY_SHIFT_MAG]),
}

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    MuJoCoModelInterface,
    SingleLegSwingWBCConfig,
    SingleLegSwingWBCQP,
    StanceWBCConfig,
    StanceWBCQP,
    swing_foothold_reference,
)


@dataclass
class StepPhase:
    leg: str
    initial_foot_pos: np.ndarray
    foothold: np.ndarray
    start_time: float


def stance_feet_for(swing_leg: str) -> tuple[str, ...]:
    return tuple(foot for foot in ALL_FEET if foot != swing_leg)


def body_shift_for_leg(robot: MuJoCoModelInterface, leg: str) -> np.ndarray:
    del robot
    return BODY_SHIFT_BY_LEG[leg]


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")
    qpos_home = robot.q

    stance_controller = StanceWBCQP(StanceWBCConfig(foot_geoms=ALL_FEET))
    swing_controllers = {
        leg: SingleLegSwingWBCQP(
            SingleLegSwingWBCConfig(
                stance_foot_geoms=stance_feet_for(leg),
                swing_foot_geom=leg,
                weight_swing_foot=1400.0,
                kp_swing=450.0,
                kd_swing=42.0,
            )
        )
        for leg in ALL_FEET
    }

    dt = float(robot.model.opt.timestep)
    duration = (PRE_SHIFT_DURATION + SWING_DURATION + HOLD_DURATION + 0.8) * len(LEG_SEQUENCE)
    steps = int(duration / dt)

    active_phase: StepPhase | None = None
    leg_index = 0
    mode = "pre-shift"
    mode_start_time = 0.0
    completed_footholds: dict[str, np.ndarray] = {}
    max_swing_pos_error = 0.0
    max_dyn_residual = 0.0
    max_stance_residual = 0.0
    max_tau = 0.0
    failed_statuses: dict[str, int] = {}
    nominal_base_xy = qpos_home[0:2].copy()

    for _ in range(steps):
        sim_time = float(robot.data.time)
        if leg_index >= len(LEG_SEQUENCE):
            break
        leg = LEG_SEQUENCE[leg_index]
        mode_time = sim_time - mode_start_time

        qpos_ref = qpos_home.copy()
        qpos_ref[0:2] = nominal_base_xy + body_shift_for_leg(robot, leg)

        if mode == "pre-shift":
            active_phase = None
            solution = stance_controller.solve(robot, qpos_ref)
            if mode_time >= PRE_SHIFT_DURATION:
                initial_pos = robot.geom_position(leg)
                active_phase = StepPhase(
                    leg=leg,
                    initial_foot_pos=initial_pos,
                    foothold=initial_pos + STEP_DELTA,
                    start_time=sim_time,
                )
                mode = "swing"
                mode_start_time = sim_time

        elif mode == "swing":
            if active_phase is None or active_phase.leg != leg:
                initial_pos = robot.geom_position(leg)
                active_phase = StepPhase(
                    leg=leg,
                    initial_foot_pos=initial_pos,
                    foothold=initial_pos + STEP_DELTA,
                    start_time=sim_time,
                )

            ref = swing_foothold_reference(
                initial_position=active_phase.initial_foot_pos,
                step_delta=STEP_DELTA,
                swing_height=SWING_HEIGHT,
                start_time=active_phase.start_time,
                duration=SWING_DURATION,
                time_s=sim_time,
            )
            solution = swing_controllers[leg].solve(robot, qpos_ref, ref.position, ref.velocity, ref.acceleration)
            swing_error = robot.geom_position(leg) - ref.position
            max_swing_pos_error = max(max_swing_pos_error, float(np.linalg.norm(swing_error)))

            swing_is_done = sim_time >= active_phase.start_time + SWING_DURATION
            foot_is_near_ground = robot.geom_position(leg)[2] <= active_phase.foothold[2] + TOUCHDOWN_Z_TOL
            if swing_is_done and foot_is_near_ground:
                completed_footholds[leg] = active_phase.foothold
                active_phase = None
                mode = "hold"
                mode_start_time = sim_time

        else:
            solution = stance_controller.solve(robot, qpos_ref)
            if mode_time >= HOLD_DURATION:
                leg_index += 1
                nominal_base_xy = robot.data.qpos[0:2].copy()
                mode = "pre-shift"
                mode_start_time = sim_time
                active_phase = None

        if solution.status not in ("solved", "solved inaccurate"):
            failed_statuses[solution.status] = failed_statuses.get(solution.status, 0) + 1
            robot.data.ctrl[:] = 0.0
        else:
            robot.data.ctrl[:] = solution.tau

        mujoco.mj_step(robot.model, robot.data)

        max_dyn_residual = max(max_dyn_residual, float(np.linalg.norm(solution.dynamics_residual)))
        max_stance_residual = max(max_stance_residual, float(np.linalg.norm(solution.stance_residual)))
        max_tau = max(max_tau, float(np.max(np.abs(solution.tau))))

    final_base_pos = robot.data.qpos[0:3].copy()

    print("=== Slow step-sequence WBC smoke test ===")
    print(f"sequence              = {LEG_SEQUENCE}")
    print(f"duration budget       = {duration:.3f} s")
    print(f"simulated time        = {robot.data.time:.3f} s")
    print(f"completed legs        = {leg_index}/{len(LEG_SEQUENCE)}")
    print(f"step delta            = {STEP_DELTA.tolist()} m")
    print(f"swing height          = {SWING_HEIGHT:.3f} m")
    print(f"final base pos        = {np.round(final_base_pos, 5).tolist()}")
    print(f"max swing pos error   = {max_swing_pos_error:.3e} m")
    print(f"max |tau|             = {max_tau:.3e} Nm")
    print(f"max dyn residual      = {max_dyn_residual:.3e}")
    print(f"max stance residual   = {max_stance_residual:.3e}")
    print(f"failed QP statuses    = {failed_statuses}")
    print("\n=== Final foot positions ===")
    for leg in ALL_FEET:
        pos = robot.geom_position(leg)
        target = completed_footholds.get(leg)
        if target is None:
            print(f"{leg}: pos={np.round(pos, 5).tolist()} target=<not stepped>")
        else:
            err = pos - target
            print(f"{leg}: pos={np.round(pos, 5).tolist()} target={np.round(target, 5).tolist()} err={np.round(err, 5).tolist()}")


if __name__ == "__main__":
    main()
