"""Run a headless FL forward-step test with three stance feet."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
STANCE_FEET = ("FR", "RL", "RR")
SWING_FOOT = "FL"
SWING_START = 1.0
SWING_DURATION = 1.2
SWING_HEIGHT = 0.06
STEP_DELTA = np.array([0.05, 0.0, 0.0])
BODY_SHIFT_XY = np.array([-0.04, -0.04])
TOUCHDOWN_Z_TOL = 0.012

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    MuJoCoModelInterface,
    SingleLegSwingWBCConfig,
    SingleLegSwingWBCQP,
    StanceWBCConfig,
    StanceWBCQP,
    swing_foothold_reference,
)


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")
    qpos_ref = robot.q
    shifted_qpos_ref = qpos_ref.copy()
    shifted_qpos_ref[0:2] += BODY_SHIFT_XY

    initial_swing_pos = robot.geom_position(SWING_FOOT)
    foothold = initial_swing_pos + STEP_DELTA

    stance_controller = StanceWBCQP(StanceWBCConfig(foot_geoms=("FL", "FR", "RL", "RR")))
    swing_controller = SingleLegSwingWBCQP(
        SingleLegSwingWBCConfig(
            stance_foot_geoms=STANCE_FEET,
            swing_foot_geom=SWING_FOOT,
            weight_swing_foot=1600.0,
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
    max_swing_pos_error = 0.0
    max_tau = 0.0
    failed_statuses: dict[str, int] = {}

    for _ in range(steps):
        sim_time = float(robot.data.time)
        foot_pos = robot.geom_position(SWING_FOOT)
        swing_is_done = sim_time >= SWING_START + SWING_DURATION
        foot_is_near_ground = foot_pos[2] <= foothold[2] + TOUCHDOWN_Z_TOL
        if touchdown_time is None and swing_is_done and foot_is_near_ground:
            touchdown_time = sim_time

        if sim_time < SWING_START or touchdown_time is not None:
            target_pos = initial_swing_pos.copy()
            solution = stance_controller.solve(robot, shifted_qpos_ref)
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
            solution = swing_controller.solve(robot, shifted_qpos_ref, ref.position, ref.velocity, ref.acceleration)

        if solution.status not in ("solved", "solved inaccurate"):
            failed_statuses[solution.status] = failed_statuses.get(solution.status, 0) + 1
            robot.data.ctrl[:] = 0.0
        else:
            robot.data.ctrl[:] = solution.tau

        mujoco.mj_step(robot.model, robot.data)

        if touchdown_time is None and sim_time >= SWING_START:
            swing_error = robot.geom_position(SWING_FOOT) - target_pos
            max_swing_pos_error = max(max_swing_pos_error, float(np.linalg.norm(swing_error)))
        max_dyn_residual = max(max_dyn_residual, float(np.linalg.norm(solution.dynamics_residual)))
        max_stance_residual = max(max_stance_residual, float(np.linalg.norm(solution.stance_residual)))
        max_tau = max(max_tau, float(np.max(np.abs(solution.tau))))

    final_swing_pos = robot.geom_position(SWING_FOOT)
    final_base_pos = robot.data.qpos[0:3].copy()
    final_foothold_error = final_swing_pos - foothold
    final_base_drift = final_base_pos - shifted_qpos_ref[0:3]

    print("=== Single-leg forward-step WBC smoke test ===")
    print(f"duration              = {duration:.3f} s")
    print(f"stance feet           = {STANCE_FEET}")
    print(f"swing foot            = {SWING_FOOT}")
    print(f"body shift xy         = {BODY_SHIFT_XY.tolist()} m")
    print(f"step delta            = {STEP_DELTA.tolist()} m")
    print(f"swing height          = {SWING_HEIGHT:.3f} m")
    print(f"touchdown time        = {touchdown_time}")
    print(f"initial swing pos     = {np.round(initial_swing_pos, 5).tolist()}")
    print(f"target foothold       = {np.round(foothold, 5).tolist()}")
    print(f"final swing pos       = {np.round(final_swing_pos, 5).tolist()}")
    print(f"final foothold error  = {np.round(final_foothold_error, 5).tolist()}")
    print(f"final base pos        = {np.round(final_base_pos, 5).tolist()}")
    print(f"final base drift      = {np.round(final_base_drift, 5).tolist()}")
    print(f"max swing pos error   = {max_swing_pos_error:.3e} m")
    print(f"max |tau|             = {max_tau:.3e} Nm")
    print(f"max dyn residual      = {max_dyn_residual:.3e}")
    print(f"max stance residual   = {max_stance_residual:.3e}")
    print(f"failed QP statuses    = {failed_statuses}")


if __name__ == "__main__":
    main()
