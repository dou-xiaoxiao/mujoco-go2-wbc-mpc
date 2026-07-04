"""Quick regression checks for the current MPC/WBC control stack."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"

FOOT_GEOMS = ("FL", "FR", "RL", "RR")
STANCE_FEET = ("FR", "RL", "RR")
SWING_FOOT = "FL"

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    CentroidalMPC,
    CentroidalMPCConfig,
    GeneralContactWBCConfig,
    GeneralContactWBCQP,
    MuJoCoModelInterface,
    SingleLegSwingWBCConfig,
    SingleLegSwingWBCQP,
    StanceWBCConfig,
    StanceWBCQP,
    current_single_leg_phase,
    single_leg_swing_schedule,
)


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")

    checks.append(check_actuation_matrix_cache(robot))
    checks.append(check_phase_semantics())
    checks.append(check_stance_wbc(robot))
    checks.append(check_single_leg_swing_wbc(robot))
    checks.append(check_general_contact_wbc_crawl_mode(robot))
    checks.append(check_general_contact_wbc_trot_mode(robot))
    checks.extend(check_mpc_schedules(robot))

    failed = [name for name, ok, _ in checks if not ok]
    print("=== Control stack validation ===")
    for name, ok, details in checks:
        status = "PASS" if ok else "FAIL"
        print(f"{status:4s}  {name:32s} {details}")

    if failed:
        raise SystemExit(f"Validation failed: {failed}")

    print("\nAll control-stack checks passed.")


def check_phase_semantics() -> tuple[str, bool, str]:
    before_touchdown = current_single_leg_phase(
        SWING_FOOT,
        current_time=0.05,
        swing_start=0.0,
        swing_duration=0.25,
        touchdown_time=0.25,
    )
    after_touchdown = current_single_leg_phase(
        SWING_FOOT,
        current_time=0.26,
        swing_start=0.0,
        swing_duration=0.25,
        touchdown_time=0.25,
    )

    schedule = single_leg_swing_schedule(
        FOOT_GEOMS,
        SWING_FOOT,
        current_time=0.0,
        horizon_steps=4,
        dt=0.1,
        swing_start=0.0,
        swing_duration=0.4,
        touchdown_time=0.25,
    )
    fl_column = schedule[:, FOOT_GEOMS.index(SWING_FOOT)].tolist()

    ok = before_touchdown == "swing" and after_touchdown == "stance" and fl_column == [False, False, False, True]
    return "contact phase semantics", ok, f"phase=({before_touchdown}, {after_touchdown}), FL={fl_column}"


def check_actuation_matrix_cache(robot: MuJoCoModelInterface) -> tuple[str, bool, str]:
    ok, max_error = robot.check_actuation_matrix_cache(atol=1.0e-12)
    return "actuation matrix cache", ok, f"max_abs_error={max_error:.2e}"


def check_stance_wbc(robot: MuJoCoModelInterface) -> tuple[str, bool, str]:
    robot.set_keyframe("home")
    controller = StanceWBCQP(StanceWBCConfig(foot_geoms=FOOT_GEOMS))
    solution = controller.solve(robot, robot.q.copy())
    dyn = float(np.linalg.norm(solution.dynamics_residual))
    stance = float(np.linalg.norm(solution.stance_residual))
    ok = is_solved(solution.status) and dyn < 1.0e-6 and stance < 1.0e-6
    return "static stance WBC", ok, f"status={solution.status}, dyn={dyn:.2e}, stance={stance:.2e}"


def check_single_leg_swing_wbc(robot: MuJoCoModelInterface) -> tuple[str, bool, str]:
    robot.set_keyframe("home")
    qpos_ref = robot.q.copy()
    qpos_ref[0:2] += np.array([-0.04, -0.04])

    swing_pos_ref = robot.geom_position(SWING_FOOT)
    swing_pos_ref[2] += 0.03

    controller = SingleLegSwingWBCQP(
        SingleLegSwingWBCConfig(stance_foot_geoms=STANCE_FEET, swing_foot_geom=SWING_FOOT)
    )
    solution = controller.solve(robot, qpos_ref, swing_pos_ref)
    dyn = float(np.linalg.norm(solution.dynamics_residual))
    stance = float(np.linalg.norm(solution.stance_residual))
    swing = float(np.linalg.norm(solution.swing_accel_error))
    ok = is_solved(solution.status) and dyn < 1.0e-5 and stance < 1.0e-5
    return "single-leg swing WBC", ok, f"status={solution.status}, dyn={dyn:.2e}, stance={stance:.2e}, swing={swing:.2e}"


def check_general_contact_wbc_crawl_mode(robot: MuJoCoModelInterface) -> tuple[str, bool, str]:
    robot.set_keyframe("home")
    qpos_ref = robot.q.copy()
    swing_pos_refs = {"FL": robot.geom_position("FL") + np.array([0.0, 0.0, 0.03])}

    controller = GeneralContactWBCQP(
        GeneralContactWBCConfig(
            stance_foot_geoms=("FR", "RL", "RR"),
            swing_foot_geoms=("FL",),
        )
    )
    solution = controller.solve(robot, qpos_ref, swing_pos_refs=swing_pos_refs)
    dyn = float(np.linalg.norm(solution.dynamics_residual))
    stance = float(np.linalg.norm(solution.stance_residual))
    swing = float(np.linalg.norm(solution.swing_accel_error))
    ok = is_solved(solution.status) and dyn < 1.0e-5 and stance < 1.0e-5
    return "general WBC crawl mode", ok, f"status={solution.status}, dyn={dyn:.2e}, stance={stance:.2e}, swing={swing:.2e}"


def check_general_contact_wbc_trot_mode(robot: MuJoCoModelInterface) -> tuple[str, bool, str]:
    robot.set_keyframe("home")
    qpos_ref = robot.q.copy()
    swing_pos_refs = {
        "FL": robot.geom_position("FL") + np.array([0.02, 0.0, 0.03]),
        "RR": robot.geom_position("RR") + np.array([0.02, 0.0, 0.03]),
    }

    controller = GeneralContactWBCQP(
        GeneralContactWBCConfig(
            stance_foot_geoms=("FR", "RL"),
            swing_foot_geoms=("FL", "RR"),
        )
    )
    solution = controller.solve(robot, qpos_ref, swing_pos_refs=swing_pos_refs)
    dyn = float(np.linalg.norm(solution.dynamics_residual))
    stance = float(np.linalg.norm(solution.stance_residual))
    swing = float(np.linalg.norm(solution.swing_accel_error))
    ok = is_solved(solution.status) and dyn < 1.0e-5 and stance < 1.0e-5
    return "general WBC trot mode", ok, f"status={solution.status}, dyn={dyn:.2e}, stance={stance:.2e}, swing={swing:.2e}"


def check_mpc_schedules(robot: MuJoCoModelInterface) -> list[tuple[str, bool, str]]:
    robot.set_keyframe("home")
    config = CentroidalMPCConfig(contact_geoms=FOOT_GEOMS, horizon_steps=10, dt=0.03, normal_force_min=5.0)
    mpc = CentroidalMPC(config)

    all_stance_schedule = np.ones((config.horizon_steps, len(FOOT_GEOMS)), dtype=bool)
    all_stance = mpc.solve(robot, robot.center_of_mass(), contact_schedule=all_stance_schedule)
    all_dyn = float(np.linalg.norm(all_stance.dynamics_residual))
    all_ok = is_solved(all_stance.status) and all_dyn < 1.0e-5

    fl_swing_schedule = all_stance_schedule.copy()
    fl_swing_schedule[:, FOOT_GEOMS.index(SWING_FOOT)] = False
    fl_swing = mpc.solve(robot, robot.center_of_mass(), contact_schedule=fl_swing_schedule)
    fl_dyn = float(np.linalg.norm(fl_swing.dynamics_residual))
    fl_forces = np.max(np.abs(fl_swing.contact_forces[:, FOOT_GEOMS.index(SWING_FOOT), :]))
    fl_ok = is_solved(fl_swing.status) and fl_dyn < 1.0e-5 and fl_forces < 1.0e-6

    return [
        ("SRB-MPC all stance", all_ok, f"status={all_stance.status}, dyn={all_dyn:.2e}"),
        ("SRB-MPC FL swing force-zero", fl_ok, f"status={fl_swing.status}, dyn={fl_dyn:.2e}, max_FL_force={fl_forces:.2e}"),
    ]


def is_solved(status: str) -> bool:
    return status in {"solved", "solved inaccurate"}


if __name__ == "__main__":
    main()
