"""Use the first horizon centroidal MPC force sample as WBC force reference."""

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
BODY_SHIFT_XY = np.array([-0.04, -0.04])

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    CentroidalMPC,
    CentroidalMPCConfig,
    MuJoCoModelInterface,
    SingleLegSwingWBCConfig,
    SingleLegSwingWBCQP,
    StanceWBCConfig,
    StanceWBCQP,
)


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")

    mpc_config = CentroidalMPCConfig(contact_geoms=FOOT_GEOMS, horizon_steps=12, dt=0.03)
    mpc = CentroidalMPC(mpc_config)

    all_stance_schedule = np.ones((mpc_config.horizon_steps, len(FOOT_GEOMS)), dtype=bool)
    all_stance_mpc = mpc.solve(robot, robot.center_of_mass(), contact_schedule=all_stance_schedule)
    stance_wbc = StanceWBCQP(StanceWBCConfig(foot_geoms=FOOT_GEOMS, weight_force=1.0)).solve(
        robot,
        robot.q,
        force_ref=all_stance_mpc.first_contact_forces,
    )

    fl_swing_schedule = all_stance_schedule.copy()
    fl_swing_schedule[:, 0] = False
    fl_swing_mpc = mpc.solve(robot, robot.center_of_mass(), contact_schedule=fl_swing_schedule)
    mpc_stance_force_ref = fl_swing_mpc.contact_forces[0, 1:, :].reshape(-1)

    qpos_ref = robot.q
    qpos_ref[0:2] += BODY_SHIFT_XY
    swing_pos_ref = robot.geom_position(SWING_FOOT)
    swing_pos_ref[2] += 0.03
    swing_wbc = SingleLegSwingWBCQP(
        SingleLegSwingWBCConfig(stance_foot_geoms=STANCE_FEET, swing_foot_geom=SWING_FOOT, weight_force=1.0)
    ).solve(
        robot,
        qpos_ref,
        swing_pos_ref,
        force_ref=mpc_stance_force_ref,
    )

    print("=== Horizon MPC first force sample -> WBC ===")
    print(f"all-stance MPC status   = {all_stance_mpc.status}")
    print(f"all-stance WBC status   = {stance_wbc.status}")
    print(f"FL-swing MPC status     = {fl_swing_mpc.status}")
    print(f"FL-swing WBC status     = {swing_wbc.status}")

    print("\n=== All-stance force comparison ===")
    print_force_comparison(
        FOOT_GEOMS,
        all_stance_mpc.first_contact_forces.reshape(len(FOOT_GEOMS), 3),
        stance_wbc.contact_forces.reshape(len(FOOT_GEOMS), 3),
    )
    print(f"||stance WBC dyn residual||   = {np.linalg.norm(stance_wbc.dynamics_residual):.3e}")
    print(f"||stance WBC contact residual|| = {np.linalg.norm(stance_wbc.stance_residual):.3e}")

    print("\n=== FL-swing force comparison ===")
    print_force_comparison(
        STANCE_FEET,
        fl_swing_mpc.contact_forces[0, 1:, :],
        swing_wbc.contact_forces.reshape(len(STANCE_FEET), 3),
    )
    print(f"||swing WBC dyn residual||    = {np.linalg.norm(swing_wbc.dynamics_residual):.3e}")
    print(f"||swing WBC contact residual|| = {np.linalg.norm(swing_wbc.stance_residual):.3e}")
    print(f"||swing accel error||         = {np.linalg.norm(swing_wbc.swing_accel_error):.3e}")


def print_force_comparison(names: tuple[str, ...], mpc_forces: np.ndarray, wbc_forces: np.ndarray) -> None:
    for name, mpc_force, wbc_force in zip(names, mpc_forces, wbc_forces):
        diff = wbc_force - mpc_force
        print(
            f"{name}: "
            f"mpc=[{mpc_force[0]:9.4f}, {mpc_force[1]:9.4f}, {mpc_force[2]:9.4f}]  "
            f"wbc=[{wbc_force[0]:9.4f}, {wbc_force[1]:9.4f}, {wbc_force[2]:9.4f}]  "
            f"diff_norm={np.linalg.norm(diff):.3e}"
        )


if __name__ == "__main__":
    main()
