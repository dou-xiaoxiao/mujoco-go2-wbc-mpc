"""Solve centroidal force QP, then use its force reference in stance WBC."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
FOOT_GEOMS = ("FL", "FR", "RL", "RR")

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import (  # noqa: E402
    CentroidalForceQP,
    CentroidalForceQPConfig,
    MuJoCoModelInterface,
    StanceWBCConfig,
    StanceWBCQP,
)


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")
    qpos_ref = robot.q

    mpc_solution = CentroidalForceQP(CentroidalForceQPConfig(contact_geoms=FOOT_GEOMS)).solve(robot)
    wbc_solution = StanceWBCQP(StanceWBCConfig(foot_geoms=FOOT_GEOMS)).solve(
        robot,
        qpos_ref,
        force_ref=mpc_solution.contact_forces,
    )

    mpc_forces = mpc_solution.contact_forces.reshape(len(FOOT_GEOMS), 3)
    wbc_forces = wbc_solution.contact_forces.reshape(len(FOOT_GEOMS), 3)

    print("=== Centroidal force reference -> stance WBC ===")
    print(f"centroidal status   = {mpc_solution.status}")
    print(f"WBC status          = {wbc_solution.status}")
    print(f"||WBC dyn residual||    = {np.linalg.norm(wbc_solution.dynamics_residual):.3e}")
    print(f"||WBC stance residual|| = {np.linalg.norm(wbc_solution.stance_residual):.3e}")

    print("\n=== Force comparison, world frame ===")
    for name, mpc_force, wbc_force in zip(FOOT_GEOMS, mpc_forces, wbc_forces):
        diff = wbc_force - mpc_force
        print(
            f"{name}: "
            f"mpc=[{mpc_force[0]:9.4f}, {mpc_force[1]:9.4f}, {mpc_force[2]:9.4f}]  "
            f"wbc=[{wbc_force[0]:9.4f}, {wbc_force[1]:9.4f}, {wbc_force[2]:9.4f}]  "
            f"diff_norm={np.linalg.norm(diff):.3e}"
        )

    print("\nData path:")
    print("centroidal QP contact_forces -> StanceWBCQP.solve(..., force_ref=contact_forces)")


if __name__ == "__main__":
    main()
