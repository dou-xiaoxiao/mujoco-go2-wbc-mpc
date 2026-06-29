"""Solve the first horizon centroidal MPC for Unitree Go2."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
FOOT_GEOMS = ("FL", "FR", "RL", "RR")

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import CentroidalMPC, CentroidalMPCConfig, MuJoCoModelInterface  # noqa: E402


def run_case(name: str, robot: MuJoCoModelInterface, config: CentroidalMPCConfig, contact_schedule: np.ndarray) -> None:
    solver = CentroidalMPC(config)
    com_ref = robot.center_of_mass()
    solution = solver.solve(robot, com_ref, contact_schedule=contact_schedule)

    forces0 = solution.first_contact_forces.reshape(len(FOOT_GEOMS), 3)
    total_force0 = np.sum(forces0, axis=0)
    weight = robot.total_mass() * abs(float(robot.model.opt.gravity[2]))
    active0 = [foot for foot, active in zip(FOOT_GEOMS, contact_schedule[0]) if active]
    swing0 = [foot for foot, active in zip(FOOT_GEOMS, contact_schedule[0]) if not active]

    print(f"\n=== {name} ===")
    print(f"status                 = {solution.status}")
    print(f"objective              = {solution.objective:.6f}")
    print(f"horizon, dt            = {config.horizon_steps}, {config.dt:.3f} s")
    print(f"active feet at k=0     = {active0}")
    print(f"swing feet at k=0      = {swing0}")
    print(f"||dynamics residual||  = {np.linalg.norm(solution.dynamics_residual):.3e}")
    print(f"inertia diag W         = {np.round(np.diag(solution.inertia_world), 6).tolist()}")

    print("\nFirst-step forces, world frame:")
    for foot, active, force in zip(FOOT_GEOMS, contact_schedule[0], forces0):
        print(
            f"{foot}: active={str(bool(active)):5s}  "
            f"f=[{force[0]:9.4f}, {force[1]:9.4f}, {force[2]:9.4f}]"
        )
    print(f"sum first-step force   = {np.round(total_force0, 6).tolist()}")
    print(f"sum_fz / weight        = {total_force0[2] / weight:.9f}")

    print("\nPredicted COM:")
    print(f"initial                = {np.round(solution.states[0, 0:3], 6).tolist()}")
    print(f"final                  = {np.round(solution.states[-1, 0:3], 6).tolist()}")
    print(f"final velocity         = {np.round(solution.states[-1, 3:6], 6).tolist()}")
    print(f"final theta            = {np.round(solution.states[-1, 6:9], 6).tolist()}")
    print(f"final omega            = {np.round(solution.states[-1, 9:12], 6).tolist()}")


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")

    config = CentroidalMPCConfig(contact_geoms=FOOT_GEOMS, horizon_steps=12, dt=0.03)
    all_stance = np.ones((config.horizon_steps, len(FOOT_GEOMS)), dtype=bool)
    fl_swing = all_stance.copy()
    fl_swing[:, 0] = False

    print("=== Horizon centroidal/SRB MPC v1 ===")
    print("state   = [com_pos, com_vel, theta, omega]")
    print("control = [f_FL, f_FR, f_RL, f_RR] at each horizon step")
    print("model   = p[k+1] = p[k] + dt v[k]")
    print("          v[k+1] = v[k] + dt (sum f_i[k] / m + g)")
    print("          theta[k+1] = theta[k] + dt omega[k]")
    print("          omega[k+1] = omega[k] + dt I^-1 sum (p_i - com) x f_i[k]")

    run_case("all-stance standing", robot, config, all_stance)
    run_case("FL swing force disabled", robot, config, fl_swing)


if __name__ == "__main__":
    main()
