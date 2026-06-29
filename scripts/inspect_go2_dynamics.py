"""Inspect Unitree Go2 floating-base dynamics quantities for WBC."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
FOOT_GEOMS = ["FL", "FR", "RL", "RR"]

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import MuJoCoModelInterface  # noqa: E402


def names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str]:
    result = []
    for idx in range(count):
        name = mujoco.mj_id2name(model, obj_type, idx)
        result.append(name if name is not None else f"<unnamed_{idx}>")
    return result


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")

    model = robot.model
    data = robot.data

    mass = robot.mass_matrix()
    h = robot.bias_forces()
    bmat = robot.actuation_matrix()
    jc = robot.stacked_geom_jacobian(FOOT_GEOMS)
    jdot_v = robot.stacked_geom_jdot_v(FOOT_GEOMS)

    print(f"model: {MODEL_PATH}")
    print("\n=== Dimensions ===")
    print(f"nq = {robot.nq}")
    print(f"nv = {robot.nv}")
    print(f"nu = {robot.nu}")

    print("\n=== Floating-base convention ===")
    print("qpos[0:3]  = base xyz")
    print("qpos[3:7]  = base quaternion [w, x, y, z]")
    print("qvel[0:3]  = base linear velocity")
    print("qvel[3:6]  = base angular velocity")
    print("qpos[7:]   = 12 joint positions")
    print("qvel[6:]   = 12 joint velocities")

    print("\n=== Joints ===")
    for idx, name in enumerate(names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)):
        qadr = model.jnt_qposadr[idx]
        dadr = model.jnt_dofadr[idx]
        print(f"{idx:02d}: {name:16s} qposadr={qadr:2d} dofadr={dadr:2d}")

    print("\n=== Actuators ===")
    for idx, name in enumerate(names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)):
        print(f"{idx:02d}: {name}")

    print("\n=== Foot geoms ===")
    for foot in FOOT_GEOMS:
        print(f"{foot}: pos={np.round(robot.geom_position(foot), 5).tolist()}")

    print("\n=== Dynamics blocks ===")
    print(f"M shape       = {mass.shape}")
    print(f"h shape       = {h.shape}")
    print(f"B shape       = {bmat.shape}")
    print(f"Jc shape      = {jc.shape}")
    print(f"Jdot_v shape  = {jdot_v.shape}")
    print(f"cond(M)       = {np.linalg.cond(mass):.3e}")
    print(f"||M-M.T||     = {np.linalg.norm(mass - mass.T):.3e}")
    print(f"B base rows   = {np.round(bmat[:6], 5).tolist()}")

    tau = data.ctrl.copy()
    vdot = np.zeros(robot.nv)
    contact_forces = np.zeros(3 * len(FOOT_GEOMS))
    residual = robot.dynamics_residual(vdot, tau, FOOT_GEOMS, contact_forces)

    print("\n=== WBC equation at home, no contact force ===")
    print("residual = M vdot + h - B tau - Jc.T f")
    print(f"||residual|| = {np.linalg.norm(residual):.6f}")
    print(f"residual     = {np.round(residual, 5).tolist()}")

    print("\nEquation form:")
    print("M(q) vdot + h(q, v) = B tau + Jc(q)^T f")


if __name__ == "__main__":
    main()
