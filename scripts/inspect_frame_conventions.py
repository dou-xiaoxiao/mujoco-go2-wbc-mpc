"""Print Go2 world/base frame helpers and foot coordinates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
FOOT_GEOMS = ("FL", "FR", "RL", "RR")

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import MuJoCoModelInterface  # noqa: E402


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")

    r_wb = robot.base_rotation_world_from_base()
    r_bw = robot.base_rotation_base_from_world()
    identity_error = np.linalg.norm(r_wb @ r_bw - np.eye(3))

    print("=== Base frame ===")
    print(f"base position p_WB = {np.round(robot.base_position(), 5).tolist()}")
    print(f"R_WB =\n{np.round(r_wb, 5)}")
    print(f"R_BW = R_WB.T\n{np.round(r_bw, 5)}")
    print(f"||R_WB R_BW - I|| = {identity_error:.3e}")

    print("\n=== Foot positions ===")
    for name in FOOT_GEOMS:
        pos_w = robot.geom_position(name)
        pos_b = robot.geom_position_in_base(name)
        roundtrip = robot.base_point_to_world(pos_b)
        err = np.linalg.norm(roundtrip - pos_w)
        print(f"{name}: p_WF={np.round(pos_w, 5).tolist()}  p_BF={np.round(pos_b, 5).tolist()}  roundtrip={err:.3e}")

    print("\nConventions:")
    print("R_WB maps base-frame vectors to world-frame vectors.")
    print("R_BW maps world-frame vectors to base-frame vectors.")
    print("mj_jacGeom jacp maps qvel to world-frame foot point velocity.")
    print("contact forces are [fx, fy, fz] in the world frame.")


if __name__ == "__main__":
    main()
