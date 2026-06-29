"""Launch a viewer and control Go2 with the first static stance WBC."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
FOOT_GEOMS = ("FL", "FR", "RL", "RR")

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import MuJoCoModelInterface, StanceWBCConfig, StanceWBCQP  # noqa: E402


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")
    qpos_ref = robot.q

    controller = StanceWBCQP(StanceWBCConfig(foot_geoms=FOOT_GEOMS))
    dt = float(robot.model.opt.timestep)

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            solution = controller.solve(robot, qpos_ref)
            if solution.status in ("solved", "solved inaccurate"):
                robot.data.ctrl[:] = solution.tau
            mujoco.mj_step(robot.model, robot.data)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
