"""Inspect the quasi-static crawl planner outputs without running dynamics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"

FOOT_GEOMS = ("FL", "FR", "RL", "RR")

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import CrawlGaitConfig, CrawlGaitPlanner, MuJoCoModelInterface  # noqa: E402


def main() -> None:
    robot = MuJoCoModelInterface(MODEL_PATH)
    robot.set_keyframe("home")

    planner = CrawlGaitPlanner(
        CrawlGaitConfig(
            foot_geoms=FOOT_GEOMS,
            sequence=("FL", "RR", "FR", "RL"),
            step_delta=(0.0, 0.0, 0.0),
        )
    )
    locked_feet = {foot: robot.geom_position(foot) for foot in FOOT_GEOMS}
    nominal_body_xy = robot.data.qpos[0:2].copy()

    print("=== Crawl planner windows ===")
    for idx, window in enumerate(planner.windows):
        stance = planner.stance_feet_for_swing(window.foot)
        target_xy = planner.body_xy_reference(nominal_body_xy, locked_feet, window.start_time, idx, idx)
        print(
            f"{idx}: swing={window.foot} start={window.start_time:.2f}s "
            f"end={window.end_time:.2f}s stance={stance} body_xy={np.round(target_xy, 5).tolist()}"
        )

    print("\n=== Horizon contact schedule samples ===")
    for time_s in (0.5, 0.9, 1.0, 1.6, 2.3, 3.0):
        active = planner.active_window_id(time_s)
        schedule = planner.contact_schedule(
            current_time=time_s,
            horizon_steps=8,
            dt=0.1,
            active_window_id=active,
        )
        print(f"\nt={time_s:.2f}s active={active}")
        for foot_idx, foot in enumerate(FOOT_GEOMS):
            flags = ["S" if active_contact else "-" for active_contact in schedule[:, foot_idx]]
            print(f"{foot}: {' '.join(flags)}")


if __name__ == "__main__":
    main()
