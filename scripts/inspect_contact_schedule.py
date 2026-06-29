"""Inspect horizon contact schedules for the FL forward-step example."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
FOOT_GEOMS = ("FL", "FR", "RL", "RR")
SWING_FOOT = "FL"
SWING_START = 1.0
SWING_DURATION = 1.2
HORIZON_STEPS = 10
MPC_DT = 0.03

sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import current_single_leg_phase, single_leg_swing_schedule  # noqa: E402


def main() -> None:
    print("=== Current phase checks ===")
    for current_time in (0.05, 0.20, 0.26):
        phase = current_single_leg_phase(
            "FL",
            current_time=current_time,
            swing_start=0.0,
            swing_duration=0.25,
            touchdown_time=0.25,
        )
        print(f"t={current_time:.2f}s, touchdown_time=0.25 -> {phase}")

    print("\n=== Horizon schedule checks ===")
    for current_time in (0.75, 0.85, 0.95, 1.00, 1.30, 2.10, 2.25):
        schedule = single_leg_swing_schedule(
            FOOT_GEOMS,
            SWING_FOOT,
            current_time=current_time,
            horizon_steps=HORIZON_STEPS,
            dt=MPC_DT,
            swing_start=SWING_START,
            swing_duration=SWING_DURATION,
        )
        print(f"\nt = {current_time:.2f}s")
        print("knot times:", [round(current_time + k * MPC_DT, 2) for k in range(HORIZON_STEPS)])
        for foot_idx, foot in enumerate(FOOT_GEOMS):
            labels = ["S" if active else "-" for active in schedule[:, foot_idx]]
            print(f"{foot}: {' '.join(labels)}")
        print(f"FL swing knots = {sum(not active for active in schedule[:, 0])}")


if __name__ == "__main__":
    main()
