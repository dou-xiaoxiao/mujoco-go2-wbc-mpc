"""Inspect which external gait contact modes the current WBC can execute."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mujoco_wbc import classify_contact_mode, named_contact_patterns  # noqa: E402


FOOT_GEOMS = ("FL", "FR", "RL", "RR")


def main() -> None:
    print("=== External reference contact-mode support ===")
    print(f"foot order: {FOOT_GEOMS}")
    print()

    for gait_name, patterns in named_contact_patterns().items():
        print(f"[{gait_name}]")
        for idx, contact_state in enumerate(patterns):
            report = classify_contact_mode(contact_state, FOOT_GEOMS)
            flags = " ".join(
                f"{foot}:{'S' if in_contact else '-'}"
                for foot, in_contact in zip(FOOT_GEOMS, contact_state)
            )
            support = "OK" if report.supported else "NEEDS"
            print(
                f"  {idx:02d}  {flags}  {support:5s}  "
                f"{report.required_wbc}  ({report.reason})"
            )
        print()


if __name__ == "__main__":
    main()
