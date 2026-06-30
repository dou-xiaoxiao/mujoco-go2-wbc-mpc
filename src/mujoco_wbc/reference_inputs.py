"""Reference interface expected from an upstream planner or policy.

This module does not generate clever plans. It defines the locomotion reference
shape that the MPC/WBC stack can consume, and checks whether a contact mode is
currently executable by the WBC implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class LocomotionReferenceFrame:
    """One time sample of an upstream locomotion reference."""

    time_s: float
    contact_state: tuple[bool, bool, bool, bool]
    base_position_ref: Array
    base_orientation_ref: Array
    com_position_ref: Array
    com_velocity_ref: Array
    foothold_targets: dict[str, Array]

    @property
    def stance_count(self) -> int:
        return sum(1 for in_contact in self.contact_state if in_contact)

    @property
    def swing_count(self) -> int:
        return len(self.contact_state) - self.stance_count


@dataclass(frozen=True)
class ModeSupportReport:
    """Whether the current WBC code can execute a contact mode."""

    mode_name: str
    contact_state: tuple[bool, bool, bool, bool]
    supported: bool
    required_wbc: str
    reason: str


def classify_contact_mode(
    contact_state: tuple[bool, bool, bool, bool],
    foot_geoms: tuple[str, ...] = ("FL", "FR", "RL", "RR"),
) -> ModeSupportReport:
    """Classify a contact state against the currently implemented WBC modes."""

    if len(contact_state) != len(foot_geoms):
        raise ValueError(f"contact_state must have length {len(foot_geoms)}, got {len(contact_state)}")

    stance_feet = tuple(foot for foot, in_contact in zip(foot_geoms, contact_state) if in_contact)
    swing_feet = tuple(foot for foot, in_contact in zip(foot_geoms, contact_state) if not in_contact)

    if len(swing_feet) == 0:
        return ModeSupportReport(
            mode_name="all-stance",
            contact_state=contact_state,
            supported=True,
            required_wbc="StanceWBCQP",
            reason="all feet are stance constraints",
        )

    if len(swing_feet) == 1 and len(stance_feet) == 3:
        return ModeSupportReport(
            mode_name=f"{swing_feet[0]}-single-swing",
            contact_state=contact_state,
            supported=True,
            required_wbc="SingleLegSwingWBCQP",
            reason="current WBC supports one swing foot and three stance feet",
        )

    if len(stance_feet) == 0:
        return ModeSupportReport(
            mode_name="flight",
            contact_state=contact_state,
            supported=False,
            required_wbc="FlightWBCQP / ballistic base tracking",
            reason="current WBC assumes at least stance contacts for contact-force constraints",
        )

    if len(swing_feet) >= 1 and len(stance_feet) >= 1:
        return ModeSupportReport(
            mode_name=f"{len(swing_feet)}-swing",
            contact_state=contact_state,
            supported=True,
            required_wbc="GeneralContactWBCQP",
            reason="generic WBC supports arbitrary non-flight stance/swing subsets",
        )

    return ModeSupportReport(
        mode_name=f"{len(swing_feet)}-swing",
        contact_state=contact_state,
        supported=False,
        required_wbc="MultiSwingWBCQP",
        reason="current WBC has no generic multi-swing task stack yet",
    )


def named_contact_patterns() -> dict[str, list[tuple[bool, bool, bool, bool]]]:
    """Representative external gait/contact patterns in FL, FR, RL, RR order."""

    return {
        "crawl": [
            (False, True, True, True),
            (True, True, True, True),
            (True, True, True, False),
            (True, True, True, True),
            (True, False, True, True),
            (True, True, True, True),
            (True, True, False, True),
            (True, True, True, True),
        ],
        "trot": [
            (False, True, True, False),
            (True, False, False, True),
        ],
        "pace": [
            (False, True, False, True),
            (True, False, True, False),
        ],
        "bound": [
            (False, False, True, True),
            (True, True, False, False),
        ],
        "pronk": [
            (False, False, False, False),
            (True, True, True, True),
        ],
        "jump": [
            (True, True, True, True),
            (False, False, False, False),
            (True, True, True, True),
        ],
    }


def validate_reference_frame(frame: LocomotionReferenceFrame, foot_geoms: tuple[str, ...] = ("FL", "FR", "RL", "RR")) -> None:
    """Validate shape-level consistency before passing a reference to control."""

    if len(frame.contact_state) != len(foot_geoms):
        raise ValueError(f"contact_state must have length {len(foot_geoms)}")
    if np.asarray(frame.base_position_ref).shape != (3,):
        raise ValueError("base_position_ref must have shape (3,)")
    if np.asarray(frame.base_orientation_ref).shape != (4,):
        raise ValueError("base_orientation_ref must have shape (4,)")
    if np.asarray(frame.com_position_ref).shape != (3,):
        raise ValueError("com_position_ref must have shape (3,)")
    if np.asarray(frame.com_velocity_ref).shape != (3,):
        raise ValueError("com_velocity_ref must have shape (3,)")
    missing = [foot for foot in foot_geoms if foot not in frame.foothold_targets]
    if missing:
        raise ValueError(f"missing foothold targets for feet: {missing}")
    for foot, target in frame.foothold_targets.items():
        if foot not in foot_geoms:
            raise ValueError(f"unknown foot in foothold_targets: {foot}")
        if np.asarray(target).shape != (3,):
            raise ValueError(f"foothold target for {foot} must have shape (3,)")
