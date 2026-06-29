"""Contact schedule utilities for MPC horizons."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def scheduled_swing_contacts(
    foot_geoms: tuple[str, ...],
    swing_windows: list[tuple[str, float, float]],
    current_time: float,
    horizon_steps: int,
    dt: float,
    completed_windows: set[int] | None = None,
) -> Array:
    """Return a horizon schedule for multiple planned single-leg swing windows.

    Each swing window is `(foot_name, start_time, duration)`. `completed_windows`
    can mark windows that touched down early, making that foot stance again for
    all future knots.
    """

    completed_windows = set() if completed_windows is None else completed_windows
    schedule = np.ones((horizon_steps, len(foot_geoms)), dtype=bool)
    foot_to_index = {foot: idx for idx, foot in enumerate(foot_geoms)}

    for window_id, (foot, start_time, duration) in enumerate(swing_windows):
        if foot not in foot_to_index:
            raise ValueError(f"swing foot {foot!r} is not in foot_geoms {foot_geoms!r}")
        if window_id in completed_windows:
            continue

        end_time = start_time + duration
        foot_idx = foot_to_index[foot]
        for step in range(horizon_steps):
            knot_time = current_time + step * dt
            if start_time <= knot_time < end_time:
                schedule[step, foot_idx] = False

    return schedule


def active_swing_window(
    swing_windows: list[tuple[str, float, float]],
    current_time: float,
    completed_windows: set[int] | None = None,
) -> tuple[int, str, float, float] | None:
    completed_windows = set() if completed_windows is None else completed_windows
    for window_id, (foot, start_time, duration) in enumerate(swing_windows):
        if window_id in completed_windows:
            continue
        if start_time <= current_time < start_time + duration:
            return window_id, foot, start_time, duration
    return None


def single_leg_swing_schedule(
    foot_geoms: tuple[str, ...],
    swing_foot: str,
    current_time: float,
    horizon_steps: int,
    dt: float,
    swing_start: float,
    swing_duration: float,
    touchdown_time: float | None = None,
) -> Array:
    """Return schedule[k, foot], where True means stance and False means swing."""

    if swing_foot not in foot_geoms:
        raise ValueError(f"swing_foot {swing_foot!r} is not in foot_geoms {foot_geoms!r}")

    schedule = np.ones((horizon_steps, len(foot_geoms)), dtype=bool)
    swing_index = foot_geoms.index(swing_foot)
    swing_end = swing_start + swing_duration

    for step in range(horizon_steps):
        knot_time = current_time + step * dt
        if touchdown_time is None:
            is_swing = swing_start <= knot_time < swing_end
        else:
            is_swing = swing_start <= knot_time < touchdown_time
        schedule[step, swing_index] = not is_swing

    return schedule


def current_single_leg_phase(
    swing_foot: str,
    current_time: float,
    swing_start: float,
    swing_duration: float,
    touchdown_time: float | None = None,
) -> str:
    del swing_foot
    swing_end = touchdown_time if touchdown_time is not None else swing_start + swing_duration
    if swing_start <= current_time < swing_end:
        return "swing"
    return "stance"
