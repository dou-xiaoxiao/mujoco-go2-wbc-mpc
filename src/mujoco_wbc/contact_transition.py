"""Contact transition utilities for stance/swing mode switches."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def smoothstep01(value: float) -> float:
    ratio = float(np.clip(value, 0.0, 1.0))
    return ratio * ratio * (3.0 - 2.0 * ratio)


def landing_ramped_force_ref(
    force_ref_all: Array,
    foot_geoms: tuple[str, ...],
    time_s: float,
    touchdown_times_by_foot: dict[str, float],
    ramp_time: float,
) -> Array:
    """Ramp force references from zero after a foot is accepted as stance."""

    if ramp_time <= 0.0:
        return np.asarray(force_ref_all, dtype=float).copy()

    forces = np.asarray(force_ref_all, dtype=float).reshape(len(foot_geoms), 3).copy()
    for foot_id, foot in enumerate(foot_geoms):
        touchdown_time = touchdown_times_by_foot.get(foot)
        if touchdown_time is None:
            continue
        elapsed = time_s - touchdown_time
        if elapsed >= ramp_time:
            continue
        forces[foot_id] *= smoothstep01(elapsed / ramp_time)
    return forces.reshape(-1)


def landing_force_zero_weights(
    foot_geoms: tuple[str, ...],
    time_s: float,
    touchdown_times_by_foot: dict[str, float],
    ramp_time: float,
    zero_weight: float,
) -> Array:
    """Add a temporary WBC penalty that releases contact force after touchdown."""

    weights = np.zeros((len(foot_geoms), 3), dtype=float)
    if ramp_time <= 0.0 or zero_weight <= 0.0:
        return weights.reshape(-1)

    for foot_id, foot in enumerate(foot_geoms):
        touchdown_time = touchdown_times_by_foot.get(foot)
        if touchdown_time is None:
            continue
        elapsed = time_s - touchdown_time
        if elapsed >= ramp_time:
            continue
        release = smoothstep01(elapsed / ramp_time)
        weights[foot_id, :] = zero_weight * (1.0 - release)
    return weights.reshape(-1)


def update_touchdown_hysteresis(
    condition_by_foot: dict[str, bool],
    time_s: float,
    hold_time: float,
    candidate_times_by_foot: dict[str, float],
) -> bool:
    """Return true only after every foot satisfies its touchdown condition long enough."""

    all_ready = True
    for foot, condition in condition_by_foot.items():
        if condition:
            candidate_times_by_foot.setdefault(foot, time_s)
            if time_s - candidate_times_by_foot[foot] < hold_time:
                all_ready = False
        else:
            candidate_times_by_foot.pop(foot, None)
            all_ready = False
    return all_ready

