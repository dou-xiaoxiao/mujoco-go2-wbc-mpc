"""Swing foot reference trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SwingReference:
    position: Array
    velocity: Array
    acceleration: Array


def smoothstep(ratio: float) -> tuple[float, float, float]:
    """Return s(r), ds/dr, d2s/dr2 for cubic smoothstep."""

    r = float(np.clip(ratio, 0.0, 1.0))
    s = 3.0 * r * r - 2.0 * r * r * r
    ds = 6.0 * r - 6.0 * r * r
    dds = 6.0 - 12.0 * r
    return s, ds, dds


def swing_foothold_reference(
    initial_position: Array,
    step_delta: Array,
    swing_height: float,
    start_time: float,
    duration: float,
    time_s: float,
) -> SwingReference:
    """Smoothly move a foot from its initial point to a target foothold.

    The horizontal component uses cubic smoothstep. The vertical clearance is
    h * sin(pi * s), where s is the same smoothstep phase. Before the swing
    starts, the reference is the initial point. After the swing finishes, the
    reference is the final foothold with zero velocity and acceleration.
    """

    p0 = np.asarray(initial_position, dtype=float)
    delta = np.asarray(step_delta, dtype=float)

    if time_s <= start_time:
        return SwingReference(position=p0.copy(), velocity=np.zeros(3), acceleration=np.zeros(3))

    if time_s >= start_time + duration:
        return SwingReference(position=p0 + delta, velocity=np.zeros(3), acceleration=np.zeros(3))

    r = (time_s - start_time) / duration
    s, ds_dr, dds_dr2 = smoothstep(r)
    sdot = ds_dr / duration
    sddot = dds_dr2 / (duration * duration)

    position = p0 + delta * s
    velocity = delta * sdot
    acceleration = delta * sddot

    sin_term = np.sin(np.pi * s)
    cos_term = np.cos(np.pi * s)
    position[2] += swing_height * sin_term
    velocity[2] += swing_height * np.pi * cos_term * sdot
    acceleration[2] += swing_height * (
        -np.pi * np.pi * sin_term * sdot * sdot + np.pi * cos_term * sddot
    )

    return SwingReference(position=position, velocity=velocity, acceleration=acceleration)
