"""Simple gait and reference planners above MPC/WBC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_schedule import scheduled_swing_contacts
from .support_polygon import body_reference_from_support
from .swing_trajectory import SwingReference, smoothstep, swing_foothold_reference


Array = np.ndarray


@dataclass(frozen=True)
class CrawlCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0


@dataclass(frozen=True)
class CrawlGaitConfig:
    foot_geoms: tuple[str, ...] = ("FL", "FR", "RL", "RR")
    sequence: tuple[str, ...] = ("FL", "RR", "FR", "RL")
    first_swing_start: float = 1.0
    swing_duration: float = 1.2
    swing_gap: float = 0.8
    swing_height: float = 0.035
    step_delta: Array | tuple[float, float, float] = (0.0, 0.0, 0.0)
    pre_shift_time: float = 0.6
    support_centroid_ratio: float = 0.85
    command: CrawlCommand | None = None
    max_step_length: float = 0.04


@dataclass(frozen=True)
class SwingWindow:
    foot: str
    start_time: float
    duration: float

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def as_tuple(self) -> tuple[str, float, float]:
        return self.foot, self.start_time, self.duration


class CrawlGaitPlanner:
    """Generate quasi-static crawl references for the MPC/WBC stack.

    This planner intentionally does not solve dynamics. It only answers the
    upper-layer task questions: which leg swings, where the body reference
    should move before that swing, and what foothold/swing reference to give
    the lower controllers.
    """

    def __init__(self, config: CrawlGaitConfig | None = None):
        self.config = config or CrawlGaitConfig()
        self._validate_config()
        self.windows = self._build_windows()

    def swing_windows(self) -> list[tuple[str, float, float]]:
        return [window.as_tuple() for window in self.windows]

    def active_window_id(
        self,
        time_s: float,
        completed_windows: set[int] | None = None,
    ) -> int | None:
        completed_windows = set() if completed_windows is None else completed_windows
        for window_id, window in enumerate(self.windows):
            if window_id in completed_windows:
                continue
            if window.start_time <= time_s < window.end_time:
                return window_id
        return None

    def should_start_window(self, time_s: float, next_window_id: int) -> bool:
        if next_window_id >= len(self.windows):
            return False
        return time_s >= self.windows[next_window_id].start_time

    def window_by_id(self, window_id: int | None) -> tuple[int, str, float, float] | None:
        if window_id is None:
            return None
        window = self.windows[window_id]
        return window_id, window.foot, window.start_time, window.duration

    def stance_feet_for_swing(self, swing_foot: str | None) -> tuple[str, ...]:
        if swing_foot is None:
            return self.config.foot_geoms
        return tuple(foot for foot in self.config.foot_geoms if foot != swing_foot)

    def target_footholds(self, initial_foot_positions: dict[str, Array]) -> dict[str, Array]:
        delta = self.step_delta()
        return {foot: np.asarray(pos, dtype=float) + delta for foot, pos in initial_foot_positions.items()}

    def step_delta(self) -> Array:
        if self.config.command is None:
            return np.asarray(self.config.step_delta, dtype=float)

        cycle_time = self.cycle_duration()
        command = self.config.command
        delta = np.array([command.vx * cycle_time, command.vy * cycle_time, 0.0], dtype=float)
        planar_norm = float(np.linalg.norm(delta[0:2]))
        if planar_norm > self.config.max_step_length:
            delta[0:2] *= self.config.max_step_length / planar_norm
        return delta

    def cycle_duration(self) -> float:
        if not self.config.sequence:
            return 0.0
        return len(self.config.sequence) * (self.config.swing_duration + self.config.swing_gap)

    def contact_schedule(
        self,
        current_time: float,
        horizon_steps: int,
        dt: float,
        completed_windows: set[int] | None = None,
        active_window_id: int | None = None,
    ) -> Array:
        visible_windows = self.swing_windows() if active_window_id is None else self.swing_windows()[: active_window_id + 1]
        schedule = scheduled_swing_contacts(
            self.config.foot_geoms,
            visible_windows,
            current_time=current_time,
            horizon_steps=horizon_steps,
            dt=dt,
            completed_windows=completed_windows,
        )
        if active_window_id is not None:
            active_foot = self.windows[active_window_id].foot
            schedule[:, self.config.foot_geoms.index(active_foot)] = False
        return schedule

    def body_xy_reference(
        self,
        nominal_body_xy: Array,
        locked_foot_positions: dict[str, Array],
        time_s: float,
        active_window_id: int | None,
        next_window_id: int,
    ) -> Array:
        nominal = np.asarray(nominal_body_xy, dtype=float)
        target = self._target_body_xy_for_window(nominal, locked_foot_positions, active_window_id, next_window_id)

        if active_window_id is not None:
            return target
        if next_window_id >= len(self.windows):
            all_support_points_xy = np.vstack([locked_foot_positions[foot][0:2] for foot in self.config.foot_geoms])
            return body_reference_from_support(
                nominal,
                all_support_points_xy,
                centroid_ratio=self.config.support_centroid_ratio,
            )

        next_window = self.windows[next_window_id]
        shift_start = next_window.start_time - self.config.pre_shift_time
        if time_s < shift_start:
            return nominal.copy()
        if time_s >= next_window.start_time:
            return target

        ratio = (time_s - shift_start) / self.config.pre_shift_time
        s, _, _ = smoothstep(ratio)
        return nominal + s * (target - nominal)

    def swing_reference(
        self,
        foot: str,
        initial_foot_positions: dict[str, Array],
        start_time: float,
        duration: float,
        time_s: float,
    ) -> SwingReference:
        return swing_foothold_reference(
            initial_position=initial_foot_positions[foot],
            step_delta=self.step_delta(),
            swing_height=self.config.swing_height,
            start_time=start_time,
            duration=duration,
            time_s=time_s,
        )

    def _target_body_xy_for_window(
        self,
        nominal_body_xy: Array,
        locked_foot_positions: dict[str, Array],
        active_window_id: int | None,
        next_window_id: int,
    ) -> Array:
        if active_window_id is not None:
            swing_foot = self.windows[active_window_id].foot
        elif next_window_id < len(self.windows):
            swing_foot = self.windows[next_window_id].foot
        else:
            return np.asarray(nominal_body_xy, dtype=float).copy()

        stance_feet = self.stance_feet_for_swing(swing_foot)
        support_points_xy = np.vstack([locked_foot_positions[foot][0:2] for foot in stance_feet])
        return body_reference_from_support(
            nominal_body_xy,
            support_points_xy,
            centroid_ratio=self.config.support_centroid_ratio,
        )

    def _build_windows(self) -> list[SwingWindow]:
        cfg = self.config
        windows = []
        stride = cfg.swing_duration + cfg.swing_gap
        for idx, foot in enumerate(cfg.sequence):
            windows.append(
                SwingWindow(
                    foot=foot,
                    start_time=cfg.first_swing_start + idx * stride,
                    duration=cfg.swing_duration,
                )
            )
        return windows

    def _validate_config(self) -> None:
        cfg = self.config
        unknown = [foot for foot in cfg.sequence if foot not in cfg.foot_geoms]
        if unknown:
            raise ValueError(f"crawl sequence contains unknown feet: {unknown}")
        if cfg.swing_duration <= 0.0:
            raise ValueError("swing_duration must be positive")
        if cfg.swing_gap < 0.0:
            raise ValueError("swing_gap must be non-negative")
        if cfg.pre_shift_time <= 0.0:
            raise ValueError("pre_shift_time must be positive")
        step_delta = np.asarray(cfg.step_delta, dtype=float)
        if step_delta.shape != (3,):
            raise ValueError(f"step_delta must have shape (3,), got {step_delta.shape}")
        if cfg.max_step_length <= 0.0:
            raise ValueError("max_step_length must be positive")
