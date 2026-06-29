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
    command_velocity_ref_scale: float = 1.0


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


@dataclass(frozen=True)
class FootholdPlan:
    window_id: int
    foot: str
    initial_position: Array
    target_position: Array


@dataclass(frozen=True)
class ReferenceBundle:
    base_position_ref: Array
    base_orientation_ref: Array
    com_position_ref: Array
    com_velocity_ref: Array


class RollingFootholdPlanner:
    """Maintain foothold targets for repeated stepping.

    The planner owns the rolling foot state: stance feet are locked at their
    latest touchdown positions; when a new swing starts, that foot's target is
    generated relative to its current locked stance position.
    """

    def __init__(
        self,
        foot_geoms: tuple[str, ...],
        initial_foot_positions: dict[str, Array],
        step_delta: Array,
        step_deltas_by_foot: dict[str, Array] | None = None,
    ):
        self.foot_geoms = foot_geoms
        self.step_delta = np.asarray(step_delta, dtype=float)
        if self.step_delta.shape != (3,):
            raise ValueError(f"step_delta must have shape (3,), got {self.step_delta.shape}")
        self.step_deltas_by_foot = {
            foot: np.asarray(delta, dtype=float).copy()
            for foot, delta in (step_deltas_by_foot or {}).items()
        }
        self.locked_positions = {
            foot: np.asarray(initial_foot_positions[foot], dtype=float).copy()
            for foot in foot_geoms
        }
        self.plans: dict[int, FootholdPlan] = {}

    def start_swing(self, window_id: int, foot: str) -> FootholdPlan:
        initial = self.locked_positions[foot].copy()
        plan = FootholdPlan(
            window_id=window_id,
            foot=foot,
            initial_position=initial,
            target_position=initial + self.step_delta_for_foot(foot),
        )
        self.plans[window_id] = plan
        return plan

    def step_delta_for_foot(self, foot: str) -> Array:
        return self.step_deltas_by_foot.get(foot, self.step_delta)

    def touchdown(self, foot: str, actual_position: Array) -> None:
        self.locked_positions[foot] = np.asarray(actual_position, dtype=float).copy()

    def target_for_window(self, window_id: int) -> Array:
        return self.plans[window_id].target_position

    def swing_reference(
        self,
        window_id: int,
        swing_height: float,
        start_time: float,
        duration: float,
        time_s: float,
    ) -> SwingReference:
        plan = self.plans[window_id]
        return swing_foothold_reference(
            initial_position=plan.initial_position,
            step_delta=plan.target_position - plan.initial_position,
            swing_height=swing_height,
            start_time=start_time,
            duration=duration,
            time_s=time_s,
        )


class BodyReferencePlanner:
    """Generate conservative body xy references from support feet."""

    def __init__(self, foot_geoms: tuple[str, ...], support_centroid_ratio: float, pre_shift_time: float):
        self.foot_geoms = foot_geoms
        self.support_centroid_ratio = float(support_centroid_ratio)
        self.pre_shift_time = float(pre_shift_time)

    def reference_xy(
        self,
        nominal_body_xy: Array,
        locked_foot_positions: dict[str, Array],
        windows: list[SwingWindow],
        stance_feet_for_swing: dict[str, tuple[str, ...]],
        time_s: float,
        active_window_id: int | None,
        next_window_id: int,
    ) -> Array:
        nominal = np.asarray(nominal_body_xy, dtype=float)
        target = self._target_body_xy_for_window(
            nominal,
            locked_foot_positions,
            windows,
            stance_feet_for_swing,
            active_window_id,
            next_window_id,
        )

        if active_window_id is not None:
            return target
        if next_window_id >= len(windows):
            return target

        next_window = windows[next_window_id]
        shift_start = next_window.start_time - self.pre_shift_time
        if time_s < shift_start:
            return nominal.copy()
        if time_s >= next_window.start_time:
            return target

        ratio = (time_s - shift_start) / self.pre_shift_time
        s, _, _ = smoothstep(ratio)
        return nominal + s * (target - nominal)

    def _target_body_xy_for_window(
        self,
        nominal_body_xy: Array,
        locked_foot_positions: dict[str, Array],
        windows: list[SwingWindow],
        stance_feet_for_swing: dict[str, tuple[str, ...]],
        active_window_id: int | None,
        next_window_id: int,
    ) -> Array:
        if active_window_id is not None:
            support_feet = stance_feet_for_swing[windows[active_window_id].foot]
        elif next_window_id < len(windows):
            support_feet = stance_feet_for_swing[windows[next_window_id].foot]
        else:
            support_feet = self.foot_geoms

        support_points_xy = np.vstack([locked_foot_positions[foot][0:2] for foot in support_feet])
        return body_reference_from_support(
            nominal_body_xy,
            support_points_xy,
            centroid_ratio=self.support_centroid_ratio,
        )


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
        self.body_reference_planner = BodyReferencePlanner(
            self.config.foot_geoms,
            self.config.support_centroid_ratio,
            self.config.pre_shift_time,
        )

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

    def stance_feet_map(self) -> dict[str, tuple[str, ...]]:
        return {foot: self.stance_feet_for_swing(foot) for foot in self.config.foot_geoms}

    def rolling_foothold_planner(self, initial_foot_positions: dict[str, Array]) -> RollingFootholdPlanner:
        return RollingFootholdPlanner(
            self.config.foot_geoms,
            initial_foot_positions,
            self.step_delta(),
            self.foothold_deltas(initial_foot_positions),
        )

    def target_footholds(self, initial_foot_positions: dict[str, Array]) -> dict[str, Array]:
        deltas = self.foothold_deltas(initial_foot_positions)
        return {foot: np.asarray(pos, dtype=float) + deltas[foot] for foot, pos in initial_foot_positions.items()}

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

    def foothold_deltas(self, initial_foot_positions: dict[str, Array]) -> dict[str, Array]:
        return {
            foot: self.foothold_delta(foot, initial_foot_positions)
            for foot in self.config.foot_geoms
        }

    def foothold_delta(self, foot: str, initial_foot_positions: dict[str, Array]) -> Array:
        delta = self.step_delta().copy()
        command = self.config.command
        if command is None or command.yaw_rate == 0.0:
            return delta

        center_xy = np.mean(
            np.vstack([np.asarray(initial_foot_positions[name], dtype=float)[0:2] for name in self.config.foot_geoms]),
            axis=0,
        )
        foot_xy = np.asarray(initial_foot_positions[foot], dtype=float)[0:2]
        offset_xy = foot_xy - center_xy
        yaw_angle = command.yaw_rate * self.cycle_duration()
        delta[0:2] += yaw_angle * np.array([-offset_xy[1], offset_xy[0]], dtype=float)
        self._limit_planar_delta(delta)
        return delta

    def cycle_duration(self) -> float:
        if not self.config.foot_geoms:
            return 0.0
        return len(self.config.foot_geoms) * (self.config.swing_duration + self.config.swing_gap)

    def _limit_planar_delta(self, delta: Array) -> None:
        planar_norm = float(np.linalg.norm(delta[0:2]))
        if planar_norm > self.config.max_step_length:
            delta[0:2] *= self.config.max_step_length / planar_norm

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
        return self.body_reference_planner.reference_xy(
            nominal_body_xy,
            locked_foot_positions,
            self.windows,
            self.stance_feet_map(),
            time_s,
            active_window_id,
            next_window_id,
        )

    def reference_bundle(
        self,
        home_qpos_ref: Array,
        home_com_ref: Array,
        nominal_body_xy: Array,
        locked_foot_positions: dict[str, Array],
        time_s: float,
        active_window_id: int | None,
        next_window_id: int,
    ) -> ReferenceBundle:
        body_xy_ref = self.body_xy_reference(
            nominal_body_xy,
            locked_foot_positions,
            time_s,
            active_window_id,
            next_window_id,
        )

        base_position_ref = np.asarray(home_qpos_ref[0:3], dtype=float).copy()
        base_position_ref[0:2] = body_xy_ref
        base_orientation_ref = np.asarray(home_qpos_ref[3:7], dtype=float).copy()

        com_position_ref = np.asarray(home_com_ref, dtype=float).copy()
        com_position_ref[0:2] += body_xy_ref - np.asarray(nominal_body_xy, dtype=float)
        com_velocity_ref = self.command_velocity_ref()

        return ReferenceBundle(
            base_position_ref=base_position_ref,
            base_orientation_ref=base_orientation_ref,
            com_position_ref=com_position_ref,
            com_velocity_ref=com_velocity_ref,
        )

    def command_velocity_ref(self) -> Array:
        command = self.config.command
        if command is None:
            return np.zeros(3, dtype=float)
        return self.config.command_velocity_ref_scale * np.array([command.vx, command.vy, 0.0], dtype=float)

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
            step_delta=self.foothold_delta(foot, initial_foot_positions),
            swing_height=self.config.swing_height,
            start_time=start_time,
            duration=duration,
            time_s=time_s,
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
        if cfg.command_velocity_ref_scale < 0.0:
            raise ValueError("command_velocity_ref_scale must be non-negative")
