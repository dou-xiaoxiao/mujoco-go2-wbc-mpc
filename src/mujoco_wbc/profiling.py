"""Small timing helpers for simulation/control loops."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator


@dataclass
class TimingStats:
    """Accumulated wall-clock timing statistics for one named code section."""

    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.count += 1
        self.total_s += elapsed_s
        self.max_s = max(self.max_s, elapsed_s)

    @property
    def mean_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return 1000.0 * self.total_s / self.count

    @property
    def max_ms(self) -> float:
        return 1000.0 * self.max_s


class LoopProfiler:
    """Collect named section timings and print compact loop diagnostics."""

    def __init__(self) -> None:
        self._stats: dict[str, TimingStats] = {}

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - start)

    def add(self, name: str, elapsed_s: float) -> None:
        self._stats.setdefault(name, TimingStats()).add(elapsed_s)

    def reset(self) -> None:
        self._stats.clear()

    def summary_lines(self) -> list[str]:
        lines = []
        for name, stats in self._stats.items():
            lines.append(f"{name}: n={stats.count} mean={stats.mean_ms:.2f}ms max={stats.max_ms:.2f}ms")
        return lines
