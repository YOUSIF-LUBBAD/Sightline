# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Small cumulative timing counters that can be shared by a session and its engine."""

import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass


@dataclass(frozen=True)
class StageTime:
    calls: int
    total_ns: int
    last_ns: int


class Timings:
    def __init__(self) -> None:
        self._stages: dict[str, StageTime] = {}
        self._lock = threading.Lock()

    def record(self, stage: str, elapsed_ns: int) -> None:
        with self._lock:
            old = self._stages.get(stage)
            self._stages[stage] = StageTime(
                1 if old is None else old.calls + 1,
                elapsed_ns if old is None else old.total_ns + elapsed_ns,
                elapsed_ns,
            )

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter_ns() - start)

    def snapshot(self) -> dict[str, StageTime]:
        with self._lock:
            return self._stages.copy()


def timed(timings: Timings | None, stage: str) -> AbstractContextManager[None]:
    return nullcontext() if timings is None else timings.measure(stage)
