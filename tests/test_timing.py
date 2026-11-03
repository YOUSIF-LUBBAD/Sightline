# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""A failed stage still records its duration and cannot erase earlier samples."""

import pytest

from sightline.timing import StageTime, Timings


def test_records_success_and_failure_with_independent_snapshots(monkeypatch) -> None:
    ticks = iter([100, 121, 130, 140])
    monkeypatch.setattr("sightline.timing.time.perf_counter_ns", lambda: next(ticks))
    timings = Timings()
    with timings.measure("ocr"):
        pass
    first = timings.snapshot()
    with pytest.raises(ValueError), timings.measure("ocr"):
        raise ValueError
    assert first["ocr"] == StageTime(1, 21, 21)
    assert timings.snapshot()["ocr"] == StageTime(2, 31, 10)
