# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Exercise the whole capture-to-input path with controlled pixels and a fake mouse."""

import numpy as np
import pytest

from sightline.frame import Frame
from sightline.geometry import Box
from sightline.locator import AmbiguousTarget, TargetNotFound
from sightline.mouse import InputError
from sightline.ocr import OcrError, TextDetection
from sightline.session import Sightline
from sightline.timing import Timings


class Source:
    def __init__(self, frame: Frame) -> None:
        self.frame = frame
        self.calls = 0

    def capture(self) -> Frame:
        self.calls += 1
        return self.frame

    def close(self) -> None:
        pass


class Engine:
    def __init__(self, detections: tuple[TextDetection, ...]) -> None:
        self.detections = detections
        self.frames: list[Frame] = []

    def read(self, frame: Frame) -> tuple[TextDetection, ...]:
        self.frames.append(frame)
        return self.detections


class Mouse:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.size = (800, 600)

    def desktop_size(self) -> tuple[int, int]:
        return self.size

    def click(self, x: int, y: int) -> None:
        self.calls.append((x, y))


@pytest.fixture
def setup_session() -> tuple[Sightline, Source, Engine, Mouse]:
    frame = Frame(np.zeros((1200, 1600, 3), dtype=np.uint8), Box(0, 0, 800, 600))
    source = Source(frame)
    engine = Engine((TextDetection("Start", Box(100, 100, 300, 200), 0.95),))
    mouse = Mouse()
    return Sightline(source, engine, mouse), source, engine, mouse


def test_timed_session_distinguishes_preview_from_execution(setup_session) -> None:
    _, source, engine, mouse = setup_session
    timings = Timings()
    session = Sightline(source, engine, mouse, timings=timings)
    session.click_text("Start")
    assert set(timings.snapshot()) == {"capture", "ocr.read", "locate", "input.map"}
    session.click_text("Start", execute=True)
    stages = timings.snapshot()
    assert stages["input.click"].calls == stages["input.validate"].calls == 1
    assert stages["capture"].calls == stages["ocr.read"].calls == 2


def test_default_is_a_dry_run(setup_session) -> None:
    session, source, engine, mouse = setup_session
    result = session.click_text("start")
    assert result.point == (100, 75)
    assert result.executed is False
    assert mouse.calls == []
    assert source.calls == 1
    assert engine.frames == [source.frame]


def test_explicit_click_uses_scaled_label_center(setup_session) -> None:
    session, source, _, mouse = setup_session
    result = session.click_text("Start", execute=True)
    assert result.executed is True
    assert mouse.calls == [(100, 75)]
    assert source.calls == 1


@pytest.mark.parametrize("reason", ["duplicate", "missing", "low confidence"])
def test_uncertain_target_never_clicks(setup_session, reason: str) -> None:
    session, _, engine, mouse = setup_session
    if reason == "duplicate":
        engine.detections *= 2
        error = AmbiguousTarget
    else:
        label = "Other" if reason == "missing" else "Start"
        engine.detections = (TextDetection(label, Box(10, 10, 20, 20), 0.1),)
        error = TargetNotFound
    with pytest.raises(error):
        session.click_text("Start", execute=True)
    assert mouse.calls == []


@pytest.mark.parametrize("age", [60, -60])
def test_stale_or_future_frame_never_clicks(setup_session, monkeypatch, age: float) -> None:
    session, source, _, mouse = setup_session
    # Keep this check independent of how long the machine has been running.
    now = 100.0
    monkeypatch.setattr("sightline.session.time.monotonic", lambda: now)
    source.frame = Frame(source.frame.pixels, source.frame.desktop_bounds, captured_at=now - age)
    with pytest.raises(InputError, match="too old"):
        session.click_text("Start", execute=True)
    assert mouse.calls == []


def test_display_resize_never_clicks(setup_session) -> None:
    session, _, _, mouse = setup_session
    mouse.size = (1024, 768)
    with pytest.raises(InputError, match="Desktop size"):
        session.click_text("Start", execute=True)
    assert mouse.calls == []


def test_offline_frame_can_be_read_but_not_clicked(setup_session) -> None:
    session, source, _, mouse = setup_session
    source.frame = Frame(source.frame.pixels)
    assert session.find_text("Start").detection.text == "Start"
    with pytest.raises(InputError, match="no desktop mapping"):
        session.click_text("Start", execute=True)
    assert mouse.calls == []


def test_out_of_bounds_detection_is_rejected(setup_session) -> None:
    session, _, engine, mouse = setup_session
    engine.detections = (TextDetection("Start", Box(-10, 5, 20, 30), 1),)
    with pytest.raises(OcrError, match="outside"):
        session.click_text("Start", execute=True)
    assert mouse.calls == []


def test_mouse_is_optional_until_execution(setup_session) -> None:
    _, source, engine, _ = setup_session
    session = Sightline(source, engine)
    assert session.click_text("Start").executed is False
    with pytest.raises(InputError, match="controller"):
        session.click_text("Start", execute=True)


@pytest.mark.parametrize("age", [0, -1, float("inf"), float("nan")])
def test_invalid_frame_age_is_rejected(setup_session, age: float) -> None:
    _, source, engine, mouse = setup_session
    with pytest.raises(ValueError, match="positive"):
        Sightline(source, engine, mouse, max_frame_age=age)
