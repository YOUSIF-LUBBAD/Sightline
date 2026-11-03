# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""A fake capture device exercises conversions without reading the real screen."""

from dataclasses import dataclass

import pytest

from sightline.desktop import MSSFrameSource
from sightline.sources import CaptureError


@dataclass
class Shot:
    width: int = 4
    height: int = 2
    rgb: bytes = bytes([255, 0, 12] * 8)


class Backend:
    def __init__(self) -> None:
        self.monitors = [{}, {"left": 0, "top": 0, "width": 2, "height": 1}]
        self.closed = False
        self.fail = False

    def grab(self, monitor: dict[str, int]) -> Shot:
        if self.fail:
            raise OSError("display disconnected")
        return Shot()

    def close(self) -> None:
        self.closed = True


def test_rgb_and_retina_mapping_and_cleanup() -> None:
    backend = Backend()
    with MSSFrameSource((2, 1), backend_factory=lambda: backend) as source:
        frame = source.capture()
        assert frame.pixels[0, 0].tolist() == [255, 0, 12]
        assert frame.to_desktop(3, 1) == (1, 0)
        assert frame.source == "desktop:primary"
    assert backend.closed
    with pytest.raises(CaptureError, match="closed"):
        source.capture()


def test_unmapped_capture_is_read_only() -> None:
    source = MSSFrameSource(backend_factory=Backend)
    assert source.capture().desktop_bounds is None
    source.close()


def test_missing_monitor_and_secondary_monitor_are_rejected() -> None:
    backend = Backend()
    backend.monitors = [{}]
    source = MSSFrameSource(backend_factory=lambda: backend)
    with pytest.raises(CaptureError, match="No monitor"):
        source.capture()
    backend.monitors = [{}, {"left": -100, "top": 0, "width": 100, "height": 100}]
    with pytest.raises(CaptureError, match="not primary"):
        source.capture()
    source.close()


def test_capture_failure_keeps_original_exception() -> None:
    backend = Backend()
    backend.fail = True
    with MSSFrameSource(backend_factory=lambda: backend) as source:
        with pytest.raises(CaptureError) as caught:
            source.capture()
    assert isinstance(caught.value.__cause__, OSError)
    assert backend.closed


def test_invalid_input_size() -> None:
    with pytest.raises(ValueError):
        MSSFrameSource((0, 5))
