# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Offline inputs should behave like capture without touching the desktop."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sightline.frame import Frame
from sightline.sources import CaptureError, FrameSource, ImageFrameSource


def test_capture_returns_fresh_frames() -> None:
    source: FrameSource = ImageFrameSource(Frame(np.zeros((5, 7, 3), dtype=np.uint8)))
    first, second = source.capture(), source.capture()
    assert first is not second
    assert second.captured_at >= first.captured_at
    np.testing.assert_array_equal(first.pixels, second.pixels)
    source.close()


def test_file_loading_converts_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    Image.new("L", (10, 8), 75).save(path)
    with ImageFrameSource.from_file(path) as source:
        frame = source.capture()
        assert frame.pixels.shape == (8, 10, 3)
        assert frame.pixels[0, 0].tolist() == [75, 75, 75]
        assert frame.desktop_bounds is None
    with pytest.raises(CaptureError, match="closed"):
        source.capture()
    source.close()  # Closing twice is safe.


def test_missing_or_invalid_image_has_a_clear_error(tmp_path: Path) -> None:
    for path in (tmp_path / "missing.png", tmp_path / "bad.png"):
        if path.name == "bad.png":
            path.write_text("not an image")
        with pytest.raises(CaptureError, match="Cannot load"):
            ImageFrameSource.from_file(path)
