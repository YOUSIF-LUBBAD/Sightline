# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Crop coordinates must never leak into full-screen input coordinates."""

import numpy as np
import pytest

from sightline.frame import Frame
from sightline.geometry import Box
from sightline.ocr import OcrError, TextDetection
from sightline.roi import read_region


class CropReader:
    def __init__(self, box: Box) -> None:
        self.box = box

    def read(self, crop):
        assert crop.desktop_bounds is None
        assert crop.pixels.shape == (4, 5, 3)
        assert crop.captured_at == 12
        return (TextDetection("Run", self.box, 0.8),)


def test_crop_offsets_fractional_detections_and_keeps_confidence() -> None:
    frame = Frame(np.zeros((15, 15, 3), dtype=np.uint8), Box(0, 0, 30, 30), captured_at=12)
    region = Box(5, 8, 10, 12)
    assert read_region(CropReader(Box(0.5, 1, 5, 4)), frame, region) == (
        TextDetection("Run", Box(5.5, 9, 10, 12), 0.8),
    )
    with pytest.raises(OcrError, match="outside"):
        read_region(CropReader(Box(0, 0, 6, 4)), frame, region)
    with pytest.raises(ValueError, match="whole"):
        read_region(CropReader(region), frame, Box(0.5, 0, 2, 5))
