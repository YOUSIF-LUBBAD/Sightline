# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Reject invalid recognition data at the engine boundary."""

import pytest

from sightline.geometry import Box
from sightline.ocr import TextDetection


@pytest.mark.parametrize("confidence", [-1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        TextDetection("Start", Box(0, 0, 5, 5), confidence)


@pytest.mark.parametrize("text", ["", "  ", "\n\t"])
def test_empty_text(text: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        TextDetection(text, Box(0, 0, 5, 5), 0.8)
