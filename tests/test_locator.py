# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""A weak or ambiguous OCR result must not silently select a button."""

import pytest

from sightline.geometry import Box
from sightline.locator import AmbiguousTarget, TargetNotFound, TextLocator
from sightline.ocr import TextDetection


def label(text: str, confidence: float = 0.9) -> TextDetection:
    return TextDetection(text, Box(0, 0, 100, 30), confidence)


def test_unicode_case_and_whitespace_normalization() -> None:
    detection = label("ＳＴＡＲＴ\n  Now")
    assert TextLocator("start now").one([detection]) is detection


def test_exact_and_contains_are_distinct() -> None:
    detection = label("Restart task")
    with pytest.raises(TargetNotFound):
        TextLocator("start").one([detection])
    assert TextLocator("start", mode="contains").one([detection]) is detection


def test_low_confidence_is_ignored() -> None:
    with pytest.raises(TargetNotFound):
        TextLocator("Start", min_confidence=0.8).one([label("Start", 0.2)])


def test_duplicate_labels_are_never_resolved_by_confidence_alone() -> None:
    with pytest.raises(AmbiguousTarget, match="2 matches"):
        TextLocator("Start").one([label("Start", 0.99), label("Start", 0.8)])


@pytest.mark.parametrize("query", ["", " \t\n"])
def test_empty_query(query: str) -> None:
    with pytest.raises(ValueError):
        TextLocator(query)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_invalid_threshold(confidence: float) -> None:
    with pytest.raises(ValueError):
        TextLocator("Start", min_confidence=confidence)
