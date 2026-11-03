# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Check the library boundary with small fake results, not downloaded models."""

import numpy as np
import pytest
from numpy.typing import NDArray

from sightline.easyocr_engine import EasyOCRConfig, EasyOCREngine, RawDetection
from sightline.frame import Frame
from sightline.geometry import Box
from sightline.ocr import OcrError


class FakeReader:
    def __init__(self, results: list[RawDetection]) -> None:
        self.results = results
        self.seen: NDArray[np.uint8] | None = None

    def readtext(
        self, image: NDArray[np.uint8], *, detail: int, paragraph: bool
    ) -> list[RawDetection]:
        assert detail == 1 and paragraph is False
        self.seen = image
        return self.results


def test_reuses_reader_converts_color_and_clips_boxes() -> None:
    reader = FakeReader([([[-2, 1], [20, 1], [20, 5], [-2, 5]], "Start", 0.9)])
    creations = []

    def factory() -> FakeReader:
        creations.append(True)
        return reader

    engine = EasyOCREngine(reader_factory=factory)
    pixels = np.full((10, 12, 3), [255, 15, 5], dtype=np.uint8)
    frame = Frame(pixels)
    assert engine.read(frame)[0].box == Box(0, 1, 12, 5)
    engine.read(frame)
    assert len(creations) == 1
    assert reader.seen is not None
    assert reader.seen[0, 0].tolist() == [5, 15, 255]
    assert frame.pixels[0, 0].tolist() == [255, 15, 5]


def test_ignores_empty_and_offscreen_labels() -> None:
    reader = FakeReader(
        [
            ([[0, 0], [2, 0], [2, 2], [0, 2]], " ", 0.5),
            ([[20, 20], [22, 20], [22, 22], [20, 22]], "Hidden", 0.9),
        ]
    )
    frame = Frame(np.zeros((10, 10, 3), dtype=np.uint8))
    assert EasyOCREngine(reader_factory=lambda: reader).read(frame) == ()


def test_malformed_results_do_not_become_click_targets() -> None:
    reader = FakeReader([([[0, 0], [2, 2]], "Start", 0.9)])
    with pytest.raises(OcrError, match="invalid detections"):
        EasyOCREngine(reader_factory=lambda: reader).read(
            Frame(np.zeros((5, 5, 3), dtype=np.uint8))
        )


def test_reader_initialization_error_is_clear() -> None:
    def broken() -> FakeReader:
        raise FileNotFoundError("missing weights")

    with pytest.raises(OcrError, match="enable downloads"):
        EasyOCREngine(reader_factory=broken).prepare()


@pytest.mark.parametrize("languages", [(), ("",), ("en", "en")])
def test_invalid_language_configuration(languages: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        EasyOCRConfig(languages=languages)


def test_downloads_are_off_by_default() -> None:
    assert EasyOCRConfig().download_enabled is False
