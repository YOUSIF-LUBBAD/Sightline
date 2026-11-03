# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Prove cache identity, bounded eviction and handling of failed reads."""

import numpy as np
import pytest

from sightline.cache import CachedOCR, CacheInfo
from sightline.frame import Frame
from sightline.geometry import Box
from sightline.ocr import OcrError, TextDetection


class Reader:
    def __init__(self):
        self.fail = False

    def read(self, frame):
        if self.fail:
            raise RuntimeError("model failed")
        return ()


def test_shape_pixels_lru_order_and_empty_hits() -> None:
    cache = CachedOCR(Reader(), capacity=2)
    a = Frame(np.zeros((2, 4, 3), dtype=np.uint8))
    b = Frame(np.zeros((4, 2, 3), dtype=np.uint8))
    c = Frame(np.ones((2, 4, 3), dtype=np.uint8))
    for frame in (a, b, Frame(a.pixels), c, b):
        assert cache.read(frame) == ()
    assert cache.info() == CacheInfo(hits=1, misses=4, entries=2)


def test_invalid_geometry_cannot_poison_a_later_cache_hit() -> None:
    class RepairsItsBox:
        def read(self, frame):
            return result

    result = (TextDetection("Start", Box(0, 0, 2, 1), 1),)
    cache = CachedOCR(RepairsItsBox())
    frame = Frame(np.zeros((1, 1, 3), dtype=np.uint8))
    with pytest.raises(OcrError, match="outside"):
        cache.read(frame)
    result = (TextDetection("Start", Box(0, 0, 1, 1), 1),)
    assert cache.read(frame) == result


def test_failing_read_is_retried() -> None:
    engine = Reader()
    cache = CachedOCR(engine)
    frame = Frame(np.zeros((1, 1, 3), dtype=np.uint8))
    engine.fail = True
    with pytest.raises(RuntimeError):
        cache.read(frame)
    engine.fail = False
    assert cache.read(frame) == ()
    assert cache.info() == CacheInfo(0, 2, 1)
