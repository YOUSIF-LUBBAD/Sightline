# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Bounded OCR result cache owned by one recognizer configuration."""

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

from sightline.frame import Frame
from sightline.ocr import OcrEngine, OcrError, TextDetection
from sightline.timing import Timings, timed


@dataclass(frozen=True)
class CacheInfo:
    hits: int
    misses: int
    entries: int


class CachedOCR:
    def __init__(
        self, engine: OcrEngine, capacity: int = 128, *, timings: Timings | None = None
    ) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Cache capacity must be a positive integer")
        self.engine = engine
        self.capacity = capacity
        self.timings = timings
        self._entries: OrderedDict[tuple[int, int, bytes], tuple[TextDetection, ...]] = (
            OrderedDict()
        )
        self._hits = self._misses = 0
        self._lock = threading.Lock()

    def info(self) -> CacheInfo:
        with self._lock:
            return CacheInfo(self._hits, self._misses, len(self._entries))

    def read(self, frame: Frame) -> tuple[TextDetection, ...]:
        with timed(self.timings, "cache.hash"):
            digest = hashlib.blake2b(memoryview(frame.pixels), digest_size=32).digest()
            key = (frame.height, frame.width, digest)
        with self._lock:
            with timed(self.timings, "cache.lookup"):
                hit = self._entries.get(key)
                if hit is not None:
                    self._entries.move_to_end(key)
                    self._hits += 1
                    return hit
                self._misses += 1
            with timed(self.timings, "ocr.infer"):
                result = self.engine.read(frame)
            if any(item.box.clipped_to(frame.pixel_bounds) != item.box for item in result):
                raise OcrError("OCR returned a detection outside its image")
            self._entries[key] = result
            if len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
            return result
